#!/usr/bin/env python3
"""Single-file BaldEagle drafter eval driver.

Two responsibilities:
  1. Boot SGLang with the chosen drafter against its target.
  2. Iterate (dataset, lang, seed) cells from the frozen master eval set,
     run prompts through SGLang, write per-cell results.

Inputs:
  - benchmark/drafters.yaml          drafter registry (target, path, status, ...)
  - cache/eval_set/eval_n<N>_seed<S>.parquet
                                     master eval set (built by scripts/build_eval_set.py)

Outputs (per (drafter, dataset, lang, seed) cell):
  - benchmark/results/{run_id}/meta.json              env + provenance
  - benchmark/results/{run_id}/manifest_snapshot.yaml exact drafter entry used
  - benchmark/results/{run_id}/requests.jsonl         per-request rows (raw)
  - benchmark/results/{run_id}/summary.json           aggregated metrics
  run_id = {target}_{drafter}_{dataset}_{lang}_n{N}_seed{S}

Resumability:
  Skip-if-summary-exists. Pass --force to overwrite.
  SGLang server is booted ONCE per drafter and reused across all (ds, lang, seed)
  cells for that drafter (saves ~60s of cold start per cell).

Usage (runs from BaldEagle's uv venv — no sglang import needed in this file):
  uv run python benchmark/eval.py --drafter aya-base --n 50
  uv run python benchmark/eval.py --drafter aya-base --datasets mgsm --langs de --n 5
  uv run python benchmark/eval.py --audit                # status report only
  uv run python benchmark/eval.py --sweep                # all ready drafters

eval.py talks to the SGLang server over HTTP (/v1/chat/completions) so it has
no sglang dependency itself. The SERVER subprocess is launched via a Python
that DOES have sglang installed — set `VENV_PYTHON` below if your sglang
install lives elsewhere than the default. Default points to the SpecForge venv
because that's where the patched sglang 0.5.9 lives on this cluster.
"""

import argparse
import datetime as _dt
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import time
import unicodedata
from typing import Iterator

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

# Ensure localhost calls bypass any inherited http_proxy (we rely on squid for HF
# access, but health/inference calls go to 127.0.0.1).
os.environ.setdefault("no_proxy", "localhost,127.0.0.1,0.0.0.0")
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1,0.0.0.0")

# ─── Constants ─────────────────────────────────────────────────────────────

REGISTRY_PATH = os.path.join(REPO, "benchmark", "drafters.yaml")
RESULTS_ROOT  = os.path.join(REPO, "benchmark", "results")

# Per-target active langs. llama31 = de/es/fr/hi (v2 story); aya/qwen = de/ja/zh/hi.
TARGET_LANGS = {
    "llama31": ["de", "es", "fr", "hi"],
    "aya":     ["de", "ja", "zh", "hi"],
    "qwen":    ["de", "ja", "zh", "hi"],
}

# Per-dataset language coverage. Cells outside the intersection of TARGET_LANGS
# and DATASET_LANGS are skipped at planning time.
DATASET_LANGS = {
    "sharegpt":      ["de", "es", "fr", "ja", "zh", "hi"],
    "marena":        ["de", "es", "fr", "ja", "zh", "hi"],
    "mgsm":          ["de", "es", "fr", "ja", "zh"],   # no hi upstream
    "xl-sum":        ["es", "fr", "ja", "zh", "hi"],   # no de upstream (BBC)
    "codemix-gsm8k": ["es", "hi", "zh"],
    "mtbench":       ["en"],
}

DATASETS_ALL = list(DATASET_LANGS.keys())

# Spec-decode hyperparams (also pinned in drafters.yaml.spec_config; redundant
# here for the audit trail in meta.json).
SPEC_NUM_STEPS    = 5
SPEC_EAGLE_TOPK   = 8
SPEC_DRAFT_TOKENS = 64

# Server boot CUDA env (matches the scratch overlay; see auto-memory pin on
# /software/CUDA having a noexcept clash with glibc).
CUDA_HOME    = "/scratch/htc/snelaturu/CUDA/cuda-12.9"
HF_HOME      = "/scratch/htc/snelaturu/CACHES/huggingface"
SQUID_PROXY  = "http://squid.zib.de:3128"
VENV_PYTHON  = "/scratch/htc/snelaturu/codebase/SpecForge-z1/.venv/bin/python"

SYSTEM_MESSAGE = (
    "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, "
    "while being safe.  Your answers should not include any harmful, unethical, racist, sexist, "
    "toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased "
    "and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, "
    "explain why instead of answering something not correct. If you don't know the answer to a "
    "question, please don't share false information."
)

# ─── Drafter registry ──────────────────────────────────────────────────────

def load_registry():
    import yaml
    with open(REGISTRY_PATH) as f:
        return yaml.safe_load(f)


def resolve_drafter(name, target_override=None):
    """Return (drafter_entry_or_none, target_name, target_entry, registry)."""
    reg = load_registry()
    if name == "none":
        target = target_override or "llama31"
        return None, target, reg["targets"][target], reg
    if name not in reg["drafters"]:
        raise SystemExit(f"unknown drafter {name!r} (known: {sorted(reg['drafters'])[:8]}…)")
    d = reg["drafters"][name]
    target = d["target"]
    return d, target, reg["targets"][target], reg


def vocab_guard(drafter_dir, target_entry, target_name):
    """Confirm the drafter's config.vocab_size matches the target's expected vocab."""
    cfg_path = os.path.join(drafter_dir, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    got = cfg.get("vocab_size")
    want = target_entry["vocab_size_check"]
    if got != want:
        raise SystemExit(f"VOCAB MISMATCH: drafter={got} != target {target_name} expects={want}")
    print(f"[guard] vocab OK: drafter={got} == target({target_name})={want}")


# ─── Coverage ──────────────────────────────────────────────────────────────

def combos_for(drafter_entry, target_name, datasets_filter=None, langs_filter=None):
    """Return list[(dataset, lang)] for a drafter under the per-target lang grid."""
    target_langs = TARGET_LANGS.get(target_name, ["de", "es", "fr", "ja", "zh", "hi"])
    focus = (drafter_entry or {}).get("focus_lang")
    out = []

    if focus:
        # Specialist: own lang only — and only if still in target's active set.
        if focus not in target_langs:
            return [("mtbench", "en")]
        for ds in DATASETS_ALL:
            if ds == "mtbench":
                continue
            if focus in DATASET_LANGS.get(ds, target_langs):
                out.append((ds, focus))
        out.append(("mtbench", "en"))
    else:
        for lang in target_langs:
            for ds in DATASETS_ALL:
                if ds == "mtbench":
                    continue
                if lang in DATASET_LANGS.get(ds, target_langs):
                    out.append((ds, lang))
        out.append(("mtbench", "en"))

    if datasets_filter:
        ok = set(datasets_filter)
        out = [c for c in out if c[0] in ok]
    if langs_filter:
        ok = set(langs_filter)
        out = [c for c in out if c[1] in ok]
    return out


# ─── Eval-set reader ───────────────────────────────────────────────────────

def eval_set_path(n_pool, seed):
    return os.path.join(REPO, f"cache/eval_set/eval_n{n_pool}_seed{seed}.parquet")


def load_prompts(dataset, lang, n, seed, n_pool=300):
    """Read the (dataset, lang) cell from the per-seed master parquet, take first n."""
    import pyarrow.parquet as pq
    path = eval_set_path(n_pool, seed)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"master eval set missing: {path}. "
            f"Run: uv run python scripts/build_eval_set.py --seeds {seed}"
        )
    if dataset == "mtbench" and not lang:
        lang = "en"
    table = pq.read_table(path, filters=[("dataset", "=", dataset), ("lang", "=", lang)])
    if table.num_rows == 0:
        raise ValueError(f"no rows in {path} for (dataset={dataset!r}, lang={lang!r})")
    rows = table.to_pylist()
    rows.sort(key=lambda r: r["sample_idx"])
    if n > 0:
        rows = rows[:n]
    return rows


# ─── Sanity checks (rule-based, no model) ──────────────────────────────────

# Cheap script-presence checks for the languages we bench. >5% target-script
# chars in a >=50-char response is the threshold; below that we flag drift.
_SCRIPT_RANGES = {
    "de": [(0x0041, 0x007A), (0x00C0, 0x00FF)],   # Latin
    "es": [(0x0041, 0x007A), (0x00C0, 0x00FF)],
    "fr": [(0x0041, 0x007A), (0x00C0, 0x00FF)],
    "ja": [(0x3040, 0x30FF), (0x4E00, 0x9FFF)],   # Hiragana/Katakana/CJK
    "zh": [(0x4E00, 0x9FFF), (0x3400, 0x4DBF)],   # CJK
    "hi": [(0x0900, 0x097F)],                     # Devanagari
    "en": [(0x0041, 0x007A)],                     # Latin
}


def check_sanity(response, prompt_text, lang, dataset, system_msg):
    """5 cheap response checks → dict[flag, bool]. Any True = sanity violation."""
    flags = {
        "too_short": False,
        "repetition": False,
        f"script_mismatch_{lang}": False,
        "prompt_echo": False,
        "system_leak": False,
    }
    if not response or len(response) < 5:
        flags["too_short"] = True
        return flags

    # Repetition: a single character dominates the response.
    counts = {}
    for ch in response:
        counts[ch] = counts.get(ch, 0) + 1
    top_frac = max(counts.values()) / len(response)
    if top_frac > 0.85:
        flags["repetition"] = True

    # Script mismatch: <5% target-script chars in a long-enough response.
    # Suppressed for codemix-* (legitimate Latin output is expected).
    if lang in _SCRIPT_RANGES and len(response) >= 50 and not dataset.startswith("codemix"):
        ranges = _SCRIPT_RANGES[lang]
        in_script = sum(1 for ch in response if any(a <= ord(ch) <= b for a, b in ranges))
        if in_script / len(response) < 0.05:
            flags[f"script_mismatch_{lang}"] = True

    # Prompt echo: response is essentially the user prompt verbatim.
    if prompt_text and len(prompt_text) >= 30:
        norm_p = re.sub(r"\s+", " ", prompt_text.strip())
        norm_r = re.sub(r"\s+", " ", response.strip())
        if norm_p in norm_r and len(norm_p) / max(len(norm_r), 1) > 0.9:
            flags["prompt_echo"] = True

    # System leak: response includes literal SYSTEM:/USER:/ASSISTANT: markers,
    # or quotes the system message verbatim. Catches the SGL frontend "default"
    # template bug (emits literal "SYSTEM:" markers).
    head = response.lstrip()[:64].upper()
    if head.startswith(("SYSTEM:", "USER:", "ASSISTANT:")):
        flags["system_leak"] = True
    if system_msg and len(system_msg) >= 30 and system_msg[:120] in response:
        flags["system_leak"] = True

    return flags


# ─── SGLang HTTP client (no sglang Python import) ─────────────────────────
# We talk to the server's OpenAI-compatible /v1/chat/completions endpoint and
# read SGLang's `meta_info` extension for spec-decode metrics. No `import sglang`
# anywhere in this file — keeps eval.py runnable from BaldEagle's uv venv.
# The SERVER subprocess still uses whatever Python has sglang installed (set
# via VENV_PYTHON below) — that's just where the server binary lives, not
# eval.py's runtime concern.

import http.client as _http
import json as _json


def _post_generate(host, port, prompt_text, max_new_tokens, temperature):
    """POST to SGLang's native /generate; return (text, meta_info_dict).
    /generate carries full meta_info including spec_verify_ct, spec_accept_rate, etc.
    """
    body = _json.dumps({
        "text": prompt_text,
        "sampling_params": {
            "max_new_tokens": max_new_tokens,
            "temperature":    temperature,
        },
    })
    conn = _http.HTTPConnection(host, port, timeout=600)
    conn.request("POST", "/generate", body=body,
                 headers={"Content-Type": "application/json"})
    r = conn.getresponse()
    raw = r.read()
    conn.close()
    if r.status != 200:
        raise RuntimeError(f"/generate HTTP {r.status}: {raw[:300]!r}")
    out = _json.loads(raw)
    return out.get("text", ""), (out.get("meta_info") or {})


def run_one_prompt(client, prompt_row, max_new_tokens, temperature, lang, dataset):
    """Apply target tokenizer's chat template, POST to SGLang /generate, parse meta_info."""
    host, port = client["host"], client["port"]
    tokenizer = client["tokenizer"]
    messages = [
        {"role": "system",  "content": prompt_row["system"] or SYSTEM_MESSAGE},
        {"role": "user",    "content": prompt_row["prompt_text"]},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    response_text, info = _post_generate(host, port, prompt_text,
                                          max_new_tokens, temperature)

    completion = int(info.get("completion_tokens", 0) or 0)
    verify = int(info.get("spec_verify_ct", 0) or 0)
    accept_length = (completion / verify) if verify > 0 else None

    sanity = check_sanity(
        response_text,
        prompt_row["prompt_text"],
        lang,
        dataset,
        prompt_row["system"] or SYSTEM_MESSAGE,
    )

    return {
        "prompt_id":            prompt_row["prompt_id"],
        "prompt_tokens":        int(info.get("prompt_tokens", 0) or 0),
        "completion_tokens":    completion,
        "spec_verify_ct":       verify,
        "accept_length":        accept_length,
        "spec_accept_rate":     info.get("spec_accept_rate"),
        "spec_accept_token_num": info.get("spec_accept_token_num"),
        "spec_draft_token_num": info.get("spec_draft_token_num"),
        "spec_accept_histogram": info.get("spec_accept_histogram"),
        "finish_reason":        (info.get("finish_reason") or {}).get("type", "")
                                 if isinstance(info.get("finish_reason"), dict)
                                 else (info.get("finish_reason") or ""),
        "response_text":        response_text,
        "sanity":               sanity,
    }


def summarize(rows, wall_clock_s, run_id):
    """Pure aggregation: rows -> summary dict."""
    n = len(rows)
    completion_sum = sum(r["completion_tokens"] for r in rows)
    verify_sum     = sum(r["spec_verify_ct"] for r in rows)
    accept_token_sum = sum(int(r["spec_accept_token_num"] or 0) for r in rows)
    draft_token_sum  = sum(int(r["spec_draft_token_num"] or 0) for r in rows)

    accept_lengths = [r["accept_length"] for r in rows if r["accept_length"] is not None]
    accept_rates   = [r["spec_accept_rate"] for r in rows if r["spec_accept_rate"] is not None]

    def _stats(xs):
        if not xs:
            return None
        xs_sorted = sorted(xs)
        return {
            "n": len(xs),
            "mean": sum(xs) / len(xs),
            "p50": xs_sorted[len(xs) // 2],
            "p90": xs_sorted[max(0, int(0.9 * len(xs)) - 1)],
            "min": xs_sorted[0],
            "max": xs_sorted[-1],
        }

    sanity_flags_total = {}
    insane = 0
    for r in rows:
        any_flag = False
        for flag, val in (r["sanity"] or {}).items():
            if val:
                sanity_flags_total[flag] = sanity_flags_total.get(flag, 0) + 1
                any_flag = True
        if any_flag:
            insane += 1

    n_valid = sum(1 for r in rows if r["completion_tokens"] > 0)
    accept_rate_overall = (accept_token_sum / draft_token_sum) if draft_token_sum > 0 else None
    tps_overall = (completion_sum / wall_clock_s) if wall_clock_s > 0 else None
    spec_active = verify_sum > 0

    # Leviathan-style derived metrics. accept_length already captured above.
    # Per the EAGLE-2 / Leviathan literature the standard reporting form is:
    #   alpha_per_token  = (E[accept_length] - 1) / num_steps   — per-drafter-token acceptance rate
    #   alpha_normalized = E[accept_length]       / (num_steps + 1) — fraction of ideal speedup
    # Both are bounded [0, 1] and directly comparable across drafters at fixed num_steps.
    #
    # Estimator choice for E[accept_length] across the cell — TWO options, both reported:
    #
    #   (A) RATIO-OF-SUMS (preferred, headline number):
    #         accept_length_overall = completion_sum / verify_sum
    #       Each verify step is weighted equally regardless of which prompt it came from.
    #       This is the unbiased per-step estimator (a Horvitz–Thompson-style ratio) and
    #       is what the EAGLE / Leviathan papers report. Long, hard prompts contribute
    #       proportionally to how many verify steps they actually consumed.
    #
    #   (B) MEAN-OF-RATIOS (diagnostic only, *_mor suffix):
    #         accept_length_mor = mean_p( completion_p / verify_p )
    #       Each prompt contributes one ratio regardless of length. Biased toward short
    #       prompts (where 1–2 lucky accepts swing the ratio). Kept for backward
    #       compatibility with prior summaries and so the gap (A) - (B) is visible —
    #       a large gap means response lengths vary a lot, which itself is informative.
    #
    # We expose `alpha_per_token` / `alpha_normalized` as the preferred (A) numbers,
    # and `alpha_per_token_mor` / `alpha_normalized_mor` as the (B) diagnostics.
    al = _stats(accept_lengths)

    accept_length_overall = (completion_sum / verify_sum) if verify_sum > 0 else None
    if accept_length_overall is not None:
        alpha_per_token  = (accept_length_overall - 1.0) / SPEC_NUM_STEPS
        alpha_normalized = accept_length_overall         / (SPEC_NUM_STEPS + 1.0)
    else:
        alpha_per_token  = None
        alpha_normalized = None

    if al:
        alpha_per_token_mor  = (al["mean"] - 1.0) / SPEC_NUM_STEPS
        alpha_normalized_mor = al["mean"]         / (SPEC_NUM_STEPS + 1.0)
    else:
        alpha_per_token_mor  = None
        alpha_normalized_mor = None

    return {
        "run_id":             run_id,
        "n":                  n,
        "n_valid":            n_valid,
        "spec_decode_active": spec_active,
        "accept_length":      al,
        # Cell-level E[accept_length] via ratio-of-sums (unbiased; preferred for the paper).
        "accept_length_overall": accept_length_overall,
        # Leviathan-style derived metrics — HEADLINE numbers (ratio-of-sums).
        "alpha_per_token":    alpha_per_token,    # (accept_length_overall - 1) / num_steps
        "alpha_normalized":   alpha_normalized,   # accept_length_overall       / (num_steps + 1)
        # Mean-of-ratios diagnostics: biased toward short prompts, kept for auditability
        # and backward comparison with prior runs. Compare against the headline numbers
        # above; a large gap ⇒ response-length variance is large within the cell.
        "alpha_per_token_mor":  alpha_per_token_mor,
        "alpha_normalized_mor": alpha_normalized_mor,
        # SGLang's tree-fraction acceptance (denominator = verify_ct × (num_draft_tokens-1)).
        # NOT Leviathan's α — this is "fraction of all tree-proposed tokens accepted",
        # naturally low because most tree branches get culled.
        "accept_rate":        _stats(accept_rates),
        "accept_rate_overall": accept_rate_overall,
        "completion_tokens":  {"sum": completion_sum, "mean": completion_sum / max(n, 1)},
        "verify_tokens":      {"sum": verify_sum, "mean": verify_sum / max(n, 1)},
        "wall_clock_s":       wall_clock_s,
        "tps_overall":        tps_overall,
        "sanity": {
            "n_insane":            insane,
            "insane_fraction":     insane / max(n, 1),
            "flags_total":         sanity_flags_total,
        },
    }


# ─── SGLang server lifecycle ───────────────────────────────────────────────

def find_free_port(start=30000):
    for port in range(start, start + 100):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("no free port in range")


def launch_sglang(target_hf, drafter_dir, port, log_path, target_extra_env=None):
    """Boot SGLang in the background; return Popen handle. Caller must terminate."""
    env = os.environ.copy()
    env.update({
        "CUDA_HOME":           CUDA_HOME,
        "CUDA_PATH":           CUDA_HOME,
        "PATH":                f"{CUDA_HOME}/bin:" + env.get("PATH", ""),
        "LD_LIBRARY_PATH":     f"{CUDA_HOME}/lib64:" + env.get("LD_LIBRARY_PATH", ""),
        "HF_HOME":             HF_HOME,
        "http_proxy":          SQUID_PROXY,
        "https_proxy":         SQUID_PROXY,
        "no_proxy":            "localhost,127.0.0.1,0.0.0.0",
        "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
    })
    if target_extra_env:
        for k, v in target_extra_env.items():
            env[k] = str(v)

    cmd = [
        VENV_PYTHON, "-m", "sglang.launch_server",
        "--model-path", target_hf,
        "--port", str(port),
        "--dtype", "bfloat16",
        "--mem-fraction-static", "0.85",
    ]
    if drafter_dir is not None:
        cmd += [
            "--context-length", "4096",
            "--speculative-algorithm", "EAGLE",
            "--speculative-draft-model-path", drafter_dir,
            "--speculative-num-steps", str(SPEC_NUM_STEPS),
            "--speculative-eagle-topk", str(SPEC_EAGLE_TOPK),
            "--speculative-num-draft-tokens", str(SPEC_DRAFT_TOKENS),
        ]

    log_f = open(log_path, "w")
    print(f"[server] launching SGLang (port {port}, log {log_path})")
    proc = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
    return proc, log_f


def wait_for_server(port, timeout_s=300):
    """Poll /health until 200 OK or timeout. Uses http.client directly to bypass
    any http_proxy env var (squid would otherwise intercept the localhost call)."""
    import http.client
    deadline = time.time() + timeout_s
    started = time.time()
    while time.time() < deadline:
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
            conn.request("GET", "/health")
            r = conn.getresponse()
            r.read()
            conn.close()
            if r.status == 200:
                print(f"[server] ready in {int(time.time() - started)}s")
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def stop_server(proc, log_f):
    print(f"[server] terminating pid={proc.pid}")
    try:
        proc.terminate()
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
    finally:
        log_f.close()


# ─── Per-cell run + write ──────────────────────────────────────────────────

def run_id_for(target, drafter, dataset, lang, n, seed):
    return f"{target}_{drafter}_{dataset}_{lang}_n{n}_seed{seed}"


def cell_done(out_dir):
    return os.path.isfile(os.path.join(out_dir, "summary.json"))


def write_meta(out_dir, drafter_entry, target_entry, target_name, drafter_name,
               registry, args, n_prompts, dataset, lang, seed):
    """Provenance trap. Captures everything you'd need a year from now."""
    drafter_path = None
    drafter_sha = None
    if drafter_entry is not None:
        drafter_path = os.path.abspath(os.path.join(REPO, drafter_entry["path"]))
        sft = os.path.join(drafter_path, "model.safetensors")
        if os.path.isfile(sft):
            h = hashlib.sha256()
            with open(sft, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            drafter_sha = h.hexdigest()

    git_sha = ""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                             capture_output=True, text=True, timeout=2)
        git_sha = out.stdout.strip()
    except Exception:
        pass

    meta = {
        "run_id":          run_id_for(target_name, drafter_name, dataset, lang,
                                      args.n, seed),
        "drafter":         drafter_name,
        "drafter_path":    drafter_path,
        "drafter_sha256":  drafter_sha,
        "target":          target_name,
        "target_hf":       target_entry["hf_name"],
        "dataset":         dataset,
        "lang":            lang,
        "n_requested":     args.n,
        "n_loaded":        n_prompts,
        "seed":            seed,
        "max_new_tokens":  args.max_new_tokens,
        "temperature":     args.temperature,
        "spec_config":     {"num_steps": SPEC_NUM_STEPS,
                            "eagle_topk": SPEC_EAGLE_TOPK,
                            "draft_tokens": SPEC_DRAFT_TOKENS},
        "client_chat_template": target_entry.get("client_chat_template"),
        "git_sha":         git_sha,
        "started_utc":     _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def write_manifest_snapshot(out_dir, drafter_name, drafter_entry, target_name,
                             target_entry, registry):
    import yaml
    snap = {
        "drafter_name":   drafter_name,
        "drafter_entry":  drafter_entry,
        "target_name":    target_name,
        "target_entry":   target_entry,
        "spec_config":    registry.get("spec_config"),
        "registry_path":  REGISTRY_PATH,
    }
    with open(os.path.join(out_dir, "manifest_snapshot.yaml"), "w") as f:
        yaml.safe_dump(snap, f, sort_keys=False, default_flow_style=False)


def run_cell(client, drafter_name, drafter_entry, target_name, target_entry,
             registry, dataset, lang, seed, args):
    """Run one (drafter, dataset, lang, seed) cell. Skip if already done unless --force."""
    run_id = run_id_for(target_name, drafter_name, dataset, lang, args.n, seed)
    out_dir = os.path.join(RESULTS_ROOT, run_id)

    if cell_done(out_dir) and not args.force:
        print(f"[skip] {run_id} (summary.json exists; pass --force to overwrite)")
        return "skipped"

    os.makedirs(out_dir, exist_ok=True)
    prompts = load_prompts(dataset, lang, args.n, seed)
    print(f"[cell] {run_id}  {len(prompts)} prompts")

    write_meta(out_dir, drafter_entry, target_entry, target_name, drafter_name,
               registry, args, len(prompts), dataset, lang, seed)
    write_manifest_snapshot(out_dir, drafter_name, drafter_entry, target_name,
                             target_entry, registry)

    rows = []
    t0 = time.time()
    for i, p in enumerate(prompts):
        try:
            row = run_one_prompt(client, p, args.max_new_tokens, args.temperature,
                                 lang, dataset)
        except Exception as e:
            print(f"  [err] prompt {i} ({p['prompt_id']}): {type(e).__name__}: {e}",
                  file=sys.stderr)
            row = {
                "prompt_id":         p["prompt_id"],
                "prompt_tokens":     0,
                "completion_tokens": 0,
                "spec_verify_ct":    0,
                "accept_length":     None,
                "spec_accept_rate":  None,
                "spec_accept_token_num": None,
                "spec_draft_token_num":  None,
                "spec_accept_histogram": None,
                "finish_reason":     "error",
                "response_text":     "",
                "sanity":            {"too_short": True},
                "error":             f"{type(e).__name__}: {e}",
            }
        rows.append(row)
        if (i + 1) % 25 == 0:
            print(f"  ... {i+1}/{len(prompts)}")
    wall = time.time() - t0
    print(f"[cell] {run_id} done in {wall:.1f}s  ({wall/max(len(rows),1):.2f}s/prompt)")

    with open(os.path.join(out_dir, "requests.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    summary = summarize(rows, wall, run_id)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Per-cell one-line summary. None-safe so baseline cells (drafter='none', no spec
    # activity ⇒ accept_length_overall / alpha_* all None) don't crash on f-string format.
    # We print BOTH the ratio-of-sums headline (alpha_per_token) and the mean-of-ratios
    # diagnostic (alpha_per_token_mor) so the gap is visible while a sweep runs.
    def _f(x, fmt=".3f"):
        return format(x, fmt) if isinstance(x, (int, float)) else "  n/a"

    al_mean = (summary['accept_length'] or {}).get('mean')
    print(f"  accept_length(mor_mean)={_f(al_mean)}  "
          f"accept_length_overall={_f(summary['accept_length_overall'])}  "
          f"α_per_token={_f(summary['alpha_per_token'])} "
          f"(mor={_f(summary['alpha_per_token_mor'])})  "
          f"α_norm={_f(summary['alpha_normalized'])}  "
          f"tree_rate={_f(summary['accept_rate_overall'])}  "
          f"sanity={summary['sanity']['n_insane']}/{summary['n']} insane")
    return "ran"


# ─── One-drafter driver ────────────────────────────────────────────────────

def run_drafter(drafter_name, args, target_override=None):
    """Boot SGLang for one drafter, iterate (ds, lang, seed) cells."""
    drafter_entry, target_name, target_entry, registry = resolve_drafter(
        drafter_name, target_override=target_override)

    # Pre-filter combos: which (dataset, lang) pairs are eligible for this drafter.
    datasets_filter = [d.strip() for d in args.datasets.split(",")] if args.datasets else None
    langs_filter    = [l.strip() for l in args.langs.split(",")] if args.langs else None
    combos = combos_for(drafter_entry, target_name, datasets_filter, langs_filter)
    seeds  = [int(s) for s in args.seeds.split(",") if s.strip()]
    print(f"[plan] drafter={drafter_name} target={target_name}  "
          f"{len(combos)} combos × {len(seeds)} seeds = {len(combos)*len(seeds)} cells")

    # Decide if all cells are already done — skip server boot entirely.
    todo = []
    for seed in seeds:
        for ds, lang in combos:
            rid = run_id_for(target_name, drafter_name, ds, lang, args.n, seed)
            out_dir = os.path.join(RESULTS_ROOT, rid)
            if not cell_done(out_dir) or args.force:
                todo.append((seed, ds, lang))
    if not todo:
        print(f"[plan] all {len(combos)*len(seeds)} cells already done; nothing to run")
        return

    # Vocab guard before booting SGLang.
    if drafter_entry is not None:
        drafter_dir = os.path.abspath(os.path.join(REPO, drafter_entry["path"]))
        vocab_guard(drafter_dir, target_entry, target_name)
    else:
        drafter_dir = None

    # Boot SGLang once and reuse across cells.
    port = args.port or find_free_port()
    log_dir = os.path.join(RESULTS_ROOT, "_server_logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(
        log_dir,
        f"sglang_{drafter_name}_{_dt.datetime.utcnow().strftime('%Y%m%d-%H%M%S')}.log",
    )
    proc, log_f = launch_sglang(
        target_hf=target_entry["hf_name"],
        drafter_dir=drafter_dir,
        port=port,
        log_path=log_path,
        target_extra_env=target_entry.get("extra_env") or {},
    )

    try:
        if not wait_for_server(port, timeout_s=300):
            print(f"[server] FAILED to come up; tail of {log_path}:", file=sys.stderr)
            with open(log_path) as f:
                tail = f.readlines()[-40:]
            sys.stderr.writelines(tail)
            raise SystemExit(1)

        # HTTP client. We hit /generate directly (which returns full meta_info
        # including spec_verify_ct etc.). Chat template is applied client-side
        # via the target's HF tokenizer — no sglang import needed in this process.
        from transformers import AutoTokenizer
        os.environ.setdefault("HF_HOME", HF_HOME)
        # Suppress benign offline-mode warnings during tokenizer init
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")
        tokenizer = AutoTokenizer.from_pretrained(target_entry["hf_name"])
        client = {
            "host":      "127.0.0.1",
            "port":      port,
            "tokenizer": tokenizer,
        }

        # Iterate combos × seeds.
        for seed in seeds:
            for ds, lang in combos:
                run_cell(client, drafter_name, drafter_entry, target_name,
                         target_entry, registry, ds, lang, seed, args)
    finally:
        stop_server(proc, log_f)


# ─── Audit subcommand ──────────────────────────────────────────────────────

def cmd_audit(args):
    reg = load_registry()
    ACTIVE = {"ready", "partial", None}
    print(f"{'name':<42s} {'status':<8s} {'target':<8s} {'on_disk':<7s} {'vocab':>7s} {'epoch':<10s} | issues")
    print("-" * 100)
    n_ok = n_bad = n_skip = n_pending = 0
    for name, d in sorted(reg["drafters"].items(), key=lambda kv: kv[0]):
        status = d.get("status", "ready")
        path = os.path.join(REPO, d["path"])
        target = d["target"]
        cfg_p = os.path.join(path, "config.json")
        sft_p = os.path.join(path, "model.safetensors")
        on_disk = os.path.isfile(cfg_p) and os.path.isfile(sft_p)
        vocab = "?"
        issues = ""
        if on_disk:
            with open(cfg_p) as f:
                vocab = json.load(f).get("vocab_size", "?")
            want = reg["targets"][target]["vocab_size_check"]
            if vocab != want:
                issues = f"vocab {vocab} != {want}"
        else:
            if status not in ("pending", "partial"):
                issues = "missing on disk"
        if issues and status in ACTIVE:
            n_bad += 1
        elif status == "skip":
            n_skip += 1
        elif status == "pending":
            n_pending += 1
        else:
            n_ok += 1
        epoch_in_path = os.path.basename(path)
        print(f"{name:<42s} {status:<8s} {target:<8s} {str(on_disk):<7s} {str(vocab):>7s} "
              f"{epoch_in_path:<10s} | {issues}")
    print()
    print(f"summary: {n_ok} ready/partial, {n_pending} pending, {n_skip} skip "
          f"({n_bad} broken among active)")
    sys.exit(1 if n_bad else 0)


# ─── Sweep subcommand ──────────────────────────────────────────────────────

def cmd_sweep(args):
    """Iterate over all status:ready|partial drafters in drafters.yaml, plus
    one no-drafter baseline per target. Same args.seeds, args.n applied."""
    reg = load_registry()
    ACTIVE = {"ready", "partial", None}
    drafters = [n for n, d in reg["drafters"].items() if d.get("status") in ACTIVE]
    print(f"[sweep] {len(drafters)} drafter(s) at n={args.n}, seeds={args.seeds}")
    for name in drafters:
        print(f"\n=== {name} ===")
        try:
            run_drafter(name, args)
        except Exception as e:
            print(f"[sweep] {name} FAILED: {type(e).__name__}: {e}", file=sys.stderr)
    # Baselines (drafter='none') per target so report.py can compute speedup.
    for tgt in sorted({d["target"] for d in reg["drafters"].values()}):
        print(f"\n=== baseline: target={tgt} drafter=none ===")
        try:
            run_drafter("none", args, target_override=tgt)
        except Exception as e:
            print(f"[sweep] none/{tgt} FAILED: {type(e).__name__}: {e}", file=sys.stderr)


# ─── Entry point ───────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drafter", help="drafter name from drafters.yaml, or 'none' for baseline")
    ap.add_argument("--target", help="target override (only used when --drafter=none)")
    ap.add_argument("--datasets", help="comma list; defaults to all in coverage")
    ap.add_argument("--langs", help="comma list; defaults to all in coverage")
    ap.add_argument("--n", type=int, default=50,
                    help="prompts per (dataset, lang) cell. Default 50 — gives ~5%% bootstrap "
                         "CI per cell on accept_length, ~3%% when post-hoc aggregated across "
                         "datasets for a per-(drafter, lang) headline.")
    ap.add_argument("--seeds", default="42", help="comma-separated seeds (default 42)")
    ap.add_argument("--port", type=int, default=0, help="SGLang port (0 = auto)")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--force", action="store_true",
                    help="re-run even if summary.json already exists")
    ap.add_argument("--audit", action="store_true",
                    help="print drafter status table and exit")
    ap.add_argument("--sweep", action="store_true",
                    help="iterate over all status:ready drafters + baselines")
    args = ap.parse_args()

    if args.audit:
        cmd_audit(args)
        return
    if args.sweep:
        cmd_sweep(args)
        return
    if not args.drafter:
        ap.error("either --drafter, --audit, or --sweep is required")
    run_drafter(args.drafter, args, target_override=args.target)


if __name__ == "__main__":
    main()

