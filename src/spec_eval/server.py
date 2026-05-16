"""SGLang server lifecycle.

We boot SGLang as a subprocess and talk to it over HTTP. This file MUST NOT
``import sglang`` — that keeps the eval venv minimal. The subprocess uses
``sys.executable -m sglang.launch_server``, so whichever venv ``spec-eval``
runs in must also have sglang installed (``uv pip install sglang[all]``).
"""

from __future__ import annotations

import http.client
import logging
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class SpecConfig:
    """EAGLE-2 hyperparams. Defaults match the older eval.py."""

    algorithm: str = "EAGLE"  # use "EAGLE3" for v3 drafters
    num_steps: int = 5
    eagle_topk: int = 8
    draft_tokens: int = 64


@dataclass
class ServerConfig:
    target: str
    draft: Optional[str] = None  # None ⇒ baseline (no spec decoding)
    port: int = 0  # 0 ⇒ pick a free port
    host: str = "127.0.0.1"
    dtype: str = "bfloat16"
    mem_fraction_static: float = 0.85
    tp_size: int = 1
    context_length: Optional[int] = 4096
    trust_remote_code: bool = False
    attention_backend: Optional[str] = None  # e.g. "fa3"
    cuda_graph_max_bs: Optional[int] = None
    extra_env: Dict[str, str] = field(default_factory=dict)
    extra_args: List[str] = field(default_factory=list)
    spec: SpecConfig = field(default_factory=SpecConfig)


def find_free_port(start: int = 30000, span: int = 200) -> int:
    for port in range(start, start + span):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"no free port in [{start}, {start + span})")


def build_launch_cmd(cfg: ServerConfig) -> List[str]:
    cmd: List[str] = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        cfg.target,
        "--host",
        cfg.host,
        "--port",
        str(cfg.port),
        "--dtype",
        cfg.dtype,
        "--mem-fraction-static",
        str(cfg.mem_fraction_static),
        "--tp-size",
        str(cfg.tp_size),
    ]
    if cfg.context_length is not None:
        cmd += ["--context-length", str(cfg.context_length)]
    if cfg.trust_remote_code:
        cmd += ["--trust-remote-code"]
    if cfg.attention_backend:
        cmd += ["--attention-backend", cfg.attention_backend]
    if cfg.cuda_graph_max_bs is not None:
        cmd += ["--cuda-graph-max-bs", str(cfg.cuda_graph_max_bs)]
    if cfg.draft:
        cmd += [
            "--speculative-algorithm",
            cfg.spec.algorithm,
            "--speculative-draft-model-path",
            cfg.draft,
            "--speculative-num-steps",
            str(cfg.spec.num_steps),
            "--speculative-eagle-topk",
            str(cfg.spec.eagle_topk),
            "--speculative-num-draft-tokens",
            str(cfg.spec.draft_tokens),
        ]
    cmd += list(cfg.extra_args)
    return cmd


def launch(cfg: ServerConfig, log_path: str) -> Tuple[subprocess.Popen, "IO"]:
    """Spawn SGLang. Caller owns teardown via :func:`stop`."""
    if cfg.port == 0:
        cfg.port = find_free_port()

    env = os.environ.copy()
    # Localhost calls bypass any inherited http_proxy / squid setup.
    env.setdefault("no_proxy", "localhost,127.0.0.1,0.0.0.0")
    env.setdefault("NO_PROXY", "localhost,127.0.0.1,0.0.0.0")
    # Allow override of long context len without aborting (carried over from
    # the old eval.py — bites on llama3.1 with very-long context drafters).
    env.setdefault("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN", "1")
    env.update({k: str(v) for k, v in cfg.extra_env.items()})

    cmd = build_launch_cmd(cfg)
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    log_f = open(log_path, "w")
    logger.info("launching SGLang on port %s (log=%s)", cfg.port, log_path)
    logger.info("  cmd: %s", " ".join(cmd))
    proc = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
    return proc, log_f


def wait_for_server(host: str, port: int, timeout_s: int = 600) -> bool:
    """Poll /health until 200 OK or timeout. Uses http.client to bypass http_proxy."""
    deadline = time.time() + timeout_s
    started = time.time()
    last_err: Optional[str] = None
    while time.time() < deadline:
        try:
            conn = http.client.HTTPConnection(host, port, timeout=2)
            conn.request("GET", "/health")
            r = conn.getresponse()
            r.read()
            conn.close()
            if r.status == 200:
                logger.info("server ready in %ds", int(time.time() - started))
                return True
        except Exception as e:  # noqa: BLE001 — server isn't up yet, retry
            last_err = f"{type(e).__name__}: {e}"
        time.sleep(2)
    logger.error("server failed to come up in %ds (last err: %s)", timeout_s, last_err)
    return False


def stop(proc: subprocess.Popen, log_f) -> None:
    if proc.poll() is not None:
        log_f.close()
        return
    logger.info("terminating SGLang pid=%s", proc.pid)
    try:
        proc.terminate()
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
    finally:
        log_f.close()


def flush_cache(host: str, port: int) -> None:
    """POST /flush_cache between tasks so radix-cache doesn't leak prompts across cells."""
    try:
        conn = http.client.HTTPConnection(host, port, timeout=10)
        conn.request("POST", "/flush_cache")
        conn.getresponse().read()
        conn.close()
    except Exception as e:  # noqa: BLE001 — flush is best-effort
        logger.warning("flush_cache failed: %s", e)
