"""Pre-flight guards. Fail loudly before booting SGLang so a 2-minute server
spin-up doesn't get spent on a config that's mathematically guaranteed to fail.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class GuardResult:
    ok: bool
    message: str


def _load_config(path_or_id: str) -> Optional[dict]:
    """Read config.json from a local dir, else fall back to HF hub."""
    if os.path.isdir(path_or_id):
        cfg_path = os.path.join(path_or_id, "config.json")
        if os.path.isfile(cfg_path):
            try:
                with open(cfg_path) as f:
                    return json.load(f)
            except Exception as e:  # noqa: BLE001
                logger.debug("local config read failed: %s", e)
                return None
        return None
    try:
        from huggingface_hub import hf_hub_download

        local = hf_hub_download(repo_id=path_or_id, filename="config.json")
        with open(local) as f:
            return json.load(f)
    except Exception as e:  # noqa: BLE001
        logger.debug("hub config lookup for %s failed: %s", path_or_id, e)
        return None


def vocab_guard(target: str, draft: Optional[str]) -> GuardResult:
    """Confirm ``draft.config.vocab_size == target.config.vocab_size``.

    EAGLE draft heads share the target's LM head. A mismatched vocab silently
    produces gibberish and a 0% acceptance rate — we'd rather refuse to boot.
    Falls back to OK with a warning if either config is unavailable.
    """
    if not draft:
        return GuardResult(True, "no draft → skipping vocab guard")
    t = _load_config(target)
    d = _load_config(draft)
    if not t or not d:
        return GuardResult(
            True,
            "could not read both configs; skipping vocab guard (will fail at server-boot if mismatched)",
        )
    tv = t.get("vocab_size")
    dv = d.get("vocab_size")
    if tv is None or dv is None:
        return GuardResult(True, f"vocab_size missing (target={tv}, draft={dv}); skipping")
    if tv != dv:
        return GuardResult(False, f"VOCAB MISMATCH: target={tv} vs draft={dv}")
    return GuardResult(True, f"vocab OK: {tv} (target == draft)")


def parse_config_tuple(spec: str) -> Tuple[int, int, int, int]:
    """Parse a SpecForge-style ``batch_size,num_steps,topk,draft_tokens`` tuple.

    Special case: ``"1,0,0,0"`` (or any ``...,0,0,0``) → baseline (no spec decoding).
    """
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != 4 or any(not p.isdigit() and not p.lstrip("-").isdigit() for p in parts):
        raise ValueError(
            f"config tuple must be 'batch_size,num_steps,topk,draft_tokens', got: {spec!r}"
        )
    bs, steps, topk, dt = (int(p) for p in parts)
    return bs, steps, topk, dt


def is_baseline_tuple(t: Tuple[int, int, int, int]) -> bool:
    """``num_steps=0`` ⇒ no spec decoding."""
    return t[1] == 0
