"""Detect EAGLE algorithm version (EAGLE-2 vs EAGLE-3) from a draft model.

Strategy, in order:
  1. Explicit override (caller passed ``algorithm="EAGLE"|"EAGLE3"``).
  2. Read the draft's ``config.json`` (local path or HF hub) and look for v3
     fingerprints — ``architectures`` ending in ``Eagle3``, an
     ``eagle_config.use_aux_hidden_state`` flag, or the literal ``"eagle3"``
     in the model_type.
  3. Filename heuristic — anything matching ``/eagle[-_]?3\\b/i``.
  4. Fall back to EAGLE-2.

Per-algorithm spec hyperparam defaults match the SGLang / SpecForge defaults:
  EAGLE  : num_steps=5, eagle_topk=8, draft_tokens=64   (verify-heavy tree)
  EAGLE3 : num_steps=3, eagle_topk=1, draft_tokens=4    (linear-chain, faster)
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AlgoDefaults:
    algorithm: str
    num_steps: int
    eagle_topk: int
    draft_tokens: int


EAGLE2_DEFAULTS = AlgoDefaults("EAGLE", num_steps=5, eagle_topk=8, draft_tokens=64)
EAGLE3_DEFAULTS = AlgoDefaults("EAGLE3", num_steps=3, eagle_topk=1, draft_tokens=4)


def defaults_for(algorithm: str) -> AlgoDefaults:
    return EAGLE3_DEFAULTS if algorithm.upper() == "EAGLE3" else EAGLE2_DEFAULTS


_V3_RE = re.compile(r"eagle[-_]?3\b", re.IGNORECASE)


def _read_local_config(draft_path: str) -> Optional[dict]:
    cfg_path = os.path.join(draft_path, "config.json")
    if os.path.isfile(cfg_path):
        try:
            with open(cfg_path) as f:
                return json.load(f)
        except Exception as e:  # noqa: BLE001
            logger.debug("failed to read %s: %s", cfg_path, e)
    return None


def _read_hub_config(draft_id: str) -> Optional[dict]:
    """Hit HF Hub for ``config.json`` without pulling weights."""
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=draft_id, filename="config.json")
        with open(path) as f:
            return json.load(f)
    except Exception as e:  # noqa: BLE001
        logger.debug("hub config lookup for %s failed: %s", draft_id, e)
        return None


def _config_says_eagle3(cfg: dict) -> bool:
    archs = cfg.get("architectures") or []
    if any(("eagle3" in a.lower()) for a in archs):
        return True
    mt = (cfg.get("model_type") or "").lower()
    if "eagle3" in mt:
        return True
    ec = cfg.get("eagle_config") or {}
    if isinstance(ec, dict) and ec.get("use_aux_hidden_state"):
        return True
    # Some EAGLE-3 configs declare aux_hidden_state at the top level.
    if cfg.get("use_aux_hidden_state"):
        return True
    return False


def detect_algorithm(draft: Optional[str], override: Optional[str] = None) -> str:
    """Return ``"EAGLE"`` or ``"EAGLE3"``. Defaults to ``"EAGLE"`` when unknown."""
    if not draft:
        return "EAGLE"
    if override:
        out = override.upper()
        if out not in ("EAGLE", "EAGLE3"):
            raise ValueError(f"unknown algorithm {override!r}; want EAGLE or EAGLE3")
        return out

    # 1. local config.json
    if os.path.isdir(draft):
        cfg = _read_local_config(draft)
        if cfg and _config_says_eagle3(cfg):
            logger.info("auto-detect: %s → EAGLE3 (local config)", draft)
            return "EAGLE3"

    # 2. filename heuristic (works for both local paths and HF ids)
    if _V3_RE.search(draft):
        logger.info("auto-detect: %s → EAGLE3 (name heuristic)", draft)
        return "EAGLE3"

    # 3. HF hub config (only if it looks like an "org/name" id; cheap network call)
    if "/" in draft and not os.path.exists(draft):
        cfg = _read_hub_config(draft)
        if cfg and _config_says_eagle3(cfg):
            logger.info("auto-detect: %s → EAGLE3 (hub config)", draft)
            return "EAGLE3"

    logger.info("auto-detect: %s → EAGLE (default)", draft)
    return "EAGLE"
