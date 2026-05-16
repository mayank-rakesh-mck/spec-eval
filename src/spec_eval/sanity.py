"""Cheap rule-based response checks. No model required.

Five flags catch the most common SGLang/drafter failure modes:
  - too_short        : response < 5 chars
  - repetition       : a single char dominates (>85% of the response)
  - prompt_echo      : response is the user prompt verbatim
  - system_leak      : response starts with SYSTEM:/USER:/ASSISTANT: or quotes the system prompt
  - generation_empty : finish_reason == "error" or completion_tokens == 0

Lifted from the old eval.py and trimmed to what's actually load-bearing.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional


def sanity_flags(
    response: str,
    prompt_text: str,
    system_msg: Optional[str] = None,
    finish_reason: str = "",
    completion_tokens: int = 0,
) -> Dict[str, bool]:
    flags = {
        "too_short": False,
        "repetition": False,
        "prompt_echo": False,
        "system_leak": False,
        "generation_empty": False,
    }
    if not response or completion_tokens == 0 or finish_reason == "error":
        flags["generation_empty"] = True
    if not response or len(response) < 5:
        flags["too_short"] = True
        return flags

    counts: Dict[str, int] = {}
    for ch in response:
        counts[ch] = counts.get(ch, 0) + 1
    if max(counts.values()) / len(response) > 0.85:
        flags["repetition"] = True

    if prompt_text and len(prompt_text) >= 30:
        norm_p = re.sub(r"\s+", " ", prompt_text.strip())
        norm_r = re.sub(r"\s+", " ", response.strip())
        if norm_p in norm_r and len(norm_p) / max(len(norm_r), 1) > 0.9:
            flags["prompt_echo"] = True

    head = response.lstrip()[:64].upper()
    if head.startswith(("SYSTEM:", "USER:", "ASSISTANT:")):
        flags["system_leak"] = True
    if system_msg and len(system_msg) >= 30 and system_msg[:120] in response:
        flags["system_leak"] = True

    return flags


def insanity_count(flags: Dict[str, bool]) -> int:
    """``1`` if any flag is set, else ``0``."""
    return 1 if any(flags.values()) else 0


def aggregate_flags(rows) -> Dict[str, Any]:
    """Roll per-row ``sanity`` dicts up into a summary."""
    n = 0
    insane = 0
    totals: Dict[str, int] = {}
    for r in rows:
        s = r.get("sanity") if isinstance(r, dict) else None
        if not s:
            continue
        n += 1
        if any(s.values()):
            insane += 1
        for k, v in s.items():
            if v:
                totals[k] = totals.get(k, 0) + 1
    return {
        "n": n,
        "n_insane": insane,
        "insane_fraction": insane / max(n, 1),
        "flags_total": totals,
    }
