"""Prompt-building helpers used by the Benchmarker subclasses.

The SpecForge sgl functions decorate with ``@sgl.function`` and rely on
sglang-side chat templates; we apply the chat template client-side via the
target HF tokenizer so the eval code doesn't need ``import sglang``.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# ─── Cache dir used for raw .jsonl downloads (gsm8k, mtbench) ──────────────

_CACHE_DIR = Path(os.environ.get("SPEC_EVAL_CACHE", ".cache")) / "downloads"


def download_and_cache(url: str, filename: Optional[str] = None) -> str:
    """Download a file once and cache it on disk. Returns the local path."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    name = filename or os.path.basename(url)
    dest = _CACHE_DIR / name
    if not dest.is_file():
        urllib.request.urlretrieve(url, dest)
    return str(dest)


def read_jsonl(path: str):
    import json

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


# ─── Prompt builders. Each Benchmarker returns one of these. ───────────────


def simple_user_prompt(
    question: str,
    system_prompt: Optional[str] = None,
    user_prefix: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Single-turn chat: optional system + one user turn."""
    msgs: List[Dict[str, str]] = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    content = question if not user_prefix else question + user_prefix
    msgs.append({"role": "user", "content": content})
    return msgs


def multi_turn_prompt(
    turns: List[str],
    system_prompt: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Pre-fill an alternating user/assistant transcript ending at user turn 1.

    For MT-Bench we generate each turn separately because the assistant reply
    for turn N is the model's own output and only known at runtime. This
    helper just sets up the *initial* messages (system + first user). The
    runner appends the model reply then the next user turn before the second
    /generate call.
    """
    msgs: List[Dict[str, str]] = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    if turns:
        msgs.append({"role": "user", "content": turns[0]})
    return msgs


def few_shot_prompt(
    question: str,
    few_shot_block: str,
    stop: Optional[List[str]] = None,  # noqa: ARG001 — caller threads stop separately
) -> List[Dict[str, str]]:
    """GSM8K-style: few-shot examples concatenated with the new question.

    SpecForge uses a raw-completion prompt (no chat template) for the
    few-shot path. We mirror that by emitting a single "user" message whose
    content is ``few_shot_block + question``; the runner will call /generate
    with ``apply_chat_template=False`` for this shape.
    """
    return [
        {
            "role": "user",
            "content": few_shot_block + question,
            "raw": True,  # runner sentinel: don't apply chat template
        }
    ]
