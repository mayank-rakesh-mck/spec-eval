"""AIME 2024 — 30 integer-answer competition problems. \\boxed{} or trailing int."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset

from spec_eval.registry import BENCHMARKS
from spec_eval.tasks.base import Benchmarker


def _extract_aime(output: str) -> Optional[str]:
    m = re.search(r"\\boxed\{([^}]+)\}", output)
    if m:
        ints = re.findall(r"\d+", m.group(1))
        if ints:
            return ints[-1]
        return m.group(1).strip()
    m = re.search(r"\\boxed\s+(\d+)", output)
    if m:
        return m.group(1)
    for pat in [
        r"(?:answer|Answer|ANSWER)[\s:]+(\d+)",
        r"(?:final\s+answer|Final\s+Answer)[\s:]+(\d+)",
        r"(?:is|equals?|=\s*)(\d+)\s*$",
    ]:
        ms = re.findall(pat, output, re.IGNORECASE)
        if ms:
            return ms[-1]
    nums = re.findall(r"\b(\d+)\b", output)
    valid = [n for n in nums if 0 <= int(n) <= 999]
    return valid[-1] if valid else None


@BENCHMARKS.register("aime")
class AIMEBenchmarker(Benchmarker):
    def __init__(self, num_samples: Optional[int] = None, subset=None, seed: int = 0):
        super().__init__(num_samples, subset, seed)

    def load_data(self) -> Tuple[List[Dict[str, Any]], List[Optional[str]]]:
        ds = load_dataset("Maxwell-Jia/AIME_2024")["train"]
        questions, labels = [], []
        for i, q in enumerate(ds):
            if self.num_samples is not None and i >= self.num_samples:
                break
            questions.append({"question": q["Problem"]})
            ans = q.get("Answer") or q.get("answer")
            labels.append(str(ans).strip() if ans is not None else None)
        return questions, labels

    def build_messages(self, question: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": question["question"]
                + "\n\nPlease reason step by step, and put your final answer within \\boxed{}.",
            }
        ]

    def get_max_new_tokens(self) -> int:
        # AIME needs room for long reasoning; SpecForge uses 32768. We default
        # to 8192 to keep eval wall-clock sane; bump via the CLI if needed.
        return 8192

    def extract_answer(self, output: str, label: Optional[Any] = None) -> Optional[str]:
        return _extract_aime(output)

    def compute_accuracy(
        self, predictions: List[Any], labels: List[Any]
    ) -> Optional[float]:
        if not labels or all(l is None for l in labels):
            return None
        correct = 0
        valid = 0
        for pred, label in zip(predictions, labels):
            if label is None:
                continue
            valid += 1
            if pred is None:
                continue
            if str(pred).strip() == str(label).strip():
                correct += 1
                continue
            try:
                if int(pred) == int(label):
                    correct += 1
            except (TypeError, ValueError):
                pass
        return correct / valid if valid else 0.0
