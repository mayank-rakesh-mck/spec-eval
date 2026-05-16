"""MATH-500 — 500 problems from the MATH dataset, gold in ``\\boxed{...}``."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset

from spec_eval.registry import BENCHMARKS
from spec_eval.tasks.base import Benchmarker


def _extract_math(output: str) -> Optional[str]:
    m = re.search(r"\\boxed\{([^}]+)\}", output)
    if m:
        return m.group(1).strip()
    m = re.search(r"\\boxed\s+([^\s]+)", output)
    if m:
        return m.group(1).strip()
    for pat in [
        r"(?:answer|Answer|ANSWER)[\s:]+([-+]?\d*\.?\d+)",
        r"(?:is|equals?|=\s*)([-+]?\d*\.?\d+)\s*$",
    ]:
        ms = re.findall(pat, output, re.IGNORECASE)
        if ms:
            return ms[-1].strip()
    nums = re.findall(r"[-+]?\d*\.?\d+", output)
    return nums[-1] if nums else None


@BENCHMARKS.register("math500")
class Math500Benchmarker(Benchmarker):
    def __init__(self, num_samples: Optional[int] = None, subset=None):
        super().__init__(num_samples, subset)

    def load_data(self) -> Tuple[List[Dict[str, Any]], List[Optional[str]]]:
        ds = load_dataset("HuggingFaceH4/MATH-500")["test"]
        questions, labels = [], []
        for i, q in enumerate(ds):
            if self.num_samples is not None and i >= self.num_samples:
                break
            questions.append({"question": q["problem"]})
            ans = q.get("answer")
            if ans is None and "solution" in q:
                ans = _extract_math(q["solution"])
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
        return 2048

    def extract_answer(self, output: str, label: Optional[Any] = None) -> Optional[str]:
        return _extract_math(output)

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
            p = str(pred).strip().lower()
            l = str(label).strip().lower()
            if p == l:
                correct += 1
                continue
            try:
                if abs(float(p) - float(l)) < 1e-6:
                    correct += 1
            except ValueError:
                pass
        return correct / valid if valid else 0.0
