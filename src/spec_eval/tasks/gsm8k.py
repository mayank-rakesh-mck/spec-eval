"""GSM8K — grade-school math. Few-shot CoT, regex-extract the final number.

Ported from SpecForge benchmarker/gsm8k.py. Same prompt format (raw few-shot,
no chat template) so accept-length numbers are comparable to SpecForge runs.
"""

from __future__ import annotations

import ast
import re
from typing import Any, Dict, List, Optional, Tuple

from spec_eval.registry import BENCHMARKS
from spec_eval.tasks.base import Benchmarker
from spec_eval.utils import download_and_cache, read_jsonl

INVALID = -9999999
_GSM8K_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/"
    "master/grade_school_math/data/test.jsonl"
)


def _one_example(line: Dict[str, Any], include_answer: bool) -> str:
    s = "Question: " + line["question"] + "\nAnswer:"
    if include_answer:
        s += " " + line["answer"]
    return s


def _few_shot_block(lines: List[Dict[str, Any]], k: int) -> str:
    return "".join(_one_example(lines[i], True) + "\n\n" for i in range(k))


def _answer_value(answer_str: str) -> int:
    s = answer_str.replace(",", "")
    nums = re.findall(r"-?\d+", s)
    if not nums:
        return INVALID
    try:
        return int(ast.literal_eval(nums[-1]))
    except (ValueError, SyntaxError):
        return INVALID


@BENCHMARKS.register("gsm8k")
class GSM8KBenchmarker(Benchmarker):
    """1.3K test problems; gold answers come from the `#### N` line."""

    def __init__(self, num_samples: Optional[int] = None, subset=None):
        super().__init__(num_samples, subset)
        self._few_shot: str = ""

    def load_data(self) -> Tuple[List[Dict[str, Any]], List[int]]:
        path = download_and_cache(_GSM8K_URL, filename="gsm8k_test.jsonl")
        lines = list(read_jsonl(path))
        self._few_shot = _few_shot_block(lines, 5)

        questions, labels = [], []
        for i, line in enumerate(lines):
            if self.num_samples is not None and i >= self.num_samples:
                break
            questions.append({"question": _one_example(line, include_answer=False)})
            labels.append(_answer_value(line["answer"]))
        assert all(l != INVALID for l in labels), "bad GSM8K labels"
        return questions, labels

    def build_messages(self, question: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Raw few-shot (no chat template) — matches SpecForge's create_few_shot_sgl_function.
        return [
            {
                "role": "user",
                "content": self._few_shot + question["question"],
                "raw": True,
            }
        ]

    def get_stop(self):
        return ["Question", "Assistant:", "<|separator|>"]

    def get_max_new_tokens(self) -> int:
        return 512

    def extract_answer(self, output: str, label: Optional[Any] = None) -> int:
        return _answer_value(output)

    def compute_accuracy(
        self, predictions: List[Any], labels: List[Any]
    ) -> Optional[float]:
        if not labels:
            return None
        correct = sum(1 for p, l in zip(predictions, labels) if p == l)
        return correct / len(labels)
