"""HellaSwag — common-sense reasoning continuation MCQ.

10K validation rows; each row picks the most plausible 4-way continuation.
We grade on the validation split (test labels are hidden).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset

from spec_eval.registry import BENCHMARKS
from spec_eval.tasks.base import Benchmarker

_TEMPLATE = """\
Choose the most plausible continuation of the following passage. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is A, B, C, or D. Think step by step before answering.

Passage: {ctx}

A) {a}
B) {b}
C) {c}
D) {d}"""


def _format(row: Dict[str, Any]) -> Tuple[str, str]:
    ctx = (row.get("activity_label", "") + " " + row.get("ctx", "")).strip()
    endings = row["endings"]
    prompt = _TEMPLATE.format(
        ctx=ctx,
        a=endings[0].strip(),
        b=endings[1].strip(),
        c=endings[2].strip(),
        d=endings[3].strip(),
    )
    return prompt, ["A", "B", "C", "D"][int(row["label"])]


@BENCHMARKS.register("hellaswag")
class HellaSwagBenchmarker(Benchmarker):
    def __init__(self, num_samples: Optional[int] = None, subset=None, seed: int = 0):
        super().__init__(num_samples, subset, seed=seed)

    def load_data(self) -> Tuple[List[Dict[str, Any]], List[str]]:
        ds = load_dataset("Rowan/hellaswag")["validation"]
        questions: List[Dict[str, Any]] = []
        labels: List[str] = []
        for i, row in enumerate(ds):
            if self.num_samples is not None and i >= self.num_samples:
                break
            # HellaSwag's `label` is sometimes a string; coerce defensively.
            if row.get("label") in (None, ""):
                continue
            q, ans = _format(row)
            questions.append({"question": q})
            labels.append(ans)
        return questions, labels

    def build_messages(self, question: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [{"role": "user", "content": question["question"]}]

    def get_max_new_tokens(self) -> int:
        return 1024

    def extract_answer(self, output: str, label: Optional[Any] = None) -> Optional[str]:
        if "Answer: " not in output:
            return None
        return output.split("Answer: ", 1)[1].strip()[:1].upper()

    def compute_accuracy(
        self, predictions: List[Any], labels: List[Any]
    ) -> Optional[float]:
        if not labels:
            return None
        correct = sum(1 for p, l in zip(predictions, labels) if p == l)
        return correct / len(labels)
