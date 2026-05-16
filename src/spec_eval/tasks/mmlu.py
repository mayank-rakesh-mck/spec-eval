"""MMLU — 4-way multiple choice across 57 subjects. CoT then ``Answer: X``."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset

from spec_eval.registry import BENCHMARKS
from spec_eval.tasks.base import Benchmarker

_TEMPLATE = """\
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{question}

A) {a}
B) {b}
C) {c}
D) {d}"""


def _format(row: Dict[str, Any]) -> Tuple[str, str]:
    choices = row["choices"]
    q = _TEMPLATE.format(
        question=row["question"].strip(),
        a=choices[0].strip(),
        b=choices[1].strip(),
        c=choices[2].strip(),
        d=choices[3].strip(),
    )
    return q, ["A", "B", "C", "D"][row["answer"]]


@BENCHMARKS.register("mmlu")
class MMLUBenchmarker(Benchmarker):
    """``--tasks mmlu:50`` runs 50 across all subjects. Use
    ``mmlu:50:high_school_physics,abstract_algebra`` to pin subsets."""

    def __init__(
        self,
        num_samples: Optional[int] = None,
        subset: Optional[List[str]] = None,
    ):
        super().__init__(num_samples, subset or ["all"])

    def load_data(self) -> Tuple[List[Dict[str, Any]], List[str]]:
        questions, labels = [], []
        for sub in self.subset or ["all"]:
            ds = load_dataset("cais/mmlu", sub)["test"]
            for i, row in enumerate(ds):
                if self.num_samples is not None and i >= self.num_samples:
                    break
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
