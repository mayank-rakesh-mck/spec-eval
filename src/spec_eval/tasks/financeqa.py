"""FinanceQA — context-grounded financial QA. No automatic grader."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset

from spec_eval.registry import BENCHMARKS
from spec_eval.tasks.base import Benchmarker

_PROMPT = """\
Given the following context:

{context}

Can you answer the following question?

{question}"""


def _question_text(row: Dict[str, Any]) -> str:
    if row.get("context"):
        return _PROMPT.format(
            context=row["context"].strip(),
            question=row["question"].strip(),
        )
    return row["question"].strip()


@BENCHMARKS.register("financeqa")
class FinanceQABenchmarker(Benchmarker):
    SHOW_ACCURACY = False

    def __init__(self, num_samples: Optional[int] = None, subset=None):
        super().__init__(num_samples, subset)

    def load_data(self) -> Tuple[List[Dict[str, Any]], List[Optional[str]]]:
        ds = load_dataset("AfterQuery/FinanceQA")["test"]
        questions: List[Dict[str, Any]] = []
        labels: List[Optional[str]] = []
        for i, row in enumerate(ds):
            if self.num_samples is not None and i >= self.num_samples:
                break
            questions.append({"question": _question_text(row)})
            labels.append(row.get("answer") or None)
        return questions, labels

    def build_messages(self, question: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [{"role": "user", "content": question["question"]}]

    def get_max_new_tokens(self) -> int:
        return 1024
