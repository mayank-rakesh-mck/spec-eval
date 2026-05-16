"""SimpleQA — short factual QA from OpenAI. No automatic grader (the official
release ships an LLM-judge); we report spec-decode metrics + leave grading to
an external script.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset

from spec_eval.registry import BENCHMARKS
from spec_eval.tasks.base import Benchmarker


@BENCHMARKS.register("simpleqa")
class SimpleQABenchmarker(Benchmarker):
    SHOW_ACCURACY = False  # SimpleQA's official scorer is LLM-judge based

    def __init__(self, num_samples: Optional[int] = None, subset=None):
        super().__init__(num_samples, subset)

    def load_data(self) -> Tuple[List[Dict[str, Any]], List[Optional[str]]]:
        ds = load_dataset("basicv8vc/SimpleQA")["test"]
        questions: List[Dict[str, Any]] = []
        labels: List[Optional[str]] = []
        for i, q in enumerate(ds):
            if self.num_samples is not None and i >= self.num_samples:
                break
            questions.append({"question": q["problem"].strip()})
            # Hold onto the gold for downstream judging; we don't grade here.
            labels.append(str(q.get("answer", "")).strip() or None)
        return questions, labels

    def build_messages(self, question: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [{"role": "user", "content": question["question"]}]

    def get_max_new_tokens(self) -> int:
        return 512
