"""LiveCodeBench — contamination-free coding benchmark.

The official ``livecodebench/code_generation`` dataset is gated; once you have
access, ``huggingface-cli login`` first. We do NOT run the LCB test harness
here — it requires installing the LCB grader separately. We just generate
completions and let you score downstream.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset

from spec_eval.registry import BENCHMARKS
from spec_eval.tasks.base import Benchmarker


@BENCHMARKS.register("livecodebench")
class LCBBenchmarker(Benchmarker):
    SHOW_ACCURACY = False

    def __init__(self, num_samples: Optional[int] = None, subset=None):
        super().__init__(num_samples, subset)

    def load_data(self) -> Tuple[List[Dict[str, Any]], List[None]]:
        ds = load_dataset("livecodebench/code_generation")["test"]
        questions: List[Dict[str, Any]] = []
        labels: List[None] = []
        for i, row in enumerate(ds):
            if self.num_samples is not None and i >= self.num_samples:
                break
            questions.append({"question": row["question_content"].strip()})
            labels.append(None)
        return questions, labels

    def build_messages(self, question: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [{"role": "user", "content": question["question"]}]

    def get_max_new_tokens(self) -> int:
        return 2048
