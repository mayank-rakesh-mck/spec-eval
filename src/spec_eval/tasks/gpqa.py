"""GPQA — graduate-level science MCQ. CoT then ``Answer: X``."""

from __future__ import annotations

import random
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
    gold = random.randint(0, 3)
    choices = [
        row["Incorrect Answer 1"],
        row["Incorrect Answer 2"],
        row["Incorrect Answer 3"],
    ]
    choices.insert(gold, row["Correct Answer"])
    q = _TEMPLATE.format(
        question=row["Question"].strip(),
        a=choices[0].strip(),
        b=choices[1].strip(),
        c=choices[2].strip(),
        d=choices[3].strip(),
    )
    return q, ["A", "B", "C", "D"][gold]


@BENCHMARKS.register("gpqa")
class GPQABenchmarker(Benchmarker):
    """The ``Idavidrein/gpqa`` dataset is gated on HF; ``huggingface-cli login``
    first. SpecForge uses the ``gpqa_main`` config."""

    def __init__(self, num_samples: Optional[int] = None, subset=None, seed: int = 0):
        super().__init__(num_samples, subset, seed)

    def load_data(self) -> Tuple[List[Dict[str, Any]], List[str]]:
        # Deterministic gold-letter shuffle per run.
        random.seed(0xC0FFEE)
        ds = load_dataset("Idavidrein/gpqa", "gpqa_main")["train"]
        questions, labels = [], []
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
        return 2048

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
