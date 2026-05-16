"""ARC-Challenge — 1,172 grade-school science MCQ. Canonical reasoning eval."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset

from spec_eval.registry import BENCHMARKS
from spec_eval.tasks.base import Benchmarker

_TEMPLATE = """\
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of the choices below. Think step by step before answering.

{question}

{choices_block}"""


def _format(row: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """ARC choices come as {'label': ['A','B','C','D'], 'text': [...]}.
    We normalise weird label sets (some rows use '1','2','3','4') to A-D."""
    choices = row.get("choices") or {}
    labels = list(choices.get("label", []))
    texts = list(choices.get("text", []))
    if not labels or not texts or len(labels) != len(texts):
        return None, None

    norm_labels: List[str] = []
    for i, lab in enumerate(labels):
        if isinstance(lab, str) and lab.upper() in ("A", "B", "C", "D", "E"):
            norm_labels.append(lab.upper())
        else:
            # numeric label → map to A..E
            norm_labels.append(chr(65 + i))

    block = "\n".join(f"{nl}) {t}" for nl, t in zip(norm_labels, texts))
    answer = row.get("answerKey")
    if answer in labels:
        ans_letter = norm_labels[labels.index(answer)]
    elif isinstance(answer, str) and answer.upper() in norm_labels:
        ans_letter = answer.upper()
    else:
        return None, None

    prompt = _TEMPLATE.format(question=row["question"].strip(), choices_block=block)
    return prompt, ans_letter


@BENCHMARKS.register("arc_challenge")
class ARCChallengeBenchmarker(Benchmarker):
    """``allenai/ai2_arc`` :: ``ARC-Challenge``. Single-letter MCQ answer."""

    def __init__(self, num_samples: Optional[int] = None, subset=None, seed: int = 0):
        super().__init__(num_samples, subset, seed=seed)

    def load_data(self) -> Tuple[List[Dict[str, Any]], List[str]]:
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge")["test"]
        questions: List[Dict[str, Any]] = []
        labels: List[str] = []
        for i, row in enumerate(ds):
            if self.num_samples is not None and i >= self.num_samples:
                break
            q, ans = _format(row)
            if q is None:
                continue
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
