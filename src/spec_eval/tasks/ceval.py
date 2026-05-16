"""C-Eval — Chinese MCQ across 52 subjects.

NOT in the English preset. Kept for parity with SpecForge; opt in explicitly
with ``--tasks ceval`` if you have a Chinese-capable target.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from datasets import concatenate_datasets, load_dataset

from spec_eval.registry import BENCHMARKS
from spec_eval.tasks.base import Benchmarker

_CEVAL_CONFIGS = [
    "accountant", "advanced_mathematics", "art_studies", "basic_medicine",
    "business_administration", "chinese_language_and_literature", "civil_servant",
    "clinical_medicine", "college_chemistry", "college_economics", "college_physics",
    "college_programming", "computer_architecture", "computer_network",
    "discrete_mathematics", "education_science", "electrical_engineer",
    "environmental_impact_assessment_engineer", "fire_engineer", "high_school_biology",
    "high_school_chemistry", "high_school_chinese", "high_school_geography",
    "high_school_history", "high_school_mathematics", "high_school_physics",
    "high_school_politics", "ideological_and_moral_cultivation", "law",
    "legal_professional", "logic", "mao_zedong_thought", "marxism",
    "metrology_engineer", "middle_school_biology", "middle_school_chemistry",
    "middle_school_geography", "middle_school_history", "middle_school_mathematics",
    "middle_school_physics", "middle_school_politics", "modern_chinese_history",
    "operating_system", "physician", "plant_protection", "probability_and_statistics",
    "professional_tour_guide", "sports_science", "tax_accountant",
    "teacher_qualification", "urban_and_rural_planner", "veterinary_medicine",
]


def _extract_letter(s: str) -> Optional[str]:
    s = s.strip().upper()
    m = re.search(r"\b([ABCD])\b", s)
    if m:
        return m.group(1)
    for pat in [r"\(([ABCD])\)", r"\[([ABCD])\]", r"答案[：:]\s*([ABCD])", r"ANSWER[：:]\s*([ABCD])"]:
        m = re.search(pat, s)
        if m:
            return m.group(1).upper()
    m = re.search(r"([ABCD])", s)
    return m.group(1) if m else None


def _format(question: str, options: List[str]) -> str:
    out = question + "\n\n选项：\n"
    for i, opt in enumerate(options):
        out += f"{chr(65 + i)}. {opt}\n"
    out += "\n请从A、B、C、D中选择一个答案。"
    return out


@BENCHMARKS.register("ceval")
class CEvalBenchmarker(Benchmarker):
    LANGUAGE = "zh"

    def __init__(
        self,
        num_samples: Optional[int] = None,
        subset: Optional[List[str]] = None,
    ):
        super().__init__(num_samples, subset)

    def load_data(self) -> Tuple[List[Dict[str, Any]], List[str]]:
        subsets = self.subset if self.subset else _CEVAL_CONFIGS
        if subsets == ["all"]:
            subsets = _CEVAL_CONFIGS

        loaded = []
        for cfg in subsets:
            if cfg not in _CEVAL_CONFIGS:
                raise ValueError(f"unknown C-Eval subset: {cfg}")
            try:
                loaded.append(load_dataset("ceval/ceval-exam", name=cfg, split="test"))
            except Exception as e:  # noqa: BLE001 — some splits may be unavailable
                print(f"[ceval] skipping {cfg}: {e}")
        if not loaded:
            return [], []
        ds = concatenate_datasets(loaded)

        questions: List[Dict[str, Any]] = []
        labels: List[str] = []
        for i, item in enumerate(ds):
            if self.num_samples is not None and i >= self.num_samples:
                break
            qtext = item.get("question") or item.get("inputs") or item.get("problem")
            if not qtext:
                continue
            options = [
                item.get("A", ""),
                item.get("B", ""),
                item.get("C", ""),
                item.get("D", ""),
            ]
            options = [o for o in options if o]
            if len(options) < 2:
                continue
            ans = (item.get("answer") or item.get("target") or "").strip().upper()
            if ans not in {"A", "B", "C", "D"}:
                continue
            questions.append({"question": _format(qtext, options)})
            labels.append(ans)
        return questions, labels

    def build_messages(self, question: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [{"role": "user", "content": question["question"]}]

    def get_max_new_tokens(self) -> int:
        return 256

    def extract_answer(self, output: str, label: Optional[Any] = None) -> Optional[str]:
        return _extract_letter(output)

    def compute_accuracy(
        self, predictions: List[Any], labels: List[Any]
    ) -> Optional[float]:
        if not labels:
            return None
        correct = sum(1 for p, l in zip(predictions, labels) if p == l)
        valid = sum(1 for p in predictions if p is not None)
        return correct / valid if valid else 0.0
