"""MT-Bench — 80 chat questions × 2 turns. No automatic judge; we report
spec-decode metrics only. Use an external LLM-judge for quality scoring.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from spec_eval.registry import BENCHMARKS
from spec_eval.tasks.base import Benchmarker
from spec_eval.utils import download_and_cache, read_jsonl

_MTBENCH_URL = (
    "https://raw.githubusercontent.com/lm-sys/FastChat/"
    "main/fastchat/llm_judge/data/mt_bench/question.jsonl"
)

SYSTEM_PROMPT = (
    "You are a helpful, respectful and honest assistant. Always answer as helpfully "
    "as possible, while being safe. Your answers should not include any harmful, "
    "unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure "
    "that your responses are socially unbiased and positive in nature.\n\n"
    "If a question does not make any sense, or is not factually coherent, explain "
    "why instead of answering something not correct. If you don't know the answer "
    "to a question, please don't share false information."
)


@BENCHMARKS.register("mtbench")
class MTBenchBenchmarker(Benchmarker):
    SHOW_ACCURACY = False  # no automatic judge

    def __init__(self, num_samples: Optional[int] = None, subset=None):
        super().__init__(num_samples, subset)

    def load_data(self) -> Tuple[List[Dict[str, Any]], List[None]]:
        path = download_and_cache(_MTBENCH_URL, filename="mtbench.jsonl")
        rows = list(read_jsonl(path))
        if self.num_samples is not None:
            rows = rows[: self.num_samples]
        questions = [
            {"question_1": r["turns"][0], "question_2": r["turns"][1]} for r in rows
        ]
        return questions, [None] * len(questions)

    def build_messages(self, question: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question["question_1"]},
        ]

    def get_answer_keys(self) -> List[str]:
        return ["answer_1", "answer_2"]

    def get_max_new_tokens(self) -> int:
        return 1024
