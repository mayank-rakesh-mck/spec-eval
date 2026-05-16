"""MMStar — multimodal benchmark.

NOT in the English/text preset. Kept for parity with SpecForge but the
text-only ``/generate`` codepath we use can't drive a VLM properly — we'd
need to switch to ``/v1/chat/completions`` with image_url content blocks
(SGLang supports this for VLM targets). For now this raises a clear error
so it's obvious the scaffold isn't wired end-to-end.
"""

from __future__ import annotations

import os
import re
import shutil
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset

from spec_eval.registry import BENCHMARKS
from spec_eval.tasks.base import Benchmarker


def _extract_letter(output: str, options: Optional[List[str]] = None) -> Optional[str]:
    s = output.strip().upper()
    m = re.search(r"\b([A-Z])\b", s)
    if m:
        letter = m.group(1)
        if options:
            max_opt = chr(64 + len(options))
            if "A" <= letter <= max_opt:
                return letter
        elif "A" <= letter <= "D":
            return letter
    for pat in [r"\(([A-Z])\)", r"\[([A-Z])\]", r"ANSWER[：:]\s*([A-Z])"]:
        m = re.search(pat, s)
        if m:
            return m.group(1)
    return None


@BENCHMARKS.register("mmstar")
class MMStarBenchmarker(Benchmarker):
    REQUIRES_VISION = True
    LANGUAGE = "multi"

    def __init__(self, num_samples: Optional[int] = None, subset=None):
        super().__init__(num_samples, subset)
        self.cache_dir: Optional[str] = None
        self.options_list: List[List[str]] = []

    def load_data(self) -> Tuple[List[Dict[str, Any]], List[Optional[str]]]:
        self.cache_dir = os.path.join(".cache", "mmstar")
        os.makedirs(self.cache_dir, exist_ok=True)
        ds = load_dataset("Lin-Chen/MMStar")["val"]
        questions: List[Dict[str, Any]] = []
        labels: List[Optional[str]] = []
        for i, q in enumerate(ds):
            if self.num_samples is not None and i >= self.num_samples:
                break
            image = q["image"]
            image_path = os.path.join(self.cache_dir, q["meta_info"]["image_path"])
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            image.convert("RGB").save(image_path, "JPEG")

            qfull = q["question"]
            if "Options:" in qfull:
                qtext, opt_block = qfull.split("Options:", 1)
                opts = []
                for line in opt_block.strip().split("\n"):
                    line = line.strip()
                    if line and re.match(r"^[A-Z]\.", line):
                        opts.append(re.sub(r"^[A-Z]\.\s*", "", line).strip())
                self.options_list.append(opts)
                qtext = qtext.strip()
            else:
                qtext = qfull.strip()
                self.options_list.append([])

            questions.append({"image_path": image_path, "question": qtext})

            ans = (q.get("answer") or "").strip().upper()
            labels.append(ans if ans and len(ans) == 1 and "A" <= ans <= "Z" else None)
        return questions, labels

    def build_messages(self, question: Dict[str, Any]) -> List[Dict[str, Any]]:
        # The text-only /generate path can't carry an image; a VLM-aware
        # client (one that posts to /v1/chat/completions with image_url content)
        # would be needed. Surface a clear error so users don't get silent
        # garbage output.
        raise NotImplementedError(
            "MMStar requires a VLM-aware client (image content). This pipeline's "
            "/generate codepath is text-only — open an issue or wire a "
            "multimodal endpoint to use this benchmark."
        )

    def extract_answer(self, output: str, label: Optional[Any] = None) -> Optional[str]:
        return _extract_letter(output)

    def compute_accuracy(
        self, predictions: List[Any], labels: List[Any]
    ) -> Optional[float]:
        if not labels or all(l is None for l in labels):
            return None
        correct = sum(
            1 for p, l in zip(predictions, labels) if p and l and p == l
        )
        valid = sum(1 for l in labels if l is not None)
        return correct / valid if valid else 0.0

    def run(self, *args, **kwargs):
        try:
            return super().run(*args, **kwargs)
        finally:
            if self.cache_dir and os.path.exists(self.cache_dir):
                shutil.rmtree(self.cache_dir, ignore_errors=True)
