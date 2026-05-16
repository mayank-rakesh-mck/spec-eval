"""HumanEval — Python code generation. exec-based pass@1.

Ported from SpecForge benchmarker/humaneval.py. The model is given the
prompt (function signature + docstring); we concatenate its completion and
run the embedded test in a SUBPROCESS with a timeout so a runaway loop in
generated code can't wedge the eval. Do NOT run this against a shared host
without a sandbox.
"""

from __future__ import annotations

import re
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset

from spec_eval.registry import BENCHMARKS
from spec_eval.tasks.base import Benchmarker

_EXEC_TIMEOUT_S = 10


def _extract_code(output: str) -> Optional[str]:
    """Pull Python out of ```...``` blocks or grab a bare def block."""
    m = re.search(r"```(?:python)?\n(.*?)```", output, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"(def\s+\w+\([^)]*\):.*?)(?=\n\ndef\s+|\Z)", output, re.DOTALL)
    if m:
        return m.group(1).strip()
    return output.strip() or None


def _check_correctness(code: str, test_code: str, entry_point: str) -> bool:
    """Run ``code + test_code + check(entry_point)`` in a subprocess."""
    program = code + "\n\n" + test_code + f"\n\ncheck({entry_point})\n"
    try:
        r = subprocess.run(
            [sys.executable, "-I", "-c", program],
            capture_output=True,
            timeout=_EXEC_TIMEOUT_S,
        )
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:  # noqa: BLE001 — any spawn failure counts as a fail
        return False


@BENCHMARKS.register("humaneval")
class HumanEvalBenchmarker(Benchmarker):
    """164 problems; exec-based pass@1."""

    def __init__(self, num_samples: Optional[int] = None, subset=None):
        super().__init__(num_samples, subset)

    def load_data(self) -> Tuple[List[Dict[str, Any]], List[Optional[Dict[str, str]]]]:
        ds = load_dataset("openai/openai_humaneval")["test"]
        questions: List[Dict[str, Any]] = []
        labels: List[Optional[Dict[str, str]]] = []
        for i, q in enumerate(ds):
            if self.num_samples is not None and i >= self.num_samples:
                break
            questions.append({"question": q["prompt"]})
            labels.append(
                {
                    "prompt": q["prompt"],
                    "test": q.get("test", ""),
                    "entry_point": q.get("entry_point", ""),
                    "canonical_solution": q.get("canonical_solution", ""),
                }
            )
        return questions, labels

    def build_messages(self, question: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [{"role": "user", "content": question["question"]}]

    def get_max_new_tokens(self) -> int:
        return 1024

    def extract_answer(self, output: str, label: Optional[Any] = None) -> Optional[str]:
        return _extract_code(output)

    def compute_accuracy(
        self, predictions: List[Any], labels: List[Any]
    ) -> Optional[float]:
        if not labels or all(l is None for l in labels):
            return None
        correct = 0
        valid = 0
        for pred, label in zip(predictions, labels):
            if not label:
                continue
            valid += 1
            if pred is None:
                continue
            pred_str = str(pred).strip()
            entry_point = label.get("entry_point", "")
            prompt = label.get("prompt", "")

            # Reconstruct a runnable program. If pred already redefines the
            # entry_point, we trust it; otherwise we append it after the
            # prompt (which contains the function header).
            if pred_str.startswith("def "):
                m = re.match(r"def\s+(\w+)\s*\(", pred_str)
                if m and entry_point and m.group(1) == entry_point:
                    full_code = pred_str
                else:
                    full_code = prompt + "\n" + pred_str
            else:
                full_code = prompt + "\n" + pred_str

            if label.get("test") and _check_correctness(
                full_code, label["test"], entry_point
            ):
                correct += 1
        return correct / valid if valid else 0.0
