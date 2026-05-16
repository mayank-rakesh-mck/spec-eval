"""Importing this package populates the BENCHMARKS registry as a side effect."""

from spec_eval.tasks import (
    aime,  # noqa: F401
    arc_challenge,  # noqa: F401
    ceval,  # noqa: F401
    financeqa,  # noqa: F401
    gpqa,  # noqa: F401
    gsm8k,  # noqa: F401
    hellaswag,  # noqa: F401
    humaneval,  # noqa: F401
    livecodebench,  # noqa: F401
    math500,  # noqa: F401
    mmlu,  # noqa: F401
    mmstar,  # noqa: F401
    mtbench,  # noqa: F401
    simpleqa,  # noqa: F401
)
from spec_eval.tasks.base import Benchmarker

__all__ = ["Benchmarker"]
