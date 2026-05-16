"""Benchmark registry. Mirrors SpecForge's benchmarker/registry.py."""

from __future__ import annotations

from typing import Dict, Type


class BenchmarkRegistry:
    """Decorator-based registry: ``@BENCHMARKS.register("gsm8k")`` on a Benchmarker subclass."""

    def __init__(self) -> None:
        self.benchmarks: Dict[str, Type] = {}

    def register(self, name: str):
        def wrapper(cls):
            if name in self.benchmarks and self.benchmarks[name] is not cls:
                raise ValueError(f"benchmark {name!r} already registered to {self.benchmarks[name]}")
            self.benchmarks[name] = cls
            cls.NAME = name  # convenience: keep the registry key on the class
            return cls

        return wrapper

    def get(self, name: str) -> Type:
        if name not in self.benchmarks:
            raise KeyError(
                f"unknown benchmark {name!r}. Known: {sorted(self.benchmarks)}"
            )
        return self.benchmarks[name]

    def names(self):
        return sorted(self.benchmarks)


BENCHMARKS = BenchmarkRegistry()
