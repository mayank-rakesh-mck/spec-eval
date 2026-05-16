"""Benchmarker ABC. Same shape as SpecForge's benchmarker/base.py, with three
differences:

1. We don't take an SGL ``@sgl.function`` — the subclass returns *messages*
   (or a raw prompt string) and the runner POSTs to /generate.
2. We don't hold ``num_threads``: client concurrency is configured on the
   ``AsyncSGLangClient`` instance, not the task.
3. Each request row carries a ``sanity`` dict for cheap rule-based QC.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, Union

from spec_eval.client import (
    AsyncSGLangClient,
    GenerateResult,
    SGLangClient,
)
from spec_eval.metrics import BenchmarkMetrics, compute_metrics
from spec_eval.sanity import sanity_flags

logger = logging.getLogger(__name__)


class Benchmarker(ABC):
    """Base class. Override at minimum ``load_data`` and ``build_messages``.

    Subclass contract
    -----------------
    load_data() -> (questions, labels)
        questions : list[dict]    — task-specific, must be JSON-serialisable
        labels    : list[Any]     — gold answer per question (None if not gradable)

    build_messages(question) -> list[dict]
        OpenAI-style chat messages. The runner applies the target's chat
        template via the HF tokenizer. If the first message carries
        ``{"raw": True}``, the runner sends the bare content to /generate
        without templating (few-shot GSM8K shape).

    Optional overrides
    ------------------
    extract_answer(output, label) -> answer
    compute_accuracy(predictions, labels) -> float | None
    get_answer_keys() -> list[str] | None      # multi-turn (MT-Bench)
    get_max_new_tokens() -> int
    get_stop() -> list[str] | None
    get_system_prompt() -> str | None          # for sanity / system-leak check
    """

    NAME: ClassVar[str] = ""
    SHOW_ACCURACY: ClassVar[bool] = True
    REQUIRES_VISION: ClassVar[bool] = False
    LANGUAGE: ClassVar[str] = "en"

    def __init__(
        self,
        num_samples: Optional[int] = None,
        subset: Optional[List[str]] = None,
        seed: int = 0,
    ):
        self.num_samples = num_samples
        self.subset = subset
        self.seed = seed
        self.questions: List[Dict[str, Any]] = []
        self.labels: List[Any] = []
        # Cell context (populated by run()); used by _finalize for α derivations.
        self._cell_num_steps: Optional[int] = None
        self._cell_batch_size: Optional[int] = None

    # ─── To be implemented ────────────────────────────────────────────────

    @abstractmethod
    def load_data(self) -> Tuple[List[Dict[str, Any]], List[Any]]:
        ...

    @abstractmethod
    def build_messages(self, question: Dict[str, Any]) -> List[Dict[str, Any]]:
        ...

    # ─── Optional hooks ───────────────────────────────────────────────────

    def extract_answer(self, output: str, label: Optional[Any] = None) -> Any:
        return output

    def compute_accuracy(
        self, predictions: List[Any], labels: List[Any]
    ) -> Optional[float]:
        return None

    def get_answer_keys(self) -> Optional[List[str]]:
        return None

    def get_max_new_tokens(self) -> int:
        return 2048

    def get_stop(self) -> Optional[List[str]]:
        return None

    def get_temperature(self) -> float:
        return 0.0

    def get_system_prompt(self) -> Optional[str]:
        return None

    # ─── Driver. Called by the runner. ────────────────────────────────────

    def run(
        self,
        client: Union[SGLangClient, AsyncSGLangClient],
        tokenizer,
        on_request_row: Optional[Callable[[Dict[str, Any]], None]] = None,
        num_steps: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> Tuple[BenchmarkMetrics, List[Dict[str, Any]]]:
        """Load → generate → score. Returns (metrics, per-request rows).

        Args:
            client: sync or async SGLang client.
            tokenizer: HF tokenizer for the target (applies chat template).
            on_request_row: optional per-row callback.
            num_steps: ``--speculative-num-steps`` of the running cell. Used
                to derive Leviathan-style α at aggregation time.
            batch_size: batch size of the running cell, used to look up
                ``step_time_dict[<bs>]`` from ``/server_info``.
        """
        logger.info("[%s] loading data", self.NAME)
        self.questions, self.labels = self.load_data()
        if not self.questions:
            logger.warning("[%s] no questions; skipping", self.NAME)
            return BenchmarkMetrics(), []

        self._cell_num_steps = num_steps
        self._cell_batch_size = batch_size

        if isinstance(client, AsyncSGLangClient):
            return asyncio.run(self._run_async(client, tokenizer, on_request_row))
        return self._run_sync(client, tokenizer, on_request_row)

    # ─── Sync path ────────────────────────────────────────────────────────

    def _run_sync(
        self,
        client: SGLangClient,
        tokenizer,
        on_request_row: Optional[Callable[[Dict[str, Any]], None]],
    ) -> Tuple[BenchmarkMetrics, List[Dict[str, Any]]]:
        max_new = self.get_max_new_tokens()
        stop = self.get_stop()
        temperature = self.get_temperature()
        answer_keys = self.get_answer_keys()
        is_multi_turn = bool(answer_keys and len(answer_keys) > 1)

        rows: List[Dict[str, Any]] = []
        predictions: List[Any] = []

        t0 = time.perf_counter()
        for i, q in enumerate(self.questions):
            try:
                if is_multi_turn:
                    row, pred = self._run_multi_turn_sync(
                        client, tokenizer, q, answer_keys, max_new, temperature, stop
                    )
                else:
                    row, pred = self._run_single_sync(
                        client, tokenizer, q, max_new, temperature, stop
                    )
            except Exception as e:  # noqa: BLE001
                logger.exception("[%s] prompt %d failed: %s", self.NAME, i, e)
                row = self._error_row(e)
                pred = None
            self._attach_meta(row, i, pred, q)
            rows.append(row)
            predictions.append(pred)
            if on_request_row is not None:
                on_request_row(row)
            if (i + 1) % 25 == 0 or (i + 1) == len(self.questions):
                logger.info("[%s] %d/%d", self.NAME, i + 1, len(self.questions))

        latency = time.perf_counter() - t0
        # Fetch step_time stats from /server_info (best-effort).
        step_time_ms: Optional[float] = None
        try:
            from spec_eval.client import extract_step_time_p20_ms

            step_time_ms = extract_step_time_p20_ms(
                client.server_info(), self._cell_batch_size or 1
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("server_info fetch failed: %s", e)
        return self._finalize(rows, predictions, latency, answer_keys, step_time_ms)

    # ─── Async path ───────────────────────────────────────────────────────

    async def _run_async(
        self,
        client: AsyncSGLangClient,
        tokenizer,
        on_request_row: Optional[Callable[[Dict[str, Any]], None]],
    ) -> Tuple[BenchmarkMetrics, List[Dict[str, Any]]]:
        max_new = self.get_max_new_tokens()
        stop = self.get_stop()
        temperature = self.get_temperature()
        answer_keys = self.get_answer_keys()
        is_multi_turn = bool(answer_keys and len(answer_keys) > 1)

        async def _one(i: int, q: Dict[str, Any]) -> Tuple[int, Dict[str, Any], Any]:
            try:
                if is_multi_turn:
                    row, pred = await self._run_multi_turn_async(
                        client, tokenizer, q, answer_keys, max_new, temperature, stop
                    )
                else:
                    row, pred = await self._run_single_async(
                        client, tokenizer, q, max_new, temperature, stop
                    )
            except Exception as e:  # noqa: BLE001
                logger.exception("[%s] prompt %d failed: %s", self.NAME, i, e)
                row = self._error_row(e)
                pred = None
            self._attach_meta(row, i, pred, q)
            if on_request_row is not None:
                on_request_row(row)
            return i, row, pred

        t0 = time.perf_counter()
        # Fetch /server_info INSIDE the same client context (it's the async
        # client's lifespan). We don't want to re-enter the context, so we
        # capture step_time after asyncio.gather completes but before exit.
        step_time_ms: Optional[float] = None
        async with client:
            tasks = [asyncio.create_task(_one(i, q)) for i, q in enumerate(self.questions)]
            results = await asyncio.gather(*tasks)
            try:
                from spec_eval.client import extract_step_time_p20_ms

                info = await client.server_info()
                step_time_ms = extract_step_time_p20_ms(
                    info, self._cell_batch_size or 1
                )
            except Exception as e:  # noqa: BLE001
                logger.debug("server_info fetch failed: %s", e)
        latency = time.perf_counter() - t0

        results.sort(key=lambda r: r[0])
        rows = [r[1] for r in results]
        predictions = [r[2] for r in results]
        logger.info("[%s] %d/%d done (concurrency=%d)",
                    self.NAME, len(rows), len(self.questions), client.concurrency)
        return self._finalize(rows, predictions, latency, answer_keys, step_time_ms)

    # ─── Inner steps ──────────────────────────────────────────────────────

    def _apply_template(self, tokenizer, messages: List[Dict[str, Any]]) -> str:
        if messages and messages[0].get("raw"):
            return messages[0]["content"]
        clean = [{"role": m["role"], "content": m["content"]} for m in messages]
        return tokenizer.apply_chat_template(
            clean, tokenize=False, add_generation_prompt=True
        )

    def _run_single_sync(self, client, tokenizer, q, max_new, temperature, stop):
        msgs = self.build_messages(q)
        prompt_text = self._apply_template(tokenizer, msgs)
        r = client.generate(prompt_text, max_new, temperature, stop)
        return self._row_from_result(r, prompt_text), self.extract_answer(r.text, None)

    async def _run_single_async(self, client, tokenizer, q, max_new, temperature, stop):
        msgs = self.build_messages(q)
        prompt_text = self._apply_template(tokenizer, msgs)
        r = await client.generate(prompt_text, max_new, temperature, stop)
        return self._row_from_result(r, prompt_text), self.extract_answer(r.text, None)

    def _run_multi_turn_sync(
        self, client, tokenizer, q, answer_keys, max_new, temperature, stop
    ):
        messages = self.build_messages(q)
        results: List[GenerateResult] = []
        for turn_idx, _key in enumerate(answer_keys):
            prompt_text = self._apply_template(tokenizer, messages)
            r = client.generate(prompt_text, max_new, temperature, stop)
            results.append(r)
            messages.append({"role": "assistant", "content": r.text})
            next_q_key = f"question_{turn_idx + 2}"
            if next_q_key in q:
                messages.append({"role": "user", "content": q[next_q_key]})
        return self._merge_multi_turn(results, answer_keys)

    async def _run_multi_turn_async(
        self, client, tokenizer, q, answer_keys, max_new, temperature, stop
    ):
        messages = self.build_messages(q)
        results: List[GenerateResult] = []
        for turn_idx, _key in enumerate(answer_keys):
            prompt_text = self._apply_template(tokenizer, messages)
            r = await client.generate(prompt_text, max_new, temperature, stop)
            results.append(r)
            messages.append({"role": "assistant", "content": r.text})
            next_q_key = f"question_{turn_idx + 2}"
            if next_q_key in q:
                messages.append({"role": "user", "content": q[next_q_key]})
        return self._merge_multi_turn(results, answer_keys)

    def _merge_multi_turn(
        self, results: List[GenerateResult], answer_keys: List[str]
    ) -> Tuple[Dict[str, Any], Any]:
        """Flatten an N-turn conversation into one row. Token/spec counts sum
        across turns; latency percentiles use the max so we don't double-count."""
        def _maxnone(xs):
            xs = [x for x in xs if isinstance(x, (int, float))]
            return max(xs) if xs else None

        primary = self.extract_answer(results[0].text, None)
        responses = {answer_keys[i]: results[i].text for i in range(len(results))}
        h_total = None
        hs = [r.spec_accept_histogram for r in results if r.spec_accept_histogram]
        if hs:
            from spec_eval.metrics import _sum_histograms  # avoid cycle

            h_total = _sum_histograms(hs)

        row: Dict[str, Any] = {
            "prompt_tokens": sum(r.prompt_tokens for r in results),
            "completion_tokens": sum(r.completion_tokens for r in results),
            "cached_tokens": sum(r.cached_tokens for r in results),
            "spec_verify_ct": sum(r.spec_verify_ct for r in results),
            "spec_accept_token_num": sum(r.spec_accept_token_num for r in results),
            "spec_draft_token_num": sum(r.spec_draft_token_num for r in results),
            "spec_accept_histogram": h_total,
            "spec_accept_length": None,  # per-request only; cell-level reagg.
            "spec_accept_rate": None,
            "e2e_latency": sum(r.e2e_latency or 0.0 for r in results) or None,
            "inference_time": sum(r.inference_time or 0.0 for r in results) or None,
            "queue_time": _maxnone([r.queue_time for r in results]),
            "decode_throughput": _maxnone([r.decode_throughput for r in results]),
            "total_retractions": sum(r.total_retractions for r in results),
            "response_text": "\n\n---\n\n".join(r.text for r in results),
            "per_turn_responses": responses,
            "finish_reason": results[-1].finish_reason,
        }
        return row, primary

    def _row_from_result(
        self, result: GenerateResult, prompt_text: str = ""
    ) -> Dict[str, Any]:
        """Capture every spec/latency/cache field SGLang exposes in ``meta_info``.

        Older SGLang builds won't carry the newer fields (``cached_tokens``,
        ``e2e_latency``, ``spec_accept_histogram``, etc.) — properties return
        ``None`` / ``0`` so the rest of the pipeline degrades gracefully.
        """
        return {
            # Token counts
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "cached_tokens": result.cached_tokens,
            # Speculative-decoding metrics
            "spec_verify_ct": result.spec_verify_ct,
            "spec_accept_length": result.spec_accept_length,
            "spec_accept_rate": result.spec_accept_rate,
            "spec_accept_token_num": result.spec_accept_token_num,
            "spec_draft_token_num": result.spec_draft_token_num,
            "spec_accept_histogram": result.spec_accept_histogram,
            # Latency / timing
            "e2e_latency": result.e2e_latency,
            "inference_time": result.inference_time,
            "queue_time": result.queue_time,
            "decode_throughput": result.decode_throughput,
            # Memory pressure
            "total_retractions": result.total_retractions,
            # Misc
            "request_id": result.request_id,
            "finish_reason": result.finish_reason,
            "response_text": result.text,
            "_prompt_text": prompt_text,
        }

    def _error_row(self, e: Exception) -> Dict[str, Any]:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_tokens": 0,
            "spec_verify_ct": 0,
            "spec_accept_token_num": 0,
            "spec_draft_token_num": 0,
            "spec_accept_histogram": None,
            "e2e_latency": None,
            "inference_time": None,
            "queue_time": None,
            "decode_throughput": None,
            "total_retractions": 0,
            "response_text": "",
            "finish_reason": "error",
            "error": f"{type(e).__name__}: {e}",
        }

    def _attach_meta(self, row: Dict[str, Any], idx: int, pred: Any, q: Dict[str, Any]) -> None:
        row["idx"] = idx
        row["label"] = self.labels[idx] if idx < len(self.labels) else None
        row["prediction"] = pred
        prompt_for_sanity = row.pop("_prompt_text", q.get("question", ""))
        row["sanity"] = sanity_flags(
            row.get("response_text", ""),
            prompt_for_sanity,
            self.get_system_prompt(),
            row.get("finish_reason", ""),
            row.get("completion_tokens", 0),
        )

    def _finalize(
        self,
        rows: List[Dict[str, Any]],
        predictions: List[Any],
        latency: float,
        answer_keys: Optional[List[str]],
        step_time_p20_ms: Optional[float] = None,
    ) -> Tuple[BenchmarkMetrics, List[Dict[str, Any]]]:
        metrics = compute_metrics(
            rows,
            latency,
            answer_keys=answer_keys,
            num_steps=self._cell_num_steps,
            step_time_p20_ms=step_time_p20_ms,
        )
        from spec_eval.sanity import aggregate_flags  # local import to avoid cycle

        metrics.sanity = aggregate_flags(rows)
        # Stamp the gen-config we actually used onto the metrics so a year
        # from now `metrics.json` still says how the cell was sampled.
        metrics.max_new_tokens = self.get_max_new_tokens()
        metrics.temperature = self.get_temperature()
        metrics.stop = self.get_stop()
        if self.labels and any(l is not None for l in self.labels):
            acc = self.compute_accuracy(predictions, self.labels)
            if acc is not None:
                metrics.accuracy = acc
                metrics.num_valid_predictions = sum(1 for p in predictions if p is not None)
        return metrics, rows
