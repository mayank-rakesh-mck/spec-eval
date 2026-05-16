"""HTTP client for SGLang /generate.

Why /generate and not /v1/chat/completions? /generate returns the full
``meta_info`` payload that carries ``spec_verify_ct``, ``spec_accept_token_num``,
``completion_tokens`` — which is exactly what we need for accept-length / α.
The OpenAI-compat endpoint strips those fields.

Sync + async variants, both with retry/backoff on transient errors.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass
class GenerateResult:
    """Wrapper around the ``/generate`` response. ``meta_info`` carries every
    speculative-decoding + latency + cache field that SGLang exposes; we surface
    them as defensive @property accessors (return ``None`` / ``0`` when missing
    so older SGLang versions don't blow up the eval)."""

    text: str
    meta_info: Dict[str, Any]

    # ─── Token counts ─────────────────────────────────────────────────────

    @property
    def completion_tokens(self) -> int:
        return int(self.meta_info.get("completion_tokens", 0) or 0)

    @property
    def prompt_tokens(self) -> int:
        return int(self.meta_info.get("prompt_tokens", 0) or 0)

    @property
    def cached_tokens(self) -> int:
        """Radix-cache hits. Skews per-request throughput numbers if ignored."""
        return int(self.meta_info.get("cached_tokens", 0) or 0)

    # ─── Speculative-decoding metrics (PR #18332 + earlier) ───────────────

    @property
    def spec_verify_ct(self) -> int:
        return int(self.meta_info.get("spec_verify_ct", 0) or 0)

    @property
    def spec_accept_length(self) -> Optional[float]:
        """Server-side E[α] for this request (= completion / verify_ct)."""
        v = self.meta_info.get("spec_accept_length")
        return float(v) if isinstance(v, (int, float)) else None

    @property
    def spec_accept_rate(self) -> Optional[float]:
        v = self.meta_info.get("spec_accept_rate")
        return float(v) if isinstance(v, (int, float)) else None

    @property
    def spec_accept_token_num(self) -> int:
        return int(self.meta_info.get("spec_accept_token_num", 0) or 0)

    @property
    def spec_draft_token_num(self) -> int:
        return int(self.meta_info.get("spec_draft_token_num", 0) or 0)

    @property
    def spec_accept_histogram(self) -> Optional[List[int]]:
        """Distribution of #accepted-drafts-per-step. Example: ``[3, 6, 3, 2]``
        means 3 steps with 0 accepted, 6 with 1, 3 with 2, 2 with 3.
        Added to SGLang in PR #18332 (Feb 2026); older builds return None."""
        h = self.meta_info.get("spec_accept_histogram")
        if h is None:
            return None
        try:
            return [int(x) for x in h]
        except (TypeError, ValueError):
            return None

    # ─── Latency / timing (newer SGLang exposes these) ────────────────────

    @property
    def e2e_latency(self) -> Optional[float]:
        v = self.meta_info.get("e2e_latency")
        return float(v) if isinstance(v, (int, float)) else None

    @property
    def inference_time(self) -> Optional[float]:
        """Pure inference (excludes queue + network)."""
        v = self.meta_info.get("inference_time")
        return float(v) if isinstance(v, (int, float)) else None

    @property
    def queue_time(self) -> Optional[float]:
        v = self.meta_info.get("queue_time")
        return float(v) if isinstance(v, (int, float)) else None

    @property
    def decode_throughput(self) -> Optional[float]:
        """Server's per-request decode tok/s."""
        v = self.meta_info.get("decode_throughput")
        return float(v) if isinstance(v, (int, float)) else None

    @property
    def total_retractions(self) -> int:
        """Number of times this request got retracted under memory pressure."""
        return int(self.meta_info.get("total_retractions", 0) or 0)

    # ─── Misc ─────────────────────────────────────────────────────────────

    @property
    def finish_reason(self) -> str:
        fr = self.meta_info.get("finish_reason")
        if isinstance(fr, dict):
            return str(fr.get("type", ""))
        return str(fr or "")

    @property
    def request_id(self) -> Optional[str]:
        return self.meta_info.get("id")


def _build_sampling_params(
    max_new_tokens: int,
    temperature: float,
    stop: Optional[List[str]],
    top_p: Optional[float],
    repetition_penalty: Optional[float],
) -> Dict[str, Any]:
    sp: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
    }
    if stop:
        sp["stop"] = stop
    if top_p is not None:
        sp["top_p"] = top_p
    if repetition_penalty is not None:
        sp["repetition_penalty"] = repetition_penalty
    return sp


class SGLangClient:
    """Thin sync wrapper. Use this for sequential dispatch. For
    concurrency, see :class:`AsyncSGLangClient`."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 30000,
        timeout: float = 600.0,
        max_retries: int = 3,
        backoff_base: float = 0.5,
    ):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self._client = httpx.Client(
            base_url=self.base_url, timeout=timeout, trust_env=False
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def generate(
        self,
        prompt_text: str,
        max_new_tokens: int,
        temperature: float = 0.0,
        stop: Optional[List[str]] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
    ) -> GenerateResult:
        payload = {
            "text": prompt_text,
            "sampling_params": _build_sampling_params(
                max_new_tokens, temperature, stop, top_p, repetition_penalty
            ),
        }
        last_err: Optional[str] = None
        for attempt in range(self.max_retries):
            try:
                r = self._client.post("/generate", json=payload)
                if r.status_code == 200:
                    out = r.json()
                    return GenerateResult(
                        text=out.get("text", ""), meta_info=out.get("meta_info") or {}
                    )
                last_err = f"HTTP {r.status_code}: {r.text[:300]!r}"
            except (httpx.HTTPError, httpx.TransportError) as e:  # noqa: BLE001
                last_err = f"{type(e).__name__}: {e}"

            if attempt < self.max_retries - 1:
                delay = self.backoff_base * (2**attempt) + random.uniform(0, 0.2)
                logger.warning(
                    "generate retry %d/%d after %s (sleep %.2fs)",
                    attempt + 1, self.max_retries, last_err, delay,
                )
                time.sleep(delay)
        raise RuntimeError(f"/generate failed after {self.max_retries} attempts: {last_err}")

    def flush_cache(self) -> None:
        try:
            self._client.post("/flush_cache")
        except Exception as e:  # noqa: BLE001 — flush is best-effort
            logger.warning("flush_cache failed: %s", e)

    def server_info(self) -> Optional[Dict[str, Any]]:
        """Fetch ``/server_info``. Used to pull ``step_time_dict`` (SpecForge
        convention: 20th-percentile step time → speed estimate). Requires
        ``SGLANG_RECORD_STEP_TIME=1`` in the server's env."""
        try:
            r = self._client.get("/server_info")
            if r.status_code == 200:
                return r.json()
        except Exception as e:  # noqa: BLE001
            logger.warning("server_info failed: %s", e)
        return None


class AsyncSGLangClient:
    """Bounded-concurrency async client.

    SGLang batches server-side; client concurrency lets that batching kick in.
    Use ``concurrency=1`` (default) to preserve sequential semantics — useful
    when you want a clean per-prompt latency measurement.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 30000,
        concurrency: int = 1,
        timeout: float = 600.0,
        max_retries: int = 3,
        backoff_base: float = 0.5,
    ):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.concurrency = max(1, concurrency)
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self._client: Optional[httpx.AsyncClient] = None
        self._timeout = timeout
        self._sema: Optional[asyncio.Semaphore] = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            base_url=self.base_url, timeout=self._timeout, trust_env=False
        )
        self._sema = asyncio.Semaphore(self.concurrency)
        return self

    async def __aexit__(self, *exc):
        if self._client is not None:
            await self._client.aclose()

    async def generate(
        self,
        prompt_text: str,
        max_new_tokens: int,
        temperature: float = 0.0,
        stop: Optional[List[str]] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
    ) -> GenerateResult:
        assert self._client is not None and self._sema is not None
        payload = {
            "text": prompt_text,
            "sampling_params": _build_sampling_params(
                max_new_tokens, temperature, stop, top_p, repetition_penalty
            ),
        }
        last_err: Optional[str] = None
        async with self._sema:
            for attempt in range(self.max_retries):
                try:
                    r = await self._client.post("/generate", json=payload)
                    if r.status_code == 200:
                        out = r.json()
                        return GenerateResult(
                            text=out.get("text", ""),
                            meta_info=out.get("meta_info") or {},
                        )
                    last_err = f"HTTP {r.status_code}: {r.text[:300]!r}"
                except (httpx.HTTPError, httpx.TransportError) as e:  # noqa: BLE001
                    last_err = f"{type(e).__name__}: {e}"
                if attempt < self.max_retries - 1:
                    delay = self.backoff_base * (2**attempt) + random.uniform(0, 0.2)
                    logger.warning(
                        "generate retry %d/%d after %s",
                        attempt + 1, self.max_retries, last_err,
                    )
                    await asyncio.sleep(delay)
        raise RuntimeError(
            f"/generate failed after {self.max_retries} attempts: {last_err}"
        )

    async def flush_cache(self) -> None:
        assert self._client is not None
        try:
            await self._client.post("/flush_cache")
        except Exception as e:  # noqa: BLE001
            logger.warning("flush_cache failed: %s", e)

    async def server_info(self) -> Optional[Dict[str, Any]]:
        assert self._client is not None
        try:
            r = await self._client.get("/server_info")
            if r.status_code == 200:
                return r.json()
        except Exception as e:  # noqa: BLE001
            logger.warning("server_info failed: %s", e)
        return None


def extract_step_time_p20_ms(server_info: Optional[Dict[str, Any]], batch_size: int) -> Optional[float]:
    """Pull the 20th-percentile step time (ms) for a given batch size.

    Matches SpecForge's ``bench_speculative.py`` convention (np.percentile(..., 20))
    — 20% is more stable than the median against the initial slow steps right
    after server warmup.
    """
    if not server_info:
        return None
    states = server_info.get("internal_states")
    if not states:
        return None
    try:
        d = states[0].get("step_time_dict") or {}
    except (IndexError, AttributeError, TypeError):
        return None
    times = d.get(str(batch_size)) or d.get(batch_size)
    if not times:
        return None
    try:
        import numpy as np  # local import to avoid hard dep at client import time

        return float(np.percentile(times, 20)) * 1000.0  # → ms
    except Exception:  # noqa: BLE001
        return None
