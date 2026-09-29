"""Shared HTTP client: per-tool rate limiting, retries with backoff on 429/5xx, disk cache."""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from mash_agent.cache import DiskCache
from mash_agent.ratelimit import RateLimiter

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class ToolError(RuntimeError):
    """Raised when an upstream API call fails after retries."""


class ApiClient:
    def __init__(
        self,
        base_url: str,
        limiter: RateLimiter,
        cache: DiskCache | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float = 20.0,
        max_retries: int = 4,
        backoff_base: float = 1.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._client = httpx.AsyncClient(base_url=base_url, transport=transport, timeout=timeout)
        self._limiter = limiter
        self._cache = cache
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._sleep = sleep
        self._base_url = base_url

    async def aclose(self) -> None:
        await self._client.aclose()

    async def get_json(self, path: str, params: dict[str, Any]) -> Any:
        return await self._get(path, params, as_text=False)

    async def get_text(self, path: str, params: dict[str, Any]) -> str:
        result: str = await self._get(path, params, as_text=True)
        return result

    async def _get(self, path: str, params: dict[str, Any], *, as_text: bool) -> Any:
        key = DiskCache.make_key(self._base_url, path, params, as_text)
        if self._cache is not None and (hit := self._cache.get(key)) is not None:
            return hit
        last: str = "no attempt made"
        for attempt in range(self._max_retries + 1):
            await self._limiter.acquire()
            try:
                resp = await self._client.get(path, params=params)
            except httpx.TransportError as exc:
                last = f"transport error: {exc!r}"
            else:
                if resp.status_code == 404:
                    return "" if as_text else {}
                if resp.status_code not in RETRYABLE_STATUS:
                    resp.raise_for_status()
                    value = resp.text if as_text else resp.json()
                    if self._cache is not None:
                        self._cache.set(key, value)
                    return value
                last = f"HTTP {resp.status_code}"
                retry_after = resp.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    await self._sleep(float(retry_after))
                    continue
            if attempt < self._max_retries:
                await self._sleep(self._backoff_base * 2**attempt)
        raise ToolError(f"GET {self._base_url}{path} failed after retries: {last}")
