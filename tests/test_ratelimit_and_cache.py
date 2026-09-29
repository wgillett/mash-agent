import httpx
import pytest

from mash_agent.cache import DiskCache
from mash_agent.mcp_servers.client import ApiClient, ToolError
from mash_agent.ratelimit import RateLimiter


async def test_limiter_spaces_calls() -> None:
    now = 0.0
    sleeps: list[float] = []

    async def fake_sleep(s: float) -> None:
        sleeps.append(s)

    limiter = RateLimiter(2.0, clock=lambda: now, sleep=fake_sleep)
    for _ in range(3):
        await limiter.acquire()
    assert sleeps == pytest.approx([0.5, 1.0])


def test_limiter_rejects_nonpositive_rate() -> None:
    with pytest.raises(ValueError):
        RateLimiter(0)


def test_cache_roundtrip_and_ttl(tmp_path: pytest.TempPathFactory) -> None:
    cache = DiskCache(tmp_path, ttl_seconds=100)  # type: ignore[arg-type]
    key = DiskCache.make_key("a", {"b": 1})
    assert cache.get(key) is None
    cache.set(key, {"x": [1, 2]})
    assert cache.get(key) == {"x": [1, 2]}
    expired = DiskCache(tmp_path, ttl_seconds=-1)  # type: ignore[arg-type]
    assert expired.get(key) is None


async def _client(handler: httpx.MockTransport, sleeps: list[float]) -> ApiClient:
    async def fake_sleep(s: float) -> None:
        sleeps.append(s)

    return ApiClient(
        "https://example.test",
        RateLimiter(1000),
        transport=handler,
        max_retries=3,
        sleep=fake_sleep,
    )


async def test_retries_on_429_then_succeeds() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 3:
            return httpx.Response(429)
        return httpx.Response(200, json={"ok": True})

    sleeps: list[float] = []
    client = await _client(httpx.MockTransport(handler), sleeps)
    assert await client.get_json("/x", {}) == {"ok": True}
    assert calls == 3
    assert sleeps == [1.0, 2.0]


async def test_gives_up_after_retries() -> None:
    client = await _client(httpx.MockTransport(lambda r: httpx.Response(503)), [])
    with pytest.raises(ToolError):
        await client.get_json("/x", {})


async def test_respects_retry_after_header() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"Retry-After": "7"})
        return httpx.Response(200, json={})

    sleeps: list[float] = []
    client = await _client(httpx.MockTransport(handler), sleeps)
    await client.get_json("/x", {"a": 1})
    assert sleeps == [7.0]


async def test_cache_avoids_second_request(tmp_path: pytest.TempPathFactory) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"n": calls})

    client = ApiClient(
        "https://example.test",
        RateLimiter(1000),
        cache=DiskCache(tmp_path),  # type: ignore[arg-type]
        transport=httpx.MockTransport(handler),
    )
    assert await client.get_json("/x", {}) == {"n": 1}
    assert await client.get_json("/x", {}) == {"n": 1}
    assert calls == 1


async def test_concurrent_sets_same_key_leave_valid_entry(tmp_path: pytest.TempPathFactory) -> None:
    import asyncio

    cache = DiskCache(tmp_path)  # type: ignore[arg-type]
    key = DiskCache.make_key("k")
    await asyncio.gather(*(asyncio.to_thread(cache.set, key, {"i": i}) for i in range(50)))
    assert cache.get(key) in [{"i": i} for i in range(50)]
    assert [p.name for p in tmp_path.iterdir()] == [f"{key}.json"]  # type: ignore[attr-defined]


def test_expired_entry_is_deleted_on_read(tmp_path: pytest.TempPathFactory) -> None:
    DiskCache(tmp_path).set("k", 1)  # type: ignore[arg-type]
    assert DiskCache(tmp_path, ttl_seconds=-1).get("k") is None  # type: ignore[arg-type]
    assert list(tmp_path.iterdir()) == []  # type: ignore[attr-defined]
