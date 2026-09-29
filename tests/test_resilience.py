import asyncio

from mash_agent.graph.resilience import run_resilient


async def test_success_first_try() -> None:
    async def fn() -> int:
        return 7

    r = await run_resilient(fn, timeout_s=1, max_attempts=3, backoff_s=1)
    assert (r.ok, r.value, r.attempts, r.retry_errors) == (True, 7, 1, [])


async def test_retries_with_doubling_backoff_then_succeeds() -> None:
    calls = 0
    sleeps: list[float] = []

    async def fake_sleep(s: float) -> None:
        sleeps.append(s)

    async def fn() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise RuntimeError(f"boom {calls}")
        return "ok"

    r = await run_resilient(fn, timeout_s=1, max_attempts=3, backoff_s=1, sleep=fake_sleep)
    assert (r.ok, r.value, r.attempts) == (True, "ok", 3)
    assert sleeps == [1, 2]
    assert r.retry_errors == ["RuntimeError: boom 1", "RuntimeError: boom 2"]


async def test_exhausted_attempts_return_error_instead_of_raising() -> None:
    async def fn() -> None:
        raise ValueError("nope")

    async def no_sleep(s: float) -> None:
        return None

    r = await run_resilient(fn, timeout_s=1, max_attempts=2, backoff_s=1, sleep=no_sleep)
    assert (r.ok, r.value, r.attempts, r.error) == (False, None, 2, "ValueError: nope")
    assert r.retry_errors == ["ValueError: nope"]


async def test_timeout_is_enforced_per_attempt() -> None:
    async def fn() -> None:
        await asyncio.sleep(10)

    async def no_sleep(s: float) -> None:
        return None

    r = await run_resilient(fn, timeout_s=0.02, max_attempts=2, backoff_s=1, sleep=no_sleep)
    assert not r.ok and r.attempts == 2
    assert r.error == "timeout after 0.02s"
