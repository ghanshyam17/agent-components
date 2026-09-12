"""Per-endpoint rate limiting: a token bucket (rps) + a concurrency semaphore.

Both are optional; an endpoint with neither limiter is uncapped.
"""
from __future__ import annotations

import asyncio
import time


class TokenBucket:
    """Async token bucket refilled at `rate` tokens/sec, capacity `capacity`.

    `acquire()` blocks until a token is available, then consumes one.
    """

    def __init__(self, rate: float, capacity: float | None = None):
        self.rate = rate
        self.capacity = capacity if capacity is not None else max(1.0, rate)
        self.tokens = self.capacity
        self._last = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        while True:
            async with self._lock:
                now = time.monotonic()
                self.tokens = min(self.capacity, self.tokens + (now - self._last) * self.rate)
                self._last = now
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
                # need to wait for (1 - tokens) / rate seconds
                wait = (1.0 - self.tokens) / self.rate
            await asyncio.sleep(wait)


class Limiter:
    """Combines an optional rps bucket and an optional concurrency semaphore.

    Usage::

        await limiter.acquire()
        try:
            ...
        finally:
            limiter.release()
    """

    def __init__(self, rps: float | None = None, max_concurrency: int | None = None):
        self.bucket = TokenBucket(rps, rps) if rps else None
        self.sem = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    async def acquire(self) -> None:
        if self.sem is not None:
            await self.sem.acquire()
        if self.bucket is not None:
            await self.bucket.acquire()

    def release(self) -> None:
        if self.sem is not None:
            self.sem.release()