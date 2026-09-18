"""Small process-local rate limiter for development/single-instance deployments.

For multiple production instances, replace this with Redis/API-gateway rate
limiting so limits are shared across instances.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from threading import Lock

from fastapi import HTTPException, Request, status

_events: dict[str, deque[float]] = defaultdict(deque)
_lock = Lock()


def enforce_rate_limit(request: Request, limit: int, window_seconds: int, bucket: str) -> None:
    client = request.client.host if request.client else "unknown"
    key = f"{bucket}:{client}"
    now = time.monotonic()
    cutoff = now - window_seconds
    with _lock:
        events = _events[key]
        while events and events[0] <= cutoff:
            events.popleft()
        if len(events) >= limit:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many requests. Please try again later.",
            )
        events.append(now)
