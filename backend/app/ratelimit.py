from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from fastapi import status
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

if TYPE_CHECKING:
    from .config import Settings

ClockFn = Callable[[], float]
SleepFn = Callable[[float], None]

_MAX_TRACKED_BUCKETS = 4096
_PRUNE_EVERY_N_CHECKS = 256


@dataclass(frozen=True)
class RateLimitRule:
    """Sliding-window rate limit definition for an API route group."""

    name: str
    limit: int
    window_seconds: int


@dataclass(frozen=True)
class RateLimitDecision:
    """Outcome of a single rate-limit evaluation."""

    allowed: bool
    limit: int
    remaining: int
    reset_after_seconds: float
    retry_after_seconds: int
    rule_name: str


class SlidingWindowRateLimiter:
    """Simple thread-safe, in-memory sliding-window rate limiter keyed by (rule, client)."""

    def __init__(
        self,
        *,
        clock: ClockFn = time.monotonic,
        max_buckets: int = _MAX_TRACKED_BUCKETS,
    ) -> None:
        self._clock = clock
        self._max_buckets = max(16, int(max_buckets))
        self._lock = threading.Lock()
        self._buckets: dict[tuple[str, str], tuple[int, deque[float]]] = {}
        self._check_count = 0

    def check(self, rule: RateLimitRule, client_key: str) -> RateLimitDecision:
        if rule.limit <= 0 or rule.window_seconds <= 0:
            return RateLimitDecision(
                allowed=True,
                limit=0,
                remaining=0,
                reset_after_seconds=0.0,
                retry_after_seconds=0,
                rule_name=rule.name,
            )

        now = self._clock()
        cutoff = now - rule.window_seconds
        bucket_key = (rule.name, client_key)

        with self._lock:
            self._check_count += 1
            if (
                self._check_count % _PRUNE_EVERY_N_CHECKS == 0
                or len(self._buckets) > self._max_buckets
            ):
                self._prune_expired_locked(now)

            entry = self._buckets.get(bucket_key)
            if entry is None:
                timestamps: deque[float] = deque()
                self._buckets[bucket_key] = (rule.window_seconds, timestamps)
            else:
                _, timestamps = entry
                if entry[0] != rule.window_seconds:
                    self._buckets[bucket_key] = (rule.window_seconds, timestamps)

            while timestamps and timestamps[0] <= cutoff:
                timestamps.popleft()

            if len(timestamps) < rule.limit:
                timestamps.append(now)
                remaining = max(0, rule.limit - len(timestamps))
                reset_after = max(0.0, (timestamps[0] + rule.window_seconds) - now)
                return RateLimitDecision(
                    allowed=True,
                    limit=rule.limit,
                    remaining=remaining,
                    reset_after_seconds=reset_after,
                    retry_after_seconds=0,
                    rule_name=rule.name,
                )

            reset_after = max(0.0, (timestamps[0] + rule.window_seconds) - now)
            retry_after = max(1, math.ceil(reset_after))
            return RateLimitDecision(
                allowed=False,
                limit=rule.limit,
                remaining=0,
                reset_after_seconds=reset_after,
                retry_after_seconds=retry_after,
                rule_name=rule.name,
            )

    def reset(self) -> None:
        with self._lock:
            self._buckets.clear()
            self._check_count = 0

    def _prune_expired_locked(self, now: float) -> None:
        expired_keys: list[tuple[str, str]] = []
        for key, (window_seconds, timestamps) in self._buckets.items():
            cutoff = now - window_seconds
            while timestamps and timestamps[0] <= cutoff:
                timestamps.popleft()
            if not timestamps:
                expired_keys.append(key)
        for key in expired_keys:
            self._buckets.pop(key, None)

        # Hard safety cap to bound memory under spoofed-IP floods.
        if len(self._buckets) > self._max_buckets:
            overflow = len(self._buckets) - self._max_buckets
            oldest_keys = sorted(
                self._buckets.keys(),
                key=lambda k: self._buckets[k][1][-1] if self._buckets[k][1] else 0.0,
            )[:overflow]
            for key in oldest_keys:
                self._buckets.pop(key, None)


class OutboundRateLimiter:
    """Thread-safe sliding-window pacing limiter for outbound provider API calls."""

    def __init__(
        self,
        max_requests: int,
        window_seconds: float = 60.0,
        *,
        clock: ClockFn = time.monotonic,
    ) -> None:
        self.max_requests = max(0, int(max_requests))
        self.window_seconds = max(0.0, float(window_seconds))
        self._clock = clock
        self._lock = threading.Lock()
        self._timestamps: deque[float] = deque()
        self._simulated_advance = 0.0

    def acquire(
        self,
        *,
        sleep_fn: SleepFn = time.sleep,
        on_wait: Callable[[float], None] | None = None,
    ) -> float:
        """Wait if necessary until a request slot is available; return total seconds waited."""
        if self.max_requests <= 0 or self.window_seconds <= 0.0:
            return 0.0

        total_waited = 0.0
        while True:
            with self._lock:
                now = self._clock() + self._simulated_advance
                cutoff = now - self.window_seconds
                while self._timestamps and self._timestamps[0] <= cutoff:
                    self._timestamps.popleft()

                if len(self._timestamps) < self.max_requests:
                    self._timestamps.append(now)
                    return total_waited

                wait_seconds = max(0.0, (self._timestamps[0] + self.window_seconds) - now)

            if wait_seconds <= 0.0:
                continue

            if on_wait is not None:
                on_wait(wait_seconds)
            before_sleep = self._clock()
            sleep_fn(wait_seconds)
            after_sleep = self._clock()
            if wait_seconds >= 0.001 and (after_sleep - before_sleep) < 1e-6:
                with self._lock:
                    self._simulated_advance += wait_seconds
            total_waited += wait_seconds

    def reset(self) -> None:
        with self._lock:
            self._timestamps.clear()
            self._simulated_advance = 0.0


def _get_header(scope: Scope, header_name: bytes) -> str:
    for name, value in scope.get("headers") or ():
        if name.lower() == header_name:
            try:
                return value.decode("latin-1").strip()
            except UnicodeDecodeError:
                return ""
    return ""


def extract_client_ip(scope: Scope, *, trust_proxy_headers: bool = True) -> str:
    """Extract client IP from proxy headers (when trusted) or ASGI scope client."""
    if trust_proxy_headers:
        forwarded_for = _get_header(scope, b"x-forwarded-for")
        if forwarded_for:
            first_ip = forwarded_for.split(",")[0].strip()
            if first_ip:
                return first_ip
        real_ip = _get_header(scope, b"x-real-ip")
        if real_ip:
            return real_ip

    client = scope.get("client")
    if client and len(client) >= 1 and client[0]:
        return str(client[0]).strip()
    return "unknown"


def resolve_api_rate_limit_rule(
    method: str,
    path: str,
    settings: Settings,
) -> RateLimitRule | None:
    """Map an HTTP request to its rate-limit rule, or None for non-API / OPTIONS requests."""
    normalized_method = method.upper()
    if normalized_method == "OPTIONS":
        return None

    normalized_path = path.rstrip("/") or "/"
    if normalized_path != "/api" and not normalized_path.startswith("/api/"):
        return None

    window = settings.rate_limit_window_seconds

    if normalized_path == "/api/health":
        return RateLimitRule(
            name="health",
            limit=settings.rate_limit_health_per_window,
            window_seconds=window,
        )

    if normalized_path == "/api/auth/login" and normalized_method == "POST":
        return RateLimitRule(
            name="auth_login",
            limit=settings.rate_limit_auth_login_per_window,
            window_seconds=window,
        )

    if normalized_path == "/api/auth" or normalized_path.startswith("/api/auth/"):
        return RateLimitRule(
            name="auth",
            limit=settings.rate_limit_auth_per_window,
            window_seconds=window,
        )

    if normalized_path == "/api/jobs" and normalized_method == "POST":
        return RateLimitRule(
            name="jobs_create",
            limit=settings.rate_limit_jobs_create_per_window,
            window_seconds=window,
        )

    if (
        normalized_path == "/api/jobs"
        or normalized_path.startswith("/api/jobs/")
        or normalized_path == "/api/queue"
        or normalized_path.startswith("/api/queue/")
    ):
        if normalized_method in {"GET", "HEAD"}:
            return RateLimitRule(
                name="jobs_read",
                limit=settings.rate_limit_jobs_read_per_window,
                window_seconds=window,
            )
        return RateLimitRule(
            name="queue_control",
            limit=settings.rate_limit_queue_control_per_window,
            window_seconds=window,
        )

    return RateLimitRule(
        name="api_default",
        limit=settings.rate_limit_default_per_window,
        window_seconds=window,
    )


class RateLimitMiddleware:
    """Pure ASGI sliding-window rate limiter covering all /api/* HTTP routes."""

    def __init__(
        self,
        app: ASGIApp,
        settings: Settings,
        limiter: SlidingWindowRateLimiter | None = None,
        *,
        wall_clock: ClockFn = time.time,
    ) -> None:
        self.app = app
        self.settings = settings
        self.limiter = limiter or SlidingWindowRateLimiter()
        self._wall_clock = wall_clock

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not self.settings.rate_limit_enabled:
            await self.app(scope, receive, send)
            return

        method = str(scope.get("method", "GET"))
        path = str(scope.get("path", ""))
        rule = resolve_api_rate_limit_rule(method, path, self.settings)
        if rule is None or rule.limit <= 0:
            await self.app(scope, receive, send)
            return

        client_ip = extract_client_ip(
            scope,
            trust_proxy_headers=self.settings.rate_limit_trust_proxy_headers,
        )
        decision = self.limiter.check(rule, client_ip)
        reset_epoch = int(math.ceil(self._wall_clock() + decision.reset_after_seconds))

        rate_headers: list[tuple[bytes, bytes]] = [
            (b"x-ratelimit-limit", str(decision.limit).encode("ascii")),
            (b"x-ratelimit-remaining", str(decision.remaining).encode("ascii")),
            (b"x-ratelimit-reset", str(reset_epoch).encode("ascii")),
        ]

        if not decision.allowed:
            retry_after = str(decision.retry_after_seconds)
            response = JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": (
                        f"Rate limit exceeded. Please try again in "
                        f"{decision.retry_after_seconds} second(s)."
                    )
                },
                headers={
                    "Retry-After": retry_after,
                    "X-RateLimit-Limit": str(decision.limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(reset_epoch),
                },
            )
            await response(scope, receive, send)
            return

        async def send_with_rate_limit_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers") or [])
                headers.extend(rate_headers)
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_rate_limit_headers)


__all__ = [
    "ClockFn",
    "OutboundRateLimiter",
    "RateLimitDecision",
    "RateLimitMiddleware",
    "RateLimitRule",
    "SleepFn",
    "SlidingWindowRateLimiter",
    "extract_client_ip",
    "resolve_api_rate_limit_rule",
]
