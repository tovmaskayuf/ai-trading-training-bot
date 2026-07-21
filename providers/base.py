"""Shared async HTTP plumbing for all upstream providers."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any
from urllib.parse import urlsplit

import httpx

log = logging.getLogger("providers")

_client: httpx.AsyncClient | None = None

TIMEOUT = httpx.Timeout(15.0, connect=8.0)
MAX_RETRIES = 3


def client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            timeout=TIMEOUT,
            headers={"User-Agent": "ai-trading-training-bot/1.0"},
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )
    return _client


async def aclose() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


class ProviderError(RuntimeError):
    def __init__(self, message: str, *, status: int | None = None,
                 permanent: bool = False, rate_limited: bool = False,
                 retry_after: float = 0.0):
        super().__init__(message)
        self.status = status
        self.permanent = permanent
        self.rate_limited = rate_limited
        self.retry_after = retry_after


# Per-host cooldowns. A rate-limited host stays skipped until its window ends
# instead of being hammered through it. Binance escalates repeated 429s to an
# outright IP ban (HTTP 418) lasting tens of minutes, and on shared cloud
# egress the budget is spent by other tenants too, so backing off on the first
# refusal is the difference between a slow minute and a half-hour outage.
_cooldown_until: dict[str, float] = {}

# Binance states the ban expiry as an epoch-millisecond timestamp in the body.
_BAN_UNTIL_RE = re.compile(r"banned until (\d{10,})")


def _host(url: str) -> str:
    return urlsplit(url).netloc


def cooldown_remaining(url: str) -> float:
    """Seconds left before this host may be called again."""
    return max(0.0, _cooldown_until.get(_host(url), 0.0) - time.time())


def cooldowns() -> dict[str, int]:
    """Active cooldowns in seconds remaining, for diagnostics."""
    now = time.time()
    return {h: int(t - now) for h, t in _cooldown_until.items() if t > now}


def _note_rate_limit(url: str, resp: Any) -> float:
    """Record a cooldown from the response and return its length in seconds."""
    host = _host(url)
    body = (resp.text or "")[:300]

    seconds = 0.0
    m = _BAN_UNTIL_RE.search(body)
    if m:                                   # explicit ban expiry
        seconds = max(0.0, int(m.group(1)) / 1000 - time.time())
    if not seconds:
        try:
            seconds = float(resp.headers.get("Retry-After", "") or 0)
        except ValueError:
            seconds = 0.0
    if not seconds:
        seconds = 60.0                      # conservative default

    seconds = min(seconds, 3600.0)
    _cooldown_until[host] = time.time() + seconds
    log.warning("%s rate-limited (%s); backing off %.0fs",
                host, resp.status_code, seconds)
    return seconds


async def request(method: str, url: str, **kw: Any) -> Any:
    """Issue a request with exponential backoff, returning parsed JSON.

    Retries transport errors and 5xx. Rate-limit responses (429, and Binance's
    418 ban) are **not** retried: retrying is what escalates a soft limit into
    a ban, since each attempt spends more of the same budget. The host is put
    on cooldown and the caller degrades for that cycle instead.
    """
    wait = cooldown_remaining(url)
    if wait > 0:
        raise ProviderError(
            f"{_host(url)} is rate-limited for another {wait:.0f}s",
            status=429, permanent=True, rate_limited=True)

    delay = 1.0
    last: Exception | None = None

    for attempt in range(MAX_RETRIES):
        try:
            resp = await client().request(method, url, **kw)
            if resp.status_code in (418, 429):
                secs = _note_rate_limit(url, resp)
                raise ProviderError(
                    f"{resp.status_code} from {url}: {resp.text[:200]}",
                    status=resp.status_code, permanent=True, rate_limited=True,
                    retry_after=secs,
                )
            if resp.status_code >= 500:
                raise ProviderError(f"{resp.status_code} from {url}",
                                    status=resp.status_code)
            if resp.status_code >= 400:
                raise ProviderError(
                    f"{resp.status_code} from {url}: {resp.text[:200]}",
                    status=resp.status_code, permanent=True,
                )
            return resp.json()
        except ProviderError as e:
            if e.permanent:
                raise
            last = e
        except (httpx.HTTPError, ValueError) as e:
            last = e

        if attempt < MAX_RETRIES - 1:
            await asyncio.sleep(delay)
            delay *= 2

    raise ProviderError(f"failed after {MAX_RETRIES} attempts: {last}")


async def get(url: str, **kw: Any) -> Any:
    return await request("GET", url, **kw)


async def post(url: str, **kw: Any) -> Any:
    return await request("POST", url, **kw)
