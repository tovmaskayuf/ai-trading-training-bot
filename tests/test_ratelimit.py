"""Rate-limit handling.

Binance escalates repeated 429s into an outright IP ban (HTTP 418) lasting
tens of minutes, and on shared cloud egress the per-IP budget is spent by other
tenants too. Retrying a rate-limit response is what causes the escalation: each
attempt spends more of the same budget. These checks pin the behaviour that a
real 30-minute ban forced.

No network access: responses are faked, so this runs anywhere and cannot itself
contribute to a ban.
"""

import asyncio
import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from providers import base  # noqa: E402

failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(("PASS  " if cond else "FAIL  ") + name +
          (f"\n      {detail}" if detail and not cond else ""))
    if not cond:
        failures.append(name)


class FakeResponse:
    def __init__(self, status, text="", headers=None):
        self.status_code = status
        self.text = text
        self.headers = headers or {}

    def json(self):
        return {"ok": True}


def install(handler):
    base._client = types.SimpleNamespace(request=handler)
    base._cooldown_until.clear()


async def run() -> None:
    calls = {"n": 0}

    # --- 418 ban: parsed expiry, no retry, host put on cooldown ------------
    ban_at = int((time.time() + 120) * 1000)

    async def banned(method, url, **kw):
        calls["n"] += 1
        return FakeResponse(418,
                            '{"code":-1003,"msg":"Way too much request weight used; '
                            f'IP banned until {ban_at}."}}')

    install(banned)
    try:
        await base.get("https://api.binance.com/api/v3/klines")
        check("418 raises", False)
    except base.ProviderError as e:
        check("418 raises ProviderError", True)
        check("flagged as rate limited", e.rate_limited is True)
        check("not retried -- retrying is what causes the ban",
              calls["n"] == 1, f"{calls['n']} attempts")
        check("ban expiry parsed from the body",
              110 < e.retry_after < 130, f"{e.retry_after:.0f}s")

    before = calls["n"]
    for _ in range(5):
        try:
            await base.get("https://api.binance.com/api/v3/klines")
        except base.ProviderError:
            pass
    check("later calls short-circuit without hitting the network",
          calls["n"] == before, f"{calls['n'] - before} extra requests")
    check("cooldown is visible for diagnostics",
          "api.binance.com" in base.cooldowns())
    check("a different host is unaffected",
          base.cooldown_remaining("https://api.hyperliquid.xyz/info") == 0)

    # --- 429 with Retry-After ---------------------------------------------
    calls["n"] = 0

    async def limited(method, url, **kw):
        calls["n"] += 1
        return FakeResponse(429, "slow down", {"Retry-After": "45"})

    install(limited)
    try:
        await base.get("https://api.coingecko.com/api/v3/simple/price")
        check("429 raises", False)
    except base.ProviderError as e:
        check("Retry-After is honoured", 40 < e.retry_after < 50,
              f"{e.retry_after:.0f}s")
        check("429 is not retried either", calls["n"] == 1, f"{calls['n']} attempts")

    # --- 429 without Retry-After falls back to a default -------------------
    calls["n"] = 0

    async def bare(method, url, **kw):
        calls["n"] += 1
        return FakeResponse(429, "slow down")

    install(bare)
    try:
        await base.get("https://api.example.com/x")
    except base.ProviderError as e:
        check("missing Retry-After uses a conservative default",
              e.retry_after >= 60, f"{e.retry_after:.0f}s")

    # --- 5xx is transient and *should* still retry -------------------------
    calls["n"] = 0

    async def failing(method, url, **kw):
        calls["n"] += 1
        return FakeResponse(503, "unavailable")

    install(failing)
    try:
        await base.get("https://api.example.org/y")
    except base.ProviderError:
        pass
    check("5xx still retries -- transient, not a budget problem",
          calls["n"] == base.MAX_RETRIES, f"{calls['n']} attempts")

    # --- a permanent 4xx must not create a cooldown ------------------------
    calls["n"] = 0

    async def bad_symbol(method, url, **kw):
        calls["n"] += 1
        return FakeResponse(400, '{"code":-1121,"msg":"Invalid symbol."}')

    install(bad_symbol)
    try:
        await base.get("https://api.binance.com/api/v3/klines")
    except base.ProviderError as e:
        check("400 fails immediately", calls["n"] == 1)
        check("400 is not treated as rate limiting", e.rate_limited is False)
    check("a bad request does not block the host",
          base.cooldown_remaining("https://api.binance.com/api/v3/klines") == 0)

    base._client = None
    base._cooldown_until.clear()


def main() -> None:
    asyncio.run(run())
    print()
    if failures:
        print(f"{len(failures)} FAILURE(S): {failures}")
        sys.exit(1)
    print("all rate-limit checks passed")


if __name__ == "__main__":
    main()
