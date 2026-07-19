"""Technical indicators on plain Python lists.

Deliberately dependency-free: 15 assets x a few hundred candles is trivial
compute, and avoiding numpy/pandas keeps the install trivial on new Python
releases where compiled wheels may lag.

Every function returns None when there is not enough data rather than raising,
so a cold start degrades gracefully instead of crashing the engine.
"""

from __future__ import annotations

import math
from statistics import fmean, pstdev

Series = list[float]


def sma(values: Series, period: int) -> float | None:
    if len(values) < period:
        return None
    return fmean(values[-period:])


def ema_series(values: Series, period: int) -> Series | None:
    """Full EMA series, seeded with an SMA of the first `period` values."""
    if len(values) < period:
        return None
    k = 2 / (period + 1)
    out = [fmean(values[:period])]
    for v in values[period:]:
        out.append(v * k + out[-1] * (1 - k))
    return out


def ema(values: Series, period: int) -> float | None:
    s = ema_series(values, period)
    return s[-1] if s else None


def rsi(values: Series, period: int = 14) -> float | None:
    """Wilder's RSI. Returns 0-100."""
    if len(values) < period + 1:
        return None

    deltas = [values[i] - values[i - 1] for i in range(1, len(values))]
    gains = [max(d, 0.0) for d in deltas]
    losses = [max(-d, 0.0) for d in deltas]

    avg_gain = fmean(gains[:period])
    avg_loss = fmean(losses[:period])

    # Wilder smoothing over the remainder.
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd(values: Series, fast: int = 12, slow: int = 26,
         signal: int = 9) -> dict[str, float] | None:
    """Returns {macd, signal, hist, hist_slope}.

    hist_slope is the change in histogram vs the prior bar -- it distinguishes
    'negative but improving' from 'negative and deteriorating', which matters
    a lot for the momentum score.
    """
    if len(values) < slow + signal:
        return None

    fast_s = ema_series(values, fast)
    slow_s = ema_series(values, slow)
    if not fast_s or not slow_s:
        return None

    # ema_series outputs are offset by their seeding periods; align on the tail.
    n = min(len(fast_s), len(slow_s))
    macd_line = [fast_s[-n + i] - slow_s[-n + i] for i in range(n)]

    sig_s = ema_series(macd_line, signal)
    if not sig_s:
        return None

    m = min(len(macd_line), len(sig_s))
    hist = [macd_line[-m + i] - sig_s[-m + i] for i in range(m)]

    return {
        "macd": macd_line[-1],
        "signal": sig_s[-1],
        "hist": hist[-1],
        "hist_slope": hist[-1] - hist[-2] if len(hist) >= 2 else 0.0,
    }


def true_ranges(highs: Series, lows: Series, closes: Series) -> Series | None:
    if len(highs) < 2 or not (len(highs) == len(lows) == len(closes)):
        return None
    return [
        max(highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]))
        for i in range(1, len(highs))
    ]


def atr(highs: Series, lows: Series, closes: Series, period: int = 14) -> float | None:
    """Wilder's Average True Range in price units."""
    tr = true_ranges(highs, lows, closes)
    if not tr or len(tr) < period:
        return None
    val = fmean(tr[:period])
    for t in tr[period:]:
        val = (val * (period - 1) + t) / period
    return val


def log_returns(values: Series) -> Series:
    return [
        math.log(values[i] / values[i - 1])
        for i in range(1, len(values))
        if values[i - 1] > 0 and values[i] > 0
    ]


def realized_vol(values: Series, periods_per_year: int = 8760) -> float | None:
    """Annualized realized volatility as a fraction (0.8 == 80%).

    periods_per_year defaults to hourly bars (24 * 365).
    """
    r = log_returns(values)
    if len(r) < 2:
        return None
    return pstdev(r) * math.sqrt(periods_per_year)


def max_drawdown(values: Series) -> float | None:
    """Largest peak-to-trough decline as a positive fraction."""
    if len(values) < 2:
        return None
    peak = values[0]
    worst = 0.0
    for v in values:
        peak = max(peak, v)
        if peak > 0:
            worst = max(worst, (peak - v) / peak)
    return worst


def sharpe(values: Series, periods_per_year: int = 8760) -> float | None:
    """Annualized return/volatility ratio. Risk-free rate assumed zero."""
    r = log_returns(values)
    if len(r) < 2:
        return None
    sd = pstdev(r)
    if sd == 0:
        return 0.0
    return (fmean(r) / sd) * math.sqrt(periods_per_year)


def pct_change(values: Series, periods: int) -> float | None:
    """Percent change over the last `periods` bars."""
    if len(values) < periods + 1 or values[-periods - 1] == 0:
        return None
    return (values[-1] / values[-periods - 1] - 1) * 100


def range_position(values: Series, periods: int) -> float | None:
    """Where the latest value sits in its recent range, 0-100.

    0 == at the period low, 100 == at the period high.
    """
    if len(values) < periods:
        return None
    window = values[-periods:]
    lo, hi = min(window), max(window)
    if hi == lo:
        return 50.0
    return (values[-1] - lo) / (hi - lo) * 100


def pct_rank(value: float, population: list[float]) -> float:
    """Percentile rank of `value` within `population`, 0-100."""
    if not population:
        return 50.0
    below = sum(1 for p in population if p < value)
    equal = sum(1 for p in population if p == value)
    return (below + 0.5 * equal) / len(population) * 100


def clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


def scale(value: float, lo: float, hi: float, invert: bool = False) -> float:
    """Map a raw value onto 0-100 by linear interpolation between lo and hi.

    Values outside [lo, hi] clamp to the ends. With invert=True, lo maps to 100
    (used for risk metrics where smaller is better).
    """
    if hi == lo:
        return 50.0
    pos = (value - lo) / (hi - lo) * 100
    pos = clamp(pos)
    return 100 - pos if invert else pos
