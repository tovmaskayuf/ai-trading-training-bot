"""Indicator correctness checks against hand-computed / published values."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analytics import indicators as ind

failures: list[str] = []


def check(name: str, got, want, tol=1e-6):
    if want is None:
        ok = got is None
    elif got is None:
        ok = False
    else:
        ok = abs(got - want) <= tol
    print(f"{'PASS' if ok else 'FAIL'}  {name}: got={got} want={want}")
    if not ok:
        failures.append(name)


# --- RSI against Wilder's published series ---------------------------------
# Standard reference dataset; RSI(14) at the 15th close is ~70.53.
wilder = [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42,
          45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28]
check("rsi(14) wilder", ind.rsi(wilder, 14), 70.53, tol=0.1)

# Monotonic rise has no losses -> RSI pins at 100.
check("rsi all gains", ind.rsi([float(i) for i in range(1, 20)], 14), 100.0)
# Flat series has neither gains nor losses -> neutral.
check("rsi flat", ind.rsi([50.0] * 20, 14), 50.0)
# Insufficient data must return None, not raise.
check("rsi short", ind.rsi([1.0, 2.0, 3.0], 14), None)

# --- EMA -------------------------------------------------------------------
# EMA of a constant series is that constant.
check("ema constant", ind.ema([10.0] * 30, 10), 10.0, tol=1e-9)
# Hand-computed: seed=SMA(1,2,3)=2, k=0.5; next=4*0.5+2*0.5=3; then 5*.5+3*.5=4
check("ema hand-calc", ind.ema([1.0, 2.0, 3.0, 4.0, 5.0], 3), 4.0, tol=1e-9)
check("ema short", ind.ema([1.0, 2.0], 10), None)

# --- MACD ------------------------------------------------------------------
# On a constant series every EMA equals the constant, so macd/signal/hist = 0.
m = ind.macd([100.0] * 60)
check("macd constant line", m["macd"] if m else None, 0.0, tol=1e-9)
check("macd constant hist", m["hist"] if m else None, 0.0, tol=1e-9)
check("macd short", ind.macd([1.0] * 10), None)

# A steadily rising series must produce a positive MACD line.
rising = ind.macd([float(i) for i in range(1, 80)])
print(f"{'PASS' if rising and rising['macd'] > 0 else 'FAIL'}  "
      f"macd rising is positive: {rising['macd'] if rising else None}")
if not (rising and rising["macd"] > 0):
    failures.append("macd rising positive")

# --- ATR -------------------------------------------------------------------
# Every bar has high-low = 2 and no gaps, so ATR = 2.
n = 30
highs = [11.0] * n
lows = [9.0] * n
closes = [10.0] * n
check("atr constant range", ind.atr(highs, lows, closes, 14), 2.0, tol=1e-9)
check("atr short", ind.atr([1.0], [1.0], [1.0], 14), None)

# --- Drawdown --------------------------------------------------------------
# 100 -> 50 is a 50% drawdown, and recovery does not erase the historical max.
check("max_drawdown", ind.max_drawdown([100.0, 80.0, 50.0, 90.0]), 0.5, tol=1e-9)
check("max_drawdown monotonic", ind.max_drawdown([1.0, 2.0, 3.0]), 0.0, tol=1e-9)

# --- Range position --------------------------------------------------------
check("range_position at high", ind.range_position([1.0, 2.0, 3.0], 3), 100.0)
check("range_position at low", ind.range_position([3.0, 2.0, 1.0], 3), 0.0)
check("range_position mid", ind.range_position([1.0, 3.0, 2.0], 3), 50.0)
check("range_position flat", ind.range_position([5.0, 5.0, 5.0], 3), 50.0)

# --- Percentile rank -------------------------------------------------------
check("pct_rank median", ind.pct_rank(3.0, [1.0, 2.0, 3.0, 4.0, 5.0]), 50.0)
check("pct_rank top", ind.pct_rank(9.0, [1.0, 2.0, 3.0]), 100.0)
check("pct_rank empty", ind.pct_rank(1.0, []), 50.0)

# --- Scale -----------------------------------------------------------------
check("scale mid", ind.scale(50, 0, 100), 50.0)
check("scale clamps high", ind.scale(500, 0, 100), 100.0)
check("scale clamps low", ind.scale(-50, 0, 100), 0.0)
check("scale inverted", ind.scale(0, 0, 100, invert=True), 100.0)
check("scale degenerate", ind.scale(5, 10, 10), 50.0)

# --- pct_change ------------------------------------------------------------
check("pct_change +10%", ind.pct_change([100.0, 110.0], 1), 10.0, tol=1e-9)
check("pct_change short", ind.pct_change([100.0], 5), None)

print()
if failures:
    print(f"{len(failures)} FAILURE(S): {failures}")
    sys.exit(1)
print("all indicator checks passed")
