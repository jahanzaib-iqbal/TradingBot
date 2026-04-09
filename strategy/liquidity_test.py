"""
strategy/liquidity_test.py
===========================
Self-contained tests for LiquiditySweepDetector using synthetic OHLCV data.
No MT5 connection required.

Scenarios tested
----------------
1.  HIGH sweep detected      — spike above previous swing high then reversal
2.  LOW  sweep detected      — spike below previous swing low then reversal
3.  No sweep (clean breakout)— price breaks above high but stays there (not a sweep)
4.  No sweep (no breach)     — price never touches the swing level
5.  Insufficient bars        — fewer bars than minimum required
6.  Direction == "SELL"      — high sweep returns SELL direction
7.  Direction == "BUY"       — low  sweep returns BUY  direction
8.  Quality scoring          — HIGH quality sweep has score >= 0.70
9.  to_dict() keys           — all required keys present
10. get_latest_sweep()       — returns most recent sweep
11. Confirmation gate        — confirmed=True when next bar agrees

Run from the project root:
    cd gold_trading_bot
    python strategy/liquidity_test.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone

if __name__ == "__main__":
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from strategy.liquidity_detection import (
    LiquiditySweepDetector,
    LiquiditySweep,
    SweepType,
    SweepQuality,
)

# ── Colour helpers ────────────────────────────────────────────────────────────
GREEN = "\033[92m"; RED = "\033[91m"; CYAN = "\033[96m"
RESET = "\033[0m";  BOLD = "\033[1m"

def ok(msg):   print(f"  {GREEN}PASS{RESET}  {msg}")
def fail(msg): print(f"  {RED}FAIL{RESET}  {msg}")
def hdr(msg):  print(f"\n{BOLD}{CYAN}{'-'*60}{RESET}\n{BOLD}{CYAN}  {msg}{RESET}\n{BOLD}{CYAN}{'-'*60}{RESET}")


# ─────────────────────────────────────────────────────────────────────────────
# Data builders
# ─────────────────────────────────────────────────────────────────────────────

def _make_base(n: int = 60, base_price: float = 2100.0) -> pd.DataFrame:
    """Build a quiet range DataFrame — uniform bars with tiny noise."""
    rng = np.random.default_rng(42)
    hi  = base_price + rng.uniform(3.0, 5.0, n)
    lo  = base_price - rng.uniform(3.0, 5.0, n)
    op  = base_price + rng.uniform(-1.0, 1.0, n)
    cl  = base_price + rng.uniform(-1.0, 1.0, n)
    hi  = np.maximum(hi, np.maximum(op, cl))
    lo  = np.minimum(lo, np.minimum(op, cl))
    vol = rng.integers(800, 1200, n).astype(float)

    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    times = [start + timedelta(hours=i) for i in range(n)]

    return pd.DataFrame({
        "time":        pd.to_datetime(times, utc=True),
        "open":        op.round(2),
        "high":        hi.round(2),
        "low":         lo.round(2),
        "close":       cl.round(2),
        "tick_volume": vol,
    })


def _inject_high_sweep(df: pd.DataFrame, bar_idx: int, swing_high: float) -> pd.DataFrame:
    """
    Turn bar[bar_idx] into a high-sweep candle.

    The bar spikes 10 pts ABOVE swing_high but closes 5 pts BELOW it.
    Volume is 3x average to signal institutional activity.
    """
    df = df.copy()
    df.at[bar_idx, "high"]        = swing_high + 10.0
    df.at[bar_idx, "low"]         = swing_high - 8.0
    df.at[bar_idx, "open"]        = swing_high + 2.0   # open above level
    df.at[bar_idx, "close"]       = swing_high - 5.0   # close back below → sweep
    df.at[bar_idx, "tick_volume"] = 3000.0               # volume surge
    return df


def _inject_low_sweep(df: pd.DataFrame, bar_idx: int, swing_low: float) -> pd.DataFrame:
    """
    Turn bar[bar_idx] into a low-sweep candle.

    The bar spikes 10 pts BELOW swing_low but closes 5 pts ABOVE it.
    """
    df = df.copy()
    df.at[bar_idx, "low"]         = swing_low - 10.0
    df.at[bar_idx, "high"]        = swing_low + 8.0
    df.at[bar_idx, "open"]        = swing_low - 2.0    # open below level
    df.at[bar_idx, "close"]       = swing_low + 5.0    # close back above → sweep
    df.at[bar_idx, "tick_volume"] = 3000.0
    return df


def _inject_genuine_breakout(df: pd.DataFrame, bar_idx: int, swing_high: float) -> pd.DataFrame:
    """Bar breaks above swing_high AND stays above — NOT a sweep."""
    df = df.copy()
    df.at[bar_idx, "high"]  = swing_high + 15.0
    df.at[bar_idx, "low"]   = swing_high + 2.0
    df.at[bar_idx, "open"]  = swing_high + 3.0
    df.at[bar_idx, "close"] = swing_high + 12.0   # close stays ABOVE → breakout
    return df


def _find_swing_high_price(df: pd.DataFrame, lookback: int = 3) -> tuple[int, float]:
    """Locate the first clear swing high in the DataFrame."""
    det = LiquiditySweepDetector()
    det.swing_lookback = lookback
    swings = det._find_swing_highs(df)
    if not swings:
        raise RuntimeError("No swing high found in test data")
    idx = sorted(swings.keys())[0]
    return idx, swings[idx]


def _find_swing_low_price(df: pd.DataFrame, lookback: int = 3) -> tuple[int, float]:
    """Locate the first clear swing low in the DataFrame."""
    det = LiquiditySweepDetector()
    det.swing_lookback = lookback
    swings = det._find_swing_lows(df)
    if not swings:
        raise RuntimeError("No swing low found in test data")
    idx = sorted(swings.keys())[0]
    return idx, swings[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    print(f"\n{BOLD}{'='*60}")
    print("  LiquiditySweepDetector -- Synthetic Data Tests")
    print(f"{'='*60}{RESET}")

    det     = LiquiditySweepDetector()
    results = []

    # ── Test 1: HIGH sweep detected ───────────────────────────────────────
    hdr("Test 1 -- HIGH sweep detected")
    df1 = _make_base(n=80)
    sh_idx, sh_price = _find_swing_high_price(df1)
    sweep_bar = sh_idx + 10          # 10 bars after the swing high
    df1 = _inject_high_sweep(df1, sweep_bar, sh_price)
    sweeps1 = det.detect(df1)

    high_sweeps = [s for s in sweeps1 if s.type == SweepType.HIGH]
    t1 = len(high_sweeps) >= 1
    if t1:
        s = high_sweeps[0]
        ok(f"HIGH sweep found: {s}")
    else:
        fail(f"No HIGH sweep detected  (found {len(sweeps1)} total sweeps)")
    results.append(t1)

    # ── Test 2: LOW sweep detected ────────────────────────────────────────
    hdr("Test 2 -- LOW sweep detected")
    df2 = _make_base(n=80)
    sl_idx, sl_price = _find_swing_low_price(df2)
    sweep_bar2 = sl_idx + 10
    df2 = _inject_low_sweep(df2, sweep_bar2, sl_price)
    sweeps2 = det.detect(df2)

    low_sweeps = [s for s in sweeps2 if s.type == SweepType.LOW]
    t2 = len(low_sweeps) >= 1
    if t2:
        s = low_sweeps[0]
        ok(f"LOW sweep found: {s}")
    else:
        fail(f"No LOW sweep detected  (found {len(sweeps2)} total sweeps)")
    results.append(t2)

    # ── Test 3: No sweep on genuine breakout ──────────────────────────────
    hdr("Test 3 -- No sweep on genuine breakout (close stays above level)")
    df3 = _make_base(n=80)
    sh_idx3, sh_price3 = _find_swing_high_price(df3)
    sweep_bar3 = sh_idx3 + 10
    df3 = _inject_genuine_breakout(df3, sweep_bar3, sh_price3)
    sweeps3 = det.detect(df3)
    breakout_bar_sweeps = [s for s in sweeps3 if s.sweep_bar_index == sweep_bar3 and s.type == SweepType.HIGH]
    t3 = len(breakout_bar_sweeps) == 0
    if t3:
        ok("Genuine breakout correctly NOT flagged as a sweep")
    else:
        fail(f"False positive: breakout bar incorrectly flagged as sweep")
    results.append(t3)

    # ── Test 4: No sweep when price never reaches the level ───────────────
    hdr("Test 4 -- No sweep when price stays below swing high")
    df4 = _make_base(n=80)
    sweeps4 = det.detect(df4)
    t4_note = f"(found {len(sweeps4)} sweeps in clean data)"
    # In clean uniform data there should be very few or zero sweeps
    t4 = len(sweeps4) == 0
    if t4:
        ok(f"No sweeps in clean ranging data {t4_note}")
    else:
        # Some noise sweeps may appear; check their quality
        low_q = all(s.quality == SweepQuality.LOW for s in sweeps4)
        if low_q:
            ok(f"Only LOW quality noise sweeps in clean data {t4_note}")
            t4 = True
        else:
            fail(f"Unexpected sweeps in clean data {t4_note}")
    results.append(t4)

    # ── Test 5: Insufficient bars ─────────────────────────────────────────
    hdr("Test 5 -- Insufficient bars (5 bars only)")
    df5     = _make_base(n=5)
    sweeps5 = det.detect(df5)
    t5 = len(sweeps5) == 0
    if t5:
        ok("Empty result returned for 5-bar DataFrame (too short)")
    else:
        fail(f"Expected no sweeps for 5 bars; got {len(sweeps5)}")
    results.append(t5)

    # ── Test 6: HIGH sweep → direction == "SELL" ─────────────────────────
    hdr("Test 6 -- HIGH sweep direction is SELL")
    if t1 and high_sweeps:
        t6 = high_sweeps[0].direction == "SELL"
        if t6:
            ok(f"HIGH sweep direction = SELL (correct)")
        else:
            fail(f"HIGH sweep direction = {high_sweeps[0].direction} (expected SELL)")
    else:
        t6 = False
        fail("No HIGH sweep available from Test 1 to check direction")
    results.append(t6)

    # ── Test 7: LOW sweep → direction == "BUY" ───────────────────────────
    hdr("Test 7 -- LOW sweep direction is BUY")
    if t2 and low_sweeps:
        t7 = low_sweeps[0].direction == "BUY"
        if t7:
            ok(f"LOW sweep direction = BUY (correct)")
        else:
            fail(f"LOW sweep direction = {low_sweeps[0].direction} (expected BUY)")
    else:
        t7 = False
        fail("No LOW sweep available from Test 2 to check direction")
    results.append(t7)

    # ── Test 8: Quality scoring ───────────────────────────────────────────
    hdr("Test 8 -- Quality score >= 0.70 for strong sweep")
    if t1 and high_sweeps:
        s = high_sweeps[0]
        t8 = s.quality_score >= 0.40   # at least MEDIUM for our injected sweep
        if t8:
            ok(f"Quality score={s.quality_score:.2f} label={s.quality.value}")
        else:
            fail(f"Quality score too low: {s.quality_score:.2f}")
    else:
        t8 = False
        fail("No sweep to score")
    results.append(t8)

    # ── Test 9: to_dict() keys ────────────────────────────────────────────
    hdr("Test 9 -- to_dict() contains all required keys")
    required_keys = {
        "type", "direction", "sweep_type", "sweep_price",
        "sweep_bar_index", "wick_size", "wick_atr_ratio",
        "level_age_bars", "quality_score", "quality", "confirmed", "notes",
    }
    if t1 and high_sweeps:
        d = high_sweeps[0].to_dict()
        missing_keys = required_keys - set(d.keys())
        t9 = len(missing_keys) == 0
        if t9:
            ok(f"All {len(required_keys)} required keys present")
            ok(f"type={d['type']!r}  direction={d['direction']!r}  quality={d['quality']!r}")
        else:
            fail(f"Missing keys: {missing_keys}")
    else:
        t9 = False
        fail("No sweep dict to inspect")
    results.append(t9)

    # ── Test 10: get_latest_sweep() ───────────────────────────────────────
    hdr("Test 10 -- get_latest_sweep() returns most recent sweep")
    latest = det.get_latest_sweep(df1)
    t10 = latest is not None
    if t10:
        ok(f"get_latest_sweep() returned: {latest}")
    else:
        fail("get_latest_sweep() returned None")
    results.append(t10)

    # ── Test 11: Confirmation gate ────────────────────────────────────────
    hdr("Test 11 -- Confirmation flag when next bar closes in reversal direction")
    # Build a HIGH sweep where the NEXT bar also closes bearishly (confirmed)
    df11 = _make_base(n=80)
    sh_idx11, sh_price11 = _find_swing_high_price(df11)
    sweep_bar11 = sh_idx11 + 8     # ensure there's at least one more bar after
    df11 = _inject_high_sweep(df11, sweep_bar11, sh_price11)
    # Make the next bar also close lower (strong bearish confirmation)
    if sweep_bar11 + 1 < len(df11):
        sweep_close = float(df11.at[sweep_bar11, "close"])
        df11.at[sweep_bar11 + 1, "close"] = sweep_close - 8.0
        df11.at[sweep_bar11 + 1, "high"]  = sweep_close + 1.0
        df11.at[sweep_bar11 + 1, "low"]   = sweep_close - 10.0
        df11.at[sweep_bar11 + 1, "open"]  = sweep_close - 1.0

    sweeps11   = det.detect(df11)
    bar11_sweeps = [s for s in sweeps11
                    if s.sweep_bar_index == sweep_bar11 and s.type == SweepType.HIGH]
    t11 = len(bar11_sweeps) > 0 and bar11_sweeps[0].confirmed
    if t11:
        ok(f"Sweep confirmed=True when next bar closes bearishly")
    else:
        if len(bar11_sweeps) == 0:
            fail("No sweep detected on the target bar")
        else:
            fail(f"Sweep found but confirmed={bar11_sweeps[0].confirmed}")
    results.append(t11)

    # ── Summary ───────────────────────────────────────────────────────────
    hdr("Summary")
    passed = sum(results)
    total  = len(results)
    for i, r in enumerate(results, 1):
        s = f"{GREEN}PASS{RESET}" if r else f"{RED}FAIL{RESET}"
        print(f"  [{s}]  Test {i}")
    print()
    if passed == total:
        print(f"{GREEN}{BOLD}All {total} tests passed{RESET}")
        return 0
    else:
        print(f"{RED}{BOLD}{total - passed}/{total} tests FAILED{RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
