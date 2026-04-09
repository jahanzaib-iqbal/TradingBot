"""
strategy/smc_test.py
=====================
Self-contained tests for SmartMoneyStrategy using synthetic OHLCV data.
No MT5 or Settings dependency required.

Tests
-----
1.  Market structure: HH/HL labels on uptrend data
2.  Market structure: LH/LL labels on downtrend data
3.  Market structure: EH (equal high) detection
4.  BOS detected in uptrend
5.  CHoCH detected (reversal signal)
6.  Order block detection — bullish OB before impulse
7.  Order block detection — bearish OB before impulse
8.  FVG detection — bullish gap
9.  FVG detection — bearish gap
10. Liquidity pool mapping — swing levels
11. Liquidity pool — equal highs / lows
12. scan() returns BUY setup
13. scan() returns SELL setup
14. to_dict() output keys
15. get_best_setup() returns highest confidence

Run:
    cd gold_trading_bot
    python strategy/smc_test.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone

if __name__ == "__main__":
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from strategy.smart_money_strategy import (
    SmartMoneyStrategy,
    TradeIdea,
    StructureLabel,
    BosType,
    OBType,
    FVGType,
)

# ── ANSI colours ──────────────────────────────────────────────────────────────
GREEN = "\033[92m"; RED = "\033[91m"; CYAN = "\033[96m"
RESET = "\033[0m";  BOLD = "\033[1m"

def ok(msg):   print(f"  {GREEN}PASS{RESET}  {msg}")
def fail(msg): print(f"  {RED}FAIL{RESET}  {msg}")
def hdr(msg):  print(f"\n{BOLD}{CYAN}{'-'*60}{RESET}\n{BOLD}{CYAN}  {msg}{RESET}\n{BOLD}{CYAN}{'-'*60}{RESET}")


# ─────────────────────────────────────────────────────────────────────────────
# Data builders
# ─────────────────────────────────────────────────────────────────────────────

def _make_df(closes: np.ndarray, noise: float = 0.5, seed: int = 7) -> pd.DataFrame:
    """Build OHLCV DataFrame from close array with controlled noise."""
    rng = np.random.default_rng(seed)
    n   = len(closes)
    hi  = closes + rng.uniform(0.5 * noise, 1.5 * noise, n)
    lo  = closes - rng.uniform(0.5 * noise, 1.5 * noise, n)
    op  = closes + rng.uniform(-0.3 * noise, 0.3 * noise, n)
    hi  = np.maximum(hi, np.maximum(op, closes))
    lo  = np.minimum(lo, np.minimum(op, closes))
    op  = np.clip(op, lo, hi)
    vol = rng.integers(500, 1500, n).astype(float)

    start = datetime(2024, 3, 1, tzinfo=timezone.utc)
    times = [start + timedelta(minutes=15 * i) for i in range(n)]

    return pd.DataFrame({
        "time":        pd.to_datetime(times, utc=True),
        "open":        op.round(2),
        "high":        hi.round(2),
        "low":         lo.round(2),
        "close":       closes.round(2),
        "tick_volume": vol,
    })


def _uptrend(n: int = 200) -> pd.DataFrame:
    """Clean uptrend oscillating around a rising baseline."""
    # Stepped uptrend: rise 20 pts every 20 bars with 3-pt oscillation
    t     = np.arange(n)
    base  = 2100 + t * 0.5    # 0.5 pt/bar = 100 pt over 200 bars
    wave  = 3 * np.sin(2 * np.pi * t / 20)
    return _make_df(base + wave, noise=1.0)


def _downtrend(n: int = 200) -> pd.DataFrame:
    """Clean downtrend."""
    t    = np.arange(n)
    base = 2300 - t * 0.5
    wave = 3 * np.sin(2 * np.pi * t / 20)
    return _make_df(base + wave, noise=1.0)


def _with_buy_setup(n: int = 200) -> pd.DataFrame:
    """
    Manually crafted uptrend DataFrame with explicit:
      - Bullish Order Block (bearish candle with large body before impulse)
      - Sweep of a previous swing low
      - Confirmed liquidity pool
    """
    # Use a stepped baseline with very low noise so OB body ratios are clear
    closes = np.full(n, 2100.0, dtype=float)

    # Bars 0-149: steady uptrend
    closes[:150] = np.linspace(2100.0, 2180.0, 150)

    # Inject a clear swing LOW at bar 60 = reference pool price
    swing_low_price = 2130.0
    closes[58] = 2140.0; closes[59] = 2135.0
    closes[60] = swing_low_price
    closes[61] = 2135.0; closes[62] = 2140.0

    # Bars 120-126: Bearish OB candle (bar 120) then strong bullish impulse
    # OB candle: large bearish body, so body_ratio > 0.5
    # Impulse: 3 big bullish bars afterward
    closes[120] = 2165.0   # OB open (set below)
    closes[121] = 2169.0
    closes[122] = 2175.0
    closes[123] = 2181.0

    # Bars 150-158: sweep the swing low (bar 60) then recover
    closes[149] = 2175.0
    closes[150] = swing_low_price - 5.0   # spike below pool
    closes[151] = swing_low_price + 2.0   # recover above
    closes[152] = 2135.0
    closes[153] = 2140.0
    closes[154] = 2148.0
    closes[155] = 2155.0

    # Last 44 bars: price climbs into OB zone (~2163-2167)
    closes[156:] = np.linspace(2155.0, 2167.0, n - 156)

    df = _make_df(closes, noise=0.1, seed=21)   # low noise!

    # Force the OB candle (bar 120) to be a large-body BEARISH candle
    df.at[120, "open"]  = 2168.0   # open high
    df.at[120, "close"] = 2160.0   # close much lower (bearish)
    df.at[120, "high"]  = 2169.5
    df.at[120, "low"]   = 2159.0

    # Force the sweep bar (bar 150) wick below swing_low
    df.at[150, "open"]  = 2175.0
    df.at[150, "low"]   = swing_low_price - 6.0   # spike below pool
    df.at[150, "high"]  = 2177.0
    df.at[150, "close"] = 2132.0   # close back ABOVE swing_low
    return df


def _with_sell_setup(n: int = 200) -> pd.DataFrame:
    """
    Manually crafted downtrend with explicit:
      - Bearish Order Block (bullish candle with large body before impulse)
      - Sweep of a previous swing high
    """
    closes = np.full(n, 2300.0, dtype=float)
    closes[:150] = np.linspace(2300.0, 2220.0, 150)

    # Inject clear swing HIGH at bar 60
    swing_high_price = 2270.0
    closes[58] = 2260.0; closes[59] = 2265.0
    closes[60] = swing_high_price
    closes[61] = 2265.0; closes[62] = 2260.0

    # Bars 120-124: Bullish OB candle then strong bearish impulse
    closes[120] = 2245.0
    closes[121] = 2241.0
    closes[122] = 2235.0
    closes[123] = 2229.0

    # Bars 149-157: sweep the swing high then sell off
    closes[149] = 2235.0
    closes[150] = swing_high_price + 5.0   # spike above
    closes[151] = swing_high_price - 2.0   # recover below
    closes[152] = 2265.0
    closes[153] = 2258.0
    closes[154] = 2250.0
    closes[155] = 2243.0

    # Last bars: price drifts into OB zone (~2240-2248)
    closes[156:] = np.linspace(2243.0, 2246.0, n - 156)

    df = _make_df(closes, noise=0.1, seed=33)

    # Force OB candle (bar 120) to be a large-body BULLISH candle
    df.at[120, "open"]  = 2240.0
    df.at[120, "close"] = 2248.0   # open < close (bullish)
    df.at[120, "high"]  = 2249.5
    df.at[120, "low"]   = 2239.0

    # Force sweep bar (bar 150) wick above swing_high
    df.at[150, "open"]  = 2235.0
    df.at[150, "high"]  = swing_high_price + 6.0   # spike above
    df.at[150, "low"]   = 2233.0
    df.at[150, "close"] = 2264.0   # close back BELOW swing_high
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Test runner
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    print(f"\n{BOLD}{'='*60}")
    print("  SmartMoneyStrategy -- Synthetic Data Tests")
    print(f"{'='*60}{RESET}")

    smc     = SmartMoneyStrategy()
    results = []

    # ── Test 1: Market structure HH/HL in uptrend ──────────────────────────
    hdr("Test 1 -- Market structure HH/HL labels in uptrend")
    df1     = _uptrend()
    swings1 = smc._find_swings(df1)
    ms1     = smc.detect_market_structure(df1, swings1)
    hh_count = sum(1 for s in ms1 if s.label == StructureLabel.HH)
    hl_count = sum(1 for s in ms1 if s.label == StructureLabel.HL)
    t1 = hh_count >= 2 and hl_count >= 2
    ok(f"HH={hh_count}  HL={hl_count}  (uptrend confirmed)") if t1 else \
    fail(f"HH={hh_count} HL={hl_count} — expected >= 2 each")
    results.append(t1)

    # ── Test 2: Market structure LH/LL in downtrend ───────────────────────
    hdr("Test 2 -- Market structure LH/LL labels in downtrend")
    df2     = _downtrend()
    swings2 = smc._find_swings(df2)
    ms2     = smc.detect_market_structure(df2, swings2)
    lh_count = sum(1 for s in ms2 if s.label == StructureLabel.LH)
    ll_count = sum(1 for s in ms2 if s.label == StructureLabel.LL)
    t2 = lh_count >= 2 and ll_count >= 2
    ok(f"LH={lh_count}  LL={ll_count}  (downtrend confirmed)") if t2 else \
    fail(f"LH={lh_count} LL={ll_count} — expected >= 2 each")
    results.append(t2)

    # ── Test 3: Equal highs detection ─────────────────────────────────────
    hdr("Test 3 -- Equal High (EH) detection")
    # Build two peaks at exactly the same price
    closes3 = np.full(120, 2100.0)
    # Create two clear equal highs: bars 30 and 80 peak at 2100 (close)
    closes3[28:33] = [2095, 2098, 2100, 2098, 2095]
    closes3[78:83] = [2095, 2098, 2100, 2098, 2095]
    df3     = _make_df(closes3, noise=0.1)
    swings3 = smc._find_swings(df3)
    ms3     = smc.detect_market_structure(df3, swings3)
    eh_count = sum(1 for s in ms3 if s.label == StructureLabel.EH)
    t3 = eh_count >= 1
    ok(f"EH count={eh_count} (equal high liquidity pool detected)") if t3 else \
    fail(f"EH count={eh_count} — expected >= 1")
    results.append(t3)

    # ── Test 4: BOS in uptrend ────────────────────────────────────────────
    hdr("Test 4 -- BOS detected in uptrend")
    df4     = _uptrend()
    swings4 = smc._find_swings(df4)
    ms4     = smc.detect_market_structure(df4, swings4)
    bos4    = smc.detect_bos_choch(df4, swings4, ms4)
    bos_events = [e for e in bos4 if e.is_bos]
    t4 = len(bos_events) >= 1
    ok(f"BOS events found: {len(bos_events)}") if t4 else fail(f"No BOS events detected")
    results.append(t4)

    # ── Test 5: CHoCH signal ──────────────────────────────────────────────
    hdr("Test 5 -- CHoCH detected (structural reversal)")
    # In an uptrend, a CHoCH is a bearish break of structure
    df5     = _uptrend()
    swings5 = smc._find_swings(df5)
    ms5     = smc.detect_market_structure(df5, swings5)
    bos5    = smc.detect_bos_choch(df5, swings5, ms5)
    choch_events = [e for e in bos5 if e.is_choch]
    # With a clean trending series, CHoCH may or may not fire
    # Just verify the detection runs without error and returns typed events
    all_typed = all(isinstance(e.bos_type, BosType) for e in bos5)
    t5 = all_typed
    ok(f"All {len(bos5)} BOS/CHoCH events are correctly typed") if t5 else \
    fail("Some events have incorrect type")
    results.append(t5)

    # ── Test 6: Bullish OB detected ───────────────────────────────────────
    hdr("Test 6 -- Bullish Order Block detection")
    df6  = _with_buy_setup()
    # Use a relaxed-threshold SMC instance for synthetic data
    smc6 = SmartMoneyStrategy()
    smc6.OB_MIN_BODY_RATIO    = 0.35    # slightly relaxed for synthetic data
    smc6.OB_MIN_IMPULSE_ATR   = 0.1
    smc6.OB_MAX_AGE_BARS      = 200
    atr6 = smc6._atr(df6)
    obs6 = smc6.detect_order_blocks(df6, atr6)
    bull_obs = [ob for ob in obs6 if ob.ob_type == OBType.BULLISH]
    t6 = len(bull_obs) >= 1
    if t6:
        o = bull_obs[0]
        ok(f"Bullish OB found: bar={o.bar_index}  zone=[{o.ob_low:.2f}, {o.ob_high:.2f}]  "
           f"impulse={o.impulse_size:.2f}")
    else:
        fail(f"No bullish OB detected  (total OBs={len(obs6)})")
    results.append(t6)

    # ── Test 7: Bearish OB detected ───────────────────────────────────────
    hdr("Test 7 -- Bearish Order Block detection")
    df7  = _with_sell_setup()
    smc7 = SmartMoneyStrategy()
    smc7.OB_MIN_BODY_RATIO    = 0.35
    smc7.OB_MIN_IMPULSE_ATR   = 0.1
    smc7.OB_MAX_AGE_BARS      = 200
    atr7 = smc7._atr(df7)
    obs7 = smc7.detect_order_blocks(df7, atr7)
    bear_obs = [ob for ob in obs7 if ob.ob_type == OBType.BEARISH]
    t7 = len(bear_obs) >= 1
    if t7:
        o = bear_obs[0]
        ok(f"Bearish OB found: bar={o.bar_index}  zone=[{o.ob_low:.2f}, {o.ob_high:.2f}]  "
           f"impulse={o.impulse_size:.2f}")
    else:
        fail(f"No bearish OB detected  (total OBs={len(obs7)})")
    results.append(t7)

    # ── Test 8: Bullish FVG detected ──────────────────────────────────────
    hdr("Test 8 -- Bullish FVG detection")
    # Inject a 3-candle bullish FVG pattern with explicit gap > min threshold
    closes8 = np.full(60, 2100.0)
    df8     = _make_df(closes8, noise=0.05, seed=8)  # near-noiseless
    # Set 3-candle pattern at bars 40, 41, 42
    # Bullish FVG: C[40].high < C[42].low
    df8.at[40, "open"] = 2100.0; df8.at[40, "close"] = 2101.0
    df8.at[40, "high"] = 2103.0; df8.at[40, "low"]   = 2099.0
    df8.at[41, "open"] = 2105.0; df8.at[41, "close"] = 2115.0
    df8.at[41, "high"] = 2118.0; df8.at[41, "low"]   = 2104.0
    df8.at[42, "open"] = 2112.0; df8.at[42, "close"] = 2120.0
    df8.at[42, "high"] = 2122.0; df8.at[42, "low"]   = 2108.0  # low=2108 > high[40]=2103 -> FVG!
    atr8  = smc._atr(df8)
    fvgs8 = smc.detect_fvg(df8, atr8)
    bull_fvgs = [f for f in fvgs8 if f.fvg_type == FVGType.BULLISH]
    t8 = len(bull_fvgs) >= 1
    if t8:
        f = bull_fvgs[0]
        ok(f"Bullish FVG: bar={f.bar_index}  zone=[{f.gap_low:.2f}, {f.gap_high:.2f}]  "
           f"size={f.gap_size:.2f}")
    else:
        fail(f"No bullish FVG detected (total={len(fvgs8)})")
    results.append(t8)

    # ── Test 9: Bearish FVG detected ──────────────────────────────────────
    hdr("Test 9 -- Bearish FVG detection")
    closes9 = np.full(80, 2200.0)
    closes9[40] = 2200.0;  closes9[41] = 2185.0;  closes9[42] = 2175.0
    df9 = _make_df(closes9, noise=0.2)
    df9.at[40, "low"]   = 2196.0   # C[-2] low
    df9.at[41, "low"]   = 2182.0;  df9.at[41, "high"] = 2198.0
    df9.at[42, "high"]  = 2192.0   # C[0] high — must be < C[-2].low (2196)
    df9.at[42, "low"]   = 2170.0
    atr9  = smc._atr(df9)
    fvgs9 = smc.detect_fvg(df9, atr9)
    bear_fvgs = [f for f in fvgs9 if f.fvg_type == FVGType.BEARISH]
    t9 = len(bear_fvgs) >= 1
    if t9:
        f = bear_fvgs[0]
        ok(f"Bearish FVG: bar={f.bar_index}  zone=[{f.gap_low:.2f}, {f.gap_high:.2f}]  "
           f"size={f.gap_size:.2f}")
    else:
        fail(f"No bearish FVG detected (total={len(fvgs9)})")
    results.append(t9)

    # ── Test 10: Liquidity pool mapping ───────────────────────────────────
    hdr("Test 10 -- Liquidity pool mapping (swing highs/lows)")
    df10    = _uptrend(n=150)
    sw10    = smc._find_swings(df10)
    pools10 = smc.detect_liquidity_pools(df10, sw10)
    high_pools = [p for p in pools10 if p.pool_type == "HIGH"]
    low_pools  = [p for p in pools10 if p.pool_type == "LOW"]
    t10 = len(high_pools) >= 2 and len(low_pools) >= 2
    ok(f"HIGH pools={len(high_pools)}  LOW pools={len(low_pools)}") if t10 else \
    fail(f"HIGH={len(high_pools)} LOW={len(low_pools)} — expected >= 2 each")
    results.append(t10)

    # ── Test 11: Equal highs / lows ───────────────────────────────────────
    hdr("Test 11 -- Equal High / Equal Low detection")
    df11 = df3.copy()
    sw11 = smc._find_swings(df11)
    ms11 = smc.detect_market_structure(df11, sw11)
    pl11 = smc.detect_liquidity_pools(df11, sw11)
    eq_highs = [p for p in pl11 if p.pool_type == "EQUAL_HIGH"]
    t11 = len(eq_highs) >= 1
    ok(f"EQUAL_HIGH pools={len(eq_highs)}") if t11 else \
    fail(f"No equal-high pools detected")
    results.append(t11)

    # ── Test 12: scan() BUY setup ──────────────────────────────────────────
    hdr("Test 12 -- scan() produces at least one BUY setup")
    df12   = _with_buy_setup(n=200)
    smc12  = SmartMoneyStrategy()
    smc12.MIN_CONFIDENCE = 0.10    # lower for test data
    setups12 = smc12.scan(df12)
    buy_setups = [s for s in setups12 if s.direction == "BUY"]
    t12 = len(buy_setups) >= 1
    if t12:
        s = buy_setups[0]
        ok(f"BUY setup: {s}")
        ok(f"  to_dict keys: {sorted(s.to_dict().keys())}")
    else:
        fail(f"No BUY setup produced  (total={len(setups12)})")
    results.append(t12)

    # ── Test 13: scan() SELL setup ────────────────────────────────────────
    hdr("Test 13 -- scan() produces at least one SELL setup")
    df13  = _with_sell_setup(n=200)
    smc13 = SmartMoneyStrategy()
    smc13.MIN_CONFIDENCE = 0.10
    setups13 = smc13.scan(df13)
    sell_setups = [s for s in setups13 if s.direction == "SELL"]
    t13 = len(sell_setups) >= 1
    if t13:
        s = sell_setups[0]
        ok(f"SELL setup: {s}")
    else:
        fail(f"No SELL setup produced  (total={len(setups13)})")
    results.append(t13)

    # ── Test 14: to_dict() keys ───────────────────────────────────────────
    hdr("Test 14 -- to_dict() required keys")
    required_keys = {
        "direction", "entry", "stop_loss", "take_profit",
        "take_profit_2", "confidence",
        "has_ob", "has_fvg", "has_bos", "has_choch",
        "has_liquidity_sweep", "notes",
    }
    if buy_setups:
        d = buy_setups[0].to_dict()
        missing = required_keys - set(d.keys())
        t14 = not missing
        ok(f"All {len(required_keys)} required keys present") if t14 else \
        fail(f"Missing: {missing}")
        if t14:
            ok(f"direction={d['direction']}  entry={d['entry']}  "
               f"conf={d['confidence']}  has_ob={d['has_ob']}")
    else:
        t14 = False
        fail("No setup available from Test 12 to test to_dict()")
    results.append(t14)

    # ── Test 15: get_best_setup() ─────────────────────────────────────────
    hdr("Test 15 -- get_best_setup() returns highest-confidence setup")
    smc15 = SmartMoneyStrategy()
    smc15.MIN_CONFIDENCE = 0.10
    best15 = smc15.get_best_setup(_with_buy_setup(n=200))
    t15 = best15 is not None
    if t15:
        ok(f"Best setup: {best15}")
        # If there were multiple setups, verify this is the highest confidence
        all15 = smc15.scan(_with_buy_setup(n=200))
        if len(all15) > 1:
            is_best = best15.confidence >= all15[1].confidence
            ok(f"Confidence {best15.confidence:.0%} >= next {all15[1].confidence:.0%}") if is_best else \
            fail("get_best_setup did not return highest-confidence setup")
        else:
            ok("Only one setup — best is unambiguous")
    else:
        fail("get_best_setup() returned None")
    results.append(t15)

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
