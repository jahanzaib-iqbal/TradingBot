"""
strategy/liquidity_map_test.py
================================
Self-contained tests for LiquidityMapper and LiquidityMap.
No MT5, Telegram, or live data required.

Tests
-----
 1.  ZoneType enum has all 10 expected values
 2.  LiquidityZone.to_dict() returns required keys
 3.  LiquidityZone.is_below is complement of is_above
 4.  LiquidityMap.nearest_above/below return correct zones
 5.  LiquidityMap.zones_between() filters correctly
 6.  LiquidityMap.summary string is non-empty
 7.  LiquidityMapper constructs with default settings
 8.  _compute_atr returns positive on valid OHLCV
 9.  _find_swings detects clear swing highs correctly
10.  _detect_equal_highs: clustered swing highs → EQUAL_HIGHS zones
11.  _detect_equal_lows:  clustered swing lows  → EQUAL_LOWS  zones
12.  _detect_session_levels: returns SESSION_HIGH + SESSION_LOW
13.  _detect_weekly_levels: returns WEEKLY_HIGH + WEEKLY_LOW
14.  _detect_round_numbers: returns level at nearest $50 above and below
15.  _detect_swing_levels: at least one SWING_HIGH and SWING_LOW detected
16.  _merge_nearby_zones: overlapping zones collapsed into one
17.  map() returns a valid LiquidityMap with zones above and below
18.  liquidity_above sorted nearest first (ascending price)
19.  liquidity_below sorted nearest first (descending price)
20.  is_entry_safe() blocks entry too close to strong zone
21.  is_entry_safe() approves clear entry away from zones
22.  get_nearest_tp_target() returns valid TP price

Run:
    cd gold_trading_bot
    python strategy/liquidity_map_test.py
"""

from __future__ import annotations

import sys
import os
from datetime import datetime, timezone, timedelta

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

# ── ANSI colours ─────────────────────────────────────────────────────────────
GREEN = "\033[92m"; RED = "\033[91m"; CYAN = "\033[96m"
RESET = "\033[0m";  BOLD = "\033[1m"

def ok(msg):   print(f"  {GREEN}PASS{RESET}  {msg}")
def fail(msg): print(f"  {RED}FAIL{RESET}  {msg}")
def hdr(msg):  print(f"\n{BOLD}{CYAN}{'-'*60}{RESET}\n{BOLD}{CYAN}  {msg}{RESET}\n{BOLD}{CYAN}{'-'*60}{RESET}")


# ─────────────────────────────────────────────────────────────────────────────
# Test helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cfg():
    from config.settings import Settings
    return Settings()


def _make_ohlcv(
    n:            int   = 200,
    base_price:   float = 2350.0,
    trend:        float = 0.0,
    noise:        float = 5.0,
    seed:         int   = 42,
    equal_highs_at: float | None = None,
    equal_lows_at:  float | None = None,
    add_time:     bool  = True,
) -> pd.DataFrame:
    """
    Synthetic OHLCV data with optional equal-high / equal-low clusters.

    equal_highs_at: if set, inserts 3 swing highs pinned to this price
    equal_lows_at:  if set, inserts 3 swing lows pinned to this price
    """
    rng    = np.random.default_rng(seed)
    prices = base_price + trend * np.arange(n) + rng.normal(0, noise, n).cumsum()

    highs  = prices + rng.uniform(2, 8, n)
    lows   = prices - rng.uniform(2, 8, n)
    opens  = prices + rng.uniform(-3, 3, n)
    vols   = rng.integers(200, 1000, n).astype(float)

    # Insert equal highs (3 clear swing peaks at the same price)
    if equal_highs_at is not None:
        for idx in [int(n * 0.25), int(n * 0.50), int(n * 0.75)]:
            idx = max(2, min(idx, n - 3))   # guard bounds
            highs[idx]       = equal_highs_at
            highs[idx - 1]   = equal_highs_at - 3
            highs[idx + 1]   = equal_highs_at - 3
            prices[idx]      = equal_highs_at - 1

    # Insert equal lows
    if equal_lows_at is not None:
        for idx in [int(n * 0.28), int(n * 0.53), int(n * 0.78)]:
            idx = max(2, min(idx, n - 3))
            lows[idx]        = equal_lows_at
            lows[idx - 1]    = equal_lows_at + 3
            lows[idx + 1]    = equal_lows_at + 3
            prices[idx]      = equal_lows_at + 1

    df = pd.DataFrame({
        "open":        opens,
        "high":        highs,
        "low":         lows,
        "close":       prices,
        "tick_volume": vols,
    })

    if add_time:
        start = datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc)
        df["time"] = [start + timedelta(hours=i) for i in range(n)]

    return df


def _mapper(cfg=None, **kwargs):
    from strategy.liquidity_map import LiquidityMapper
    return LiquidityMapper(cfg or _cfg(), **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    print(f"\n{BOLD}{'='*60}")
    print("  LiquidityMapper — Institutional Liquidity Map Tests")
    print(f"{'='*60}{RESET}")

    from strategy.liquidity_map import (
        LiquidityMapper, LiquidityMap, LiquidityZone, ZoneType,
    )

    results = []
    cfg     = _cfg()

    # ── Test 1: ZoneType enum completeness ───────────────────────────────────
    hdr("Test 1 -- ZoneType has all 10 expected values")
    expected_types = {
        "EQUAL_HIGHS", "EQUAL_LOWS",
        "SESSION_HIGH", "SESSION_LOW",
        "WEEKLY_HIGH",  "WEEKLY_LOW",
        "ROUND_NUMBER", "HALF_NUMBER",
        "SWING_HIGH",   "SWING_LOW",
    }
    actual_types = {z.value for z in ZoneType}
    t1 = expected_types == actual_types
    ok(f"All 10 ZoneType values present") if t1 else \
    fail(f"Missing: {expected_types - actual_types}  Extra: {actual_types - expected_types}")
    results.append(t1)

    # ── Test 2: LiquidityZone.to_dict() keys ──────────────────────────────────
    hdr("Test 2 -- LiquidityZone.to_dict() has required keys")
    zone = LiquidityZone(
        price=2350.0, zone_high=2351.0, zone_low=2349.0,
        zone_type=ZoneType.EQUAL_HIGHS, strength=0.85,
        touches=3, age_bars=10, description="test", is_above=True,
    )
    d2 = zone.to_dict()
    required2 = {"price", "zone_high", "zone_low", "type", "strength", "touches", "age_bars", "description"}
    miss2 = required2 - set(d2.keys())
    t2 = not miss2
    ok(f"All {len(required2)} keys present: {list(d2.keys())}") if t2 else \
    fail(f"Missing: {miss2}")
    results.append(t2)

    # ── Test 3: is_below is complement of is_above ────────────────────────────
    hdr("Test 3 -- LiquidityZone.is_below is complement of is_above")
    z_above = LiquidityZone(price=2360.0, zone_high=2361.0, zone_low=2359.0,
                            zone_type=ZoneType.SWING_HIGH, strength=0.7, is_above=True)
    z_below = LiquidityZone(price=2340.0, zone_high=2341.0, zone_low=2339.0,
                            zone_type=ZoneType.SWING_LOW, strength=0.7, is_above=False)
    t3 = (not z_above.is_below and z_below.is_below and
          z_above.is_above and not z_below.is_above)
    ok(f"is_above/is_below complement works") if t3 else fail("Complement broken")
    results.append(t3)

    # ── Test 4: LiquidityMap.nearest_above / nearest_below ───────────────────
    hdr("Test 4 -- LiquidityMap.nearest_above / nearest_below")
    lmap4 = LiquidityMap(
        liquidity_above=[
            LiquidityZone(2360.0, 2361.0, 2359.0, ZoneType.ROUND_NUMBER, 0.75, is_above=True),
            LiquidityZone(2400.0, 2401.0, 2399.0, ZoneType.ROUND_NUMBER, 0.75, is_above=True),
        ],
        liquidity_below=[
            LiquidityZone(2340.0, 2341.0, 2339.0, ZoneType.ROUND_NUMBER, 0.75, is_above=False),
            LiquidityZone(2300.0, 2301.0, 2299.0, ZoneType.ROUND_NUMBER, 0.75, is_above=False),
        ],
        current_price=2350.0,
    )
    t4 = (lmap4.nearest_above is not None and abs(lmap4.nearest_above.price - 2360.0) < 1e-3 and
          lmap4.nearest_below is not None and abs(lmap4.nearest_below.price - 2340.0) < 1e-3)
    ok(f"nearest_above={lmap4.nearest_above.price:.2f}  nearest_below={lmap4.nearest_below.price:.2f}") if t4 else \
    fail(f"above={lmap4.nearest_above}  below={lmap4.nearest_below}")
    results.append(t4)

    # ── Test 5: zones_between ─────────────────────────────────────────────────
    hdr("Test 5 -- LiquidityMap.zones_between() filters correctly")
    between5 = lmap4.zones_between(2350.0, 2370.0)
    t5 = len(between5) == 1 and between5[0].price == 2360.0
    ok(f"zones_between(2350, 2370) = {[z.price for z in between5]}") if t5 else \
    fail(f"Got {[z.price for z in between5]}")
    results.append(t5)

    # ── Test 6: summary string ────────────────────────────────────────────────
    hdr("Test 6 -- LiquidityMap.summary is non-empty")
    s6 = lmap4.summary
    t6 = len(s6) > 10 and "above" in s6 and "below" in s6
    ok(f"summary = '{s6}'") if t6 else fail(f"Unexpected summary: '{s6}'")
    results.append(t6)

    # ── Test 7: LiquidityMapper constructs ───────────────────────────────────
    hdr("Test 7 -- LiquidityMapper constructs with default settings")
    mapper = _mapper(cfg)
    t7 = repr(mapper.cfg) is not None
    ok(f"LiquidityMapper constructed: equal_tol={mapper.equal_tol:.4f}  swing_lb={mapper.swing_lookback}")
    results.append(True)

    # ── Test 8: _compute_atr positive on valid data ───────────────────────────
    hdr("Test 8 -- _compute_atr returns positive on valid OHLCV")
    df8  = _make_ohlcv(60)
    atr8 = LiquidityMapper._compute_atr(df8)
    t8   = 0.1 < atr8 < 100.0
    ok(f"ATR={atr8:.3f}") if t8 else fail(f"ATR={atr8} out of range")
    results.append(t8)

    # ── Test 9: _find_swings detects clear highs ──────────────────────────────
    hdr("Test 9 -- _find_swings detects clear swing highs correctly")
    df9 = _make_ohlcv(100, equal_highs_at=2380.0)
    mapper9 = LiquidityMapper(cfg, swing_lookback=2)
    swings9 = mapper9._find_swings(df9, "high", "max")
    # Should detect swings near bars 40, 80, 130 (clamped to 99 for n=100)
    t9 = len(swings9) >= 2
    ok(f"Found {len(swings9)} swing highs") if t9 else fail(f"Only {len(swings9)} swing highs found")
    results.append(t9)

    # ── Test 10: equal_highs detection ───────────────────────────────────────
    hdr("Test 10 -- _detect_equal_highs detects clustered swing highs")
    df10 = _make_ohlcv(200, equal_highs_at=2380.0)
    mapper10 = LiquidityMapper(cfg, equal_tolerance=0.10, swing_lookback=2)
    current10 = 2350.0
    tol10     = current10 * 0.001
    eq_highs  = mapper10._detect_equal_highs(df10, current10, tol10 * 10)
    t10 = any(z.zone_type == ZoneType.EQUAL_HIGHS for z in eq_highs)
    ok(f"Detected {len(eq_highs)} equal-high zones  types={[z.zone_type.value for z in eq_highs]}") if t10 else \
    fail(f"No EQUAL_HIGHS detected  (swing highs: {mapper10._find_swings(df10,'high','max')})")
    results.append(t10)

    # ── Test 11: equal_lows detection ────────────────────────────────────────
    hdr("Test 11 -- _detect_equal_lows detects clustered swing lows")
    df11 = _make_ohlcv(200, equal_lows_at=2320.0)
    eq_lows = mapper10._detect_equal_lows(df11, 2350.0, tol10 * 10)
    t11 = any(z.zone_type == ZoneType.EQUAL_LOWS for z in eq_lows)
    ok(f"Detected {len(eq_lows)} equal-low zones") if t11 else \
    fail(f"No EQUAL_LOWS detected")
    results.append(t11)

    # ── Test 12: session levels ───────────────────────────────────────────────
    hdr("Test 12 -- _detect_session_levels returns SESSION_HIGH + SESSION_LOW")
    df12    = _make_ohlcv(200, base_price=2350.0, add_time=True)
    mapper12 = LiquidityMapper(cfg)
    sess12   = mapper12._detect_session_levels(df12, 2350.0)
    types12  = {z.zone_type for z in sess12}
    t12 = ZoneType.SESSION_HIGH in types12 and ZoneType.SESSION_LOW in types12
    ok(f"Session zones: {[z.zone_type.value for z in sess12]}") if t12 else \
    fail(f"Got: {[z.zone_type.value for z in sess12]}")
    results.append(t12)

    # ── Test 13: weekly levels ────────────────────────────────────────────────
    hdr("Test 13 -- _detect_weekly_levels returns WEEKLY_HIGH + WEEKLY_LOW")
    df13   = _make_ohlcv(300, base_price=2350.0, add_time=True)
    week13 = mapper12._detect_weekly_levels(df13, 2350.0)
    types13 = {z.zone_type for z in week13}
    t13 = ZoneType.WEEKLY_HIGH in types13 and ZoneType.WEEKLY_LOW in types13
    ok(f"Weekly zones: {[z.zone_type.value for z in week13]}") if t13 else \
    fail(f"Got: {[z.zone_type.value for z in week13]}")
    results.append(t13)

    # ── Test 14: round numbers ────────────────────────────────────────────────
    hdr("Test 14 -- _detect_round_numbers returns $50 levels above and below")
    atr14  = 10.0
    round14 = mapper12._detect_round_numbers(2355.0, atr14)
    above14 = [z for z in round14 if z.is_above]
    below14 = [z for z in round14 if not z.is_above]
    # Nearest $50 above 2355 = 2400; nearest below = 2350
    t14_above = any(z.price == 2400.0 for z in above14)
    t14_below = any(z.price == 2350.0 for z in below14)
    t14 = t14_above and t14_below
    ok(f"Rounds above={[z.price for z in above14[:3]]}  below={[z.price for z in below14[:3]]}") if t14 else \
    fail(f"above has 2400={t14_above}  below has 2350={t14_below}")
    results.append(t14)

    # ── Test 15: swing_levels ─────────────────────────────────────────────────
    hdr("Test 15 -- _detect_swing_levels returns SWING_HIGH and SWING_LOW")
    df15    = _make_ohlcv(150, base_price=2350.0)
    sw15    = mapper12._detect_swing_levels(df15, 2350.0, 1.5)
    types15 = {z.zone_type for z in sw15}
    t15 = ZoneType.SWING_HIGH in types15 and ZoneType.SWING_LOW in types15
    ok(f"Swing zones: {len([z for z in sw15 if z.zone_type==ZoneType.SWING_HIGH])} highs  "
       f"{len([z for z in sw15 if z.zone_type==ZoneType.SWING_LOW])} lows") if t15 else \
    fail(f"Got types: {types15}")
    results.append(t15)

    # ── Test 16: merge_nearby_zones ───────────────────────────────────────────
    hdr("Test 16 -- _merge_nearby_zones collapses overlapping zones")
    z_a = LiquidityZone(2370.0, 2371.5, 2368.5, ZoneType.SWING_HIGH,    0.65, is_above=True)
    z_b = LiquidityZone(2370.5, 2372.0, 2369.0, ZoneType.ROUND_NUMBER,  0.75, is_above=True)
    z_c = LiquidityZone(2400.0, 2402.0, 2398.0, ZoneType.WEEKLY_HIGH,   0.85, is_above=True)
    merged16 = mapper12._merge_nearby_zones([z_a, z_b, z_c], tol_pts=2.0)
    t16 = len(merged16) == 2   # z_a + z_b merge into 1, z_c stays
    ok(f"Merged {3} zones → {len(merged16)} zones  prices={[z.price for z in merged16]}") if t16 else \
    fail(f"Expected 2 merged zones, got {len(merged16)}: {merged16}")
    results.append(t16)

    # ── Test 17: map() returns valid LiquidityMap ─────────────────────────────
    hdr("Test 17 -- map() returns valid LiquidityMap with zones")
    df17   = _make_ohlcv(300, base_price=2350.0, add_time=True,
                          equal_highs_at=2380.0, equal_lows_at=2320.0)
    lmap17 = mapper12.map(df17)
    t17 = (
        isinstance(lmap17, LiquidityMap) and
        len(lmap17.liquidity_above) >= 1 and
        len(lmap17.liquidity_below) >= 1 and
        lmap17.current_price > 0
    )
    ok(f"LiquidityMap: {lmap17.summary}") if t17 else \
    fail(f"above={len(lmap17.liquidity_above)}  below={len(lmap17.liquidity_below)}  price={lmap17.current_price}")
    results.append(t17)

    # ── Test 18: above zones sorted ascending (nearest first) ────────────────
    hdr("Test 18 -- liquidity_above sorted ascending (nearest first)")
    prices_above = [z.price for z in lmap17.liquidity_above]
    t18 = prices_above == sorted(prices_above)
    ok(f"liquidity_above prices (first 5): {prices_above[:5]}") if t18 else \
    fail(f"Not sorted ascending: {prices_above[:5]}")
    results.append(t18)

    # ── Test 19: below zones sorted descending (nearest first) ───────────────
    hdr("Test 19 -- liquidity_below sorted descending (nearest first)")
    prices_below = [z.price for z in lmap17.liquidity_below]
    t19 = prices_below == sorted(prices_below, reverse=True)
    ok(f"liquidity_below prices (first 5): {prices_below[:5]}") if t19 else \
    fail(f"Not sorted descending: {prices_below[:5]}")
    results.append(t19)

    # ── Test 20: is_entry_safe BLOCKS entry too close to strong zone ──────────
    hdr("Test 20 -- is_entry_safe() blocks entry too close to strong zone")
    lmap20 = LiquidityMap(
        liquidity_above=[
            LiquidityZone(2357.0, 2358.0, 2356.0, ZoneType.EQUAL_HIGHS, 0.90, is_above=True),
        ],
        liquidity_below=[],
        current_price=2350.0,
    )
    safe20, reason20 = mapper12.is_entry_safe(
        entry_price=2354.0, direction="BUY",
        lmap=lmap20, atr=10.0, min_distance_pts=5.0
    )
    t20 = not safe20   # should be blocked (zone at 2357 is only 3 pts above entry)
    ok(f"Blocked: {reason20[:60]}") if t20 else fail(f"Should be blocked; is_safe={safe20}")
    results.append(t20)

    # ── Test 21: is_entry_safe APPROVES clear entry ───────────────────────────
    hdr("Test 21 -- is_entry_safe() approves entry far from strong zones")
    lmap21 = LiquidityMap(
        liquidity_above=[
            LiquidityZone(2400.0, 2401.0, 2399.0, ZoneType.ROUND_NUMBER, 0.75, is_above=True),
        ],
        liquidity_below=[],
        current_price=2350.0,
    )
    safe21, reason21 = mapper12.is_entry_safe(
        entry_price=2355.0, direction="BUY",
        lmap=lmap21, atr=10.0, min_distance_pts=5.0
    )
    t21 = safe21  # zone at 2400 is 45 pts away — safe
    ok(f"Approved: {reason21}") if t21 else fail(f"Should be approved; reason={reason21}")
    results.append(t21)

    # ── Test 22: get_nearest_tp_target returns valid TP price ─────────────────
    hdr("Test 22 -- get_nearest_tp_target() returns valid TP price")
    lmap22 = LiquidityMap(
        liquidity_above=[
            LiquidityZone(2380.0, 2381.0, 2379.0, ZoneType.EQUAL_HIGHS,  0.90, is_above=True),
            LiquidityZone(2400.0, 2401.0, 2399.0, ZoneType.ROUND_NUMBER,  0.75, is_above=True),
        ],
        liquidity_below=[
            LiquidityZone(2330.0, 2331.0, 2329.0, ZoneType.EQUAL_LOWS,   0.90, is_above=False),
        ],
        current_price=2355.0,
    )
    # BUY from 2355, SL 20 pts → need ≥ 1.5×20=30 pts above → TP ≥ 2385
    # Nearest qualifying zone: 2400 (2380 is only 25 pts away < 30)
    tp22_buy = mapper12.get_nearest_tp_target(
        entry_price=2355.0, direction="BUY",
        lmap=lmap22, min_rr=1.5, stop_distance=20.0,
    )
    # SELL from 2355, SL 20 pts → need ≥ 30 pts below → TP ≤ 2325
    # Only zone below is at 2330 (25 pts away < 30) → should return None
    tp22_sell = mapper12.get_nearest_tp_target(
        entry_price=2355.0, direction="SELL",
        lmap=lmap22, min_rr=1.5, stop_distance=20.0,
    )
    t22 = (
        tp22_buy is not None and abs(tp22_buy - 2400.0) < 1e-3 and
        tp22_sell is None   # 2330 is only 25 pts away, doesn't meet 30-pt minimum
    )
    ok(f"BUY TP={tp22_buy}  SELL TP={tp22_sell}  (correct)") if t22 else \
    fail(f"BUY TP={tp22_buy} (expected 2400)  SELL TP={tp22_sell} (expected None)")
    results.append(t22)

    # ── Summary ───────────────────────────────────────────────────────────────
    hdr("Summary")
    passed = sum(results)
    total  = len(results)

    # Print full map from test 17
    print("\n  -- Sample map() output (Test 17) --")
    lmap17.print_summary()

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
