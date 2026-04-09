"""
risk/position_sizing_test.py
==============================
Self-contained tests for PositionSizer — no MT5, no Settings required.

Scenarios tested
----------------
1.  Basic BUY — standard $10k, 1%, $10 SL distance -> 0.10 lots
2.  Basic SELL — same parameters, opposite direction
3.  Risk amount — dollar amount check
4.  Lot flooring — tiny account raises to MIN_LOT
5.  Lot capping — massive risk hits MAX_LOT ceiling
6.  R:R TP1 only — reward ratio with single target
7.  R:R TP1 + TP2 — both targets, primary = TP2
8.  Invalid: entry == stop (zero distance)
9.  Invalid: negative balance
10. Invalid: TP wrong side of entry
11. monetary_value_per_point  — $1 per point per lot for XAUUSD
12. lot_size_only() helper
13. max_lots_for_balance() breakdown
14. to_dict() keys
15. Real-world example — $5k account, 0.5%, $15 SL
16. Lot floor / rounding — verify floor-down, not nearest-round

Run from the project root:
    cd gold_trading_bot
    python risk/position_sizing_test.py
"""

from __future__ import annotations

import sys

if __name__ == "__main__":
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from risk.position_sizing import (
    PositionSizer,
    PositionSize,
    PositionSizingError,
    XAUUSD_CONTRACT_SIZE,
    XAUUSD_POINT_SIZE,
    XAUUSD_POINT_VALUE_PER_LOT,
)

# ── ANSI colours ──────────────────────────────────────────────────────────────
GREEN = "\033[92m"; RED = "\033[91m"; CYAN = "\033[96m"
RESET = "\033[0m";  BOLD = "\033[1m"

def ok(msg):   print(f"  {GREEN}PASS{RESET}  {msg}")
def fail(msg): print(f"  {RED}FAIL{RESET}  {msg}")
def hdr(msg):  print(f"\n{BOLD}{CYAN}{'-'*60}{RESET}\n{BOLD}{CYAN}  {msg}{RESET}\n{BOLD}{CYAN}{'-'*60}{RESET}")


def approx(a: float, b: float, tol: float = 1e-4) -> bool:
    """Return True if a is within tol of b."""
    return abs(a - b) <= tol


def main() -> int:
    print(f"\n{BOLD}{'='*60}")
    print("  PositionSizer -- XAUUSD Tests")
    print(f"{'='*60}{RESET}")

    # Standard sizer (no Settings dependency)
    sizer = PositionSizer()
    sizer.min_lot = 0.01
    sizer.max_lot = 5.00

    results = []

    # ── Test 1: Basic BUY ─────────────────────────────────────────────────
    hdr("Test 1 -- Basic BUY  ($10k, 1%, entry=2400, SL=2390)")
    # Formula: risk=$100 / (stop=$10 x 100oz) = 0.10 lots
    ps = sizer.calculate(entry_price=2400.00, stop_loss=2390.00,
                         account_balance=10_000.00, risk_pct=1.0)
    print(f"  {ps}")
    t1 = (
        ps.direction == "BUY" and
        ps.lot_size == 0.10 and
        approx(ps.stop_distance_usd, 10.00) and
        approx(ps.risk_amount_usd, 100.00)  # 0.10 lot × 100 oz × $10 = $100
    )
    ok(f"lots={ps.lot_size}  risk=${ps.risk_amount_usd}  dir={ps.direction}") if t1 else \
    fail(f"lots={ps.lot_size} (exp 0.10)  risk=${ps.risk_amount_usd} (exp $100.00)  dir={ps.direction}")
    results.append(t1)

    # ── Test 2: Basic SELL ────────────────────────────────────────────────
    hdr("Test 2 -- Basic SELL ($10k, 1%, entry=2400, SL=2410)")
    ps2 = sizer.calculate(entry_price=2400.00, stop_loss=2410.00,
                          account_balance=10_000.00, risk_pct=1.0)
    print(f"  {ps2}")
    t2 = (
        ps2.direction == "SELL" and
        ps2.lot_size == 0.10 and
        approx(ps2.risk_amount_usd, 100.00)
    )
    ok(f"lots={ps2.lot_size}  risk=${ps2.risk_amount_usd}  dir={ps2.direction}") if t2 else \
    fail(f"lots={ps2.lot_size} (exp 0.10)  dir={ps2.direction} (exp SELL)")
    results.append(t2)

    # ── Test 3: Risk amount ───────────────────────────────────────────────
    hdr("Test 3 -- risk_amount() standalone")
    r3 = sizer.risk_amount(account_balance=25_000, risk_pct=0.5)
    t3 = approx(r3, 125.00)
    ok(f"risk_amount(25000, 0.5%) = ${r3}") if t3 else fail(f"Got ${r3}, expected $125.00")
    results.append(t3)

    # ── Test 4: Lot flooring (tiny account) ──────────────────────────────
    hdr("Test 4 -- Lot floored to MIN_LOT (tiny account, huge stop)")
    # $500 × 0.5% = $2.50 risk; SL = $50; raw_lot = 2.5 / 5000 = 0.0005 -> floor to 0.01
    ps4 = sizer.calculate(entry_price=2400.00, stop_loss=2350.00,
                          account_balance=500.00, risk_pct=0.5)
    print(f"  {ps4}")
    t4 = ps4.lot_size == 0.01 and ps4.lot_floored
    ok(f"lot={ps4.lot_size} floored={ps4.lot_floored}") if t4 else \
    fail(f"lot={ps4.lot_size} floored={ps4.lot_floored} (expected 0.01 + floored=True)")
    results.append(t4)

    # ── Test 5: Lot capping (max_lot) ─────────────────────────────────────
    hdr("Test 5 -- Lot capped to MAX_LOT")
    # $1M × 5% = $50,000 risk; SL = $1; raw_lot = 50000 / 100 = 500 -> cap to 5.0
    ps5 = sizer.calculate(entry_price=2400.00, stop_loss=2399.00,
                          account_balance=1_000_000.00, risk_pct=5.0)
    print(f"  {ps5}")
    t5 = ps5.lot_size == 5.00 and ps5.lot_capped
    ok(f"lot={ps5.lot_size} capped={ps5.lot_capped}") if t5 else \
    fail(f"lot={ps5.lot_size} capped={ps5.lot_capped} (expected 5.00 + capped=True)")
    results.append(t5)

    # ── Test 6: R:R with TP1 only ─────────────────────────────────────────
    hdr("Test 6 -- R:R ratio with TP1 only")
    # SL=-10, TP1=+20 -> R:R = 2.0
    ps6 = sizer.calculate(entry_price=2400.00, stop_loss=2390.00,
                          take_profit_1=2420.00,
                          account_balance=10_000.00, risk_pct=1.0)
    print(f"  {ps6}")
    t6 = (
        ps6.rr_ratio_tp1 == 2.0 and
        ps6.rr_ratio_tp2 is None and
        ps6.rr_ratio == 2.0
    )
    ok(f"rr_tp1={ps6.rr_ratio_tp1}  rr_tp2={ps6.rr_ratio_tp2}  primary={ps6.rr_ratio}") if t6 else \
    fail(f"rr_tp1={ps6.rr_ratio_tp1} (exp 2.0)  rr={ps6.rr_ratio} (exp 2.0)")
    results.append(t6)

    # ── Test 7: R:R TP1 + TP2 ────────────────────────────────────────────
    hdr("Test 7 -- R:R ratio with both TP1 and TP2")
    # SL=-10, TP1=+20 (2:1), TP2=+40 (4:1)
    ps7 = sizer.calculate(entry_price=2400.00, stop_loss=2390.00,
                          take_profit_1=2420.00, take_profit_2=2440.00,
                          account_balance=10_000.00, risk_pct=1.0)
    print(f"  {ps7}")
    t7 = (
        ps7.rr_ratio_tp1 == 2.0 and
        ps7.rr_ratio_tp2 == 4.0 and
        ps7.rr_ratio == 4.0   # primary = TP2
    )
    ok(f"rr_tp1={ps7.rr_ratio_tp1}  rr_tp2={ps7.rr_ratio_tp2}  primary={ps7.rr_ratio}") if t7 else \
    fail(f"rr_tp1={ps7.rr_ratio_tp1} rr_tp2={ps7.rr_ratio_tp2} primary={ps7.rr_ratio}")
    results.append(t7)

    # ── Test 8: Invalid — entry == stop ───────────────────────────────────
    hdr("Test 8 -- PositionSizingError when entry == stop_loss")
    t8 = False
    try:
        sizer.calculate(entry_price=2400.00, stop_loss=2400.00,
                        account_balance=10_000.00, risk_pct=1.0)
        fail("Expected PositionSizingError — none raised")
    except PositionSizingError as e:
        ok(f"PositionSizingError raised: {str(e)[:60]}...")
        t8 = True
    results.append(t8)

    # ── Test 9: Invalid — negative balance ────────────────────────────────
    hdr("Test 9 -- PositionSizingError on negative balance")
    t9 = False
    try:
        sizer.calculate(entry_price=2400.00, stop_loss=2390.00,
                        account_balance=-1000.00, risk_pct=1.0)
        fail("Expected PositionSizingError — none raised")
    except PositionSizingError as e:
        ok(f"PositionSizingError raised: {str(e)[:60]}...")
        t9 = True
    results.append(t9)

    # ── Test 10: Invalid — TP on wrong side ───────────────────────────────
    hdr("Test 10 -- PositionSizingError: BUY trade with TP below entry")
    t10 = False
    try:
        sizer.calculate(entry_price=2400.00, stop_loss=2390.00,
                        take_profit_1=2380.00,  # BELOW entry -> invalid for BUY
                        account_balance=10_000.00, risk_pct=1.0)
        fail("Expected PositionSizingError — none raised")
    except PositionSizingError as e:
        ok(f"PositionSizingError raised: {str(e)[:60]}...")
        t10 = True
    results.append(t10)

    # ── Test 11: monetary_value_per_point ─────────────────────────────────
    hdr("Test 11 -- monetary_value_per_point ($1 per 0.01 per lot for XAUUSD)")
    # 0.10 lot x 100 oz x $0.01 = $0.10 per point
    v11 = sizer.monetary_value_per_point(0.10)
    t11 = approx(v11, 0.10)
    ok(f"0.10 lot -> ${v11:.4f}/point  (expected $0.1000)") if t11 else fail(f"Got ${v11}")
    # Also check 1.00 lot = $1.00/point
    v11b = sizer.monetary_value_per_point(1.00)
    t11b = approx(v11b, 1.00)
    ok(f"1.00 lot -> ${v11b:.4f}/point  (expected $1.0000)") if t11b else fail(f"Got ${v11b}")
    results.append(t11 and t11b)

    # ── Test 12: lot_size_only() helper ──────────────────────────────────
    hdr("Test 12 -- lot_size_only() quick helper")
    lot12 = sizer.lot_size_only(entry_price=2400.00, stop_loss=2390.00,
                                account_balance=10_000.00, risk_pct=1.0)
    t12 = lot12 == 0.10
    ok(f"lot_size_only() = {lot12}") if t12 else fail(f"Got {lot12}, expected 0.10")
    results.append(t12)

    # ── Test 13: max_lots_for_balance() ──────────────────────────────────
    hdr("Test 13 -- max_lots_for_balance() at 3 risk levels")
    breakdown = sizer.max_lots_for_balance(entry_price=2400.00, stop_loss=2390.00,
                                           account_balance=10_000.00)
    required_keys = {"conservative_0.5pct", "standard_1pct", "aggressive_2pct", "max_allowed"}
    t13 = required_keys.issubset(set(breakdown.keys()))
    if t13:
        ok(f"Keys: {sorted(breakdown.keys())}")
        for k, v in sorted(breakdown.items()):
            ok(f"  {k} = {v} lots")
    else:
        fail(f"Missing keys: {required_keys - set(breakdown.keys())}")
    results.append(t13)

    # ── Test 14: to_dict() ────────────────────────────────────────────────
    hdr("Test 14 -- to_dict() required keys")
    required = {
        "lot_size", "risk_amount_usd", "actual_risk_pct",
        "stop_distance_usd", "stop_distance_pts",
        "rr_ratio", "rr_ratio_tp1", "rr_ratio_tp2",
        "entry_price", "stop_loss", "take_profit_1", "take_profit_2",
        "direction", "monetary_value_1pt", "lot_capped", "lot_floored",
    }
    d14 = ps7.to_dict()
    missing = required - set(d14.keys())
    t14 = not missing
    ok(f"All {len(required)} keys present") if t14 else fail(f"Missing: {missing}")
    if t14:
        ok(f"lot={d14['lot_size']}  risk=${d14['risk_amount_usd']}  "
           f"rr={d14['rr_ratio']}  dir={d14['direction']}")
    results.append(t14)

    # ── Test 15: Real-world example ───────────────────────────────────────
    hdr("Test 15 -- Real-world: $5k account, 0.5%, SL=$15")
    # risk = $5000 x 0.5% = $25; lot = $25 / ($15 * 100) = 0.01666 -> floor to 0.01
    ps15 = sizer.calculate(entry_price=2350.00, stop_loss=2335.00,
                           take_profit_1=2380.00, take_profit_2=2410.00,
                           account_balance=5_000.00, risk_pct=0.5)
    print(f"  {ps15}")
    expected_lot15 = 0.01    # floor(0.01666, 0.01) = 0.01
    t15 = ps15.lot_size == expected_lot15
    ok(f"lot={ps15.lot_size} (exp {expected_lot15})  rr_tp1={ps15.rr_ratio_tp1}  rr_tp2={ps15.rr_ratio_tp2}") if t15 \
    else fail(f"lot={ps15.lot_size}, expected {expected_lot15}")
    results.append(t15)

    # ── Test 16: Lot floor precision (not nearest-round) ──────────────────
    hdr("Test 16 -- Floor to step (not round) — 0.179 -> 0.17, not 0.18")
    # Force a raw_lot of 0.179...:
    # lot = risk / (stop * 100)
    # 0.179 = risk / (stop * 100)  -> with stop=$5.58 and risk=$100 → 100/(5.58*100) ≈ 0.179
    ps16 = sizer.calculate(entry_price=2400.00, stop_loss=2394.42,
                           account_balance=10_000.00, risk_pct=1.0)
    # raw = 100 / (5.58 * 100) = 0.1792...; floor to 0.01 step -> 0.17
    print(f"  {ps16}  (raw_close_to=0.179)")
    t16 = ps16.lot_size <= ps16.lot_size  # floored means it can't be higher than raw
    # More precise check: compare actual risk to intended risk
    actual_slots = ps16.risk_amount_usd
    intended     = 100.00
    t16 = actual_slots <= intended + 0.01   # actual risk must not EXCEED intended
    ok(f"lot={ps16.lot_size}  actual_risk=${ps16.risk_amount_usd:.2f} <= intended=${intended:.2f}") if t16 else \
    fail(f"Actual risk ${ps16.risk_amount_usd:.2f} exceeds intended ${intended:.2f}")
    results.append(t16)

    # ── Summary ───────────────────────────────────────────────────────────
    hdr("Summary")
    passed = sum(results)
    total  = len(results)
    for i, r in enumerate(results, 1):
        s = f"{GREEN}PASS{RESET}" if r else f"{RED}FAIL{RESET}"
        print(f"  [{s}]  Test {i}")
    print()

    # Print the contract spec for reference
    print(f"  {CYAN}XAUUSD contract spec:{RESET}")
    print(f"    Contract size     : {XAUUSD_CONTRACT_SIZE:.0f} oz/lot")
    print(f"    Point size        : ${XAUUSD_POINT_SIZE}")
    print(f"    Point value/lot   : ${XAUUSD_POINT_VALUE_PER_LOT:.2f}")
    print()

    if passed == total:
        print(f"{GREEN}{BOLD}All {total} tests passed{RESET}")
        return 0
    else:
        print(f"{RED}{BOLD}{total - passed}/{total} tests FAILED{RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
