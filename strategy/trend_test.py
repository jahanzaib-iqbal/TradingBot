"""
strategy/trend_test.py
=======================
Self-contained tests for TrendDetector using synthetic OHLCV data.
No MT5 connection required.

Scenarios tested
----------------
1. BULLISH STRONG   — sustained uptrend, price far above both EMAs
2. BULLISH WEAK     — price just crossed above both EMAs (early trend)
3. BEARISH STRONG   — sustained downtrend, price far below both EMAs
4. NEUTRAL          — price between EMA_50 and EMA_200
5. UNKNOWN          — insufficient bars
6. Golden Cross     — EMA_50 crossing above EMA_200
7. Death Cross      — EMA_50 crossing below EMA_200
8. add_emas()       — verify column names and values
9. get_ema_values() — quick EMA snapshot
10. allows_direction() — trade gate logic

Run from the project root:
    cd gold_trading_bot
    python strategy/trend_test.py
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone

if __name__ == "__main__":
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from strategy.trend_detection import TrendDetector, TrendDirection, TrendStrengthLabel

# ── Colour helpers (ASCII-safe for Windows cp1252) ───────────────────────────
GREEN = "\033[92m"; RED = "\033[91m"; CYAN = "\033[96m"
RESET = "\033[0m";  BOLD = "\033[1m"

def ok(msg):   print(f"  {GREEN}PASS{RESET}  {msg}")
def fail(msg): print(f"  {RED}FAIL{RESET}  {msg}")
def hdr(msg):  print(f"\n{BOLD}{CYAN}{'-'*60}{RESET}\n{BOLD}{CYAN}  {msg}{RESET}\n{BOLD}{CYAN}{'-'*60}{RESET}")

# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data builders
# ─────────────────────────────────────────────────────────────────────────────

def _make_df(closes: np.ndarray, noise: float = 0.5) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame from an array of close prices."""
    rng = np.random.default_rng(seed=99)
    n   = len(closes)
    hi  = closes + rng.uniform(noise * 0.5, noise * 1.5, n)
    lo  = closes - rng.uniform(noise * 0.5, noise * 1.5, n)
    op  = closes + rng.uniform(-noise * 0.3, noise * 0.3, n)
    hi  = np.maximum(hi, closes)
    lo  = np.minimum(lo, closes)
    op  = np.clip(op, lo, hi)

    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    times = [start + timedelta(hours=i) for i in range(n)]

    return pd.DataFrame({
        "time":        pd.to_datetime(times, utc=True),
        "open":        op.round(2),
        "high":        hi.round(2),
        "low":         lo.round(2),
        "close":       closes.round(2),
        "tick_volume": rng.integers(500, 2000, n),
    })


def _bullish_strong(n: int = 300) -> pd.DataFrame:
    """Price rises from 2000 to 2300 (50% above EMA200 by the end)."""
    return _make_df(np.linspace(2000, 2300, n), noise=2.0)


def _bullish_weak(n: int = 220) -> pd.DataFrame:
    """
    Price was flat at 2100 for the first 210 bars (to warm up EMAs),
    then nudges up to 2105 on the last 10 bars.
    The move is small so price is just above both EMAs -> BULLISH WEAK.
    """
    base = np.full(n, 2100.0)
    base[-10:] = np.linspace(2100.0, 2105.0, 10)
    return _make_df(base, noise=0.5)


def _bearish_strong(n: int = 300) -> pd.DataFrame:
    """Price falls from 2300 to 2000 (50% below EMA200 by the end)."""
    return _make_df(np.linspace(2300, 2000, n), noise=2.0)


def _neutral_between(n: int = 300) -> pd.DataFrame:
    """
    Build a DataFrame where EMA_50 is ABOVE the final close price
    but EMA_200 is BELOW it.

    Strategy: start high (EMA200 warm-ups high), then drop price so the
    recent EMA50 sits above current price while EMA200 lags below it.
    """
    # First 260 bars at 2200 (warms up both EMAs near 2200).
    # Then drop price to 2150 for bars 260-290 (EMA50 starts to fall,
    # EMA200 still near 2200). Last 10 bars price recovers to 2180.
    # Result: close~2180, EMA50~2175 (falling), EMA200~2195
    #         => close < EMA200 but close > EMA50 => NEUTRAL
    prices = np.full(n, 2200.0)
    prices[260:290] = np.linspace(2200, 2150, 30)
    prices[290:]    = np.linspace(2150, 2180, n - 290)
    return _make_df(prices, noise=1.0)


def _insufficient(n: int = 100) -> pd.DataFrame:
    """Only 100 bars — below TrendDetector.MIN_BARS_REQUIRED (210)."""
    return _make_df(np.linspace(2100, 2150, n), noise=2.0)


def _golden_cross_at_end(n: int = 310) -> pd.DataFrame:
    """
    Build a scenario where EMA_50 crosses above EMA_200 on the last bar.

    Steps:
      - Bars 0-250: price at 1900 (both EMAs settle near 1900, EMA50 belowEMA200 naturally at same level)
      - Bars 250-300: price jumps to 2100 (EMA50 surges up faster)
      - Bar 300: EMA50 finally crosses above EMA200
    """
    prices = np.full(n, 1900.0)
    prices[250:] = np.linspace(1900, 2100, n - 250)
    return _make_df(prices, noise=0.5)


def _death_cross_at_end(n: int = 310) -> pd.DataFrame:
    """EMA_50 crosses below EMA_200 on the last bar."""
    prices = np.full(n, 2200.0)
    prices[250:] = np.linspace(2200, 2000, n - 250)
    return _make_df(prices, noise=0.5)


# ─────────────────────────────────────────────────────────────────────────────
# Test runner
# ─────────────────────────────────────────────────────────────────────────────

def run_test(
    name: str,
    df: pd.DataFrame,
    expected_direction: TrendDirection,
    expected_strength_label: TrendStrengthLabel | None = None,
    det: TrendDetector | None = None,
) -> bool:
    if det is None:
        det = TrendDetector()

    result = det.detect(df)
    d = result.to_dict()

    dir_ok = result.direction == expected_direction
    lbl_ok = (
        expected_strength_label is None or
        result.strength_label == expected_strength_label
    )
    passed = dir_ok and lbl_ok

    status = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
    print(f"\n  [{status}]  {name}")

    exp_lbl = f", strength_label={expected_strength_label.value}" if expected_strength_label else ""
    print(f"         Expected  direction={expected_direction.value}{exp_lbl}")
    print(f"         Got       direction={d['direction']}, strength={d['strength']:.2f} ({d['strength_label']})")
    print(f"         EMA50={d['ema_50']}  EMA200={d['ema_200']}  close={d['close']}")
    print(f"         sep={d['ema_separation_pct']:+.3f}%  slope={d['ema50_slope']:+.1f}deg  "
          f"GX={d['golden_cross']}  DX={d['death_cross']}")

    return passed


def main() -> int:
    print(f"\n{BOLD}{'='*60}")
    print("  TrendDetector -- Synthetic Data Tests")
    print(f"{'='*60}{RESET}")

    det     = TrendDetector()
    results = []

    # ── Test 1: BULLISH STRONG ────────────────────────────────────────────
    hdr("Test 1 -- BULLISH STRONG (price far above both EMAs)")
    results.append(run_test(
        "Uptrend 2000->2300 over 300 bars",
        _bullish_strong(),
        expected_direction     = TrendDirection.BULLISH,
        expected_strength_label= TrendStrengthLabel.STRONG,
    ))

    # ── Test 2: BULLISH WEAK ──────────────────────────────────────────────
    hdr("Test 2 -- BULLISH WEAK (price just nudged above both EMAs)")
    results.append(run_test(
        "Flat at 2100 then tiny nudge to 2105 (last 10 bars)",
        _bullish_weak(),
        expected_direction     = TrendDirection.BULLISH,
        expected_strength_label= TrendStrengthLabel.WEAK,
    ))

    # ── Test 3: BEARISH STRONG ────────────────────────────────────────────
    hdr("Test 3 -- BEARISH STRONG (price far below both EMAs)")
    results.append(run_test(
        "Downtrend 2300->2000 over 300 bars",
        _bearish_strong(),
        expected_direction     = TrendDirection.BEARISH,
        expected_strength_label= TrendStrengthLabel.STRONG,
    ))

    # ── Test 4: NEUTRAL ───────────────────────────────────────────────────
    hdr("Test 4 -- NEUTRAL (price caught between EMA_50 and EMA_200)")
    results.append(run_test(
        "Price between EMA50 and EMA200 after a pullback",
        _neutral_between(),
        expected_direction     = TrendDirection.NEUTRAL,
    ))

    # ── Test 5: UNKNOWN (insufficient bars) ───────────────────────────────
    hdr("Test 5 -- UNKNOWN (100 bars < MIN_BARS_REQUIRED 210)")
    results.append(run_test(
        "Only 100 bars provided",
        _insufficient(),
        expected_direction     = TrendDirection.UNKNOWN,
    ))

    # ── Test 6: Golden Cross ──────────────────────────────────────────────
    hdr("Test 6 -- Golden Cross detection (EMA50 crossed above EMA200 in series)")
    det_gx = TrendDetector()
    df_gx  = _golden_cross_at_end(n=310)
    df_gx_emas = det_gx.add_emas(df_gx)
    # Scan the entire series for a golden cross (EMA50 crosses above EMA200)
    diff   = df_gx_emas["ema_50"] - df_gx_emas["ema_200"]
    cross_mask = (diff.shift(1) <= 0) & (diff > 0)
    n_crosses  = int(cross_mask.sum())
    final_ep   = float(diff.iloc[-1])
    # Also verify the endpoint: EMA50 should now be ABOVE EMA200
    passed_gx = (n_crosses >= 1) and (final_ep > 0)
    if passed_gx:
        ok(f"Golden Cross found {n_crosses} time(s) in series")
        ok(f"Final state: EMA50={df_gx_emas['ema_50'].iloc[-1]:.2f} > EMA200={df_gx_emas['ema_200'].iloc[-1]:.2f}")
    else:
        fail(f"No Golden Cross in series ({n_crosses} found, final diff={final_ep:+.4f})")
    results.append(passed_gx)

    # ── Test 7: Death Cross ───────────────────────────────────────────────
    hdr("Test 7 -- Death Cross detection (EMA50 crossed below EMA200 in series)")
    det_dx = TrendDetector()
    df_dx  = _death_cross_at_end(n=310)
    df_dx_emas = det_dx.add_emas(df_dx)
    diff_d = df_dx_emas["ema_50"] - df_dx_emas["ema_200"]
    cross_mask_d = (diff_d.shift(1) >= 0) & (diff_d < 0)
    n_crosses_d  = int(cross_mask_d.sum())
    final_dp     = float(diff_d.iloc[-1])
    passed_dx = (n_crosses_d >= 1) and (final_dp < 0)
    if passed_dx:
        ok(f"Death Cross found {n_crosses_d} time(s) in series")
        ok(f"Final state: EMA50={df_dx_emas['ema_50'].iloc[-1]:.2f} < EMA200={df_dx_emas['ema_200'].iloc[-1]:.2f}")
    else:
        fail(f"No Death Cross in series ({n_crosses_d} found, final diff={final_dp:+.4f})")
    results.append(passed_dx)

    # ── Test 8: add_emas() ────────────────────────────────────────────────
    hdr("Test 8 -- add_emas() column names and sanity")
    df_raw = _bullish_strong()
    df_out = det.add_emas(df_raw)
    t8 = True
    if "ema_50" in df_out.columns and "ema_200" in df_out.columns:
        ok("Columns ema_50 and ema_200 added")
    else:
        fail(f"Columns missing: {set(df_out.columns)}")
        t8 = False
    if df_out["ema_50"].iloc[-1] > df_out["ema_200"].iloc[-1]:
        ok("In uptrend: EMA_50 > EMA_200 (correct stacking)")
    else:
        fail("EMA stacking incorrect for uptrend data")
        t8 = False
    results.append(t8)

    # ── Test 9: get_ema_values() ──────────────────────────────────────────
    hdr("Test 9 -- get_ema_values() quick snapshot")
    df_bull = _bullish_strong()
    ema_vals = det.get_ema_values(df_bull)
    required = {"ema_50", "ema_200", "close"}
    t9 = required.issubset(set(ema_vals.keys()))
    if t9:
        ok(f"Keys present: {sorted(ema_vals.keys())}")
        ok(f"Values: close={ema_vals['close']}  ema_50={ema_vals['ema_50']}  ema_200={ema_vals['ema_200']}")
    else:
        fail(f"Missing keys: {required - set(ema_vals.keys())}")
    results.append(t9)

    # ── Test 10: allows_direction() ───────────────────────────────────────
    hdr("Test 10 -- allows_direction() trade gate")
    bull_r = det.detect(_bullish_strong())
    bear_r = det.detect(_bearish_strong())
    t10 = True
    if bull_r.allows_direction("BUY") and not bull_r.allows_direction("SELL"):
        ok("Bullish trend: BUY allowed, SELL blocked")
    else:
        fail(f"Bullish gate wrong  BUY={bull_r.allows_direction('BUY')} SELL={bull_r.allows_direction('SELL')}")
        t10 = False
    if bear_r.allows_direction("SELL") and not bear_r.allows_direction("BUY"):
        ok("Bearish trend: SELL allowed, BUY blocked")
    else:
        fail(f"Bearish gate wrong  BUY={bear_r.allows_direction('BUY')} SELL={bear_r.allows_direction('SELL')}")
        t10 = False
    results.append(t10)

    # ── Test 11: detect_as_dict() format ─────────────────────────────────
    hdr("Test 11 -- detect_as_dict() output format")
    d_out = det.detect_as_dict(_bullish_strong())
    required_keys = {
        "direction", "strength", "strength_label",
        "ema_50", "ema_200", "close",
        "ema_separation_pct", "price_vs_ema50_pct", "price_vs_ema200_pct",
        "ema50_slope", "golden_cross", "death_cross", "notes",
    }
    missing = required_keys - set(d_out.keys())
    if not missing:
        ok(f"All {len(required_keys)} required keys present")
        ok(f"direction={d_out['direction']!r}  strength={d_out['strength']}  label={d_out['strength_label']!r}")
    else:
        fail(f"Missing keys: {missing}")
    results.append(not bool(missing))

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
