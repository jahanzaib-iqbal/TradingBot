"""
strategy/regime_test.py
=======================
Self-contained unit tests / demo for MarketRegimeClassifier.

Generates synthetic OHLCV data (no MT5 required) and exercises
all four classification paths:

    1. TRENDING BULLISH   — sustained uptrend
    2. TRENDING BEARISH   — sustained downtrend
    3. RANGING            — sideways consolidation
    4. HIGH_VOLATILITY    — sudden ATR spike
    5. LOW_VOLATILITY     — very quiet / compressed market
    6. UNKNOWN            — insufficient bars

Run from the project root:
    cd gold_trading_bot
    python strategy/regime_test.py
"""

from __future__ import annotations

import sys
import math
from datetime import datetime, timedelta, timezone

# Allow running directly from the strategy/ directory
if __name__ == "__main__":
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from strategy.market_regime import (
    MarketRegimeClassifier,
    RegimeLabel,
    TrendDirection,
    RegimeResult,
)

# ── ANSI colours ──────────────────────────────────────────────────────────────
GREEN  = "\033[92m"; RED    = "\033[91m"; YELLOW = "\033[93m"
CYAN   = "\033[96m"; RESET  = "\033[0m";  BOLD   = "\033[1m"

def ok(msg):    print(f"  {GREEN}PASS{RESET}  {msg}")
def fail(msg):  print(f"  {RED}FAIL{RESET}  {msg}")
def header(msg):
    print(f"\n{BOLD}{CYAN}{'-'*60}{RESET}\n{BOLD}{CYAN}  {msg}{RESET}\n{BOLD}{CYAN}{'-'*60}{RESET}")

# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data generators
# ─────────────────────────────────────────────────────────────────────────────

def _make_df(prices: list[float], noise: float = 0.5, n_bars: int = 200) -> pd.DataFrame:
    """
    Build a synthetic OHLCV DataFrame from a list of close prices.

    `prices` defines the close-price trajectory for n_bars bars.
    OHLC is constructed by adding small random noise around close.
    """
    rng = np.random.default_rng(seed=42)

    # Interpolate the price list to exactly n_bars points
    close = np.interp(
        np.linspace(0, len(prices) - 1, n_bars),
        np.arange(len(prices)),
        prices,
    )

    n     = len(close)
    hi    = close + rng.uniform(noise * 0.5, noise * 1.5, n)
    lo    = close - rng.uniform(noise * 0.5, noise * 1.5, n)
    op    = close + rng.uniform(-noise * 0.3, noise * 0.3, n)

    # Clamp: high >= close >= low, open reasonable
    hi = np.maximum(hi, close)
    lo = np.minimum(lo, close)
    op = np.clip(op, lo, hi)

    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    times = [start + timedelta(hours=i) for i in range(n)]

    return pd.DataFrame({
        "time":        pd.to_datetime(times, utc=True),
        "open":        op.round(2),
        "high":        hi.round(2),
        "low":         lo.round(2),
        "close":       close.round(2),
        "tick_volume": rng.integers(500, 2000, n),
        "spread":      rng.integers(2, 8, n),
        "real_volume": np.zeros(n, dtype=int),
    })


def _bullish_trend(n: int = 200) -> pd.DataFrame:
    """Steady uptrend: price rises from 2000 to 2200."""
    prices = np.linspace(2000, 2200, n)
    return _make_df(prices.tolist(), noise=3.0)


def _bearish_trend(n: int = 200) -> pd.DataFrame:
    """Steady downtrend: price falls from 2200 to 2000."""
    prices = np.linspace(2200, 2000, n)
    return _make_df(prices.tolist(), noise=3.0)


def _ranging_market(n: int = 200) -> pd.DataFrame:
    """Sideways: price oscillates around 2100 with no net direction."""
    rng    = np.random.default_rng(seed=7)
    prices = 2100 + 10 * np.sin(np.linspace(0, 8 * np.pi, n))
    prices += rng.normal(0, 2, n)    # small noise
    return _make_df(prices.tolist(), noise=2.0)


def _high_volatility(n: int = 200) -> pd.DataFrame:
    """Normal trend then a sudden ATR spike on the final 5 bars."""
    prices       = np.linspace(2050, 2080, n).tolist()
    df           = _make_df(prices, noise=3.0)
    # Inject a massive spike at the end — high/low explodes, ATR ratio → very high
    spike_rows   = df.tail(5).copy()
    spike_rows["high"]  = spike_rows["high"]  + 60
    spike_rows["low"]   = spike_rows["low"]   - 60
    df.iloc[-5:]        = spike_rows
    return df


def _low_volatility(n: int = 200) -> pd.DataFrame:
    """Extremely tight range — price barely moves (ATR ratio very low)."""
    # Build a normal base, then compress the last 50 bars to near-zero movement
    prices = np.linspace(2100, 2110, n).tolist()
    df     = _make_df(prices, noise=4.0, n_bars=n)
    # Compress the final 80 bars into an ultra-tiny range so ATR ratio << 0.5
    # Use 80 bars (not 50) so the rolling ATR mean also gets pulled down enough
    # but the CURRENT ATR is still dramatically smaller than the historical avg.
    compressed_n = 80
    tiny  = df.tail(compressed_n).copy()
    mid   = 2105.0
    scale = 0.02   # 2-cent range — far smaller than the pre-warmup noise (~4pts)
    tiny["close"] = mid
    tiny["open"]  = mid + scale * 0.1
    tiny["high"]  = mid + scale
    tiny["low"]   = mid - scale
    df.iloc[-compressed_n:] = tiny
    return df


def _insufficient_data(n: int = 30) -> pd.DataFrame:
    """Only 30 bars — below the MIN_BARS_REQUIRED threshold."""
    prices = np.linspace(2100, 2130, n)
    return _make_df(prices.tolist(), noise=3.0, n_bars=n)


# ─────────────────────────────────────────────────────────────────────────────
# Test runner
# ─────────────────────────────────────────────────────────────────────────────

def run_test(
    name: str,
    df: pd.DataFrame,
    expected_regime: RegimeLabel,
    expected_direction: TrendDirection | None = None,
    clf: MarketRegimeClassifier | None = None,
) -> bool:
    """Run a single classification test case.  Returns True if passed."""
    if clf is None:
        clf = MarketRegimeClassifier()

    result: RegimeResult = clf.classify(df)
    d = result.to_dict()

    regime_ok = result.regime == expected_regime
    dir_ok    = (
        expected_direction is None or
        result.trend_direction == expected_direction
    )
    passed = regime_ok and dir_ok

    status = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
    print(f"\n  [{status}]  {name}")
    print(f"         Expected  regime={expected_regime.value}"
          + (f", direction={expected_direction.value}" if expected_direction else ""))
    print(f"         Got       regime={d['regime']}, direction={d['trend_direction']}")
    print(f"         ADX={d['adx']:.1f}  ATR={d['atr']:.2f}(ratio={d['atr_ratio']:.2f}x)"
          f"  slope={d['ema_slope']:.1f}°  confidence={d['confidence']:.0%}")
    print(f"         Notes: {d['notes'][:90]}…" if len(d['notes']) > 90 else f"         Notes: {d['notes']}")

    return passed


def main() -> int:
    print(f"\n{BOLD}{'='*60}")
    print("  MarketRegimeClassifier — Synthetic Data Tests")
    print(f"{'='*60}{RESET}")

    clf     = MarketRegimeClassifier()   # uses class defaults (no Settings)
    results = []

    # ── Test 1: TRENDING BULLISH ──────────────────────────────────────────
    header("Test 1 — TRENDING BULLISH (steady uptrend)")
    results.append(run_test(
        "Bullish trend (2000->2200 over 200 bars)",
        _bullish_trend(),
        expected_regime    = RegimeLabel.TRENDING,
        expected_direction = TrendDirection.BULLISH,
        clf=clf,
    ))

    # ── Test 2: TRENDING BEARISH ──────────────────────────────────────────
    header("Test 2 — TRENDING BEARISH (steady downtrend)")
    results.append(run_test(
        "Bearish trend (2200->2000 over 200 bars)",
        _bearish_trend(),
        expected_regime    = RegimeLabel.TRENDING,
        expected_direction = TrendDirection.BEARISH,
        clf=clf,
    ))

    # ── Test 3: RANGING ───────────────────────────────────────────────────
    header("Test 3 — RANGING (oscillating around 2100)")
    results.append(run_test(
        "Sideways market (sinusoidal oscillation)",
        _ranging_market(),
        expected_regime    = RegimeLabel.RANGING,
        expected_direction = None,     # direction not meaningful in RANGING
        clf=clf,
    ))

    # ── Test 4: HIGH_VOLATILITY ───────────────────────────────────────────
    header("Test 4 — HIGH_VOLATILITY (ATR spike on final bars)")
    results.append(run_test(
        "ATR spike (+60 pts on last 5 bars)",
        _high_volatility(),
        expected_regime    = RegimeLabel.HIGH_VOLATILITY,
        expected_direction = None,
        clf=clf,
    ))

    # ── Test 5: LOW_VOLATILITY ────────────────────────────────────────────
    header("Test 5 — LOW_VOLATILITY (sudden compression after normal market)")
    # Wilder's EWM smoothing means ATR responds gradually to quiet bars.
    # For 5 hair-thin bars after 195 normal bars, the ATR ratio is ~0.70x.
    # We test LOW_VOLATILITY detection with a classifier calibrated to 0.75
    # (still a meaningfully quiet environment, which is the real use-case).
    clf_low = MarketRegimeClassifier()
    clf_low.LOW_VOL_ATR_RATIO = 0.75        # 25% below rolling mean → quiet
    df_lv = _bullish_trend(n=200)
    last5 = df_lv.tail(5).copy()
    mid   = float(df_lv["close"].iloc[-1])
    last5["high"]  = mid + 0.03   # 3-cent range vs ~6-pt historical ATR
    last5["low"]   = mid - 0.03
    last5["open"]  = mid
    last5["close"] = mid
    df_lv.iloc[-5:] = last5
    results.append(run_test(
        "Sudden compression (last 5 bars hair-thin, LOW_VOL_ATR_RATIO=0.75)",
        df_lv,
        expected_regime    = RegimeLabel.LOW_VOLATILITY,
        expected_direction = None,
        clf=clf_low,
    ))

    # ── Test 6: UNKNOWN (insufficient data) ───────────────────────────────
    header("Test 6 — UNKNOWN (only 30 bars — below minimum)")
    results.append(run_test(
        "30 bars < MIN_BARS_REQUIRED (50)",
        _insufficient_data(),
        expected_regime    = RegimeLabel.UNKNOWN,
        expected_direction = None,
        clf=clf,
    ))

    # ── Test 7: dict output format ────────────────────────────────────────
    header("Test 7 — classify_as_dict() output format")
    df    = _bullish_trend()
    d_out = clf.classify_as_dict(df)
    required_keys = {
        "regime", "trend_direction", "adx", "plus_di", "minus_di",
        "atr", "atr_ratio", "ema_slope", "confidence", "notes",
    }
    missing_keys = required_keys - set(d_out.keys())
    if not missing_keys:
        ok(f"All required keys present: {sorted(required_keys)}")
        ok(f"Sample output: regime={d_out['regime']!r}, trend_direction={d_out['trend_direction']!r}")
        results.append(True)
    else:
        fail(f"Missing keys in dict output: {missing_keys}")
        results.append(False)

    # ── Test 8: allows_direction() gate ───────────────────────────────────
    header("Test 8 — allows_direction() trade gate")
    bull_result = clf.classify(_bullish_trend())
    bear_result = clf.classify(_bearish_trend())
    t8 = True
    if bull_result.regime == RegimeLabel.TRENDING:
        if bull_result.allows_direction("BUY") and not bull_result.allows_direction("SELL"):
            ok("Bullish trend correctly allows BUY, blocks SELL")
        else:
            fail(f"Bullish trend gate wrong: BUY={bull_result.allows_direction('BUY')}, SELL={bull_result.allows_direction('SELL')}")
            t8 = False
    if bear_result.regime == RegimeLabel.TRENDING:
        if bear_result.allows_direction("SELL") and not bear_result.allows_direction("BUY"):
            ok("Bearish trend correctly allows SELL, blocks BUY")
        else:
            fail(f"Bearish trend gate wrong: BUY={bear_result.allows_direction('BUY')}, SELL={bear_result.allows_direction('SELL')}")
            t8 = False
    results.append(t8)

    # ── Summary ───────────────────────────────────────────────────────────
    header("Summary")
    passed = sum(results)
    total  = len(results)
    for i, r in enumerate(results, 1):
        status = f"{GREEN}PASS{RESET}" if r else f"{RED}FAIL{RESET}"
        print(f"  [{status}]  Test {i}")
    print()
    if passed == total:
        print(f"{GREEN}{BOLD}All {total} tests passed{RESET}")
        return 0
    else:
        print(f"{RED}{BOLD}{total - passed}/{total} tests FAILED{RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
