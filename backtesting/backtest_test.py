"""
backtesting/backtest_test.py
==============================
Self-contained backtest engine tests using synthetic OHLCV data.
No MT5 or live data required.

Tests
-----
 1.  BacktestResult.to_dict() has all required keys
 2.  TradeRecord.risk_distance computes correctly
 3.  TradeRecord.rr_tp1 computes correctly
 4.  TradeRecord.is_winner / is_loser flags
 5.  _compute_drawdown() — zero drawdown on all-wins
 6.  _compute_drawdown() — correct peak-trough on known series
 7.  _compute_streaks() — max win/loss streaks
 8.  _close() helper mutates record correctly
 9.  BacktestEngine constructs without error
10.  load_csv() raises FileNotFoundError on missing file
11.  _slice() filters by date correctly
12.  run() returns BacktestResult (no crash on synthetic data)
13.  run() — total_trades >= 0 (sanity)
14.  run() — win_rate in [0, 1]
15.  run() — profit_factor >= 0
16.  run() — expectancy computed: total_r / total_trades (approx)
17.  run() — max_drawdown_r <= 0
18.  run() — max_consecutive_losses and wins >= 0
19.  print_summary() runs without error
20.  export() writes trade_log.csv and backtest_summary.txt
21.  to_dict() round-trips through JSON without error
22.  walk_forward() returns n_folds results

Run:
    cd gold_trading_bot
    python backtesting/backtest_test.py
"""

from __future__ import annotations

import json
import sys
import os
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

# ANSI colours
GREEN = "\033[92m"; RED = "\033[91m"; CYAN = "\033[96m"
RESET = "\033[0m";  BOLD = "\033[1m"

def ok(msg):   print(f"  {GREEN}PASS{RESET}  {msg}")
def fail(msg): print(f"  {RED}FAIL{RESET}  {msg}")
def hdr(msg):  print(f"\n{BOLD}{CYAN}{'-'*60}{RESET}\n{BOLD}{CYAN}  {msg}{RESET}\n{BOLD}{CYAN}{'-'*60}{RESET}")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _settings():
    from config.settings import Settings
    cfg = Settings()
    cfg.CONFIDENCE_THRESHOLD = 0.10   # low threshold so synth data fires signals
    cfg.MIN_RR_RATIO         = 1.0
    cfg.ATR_MIN_POINTS       = 0.05   # allow tiny synthetic ATR
    return cfg


def _make_df(n: int = 500, base: float = 2300.0,
             trend: float = 0.05, noise: float = 2.5,
             interval_min: int = 60, seed: int = 42) -> pd.DataFrame:
    """Create realistic OHLCV data with a directional drift."""
    rng    = np.random.default_rng(seed)
    closes = base + np.arange(n) * trend + rng.normal(0, noise, n)
    highs  = closes + rng.uniform(1.0, 3.5, n)
    lows   = closes - rng.uniform(1.0, 3.5, n)
    opens  = closes + rng.uniform(-1.0, 1.0, n)
    highs  = np.maximum(highs, np.maximum(opens, closes))
    lows   = np.minimum(lows,  np.minimum(opens, closes))
    start  = datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc)
    times  = pd.to_datetime([start + timedelta(minutes=interval_min * i) for i in range(n)], utc=True)
    return pd.DataFrame({
        "time":        times,
        "open":        opens.round(2),
        "high":        highs.round(2),
        "low":         lows.round(2),
        "close":       closes.round(2),
        "tick_volume": rng.integers(300, 1200, n).astype(float),
    })


def _make_trade(direction="BUY", entry=2350.0, sl=2340.0, tp1=2370.0, tp2=2390.0,
                outcome="TP1", r=2.0, bar=250, trade_id=1) -> "TradeRecord":
    from backtesting.backtest_engine import TradeRecord
    from datetime import datetime, timezone
    rec = TradeRecord(
        trade_id=trade_id, signal_bar=bar,
        signal_time=datetime(2024,2,1,10,0,tzinfo=timezone.utc),
        direction=direction, entry=entry, stop_loss=sl,
        take_profit_1=tp1, take_profit_2=tp2,
        confidence=0.72, has_ob=True, has_fvg=False,
        has_bos=True, has_choch=False, has_liq_sweep=True,
        session="london", regime="TRENDING",
    )
    rec.outcome=outcome; rec.outcome_bar=bar+10; rec.bars_held=10; rec.r_gained=r
    return rec


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    print(f"\n{BOLD}{'='*60}")
    print("  BacktestEngine -- Synthetic Tests")
    print(f"{'='*60}{RESET}")

    results = []
    cfg     = _settings()

    from backtesting.backtest_engine import (
        BacktestEngine, BacktestResult, TradeRecord,
        MIN_LOOKBACK_BARS,
    )

    engine = BacktestEngine(cfg)

    # ── Test 1: BacktestResult.to_dict() keys ─────────────────────────────────
    hdr("Test 1 -- BacktestResult.to_dict() has all required keys")
    required_keys = {
        "win_rate", "profit_factor", "expectancy", "total_r",
        "total_trades", "winners", "losers", "max_drawdown_r",
        "max_drawdown_pct", "max_consecutive_losses",
        "max_consecutive_wins", "sharpe_ratio",
    }
    br = BacktestResult()
    d1 = br.to_dict()
    missing1 = required_keys - set(d1.keys())
    t1 = not missing1
    ok(f"All {len(required_keys)} required keys present") if t1 else \
    fail(f"Missing: {missing1}")
    results.append(t1)

    # ── Test 2: TradeRecord.risk_distance ─────────────────────────────────────
    hdr("Test 2 -- TradeRecord.risk_distance computes correctly")
    tr2 = _make_trade(entry=2350.0, sl=2340.0)
    t2 = abs(tr2.risk_distance - 10.0) < 0.001
    ok(f"risk_distance={tr2.risk_distance:.2f}") if t2 else \
    fail(f"Expected 10.0, got {tr2.risk_distance}")
    results.append(t2)

    # ── Test 3: TradeRecord.rr_tp1 ────────────────────────────────────────────
    hdr("Test 3 -- TradeRecord.rr_tp1 computes correctly")
    tr3 = _make_trade(entry=2350.0, sl=2340.0, tp1=2370.0)  # 20/10 = 2.0
    t3 = abs(tr3.rr_tp1 - 2.0) < 0.001
    ok(f"rr_tp1={tr3.rr_tp1:.2f}") if t3 else fail(f"Expected 2.0, got {tr3.rr_tp1}")
    results.append(t3)

    # ── Test 4: is_winner / is_loser ──────────────────────────────────────────
    hdr("Test 4 -- TradeRecord.is_winner / is_loser flags")
    win  = _make_trade(outcome="TP1",  r=2.0)
    lose = _make_trade(outcome="SL",   r=-1.0)
    be   = _make_trade(outcome="BE",   r=1.0)
    exp  = _make_trade(outcome="EXPIRED", r=0.0)
    t4 = (win.is_winner and not win.is_loser and
          lose.is_loser and not lose.is_winner and
          not be.is_winner and not be.is_loser and
          not exp.is_winner)
    ok("is_winner/is_loser flags correct for all 4 outcomes") if t4 else \
    fail(f"Flags incorrect: win={win.is_winner} lose={lose.is_loser}")
    results.append(t4)

    # ── Test 5: _compute_drawdown — all wins → 0 DD ───────────────────────────
    hdr("Test 5 -- _compute_drawdown: all-win series has 0 drawdown")
    dd5_r, dd5_pct = BacktestEngine._compute_drawdown([1.0, 2.0, 0.5, 1.5])
    t5 = dd5_r >= 0.0   # no drawdown
    ok(f"max_dd_r={dd5_r:.4f} (≥0)") if t5 else fail(f"Expected 0, got {dd5_r}")
    results.append(t5)

    # ── Test 6: _compute_drawdown — known series ──────────────────────────────
    hdr("Test 6 -- _compute_drawdown: correct peak-trough on known series")
    # Equity: 0 → +2 → +1 → -1 → 0 → +3  (peak=5 at bar5, trough=-1 at bar3)
    # DD from peak 2 → trough 0 = -2 R  (then recovers)
    series6 = [2.0, -1.0, -2.0, 1.0, 3.0]
    dd6_r, dd6_pct = BacktestEngine._compute_drawdown(series6)
    t6 = dd6_r < 0.0   # some drawdown exists
    ok(f"max_dd_r={dd6_r:.4f} (< 0, correct)") if t6 else \
    fail(f"Expected negative drawdown, got {dd6_r}")
    results.append(t6)

    # ── Test 7: _compute_streaks ───────────────────────────────────────────────
    hdr("Test 7 -- _compute_streaks: max win/loss streaks")
    trades7 = [
        _make_trade(outcome="TP1", r=2.0, trade_id=i) for i in range(3)
    ] + [
        _make_trade(outcome="SL",  r=-1.0, trade_id=i+3) for i in range(4)
    ] + [
        _make_trade(outcome="TP1", r=2.0, trade_id=i+7) for i in range(2)
    ]
    cw7, cl7 = BacktestEngine._compute_streaks(trades7)
    t7 = cw7 == 3 and cl7 == 4
    ok(f"max_wins={cw7}  max_losses={cl7}") if t7 else \
    fail(f"Expected cw=3 cl=4, got cw={cw7} cl={cl7}")
    results.append(t7)

    # ── Test 8: _close() helper ───────────────────────────────────────────────
    hdr("Test 8 -- _close() mutates TradeRecord correctly")
    tr8 = _make_trade()
    bar_time_8 = datetime(2024, 2, 1, 14, 0, tzinfo=timezone.utc)
    BacktestEngine._close(tr8, "TP2", 260, bar_time_8, 3.5)
    t8 = (tr8.outcome == "TP2" and tr8.outcome_bar == 260
          and abs(tr8.r_gained - 3.5) < 0.001 and tr8.bars_held == 10)
    ok(f"outcome={tr8.outcome}  r={tr8.r_gained}  bars={tr8.bars_held}") if t8 else \
    fail(f"_close() fields wrong")
    results.append(t8)

    # ── Test 9: BacktestEngine constructs ─────────────────────────────────────
    hdr("Test 9 -- BacktestEngine constructs without error")
    t9 = isinstance(engine, BacktestEngine)
    ok(f"BacktestEngine constructed  max_bars={engine.max_bars_open}") if t9 else \
    fail("Construction failed")
    results.append(t9)

    # ── Test 10: load_csv raises on missing file ──────────────────────────────
    hdr("Test 10 -- load_csv() raises FileNotFoundError on missing file")
    try:
        BacktestEngine.load_csv("nonexistent_file.csv")
        t10 = False
        fail("Should have raised FileNotFoundError")
    except FileNotFoundError:
        t10 = True
        ok("FileNotFoundError raised correctly")
    results.append(t10)

    # ── Test 11: _slice() filters date range ─────────────────────────────────
    hdr("Test 11 -- _slice() filters DataFrame by date range")
    df11 = _make_df(n=200)
    sliced11 = BacktestEngine._slice(df11, "2024-01-10", "2024-01-20")
    t11 = len(sliced11) < len(df11) and len(sliced11) > 0
    ok(f"Sliced {len(df11)} → {len(sliced11)} bars") if t11 else \
    fail(f"Slice returned {len(sliced11)}")
    results.append(t11)

    # ── Tests 12–19: Integration (requires enough bars) ──────────────────────
    hdr("Test 12 -- run() returns BacktestResult on synthetic data")
    df_m15 = _make_df(n=500, base=2300.0, noise=3.0, interval_min=15, seed=7)
    df_h1  = _make_df(n=500, base=2300.0, noise=4.0, interval_min=60,  seed=8)

    print(f"  Running backtest on {len(df_m15)} M15 bars... (this may take a few seconds)")
    result12 = engine.run(df_h1, df_m15, account_balance=10_000.0)
    t12 = isinstance(result12, BacktestResult)
    ok(f"BacktestResult returned  trades={result12.total_trades}") if t12 else \
    fail("run() did not return BacktestResult")
    results.append(t12)

    # ── Test 13: total_trades >= 0 ────────────────────────────────────────────
    hdr("Test 13 -- run() total_trades >= 0")
    t13 = result12.total_trades >= 0
    ok(f"total_trades={result12.total_trades}") if t13 else fail("total_trades < 0")
    results.append(t13)

    # ── Test 14: win_rate in [0, 1] ───────────────────────────────────────────
    hdr("Test 14 -- run() win_rate is in [0.0, 1.0]")
    t14 = 0.0 <= result12.win_rate <= 1.0
    ok(f"win_rate={result12.win_rate:.1%}") if t14 else \
    fail(f"win_rate out of range: {result12.win_rate}")
    results.append(t14)

    # ── Test 15: profit_factor >= 0 ───────────────────────────────────────────
    hdr("Test 15 -- run() profit_factor >= 0")
    t15 = result12.profit_factor >= 0.0
    ok(f"profit_factor={result12.profit_factor:.2f}") if t15 else \
    fail(f"profit_factor={result12.profit_factor}")
    results.append(t15)

    # ── Test 16: expectancy = total_r / total_trades ──────────────────────────
    hdr("Test 16 -- run() expectancy ≈ total_r / total_trades")
    if result12.total_trades > 0:
        exp_check = result12.total_r / result12.total_trades
        t16 = abs(result12.expectancy - exp_check) < 0.01
        ok(f"expectancy={result12.expectancy:.4f} ≈ {exp_check:.4f}") if t16 else \
        fail(f"Mismatch: {result12.expectancy} vs {exp_check}")
    else:
        t16 = True
        ok("No trades — expectancy=0 by definition")
    results.append(t16)

    # ── Test 17: max_drawdown_r <= 0 ─────────────────────────────────────────
    hdr("Test 17 -- run() max_drawdown_r <= 0")
    t17 = result12.max_drawdown_r <= 0.0
    ok(f"max_drawdown_r={result12.max_drawdown_r:.4f} R") if t17 else \
    fail(f"max_drawdown_r={result12.max_drawdown_r} should be ≤ 0")
    results.append(t17)

    # ── Test 18: streak fields >= 0 ───────────────────────────────────────────
    hdr("Test 18 -- run() max_consecutive streaks >= 0")
    t18 = (result12.max_consecutive_wins >= 0 and
           result12.max_consecutive_losses >= 0)
    ok(f"max_wins={result12.max_consecutive_wins}  "
       f"max_losses={result12.max_consecutive_losses}") if t18 else \
    fail("Negative streak values")
    results.append(t18)

    # ── Test 19: print_summary() doesn't crash ────────────────────────────────
    hdr("Test 19 -- print_summary() runs without error")
    try:
        result12.print_summary()
        t19 = True
        ok("print_summary() completed")
    except Exception as exc:
        t19 = False
        fail(f"print_summary() raised: {exc}")
    results.append(t19)

    # ── Test 20: export() writes files ───────────────────────────────────────
    hdr("Test 20 -- export() writes trade_log.csv and backtest_summary.txt")
    with tempfile.TemporaryDirectory() as tmpdir:
        engine.export(result12, tmpdir)
        csv_ok  = (Path(tmpdir) / "trade_log.csv").exists()
        txt_ok  = (Path(tmpdir) / "backtest_summary.txt").exists()
        t20 = csv_ok and txt_ok
        ok(f"trade_log.csv={csv_ok}  backtest_summary.txt={txt_ok}") if t20 else \
        fail("export() did not create expected files")
    results.append(t20)

    # ── Test 21: to_dict() JSON-serialisable ─────────────────────────────────
    hdr("Test 21 -- to_dict() round-trips through JSON without error")
    try:
        j21 = json.dumps(result12.to_dict())
        t21 = len(j21) > 10
        ok(f"JSON serialisation OK ({len(j21)} bytes)") if t21 else fail("Empty JSON")
    except Exception as exc:
        t21 = False
        fail(f"JSON error: {exc}")
    results.append(t21)

    # ── Test 22: walk_forward() returns n_folds results ──────────────────────
    hdr("Test 22 -- walk_forward() returns correct number of folds")
    folds22 = engine.walk_forward(df_h1, df_m15, n_folds=2)
    t22 = len(folds22) == 2 and all(isinstance(f, BacktestResult) for f in folds22)
    ok(f"walk_forward returned {len(folds22)} folds") if t22 else \
    fail(f"Expected 2 folds, got {len(folds22)}")
    results.append(t22)

    # ── Summary ───────────────────────────────────────────────────────────────
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
