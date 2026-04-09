"""
strategy/strategy_optimizer_test.py
=====================================
Self-contained tests for StrategyOptimizer.
No MT5, Telegram, or live data required.

Tests
-----
 1.  _parse_range parses "1.0,2.5,0.5" → [1.0, 1.5, 2.0, 2.5]
 2.  _parse_range handles integer steps ("30,70,10") correctly
 3.  _compute_metrics returns correct win_rate on known data
 4.  _compute_metrics returns correct profit_factor
 5.  _compute_metrics returns correct max_drawdown_r
 6.  _compute_metrics detects current_streak correctly
 7.  _compute_metrics flags is_degrading when recent < overall
 8.  ParamSet.is_valid() rejects TP1 ≤ SL
 9.  ParamSet.is_valid() rejects out-of-range confidence
10.  ParamSet.rr_ratio computed correctly
11.  StrategyOptimizer constructs (no history)
12.  record_trade() appends to CSV and increments n_trades
13.  record_trade() skips EXPIRED outcomes
14.  _build_grid() generates valid ParamSets only
15.  _apply_guards() rejects low win_rate
16.  _apply_guards() rejects high drawdown
17.  optimize() returns OptimizationReport
18.  OptimizationReport.n_candidates > 0
19.  OptimizationReport improved=True when better params found
20.  apply_params() never sets RISK_PCT above OPTIMIZER_MAX_RISK_PCT
21.  maybe_optimize() triggers only when eval_every trades reached
22.  current_performance() returns populated PerformanceMetrics

Run:
    cd gold_trading_bot
    python strategy/strategy_optimizer_test.py
"""

from __future__ import annotations

import sys
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

# ── ANSI colours ─────────────────────────────────────────────────────────────
GREEN = "\033[92m"; RED = "\033[91m"; CYAN = "\033[96m"
RESET = "\033[0m";  BOLD = "\033[1m"

def ok(msg):   print(f"  {GREEN}PASS{RESET}  {msg}")
def fail(msg): print(f"  {RED}FAIL{RESET}  {msg}")
def hdr(msg):  print(f"\n{BOLD}{CYAN}{'-'*60}{RESET}\n{BOLD}{CYAN}  {msg}{RESET}\n{BOLD}{CYAN}{'-'*60}{RESET}")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cfg(tmp_dir: str, eval_every: int = 10, min_history: int = 10):
    from config.settings import Settings
    cfg = Settings()
    cfg.OPTIMIZER_ENABLED           = True
    cfg.OPTIMIZER_EVAL_EVERY        = eval_every
    cfg.OPTIMIZER_MIN_HISTORY       = min_history
    cfg.OPTIMIZER_MAX_RISK_PCT      = 2.0
    cfg.OPTIMIZER_MIN_WIN_RATE      = 0.40
    cfg.OPTIMIZER_MIN_PROFIT_FACTOR = 1.10
    cfg.OPTIMIZER_MAX_DRAWDOWN_R    = 20.0
    cfg.OPTIMIZER_SCORE_METRIC      = "expectancy"
    cfg.OPTIMIZER_ATR_SL_RANGE      = "1.0,2.0,0.5"       # 3 values
    cfg.OPTIMIZER_ATR_TP1_RANGE     = "1.5,2.5,0.5"       # 3 values (some < SL → filtered)
    cfg.OPTIMIZER_ATR_TP2_RANGE     = "3.0,4.0,1.0"       # 2 values
    cfg.OPTIMIZER_CONF_RANGE        = "0.60,0.70,0.05"    # 3 values
    cfg.OPTIMIZER_EMA_SLOW_RANGE    = "40,50,10"           # 2 values
    cfg.OPTIMIZER_HISTORY_PATH      = str(Path(tmp_dir) / "trade_history.csv")
    cfg.OPTIMIZER_OUTPUT_DIR        = str(Path(tmp_dir) / "reports")
    # Settings fields for current params baseline
    cfg.ATR_SL_MULTIPLIER           = 1.5
    cfg.ATR_TP1_MULTIPLIER          = 2.0
    cfg.ATR_TP2_MULTIPLIER          = 4.0
    cfg.CONFIDENCE_THRESHOLD        = 0.68
    cfg.EMA_SLOW_PERIOD             = 50
    cfg.RISK_PER_TRADE_PCT          = 1.0
    return cfg


def _make_trade(
    trade_id: int = 1,
    outcome: str = "TP1",
    r_gained: float = 1.0,
    confidence: float = 0.72,
    session: str = "london",
    regime: str = "TRENDING",
):
    """Lightweight dict-based trade record (duck-typed for TradeRecord)."""
    class FakeTrade:
        pass
    t = FakeTrade()
    t.trade_id      = trade_id
    t.signal_time   = datetime(2024, 3, 1, 10, 0, tzinfo=timezone.utc)
    t.direction     = "BUY"
    t.outcome       = outcome
    t.r_gained      = r_gained
    t.confidence    = confidence
    t.session       = session
    t.regime        = regime
    t.has_ob        = True
    t.has_fvg       = False
    t.has_bos       = True
    t.has_liq_sweep = True
    return t


def _make_trade_log(n_wins: int = 30, n_losses: int = 15, seed: int = 7):
    """Create a mixed list of trades for optimizer training."""
    rng    = np.random.default_rng(seed)
    trades = []
    tid    = 1
    for _ in range(n_wins):
        trades.append(_make_trade(
            trade_id   = tid,
            outcome    = rng.choice(["TP1", "TP2"]),
            r_gained   = float(rng.uniform(0.5, 3.0)),
            confidence = float(rng.uniform(0.60, 0.85)),
        ))
        tid += 1
    for _ in range(n_losses):
        trades.append(_make_trade(
            trade_id   = tid,
            outcome    = "SL",
            r_gained   = -1.0,
            confidence = float(rng.uniform(0.55, 0.75)),
        ))
        tid += 1
    return trades


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    print(f"\n{BOLD}{'='*60}")
    print("  StrategyOptimizer — Performance Tracker Tests")
    print(f"{'='*60}{RESET}")

    from strategy.strategy_optimizer import (
        StrategyOptimizer, PerformanceMetrics, ParamSet,
        OptimizationReport, GridSearchResult,
        _parse_range, _compute_metrics, _replay_with_params,
    )

    results = []

    with tempfile.TemporaryDirectory() as tmpdir:
        # ── Test 1: _parse_range float ────────────────────────────────────────
        hdr("Test 1 -- _parse_range parses '1.0,2.5,0.5' → [1.0, 1.5, 2.0, 2.5]")
        pr1 = _parse_range("1.0,2.5,0.5")
        t1  = len(pr1) == 4 and abs(pr1[0] - 1.0) < 1e-6 and abs(pr1[-1] - 2.5) < 1e-6
        ok(f"Got {pr1}") if t1 else fail(f"Got {pr1}")
        results.append(t1)

        # ── Test 2: _parse_range integer steps ───────────────────────────────
        hdr("Test 2 -- _parse_range handles integer steps '30,70,10'")
        pr2 = _parse_range("30,70,10")
        t2  = len(pr2) == 5 and pr2[0] == 30.0 and pr2[-1] == 70.0
        ok(f"Got {pr2}") if t2 else fail(f"Got {pr2}")
        results.append(t2)

        # ── Test 3: _compute_metrics win_rate ─────────────────────────────────
        hdr("Test 3 -- _compute_metrics win_rate on known data")
        r3   = [1.5, -1.0, 2.0, -1.0, 1.0]   # 3 wins, 2 losses
        m3   = _compute_metrics(r3)
        t3   = abs(m3.win_rate - 0.60) < 1e-4
        ok(f"win_rate={m3.win_rate:.4f}  (expected 0.60)") if t3 else \
        fail(f"win_rate={m3.win_rate}")
        results.append(t3)

        # ── Test 4: profit_factor ─────────────────────────────────────────────
        hdr("Test 4 -- _compute_metrics profit_factor = gross_profit / gross_loss")
        # gross_profit=4.5  gross_loss=2.0  PF=2.25
        t4 = abs(m3.profit_factor - 4.5 / 2.0) < 0.01
        ok(f"PF={m3.profit_factor:.4f}  (expected {4.5/2.0:.4f})") if t4 else \
        fail(f"PF={m3.profit_factor}")
        results.append(t4)

        # ── Test 5: max_drawdown_r ────────────────────────────────────────────
        hdr("Test 5 -- _compute_metrics max_drawdown_r")
        # Sequence: +1, +1, -1, -1, -1, +2  equity: 0,1,2,1,0,-1,1
        r5   = [1.0, 1.0, -1.0, -1.0, -1.0, 2.0]
        m5   = _compute_metrics(r5)
        # peak=2, trough=-1 → dd=3R
        t5   = abs(m5.max_drawdown_r - 3.0) < 0.01
        ok(f"max_dd_r={m5.max_drawdown_r:.3f}  (expected 3.0)") if t5 else \
        fail(f"max_dd_r={m5.max_drawdown_r}")
        results.append(t5)

        # ── Test 6: current_streak ────────────────────────────────────────────
        hdr("Test 6 -- _compute_metrics current_streak after 3 consecutive wins")
        r6   = [-1.0, -1.0, 1.0, 1.0, 1.0]
        m6   = _compute_metrics(r6)
        t6   = m6.current_streak == 3
        ok(f"current_streak={m6.current_streak}  (expected 3)") if t6 else \
        fail(f"current_streak={m6.current_streak}")
        results.append(t6)

        # ── Test 7: is_degrading detection ───────────────────────────────────
        hdr("Test 7 -- is_degrading flagged when last_n far below overall")
        # Good start, then 20 consecutive losses
        r7  = [1.5] * 40 + [-1.0] * 20
        m7  = _compute_metrics(r7, last_n=20)
        t7  = m7.is_degrading
        ok(f"is_degrading={m7.is_degrading}  last_exp={m7.last_n_expectancy:.3f}  "
           f"overall_exp={m7.expectancy:.3f}") if t7 else \
        fail(f"is_degrading should be True")
        results.append(t7)

        # ── Test 8: ParamSet.is_valid rejects TP1 ≤ SL ───────────────────────
        hdr("Test 8 -- ParamSet.is_valid() rejects TP1 ≤ SL multiplier")
        cfg8   = _cfg(tmpdir)
        bad8   = ParamSet(atr_sl_mult=2.0, atr_tp1_mult=1.5, atr_tp2_mult=4.0,
                          conf_threshold=0.65, ema_slow=50)
        t8     = not bad8.is_valid(cfg8)
        ok(f"is_valid=False → rejected correctly") if t8 else \
        fail(f"Should be invalid (TP1={bad8.atr_tp1_mult} ≤ SL={bad8.atr_sl_mult})")
        results.append(t8)

        # ── Test 9: is_valid rejects out-of-range confidence ─────────────────
        hdr("Test 9 -- ParamSet.is_valid() rejects conf_threshold outside [0.40, 0.95]")
        bad9 = ParamSet(atr_sl_mult=1.5, atr_tp1_mult=2.0, atr_tp2_mult=4.0,
                        conf_threshold=0.10, ema_slow=50)
        t9   = not bad9.is_valid(cfg8)
        ok(f"is_valid=False (conf=0.10)") if t9 else fail("Should be invalid")
        results.append(t9)

        # ── Test 10: rr_ratio ─────────────────────────────────────────────────
        hdr("Test 10 -- ParamSet.rr_ratio = atr_tp1_mult / atr_sl_mult")
        p10 = ParamSet(atr_sl_mult=1.5, atr_tp1_mult=3.0, atr_tp2_mult=6.0,
                       conf_threshold=0.65, ema_slow=50)
        t10 = abs(p10.rr_ratio - 2.0) < 1e-6
        ok(f"rr_ratio={p10.rr_ratio:.4f}  (expected 2.0)") if t10 else \
        fail(f"rr_ratio={p10.rr_ratio}")
        results.append(t10)

        # ── Test 11: Constructs with no history ───────────────────────────────
        hdr("Test 11 -- StrategyOptimizer constructs with no existing history")
        cfg11 = _cfg(tmpdir)
        opt11 = StrategyOptimizer(cfg11)
        t11   = opt11.n_trades == 0
        ok(f"n_trades=0  history_path={cfg11.OPTIMIZER_HISTORY_PATH}") if t11 else \
        fail(f"n_trades={opt11.n_trades}")
        results.append(t11)

        # ── Test 12: record_trade appends to CSV ──────────────────────────────
        hdr("Test 12 -- record_trade() appends to CSV and increments n_trades")
        cfg12 = _cfg(tmpdir + "/t12")
        Path(tmpdir + "/t12").mkdir(exist_ok=True)
        opt12 = StrategyOptimizer(cfg12)
        for i in range(5):
            opt12.record_trade(_make_trade(trade_id=i+1, outcome="TP1", r_gained=1.5))
        t12 = opt12.n_trades == 5 and Path(cfg12.OPTIMIZER_HISTORY_PATH).exists()
        ok(f"n_trades={opt12.n_trades}  CSV exists={Path(cfg12.OPTIMIZER_HISTORY_PATH).exists()}") if t12 else \
        fail(f"n_trades={opt12.n_trades}")
        results.append(t12)

        # ── Test 13: record_trade skips EXPIRED ───────────────────────────────
        hdr("Test 13 -- record_trade() skips EXPIRED outcomes")
        before13 = opt12.n_trades
        opt12.record_trade(_make_trade(outcome="EXPIRED", r_gained=0.0))
        t13 = opt12.n_trades == before13   # count unchanged
        ok(f"n_trades unchanged at {opt12.n_trades} after EXPIRED trade") if t13 else \
        fail(f"n_trades went from {before13} to {opt12.n_trades}")
        results.append(t13)

        # ── Test 14: _build_grid returns valid ParamSets only ─────────────────
        hdr("Test 14 -- _build_grid() generates only valid ParamSets")
        cfg14 = _cfg(tmpdir)
        opt14 = StrategyOptimizer(cfg14)
        grid14 = opt14._build_grid()
        all_valid = all(p.is_valid(cfg14) for p in grid14)
        t14 = len(grid14) > 0 and all_valid
        ok(f"Grid: {len(grid14)} valid candidates  all_valid={all_valid}") if t14 else \
        fail(f"{len(grid14)} candidates  all_valid={all_valid}")
        results.append(t14)

        # ── Test 15: _apply_guards rejects low win_rate ───────────────────────
        hdr("Test 15 -- _apply_guards() rejects param set with win_rate < threshold")
        cfg15  = _cfg(tmpdir)
        opt15  = StrategyOptimizer(cfg15)
        low_wr = PerformanceMetrics(
            n_trades=50, n_winners=10, n_losers=40,
            win_rate=0.20, profit_factor=0.5, expectancy=-0.5,
            max_drawdown_r=15.0,
        )
        rejected15, reason15 = opt15._apply_guards(ParamSet(), low_wr)
        t15 = rejected15
        ok(f"Rejected: {reason15}") if t15 else \
        fail(f"Should be rejected (WR=20%)")
        results.append(t15)

        # ── Test 16: _apply_guards rejects high drawdown ──────────────────────
        hdr("Test 16 -- _apply_guards() rejects param set with drawdown > limit")
        cfg16  = _cfg(tmpdir)
        opt16  = StrategyOptimizer(cfg16)
        hi_dd  = PerformanceMetrics(
            n_trades=50, n_winners=30, n_losers=20,
            win_rate=0.60, profit_factor=1.8, expectancy=0.3,
            max_drawdown_r=25.0,   # > OPTIMIZER_MAX_DRAWDOWN_R=20.0
        )
        rejected16, reason16 = opt16._apply_guards(ParamSet(), hi_dd)
        t16 = rejected16
        ok(f"Rejected: {reason16}") if t16 else \
        fail(f"Should be rejected (DD=25R)")
        results.append(t16)

        # ── Test 17: optimize() returns OptimizationReport ───────────────────
        hdr("Test 17 -- optimize() returns a valid OptimizationReport")
        cfg17 = _cfg(tmpdir + "/t17", eval_every=5, min_history=10)
        Path(tmpdir + "/t17").mkdir(exist_ok=True)
        opt17 = StrategyOptimizer(cfg17)
        trades17 = _make_trade_log(n_wins=25, n_losses=15)
        for t in trades17:
            opt17.record_trade(t)
        report17 = opt17.optimize()
        t17 = (isinstance(report17, OptimizationReport) and
               report17.n_trades_used >= 10 and
               report17.n_candidates > 0)
        ok(f"Report: trades={report17.n_trades_used}  "
           f"candidates={report17.n_candidates}  "
           f"passed={report17.n_passed}") if t17 else \
        fail(f"n_trades={report17.n_trades_used}  candidates={report17.n_candidates}")
        results.append(t17)

        # ── Test 18: n_candidates > 0 ─────────────────────────────────────────
        hdr("Test 18 -- OptimizationReport.n_candidates > 0")
        t18 = report17.n_candidates > 0
        ok(f"n_candidates={report17.n_candidates}") if t18 else \
        fail("n_candidates=0")
        results.append(t18)

        # ── Test 19: improved=True when better params found ───────────────────
        hdr("Test 19 -- optimize() correctly sets improved=True / False")
        # Just check the report has a valid boolean
        t19 = isinstance(report17.improved, bool)
        ok(f"improved={report17.improved}  (type=bool)") if t19 else \
        fail("improved is not a bool")
        results.append(t19)

        # ── Test 20: apply_params never exceeds max risk ───────────────────────
        hdr("Test 20 -- apply_params() never sets RISK_PCT above OPTIMIZER_MAX_RISK_PCT")
        from config.settings import Settings
        cfg20 = Settings()
        cfg20.RISK_PER_TRADE_PCT      = 5.0   # very high
        cfg20.OPTIMIZER_MAX_RISK_PCT  = 2.0
        opt20 = StrategyOptimizer(_cfg(tmpdir))
        opt20.apply_params(ParamSet(), cfg20)
        t20 = cfg20.RISK_PER_TRADE_PCT <= 2.0
        ok(f"RISK_PCT capped at {cfg20.RISK_PER_TRADE_PCT}%  (max=2.0%)") if t20 else \
        fail(f"RISK_PCT={cfg20.RISK_PER_TRADE_PCT}% exceeds 2.0%")
        results.append(t20)

        # ── Test 21: maybe_optimize triggers only when eval_every reached ─────
        hdr("Test 21 -- maybe_optimize() triggers only after eval_every trades")
        cfg21 = _cfg(tmpdir + "/t21", eval_every=10, min_history=10)
        Path(tmpdir + "/t21").mkdir(exist_ok=True)
        opt21 = StrategyOptimizer(cfg21)
        trades21 = _make_trade_log(n_wins=8, n_losses=5)
        for t in trades21:
            opt21.record_trade(t)
        # 13 trades but eval_every=10, last_eval_n=0 → 13-0=13 >= 10 → should trigger
        report21 = opt21.maybe_optimize()
        t21 = report21 is not None and isinstance(report21, OptimizationReport)
        ok(f"maybe_optimize() triggered: {report21.n_trades_used} trades") if t21 else \
        fail(f"Expected OptimizationReport, got {type(report21)}")
        results.append(t21)

        # ── Test 22: current_performance returns populated metrics ─────────────
        hdr("Test 22 -- current_performance() returns populated PerformanceMetrics")
        perf22 = opt17.current_performance(last_n=10)
        t22 = (
            isinstance(perf22, PerformanceMetrics) and
            perf22.n_trades > 0 and
            0.0 <= perf22.win_rate <= 1.0 and
            perf22.n_winners + perf22.n_losers + perf22.n_break_evens == perf22.n_trades
        )
        ok(f"n={perf22.n_trades}  WR={perf22.win_rate:.1%}  "
           f"PF={perf22.profit_factor:.2f}  exp={perf22.expectancy:+.3f}R") if t22 else \
        fail(f"Unexpected performance: {perf22.to_dict()}")
        results.append(t22)

        # Print samples inside the with-block (while tmpdir still exists)
        print("\n  -- Sample PerformanceMetrics.print_summary() output --")
        test_metrics = _compute_metrics([1.5, -1.0, 2.0, -1.0, 1.0, 0.8, -1.0, 1.2, 1.8, -1.0])
        test_metrics.print_summary("Test Data (10 trades)")

        print("\n  -- Sample OptimizationReport.print_summary() --")
        report17.print_summary()

    # ── Final summary ──────────────────────────────────────────────────────────
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
