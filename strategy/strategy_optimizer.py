"""
strategy/strategy_optimizer.py
================================
Continuous strategy performance evaluator and parameter optimizer.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GOAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Track every resolved trade, store it in a persistent CSV log,
and periodically evaluate strategy performance.  When enough trades
have accumulated, run a grid search over key parameters and adopt the
best-performing configuration — subject to strictly enforced risk limits.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TRACKED METRICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • Win Rate            — winners / total resolved trades
  • Average R:R         — mean R gained per winner
  • Profit Factor       — gross profit R / gross loss R
  • Expectancy          — average R per trade (winners + losers)
  • Max Drawdown (R)    — largest peak-to-trough equity decline in R
  • Consecutive losses  — current losing streak (live risk guard)
  • Sharpe Ratio        — trade-level R Sharpe (R / σ_R × √n_per_year)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GRID SEARCH PARAMETERS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Axis 1 — ATR_SL_MULTIPLIER   (SL = ATR × k)
  Axis 2 — ATR_TP1_MULTIPLIER  (TP1 = ATR × k)
  Axis 3 — ATR_TP2_MULTIPLIER  (TP2 = ATR × k)
  Axis 4 — CONFIDENCE_THRESHOLD
  Axis 5 — EMA_SLOW_PERIOD

  Grid is defined in .env via "min,max,step" strings.
  Default grid sizes are small (3–5 steps per axis) to keep runtime
  manageable.  Each candidate set is scored on the stored trade history
  using a replay simulation (not a full backtest — uses stored outcomes).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RISK GUARD (ALWAYS ENFORCED)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. RISK_PER_TRADE_PCT never exceeds OPTIMIZER_MAX_RISK_PCT (default 2%).
  2. Candidate sets with win_rate  < OPTIMIZER_MIN_WIN_RATE  are rejected.
  3. Candidate sets with PF        < OPTIMIZER_MIN_PROFIT_FACTOR are rejected.
  4. Candidate sets with drawdown  > OPTIMIZER_MAX_DRAWDOWN_R  are rejected.
  5. TP1 multiplier must exceed SL multiplier (ensures positive R:R).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EVALUATION CADENCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • record_trade(trade_record)     — call after every resolved trade
  • maybe_optimize()               — checks if OPTIMIZER_EVAL_EVERY trades
                                     have accumulated since last run and
                                     triggers optimize() if so
  • optimize()                     — force a full grid search now

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PUBLIC API
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  opt = StrategyOptimizer(settings)

  # Record a trade after it resolves (live loop or backtest post-process)
  opt.record_trade(trade_record)

  # Call every loop cycle — no-op if it's not time yet
  opt.maybe_optimize()

  # Force an immediate optimization run
  report = opt.optimize()
  report.print_summary()
  report.best_params    # → OptimizedParams (apply manually or auto-apply)

  # Apply best params back to settings (mutates settings in-place)
  opt.apply_params(report.best_params, settings)

  # Current performance snapshot
  perf = opt.current_performance()

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PERSISTENCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Trade history → CSV  (OPTIMIZER_HISTORY_PATH)
  Optimizer report     → JSON + human-readable TXT in OPTIMIZER_OUTPUT_DIR

Dependencies
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  pandas, numpy, itertools, config.settings.Settings, utils.logger
"""

from __future__ import annotations

import csv
import itertools
import json
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import Settings
from utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# CSV columns written to trade history
_TRADE_CSV_FIELDS = [
    "trade_id", "timestamp", "direction", "outcome",
    "r_gained", "confidence", "session", "regime",
    "has_ob", "has_fvg", "has_bos", "has_liq_sweep",
    "atr_sl_mult", "atr_tp1_mult", "atr_tp2_mult",
    "ema_slow", "conf_threshold",
]

# Grid axis names (must match ParamSet fields)
_GRID_AXES = [
    "atr_sl_mult",
    "atr_tp1_mult",
    "atr_tp2_mult",
    "conf_threshold",
    "ema_slow",
]

# Scoring metrics supported
_SCORE_METRICS = ("expectancy", "profit_factor", "sharpe", "calmar")


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PerformanceMetrics:
    """
    Snapshot of current strategy performance calculated from trade history.

    All R values are normalised: 1R = distance from entry to stop-loss.
    """
    n_trades:            int   = 0
    n_winners:           int   = 0
    n_losers:            int   = 0
    n_break_evens:       int   = 0

    win_rate:            float = 0.0   # 0–1
    profit_factor:       float = 0.0   # gross_profit_R / gross_loss_R
    expectancy:          float = 0.0   # mean R per trade
    avg_winner_r:        float = 0.0
    avg_loser_r:         float = 0.0
    total_r:             float = 0.0

    max_drawdown_r:      float = 0.0   # peak-to-trough in R (positive = bad)
    max_drawdown_pct:    float = 0.0   # as % of peak equity
    max_consecutive_losses: int = 0
    current_streak:      int   = 0     # + for wins, - for losses
    sharpe_ratio:        float = 0.0
    calmar_ratio:        float = 0.0

    # Window performance (last N trades)
    last_n:              int   = 0
    last_n_win_rate:     float = 0.0
    last_n_expectancy:   float = 0.0

    is_degrading:        bool  = False  # True if recent > overall performance gap is large

    def score(self, metric: str = "expectancy") -> float:
        """Return the primary score for ranking parameter sets."""
        mapping = {
            "expectancy":    self.expectancy,
            "profit_factor": self.profit_factor,
            "sharpe":        self.sharpe_ratio,
            "calmar":        self.calmar_ratio,
        }
        return mapping.get(metric, self.expectancy)

    def to_dict(self) -> dict:
        return {
            "n_trades":          self.n_trades,
            "win_rate":          round(self.win_rate,         4),
            "profit_factor":     round(self.profit_factor,    4),
            "expectancy":        round(self.expectancy,       4),
            "avg_winner_r":      round(self.avg_winner_r,     4),
            "avg_loser_r":       round(self.avg_loser_r,      4),
            "total_r":           round(self.total_r,          4),
            "max_drawdown_r":    round(self.max_drawdown_r,   4),
            "max_drawdown_pct":  round(self.max_drawdown_pct, 4),
            "max_cons_losses":   self.max_consecutive_losses,
            "sharpe_ratio":      round(self.sharpe_ratio,     4),
            "calmar_ratio":      round(self.calmar_ratio,     4),
            "is_degrading":      self.is_degrading,
        }

    def print_summary(self, title: str = "Current Performance") -> None:
        sep = "─" * 52
        print(f"\n{'═'*52}")
        print(f"  {title}")
        print(f"{'═'*52}")
        print(f"  Trades:          {self.n_trades:>6}")
        print(f"  Winners:         {self.n_winners:>6}  ({self.win_rate:.1%})")
        print(f"  Losers:          {self.n_losers:>6}")
        print(f"{sep}")
        print(f"  Win Rate:        {self.win_rate:.3f}")
        print(f"  Profit Factor:   {self.profit_factor:.3f}")
        print(f"  Expectancy:      {self.expectancy:+.3f} R/trade")
        print(f"  Total R:         {self.total_r:+.3f} R")
        print(f"{sep}")
        print(f"  Max Drawdown:    {self.max_drawdown_r:.2f} R  ({self.max_drawdown_pct:.1%})")
        print(f"  Max Consec Loss: {self.max_consecutive_losses}")
        print(f"  Sharpe:          {self.sharpe_ratio:.3f}")
        print(f"  Calmar:          {self.calmar_ratio:.3f}")
        if self.last_n > 0:
            print(f"{sep}")
            print(f"  Last {self.last_n} trades:")
            print(f"    Win Rate:      {self.last_n_win_rate:.1%}")
            print(f"    Expectancy:    {self.last_n_expectancy:+.3f} R/trade")
        if self.is_degrading:
            print(f"\n  ⚠  DEGRADING — recent performance below overall baseline")
        print(f"{'═'*52}\n")


@dataclass
class ParamSet:
    """
    A single candidate parameter configuration for grid search.

    All values correspond to Settings fields that the optimizer may adjust.
    Default values mirror the baseline configuration.
    """
    atr_sl_mult:     float = 1.5    # ATR_SL_MULTIPLIER
    atr_tp1_mult:    float = 2.0    # ATR_TP1_MULTIPLIER
    atr_tp2_mult:    float = 4.0    # ATR_TP2_MULTIPLIER
    conf_threshold:  float = 0.68   # CONFIDENCE_THRESHOLD
    ema_slow:        int   = 50     # EMA_SLOW_PERIOD

    @property
    def rr_ratio(self) -> float:
        """R:R of TP1 relative to SL."""
        return self.atr_tp1_mult / self.atr_sl_mult if self.atr_sl_mult > 0 else 0.0

    def is_valid(self, cfg: Settings) -> bool:
        """
        Hard validity checks — any failing disqualifies the set instantly.
        """
        if self.atr_tp1_mult <= self.atr_sl_mult:
            return False   # TP1 must be further than SL
        if self.atr_tp2_mult <= self.atr_tp1_mult:
            return False   # TP2 must be further than TP1
        if not (0.40 <= self.conf_threshold <= 0.95):
            return False
        if not (20 <= self.ema_slow <= 200):
            return False
        return True

    def to_dict(self) -> dict:
        return {
            "atr_sl_mult":    self.atr_sl_mult,
            "atr_tp1_mult":   self.atr_tp1_mult,
            "atr_tp2_mult":   self.atr_tp2_mult,
            "conf_threshold": self.conf_threshold,
            "ema_slow":       self.ema_slow,
            "rr_ratio_tp1":   round(self.rr_ratio, 3),
        }

    def __str__(self) -> str:
        return (
            f"ParamSet("
            f"SL={self.atr_sl_mult}× ATR  "
            f"TP1={self.atr_tp1_mult}× ATR  "
            f"TP2={self.atr_tp2_mult}× ATR  "
            f"conf≥{self.conf_threshold:.0%}  "
            f"EMA_slow={self.ema_slow})"
        )


@dataclass
class GridSearchResult:
    """Performance of one ParamSet evaluated on the trade history."""
    params:       ParamSet
    metrics:      PerformanceMetrics
    score:        float            # scalar used for ranking
    score_metric: str
    rejected:     bool    = False  # True if it failed a risk/quality guard
    reject_reason: str    = ""

    def to_dict(self) -> dict:
        return {
            **self.params.to_dict(),
            **self.metrics.to_dict(),
            "score":        round(self.score, 5),
            "score_metric": self.score_metric,
            "rejected":     self.rejected,
            "reject_reason": self.reject_reason,
        }


@dataclass
class OptimizationReport:
    """
    Complete output of one optimizer run.

    Attributes
    ----------
    run_id          : ISO timestamp of this optimization run.
    n_trades_used   : Number of trades in the history at run time.
    n_candidates    : Total grid points evaluated.
    n_passed        : Grid points passing all risk/quality guards.
    best_params     : ParamSet with highest score (or None if none passed).
    current_params  : ParamSet loaded from settings before the run.
    current_metrics : Performance of the existing parameter set.
    all_results     : All evaluated GridSearchResults, sorted best → worst.
    improved        : True if best_params outperforms current_params.
    """
    run_id:          str
    n_trades_used:   int
    n_candidates:    int                   = 0
    n_passed:        int                   = 0
    best_params:     Optional[ParamSet]    = None
    current_params:  Optional[ParamSet]    = None
    current_metrics: Optional[PerformanceMetrics] = None
    all_results:     list[GridSearchResult] = field(default_factory=list)
    improved:        bool                   = False
    score_metric:    str                    = "expectancy"

    def print_summary(self) -> None:
        sep = "─" * 60
        print(f"\n{'═'*60}")
        print(f"  OPTIMIZATION REPORT   [{self.run_id}]")
        print(f"{'═'*60}")
        print(f"  Trades used:     {self.n_trades_used}")
        print(f"  Grid candidates: {self.n_candidates}")
        print(f"  Passed guards:   {self.n_passed}")
        print(f"  Score metric:    {self.score_metric}")
        print(f"{sep}")

        if self.current_metrics:
            p_cur = self.current_params
            m_cur = self.current_metrics
            print(f"  Current params:  {p_cur}")
            print(f"  Current score:   {m_cur.score(self.score_metric):+.4f}")
            print(f"  Current WR:      {m_cur.win_rate:.1%}  PF={m_cur.profit_factor:.2f}"
                  f"  DD={m_cur.max_drawdown_r:.2f}R")
        print(f"{sep}")

        if self.best_params:
            m_best = self.all_results[0].metrics if self.all_results else None
            print(f"  BEST params:     {self.best_params}")
            if m_best:
                print(f"  Best score:      {self.all_results[0].score:+.4f}")
                print(f"  Best WR:         {m_best.win_rate:.1%}  "
                      f"PF={m_best.profit_factor:.2f}  "
                      f"DD={m_best.max_drawdown_r:.2f}R")
            if self.improved:
                print(f"\n  ✅ IMPROVEMENT FOUND — recommend applying new params")
            else:
                print(f"\n  ℹ  Current params already optimal — no change recommended")
        else:
            print(f"  ⚠  No parameter set passed all quality/risk guards")

        print(f"\n  Top 5 results:")
        for i, r in enumerate(self.all_results[:5], 1):
            flag = "✅" if not r.rejected else "❌"
            print(
                f"    {i}. {flag}  score={r.score:+.4f}  "
                f"WR={r.metrics.win_rate:.0%}  "
                f"PF={r.metrics.profit_factor:.2f}  "
                f"DD={r.metrics.max_drawdown_r:.2f}R  "
                f"SL={r.params.atr_sl_mult}×  "
                f"conf={r.params.conf_threshold:.0%}"
            )
        print(f"{'═'*60}\n")

    def to_dict(self) -> dict:
        return {
            "run_id":          self.run_id,
            "n_trades_used":   self.n_trades_used,
            "n_candidates":    self.n_candidates,
            "n_passed":        self.n_passed,
            "improved":        self.improved,
            "score_metric":    self.score_metric,
            "best_params":     self.best_params.to_dict() if self.best_params else None,
            "current_params":  self.current_params.to_dict() if self.current_params else None,
            "current_score":   (self.current_metrics.score(self.score_metric)
                                if self.current_metrics else None),
            "top_results":     [r.to_dict() for r in self.all_results[:10]],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Performance calculation helper
# ─────────────────────────────────────────────────────────────────────────────

def _compute_metrics(
    r_series:       list[float],
    last_n:         int = 20,
    trades_per_year: int = 200,
) -> PerformanceMetrics:
    """
    Compute a full PerformanceMetrics object from a list of R values.

    Parameters
    ----------
    r_series        : List of per-trade R gained (positive = win, negative = loss).
    last_n          : Window size for recent-performance comparison.
    trades_per_year : Approximate trades per year (for Sharpe annualisation).

    Returns
    -------
    PerformanceMetrics
    """
    if not r_series:
        return PerformanceMetrics()

    r_arr = np.array(r_series, dtype=float)
    n     = len(r_arr)

    winners   = r_arr[r_arr > 0]
    losers    = r_arr[r_arr < 0]
    be        = r_arr[r_arr == 0.0]

    n_win = len(winners)
    n_los = len(losers)
    n_be  = len(be)

    win_rate     = n_win / n if n > 0 else 0.0
    total_r      = float(np.sum(r_arr))
    expectancy   = float(np.mean(r_arr)) if n > 0 else 0.0
    avg_win_r    = float(np.mean(winners)) if n_win > 0 else 0.0
    avg_los_r    = float(np.mean(losers))  if n_los > 0 else 0.0
    gross_profit = float(np.sum(winners))  if n_win > 0 else 0.0
    gross_loss   = float(abs(np.sum(losers))) if n_los > 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (
        float("inf") if gross_profit > 0 else 0.0
    )

    # Max drawdown (peak-to-trough equity curve in R)
    equity       = np.cumsum(np.insert(r_arr, 0, 0.0))
    peak         = np.maximum.accumulate(equity)
    drawdowns    = peak - equity
    max_dd_r     = float(np.max(drawdowns))
    peak_max     = float(np.max(peak)) if np.max(peak) > 0 else 1.0
    max_dd_pct   = max_dd_r / peak_max

    # Consecutive losses + current streak
    max_cons_loss = 0
    cur_loss      = 0
    cur_streak    = 0    # positive = win streak, negative = loss streak
    for r in r_arr:
        if r < 0:
            cur_loss  += 1
            max_cons_loss = max(max_cons_loss, cur_loss)
            # Reset win streak; go into loss streak
            cur_streak = min(cur_streak, 0) - 1
        elif r > 0:
            cur_loss   = 0
            # Reset loss streak; go into win streak
            cur_streak = max(cur_streak, 0) + 1
        else:
            cur_loss   = 0
            cur_streak = 0

    # Sharpe (trade-level: mean_R / std_R × sqrt(trades_per_year))
    std_r        = float(np.std(r_arr)) if n > 1 else 1.0
    sharpe       = (expectancy / std_r * math.sqrt(trades_per_year)) if std_r > 0 else 0.0

    # Calmar = annualised return / max drawdown
    annual_r     = expectancy * trades_per_year
    calmar       = annual_r / max_dd_r if max_dd_r > 0 else (
        float("inf") if annual_r > 0 else 0.0
    )
    calmar       = min(calmar, 99.9)   # cap at 99.9 to avoid inf in JSON

    # Recent window
    last_r       = r_arr[-last_n:] if n >= last_n else r_arr
    last_win_r   = float(np.mean(last_r[last_r > 0])) if np.any(last_r > 0) else 0.0
    last_wr      = float(np.mean(last_r > 0))
    last_exp     = float(np.mean(last_r))

    # Degradation check: recent expectancy significantly below overall
    is_degrading = last_exp < (expectancy - 0.30) and n >= last_n

    return PerformanceMetrics(
        n_trades             = n,
        n_winners            = n_win,
        n_losers             = n_los,
        n_break_evens        = n_be,
        win_rate             = round(win_rate, 4),
        profit_factor        = round(min(profit_factor, 99.9), 4),
        expectancy           = round(expectancy, 4),
        avg_winner_r         = round(avg_win_r,  4),
        avg_loser_r          = round(avg_los_r,  4),
        total_r              = round(total_r,    4),
        max_drawdown_r       = round(max_dd_r,   4),
        max_drawdown_pct     = round(max_dd_pct, 4),
        max_consecutive_losses = max_cons_loss,
        current_streak       = cur_streak,
        sharpe_ratio         = round(sharpe,  4),
        calmar_ratio         = round(calmar,  4),
        last_n               = len(last_r),
        last_n_win_rate      = round(last_wr,  4),
        last_n_expectancy    = round(last_exp, 4),
        is_degrading         = is_degrading,
    )


def _parse_range(spec: str) -> list[float]:
    """
    Parse a "min,max,step" string into an inclusive list of floats.

    Examples
    --------
    "1.0,2.5,0.5"  →  [1.0, 1.5, 2.0, 2.5]
    "0.55,0.80,0.05" →  [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    "30,70,10"       →  [30.0, 40.0, 50.0, 60.0, 70.0]
    """
    try:
        parts = [float(p.strip()) for p in spec.split(",")]
        if len(parts) != 3:
            raise ValueError(f"Expected 'min,max,step' got {spec!r}")
        lo, hi, step = parts
        vals = []
        v = lo
        while v <= hi + 1e-9:
            vals.append(round(v, 6))
            v += step
        return vals
    except Exception as exc:
        logger.error(f"_parse_range({spec!r}) failed: {exc} — using default [lo]")
        parts = spec.split(",")
        return [float(parts[0].strip())] if parts else [1.0]


# ─────────────────────────────────────────────────────────────────────────────
# Replay simulator (fast — re-scores stored trades against a ParamSet)
# ─────────────────────────────────────────────────────────────────────────────

def _replay_with_params(
    df_history: pd.DataFrame,
    params:     ParamSet,
) -> list[float]:
    """
    Fast simulation that applies a ParamSet to the stored trade history and
    returns a corrected R series.

    Because we don't have bar data the optimizer uses the STORED OUTCOMES
    (SL / TP1 / TP2 / BE) and adjusts only for changes in R:R ratio and
    confidence threshold filtering.

    Adjustment logic
    ─────────────────
    1. If confidence  < new conf_threshold  → trade would NOT have been taken.
    2. If outcome == "TP1":  new_r = 0.5 × (atr_tp1_mult / atr_sl_mult)
                              (partial close: 50% at TP1)
    3. If outcome == "TP2":  new_r = 0.5 × (atr_tp1_mult / atr_sl_mult)
                                    + 0.5 × (atr_tp2_mult / atr_sl_mult)
    4. If outcome == "SL":   new_r = -1.0   (always 1R loss)
    5. Break-even / expired: new_r = 0.0
    """
    r_list = []
    conf_col = "conf_threshold" if "conf_threshold" in df_history.columns else "confidence"

    tp1_r  = params.atr_tp1_mult / params.atr_sl_mult
    tp2_r  = params.atr_tp2_mult / params.atr_sl_mult

    for _, row in df_history.iterrows():
        outcome = str(row.get("outcome", "EXPIRED"))
        if outcome == "EXPIRED":
            continue

        # Confidence gate — trade would be skipped under new threshold
        try:
            conf = float(row.get(conf_col, row.get("confidence", 1.0)))
        except (ValueError, TypeError):
            conf = 1.0

        if conf < params.conf_threshold:
            continue   # trade filtered out

        if outcome in ("TP1",):
            r_list.append(0.5 * tp1_r)           # partial close at TP1
        elif outcome in ("TP2",):
            r_list.append(0.5 * tp1_r + 0.5 * tp2_r)
        elif outcome in ("SL", "STOP_LOSS"):
            r_list.append(-1.0)
        else:
            r_list.append(0.0)   # BE / unknown

    return r_list


# ─────────────────────────────────────────────────────────────────────────────
# Main optimizer class
# ─────────────────────────────────────────────────────────────────────────────

class StrategyOptimizer:
    """
    Continuously tracks trade performance and periodically runs grid search
    to find better-performing parameter sets.

    Designed to be instantiated once at bot startup and kept alive for the
    duration of the session.  The trade history CSV persists across restarts.

    Example
    -------
        from strategy.strategy_optimizer import StrategyOptimizer
        from config.settings import get_settings

        cfg = get_settings()
        opt = StrategyOptimizer(cfg)

        # In the main loop, after each resolved trade:
        opt.record_trade(trade_record)
        opt.maybe_optimize()

        # After optimization, apply recommended params:
        report = opt.last_report
        if report and report.improved and report.best_params:
            opt.apply_params(report.best_params, cfg)
    """

    def __init__(self, settings: Settings) -> None:
        self.cfg            = settings
        self._history_path  = Path(settings.OPTIMIZER_HISTORY_PATH)
        self._output_dir    = Path(settings.OPTIMIZER_OUTPUT_DIR)
        self._eval_every    = settings.OPTIMIZER_EVAL_EVERY
        self._min_history   = settings.OPTIMIZER_MIN_HISTORY
        self._score_metric  = settings.OPTIMIZER_SCORE_METRIC
        self._last_eval_n   = 0       # n_trades at last optimization run
        self.last_report:   Optional[OptimizationReport] = None

        # Ensure directories exist
        self._history_path.parent.mkdir(parents=True, exist_ok=True)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Load existing history count
        self._n_trades_on_disk = self._count_history_rows()

        logger.info(
            f"StrategyOptimizer initialised  "
            f"history={self._history_path}  "
            f"eval_every={self._eval_every}  "
            f"existing_trades={self._n_trades_on_disk}"
        )

    # =========================================================================
    # PUBLIC — Record a trade
    # =========================================================================

    def record_trade(self, trade_record, current_params: Optional[ParamSet] = None) -> None:
        """
        Append a resolved trade to the persistent CSV history.

        Parameters
        ----------
        trade_record   : A TradeRecord from BacktestEngine or the live bot.
                         Must have attributes: trade_id, outcome, r_gained,
                         confidence, session, regime, has_ob, has_fvg, has_bos,
                         has_liq_sweep.
        current_params : The ParamSet active when this trade was taken.
                         If None, uses values from Settings.

        Silently skips EXPIRED trades (not resolved → uninformative).
        """
        outcome = getattr(trade_record, "outcome", "EXPIRED")
        if outcome == "EXPIRED":
            return

        p = current_params or self._params_from_settings()

        row = {
            "trade_id":       getattr(trade_record, "trade_id", ""),
            "timestamp":      (getattr(trade_record, "signal_time", None) or
                               datetime.now(timezone.utc)).isoformat(),
            "direction":      getattr(trade_record, "direction", ""),
            "outcome":        outcome,
            "r_gained":       round(getattr(trade_record, "r_gained", 0.0) or 0.0, 4),
            "confidence":     round(getattr(trade_record, "confidence", 0.0) or 0.0, 4),
            "session":        getattr(trade_record, "session", ""),
            "regime":         getattr(trade_record, "regime", ""),
            "has_ob":         int(getattr(trade_record, "has_ob",        False)),
            "has_fvg":        int(getattr(trade_record, "has_fvg",       False)),
            "has_bos":        int(getattr(trade_record, "has_bos",       False)),
            "has_liq_sweep":  int(getattr(trade_record, "has_liq_sweep", False)),
            "atr_sl_mult":    p.atr_sl_mult,
            "atr_tp1_mult":   p.atr_tp1_mult,
            "atr_tp2_mult":   p.atr_tp2_mult,
            "ema_slow":       p.ema_slow,
            "conf_threshold": p.conf_threshold,
        }

        write_header = not self._history_path.exists() or self._history_path.stat().st_size == 0

        with open(self._history_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_TRADE_CSV_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        self._n_trades_on_disk += 1
        logger.debug(
            f"record_trade: trade_id={row['trade_id']}  "
            f"outcome={outcome}  r={row['r_gained']:+.2f}  "
            f"total={self._n_trades_on_disk}"
        )

    # =========================================================================
    # PUBLIC — Conditional trigger
    # =========================================================================

    def maybe_optimize(self) -> Optional[OptimizationReport]:
        """
        Run optimization if OPTIMIZER_EVAL_EVERY new trades have been recorded
        since the last run and the minimum history threshold is met.

        Returns the OptimizationReport if a run occurred, else None.
        """
        if not self.cfg.OPTIMIZER_ENABLED:
            return None

        n = self._n_trades_on_disk
        if n < self._min_history:
            logger.debug(
                f"maybe_optimize: {n} trades < min {self._min_history} — skipping"
            )
            return None

        new_trades = n - self._last_eval_n
        if new_trades < self._eval_every:
            logger.debug(
                f"maybe_optimize: {new_trades} new trades < eval_every={self._eval_every}"
            )
            return None

        logger.info(
            f"maybe_optimize: {new_trades} new trades since last run — triggering"
        )
        return self.optimize()

    # =========================================================================
    # PUBLIC — Force optimization
    # =========================================================================

    def optimize(self) -> OptimizationReport:
        """
        Run a full grid search and return an OptimizationReport.

        Steps
        -----
        1. Load trade history from CSV
        2. Compute current performance
        3. Build the grid of candidate parameter sets
        4. Score each candidate against the history (replay)
        5. Apply risk/quality guards, rank by score metric
        6. Assemble and persist the report
        7. Return the report
        """
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        logger.info(f"StrategyOptimizer.optimize() — run_id={run_id}")

        # ── Load history ──────────────────────────────────────────────────────
        df = self._load_history()
        if df is None or len(df) < self._min_history:
            logger.warning(
                f"optimize(): only {len(df) if df is not None else 0} trades in history "
                f"(need {self._min_history}) — aborting"
            )
            return OptimizationReport(
                run_id=run_id, n_trades_used=0, score_metric=self._score_metric
            )

        n_trades = len(df)
        logger.info(f"  Loaded {n_trades} resolved trades from history")

        # ── Current performance ───────────────────────────────────────────────
        current_params  = self._params_from_settings()
        current_r       = _replay_with_params(df, current_params)
        current_metrics = _compute_metrics(current_r)

        logger.info(
            f"  Current params: {current_params}"
        )
        logger.info(
            f"  Current metrics: WR={current_metrics.win_rate:.1%}  "
            f"PF={current_metrics.profit_factor:.2f}  "
            f"exp={current_metrics.expectancy:+.3f}R  "
            f"DD={current_metrics.max_drawdown_r:.2f}R"
        )

        # ── Build grid ────────────────────────────────────────────────────────
        grid = self._build_grid()
        n_candidates = len(grid)
        logger.info(f"  Grid: {n_candidates} candidates")

        # ── Evaluate each candidate ───────────────────────────────────────────
        all_results: list[GridSearchResult] = []

        for params in grid:
            r_series = _replay_with_params(df, params)
            if len(r_series) < max(10, self._min_history // 5):
                # Too few trades would survive this filter — skip
                result = GridSearchResult(
                    params=params,
                    metrics=PerformanceMetrics(),
                    score=-9999.0,
                    score_metric=self._score_metric,
                    rejected=True,
                    reject_reason="Insufficient trades after confidence filter",
                )
                all_results.append(result)
                continue

            metrics = _compute_metrics(r_series)
            rejected, reason = self._apply_guards(params, metrics)

            score = (
                metrics.score(self._score_metric)
                if not rejected
                else -9999.0
            )

            all_results.append(GridSearchResult(
                params        = params,
                metrics       = metrics,
                score         = round(score, 6),
                score_metric  = self._score_metric,
                rejected      = rejected,
                reject_reason = reason,
            ))

        # Sort: non-rejected first, by score descending
        all_results.sort(key=lambda r: (not r.rejected, r.score), reverse=True)

        # ── Determine best ────────────────────────────────────────────────────
        valid = [r for r in all_results if not r.rejected]
        n_passed = len(valid)

        best_result = valid[0] if valid else None
        best_params = best_result.params if best_result else None

        current_score = current_metrics.score(self._score_metric)
        best_score    = best_result.score if best_result else -9999.0
        improved      = best_score > current_score + 0.01   # +0.01 dead-band

        # Update last eval count
        self._last_eval_n = n_trades

        report = OptimizationReport(
            run_id          = run_id,
            n_trades_used   = n_trades,
            n_candidates    = n_candidates,
            n_passed        = n_passed,
            best_params     = best_params,
            current_params  = current_params,
            current_metrics = current_metrics,
            all_results     = all_results,
            improved        = improved,
            score_metric    = self._score_metric,
        )

        self.last_report = report

        logger.info(
            f"  Optimization done: {n_passed}/{n_candidates} passed  "
            f"improved={improved}"
        )
        if best_params:
            logger.info(f"  Best: {best_params}  score={best_score:+.4f}")

        # ── Persist ───────────────────────────────────────────────────────────
        self._save_report(report, run_id)

        return report

    # =========================================================================
    # PUBLIC — Apply best params to settings
    # =========================================================================

    def apply_params(self, params: ParamSet, settings: Settings) -> None:
        """
        Mutate a Settings object to reflect the optimal ParamSet.

        The risk guard is re-applied here: RISK_PER_TRADE_PCT is never
        raised above OPTIMIZER_MAX_RISK_PCT.

        Parameters
        ----------
        params   : ParamSet to apply.
        settings : Settings object to mutate in-place.
        """
        settings.ATR_SL_MULTIPLIER    = params.atr_sl_mult
        settings.ATR_TP1_MULTIPLIER   = params.atr_tp1_mult
        settings.ATR_TP2_MULTIPLIER   = params.atr_tp2_mult
        settings.CONFIDENCE_THRESHOLD = round(params.conf_threshold, 4)
        settings.EMA_SLOW_PERIOD      = int(params.ema_slow)

        # Hard risk cap — optimizer must never loosen beyond this
        if hasattr(settings, "RISK_PER_TRADE_PCT"):
            settings.RISK_PER_TRADE_PCT = min(
                settings.RISK_PER_TRADE_PCT,
                settings.OPTIMIZER_MAX_RISK_PCT,
            )

        logger.info(
            f"apply_params(): applied {params}  "
            f"risk={settings.RISK_PER_TRADE_PCT}%"
        )

    # =========================================================================
    # PUBLIC — Convenience metric snapshot
    # =========================================================================

    def current_performance(self, last_n: int = 20) -> PerformanceMetrics:
        """
        Compute and return current performance metrics from the trade history.

        Parameters
        ----------
        last_n : Window size for recent-performance sub-metrics.

        Returns
        -------
        PerformanceMetrics — empty if no history available.
        """
        df = self._load_history()
        if df is None or len(df) == 0:
            return PerformanceMetrics()

        current_params = self._params_from_settings()
        r_series       = _replay_with_params(df, current_params)
        return _compute_metrics(r_series, last_n=last_n)

    @property
    def n_trades(self) -> int:
        """Total resolved trades recorded to disk."""
        return self._n_trades_on_disk

    @property
    def trades_until_next_eval(self) -> int:
        """How many more trades before the next automatic evaluation."""
        done = self._n_trades_on_disk - self._last_eval_n
        remaining = self._eval_every - done
        return max(remaining, 0)

    def clear_history(self) -> None:
        """Delete the trade history CSV (use with caution)."""
        if self._history_path.exists():
            self._history_path.unlink()
            self._n_trades_on_disk = 0
            self._last_eval_n      = 0
            logger.warning(f"Trade history cleared: {self._history_path}")

    # =========================================================================
    # PRIVATE — Grid builder
    # =========================================================================

    def _build_grid(self) -> list[ParamSet]:
        """
        Build all candidate ParamSets from the configured ranges.

        Invalid sets (TP ≤ SL, out-of-range values) are filtered here.
        """
        axes = {
            "atr_sl_mult":    _parse_range(self.cfg.OPTIMIZER_ATR_SL_RANGE),
            "atr_tp1_mult":   _parse_range(self.cfg.OPTIMIZER_ATR_TP1_RANGE),
            "atr_tp2_mult":   _parse_range(self.cfg.OPTIMIZER_ATR_TP2_RANGE),
            "conf_threshold": _parse_range(self.cfg.OPTIMIZER_CONF_RANGE),
            "ema_slow":       _parse_range(self.cfg.OPTIMIZER_EMA_SLOW_RANGE),
        }

        grid: list[ParamSet] = []
        for combo in itertools.product(*axes.values()):
            params = ParamSet(
                atr_sl_mult    = combo[0],
                atr_tp1_mult   = combo[1],
                atr_tp2_mult   = combo[2],
                conf_threshold = combo[3],
                ema_slow       = int(combo[4]),
            )
            if params.is_valid(self.cfg):
                grid.append(params)

        logger.debug(f"  _build_grid: {len(grid)} valid candidates")
        return grid

    # =========================================================================
    # PRIVATE — Risk / quality guards
    # =========================================================================

    def _apply_guards(
        self, params: ParamSet, metrics: PerformanceMetrics
    ) -> tuple[bool, str]:
        """
        Return (rejected, reason) for a candidate parameter set.

        rejected=True means this set failed at least one guard.
        """
        cfg = self.cfg

        if metrics.win_rate < cfg.OPTIMIZER_MIN_WIN_RATE:
            return (
                True,
                f"win_rate={metrics.win_rate:.1%} < min {cfg.OPTIMIZER_MIN_WIN_RATE:.0%}"
            )

        if metrics.profit_factor < cfg.OPTIMIZER_MIN_PROFIT_FACTOR:
            return (
                True,
                f"profit_factor={metrics.profit_factor:.2f} < min {cfg.OPTIMIZER_MIN_PROFIT_FACTOR}"
            )

        if metrics.max_drawdown_r > cfg.OPTIMIZER_MAX_DRAWDOWN_R:
            return (
                True,
                f"drawdown={metrics.max_drawdown_r:.2f}R > max {cfg.OPTIMIZER_MAX_DRAWDOWN_R}R"
            )

        if metrics.expectancy <= 0:
            return (True, f"negative expectancy: {metrics.expectancy:+.3f}R")

        if metrics.n_trades < max(10, self.cfg.OPTIMIZER_MIN_HISTORY // 5):
            return (True, f"too few trades: {metrics.n_trades}")

        return (False, "")

    # =========================================================================
    # PRIVATE — CSV helpers
    # =========================================================================

    def _load_history(self) -> Optional[pd.DataFrame]:
        """Load and return the trade history CSV, or None if unavailable."""
        if not self._history_path.exists():
            return None
        try:
            df = pd.read_csv(self._history_path)
            df.columns = [c.strip().lower() for c in df.columns]
            # Exclude EXPIRED
            if "outcome" in df.columns:
                df = df[df["outcome"].str.upper() != "EXPIRED"].copy()
            if "r_gained" in df.columns:
                df["r_gained"] = pd.to_numeric(df["r_gained"], errors="coerce").fillna(0.0)
            return df
        except Exception as exc:
            logger.error(f"_load_history(): failed to read {self._history_path}: {exc}")
            return None

    def _count_history_rows(self) -> int:
        """Count non-header rows in the CSV without loading the full file."""
        if not self._history_path.exists():
            return 0
        try:
            with open(self._history_path, encoding="utf-8") as f:
                return max(sum(1 for _ in f) - 1, 0)   # -1 for header
        except Exception:
            return 0

    def _params_from_settings(self) -> ParamSet:
        """Build a ParamSet from the current Settings values."""
        return ParamSet(
            atr_sl_mult    = getattr(self.cfg, "ATR_SL_MULTIPLIER",  1.5),
            atr_tp1_mult   = getattr(self.cfg, "ATR_TP1_MULTIPLIER", 2.0),
            atr_tp2_mult   = getattr(self.cfg, "ATR_TP2_MULTIPLIER", 4.0),
            conf_threshold = getattr(self.cfg, "CONFIDENCE_THRESHOLD", 0.68),
            ema_slow       = int(getattr(self.cfg, "EMA_SLOW_PERIOD", 50)),
        )

    # =========================================================================
    # PRIVATE — Report persistence
    # =========================================================================

    def _save_report(self, report: OptimizationReport, run_id: str) -> None:
        """Save the report as JSON + human-readable TXT."""
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # JSON
        json_path = self._output_dir / f"opt_{run_id}.json"
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(report.to_dict(), f, indent=2, default=str)
            logger.info(f"  Report saved → {json_path}")
        except Exception as exc:
            logger.error(f"  Failed to save JSON report: {exc}")

        # Human-readable TXT
        txt_path = self._output_dir / f"opt_{run_id}.txt"
        try:
            import io, contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                report.print_summary()
            txt_path.write_text(buf.getvalue(), encoding="utf-8")
        except Exception:
            pass   # TXT is non-critical

        # Also write a "latest" snapshot (overwritten each run)
        latest_path = self._output_dir / "latest.json"
        try:
            with open(latest_path, "w", encoding="utf-8") as f:
                json.dump(report.to_dict(), f, indent=2, default=str)
        except Exception:
            pass
