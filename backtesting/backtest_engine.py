"""
backtesting/backtest_engine.py
================================
Event-driven bar-by-bar backtesting engine for the AntiGravity Gold Bot.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHILOSOPHY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The engine replays historical bars exactly as the live bot sees them.
At each bar close it feeds the history-so-far to SignalGenerator,
replicating lookback-window constraints  (no future-leak).

Outcome simulation
------------------
After a signal fires at bar i, bars [i+1 ...] are scanned in order.
The candle's High and Low determine which price is hit first:

  BUY  signal: SL=below entry, TP=above entry
    • If bar.low  ≤ SL  →  STOP_LOSS  (worst-case: assume SL hit before TP)
    • If bar.high ≥ TP1 →  TP1_HIT
    • If bar.high ≥ TP2 →  TP2_HIT   (only checked when TP1 already hit)

  SELL signal: SL=above entry, TP=below entry
    • If bar.high ≥ SL  →  STOP_LOSS
    • If bar.low  ≤ TP1 →  TP1_HIT
    • If bar.low  ≤ TP2 →  TP2_HIT

  Partial-TP strategy (realistic):
    • On TP1 hit: close 50% of position at TP1, move SL to break-even
    • Remaining 50% runs to TP2 or is stopped out at break-even (0 R)

  If no resolution within MAX_BARS_OPEN bars → EXPIRED (0 R, excluded by default)

Performance metrics computed
-----------------------------
  total_trades        — all non-expired closed trades
  winners / losers    — trades that hit TP1 or better / hit SL
  win_rate            — winners / total_trades
  profit_factor       — gross_profit / gross_loss  (in R)
  expectancy          — average R per trade
  avg_winner_r        — average R on winning trades
  avg_loser_r         — average R on losing trades  (always ≤ -1.0 R)
  max_drawdown_pct    — peak-to-trough equity drawdown as % of peak equity
  max_drawdown_r      — same in R units
  max_consecutive_losses
  max_consecutive_wins
  sharpe_ratio        — approximate (daily R / σ_daily_R × √252)
  calmar_ratio        — annualised return / max_drawdown
  signals_per_day     — average number of signals generated per trading day
  total_r             — net R gained across all trades

Public API
----------
  engine = BacktestEngine(settings)

  # Option A — run on a DataFrame already in memory
  result = engine.run(df_h1, df_m15, start="2024-01-01", end="2024-03-31")

  # Option B — load from CSV files
  result = engine.run_from_csv(
      h1_csv  = "data/XAUUSD_H1.csv",
      m15_csv = "data/XAUUSD_M15.csv",
      start   = "2024-01-01",
      end     = "2024-03-31",
  )

  result.print_summary()            # → console
  result.to_dict()                  # → machine-readable dict
  engine.export(result, "out/")     # → CSV + txt summary

Walk-forward validation
-----------------------
  folds = engine.walk_forward(df_h1, df_m15, n_folds=4)
  for fold_result in folds:
      fold_result.print_summary()

Dependencies
------------
  pandas, numpy, strategy.*, signals.*, filters.*, risk.*
  config.settings.Settings, utils.logger
"""

from __future__ import annotations

import csv
import io
import json
import math
import contextlib
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import Settings
from filters.session_filter import SessionFilter
from filters.volatility_filter import VolatilityFilter
from risk.position_sizing import PositionSizer
from signals.signal_generator import SignalGenerator
from strategy.market_regime import MarketRegimeClassifier
from strategy.smart_money_strategy import SmartMoneyStrategy
from strategy.trend_detection import TrendDetector
from utils.logger import get_logger

logger = get_logger(__name__)

# Optional matplotlib — chart generation gracefully disabled if not installed
try:
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend — safe in headless environments
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import matplotlib.patches as mpatches
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False
    logger.warning("matplotlib not installed — equity curve chart disabled")


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Maximum bars to hold a trade open before treating it as EXPIRED
MAX_BARS_OPEN: int = 32   # 32 × 15-min = 8 hours

# Partial-TP split: fraction of position closed at TP1
TP1_CLOSE_FRACTION: float = 0.50

# R value awarded for TP1 partial close (normalised: 1R = risk distance)
# Full TP1 = 1.0 × R:R_tp1.  At 50% close it's 0.5 × R:R_tp1.
# After TP1, remaining 50% runs to TP2 or BE (0 R).

# Minimum bars of price history the scanner needs before it can produce signals
MIN_LOOKBACK_BARS: int = 210   # EMA 200 warmup


# ─────────────────────────────────────────────────────────────────────────────
# Trade record
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TradeRecord:
    """
    Represents one simulated trade from signal generation to outcome.

    All monetary fields are in R-multiples (1 R = amount risked on the trade).
    Keeping everything in R makes results account-size-agnostic.
    """
    # ── Identity ──────────────────────────────────────────────────────────────
    trade_id:       int
    signal_bar:     int               # bar index in the full DataFrame
    signal_time:    datetime
    direction:      str               # "BUY" | "SELL"

    # ── Levels ────────────────────────────────────────────────────────────────
    entry:          float
    stop_loss:      float
    take_profit_1:  float
    take_profit_2:  Optional[float]

    # ── Signal metadata ───────────────────────────────────────────────────────
    confidence:     float
    has_ob:         bool
    has_fvg:        bool
    has_bos:        bool
    has_choch:      bool
    has_liq_sweep:  bool
    session:        str
    regime:         str

    # ── Outcome (filled after simulation) ────────────────────────────────────
    outcome:        str   = "OPEN"    # "TP1" | "TP2" | "SL" | "BE" | "EXPIRED"
    outcome_bar:    int   = -1
    outcome_time:   Optional[datetime] = None
    bars_held:      int   = 0
    r_gained:       float = 0.0       # net R on this trade

    # ── Derived ───────────────────────────────────────────────────────────────
    @property
    def is_winner(self) -> bool:
        return self.outcome in ("TP1", "TP2") and self.r_gained > 0

    @property
    def is_loser(self) -> bool:
        return self.outcome == "SL"

    @property
    def risk_distance(self) -> float:
        return abs(self.entry - self.stop_loss)

    @property
    def rr_tp1(self) -> float:
        if self.risk_distance == 0:
            return 0.0
        return abs(self.take_profit_1 - self.entry) / self.risk_distance

    @property
    def rr_tp2(self) -> Optional[float]:
        if self.take_profit_2 is None or self.risk_distance == 0:
            return None
        return abs(self.take_profit_2 - self.entry) / self.risk_distance

    def to_dict(self) -> dict:
        return {
            "trade_id":       self.trade_id,
            "signal_time":    self.signal_time.isoformat() if self.signal_time else "",
            "outcome_time":   self.outcome_time.isoformat() if self.outcome_time else "",
            "direction":      self.direction,
            "entry":          round(self.entry, 2),
            "stop_loss":      round(self.stop_loss, 2),
            "take_profit_1":  round(self.take_profit_1, 2),
            "take_profit_2":  round(self.take_profit_2, 2) if self.take_profit_2 else None,
            "rr_tp1":         round(self.rr_tp1, 2),
            "rr_tp2":         round(self.rr_tp2, 2) if self.rr_tp2 else None,
            "outcome":        self.outcome,
            "bars_held":      self.bars_held,
            "r_gained":       round(self.r_gained, 4),
            "confidence":     round(self.confidence, 3),
            "has_ob":         self.has_ob,
            "has_fvg":        self.has_fvg,
            "has_bos":        self.has_bos,
            "has_choch":      self.has_choch,
            "has_liq_sweep":  self.has_liq_sweep,
            "session":        self.session,
            "regime":         self.regime,
            "signal_bar":     self.signal_bar,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Results dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    """
    Aggregated performance statistics from a completed backtest run.
    All R-based fields use 1R = initial risk per trade.
    """
    # ── Metadata ──────────────────────────────────────────────────────────────
    start_date:         str   = ""
    end_date:           str   = ""
    symbol:             str   = "XAUUSD"
    timeframe:          str   = "M15"
    total_bars:         int   = 0
    total_bars_scanned: int   = 0

    # ── Volume ────────────────────────────────────────────────────────────────
    total_signals_generated: int = 0   # raw SMC ideas before any filter
    total_trades:            int = 0   # trades that reached resolution
    winners:                 int = 0
    losers:                  int = 0
    break_evens:             int = 0
    expired:                 int = 0

    # ── Core metrics ──────────────────────────────────────────────────────────
    win_rate:          float = 0.0   # winners / total_trades
    profit_factor:     float = 0.0   # gross_profit_R / |gross_loss_R|
    expectancy:        float = 0.0   # average R per trade
    total_r:           float = 0.0   # cumulative R across all trades

    # ── R-multiples ───────────────────────────────────────────────────────────
    avg_winner_r:      float = 0.0
    avg_loser_r:       float = 0.0
    largest_winner_r:  float = 0.0
    largest_loser_r:   float = 0.0

    # ── Drawdown ──────────────────────────────────────────────────────────────
    max_drawdown_r:    float = 0.0   # peak-to-trough in R
    max_drawdown_pct:  float = 0.0   # as % of cumulative peak equity (approx)

    # ── Streaks ───────────────────────────────────────────────────────────────
    max_consecutive_wins:   int = 0
    max_consecutive_losses: int = 0

    # ── Ratios ────────────────────────────────────────────────────────────────
    sharpe_ratio:      float = 0.0   # risk-adjusted return (daily R)
    calmar_ratio:      float = 0.0   # net_r / |max_drawdown_r|

    # ── Throughput ────────────────────────────────────────────────────────────
    signals_per_day:   float = 0.0
    trading_days:      int   = 0

    # ── Trade log ────────────────────────────────────────────────────────────
    trade_log: list[TradeRecord] = field(default_factory=list)

    # ── Equity curve (cumulative R, one point per resolved trade) ─────────────
    equity_curve: list[float] = field(default_factory=list)

    # ── Monthly breakdown [{month, trades, win_rate, total_r}, ...] ───────────
    monthly_breakdown: list[dict] = field(default_factory=list)

    # =========================================================================

    def print_summary(self) -> None:
        """Print a formatted performance report to stdout."""
        sep  = "═" * 60
        sep2 = "─" * 60

        # Grade the strategy
        grade, grade_desc = self._grade()

        print(f"\n{sep}")
        print(f"  BACKTEST RESULTS — {self.symbol} {self.timeframe}   [{grade}]")
        print(f"  Period: {self.start_date}  →  {self.end_date}  ({self.trading_days} days)")
        print(f"  Strategy Grade: {grade_desc}")
        print(f"{sep}")
        print(f"  {'Bars scanned:':<28} {self.total_bars_scanned:>10,}")
        print(f"  {'Signals generated:':<28} {self.total_signals_generated:>10,}")
        print(f"  {'Trades resolved:':<28} {self.total_trades:>10,}")
        print(f"    {'Winners (TP1/TP2):':<26} {self.winners:>10,}")
        print(f"    {'Losers (SL):':<26} {self.losers:>10,}")
        print(f"    {'Break-evens:':<26} {self.break_evens:>10,}")
        print(f"    {'Expired (unresolved):':<26} {self.expired:>10,}")
        print(f"{sep2}")
        print(f"  {'PERFORMANCE METRICS':}")
        print(f"{sep2}")
        pf_ok   = "✅" if self.profit_factor >= 1.5 else ("⚠️" if self.profit_factor >= 1.0 else "❌")
        wr_ok   = "✅" if self.win_rate >= 0.40     else ("⚠️" if self.win_rate >= 0.30    else "❌")
        exp_ok  = "✅" if self.expectancy >= 0.20   else ("⚠️" if self.expectancy >= 0.0   else "❌")
        sh_ok   = "✅" if self.sharpe_ratio >= 1.0  else ("⚠️" if self.sharpe_ratio >= 0.5  else "❌")
        dd_ok   = "✅" if abs(self.max_drawdown_r) <= 5 else ("⚠️" if abs(self.max_drawdown_r) <= 10 else "❌")
        print(f"  {wr_ok}  {'Win Rate:':<24} {self.win_rate:>8.1%}")
        print(f"  {pf_ok}  {'Profit Factor:':<24} {self.profit_factor:>8.2f}")
        print(f"  {exp_ok}  {'Expectancy:':<24} {self.expectancy:>+8.3f} R/trade")
        print(f"  {sh_ok}  {'Sharpe Ratio:':<24} {self.sharpe_ratio:>8.2f}")
        print(f"  {dd_ok}  {'Max Drawdown:':<24} {self.max_drawdown_r:>+8.2f} R  ({self.max_drawdown_pct:.1f}%)")
        print(f"     {'Calmar Ratio:':<24} {self.calmar_ratio:>8.2f}")
        print(f"     {'Total R:':<24} {self.total_r:>+8.2f} R")
        print(f"     {'Signals / day:':<24} {self.signals_per_day:>8.2f}")
        print(f"{sep2}")
        print(f"  {'R-MULTIPLES':}")
        print(f"{sep2}")
        print(f"  {'Avg Winner:':<28} {self.avg_winner_r:>+10.3f} R")
        print(f"  {'Avg Loser:':<28} {self.avg_loser_r:>+10.3f} R")
        print(f"  {'Largest Winner:':<28} {self.largest_winner_r:>+10.3f} R")
        print(f"  {'Largest Loser:':<28} {self.largest_loser_r:>+10.3f} R")
        print(f"  {'Max Consec. Wins:':<28} {self.max_consecutive_wins:>10,}")
        print(f"  {'Max Consec. Losses:':<28} {self.max_consecutive_losses:>10,}")

        # Monthly breakdown (if available)
        if self.monthly_breakdown:
            print(f"{sep2}")
            print(f"  {'MONTHLY BREAKDOWN':}")
            print(f"{sep2}")
            print(f"  {'Month':<12} {'Trades':>7} {'WR':>8} {'Total R':>9}")
            print(f"  {'':─<40}")
            for row in self.monthly_breakdown:
                wr_str = f"{row['win_rate']:.0%}"
                print(
                    f"  {row['month']:<12} "
                    f"{row['trades']:>7,} "
                    f"{wr_str:>8} "
                    f"{row['total_r']:>+9.2f} R"
                )
        print(f"{sep}")

    def to_dict(self) -> dict:
        return {
            "start_date":             self.start_date,
            "end_date":               self.end_date,
            "symbol":                 self.symbol,
            "timeframe":              self.timeframe,
            "total_bars_scanned":     self.total_bars_scanned,
            "total_signals_generated":self.total_signals_generated,
            "total_trades":           self.total_trades,
            "winners":                self.winners,
            "losers":                 self.losers,
            "break_evens":            self.break_evens,
            "expired":                self.expired,
            "win_rate":               round(self.win_rate, 4),
            "profit_factor":          round(self.profit_factor, 4),
            "expectancy":             round(self.expectancy, 4),
            "total_r":                round(self.total_r, 4),
            "avg_winner_r":           round(self.avg_winner_r, 4),
            "avg_loser_r":            round(self.avg_loser_r, 4),
            "largest_winner_r":       round(self.largest_winner_r, 4),
            "largest_loser_r":        round(self.largest_loser_r, 4),
            "max_drawdown_r":         round(self.max_drawdown_r, 4),
            "max_drawdown_pct":       round(self.max_drawdown_pct, 2),
            "max_consecutive_wins":   self.max_consecutive_wins,
            "max_consecutive_losses": self.max_consecutive_losses,
            "sharpe_ratio":           round(self.sharpe_ratio, 4),
            "calmar_ratio":           round(self.calmar_ratio, 4),
            "signals_per_day":        round(self.signals_per_day, 2),
            "trading_days":           self.trading_days,
            "monthly_breakdown":      self.monthly_breakdown,
        }

    # =========================================================================
    # GRADING
    # =========================================================================

    def _grade(self) -> tuple[str, str]:
        """
        Assign a letter grade to the strategy based on combined metrics.

        Returns (grade_letter: str, grade_description: str)
        """
        score = 0
        # Win Rate
        if self.win_rate >= 0.55: score += 3
        elif self.win_rate >= 0.45: score += 2
        elif self.win_rate >= 0.35: score += 1
        # Profit Factor
        if self.profit_factor >= 2.0: score += 3
        elif self.profit_factor >= 1.5: score += 2
        elif self.profit_factor >= 1.0: score += 1
        # Expectancy
        if self.expectancy >= 0.40: score += 3
        elif self.expectancy >= 0.20: score += 2
        elif self.expectancy >= 0.05: score += 1
        # Sharpe
        if self.sharpe_ratio >= 1.5: score += 2
        elif self.sharpe_ratio >= 0.8: score += 1
        # Drawdown
        if abs(self.max_drawdown_r) <= 3: score += 2
        elif abs(self.max_drawdown_r) <= 6: score += 1

        grade_map = [
            (12, "A+", "Exceptional — production ready"),
            (10, "A",  "Excellent strategy performance"),
            (8,  "B+", "Strong — minor tuning beneficial"),
            (6,  "B",  "Good — consider parameter improvements"),
            (4,  "C",  "Marginal — significant tuning required"),
            (0,  "D",  "Poor — not recommended for live trading"),
        ]
        for threshold, letter, desc in grade_map:
            if score >= threshold:
                return letter, desc
        return "D", "Poor — not recommended for live trading"

    # =========================================================================
    # EQUITY CURVE CHART
    # =========================================================================

    def save_equity_chart(
        self,
        path: str | Path,
        title: str = "",
        dpi:   int = 150,
    ) -> Optional[Path]:
        """
        Render and save an equity curve chart as a PNG image.

        The chart contains three panels:
          1. Equity curve (cumulative R) with drawdown shading
          2. Per-trade R-multiple bar chart (green winners, red losers)
          3. Rolling win-rate (20-trade window)

        Parameters
        ----------
        path  : Output file path (.png recommended)
        title : Optional chart title supplement
        dpi   : Image resolution (default 150 — good for reports)

        Returns
        -------
        Path object of saved file, or None if matplotlib unavailable
        """
        if not _MPL_AVAILABLE:
            logger.warning("save_equity_chart: matplotlib not available — skipping")
            return None

        if not self.equity_curve:
            logger.warning("save_equity_chart: no equity data — skipping")
            return None

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)

        # ── Build data arrays ─────────────────────────────────────────────────
        equity = np.array([0.0] + self.equity_curve)          # starts at 0 R
        trades = list(range(1, len(self.equity_curve) + 1))
        r_vals = np.array([t.r_gained for t in self.trade_log
                           if t.outcome != "EXPIRED"])

        # Drawdown array
        peak   = np.maximum.accumulate(equity)
        dd     = equity - peak

        # Rolling win rate (20-trade window)
        wins   = np.array([1.0 if t.is_winner else 0.0
                           for t in self.trade_log if t.outcome != "EXPIRED"])
        window = 20
        roll_wr = np.convolve(wins, np.ones(window) / window, mode="full")[:len(wins)]
        roll_wr[:window - 1] = np.nan   # mask warmup

        # ── Styling ───────────────────────────────────────────────────────────
        bg_dark   = "#0f1117"
        bg_panel  = "#1a1d27"
        green_col = "#00d68f"
        red_col   = "#ff4757"
        gold_col  = "#ffd700"
        grey_col  = "#8b9ab0"
        line_col  = "#4a90d9"

        fig, axes = plt.subplots(
            3, 1,
            figsize=(14, 10),
            gridspec_kw={"height_ratios": [3, 2, 1.5]},
            facecolor=bg_dark,
        )
        fig.suptitle(
            f"AntiGravity Gold Bot — Backtest Report\n"
            f"{self.symbol} {self.timeframe} | "
            f"{self.start_date} → {self.end_date}"
            + (f" | {title}" if title else ""),
            color=gold_col, fontsize=13, fontweight="bold", y=0.98,
        )

        # ── Panel 1: Equity curve ──────────────────────────────────────────────
        ax1 = axes[0]
        ax1.set_facecolor(bg_panel)
        ax1.plot(range(len(equity)), equity, color=line_col, linewidth=1.8,
                 label="Equity (R)")
        ax1.fill_between(range(len(equity)), equity, peak,
                         where=(dd < 0), color=red_col, alpha=0.20,
                         label="Drawdown")
        ax1.axhline(0, color=grey_col, linewidth=0.6, linestyle="--")

        # Annotate peak and trough
        if len(equity) > 1:
            peak_idx  = int(np.argmax(equity))
            trough_idx = int(np.argmin(equity))
            ax1.annotate(
                f"Peak\n{equity[peak_idx]:+.2f}R",
                xy=(peak_idx, equity[peak_idx]),
                xytext=(peak_idx, equity[peak_idx] + max(0.5, abs(equity).max() * 0.08)),
                color=green_col, fontsize=8, ha="center",
                arrowprops=dict(arrowstyle="->", color=green_col, lw=0.8),
            )
            ax1.annotate(
                f"Trough\n{equity[trough_idx]:+.2f}R",
                xy=(trough_idx, equity[trough_idx]),
                xytext=(trough_idx, equity[trough_idx] - max(0.5, abs(equity).max() * 0.08)),
                color=red_col, fontsize=8, ha="center",
                arrowprops=dict(arrowstyle="->", color=red_col, lw=0.8),
            )

        ax1.set_ylabel("Cumulative R", color=grey_col, fontsize=10)
        ax1.tick_params(colors=grey_col)
        ax1.spines[:].set_color("#2d3346")
        ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%+.1f"))
        ax1.legend(loc="upper left", facecolor=bg_panel, labelcolor=grey_col,
                   fontsize=8, framealpha=0.8)

        # Add key stats text box
        stats_text = (
            f"WR: {self.win_rate:.1%}  │  PF: {self.profit_factor:.2f}  │  "
            f"E: {self.expectancy:+.3f}R  │  "
            f"Sharpe: {self.sharpe_ratio:.2f}  │  "
            f"Max DD: {self.max_drawdown_r:.2f}R ({self.max_drawdown_pct:.1f}%)  │  "
            f"Total: {self.total_r:+.2f}R"
        )
        ax1.text(
            0.01, 0.02, stats_text,
            transform=ax1.transAxes, fontsize=8,
            color=grey_col, verticalalignment="bottom",
            bbox=dict(facecolor=bg_dark, alpha=0.7, edgecolor="none"),
        )

        # ── Panel 2: Per-trade R bars ──────────────────────────────────────────
        ax2 = axes[1]
        ax2.set_facecolor(bg_panel)
        if len(r_vals) > 0:
            colours = [green_col if r >= 0 else red_col for r in r_vals]
            ax2.bar(trades, r_vals, color=colours, width=0.7, alpha=0.85)
            ax2.axhline(0, color=grey_col, linewidth=0.6)
            # Expectancy line
            ax2.axhline(
                self.expectancy, color=gold_col, linewidth=1.0,
                linestyle="--", label=f"Expectancy {self.expectancy:+.3f}R",
            )
            ax2.legend(loc="upper right", facecolor=bg_panel, labelcolor=grey_col,
                       fontsize=8, framealpha=0.8)
        ax2.set_ylabel("R per Trade", color=grey_col, fontsize=10)
        ax2.tick_params(colors=grey_col)
        ax2.spines[:].set_color("#2d3346")
        ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%+.1f"))

        # ── Panel 3: Rolling win rate ──────────────────────────────────────────
        ax3 = axes[2]
        ax3.set_facecolor(bg_panel)
        if len(roll_wr) > window:
            x3 = np.arange(1, len(roll_wr) + 1)
            ax3.plot(x3, roll_wr * 100, color=green_col, linewidth=1.4,
                     label=f"Rolling {window}-trade WR")
            ax3.axhline(self.win_rate * 100, color=gold_col, linewidth=0.8,
                        linestyle="--", label=f"Overall WR {self.win_rate:.1%}")
            ax3.axhline(40, color=red_col, linewidth=0.5, linestyle=":")
            ax3.set_ylim(0, 100)
            ax3.yaxis.set_major_formatter(mticker.PercentFormatter())
            ax3.legend(loc="upper right", facecolor=bg_panel, labelcolor=grey_col,
                       fontsize=8, framealpha=0.8)
        ax3.set_xlabel("Trade #", color=grey_col, fontsize=10)
        ax3.set_ylabel("Win Rate", color=grey_col, fontsize=10)
        ax3.tick_params(colors=grey_col)
        ax3.spines[:].set_color("#2d3346")

        # ── Finish ────────────────────────────────────────────────────────────
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(out, dpi=dpi, facecolor=bg_dark, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Equity chart saved: {out}")
        return out

    # =========================================================================
    # HTML REPORT
    # =========================================================================

    def generate_report(
        self,
        output_dir: str | Path = ".",
        chart_filename: str = "equity_curve.png",
    ) -> dict[str, Path]:
        """
        Generate a full backtest report package to `output_dir`:

          trade_log.csv         — one row per trade
          backtest_summary.txt  — plain-text console report
          backtest_report.json  — machine-readable JSON metrics
          equity_curve.png      — equity curve chart (3-panel)
          backtest_report.html  — rich HTML report with embedded chart

        Returns
        -------
        dict of {artifact_name: Path} for all files written
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        written: dict[str, Path] = {}

        # ── 1. Trade log CSV ─────────────────────────────────────────────────
        csv_path = out / "trade_log.csv"
        rows = [t.to_dict() for t in self.trade_log]
        keys = list(rows[0].keys()) if rows else [
            "trade_id", "signal_time", "outcome_time", "direction",
            "entry", "stop_loss", "take_profit_1", "take_profit_2",
            "rr_tp1", "rr_tp2", "outcome", "bars_held", "r_gained",
            "confidence", "has_ob", "has_fvg", "has_bos", "has_choch",
            "has_liq_sweep", "session", "regime", "signal_bar",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        written["trade_log"] = csv_path
        logger.info(f"Trade log: {csv_path} ({len(rows)} rows)")

        # ── 2. Plain-text summary ────────────────────────────────────────────
        txt_path = out / "backtest_summary.txt"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.print_summary()
        txt_path.write_text(buf.getvalue(), encoding="utf-8")
        written["summary_txt"] = txt_path
        logger.info(f"Text summary: {txt_path}")

        # ── 3. JSON metrics ──────────────────────────────────────────────────
        json_path = out / "backtest_report.json"
        json_path.write_text(
            json.dumps(self.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )
        written["json_report"] = json_path

        # ── 4. Equity curve chart ────────────────────────────────────────────
        chart_path = out / chart_filename
        chart_saved = self.save_equity_chart(chart_path)
        if chart_saved:
            written["chart"] = chart_saved

        # ── 5. HTML report ───────────────────────────────────────────────────
        html_path = out / "backtest_report.html"
        html = self._build_html_report(
            chart_filename = chart_filename if chart_saved else None
        )
        html_path.write_text(html, encoding="utf-8")
        written["html_report"] = html_path
        logger.info(f"HTML report: {html_path}")

        print(f"\n📊 Backtest report saved to: {out.resolve()}")
        for name, p in written.items():
            print(f"   {name:<18} {p.name}")

        return written

    def _build_html_report(self, chart_filename: Optional[str] = None) -> str:
        """Build a rich HTML report string."""
        grade, grade_desc = self._grade()

        grade_colour = {
            "A+": "#00d68f", "A": "#00d68f",
            "B+": "#ffd700", "B": "#ffd700",
            "C":  "#ff9f43", "D": "#ff4757",
        }.get(grade, "#8b9ab0")

        def pct(v):  return f"{v:.1%}"
        def r2(v):   return f"{v:+.2f}"
        def r3(v):   return f"{v:+.3f}"
        def f2(v):   return f"{v:.2f}"

        def badge(val, good_thr, warn_thr, higher_is_good=True):
            """Return green/amber/red badge."""
            ok = (val >= good_thr) if higher_is_good else (val <= good_thr)
            warn = (val >= warn_thr) if higher_is_good else (val <= warn_thr)
            color = "#00d68f" if ok else ("#ffd700" if warn else "#ff4757")
            return f'<span style="color:{color};font-weight:bold">{val:.2f}</span>'

        chart_html = (
            f'<img src="{chart_filename}" alt="Equity Curve" '
            f'style="width:100%;border-radius:8px;margin-top:16px;"/>'
            if chart_filename else
            '<p style="color:#8b9ab0">Chart not available (matplotlib not installed)</p>'
        )

        monthly_rows = ""
        for row in self.monthly_breakdown:
            wr   = row["win_rate"]
            tr   = row["total_r"]
            tr_c = "#00d68f" if tr >= 0 else "#ff4757"
            wr_c = "#00d68f" if wr >= 0.45 else ("#ffd700" if wr >= 0.35 else "#ff4757")
            monthly_rows += (
                f"<tr>"
                f"<td>{row['month']}</td>"
                f"<td>{row['trades']}</td>"
                f"<td style='color:{wr_c}'>{wr:.0%}</td>"
                f"<td style='color:{tr_c}'>{tr:+.2f} R</td>"
                f"</tr>"
            )

        monthly_section = (
            f"""
            <h2>Monthly Breakdown</h2>
            <table>
              <thead><tr>
                <th>Month</th><th>Trades</th>
                <th>Win Rate</th><th>Total R</th>
              </tr></thead>
              <tbody>{monthly_rows}</tbody>
            </table>
            """
            if monthly_rows else ""
        )

        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Backtest Report — {self.symbol} {self.timeframe}</title>
  <style>
    :root {{
      --bg:       #0f1117;
      --surface:  #1a1d27;
      --border:   #2d3346;
      --text:     #e0e6f0;
      --muted:    #8b9ab0;
      --gold:     #ffd700;
      --green:    #00d68f;
      --red:      #ff4757;
      --blue:     #4a90d9;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Segoe UI', 'Inter', sans-serif;
      background: var(--bg); color: var(--text);
      padding: 24px; max-width: 960px; margin: 0 auto;
    }}
    h1 {{ color: var(--gold); font-size: 1.5rem; margin-bottom: 4px; }}
    h2 {{ color: var(--blue); font-size: 1.1rem; margin: 24px 0 10px; }}
    .subtitle {{ color: var(--muted); font-size: 0.9rem; margin-bottom: 20px; }}
    .grade-badge {{
      display: inline-block;
      padding: 6px 20px;
      border-radius: 6px;
      background: {grade_colour}22;
      border: 1px solid {grade_colour};
      color: {grade_colour};
      font-size: 1.6rem;
      font-weight: bold;
      float: right;
      margin-top: -8px;
    }}
    .metrics-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
      gap: 12px;
      margin-bottom: 20px;
    }}
    .metric-card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 14px 18px;
    }}
    .metric-label {{ color: var(--muted); font-size: 0.78rem; margin-bottom: 4px; }}
    .metric-value {{ font-size: 1.4rem; font-weight: 700; }}
    .metric-sub {{ color: var(--muted); font-size: 0.78rem; margin-top: 2px; }}
    .green {{ color: var(--green); }}
    .red   {{ color: var(--red);   }}
    .gold  {{ color: var(--gold);  }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--surface);
      border-radius: 8px;
      overflow: hidden;
      font-size: 0.88rem;
    }}
    th, td {{
      padding: 9px 14px;
      border-bottom: 1px solid var(--border);
      text-align: right;
    }}
    th {{ background: #232738; color: var(--muted); font-weight: 600; }}
    tr:last-child td {{ border-bottom: none; }}
    td:first-child, th:first-child {{ text-align: left; }}
    .footer {{ color: var(--muted); font-size: 0.75rem; margin-top: 32px; }}
  </style>
</head>
<body>
  <h1>📊 Backtest Report
    <span class="grade-badge">{grade}</span>
  </h1>
  <div class="subtitle">
    {self.symbol} {self.timeframe} &nbsp;|&nbsp;
    {self.start_date} → {self.end_date} &nbsp;|&nbsp;
    {self.trading_days} trading days &nbsp;|&nbsp;
    {grade_desc}
  </div>

  {chart_html}

  <h2>Performance Metrics</h2>
  <div class="metrics-grid">
    <div class="metric-card">
      <div class="metric-label">Win Rate</div>
      <div class="metric-value {'green' if self.win_rate >= 0.45 else ('gold' if self.win_rate >= 0.35 else 'red')}">{pct(self.win_rate)}</div>
      <div class="metric-sub">{self.winners}W / {self.losers}L / {self.break_evens}BE</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Profit Factor</div>
      <div class="metric-value {'green' if self.profit_factor >= 1.5 else ('gold' if self.profit_factor >= 1.0 else 'red')}">{f2(self.profit_factor)}</div>
      <div class="metric-sub">Gross profit / gross loss</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Expectancy</div>
      <div class="metric-value {'green' if self.expectancy >= 0.2 else ('gold' if self.expectancy >= 0 else 'red')}">{r3(self.expectancy)} R</div>
      <div class="metric-sub">Average R per trade</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Total R</div>
      <div class="metric-value {'green' if self.total_r >= 0 else 'red'}">{r2(self.total_r)} R</div>
      <div class="metric-sub">{self.total_trades} resolved trades</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Sharpe Ratio</div>
      <div class="metric-value {'green' if self.sharpe_ratio >= 1.0 else ('gold' if self.sharpe_ratio >= 0.5 else 'red')}">{f2(self.sharpe_ratio)}</div>
      <div class="metric-sub">Annualised (daily R basis)</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Max Drawdown</div>
      <div class="metric-value {'green' if abs(self.max_drawdown_r) <= 3 else ('gold' if abs(self.max_drawdown_r) <= 6 else 'red')}">{r2(self.max_drawdown_r)} R</div>
      <div class="metric-sub">{self.max_drawdown_pct:.1f}% of peak equity</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Calmar Ratio</div>
      <div class="metric-value gold">{f2(self.calmar_ratio)}</div>
      <div class="metric-sub">Net R / |Max Drawdown|</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Signals / Day</div>
      <div class="metric-value">{self.signals_per_day:.2f}</div>
      <div class="metric-sub">Avg signals per trading day</div>
    </div>
  </div>

  <h2>R-Multiple Statistics</h2>
  <table>
    <thead><tr>
      <th>Metric</th><th>Value</th>
    </tr></thead>
    <tbody>
      <tr><td>Average Winner</td><td class="green">{r3(self.avg_winner_r)} R</td></tr>
      <tr><td>Average Loser</td><td class="red">{r3(self.avg_loser_r)} R</td></tr>
      <tr><td>Largest Winner</td><td class="green">{r3(self.largest_winner_r)} R</td></tr>
      <tr><td>Largest Loser</td><td class="red">{r3(self.largest_loser_r)} R</td></tr>
      <tr><td>Max Consecutive Wins</td><td>{self.max_consecutive_wins}</td></tr>
      <tr><td>Max Consecutive Losses</td><td>{self.max_consecutive_losses}</td></tr>
    </tbody>
  </table>

  {monthly_section}

  <h2>Run Details</h2>
  <table>
    <tbody>
      <tr><td>Symbol</td><td>{self.symbol}</td></tr>
      <tr><td>Timeframe</td><td>{self.timeframe}</td></tr>
      <tr><td>Period</td><td>{self.start_date} → {self.end_date}</td></tr>
      <tr><td>Bars Scanned</td><td>{self.total_bars_scanned:,}</td></tr>
      <tr><td>Signals Generated</td><td>{self.total_signals_generated:,}</td></tr>
      <tr><td>Trades Resolved</td><td>{self.total_trades:,}</td></tr>
      <tr><td>Expired (unresolved)</td><td>{self.expired}</td></tr>
    </tbody>
  </table>

  <div class="footer">
    Generated by AntiGravity Gold Trading Bot · {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}
  </div>
</body>
</html>
"""


# ─────────────────────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────────────────────

class BacktestEngine:
    """
    Event-driven bar-by-bar backtesting engine for the AntiGravity SMC strategy.

    Usage
    -----
        engine = BacktestEngine(settings)
        result = engine.run(df_h1, df_m15)
        result.print_summary()

    The engine feeds bars to SignalGenerator one at a time, accumulating a
    growing window of price history.  A signal is only fired when the window
    has at least MIN_LOOKBACK_BARS bars (prevents look-ahead).
    """

    def __init__(
        self,
        settings:      Settings,
        max_bars_open: int   = MAX_BARS_OPEN,
        tp1_fraction:  float = TP1_CLOSE_FRACTION,
        exclude_expired: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        settings       : App configuration
        max_bars_open  : Bars before open trade is marked EXPIRED
        tp1_fraction   : Fraction of position closed at TP1 (default 0.5)
        exclude_expired: Exclude EXPIRED trades from win/loss stats (default True)
        """
        self.cfg            = settings
        self.max_bars_open  = max_bars_open
        self.tp1_fraction   = tp1_fraction
        self.exclude_expired = exclude_expired

        # Build sub-components (same as live bot)
        self.session_f    = SessionFilter(settings)
        self.vol_f        = VolatilityFilter(settings)
        self.signal_gen   = SignalGenerator(
            settings,
            session_filter    = self.session_f,
            volatility_filter = self.vol_f,
        )

    # =========================================================================
    # PRIMARY PUBLIC API
    # =========================================================================

    def run(
        self,
        df_h1:      pd.DataFrame,
        df_m15:     pd.DataFrame,
        start:      str | None = None,
        end:        str | None = None,
        account_balance: float | None = None,
    ) -> BacktestResult:
        """
        Run a full backtest on pre-loaded DataFrames.

        Parameters
        ----------
        df_h1       : H1 OHLCV (trend / regime timeframe) — must have 'time' col
        df_m15      : M15 OHLCV (signal / SMC scan timeframe) — must have 'time' col
        start       : "YYYY-MM-DD" — first bar date (inclusive).  None = all data.
        end         : "YYYY-MM-DD" — last bar date (inclusive).   None = all data.
        account_balance : Starting balance for lot sizing (defaults to Settings value)

        Returns
        -------
        BacktestResult with full trade log and statistics
        """
        balance = account_balance or self.cfg.ACCOUNT_BALANCE

        # ── Slice date range ──────────────────────────────────────────────────
        df_h1_  = self._slice(df_h1,  start, end)
        df_m15_ = self._slice(df_m15, start, end)

        if len(df_m15_) < MIN_LOOKBACK_BARS:
            logger.error(
                f"run(): only {len(df_m15_)} M15 bars after date filter "
                f"(need {MIN_LOOKBACK_BARS}). Returning empty result."
            )
            return BacktestResult(start_date=start or "", end_date=end or "")

        logger.info(
            f"Backtest: {len(df_m15_)} M15 bars | "
            f"{len(df_h1_)} H1 bars | "
            f"range {df_m15_['time'].iloc[0]} → {df_m15_['time'].iloc[-1]}"
        )

        # ── Bar-by-bar simulation ─────────────────────────────────────────────
        trade_log: list[TradeRecord] = []
        trade_id  = 0
        total_sig = 0

        # Track the 'active signal' window (same logic as live bot)
        active_end_bar: int = -1   # bar index after which we allow new signals

        for i in range(MIN_LOOKBACK_BARS, len(df_m15_)):
            # Slice history up to (but not including) current bar — no look-ahead
            hist_m15 = df_m15_.iloc[:i].copy()
            bar_time = df_m15_["time"].iloc[i]

            # Align H1 history: only bars with time <= current M15 bar time
            hist_h1 = df_h1_[df_h1_["time"] <= bar_time].copy()
            if len(hist_h1) < MIN_LOOKBACK_BARS:
                continue

            # Skip if inside the hold window of a prior signal
            if i <= active_end_bar:
                continue

            now_dt = bar_time.to_pydatetime() if hasattr(bar_time, "to_pydatetime") else bar_time
            if now_dt.tzinfo is None:
                now_dt = now_dt.replace(tzinfo=timezone.utc)

            # ── Run signal generator ──────────────────────────────────────────
            try:
                signals, rejected = self.signal_gen.generate(
                    df_h1           = hist_h1,
                    df_m15          = hist_m15,
                    account_balance = balance,
                    now             = now_dt,
                )
            except Exception as exc:
                logger.warning(f"Bar {i}: signal_gen.generate() error: {exc}")
                continue

            total_sig += len(signals) + len(rejected)

            if not signals:
                continue

            # Take the highest-confidence signal only (mirrors live bot)
            best = max(signals, key=lambda s: s.confidence)

            # ── Simulate outcome ──────────────────────────────────────────────
            trade_id += 1
            record = self._make_record(trade_id, i, best, now_dt)
            outcome = self._simulate_outcome(record, df_m15_, i)
            trade_log.append(outcome)

            # Lock the engine for the duration of the trade to avoid overlapping
            active_end_bar = outcome.outcome_bar if outcome.outcome_bar > 0 else i + self.max_bars_open

            logger.debug(
                f"Bar {i} | {now_dt.strftime('%Y-%m-%d %H:%M')} | "
                f"{outcome.direction} @ {outcome.entry:.2f} → "
                f"{outcome.outcome}  {outcome.r_gained:+.2f}R"
            )

        # ── Aggregate statistics ──────────────────────────────────────────────
        result = self._compute_statistics(
            trade_log  = trade_log,
            total_sigs = total_sig,
            df_m15     = df_m15_,
            start      = start or str(df_m15_["time"].iloc[0])[:10],
            end        = end   or str(df_m15_["time"].iloc[-1])[:10],
        )
        logger.info(
            f"Backtest complete: {result.total_trades} trades | "
            f"WR={result.win_rate:.0%} | PF={result.profit_factor:.2f} | "
            f"E={result.expectancy:+.3f}R | DD={result.max_drawdown_r:.2f}R"
        )
        return result

    def run_from_csv(
        self,
        m15_csv:    str | Path,
        h1_csv:     str | Path,
        start:      str | None = None,
        end:        str | None = None,
        account_balance: float | None = None,
    ) -> BacktestResult:
        """
        Load OHLCV data from CSV files and run the backtest.

        CSV format expected:
            time, open, high, low, close, tick_volume
            (time must be parseable by pd.to_datetime)
        """
        df_m15 = self.load_csv(m15_csv)
        df_h1  = self.load_csv(h1_csv)
        return self.run(df_h1, df_m15, start=start, end=end,
                        account_balance=account_balance)

    def walk_forward(
        self,
        df_h1:      pd.DataFrame,
        df_m15:     pd.DataFrame,
        n_folds:    int   = 4,
        account_balance: float | None = None,
    ) -> list[BacktestResult]:
        """
        Split the dataset into n_folds equal time segments and backtest each.

        Useful for checking strategy robustness across different market periods.

        Returns a list of BacktestResult — one per fold.
        """
        total_bars = len(df_m15)
        fold_size  = total_bars // n_folds
        results    = []

        for fold in range(n_folds):
            i_start = fold * fold_size
            i_end   = (fold + 1) * fold_size if fold < n_folds - 1 else total_bars

            start = str(df_m15["time"].iloc[i_start])[:10]
            end   = str(df_m15["time"].iloc[i_end - 1])[:10]

            logger.info(f"Walk-forward fold {fold+1}/{n_folds}: {start} → {end}")

            fold_result = self.run(
                df_h1, df_m15, start=start, end=end,
                account_balance=account_balance,
            )
            results.append(fold_result)
            fold_result.print_summary()

        return results

    # =========================================================================
    # OUTCOME SIMULATION
    # =========================================================================

    def _simulate_outcome(
        self,
        record: TradeRecord,
        df:     pd.DataFrame,
        entry_bar: int,
    ) -> TradeRecord:
        """
        Scan forward bars to determine whether the trade hits SL, TP1, or TP2.

        Partial-TP logic:
            On TP1 hit → close `tp1_fraction` of position at TP1 R:R
                       → remaining position runs to TP2 (or stops at break-even)

        P&L in R-multiples:
            SL hit         → r_gained = -1.0
            TP1 hit only   → r_gained = +rr_tp1 × tp1_fraction
            TP2 hit        → r_gained = tp1_fraction × rr_tp1 + (1-tp1_fraction) × rr_tp2
            Break-even     → r_gained = +tp1_fraction × rr_tp1  (remainder closed at 0)
        """
        direction  = record.direction
        entry      = record.entry
        sl         = record.stop_loss
        tp1        = record.take_profit_1
        tp2        = record.take_profit_2

        rr1 = record.rr_tp1
        rr2 = record.rr_tp2

        n = len(df)
        tp1_hit = False
        be_sl   = entry   # break-even stop (moves to entry after TP1 hit)

        end_bar = min(entry_bar + self.max_bars_open, n)

        for j in range(entry_bar + 1, end_bar):
            bar_high = float(df["high"].iloc[j])
            bar_low  = float(df["low"].iloc[j])
            bar_time = df["time"].iloc[j]

            if direction == "BUY":
                # Stop loss hit (check first — worst case assumption)
                if not tp1_hit and bar_low <= sl:
                    return self._close(record, "SL", j, bar_time, -1.0)

                # Break-even stop after TP1
                if tp1_hit and bar_low <= be_sl:
                    r = self.tp1_fraction * rr1  # only TP1 portion profitable
                    return self._close(record, "BE", j, bar_time, r)

                # TP1 hit
                if not tp1_hit and bar_high >= tp1:
                    tp1_hit = True
                    be_sl   = entry   # slide SL to break-even
                    if tp2 is None:
                        # No TP2 — close full position at TP1
                        return self._close(record, "TP1", j, bar_time, rr1)

                # TP2 hit (only checked after TP1)
                if tp1_hit and tp2 is not None and bar_high >= tp2:
                    r = self.tp1_fraction * rr1 + (1 - self.tp1_fraction) * rr2
                    return self._close(record, "TP2", j, bar_time, r)

            else:  # SELL
                if not tp1_hit and bar_high >= sl:
                    return self._close(record, "SL", j, bar_time, -1.0)

                if tp1_hit and bar_high >= be_sl:
                    r = self.tp1_fraction * rr1
                    return self._close(record, "BE", j, bar_time, r)

                if not tp1_hit and bar_low <= tp1:
                    tp1_hit = True
                    be_sl   = entry
                    if tp2 is None:
                        return self._close(record, "TP1", j, bar_time, rr1)

                if tp1_hit and tp2 is not None and bar_low <= tp2:
                    r = self.tp1_fraction * rr1 + (1 - self.tp1_fraction) * rr2
                    return self._close(record, "TP2", j, bar_time, r)

        # No resolution within max_bars_open → close at last bar at entry price
        last_bar  = min(entry_bar + self.max_bars_open, n - 1)
        last_time = df["time"].iloc[last_bar]
        return self._close(record, "EXPIRED", last_bar, last_time, 0.0)

    @staticmethod
    def _close(
        record:   TradeRecord,
        outcome:  str,
        bar:      int,
        bar_time: datetime,
        r_gained: float,
    ) -> TradeRecord:
        """Mutate and return the trade record with outcome fields filled."""
        record.outcome     = outcome
        record.outcome_bar = bar
        record.outcome_time = (
            bar_time.to_pydatetime().replace(tzinfo=timezone.utc)
            if hasattr(bar_time, "to_pydatetime")
            else bar_time
        )
        record.bars_held  = bar - record.signal_bar
        record.r_gained   = round(r_gained, 4)
        return record

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def _compute_statistics(
        self,
        trade_log: list[TradeRecord],
        total_sigs: int,
        df_m15:     pd.DataFrame,
        start:      str,
        end:        str,
    ) -> BacktestResult:
        """Aggregate the trade log into a BacktestResult."""

        active_trades = [
            t for t in trade_log
            if not (self.exclude_expired and t.outcome == "EXPIRED")
        ]

        n       = len(active_trades)
        winners = [t for t in active_trades if t.is_winner]
        losers  = [t for t in active_trades if t.is_loser]
        bes     = [t for t in active_trades if t.outcome == "BE"]
        expired = [t for t in trade_log    if t.outcome == "EXPIRED"]

        win_r   = [t.r_gained for t in winners]
        loss_r  = [t.r_gained for t in losers]
        all_r   = [t.r_gained for t in active_trades]

        gross_profit = sum(r for r in all_r if r > 0)
        gross_loss   = abs(sum(r for r in all_r if r < 0))

        win_rate         = len(winners) / n if n > 0 else 0.0
        profit_factor    = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        total_r          = sum(all_r)
        expectancy       = total_r / n if n > 0 else 0.0
        avg_winner_r     = sum(win_r) / len(win_r) if win_r else 0.0
        avg_loser_r      = sum(loss_r) / len(loss_r) if loss_r else 0.0
        largest_winner_r = max(win_r,  default=0.0)
        largest_loser_r  = min(loss_r, default=0.0)

        max_dd_r, max_dd_pct = self._compute_drawdown(all_r)
        max_cw, max_cl       = self._compute_streaks(active_trades)
        sharpe               = self._compute_sharpe(all_r, df_m15)
        calmar               = total_r / abs(max_dd_r) if max_dd_r != 0 else 0.0
        equity_curve         = list(np.cumsum(all_r))
        monthly_bkdn         = self._compute_monthly_breakdown(active_trades)

        # Trading day count
        if len(df_m15) > 0:
            t0   = df_m15["time"].iloc[0]
            t1   = df_m15["time"].iloc[-1]
            days = max(1, (t1 - t0).days) if hasattr((t1 - t0), "days") else 1
        else:
            days = 1

        signals_per_day = n / days if days > 0 else 0.0

        return BacktestResult(
            start_date               = start,
            end_date                 = end,
            symbol                   = self.cfg.SYMBOL,
            timeframe                = self.cfg.SIGNAL_TIMEFRAME,
            total_bars               = len(df_m15),
            total_bars_scanned       = len(df_m15) - MIN_LOOKBACK_BARS,
            total_signals_generated  = total_sigs,
            total_trades             = n,
            winners                  = len(winners),
            losers                   = len(losers),
            break_evens              = len(bes),
            expired                  = len(expired),
            win_rate                 = round(win_rate,       4),
            profit_factor            = round(profit_factor,  4),
            expectancy               = round(expectancy,     4),
            total_r                  = round(total_r,        4),
            avg_winner_r             = round(avg_winner_r,   4),
            avg_loser_r              = round(avg_loser_r,    4),
            largest_winner_r         = round(largest_winner_r, 4),
            largest_loser_r          = round(largest_loser_r,  4),
            max_drawdown_r           = round(max_dd_r,  4),
            max_drawdown_pct         = round(max_dd_pct, 2),
            max_consecutive_wins     = max_cw,
            max_consecutive_losses   = max_cl,
            sharpe_ratio             = round(sharpe,  4),
            calmar_ratio             = round(calmar,  4),
            signals_per_day          = round(signals_per_day, 3),
            trading_days             = days,
            trade_log                = trade_log,
            equity_curve             = equity_curve,
            monthly_breakdown        = monthly_bkdn,
        )

    @staticmethod
    def _compute_drawdown(r_series: list[float]) -> tuple[float, float]:
        """
        Compute max drawdown in both R-units and percentage.

        Returns (max_drawdown_r, max_drawdown_pct).
        """
        if not r_series:
            return 0.0, 0.0

        equity  = np.cumsum([0.0] + r_series)   # starts at 0 R
        peak    = np.maximum.accumulate(equity)
        dd      = equity - peak                  # always ≤ 0
        max_dd_r = float(np.min(dd))

        # Percentage drawdown (relative to peak, treating 100R as 100% base)
        peak_nonzero = np.where(peak > 0, peak, np.nan)
        dd_pct       = np.where(peak > 0, dd / peak_nonzero * 100.0, 0.0)
        max_dd_pct   = float(np.nanmin(dd_pct))

        return max_dd_r, max_dd_pct

    @staticmethod
    def _compute_streaks(trades: list[TradeRecord]) -> tuple[int, int]:
        """Return (max_consecutive_wins, max_consecutive_losses)."""
        max_cw = max_cl = cw = cl = 0
        for t in trades:
            if t.is_winner:
                cw += 1; cl = 0
                max_cw = max(max_cw, cw)
            elif t.is_loser:
                cl += 1; cw = 0
                max_cl = max(max_cl, cl)
            else:
                cw = 0; cl = 0
        return max_cw, max_cl

    @staticmethod
    def _compute_monthly_breakdown(trades: list[TradeRecord]) -> list[dict]:
        """
        Group resolved trades by calendar month and compute per-month stats.

        Returns list of dicts: [{month, trades, winners, losers, win_rate, total_r}]
        sorted chronologically.
        """
        from collections import defaultdict
        monthly: dict[str, list] = defaultdict(list)
        for t in trades:
            if t.signal_time is None:
                continue
            key = t.signal_time.strftime("%Y-%m")
            monthly[key].append(t)

        rows = []
        for month_key in sorted(monthly):
            trades_m = monthly[month_key]
            w  = [t for t in trades_m if t.is_winner]
            l  = [t for t in trades_m if t.is_loser]
            nm = len(trades_m)
            rows.append({
                "month":     month_key,
                "trades":    nm,
                "winners":   len(w),
                "losers":    len(l),
                "win_rate":  len(w) / nm if nm > 0 else 0.0,
                "total_r":   round(sum(t.r_gained for t in trades_m), 4),
            })
        return rows

    @staticmethod
    def _compute_sharpe(r_series: list[float], df_m15: pd.DataFrame) -> float:
        """
        Approximate annualised Sharpe ratio using daily R-sums.

        Groups R-multiples by trade date and computes:
            sharpe = mean_daily_R / std_daily_R * sqrt(252)
        """
        if len(r_series) < 4:
            return 0.0
        arr = np.array(r_series)
        mu  = float(np.mean(arr))
        sig = float(np.std(arr))
        if sig == 0:
            return 0.0
        # √252 scaling (252 trading days / year)
        return round(mu / sig * math.sqrt(252), 4)

    # =========================================================================
    # DATA UTILITIES
    # =========================================================================

    @staticmethod
    def load_csv(path: str | Path) -> pd.DataFrame:
        """
        Load OHLCV data from a CSV file.

        Expected columns: time, open, high, low, close, tick_volume
        Time column must be parseable by pd.to_datetime.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"CSV not found: {p}")

        df = pd.read_csv(p, parse_dates=["time"])
        df.columns = [c.strip().lower() for c in df.columns]

        required = {"time", "open", "high", "low", "close"}
        missing  = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")

        if "tick_volume" not in df.columns:
            df["tick_volume"] = 0

        df = df.sort_values("time").reset_index(drop=True)

        # Ensure UTC-aware timestamps
        if df["time"].dt.tz is None:
            df["time"] = df["time"].dt.tz_localize("UTC")
        else:
            df["time"] = df["time"].dt.tz_convert("UTC")

        logger.info(f"Loaded {len(df):,} bars from {p.name}")
        return df

    @staticmethod
    def _slice(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
        """Filter DataFrame to [start, end] date range."""
        if start:
            df = df[df["time"] >= pd.Timestamp(start, tz="UTC")]
        if end:
            df = df[df["time"] <= pd.Timestamp(end + " 23:59:59", tz="UTC")]
        return df.reset_index(drop=True)

    # =========================================================================
    # EXPORT
    # =========================================================================

    def export(self, result: BacktestResult, output_dir: str | Path = ".") -> dict:
        """
        Export full backtest report package to `output_dir`.

        Delegates to BacktestResult.generate_report() which writes:
          trade_log.csv          — one row per trade
          backtest_summary.txt   — plain-text console report
          backtest_report.json   — machine-readable JSON
          equity_curve.png       — equity curve chart (matplotlib)
          backtest_report.html   — rich HTML report

        Returns dict of {artifact_name: Path}.
        """
        return result.generate_report(output_dir=output_dir)

    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================

    @staticmethod
    def _make_record(
        trade_id: int,
        bar_i:    int,
        signal,          # TradingSignal
        bar_time: datetime,
    ) -> TradeRecord:
        """Construct a TradeRecord from a TradingSignal."""
        return TradeRecord(
            trade_id      = trade_id,
            signal_bar    = bar_i,
            signal_time   = bar_time,
            direction     = signal.direction,
            entry         = signal.entry_price,
            stop_loss     = signal.stop_loss,
            take_profit_1 = signal.take_profit_1,
            take_profit_2 = signal.take_profit_2,
            confidence    = signal.confidence,
            has_ob        = signal.has_ob,
            has_fvg       = signal.has_fvg,
            has_bos       = signal.has_bos,
            has_choch     = signal.has_choch,
            has_liq_sweep = signal.has_liquidity_sweep,
            session       = signal.session,
            regime        = signal.regime,
        )
