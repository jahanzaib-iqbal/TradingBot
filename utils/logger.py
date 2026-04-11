"""
utils/logger.py
---------------
Centralised logging configuration for the JayBot Gold Trading Bot.

Responsibilities
────────────────
• `get_logger(name)`          — standard module-level logger factory (INFO+)
• `get_debug_logger(name)`    — verbose DEBUG-level logger writing to debug.log
• `TradingDebugLogger`        — structured helper that emits human-readable,
                                step-by-step decision traces for every signal
                                pipeline stage.

The two-file approach is intentional:
  gold_bot.log  — INFO and above for operations, monitoring, alerts
  debug.log     — DEBUG for every signal decision (why was it accepted/rejected)

This makes production logs clean while keeping full diagnostic detail available.

Usage
-----
    from utils.logger import get_logger, TradingDebugLogger

    logger = get_logger(__name__)         # for module-level INFO logging
    dbg    = TradingDebugLogger()         # once per generate() cycle

    dbg.cycle_start(now)
    dbg.pre_scan_filters(sess_ok, sess_reason, vol_ok, vol_reason,
                         news_ok, news_reason)
    dbg.context(trend_result, regime_result, liq_sweep)
    dbg.strategy_verdicts([trend_verdict, liq_verdict])
    dbg.smc_ideas(raw_ideas)
    dbg.confluence(idea.direction, confluence)
    dbg.gate_pass("trend_alignment")
    dbg.gate_reject("confidence_threshold", "62% < 68%", idea)
    dbg.confidence_breakdown(base, breakdown, final)
    dbg.signal_approved(signal)
    dbg.cycle_summary(signals, rejected)

Dependencies
────────────
  logging, logging.handlers, pathlib, sys
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Formatting constants
# ─────────────────────────────────────────────────────────────────────────────

LOG_FORMAT   = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DEBUG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
DATE_FORMAT  = "%Y-%m-%d %H:%M:%S"

_CONFIGURED_LOGGERS: set[str] = set()


# ─────────────────────────────────────────────────────────────────────────────
# Core factory functions
# ─────────────────────────────────────────────────────────────────────────────

def get_logger(
    name:     str,
    log_file: str = "logs/gold_bot.log",
    level:    str = "WARNING",
) -> logging.Logger:
    """
    Return a configured INFO-level logger.

    Writes to both stdout and a daily-rotating file (gold_bot.log).
    Safe to call multiple times — duplicate handlers are suppressed.

    Args
    ----
    name     : Logger name (use __name__ from the calling module)
    log_file : Path to the rotating log file
    level    : Logging level for this logger ("INFO" | "WARNING" | "ERROR" | "DEBUG")
    """
    logger = logging.getLogger(name)

    if name in _CONFIGURED_LOGGERS or logger.handlers:
        return logger

    _CONFIGURED_LOGGERS.add(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(getattr(logging, level.upper(), logging.INFO))
    ch.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    logger.addHandler(ch)

    # File logging has been disabled by user request.

    return logger


def get_debug_logger(
    name:       str  = "trading.debug",
    debug_file: str  = "logs/debug.log",
) -> logging.Logger:
    """
    Return a DEBUG-level logger that writes to a separate debug.log file.

    The debug log captures every decision step in the signal pipeline without
    cluttering the main gold_bot.log.

    Args
    ----
    name       : Logger name (default "trading.debug")
    debug_file : Path to the debug rotating log file
    """
    logger = logging.getLogger(name)

    if name in _CONFIGURED_LOGGERS or logger.handlers:
        return logger

    _CONFIGURED_LOGGERS.add(name)
    logger.setLevel(logging.DEBUG)

    # File logging has been disabled by user request.
    # We use NullHandler so all debug events are completely ignored and discarded.
    logger.addHandler(logging.NullHandler())

    return logger


# ─────────────────────────────────────────────────────────────────────────────
# TradingDebugLogger — structured decision trace helper
# ─────────────────────────────────────────────────────────────────────────────

class TradingDebugLogger:
    """
    Structured step-by-step decision logger for the signal pipeline.

    Emits human-readable DEBUG messages to `logs/debug.log` so you can
    trace exactly why any signal was approved or rejected.

    Every public method corresponds to one named stage in the pipeline:

        cycle_start()           — new scan cycle begins
        pre_scan_filters()      — Session / Volatility / News gate results
        context()               — Trend, Regime, Liquidity Sweep summary
        strategy_verdicts()     — Trend + LiqSweep verdict output
        smc_ideas()             — Raw SMC ideas count
        confluence()            — Confluence vote summary per idea
        gate_pass()             — A named gate was passed
        gate_reject()           — A named gate rejected the idea
        confidence_breakdown()  — Full confidence component audit trail
        signal_approved()       — Signal passed all gates
        cycle_summary()         — End of cycle totals

    Example debug.log output:
    ─────────────────────────────────────────────────────
    ╔══════════════════════════════════════════════════╗
    ║  SIGNAL CYCLE — 2024-03-20 14:00 UTC            ║
    ╚══════════════════════════════════════════════════╝
    ▶ PRE-SCAN FILTERS
       Session   : ✅ PASS — london active (07:00–15:59)
       Volatility: ✅ PASS — ATR=12.3 pts (range 5.0–40.0)
       News      : ✅ PASS — LOW risk (score=12)
    ▶ CONTEXT
       Trend     : BULLISH  strength=74%  EMA50=2318  EMA200=2285
       Regime    : TRENDING
       Liq Sweep : YES — SELL sweep at 2322.00 (BUY reversal signal)
    ▶ STRATEGY VERDICTS
       ✅ TREND            dir=BUY    conf=74%  BULLISH EMA structure
       ✅ LIQUIDITY_SWEEP  dir=BUY    conf=68%  Buy sweep confirmed
    ▶ SMC IDEAS  (2 candidate(s))
    ═══ IDEA #1 — BUY @ 2315.00 ═══════════════════════
       confluence : BUY boost=+8% active=3 agree=[TREND,SMC,LIQUIDITY_SWEEP]
       gate       : trend_alignment        ✅ PASS
       gate       : regime_gate            ✅ PASS (TRENDING)
       CONFIDENCE BREAKDOWN
         base (SMC)          :    0.720
         + trend_aligned     :   +0.100   → aligned with BULLISH
         + trend_strength    :   +0.059   → 74% strength
         + regime_trending   :   +0.050   → TRENDING market
         + session_london    :   +0.050   → London session
         + vol_optimal       :   +0.050   → ATR in range
         + liq_sweep_match   :   +0.050   → sweep confirms BUY
         + ob_fvg_overlap    :   +0.050   → OB+FVG present
         + confluence_boost  :   +0.080   → 2 strategies agree
       ─────────────────────────────────────────────
         FINAL               :    1.000  (clamped)
       gate       : confidence_threshold   ✅ PASS (100% ≥ 68%)
       gate       : rr_ratio               ✅ PASS (R:R 2.3 ≥ 2.0)
       ✅ SIGNAL APPROVED — BUY XAUUSD @ 2315.00 | conf=100% | lot=0.24
    ───────────────────────────────────────────────────
    ▶ CYCLE SUMMARY: 1 approved / 1 rejected
    """

    # Shared underlying logger — writes only to debug.log
    _debug_log: logging.Logger = get_debug_logger()

    _W = 55           # line width for separator bars

    def __init__(self, idea_counter: int = 0) -> None:
        self._idea_num = idea_counter

    # =========================================================================
    # Public stage methods
    # =========================================================================

    def cycle_start(self, now: Optional[datetime] = None) -> None:
        """Log the beginning of a new signal generation cycle."""
        ts = (now or datetime.utcnow()).strftime("%Y-%m-%d %H:%M UTC")
        self._h1(f"SIGNAL CYCLE — {ts}")

    def pre_scan_filters(
        self,
        sess_ok:     bool, sess_reason:  str,
        vol_ok:      bool, vol_reason:   str,
        news_ok:     bool, news_reason:  str,
    ) -> None:
        """Log the three pre-scan filter results."""
        self._section("PRE-SCAN FILTERS")
        self._log(f"  Session   : {self._tick(sess_ok)} — {sess_reason}")
        self._log(f"  Volatility: {self._tick(vol_ok)} — {vol_reason}")
        self._log(f"  News      : {self._tick(news_ok)} — {news_reason}")

        if not sess_ok and not vol_ok:
            self._log("  ⛔ HARD BLOCK — both session and volatility blocked")

    def context(
        self,
        trend_result:  Any,   # TrendResult
        regime_result: Any,   # RegimeResult
        liq_sweep:     Any,   # LiquiditySweep | None
    ) -> None:
        """Log the shared context computed once per cycle."""
        self._section("CONTEXT")

        # Trend
        dir_val = (
            trend_result.direction.value
            if hasattr(trend_result.direction, "value")
            else str(trend_result.direction)
        )
        self._log(
            f"  Trend     : {dir_val:<10}  "
            f"strength={trend_result.strength:.0%}  "
            f"EMA50={trend_result.ema_50:.2f}  "
            f"EMA200={trend_result.ema_200:.2f}"
        )

        # Regime
        regime_val = (
            regime_result.regime.value
            if hasattr(regime_result, "regime") and hasattr(regime_result.regime, "value")
            else str(getattr(regime_result, "regime", regime_result))
        )
        adx = getattr(regime_result, "adx", None)
        adx_str = f"  ADX={adx:.1f}" if adx is not None else ""
        self._log(f"  Regime    : {regime_val}{adx_str}")

        # Liquidity sweep
        if liq_sweep:
            sw_dir  = getattr(liq_sweep, "direction", "?")
            sw_pri  = getattr(liq_sweep, "price", 0.0)
            sw_type = getattr(liq_sweep, "sweep_type", "")
            self._log(
                f"  Liq Sweep : YES — {sw_dir} sweep @ {sw_pri:.2f}  "
                f"type={sw_type}"
            )
        else:
            self._log("  Liq Sweep : none")

    def strategy_verdicts(self, verdicts: list[Any]) -> None:
        """Log StrategyVerdict objects from Trend and Liquidity Sweep strategies."""
        self._section("STRATEGY VERDICTS")
        for v in verdicts:
            self._log(f"  {v}")

    def smc_ideas(self, ideas: list[Any]) -> None:
        """Log count (and optionally summary) of raw SMC ideas."""
        self._section(f"SMC IDEAS  ({len(ideas)} candidate(s))")
        for i, idea in enumerate(ideas, 1):
            dir_     = getattr(idea, "direction", "?")
            entry_   = getattr(idea, "entry", 0.0)
            conf_    = getattr(idea, "confidence", 0.0)
            has_ob   = "OB " if getattr(idea, "has_ob", False) else ""
            has_fvg  = "FVG " if getattr(idea, "has_fvg", False) else ""
            has_bos  = "BOS " if getattr(idea, "has_bos", False) else ""
            has_choch= "CHoCH " if getattr(idea, "has_choch", False) else ""
            has_liq  = "LiqSweep" if getattr(idea, "has_liquidity_sweep", False) else ""
            smc_tags = (has_ob + has_fvg + has_bos + has_choch + has_liq).strip()
            self._log(
                f"  #{i}  {dir_:<5} @ {entry_:.2f}  "
                f"smc_conf={conf_:.0%}  "
                f"[{smc_tags if smc_tags else 'no tags'}]"
            )

    def idea_start(self, idea_num: int, idea: Any) -> None:
        """Log the beginning of validation for one trade idea."""
        dir_   = getattr(idea, "direction", "?")
        entry_ = getattr(idea, "entry", 0.0)
        conf_  = getattr(idea, "confidence", 0.0)
        self._log(
            f"  {'═'*self._W}"
            f"\n  IDEA #{idea_num} — {dir_} @ {entry_:.2f}  "
            f"(smc_conf={conf_:.0%})"
        )

    def confluence(self, direction: str, confluence: Any) -> None:
        """Log the confluence vote result for one idea."""
        summary = (
            confluence.summary_str()
            if hasattr(confluence, "summary_str")
            else str(confluence)
        )
        conflict = getattr(confluence, "conflict_detected", False)
        icon = "❌ CONFLICT" if conflict else "✅"
        self._log(f"  confluence : {icon} {summary}")

    def gate_pass(self, gate_name: str, detail: str = "") -> None:
        """Log a gate that the idea passed."""
        detail_str = f"  ({detail})" if detail else ""
        self._log(f"  gate       : {gate_name:<28} ✅ PASS{detail_str}")

    def gate_reject(
        self,
        gate_name: str,
        reason:    str,
        idea:      Any = None,
    ) -> None:
        """Log a gate that rejected the idea — the primary debugging entry point."""
        dir_   = getattr(idea, "direction", "?") if idea else "?"
        entry_ = getattr(idea, "entry", 0.0)     if idea else 0.0
        self._log(
            f"  gate       : {gate_name:<28} ❌ REJECT\n"
            f"               Reason  : {reason}\n"
            f"               Idea    : {dir_} @ {entry_:.2f}"
        )

    def confidence_breakdown(
        self,
        base:       float,
        breakdown:  dict,
        final:      float,
        threshold:  float = 0.0,
    ) -> None:
        """
        Log the full confidence scoring audit trail.

        Parameters
        ----------
        base        : Starting SMC confidence score
        breakdown   : Dict of {component_name: delta_value} from _rescore_confidence
        final       : Final clamped confidence
        threshold   : Minimum threshold (for pass/fail indicator)
        """
        self._log(f"  CONFIDENCE BREAKDOWN")
        self._log(f"    {'base (SMC)':<30}: {base:+.3f}")

        running = base
        for component, delta in breakdown.items():
            if delta == 0.0:
                continue
            running += delta
            label = component.replace("_", " ")
            sign  = "+" if delta >= 0 else ""
            self._log(f"    {'+ ' + label:<30}: {sign}{delta:.3f}   → {min(running, 1.0):.3f}")

        self._log(f"    {'─'*44}")
        status = "✅" if final >= threshold else "❌"
        self._log(f"    {'FINAL':<30}: {final:+.3f}  {status}")

    def signal_approved(self, signal: Any) -> None:
        """Log a fully approved trading signal."""
        dir_   = getattr(signal, "direction", "?")
        symbol = getattr(signal, "symbol", "XAUUSD")
        entry_ = getattr(signal, "entry_price", 0.0)
        sl_    = getattr(signal, "stop_loss", 0.0)
        tp1_   = getattr(signal, "take_profit_1", 0.0)
        tp2_   = getattr(signal, "take_profit_2", None)
        lot_   = getattr(signal, "lot_size", 0.0)
        conf_  = getattr(signal, "confidence", 0.0)
        rr_    = getattr(signal, "rr_ratio_tp1", None)
        confl_ = getattr(signal, "confluence_count", 0)
        sess_  = getattr(signal, "session", "")
        regime_= getattr(signal, "regime", "")
        trend_ = getattr(signal, "trend_direction", "")
        risk_  = getattr(signal, "risk_amount_usd", 0.0)

        tp2_line = f"\n    TP2     : {tp2_:.2f}" if tp2_ else ""
        rr_str   = f"{rr_:.2f}" if rr_ else "N/A"

        self._log(
            f"\n  ✅ SIGNAL APPROVED\n"
            f"  {'─'*self._W}\n"
            f"    Direction : {dir_} {symbol}\n"
            f"    Entry     : {entry_:.2f}\n"
            f"    Stop Loss : {sl_:.2f}\n"
            f"    TP1       : {tp1_:.2f}{tp2_line}\n"
            f"    Lot Size  : {lot_:.2f}   Risk: ${risk_:.2f}\n"
            f"    R:R       : {rr_str}:1\n"
            f"    Confidence: {conf_:.1%}\n"
            f"    Confluence: {confl_}/3 strategies agree\n"
            f"    Session   : {sess_.upper()}\n"
            f"    Regime    : {regime_}\n"
            f"    Trend     : {trend_}\n"
            f"  {'─'*self._W}"
        )

    def signal_daily_cap(self, cap: int, signal: Any) -> None:
        """Log that a signal was blocked by the daily cap."""
        dir_   = getattr(signal, "direction", "?")
        entry_ = getattr(signal, "entry_price", 0.0)
        self._log(
            f"  ⛔ DAILY CAP REACHED ({cap})\n"
            f"     Blocked: {dir_} @ {entry_:.2f}"
        )

    def cycle_summary(self, signals: list, rejected: list) -> None:
        """Log end-of-cycle totals."""
        total = len(signals) + len(rejected)
        self._log(
            f"\n  ▶ CYCLE SUMMARY\n"
            f"    Total ideas  : {total}\n"
            f"    ✅ Approved  : {len(signals)}\n"
            f"    ❌ Rejected  : {len(rejected)}"
        )

        # Rejection breakdown by stage
        if rejected:
            stages: dict[str, int] = {}
            for r in rejected:
                stage = getattr(r, "stage", "unknown")
                stages[stage] = stages.get(stage, 0) + 1
            self._log(f"    Rejection stages:")
            for stage, count in sorted(stages.items(), key=lambda x: -x[1]):
                self._log(f"      {stage:<30}: {count}")

        self._log(f"  {'═'*self._W}\n")

    # =========================================================================
    # Convenience wrappers matching common gate names
    # =========================================================================

    def log_regime(
        self,
        regime_label: Any,
        adx: float = 0.0,
        detail:  str = "",
    ) -> None:
        """Standalone regime classification log entry."""
        label = regime_label.value if hasattr(regime_label, "value") else str(regime_label)
        adx_str = f"  ADX={adx:.1f}" if adx else ""
        self._log(f"  [REGIME] {label}{adx_str}  {detail}")

    def log_trend(
        self,
        direction:  str,
        strength:   float,
        ema_50:     float = 0.0,
        ema_200:    float = 0.0,
        slope_deg:  float = 0.0,
    ) -> None:
        """Standalone trend detection log entry."""
        self._log(
            f"  [TREND] {direction}  "
            f"strength={strength:.0%}  "
            f"EMA50={ema_50:.2f}  EMA200={ema_200:.2f}  "
            f"slope={slope_deg:.1f}°"
        )

    def log_filter_result(
        self,
        filter_name: str,
        allowed:     bool,
        reason:      str,
    ) -> None:
        """Generic filter result log entry."""
        icon = "✅" if allowed else "❌"
        self._log(f"  [{filter_name.upper()}] {icon} {reason}")

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _log(self, msg: str) -> None:
        """Write to debug.log at DEBUG level."""
        self._debug_log.debug(msg)

    def _h1(self, title: str) -> None:
        """Write a prominent header to debug.log."""
        bar = "═" * self._W
        self._log(f"\n╔{bar}╗\n║  {title:<{self._W-2}}║\n╚{bar}╝")

    def _section(self, name: str) -> None:
        """Write a section divider."""
        self._log(f"▶ {name}")

    @staticmethod
    def _tick(ok: bool) -> str:
        return "✅ PASS" if ok else "❌ FAIL"


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton (convenience access)
# ─────────────────────────────────────────────────────────────────────────────

#: A shared TradingDebugLogger instance importable directly from the module.
#: Import as:  from utils.logger import trade_debug
trade_debug = TradingDebugLogger()
