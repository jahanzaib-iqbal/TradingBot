"""
main.py
========
AntiGravity Gold Trading Signal Bot — Entry Point & Main Loop.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ARCHITECTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  main.py
    │
    └─ TradingBot.run()          ← async main loop (1-min heartbeat)
          │
          ├─ STARTUP
          │    ├─ connect MT5
          │    ├─ validate config
          │    └─ send "Bot started" Telegram
          │
          ├─ EACH CYCLE  (every 60 seconds)
          │    ├─ 1. Fetch multi-TF data  (H4/H1/M15)
          │    ├─ 2. Session filter       → skip if Asian/closed
          │    ├─ 3. News filter          → skip near high-impact events
          │    ├─ 4. Volatility filter    → skip if ATR too low/high
          │    ├─ 5. Daily risk gate      → skip if loss/cap exceeded
          │    ├─ 6. Active-signal guard  → skip if signal already live
          │    ├─ 7. SignalGenerator      → trend + regime + SMC scan
          │    ├─ 8. Validate signals     → R:R, confidence, direction
          │    └─ 9. Telegram dispatch    → send + record
          │
          └─ SHUTDOWN
               ├─ disconnect MT5
               └─ send daily summary to Telegram

ACTIVE-SIGNAL GUARD
━━━━━━━━━━━━━━━━━━━
The bot tracks the most-recently sent signal and blocks new signals
of the SAME direction until the trade is considered resolved.

Resolution conditions (conservative defaults):
  - OPPOSITE signal detected            → prior signal invalidated
  - N bars elapsed since signal (M15)   → signal expired
  - User runs with --reset-signal flag  → manual clear

The guard ensures "only one active signal at a time" as required.

CLI Options
━━━━━━━━━━━
  python main.py                    Live mode (real Telegram)
  python main.py --dry-run          Live data, no Telegram dispatch
  python main.py --interval 60      Override loop interval (seconds)
  python main.py --reset-signal     Clear active signal on startup
  python main.py --no-mt5           Skip MT5; use synthetic demo data

Dependencies
━━━━━━━━━━━━
  config.settings         → Settings
  data.mt5_data           → MT5DataProvider
  filters.*               → SessionFilter, VolatilityFilter, NewsFilter
  signals.signal_generator→ SignalGenerator, TradingSignal
  notifications.telegram_bot → TelegramNotifier
  utils.logger            → get_logger
"""

from __future__ import annotations

import argparse
import asyncio
import signal as _signal
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional

# ── Project imports ───────────────────────────────────────────────────────────
from config.settings import Settings
from data.mt5_data import MT5DataProvider, MT5Error
from filters.news_filter import NewsFilter
from filters.session_filter import SessionFilter
from filters.volatility_filter import VolatilityFilter
from notifications.discord_notifier import send_trade_signal, send_bot_started, send_bot_stopped, send_daily_report
from signals.signal_generator import SignalGenerator, TradingSignal
from utils.logger import get_logger
from data.trade_tracker import TradeTracker

logger = get_logger(__name__)

# Bot version tag (update when releasing)
BOT_VERSION = "1.0.0"


# ─────────────────────────────────────────────────────────────────────────────
# Active-signal state
# ─────────────────────────────────────────────────────────────────────────────

# Number of M15 bars after which a signal is considered "expired / closed"
# 16 bars × 15 min = 4 hours max hold window
_SIGNAL_EXPIRY_BARS: int = 16


@dataclass
class ActiveSignalState:
    """
    Tracks the most-recently dispatched signal to enforce the
    'one active signal at a time' rule.

    The signal is considered ACTIVE until one of:
      1. An opposite-direction signal fires  → invalidates prior trade
      2. `expiry_time` is reached           → trade window timed out
      3. Bot is restarted with --reset-signal
    """
    signal:      TradingSignal
    sent_at:     datetime
    expiry_time: datetime

    def is_expired(self, now: datetime | None = None) -> bool:
        now = now or datetime.now(timezone.utc)
        return now >= self.expiry_time

    def is_opposite(self, new_direction: str) -> bool:
        return new_direction != self.signal.direction

    def __str__(self) -> str:
        remaining = (self.expiry_time - datetime.now(timezone.utc)).total_seconds()
        return (
            f"ActiveSignal({self.signal.direction} @ {self.signal.entry_price:.2f} | "
            f"conf={self.signal.confidence:.0%} | "
            f"expires in {max(0, remaining / 60):.0f} min)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Daily statistics tracker
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DailyStats:
    """Accumulates per-day metrics for the end-of-day summary Telegram message."""
    date:               str   = ""
    signals_sent:       int   = 0
    buy_signals:        int   = 0
    sell_signals:       int   = 0
    confidences:        list  = field(default_factory=list)
    regimes_seen:       set   = field(default_factory=set)
    sessions_seen:      set   = field(default_factory=set)
    cycles_run:         int   = 0
    cycles_skipped:     int   = 0
    errors:             int   = 0
    mt5_reconnects:     int   = 0

    def reset(self, date: str) -> None:
        self.date           = date
        self.signals_sent   = 0
        self.buy_signals    = 0
        self.sell_signals   = 0
        self.confidences    = []
        self.regimes_seen   = set()
        self.sessions_seen  = set()
        self.cycles_run     = 0
        self.cycles_skipped = 0
        self.errors         = 0
        self.mt5_reconnects = 0

    def record_signal(self, sig: TradingSignal) -> None:
        self.signals_sent += 1
        if sig.direction == "BUY":
            self.buy_signals += 1
        else:
            self.sell_signals += 1
        self.confidences.append(sig.confidence)
        if sig.session:
            self.sessions_seen.add(sig.session)
        if sig.regime:
            self.regimes_seen.add(sig.regime)

    def to_summary_dict(self) -> dict:
        confs = self.confidences
        return {
            "date":               self.date,
            "signals_sent":       self.signals_sent,
            "buy_signals":        self.buy_signals,
            "sell_signals":       self.sell_signals,
            "avg_confidence":     sum(confs) / len(confs) if confs else 0.0,
            "highest_confidence": max(confs, default=0.0),
            "regimes_seen":       sorted(self.regimes_seen),
            "sessions_seen":      sorted(self.sessions_seen),
            "errors":             self.errors,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Demo / synthetic data provider (used with --no-mt5)
# ─────────────────────────────────────────────────────────────────────────────

def _make_demo_df(n: int = 600, base: float = 2350.0,
                  trend: float = 0.06, noise: float = 3.0,
                  interval_min: int = 60) -> "pd.DataFrame":
    """Generate synthetic OHLCV data for demo/dry-run without MT5."""
    import numpy as np
    import pandas as pd

    rng    = __import__("numpy").random.default_rng(42)
    t      = __import__("numpy").arange(n, dtype=float)
    close  = base + t * trend + rng.normal(0, noise, n)
    high   = close + rng.uniform(1.0, 4.0, n)
    low    = close - rng.uniform(1.0, 4.0, n)
    open_  = close + rng.uniform(-1.5, 1.5, n)
    high   = np.maximum(high, np.maximum(open_, close))
    low    = np.minimum(low,  np.minimum(open_, close))
    start  = datetime(2024, 3, 1, tzinfo=timezone.utc)
    times  = pd.to_datetime([
        start + timedelta(minutes=interval_min * i) for i in range(n)
    ], utc=True)
    return pd.DataFrame({
        "time":        times,
        "open":        open_.round(2),
        "high":        high.round(2),
        "low":         low.round(2),
        "close":       close.round(2),
        "tick_volume": rng.integers(300, 1200, n).astype(float),
    })


# ─────────────────────────────────────────────────────────────────────────────
# Main Bot class
# ─────────────────────────────────────────────────────────────────────────────

class TradingBot:
    """
    Complete async trading bot lifecycle manager.

    Responsibilities
    ----------------
    - Component initialisation and MT5 connection management
    - The main 60-second polling loop
    - Pipeline orchestration per cycle
    - Active-signal guard (one signal at a time)
    - Daily stats tracking and end-of-day summary dispatch
    - Graceful shutdown and cleanup

    Usage
    -----
        bot = TradingBot(cfg, dry_run=True)
        asyncio.run(bot.run())
    """

    def __init__(
        self,
        cfg:           Settings,
        dry_run:       bool = False,
        loop_interval: int  | None = None,
        no_mt5:        bool = False,
        reset_signal:  bool = False,
    ) -> None:
        self.cfg           = cfg
        self.dry_run       = dry_run
        self.no_mt5        = no_mt5
        self.loop_interval = loop_interval or cfg.LOOP_INTERVAL_SECONDS
        self._running      = False

        # ── Sub-components ────────────────────────────────────────────────────
        self.data          = MT5DataProvider(cfg)

        self.session_f     = SessionFilter(cfg)
        self.volatility_f  = VolatilityFilter(cfg)
        self.news_f        = NewsFilter(cfg, safe_mode=True)  # no live news in v1

        self.signal_gen    = SignalGenerator(
            cfg,
            session_filter    = self.session_f,
            volatility_filter = self.volatility_f,
            news_filter       = self.news_f,
        )



        # ── State ─────────────────────────────────────────────────────────────
        self._active_signal: Optional[ActiveSignalState] = None
        self._stats         = DailyStats()
        self._start_time    = datetime.now(timezone.utc)
        
        self.executor       = ThreadPoolExecutor(max_workers=3)
        self.tracker        = TradeTracker()
        self._report_sent_today = False

        if reset_signal:
            logger.info("--reset-signal: active signal cleared on startup")
            self._active_signal = None

    # =========================================================================
    # PUBLIC — Lifecycle
    # =========================================================================

    async def run(self) -> None:
        """
        Start the bot.  Blocks until shutdown (Ctrl-C / SIGTERM / error).

        Flow:
            startup()
            loop:
                wait for next aligned minute
                _run_cycle()
                sleep(interval)
            shutdown()
        """
        try:
            await self._startup()
            self._running = True
            await self._loop()
        except asyncio.CancelledError:
            logger.info("Cancelled — shutting down")
        except Exception as exc:
            logger.critical(f"Fatal error in run(): {exc}", exc_info=True)
            await self._send_error_alert("FATAL_ERROR", str(exc))
        finally:
            await self._shutdown()

    # =========================================================================
    # STARTUP & SHUTDOWN
    # =========================================================================

    async def _startup(self) -> None:
        """Initialise connections and announce bot start."""
        logger.info("=" * 60)
        logger.info(f"  AntiGravity Gold Bot v{BOT_VERSION}  —  starting up")
        logger.info("=" * 60)
        logger.info(f"  Symbol:        {self.cfg.SYMBOL}")
        logger.info(f"  Timeframes:    {self.cfg.TREND_TIMEFRAME} / "
                    f"{self.cfg.SIGNAL_TIMEFRAME} / {self.cfg.ENTRY_TIMEFRAME}")
        logger.info(f"  Loop interval: {self.loop_interval}s")
        logger.info(f"  Dry-run:       {self.dry_run}")
        logger.info(f"  No-MT5 mode:   {self.no_mt5}")
        logger.info(f"  Conf threshold:{self.cfg.CONFIDENCE_THRESHOLD:.0%}")
        logger.info(f"  Min R:R:       {self.cfg.MIN_RR_RATIO}:1")
        logger.info(f"  Max signals/d: UNLIMITED (24/7 Mode)")
        logger.info("=" * 60)
        print(f"  Jahanzaib Gold Bot v{BOT_VERSION}  -  Started, Best of Luck :) ")

        if not self.no_mt5:
            self._connect_mt5()

        # Reset daily stats
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self._stats.reset(today)

        # Announce on Discord
        send_bot_started(symbol=self.cfg.SYMBOL, version=BOT_VERSION)

    async def _shutdown(self) -> None:
        """Cleanup connections and send daily summary."""
        logger.info("Shutting down …")

        if not self.no_mt5:
            try:
                self.data.disconnect()
                logger.info("MT5 disconnected")
            except Exception as exc:
                logger.warning(f"MT5 disconnect error: {exc}")

        send_bot_stopped(reason="Normal shutdown")
        self.executor.shutdown(wait=False)
        logger.info("Bot stopped.")

    # =========================================================================
    # MAIN LOOP
    # =========================================================================

    async def _loop(self) -> None:
        """
        Main polling loop.  Runs forever until _running is False.

        On each tick:
          1. Check if UTC date rolled over → reset daily stats + send summary
          2. Execute one full analysis cycle
          3. Sleep for loop_interval seconds
        """
        last_date = datetime.now(timezone.utc).date()

        while self._running:
            now = datetime.now(timezone.utc)

            # ── Date rollover ─────────────────────────────────────────────────
            now_pkt = now + timedelta(hours=5)
            if now.date() != last_date:
                logger.info("Date rollover — resetting daily stats")
                await self._on_date_rollover(str(last_date))
                last_date = now.date()
                
            # Check for 23:59 PKT daily report
            if now_pkt.hour == 23 and now_pkt.minute == 59:
                if not self._report_sent_today:
                    self._report_sent_today = True
                    date_str = self.tracker.get_pkt_date_str(now)
                    self.executor.submit(self._generate_and_send_report, date_str)
            elif now_pkt.hour == 0:
                self._report_sent_today = False

            # ── Analysis cycle ────────────────────────────────────────────────
            try:
                await self._run_cycle(now)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._stats.errors += 1
                logger.error(f"Cycle error: {exc}", exc_info=True)
                await self._send_error_alert("CYCLE_ERROR", str(exc))

            # ── Sleep until next cycle ────────────────────────────────────────
            await asyncio.sleep(self.loop_interval)

    # =========================================================================
    # ONE ANALYSIS CYCLE
    # =========================================================================

    async def _run_cycle(self, now: datetime) -> None:
        """
        Execute one complete signal-detection cycle.

        Gates are applied in order — any failure short-circuits the pipeline
        without errors (it's a normal, expected condition).

        Returns without sending a signal if any gate fails.
        """
        self._stats.cycles_run += 1
        cycle_start = datetime.now(timezone.utc)
        logger.info(f"── Cycle #{self._stats.cycles_run}  {now.strftime('%H:%M:%S UTC')} ──")

        # ─────────────────────────────────────────────────────────────────────
        # GATE 1 — Session check
        # ─────────────────────────────────────────────────────────────────────
        sess_ok, sess_reason = self.session_f.is_allowed(now)
        session_name = self.session_f.current_session(now)
        logger.info(f"  Session: {session_name.upper()}  allowed={sess_ok}")
        if not sess_ok:
            logger.info(f"  → Skip: {sess_reason}")
            self._stats.cycles_skipped += 1
            return

        # ─────────────────────────────────────────────────────────────────────
        # GATE 2 — News check
        # ─────────────────────────────────────────────────────────────────────
        news_ok, news_reason = self.news_f.is_allowed(now)
        if not news_ok:
            logger.info(f"  → Skip (news): {news_reason}")
            self._stats.cycles_skipped += 1
            return

        # ─────────────────────────────────────────────────────────────────────
        # STEP 3 — Fetch market data
        # ─────────────────────────────────────────────────────────────────────
        df_trend, df_signal, df_entry = self._fetch_data()
        if df_trend is None or df_signal is None:
            logger.warning("  → Skip: data fetch failed")
            self._stats.cycles_skipped += 1
            return
            
        # Update current open trades asynchronously non-blocking
        if df_entry is not None and not df_entry.empty:
            self.executor.submit(self.tracker.update_open_trades, df_entry.copy())

        # ─────────────────────────────────────────────────────────────────────
        # GATE 4 — Volatility check (on signal-TF bars)
        # ─────────────────────────────────────────────────────────────────────
        vol_ok, vol_reason = self.volatility_f.is_allowed(df_signal)
        atr = self.volatility_f.compute_atr(df_signal)
        logger.info(f"  ATR: {atr:.2f}  volatility_ok={vol_ok}")
        if not vol_ok:
            logger.info(f"  → Skip: {vol_reason}")
            self._stats.cycles_skipped += 1
            return

        # ─────────────────────────────────────────────────────────────────────
        # GATE 5 — Daily cap check (DISABLED PER USER REQUEST)
        # ─────────────────────────────────────────────────────────────────────
        # User requested no maximum signals limit.

        # ─────────────────────────────────────────────────────────────────────
        # GATE 6 — Active-signal guard
        # ─────────────────────────────────────────────────────────────────────
        if self._active_signal:
            if self._active_signal.is_expired(now):
                logger.info(f"  Active signal expired — cleared: {self._active_signal}")
                self._active_signal = None
            else:
                logger.info(f"  → Skip: {self._active_signal} still active")
                self._stats.cycles_skipped += 1
                return

        # ─────────────────────────────────────────────────────────────────────
        # STEP 7 — Run SignalGenerator
        # ─────────────────────────────────────────────────────────────────────
        account_balance = self._get_account_balance()

        signals, rejected = self.signal_gen.generate(
            df_h1           = df_trend,
            df_m15          = df_signal,
            account_balance = account_balance,
            df_m5           = df_entry,
            now             = now,
        )

        logger.info(
            f"  SignalGen: {len(signals)} approved, {len(rejected)} rejected"
        )

        # Log rejection reasons at debug level for auditing
        for rej in rejected:
            logger.debug(f"    Rejected [{rej.stage}]: {rej.reason}")

        # ─────────────────────────────────────────────────────────────────────
        # STEP 8 — Dispatch the best signal
        # ─────────────────────────────────────────────────────────────────────
        if signals:
            # Pick highest-confidence signal
            best = max(signals, key=lambda s: s.confidence)
            await self._dispatch_signal(best, now)

    # =========================================================================
    # SIGNAL DISPATCH
    # =========================================================================

    async def _dispatch_signal(
        self, signal: TradingSignal, now: datetime
    ) -> None:
        """
        Send a validated signal to Telegram and record it as the active signal.

        Active-signal guard logic:
          - If an opposite-direction signal is detected, the prior signal is
            cancelled / logged before the new one is activated.
          - After dispatch, the signal is locked for _SIGNAL_EXPIRY_BARS × 15 min.
        """
        expiry_minutes = _SIGNAL_EXPIRY_BARS * 15   # 4 hours default
        expiry_time    = now + timedelta(minutes=expiry_minutes)

        # If an opposite signal fires (e.g., prior BUY now SELL) log the event
        if self._active_signal and self._active_signal.is_opposite(signal.direction):
            logger.info(
                f"Opposite signal — prior {self._active_signal.signal.direction} "
                f"invalidated by new {signal.direction}"
            )
            self._active_signal = None

        # Record stats
        self._stats.record_signal(signal)

        # Log to console (always)
        logger.info(f"")
        logger.info(f"  ★ SIGNAL APPROVED ★")
        logger.info(f"  {signal}")
        logger.info(f"")

        # Send to Discord
        success = send_trade_signal(signal.to_dict())

        if success:
            # Lock in the active-signal guard
            self._active_signal = ActiveSignalState(
                signal      = signal,
                sent_at     = now,
                expiry_time = expiry_time,
            )
            logger.info(
                f"  Signal dispatched — active until "
                f"{expiry_time.strftime('%H:%M UTC')} "
                f"({_SIGNAL_EXPIRY_BARS} bars)"
            )
            
            # Submit trade to tracker asynchronously AFTER discord notification
            self.executor.submit(self.tracker.add_trade, signal.to_dict())
            
        else:
            logger.error("  Discord send failed — signal NOT locked as active")

    # =========================================================================
    # DATA FETCHING
    # =========================================================================

    def _fetch_data(
        self,
    ) -> tuple[Optional["pd.DataFrame"],
               Optional["pd.DataFrame"],
               Optional["pd.DataFrame"]]:
        """
        Fetch three timeframes of OHLCV data.

        Returns (df_trend, df_signal, df_entry):
            df_trend  = H4/H1 for trend + regime context
            df_signal = H1/M15 for SMC scan (primary)
            df_entry  = M15/M5 for entry refinement (optional)

        Returns (None, None, None) on any data error.
        """
        if self.no_mt5:
            return self._fetch_demo_data()

        symbol = self.cfg.SYMBOL
        n      = self.cfg.BARS_TO_FETCH

        try:
            df_trend  = self.data.get_candles(symbol, self.cfg.TREND_TIMEFRAME,  count=n)
            df_signal = self.data.get_candles(symbol, self.cfg.SIGNAL_TIMEFRAME, count=n)
            df_entry  = self.data.get_candles(symbol, self.cfg.ENTRY_TIMEFRAME,  count=min(n, 300))

            # Validate minimum bar counts
            min_bars = 210  # EMA 200 warmup minimum
            if len(df_trend) < min_bars or len(df_signal) < min_bars:
                logger.warning(
                    f"Insufficient bars: trend={len(df_trend)}, "
                    f"signal={len(df_signal)}  (need {min_bars})"
                )
                return None, None, None

            logger.debug(
                f"Data fetched: {self.cfg.TREND_TIMEFRAME}={len(df_trend)} bars "
                f"{self.cfg.SIGNAL_TIMEFRAME}={len(df_signal)} bars "
                f"{self.cfg.ENTRY_TIMEFRAME}={len(df_entry)} bars"
            )
            return df_trend, df_signal, df_entry

        except MT5Error as exc:
            self._stats.errors += 1
            logger.error(f"MT5 data fetch error: {exc}")
            # Attempt reconnection
            if self._attempt_reconnect():
                return None, None, None
            return None, None, None

        except Exception as exc:
            self._stats.errors += 1
            logger.error(f"Unexpected data fetch error: {exc}", exc_info=True)
            return None, None, None

    def _fetch_demo_data(
        self,
    ) -> tuple["pd.DataFrame", "pd.DataFrame", "pd.DataFrame"]:
        """Return synthetic data when running in --no-mt5 mode."""
        df_trend  = _make_demo_df(n=600, interval_min=60)    # H1 equivalent
        df_signal = _make_demo_df(n=400, interval_min=15)    # M15
        df_entry  = _make_demo_df(n=300, interval_min=5)     # M5
        return df_trend, df_signal, df_entry

    # =========================================================================
    # MT5 HELPERS
    # =========================================================================

    def _connect_mt5(self) -> None:
        """Connect to MT5 terminal.  Exits if connection fails."""
        try:
            self.data.connect_to_mt5()
            logger.info("MT5 connected successfully")
        except MT5Error as exc:
            logger.critical(f"MT5 connection failed: {exc}")
            sys.exit(1)

    def _attempt_reconnect(self) -> bool:
        """Try to reconnect to MT5.  Returns True if retry should happen."""
        try:
            self.data.reconnect()
            self._stats.mt5_reconnects += 1
            logger.info("MT5 reconnected successfully")
            return True
        except Exception as exc:
            logger.error(f"MT5 reconnect failed: {exc}")
            return False

    def _get_account_balance(self) -> float:
        """Fetch live account balance; fall back to Settings value on error."""
        if self.no_mt5:
            return self.cfg.ACCOUNT_BALANCE

        try:
            info = self.data.get_account_info()
            balance = getattr(info, "balance", None) or getattr(info, "equity", None)
            if balance and balance > 0:
                return float(balance)
        except Exception as exc:
            logger.warning(f"Could not fetch account balance: {exc}")

        return self.cfg.ACCOUNT_BALANCE

    # =========================================================================
    # DATE ROLLOVER & ERROR ALERTS
    # =========================================================================

    async def _on_date_rollover(self, date_str: str) -> None:
        """Send the daily summary and reset counters at UTC midnight."""
        summary = self._stats.to_summary_dict()
        summary["date"] = date_str   # use the *old* date for the summary
        self._stats.reset(datetime.now(timezone.utc).strftime("%Y-%m-%d"))
        # Clear expired active signal on new day
        self._active_signal = None
        logger.info("Daily stats reset for new UTC day")

    def _generate_and_send_report(self, date_str: str) -> None:
        """Run asynchronously to compile and dispatch the daily Discord report."""
        try:
            trades = self.tracker.get_daily_trades(date_str)
            daily_p = self.tracker.get_daily_performance(date_str)
            life_p = self.tracker.get_overall_performance()
            send_daily_report(date_str, trades, daily_p, life_p)
            logger.info(f"Daily Discord Report dispatched for PKT date: {date_str}")
        except Exception as e:
            logger.error(f"Failed to compile and send daily report: {e}")

    async def _send_error_alert(self, error_type: str, detail: str) -> None:
        """Error alerts not routed to discord."""
        pass

    def stop(self) -> None:
        """Signal the loop to exit cleanly on the next iteration."""
        logger.info("stop() called — setting _running=False")
        self._running = False


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AntiGravity Gold Trading Signal Bot",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Fetch live data but print signals to console instead of Telegram",
    )
    parser.add_argument(
        "--no-mt5", action="store_true",
        help="Skip MT5 connection; use synthetic demo data (for local testing)",
    )
    parser.add_argument(
        "--interval", type=int, default=None, metavar="SECONDS",
        help="Override loop interval in seconds (default: Settings.LOOP_INTERVAL_SECONDS)",
    )
    parser.add_argument(
        "--reset-signal", action="store_true",
        help="Clear the active-signal guard on startup",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    cfg  = Settings()

    bot = TradingBot(
        cfg           = cfg,
        dry_run       = args.dry_run,
        no_mt5        = args.no_mt5,
        loop_interval = args.interval,
        reset_signal  = args.reset_signal,
    )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # ── Graceful shutdown handlers ────────────────────────────────────────────
    def _shutdown_handler(signum, frame):           # SIGINT / SIGTERM
        logger.info(f"Signal {signum} received — requesting shutdown …")
        bot.stop()
        for task in asyncio.all_tasks(loop):
            task.cancel()

    import signal as _sig
    for s in (_sig.SIGINT, _sig.SIGTERM):
        try:
            # Unix: add_signal_handler (async-safe)
            loop.add_signal_handler(s, lambda: _shutdown_handler(s, None))
        except (NotImplementedError, AttributeError):
            # Windows: fall back to stdlib signal.signal
            _sig.signal(s, _shutdown_handler)

    # ── Run ───────────────────────────────────────────────────────────────────
    try:
        loop.run_until_complete(bot.run())
    except (KeyboardInterrupt, SystemExit):
        logger.info("KeyboardInterrupt — exiting")
    finally:
        # Cancel remaining tasks
        pending = asyncio.all_tasks(loop)
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()
        logger.info("Event loop closed. Goodbye.")


if __name__ == "__main__":
    main()
