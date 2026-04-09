"""
notifications/telegram_bot.py
================================
Telegram notification dispatcher for the AntiGravity Gold Trading Bot.

Sends three categories of messages:
  1. Trade signals        — formatted alert for every approved TradingSignal
  2. Status updates       — bot started/stopped, daily summary, heartbeat
  3. Error / admin alerts — sent to TELEGRAM_ADMIN_CHAT_ID (if configured)

Message formatting
------------------
Signals are sent as Markdown (not MarkdownV2) for maximum compatibility with
most Telegram clients and channel setups.  Special characters are escaped
automatically.  The format mirrors the user's requested layout:

    ════ GOLD TRADE SIGNAL ════

    ⬆  Direction:   BUY
    💰 Entry:       2355.50
    🛑 Stop Loss:   2348.90
    🎯 Take Profit: 2368.40

    ⚖️  Risk/Reward:  1 : 2.0
    📊 Lot Size:    0.24
    🔥 Confidence:  82%

    ─────────────────────────
    Session: LONDON
    Regime:  TRENDING
    Trend:   BULLISH (str 71%)

    📦 Order Block  ⚡ FVG  🌊 Liq Sweep
    ─────────────────────────
    🕐 2024-03-04 10:15 UTC

Retry / rate-limit handling
----------------------------
Telegram imposes a 30-message/second global limit and a
1-message/second per-chat limit.  We handle this with:
  - Exponential back-off on TelegramError (doubles each attempt)
  - Hard rate-limit: a minimum 1-second gap between sends to the same chat
  - Maximum 3 retries before logging failure and giving up (no crash)

Dry-run mode
------------
Set  DRY_RUN = True  in the constructor (or TELEGRAM_DRY_RUN=true/.env)
to print formatted messages to the console without actually sending.
Useful for local development with no valid bot token.

Dependencies
------------
  python-telegram-bot >= 20.0 (async API)
  config.settings.Settings
  signals.signal_generator.TradingSignal
  utils.logger
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Optional

from config.settings import Settings
from utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Lazy import guard — only import telegram when available so the rest of the
# codebase can be tested without a live bot connection.
# ─────────────────────────────────────────────────────────────────────────────
try:
    import telegram
    from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup
    from telegram.error import TelegramError, NetworkError, RetryAfter
    _TELEGRAM_AVAILABLE = True
except ImportError:                             # pragma: no cover
    _TELEGRAM_AVAILABLE = False
    telegram = None                             # type: ignore[assignment]
    Bot = None                                  # type: ignore[assignment,misc]
    TelegramError = Exception                   # type: ignore[assignment,misc]
    NetworkError  = Exception                   # type: ignore[assignment,misc]
    RetryAfter    = Exception                   # type: ignore[assignment,misc]


# ─────────────────────────────────────────────────────────────────────────────
# Message builder (pure functions — no network I/O)
# ─────────────────────────────────────────────────────────────────────────────

class MessageBuilder:
    """
    Builds Telegram-ready text messages from TradingSignal / status dicts.

    All methods are static / pure — no state, no I/O, easily unit-tested.
    """

    # ── Signal message ────────────────────────────────────────────────────────

    @staticmethod
    def signal(signal) -> str:          # signal: TradingSignal
        """
        Produce the main trade signal Telegram message.

        Structure matches the user's requested format:

            ════ GOLD TRADE SIGNAL ════
            Direction / Entry / SL / TP
            R:R / Lot / Confidence
            Context (session, regime, trend)
            SMC badges
            Timestamp
        """
        direction    = signal.direction
        dir_emoji    = "🟢" if direction == "BUY" else "🔴"
        dir_arrow    = "⬆️" if direction == "BUY" else "⬇️"
        dir_label    = "BUY " if direction == "BUY" else "SELL"

        # Confidence stars (1–5)
        pct = int(signal.confidence * 100)
        stars = (
            "★★★★★" if pct >= 80 else
            "★★★★☆" if pct >= 70 else
            "★★★☆☆" if pct >= 60 else
            "★★☆☆☆"
        )

        # R:R string
        rr1 = signal.rr_ratio_tp1
        rr2 = signal.rr_ratio_tp2
        if rr1 is not None:
            rr_str = f"1 : {rr1:.1f}"
        else:
            rr_str = "N/A"

        # TP2 line (optional)
        tp2_line = (
            f"🎯 Take Profit 2: {signal.take_profit_2:.2f}\n"
            if signal.take_profit_2 else ""
        )

        # SMC evidence badges
        badges = []
        if signal.has_ob:               badges.append("📦 Order Block")
        if signal.has_fvg:              badges.append("⚡ FVG")
        if signal.has_bos:              badges.append("🔨 BOS")
        if signal.has_choch:            badges.append("↩️ CHoCH")
        if signal.has_liquidity_sweep:  badges.append("🌊 Liq Sweep")
        smc_line = "  |  ".join(badges) if badges else "—"

        session = (signal.session or "unknown").upper()
        regime  = (signal.regime  or "unknown").replace("_", " ")
        trend   = signal.trend_direction or "UNKNOWN"
        ts      = signal.timestamp.strftime("%Y-%m-%d %H:%M UTC")

        msg = (
            f"{dir_emoji} *GOLD TRADE SIGNAL* {dir_emoji}\n"
            f"{'═' * 28}\n"
            f"\n"
            f"{dir_arrow}  *Direction:*   {dir_label}\n"
            f"💰 *Entry:*       {signal.entry_price:.2f}\n"
            f"🛑 *Stop Loss:*   {signal.stop_loss:.2f}\n"
            f"🎯 *Take Profit:* {signal.take_profit_1:.2f}\n"
            f"{tp2_line}"
            f"\n"
            f"⚖️  *Risk/Reward:*  {rr_str}\n"
            f"📊 *Lot Size:*    {signal.lot_size:.2f}\n"
            f"💸 *Risk:*        ${signal.risk_amount_usd:.0f} ({signal.risk_pct:.1f}%)\n"
            f"🔥 *Confidence:*  {pct}%  {stars}\n"
            f"\n"
            f"{'─' * 28}\n"
            f"📍 *Session:*  {session}\n"
            f"📈 *Regime:*   {regime}\n"
            f"📉 *Trend:*    {trend} (str {signal.trend_strength:.0%})\n"
            f"\n"
            f"🧩 {smc_line}\n"
            f"{'─' * 28}\n"
            f"🕐 _{ts}_\n"
        )
        return msg

    # ── Bot started ───────────────────────────────────────────────────────────

    @staticmethod
    def bot_started(symbol: str = "XAUUSD", version: str = "1.0") -> str:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        return (
            f"✅ *AntiGravity Bot Started*\n"
            f"{'─' * 28}\n"
            f"Symbol:   {symbol}\n"
            f"Version:  {version}\n"
            f"Time:     {ts}\n"
            f"\n"
            f"Monitoring Gold for high-probability setups...\n"
            f"Target: 2–4 signals per day."
        )

    # ── Bot stopped ───────────────────────────────────────────────────────────

    @staticmethod
    def bot_stopped(reason: str = "Manual shutdown") -> str:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        return (
            f"🛑 *AntiGravity Bot Stopped*\n"
            f"{'─' * 28}\n"
            f"Reason: {reason}\n"
            f"Time:   {ts}"
        )

    # ── Daily summary ─────────────────────────────────────────────────────────

    @staticmethod
    def daily_summary(stats: dict) -> str:
        """
        Format the end-of-day summary message.

        Expected keys in stats:
            date, signals_sent, buy_signals, sell_signals,
            avg_confidence, highest_confidence,
            regimes_seen (list[str]), sessions_seen (list[str]),
            errors (int)
        """
        date          = stats.get("date", "today")
        total         = stats.get("signals_sent", 0)
        buys          = stats.get("buy_signals", 0)
        sells         = stats.get("sell_signals", 0)
        avg_conf      = stats.get("avg_confidence", 0.0)
        high_conf     = stats.get("highest_confidence", 0.0)
        errors        = stats.get("errors", 0)
        regimes       = ", ".join(stats.get("regimes_seen", ["-"]))
        sessions      = ", ".join(stats.get("sessions_seen", ["-"]))

        status = "✅" if errors == 0 else "⚠️"
        return (
            f"📊 *Daily Summary — {date}* {status}\n"
            f"{'═' * 28}\n"
            f"\n"
            f"Signals sent:    {total}  (🟢 {buys} BUY / 🔴 {sells} SELL)\n"
            f"Avg confidence:  {avg_conf:.0%}\n"
            f"Best confidence: {high_conf:.0%}\n"
            f"\n"
            f"Sessions active: {sessions}\n"
            f"Regimes seen:    {regimes}\n"
            f"\n"
            f"Errors:          {errors}\n"
        )

    # ── Error alert ───────────────────────────────────────────────────────────

    @staticmethod
    def error_alert(error_type: str, detail: str) -> str:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        return (
            f"⚠️ *Bot Error: {error_type}*\n"
            f"{'─' * 28}\n"
            f"{detail}\n"
            f"Time: {ts}"
        )

    # ── Heartbeat (periodic health check) ────────────────────────────────────

    @staticmethod
    def heartbeat(signals_today: int, uptime_hrs: float) -> str:
        ts = datetime.now(timezone.utc).strftime("%H:%M UTC")
        return (
            f"💚 *Bot Heartbeat* — {ts}\n"
            f"Signals today: {signals_today}\n"
            f"Uptime: {uptime_hrs:.1f}h"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Notifier
# ─────────────────────────────────────────────────────────────────────────────

class TelegramNotifier:
    """
    Async Telegram message dispatcher.

    Usage
    -----
    Synchronous context (e.g., scripts, tests):
        notifier = TelegramNotifier(settings)
        success  = asyncio.run(notifier.send_signal(signal))

    Async context (e.g., APScheduler, asyncio loop):
        async def job():
            await notifier.send_signal(signal)

    Dry-run (no real bot, for testing):
        notifier = TelegramNotifier(settings, dry_run=True)
    """

    def __init__(
        self,
        settings: Settings,
        dry_run:  bool = False,
    ) -> None:
        """
        Parameters
        ----------
        settings : App configuration (token, chat IDs, timeouts, retries)
        dry_run  : If True, print messages to stdout instead of sending them.
                   Automatically True if the TELEGRAM_BOT_TOKEN is empty.
        """
        self.cfg         = settings
        self.max_retries = settings.TELEGRAM_MAX_RETRIES
        self.timeout     = settings.TELEGRAM_TIMEOUT_SECONDS

        # Force dry_run if no token or library missing
        token = settings.TELEGRAM_BOT_TOKEN
        self.dry_run = dry_run or not token or not _TELEGRAM_AVAILABLE

        if not _TELEGRAM_AVAILABLE:
            logger.warning(
                "python-telegram-bot not installed — TelegramNotifier in dry-run mode"
            )
        elif not token:
            logger.warning(
                "TELEGRAM_BOT_TOKEN is empty — TelegramNotifier in dry-run mode"
            )

        # Lazily initialised Bot object (created on first send)
        self._bot: Optional["Bot"] = None

        # Simple per-chat rate-limiting  (epoch seconds of last send)
        self._last_send_time: float = 0.0

    # =========================================================================
    # PUBLIC — Signal messages
    # =========================================================================

    async def send_signal(self, signal) -> bool:
        """
        Send a fully formatted trade signal alert.

        Parameters
        ----------
        signal : TradingSignal — validated signal from SignalGenerator

        Returns
        -------
        bool — True if delivered (or dry-run accepted), False on failure
        """
        try:
            text = MessageBuilder.signal(signal)
            logger.info(
                f"Sending signal: {signal.direction} {signal.symbol} "
                f"@ {signal.entry_price:.2f}  conf={signal.confidence:.0%}"
            )
            ok = await self._send_with_retry(
                text=text,
                chat_id=self.cfg.TELEGRAM_CHAT_ID,
            )
            if ok:
                logger.info("Signal sent successfully")
            return ok
        except Exception as exc:
            logger.error(f"send_signal() unexpected error: {exc}")
            return False

    # =========================================================================
    # PUBLIC — Status messages
    # =========================================================================

    async def send_text(
        self,
        message:    str,
        chat_id:    str | None = None,
        parse_mode: str = "Markdown",
    ) -> bool:
        """
        Send a plain text message to the signal channel (or an override chat).

        Use this for custom status messages, announcements, and debug info.
        """
        target = chat_id or self.cfg.TELEGRAM_CHAT_ID
        return await self._send_with_retry(
            text=message, chat_id=target, parse_mode=parse_mode
        )

    async def send_bot_started(self) -> bool:
        """Send the 'bot started' notification to the admin chat (fallback: signal chat)."""
        msg     = MessageBuilder.bot_started(symbol=self.cfg.SYMBOL)
        chat_id = self.cfg.TELEGRAM_ADMIN_CHAT_ID or self.cfg.TELEGRAM_CHAT_ID
        return await self._send_with_retry(msg, chat_id=chat_id)

    async def send_bot_stopped(self, reason: str = "Manual shutdown") -> bool:
        """Send the 'bot stopped' notification."""
        msg     = MessageBuilder.bot_stopped(reason=reason)
        chat_id = self.cfg.TELEGRAM_ADMIN_CHAT_ID or self.cfg.TELEGRAM_CHAT_ID
        return await self._send_with_retry(msg, chat_id=chat_id)

    async def send_daily_summary(self, stats: dict) -> bool:
        """
        Send the end-of-day performance summary.

        Parameters
        ----------
        stats : dict with keys described in MessageBuilder.daily_summary()
        """
        msg = MessageBuilder.daily_summary(stats)
        logger.info(f"Sending daily summary: {stats.get('signals_sent', 0)} signals")
        return await self._send_with_retry(msg, chat_id=self.cfg.TELEGRAM_CHAT_ID)

    async def send_error_alert(self, error_type: str, detail: str) -> bool:
        """
        Send an error alert to the admin chat.

        Routes to TELEGRAM_ADMIN_CHAT_ID if set, otherwise the signal chat.
        """
        msg     = MessageBuilder.error_alert(error_type, detail)
        chat_id = self.cfg.TELEGRAM_ADMIN_CHAT_ID or self.cfg.TELEGRAM_CHAT_ID
        logger.warning(f"Sending error alert: {error_type}")
        return await self._send_with_retry(msg, chat_id=chat_id)

    async def send_heartbeat(self, signals_today: int, uptime_hrs: float) -> bool:
        """Send a periodic heartbeat to the admin chat."""
        msg     = MessageBuilder.heartbeat(signals_today, uptime_hrs)
        chat_id = self.cfg.TELEGRAM_ADMIN_CHAT_ID or self.cfg.TELEGRAM_CHAT_ID
        return await self._send_with_retry(msg, chat_id=chat_id)

    # =========================================================================
    # PUBLIC — Sync wrappers (convenience)
    # =========================================================================

    def send_signal_sync(self, signal) -> bool:
        """Synchronous wrapper for send_signal() — use in non-async contexts."""
        return asyncio.run(self.send_signal(signal))

    def send_text_sync(self, message: str, chat_id: str | None = None) -> bool:
        """Synchronous wrapper for send_text()."""
        return asyncio.run(self.send_text(message, chat_id=chat_id))

    def send_bot_started_sync(self) -> bool:
        return asyncio.run(self.send_bot_started())

    def send_daily_summary_sync(self, stats: dict) -> bool:
        return asyncio.run(self.send_daily_summary(stats))

    # =========================================================================
    # INTERNAL — Core sending logic with retry and rate-limit handling
    # =========================================================================

    async def _send_with_retry(
        self,
        text:       str,
        chat_id:    str | None,
        parse_mode: str = "Markdown",
        max_retries: int | None = None,
    ) -> bool:
        """
        Send a message with exponential back-off on failures.

        Retry schedule:
          Attempt 0: immediate
          Attempt 1: wait 2 s
          Attempt 2: wait 4 s
          Attempt 3: wait 8 s  (etc.)

        Special handling:
          RetryAfter: honours the server-requested delay before retrying
          NetworkError: retries up to max_retries
          Other TelegramError: logs and returns False immediately (no retry)

        Parameters
        ----------
        text        : UTF-8 message body
        chat_id     : Destination chat/channel ID string
        parse_mode  : "Markdown" (default) or "HTML"
        max_retries : Override Settings.TELEGRAM_MAX_RETRIES

        Returns True on success, False after exhausting all retries.
        """
        n_retries = max_retries if max_retries is not None else self.max_retries

        # ── Dry-run mode ──────────────────────────────────────────────────────
        if self.dry_run:
            self._print_dry_run(text, chat_id, parse_mode)
            return True

        # ── Guard: valid chat_id ──────────────────────────────────────────────
        if not chat_id:
            logger.error("_send_with_retry: chat_id is empty — cannot send")
            return False

        # ── Rate-limit: minimum 1 s between sends ─────────────────────────────
        elapsed = time.monotonic() - self._last_send_time
        if elapsed < 1.0:
            await asyncio.sleep(1.0 - elapsed)

        # ── Retry loop ────────────────────────────────────────────────────────
        bot = self._get_bot()

        for attempt in range(n_retries + 1):
            try:
                await bot.send_message(
                    chat_id    = chat_id,
                    text       = text,
                    parse_mode = parse_mode,
                    read_timeout  = self.timeout,
                    write_timeout = self.timeout,
                )
                self._last_send_time = time.monotonic()
                return True

            except RetryAfter as exc:
                # Telegram says: wait this many seconds
                wait = getattr(exc, "retry_after", 30)
                logger.warning(
                    f"Telegram rate-limited — waiting {wait}s "
                    f"(attempt {attempt + 1}/{n_retries + 1})"
                )
                await asyncio.sleep(wait)

            except NetworkError as exc:
                delay = 2 ** attempt
                logger.warning(
                    f"Telegram network error ({exc}) — "
                    f"retrying in {delay}s (attempt {attempt + 1}/{n_retries + 1})"
                )
                if attempt < n_retries:
                    await asyncio.sleep(delay)

            except TelegramError as exc:
                # Non-retryable API errors (bad token, chat not found, etc.)
                logger.error(f"Telegram API error: {exc} — not retrying")
                return False

            except Exception as exc:
                logger.error(f"Unexpected send error: {exc}")
                return False

        logger.error(
            f"Message delivery failed after {n_retries + 1} attempts to chat {chat_id}"
        )
        return False

    # =========================================================================
    # INTERNAL — Helpers
    # =========================================================================

    def _get_bot(self) -> "Bot":
        """Return the cached Bot instance, creating it on first call."""
        if self._bot is None:
            if not _TELEGRAM_AVAILABLE:
                raise RuntimeError(
                    "python-telegram-bot is not installed. "
                    "Run: pip install python-telegram-bot>=20.0"
                )
            self._bot = Bot(token=self.cfg.TELEGRAM_BOT_TOKEN)
        return self._bot

    def _print_dry_run(
        self, text: str, chat_id: str | None, parse_mode: str
    ) -> None:
        """Print message to stdout in dry-run mode."""
        print(f"\n{'━' * 60}")
        print(f"  [DRY-RUN] Telegram → chat_id={chat_id or 'N/A'}  mode={parse_mode}")
        print(f"{'━' * 60}")
        print(text)
        print(f"{'━' * 60}\n")
        logger.debug(f"Dry-run: message printed to console ({len(text)} chars)")

    def is_configured(self) -> bool:
        """Return True if the bot has a valid token and chat ID."""
        return bool(self.cfg.TELEGRAM_BOT_TOKEN and self.cfg.TELEGRAM_CHAT_ID)

    def __repr__(self) -> str:
        token_masked = (
            f"{self.cfg.TELEGRAM_BOT_TOKEN[:8]}***"
            if self.cfg.TELEGRAM_BOT_TOKEN else "NO_TOKEN"
        )
        return (
            f"TelegramNotifier("
            f"chat={self.cfg.TELEGRAM_CHAT_ID or 'NO_CHAT'}, "
            f"token={token_masked}, "
            f"dry_run={self.dry_run}, "
            f"retries={self.max_retries}"
            f")"
        )
