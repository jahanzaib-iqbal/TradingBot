"""
notifications/telegram_bot_test.py
====================================
Self-contained tests for TelegramNotifier and MessageBuilder.
Runs entirely in dry-run mode — no real Telegram credentials needed.

Tests
-----
 1. MessageBuilder.signal() — BUY message contains required fields
 2. MessageBuilder.signal() — SELL message direction correct
 3. MessageBuilder.signal() — confidence stars: >=80% gets 5 stars
 4. MessageBuilder.signal() — TP2 line present when provided
 5. MessageBuilder.signal() — TP2 line absent when None
 6. MessageBuilder.signal() — SMC badge line includes OB, FVG, BOS
 7. MessageBuilder.bot_started() — contains symbol and timestamp
 8. MessageBuilder.bot_stopped() — contains reason
 9. MessageBuilder.daily_summary() — contains all stat fields
10. MessageBuilder.error_alert() — contains error type and detail
11. MessageBuilder.heartbeat() — contains signals_today count
12. TelegramNotifier — constructs in dry_run mode (no token)
13. TelegramNotifier.is_configured() — False when no token
14. send_signal() dry-run — returns True and prints output
15. send_text() dry-run — returns True
16. send_bot_started() dry-run — returns True
17. send_daily_summary() dry-run — returns True
18. send_error_alert() dry-run — returns True
19. send_heartbeat() dry-run — returns True
20. send_signal_sync() — sync wrapper works correctly
21. Retry guard: no chat_id returns False
22. Repr format contains key info

Run:
    cd gold_trading_bot
    python notifications/telegram_bot_test.py
"""

from __future__ import annotations

import asyncio
import sys
import os
from datetime import datetime, timezone

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── ANSI colours ──────────────────────────────────────────────────────────────
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
    # Ensure dry-run by clearing any token that might be in .env
    cfg.TELEGRAM_BOT_TOKEN  = ""
    cfg.TELEGRAM_CHAT_ID    = "123456789"
    cfg.TELEGRAM_ADMIN_CHAT_ID = ""
    return cfg


def _make_buy_signal():
    """Construct a BUY TradingSignal without running the full pipeline."""
    from signals.signal_generator import TradingSignal
    return TradingSignal(
        symbol           = "XAUUSD",
        direction        = "BUY",
        timestamp        = datetime(2024, 3, 4, 10, 15, tzinfo=timezone.utc),
        entry_price      = 2355.50,
        stop_loss        = 2348.90,
        take_profit_1    = 2368.40,
        take_profit_2    = 2381.30,
        lot_size         = 0.24,
        risk_amount_usd  = 156.0,
        risk_pct         = 1.0,
        rr_ratio_tp1     = 1.94,
        rr_ratio_tp2     = 3.88,
        monetary_value_1pt = 0.24,
        confidence       = 0.82,
        smc_confidence   = 0.70,
        trend_strength   = 0.71,
        regime           = "TRENDING",
        session          = "london",
        trend_direction  = "BULLISH",
        has_ob           = True,
        has_fvg          = True,
        has_bos          = True,
        has_choch        = False,
        has_liquidity_sweep = True,
        notes            = "Bullish OB + FVG + liquidity sweep",
    )


def _make_sell_signal():
    from signals.signal_generator import TradingSignal
    return TradingSignal(
        symbol           = "XAUUSD",
        direction        = "SELL",
        timestamp        = datetime(2024, 3, 4, 14, 30, tzinfo=timezone.utc),
        entry_price      = 2355.50,
        stop_loss        = 2362.10,
        take_profit_1    = 2342.60,
        take_profit_2    = None,
        lot_size         = 0.18,
        risk_amount_usd  = 119.0,
        risk_pct         = 1.0,
        rr_ratio_tp1     = 1.95,
        rr_ratio_tp2     = None,
        monetary_value_1pt = 0.18,
        confidence       = 0.72,
        smc_confidence   = 0.62,
        trend_strength   = 0.58,
        regime           = "TRENDING",
        session          = "overlap",
        trend_direction  = "BEARISH",
        has_ob           = True,
        has_fvg          = False,
        has_bos          = False,
        has_choch        = True,
        has_liquidity_sweep = True,
        notes            = "Bearish OB + CHoCH + liquidity sweep",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    print(f"\n{BOLD}{'='*60}")
    print("  TelegramNotifier -- Message Format & Dry-Run Tests")
    print(f"{'='*60}{RESET}")

    results = []
    cfg     = _settings()

    from notifications.telegram_bot import TelegramNotifier, MessageBuilder

    buy_sig  = _make_buy_signal()
    sell_sig = _make_sell_signal()

    # ── Test 1: BUY message has required fields ───────────────────────────────
    hdr("Test 1 -- BUY signal message contains all required fields")
    msg1 = MessageBuilder.signal(buy_sig)
    required = ["GOLD TRADE SIGNAL", "BUY", "2355.50", "2348.90", "2368.40",
                "0.24", "82%", "1 : 1.9", "LONDON"]
    missing1 = [r for r in required if r not in msg1]
    t1 = not missing1
    ok(f"All {len(required)} required strings present") if t1 else \
    fail(f"Missing: {missing1}")
    results.append(t1)

    # ── Test 2: SELL message direction correct ────────────────────────────────
    hdr("Test 2 -- SELL signal message shows correct direction")
    msg2 = MessageBuilder.signal(sell_sig)
    t2 = "SELL" in msg2 and "⬇️" in msg2 and "🔴" in msg2
    ok("SELL / arrow / red circle present") if t2 else fail("Direction markers missing")
    results.append(t2)

    # ── Test 3: Confidence stars ──────────────────────────────────────────────
    hdr("Test 3 -- Confidence >= 80% shows 5 stars")
    t3 = "★★★★★" in msg1   # 82% -> 5 stars
    ok("5 stars for 82% confidence") if t3 else fail(f"Expected ★★★★★ in:\n{msg1[:300]}")
    results.append(t3)

    # ── Test 4: TP2 line present when provided ────────────────────────────────
    hdr("Test 4 -- TP2 line present when take_profit_2 is set")
    t4 = "2381.30" in msg1 and "Take Profit 2" in msg1
    ok("TP2 line found with correct price") if t4 else fail(f"TP2 line missing from BUY msg")
    results.append(t4)

    # ── Test 5: TP2 line absent when None ────────────────────────────────────
    hdr("Test 5 -- TP2 line absent when take_profit_2 is None")
    t5 = "Take Profit 2" not in msg2
    ok("TP2 line correctly omitted for SELL (no TP2)") if t5 else fail("TP2 line should not appear")
    results.append(t5)

    # ── Test 6: SMC badges ────────────────────────────────────────────────────
    hdr("Test 6 -- SMC badges for OB, FVG, BOS, Liq Sweep in BUY message")
    expected_badges = ["Order Block", "FVG", "BOS", "Liq Sweep"]
    missing6 = [b for b in expected_badges if b not in msg1]
    t6 = not missing6
    ok(f"All SMC badges present: {expected_badges}") if t6 else \
    fail(f"Missing badges: {missing6}")
    results.append(t6)

    # ── Test 7: bot_started ───────────────────────────────────────────────────
    hdr("Test 7 -- MessageBuilder.bot_started() contains symbol and timestamp")
    msg7 = MessageBuilder.bot_started(symbol="XAUUSD")
    t7 = "XAUUSD" in msg7 and "Bot Started" in msg7 and "UTC" in msg7
    ok("bot_started message valid") if t7 else fail(f"bot_started missing expected content")
    results.append(t7)

    # ── Test 8: bot_stopped ───────────────────────────────────────────────────
    hdr("Test 8 -- MessageBuilder.bot_stopped() contains reason")
    msg8 = MessageBuilder.bot_stopped(reason="User request")
    t8 = "User request" in msg8 and "Stopped" in msg8
    ok("bot_stopped message valid") if t8 else fail("bot_stopped missing reason")
    results.append(t8)

    # ── Test 9: daily_summary ─────────────────────────────────────────────────
    hdr("Test 9 -- MessageBuilder.daily_summary() contains all stat fields")
    stats = {
        "date": "2024-03-04",
        "signals_sent": 3,
        "buy_signals":  2,
        "sell_signals": 1,
        "avg_confidence": 0.74,
        "highest_confidence": 0.82,
        "regimes_seen": ["TRENDING"],
        "sessions_seen": ["london", "overlap"],
        "errors": 0,
    }
    msg9 = MessageBuilder.daily_summary(stats)
    checks9 = ["2024-03-04", "3", "74%", "82%", "TRENDING", "london"]
    missing9 = [c for c in checks9 if c not in msg9]
    t9 = not missing9
    ok("Daily summary contains all expected fields") if t9 else \
    fail(f"Missing: {missing9}")
    results.append(t9)

    # ── Test 10: error_alert ──────────────────────────────────────────────────
    hdr("Test 10 -- MessageBuilder.error_alert() contains type and detail")
    msg10 = MessageBuilder.error_alert("MT5_DISCONNECT", "Connection lost to broker server")
    t10 = "MT5_DISCONNECT" in msg10 and "Connection lost" in msg10
    ok("Error alert message valid") if t10 else fail("Error alert missing content")
    results.append(t10)

    # ── Test 11: heartbeat ────────────────────────────────────────────────────
    hdr("Test 11 -- MessageBuilder.heartbeat() contains signals and uptime")
    msg11 = MessageBuilder.heartbeat(signals_today=2, uptime_hrs=4.5)
    t11 = "2" in msg11 and "4.5" in msg11 and "Heartbeat" in msg11
    ok("Heartbeat message valid") if t11 else fail("Heartbeat missing expected values")
    results.append(t11)

    # ── Test 12: Constructor dry_run ──────────────────────────────────────────
    hdr("Test 12 -- TelegramNotifier constructs correctly in dry_run mode")
    notifier = TelegramNotifier(cfg, dry_run=True)
    t12 = notifier.dry_run is True
    ok(f"dry_run=True confirmed  repr={notifier!r}") if t12 else fail("dry_run should be True")
    results.append(t12)

    # ── Test 13: is_configured() ──────────────────────────────────────────────
    hdr("Test 13 -- is_configured() returns False with no token")
    t13 = not notifier.is_configured()   # token is "" from _settings()
    ok("is_configured()=False (empty token)") if t13 else fail("should be False with empty token")
    results.append(t13)

    # ── Test 14–19: Async sends (all dry-run) ────────────────────────────────

    async def run_async_tests():
        r = []

        # Test 14: send_signal
        hdr("Test 14 -- send_signal() dry-run returns True")
        ok14 = await notifier.send_signal(buy_sig)
        r.append(ok14)
        ok(f"send_signal returned True") if ok14 else fail("send_signal returned False")

        # Test 15: send_text
        hdr("Test 15 -- send_text() dry-run returns True")
        ok15 = await notifier.send_text("Test status update message")
        r.append(ok15)
        ok("send_text returned True") if ok15 else fail("send_text returned False")

        # Test 16: send_bot_started
        hdr("Test 16 -- send_bot_started() dry-run returns True")
        ok16 = await notifier.send_bot_started()
        r.append(ok16)
        ok("send_bot_started returned True") if ok16 else fail("send_bot_started returned False")

        # Test 17: send_daily_summary
        hdr("Test 17 -- send_daily_summary() dry-run returns True")
        ok17 = await notifier.send_daily_summary(stats)
        r.append(ok17)
        ok("send_daily_summary returned True") if ok17 else fail("returned False")

        # Test 18: send_error_alert
        hdr("Test 18 -- send_error_alert() dry-run returns True")
        ok18 = await notifier.send_error_alert("TEST_ERROR", "Unit test error")
        r.append(ok18)
        ok("send_error_alert returned True") if ok18 else fail("returned False")

        # Test 19: send_heartbeat
        hdr("Test 19 -- send_heartbeat() dry-run returns True")
        ok19 = await notifier.send_heartbeat(signals_today=2, uptime_hrs=3.5)
        r.append(ok19)
        ok("send_heartbeat returned True") if ok19 else fail("returned False")

        return r

    async_results = asyncio.run(run_async_tests())
    results.extend(async_results)

    # ── Test 20: sync wrapper ─────────────────────────────────────────────────
    hdr("Test 20 -- send_signal_sync() sync wrapper works")
    notifier2 = TelegramNotifier(cfg, dry_run=True)
    ok20 = notifier2.send_signal_sync(sell_sig)
    t20 = ok20 is True
    ok("send_signal_sync returned True") if t20 else fail("sync wrapper failed")
    results.append(t20)

    # ── Test 21: no chat_id returns False ─────────────────────────────────────
    hdr("Test 21 -- _send_with_retry fails gracefully when chat_id is empty")
    async def _test21():
        n = TelegramNotifier(cfg, dry_run=False)
        n.dry_run = False   # force real path (but no bot)
        # Call with empty chat_id — should return False, not raise
        return await n._send_with_retry("hello", chat_id="", parse_mode="Markdown")

    # It won't reach the bot since chat_id guard fires first
    cfg21 = _settings()
    notifier21 = TelegramNotifier(cfg21, dry_run=False)
    notifier21.dry_run = False
    result21 = asyncio.run(notifier21._send_with_retry("hi", chat_id=""))
    t21 = result21 is False
    ok("Empty chat_id correctly returns False") if t21 else fail(f"Expected False, got {result21}")
    results.append(t21)

    # ── Test 22: repr ─────────────────────────────────────────────────────────
    hdr("Test 22 -- repr() contains key info")
    r22 = repr(notifier)
    t22 = "TelegramNotifier" in r22 and "dry_run=True" in r22 and "retries=" in r22
    ok(r22) if t22 else fail(f"repr missing expected fields: {r22}")
    results.append(t22)

    # ── Summary ───────────────────────────────────────────────────────────────
    hdr("Summary")
    passed = sum(results)
    total  = len(results)
    for i, r in enumerate(results, 1):
        s = f"{GREEN}PASS{RESET}" if r else f"{RED}FAIL{RESET}"
        print(f"  [{s}]  Test {i}")
    print()

    # Print one full signal message for visual inspection
    print(f"\n{BOLD}{'─'*60}")
    print("  Sample BUY Signal Message (as it would appear on Telegram):")
    print(f"{'─'*60}{RESET}")
    print(MessageBuilder.signal(buy_sig))
    print(f"{BOLD}{'─'*60}{RESET}")

    if passed == total:
        print(f"\n{GREEN}{BOLD}All {total} tests passed{RESET}")
        return 0
    else:
        print(f"\n{RED}{BOLD}{total - passed}/{total} tests FAILED{RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
