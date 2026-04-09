"""
data/mt5_connection_test.py
===========================
Standalone smoke-test script for the MT5DataProvider.

Run from the project root (with your .env configured):

    python -m data.mt5_connection_test

Or directly:

    cd gold_trading_bot
    python data/mt5_connection_test.py

What it tests
-------------
1. MT5 package availability
2. Terminal connection + authentication
3. Symbol availability for XAUUSD
4. Live tick / latest price fetch
5. Candle fetch for M5, M15, H1 (20 bars each)
6. DataFrame shape, columns, and dtypes
7. Account info retrieval
8. Clean disconnection

Exit codes
----------
0  — All checks passed
1  — One or more checks failed
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone

# Allow running this file directly from the data/ directory
if __name__ == "__main__":
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from config.settings import Settings
from data.mt5_data import (
    MT5DataProvider,
    MT5ConnectionError,
    MT5DataError,
    MT5NotAvailableError,
    MT5SymbolError,
    TickPrice,
    AccountInfo,
)
from utils.logger import get_logger

logger = get_logger("mt5_connection_test", level="DEBUG")

# ─────────────────────────────────────────────────────────────────────────────
# ANSI colour helpers (works on Windows Terminal / PowerShell)
# ─────────────────────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def ok(msg: str)   -> None: print(f"  {GREEN}✅ PASS{RESET}  {msg}")
def fail(msg: str) -> None: print(f"  {RED}❌ FAIL{RESET}  {msg}")
def info(msg: str) -> None: print(f"  {CYAN}ℹ  INFO{RESET}  {msg}")
def warn(msg: str) -> None: print(f"  {YELLOW}⚠  WARN{RESET}  {msg}")
def header(msg: str) -> None:
    print(f"\n{BOLD}{CYAN}{'─' * 60}{RESET}")
    print(f"{BOLD}{CYAN}  {msg}{RESET}")
    print(f"{BOLD}{CYAN}{'─' * 60}{RESET}")


# ─────────────────────────────────────────────────────────────────────────────
# Individual checks
# ─────────────────────────────────────────────────────────────────────────────

def check_mt5_package() -> bool:
    """Verify the MetaTrader5 package is importable."""
    header("Check 1 — MetaTrader5 Package")
    try:
        import MetaTrader5 as mt5  # noqa: F401
        ver = getattr(mt5, "__version__", "unknown")
        ok(f"MetaTrader5 package importable  (version: {ver})")
        return True
    except ImportError as exc:
        fail(f"Cannot import MetaTrader5: {exc}")
        warn("Install with:  pip install MetaTrader5  (Windows only)")
        return False


def check_connection(provider: MT5DataProvider) -> bool:
    """Attempt to connect to the MT5 terminal."""
    header("Check 2 — Terminal Connection & Authentication")
    try:
        provider.connect_to_mt5()
        ok("MT5 terminal connected and authenticated")
        return True
    except MT5NotAvailableError as exc:
        fail(f"MT5 package not available: {exc}")
        return False
    except MT5ConnectionError as exc:
        fail(f"Connection failed: {exc}")
        warn("Ensure MT5 terminal is running and credentials in .env are correct.")
        return False
    except Exception as exc:
        fail(f"Unexpected error: {exc}")
        return False


def check_symbol(provider: MT5DataProvider, symbol: str) -> bool:
    """Confirm the symbol is available and has a live tick."""
    header(f"Check 3 — Symbol Availability ({symbol})")
    available = provider.is_symbol_available(symbol)
    if available:
        ok(f"Symbol '{symbol}' is available in MarketWatch")
    else:
        fail(f"Symbol '{symbol}' NOT found in the terminal")
        warn("Common variants: XAUUSD, GOLD, XAUUSDm  — check your broker's name.")
    return available


def check_latest_price(provider: MT5DataProvider, symbol: str) -> bool:
    """Fetch and display the live bid/ask tick."""
    header(f"Check 4 — Latest Price ({symbol})")
    try:
        tick: TickPrice = provider.get_latest_price(symbol)
        ok(f"Tick received: {tick}")
        info(f"  bid={tick.bid:.2f}  |  ask={tick.ask:.2f}  |  mid={tick.mid:.2f}")
        info(f"  spread={tick.spread_pts} pts  |  time={tick.time_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        return True
    except MT5DataError as exc:
        fail(f"Tick fetch failed: {exc}")
        return False
    except Exception as exc:
        fail(f"Unexpected error: {exc}")
        return False


def check_candles(provider: MT5DataProvider, symbol: str, timeframe: str, count: int = 20) -> bool:
    """Fetch candles and validate the resulting DataFrame."""
    header(f"Check 5.{timeframe} — Candles ({symbol} {timeframe}, {count} bars)")
    try:
        df: pd.DataFrame = provider.get_candles(symbol, timeframe, count)

        # ── Shape ─────────────────────────────────────────────────────────
        if df.shape[0] < 2:
            fail(f"Expected ≥ 2 rows, got {df.shape[0]}")
            return False
        ok(f"DataFrame shape: {df.shape[0]} rows × {df.shape[1]} columns")

        # ── Required columns ──────────────────────────────────────────────
        required = ["time", "open", "high", "low", "close", "tick_volume",
                    "spread", "real_volume", "range", "body", "body_ratio", "is_bullish"]
        missing  = [c for c in required if c not in df.columns]
        if missing:
            fail(f"Missing columns: {missing}")
            return False
        ok(f"All required columns present: {required}")

        # ── Dtypes ───────────────────────────────────────────────────────
        assert df["time"].dtype.tz is not None, "time column must be UTC-aware"
        assert df["close"].dtype == "float64",  "close must be float64"
        assert df["tick_volume"].dtype == "int64", "tick_volume must be int64"
        ok("Column dtypes correct (UTC-aware time, float64 OHLC, int64 volume)")

        # ── Sanity: OHLC constraints ───────────────────────────────────
        invalid_hl = df[df["high"] < df["low"]]
        if not invalid_hl.empty:
            fail(f"{len(invalid_hl)} bars have high < low")
            return False
        ok("OHLC integrity: high ≥ low for all bars")

        # ── Sample output ─────────────────────────────────────────────
        last = df.iloc[-1]
        info(
            f"Most recent bar  |  time={last['time'].strftime('%Y-%m-%d %H:%M')} UTC  |  "
            f"O={last['open']:.2f}  H={last['high']:.2f}  "
            f"L={last['low']:.2f}  C={last['close']:.2f}  "
            f"vol={last['tick_volume']}"
        )
        info(
            f"  range={last['range']:.2f}  body={last['body']:.2f}  "
            f"body_ratio={last['body_ratio']:.2%}  "
            f"{'🟢 Bullish' if last['is_bullish'] else '🔴 Bearish'}"
        )
        return True

    except (MT5DataError, MT5SymbolError, ValueError) as exc:
        fail(f"Candle fetch failed: {exc}")
        return False
    except AssertionError as exc:
        fail(f"DataFrame validation failed: {exc}")
        return False
    except Exception as exc:
        fail(f"Unexpected error: {exc}")
        return False


def check_multi_timeframe(provider: MT5DataProvider, symbol: str) -> bool:
    """Fetch all three standard timeframes in one call."""
    header(f"Check 6 — Multi-Timeframe Fetch ({symbol})")
    try:
        data = provider.get_multi_timeframe(
            symbol=symbol,
            timeframes=["M5", "M15", "H1"],
            count=50,
        )
        for tf, df in data.items():
            ok(f"  {tf}: {len(df)} bars  |  latest={df['time'].iloc[-1].strftime('%Y-%m-%d %H:%M')} UTC")
        return True
    except Exception as exc:
        fail(f"Multi-TF fetch failed: {exc}")
        return False


def check_account_info(provider: MT5DataProvider) -> bool:
    """Retrieve and display account balance / equity."""
    header("Check 7 — Account Info")
    try:
        acct: AccountInfo = provider.get_account_info()
        ok(f"Account info retrieved: {acct}")
        info(f"  Free margin: {acct.free_margin:,.2f} {acct.currency}")
        return True
    except MT5DataError as exc:
        fail(f"Account info failed: {exc}")
        return False
    except Exception as exc:
        fail(f"Unexpected error: {exc}")
        return False


def check_disconnect(provider: MT5DataProvider) -> bool:
    """Verify clean disconnection."""
    header("Check 8 — Disconnection")
    try:
        provider.disconnect()
        ok("MT5 terminal disconnected cleanly")
        return True
    except Exception as exc:
        fail(f"Disconnection error: {exc}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    print(f"\n{BOLD}{'=' * 60}")
    print("  AntiGravity Gold Bot — MT5 Data Layer Smoke Test")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"{'=' * 60}{RESET}\n")

    # Use Settings without calling validate() so missing Telegram credentials
    # don't block a data-layer test.
    cfg      = Settings()
    provider = MT5DataProvider(cfg)
    symbol   = cfg.SYMBOL

    results: dict[str, bool] = {}

    # Check 1: package
    results["pkg"] = check_mt5_package()
    if not results["pkg"]:
        header("Aborting — MT5 package not installed")
        return 1

    # Check 2: connection
    results["connect"] = check_connection(provider)
    if not results["connect"]:
        header("Aborting — cannot connect to MT5")
        return 1

    # Checks 3–8: data
    results["symbol"]  = check_symbol(provider, symbol)
    results["tick"]    = check_latest_price(provider, symbol)
    results["m5"]      = check_candles(provider, symbol, "M5",  count=30)
    results["m15"]     = check_candles(provider, symbol, "M15", count=30)
    results["h1"]      = check_candles(provider, symbol, "H1",  count=50)
    results["multi"]   = check_multi_timeframe(provider, symbol)
    results["account"] = check_account_info(provider)
    results["disco"]   = check_disconnect(provider)

    # Summary
    header("Test Summary")
    passed = sum(results.values())
    total  = len(results)

    for name, result in results.items():
        status = f"{GREEN}PASS{RESET}" if result else f"{RED}FAIL{RESET}"
        print(f"  [{status}]  {name}")

    print()
    if passed == total:
        print(f"{GREEN}{BOLD}All {total} checks passed ✅{RESET}")
        return 0
    else:
        failed = total - passed
        print(f"{RED}{BOLD}{failed}/{total} checks FAILED ❌{RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
