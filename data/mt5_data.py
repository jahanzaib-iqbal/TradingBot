"""
data/mt5_data.py
================
MetaTrader 5 data provider for the JayBot Gold Trading Bot.

Public API
----------
    provider = MT5DataProvider(settings)

    provider.connect_to_mt5()                          # connect + authenticate
    provider.get_latest_price("XAUUSD")                # → TickPrice dataclass
    provider.get_candles("XAUUSD", "H1", count=200)    # → pd.DataFrame (OHLCV)
    provider.get_account_info()                         # → AccountInfo dataclass
    provider.disconnect()                               # graceful shutdown

Supported timeframes
--------------------
    M1, M5, M15, M30, H1, H4, D1, W1, MN1

DataFrame columns returned by get_candles()
-------------------------------------------
    time        datetime64[ns, UTC]  — bar open time (UTC-aware)
    open        float64
    high        float64
    low         float64
    close       float64
    tick_volume int64                — M5/MT5 tick count (proxy for volume)
    spread      int64                — spread in points at bar open
    real_volume int64                — broker real volume (0 for most brokers)

Notes
-----
- The MetaTrader5 Python package is Windows-only.
- All timestamps are converted to UTC-aware datetime objects.
- Connection state is tracked internally; call connect_to_mt5() once at startup.
- Reconnect logic retries up to Settings.MT5_MAX_RECONNECT_ATTEMPTS times with
  exponential back-off before raising MT5ConnectionError.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Conditional import — MetaTrader5 is Windows-only and may not be installed
# in CI / non-Windows environments.  We guard the import so unit tests that
# mock mt5 can still import this module.
# ---------------------------------------------------------------------------
try:
    import MetaTrader5 as mt5             # type: ignore[import]
    _MT5_AVAILABLE = True
except ImportError:
    mt5 = None                            # type: ignore[assignment]
    _MT5_AVAILABLE = False

from config.settings import Settings
from utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Custom Exceptions
# ─────────────────────────────────────────────────────────────────────────────

class MT5Error(Exception):
    """Base exception for all MT5 data-layer errors."""


class MT5NotAvailableError(MT5Error):
    """Raised when the MetaTrader5 package is not installed."""


class MT5ConnectionError(MT5Error):
    """Raised when the bot cannot connect or authenticate to MT5."""


class MT5DataError(MT5Error):
    """Raised when a data request to MT5 fails or returns empty results."""


class MT5SymbolError(MT5Error):
    """Raised when a requested symbol is not available in the MT5 terminal."""


# ─────────────────────────────────────────────────────────────────────────────
# Return-type dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TickPrice:
    """
    Represents a live price snapshot from MT5.

    Attributes
    ----------
    symbol      : trading instrument name (e.g. "XAUUSD")
    bid         : best bid price
    ask         : best ask price
    spread_pts  : spread in points (ask - bid) / point_size
    last        : last traded price (may be 0 for some brokers)
    time_utc    : UTC timestamp of the tick
    """
    symbol: str
    bid: float
    ask: float
    spread_pts: int
    last: float
    time_utc: datetime

    @property
    def mid(self) -> float:
        """Mid-point price: (bid + ask) / 2."""
        return (self.bid + self.ask) / 2.0

    @property
    def spread_price(self) -> float:
        """Spread expressed in price units (not points)."""
        return self.ask - self.bid

    def __str__(self) -> str:
        return (
            f"{self.symbol}  bid={self.bid:.2f}  ask={self.ask:.2f}"
            f"  spread={self.spread_pts}pts  @ {self.time_utc.strftime('%H:%M:%S')} UTC"
        )


@dataclass(frozen=True)
class AccountInfo:
    """
    Snapshot of the connected MT5 trading account.

    Attributes
    ----------
    login       : account login number
    name        : account holder display name
    server      : broker server name
    balance     : cash balance (USD)
    equity      : equity = balance + floating P&L
    margin      : used margin (USD)
    free_margin : available margin for new positions
    currency    : account currency (e.g. "USD")
    leverage    : account leverage (e.g. 100)
    """
    login: int
    name: str
    server: str
    balance: float
    equity: float
    margin: float
    free_margin: float
    currency: str
    leverage: int

    def __str__(self) -> str:
        return (
            f"Account #{self.login} ({self.name}) | "
            f"Balance: {self.balance:,.2f} {self.currency} | "
            f"Equity: {self.equity:,.2f} | "
            f"Leverage: 1:{self.leverage}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Timeframe mapping
# ─────────────────────────────────────────────────────────────────────────────

# Maps human-readable timeframe strings → MT5 TIMEFRAME_* constants.
# Populated lazily once mt5 is confirmed available.
_TIMEFRAME_MAP: dict[str, int] = {}

def _build_timeframe_map() -> dict[str, int]:
    """Build the timeframe string → MT5 constant mapping."""
    if not _MT5_AVAILABLE:
        return {}
    return {
        "M1":  mt5.TIMEFRAME_M1,
        "M2":  mt5.TIMEFRAME_M2,
        "M3":  mt5.TIMEFRAME_M3,
        "M4":  mt5.TIMEFRAME_M4,
        "M5":  mt5.TIMEFRAME_M5,
        "M6":  mt5.TIMEFRAME_M6,
        "M10": mt5.TIMEFRAME_M10,
        "M12": mt5.TIMEFRAME_M12,
        "M15": mt5.TIMEFRAME_M15,
        "M20": mt5.TIMEFRAME_M20,
        "M30": mt5.TIMEFRAME_M30,
        "H1":  mt5.TIMEFRAME_H1,
        "H2":  mt5.TIMEFRAME_H2,
        "H3":  mt5.TIMEFRAME_H3,
        "H4":  mt5.TIMEFRAME_H4,
        "H6":  mt5.TIMEFRAME_H6,
        "H8":  mt5.TIMEFRAME_H8,
        "H12": mt5.TIMEFRAME_H12,
        "D1":  mt5.TIMEFRAME_D1,
        "W1":  mt5.TIMEFRAME_W1,
        "MN1": mt5.TIMEFRAME_MN1,
    }

# Friendly names used in log messages
_TIMEFRAME_LABELS: dict[str, str] = {
    "M1": "1-Minute",   "M5": "5-Minute",   "M15": "15-Minute",
    "M30": "30-Minute", "H1": "1-Hour",      "H4": "4-Hour",
    "D1":  "Daily",     "W1": "Weekly",      "MN1": "Monthly",
}


# ─────────────────────────────────────────────────────────────────────────────
# DataFrame column definitions
# ─────────────────────────────────────────────────────────────────────────────

# Columns returned by mt5.copy_rates_*  (in order)
_RAW_MT5_COLUMNS = ["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]

# Derived / computed columns added during normalisation
_EXTRA_COLUMNS = ["range", "body", "body_ratio", "is_bullish"]


# ─────────────────────────────────────────────────────────────────────────────
# Main Provider Class
# ─────────────────────────────────────────────────────────────────────────────

class MT5DataProvider:
    """
    Manages all data retrieval from the MetaTrader 5 terminal.

    Lifecycle
    ---------
    1. Instantiate:  provider = MT5DataProvider(settings)
    2. Connect:      provider.connect_to_mt5()
    3. Use:          df = provider.get_candles("XAUUSD", "H1", 200)
                     tick = provider.get_latest_price("XAUUSD")
    4. Shutdown:     provider.disconnect()

    Thread safety
    -------------
    The MT5 Python bridge is single-threaded.  Do not call this class from
    multiple threads simultaneously.
    """

    def __init__(self, settings: Settings) -> None:
        self.cfg = settings
        self._connected: bool = False
        self._timeframe_map: dict[str, int] = {}

        # Validate MT5 availability at construction time (not connection time)
        if not _MT5_AVAILABLE:
            logger.error(
                "MetaTrader5 package is not installed. "
                "Run: pip install MetaTrader5  (Windows only)"
            )

    # =========================================================================
    # CONNECTION
    # =========================================================================

    def connect_to_mt5(self) -> bool:
        """
        Initialise the MT5 terminal and authenticate with the configured account.

        Behaviour
        ---------
        - Attempts to connect up to Settings.MT5_MAX_RECONNECT_ATTEMPTS times.
        - Uses linear back-off: waits attempt × 2 seconds between retries.
        - On success, sets self._connected = True and returns True.
        - On final failure, raises MT5ConnectionError.

        Returns
        -------
        bool
            True if connection and login both succeeded.

        Raises
        ------
        MT5NotAvailableError
            If the MetaTrader5 package is not installed.
        MT5ConnectionError
            If all connection attempts fail.
        """
        if not _MT5_AVAILABLE:
            raise MT5NotAvailableError(
                "MetaTrader5 package is not installed. "
                "Install it with: pip install MetaTrader5"
            )

        if self._connected:
            logger.debug("MT5 already connected — skipping reinitialisation.")
            return True

        # Build timeframe map now that mt5 is confirmed importable
        self._timeframe_map = _build_timeframe_map()

        max_attempts = self.cfg.MT5_MAX_RECONNECT_ATTEMPTS
        last_error: str = ""

        for attempt in range(1, max_attempts + 1):
            logger.info(
                "Connecting to MT5 terminal"
                f" (attempt {attempt}/{max_attempts})"
                f"  server={self.cfg.MT5_SERVER!r}"
                f"  login={self.cfg.MT5_LOGIN}"
            )

            try:
                # ── Step 1: Initialise the terminal process ─────────────────
                init_kwargs: dict = {"timeout": self.cfg.MT5_TIMEOUT_SECONDS * 1000}
                if self.cfg.MT5_PATH:
                    init_kwargs["path"] = self.cfg.MT5_PATH

                if not mt5.initialize(**init_kwargs):
                    last_error = self._format_mt5_error("initialize()")
                    logger.warning(f"MT5 initialize() failed: {last_error}")
                    self._sleep_before_retry(attempt, max_attempts)
                    continue

                # ── Step 2: Authenticate ────────────────────────────────────
                if self.cfg.MT5_LOGIN and self.cfg.MT5_PASSWORD and self.cfg.MT5_SERVER:
                    authorized = mt5.login(
                        login=self.cfg.MT5_LOGIN,
                        password=self.cfg.MT5_PASSWORD,
                        server=self.cfg.MT5_SERVER,
                    )
                    if not authorized:
                        last_error = self._format_mt5_error("login()")
                        logger.warning(
                            f"MT5 login failed for account #{self.cfg.MT5_LOGIN}: {last_error}"
                        )
                        mt5.shutdown()
                        self._sleep_before_retry(attempt, max_attempts)
                        continue
                else:
                    logger.info(
                        "MT5_LOGIN / MT5_PASSWORD / MT5_SERVER not set — "
                        "connecting to already-open terminal without login."
                    )

                # ── Success ─────────────────────────────────────────────────
                self._connected = True
                terminal_info = mt5.terminal_info()
                account_info  = mt5.account_info()

                logger.info(
                    f"✅ MT5 connected  |  "
                    f"terminal_build={getattr(terminal_info, 'build', 'N/A')}  |  "
                    f"login={getattr(account_info, 'login', 'N/A')}  |  "
                    f"server={getattr(account_info, 'server', 'N/A')}  |  "
                    f"currency={getattr(account_info, 'currency', 'N/A')}  |  "
                    f"balance={getattr(account_info, 'balance', 0):,.2f}"
                )
                return True

            except Exception as exc:
                last_error = str(exc)
                logger.error(
                    f"Unexpected exception during MT5 connection attempt {attempt}: {exc}",
                    exc_info=True,
                )
                self._sleep_before_retry(attempt, max_attempts)

        # All attempts exhausted
        raise MT5ConnectionError(
            f"Failed to connect to MT5 after {max_attempts} attempts. "
            f"Last error: {last_error}"
        )

    def disconnect(self) -> None:
        """
        Shut down the MT5 terminal connection cleanly.

        Safe to call even if not currently connected — logs a warning and
        returns without raising.
        """
        if not self._connected:
            logger.debug("disconnect() called but MT5 was not connected — no-op.")
            return

        try:
            mt5.shutdown()
            self._connected = False
            logger.info("MT5 terminal connection closed.")
        except Exception as exc:
            logger.error(f"Error during MT5 shutdown: {exc}", exc_info=True)

    def reconnect(self) -> bool:
        """
        Force a full disconnect → connect cycle.

        Useful when a data call returns an error suggesting the connection
        has been silently dropped.

        Returns
        -------
        bool
            True if reconnection succeeded.
        """
        logger.warning("Initiating MT5 reconnection …")
        self._connected = False
        try:
            mt5.shutdown()
        except Exception:
            pass
        return self.connect_to_mt5()

    # =========================================================================
    # PRICE DATA
    # =========================================================================

    def get_latest_price(self, symbol: Optional[str] = None) -> TickPrice:
        """
        Fetch the current bid/ask tick for a symbol.

        Parameters
        ----------
        symbol : str, optional
            Instrument name.  Defaults to Settings.SYMBOL ("XAUUSD").

        Returns
        -------
        TickPrice
            Dataclass with bid, ask, spread, mid, and UTC timestamp.

        Raises
        ------
        MT5ConnectionError
            If MT5 is not connected.
        MT5SymbolError
            If the symbol is not available in the terminal.
        MT5DataError
            If the tick request returns no data.
        """
        symbol = symbol or self.cfg.SYMBOL
        self._assert_connected("get_latest_price")
        self._ensure_symbol_selected(symbol)

        logger.debug(f"Fetching latest tick for {symbol} …")

        tick = mt5.symbol_info_tick(symbol)

        if tick is None:
            err = self._format_mt5_error(f"symbol_info_tick({symbol})")
            raise MT5DataError(f"No tick data returned for {symbol}. MT5 error: {err}")

        # Convert the MT5 tick timestamp (POSIX seconds, broker timezone) to UTC
        tick_utc = datetime.fromtimestamp(tick.time, tz=timezone.utc)

        # Calculate spread in points
        point = self._get_symbol_point(symbol)
        spread_pts = round((tick.ask - tick.bid) / point) if point else 0

        result = TickPrice(
            symbol=symbol,
            bid=round(tick.bid, self.cfg.SYMBOL_DIGITS),
            ask=round(tick.ask, self.cfg.SYMBOL_DIGITS),
            spread_pts=spread_pts,
            last=round(getattr(tick, "last", 0.0), self.cfg.SYMBOL_DIGITS),
            time_utc=tick_utc,
        )

        logger.debug(f"Latest price: {result}")
        return result

    # =========================================================================
    # CANDLE (OHLCV) DATA
    # =========================================================================

    def get_candles(
        self,
        symbol: Optional[str] = None,
        timeframe: str = "H1",
        count: int = 500,
    ) -> pd.DataFrame:
        """
        Fetch the most recent `count` closed OHLCV bars for `symbol`.

        Parameters
        ----------
        symbol : str, optional
            Trading instrument.  Defaults to Settings.SYMBOL.
        timeframe : str
            Bar timeframe string.  Supported values:
            M1, M5, M15, M30, H1, H4, D1, W1, MN1
        count : int
            How many bars to fetch.  The most recent bar (index 0 from MT5,
            last row in the returned DataFrame) may be incomplete (forming).
            The second-to-last bar is the most recently *closed* bar.
            Minimum: 2  |  Maximum: limited by MT5 history depth.

        Returns
        -------
        pd.DataFrame
            Rows sorted oldest → newest, indexed 0 … count-1.

            Core columns:
                time        datetime64[ns, UTC]
                open        float64
                high        float64
                low         float64
                close       float64
                tick_volume int64
                spread      int64
                real_volume int64

            Derived columns (added for convenience):
                range       float64  — high − low
                body        float64  — |close − open|
                body_ratio  float64  — body / range  (0 if range == 0)
                is_bullish  bool     — close > open

        Raises
        ------
        MT5ConnectionError
            If the MT5 terminal is not connected.
        MT5SymbolError
            If the symbol is not selectable in the terminal.
        MT5DataError
            If MT5 returns no bars or fewer than 2 bars.
        ValueError
            If `timeframe` is not in the supported set.
        """
        symbol    = symbol or self.cfg.SYMBOL
        timeframe = timeframe.upper()

        self._assert_connected("get_candles")
        self._validate_timeframe(timeframe)
        self._ensure_symbol_selected(symbol)

        mt5_tf   = self._timeframe_map[timeframe]
        tf_label = _TIMEFRAME_LABELS.get(timeframe, timeframe)

        logger.debug(
            f"Fetching {count} × {tf_label} bars for {symbol} …"
        )

        # copy_rates_from_pos(symbol, timeframe, start_pos, count)
        # start_pos=0 → start from the most recent bar
        rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, count)

        if rates is None or len(rates) == 0:
            err = self._format_mt5_error(f"copy_rates_from_pos({symbol}, {timeframe})")
            raise MT5DataError(
                f"No candle data returned for {symbol} {timeframe}. "
                f"MT5 error: {err}"
            )

        if len(rates) < 2:
            raise MT5DataError(
                f"Insufficient candle data for {symbol} {timeframe}: "
                f"received {len(rates)} bar(s), need at least 2."
            )

        df = self._rates_to_dataframe(rates, symbol, timeframe)

        logger.debug(
            f"✅ {len(df)} bars fetched for {symbol} {timeframe}  |  "
            f"oldest={df['time'].iloc[0].strftime('%Y-%m-%d %H:%M')} UTC  |  "
            f"newest={df['time'].iloc[-1].strftime('%Y-%m-%d %H:%M')} UTC"
        )

        return df

    def get_multi_timeframe(
        self,
        symbol: Optional[str] = None,
        timeframes: Optional[list[str]] = None,
        count: int = 500,
    ) -> dict[str, pd.DataFrame]:
        """
        Convenience method — fetch candles for multiple timeframes in one call.

        Parameters
        ----------
        symbol : str, optional
            Defaults to Settings.SYMBOL.
        timeframes : list[str], optional
            Defaults to [TREND_TIMEFRAME, SIGNAL_TIMEFRAME, ENTRY_TIMEFRAME]
            from Settings (e.g. ["H4", "H1", "M15"]).
        count : int
            Number of bars to fetch per timeframe.

        Returns
        -------
        dict[str, pd.DataFrame]
            Mapping of timeframe string → DataFrame, e.g.:
            {
                "H4":  <DataFrame>,
                "H1":  <DataFrame>,
                "M15": <DataFrame>,
            }
        """
        symbol = symbol or self.cfg.SYMBOL
        if timeframes is None:
            timeframes = [
                self.cfg.TREND_TIMEFRAME,
                self.cfg.SIGNAL_TIMEFRAME,
                self.cfg.ENTRY_TIMEFRAME,
            ]

        result: dict[str, pd.DataFrame] = {}

        for tf in timeframes:
            try:
                result[tf] = self.get_candles(symbol, tf, count)
            except MT5Error as exc:
                logger.error(
                    f"Failed to fetch {tf} candles for {symbol}: {exc}"
                )
                raise

        logger.info(
            f"Multi-TF fetch complete for {symbol}: "
            + ", ".join(f"{tf}={len(df)}bars" for tf, df in result.items())
        )
        return result

    # =========================================================================
    # ACCOUNT INFO
    # =========================================================================

    def get_account_info(self) -> AccountInfo:
        """
        Retrieve live account balance, equity, and margin from MT5.

        Returns
        -------
        AccountInfo
            Dataclass with login, balance, equity, margin, currency, leverage.

        Raises
        ------
        MT5ConnectionError
            If MT5 is not connected.
        MT5DataError
            If account_info() returns None.
        """
        self._assert_connected("get_account_info")
        logger.debug("Fetching MT5 account info …")

        info = mt5.account_info()
        if info is None:
            err = self._format_mt5_error("account_info()")
            raise MT5DataError(f"Failed to retrieve account info. MT5 error: {err}")

        result = AccountInfo(
            login=info.login,
            name=info.name,
            server=info.server,
            balance=info.balance,
            equity=info.equity,
            margin=info.margin,
            free_margin=info.margin_free,
            currency=info.currency,
            leverage=info.leverage,
        )

        logger.debug(f"Account info: {result}")
        return result

    # =========================================================================
    # SYMBOL UTILITIES
    # =========================================================================

    def is_symbol_available(self, symbol: str) -> bool:
        """
        Check whether a symbol exists and is enabled in the MT5 terminal.

        Parameters
        ----------
        symbol : str
            Instrument name to check.

        Returns
        -------
        bool
        """
        if not self._connected:
            return False
        info = mt5.symbol_info(symbol)
        return info is not None

    def get_symbol_info(self, symbol: Optional[str] = None) -> dict:
        """
        Return a dictionary of key symbol properties.

        Returned keys: name, point, digits, spread, trade_mode,
                       volume_min, volume_max, volume_step, description.

        Parameters
        ----------
        symbol : str, optional
            Defaults to Settings.SYMBOL.

        Returns
        -------
        dict

        Raises
        ------
        MT5ConnectionError
            If MT5 is not connected.
        MT5SymbolError
            If the symbol is not found.
        """
        symbol = symbol or self.cfg.SYMBOL
        self._assert_connected("get_symbol_info")

        info = mt5.symbol_info(symbol)
        if info is None:
            raise MT5SymbolError(
                f"Symbol '{symbol}' not found in the MT5 terminal. "
                "Check that the symbol is added to MarketWatch."
            )

        return {
            "name":        info.name,
            "point":       info.point,
            "digits":      info.digits,
            "spread":      info.spread,
            "trade_mode":  info.trade_mode,
            "volume_min":  info.volume_min,
            "volume_max":  info.volume_max,
            "volume_step": info.volume_step,
            "description": info.description,
        }

    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================

    def _rates_to_dataframe(
        self,
        rates: object,          # numpy structured array from MT5
        symbol: str,
        timeframe: str,
    ) -> pd.DataFrame:
        """
        Convert a raw MT5 rates array into a clean, typed pandas DataFrame.

        Steps performed
        ---------------
        1. Build DataFrame from the numpy structured array
        2. Rename columns to standard names
        3. Convert `time` from POSIX seconds to UTC-aware datetime
        4. Cast numeric columns to float64 / int64
        5. Sort oldest → newest (MT5 returns newest first)
        6. Add derived helper columns: range, body, body_ratio, is_bullish
        7. Reset index to 0 … N-1
        """
        df = pd.DataFrame(rates)

        # MT5 column order: time, open, high, low, close, tick_volume, spread, real_volume
        df.columns = _RAW_MT5_COLUMNS

        # ── Timestamp conversion ──────────────────────────────────────────────
        # MT5 times are POSIX seconds in broker timezone (usually UTC or UTC+2/3).
        # We convert to UTC-aware pandas Timestamps.
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)

        # ── Type casting ──────────────────────────────────────────────────────
        for col in ("open", "high", "low", "close"):
            df[col] = df[col].astype("float64").round(self.cfg.SYMBOL_DIGITS)

        for col in ("tick_volume", "spread", "real_volume"):
            df[col] = df[col].astype("int64")

        # ── Sort oldest → newest ─────────────────────────────────────────────
        # MT5 copy_rates_from_pos returns newest bar at index 0 in the numpy array,
        # but after DataFrame construction the order might vary — sort to be safe.
        df.sort_values("time", ascending=True, inplace=True)
        df.reset_index(drop=True, inplace=True)

        # ── Derived columns ───────────────────────────────────────────────────
        df["range"]      = (df["high"] - df["low"]).round(self.cfg.SYMBOL_DIGITS)
        df["body"]       = (df["close"] - df["open"]).abs().round(self.cfg.SYMBOL_DIGITS)
        df["body_ratio"] = (
            (df["body"] / df["range"].replace(0, float("nan")))
            .fillna(0.0)
            .round(4)
        )
        df["is_bullish"] = df["close"] > df["open"]

        return df

    def _validate_timeframe(self, timeframe: str) -> None:
        """Raise ValueError if timeframe is not in the supported set."""
        if not self._timeframe_map:
            # Map may be empty if MT5 not available; skip validation
            return
        if timeframe not in self._timeframe_map:
            supported = sorted(self._timeframe_map.keys())
            raise ValueError(
                f"Unsupported timeframe: {timeframe!r}. "
                f"Supported values: {supported}"
            )

    def _ensure_symbol_selected(self, symbol: str) -> None:
        """
        Add `symbol` to MarketWatch and enable it if not already visible.

        MT5 will not return data for symbols that have been removed from
        MarketWatch, even if they are valid instruments.
        """
        info = mt5.symbol_info(symbol)
        if info is None:
            raise MT5SymbolError(
                f"Symbol '{symbol}' not found in the MT5 terminal. "
                "Verify the broker's exact symbol name (e.g. 'XAUUSD', 'GOLD', 'XAUUSDm')."
            )

        if not info.visible:
            # Enable the symbol in MarketWatch so data requests work
            if not mt5.symbol_select(symbol, True):
                err = self._format_mt5_error(f"symbol_select({symbol})")
                raise MT5SymbolError(
                    f"Failed to enable symbol '{symbol}' in MarketWatch. "
                    f"MT5 error: {err}"
                )
            logger.debug(f"Symbol '{symbol}' added to MarketWatch.")

    def _get_symbol_point(self, symbol: str) -> float:
        """Return the point size for `symbol`, or Settings.SYMBOL_POINT as default."""
        try:
            info = mt5.symbol_info(symbol)
            return info.point if info else self.cfg.SYMBOL_POINT
        except Exception:
            return self.cfg.SYMBOL_POINT

    def _format_mt5_error(self, context: str) -> str:
        """
        Retrieve the last MT5 error and format it as a human-readable string.

        Parameters
        ----------
        context : str
            Description of the MT5 call that failed (for log clarity).

        Returns
        -------
        str
            Formatted error string, e.g. "code=10004 (Trade timeout)"
        """
        try:
            code, description = mt5.last_error()
            return f"code={code} ({description}) [in {context}]"
        except Exception:
            return f"unknown error [in {context}]"

    def _assert_connected(self, caller: str) -> None:
        """Raise MT5ConnectionError if not connected."""
        if not self._connected:
            raise MT5ConnectionError(
                f"{caller}() called but MT5 is not connected. "
                "Call connect_to_mt5() first."
            )

    def _sleep_before_retry(self, attempt: int, max_attempts: int) -> None:
        """Wait between retry attempts — skips sleep on final attempt."""
        if attempt < max_attempts:
            wait = attempt * 2          # 2s, 4s, 6s, 8s …
            logger.info(f"Waiting {wait}s before retry …")
            time.sleep(wait)

    # =========================================================================
    # CONTEXT MANAGER SUPPORT
    # =========================================================================

    def __enter__(self) -> "MT5DataProvider":
        """Support  with MT5DataProvider(cfg) as provider:  usage."""
        self.connect_to_mt5()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Automatically disconnect on context manager exit."""
        self.disconnect()

    # =========================================================================
    # REPR
    # =========================================================================

    def __repr__(self) -> str:
        state = "connected" if self._connected else "disconnected"
        return (
            f"MT5DataProvider("
            f"symbol={self.cfg.SYMBOL!r}, "
            f"state={state})"
        )
