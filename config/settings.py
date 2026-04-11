"""
config/settings.py
==================
Central configuration hub for the AntiGravity Gold Trading Bot.

All parameters are loaded from environment variables (via a .env file) so the
bot can be reconfigured without touching source code.  Every parameter has a
sensible default for paper-trading / development.

Usage
-----
    from config.settings import get_settings

    cfg = get_settings()                  # returns a cached singleton
    print(cfg.TELEGRAM_BOT_TOKEN)
    print(cfg.RISK_PER_TRADE_PCT)

Environment file
----------------
Copy  .env.example  →  .env  and fill in your values before running the bot.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import List

from dotenv import load_dotenv

# Load .env as early as possible so os.getenv() calls below pick up values.
load_dotenv(override=False)


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _env_float(key: str, default: float) -> float:
    """Read an environment variable and parse it as float."""
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError as exc:
        raise ValueError(f"[Settings] '{key}' must be a number. Got: {val!r}") from exc


def _env_int(key: str, default: int) -> int:
    """Read an environment variable and parse it as int."""
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError as exc:
        raise ValueError(f"[Settings] '{key}' must be an integer. Got: {val!r}") from exc


def _env_list(key: str, default: list[str]) -> list[str]:
    """Read a comma-separated environment variable as a list of strings."""
    val = os.getenv(key)
    if val is None:
        return default
    return [item.strip() for item in val.split(",") if item.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# Settings class
# ─────────────────────────────────────────────────────────────────────────────

class Settings:
    """
    Immutable configuration object built from environment variables.

    Instantiate once via  get_settings()  (singleton) rather than calling
    Settings() directly in every module.
    """

    # The Discord Webhook URL for sending signals
    DISCORD_WEBHOOK_URL: str = os.getenv("DISCORD_WEBHOOK_URL", "")

    # =========================================================================
    # METATRADER 5
    # =========================================================================

    # Your MT5 account login number (integer).
    MT5_LOGIN: int = _env_int("MT5_LOGIN", 0)

    # MT5 account password (plain text — keep in .env, never commit to git).
    MT5_PASSWORD: str = os.getenv("MT5_PASSWORD", "")

    # The broker's MT5 server name (e.g. "ICMarkets-Demo", "XM-MT5-Demo3").
    MT5_SERVER: str = os.getenv("MT5_SERVER", "")

    # Full path to the MT5 terminal executable.
    # Example: "C:\\Program Files\\MetaTrader 5\\terminal64.exe"
    MT5_PATH: str = os.getenv("MT5_PATH", "")

    # How long (seconds) to wait for the MT5 terminal to initialise before
    # declaring a connection failure.
    MT5_TIMEOUT_SECONDS: int = _env_int("MT5_TIMEOUT_SECONDS", 30)

    # Maximum number of automatic reconnection attempts if MT5 goes offline.
    MT5_MAX_RECONNECT_ATTEMPTS: int = _env_int("MT5_MAX_RECONNECT_ATTEMPTS", 5)

    # =========================================================================
    # TRADING SYMBOL
    # =========================================================================

    # The traded instrument.  Must match the broker's exact symbol name.
    # Common variants: "XAUUSD", "GOLD", "XAUUSDm" (micro account suffix).
    SYMBOL: str = os.getenv("SYMBOL", "XAUUSD")

    # Pip / point size for the symbol in price terms (1 point = 0.01 for XAUUSD).
    SYMBOL_POINT: float = _env_float("SYMBOL_POINT", 0.01)

    # Digit precision for price display (2 decimals = "$2345.67").
    SYMBOL_DIGITS: int = _env_int("SYMBOL_DIGITS", 2)

    # =========================================================================
    # TIMEFRAMES
    # =========================================================================

    # Higher-timeframe used to establish macro trend and market regime.
    # Valid values: "M1","M5","M15","M30","H1","H4","D1","W1","MN1"
    TREND_TIMEFRAME: str = os.getenv("TREND_TIMEFRAME", "H4")

    # Intermediate timeframe where SMC setups (OB, FVG, BOS) are identified.
    SIGNAL_TIMEFRAME: str = os.getenv("SIGNAL_TIMEFRAME", "H1")

    # Lower timeframe used to refine the precise entry price within the zone.
    ENTRY_TIMEFRAME: str = os.getenv("ENTRY_TIMEFRAME", "M15")

    # Number of historical bars to fetch per timeframe for indicator warmup.
    # Rule of thumb: at least 3× the longest indicator period (e.g. EMA 200 → 600).
    BARS_TO_FETCH: int = _env_int("BARS_TO_FETCH", 600)

    # =========================================================================
    # ACCOUNT & RISK
    # =========================================================================

    # Approximate starting account balance in USD.
    # Used for lot-size calculations when live balance cannot be fetched from MT5.
    ACCOUNT_BALANCE: float = _env_float("ACCOUNT_BALANCE", 10_000.0)

    # Percentage of account balance to risk on each individual trade.
    # Example: 1.0 → risk $100 on a $10,000 account (per trade).
    # Recommended range: 0.5 – 2.0 for conservative intraday trading.
    RISK_PER_TRADE_PCT: float = _env_float("RISK_PER_TRADE_PCT", 1.0)

    # Disabled: MAX_DAILY_LOSS_PCT
    # Disabled: MAX_SIGNALS_PER_DAY

    # Minimum acceptable reward-to-risk ratio for a signal to be published.
    # Signals with R:R below this value are discarded.
    # Example: 2.0 → only take trades where potential gain ≥ 2× potential loss.
    MIN_RR_RATIO: float = _env_float("MIN_RR_RATIO", 1.2)

    # Maximum lot size the bot will ever suggest (safety ceiling).
    MAX_LOT_SIZE: float = _env_float("MAX_LOT_SIZE", 5.0)

    # Minimum lot size (broker floor; most brokers allow 0.01).
    MIN_LOT_SIZE: float = _env_float("MIN_LOT_SIZE", 0.01)

    # =========================================================================
    # SIGNAL CONFIDENCE THRESHOLD
    # =========================================================================

    # Composite confluence score (0.0 – 1.0) that a setup must achieve before
    # a signal is generated.  Score is built from weighted SMC factors:
    #   • Market regime alignment   (25 %)
    #   • Order Block quality       (25 %)
    #   • Fair Value Gap overlap    (20 %)
    #   • BOS / ChoCH confirmation  (15 %)
    #   • Liquidity target distance (15 %)
    #
    # Recommended: 0.65 – 0.75 for selective, high-quality signals.
    # Lower values → more signals, lower average win rate.
    # Higher values → fewer signals, higher average win rate (fewer setups qualify).
    CONFIDENCE_THRESHOLD: float = _env_float("CONFIDENCE_THRESHOLD", 0.68)

    # =========================================================================
    # SESSION WINDOWS  (all times in UTC, 24-hour "HH:MM" format)
    # =========================================================================

    # Asian session — Gold liquidity is generally low; disabled by default.
    ASIAN_SESSION_START: str = os.getenv("ASIAN_SESSION_START", "00:00")
    ASIAN_SESSION_END: str   = os.getenv("ASIAN_SESSION_END",   "06:59")

    # London session — highest Gold liquidity in Europe; primary session.
    LONDON_SESSION_START: str = os.getenv("LONDON_SESSION_START", "07:00")
    LONDON_SESSION_END: str   = os.getenv("LONDON_SESSION_END",   "15:59")

    # New York session — strong USD-driven volatility; secondary session.
    NEW_YORK_SESSION_START: str = os.getenv("NEW_YORK_SESSION_START", "12:00")
    NEW_YORK_SESSION_END: str   = os.getenv("NEW_YORK_SESSION_END",   "20:59")

    # London–New York overlap — highest intraday liquidity; best for scalps.
    OVERLAP_START: str = os.getenv("OVERLAP_START", "12:00")
    OVERLAP_END: str   = os.getenv("OVERLAP_END",   "15:59")

    # Comma-separated list of sessions during which signals ARE allowed.
    # Options: "asian", "london", "new_york", "overlap"
    # Default: London and New York only (no Asian session for Gold).
    ACTIVE_SESSIONS: List[str] = _env_list(
        "ACTIVE_SESSIONS", ["london", "new_york"]
    )

    # =========================================================================
    # ATR (Average True Range) SETTINGS
    # =========================================================================

    # Number of bars used to calculate ATR.
    # Standard value: 14 (matches MT4/MT5 default).
    ATR_PERIOD: int = _env_int("ATR_PERIOD", 14)

    # Minimum ATR value (in price points) required to consider the market
    # tradeable.  Below this the market is too quiet / choppy.
    # For XAUUSD: a typical quiet bar ATR is ~3–5 pts; active is 8–20 pts.
    ATR_MIN_POINTS: float = _env_float("ATR_MIN_POINTS", 5.0)

    # Maximum ATR value beyond which the market is considered too erratic
    # (e.g., during a surprise news spike).  Signals are blocked above this.
    ATR_MAX_POINTS: float = _env_float("ATR_MAX_POINTS", 40.0)

    # ATR multiplier used when calculating stop-loss distance from an Order Block.
    # stop_loss_distance = ATR × ATR_SL_MULTIPLIER
    ATR_SL_MULTIPLIER: float = _env_float("ATR_SL_MULTIPLIER", 1.5)

    # ATR multiplier used for Take Profit 1 (first partial target).
    # tp1_distance = ATR × ATR_TP1_MULTIPLIER
    ATR_TP1_MULTIPLIER: float = _env_float("ATR_TP1_MULTIPLIER", 2.0)

    # ATR multiplier used for Take Profit 2 (final target / liquidity level).
    # tp2_distance = ATR × ATR_TP2_MULTIPLIER
    ATR_TP2_MULTIPLIER: float = _env_float("ATR_TP2_MULTIPLIER", 4.0)

    # =========================================================================
    # EMA (Exponential Moving Average) SETTINGS
    # =========================================================================

    # Fast EMA period — reacts quickly to recent price changes.
    # Used for: short-term bias, golden/death cross detection.
    EMA_FAST_PERIOD: int = _env_int("EMA_FAST_PERIOD", 20)

    # Slow EMA period — filters out short-term noise.
    # Used for: trend confirmation against the fast EMA.
    EMA_SLOW_PERIOD: int = _env_int("EMA_SLOW_PERIOD", 50)

    # Macro EMA period — defines the long-term trend direction.
    # Signals are only taken in the direction of price relative to this EMA.
    EMA_MACRO_PERIOD: int = _env_int("EMA_MACRO_PERIOD", 200)

    # Minimum EMA slope angle (degrees, approximated) for the trend to be
    # considered genuinely directional vs. flat / sideways.
    # Computed as: arctan(Δema / lookback_bars) × (180 / π)
    EMA_MIN_SLOPE_ANGLE: float = _env_float("EMA_MIN_SLOPE_ANGLE", 5.0)

    # Number of bars to look back when computing EMA slope angle.
    EMA_SLOPE_LOOKBACK: int = _env_int("EMA_SLOPE_LOOKBACK", 5)

    # =========================================================================
    # ADX SETTINGS  (for regime classification)
    # =========================================================================

    # Period for the ADX (Average Directional Index) calculation.
    ADX_PERIOD: int = _env_int("ADX_PERIOD", 14)

    # ADX threshold above which the market is classified as TRENDING.
    # ADX < threshold → RANGING; ADX ≥ threshold → TRENDING_UP or TRENDING_DOWN.
    ADX_TREND_THRESHOLD: float = _env_float("ADX_TREND_THRESHOLD", 25.0)

    # =========================================================================
    # NEWS / ECONOMIC CALENDAR FILTER
    # =========================================================================

    # Generic API key field (legacy — keep for backward compatibility).
    # Prefer the provider-specific keys below.
    NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")

    # ── Provider API keys ─────────────────────────────────────────────────────

    # Finnhub free API key.  Sign up at https://finnhub.io (free tier).
    # Used for: real-time company news and economic calendar events.
    # Quota: 60 calls/min on free tier.
    FINNHUB_API_KEY: str = os.getenv("FINNHUB_API_KEY", "")

    # NewsAPI.org API key.  Sign up at https://newsapi.org (free tier).
    # Used for: keyword-based headline scanning (Fed, CPI, FOMC, war, etc.)
    # Quota: 100 calls/day on free tier.
    NEWSAPI_KEY: str = os.getenv("NEWSAPI_KEY", "")

    # ── Blackout windows ─────────────────────────────────────────────────────

    # Minutes BEFORE a scheduled high-impact event during which signals block.
    NEWS_BLACKOUT_BEFORE_MIN: int = _env_int("NEWS_BLACKOUT_BEFORE_MIN", 30)

    # Minutes AFTER a high-impact event during which signals block.
    NEWS_BLACKOUT_AFTER_MIN: int = _env_int("NEWS_BLACKOUT_AFTER_MIN", 15)

    # ── News risk scoring ─────────────────────────────────────────────────────

    # How long (minutes) to cache fetched news data from providers.
    # Avoids hammering the API on every loop cycle.
    NEWS_CACHE_TTL_MIN: int = _env_int("NEWS_CACHE_TTL_MIN", 10)

    # Lookahead window (minutes) for the Finnhub economic calendar.
    # Events scheduled within this window are evaluated for risk.
    NEWS_LOOKAHEAD_MIN: int = _env_int("NEWS_LOOKAHEAD_MIN", 120)

    # Minimum risk score [0–100] to classify overall news risk as "HIGH".
    # Signals are BLOCKED when news_risk == "HIGH".
    NEWS_HIGH_RISK_THRESHOLD: int = _env_int("NEWS_HIGH_RISK_THRESHOLD", 70)

    # Minimum risk score to classify as "MEDIUM" (confidence penalty applied).
    NEWS_MEDIUM_RISK_THRESHOLD: int = _env_int("NEWS_MEDIUM_RISK_THRESHOLD", 35)

    # Comma-separated currencies whose events are tracked.
    # Gold reacts primarily to USD and, to a lesser extent, EUR/GBP.
    NEWS_TRACKED_CURRENCIES: str = os.getenv("NEWS_TRACKED_CURRENCIES", "USD,XAU")


    # =========================================================================
    # SMART MONEY CONCEPTS — DETECTION PARAMETERS
    # =========================================================================

    # Minimum number of consecutive candles required to confirm an impulsive
    # move away from an Order Block (strength qualifier).
    OB_MIN_IMPULSE_CANDLES: int = _env_int("OB_MIN_IMPULSE_CANDLES", 3)

    # Minimum body-to-range ratio for a candle to qualify as a valid Order Block.
    # body_ratio = candle_body / (high − low)
    OB_MIN_BODY_RATIO: float = _env_float("OB_MIN_BODY_RATIO", 0.3)

    # Maximum number of bars to look back when searching for Order Blocks.
    OB_LOOKBACK_BARS: int = _env_int("OB_LOOKBACK_BARS", 300)

    # Minimum size of a Fair Value Gap expressed as a fraction of ATR.
    # fvg_size_points ≥ FVG_MIN_ATR_FRACTION × ATR
    FVG_MIN_ATR_FRACTION: float = _env_float("FVG_MIN_ATR_FRACTION", 0.2)

    # Maximum number of bars to look back when searching for Fair Value Gaps.
    FVG_LOOKBACK_BARS: int = _env_int("FVG_LOOKBACK_BARS", 300)

    # =========================================================================
    # SCHEDULING
    # =========================================================================

    # How often (in seconds) the main analysis loop wakes up to check for new
    # bar closes and potential setups.
    # 900 seconds = 15 minutes (aligns with M15 bar close).
    LOOP_INTERVAL_SECONDS: int = _env_int("LOOP_INTERVAL_SECONDS", 900)

    # UTC time (HH:MM) at which the daily summary is prepared.
    DAILY_SUMMARY_TIME_UTC: str = os.getenv("DAILY_SUMMARY_TIME_UTC", "21:00")

    # =========================================================================
    # LOGGING
    # =========================================================================

    # Logging verbosity: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Path to the rotating log file (relative to the project root).
    LOG_FILE: str = os.getenv("LOG_FILE", "logs/gold_bot.log")

    # Number of days to retain old log files before deletion.
    LOG_RETENTION_DAYS: int = _env_int("LOG_RETENTION_DAYS", 30)

    # =========================================================================
    # BACKTESTING
    # =========================================================================

    # Directory where historical CSV data files are stored.
    # Expected filename format: <SYMBOL>_<TIMEFRAME>.csv  e.g. XAUUSD_H1.csv
    BACKTEST_DATA_DIR: str = os.getenv("BACKTEST_DATA_DIR", "data/historical")

    # Directory where backtest result CSVs and summary reports are written.
    BACKTEST_OUTPUT_DIR: str = os.getenv("BACKTEST_OUTPUT_DIR", "backtesting/results")

    # Starting virtual account balance for backtests (USD).
    BACKTEST_INITIAL_BALANCE: float = _env_float("BACKTEST_INITIAL_BALANCE", 10_000.0)

    # =========================================================================
    # AI PROBABILITY MODEL
    # =========================================================================

    # Enable or disable the AI probability filter in the signal pipeline.
    # When True, every signal must also pass the AI probability gate before dispatch.
    AI_USE_PROBABILITY_FILTER: bool = os.getenv("AI_USE_PROBABILITY_FILTER", "false").lower() == "true"

    # Minimum win-probability score required to approve a signal.
    # Signals scoring below this are rejected even if SMC confidence passes.
    # Range: 0.0 – 1.0  |  Recommended: 0.55 – 0.70
    AI_PROBABILITY_THRESHOLD: float = _env_float("AI_PROBABILITY_THRESHOLD", 0.60)

    # Path (relative to project root) where the trained model is saved / loaded.
    # Format: joblib pickle (.pkl or .joblib)
    AI_MODEL_PATH: str = os.getenv("AI_MODEL_PATH", "strategy/models/rf_probability.joblib")

    # Minimum number of historical trade records needed before training is allowed.
    # Training on too few samples produces unreliable models.
    AI_MIN_TRAIN_SAMPLES: int = _env_int("AI_MIN_TRAIN_SAMPLES", 50)

    # Number of trees in the RandomForestClassifier.
    # More trees → better accuracy, slower training.  Recommended: 100 – 500.
    AI_N_ESTIMATORS: int = _env_int("AI_N_ESTIMATORS", 200)

    # Random seed for reproducible model training.
    AI_RANDOM_SEED: int = _env_int("AI_RANDOM_SEED", 42)

    # =========================================================================
    # STRATEGY OPTIMIZER
    # =========================================================================

    # Enable auto-optimization.  When True, the optimizer will re-evaluate and
    # potentially adjust parameters every OPTIMIZER_EVAL_EVERY trades.
    OPTIMIZER_ENABLED: bool = os.getenv("OPTIMIZER_ENABLED", "false").lower() == "true"

    # Number of resolved trades between optimizer evaluations.
    # Smaller = more responsive but noisier.  Recommended: 50–200.
    OPTIMIZER_EVAL_EVERY: int = _env_int("OPTIMIZER_EVAL_EVERY", 100)

    # CSV file where live trade history is stored for optimizer analysis.
    # Path is relative to the project root.
    OPTIMIZER_HISTORY_PATH: str = os.getenv(
        "OPTIMIZER_HISTORY_PATH", "strategy/optimizer_data/trade_history.csv"
    )

    # Directory where optimizer run reports and param snapshots are saved.
    OPTIMIZER_OUTPUT_DIR: str = os.getenv(
        "OPTIMIZER_OUTPUT_DIR", "strategy/optimizer_data/reports"
    )

    # Minimum number of trades in history before the first optimization run.
    # Prevents optimizing on statistically insignificant samples.
    OPTIMIZER_MIN_HISTORY: int = _env_int("OPTIMIZER_MIN_HISTORY", 50)

    # Hard ceiling on account risk per trade that the optimizer may recommend.
    # Even if grid search finds a "better" configuration, RISK_PER_TRADE_PCT
    # will never be raised above this limit.
    OPTIMIZER_MAX_RISK_PCT: float = _env_float("OPTIMIZER_MAX_RISK_PCT", 2.0)

    # Minimum acceptable win rate (0–1) for a parameter set to be adopted.
    # Parameter sets producing a win rate below this floor are rejected.
    OPTIMIZER_MIN_WIN_RATE: float = _env_float("OPTIMIZER_MIN_WIN_RATE", 0.40)

    # Minimum acceptable profit factor for a parameter set to be adopted.
    OPTIMIZER_MIN_PROFIT_FACTOR: float = _env_float("OPTIMIZER_MIN_PROFIT_FACTOR", 1.20)

    # Maximum drawdown (in R) allowed for a candidate parameter set.
    OPTIMIZER_MAX_DRAWDOWN_R: float = _env_float("OPTIMIZER_MAX_DRAWDOWN_R", 10.0)

    # Grid search ranges — ATR SL multiplier
    # Format: "min,max,step"  e.g. "1.0,2.5,0.5" → [1.0, 1.5, 2.0, 2.5]
    OPTIMIZER_ATR_SL_RANGE: str = os.getenv("OPTIMIZER_ATR_SL_RANGE", "1.0,2.5,0.5")

    # Grid search ranges — ATR TP1 multiplier
    OPTIMIZER_ATR_TP1_RANGE: str = os.getenv("OPTIMIZER_ATR_TP1_RANGE", "1.5,3.0,0.5")

    # Grid search ranges — ATR TP2 multiplier
    OPTIMIZER_ATR_TP2_RANGE: str = os.getenv("OPTIMIZER_ATR_TP2_RANGE", "3.0,6.0,1.0")

    # Grid search ranges — confidence threshold
    OPTIMIZER_CONF_RANGE: str = os.getenv("OPTIMIZER_CONF_RANGE", "0.55,0.80,0.05")

    # Grid search ranges — EMA slow period
    OPTIMIZER_EMA_SLOW_RANGE: str = os.getenv("OPTIMIZER_EMA_SLOW_RANGE", "30,70,10")

    # Scoring function used to rank parameter sets during grid search.
    # Options: "expectancy" | "profit_factor" | "sharpe" | "calmar"
    OPTIMIZER_SCORE_METRIC: str = os.getenv("OPTIMIZER_SCORE_METRIC", "expectancy")

    # =========================================================================
    # VALIDATION
    # =========================================================================

    def validate(self) -> None:
        """
        Raise ValueError with a clear message if any critical settings are
        missing or misconfigured.

        Call this once at bot startup (before the main loop begins) so that
        missing credentials surface immediately instead of at the first send.
        """
        errors: list[str] = []

        if not self.DISCORD_WEBHOOK_URL:
            errors.append("DISCORD_WEBHOOK_URL is not set.")

        if self.RISK_PER_TRADE_PCT <= 0 or self.RISK_PER_TRADE_PCT > 10:
            errors.append(
                f"RISK_PER_TRADE_PCT must be between 0 and 10. Got: {self.RISK_PER_TRADE_PCT}"
            )
        if self.CONFIDENCE_THRESHOLD < 0.0 or self.CONFIDENCE_THRESHOLD > 1.0:
            errors.append(
                f"CONFIDENCE_THRESHOLD must be between 0.0 and 1.0. Got: {self.CONFIDENCE_THRESHOLD}"
            )
        if self.ATR_MIN_POINTS >= self.ATR_MAX_POINTS:
            errors.append(
                f"ATR_MIN_POINTS ({self.ATR_MIN_POINTS}) must be less than "
                f"ATR_MAX_POINTS ({self.ATR_MAX_POINTS})."
            )
        if self.EMA_FAST_PERIOD >= self.EMA_SLOW_PERIOD:
            errors.append(
                f"EMA_FAST_PERIOD ({self.EMA_FAST_PERIOD}) must be less than "
                f"EMA_SLOW_PERIOD ({self.EMA_SLOW_PERIOD})."
            )
        if self.EMA_SLOW_PERIOD >= self.EMA_MACRO_PERIOD:
            errors.append(
                f"EMA_SLOW_PERIOD ({self.EMA_SLOW_PERIOD}) must be less than "
                f"EMA_MACRO_PERIOD ({self.EMA_MACRO_PERIOD})."
            )
        if self.MIN_RR_RATIO < 1.0:
            errors.append(
                f"MIN_RR_RATIO ({self.MIN_RR_RATIO}) must be at least 1.0."
            )

        if errors:
            bullet_list = "\n  • ".join(errors)
            raise ValueError(
                f"[Settings] Configuration errors detected:\n  • {bullet_list}"
            )

    def __repr__(self) -> str:
        return (
            f"Settings("
            f"symbol={self.SYMBOL!r}, "
            f"signal_tf={self.SIGNAL_TIMEFRAME!r}, "
            f"risk_pct={self.RISK_PER_TRADE_PCT}, "
            f"confidence={self.CONFIDENCE_THRESHOLD}, "
            f"sessions={self.ACTIVE_SESSIONS}"
            f")"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Singleton accessor
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the application-wide Settings singleton.

    The first call constructs and validates the object; subsequent calls return
    the cached instance without re-reading the environment.

    Example
    -------
        from config.settings import get_settings

        cfg = get_settings()
        print(cfg.SYMBOL)
    """
    cfg = Settings()
    cfg.validate()
    return cfg
