# Configuration Reference

Complete reference for all environment variables supported by `config/settings.py`.

Copy `.env.example` → `.env` and fill in your values before starting the bot.

---

## Telegram

| Variable | Type | Default | Description |
|---|---|---|---|
| `TELEGRAM_BOT_TOKEN` | string | *(required)* | HTTP API token from @BotFather |
| `TELEGRAM_CHAT_ID` | string | *(required)* | Target channel / group / user ID |
| `TELEGRAM_ADMIN_CHAT_ID` | string | `""` | Optional admin chat for error alerts |
| `TELEGRAM_TIMEOUT_SECONDS` | int | `20` | API request timeout |
| `TELEGRAM_MAX_RETRIES` | int | `3` | Retry attempts on failed sends |

---

## MetaTrader 5

| Variable | Type | Default | Description |
|---|---|---|---|
| `MT5_LOGIN` | int | `0` | MT5 account number |
| `MT5_PASSWORD` | string | `""` | MT5 account password |
| `MT5_SERVER` | string | `""` | Broker server name (e.g. `ICMarkets-Demo`) |
| `MT5_PATH` | string | `""` | Full path to `terminal64.exe` |
| `MT5_TIMEOUT_SECONDS` | int | `30` | Connection timeout |
| `MT5_MAX_RECONNECT_ATTEMPTS` | int | `5` | Auto-reconnect attempts |

---

## Symbol

| Variable | Type | Default | Description |
|---|---|---|---|
| `SYMBOL` | string | `XAUUSD` | Traded instrument (must match broker exactly) |
| `SYMBOL_POINT` | float | `0.01` | Minimum price movement (1 point) |
| `SYMBOL_DIGITS` | int | `2` | Decimal places for price display |

---

## Timeframes

| Variable | Type | Default | Description |
|---|---|---|---|
| `TREND_TIMEFRAME` | string | `H4` | Macro trend context |
| `SIGNAL_TIMEFRAME` | string | `H1` | SMC setup detection |
| `ENTRY_TIMEFRAME` | string | `M15` | Entry refinement |
| `BARS_TO_FETCH` | int | `600` | Historical bars per timeframe |

Valid timeframe values: `M1` `M5` `M15` `M30` `H1` `H4` `D1` `W1` `MN1`

---

## Account & Risk

| Variable | Type | Default | Description |
|---|---|---|---|
| `ACCOUNT_BALANCE` | float | `10000.0` | Account balance in USD |
| `RISK_PER_TRADE_PCT` | float | `1.0` | % of balance risked per trade |
| `MAX_DAILY_LOSS_PCT` | float | `3.0` | Daily loss limit before bot stops |
| `MAX_SIGNALS_PER_DAY` | int | `4` | Maximum signals per calendar day |
| `MIN_RR_RATIO` | float | `2.0` | Minimum reward-to-risk ratio |
| `MAX_LOT_SIZE` | float | `5.0` | Safety ceiling on lot size |
| `MIN_LOT_SIZE` | float | `0.01` | Broker minimum lot size |

> **Recommended risk**: `RISK_PER_TRADE_PCT` between `0.5` and `2.0` for conservative intraday trading.

---

## Signal Confidence

| Variable | Type | Default | Description |
|---|---|---|---|
| `CONFIDENCE_THRESHOLD` | float | `0.68` | Minimum confluence score (0.0 – 1.0) |

The confluence score is a weighted combination of:

| Factor | Weight |
|--------|--------|
| Market regime alignment | 25% |
| Order Block quality | 25% |
| Fair Value Gap overlap | 20% |
| BOS / ChoCH confirmation | 15% |
| Liquidity target quality | 15% |

> **Tuning guide**: `0.65` → more signals / lower win rate; `0.75` → fewer signals / higher win rate.

---

## Session Windows (UTC)

| Variable | Type | Default | Description |
|---|---|---|---|
| `LONDON_SESSION_START` | HH:MM | `07:00` | London open |
| `LONDON_SESSION_END` | HH:MM | `15:59` | London close |
| `NEW_YORK_SESSION_START` | HH:MM | `12:00` | New York open |
| `NEW_YORK_SESSION_END` | HH:MM | `20:59` | New York close |
| `ASIAN_SESSION_START` | HH:MM | `00:00` | Asian open |
| `ASIAN_SESSION_END` | HH:MM | `06:59` | Asian close |
| `OVERLAP_START` | HH:MM | `12:00` | London-NY overlap start |
| `OVERLAP_END` | HH:MM | `15:59` | London-NY overlap end |
| `ACTIVE_SESSIONS` | list | `london,new_york` | Comma-separated allowed sessions |

> **Note**: All times are in **UTC**. Adjust for DST manually or use a broker on UTC-fixed servers.

---

## ATR Settings

| Variable | Type | Default | Description |
|---|---|---|---|
| `ATR_PERIOD` | int | `14` | Bars for ATR calculation |
| `ATR_MIN_POINTS` | float | `5.0` | Minimum ATR to trade (too quiet below) |
| `ATR_MAX_POINTS` | float | `40.0` | Maximum ATR to trade (too erratic above) |
| `ATR_SL_MULTIPLIER` | float | `1.5` | Stop loss = ATR × this |
| `ATR_TP1_MULTIPLIER` | float | `2.0` | Take Profit 1 = ATR × this |
| `ATR_TP2_MULTIPLIER` | float | `4.0` | Take Profit 2 = ATR × this |

---

## EMA Settings

| Variable | Type | Default | Description |
|---|---|---|---|
| `EMA_FAST_PERIOD` | int | `20` | Fast EMA (short-term bias) |
| `EMA_SLOW_PERIOD` | int | `50` | Slow EMA (trend filter) |
| `EMA_MACRO_PERIOD` | int | `200` | Macro EMA (long-term trend) |
| `EMA_MIN_SLOPE_ANGLE` | float | `5.0` | Minimum slope angle (degrees) |
| `EMA_SLOPE_LOOKBACK` | int | `5` | Bars used for slope calculation |

> **Constraint**: `EMA_FAST_PERIOD` < `EMA_SLOW_PERIOD` < `EMA_MACRO_PERIOD` — the bot validates this at startup.

---

## ADX Settings

| Variable | Type | Default | Description |
|---|---|---|---|
| `ADX_PERIOD` | int | `14` | ADX calculation period |
| `ADX_TREND_THRESHOLD` | float | `25.0` | ADX ≥ this → market is trending |

---

## News Filter

| Variable | Type | Default | Description |
|---|---|---|---|
| `NEWS_API_KEY` | string | `""` | Economic calendar API key |
| `NEWS_BLACKOUT_BEFORE_MIN` | int | `30` | Block signals N min before high-impact news |
| `NEWS_BLACKOUT_AFTER_MIN` | int | `15` | Block signals N min after high-impact news |

---

## SMC Detection Parameters

| Variable | Type | Default | Description |
|---|---|---|---|
| `OB_MIN_IMPULSE_CANDLES` | int | `3` | Min candles in impulse after OB |
| `OB_MIN_BODY_RATIO` | float | `0.5` | Min candle body / range ratio for OB |
| `OB_LOOKBACK_BARS` | int | `50` | Bars to scan for Order Blocks |
| `FVG_MIN_ATR_FRACTION` | float | `0.3` | Minimum FVG size as fraction of ATR |
| `FVG_LOOKBACK_BARS` | int | `30` | Bars to scan for Fair Value Gaps |

---

## Scheduling

| Variable | Type | Default | Description |
|---|---|---|---|
| `LOOP_INTERVAL_SECONDS` | int | `900` | Main loop cadence (900 = 15 min) |
| `DAILY_SUMMARY_TIME_UTC` | HH:MM | `21:00` | Time to send daily summary message |

---

## Logging

| Variable | Type | Default | Description |
|---|---|---|---|
| `LOG_LEVEL` | string | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `LOG_FILE` | string | `logs/gold_bot.log` | Rotating log file path |
| `LOG_RETENTION_DAYS` | int | `30` | Days to keep old log files |

---

## Backtesting

| Variable | Type | Default | Description |
|---|---|---|---|
| `BACKTEST_DATA_DIR` | string | `data/historical` | Historical CSV data directory |
| `BACKTEST_OUTPUT_DIR` | string | `backtesting/results` | Results output directory |
| `BACKTEST_INITIAL_BALANCE` | float | `10000.0` | Starting virtual balance (USD) |

Expected CSV filename format: `XAUUSD_H1.csv`, `XAUUSD_H4.csv`, `XAUUSD_M15.csv`

---

*Last updated: April 2026 | JayBot Trading Bot v1.0*
