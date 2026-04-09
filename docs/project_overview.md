# AntiGravity Gold Trading Bot — Project Overview

> A modular, signal-only Python bot for XAUUSD (Gold) intraday trading.
> Signals are published to Telegram. No automated trade execution.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Signal Pipeline](#signal-pipeline)
3. [Module Reference](#module-reference)
4. [Data Flow Diagram](#data-flow-diagram)
5. [Configuration System](#configuration-system)
6. [Filter System](#filter-system)
7. [Risk Management](#risk-management)
8. [Extending the Bot](#extending-the-bot)

---

## Architecture Overview

The bot is structured as a **layered pipeline** where each layer has a single, clearly defined responsibility:

```
┌──────────────────────────────────────────────────────────────┐
│                        main.py                               │
│              Async event loop + CLI entry point              │
└────────────────────────────┬─────────────────────────────────┘
                             │ orchestrates
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
   [Data Layer]       [Strategy Layer]     [Filter Layer]
   mt5_data.py        market_regime.py     news_filter.py
   news_data.py       smart_money_         volatility_
                      strategy.py          filter.py
                      trend_detection.py   session_filter.py
         │                   │
         └─────────┬─────────┘
                   ▼
           [Signal Layer]
         signal_generator.py
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
   [Risk Layer]      [Notification Layer]
   risk_manager.py   telegram_bot.py
   position_sizing.py
```

Every layer communicates through **typed dataclasses** (`TradeIdea`, `TradingSignal`) — no shared global state.

---

## Signal Pipeline

A full cycle runs every **15 minutes** (configurable via `LOOP_INTERVAL_SECONDS`):

| Step | Module | Action |
|------|--------|--------|
| 1 | `mt5_data.py` | Fetch H4, H1, M15 OHLCV bars from MT5 |
| 2 | `session_filter.py` | Reject if outside London / NY session |
| 3 | `news_filter.py` | Reject if within news blackout window |
| 4 | `volatility_filter.py` | Reject if ATR outside min–max bounds |
| 5 | `risk_manager.py` | Reject if daily cap or loss limit hit |
| 6 | `market_regime.py` | Classify trend: TRENDING_UP / DOWN / RANGING |
| 7 | `trend_detection.py` | Compute EMA alignment and slope |
| 8 | `smart_money_strategy.py` | Detect OB, FVG, BOS, liquidity pools |
| 9 | `signal_generator.py` | Compute entry, SL, TP1, TP2, lot size, R:R |
| 10 | `telegram_bot.py` | Publish formatted signal to Telegram |
| 11 | `risk_manager.py` | Increment daily signal counter |

---

## Module Reference

### `config/settings.py`
Single source of truth for all configuration. Reads from `.env` on startup. Contains full documentation for every parameter as inline comments. See [configuration.md](configuration.md) for a full parameter reference table.

### `data/mt5_data.py`
Wraps the MetaTrader5 Python bridge. Provides:
- `connect()` / `disconnect()` lifecycle
- `get_bars(symbol, timeframe, count)` → `pd.DataFrame`
- `get_current_price(symbol)` → `dict`

### `data/news_data.py`
Economic calendar provider. Fetches high-impact USD/XAU events and determines whether the current time falls in a blackout window.

### `strategy/market_regime.py`
Classifies the market into `TRENDING_UP`, `TRENDING_DOWN`, or `RANGING` using ADX and price swing structure (HH/HL vs LL/LH).

### `strategy/smart_money_strategy.py`
Core SMC engine. Identifies:
- **Order Blocks** — last opposing candle before an impulse
- **Fair Value Gaps** — three-candle price imbalances
- **Break of Structure** — structural high/low breaches
- **Liquidity Pools** — equal highs/lows, previous session extremes

Outputs a `TradeIdea` dataclass with a `confluence_score`.

### `strategy/trend_detection.py`
EMA ribbon analysis (EMA 20/50/200). Provides golden/death cross detection, alignment checks, and slope-angle computation.

### `risk/position_sizing.py`
Fixed-fractional lot calculator.
`lot = (balance × risk_pct) / (stop_loss_points × pip_value_per_lot)`

### `risk/risk_manager.py`
Persistent daily gate. Tracks signal count and estimated P&L. State survives bot restarts via a JSON file. Resets at UTC midnight.

### `signals/signal_generator.py`
Assembles a `TradingSignal` from a `TradeIdea`. Validates R:R ratio ≥ `MIN_RR_RATIO` and confluence ≥ `CONFIDENCE_THRESHOLD` before publishing.

### `notifications/telegram_bot.py`
Async Telegram dispatcher (python-telegram-bot v20+). Includes exponential back-off retry logic and supports daily summary messages.

### `filters/news_filter.py`
Blocks signals within `NEWS_BLACKOUT_BEFORE_MIN` minutes before and `NEWS_BLACKOUT_AFTER_MIN` minutes after any high-impact USD/XAU event.

### `filters/volatility_filter.py`
Computes ATR on the signal timeframe. Rejects setups when ATR is below `ATR_MIN_POINTS` (too quiet) or above `ATR_MAX_POINTS` (too erratic).

### `filters/session_filter.py`
Enforces the `ACTIVE_SESSIONS` whitelist. Uses UTC time; session window boundaries are configurable via `.env`.

### `backtesting/backtest_engine.py`
Bar-by-bar historical simulator. Uses the **same** filter stack as live mode (no look-ahead bias by design). Produces trade logs, win rate, profit factor, and drawdown stats.

### `utils/logger.py`
Centralised logger factory. Creates rotating daily log files with a consistent format: `timestamp | level | module | message`.

### `utils/helpers.py`
Shared utilities: `retry()` decorator, `round_price()`, `safe_divide()`, `utc_now()`, DataFrame column validators.

---

## Data Flow Diagram

```
MT5 Terminal ──OHLCV──► mt5_data.py ──DataFrame──►┐
                                                    │
Economic Calendar ──events──► news_data.py         │
                                                    │
                         ┌──────────────────────────┘
                         │
                         ▼
               market_regime.py  ──RegimeLabel──►┐
               trend_detection.py ──EMA metrics──►│
                                                  │
                         smart_money_strategy.py ◄┘
                                   │
                               TradeIdea
                                   │
                 ┌─────────────────▼──────────────┐
                 │         Filter Layer            │
                 │  session · news · volatility    │
                 └─────────────────┬──────────────┘
                                   │ (pass)
                 ┌─────────────────▼──────────────┐
                 │         Risk Gate               │
                 │  risk_manager · position_sizer  │
                 └─────────────────┬──────────────┘
                                   │ (pass)
                         signal_generator.py
                                   │
                             TradingSignal
                                   │
                         telegram_bot.py ──► 📲 Telegram
```

---

## Configuration System

All parameters are controlled through the `.env` file. The `Settings` class in `config/settings.py`:

1. Reads every variable from the environment (falls back to safe defaults)
2. Validates critical values at startup (raises `ValueError` on misconfiguration)
3. Is exposed as a **singleton** via `get_settings()` — all modules share one instance

See [configuration.md](configuration.md) for the full parameter reference.

---

## Filter System

All filters follow the same interface:

```python
def is_allowed(self, ...) -> Tuple[bool, str]:
    # Returns (True, "") if the check passes
    # Returns (False, "reason message") if blocked
```

This makes filter results **loggable** and **auditable** — every rejection is recorded with a reason.

---

## Risk Management

Two complementary layers:

| Layer | Where | What it controls |
|-------|-------|-----------------|
| **Per-trade** | `position_sizing.py` | Lot size (capped by `RISK_PER_TRADE_PCT`) |
| **Daily** | `risk_manager.py` | Signal cap (`MAX_SIGNALS_PER_DAY`) + daily loss limit (`MAX_DAILY_LOSS_PCT`) |

The daily state is written to `data/risk_state.json` so a bot restart mid-day doesn't reset the counters.

---

## Extending the Bot

### Add a new filter
1. Create `filters/my_filter.py` implementing `is_allowed() -> Tuple[bool, str]`
2. Add it to `filters/__init__.py`
3. Call it in `signal_generator.py`'s validation chain

### Add a new indicator
1. Add it to `strategy/trend_detection.py` or create a new file in `strategy/`
2. Incorporate its output into `SmartMoneyStrategy._score_confluence()`

### Add an ML signal scorer
1. Train a model using `backtesting/` output CSVs
2. Save with `joblib.dump()`
3. Load in `signal_generator.py` and use its probability as a `confluence_score` override

---

*Last updated: April 2026 | AntiGravity Trading Bot v1.0*
