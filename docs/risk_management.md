# Risk Management Documentation

How the AntiGravity Gold Bot manages and limits financial risk.

---

## Overview

The bot implements **two independent risk layers** that work together:

1. **Per-trade layer** (`position_sizing.py`) — controls how much is risked on each individual trade
2. **Daily layer** (`risk_manager.py`) — controls the overall day's exposure and stops the bot if limits are breached

No trade is ever executed automatically — signals are Telegram-only — but accurate sizing makes it easy for traders to place the exact recommended lot size.

---

## Per-Trade Risk: Fixed-Fractional Position Sizing

### Formula

```
risk_amount = account_balance × (RISK_PER_TRADE_PCT / 100)
lot_size    = risk_amount / (stop_loss_points × pip_value_per_lot)
```

### XAUUSD pip value reference

| Lot Size | Value per point | Value per 10 points |
|----------|----------------|---------------------|
| 1.00 lot | $10.00 | $100.00 |
| 0.10 lot | $1.00  | $10.00  |
| 0.01 lot | $0.10  | $1.00   |

> For XAUUSD: 1 full lot = 100 troy oz; 1 point = $0.01 price change = $1 per 0.01 lot.

### Example calculation

```
Account balance:       $10,000
Risk per trade:        1% → $100
Stop loss distance:    20 points (e.g., entry at 2350.00, SL at 2330.00)
Pip value (1.0 lot):  $10 per point

Lot size = $100 / (20 × $10) = 0.50 lots
```

### Lot size limits

- Maximum: `MAX_LOT_SIZE` (default `5.0`) — absolute safety ceiling
- Minimum: `MIN_LOT_SIZE` (default `0.01`) — broker floor
- Result is rounded to 2 decimal places

---

## Daily Risk Gate

The `RiskManager` class maintains a persistent state file (`data/risk_state.json`) that resets at **UTC midnight**.

### Gate conditions — all must pass

| Check | Condition | Setting |
|-------|-----------|---------|
| Signal count | `signals_today < MAX_SIGNALS_PER_DAY` | `MAX_SIGNALS_PER_DAY = 4` |
| Daily loss | `daily_loss_pct < MAX_DAILY_LOSS_PCT` | `MAX_DAILY_LOSS_PCT = 3.0` |

If either condition fails, `is_trading_allowed()` returns `False` and no more signals are published for the rest of the day.

### State persistence

```json
{
  "date": "2026-04-08",
  "signals_today": 2,
  "daily_loss_pct": 1.2
}
```

The file is written after every signal and after every P&L update. A bot restart mid-day correctly resumes from the saved state.

---

## Reward-to-Risk Minimum

Every signal must satisfy:

```
R:R ratio = |TP1 - entry| / |entry - SL|  ≥  MIN_RR_RATIO (default: 2.0)
```

Signals that don't meet this threshold are **silently discarded** before hitting the Telegram dispatcher. A debug log entry is written explaining the rejection.

---

## Risk Pyramid (visual)

```
            ┌─────────────────────────────────────────────┐
            │    SIGNAL REJECTED (pre-publication)         │
            │  • R:R < MIN_RR_RATIO                        │
            │  • Confidence < CONFIDENCE_THRESHOLD          │
            └─────────────────────────────────────────────┘
                                   │
            ┌──────────────────────▼──────────────────────┐
            │    DAILY GATE (risk_manager.py)               │
            │  • signals_today ≥ MAX_SIGNALS_PER_DAY       │
            │  • daily_loss_pct ≥ MAX_DAILY_LOSS_PCT        │
            └─────────────────────────────────────────────┘
                                   │
            ┌──────────────────────▼──────────────────────┐
            │    FILTER LAYER                              │
            │  • Session out of ACTIVE_SESSIONS            │
            │  • News blackout window active               │
            │  • ATR outside MIN/MAX bounds                │
            └─────────────────────────────────────────────┘
                                   │
            ┌──────────────────────▼──────────────────────┐
            │    PER-TRADE SIZING (position_sizing.py)      │
            │  • lot = balance × risk% / (SL × pip_val)   │
            │  • clamped to MIN/MAX_LOT_SIZE               │
            └─────────────────────────────────────────────┘
                                   │
                          📲 Published to Telegram
```

---

## Recommended Settings by Account Size

| Account | `RISK_PER_TRADE_PCT` | `MAX_DAILY_LOSS_PCT` | Expected daily risk |
|---------|---------------------|---------------------|---------------------|
| $1,000  | 0.5% | 2% | $5–$20 |
| $5,000  | 1.0% | 3% | $50–$150 |
| $10,000 | 1.0% | 3% | $100–$300 |
| $50,000 | 0.5% | 2% | $250–$750 |

---

*Last updated: April 2026 | AntiGravity Trading Bot v1.0*
