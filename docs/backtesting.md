# Backtesting Guide

How to validate the strategy on historical data before running live.

---

## Overview

The `BacktestEngine` in `backtesting/backtest_engine.py` simulates the full live pipeline bar by bar on historical OHLCV data. It uses the **same filter stack, risk rules, and signal generation logic** as the live bot — there is no look-ahead bias by architecture.

---

## Data Requirements

### CSV format

Place historical data CSVs in `BACKTEST_DATA_DIR` (default: `data/historical/`).

Expected columns:

```
time,open,high,low,close,tick_volume
2024-01-02 07:00:00,2063.45,2065.20,2062.10,2064.80,1250
2024-01-02 08:00:00,2064.80,2070.55,2063.90,2068.30,1890
...
```

Expected filenames:
- `XAUUSD_H4.csv` — trend timeframe
- `XAUUSD_H1.csv` — signal timeframe
- `XAUUSD_M15.csv` — entry timeframe

### Exporting data from MT5

In MT5 → Tools → History Center → XAUUSD → select timeframe → Export.
Alternatively use `mt5_data.py`'s `get_bars()` to export via the Python bridge.

---

## Running a Backtest

```bash
python main.py --backtest
```

Or programmatically:

```python
from backtesting.backtest_engine import BacktestEngine
from config.settings import get_settings
from pathlib import Path

cfg = get_settings()
engine = BacktestEngine(cfg, data_path=Path(cfg.BACKTEST_DATA_DIR))

result = engine.run(start_date="2024-01-01", end_date="2024-12-31")

print(f"Win Rate:       {result.win_rate:.1%}")
print(f"Profit Factor:  {result.profit_factor:.2f}")
print(f"Max Drawdown:   {result.max_drawdown_pct:.1%}")
print(f"Signals/Day:    {result.signals_per_day:.1f}")

engine.export_results(result, output_dir=Path(cfg.BACKTEST_OUTPUT_DIR))
```

---

## Output Files

After a run, results are written to `BACKTEST_OUTPUT_DIR`:

| File | Description |
|------|-------------|
| `trade_log.csv` | Every signal with entry, SL, TP1, TP2, outcome, R gained |
| `summary.txt` | Human-readable performance report |
| `equity_curve.csv` | Cumulative balance at each trade |

---

## Performance Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Win Rate** | % of signals that hit TP1 | ≥ 50% |
| **Profit Factor** | Gross profit / Gross loss | ≥ 1.5 |
| **Average R:R** | Average reward per trade (in R multiples) | ≥ 1.5R |
| **Max Consecutive Losses** | Longest losing streak | ≤ 5 |
| **Max Drawdown** | Peak-to-trough equity decline | ≤ 15% |
| **Signals per Day** | Average daily signal count | 2–4 |

---

## Walk-Forward Validation

To avoid overfitting, use walk-forward splits:

1. Train period: Jan–Sep (optimize `CONFIDENCE_THRESHOLD`, `ATR_SL_MULTIPLIER`)
2. Test period:  Oct–Dec (hold out, never optimized on)
3. Compare statistics — if test period is significantly worse, the strategy is overfitted

---

## Parameter Sensitivity Analysis

Key parameters to stress-test:

| Parameter | Range to test | Impact |
|-----------|--------------|--------|
| `CONFIDENCE_THRESHOLD` | 0.55 → 0.80 | Signal volume vs quality trade-off |
| `MIN_RR_RATIO` | 1.5 → 3.0 | Win rate vs average win size |
| `ATR_SL_MULTIPLIER` | 1.0 → 2.5 | Stop size (wider = fewer stops hit, larger position risk) |
| `OB_MIN_IMPULSE_CANDLES` | 2 → 5 | Order Block strictness |

---

*Last updated: April 2026 | AntiGravity Trading Bot v1.0*
