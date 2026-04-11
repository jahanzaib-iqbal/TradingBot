# Alpha — Gold (XAUUSD) Intraday Trading Signal Bot

> **Signal-only bot. No automated trade execution.**  
> All trade ideas are validated, scored, and delivered to a Telegram channel or group.

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)](https://python.org)
[![Telegram](https://img.shields.io/badge/Telegram-Bot%20API-26A5E4?logo=telegram)](https://core.telegram.org/bots)
[![MT5](https://img.shields.io/badge/MetaTrader5-5.0%2B-orange)](https://www.metaquotes.net)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Table of Contents

1. [Overview](#overview)
2. [Signal Pipeline](#signal-pipeline)
3. [Project Structure](#project-structure)
4. [Requirements](#requirements)
5. [Installation](#installation)
6. [MetaTrader 5 Setup](#metatrader-5-setup)
7. [Discord Webhook Setup](#discord-webhook-setup)
8. [Environment Variables](#environment-variables)
9. [Running the Bot](#running-the-bot)
10. [Backtesting](#backtesting)
11. [Deploying to Railway](#deploying-to-railway)
12. [Deploying to Render](#deploying-to-render)
13. [Strategy Overview](#strategy-overview)
14. [Performance Tracking & Reports](#performance-tracking--reports)
15. [Risk Management](#risk-management)
16. [Disclaimer](#disclaimer)

---

## Overview

**JayBot** is a modular Python trading signal bot for **XAUUSD (Gold)** intraday trading.

It analyses multi-timeframe price data using **Smart Money Concepts (SMC)** — Order Blocks, Fair Value Gaps, Liquidity Sweeps, and Break of Structure — then publishes **24/7 high-probability trade ideas** directly to Discord.

**Key design decisions:**
- ✅ Signal-only — no positions are ever opened
- ✅ Fully modular — every component is replaceable
- ✅ Dry-run mode — test locally without a live bot token
- ✅ Walk-forward backtesting built-in
- ✅ Confidence scoring (0–100%) on every signal

---

## Signal Pipeline

```
MT5 Data  ──  H4 (Trend) + H1 (Regime) + M15 (SMC Scan)
                       │
              ┌────────▼────────┐
              │  Session Filter │  London · New York · Overlap only
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │  News Filter    │  Block 30 min before / 15 min after events
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │ Volatility Gate │  ATR must be 5 – 40 pts (not dead / not spiking)
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │  Daily Cap Gate │  Max 4 signals / day enforced
              └────────┬────────┘
                       │
              ┌────────▼────────────────────────────────────────┐
              │  Smart Money Strategy (SMC Engine)               │
              │  Order Blocks · FVG · BOS/CHoCH · Liq Pools     │
              └────────┬────────────────────────────────────────┘
                       │
              ┌────────▼────────┐
              │ Confidence Score│  0–100%  (weighted multi-factor)
              └────────┬────────┘
                       │  ≥ 68% threshold
              ┌────────▼────────┐
              │  Risk Manager   │  Lot size · SL · TP1 · TP2 · R:R
              └────────┬────────┘
                       │
                       ▼
               ┌────────▼────────┐
               │  Trade Tracker  │  Async Database Storage + Live M5 Pricing
               └────────┬────────┘
                       │
                       ▼
                📲  Discord Webhook (Signals + Daily Reports)
```

---

## Project Structure

```
gold_trading_bot/
├── main.py                        # Async bot loop — entry point
├── requirements.txt
├── .env.example                   # Template for environment variables
│
├── config/
│   └── settings.py                # Centralised configuration (all .env keys)
│
├── data/
│   ├── mt5_data.py                # MT5 multi-timeframe OHLCV provider
│   └── news_data.py               # Economic calendar / news event provider
│
├── strategy/
│   ├── market_regime.py           # ATR + ADX regime classifier
│   ├── smart_money_strategy.py    # SMC engine (OB, FVG, BOS, CHoCH, Liq)
│   ├── trend_detection.py         # EMA 50/200 trend + strength scoring
│   └── liquidity_detection.py     # Liquidity sweep identification
│
├── filters/
│   ├── session_filter.py          # London / New York / Overlap time gates
│   ├── volatility_filter.py       # Wilder ATR classification + guard
│   └── news_filter.py             # High-impact event blackout windows
│
├── risk/
│   ├── position_sizing.py         # Fixed-fractional lot size calculator
│   └── risk_manager.py            # Daily cap + daily loss-limit gate
│
├── signals/
│   └── signal_generator.py        # TradeIdea → TradingSignal pipeline
│
├── notifications/
│   └── discord_notifier.py        # Discord webhook dispatcher
│
├── backtesting/
│   └── backtest_engine.py         # Bar-by-bar simulation engine
│
├── utils/
│   ├── logger.py                  # Rotating file + console logger
│   └── helpers.py                 # Price utils, retry decorator
│
└── docs/
    └── code_overview.md           # Developer reference
```

---

## Requirements

| Dependency | Minimum Version | Purpose |
|---|---|---|
| Python | **3.11** | f-strings, `tomllib`, `match` syntax |
| MetaTrader5 | 5.0.45 | OHLCV + tick data (Windows only) |
| pandas | 2.2.0 | DataFrames |
| numpy | 1.26.0 | Numerical ops |
| pandas-ta | 0.3.14b | ATR, ADX, EMA |
| scipy | 1.13.0 | Linear regression slope |

| APScheduler | 3.10.4 | Scheduled jobs |
| python-dotenv | 1.0.0 | `.env` loading |
| scikit-learn | 1.4.0 | Future ML scoring |

> **Note:** `MetaTrader5` is **Windows-only**. On Linux/macOS use `--no-mt5` (demo data mode) or run inside a Windows VM / VPS.

---

## Installation

### Step 1 — Clone the repository

```bash
git clone https://github.com/your-username/TradingBot.git
```

### Step 2 — Create a virtual environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3.11 -m venv .venv
source .venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **MetaTrader5 on Windows only.** If you are on Linux/macOS skip it:
> ```bash
> pip install -r requirements.txt --ignore-requires-python
> # or edit requirements.txt and remove the MetaTrader5 line
> ```

### Step 4 — Copy the environment file

```bash
cp .env.example .env
```

Fill in your credentials (see [Environment Variables](#environment-variables) below).

### Step 5 — Verify the installation

```bash
# Run all synthetic unit tests (no MT5 or Telegram required)
python -m signals.signal_generator_test
python -m backtesting.backtest_test
```

All tests should print **All 22 tests passed**.

---

## MetaTrader 5 Setup

### 1. Install MetaTrader 5

Download MT5 from your broker or from [metaquotes.net](https://www.metatrader5.com/en/download).

### 2. Enable Algorithmic Trading

Inside MT5:  
`Tools → Options → Expert Advisors → ✅ Allow algorithmic trading`

### 3. Add XAUUSD to Market Watch

`View → Market Watch → right-click → Show All` then find **XAUUSD** (or your broker's symbol variant like `XAUUSDm`, `GOLD`).

### 4. Find your broker's symbol name

```python
# Run once to list available symbols
import MetaTrader5 as mt5
mt5.initialize()
symbols = [s.name for s in mt5.symbols_get() if "XAU" in s.name or "GOLD" in s.name]
print(symbols)
mt5.shutdown()
```

Set this name as `SYMBOL=` in your `.env`.

### 5. Configure `.env` credentials

```env
MT5_LOGIN=12345678              # Your MT5 account number
MT5_PASSWORD=your_password      # MT5 account password
MT5_SERVER=ICMarkets-Demo       # Exact server name from MT5 login screen
MT5_PATH=C:\Program Files\MetaTrader 5\terminal64.exe
```

> 💡 The MT5 terminal **must be running and logged in** before starting the bot. The Python API wraps the desktop terminal — it cannot log in independently.

---

## Discord Webhook Setup

### 1. Open Discord

Open the Discord app or web browser and navigate to your server.

### 2. Go to Server Settings

Click on your server name in the top left corner and select **Server Settings**.

### 3. Integrations → Webhooks

Navigate to the **Integrations** tab on the left menu, then click on **Webhooks**.

### 4. Create webhook

Click the **New Webhook** button. Give it a name and select the channel where you want the bot to send signals.

### 5. Copy URL

Click **Copy Webhook URL** for the webhook you just created.

### 6. Paste in settings.py

Paste the URL into your `.env` file under the key `DISCORD_WEBHOOK_URL` (which is loaded in settings.py).

---

## Environment Variables

All configuration lives in `.env`. Copy `.env.example` and fill in your values.

### Discord

| Variable | Default | Description |
|---|---|---|
| `DISCORD_WEBHOOK_URL` | *(required)* | Paste your Discord Webhook URL here |

### MetaTrader 5

| Variable | Default | Description |
|---|---|---|
| `MT5_LOGIN` | *(required)* | Account login number |
| `MT5_PASSWORD` | *(required)* | Account password |
| `MT5_SERVER` | *(required)* | Broker server name |
| `MT5_PATH` | *(required)* | Path to `terminal64.exe` |
| `MT5_TIMEOUT_SECONDS` | `30` | Connection timeout |
| `MT5_MAX_RECONNECT_ATTEMPTS` | `5` | Auto-reconnect retries |

### Symbol & Timeframes

| Variable | Default | Description |
|---|---|---|
| `SYMBOL` | `XAUUSD` | Broker symbol name |
| `SYMBOL_POINT` | `0.01` | Point size (1 point = $0.01 for Gold) |
| `TREND_TIMEFRAME` | `H4` | Macro trend context |
| `SIGNAL_TIMEFRAME` | `H1` | SMC scan timeframe |
| `ENTRY_TIMEFRAME` | `M15` | Entry refinement timeframe |
| `BARS_TO_FETCH` | `600` | Historical bars per request |

### Account & Risk

| Variable | Default | Description |
|---|---|---|
| `ACCOUNT_BALANCE` | `10000.0` | Balance used when live value unavailable |
| `RISK_PER_TRADE_PCT` | `1.0` | % of balance risked per trade |
| `MAX_DAILY_LOSS_PCT` | `3.0` | Stop trading after this % drawdown |
| `MAX_SIGNALS_PER_DAY` | `4` | Hard cap on daily signal count |
| `MIN_RR_RATIO` | `2.0` | Minimum reward-to-risk ratio |

### Signal Quality

| Variable | Default | Description |
|---|---|---|
| `CONFIDENCE_THRESHOLD` | `0.68` | Minimum confidence score (0.0–1.0) |

### Session Windows (UTC)

| Variable | Default | Description |
|---|---|---|
| `LONDON_SESSION_START` | `07:00` | London open |
| `LONDON_SESSION_END` | `15:59` | London close |
| `NEW_YORK_SESSION_START` | `12:00` | New York open |
| `NEW_YORK_SESSION_END` | `20:59` | New York close |
| `ACTIVE_SESSIONS` | `london,new_york` | Comma-separated allowed sessions |

### ATR / Volatility

| Variable | Default | Description |
|---|---|---|
| `ATR_PERIOD` | `14` | Wilder ATR period |
| `ATR_MIN_POINTS` | `5.0` | Block if ATR below this (market too quiet) |
| `ATR_MAX_POINTS` | `40.0` | Block if ATR above this (news spike) |
| `ATR_SL_MULTIPLIER` | `1.5` | Stop loss = ATR × this |
| `ATR_TP1_MULTIPLIER` | `2.0` | TP1 = ATR × this |
| `ATR_TP2_MULTIPLIER` | `4.0` | TP2 = ATR × this |

### News Filter

| Variable | Default | Description |
|---|---|---|
| `NEWS_API_KEY` | `""` | API key for news provider (optional) |
| `NEWS_BLACKOUT_BEFORE_MIN` | `30` | Block signals N min before event |
| `NEWS_BLACKOUT_AFTER_MIN` | `15` | Block signals N min after event |

### Scheduling

| Variable | Default | Description |
|---|---|---|
| `LOOP_INTERVAL_SECONDS` | `900` | Bot scan interval (900 = 15 min) |
| `DAILY_SUMMARY_TIME_UTC` | `21:00` | When to send the daily summary |

### Logging

| Variable | Default | Description |
|---|---|---|
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `LOG_FILE` | `logs/gold_bot.log` | Log file path |
| `LOG_RETENTION_DAYS` | `30` | Days before log rotation |

---

## Running the Bot

### Live mode (real MT5 + Discord)

```bash
python main.py
```

### Dry-run (real MT5 data, no Discord dispatch)

Messages are printed to the console instead of being sent. Useful for validating signal quality before going live.

```bash
python main.py --dry-run
```

### Demo mode (no MT5 required)

Generates synthetic OHLCV data internally. Useful on Linux, macOS, or CI environments where MetaTrader5 cannot be installed.

```bash
python main.py --no-mt5 --dry-run
```

### Override loop interval

```bash
python main.py --interval 60       # Scan every 60 seconds instead of 900
```

### Clear active-signal lock on startup

The bot tracks which signal is currently "live" and prevents re-entry. Use this flag to reset that state without restarting:

```bash
python main.py --reset-signal
```

### All flags combined

```bash
python main.py --dry-run --no-mt5 --interval 60 --reset-signal
```

---

## Backtesting

### Prepare historical data

Export OHLCV bars from MT5 as CSV files:

1. In MT5: `File → Open Data Folder → history → <broker>` — export from the terminal  
   **or** use the built-in script to save via Python:

```python
from data.mt5_data import MT5DataProvider
from config.settings import Settings

cfg      = Settings()
provider = MT5DataProvider(cfg)
provider.connect_to_mt5()

df_h1  = provider.get_candles("XAUUSD", "H1",  count=5000)
df_m15 = provider.get_candles("XAUUSD", "M15", count=10000)

df_h1.to_csv("data/historical/XAUUSD_H1.csv",  index=False)
df_m15.to_csv("data/historical/XAUUSD_M15.csv", index=False)
provider.disconnect()
```

### Run a backtest

```python
from backtesting.backtest_engine import BacktestEngine
from config.settings import Settings

cfg    = Settings()
engine = BacktestEngine(cfg)

result = engine.run_from_csv(
    h1_csv  = "data/historical/XAUUSD_H1.csv",
    m15_csv = "data/historical/XAUUSD_M15.csv",
    start   = "2024-01-01",
    end     = "2024-06-30",
)

result.print_summary()
```

**Example output:**
```
════════════════════════════════════════════════════════
  BACKTEST RESULTS — XAUUSD H1
  2024-01-01  →  2024-06-30
════════════════════════════════════════════════════════
  Bars scanned:          3,390
  Signals generated:        89
  Trades resolved:          31
    Winners:                19  (TP1/TP2)
    Losers:                 10  (SL)
    Break-evens:             2
────────────────────────────────────────────────────────
  Win Rate:             61.3%
  Profit Factor:         1.82
  Expectancy:          +0.412 R / trade
  Total R:             +12.77 R
────────────────────────────────────────────────────────
  Max Drawdown:          -3.20 R  (-4.8%)
  Max consec. losses:        3
  Sharpe Ratio:           1.24
```

### Walk-forward validation (robustness check)

Splits data into time periods and tests each independently:

```python
result_folds = engine.walk_forward(df_h1, df_m15, n_folds=4)
for i, fold in enumerate(result_folds, 1):
    print(f"\n── Fold {i} ──")
    fold.print_summary()
```

### Export results to CSV + text report

```python
engine.export(result, output_dir="backtesting/results/")
# Writes:
#   backtesting/results/trade_log.csv
#   backtesting/results/backtest_summary.txt
```

### Tune key parameters for backtesting

Edit `.env` or pass overrides directly:

```env
CONFIDENCE_THRESHOLD=0.60    # Lower to get more trades
MIN_RR_RATIO=1.5             # Lower to allow tighter setups
ATR_MIN_POINTS=3.0           # Include quieter periods
```

---

## Deploying to Railway

> ⚠️ **Important:** MetaTrader5 is Windows-only and cannot run on Railway's Linux containers. Deploy in `--no-mt5` mode with a data feed from an alternative provider, or run the bot on a **Windows VPS** and use Railway only for the web dashboard / monitoring layer.

### Steps

**1. Push your code to GitHub**

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/your-username/TradingBot.git
git push -u origin main
```

**2. Create a Railway account**

Go to [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub repo**.

**3. Set environment variables in Railway**

In your Railway project → **Variables** tab, add all variables from `.env.example`:

```
DISCORD_WEBHOOK_URL  = https://discord.com/api/webhooks/...
SYMBOL               = XAUUSD
LOG_LEVEL            = INFO
# Leave MT5 variables blank — use --no-mt5 flag
```

**4. Create a `Procfile`** in the project root:

```
# Procfile
worker: python gold_trading_bot/main.py --no-mt5 --interval 60
```

**or** set the start command in the Railway dashboard:

```
python gold_trading_bot/main.py --no-mt5 --interval 60
```

**5. Create `runtime.txt`** (specify Python version):

```
python-3.11.9
```

**6. Deploy**

Railway auto-deploys on every git push. Monitor logs in the Railway dashboard under **Deployments → View Logs**.

---

## Deploying to Render

> ⚠️ **Same MT5 limitation:** Render runs Linux. Use `--no-mt5` or a Windows VPS for full MT5 support.

### Steps

**1. Create a Render account**

Go to [render.com](https://render.com) → **New +** → **Background Worker**.

**2. Connect your GitHub repository**

Select the repo and branch to deploy from.

**3. Configure the service**

| Setting | Value |
|---|---|
| **Name** | `JayBot` |
| **Environment** | `Python` |
| **Build Command** | `pip install -r gold_trading_bot/requirements.txt` |
| **Start Command** | `python gold_trading_bot/main.py --no-mt5 --interval 60` |
| **Plan** | Starter (free) or Standard |

**4. Add environment variables**

In the **Environment** tab, add all your secrets:

```
DISCORD_WEBHOOK_URL  = https://discord.com/api/webhooks/...
CONFIDENCE_THRESHOLD = 0.68
MIN_RR_RATIO         = 2.0
LOG_LEVEL            = INFO
```

**5. Deploy**

Click **Create Background Worker**. Render will clone, install, and start your bot.

**6. Monitor logs**

In the Render dashboard → **Logs** tab. The bot prints every cycle result:

```
── Cycle #12  13:30:00 UTC ──
Session: OVERLAP  allowed=True
ATR: 8.45  volatility_ok=True
SignalGen: 1 approved, 3 rejected
★ SIGNAL APPROVED ★
BUY XAUUSD @ 2355.50 | conf=74% | regime=TRENDING
```

### Using a Windows VPS (recommended for full MT5 integration)

For **full live trading signals with MT5**, deploy on a Windows VPS:

| Provider | Specs | Cost |
|---|---|---|
| [AWS EC2](https://aws.amazon.com/ec2/) | `t3.medium` Windows | ~$30/mo |
| [Vultr](https://vultr.com) | 2 vCPU / 4 GB RAM Windows | ~$24/mo |
| [Contabo](https://contabo.com) | 4 vCPU / 8 GB RAM Windows | ~$14/mo |
| [DigitalOcean](https://digitalocean.com) | 2 vCPU / 4 GB RAM Windows | ~$28/mo |

On the Windows VPS:

```powershell
# 1. Install Python 3.11
# 2. Install MetaTrader 5 and log in
# 3. Clone and install the bot
git clone <repo-url>
pip install -r requirements.txt

# 4. Create .env with MT5 + Telegram credentials
# 5. Run the bot
python main.py

# 6. To keep running after disconnect — use Task Scheduler or NSSM:
nssm install JayBot "python" "C:\path\to\gold_trading_bot\main.py"
nssm start JayBot
```

---

## Strategy Overview

| SMC Concept | What It Detects | Confidence Bonus |
|---|---|---|
| **Order Block (OB)** | Last opposing candle before impulse move | +5% |
| **Fair Value Gap (FVG)** | 3-candle price imbalance zone | +5% with OB |
| **Break of Structure (BOS)** | Swing high/low broken in trend direction | +3% |
| **Change of Character (CHoCH)** | Early trend reversal signal | +3% |
| **Liquidity Sweep** | Stop-hunt wick below swing low / above swing high | +3% |
| **Trend Alignment** | EMA 50/200 confirms signal direction | +10% |
| **Market Regime** | ADX + price structure: TRENDING preferred | +5% |

---

## Performance Tracking & Reports

**JayBot includes a built-in asynchronous Trade Tracker.**
Whenever a signal is sent to Discord, it is simultaneously recorded in a persistent lightweight database (`data/trades.sqlite`). 

The bot runs a non-blocking background thread that updates open trades against precise M5 price ticks. If the price wick sweeps your predetermined Stop Loss or Take Profit bounds, it correctly marks the trade as WON/LOSS and calculates your strict $ Risk-to-Reward Ratio (R).

### Auto-Reporting
At 11:59 PM Pakistan Time (`23:59 PKT`) exactly every night, the bot calculates your Daily and Lifetime Win Rates, formats them into a clean Markdown Table, and automatically dispatches the total PnL report to your Discord channel.

---

## Risk Management

| Parameter | Default | Notes |
|---|---|---|
| Risk per trade | **1%** of balance | Adjust via `RISK_PER_TRADE_PCT` |
| Lot size | Fixed-fractional | Calibrated for XAUUSD (100 oz/lot) |
| Stop loss | `1.5 × ATR` | Market-adaptive |
| Take Profit 1 | `2.0 × ATR` | 50% of position closed |
| Take Profit 2 | `4.0 × ATR` | Remainder runs to TP2 |
| Min R:R | **2.0 : 1** | Signals below this are rejected |
| Max signals/day | **4** | Hard cap |
| Max daily loss | **3%** | Bot pauses for the day |

---

## Disclaimer

This software is for **educational and informational purposes only**.  
It does **not** constitute financial advice.  
Past backtest performance does **not** guarantee future results.  
Gold (XAUUSD) trading involves significant risk of loss.  
**Use at your own risk.**

---

## License

MIT License — see [LICENSE](LICENSE) for full text.

---

*Built with ❤️ using Python, Smart Money Concepts, and a healthy respect for risk management.*
