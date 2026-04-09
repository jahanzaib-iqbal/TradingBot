# AntiGravity — Gold Trading Signal Bot

> **Signal-only.** No automated execution. All signals published to Telegram.

---

## What is this?

**AntiGravity** is a modular Python bot that analyses the XAUUSD (Gold) market and publishes
intraday trade signals to a Telegram channel. It is designed for traders who want
data-driven, algorithmically filtered entry ideas while keeping full manual control
over trade execution.

The bot generates **2–4 high-quality signals per day** using a Smart Money Concepts
methodology — Order Blocks, Fair Value Gaps, Break of Structure, and liquidity pool targeting.

---

## Quick Links

| Resource | Location |
|----------|----------|
| 📐 Architecture & modules | [`docs/project_overview.md`](gold_trading_bot/docs/project_overview.md) |
| ⚙️ All configuration options | [`docs/configuration.md`](gold_trading_bot/docs/configuration.md) |
| 📊 Strategy explanation | [`docs/strategy.md`](gold_trading_bot/docs/strategy.md) |
| 🛡️ Risk management | [`docs/risk_management.md`](gold_trading_bot/docs/risk_management.md) |
| 🕵️ Backtesting guide | [`docs/backtesting.md`](gold_trading_bot/docs/backtesting.md) |
| 📲 Telegram setup | [`docs/telegram_setup.md`](gold_trading_bot/docs/telegram_setup.md) |

---

## Requirements

- **Python** 3.11+
- **Windows OS** (MetaTrader 5 Python bridge is Windows-only)
- A MetaTrader 5 account (demo or live) at any supported broker
- A Telegram bot token (free — create via @BotFather)

---

## Getting Started

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd AntiGravity_TradingBot/gold_trading_bot

# 2. Create a virtual environment
python -m venv .venv
.venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure the bot
copy .env.example .env
# → Edit .env with your MT5 credentials, Telegram token, and risk settings

# 5. Run
python main.py                # Live mode
python main.py --dry-run      # No Telegram dispatch (for testing)
python main.py --backtest     # Historical simulation
```

---

## Disclaimer

This software is provided for **educational and informational purposes only**.
It does not constitute financial advice, investment advice, or any recommendation
to buy or sell any financial instrument.

Trading foreign exchange, commodities, and CFDs carries significant risk of loss
and may not be suitable for all investors. Past backtest performance is not
indicative of future results. Always trade with capital you can afford to lose.

---

## License

MIT License — see [`LICENSE`](LICENSE) for details.
