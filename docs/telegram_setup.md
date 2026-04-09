# Telegram Setup Guide

How to create and configure the Telegram bot that delivers trading signals.

---

## Step 1: Create a Telegram Bot

1. Open Telegram and search for **@BotFather**
2. Send `/newbot`
3. Choose a name (e.g., `AntiGravity Gold Signals`)
4. Choose a username ending in `_bot` (e.g., `antigravity_gold_bot`)
5. BotFather will reply with your **HTTP API token** — copy it

Set it in `.env`:
```
TELEGRAM_BOT_TOKEN=123456789:ABCdefGhIjKlMnOpQrStUvWxYz
```

---

## Step 2: Get Your Chat ID

### For a personal / test chat (DM to the bot)
1. Start a conversation with your bot (send `/start`)
2. Visit: `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates`
3. Look for `"chat":{"id": 123456789}` — that's your chat ID

```
TELEGRAM_CHAT_ID=123456789
```

### For a public/private **channel**
1. Add the bot as an **administrator** of the channel
2. Forward any message from the channel to @userinfobot
3. The ID will start with `-100` e.g. `-1001234567890`

```
TELEGRAM_CHAT_ID=-1001234567890
```

### For a **group**
1. Add the bot to the group
2. Send a message in the group
3. Visit `getUpdates` — look for `"chat":{"id": -987654321}`

```
TELEGRAM_CHAT_ID=-987654321
```

---

## Step 3: Test the Connection

Run the bot in dry-run mode and check the Telegram channel:

```bash
python main.py --dry-run
```

Or test the Telegram connection directly:

```python
from notifications.telegram_bot import TelegramNotifier
from config.settings import get_settings
import asyncio

cfg = get_settings()
notifier = TelegramNotifier(cfg)
asyncio.run(notifier.send_text("✅ AntiGravity Bot connected successfully!"))
```

---

## Signal Message Format

Each signal is published as a formatted Telegram message:

```
🥇 XAUUSD Signal — BUY

📍 Entry Zone:    2,345.00 – 2,347.50
🛑 Stop Loss:     2,330.00
🎯 Take Profit 1: 2,365.00
🎯 Take Profit 2: 2,390.00

📊 R:R Ratio:     2.5 : 1
📦 Lot Size:      0.25
⚡ Confidence:    74%
🕐 Session:       London–NY Overlap

⚙️ Context:
• Regime: TRENDING_UP (H4)
• Setup: Bullish Order Block + FVG
• Target: Previous session high liquidity

⏰ Signal time: 2026-04-08 14:30 UTC

⚠️ Not financial advice. Trade at your own risk.
```

---

## Admin Alerts

Set `TELEGRAM_ADMIN_CHAT_ID` to your personal Telegram user ID to receive:
- Bot startup notifications
- Error alerts (MT5 disconnection, API failures)
- Daily summary at `DAILY_SUMMARY_TIME_UTC`

---

## Rate Limits

Telegram enforces a rate limit of **30 messages per second** (global) and **1 message per second per chat**. The bot handles this automatically via the `_send_with_retry()` method with exponential back-off.

---

*Last updated: April 2026 | AntiGravity Trading Bot v1.0*
