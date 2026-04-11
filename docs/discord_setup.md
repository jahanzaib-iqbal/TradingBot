# Discord Setup Guide

How to configure the Discord Webhook that delivers trading signals and the daily performance reports.

---

## Step 1: Create a Discord Webhook

1. Open your Discord Server where you want to receive the signals.
2. Go to **Server Settings** (click your server name in the top left).
3. Select **Integrations** in the left menu.
4. Click on **Webhooks** and then click **New Webhook**.
5. Name the webhook (e.g., `AntiGravity Gold Bot`).
6. Select the specific Discord **Channel** where you want the messages dropped.
7. Click **Copy Webhook URL** and save your changes.

---

## Step 2: Add URL to your Configuration

In your project root, open your `.env` file and paste the copied URL directly into the configured variable:

```env
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/your-webhook-id/your-webhook-token
```

---

## Signal Message Format

Whenever the bot detects a high-probability Smart Money setup via the H4/H1/M15 timeframe analysis, it pushes an embedded message straight to Discord in less than a second. 

The format looks exactly like this:

```
🥇 XAUUSD Signal — BUY

📍 Entry Zone:    2,345.00 – 2,347.50
🛑 Stop Loss:     2,330.00
🎯 Take Profit 1: 2,365.00
🎯 Take Profit 2: 2,390.00

📊 R:R Ratio:     2.5 : 1
📦 Lot Size:      0.25 (1% Risk)
⚡ Confidence:    74%
🕐 Session:       London–NY Overlap

⚙️ Context:
• Regime: TRENDING_UP (H4)
• Setup: Bullish Order Block + FVG
• Target: Previous session high liquidity

⏰ Signal time: 2026-04-12 14:30 UTC
```

---

## The Dynamic Trade Tracker

AntiGravity features a persistent background database (`trade_history.sqlite`) that evaluates the actual M5 charts against the Stop Loss and Take Profit levels of every signal sent. 

Because the tracking system relies on background **Threading** (`ThreadPoolExecutor`), it never delays the instantaneous transmission of the signal logic to the Discord Webhook.

---

## Daily Trading Report

At **23:59 Pakistan Standard Time (PKT)** exactly, the bot halts briefly to generate a consolidated Markdown Table outlining exactly how the signals from the day performed based on the Tracker results.

The report format generated follows this exact monospace structure inside a Discord Markdown Embedded Frame:

```
📊 DAILY TRADING REPORT (XAUUSD)

**Date:** 2026-04-12

**TRADES:**
# | Dir | Entry  | Exit   | SL     | TP     | RR   
--|-----|--------|--------|--------|--------|------
1 | BUY | 2350.5 | 2365.0 | 2345.0 | 2365.0 | +2.6R
2 | SELL| 2362.0 | OPEN   | 2368.0 | 2350.0 | PEND 

**DAILY PERFORMANCE:**
Trades: 2
Wins: 1 | Loss: 0
Win Rate: 100.0%
Total R: +2.6R

**LIFETIME PERFORMANCE:**
Trades: 15
Wins: 10 | Loss: 5
Win Rate: 66.6%
Total R: +12.4R
```

---

## Important Admin Notes

*   Ensure you leave the terminal executing the script open 24/7 if deployed locally, or hosted on AWS/Render.
*   Discord Webhooks only transmit outwards (1-way). The bot does not respond to typed text commands.
*   If the bot's MT5 connection drops, it will repeatedly auto-reconnect and dispatch a startup success confirmation embed to Discord automatically.

*Last updated: April 2026 | AntiGravity Trading Bot v1.0*
