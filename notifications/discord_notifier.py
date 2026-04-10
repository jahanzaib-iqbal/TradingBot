"""
notifications/discord_notifier.py
================================
Discord webhook notification dispatcher.
"""
import requests
import logging
import time

from config.settings import get_settings

logger = logging.getLogger(__name__)

def send_trade_signal(signal: dict) -> bool:
    """
    Sends a trade signal to Discord via webhook using an Embed format.
    Includes 1 retry on failure.
    """
    settings = get_settings()
    webhook_url = settings.DISCORD_WEBHOOK_URL
    if not webhook_url:
        logger.error("Discord webhook URL not configured.")
        return False

    direction = signal.get("direction", "UNKNOWN")
    
    # Set embed color (Green for BUY, Red for SELL)
    if direction == "BUY":
        color = 0x00FF00  # Green
    elif direction == "SELL":
        color = 0xFF0000  # Red
    else:
        color = 0x808080  # Gray (fallback)

    dir_emoji = "🟢" if direction == "BUY" else "🔴"
    action_text = "BUY 📈" if direction == "BUY" else "SELL 📉"

    # Format fields
    entry = signal.get("entry_price")
    sl = signal.get("stop_loss")
    tp = signal.get("take_profit_1")
    rr = signal.get("rr_ratio_tp1")
    lot_size = signal.get("lot_size")
    confidence = signal.get("confidence", 0) * 100
    timestamp = signal.get("timestamp")

    embed = {
        "title": f"{dir_emoji} JayBot | XAUUSD {action_text}",
        "description": "🚨 **High Probability Setup Detected** 🚨",
        "color": color,
        "fields": [
            {"name": "🎯 Direction", "value": f"**{direction}**", "inline": True},
            {"name": "💰 Entry", "value": f"**{entry:.2f}**" if entry is not None else "N/A", "inline": True},
            {"name": "🛑 Stop Loss", "value": f"**{sl:.2f}**" if sl is not None else "N/A", "inline": True},
            {"name": "✅ Take Profit", "value": f"**{tp:.2f}**" if tp is not None else "N/A", "inline": True},
            {"name": "⚖️ Risk/Reward", "value": f"**1:{rr:.1f}**" if rr is not None else "N/A", "inline": True},
            {"name": "📊 Lot Size", "value": f"**{lot_size}**" if lot_size is not None else "N/A", "inline": True},
            {"name": "🔥 Confidence", "value": f"**{confidence:.0f}%**" if confidence is not None else "N/A", "inline": True}
        ],
        "footer": {
            "text": "🧩 Smart Money • 📈 Macro • 🔗 Correlation"
        }
    }
    
    if timestamp:
        embed["timestamp"] = timestamp

    payload = {"embeds": [embed]}

    # Try sending with 1 retry
    for attempt in range(2):
        try:
            response = requests.post(webhook_url, json=payload, timeout=10)
            if response.status_code in (200, 204):
                logger.info("Successfully sent Discord signal.")
                return True
            else:
                logger.warning(f"Discord webhook failed with status {response.status_code}: {response.text}")
        except requests.RequestException as e:
            logger.warning(f"Discord webhook exception: {e}")
            
        if attempt == 0:
            logger.info("Retrying Discord webhook in 2 seconds...")
            time.sleep(2)
            
    logger.error("Failed to send Discord signal after 2 attempts.")
    return False

def send_bot_started(symbol: str, version: str) -> bool:
    """Send bot startup notification via Discord."""
    settings = get_settings()
    webhook_url = settings.DISCORD_WEBHOOK_URL
    if not webhook_url:
        return False
        
    embed = {
        "title": "✅ JayBot Started",
        "description": f"**Symbol:** {symbol}\n**Version:** {version}\n\nMonitoring Gold for high-probability setups...",
        "color": 0x00FF00,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    
    try:
        requests.post(webhook_url, json={"embeds": [embed]}, timeout=10)
        return True
    except requests.RequestException:
        return False

def send_bot_stopped(reason: str) -> bool:
    """Send bot shutdown notification via Discord."""
    settings = get_settings()
    webhook_url = settings.DISCORD_WEBHOOK_URL
    if not webhook_url:
        return False
        
    embed = {
        "title": "🛑 JayBot Stopped",
        "description": f"**Reason:** {reason}",
        "color": 0xFF0000,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    
    try:
        requests.post(webhook_url, json={"embeds": [embed]}, timeout=10)
        return True
    except requests.RequestException:
        return False
