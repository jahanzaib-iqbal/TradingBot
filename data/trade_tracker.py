import sqlite3
import threading
import logging
import pandas as pd
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

class TradeTracker:
    def __init__(self, db_path="data/trade_history.sqlite"):
        self.db_path = db_path
        self._lock = threading.RLock()
        
        # Initialize database
        with self._lock:
            with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date TEXT,
                        timestamp TEXT,
                        direction TEXT,
                        entry_price REAL,
                        stop_loss REAL,
                        take_profit REAL,
                        exit_price REAL,
                        result TEXT,
                        rr REAL,
                        status TEXT
                    )
                """)
                conn.commit()

    def get_pkt_date_str(self, dt_utc: datetime = None) -> str:
        """Returns YYYY-MM-DD for Pakistan Time (UTC+5)"""
        if dt_utc is None:
            dt_utc = datetime.now(timezone.utc)
        dt_pkt = dt_utc + timedelta(hours=5)
        return dt_pkt.strftime("%Y-%m-%d")

    def add_trade(self, signal: dict):
        """Called asynchronously. Save newly dispatched signal to DB."""
        now = datetime.now(timezone.utc)
        date_str = self.get_pkt_date_str(now)
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S UTC")
        
        try:
            with self._lock:
                with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
                    conn.execute("""
                        INSERT INTO trades (
                            date, timestamp, direction, entry_price, 
                            stop_loss, take_profit, status
                        ) VALUES (?, ?, ?, ?, ?, ?, 'PENDING')
                    """, (
                        date_str, 
                        timestamp, 
                        signal.get("direction", "UNKNOWN"), 
                        signal.get("entry_price", 0.0), 
                        signal.get("stop_loss", 0.0), 
                        signal.get("take_profit_1", 0.0)
                    ))
                    conn.commit()
        except Exception as e:
            logger.error(f"TradeTracker failed to add trade: {e}")

    def update_open_trades(self, df_m5: pd.DataFrame):
        """Called asynchronously. Check active trades against recent candles."""
        if df_m5 is None or df_m5.empty:
            return

        with self._lock:
            with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM trades WHERE status = 'PENDING'")
                open_trades = cursor.fetchall()
                
                if not open_trades:
                    return

                for trade in open_trades:
                    self._check_trade_against_candles(trade, df_m5, conn)

    def _check_trade_against_candles(self, trade: sqlite3.Row, df: pd.DataFrame, conn: sqlite3.Connection):
        trade_id = trade["id"]
        direction = trade["direction"]
        sl = trade["stop_loss"]
        tp = trade["take_profit"]
        entry = trade["entry_price"]

        hit_tp = False
        hit_sl = False
        exit_price = None

        # Check each M5 candle chronologically
        for _, row in df.iterrows():
            high, low = row["high"], row["low"]

            if direction == "BUY":
                if low <= sl:
                    hit_sl = True
                    exit_price = sl
                elif high >= tp:
                    hit_tp = True
                    exit_price = tp

            elif direction == "SELL":
                if high >= sl:
                    hit_sl = True
                    exit_price = sl
                elif low <= tp:
                    hit_tp = True
                    exit_price = tp

            # If both hit in the exact same candle, conservatively assume SL hit first.
            if hit_sl and hit_tp:
                hit_tp = False
                exit_price = sl

            if hit_sl or hit_tp:
                break
        
        if not hit_sl and not hit_tp:
            return # Still Pending

        # Calculate RR
        result = "WIN" if hit_tp else "LOSS"
        
        if direction == "BUY":
            risk = entry - sl
            reward = exit_price - entry
        else: # SELL
            risk = sl - entry
            reward = entry - exit_price

        # Safeguard division by zero
        if risk == 0:
            rr = 0.0
        else:
            rr = reward / risk if result == "WIN" else -1.0
            
        rr = round(rr, 2)

        cursor = conn.cursor()
        cursor.execute("""
            UPDATE trades 
            SET status = 'CLOSED', result = ?, exit_price = ?, rr = ? 
            WHERE id = ?
        """, (result, exit_price, rr, trade_id))
        conn.commit()

    def get_daily_performance(self, date_str: str) -> dict:
        """Returns statistics for a specific PKT date"""
        return self._compute_performance("SELECT * FROM trades WHERE date = ?", (date_str,))

    def get_overall_performance(self) -> dict:
        """Returns lifetime statistics"""
        return self._compute_performance("SELECT * FROM trades", ())

    def get_daily_trades(self, date_str: str) -> list[dict]:
        with self._lock:
            with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM trades WHERE date = ?", (date_str,))
                return [dict(row) for row in cursor.fetchall()]

    def _compute_performance(self, query: str, params: tuple) -> dict:
        with self._lock:
            with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(query, params)
                trades = cursor.fetchall()

        total = len(trades)
        wins = sum(1 for t in trades if t["result"] == "WIN")
        losses = sum(1 for t in trades if t["result"] == "LOSS")
        total_r = sum(t["rr"] for t in trades if t["status"] == "CLOSED" and t["rr"] is not None)
        
        closed_trades = wins + losses
        win_rate = (wins / closed_trades * 100) if closed_trades > 0 else 0.0

        return {
            "total": total,
            "wins": wins,
            "losses": losses,
            "win_rate": round(win_rate, 1),
            "total_r": round(total_r, 2)
        }
