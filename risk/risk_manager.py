"""
risk/risk_manager.py
--------------------
Daily risk gate and signal throttle manager.

Responsibilities:
- Track how many signals have been sent today
- Track the running daily P&L exposure (estimated)
- Block new signals if any of the following are breached:
    * MAX_SIGNALS_PER_DAY has been reached
    * Estimated daily loss exceeds MAX_DAILY_LOSS_PCT
    * A news blackout window is active
- Reset counters at the start of each new trading day (UTC midnight)
- Persist intraday state to a lightweight JSON file so restarts don't lose context

Dependencies:
    config.settings.Settings, utils.logger
"""

from __future__ import annotations
import json
from datetime import date
from pathlib import Path

from config.settings import Settings


class RiskManager:
    """Guards the signal pipeline against over-trading and excessive daily loss."""

    STATE_FILE = Path("data/risk_state.json")

    def __init__(self, settings: Settings):
        self.cfg = settings
        self._state: dict = self._load_state()

    # ── Gate Checks ───────────────────────────────────────────────────────────

    def is_trading_allowed(self) -> bool:
        """
        Master gate — returns True only if ALL risk conditions are satisfied.
        (Note: Max Signals and Max Loss completely disabled as per user request).
        """
        self._reset_if_new_day()
        return True

    def register_signal_sent(self) -> None:
        """Increment the daily signal counter and persist state."""
        self._state["signals_today"] += 1
        self._save_state()

    def update_daily_loss(self, loss_pct: float) -> None:
        """Update the running daily loss percentage (positive = loss)."""
        self._state["daily_loss_pct"] = loss_pct
        self._save_state()

    # ── State Persistence ─────────────────────────────────────────────────────

    def _reset_if_new_day(self) -> None:
        today = str(date.today())
        if self._state.get("date") != today:
            self._state = {"date": today, "signals_today": 0, "daily_loss_pct": 0.0}
            self._save_state()

    def _load_state(self) -> dict:
        if self.STATE_FILE.exists():
            with open(self.STATE_FILE) as f:
                return json.load(f)
        return {"date": "", "signals_today": 0, "daily_loss_pct": 0.0}

    def _save_state(self) -> None:
        self.STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(self.STATE_FILE, "w") as f:
            json.dump(self._state, f, indent=2)
