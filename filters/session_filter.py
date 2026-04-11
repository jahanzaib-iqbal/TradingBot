"""
filters/session_filter.py
==========================
Trading session time-window filter for XAUUSD.

Gold liquidity and volatility are heavily session-dependent:
  - Asian session   (00:00-07:00 UTC) — thin liquidity, small ranges, many false signals
  - London session  (07:00-16:00 UTC) — high liquidity, institutional activity, primary session
  - NY session      (12:00-21:00 UTC) — USD-driven moves, strong follow-through
  - London/NY overlap (12:00-16:00)   — BEST window: highest volume, clearest setups

Signal confidence adjustment by session
-----------------------------------------
  OVERLAP  → +0.10  (optimal — max liquidity, two major markets open)
  LONDON   → +0.05  (excellent — institutional flow dominates)
  NEW_YORK → +0.03  (good — USD volatility creates clean setups)
  ASIAN    → -0.10  (poor — low liquidity, choppy; blocked by default)
  CLOSED   →  0.00  (blocked)

Dependencies: datetime, config.settings.Settings
"""

from __future__ import annotations

from datetime import datetime, time, timezone
from typing import Tuple

from config.settings import Settings
from utils.logger import get_logger

logger = get_logger(__name__)

# UTC session boundaries (inclusive start, inclusive end)
SESSION_WINDOWS: dict[str, tuple[time, time]] = {
    "asian":    (time(0, 0),  time(6, 59)),
    "london":   (time(7, 0),  time(15, 59)),
    "overlap":  (time(12, 0), time(15, 59)),   # London–NY overlap (highest quality)
    "new_york": (time(12, 0), time(20, 59)),
}

# Confidence delta each session contributes to a signal's final score
SESSION_CONFIDENCE_DELTA: dict[str, float] = {
    "overlap":  0.10,
    "london":   0.05,
    "new_york": 0.03,
    "asian":    -0.10,
    "closed":   0.00,
}


class SessionFilter:
    """
    Restricts signal generation to configured high-liquidity sessions.

    Usage
    -----
        sf = SessionFilter(settings)
        allowed, session = sf.is_allowed()
        confidence_boost = sf.confidence_delta()
    """

    def __init__(self, settings: Settings) -> None:
        self.cfg = settings

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def current_session(self, now: datetime | None = None) -> str:
        """
        Return the name of the current trading session based on UTC time.

        Return values (in priority order):
            "overlap"  — London/NY overlap (12:00–15:59 UTC)
            "london"   — London session   (07:00–15:59 UTC)
            "new_york" — New York session (12:00–20:59 UTC)
            "asian"    — Asian session    (00:00–06:59 UTC)
            "closed"   — Outside all defined windows

        Note: "overlap" takes priority over individual "london" / "new_york"
        because it represents the highest-quality sub-window.
        """
        utc_now = self._utc_time(now)

        # Check overlap first (highest priority sub-session)
        if self._in_window(utc_now, "overlap"):
            return "overlap"

        if self._in_window(utc_now, "london"):
            return "london"

        if self._in_window(utc_now, "new_york"):
            return "new_york"

        if self._in_window(utc_now, "asian"):
            return "asian"

        return "closed"

    def is_allowed(self, now: datetime | None = None) -> Tuple[bool, str]:
        """
        Always returns True per user request: "I dont want to restrict the bot to a session window, it should gave signals 24/7".
        """
        session = self.current_session(now)
        return True, f"Session '{session}' (24/7 mode active)"

    def confidence_delta(self, now: datetime | None = None) -> float:
        """
        Return the confidence score adjustment for the current session.

        Positive values increase signal confidence; negative reduce it.
        """
        session = self.current_session(now)
        return SESSION_CONFIDENCE_DELTA.get(session, 0.0)

    def session_info(self, now: datetime | None = None) -> dict:
        """Return a dict summarising the current session state."""
        session   = self.current_session(now)
        allowed, reason = self.is_allowed(now)
        return {
            "session":          session,
            "allowed":          allowed,
            "reason":           reason if not allowed else "",
            "confidence_delta": self.confidence_delta(now),
        }

    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================

    def _in_window(self, t: time, session: str) -> bool:
        start, end = SESSION_WINDOWS[session]
        return start <= t <= end

    @staticmethod
    def _utc_time(now: datetime | None) -> time:
        """Extract the UTC time component from a datetime (or use now)."""
        if now is None:
            now = datetime.now(timezone.utc)
        elif now.tzinfo is None:
            # Treat naive datetimes as UTC
            now = now.replace(tzinfo=timezone.utc)
        return now.astimezone(timezone.utc).time()
