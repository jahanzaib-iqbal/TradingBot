"""
filters/news_filter.py
========================
Fundamental news event guard for XAUUSD signal generation.

Why filter around news?
------------------------
High-impact macroeconomic events (NFP, CPI, FOMC, Fed chair speeches,
GDP, PCE) cause sudden, large Gold price moves that cannot be modelled
by technical SMC setups.  Entering just before a news release means:
  - Wide spreads (broker markup during volatility)
  - Slippage on entry and stop-loss execution
  - Whipsaw reversals after the initial spike

Strategy: block new signals in a configurable window BEFORE and AFTER
high-impact USD/XAU events.

Blocking windows (configurable in Settings)
--------------------------------------------
  NEWS_BLACKOUT_BEFORE_MIN = 30   (no new signals 30 min before event)
  NEWS_BLACKOUT_AFTER_MIN  = 15   (no new signals 15 min after event)

High-impact events tracked for XAUUSD
--------------------------------------
  USD: NFP, CPI, Core CPI, PPI, FOMC statements, Fed chair speeches,
       GDP, PCE, Retail Sales, ISM, ADP payrolls
  XAU: None scheduled (Gold reacts to USD events, not its own)

NewsDataProvider integration
------------------------------
When NewsDataProvider is available and connected, this filter queries it
for a live economic calendar.  When unavailable (no API key, offline),
the filter falls back to SAFE_MODE = True (i.e., always returns allowed).

Note: In safe_mode, set the `safe_mode` parameter to False to make the
filter block ALL signals (ultra-conservative mode for paper trading).

Dependencies: data.news_data.NewsDataProvider, config.settings.Settings
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, Tuple

from config.settings import Settings
from utils.logger import get_logger

logger = get_logger(__name__)


class NewsFilter:
    """
    Prevents signal generation during high-impact news windows.

    Usage
    -----
        nf = NewsFilter(settings)
        allowed, reason = nf.is_allowed()
        delta = nf.confidence_delta()

    Without a live news provider the filter runs in safe_mode=True
    (no blocking) unless explicitly set to safe_mode=False.
    """

    def __init__(
        self,
        settings:    Settings,
        news_provider = None,   # Optional[NewsDataProvider] — avoids hard import
        safe_mode:   bool = True,
    ) -> None:
        """
        Parameters
        ----------
        settings      : App settings
        news_provider : Optional NewsDataProvider instance.
                        If None, the filter runs in safe_mode.
        safe_mode     : If True and no news_provider, always return allowed.
                        If False and no news_provider, always return blocked.
        """
        self.cfg           = settings
        self.news          = news_provider
        self.safe_mode     = safe_mode
        self.blackout_pre  = settings.NEWS_BLACKOUT_BEFORE_MIN   # default 30
        self.blackout_post = settings.NEWS_BLACKOUT_AFTER_MIN    # default 15

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def is_allowed(self, now: datetime | None = None) -> Tuple[bool, str]:
        """
        Check whether signal generation is permitted from a news perspective.

        Returns
        -------
        (True, "")              — no blocking event detected
        (False, reason_string)  — near a high-impact event; signal blocked
        """
        if now is None:
            now = datetime.now(timezone.utc)

        # No news provider attached
        if self.news is None:
            if self.safe_mode:
                logger.debug("NewsFilter: no provider — safe_mode=True, allowing signal")
                return True, ""
            else:
                return False, "NewsFilter: no news provider — blocking in conservative mode"

        # Query live provider
        try:
            blocked, reason = self._query_provider(now)
            return (not blocked), reason if blocked else ""
        except Exception as exc:
            logger.warning(f"NewsFilter: provider query failed ({exc}). Falling back to safe_mode={self.safe_mode}")
            if self.safe_mode:
                return True, ""
            return False, f"News provider error: {exc}"

    def confidence_delta(self, now: datetime | None = None) -> float:
        """
        Return the confidence adjustment based on proximity to news events.

        Well away from events → 0.00 (no effect)
        Within 60–30 min pre  → -0.05 (approaching — slight caution)
        Within blackout window → -0.20 (should be blocked, but if somehow
                                         not, heavily penalise confidence)
        """
        if self.news is None:
            return 0.0

        try:
            minutes_to_next = self._minutes_to_next_event(now)
            if minutes_to_next is None:
                return 0.0
            if minutes_to_next <= self.blackout_pre:
                return -0.20
            if minutes_to_next <= 60:
                return -0.05
            return 0.00
        except Exception:
            return 0.00

    def get_next_event_info(self) -> str:
        """Return a human-friendly string about the next upcoming news event."""
        if self.news is None:
            return "No news provider configured"
        try:
            event = self.news.get_next_event()
            if event is None:
                return "No upcoming high-impact events found"
            name = getattr(event, "name", str(event))
            time_ = getattr(event, "time", "unknown time")
            return f"Next event: {name} at {time_} UTC"
        except Exception as exc:
            return f"Could not fetch next event: {exc}"

    def news_info(self, now: datetime | None = None) -> dict:
        """Return a summary dict of the current news state."""
        allowed, reason = self.is_allowed(now)
        return {
            "allowed":          allowed,
            "reason":           reason if not allowed else "",
            "confidence_delta": self.confidence_delta(now),
            "next_event":       self.get_next_event_info(),
            "safe_mode":        self.safe_mode,
        }

    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================

    def _query_provider(self, now: datetime) -> Tuple[bool, str]:
        """
        Ask the live news provider whether we are in a blackout window.

        Returns (blocked: bool, reason: str).
        """
        # Prefer provider's own is_news_blackout() method if it exists
        if hasattr(self.news, "is_news_blackout"):
            blocked = self.news.is_news_blackout(now)
            if blocked:
                event_info = self.get_next_event_info()
                return True, f"News blackout active — {event_info}"
            return False, ""

        # Fallback: check minutes to next event manually
        minutes_to = self._minutes_to_next_event(now)
        if minutes_to is not None and minutes_to <= self.blackout_pre:
            return True, f"High-impact event in {minutes_to:.0f} min"

        return False, ""

    def _minutes_to_next_event(self, now: datetime | None) -> Optional[float]:
        """Return minutes until the next high-impact event, or None if unknown."""
        if self.news is None:
            return None
        if now is None:
            now = datetime.now(timezone.utc)
        try:
            event = self.news.get_next_event()
            if event is None:
                return None
            event_time = getattr(event, "time", None)
            if event_time is None:
                return None
            delta = (event_time - now).total_seconds() / 60.0
            return max(delta, 0.0)
        except Exception:
            return None
