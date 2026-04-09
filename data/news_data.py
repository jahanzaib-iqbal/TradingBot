"""
data/news_data.py
==================
Economic news intelligence layer for the AntiGravity Gold Trading Bot.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GOAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Fetch and score economic news from multiple free APIs, then
provide the NewsFilter with a real-time risk assessment so that
signal generation is automatically halted before and after
high-impact events that move Gold prices.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATA SOURCES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Source A — Finnhub Economic Calendar  (finnhub.io)
  • Free tier: 60 API calls/minute
  • Provides scheduled economic events with impact level
  • Endpoint: /calendar/economic
  • Tracks: USD, EUR, GBP events with "high"/"medium"/"low" impact
  • API key:  Settings.FINNHUB_API_KEY

Source B — NewsAPI.org  (newsapi.org)
  • Free tier: 100 calls/day
  • Provides news headline scanning for sentiment keywords
  • Endpoint: /v2/everything
  • Used for:  unscheduled risk events (surprise announcements,
               geopolitical shocks, Fed speeches, war escalation)
  • API key:  Settings.NEWSAPI_KEY

Fallback — no API keys provided:
  • SAFE_MODE is engaged → all signals allowed (no blocking)
  • Warning logged on every call

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RISK SCORING MODEL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Final risk_score is an integer 0–100.

Inputs:
  1. Calendar events (Finnhub):
       time-to-event < blackout_before_min
         AND impact == "high"   → +60
         AND impact == "medium" → +20
       Still ahead (30–120 min) AND impact == "high" → +25
       Currency weight: USD → 1.0×, EUR/GBP → 0.5×

  2. Headline keywords (NewsAPI):
       Each matched keyword family adds a weighted score:
         Federal Reserve / FOMC / rate decision → +30
         Inflation / CPI / PCE / Core PCE       → +20
         War / military / geopolitical           → +20
         Unemployment / NFP / payrolls           → +15
         USD / DXY / dollar                      → +10
         (each capped: one match per family)

  3. Recency multiplier (NewsAPI):
       headline published < 1 h ago  → 1.00×
       published < 4 h ago           → 0.60×
       published < 12 h ago          → 0.30×
       older                         → 0.10×

  4. Score cap → clamped to [0, 100]

Risk classification:
  score ≥ NEWS_HIGH_RISK_THRESHOLD    (default 70) → "HIGH"   → BLOCK
  score ≥ NEWS_MEDIUM_RISK_THRESHOLD  (default 35) → "MEDIUM" → warn + penalise confidence
  score <  NEWS_MEDIUM_RISK_THRESHOLD              → "LOW"    → allow

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  news_provider.get_risk_assessment() →
    {
      "news_risk":     "HIGH" | "MEDIUM" | "LOW",
      "risk_score":    82,
      "blocked":       True,
      "reason":        "FOMC statement in 18 min (high impact USD)",
      "next_event":    {"name": "FOMC", "time": "2024-03-20T18:00:00Z",
                        "currency": "USD", "impact": "high",
                        "minutes_away": 18},
      "headlines":     ["Fed expected to hold rates...", ...],
      "cached":        False,
      "as_of":         "2024-03-20T17:42:00Z",
    }

  Integrates directly with NewsFilter via:
    news_provider.is_news_blackout(now) → bool
    news_provider.get_next_event()      → NewsEvent | None

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CACHING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Both API responses are cached independently using
Settings.NEWS_CACHE_TTL_MIN (default 10 minutes) to avoid
burning free-tier quotas.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PUBLIC API
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  provider = NewsDataProvider(settings)

  # Check if the current moment is in a blackout window
  blocked: bool = provider.is_news_blackout()

  # Full risk assessment dict
  assessment: dict = provider.get_risk_assessment()

  # Next scheduled high-impact event (compatible with NewsFilter)
  event: NewsEvent | None = provider.get_next_event()

  # Upcoming events in lookahead window
  events: list[NewsEvent] = provider.get_upcoming_events(hours_ahead=2)

  # Manually invalidate cache (e.g., on bot restart)
  provider.clear_cache()

Dependencies
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  requests, config.settings.Settings, utils.logger
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional

from config.settings import Settings
from utils.logger import get_logger

logger = get_logger(__name__)

# Optional import — graceful degradation if requests not installed
try:
    import requests as _requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _requests = None          # type: ignore[assignment]
    _REQUESTS_AVAILABLE = False
    logger.warning("data.news_data: 'requests' not installed — API calls disabled")


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

FINNHUB_CALENDAR_URL = "https://finnhub.io/api/v1/calendar/economic"
NEWSAPI_EVERYTHING_URL = "https://newsapi.org/v2/everything"

# Maximum HTTP timeout for API calls (seconds)
HTTP_TIMEOUT = 8

# Maximum number of headlines to request from NewsAPI per call
NEWSAPI_MAX_ARTICLES = 10

# ─────────────────────────────────────────────────────────────────────────────
# Keyword families and their base weights for headline scoring
# ─────────────────────────────────────────────────────────────────────────────

# Each entry: (family_name, [keywords], base_score_contribution)
# Score is added once per family (de-duplicated), scaled by recency multiplier.
_KEYWORD_FAMILIES: list[tuple[str, list[str], int]] = [
    (
        "fed_fomc",
        [
            "federal reserve", "fomc", "fed chair", "jerome powell",
            "rate decision", "rate hike", "rate cut", "interest rate",
            "monetary policy", "fed minutes", "fed statement",
        ],
        30,
    ),
    (
        "inflation",
        [
            "inflation", "cpi", "core cpi", "pce", "core pce",
            "consumer price", "price index", "price pressure",
        ],
        20,
    ),
    (
        "war_geopolitics",
        [
            "war", "military strike", "conflict", "nato", "russia", "ukraine",
            "middle east", "israel", "iran", "sanctions", "geopolitical",
            "gulf", "escalation",
        ],
        20,
    ),
    (
        "employment",
        [
            "nfp", "non-farm payroll", "payrolls", "unemployment",
            "jobless claims", "adp", "labor market", "jobs report",
        ],
        15,
    ),
    (
        "usd_dxy",
        [
            "usd", "dollar", "dxy", "dollar index",
            "currency", "fx",
        ],
        10,
    ),
    (
        "recession_gdp",
        [
            "recession", "gdp", "gross domestic", "contraction",
            "economic slowdown", "financial crisis",
        ],
        15,
    ),
    (
        "gold_specific",
        [
            "gold rally", "gold surge", "gold price", "xauusd",
            "safe haven", "precious metal",
        ],
        10,
    ),
]

# High-impact USD event name fragments (maps to Finnhub event matching)
_HIGH_IMPACT_USD_EVENTS: list[str] = [
    "nfp", "non-farm", "payroll",
    "cpi", "core cpi", "pce", "core pce",
    "fomc", "federal reserve", "rate decision",
    "gdp", "gross domestic",
    "retail sales", "ism",
    "adp", "jobless", "unemployment",
    "inflation",
]


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NewsEvent:
    """
    A scheduled economic event from the Finnhub calendar.

    Attributes
    ----------
    name       : e.g. "Non-Farm Payrolls"
    currency   : e.g. "USD"
    impact     : "high" | "medium" | "low"
    time       : UTC datetime of the event
    actual     : Reported value (if already released), else None
    estimate   : Consensus estimate, else None
    """
    name:     str
    currency: str
    impact:   str       # "high" | "medium" | "low"
    time:     datetime
    actual:   Optional[str] = None
    estimate: Optional[str] = None

    @property
    def minutes_away(self) -> float:
        """Minutes until (or since, if negative) this event fires."""
        now = datetime.now(timezone.utc)
        return (self.time - now).total_seconds() / 60.0

    @property
    def is_high_impact(self) -> bool:
        return self.impact.lower() == "high"

    @property
    def is_upcoming(self) -> bool:
        """True if the event has not yet occurred."""
        return self.minutes_away > 0

    def to_dict(self) -> dict:
        return {
            "name":         self.name,
            "currency":     self.currency,
            "impact":       self.impact,
            "time":         self.time.isoformat(),
            "minutes_away": round(self.minutes_away, 1),
            "actual":       self.actual,
            "estimate":     self.estimate,
        }

    def __str__(self) -> str:
        mins = self.minutes_away
        arrow = f"in {mins:.0f} min" if mins > 0 else f"{abs(mins):.0f} min ago"
        return f"NewsEvent({self.name!r} [{self.currency}/{self.impact}] {arrow})"


@dataclass
class NewsRiskAssessment:
    """
    Composite news risk output.  This is the canonical return type of
    NewsDataProvider.get_risk_assessment().
    """
    news_risk:   str    # "HIGH" | "MEDIUM" | "LOW"
    risk_score:  int    # 0–100
    blocked:     bool   # True when news_risk == "HIGH"
    reason:      str    # human-readable explanation
    next_event:  Optional[NewsEvent]  = None
    headlines:   list[str]            = field(default_factory=list)
    cached:      bool                 = False
    as_of:       Optional[datetime]   = None

    def to_dict(self) -> dict:
        """Return the canonical output structure requested by the user."""
        return {
            "news_risk":    self.news_risk,
            "risk_score":   self.risk_score,
            "blocked":      self.blocked,
            "reason":       self.reason,
            "next_event":   self.next_event.to_dict() if self.next_event else None,
            "headlines":    self.headlines,
            "cached":       self.cached,
            "as_of":        self.as_of.isoformat() if self.as_of else None,
        }

    def __str__(self) -> str:
        return (
            f"NewsRisk({self.news_risk} score={self.risk_score} "
            f"blocked={self.blocked}  {self.reason[:60]})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main provider
# ─────────────────────────────────────────────────────────────────────────────

class NewsDataProvider:
    """
    Fetches and scores economic news from Finnhub and NewsAPI.

    Designed as a drop-in replacement for the stub that was previously here.
    Compatible with NewsFilter's expected interface:
      • is_news_blackout(now) → bool
      • get_next_event()      → NewsEvent | None

    Thread-safety: a simple timestamp-based TTL cache is used.  For
    concurrent access, wrap calls in a threading.Lock if needed.

    Usage
    -----
        cfg      = Settings()
        provider = NewsDataProvider(cfg)

        assessment = provider.get_risk_assessment()
        print(assessment.to_dict())
        # {"news_risk": "HIGH", "risk_score": 78, "blocked": True, ...}

        if provider.is_news_blackout():
            logger.info("Signals blocked — high-impact news")
    """

    def __init__(self, settings: Settings) -> None:
        self.cfg               = settings
        self._finnhub_key      = settings.FINNHUB_API_KEY or settings.NEWS_API_KEY
        self._newsapi_key      = settings.NEWSAPI_KEY
        self._blackout_pre     = settings.NEWS_BLACKOUT_BEFORE_MIN   # default 30
        self._blackout_post    = settings.NEWS_BLACKOUT_AFTER_MIN    # default 15
        self._lookahead_min    = settings.NEWS_LOOKAHEAD_MIN         # default 120
        self._cache_ttl_sec    = settings.NEWS_CACHE_TTL_MIN * 60    # convert to seconds
        self._high_threshold   = settings.NEWS_HIGH_RISK_THRESHOLD   # default 70
        self._medium_threshold = settings.NEWS_MEDIUM_RISK_THRESHOLD # default 35
        self._tracked_ccy      = [
            c.strip().upper()
            for c in settings.NEWS_TRACKED_CURRENCIES.split(",")
            if c.strip()
        ]

        # Internal caches
        self._calendar_cache:    list[NewsEvent] = []
        self._calendar_ts:       float           = 0.0   # UNIX timestamp

        self._headlines_cache:   list[dict]      = []    # raw article dicts
        self._headlines_ts:      float           = 0.0

        self._assessment_cache:  Optional[NewsRiskAssessment] = None
        self._assessment_ts:     float           = 0.0

        logger.info(
            f"NewsDataProvider initialised  "
            f"finnhub={'yes' if self._finnhub_key else 'NO'}  "
            f"newsapi={'yes' if self._newsapi_key else 'NO'}  "
            f"blackout_pre={self._blackout_pre}min  "
            f"blackout_post={self._blackout_post}min"
        )

    # =========================================================================
    # PRIMARY PUBLIC API — NewsFilter integration
    # =========================================================================

    def is_news_blackout(self, now: Optional[datetime] = None) -> bool:
        """
        Return True if the current moment falls within a pre/post high-impact
        news blackout window.

        Compatible with NewsFilter._query_provider() which calls this method.
        """
        assessment = self.get_risk_assessment(now=now)
        return assessment.blocked

    def get_next_event(self) -> Optional[NewsEvent]:
        """
        Return the nearest upcoming high-impact event, or None.

        Compatible with NewsFilter._minutes_to_next_event() which expects
        an object with a `.time` attribute (datetime).
        """
        events = self._get_calendar(high_only=True)
        upcoming = sorted(
            [e for e in events if e.is_upcoming],
            key=lambda e: e.time,
        )
        return upcoming[0] if upcoming else None

    def get_upcoming_events(self, hours_ahead: int = 2) -> list[NewsEvent]:
        """
        Return all tracked events within the next `hours_ahead` hours.

        Parameters
        ----------
        hours_ahead : How far ahead to look (default 2 hours)

        Returns
        -------
        List[NewsEvent] — sorted by time ascending
        """
        events  = self._get_calendar(high_only=False)
        cutoff  = datetime.now(timezone.utc) + timedelta(hours=hours_ahead)
        upcoming = [
            e for e in events
            if e.is_upcoming and e.time <= cutoff
        ]
        return sorted(upcoming, key=lambda e: e.time)

    def get_risk_assessment(
        self, now: Optional[datetime] = None
    ) -> NewsRiskAssessment:
        """
        Compute and return the current news risk assessment.

        Results are cached for NEWS_CACHE_TTL_MIN minutes.

        Parameters
        ----------
        now : Override current UTC time (for testing)

        Returns
        -------
        NewsRiskAssessment — always returns a valid object, never raises
        """
        now = now or datetime.now(timezone.utc)

        # Return cached assessment if still fresh
        if (self._assessment_cache is not None and
                time.monotonic() - self._assessment_ts < self._cache_ttl_sec):
            cached = NewsRiskAssessment(
                news_risk   = self._assessment_cache.news_risk,
                risk_score  = self._assessment_cache.risk_score,
                blocked     = self._assessment_cache.blocked,
                reason      = self._assessment_cache.reason,
                next_event  = self._assessment_cache.next_event,
                headlines   = self._assessment_cache.headlines,
                cached      = True,
                as_of       = self._assessment_cache.as_of,
            )
            return cached

        # No keys at all → safe by default
        if not self._finnhub_key and not self._newsapi_key:
            return self._make_low_risk(
                reason="No API keys configured — news filter disabled",
                as_of=now,
            )

        # Gather scores from both sources
        calendar_score, calendar_reason, next_event = self._score_calendar(now)
        headline_score, headlines                   = self._score_headlines(now)

        total_score = min(calendar_score + headline_score, 100)

        # Classify
        if total_score >= self._high_threshold:
            risk_level = "HIGH"
            blocked    = True
            reason     = (
                calendar_reason
                if calendar_reason
                else f"High-risk headlines detected (score={total_score})"
            )
        elif total_score >= self._medium_threshold:
            risk_level = "MEDIUM"
            blocked    = False
            reason     = (
                calendar_reason
                if calendar_reason
                else f"Elevated news risk (score={total_score})"
            )
        else:
            risk_level = "LOW"
            blocked    = False
            reason     = "No high-impact events detected"

        assessment = NewsRiskAssessment(
            news_risk   = risk_level,
            risk_score  = total_score,
            blocked     = blocked,
            reason      = reason,
            next_event  = next_event,
            headlines   = [h.get("title", "") for h in headlines[:5]],
            cached      = False,
            as_of       = now,
        )

        # Cache result
        self._assessment_cache = assessment
        self._assessment_ts    = time.monotonic()

        logger.info(
            f"NewsDataProvider: risk={risk_level} "
            f"score={total_score} "
            f"cal={calendar_score} "
            f"hl={headline_score} "
            f"blocked={blocked}"
        )

        return assessment

    def clear_cache(self) -> None:
        """Manually invalidate all caches (useful after bot restart)."""
        self._calendar_cache   = []
        self._calendar_ts      = 0.0
        self._headlines_cache  = []
        self._headlines_ts     = 0.0
        self._assessment_cache = None
        self._assessment_ts    = 0.0
        logger.info("NewsDataProvider: caches cleared")

    # =========================================================================
    # SCORING — Calendar (Finnhub)
    # =========================================================================

    def _score_calendar(
        self, now: datetime
    ) -> tuple[int, str, Optional[NewsEvent]]:
        """
        Score scheduled economic events from the Finnhub calendar.

        Returns
        -------
        (score: int, reason: str, next_high_impact_event: NewsEvent | None)
        """
        events = self._get_calendar(high_only=False)
        if not events:
            return 0, "", None

        score      = 0
        reason     = ""
        next_event = None

        for event in sorted(events, key=lambda e: e.time):
            mins = (event.time - now).total_seconds() / 60.0

            # Currency weight
            ccy_weight = (
                1.0 if event.currency in ("USD", "XAU")
                else 0.5 if event.currency in ("EUR", "GBP")
                else 0.2
            )

            # Blackout window (pre + post)
            in_pre_window  = 0 < mins <= self._blackout_pre
            in_post_window = -self._blackout_post <= mins <= 0

            if event.is_high_impact and (in_pre_window or in_post_window):
                delta = int(75 * ccy_weight)   # 75 pts for USD in blackout window
                score += delta
                window_desc = (
                    f"in {mins:.0f} min" if in_pre_window
                    else f"{abs(mins):.0f} min ago"
                )
                reason = (
                    f"{event.name} [{event.currency}/high] "
                    f"{window_desc}"
                )
                if next_event is None or event.time < next_event.time:
                    next_event = event

            elif event.is_high_impact and 0 < mins <= self._lookahead_min:
                delta = int(25 * ccy_weight)
                score += delta
                if next_event is None or event.time < next_event.time:
                    next_event = event

            elif event.impact.lower() == "medium" and in_pre_window:
                delta = int(20 * ccy_weight)
                score += delta

        return min(score, 90), reason, next_event   # cap calendar at 90

    # =========================================================================
    # SCORING — Headlines (NewsAPI)
    # =========================================================================

    def _score_headlines(
        self, now: datetime
    ) -> tuple[int, list[dict]]:
        """
        Score raw news headlines using keyword family matching.

        Returns
        -------
        (score: int, matched_articles: list[dict])
        """
        articles = self._get_headlines()
        if not articles:
            return 0, []

        total_score   = 0
        matched       = []
        families_hit  = set()

        for article in articles:
            title_body = (
                (article.get("title") or "")
                + " "
                + (article.get("description") or "")
            ).lower()

            # Recency multiplier
            pub_str = article.get("publishedAt", "")
            multiplier = self._recency_multiplier(pub_str, now)

            for family_name, keywords, base_score in _KEYWORD_FAMILIES:
                if family_name in families_hit:
                    continue   # only count each family once
                if any(kw in title_body for kw in keywords):
                    contrib = int(base_score * multiplier)
                    total_score  += contrib
                    families_hit.add(family_name)
                    if article not in matched:
                        matched.append(article)

        return min(total_score, 60), matched   # cap headlines at 60

    # =========================================================================
    # CALENDAR FETCHER (Finnhub)
    # =========================================================================

    def _get_calendar(self, high_only: bool = True) -> list[NewsEvent]:
        """
        Return Finnhub calendar events, using the cache when valid.

        Parameters
        ----------
        high_only : If True, only returns "high" impact events
        """
        if not self._finnhub_key:
            return []

        # Use cache if fresh
        if (self._calendar_cache and
                time.monotonic() - self._calendar_ts < self._cache_ttl_sec):
            events = self._calendar_cache
        else:
            events = self._fetch_finnhub_calendar()
            self._calendar_cache = events
            self._calendar_ts    = time.monotonic()

        if high_only:
            events = [e for e in events if e.is_high_impact]

        # Only keep tracked currencies
        events = [
            e for e in events
            if e.currency.upper() in self._tracked_ccy
            or e.currency.upper() in ("USD", "EUR", "GBP")
        ]

        return events

    def _fetch_finnhub_calendar(self) -> list[NewsEvent]:
        """
        Call the Finnhub economic calendar API and return a list of NewsEvents.

        Endpoint: GET /calendar/economic?from=YYYY-MM-DD&to=YYYY-MM-DD
        """
        if not _REQUESTS_AVAILABLE:
            return []

        now   = datetime.now(timezone.utc)
        from_ = now.strftime("%Y-%m-%d")
        to_   = (now + timedelta(days=1)).strftime("%Y-%m-%d")

        params = {
            "from":  from_,
            "to":    to_,
            "token": self._finnhub_key,
        }

        try:
            resp = _requests.get(
                FINNHUB_CALENDAR_URL, params=params, timeout=HTTP_TIMEOUT
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning(f"Finnhub calendar fetch failed: {exc}")
            return []

        events: list[NewsEvent] = []
        raw_events = data.get("economicCalendar", []) or []

        for item in raw_events:
            try:
                event_time = self._parse_finnhub_time(item.get("time", ""))
                if event_time is None:
                    continue

                currency = (item.get("country") or "").upper()
                impact   = (item.get("impact") or "low").lower()
                name     = item.get("event") or item.get("name") or "Unknown Event"

                # Remap Finnhub country codes to currency codes
                _country_to_ccy = {
                    "US": "USD", "EU": "EUR", "GB": "GBP",
                    "CA": "CAD", "AU": "AUD", "JP": "JPY",
                }
                currency = _country_to_ccy.get(currency, currency)

                events.append(NewsEvent(
                    name     = name,
                    currency = currency,
                    impact   = impact,
                    time     = event_time,
                    actual   = str(item.get("actual", "")) or None,
                    estimate = str(item.get("estimate", "")) or None,
                ))
            except Exception as exc:
                logger.debug(f"Skipping malformed Finnhub event: {exc}")
                continue

        logger.info(
            f"Finnhub calendar: fetched {len(events)} events "
            f"for {from_}→{to_}"
        )
        return events

    @staticmethod
    def _parse_finnhub_time(time_str: str) -> Optional[datetime]:
        """Parse Finnhub timestamp string to UTC datetime."""
        if not time_str:
            return None
        formats = [
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d",
        ]
        for fmt in formats:
            try:
                dt = datetime.strptime(time_str, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc)
            except ValueError:
                continue
        return None

    # =========================================================================
    # HEADLINE FETCHER (NewsAPI)
    # =========================================================================

    def _get_headlines(self) -> list[dict]:
        """Return NewsAPI headlines, using the cache when valid."""
        if not self._newsapi_key:
            return []

        if (self._headlines_cache and
                time.monotonic() - self._headlines_ts < self._cache_ttl_sec):
            return self._headlines_cache

        articles = self._fetch_newsapi_headlines()
        self._headlines_cache = articles
        self._headlines_ts    = time.monotonic()
        return articles

    def _fetch_newsapi_headlines(self) -> list[dict]:
        """
        Call NewsAPI /v2/everything with a Gold-relevant keyword query.

        We build a combined OR query from the most market-moving keyword
        families to minimise API calls while maximising relevant results.
        """
        if not _REQUESTS_AVAILABLE:
            return []

        query = (
            "Federal Reserve OR FOMC OR \"interest rates\" "
            "OR CPI OR inflation OR \"non-farm payroll\" "
            "OR war OR sanctions OR geopolitical "
            "OR USD OR \"dollar index\" "
            "OR gold price OR XAUUSD"
        )

        params = {
            "q":        query,
            "language": "en",
            "sortBy":   "publishedAt",
            "pageSize": NEWSAPI_MAX_ARTICLES,
            "apiKey":   self._newsapi_key,
        }

        try:
            resp = _requests.get(
                NEWSAPI_EVERYTHING_URL, params=params, timeout=HTTP_TIMEOUT
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning(f"NewsAPI fetch failed: {exc}")
            return []

        articles = data.get("articles", []) or []
        logger.info(f"NewsAPI: fetched {len(articles)} headlines")
        return articles

    # =========================================================================
    # HELPERS
    # =========================================================================

    @staticmethod
    def _recency_multiplier(pub_str: str, now: datetime) -> float:
        """
        Return a 0–1 multiplier based on how recently the article was published.

        < 1 h  → 1.00   (very fresh — full weight)
        < 4 h  → 0.60
        < 12 h → 0.30
        older  → 0.10
        """
        if not pub_str:
            return 0.30   # unknown → moderate

        # Try ISO 8601: "2024-03-20T18:00:00Z"
        try:
            dt_str  = pub_str.replace("Z", "+00:00")
            pub_dt  = datetime.fromisoformat(dt_str)
            if pub_dt.tzinfo is None:
                pub_dt = pub_dt.replace(tzinfo=timezone.utc)
            hours_ago = (now - pub_dt).total_seconds() / 3600.0
            if hours_ago < 1:   return 1.00
            if hours_ago < 4:   return 0.60
            if hours_ago < 12:  return 0.30
            return 0.10
        except Exception:
            return 0.30

    def _make_low_risk(self, reason: str, as_of: datetime) -> NewsRiskAssessment:
        """Return a canned LOW-risk assessment with no API calls."""
        return NewsRiskAssessment(
            news_risk   = "LOW",
            risk_score  = 0,
            blocked     = False,
            reason      = reason,
            next_event  = None,
            headlines   = [],
            cached      = False,
            as_of       = as_of,
        )

    # =========================================================================
    # LEGACY compatibility shims (keep old method names working)
    # =========================================================================

    def get_upcoming_events_df(self, hours_ahead: int = 12):
        """
        Return upcoming events as a pandas DataFrame.
        Legacy interface kept for backward compatibility with old callers.
        """
        try:
            import pandas as pd
        except ImportError:
            return None

        events = self.get_upcoming_events(hours_ahead=hours_ahead)
        if not events:
            return pd.DataFrame(columns=["event_name", "currency", "impact", "event_time_utc"])

        return pd.DataFrame([{
            "event_name":    e.name,
            "currency":      e.currency,
            "impact":        e.impact,
            "event_time_utc": e.time,
        } for e in events])
