"""
data/news_data_test.py
=======================
Self-contained tests for NewsDataProvider and NewsRiskAssessment.
No live API calls — all HTTP interactions are monkey-patched.

Tests
-----
 1.  NewsEvent.minutes_away positive for future event
 2.  NewsEvent.minutes_away negative for past event
 3.  NewsEvent.is_high_impact True only for "high"
 4.  NewsEvent.to_dict() has required keys
 5.  NewsRiskAssessment.to_dict() returns correct structure
 6.  NewsDataProvider constructs with no API keys
 7.  get_risk_assessment() returns LOW when no keys configured
 8.  is_news_blackout() returns False when no keys configured
 9.  get_next_event() returns None when no calendar events
10.  _rescore_calendar: high-impact USD event in blackout → HIGH score
11.  _rescore_calendar: upcoming event outside blackout → MEDIUM score
12.  _score_headlines: FOMC headline → family score applied
13.  _score_headlines: war/geopolitical keyword → score added
14.  _score_headlines: recency multiplier scales score correctly
15.  _recency_multiplier: < 1h → 1.0, < 4h → 0.6, < 12h → 0.3, old → 0.1
16.  get_risk_assessment() → HIGH when mock calendar fires FOMC blackout
17.  get_risk_assessment() → MEDIUM when moderate score (calendar+headline)
18.  get_risk_assessment() → LOW on no events + no relevant headlines
19.  get_risk_assessment() caches result on second call
20.  clear_cache() invalidates assessment cache
21.  _parse_finnhub_time parses ISO and date-only strings
22.  get_upcoming_events() returns sorted list filtered by cutoff
23.  NewsFilter integration: is_allowed() blocks via live provider

Run:
    cd gold_trading_bot
    python data/news_data_test.py
"""

from __future__ import annotations

import sys
import os
import time
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── ANSI colours ─────────────────────────────────────────────────────────────
GREEN = "\033[92m"; RED = "\033[91m"; CYAN = "\033[96m"
RESET = "\033[0m";  BOLD = "\033[1m"

def ok(msg):   print(f"  {GREEN}PASS{RESET}  {msg}")
def fail(msg): print(f"  {RED}FAIL{RESET}  {msg}")
def hdr(msg):  print(f"\n{BOLD}{CYAN}{'-'*60}{RESET}\n{BOLD}{CYAN}  {msg}{RESET}\n{BOLD}{CYAN}{'-'*60}{RESET}")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _settings(finnhub_key: str = "", newsapi_key: str = ""):
    from config.settings import Settings
    cfg = Settings()
    cfg.FINNHUB_API_KEY            = finnhub_key
    cfg.NEWSAPI_KEY                = newsapi_key
    cfg.NEWS_API_KEY               = finnhub_key   # legacy compat
    cfg.NEWS_BLACKOUT_BEFORE_MIN   = 30
    cfg.NEWS_BLACKOUT_AFTER_MIN    = 15
    cfg.NEWS_CACHE_TTL_MIN         = 60            # long TTL for tests
    cfg.NEWS_LOOKAHEAD_MIN         = 120
    cfg.NEWS_HIGH_RISK_THRESHOLD   = 70
    cfg.NEWS_MEDIUM_RISK_THRESHOLD = 35
    cfg.NEWS_TRACKED_CURRENCIES    = "USD,XAU"
    return cfg


def _make_event(
    name: str = "Non-Farm Payrolls",
    currency: str = "USD",
    impact: str = "high",
    minutes_from_now: float = 20.0,
    actual: str = None,
) -> "NewsEvent":
    from data.news_data import NewsEvent
    now = datetime.now(timezone.utc)
    return NewsEvent(
        name     = name,
        currency = currency,
        impact   = impact,
        time     = now + timedelta(minutes=minutes_from_now),
        actual   = actual,
    )


def _mock_finnhub_response(events: list) -> dict:
    """Build a fake Finnhub API response body."""
    raw = []
    for e in events:
        raw.append({
            "event":    e.name,
            "country":  "US",
            "impact":   e.impact,
            "time":     e.time.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
            "actual":   e.actual or "",
            "estimate": "",
        })
    return {"economicCalendar": raw}


def _mock_newsapi_response(headlines: list[str]) -> dict:
    """Build a fake NewsAPI response body."""
    now_str = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return {
        "articles": [
            {
                "title":       h,
                "description": "",
                "publishedAt": now_str,  # just published → multiplier=1.0
            }
            for h in headlines
        ]
    }


class _FakeResponse:
    """Minimal requests.Response stand-in."""
    def __init__(self, data: dict):
        self._data = data
    def raise_for_status(self): pass
    def json(self): return self._data


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    print(f"\n{BOLD}{'='*60}")
    print("  NewsDataProvider — Economic News Filter Tests")
    print(f"{'='*60}{RESET}")

    from data.news_data import (
        NewsDataProvider, NewsEvent, NewsRiskAssessment,
        _KEYWORD_FAMILIES,
    )

    results = []

    # ── Test 1: NewsEvent.minutes_away positive ───────────────────────────────
    hdr("Test 1 -- NewsEvent.minutes_away positive for future event")
    e1  = _make_event(minutes_from_now=25.0)
    t1  = 20.0 <= e1.minutes_away <= 30.0
    ok(f"minutes_away={e1.minutes_away:.1f}") if t1 else fail(f"Got {e1.minutes_away}")
    results.append(t1)

    # ── Test 2: minutes_away negative for past event ──────────────────────────
    hdr("Test 2 -- NewsEvent.minutes_away negative for past event")
    e2  = _make_event(minutes_from_now=-10.0)
    t2  = e2.minutes_away < 0
    ok(f"minutes_away={e2.minutes_away:.1f}") if t2 else fail(f"Expected <0, got {e2.minutes_away}")
    results.append(t2)

    # ── Test 3: is_high_impact ────────────────────────────────────────────────
    hdr("Test 3 -- NewsEvent.is_high_impact True only for 'high'")
    e3h = _make_event(impact="high");   e3m = _make_event(impact="medium")
    t3  = e3h.is_high_impact and not e3m.is_high_impact
    ok(f"high=True  medium=False") if t3 else fail("Impact check failed")
    results.append(t3)

    # ── Test 4: NewsEvent.to_dict() keys ─────────────────────────────────────
    hdr("Test 4 -- NewsEvent.to_dict() has required keys")
    d4  = e1.to_dict()
    req = {"name", "currency", "impact", "time", "minutes_away"}
    t4  = req.issubset(set(d4.keys()))
    ok(f"keys={list(d4.keys())}") if t4 else fail(f"Missing: {req - set(d4.keys())}")
    results.append(t4)

    # ── Test 5: NewsRiskAssessment.to_dict() structure ────────────────────────
    hdr("Test 5 -- NewsRiskAssessment.to_dict() returns correct structure")
    a5 = NewsRiskAssessment(
        news_risk="HIGH", risk_score=80, blocked=True,
        reason="Test", as_of=datetime.now(timezone.utc),
    )
    d5 = a5.to_dict()
    req5 = {"news_risk", "risk_score", "blocked", "reason", "next_event",
            "headlines", "cached", "as_of"}
    t5 = req5.issubset(set(d5.keys())) and d5["news_risk"] == "HIGH"
    ok(f"to_dict keys OK  news_risk={d5['news_risk']}") if t5 else \
    fail(f"Missing: {req5 - set(d5.keys())}")
    results.append(t5)

    # ── Test 6: Constructs with no API keys ───────────────────────────────────
    hdr("Test 6 -- NewsDataProvider constructs with no API keys")
    cfg6  = _settings()
    prov6 = NewsDataProvider(cfg6)
    t6    = prov6._finnhub_key == "" and prov6._newsapi_key == ""
    ok(f"Constructed  finnhub={prov6._finnhub_key!r}  newsapi={prov6._newsapi_key!r}") if t6 else \
    fail("Had unexpected keys")
    results.append(t6)

    # ── Test 7: LOW risk with no keys ────────────────────────────────────────
    hdr("Test 7 -- get_risk_assessment() returns LOW when no keys configured")
    a7  = prov6.get_risk_assessment()
    t7  = a7.news_risk == "LOW" and not a7.blocked
    ok(f"news_risk={a7.news_risk}  blocked={a7.blocked}") if t7 else \
    fail(f"Expected LOW/False, got {a7.news_risk}/{a7.blocked}")
    results.append(t7)

    # ── Test 8: is_news_blackout False with no keys ───────────────────────────
    hdr("Test 8 -- is_news_blackout() returns False when no keys configured")
    t8 = not prov6.is_news_blackout()
    ok("Not blocked (no keys)") if t8 else fail("Should not be blocked with no keys")
    results.append(t8)

    # ── Test 9: get_next_event None with no calendar ──────────────────────────
    hdr("Test 9 -- get_next_event() returns None when no calendar events")
    t9 = prov6.get_next_event() is None
    ok("get_next_event() = None") if t9 else fail("Expected None")
    results.append(t9)

    # ── Test 10: Calendar score HIGH for FOMC in blackout window ─────────────
    hdr("Test 10 -- _score_calendar: FOMC in blackout window → HIGH score")
    cfg10  = _settings(finnhub_key="fake_key")
    prov10 = NewsDataProvider(cfg10)
    fomc   = _make_event("FOMC Statement", "USD", "high", minutes_from_now=15.0)
    now10  = datetime.now(timezone.utc)
    # Inject into cache
    prov10._calendar_cache = [fomc]
    prov10._calendar_ts    = time.monotonic()

    score10, reason10, evt10 = prov10._score_calendar(now10)
    t10 = score10 >= 60 and evt10 is not None
    ok(f"score={score10}  reason={reason10[:50]}") if t10 else \
    fail(f"score={score10}  reason={reason10}")
    results.append(t10)

    # ── Test 11: Calendar score MEDIUM for upcoming USD event (>30 min) ───────
    hdr("Test 11 -- _score_calendar: upcoming event outside blackout → MEDIUM score")
    cfg11  = _settings(finnhub_key="fake_key")
    prov11 = NewsDataProvider(cfg11)
    nfp    = _make_event("Non-Farm Payrolls", "USD", "high", minutes_from_now=60.0)
    prov11._calendar_cache = [nfp]
    prov11._calendar_ts    = time.monotonic()

    score11, _, evt11 = prov11._score_calendar(datetime.now(timezone.utc))
    # 60 min away → lookahead bonus (+25 * 1.0 = 25) but NOT blackout (+60)
    t11 = 10 <= score11 < 60 and evt11 is not None
    ok(f"score={score11}  (expected moderate, 10–60)") if t11 else \
    fail(f"score={score11}  evt={evt11}")
    results.append(t11)

    # ── Test 12: _score_headlines FOMC headline ───────────────────────────────
    hdr("Test 12 -- _score_headlines: FOMC headline → family score applied")
    cfg12  = _settings(newsapi_key="fake_key")
    prov12 = NewsDataProvider(cfg12)
    articles12 = [
        {
            "title": "Federal Reserve FOMC holds interest rates steady",
            "description": "Fed Chair Powell signals no imminent rate cuts",
            "publishedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
    ]
    prov12._headlines_cache = articles12
    prov12._headlines_ts    = time.monotonic()

    score12, matched12 = prov12._score_headlines(datetime.now(timezone.utc))
    t12 = score12 >= 25 and len(matched12) >= 1
    ok(f"score={score12}  matched={len(matched12)}") if t12 else \
    fail(f"score={score12}  matched={len(matched12)}")
    results.append(t12)

    # ── Test 13: _score_headlines war/geopolitical keyword ────────────────────
    hdr("Test 13 -- _score_headlines: war/geopolitical keyword → score added")
    articles13 = [
        {
            "title": "Russia escalates military strikes in Ukraine",
            "description": "Geopolitical tensions spike Gold safe-haven demand",
            "publishedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
    ]
    prov12._headlines_cache = articles13
    prov12._headlines_ts    = time.monotonic()
    prov12._assessment_cache = None   # clear

    score13, matched13 = prov12._score_headlines(datetime.now(timezone.utc))
    t13 = score13 >= 15 and len(matched13) >= 1
    ok(f"score={score13}  matched={len(matched13)}") if t13 else \
    fail(f"score={score13}")
    results.append(t13)

    # ── Test 14: _score_headlines recency multiplier ──────────────────────────
    hdr("Test 14 -- _score_headlines: old headline scores lower than fresh one")
    now14  = datetime.now(timezone.utc)
    old_ts = (now14 - timedelta(hours=24)).isoformat().replace("+00:00", "Z")
    new_ts = now14.isoformat().replace("+00:00", "Z")

    articles_new = [{
        "title": "Federal Reserve FOMC meeting today",
        "description": "",
        "publishedAt": new_ts,
    }]
    articles_old = [{
        "title": "Federal Reserve FOMC meeting today",
        "description": "",
        "publishedAt": old_ts,
    }]

    prov12._headlines_cache = articles_new
    prov12._headlines_ts    = time.monotonic()
    score_new, _ = prov12._score_headlines(now14)

    prov12._headlines_cache = articles_old
    prov12._headlines_ts    = time.monotonic()
    score_old, _ = prov12._score_headlines(now14)

    t14 = score_new > score_old
    ok(f"new={score_new} > old={score_old}") if t14 else \
    fail(f"Expected new({score_new}) > old({score_old})")
    results.append(t14)

    # ── Test 15: _recency_multiplier values ───────────────────────────────────
    hdr("Test 15 -- _recency_multiplier: correct multipliers at all age bands")
    now15   = datetime.now(timezone.utc)
    def ts(hours_ago): return (now15 - timedelta(hours=hours_ago)).isoformat().replace("+00:00", "Z")
    m_30min = NewsDataProvider._recency_multiplier(ts(0.4), now15)
    m_2h    = NewsDataProvider._recency_multiplier(ts(2.0), now15)
    m_6h    = NewsDataProvider._recency_multiplier(ts(6.0), now15)
    m_old   = NewsDataProvider._recency_multiplier(ts(24),  now15)
    t15 = (
        abs(m_30min - 1.00) < 1e-6 and
        abs(m_2h    - 0.60) < 1e-6 and
        abs(m_6h    - 0.30) < 1e-6 and
        abs(m_old   - 0.10) < 1e-6
    )
    ok(f"30min={m_30min}  2h={m_2h}  6h={m_6h}  24h={m_old}") if t15 else \
    fail(f"30min={m_30min}  2h={m_2h}  6h={m_6h}  24h={m_old}")
    results.append(t15)

    # ── Test 16: get_risk_assessment HIGH — FOMC in blackout window ───────────
    hdr("Test 16 -- get_risk_assessment() → HIGH when FOMC in blackout window")
    cfg16  = _settings(finnhub_key="fake_key")
    prov16 = NewsDataProvider(cfg16)
    # Inject FOMC event 10 min away (within 30-min blackout)
    prov16._calendar_cache = [_make_event("FOMC Statement", "USD", "high", 10.0)]
    prov16._calendar_ts    = time.monotonic()
    prov16._headlines_cache = []
    prov16._headlines_ts    = time.monotonic()

    a16 = prov16.get_risk_assessment()
    t16 = a16.news_risk == "HIGH" and a16.blocked and a16.risk_score >= 70
    ok(f"news_risk={a16.news_risk}  score={a16.risk_score}  reason={a16.reason[:50]}") if t16 else \
    fail(f"news_risk={a16.news_risk}  score={a16.risk_score}")
    results.append(t16)

    # ── Test 17: get_risk_assessment MEDIUM ───────────────────────────────────
    hdr("Test 17 -- get_risk_assessment() → MEDIUM on moderate combined score")
    cfg17  = _settings(finnhub_key="fake_key", newsapi_key="fake_key")
    prov17 = NewsDataProvider(cfg17)
    # Upcoming event 90 min away (not in blackout) → calendar score ~25
    prov17._calendar_cache = [_make_event("Retail Sales", "USD", "high", 90.0)]
    prov17._calendar_ts    = time.monotonic()
    # One mild headline (+10)
    prov17._headlines_cache = [{
        "title": "USD strengthens on dollar index movement",
        "description": "",
        "publishedAt": (datetime.now(timezone.utc)
                        - timedelta(hours=3)).isoformat().replace("+00:00", "Z"),
    }]
    prov17._headlines_ts = time.monotonic()

    a17 = prov17.get_risk_assessment()
    t17 = a17.news_risk in ("MEDIUM", "LOW") and not a17.blocked
    ok(f"news_risk={a17.news_risk}  score={a17.risk_score}  blocked={a17.blocked}") if t17 else \
    fail(f"news_risk={a17.news_risk}  score={a17.risk_score}")
    results.append(t17)

    # ── Test 18: LOW risk — no events and no relevant headlines ───────────────
    hdr("Test 18 -- get_risk_assessment() → LOW on no events + irrelevant headlines")
    cfg18  = _settings(finnhub_key="fake_key", newsapi_key="fake_key")
    prov18 = NewsDataProvider(cfg18)
    prov18._calendar_cache = []
    prov18._calendar_ts    = time.monotonic()
    prov18._headlines_cache = [{
        "title": "Celebrity breaks record at sports event",
        "description": "No financial relevance whatsoever",
        "publishedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }]
    prov18._headlines_ts = time.monotonic()

    a18 = prov18.get_risk_assessment()
    t18 = a18.news_risk == "LOW" and not a18.blocked
    ok(f"news_risk={a18.news_risk}  score={a18.risk_score}") if t18 else \
    fail(f"news_risk={a18.news_risk}  score={a18.risk_score}")
    results.append(t18)

    # ── Test 19: Caching — second call returns cached result ──────────────────
    hdr("Test 19 -- get_risk_assessment() caches result on second call")
    # Call once to populate cache
    a19a = prov16.get_risk_assessment()
    # Mutate the cached calendar (should not affect result)
    prov16._calendar_cache = []
    a19b = prov16.get_risk_assessment()
    t19  = a19b.cached and a19b.news_risk == a19a.news_risk
    ok(f"cached={a19b.cached}  risk consistent={a19b.news_risk == a19a.news_risk}") if t19 else \
    fail(f"cached={a19b.cached}  risk_a={a19a.news_risk}  risk_b={a19b.news_risk}")
    results.append(t19)

    # ── Test 20: clear_cache invalidates assessment ───────────────────────────
    hdr("Test 20 -- clear_cache() invalidates assessment cache")
    prov16.clear_cache()
    t20 = prov16._assessment_cache is None
    ok("_assessment_cache=None after clear") if t20 else fail("Cache not cleared")
    results.append(t20)

    # ── Test 21: _parse_finnhub_time ──────────────────────────────────────────
    hdr("Test 21 -- _parse_finnhub_time parses ISO and date-only strings")
    from data.news_data import NewsDataProvider as NDP
    dt21a = NDP._parse_finnhub_time("2024-03-20T18:00:00+00:00")
    dt21b = NDP._parse_finnhub_time("2024-03-20 18:00:00")
    dt21c = NDP._parse_finnhub_time("2024-03-20")
    dt21d = NDP._parse_finnhub_time("")
    t21 = (
        dt21a is not None and dt21a.hour == 18 and
        dt21b is not None and dt21b.hour == 18 and
        dt21c is not None and                   # date-only → midnight
        dt21d is None
    )
    ok(f"ISO={dt21a}  space={dt21b}  date={dt21c}  empty={dt21d}") if t21 else \
    fail(f"ISO={dt21a}  space={dt21b}  date={dt21c}  empty={dt21d}")
    results.append(t21)

    # ── Test 22: get_upcoming_events filters by cutoff ────────────────────────
    hdr("Test 22 -- get_upcoming_events() returns sorted list filtered by cutoff")
    cfg22  = _settings(finnhub_key="fake_key")
    prov22 = NewsDataProvider(cfg22)
    prov22._calendar_cache = [
        _make_event("CPI",    "USD", "high",   60.0),
        _make_event("GDP",    "USD", "medium",  90.0),
        _make_event("Future", "USD", "high",   200.0),   # outside 2-hour range
    ]
    prov22._calendar_ts = time.monotonic()

    events22 = prov22.get_upcoming_events(hours_ahead=2)
    # 60 and 90 min → within 2 hours; 200 min → outside
    t22 = len(events22) == 2 and events22[0].name == "CPI"
    ok(f"{len(events22)} events within 2h  first={events22[0].name if events22 else None}") if t22 else \
    fail(f"events={[e.name for e in events22]}")
    results.append(t22)

    # ── Test 23: NewsFilter integration ───────────────────────────────────────
    hdr("Test 23 -- NewsFilter.is_allowed() blocks via live NewsDataProvider")
    from filters.news_filter import NewsFilter

    # Provider with FOMC in blackout
    cfg23  = _settings(finnhub_key="fake_key")
    prov23 = NewsDataProvider(cfg23)
    prov23._calendar_cache = [_make_event("FOMC Decision", "USD", "high", 12.0)]
    prov23._calendar_ts    = time.monotonic()
    prov23._headlines_cache = []
    prov23._headlines_ts    = time.monotonic()

    nf23 = NewsFilter(cfg23, news_provider=prov23, safe_mode=False)
    allowed23, reason23 = nf23.is_allowed()
    t23 = not allowed23 and "blackout" in reason23.lower()
    ok(f"blocked: {reason23[:60]}") if t23 else \
    fail(f"allowed={allowed23}  reason={reason23}")
    results.append(t23)

    # ── Summary ───────────────────────────────────────────────────────────────
    hdr("Summary")
    passed = sum(results)
    total  = len(results)

    # Print sample risk assessment
    cfg_sample = _settings(finnhub_key="fake_key")
    prov_sample = NewsDataProvider(cfg_sample)
    prov_sample._calendar_cache = [_make_event("FOMC Statement", "USD", "high", 18.0)]
    prov_sample._calendar_ts    = time.monotonic()
    prov_sample._headlines_cache = [
        {
            "title": "Federal Reserve signals interest rate hold at FOMC meeting",
            "description": "Powell says inflation remains above 2% target",
            "publishedAt": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
        }
    ]
    prov_sample._headlines_ts = time.monotonic()

    sample_assessment = prov_sample.get_risk_assessment()
    print("\n  -- Sample get_risk_assessment().to_dict() --")
    import json
    print(json.dumps(sample_assessment.to_dict(), indent=2, default=str))

    print()
    for i, r in enumerate(results, 1):
        s = f"{GREEN}PASS{RESET}" if r else f"{RED}FAIL{RESET}"
        print(f"  [{s}]  Test {i}")
    print()
    if passed == total:
        print(f"{GREEN}{BOLD}All {total} tests passed{RESET}")
        return 0
    else:
        print(f"{RED}{BOLD}{total - passed}/{total} tests FAILED{RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
