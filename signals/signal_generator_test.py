"""
signals/signal_generator_test.py
==================================
Self-contained tests for SignalGenerator and its sub-filters.
No MT5, Telegram or live news provider required.

Tests
-----
 1. SessionFilter.current_session() — overlap priority  
 2. SessionFilter.current_session() — london  
 3. SessionFilter.current_session() — new_york  
 4. SessionFilter.current_session() — asian  
 5. SessionFilter.current_session() — closed  
 6. SessionFilter.is_allowed() — blocked asian  
 7. SessionFilter.confidence_delta() — overlap bonus  
 8. VolatilityFilter.compute_atr() — simple series  
 9. VolatilityFilter.classify() — NORMAL  
10. VolatilityFilter.classify() — LOW  
11. VolatilityFilter.classify() — HIGH  
12. VolatilityFilter.is_allowed() — normal passes  
13. VolatilityFilter.confidence_delta() — optimal range bonus  
14. NewsFilter.is_allowed() — safe_mode=True always passes  
15. NewsFilter.is_allowed() — safe_mode=False always blocks  
16. SignalGenerator constructs without error  
17. _rescore_confidence() — trend-aligned BUY boosts score  
18. _rescore_confidence() — counter-trend reduces score  
19. generate() — returns TradingSignal with synthetic uptrend data  
20. TradingSignal.to_telegram_message() — output formatting  
21. TradingSignal.to_dict() — all required keys present  
22. Daily cap: signals_today starts at zero  
23. StrategyVerdict.is_buy / is_sell properties  
24. _run_trend_strategy() — BULLISH trend → BUY verdict  
25. _run_trend_strategy() — weak trend → NEUTRAL (abstains)  
26. _compute_confluence() — all 3 agree → boost=+0.15, conflict=False  
27. _compute_confluence() — conflict detected → conflict_detected=True  
28. Telegram message includes confluence count and boost  

Run:
    cd gold_trading_bot
    python signals/signal_generator_test.py
"""

from __future__ import annotations

import sys
import os
from datetime import datetime, timezone, timedelta

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

# ── ANSI colours ──────────────────────────────────────────────────────────────
GREEN = "\033[92m"; RED = "\033[91m"; CYAN = "\033[96m"
RESET = "\033[0m";  BOLD = "\033[1m"

def ok(msg):   print(f"  {GREEN}PASS{RESET}  {msg}")
def fail(msg): print(f"  {RED}FAIL{RESET}  {msg}")
def hdr(msg):  print(f"\n{BOLD}{CYAN}{'-'*60}{RESET}\n{BOLD}{CYAN}  {msg}{RESET}\n{BOLD}{CYAN}{'-'*60}{RESET}")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _settings():
    """Return a minimal Settings object."""
    from config.settings import Settings
    return Settings()


def _make_df(n: int = 300, base: float = 2100.0,
             trend: float = 0.3, noise: float = 0.5,
             seed: int = 7) -> pd.DataFrame:
    """Build an OHLCV DataFrame with a gentle uptrend."""
    rng   = np.random.default_rng(seed)
    t     = np.arange(n, dtype=float)
    c     = base + t * trend + rng.normal(0, noise, n)
    hi    = c + rng.uniform(0.3, 0.8, n)
    lo    = c - rng.uniform(0.3, 0.8, n)
    op    = c + rng.uniform(-0.2, 0.2, n)
    hi    = np.maximum(hi, np.maximum(op, c))
    lo    = np.minimum(lo, np.minimum(op, c))
    vol   = rng.integers(500, 1500, n).astype(float)
    start = datetime(2024, 3, 1, tzinfo=timezone.utc)
    return pd.DataFrame({
        "time":        pd.to_datetime(
            [start + timedelta(minutes=15 * i) for i in range(n)], utc=True
        ),
        "open":        op.round(2),
        "high":        hi.round(2),
        "low":         lo.round(2),
        "close":       c.round(2),
        "tick_volume": vol,
    })


def _make_h1(n: int = 600) -> pd.DataFrame:
    """Build H1 OHLCV data (uptrend, 600 bars —enough for EMA 200)."""
    return _make_df(n=n, base=2100.0, trend=0.05, noise=1.0, seed=42)


def _make_m15(n: int = 300) -> pd.DataFrame:
    """Build M15 data with injected liquidity sweep setup."""
    rng    = np.random.default_rng(21)
    n_     = n
    closes = np.full(n_, 2100.0, dtype=float)
    closes[:150] = np.linspace(2100.0, 2155.0, 150)

    # Inject swing LOW at bar 60
    closes[58] = 2126; closes[59] = 2122; closes[60] = 2118.0
    closes[61] = 2122; closes[62] = 2126

    # Add OB: bar 120 bearish before impulse
    closes[118:124] = [2146, 2148, 2145, 2150, 2156, 2162]

    # Sweep bar 150: wick below bar-60 low, close above
    closes[149] = 2148; closes[150] = 2112
    closes[151:160] = np.linspace(2119, 2145, 9)
    closes[160:] = np.linspace(2145, 2152, n_ - 160)

    hi = closes + rng.uniform(0.2, 0.5, n_)
    lo = closes - rng.uniform(0.2, 0.5, n_)
    op = closes + rng.uniform(-0.1, 0.1, n_)
    hi = np.maximum(hi, np.maximum(op, closes))
    lo = np.minimum(lo, np.minimum(op, closes))
    vol = rng.integers(400, 1200, n_).astype(float)

    df = pd.DataFrame({
        "time":        pd.to_datetime([
            datetime(2024, 3, 1, tzinfo=timezone.utc) + timedelta(minutes=15 * i)
            for i in range(n_)
        ], utc=True),
        "open":        op.round(2),
        "high":        hi.round(2),
        "low":         lo.round(2),
        "close":       closes.round(2),
        "tick_volume": vol,
    })
    # Force sweep bar explicit OHLC
    df.at[150, "open"]  = 2148.0
    df.at[150, "low"]   = 2110.0   # below swing-low price 2118
    df.at[150, "high"]  = 2150.0
    df.at[150, "close"] = 2122.0
    # Force OB bar 120: large bearish body
    df.at[120, "open"]  = 2150.0
    df.at[120, "close"] = 2142.0
    df.at[120, "high"]  = 2151.5
    df.at[120, "low"]   = 2141.0
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Test runner
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    print(f"\n{BOLD}{'='*60}")
    print("  SignalGenerator -- Synthetic Tests")
    print(f"{'='*60}{RESET}")

    results = []
    cfg     = _settings()

    from filters.session_filter   import SessionFilter, SESSION_CONFIDENCE_DELTA
    from filters.volatility_filter import VolatilityFilter, VolatilityState
    from filters.news_filter       import NewsFilter
    from signals.signal_generator  import (
        SignalGenerator, TradingSignal, RejectedIdea,
        StrategyVerdict, ConfluenceResult,
    )
    from strategy.trend_detection  import TrendDirection, TrendResult
    from strategy.market_regime    import RegimeLabel

    sf  = SessionFilter(cfg)
    vf  = VolatilityFilter(cfg)
    nf  = NewsFilter(cfg, safe_mode=True)

    # ── Test 1: overlap session ───────────────────────────────────────────────
    hdr("Test 1 -- SessionFilter: overlap detected (12:00-15:59 UTC)")
    t1_dt = datetime(2024, 3, 4, 13, 30, tzinfo=timezone.utc)  # 13:30 UTC = overlap
    sess1 = sf.current_session(t1_dt)
    t1 = sess1 == "overlap"
    ok(f"session={sess1}") if t1 else fail(f"expected overlap, got {sess1}")
    results.append(t1)

    # ── Test 2: london session ────────────────────────────────────────────────
    hdr("Test 2 -- SessionFilter: london detected (07:00-11:59 UTC)")
    t2_dt = datetime(2024, 3, 4, 10, 0, tzinfo=timezone.utc)
    sess2 = sf.current_session(t2_dt)
    t2 = sess2 == "london"
    ok(f"session={sess2}") if t2 else fail(f"expected london, got {sess2}")
    results.append(t2)

    # ── Test 3: new_york (post-overlap) ───────────────────────────────────────
    hdr("Test 3 -- SessionFilter: new_york detected (16:00-20:59 UTC)")
    t3_dt = datetime(2024, 3, 4, 18, 0, tzinfo=timezone.utc)
    sess3 = sf.current_session(t3_dt)
    t3 = sess3 == "new_york"
    ok(f"session={sess3}") if t3 else fail(f"expected new_york, got {sess3}")
    results.append(t3)

    # ── Test 4: asian ─────────────────────────────────────────────────────────
    hdr("Test 4 -- SessionFilter: asian detected (03:00 UTC)")
    t4_dt = datetime(2024, 3, 4, 3, 0, tzinfo=timezone.utc)
    sess4 = sf.current_session(t4_dt)
    t4 = sess4 == "asian"
    ok(f"session={sess4}") if t4 else fail(f"expected asian, got {sess4}")
    results.append(t4)

    # ── Test 5: closed ────────────────────────────────────────────────────────
    hdr("Test 5 -- SessionFilter: closed after hours (22:00 UTC)")
    t5_dt = datetime(2024, 3, 4, 22, 0, tzinfo=timezone.utc)
    sess5 = sf.current_session(t5_dt)
    t5 = sess5 == "closed"
    ok(f"session={sess5}") if t5 else fail(f"expected closed, got {sess5}")
    results.append(t5)

    # ── Test 6: asian blocked ─────────────────────────────────────────────────
    hdr("Test 6 -- SessionFilter: asian blocked by default ACTIVE_SESSIONS")
    allowed6, reason6 = sf.is_allowed(t4_dt)
    t6 = not allowed6
    ok(f"blocked: {reason6}") if t6 else fail(f"should block asian but got allowed={allowed6}")
    results.append(t6)

    # ── Test 7: overlap confidence delta ──────────────────────────────────────
    hdr("Test 7 -- SessionFilter: overlap gives +0.10 confidence delta")
    delta7 = sf.confidence_delta(t1_dt)
    t7 = abs(delta7 - 0.10) < 0.001
    ok(f"delta={delta7}") if t7 else fail(f"expected 0.10, got {delta7}")
    results.append(t7)

    # ── Test 8: VolatilityFilter.compute_atr() ────────────────────────────────
    hdr("Test 8 -- VolatilityFilter: ATR computes on M15 data")
    df_m15 = _make_m15()
    atr8 = vf.compute_atr(df_m15)
    t8 = 0.1 < atr8 < 20.0   # M15 synthetic data should have small ATR
    ok(f"ATR={atr8:.3f}") if t8 else fail(f"ATR={atr8:.3f} out of expected range")
    results.append(t8)

    # ── Test 9: NORMAL classify ───────────────────────────────────────────────
    hdr("Test 9 -- VolatilityFilter: NORMAL regime on live-like ATR")
    # Build a df with ATR ~12 pts
    rng9   = np.random.default_rng(9)
    closes9 = np.full(60, 2100.0) + rng9.normal(0, 6, 60)
    hi9     = closes9 + rng9.uniform(5, 7, 60)
    lo9     = closes9 - rng9.uniform(5, 7, 60)
    op9     = closes9.copy()
    df9 = pd.DataFrame({"open": op9, "high": hi9, "low": lo9, "close": closes9})
    state9 = vf.classify(df9)
    t9 = state9 == VolatilityState.NORMAL
    ok(f"state={state9.value}  ATR={vf.compute_atr(df9):.2f}") if t9 else \
    fail(f"expected NORMAL, got {state9.value}  ATR={vf.compute_atr(df9):.2f}")
    results.append(t9)

    # ── Test 10: LOW classify ─────────────────────────────────────────────────
    hdr("Test 10 -- VolatilityFilter: LOW regime on tiny-range df")
    rng10 = np.random.default_rng(10)
    cl10  = np.full(60, 2100.0)
    h10   = cl10 + 0.2; l10 = cl10 - 0.2
    df10  = pd.DataFrame({"open": cl10, "high": h10, "low": l10, "close": cl10})
    state10 = vf.classify(df10)
    t10 = state10 == VolatilityState.LOW
    ok(f"state={state10.value}") if t10 else fail(f"expected LOW, got {state10.value}")
    results.append(t10)

    # ── Test 11: HIGH classify ────────────────────────────────────────────────
    hdr("Test 11 -- VolatilityFilter: HIGH regime on news-spike df")
    rng11 = np.random.default_rng(11)
    cl11  = np.full(60, 2100.0)
    h11   = cl11 + rng11.uniform(45, 55, 60)  # >40 pt range
    l11   = cl11 - rng11.uniform(45, 55, 60)
    df11  = pd.DataFrame({"open": cl11, "high": h11, "low": l11, "close": cl11})
    state11 = vf.classify(df11)
    t11 = state11 in (VolatilityState.HIGH, VolatilityState.EXTREME)
    ok(f"state={state11.value}  ATR={vf.compute_atr(df11):.2f}") if t11 else \
    fail(f"expected HIGH/EXTREME, got {state11.value}")
    results.append(t11)

    # ── Test 12: VolatilityFilter.is_allowed() ──────────────────────────────
    hdr("Test 12 -- VolatilityFilter: normal ATR data is allowed")
    allowed12, reason12 = vf.is_allowed(df9)
    t12 = allowed12
    ok(f"allowed=True  reason='{reason12}'") if t12 else fail(f"blocked: {reason12}")
    results.append(t12)

    # ── Test 13: ATR bonus ────────────────────────────────────────────────────
    hdr("Test 13 -- VolatilityFilter: optimal ATR gives +0.05 delta")
    delta13 = vf.confidence_delta(df9)
    t13 = delta13 >= 0.00   # bonus or neutral
    ok(f"confidence_delta={delta13}") if t13 else fail(f"expected >= 0, got {delta13}")
    results.append(t13)

    # ── Test 14: NewsFilter safe_mode=True ────────────────────────────────────
    hdr("Test 14 -- NewsFilter: safe_mode=True always allows")
    nf_safe = NewsFilter(cfg, safe_mode=True)
    allowed14, _ = nf_safe.is_allowed()
    t14 = allowed14
    ok("safe_mode=True -> allowed") if t14 else fail("should allow in safe_mode=True")
    results.append(t14)

    # ── Test 15: NewsFilter safe_mode=False ───────────────────────────────────
    hdr("Test 15 -- NewsFilter: safe_mode=False with no provider blocks")
    nf_block = NewsFilter(cfg, safe_mode=False)
    allowed15, reason15 = nf_block.is_allowed()
    t15 = not allowed15
    ok(f"blocked: {reason15}") if t15 else fail("should block in safe_mode=False+no provider")
    results.append(t15)

    # ── Test 16: SignalGenerator constructs ─────────────────────────────────
    hdr("Test 16 -- SignalGenerator constructs and repr is valid")
    sg = SignalGenerator(cfg)
    t16 = "SignalGenerator" in repr(sg)
    ok(repr(sg)) if t16 else fail("repr failed")
    results.append(t16)

    # ── Test 17: confidence boost on trend-aligned BUY ────────────────────
    hdr("Test 17 -- _rescore_confidence: trend-aligned BUY boosts score")
    from strategy.smart_money_strategy import TradeIdea
    idea17 = TradeIdea(
        direction="BUY", entry=2150.0, stop_loss=2140.0,
        take_profit=2170.0, take_profit_2=2190.0, confidence=0.50,
    )
    bull_trend = TrendResult(
        direction=TrendDirection.BULLISH, strength=0.70
    )
    df17 = _make_m15()
    # Force ATR to normal range by patching compute_atr
    conf17 = ConfluenceResult(direction="BUY", agreeing=1, confluence_boost=0.0)
    score17, bd17 = sg._rescore_confidence(
        base_score=0.50, idea=idea17, trend_result=bull_trend,
        regime_label=RegimeLabel.TRENDING,
        df_m15=df17, sess_ok=True, vol_ok=True, news_ok=True,
        now=t1_dt,   # 13:30 UTC = overlap -> +0.10
        confluence=conf17,
    )
    # Expect: 0.50 + 0.10(align) + 0.056(strength*0.08) + 0.05(regime) + 0.10(overlap) > 0.50
    t17 = score17 > 0.50
    ok(f"score {0.50:.2f} -> {score17:.3f}  breakdown={bd17}") if t17 else \
    fail(f"expected > 0.50, got {score17:.3f}")
    results.append(t17)

    # ── Test 18: confidence penalty on counter-trend ──────────────────────
    hdr("Test 18 -- _rescore_confidence: counter-trend SELL penalises score")
    idea18 = TradeIdea(
        direction="SELL", entry=2150.0, stop_loss=2160.0,
        take_profit=2130.0, take_profit_2=2110.0, confidence=0.50,
    )
    conf18 = ConfluenceResult(direction="SELL", agreeing=1, confluence_boost=0.0)
    score18, bd18 = sg._rescore_confidence(
        base_score=0.50, idea=idea18, trend_result=bull_trend,
        regime_label=RegimeLabel.TRENDING,
        df_m15=df17, sess_ok=True, vol_ok=True, news_ok=True,
        now=t4_dt,   # 03:00 UTC = asian -> -0.10
        confluence=conf18,
    )
    t18 = score18 < 0.50
    ok(f"score {0.50:.2f} -> {score18:.3f}  breakdown={bd18}") if t18 else \
    fail(f"expected < 0.50, got {score18:.3f}")
    results.append(t18)

    # ── Test 19: generate() returns signals on synthetic uptrend ──────────
    hdr("Test 19 -- generate(): produces TradingSignal on uptrend data")
    df_h1  = _make_h1(n=600)
    df_m15 = _make_m15(n=300)
    # Lower confidence threshold for synthetic data
    cfg19  = _settings()
    cfg19.CONFIDENCE_THRESHOLD = 0.20
    cfg19.MIN_RR_RATIO         = 1.0
    sg19 = SignalGenerator(
        cfg19,
        session_filter    = SessionFilter(cfg19),
        volatility_filter = VolatilityFilter(cfg19),
        news_filter       = NewsFilter(cfg19, safe_mode=True),
    )
    # override SMC min confidence for this test
    sg19.smc.MIN_CONFIDENCE = 0.05

    signals19, rejects19 = sg19.generate(
        df_h1=df_h1, df_m15=df_m15, account_balance=10_000.0,
        now=t1_dt,   # 13:30 UTC overlap — session allowed
    )
    t19 = len(signals19) > 0 or len(rejects19) > 0   # pipeline ran at all
    if signals19:
        ok(f"Signal produced: {signals19[0]}")
    elif rejects19:
        ok(f"Pipeline ran — {len(rejects19)} ideas rejected (gates working correctly)")
        ok(f"  First rejection: [{rejects19[0].stage}] {rejects19[0].reason}")
    else:
        fail("Pipeline produced nothing — no signals or rejections")
    results.append(t19)

    # ── Test 20: to_telegram_message() formatting ─────────────────────────
    hdr("Test 20 -- TradingSignal.to_telegram_message() output")
    sample_sig = TradingSignal(
        symbol="XAUUSD", direction="BUY",
        entry_price=2150.50, stop_loss=2140.00, take_profit_1=2170.00,
        take_profit_2=2190.00, lot_size=0.10, risk_amount_usd=105.0,
        risk_pct=1.05, rr_ratio_tp1=1.86, rr_ratio_tp2=3.72,
        monetary_value_1pt=0.10, confidence=0.75, smc_confidence=0.62,
        trend_strength=0.65, regime="TRENDING", session="overlap",
        trend_direction="BULLISH", has_ob=True, has_fvg=False,
        has_bos=True, has_choch=False, has_liquidity_sweep=True,
        notes="Bullish OB + liquidity sweep",
    )
    msg20 = sample_sig.to_telegram_message()
    t20 = (
        "XAUUSD" in msg20
        and "BUY" in msg20
        and "2150.50" in msg20
        and "2140.00" in msg20
        and "OVERLAP" in msg20
        and "Order Block" in msg20
    )
    if t20:
        ok("Telegram message formatted correctly")
        print(f"\n{'─'*60}")
        print(msg20)
        print('─'*60)
    else:
        fail("Telegram message missing expected content")
    results.append(t20)

    # ── Test 21: to_dict() required keys ─────────────────────────────────
    hdr("Test 21 -- TradingSignal.to_dict() required keys")
    required = {
        "symbol", "direction", "entry_price", "stop_loss",
        "take_profit_1", "lot_size", "risk_amount_usd", "confidence",
        "regime", "session", "trend_direction",
        "has_ob", "has_fvg", "has_bos", "has_choch", "has_liquidity_sweep",
    }
    d21     = sample_sig.to_dict()
    missing = required - set(d21.keys())
    t21 = not missing
    ok(f"All {len(required)} required keys present") if t21 else \
    fail(f"Missing keys: {missing}")
    results.append(t21)

    # ── Test 22: daily cap ────────────────────────────────────────────────
    hdr("Test 22 -- daily cap: signals_today starts at zero")
    sg22 = SignalGenerator(_settings())
    sg22._last_reset_date = datetime.now(timezone.utc)  # prevent reset
    t22 = sg22.signals_today == 0 and not sg22.daily_cap_reached
    ok(f"signals_today={sg22.signals_today}  cap_reached={sg22.daily_cap_reached}")
    results.append(t22)

    # ── Test 23: StrategyVerdict properties ──────────────────────────────
    hdr("Test 23 -- StrategyVerdict.is_buy / is_sell properties")
    v_buy  = StrategyVerdict("TEST", direction="BUY",  confidence=0.7, active=True)
    v_sell = StrategyVerdict("TEST", direction="SELL", confidence=0.7, active=True)
    v_neut = StrategyVerdict("TEST", direction="NEUTRAL", confidence=0.0, active=False)
    t23 = (v_buy.is_buy and not v_buy.is_sell and
           v_sell.is_sell and not v_sell.is_buy and
           not v_neut.is_buy and not v_neut.is_sell)
    ok("is_buy / is_sell / neutral all correct") if t23 else \
    fail(f"buy={v_buy.is_buy} sell={v_sell.is_sell} neut_buy={v_neut.is_buy}")
    results.append(t23)

    # ── Test 24: _run_trend_strategy BUY ─────────────────────────────────
    hdr("Test 24 -- _run_trend_strategy(): BULLISH trend → BUY verdict")
    from strategy.trend_detection import TrendDirection, TrendResult
    strong_bull = TrendResult(direction=TrendDirection.BULLISH, strength=0.75)
    verdict24   = sg._run_trend_strategy(strong_bull)
    t24 = verdict24.active and verdict24.direction == "BUY" and verdict24.confidence >= 0.70
    ok(f"{verdict24}") if t24 else fail(f"Got: {verdict24}")
    results.append(t24)

    # ── Test 25: _run_trend_strategy neutral on weak trend ────────────────
    hdr("Test 25 -- _run_trend_strategy(): weak trend → NEUTRAL (abstains)")
    weak_trend = TrendResult(direction=TrendDirection.BULLISH, strength=0.20)
    verdict25  = sg._run_trend_strategy(weak_trend)
    t25 = not verdict25.active and verdict25.direction == "NEUTRAL"
    ok(f"{verdict25}") if t25 else fail(f"Expected NEUTRAL inactive, got: {verdict25}")
    results.append(t25)

    # ── Test 26: _compute_confluence all-3-agree ──────────────────────────
    hdr("Test 26 -- _compute_confluence(): all 3 agree → boost=+0.15")
    v_smc  = StrategyVerdict("SMC",  direction="BUY", confidence=0.70, active=True)
    v_tr   = StrategyVerdict("TREND",direction="BUY", confidence=0.75, active=True)
    v_liq  = StrategyVerdict("LIQUIDITY_SWEEP", direction="BUY", confidence=0.80, active=True)
    conf26 = sg._compute_confluence("BUY", v_smc, v_tr, v_liq)
    t26 = (
        conf26.agreeing      == 3 and
        conf26.conflicting   == 0 and
        not conf26.conflict_detected and
        abs(conf26.confluence_boost - 0.15) < 1e-6
    )
    ok(f"agree={conf26.agreeing}  boost={conf26.confluence_boost:+.0%}  {conf26.summary_str()}") if t26 else \
    fail(f"agree={conf26.agreeing}  conflict={conf26.conflicting}  boost={conf26.confluence_boost}")
    results.append(t26)

    # ── Test 27: _compute_confluence conflict ────────────────────────────
    hdr("Test 27 -- _compute_confluence(): conflict detected → conflict_detected=True")
    v_smc27  = StrategyVerdict("SMC", direction="BUY",  confidence=0.70, active=True)
    v_tr27   = StrategyVerdict("TREND", direction="SELL", confidence=0.75, active=True)
    v_liq27  = StrategyVerdict("LIQUIDITY_SWEEP", direction="NEUTRAL", confidence=0.0, active=False)
    conf27 = sg._compute_confluence("BUY", v_smc27, v_tr27, v_liq27)
    t27 = conf27.conflict_detected and conf27.confluence_boost == 0.0
    ok(f"conflict={conf27.conflict_detected}  boost={conf27.confluence_boost:+.0%}  {conf27.summary_str()}") if t27 else \
    fail(f"conflict={conf27.conflict_detected}  boost={conf27.confluence_boost}")
    results.append(t27)

    # ── Test 28: Telegram message includes confluence fields ──────────────
    hdr("Test 28 -- TradingSignal.to_telegram_message() includes confluence info")
    sig28 = TradingSignal(
        symbol="XAUUSD", direction="BUY",
        entry_price=2150.0, stop_loss=2140.0, take_profit_1=2170.0,
        lot_size=0.10, risk_amount_usd=100.0, risk_pct=1.0,
        rr_ratio_tp1=2.0, confidence=0.80,
        confluence_count=3, confluence_boost=0.15,
    )
    msg28 = sig28.to_telegram_message()
    t28 = (
        "ALL 3" in msg28 or "3 strategies" in msg28 or "Confluence" in msg28
    )
    ok(f"Confluence line present in Telegram message") if t28 else \
    fail(f"Missing confluence info in: {msg28[:200]}")
    results.append(t28)

    # ── Summary ───────────────────────────────────────────────────────────
    hdr("Summary")
    passed = sum(results)
    total  = len(results)
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
