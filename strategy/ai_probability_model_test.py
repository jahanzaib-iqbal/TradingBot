"""
strategy/ai_probability_model_test.py
=======================================
Self-contained tests for TradeProbabilityModel.
No MT5, Telegram, or live data required.

Tests
-----
 1.  FEATURE_NAMES has exactly 12 entries
 2.  _session_to_code maps all 5 sessions correctly
 3.  _compute_atr returns positive value on valid OHLCV
 4.  _compute_atr returns 0.0 on insufficient bars
 5.  build_feature_vector_from_record — winner → label=1
 6.  build_feature_vector_from_record — loser  → label=0
 7.  build_feature_vector_from_record — expired excluded
 8.  TradeProbabilityModel constructs (sklearn available)
 9.  is_ready=False before training
10.  predict_probability passthrough (no model) → always approved
11.  train_model() raises on insufficient samples
12.  train_model() raises on missing sklearn (mocked)
13.  train_model() succeeds on synthetic trade records → returns TrainingReport
14.  TrainingReport.to_dict() has all required keys
15.  TrainingReport.print_summary() runs without error
16.  predict_probability() after training → probability in [0, 1]
17.  predict_probability() below threshold → approved=False
18.  predict_from_record() works after training
19.  save_model() saves file to disk
20.  load_model() restores model from disk
21.  PredictionResult.to_dict() has required keys
22.  evaluate_on_backtest() returns accuracy metric

Run:
    cd gold_trading_bot
    python strategy/ai_probability_model_test.py
"""

from __future__ import annotations

import sys
import os
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

# ANSI colours
GREEN = "\033[92m"; RED = "\033[91m"; CYAN = "\033[96m"
RESET = "\033[0m";  BOLD = "\033[1m"

def ok(msg):   print(f"  {GREEN}PASS{RESET}  {msg}")
def fail(msg): print(f"  {RED}FAIL{RESET}  {msg}")
def hdr(msg):  print(f"\n{BOLD}{CYAN}{'-'*60}{RESET}\n{BOLD}{CYAN}  {msg}{RESET}\n{BOLD}{CYAN}{'-'*60}{RESET}")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _settings(threshold: float = 0.60):
    from config.settings import Settings
    cfg = Settings()
    cfg.AI_PROBABILITY_THRESHOLD = threshold
    cfg.AI_MIN_TRAIN_SAMPLES     = 20     # low for testing
    cfg.AI_N_ESTIMATORS          = 10     # fast for testing
    cfg.AI_RANDOM_SEED           = 42
    cfg.AI_MODEL_PATH            = "strategy/models/test_rf.joblib"
    return cfg


def _make_ohlcv(n: int = 60, noise: float = 3.0) -> pd.DataFrame:
    rng = np.random.default_rng(99)
    c   = 2350.0 + rng.normal(0, noise, n)
    h   = c + rng.uniform(1, 4, n)
    l   = c - rng.uniform(1, 4, n)
    o   = c + rng.uniform(-1, 1, n)
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c,
                         "tick_volume": rng.integers(300, 1000, n).astype(float)})


def _make_trade_record(outcome="TP1", direction="BUY", trade_id=1,
                       session="london", regime="TRENDING",
                       has_ob=True, has_fvg=False, has_bos=True,
                       has_liq_sweep=True, confidence=0.72,
                       entry=2350.0, sl=2340.0, tp1=2370.0) -> "TradeRecord":
    from backtesting.backtest_engine import TradeRecord
    rec = TradeRecord(
        trade_id=trade_id, signal_bar=200,
        signal_time=datetime(2024, 2, 1, 10, 0, tzinfo=timezone.utc),
        direction=direction, entry=entry, stop_loss=sl,
        take_profit_1=tp1, take_profit_2=None,
        confidence=confidence,
        has_ob=has_ob, has_fvg=has_fvg, has_bos=has_bos,
        has_choch=False, has_liq_sweep=has_liq_sweep,
        session=session, regime=regime,
    )
    rec.outcome=outcome; rec.outcome_bar=210; rec.bars_held=10
    rec.r_gained = 2.0 if outcome in ("TP1","TP2") else -1.0
    return rec


def _make_signal():
    from signals.signal_generator import TradingSignal
    return TradingSignal(
        symbol="XAUUSD", direction="BUY",
        entry_price=2350.0, stop_loss=2340.0,
        take_profit_1=2370.0, take_profit_2=2390.0,
        lot_size=0.10, risk_amount_usd=100.0, risk_pct=1.0,
        rr_ratio_tp1=2.0, rr_ratio_tp2=4.0, monetary_value_1pt=0.10,
        confidence=0.72, smc_confidence=0.62, trend_strength=0.65,
        regime="TRENDING", session="overlap", trend_direction="BULLISH",
        has_ob=True, has_fvg=False, has_bos=True,
        has_choch=False, has_liquidity_sweep=True,
    )


def _make_trend_result(strength=0.65):
    from strategy.trend_detection import TrendResult, TrendDirection
    return TrendResult(
        direction=TrendDirection.BULLISH,
        strength=strength,
        ema_50=2345.0, ema_200=2330.0,
        close=2350.0,
    )


def _make_regime_result(adx=28.0):
    from strategy.market_regime import RegimeResult, RegimeLabel, TrendDirection as RD
    return RegimeResult(
        regime=RegimeLabel.TRENDING,
        trend_direction=RD.BULLISH,
        adx=adx, plus_di=22.0, minus_di=14.0,
        atr=8.5, atr_ratio=1.1, ema_slope=2.1,
        confidence=0.72,
    )


def _make_synthetic_trade_log(n_wins: int = 30, n_losses: int = 20) -> list:
    """Create a varied set of TradeRecord objects for model training."""
    records = []
    rng = np.random.default_rng(7)
    trade_id = 1
    sessions  = ["london", "new_york", "overlap", "asian"]
    regimes   = ["TRENDING", "RANGING"]

    for i in range(n_wins):
        records.append(_make_trade_record(
            outcome       = rng.choice(["TP1", "TP2"]),
            direction     = rng.choice(["BUY", "SELL"]),
            trade_id      = trade_id,
            session       = rng.choice(sessions),
            regime        = rng.choice(regimes),
            has_ob        = bool(rng.integers(0, 2)),
            has_fvg       = bool(rng.integers(0, 2)),
            has_bos       = bool(rng.integers(0, 2)),
            has_liq_sweep = bool(rng.integers(0, 2)),
            confidence    = float(rng.uniform(0.55, 0.85)),
        ))
        trade_id += 1

    for i in range(n_losses):
        records.append(_make_trade_record(
            outcome       = "SL",
            direction     = rng.choice(["BUY", "SELL"]),
            trade_id      = trade_id,
            session       = rng.choice(sessions),
            regime        = rng.choice(regimes),
            has_ob        = bool(rng.integers(0, 2)),
            has_fvg       = bool(rng.integers(0, 2)),
            has_bos       = bool(rng.integers(0, 2)),
            has_liq_sweep = bool(rng.integers(0, 2)),
            confidence    = float(rng.uniform(0.40, 0.70)),
        ))
        trade_id += 1

    return records


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    print(f"\n{BOLD}{'='*60}")
    print("  TradeProbabilityModel -- AI Predictor Tests")
    print(f"{'='*60}{RESET}")

    from strategy.ai_probability_model import (
        TradeProbabilityModel, TrainingReport, PredictionResult,
        FEATURE_NAMES, _session_to_code, _compute_atr,
        build_feature_vector, build_feature_vector_from_record,
    )

    results = []
    cfg     = _settings()

    # ── Test 1: 12 feature names ──────────────────────────────────────────────
    hdr("Test 1 -- FEATURE_NAMES has exactly 12 entries")
    t1 = len(FEATURE_NAMES) == 12
    ok(f"FEATURE_NAMES = {FEATURE_NAMES}") if t1 else \
    fail(f"Expected 12, got {len(FEATURE_NAMES)}")
    results.append(t1)

    # ── Test 2: session encoding ──────────────────────────────────────────────
    hdr("Test 2 -- _session_to_code maps all 5 sessions correctly")
    mapping = {
        "closed": 0, "asian": 1, "london": 2,
        "new_york": 3, "overlap": 4,
    }
    t2 = all(_session_to_code(s) == v for s, v in mapping.items())
    ok(f"All 5 sessions mapped correctly: {mapping}") if t2 else \
    fail(f"Session mapping wrong")
    results.append(t2)

    # ── Test 3: ATR on valid data ─────────────────────────────────────────────
    hdr("Test 3 -- _compute_atr returns positive on valid OHLCV")
    df3  = _make_ohlcv(60)
    atr3 = _compute_atr(df3)
    t3   = 0.1 < atr3 < 50.0
    ok(f"ATR={atr3:.3f}") if t3 else fail(f"ATR out of range: {atr3}")
    results.append(t3)

    # ── Test 4: ATR returns 0 on too few bars ─────────────────────────────────
    hdr("Test 4 -- _compute_atr returns 0.0 on ≤ period bars")
    df4  = _make_ohlcv(5)
    atr4 = _compute_atr(df4, period=14)
    t4   = atr4 == 0.0
    ok(f"ATR={atr4} (correctly 0.0)") if t4 else fail(f"Expected 0.0, got {atr4}")
    results.append(t4)

    # ── Test 5: winner → label=1 ──────────────────────────────────────────────
    hdr("Test 5 -- build_feature_vector_from_record: TP1 → label=1")
    rec5 = _make_trade_record(outcome="TP1")
    feat5, lab5 = build_feature_vector_from_record(rec5)
    t5 = lab5 == 1 and feat5.shape == (12,)
    ok(f"label={lab5}  feat_shape={feat5.shape}") if t5 else \
    fail(f"label={lab5}  shape={feat5.shape}")
    results.append(t5)

    # ── Test 6: loser → label=0 ───────────────────────────────────────────────
    hdr("Test 6 -- build_feature_vector_from_record: SL → label=0")
    rec6 = _make_trade_record(outcome="SL")
    feat6, lab6 = build_feature_vector_from_record(rec6)
    t6 = lab6 == 0
    ok(f"label={lab6}") if t6 else fail(f"Expected 0, got {lab6}")
    results.append(t6)

    # ── Test 7: expired excluded in _load_training_data ──────────────────────
    hdr("Test 7 -- EXPIRED outcomes are excluded from training data")
    expired_records = [
        _make_trade_record(outcome="EXPIRED", trade_id=i) for i in range(5)
    ] + [_make_trade_record(outcome="TP1", trade_id=10), _make_trade_record(outcome="SL", trade_id=11)]
    model7 = TradeProbabilityModel(cfg)
    X7, y7 = model7._load_training_data(expired_records)
    t7 = len(X7) == 2   # only 2 non-expired
    ok(f"Loaded {len(X7)} non-expired records (excluded 5 EXPIRED)") if t7 else \
    fail(f"Expected 2, got {len(X7)}")
    results.append(t7)

    # ── Test 8: constructs correctly ──────────────────────────────────────────
    hdr("Test 8 -- TradeProbabilityModel constructs without error")
    model = TradeProbabilityModel(cfg)
    t8 = repr(model).startswith("TradeProbabilityModel(")
    ok(repr(model)) if t8 else fail("repr failed")
    results.append(t8)

    # ── Test 9: is_ready=False before training ────────────────────────────────
    hdr("Test 9 -- is_ready=False before training")
    t9 = not model.is_ready
    ok("is_ready=False before training") if t9 else fail(f"is_ready={model.is_ready}")
    results.append(t9)

    # ── Test 10: passthrough predict (no model) ───────────────────────────────
    hdr("Test 10 -- predict_probability passthrough → always approved")
    sig    = _make_signal()
    trend  = _make_trend_result()
    regime = _make_regime_result()
    df_m15 = _make_ohlcv()
    pred10 = model.predict_probability(sig, trend, regime, df_m15)
    t10 = pred10.probability == 1.0 and pred10.approved and not pred10.model_ready
    ok(f"passthrough: prob={pred10.probability}  approved={pred10.approved}") if t10 else \
    fail(f"{pred10}")
    results.append(t10)

    # ── Test 11: train_model raises on insufficient samples ───────────────────
    hdr("Test 11 -- train_model() raises ValueError on too few samples")
    try:
        model.train_model([_make_trade_record("TP1")])  # only 1 sample
        t11 = False
        fail("Should have raised ValueError")
    except ValueError as exc:
        t11 = True
        ok(f"ValueError raised: {str(exc)[:60]}")
    except Exception as exc:
        t11 = False
        fail(f"Wrong exception: {type(exc).__name__}: {exc}")
    results.append(t11)

    # ── Test 12: train_model raises if sklearn not available ──────────────────
    hdr("Test 12 -- train_model() raises RuntimeError when sklearn mocked out")
    import strategy.ai_probability_model as _mod
    orig = _mod._SKLEARN_AVAILABLE
    _mod._SKLEARN_AVAILABLE = False
    try:
        model.train_model([])
        t12 = False
        fail("Should have raised RuntimeError")
    except RuntimeError as exc:
        t12 = True
        ok(f"RuntimeError raised: {str(exc)[:60]}")
    finally:
        _mod._SKLEARN_AVAILABLE = orig
    results.append(t12)

    # ── Test 13: train_model() succeeds on synthetic records ─────────────────
    hdr("Test 13 -- train_model() succeeds on 50+ synthetic trade records")
    trade_log = _make_synthetic_trade_log(n_wins=40, n_losses=30)  # more to ensure both classes in hold-out
    print(f"  Training on {len(trade_log)} records...")
    report = None
    try:
        report = model.train_model(trade_log, test_split=0.20, cv_folds=3)
        t13    = isinstance(report, TrainingReport) and report.n_samples >= 50
        ok(f"TrainingReport: n={report.n_samples}  "
           f"CV_acc={report.cv_accuracy_mean:.3f}  "
           f"test_acc={report.test_accuracy:.3f}") if t13 else \
        fail(f"n_samples={report.n_samples}")
    except Exception as exc:
        t13 = False
        fail(f"train_model() raised: {type(exc).__name__}: {exc}")
    results.append(t13)

    # ── Test 14: TrainingReport.to_dict() keys ────────────────────────────────
    hdr("Test 14 -- TrainingReport.to_dict() has all required keys")
    if report is None:
        t14 = False
        fail("No report from Test 13 — skipping")
    else:
        required14 = {
            "n_samples", "n_winners", "n_losers", "class_balance_pct",
            "cv_accuracy_mean", "cv_accuracy_std", "cv_roc_auc_mean",
            "test_accuracy", "test_roc_auc", "feature_importances",
        }
        d14     = report.to_dict()
        miss14  = required14 - set(d14.keys())
        t14 = not miss14
        ok(f"All {len(required14)} required keys present") if t14 else \
        fail(f"Missing: {miss14}")
    results.append(t14)

    # ── Test 15: print_summary() doesn't crash ────────────────────────────────
    hdr("Test 15 -- TrainingReport.print_summary() runs without error")
    if report is None:
        t15 = False; fail("No report — skipping")
    else:
        try:
            report.print_summary()
            t15 = True
            ok("print_summary() completed")
        except Exception as exc:
            t15 = False
            fail(f"Raised: {exc}")
    results.append(t15)

    # ── Test 16: predict after training → probability in [0, 1] ──────────────
    hdr("Test 16 -- predict_probability() returns prob in [0.0, 1.0] after training")
    pred16 = model.predict_probability(sig, trend, regime, df_m15)
    t16    = 0.0 <= pred16.probability <= 1.0 and pred16.model_ready
    ok(f"prob={pred16.probability:.3f}  approved={pred16.approved}") if t16 else \
    fail(f"prob={pred16.probability}  model_ready={pred16.model_ready}")
    results.append(t16)

    # ── Test 17: signal below threshold → rejected ────────────────────────────
    hdr("Test 17 -- probability below threshold → approved=False")
    cfg17           = _settings(threshold=0.99)   # impossibly high threshold
    model17         = TradeProbabilityModel(cfg17)
    model17._pipeline  = model._pipeline           # borrow the trained pipeline
    model17._is_fitted = True
    pred17 = model17.predict_probability(sig, trend, regime, df_m15)
    t17    = not pred17.approved   # prob < 0.99 → blocked
    ok(f"prob={pred17.probability:.3f} < 0.99 → rejected (correct)") if t17 else \
    fail(f"Should be rejected at threshold=0.99, got approved={pred17.approved}")
    results.append(t17)

    # ── Test 18: predict_from_record() works after training ──────────────────
    hdr("Test 18 -- predict_from_record() works on TradeRecord")
    rec18   = _make_trade_record("TP1", has_ob=True, has_fvg=True, confidence=0.80)
    pred18  = model.predict_from_record(rec18)
    t18     = 0.0 <= pred18.probability <= 1.0 and pred18.model_ready
    ok(f"prob={pred18.probability:.3f}  approved={pred18.approved}") if t18 else \
    fail(f"predict_from_record failed: {pred18}")
    results.append(t18)

    # ── Tests 19–20: save / load ──────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_model.joblib"

        hdr("Test 19 -- save_model() saves file to disk")
        try:
            saved = model.save_model(save_path)
            t19   = save_path.exists()
            ok(f"Model saved to {saved}") if t19 else fail("File not created")
        except Exception as exc:
            t19 = False
            fail(f"save_model raised: {exc}")
        results.append(t19)

        hdr("Test 20 -- load_model() restores a saved model")
        try:
            model20 = TradeProbabilityModel(cfg)
            model20.load_model(save_path)
            pred20  = model20.predict_probability(sig, trend, regime, df_m15)
            t20     = model20.is_ready and 0.0 <= pred20.probability <= 1.0
            ok(f"Loaded model: prob={pred20.probability:.3f}") if t20 else \
            fail(f"is_ready={model20.is_ready}  prob={pred20.probability}")
        except Exception as exc:
            t20 = False
            fail(f"load_model raised: {exc}")
        results.append(t20)

    # ── Test 21: PredictionResult.to_dict() ──────────────────────────────────
    hdr("Test 21 -- PredictionResult.to_dict() has required keys")
    required21 = {"probability", "approved", "threshold", "model_ready"}
    d21        = pred16.to_dict()
    miss21     = required21 - set(d21.keys())
    t21        = not miss21
    ok(f"All {len(required21)} required keys: {d21}") if t21 else \
    fail(f"Missing: {miss21}")
    results.append(t21)

    # ── Test 22: evaluate_on_backtest() ──────────────────────────────────────
    hdr("Test 22 -- evaluate_on_backtest() returns accuracy metric")
    eval22 = model.evaluate_on_backtest(trade_log)
    t22    = (eval22.get("status") == "evaluated" and
              "accuracy" in eval22 and
              0.0 <= eval22["accuracy"] <= 1.0)
    ok(f"status={eval22['status']}  accuracy={eval22.get('accuracy'):.3f}  "
       f"roc_auc={eval22.get('roc_auc'):.3f}") if t22 else \
    fail(f"evaluate result: {eval22}")
    results.append(t22)

    # ── Summary ───────────────────────────────────────────────────────────────
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
