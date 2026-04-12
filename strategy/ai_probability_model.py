"""
strategy/ai_probability_model.py
==================================
AI-powered trade win-probability predictor for the JayBot Gold Bot.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GOAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Predict the probability that a given trade signal will reach
Take Profit before Stop Loss, using an ensemble of market-context
features extracted at signal generation time.

Acts as an additional gate on top of the SMC confidence score –
both must pass for a signal to be published to Telegram.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODEL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Classifier : sklearn RandomForestClassifier
  Target     : binary  0 = SL hit  |  1 = TP1 or TP2 hit
  Output     : probability(class=1) via predict_proba()

  Calibration: Platt scaling (CalibratedClassifierCV) is applied
               after training to ensure probabilities are
               well-calibrated (not just over-confident raw RF scores).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FEATURE VECTOR  (12 features)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1.  trend_strength        — EMA 50/200 composite strength (0–1)
  2.  atr                   — Wilder ATR at signal time (price pts)
  3.  adx                   — Average Directional Index (0–100)
  4.  ema_distance_pct      — % distance of entry from EMA 200
  5.  has_liquidity_sweep   — binary  0/1
  6.  has_order_block       — binary  0/1
  7.  has_fvg               — binary  0/1
  8.  has_bos               — binary  0/1
  9.  session               — ordinal  0=closed 1=asian 2=london
                                        3=new_york 4=overlap
  10. volatility_state       — ordinal  0=low 1=normal 2=high 3=extreme
  11. rr_ratio              — reward-to-risk ratio (R:R to TP1)
  12. smc_confidence        — raw SMC engine confidence score (0–1)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PUBLIC API
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  model = TradeProbabilityModel(settings)

  # Train from backtest trade log (list[TradeRecord] or CSV path)
  report = model.train_model(trade_records)

  # Predict on a single TradingSignal (returns 0.0–1.0)
  result = model.predict_probability(signal, trend_result, regime_result, df_m15)
  # → {"probability": 0.73, "approved": True, "threshold": 0.60}

  # Persist / restore
  model.save_model("strategy/models/rf_probability.joblib")
  model.load_model("strategy/models/rf_probability.joblib")

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TRAINING WORKFLOW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. Run BacktestEngine.run() to generate a trade_log
  2. Call model.train_model(trade_log)         ← fits RF
  3. Inspect report (accuracy, feature importance, CV scores)
  4. Call model.save_model()                   ← persists to disk
  5. At bot startup: model.load_model()        ← restores weights
  6. Set AI_USE_PROBABILITY_FILTER=true in .env

Dependencies
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  scikit-learn >= 1.4.0
  joblib >= 1.3.0
  pandas, numpy
  config.settings, utils.logger
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import Settings
from utils.logger import get_logger

logger = get_logger(__name__)


# ── sklearn — lazy import guard so module loads even without sklearn ───────────
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import (
        accuracy_score, classification_report,
        roc_auc_score, confusion_matrix,
    )
    import joblib as _joblib
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    logger.warning(
        "scikit-learn / joblib not installed — "
        "TradeProbabilityModel will run in PASSTHROUGH mode. "
        "Install with: pip install scikit-learn joblib"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Feature names (order defines the vector — MUST stay consistent)
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_NAMES: list[str] = [
    "trend_strength",       # 0  — EMA composite strength  0–1
    "atr",                  # 1  — Wilder ATR in price points
    "adx",                  # 2  — Average Directional Index 0–100
    "ema_distance_pct",     # 3  — |entry - EMA200| / EMA200 × 100
    "has_liquidity_sweep",  # 4  — binary 0/1
    "has_order_block",      # 5  — binary 0/1
    "has_fvg",              # 6  — binary 0/1
    "has_bos",              # 7  — binary 0/1
    "session_code",         # 8  — ordinal: closed=0 asian=1 london=2 ny=3 overlap=4
    "volatility_state",     # 9  — ordinal: low=0 normal=1 high=2 extreme=3
    "rr_ratio",             # 10 — reward-to-risk to TP1
    "smc_confidence",       # 11 — raw SMC engine score 0–1
]

_SESSION_CODE: dict[str, int] = {
    "closed": 0, "asian": 1, "london": 2, "new_york": 3, "overlap": 4,
}
_VOLATILITY_CODE: dict[str, int] = {
    "LOW": 0, "NORMAL": 1, "HIGH": 2, "EXTREME": 3,
}


# ─────────────────────────────────────────────────────────────────────────────
# Training report dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainingReport:
    """Summary of a model training run."""
    n_samples:          int     = 0
    n_winners:          int     = 0
    n_losers:           int     = 0
    class_balance_pct:  float   = 0.0     # winners / total
    cv_accuracy_mean:   float   = 0.0     # mean 5-fold CV accuracy
    cv_accuracy_std:    float   = 0.0
    cv_roc_auc_mean:    float   = 0.0
    test_accuracy:      float   = 0.0     # hold-out 20% test set
    test_roc_auc:       float   = 0.0
    feature_importances: dict   = field(default_factory=dict)
    confusion_matrix:    list   = field(default_factory=list)  # [[TN,FP],[FN,TP]]
    classification_report: str  = ""

    def print_summary(self) -> None:
        sep = "─" * 52
        print(f"\n{'═'*52}")
        print("  AI Probability Model — Training Report")
        print(f"{'═'*52}")
        print(f"  Samples:          {self.n_samples:>8,}")
        print(f"  Winners (TP hit): {self.n_winners:>8,}  ({self.class_balance_pct:.1%})")
        print(f"  Losers  (SL hit): {self.n_losers:>8,}")
        print(f"{sep}")
        print(f"  5-Fold CV Accuracy: {self.cv_accuracy_mean:.3f} ± {self.cv_accuracy_std:.3f}")
        print(f"  5-Fold CV ROC-AUC:  {self.cv_roc_auc_mean:.3f}")
        print(f"{sep}")
        print(f"  Hold-out Accuracy:  {self.test_accuracy:.3f}")
        print(f"  Hold-out ROC-AUC:   {self.test_roc_auc:.3f}")
        print(f"{sep}")
        print("  Feature Importances (top 10):")
        sorted_fi = sorted(
            self.feature_importances.items(), key=lambda x: x[1], reverse=True
        )
        for name, imp in sorted_fi[:10]:
            bar = "█" * int(imp * 40)
            print(f"    {name:<28} {imp:.3f}  {bar}")
        print(f"{sep}")
        if self.confusion_matrix:
            cm = self.confusion_matrix
            print(f"  Confusion Matrix (hold-out):")
            print(f"    TN={cm[0][0]}  FP={cm[0][1]}")
            print(f"    FN={cm[1][0]}  TP={cm[1][1]}")
        print(f"{'═'*52}\n")

    def to_dict(self) -> dict:
        return {
            "n_samples":           self.n_samples,
            "n_winners":           self.n_winners,
            "n_losers":            self.n_losers,
            "class_balance_pct":   round(self.class_balance_pct, 4),
            "cv_accuracy_mean":    round(self.cv_accuracy_mean,  4),
            "cv_accuracy_std":     round(self.cv_accuracy_std,   4),
            "cv_roc_auc_mean":     round(self.cv_roc_auc_mean,   4),
            "test_accuracy":       round(self.test_accuracy,     4),
            "test_roc_auc":        round(self.test_roc_auc,      4),
            "feature_importances": {k: round(v, 4) for k, v in
                                    self.feature_importances.items()},
        }


# ─────────────────────────────────────────────────────────────────────────────
# Prediction result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PredictionResult:
    """
    Output from predict_probability().

    Attributes
    ----------
    probability : float
        Model's estimated win probability (0.0 – 1.0).
    approved    : bool
        True if probability >= threshold.
    threshold   : float
        The threshold value used for this prediction.
    model_ready : bool
        False if no model was loaded (passthrough mode → always approved).
    features    : dict
        The feature vector that was fed to the model.
    """
    probability: float = 0.0
    approved:    bool  = True
    threshold:   float = 0.60
    model_ready: bool  = False
    features:    dict  = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "probability": round(self.probability, 4),
            "approved":    self.approved,
            "threshold":   self.threshold,
            "model_ready": self.model_ready,
        }

    def __str__(self) -> str:
        status = "✅ APPROVED" if self.approved else "❌ REJECTED"
        ready  = "model" if self.model_ready else "passthrough"
        return (
            f"AI({ready}): probability={self.probability:.1%}  "
            f"threshold={self.threshold:.0%}  {status}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Feature builder helpers
# ─────────────────────────────────────────────────────────────────────────────

def _session_to_code(session: str | None) -> int:
    return _SESSION_CODE.get((session or "closed").lower(), 0)


def _volatility_to_code(df_m15: pd.DataFrame, atr_val: float) -> int:
    """Classify ATR into the 4 VolatilityState buckets using same thresholds as VolatilityFilter."""
    try:
        from filters.volatility_filter import VolatilityFilter, VolatilityState
        from config.settings import Settings
        state = VolatilityFilter(Settings()).classify(df_m15)
        return _VOLATILITY_CODE.get(state.value, 1)
    except Exception:
        # Fallback: classify by raw ATR value
        if atr_val < 5.0:   return 0   # LOW
        if atr_val < 40.0:  return 1   # NORMAL
        if atr_val < 80.0:  return 2   # HIGH
        return 3                        # EXTREME


def _compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Wilder ATR from OHLCV DataFrame. Returns 0.0 on error."""
    try:
        if len(df) < period + 1:
            return 0.0
        highs  = df["high"].values.astype(float)
        lows   = df["low"].values.astype(float)
        closes = df["close"].values.astype(float)
        trs    = np.maximum(highs[1:] - lows[1:],
                  np.maximum(abs(highs[1:] - closes[:-1]),
                             abs(lows[1:]  - closes[:-1])))
        atr = float(np.mean(trs[-period:]))
        for tr in trs[-period:]:
            atr = (atr * (period - 1) + tr) / period
        return round(atr, 4)
    except Exception:
        return 0.0


def build_feature_vector(
    signal,               # TradingSignal
    trend_result,         # TrendResult  (strategy.trend_detection)
    regime_result,        # RegimeResult (strategy.market_regime)
    df_m15: pd.DataFrame,
) -> np.ndarray:
    """
    Convert a TradingSignal + market context into the 12-feature numpy vector
    expected by the RandomForest.

    Parameters
    ----------
    signal        : TradingSignal from SignalGenerator
    trend_result  : TrendResult from TrendDetector.detect()
    regime_result : RegimeResult from MarketRegimeClassifier.classify()
    df_m15        : M15 OHLCV DataFrame (for ATR computation + volatility state)

    Returns
    -------
    np.ndarray shape (12,)
    """
    atr = _compute_atr(df_m15)

    # EMA 200 distance
    ema200 = getattr(trend_result, "ema_200", 0.0) or 0.0
    if ema200 > 0:
        ema_dist_pct = abs(signal.entry_price - ema200) / ema200 * 100.0
    else:
        ema_dist_pct = 0.0

    vol_code = _volatility_to_code(df_m15, atr)

    rr = getattr(signal, "rr_ratio_tp1", None) or 0.0

    features = np.array([
        float(getattr(trend_result,  "strength",      0.0) or 0.0),   # 0
        float(atr),                                                     # 1
        float(getattr(regime_result, "adx",           0.0) or 0.0),   # 2
        float(ema_dist_pct),                                            # 3
        float(1 if getattr(signal, "has_liquidity_sweep", False) else 0),  # 4
        float(1 if getattr(signal, "has_ob",           False) else 0),     # 5
        float(1 if getattr(signal, "has_fvg",          False) else 0),     # 6
        float(1 if getattr(signal, "has_bos",          False) else 0),     # 7
        float(_session_to_code(getattr(signal, "session", None))),          # 8
        float(vol_code),                                                     # 9
        float(rr),                                                           # 10
        float(getattr(signal, "smc_confidence", 0.0) or 0.0),              # 11
    ], dtype=np.float64)

    return features


def build_feature_vector_from_record(record) -> tuple[np.ndarray, int]:
    """
    Build a feature vector from a backtest TradeRecord.

    Used during training — the TradeRecord already contains all relevant
    SMC metadata recorded at signal time.

    Returns (feature_vector, label) where label = 1 (win) or 0 (loss).
    """
    # Map session string to ordinal
    sess_code = _session_to_code(getattr(record, "session", None))

    # Regime → volatility proxy (we don't have ATR in TradeRecord directly)
    # Use regime string as an ordinal proxy
    regime = (getattr(record, "regime", "") or "").upper()
    vol_code = {
        "LOW_VOLATILITY":  0,
        "NORMAL":          1,
        "TRENDING":        1,
        "RANGING":         1,
        "HIGH_VOLATILITY": 2,
        "EXTREME":         3,
    }.get(regime, 1)

    # R:R from TradeRecord
    rr = getattr(record, "rr_tp1", 0.0) or 0.0

    features = np.array([
        0.5,                                                           # trend_strength (unknown)
        0.0,                                                           # atr (unknown at record level)
        25.0,                                                          # adx (unknown; use neutral)
        0.5,                                                           # ema_distance_pct (unknown)
        float(1 if getattr(record, "has_liq_sweep", False) else 0),   # 4
        float(1 if getattr(record, "has_ob",        False) else 0),   # 5
        float(1 if getattr(record, "has_fvg",       False) else 0),   # 6
        float(1 if getattr(record, "has_bos",       False) else 0),   # 7
        float(sess_code),                                              # 8
        float(vol_code),                                               # 9
        float(rr),                                                     # 10
        float(getattr(record, "confidence", 0.5) or 0.5),             # 11  (total signal conf)
    ], dtype=np.float64)

    outcome = getattr(record, "outcome", "EXPIRED")
    label   = 1 if outcome in ("TP1", "TP2") else 0
    return features, label


def build_feature_vector_from_dict(row: dict) -> tuple[np.ndarray, int]:
    """
    Build a feature vector from a CSV trade-log row (as dict).

    Allows training directly from trade_log.csv exported by BacktestEngine.
    """
    def _get(key, default=0.0):
        val = row.get(key, default)
        try:
            return float(val) if val not in (None, "", "None") else float(default)
        except (ValueError, TypeError):
            return float(default)

    sess_code = _session_to_code(str(row.get("session", "closed")))
    regime    = str(row.get("regime", "")).upper()
    vol_code  = {"LOW_VOLATILITY": 0, "HIGH_VOLATILITY": 2, "EXTREME": 3}.get(regime, 1)
    outcome   = str(row.get("outcome", "EXPIRED"))
    label     = 1 if outcome in ("TP1", "TP2") else 0

    features = np.array([
        _get("trend_strength", 0.5),
        _get("atr", 0.0),
        _get("adx", 25.0),
        _get("ema_distance_pct", 0.5),
        _get("has_liq_sweep", 0.0),
        _get("has_ob", 0.0),
        _get("has_fvg", 0.0),
        _get("has_bos", 0.0),
        float(sess_code),
        float(vol_code),
        _get("rr_tp1", 2.0),
        _get("confidence", 0.5),
    ], dtype=np.float64)

    return features, label


# ─────────────────────────────────────────────────────────────────────────────
# Main model class
# ─────────────────────────────────────────────────────────────────────────────

class TradeProbabilityModel:
    """
    RandomForest-based trade win-probability predictor.

    Lifecycle
    ---------
    1. Instantiate once at startup
    2. Either:
       (a) train_model(records) + save_model()  ← first time
       (b) load_model()                          ← subsequent startups
    3. Call predict_probability() on every TradingSignal before dispatch

    Passthrough Mode
    ----------------
    If no model is loaded (and sklearn is unavailable), predict_probability()
    returns probability=1.0 and approved=True, so the bot runs normally
    without AI filtering.  This is safe for live use before enough training
    data has been collected.
    """

    # Minimum samples per class to attempt training
    _MIN_PER_CLASS: int = 10

    def __init__(self, settings: Settings) -> None:
        self.cfg        = settings
        self.threshold  = settings.AI_PROBABILITY_THRESHOLD
        self._pipeline: Optional[object] = None   # sklearn Pipeline once trained
        self._is_fitted = False
        self._feature_names = FEATURE_NAMES[:]

        # Try to auto-load model from configured path
        model_path = Path(settings.AI_MODEL_PATH)
        if model_path.exists():
            try:
                self.load_model(model_path)
            except Exception as exc:
                logger.warning(f"Auto-load of AI model failed: {exc}")

    # =========================================================================
    # PUBLIC — train_model
    # =========================================================================

    def train_model(
        self,
        trade_records,              # list[TradeRecord] | list[dict] | str/Path (CSV)
        test_split: float   = 0.20,
        cv_folds:   int     = 5,
    ) -> TrainingReport:
        """
        Fit the RandomForest on historical trade outcomes.

        Parameters
        ----------
        trade_records : One of:
                        • list[TradeRecord]   — from BacktestEngine.trade_log
                        • list[dict]          — rows from trade_log.csv
                        • str / Path          — path to trade_log.csv
        test_split    : Fraction of data held out for final evaluation
        cv_folds      : Number of cross-validation folds

        Returns
        -------
        TrainingReport with CV scores, feature importances, confusion matrix.

        Raises
        ------
        RuntimeError  if scikit-learn is not installed
        ValueError    if not enough samples or badly imbalanced classes
        """
        if not _SKLEARN_AVAILABLE:
            raise RuntimeError(
                "scikit-learn is not installed. "
                "Run: pip install scikit-learn joblib"
            )

        # ── Load data ─────────────────────────────────────────────────────────
        X, y = self._load_training_data(trade_records)
        n    = len(y)

        if n < self.cfg.AI_MIN_TRAIN_SAMPLES:
            raise ValueError(
                f"Need at least {self.cfg.AI_MIN_TRAIN_SAMPLES} samples "
                f"for training. Got {n}. "
                f"Run more backtests or lower AI_MIN_TRAIN_SAMPLES."
            )

        n_pos = int(np.sum(y))
        n_neg = n - n_pos

        if n_pos < self._MIN_PER_CLASS or n_neg < self._MIN_PER_CLASS:
            raise ValueError(
                f"Need at least {self._MIN_PER_CLASS} samples per class. "
                f"Got {n_pos} winners and {n_neg} losers. "
                f"Collect more diverse trade data."
            )

        logger.info(
            f"Training AI model: {n} samples "
            f"({n_pos} winners / {n_neg} losers  "
            f"balance={n_pos/n:.1%})"
        )

        # ── Train / test split (time-based — no shuffle to prevent leakage) ──
        split_idx = int(n * (1 - test_split))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # ── Build pipeline: StandardScaler → RF → Platt calibration ──────────
        rf = RandomForestClassifier(
            n_estimators  = self.cfg.AI_N_ESTIMATORS,
            max_depth      = 8,
            min_samples_leaf = 5,
            class_weight   = "balanced",   # handle imbalanced win/loss ratios
            random_state   = self.cfg.AI_RANDOM_SEED,
            n_jobs         = -1,
        )

        # CalibratedClassifierCV improves probability estimates
        calibrated_rf = CalibratedClassifierCV(rf, method="sigmoid", cv=3)

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", calibrated_rf),
        ])

        # ── Cross-validation ──────────────────────────────────────────────────
        cv_strategy = StratifiedKFold(
            n_splits=min(cv_folds, n_pos),  # can't have more folds than positive class
            shuffle=False,                   # no shuffle — preserve time order
        )

        cv_acc = cross_val_score(
            pipeline, X_train, y_train,
            cv=cv_strategy, scoring="accuracy", n_jobs=-1,
        )
        cv_auc = cross_val_score(
            pipeline, X_train, y_train,
            cv=cv_strategy, scoring="roc_auc", n_jobs=-1,
        )

        logger.info(
            f"  CV accuracy: {cv_acc.mean():.3f} ± {cv_acc.std():.3f}  "
            f"ROC-AUC: {cv_auc.mean():.3f}"
        )

        # ── Final fit on full training set ────────────────────────────────────
        pipeline.fit(X_train, y_train)
        self._pipeline  = pipeline
        self._is_fitted = True

        # ── Hold-out evaluation ───────────────────────────────────────────────
        y_pred      = pipeline.predict(X_test)
        y_prob      = pipeline.predict_proba(X_test)[:, 1]
        test_acc    = accuracy_score(y_test, y_pred)
        try:
            test_roc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.0
        except Exception:
            test_roc = 0.0
        conf_matrix = confusion_matrix(y_test, y_pred).tolist()
        cls_report  = classification_report(
            y_test, y_pred,
            target_names=["SL", "TP"],
            zero_division=0,  # suppress UndefinedMetricWarning
        )

        # ── Feature importances (from underlying RF) ──────────────────────────
        # CalibratedClassifierCV stores fitted estimators in calibrated_classifiers_
        # each of which has a .estimator attribute pointing to the fitted RF.
        try:
            calibrated_clf = pipeline.named_steps["classifier"]
            # sklearn 1.2+: calibrated_classifiers_ contains CalibratedClassifier objects
            fitted_rf = calibrated_clf.calibrated_classifiers_[0].estimator
            fi = dict(zip(self._feature_names, fitted_rf.feature_importances_))
        except Exception:
            # Fallback: uniform importances if we can't extract them
            fi = {name: 1.0 / len(self._feature_names) for name in self._feature_names}

        report = TrainingReport(
            n_samples             = n,
            n_winners             = n_pos,
            n_losers              = n_neg,
            class_balance_pct     = n_pos / n,
            cv_accuracy_mean      = float(cv_acc.mean()),
            cv_accuracy_std       = float(cv_acc.std()),
            cv_roc_auc_mean       = float(cv_auc.mean()),
            test_accuracy         = test_acc,
            test_roc_auc          = test_roc,
            feature_importances   = fi,
            confusion_matrix      = conf_matrix,
            classification_report = cls_report,
        )

        logger.info(
            f"Model trained: test_acc={test_acc:.3f}  "
            f"test_auc={test_roc:.3f}"
        )
        return report

    # =========================================================================
    # PUBLIC — predict_probability
    # =========================================================================

    def predict_probability(
        self,
        signal,               # TradingSignal
        trend_result,         # TrendResult
        regime_result,        # RegimeResult
        df_m15: pd.DataFrame,
    ) -> PredictionResult:
        """
        Predict the win probability for a validated TradingSignal.

        Parameters
        ----------
        signal        : TradingSignal from SignalGenerator
        trend_result  : TrendResult from TrendDetector.detect()
        regime_result : RegimeResult from MarketRegimeClassifier.classify()
        df_m15        : M15 OHLCV bars (last N bars at signal time)

        Returns
        -------
        PredictionResult with:
            probability : float 0.0 – 1.0
            approved    : bool  (probability >= threshold)
            model_ready : bool
            features    : dict (feature vector for auditing)
        """
        # ── Passthrough if no model ───────────────────────────────────────────
        if not self._is_fitted or not _SKLEARN_AVAILABLE:
            return PredictionResult(
                probability = 1.0,
                approved    = True,
                threshold   = self.threshold,
                model_ready = False,
                features    = {},
            )

        # ── Build feature vector ──────────────────────────────────────────────
        feat_vec = build_feature_vector(signal, trend_result, regime_result, df_m15)
        feat_dict = dict(zip(self._feature_names, feat_vec.tolist()))

        # ── Predict ───────────────────────────────────────────────────────────
        try:
            prob = float(
                self._pipeline.predict_proba(feat_vec.reshape(1, -1))[0, 1]
            )
        except Exception as exc:
            logger.error(f"predict_probability error: {exc}")
            prob = 0.5   # neutral — don't block on error

        approved = prob >= self.threshold

        logger.debug(
            f"AI: prob={prob:.1%}  threshold={self.threshold:.0%}  "
            f"{'✅' if approved else '❌'}"
        )

        return PredictionResult(
            probability = round(prob, 4),
            approved    = approved,
            threshold   = self.threshold,
            model_ready = True,
            features    = feat_dict,
        )

    def predict_from_record(self, record) -> PredictionResult:
        """
        Convenience: predict from a TradeRecord (backtest validation).

        Lets you evaluate model accuracy on out-of-sample backtest trades
        without needing the live signal pipeline.
        """
        if not self._is_fitted or not _SKLEARN_AVAILABLE:
            return PredictionResult(probability=1.0, approved=True,
                                    model_ready=False)

        feat_vec, _ = build_feature_vector_from_record(record)
        feat_dict   = dict(zip(self._feature_names, feat_vec.tolist()))

        try:
            prob = float(
                self._pipeline.predict_proba(feat_vec.reshape(1, -1))[0, 1]
            )
        except Exception as exc:
            logger.error(f"predict_from_record error: {exc}")
            prob = 0.5

        return PredictionResult(
            probability = round(prob, 4),
            approved    = prob >= self.threshold,
            threshold   = self.threshold,
            model_ready = True,
            features    = feat_dict,
        )

    # =========================================================================
    # PUBLIC — save / load
    # =========================================================================

    def save_model(self, path: str | Path | None = None) -> Path:
        """
        Persist the trained pipeline to disk using joblib.

        Parameters
        ----------
        path : File path to save to.  Defaults to Settings.AI_MODEL_PATH.
               Parent directory is created if needed.

        Returns
        -------
        Path — resolved absolute path of the saved file.

        Raises
        ------
        RuntimeError  if no model has been trained yet.
        """
        if not self._is_fitted:
            raise RuntimeError("No model to save — call train_model() first.")
        if not _SKLEARN_AVAILABLE:
            raise RuntimeError("joblib not available — cannot save model.")

        save_path = Path(path or self.cfg.AI_MODEL_PATH)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "pipeline":      self._pipeline,
            "feature_names": self._feature_names,
            "threshold":     self.threshold,
            "sklearn_ver":   _joblib.__version__ if hasattr(_joblib, "__version__") else "?",
        }
        _joblib.dump(payload, save_path)
        logger.info(f"AI model saved → {save_path.resolve()}")
        return save_path.resolve()

    def load_model(self, path: str | Path | None = None) -> None:
        """
        Load a previously saved pipeline from disk.

        Parameters
        ----------
        path : File path to load from.  Defaults to Settings.AI_MODEL_PATH.

        Raises
        ------
        FileNotFoundError  if the file does not exist.
        RuntimeError       if joblib is not available.
        """
        if not _SKLEARN_AVAILABLE:
            raise RuntimeError("joblib not available — cannot load model.")

        load_path = Path(path or self.cfg.AI_MODEL_PATH)
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")

        payload = _joblib.load(load_path)

        self._pipeline      = payload["pipeline"]
        self._feature_names = payload.get("feature_names", FEATURE_NAMES[:])
        self.threshold      = payload.get("threshold", self.threshold)
        self._is_fitted     = True

        logger.info(
            f"AI model loaded ← {load_path.resolve()}  "
            f"(threshold={self.threshold:.0%})"
        )

    # =========================================================================
    # PUBLIC — batch evaluation
    # =========================================================================

    def evaluate_on_backtest(self, trade_records) -> dict:
        """
        Apply the model to a list of TradeRecords and report accuracy.

        Useful for validating that the saved model still matches the live
        strategy's win distribution after a re-backtest.

        Returns a dict with accuracy, precision, recall, AUC.
        """
        if not self._is_fitted or not _SKLEARN_AVAILABLE:
            return {"status": "no_model"}

        results = []
        for rec in trade_records:
            outcome = getattr(rec, "outcome", "EXPIRED")
            if outcome == "EXPIRED":
                continue
            feat_vec, label = build_feature_vector_from_record(rec)
            prob = float(
                self._pipeline.predict_proba(feat_vec.reshape(1, -1))[0, 1]
            )
            results.append({"prob": prob, "label": label})

        if not results:
            return {"status": "no_resolved_trades"}

        df    = pd.DataFrame(results)
        y_tr  = df["label"].values
        y_pb  = df["prob"].values
        y_p   = (y_pb >= self.threshold).astype(int)

        acc   = float(np.mean(y_p == y_tr))
        auc   = float(roc_auc_score(y_tr, y_pb)) if len(np.unique(y_tr)) > 1 else 0.0

        return {
            "status":   "evaluated",
            "n_trades": len(results),
            "accuracy": round(acc, 4),
            "roc_auc":  round(auc, 4),
            "threshold": self.threshold,
        }

    # =========================================================================
    # PROPERTIES
    # =========================================================================

    @property
    def is_ready(self) -> bool:
        """True if a fitted model is available for predictions."""
        return self._is_fitted and _SKLEARN_AVAILABLE

    @property
    def feature_names(self) -> list[str]:
        return self._feature_names[:]

    def __repr__(self) -> str:
        status = f"fitted threshold={self.threshold:.0%}" if self._is_fitted else "not fitted"
        return (
            f"TradeProbabilityModel("
            f"sklearn={_SKLEARN_AVAILABLE}, "
            f"{status})"
        )

    # =========================================================================
    # PRIVATE — data loading
    # =========================================================================

    def _load_training_data(self, source) -> tuple[np.ndarray, np.ndarray]:
        """
        Accept multiple input formats and return (X, y).

        Supports:
          • list[TradeRecord]   — backtest engine output
          • list[dict]          — rows with dict keys matching TradeRecord fields
          • str | Path          — CSV path (trade_log.csv from engine.export())
        """
        X_list, y_list = [], []

        if isinstance(source, (str, Path)):
            # ── CSV mode ──────────────────────────────────────────────────────
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"Training CSV not found: {path}")
            df = pd.read_csv(path)
            df.columns = [c.strip().lower() for c in df.columns]
            for _, row in df.iterrows():
                d = row.to_dict()
                # Remap CSV column names to TradeRecord attribute names
                d["has_liq_sweep"] = d.get("has_liq_sweep", d.get("has_liq_sweep", 0))
                feat, label = build_feature_vector_from_dict(d)
                X_list.append(feat)
                y_list.append(label)

        elif source and hasattr(source[0], "outcome"):
            # ── TradeRecord list ──────────────────────────────────────────────
            for rec in source:
                outcome = getattr(rec, "outcome", "EXPIRED")
                if outcome == "EXPIRED":
                    continue   # exclude unresolved trades
                feat, label = build_feature_vector_from_record(rec)
                X_list.append(feat)
                y_list.append(label)

        elif source and isinstance(source[0], dict):
            # ── dict list ─────────────────────────────────────────────────────
            for row in source:
                if row.get("outcome", "EXPIRED") == "EXPIRED":
                    continue
                feat, label = build_feature_vector_from_dict(row)
                X_list.append(feat)
                y_list.append(label)

        else:
            raise TypeError(
                "trade_records must be list[TradeRecord], list[dict], or a CSV path."
            )

        if not X_list:
            raise ValueError(
                "No valid (non-EXPIRED) trade records found in the training data."
            )

        return np.array(X_list, dtype=np.float64), np.array(y_list, dtype=int)
