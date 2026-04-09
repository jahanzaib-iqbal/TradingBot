"""
filters/volatility_filter.py
==============================
ATR-based volatility guard for XAUUSD signal generation.

Why filter on volatility?
--------------------------
XAUUSD can be either too quiet (ATR < 5 pts — choppy, no directional movement,
stop-hunting everywhere) or too violent (ATR > 40 pts — news spike,
erratic wicks, spreads widen). Both extremes destroy signal quality.

The sweet spot for Gold intraday signals is ATR between 5 and 40 points
(= 500–4000 pips in MT5 notation).

Volatility state and confidence
---------------------------------
  NORMAL  (within range)  → no penalty
  LOW     (ATR too small) → confidence -0.10; signal blocked
  HIGH    (ATR too large) → confidence -0.15; signal blocked
  EXTREME (ATR > 2x max) → confidence -0.30; always blocked

Volatility confirmation scoring (for signal confidence)
-------------------------------------------------------
When ATR is in the optimal trading range, a score bonus is awarded:
  ATR in (optimal_low, optimal_high) → +0.05

Where optimal_low  = ATR_MIN_POINTS × 1.5
      optimal_high = ATR_MAX_POINTS × 0.6

Default for XAUUSD: optimal_low=7.5, optimal_high=24.0

Dependencies: pandas, numpy, config.settings.Settings
"""

from __future__ import annotations

from enum import Enum
from typing import Tuple

import numpy as np
import pandas as pd

from config.settings import Settings
from utils.logger import get_logger

logger = get_logger(__name__)


class VolatilityState(str, Enum):
    """Current volatility environment classification."""
    NORMAL  = "NORMAL"   # ATR within tradeable range
    LOW     = "LOW"      # ATR too small — market too quiet/choppy
    HIGH    = "HIGH"     # ATR too large — news spike or erratic  
    EXTREME = "EXTREME"  # ATR > 2× max threshold — do not trade


class VolatilityFilter:
    """
    Guards signals against extreme low or high volatility environments.

    Usage
    -----
        vf      = VolatilityFilter(settings)
        allowed, reason = vf.is_allowed(df)
        state   = vf.classify(df)
        delta   = vf.confidence_delta(df)
    """

    def __init__(self, settings: Settings) -> None:
        self.cfg     = settings
        self.period  = settings.ATR_PERIOD
        self.min_atr = settings.ATR_MIN_POINTS   # default 5.0
        self.max_atr = settings.ATR_MAX_POINTS   # default 40.0

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def compute_atr(self, df: pd.DataFrame, period: int | None = None) -> float:
        """
        Compute the most-recent ATR value using Wilder's smoothing.

        Parameters
        ----------
        df     : OHLCV DataFrame with columns [high, low, close]
        period : Override for Settings.ATR_PERIOD

        Returns
        -------
        float — Latest ATR value in price points (e.g. 12.5 for XAUUSD)
        """
        p = period or self.period
        req = {"high", "low", "close"}
        if not req.issubset(df.columns):
            logger.error(f"compute_atr() — missing columns: {req - set(df.columns)}")
            return 0.0

        if len(df) < p:
            logger.warning(f"compute_atr() — only {len(df)} bars, need {p}. Returning 0.")
            return 0.0

        high  = df["high"]
        low   = df["low"]
        close = df["close"]

        # True Range = max(H-L, |H-prev_C|, |L-prev_C|)
        hl    = high - low
        h_pc  = (high - close.shift(1)).abs()
        l_pc  = (low  - close.shift(1)).abs()
        tr    = pd.concat([hl, h_pc, l_pc], axis=1).max(axis=1)

        # Wilder's smoothing (same as MT5 ATR)
        atr = tr.ewm(alpha=1.0 / p, min_periods=p, adjust=False).mean()

        val = float(atr.iloc[-1])
        if np.isnan(val):
            return 0.0
        return round(val, 4)

    def classify(self, df: pd.DataFrame) -> VolatilityState:
        """
        Classify the current volatility environment.

        Returns
        -------
        VolatilityState — NORMAL / LOW / HIGH / EXTREME
        """
        atr = self.compute_atr(df)
        if atr == 0.0:
            return VolatilityState.LOW

        if atr > self.max_atr * 2.0:
            return VolatilityState.EXTREME
        if atr > self.max_atr:
            return VolatilityState.HIGH
        if atr < self.min_atr:
            return VolatilityState.LOW
        return VolatilityState.NORMAL

    def is_allowed(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Check whether the current ATR is within an acceptable trading range.

        Returns
        -------
        (True, "")             — signal generation permitted
        (False, reason_string) — ATR outside bounds; signal blocked
        """
        atr   = self.compute_atr(df)
        state = self.classify(df)

        if state == VolatilityState.LOW:
            return (
                False,
                f"ATR too low ({atr:.2f} pts < min {self.min_atr:.1f} pts) "
                "— market too quiet/choppy for reliable signals"
            )

        if state == VolatilityState.HIGH:
            return (
                False,
                f"ATR too high ({atr:.2f} pts > max {self.max_atr:.1f} pts) "
                "— possible news spike; waiting for volatility to normalise"
            )

        if state == VolatilityState.EXTREME:
            return (
                False,
                f"ATR extreme ({atr:.2f} pts >> {self.max_atr:.1f} pts limit) "
                "— major news event; trading suspended"
            )

        return True, ""

    def confidence_delta(self, df: pd.DataFrame) -> float:
        """
        Return the confidence score adjustment based on volatility quality.

        Signal confidence adjustments
        ------------------------------
        NORMAL in optimal range → +0.05 (ATR confirms good trading conditions)
        NORMAL outside optimal  →  0.00 (acceptable but not ideal)
        LOW / HIGH / EXTREME    → -0.10 to -0.30 (quality penalty)
        """
        atr   = self.compute_atr(df)
        state = self.classify(df)

        if state == VolatilityState.EXTREME:
            return -0.30
        if state == VolatilityState.HIGH:
            return -0.15
        if state == VolatilityState.LOW:
            return -0.10

        # NORMAL — check if ATR is in the "sweet spot"
        optimal_low  = self.min_atr * 1.5   # e.g. 7.5
        optimal_high = self.max_atr * 0.60  # e.g. 24.0
        if optimal_low <= atr <= optimal_high:
            return 0.05   # bonus for ideal volatility conditions

        return 0.0

    def volatility_info(self, df: pd.DataFrame) -> dict:
        """Return a dict summarising current volatility state."""
        atr   = self.compute_atr(df)
        state = self.classify(df)
        allowed, reason = self.is_allowed(df)
        return {
            "atr":              round(atr, 2),
            "state":            state.value,
            "allowed":          allowed,
            "reason":           reason if not allowed else "",
            "confidence_delta": self.confidence_delta(df),
            "atr_min":          self.min_atr,
            "atr_max":          self.max_atr,
        }
