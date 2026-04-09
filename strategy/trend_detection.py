"""
strategy/trend_detection.py
============================
EMA-based trend detector for XAUUSD.

Core rules
----------
  BULLISH  — close > EMA_50  AND  close > EMA_200
  BEARISH  — close < EMA_50  AND  close < EMA_200
  NEUTRAL  — price is between EMA_50 and EMA_200 (transition / mixed zone)

Trend strength
--------------
Strength is a composite 0.0–1.0 score built from three sub-signals:

  1. EMA separation  — how far apart EMA_50 and EMA_200 are
                       (wide separation = mature, confirmed trend)
  2. Price distance  — how far price is above/below BOTH EMAs
                       (far above = strong conviction)
  3. EMA_50 slope    — how steeply the faster EMA is climbing/falling
                       (steep slope = accelerating trend)

Strength bands
--------------
  0.00 – 0.33  → WEAK    (nascent or fading trend)
  0.33 – 0.66  → MODERATE
  0.66 – 1.00  → STRONG

Cross events
------------
  Golden Cross — EMA_50 crosses ABOVE EMA_200  (long-term bullish signal)
  Death Cross  — EMA_50 crosses BELOW EMA_200  (long-term bearish signal)

Public API
----------
    detector = TrendDetector()
    result   = detector.detect(df)         # → TrendResult
    output   = detector.detect_as_dict(df) # → plain dict

    # Or get just what you need:
    df_with_emas = detector.add_emas(df)   # df extended with ema_50, ema_200
    direction    = detector.get_direction(df)  # → TrendDirection
    strength     = detector.get_strength(df)   # → float

Dependencies
------------
    pandas, numpy  (no external TA library required)
    utils.logger
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────

class TrendDirection(str, Enum):
    """
    Directional bias returned by TrendDetector.

    BULLISH  → close > EMA_50 AND close > EMA_200
    BEARISH  → close < EMA_50 AND close < EMA_200
    NEUTRAL  → price sits between the two EMAs (mixed signal)
    UNKNOWN  → insufficient data to determine direction
    """
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"
    UNKNOWN = "UNKNOWN"


class TrendStrengthLabel(str, Enum):
    """Human-readable strength band mapped from the 0–1 score."""
    STRONG   = "STRONG"    # score > 0.66
    MODERATE = "MODERATE"  # 0.33 < score ≤ 0.66
    WEAK     = "WEAK"      # score ≤ 0.33
    UNKNOWN  = "UNKNOWN"   # no data


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrendResult:
    """
    Full output of TrendDetector.detect().

    Attributes
    ----------
    direction       : BULLISH / BEARISH / NEUTRAL / UNKNOWN
    strength        : composite score 0.0 – 1.0
    strength_label  : STRONG / MODERATE / WEAK / UNKNOWN
    ema_50          : current EMA 50 value
    ema_200         : current EMA 200 value
    close           : current closing price
    ema_separation_pct  : (EMA_50 − EMA_200) / EMA_200 × 100
                          positive → EMA_50 above EMA_200 (bullish stack)
                          negative → EMA_50 below EMA_200 (bearish stack)
    price_vs_ema50_pct  : (close − EMA_50) / EMA_50 × 100
    price_vs_ema200_pct : (close − EMA_200) / EMA_200 × 100
    ema50_slope     : EMA_50 slope angle in degrees (linear regression)
    golden_cross    : True if EMA_50 just crossed above EMA_200
    death_cross     : True if EMA_50 just crossed below EMA_200
    bars_analysed   : rows in the input DataFrame
    notes           : plain-English explanation of the classification
    """
    direction:      TrendDirection      = TrendDirection.UNKNOWN
    strength:       float               = 0.0
    strength_label: TrendStrengthLabel  = TrendStrengthLabel.UNKNOWN

    # Indicator snapshot
    ema_50:  float = 0.0
    ema_200: float = 0.0
    close:   float = 0.0

    # Derived distances (percentages)
    ema_separation_pct:  float = 0.0   # (EMA50 - EMA200) / EMA200 * 100
    price_vs_ema50_pct:  float = 0.0   # (close - EMA50)  / EMA50  * 100
    price_vs_ema200_pct: float = 0.0   # (close - EMA200) / EMA200 * 100

    # EMA slope
    ema50_slope: float = 0.0           # degrees

    # Cross events
    golden_cross: bool = False
    death_cross:  bool = False

    # Meta
    bars_analysed: int = 0
    notes:         str = ""

    # ── Helpers ───────────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """
        Return a plain dictionary representation suitable for logging,
        Telegram messages, and the signal generator.

        Example output
        --------------
            {
                "direction":          "BULLISH",
                "strength":           0.74,
                "strength_label":     "STRONG",
                "ema_50":             2285.40,
                "ema_200":            2241.10,
                "close":              2312.00,
                "ema_separation_pct": 1.98,
                "price_vs_ema50_pct": 1.16,
                "price_vs_ema200_pct":3.17,
                "ema50_slope":        8.3,
                "golden_cross":       False,
                "death_cross":        False,
                "notes":              "BULLISH — price above both EMAs..."
            }
        """
        return {
            "direction":           self.direction.value,
            "strength":            round(self.strength, 3),
            "strength_label":      self.strength_label.value,
            "ema_50":              round(self.ema_50, 2),
            "ema_200":             round(self.ema_200, 2),
            "close":               round(self.close, 2),
            "ema_separation_pct":  round(self.ema_separation_pct, 3),
            "price_vs_ema50_pct":  round(self.price_vs_ema50_pct, 3),
            "price_vs_ema200_pct": round(self.price_vs_ema200_pct, 3),
            "ema50_slope":         round(self.ema50_slope, 2),
            "golden_cross":        self.golden_cross,
            "death_cross":         self.death_cross,
            "bars_analysed":       self.bars_analysed,
            "notes":               self.notes,
        }

    def allows_direction(self, trade_direction: str) -> bool:
        """
        Return True if a trade in `trade_direction` is permitted.

        In a confirmed BULLISH trend only BUY signals pass.
        In a confirmed BEARISH trend only SELL signals pass.
        NEUTRAL / UNKNOWN allow both directions (no strong opinion).

        Parameters
        ----------
        trade_direction : "BUY" or "SELL"
        """
        td = trade_direction.upper()
        if self.direction == TrendDirection.BULLISH:
            return td == "BUY"
        if self.direction == TrendDirection.BEARISH:
            return td == "SELL"
        return True   # NEUTRAL / UNKNOWN: no filter applied

    def __str__(self) -> str:
        sep_sign = "+" if self.ema_separation_pct >= 0 else ""
        return (
            f"TrendResult("
            f"direction={self.direction.value}, "
            f"strength={self.strength:.0%}({self.strength_label.value}), "
            f"EMA50={self.ema_50:.2f}, EMA200={self.ema_200:.2f}, "
            f"sep={sep_sign}{self.ema_separation_pct:.2f}%, "
            f"slope={self.ema50_slope:.1f}deg"
            f"{'  [GOLDEN CROSS]' if self.golden_cross else ''}"
            f"{'  [DEATH CROSS]'  if self.death_cross  else ''}"
            f")"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Detector
# ─────────────────────────────────────────────────────────────────────────────

class TrendDetector:
    """
    Detects trend direction and strength from OHLCV data using EMA 50 / 200.

    Decision logic
    --------------

    Step 1 — Compute EMA_50 and EMA_200 from the close series.

    Step 2 — Classify direction by comparing the current close to both EMAs:

        close > EMA_50  AND  close > EMA_200  →  BULLISH
        close < EMA_50  AND  close < EMA_200  →  BEARISH
        otherwise                             →  NEUTRAL
             (price between the two EMAs — market in transition)

    Step 3 — Measure trend strength from three normalised sub-scores:

        Sub-score A  EMA separation
        ────────────────────────────
        separation = |EMA_50 − EMA_200| / EMA_200 × 100  (percent)
        A wide separation means the trend has been running for a while and
        the two moving averages are "pulling apart" — a sign of conviction.
        Scaled against SEPARATION_STRONG_PCT (default 2.0 % for XAUUSD H1).

        Sub-score B  Price distance from both EMAs
        ────────────────────────────────────────────
        avg_pct = mean(|close − EMA_50|/EMA_50, |close − EMA_200|/EMA_200) × 100
        Scaled against PRICE_DIST_STRONG_PCT (default 3.0 %).

        Sub-score C  EMA_50 slope angle
        ────────────────────────────────
        A steep EMA slope means momentum is accelerating in the trend direction.
        Linear regression is fitted to the last SLOPE_LOOKBACK bars of EMA_50.
        The slope is normalised by price and converted to degrees.
        Scaled against SLOPE_STRONG_DEG (default 10.0 degrees).

        Final strength = weighted average of A, B, C:
            strength = 0.40 × A  +  0.35 × B  +  0.25 × C

    Step 4 — Cross detection (golden / death cross):
        A crossover is detected by comparing EMA_50 vs EMA_200 on the
        current bar vs the previous bar (sign change of the difference).
    """

    # ── Calibration constants ───────────────────────────────────────────────
    # These are the "fully strong" reference values used to normalise each
    # sub-score to [0, 1].  Values empirically calibrated for XAUUSD H1.
    # Override them if using a different symbol or timeframe.

    # EMA separation at which strength score A saturates at 1.0.
    # 2 % of EMA_200 means EMA_50 is 2 % above/below EMA_200 — typical for
    # a mature H1 trend on Gold.
    SEPARATION_STRONG_PCT: float = 2.0

    # Average price distance (from both EMAs) at which score B saturates at 1.0.
    PRICE_DIST_STRONG_PCT: float = 3.0

    # EMA_50 slope angle (degrees) at which score C saturates at 1.0.
    SLOPE_STRONG_DEG: float = 10.0

    # Number of bars used for slope linear regression.
    SLOPE_LOOKBACK: int = 5

    # Minimum bars the DataFrame must contain for a valid result.
    MIN_BARS_REQUIRED: int = 210   # ensures EMA_200 has at least 10 warm-up bars

    # Strength label thresholds.
    STRONG_THRESHOLD:   float = 0.66
    MODERATE_THRESHOLD: float = 0.33

    def __init__(
        self,
        slow_period:  int = 50,
        macro_period: int = 200,
        slope_lookback: int = 5,
    ) -> None:
        """
        Parameters
        ----------
        slow_period   : EMA period for the intermediate moving average (default 50)
        macro_period  : EMA period for the macro trend filter (default 200)
        slope_lookback: bars used in the EMA slope linear regression (default 5)

        Note: a "fast" EMA (e.g. EMA 20) is intentionally NOT part of the direction
        rules — only EMA_50 and EMA_200 determine direction.  EMA_20 is only used
        by MarketRegimeClassifier for slope computation.
        """
        if slow_period >= macro_period:
            raise ValueError(
                f"slow_period ({slow_period}) must be less than "
                f"macro_period ({macro_period})."
            )
        self.slow_period   = slow_period    # EMA 50
        self.macro_period  = macro_period   # EMA 200
        self.SLOPE_LOOKBACK = slope_lookback
        self.MIN_BARS_REQUIRED = macro_period + 10  # at least 10 fully-settled bars

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def detect(self, df: pd.DataFrame) -> TrendResult:
        """
        Run the full trend detection pipeline on an OHLCV DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain at minimum the column: close.
            Rows sorted oldest → newest (standard get_candles() output).

        Returns
        -------
        TrendResult
            Direction, strength, EMA values, distances, slope, cross events.
        """
        # ── Guard: required columns ───────────────────────────────────────
        if "close" not in df.columns:
            logger.error("detect() — 'close' column missing from DataFrame.")
            return TrendResult(
                direction=TrendDirection.UNKNOWN,
                notes="Missing 'close' column in DataFrame.",
            )

        # ── Guard: minimum bar count ──────────────────────────────────────
        if len(df) < self.MIN_BARS_REQUIRED:
            logger.warning(
                f"detect() — only {len(df)} bars; need {self.MIN_BARS_REQUIRED}. "
                "EMA_200 has insufficient warmup — returning UNKNOWN."
            )
            return TrendResult(
                direction=TrendDirection.UNKNOWN,
                bars_analysed=len(df),
                notes=(
                    f"Insufficient bars: {len(df)} provided, "
                    f"{self.MIN_BARS_REQUIRED} required for EMA_{self.macro_period}."
                ),
            )

        df = df.copy().reset_index(drop=True)

        # ── Step 1 — Compute EMAs ──────────────────────────────────────────
        ema_slow  = self._ema(df["close"], self.slow_period)    # EMA 50
        ema_macro = self._ema(df["close"], self.macro_period)   # EMA 200

        # Current-bar snapshots (last row)
        current_close   = float(df["close"].iloc[-1])
        current_ema50   = float(ema_slow.iloc[-1])
        current_ema200  = float(ema_macro.iloc[-1])

        logger.debug(
            f"detect()  close={current_close:.2f}  "
            f"EMA{self.slow_period}={current_ema50:.2f}  "
            f"EMA{self.macro_period}={current_ema200:.2f}"
        )

        # ── Step 2 — Classify direction ────────────────────────────────────
        #
        # Rule 1 — BULLISH:
        #   Price must be above BOTH the slow (50) and macro (200) EMA.
        #   This confirms the market has bullish momentum on both the
        #   intermediate and the long-term timeframe.
        #
        # Rule 2 — BEARISH:
        #   Price must be below BOTH EMAs — confirms bearish conviction
        #   across both timeframes.
        #
        # Rule 3 — NEUTRAL:
        #   Price sits between the two EMAs.  This is a transitional zone:
        #   the market may be reversing or simply compressing.  Neither
        #   strong bull nor bear signals should be taken here.
        #
        price_above_ema50  = current_close > current_ema50
        price_above_ema200 = current_close > current_ema200

        if price_above_ema50 and price_above_ema200:
            direction = TrendDirection.BULLISH
        elif (not price_above_ema50) and (not price_above_ema200):
            direction = TrendDirection.BEARISH
        else:
            # Price is between EMA_50 and EMA_200 → mixed / transition
            direction = TrendDirection.NEUTRAL

        # ── Step 3 — Derived distance metrics ─────────────────────────────
        # These are used for both strength scoring and the output dict.

        # EMA separation: how far the two EMAs have diverged.
        # Positive  → EMA_50 above EMA_200  (bullish stacking)
        # Negative  → EMA_50 below EMA_200  (bearish stacking)
        ema_separation_pct = (
            (current_ema50 - current_ema200) / current_ema200 * 100.0
            if current_ema200 != 0 else 0.0
        )

        # How far is price above/below each EMA, expressed as a percentage?
        price_vs_ema50_pct  = (
            (current_close - current_ema50) / current_ema50 * 100.0
            if current_ema50 != 0 else 0.0
        )
        price_vs_ema200_pct = (
            (current_close - current_ema200) / current_ema200 * 100.0
            if current_ema200 != 0 else 0.0
        )

        # ── Step 4 — EMA_50 slope angle ────────────────────────────────────
        ema50_slope = self._slope_angle(ema_slow, self.SLOPE_LOOKBACK)

        # ── Step 5 — Cross detection ───────────────────────────────────────
        #
        # A Golden Cross occurs when EMA_50 crosses ABOVE EMA_200.
        # The test compares the sign of (EMA_50 − EMA_200) on the current bar
        # versus the previous bar.  A sign flip from negative to positive is
        # a fresh golden cross.
        #
        # A Death Cross is the opposite: EMA_50 crosses BELOW EMA_200.
        #
        golden_cross, death_cross = self._detect_crosses(ema_slow, ema_macro)

        # ── Step 6 — Compute strength ──────────────────────────────────────
        #
        # Strength is only meaningful when direction is BULLISH or BEARISH.
        # For NEUTRAL markets strength is set to 0 (no trend to measure).
        #
        if direction in (TrendDirection.BULLISH, TrendDirection.BEARISH):
            strength = self._compute_strength(
                ema_separation_pct = ema_separation_pct,
                price_vs_ema50_pct = price_vs_ema50_pct,
                price_vs_ema200_pct= price_vs_ema200_pct,
                ema50_slope        = ema50_slope,
                direction          = direction,
            )
        else:
            strength = 0.0

        strength_label = self._strength_label(strength, direction)

        # ── Step 7 — Build notes string ────────────────────────────────────
        notes = self._build_notes(
            direction          = direction,
            strength           = strength,
            strength_label     = strength_label,
            current_close      = current_close,
            current_ema50      = current_ema50,
            current_ema200     = current_ema200,
            ema_separation_pct = ema_separation_pct,
            ema50_slope        = ema50_slope,
            golden_cross       = golden_cross,
            death_cross        = death_cross,
        )

        result = TrendResult(
            direction          = direction,
            strength           = round(strength, 3),
            strength_label     = strength_label,
            ema_50             = round(current_ema50, 2),
            ema_200            = round(current_ema200, 2),
            close              = round(current_close, 2),
            ema_separation_pct = round(ema_separation_pct, 3),
            price_vs_ema50_pct = round(price_vs_ema50_pct, 3),
            price_vs_ema200_pct= round(price_vs_ema200_pct, 3),
            ema50_slope        = round(ema50_slope, 2),
            golden_cross       = golden_cross,
            death_cross        = death_cross,
            bars_analysed      = len(df),
            notes              = notes,
        )

        logger.info(f"Trend: {result}")
        return result

    def detect_as_dict(self, df: pd.DataFrame) -> dict:
        """
        Convenience wrapper — return detect() result as a plain dict.

        Example
        -------
            {
                "direction":          "BULLISH",
                "strength":           0.74,
                "strength_label":     "STRONG",
                "ema_50":             2285.40,
                "ema_200":            2241.10,
                "close":              2312.00,
                "ema_separation_pct": 1.98,
                "price_vs_ema50_pct": 1.16,
                "price_vs_ema200_pct":3.17,
                "ema50_slope":        8.3,
                "golden_cross":       False,
                "death_cross":        False,
                ...
            }
        """
        return self.detect(df).to_dict()

    def add_emas(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return a copy of `df` with EMA columns appended.

        Added columns
        -------------
        ema_50   : EMA using slow_period  (default 50)
        ema_200  : EMA using macro_period (default 200)

        Parameters
        ----------
        df : pd.DataFrame
            Must contain a 'close' column.

        Returns
        -------
        pd.DataFrame
            Original DataFrame + ema_50, ema_200 columns.
        """
        if "close" not in df.columns:
            raise ValueError("DataFrame must contain a 'close' column.")

        out = df.copy()
        out[f"ema_{self.slow_period}"]  = self._ema(out["close"], self.slow_period)
        out[f"ema_{self.macro_period}"] = self._ema(out["close"], self.macro_period)
        return out

    def get_direction(self, df: pd.DataFrame) -> TrendDirection:
        """Quick accessor — return only the trend direction."""
        return self.detect(df).direction

    def get_strength(self, df: pd.DataFrame) -> float:
        """Quick accessor — return only the strength score (0.0 – 1.0)."""
        return self.detect(df).strength

    def get_ema_values(self, df: pd.DataFrame) -> dict:
        """
        Return current EMA values without running the full detect() pipeline.

        Returns
        -------
        dict
            {"ema_50": float, "ema_200": float, "close": float}
        """
        if "close" not in df.columns:
            raise ValueError("DataFrame must contain a 'close' column.")

        ema50  = float(self._ema(df["close"], self.slow_period).iloc[-1])
        ema200 = float(self._ema(df["close"], self.macro_period).iloc[-1])
        return {
            f"ema_{self.slow_period}":  round(ema50,  2),
            f"ema_{self.macro_period}": round(ema200, 2),
            "close": round(float(df["close"].iloc[-1]), 2),
        }

    # =========================================================================
    # INDICATOR CALCULATIONS
    # =========================================================================

    def _ema(self, series: pd.Series, period: int) -> pd.Series:
        """
        Compute a standard Exponential Moving Average.

        Uses pandas ewm(span=period) which sets:
            alpha = 2 / (period + 1)

        This matches the EMA formula used in MT5, TradingView, and most
        charting platforms (NOT Wilder's smoothing).

        Parameters
        ----------
        series : pd.Series  — typically df["close"]
        period : int        — span of the EMA (e.g. 50, 200)

        Returns
        -------
        pd.Series           — same index as `series`; first rows are NaN
        """
        return series.ewm(span=period, min_periods=period, adjust=False).mean()

    def _slope_angle(self, ema_series: pd.Series, lookback: int) -> float:
        """
        Compute the slope of the EMA over the last `lookback` bars and
        express it in degrees.

        Method
        ------
        1. Extract the last `lookback` non-NaN values from `ema_series`.
        2. Fit a straight line (linear regression) through those values.
        3. Normalise the slope by the mean EMA value so the result is
           scale-independent (works regardless of whether XAUUSD is at $200 or $2000).
        4. Multiply by 1000 to amplify — without this, the normalised slope
           is on the order of 0.0001, giving angles indistinguishable from 0.
        5. Take arctan → degrees.

        Returns
        -------
        float
             Positive  → EMA rising  (bullish momentum)
             Negative  → EMA falling (bearish momentum)
             Near zero → flat        (no momentum)
        """
        valid = ema_series.dropna()
        if len(valid) < lookback:
            return 0.0

        tail   = valid.iloc[-lookback:].values.astype(float)
        x      = np.arange(len(tail), dtype=float)
        slope, _ = np.polyfit(x, tail, 1)

        mean_val = float(np.mean(tail))
        if mean_val == 0.0:
            return 0.0

        # Normalise and convert to degrees
        # Scale factor 1000: empirically good for XAUUSD H1 (typical slopes ~0.0002)
        return math.degrees(math.atan((slope / mean_val) * 1000))

    def _detect_crosses(
        self,
        ema_slow:  pd.Series,   # EMA 50
        ema_macro: pd.Series,   # EMA 200
    ) -> tuple[bool, bool]:
        """
        Detect whether a Golden Cross or Death Cross just occurred.

        A cross is detected by comparing the difference (EMA_50 − EMA_200)
        on the most recent bar to the bar before it.

        Golden Cross : difference flipped from negative → positive
                       (EMA_50 just crossed ABOVE EMA_200)

        Death Cross  : difference flipped from positive → negative
                       (EMA_50 just crossed BELOW EMA_200)

        Returns
        -------
        (golden_cross: bool, death_cross: bool)
        """
        diff = ema_slow - ema_macro

        # Need at least two valid (non-NaN) values to compare
        valid_diff = diff.dropna()
        if len(valid_diff) < 2:
            return False, False

        prev_diff = float(valid_diff.iloc[-2])
        curr_diff = float(valid_diff.iloc[-1])

        golden_cross = (prev_diff <= 0.0) and (curr_diff > 0.0)
        death_cross  = (prev_diff >= 0.0) and (curr_diff < 0.0)

        if golden_cross:
            logger.info(
                f"Golden Cross detected! "
                f"EMA{self.slow_period} crossed above EMA{self.macro_period}."
            )
        if death_cross:
            logger.info(
                f"Death Cross detected! "
                f"EMA{self.slow_period} crossed below EMA{self.macro_period}."
            )

        return golden_cross, death_cross

    # =========================================================================
    # STRENGTH SCORING
    # =========================================================================

    def _compute_strength(
        self,
        ema_separation_pct:  float,
        price_vs_ema50_pct:  float,
        price_vs_ema200_pct: float,
        ema50_slope:         float,
        direction:           TrendDirection,
    ) -> float:
        """
        Build the composite trend strength score [0.0, 1.0].

        Sub-score A — EMA separation (weight 40 %)
        -------------------------------------------
        For a BULLISH trend: EMA_50 should be above EMA_200 (positive separation).
        For a BEARISH trend: EMA_50 should be below EMA_200 (negative separation).
        The sub-score rewards direction-consistent separation.

        Sub-score B — Price distance from EMAs (weight 35 %)
        ------------------------------------------------------
        Takes the average of |price_vs_ema50| and |price_vs_ema200|.
        A price that is far above BOTH EMAs in a bull trend = high conviction.
        We use the absolute raw values (not direction-checked) because we only
        reach this method when direction is already confirmed BULLISH/BEARISH.

        Sub-score C — EMA_50 slope (weight 25 %)
        -------------------------------------------
        A positive slope in a bull trend (or negative slope in bear) adds confidence.
        Slope in the opposite direction reduces sub-score C toward zero.

        All sub-scores are clamped to [0, 1] before weighting.
        """
        # Sub-score A: EMA separation
        # Signed separation so counter-directional stacking scores zero/negative
        if direction == TrendDirection.BULLISH:
            signed_sep = ema_separation_pct           # positive = EMA50 above EMA200
        else:
            signed_sep = -ema_separation_pct          # flip sign for BEARISH

        score_a = min(max(signed_sep / self.SEPARATION_STRONG_PCT, 0.0), 1.0)

        # Sub-score B: Average price distance from both EMAs (absolute values)
        avg_price_dist = (abs(price_vs_ema50_pct) + abs(price_vs_ema200_pct)) / 2.0
        score_b = min(avg_price_dist / self.PRICE_DIST_STRONG_PCT, 1.0)

        # Sub-score C: EMA_50 slope
        # For BULLISH: positive slope adds strength; negative slope subtracts.
        # Scale by SLOPE_STRONG_DEG, clamp to [0, 1].
        if direction == TrendDirection.BULLISH:
            signed_slope = ema50_slope                # positive slope is bullish
        else:
            signed_slope = -ema50_slope               # negative slope is bearish

        score_c = min(max(signed_slope / self.SLOPE_STRONG_DEG, 0.0), 1.0)

        # Weighted composite
        strength = 0.40 * score_a + 0.35 * score_b + 0.25 * score_c
        return min(max(strength, 0.0), 1.0)

    def _strength_label(
        self, strength: float, direction: TrendDirection
    ) -> TrendStrengthLabel:
        """Map a 0–1 strength score to a human-readable band."""
        if direction in (TrendDirection.UNKNOWN, TrendDirection.NEUTRAL):
            return TrendStrengthLabel.UNKNOWN
        if strength >= self.STRONG_THRESHOLD:
            return TrendStrengthLabel.STRONG
        if strength >= self.MODERATE_THRESHOLD:
            return TrendStrengthLabel.MODERATE
        return TrendStrengthLabel.WEAK

    # =========================================================================
    # NOTES BUILDER
    # =========================================================================

    def _build_notes(
        self,
        direction:          TrendDirection,
        strength:           float,
        strength_label:     TrendStrengthLabel,
        current_close:      float,
        current_ema50:      float,
        current_ema200:     float,
        ema_separation_pct: float,
        ema50_slope:        float,
        golden_cross:       bool,
        death_cross:        bool,
    ) -> str:
        """
        Build a human-readable explanation of the detection result.
        Used in log messages, tests, and Telegram debug messages.
        """
        ema50_lbl  = f"EMA{self.slow_period}"
        ema200_lbl = f"EMA{self.macro_period}"
        sep_sign   = "+" if ema_separation_pct >= 0 else ""

        if direction == TrendDirection.BULLISH:
            position = (
                f"close={current_close:.2f} > {ema50_lbl}={current_ema50:.2f} "
                f"> {ema200_lbl}={current_ema200:.2f}"
            )
            note = (
                f"BULLISH — {position}. "
                f"EMA separation: {sep_sign}{ema_separation_pct:.2f}% "
                f"(EMA50 stacked above EMA200). "
                f"EMA50 slope: {ema50_slope:+.1f} deg. "
                f"Strength: {strength:.0%} ({strength_label.value})."
            )
        elif direction == TrendDirection.BEARISH:
            position = (
                f"close={current_close:.2f} < {ema50_lbl}={current_ema50:.2f} "
                f"< {ema200_lbl}={current_ema200:.2f}"
            )
            note = (
                f"BEARISH — {position}. "
                f"EMA separation: {sep_sign}{ema_separation_pct:.2f}% "
                f"(EMA50 stacked below EMA200). "
                f"EMA50 slope: {ema50_slope:+.1f} deg. "
                f"Strength: {strength:.0%} ({strength_label.value})."
            )
        else:
            # NEUTRAL: price is between the two EMAs
            note = (
                f"NEUTRAL — close={current_close:.2f} sits between "
                f"{ema50_lbl}={current_ema50:.2f} and "
                f"{ema200_lbl}={current_ema200:.2f}. "
                f"Market in transition / EMA compression zone. "
                f"Avoid directional signals until alignment clarifies."
            )

        if golden_cross:
            note += f"  [!] GOLDEN CROSS — EMA{self.slow_period} just crossed above EMA{self.macro_period}."
        if death_cross:
            note += f"  [!] DEATH CROSS — EMA{self.slow_period} just crossed below EMA{self.macro_period}."

        return note

    # =========================================================================
    # REPR
    # =========================================================================

    def __repr__(self) -> str:
        return (
            f"TrendDetector("
            f"EMA{self.slow_period}/EMA{self.macro_period}, "
            f"slope_lookback={self.SLOPE_LOOKBACK}, "
            f"min_bars={self.MIN_BARS_REQUIRED}"
            f")"
        )
