"""
strategy/market_regime.py
==========================
Market regime classifier for XAUUSD.

This module analyses raw OHLCV data and classifies the current market
environment into one of four mutually exclusive regimes:

    TRENDING       — strong directional move (ADX high, EMA slope steep)
    RANGING        — sideways / consolidation (ADX low, EMA slope flat)
    HIGH_VOLATILITY — ATR abnormally above its rolling average (news spike, breakout)
    LOW_VOLATILITY  — ATR abnormally below its rolling average (dead market, pre-session)

Within TRENDING the classifier also determines directional bias:

    BULLISH  — price trending upward  (close > EMA200, +DI > -DI, slope positive)
    BEARISH  — price trending downward (close < EMA200, -DI > +DI, slope negative)

Public API
----------
    classifier = MarketRegimeClassifier(settings)
    result     = classifier.classify(df)        # → RegimeResult
    direction  = classifier.get_trend_direction(df)  # → TrendDirection enum

    # Or use the convenience dict-output method:
    output = classifier.classify_as_dict(df)
    # → {"regime": "TRENDING", "trend_direction": "BULLISH", ...}

Indicator logic overview
------------------------
                       ┌──────────────────────────────────┐
                       │  Step 1 — Volatility gate (ATR)  │
                       │  atr_ratio = current ATR /        │
                       │             rolling_mean(ATR, N)  │
                       │                                   │
                       │  atr_ratio > HIGH_VOL_THRESHOLD   │
                       │       →  HIGH_VOLATILITY          │
                       │                                   │
                       │  atr_ratio < LOW_VOL_THRESHOLD    │
                       │       →  LOW_VOLATILITY           │
                       └──────────────┬───────────────────┘
                                      │ (normal volatility)
                       ┌──────────────▼───────────────────┐
                       │  Step 2 — Trend test (ADX)       │
                       │                                   │
                       │  ADX ≥ adx_trend_threshold (25)  │
                       │       →  TRENDING                 │
                       │                                   │
                       │  ADX < adx_trend_threshold        │
                       │       →  RANGING                  │
                       └──────────────┬───────────────────┘
                                      │ (TRENDING only)
                       ┌──────────────▼───────────────────┐
                       │  Step 3 — Direction (+DI / -DI,  │
                       │            EMA slope, price vs    │
                       │            EMA200)                │
                       │                                   │
                       │  Majority vote:                   │
                       │    BULLISH or BEARISH             │
                       └──────────────────────────────────┘

Dependencies
------------
    pandas, numpy     — pure-Python (no extra install required)
    config.settings   — for default periods and thresholds
    utils.logger      — structured logging

Note on pandas-ta
-----------------
This module intentionally implements ATR and ADX from scratch using
only pandas + numpy so it works even if pandas-ta is not installed.
If pandas-ta IS installed, you can call the `_compute_atr_pdt()` and
`_compute_adx_pdt()` helpers which delegate to it for extra speed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import Settings
from utils.logger import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────

class RegimeLabel(str, Enum):
    """Four mutually exclusive market regime states."""
    TRENDING        = "TRENDING"
    RANGING         = "RANGING"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY  = "LOW_VOLATILITY"
    UNKNOWN         = "UNKNOWN"        # fallback when data is insufficient


class TrendDirection(str, Enum):
    """Directional bias — only meaningful when regime == TRENDING."""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"    # direction unclear (low-conviction TRENDING)


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RegimeResult:
    """
    Full output of the market regime classifier.

    Attributes
    ----------
    regime          : one of TRENDING / RANGING / HIGH_VOLATILITY / LOW_VOLATILITY
    trend_direction : BULLISH / BEARISH / NEUTRAL
                      (always NEUTRAL when regime != TRENDING)
    adx             : latest ADX value  (0 – 100)
    plus_di         : latest +DI value
    minus_di        : latest -DI value
    atr             : latest ATR value (in price points)
    atr_ratio       : current ATR / rolling-mean ATR  (1.0 = average)
    ema_slope       : fast-EMA slope angle in degrees
    confidence      : composite confidence score 0.0 – 1.0
    bars_analysed   : number of bars in the input DataFrame
    notes           : human-readable explanation of the classification
    """
    regime:          RegimeLabel   = RegimeLabel.UNKNOWN
    trend_direction: TrendDirection = TrendDirection.NEUTRAL

    # Indicator snapshot
    adx:       float = 0.0
    plus_di:   float = 0.0
    minus_di:  float = 0.0
    atr:       float = 0.0
    atr_ratio: float = 1.0
    ema_slope: float = 0.0   # degrees

    # Meta
    confidence:    float = 0.0
    bars_analysed: int   = 0
    notes:         str   = ""

    # ── Dict / str output helpers ──────────────────────────────────────────

    def to_dict(self) -> dict:
        """
        Return the primary classification as a plain dict.

        Example
        -------
            {
                "regime":          "TRENDING",
                "trend_direction": "BULLISH",
                "adx":             32.4,
                "atr":             12.7,
                "atr_ratio":       1.08,
                "ema_slope":       7.2,
                "confidence":      0.81,
                "notes":           "Strong bullish trend ..."
            }
        """
        return {
            "regime":          self.regime.value,
            "trend_direction": self.trend_direction.value,
            "adx":             round(self.adx, 2),
            "plus_di":         round(self.plus_di, 2),
            "minus_di":        round(self.minus_di, 2),
            "atr":             round(self.atr, 2),
            "atr_ratio":       round(self.atr_ratio, 3),
            "ema_slope":       round(self.ema_slope, 2),
            "confidence":      round(self.confidence, 3),
            "bars_analysed":   self.bars_analysed,
            "notes":           self.notes,
        }

    def is_tradeable(self) -> bool:
        """
        Return True if the regime permits signal generation.

        Rules:
        - HIGH_VOLATILITY signals are always blocked (risk of news-spike entry)
        - LOW_VOLATILITY signals are blocked (market not moving, spreads wide)
        - TRENDING and RANGING are tradeable (different strategy sub-modes)
        """
        return self.regime in (RegimeLabel.TRENDING, RegimeLabel.RANGING)

    def allows_direction(self, direction: str) -> bool:
        """
        Return True if the given trade direction (BUY/SELL) is permitted.

        In TRENDING markets only the trend direction is allowed.
        In non-TRENDING markets both directions are considered.

        Parameters
        ----------
        direction : "BUY" or "SELL"
        """
        if self.regime != RegimeLabel.TRENDING:
            return True   # RANGING allows both directions
        if self.trend_direction == TrendDirection.NEUTRAL:
            return True   # Low-conviction trend — allow both
        if direction.upper() == "BUY":
            return self.trend_direction == TrendDirection.BULLISH
        if direction.upper() == "SELL":
            return self.trend_direction == TrendDirection.BEARISH
        return False

    def __str__(self) -> str:
        return (
            f"RegimeResult("
            f"regime={self.regime.value}, "
            f"direction={self.trend_direction.value}, "
            f"ADX={self.adx:.1f}, "
            f"ATR={self.atr:.2f}({self.atr_ratio:.2f}x), "
            f"slope={self.ema_slope:.1f}°, "
            f"confidence={self.confidence:.0%}"
            f")"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Classifier
# ─────────────────────────────────────────────────────────────────────────────

class MarketRegimeClassifier:
    """
    Classifies the current XAUUSD market environment from an OHLCV DataFrame.

    The classifier uses three independent indicator layers:

    Layer 1 — Volatility (ATR)
        Measures the ratio of the current ATR to its rolling average.
        An abnormally high ratio flags HIGH_VOLATILITY (news/spike).
        An abnormally low ratio flags LOW_VOLATILITY (dead session).

    Layer 2 — Trend strength (ADX)
        ADX (Wilder's Average Directional Index) quantifies how strongly
        price is trending, independent of direction.
        ADX ≥ adx_trend_threshold  →  market is TRENDING.
        ADX <  adx_trend_threshold  →  market is RANGING.

    Layer 3 — Trend direction (+DI/-DI, EMA slope, price vs EMA 200)
        Only computed when TRENDING is confirmed.
        Three sub-signals vote on the direction (BULLISH / BEARISH):
            a) Directional Index: +DI > -DI → bullish vote
            b) EMA slope:         slope > +threshold → bullish vote
            c) Price vs EMA 200:  close > ema200 → bullish vote
        Majority of the three sub-signals determines the final direction.
    """

    # ── Classification thresholds ──────────────────────────────────────────
    # These are stored as instance attributes so they can be overridden
    # at construction time or patched in unit tests.

    # ATR ratio above this → HIGH_VOLATILITY (2.0 = twice the recent average)
    HIGH_VOL_ATR_RATIO: float = 2.0
    # ATR ratio below this → LOW_VOLATILITY (0.5 = half the recent average)
    LOW_VOL_ATR_RATIO:  float = 0.5
    # Bars to average ATR over for the ratio denominator
    ATR_RATIO_LOOKBACK: int   = 20
    # EMA slope angle (degrees) above which the slope is "significant"
    SLOPE_SIGNIFICANCE_ANGLE: float = 3.0

    # Minimum number of bars required to produce a valid result
    MIN_BARS_REQUIRED: int = 50

    def __init__(self, settings: Optional[Settings] = None) -> None:
        """
        Parameters
        ----------
        settings : Settings, optional
            If provided, ADX period, ADX threshold, ATR period, and EMA
            periods are read from the settings object.
            If None, class-level defaults are used (good for unit tests).
        """
        if settings is not None:
            self.adx_period   : int   = settings.ADX_PERIOD            # default 14
            self.adx_threshold: float = settings.ADX_TREND_THRESHOLD   # default 25.0
            self.atr_period   : int   = settings.ATR_PERIOD            # default 14
            self.ema_fast     : int   = settings.EMA_FAST_PERIOD       # default 20
            self.ema_slow     : int   = settings.EMA_SLOW_PERIOD       # default 50
            self.ema_macro    : int   = settings.EMA_MACRO_PERIOD      # default 200
            self.slope_lookback: int  = settings.EMA_SLOPE_LOOKBACK    # default 5
        else:
            # Standalone defaults — matches standard trading platform settings
            self.adx_period    = 14
            self.adx_threshold = 25.0
            self.atr_period    = 14
            self.ema_fast      = 20
            self.ema_slow      = 50
            self.ema_macro     = 200
            self.slope_lookback = 5

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def classify(self, df: pd.DataFrame) -> RegimeResult:
        """
        Classify the market regime from an OHLCV DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns: open, high, low, close.
            Rows should be sorted oldest → newest (standard get_candles output).
            Minimum rows: MIN_BARS_REQUIRED (50).

        Returns
        -------
        RegimeResult
            Full classification result with all indicator values embedded.

        Notes
        -----
        The most recent (last) row is used as the "current bar" for
        indicator snapshot values.  Earlier rows provide history for
        rolling calculations.
        """
        # ── Guard: check required columns ─────────────────────────────────
        required = {"open", "high", "low", "close"}
        missing  = required - set(df.columns)
        if missing:
            logger.error(f"classify() — missing columns: {missing}")
            return RegimeResult(
                regime=RegimeLabel.UNKNOWN,
                notes=f"Missing required DataFrame columns: {missing}",
            )

        # ── Guard: check minimum bar count ────────────────────────────────
        if len(df) < self.MIN_BARS_REQUIRED:
            logger.warning(
                f"classify() — only {len(df)} bars available; "
                f"need at least {self.MIN_BARS_REQUIRED}. Returning UNKNOWN."
            )
            return RegimeResult(
                regime=RegimeLabel.UNKNOWN,
                bars_analysed=len(df),
                notes=(
                    f"Insufficient data: {len(df)} bars, "
                    f"need {self.MIN_BARS_REQUIRED}."
                ),
            )

        # Work on a copy to avoid mutating the caller's DataFrame
        df = df.copy().reset_index(drop=True)

        logger.debug(
            f"classify() — analysing {len(df)} bars  "
            f"(ADX_period={self.adx_period}, "
            f"ADX_threshold={self.adx_threshold}, "
            f"ATR_period={self.atr_period})"
        )

        # ── Step 1 — Compute indicators ───────────────────────────────────
        atr_series              = self._compute_atr(df)
        adx_series, pdi, mdi   = self._compute_adx(df)
        ema_fast_series         = self._compute_ema(df["close"], self.ema_fast)
        ema_macro_series        = self._compute_ema(df["close"], self.ema_macro)

        # Snapshot: values at the most recent (last) bar
        current_atr   = float(atr_series.iloc[-1])
        current_adx   = float(adx_series.iloc[-1])
        current_pdi   = float(pdi.iloc[-1])
        current_mdi   = float(mdi.iloc[-1])
        current_close = float(df["close"].iloc[-1])
        current_ema200 = float(ema_macro_series.iloc[-1])

        # ATR ratio: current ATR vs its rolling mean (tells us if volatility is abnormal)
        atr_rolling_mean = (
            atr_series
            .rolling(window=self.ATR_RATIO_LOOKBACK, min_periods=1)
            .mean()
        )
        atr_ratio       = float(current_atr / atr_rolling_mean.iloc[-1]) \
                          if float(atr_rolling_mean.iloc[-1]) > 0 else 1.0

        # EMA slope angle in degrees
        ema_slope_angle = self._compute_slope_angle(
            ema_fast_series,
            lookback=self.slope_lookback,
        )

        # ── Step 2 — Volatility gate (Layer 1) ───────────────────────────
        #
        # We check volatility FIRST because an abnormally volatile market
        # distorts ADX and slope readings.  A high-impact news bar, for
        # example, will show a huge ADX spike that looks "TRENDING" but is
        # actually a one-bar anomaly — not a tradeable trend.
        #
        if atr_ratio >= self.HIGH_VOL_ATR_RATIO:
            # ATR is at least 2× its recent average → likely news spike or
            # breakout from a long compression.  Block signals.
            result = RegimeResult(
                regime          = RegimeLabel.HIGH_VOLATILITY,
                trend_direction = TrendDirection.NEUTRAL,
                adx       = current_adx,
                plus_di   = current_pdi,
                minus_di  = current_mdi,
                atr       = current_atr,
                atr_ratio = atr_ratio,
                ema_slope = ema_slope_angle,
                confidence    = self._score_volatility_confidence(atr_ratio, high=True),
                bars_analysed = len(df),
                notes=(
                    f"ATR ratio {atr_ratio:.2f}x exceeds HIGH_VOL threshold "
                    f"({self.HIGH_VOL_ATR_RATIO:.1f}x). "
                    f"ATR={current_atr:.2f}. Likely news spike or breakout — "
                    "signal generation blocked."
                ),
            )
            logger.info(f"Regime: {result}")
            return result

        if atr_ratio <= self.LOW_VOL_ATR_RATIO:
            # ATR is less than half its recent average → market is extremely
            # quiet.  Spreads are relatively wide and moves are small — bad
            # conditions for intraday signal quality.
            result = RegimeResult(
                regime          = RegimeLabel.LOW_VOLATILITY,
                trend_direction = TrendDirection.NEUTRAL,
                adx       = current_adx,
                plus_di   = current_pdi,
                minus_di  = current_mdi,
                atr       = current_atr,
                atr_ratio = atr_ratio,
                ema_slope = ema_slope_angle,
                confidence    = self._score_volatility_confidence(atr_ratio, high=False),
                bars_analysed = len(df),
                notes=(
                    f"ATR ratio {atr_ratio:.2f}x below LOW_VOL threshold "
                    f"({self.LOW_VOL_ATR_RATIO:.1f}x). "
                    f"ATR={current_atr:.2f}. Market too quiet — "
                    "signal generation blocked."
                ),
            )
            logger.info(f"Regime: {result}")
            return result

        # ── Step 3 — Trend test (Layer 2 — ADX) ──────────────────────────
        #
        # ADX measures the STRENGTH of a trend, not its direction.
        # Classic interpretation:
        #   ADX < 20     →  no trend (weak / ranging)
        #   ADX 20 – 25  →  potential trend developing
        #   ADX ≥ 25     →  confirmed trend
        #   ADX ≥ 40     →  strong trend
        #   ADX ≥ 60     →  very strong / potentially exhausting trend
        #
        # We use the configurable adx_threshold (default 25) as the
        # cutoff between RANGING and TRENDING.
        #
        is_trending = current_adx >= self.adx_threshold

        if not is_trending:
            # ── RANGING ───────────────────────────────────────────────────
            # ADX below threshold: price is moving sideways with no clear
            # directional conviction.  SMC setups in ranging markets target
            # the opposite edge of the range.
            confidence = self._score_ranging_confidence(current_adx, self.adx_threshold)
            result = RegimeResult(
                regime          = RegimeLabel.RANGING,
                trend_direction = TrendDirection.NEUTRAL,
                adx       = current_adx,
                plus_di   = current_pdi,
                minus_di  = current_mdi,
                atr       = current_atr,
                atr_ratio = atr_ratio,
                ema_slope = ema_slope_angle,
                confidence    = confidence,
                bars_analysed = len(df),
                notes=(
                    f"ADX={current_adx:.1f} < threshold {self.adx_threshold} → RANGING. "
                    f"EMA slope={ema_slope_angle:.1f}°. "
                    f"ATR ratio={atr_ratio:.2f}x (normal). "
                    "Counter-trend and range-fade setups applicable."
                ),
            )
            logger.info(f"Regime: {result}")
            return result

        # ── Step 4 — Direction vote (Layer 3) ────────────────────────────
        #
        # ADX confirms a trend exists.  Now determine if it is BULLISH or
        # BEARISH using three independent sub-signals:
        #
        #   Vote A — Directional Index (+DI vs -DI):
        #       +DI > -DI → bulls are in control  (bullish vote)
        #       -DI > +DI → bears are in control  (bearish vote)
        #
        #   Vote B — EMA slope:
        #       slope >  +SLOPE_SIGNIFICANCE_ANGLE → bullish vote
        #       slope <  -SLOPE_SIGNIFICANCE_ANGLE → bearish vote
        #       |slope| ≤ SLOPE_SIGNIFICANCE_ANGLE → abstain (flat EMA)
        #
        #   Vote C — Price vs EMA 200:
        #       close > ema200 → price above macro trend line → bullish vote
        #       close < ema200 → price below macro trend line → bearish vote
        #
        # A simple majority (2 out of 3) decides the direction.
        # Ties → NEUTRAL (low-conviction trend, both directions acceptable).
        #
        direction, direction_votes = self._determine_direction(
            plus_di      = current_pdi,
            minus_di     = current_mdi,
            ema_slope    = ema_slope_angle,
            current_close= current_close,
            ema200       = current_ema200,
        )

        # Confidence for TRENDING: combines ADX level and directional agreement
        confidence = self._score_trending_confidence(
            adx       = current_adx,
            adx_threshold = self.adx_threshold,
            votes_for = direction_votes,
        )

        # Build a readable explanation of which votes agreed
        vote_details = self._format_vote_details(
            plus_di=current_pdi, minus_di=current_mdi,
            ema_slope=ema_slope_angle, close=current_close, ema200=current_ema200,
        )

        result = RegimeResult(
            regime          = RegimeLabel.TRENDING,
            trend_direction = direction,
            adx       = current_adx,
            plus_di   = current_pdi,
            minus_di  = current_mdi,
            atr       = current_atr,
            atr_ratio = atr_ratio,
            ema_slope = ema_slope_angle,
            confidence    = confidence,
            bars_analysed = len(df),
            notes=(
                f"ADX={current_adx:.1f} ≥ {self.adx_threshold} → TRENDING {direction.value}. "
                f"EMA slope={ema_slope_angle:.1f}°. "
                f"ATR ratio={atr_ratio:.2f}x (normal). "
                f"Direction votes: {vote_details}"
            ),
        )
        logger.info(f"Regime: {result}")
        return result

    def classify_as_dict(self, df: pd.DataFrame) -> dict:
        """
        Convenience wrapper — returns classify() result as a plain dict.

        Primary keys match the format requested:
            {"regime": "TRENDING", "trend_direction": "BULLISH", ...}

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV data (same requirements as classify()).

        Returns
        -------
        dict
        """
        return self.classify(df).to_dict()

    def get_trend_direction(self, df: pd.DataFrame) -> TrendDirection:
        """
        Quick accessor — return the trend direction without the full result.

        Returns
        -------
        TrendDirection
            BULLISH / BEARISH / NEUTRAL

        Notes
        -----
        NEUTRAL is returned both for non-TRENDING regimes and for
        TRENDING markets with ambiguous directional votes.
        """
        return self.classify(df).trend_direction

    def get_direction_as_int(self, df: pd.DataFrame) -> int:
        """
        Return trend direction as an integer gate value.

        Returns
        -------
        int
            +1  BULLISH
            -1  BEARISH
             0  NEUTRAL / RANGING / not directional
        """
        direction = self.get_trend_direction(df)
        if direction == TrendDirection.BULLISH:
            return 1
        if direction == TrendDirection.BEARISH:
            return -1
        return 0

    # =========================================================================
    # INDICATOR CALCULATIONS  (pure pandas / numpy — no external TA library)
    # =========================================================================

    def _compute_atr(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute Wilder's Average True Range (ATR).

        True Range (TR) is the greatest of:
            |high − low|          — current bar's range
            |high − prev_close|   — gap-up then current range
            |low  − prev_close|   — gap-down then current range

        ATR = Wilder's smoothed moving average of TR over `atr_period` bars.
        Wilder's smoothing is equivalent to an EMA with alpha = 1 / period.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns: high, low, close.

        Returns
        -------
        pd.Series
            ATR values, same index as df.  First (atr_period − 1) rows are NaN.
        """
        high  = df["high"]
        low   = df["low"]
        close = df["close"]

        # True Range components
        hl   = high - low                                          # current range
        h_pc = (high - close.shift(1)).abs()                       # high − prev_close
        l_pc = (low  - close.shift(1)).abs()                       # low  − prev_close

        tr = pd.concat([hl, h_pc, l_pc], axis=1).max(axis=1)

        # Wilder's smoothing: seed with simple mean of the first `period` TRs,
        # then apply alpha = 1/period smoothing for all subsequent bars.
        # This matches the MT5 / TradingView ATR formula exactly.
        atr = tr.ewm(
            alpha=1.0 / self.atr_period,
            min_periods=self.atr_period,
            adjust=False,
        ).mean()

        return atr

    def _compute_adx(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Compute Wilder's ADX (Average Directional Index) along with
        +DI and -DI directional indicators.

        Algorithm
        ---------
        1. True Range (TR) — same as ATR calculation.
        2. Directional Movement:
               +DM = max(high − prev_high, 0)  if > (prev_low − low) else 0
               -DM = max(prev_low − low,   0)  if > (high − prev_high) else 0
        3. Smooth +DM, -DM, and TR with Wilder's period.
        4. +DI = 100 × (+DM_smooth / TR_smooth)
           -DI = 100 × (-DM_smooth / TR_smooth)
        5. DX  = 100 × |+DI − -DI| / (+DI + -DI)
        6. ADX = Wilder's smooth of DX.

        Returns
        -------
        (adx, plus_di, minus_di) — all pd.Series, same index as df.
        """
        high  = df["high"]
        low   = df["low"]
        close = df["close"]
        n     = self.adx_period
        alpha = 1.0 / n

        # Directional Movement
        up_move   = high - high.shift(1)    # how much higher this bar vs previous
        down_move = low.shift(1) - low      # how much lower this bar vs previous

        # +DM: upward move is significant only when it dominates the down move
        plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        plus_dm_s  = pd.Series(plus_dm,  index=df.index, dtype=float)
        minus_dm_s = pd.Series(minus_dm, index=df.index, dtype=float)

        # True Range (recompute here to keep this method self-contained)
        hl   = high - low
        h_pc = (high - close.shift(1)).abs()
        l_pc = (low  - close.shift(1)).abs()
        tr   = pd.concat([hl, h_pc, l_pc], axis=1).max(axis=1)

        # Wilder's smoothing for TR, +DM, -DM
        tr_smooth    = tr.ewm(alpha=alpha, min_periods=n, adjust=False).mean()
        plus_dm_sm   = plus_dm_s.ewm(alpha=alpha, min_periods=n, adjust=False).mean()
        minus_dm_sm  = minus_dm_s.ewm(alpha=alpha, min_periods=n, adjust=False).mean()

        # Directional Indices (expressed as percentages)
        # Replace zero denominator with NaN to avoid division warnings,
        # then fill with 0 (no movement in either direction).
        plus_di  = 100.0 * plus_dm_sm  / tr_smooth.replace(0, np.nan).fillna(1)
        minus_di = 100.0 * minus_dm_sm / tr_smooth.replace(0, np.nan).fillna(1)

        # DX: spread between +DI and -DI relative to their sum
        di_sum   = (plus_di + minus_di).replace(0, np.nan)
        dx       = 100.0 * (plus_di - minus_di).abs() / di_sum.fillna(1)

        # ADX: Wilder's smooth of DX
        adx = dx.ewm(alpha=alpha, min_periods=n, adjust=False).mean()

        return adx, plus_di, minus_di

    def _compute_ema(self, series: pd.Series, period: int) -> pd.Series:
        """
        Compute a standard Exponential Moving Average.

        Uses pandas ewm(span=period) which corresponds to:
            alpha = 2 / (period + 1)

        This differs slightly from Wilder's alpha (1/period) used in
        ATR/ADX, but matches the EMA formula used in MT5, TradingView,
        and most charting platforms.

        Parameters
        ----------
        series : pd.Series
            Typically df["close"].
        period : int
            EMA period (e.g. 20, 50, 200).

        Returns
        -------
        pd.Series
        """
        return series.ewm(span=period, min_periods=period, adjust=False).mean()

    def _compute_slope_angle(
        self,
        ema_series: pd.Series,
        lookback: int,
    ) -> float:
        """
        Compute the slope of the EMA over the last `lookback` bars and
        convert it to degrees for human-readable interpretation.

        Method
        ------
        A linear regression is fitted to the last `lookback` values of the
        EMA series.  The resulting slope (change in EMA per bar) is divided
        by the mean EMA value to normalise it (otherwise the angle would be
        dramatically different for XAUUSD at $2000 vs $200).

        The normalised slope is then converted to degrees via arctan.

        Returns
        -------
        float
            Angle in degrees.
             Positive → EMA rising (bullish).
             Negative → EMA falling (bearish).
             Near zero → flat (no trend).
        """
        # Need at least `lookback` valid (non-NaN) values
        valid = ema_series.dropna()
        if len(valid) < lookback:
            return 0.0

        # Take the last `lookback` bars
        tail = valid.iloc[-lookback:].values.astype(float)

        # x = bar indices 0, 1, 2, … (lookback−1)
        x = np.arange(len(tail), dtype=float)

        # Linear regression: slope, intercept = polyfit(x, y, degree=1)
        slope, _ = np.polyfit(x, tail, 1)

        # Normalise by the mean EMA value so slope is a fraction per bar
        mean_ema = float(np.mean(tail))
        if mean_ema == 0:
            return 0.0

        normalised_slope = slope / mean_ema

        # Scale to degrees: multiply by a factor so "visually obvious" trends
        # register at ≥ 5°.  Factor of 1000 is empirically good for intraday
        # XAUUSD on H1 (typical normalised slopes are 0.0001 – 0.001).
        angle = math.degrees(math.atan(normalised_slope * 1000))

        return angle

    # =========================================================================
    # CLASSIFICATION HELPERS
    # =========================================================================

    def _determine_direction(
        self,
        plus_di: float,
        minus_di: float,
        ema_slope: float,
        current_close: float,
        ema200: float,
    ) -> tuple[TrendDirection, int]:
        """
        Conduct a 3-vote majority ballot to determine trend direction.

        Returns
        -------
        (TrendDirection, int)
            direction   — BULLISH / BEARISH / NEUTRAL
            votes_for   — number of votes that agreed with the winner (1–3)
                          Used for confidence scoring.
        """
        bullish_votes = 0
        bearish_votes = 0

        # ── Vote A: Directional Index ──────────────────────────────────────
        # +DI > -DI means upward directional movement dominates.
        if plus_di > minus_di:
            bullish_votes += 1
        elif minus_di > plus_di:
            bearish_votes += 1
        # Equal DI → abstain (no vote added to either side)

        # ── Vote B: EMA slope ──────────────────────────────────────────────
        # The EMA slope shows the momentum direction of recent price action.
        # Only count the vote if the slope is steep enough to be meaningful.
        if ema_slope > self.SLOPE_SIGNIFICANCE_ANGLE:
            bullish_votes += 1
        elif ema_slope < -self.SLOPE_SIGNIFICANCE_ANGLE:
            bearish_votes += 1
        # Flat EMA → abstain

        # ── Vote C: Price vs EMA 200 ───────────────────────────────────────
        # The 200-period EMA is the classic macro trend filter.
        # Price above EMA200 → long-term bullish bias.
        if math.isfinite(ema200) and ema200 > 0:
            if current_close > ema200:
                bullish_votes += 1
            elif current_close < ema200:
                bearish_votes += 1

        # ── Majority vote ──────────────────────────────────────────────────
        if bullish_votes > bearish_votes:
            return TrendDirection.BULLISH, bullish_votes
        elif bearish_votes > bullish_votes:
            return TrendDirection.BEARISH, bearish_votes
        else:
            # Tied votes → direction unclear; still mark as TRENDING
            # but with NEUTRAL direction (both signal directions permitted)
            return TrendDirection.NEUTRAL, 0

    def _format_vote_details(
        self,
        plus_di: float,
        minus_di: float,
        ema_slope: float,
        close: float,
        ema200: float,
    ) -> str:
        """Build a human-readable string showing how each vote was cast."""
        di_vote    = "BULL(+DI>-DI)" if plus_di > minus_di else ("BEAR(-DI>+DI)" if minus_di > plus_di else "ABSTAIN(=DI)")
        slope_vote = "BULL" if ema_slope > self.SLOPE_SIGNIFICANCE_ANGLE else ("BEAR" if ema_slope < -self.SLOPE_SIGNIFICANCE_ANGLE else "ABSTAIN(flat)")
        macro_vote = "BULL(>EMA200)" if close > ema200 else "BEAR(<EMA200)"
        return f"DI={di_vote}, slope={slope_vote}, macro={macro_vote}"

    # =========================================================================
    # CONFIDENCE SCORING
    # =========================================================================

    def _score_trending_confidence(
        self,
        adx: float,
        adx_threshold: float,
        votes_for: int,
    ) -> float:
        """
        Compute a confidence score [0.0, 1.0] for a TRENDING classification.

        Score combines:
        - ADX surplus above the threshold (how far above 25 is the ADX?)
        - Directional vote agreement (all 3 agreed → max direction confidence)

        The ADX component is normalised so that:
            ADX = threshold       → 0 % ADX confidence
            ADX = threshold + 20  → 100 % ADX confidence  (cap at 40+)

        The vote component:
            3 / 3 votes → 100 %
            2 / 3 votes →  67 %
            1 / 3 votes →  33 %
            0 / 3 votes →   0 %
        """
        # ADX component: how far above the threshold is the ADX?
        adx_surplus    = max(0.0, adx - adx_threshold)
        adx_component  = min(adx_surplus / 20.0, 1.0)   # saturates at +20 pts

        # Direction component: fraction of votes that agreed
        direction_component = votes_for / 3.0

        # Weighted average: ADX strength 60%, direction agreement 40%
        return round(0.60 * adx_component + 0.40 * direction_component, 3)

    def _score_ranging_confidence(self, adx: float, threshold: float) -> float:
        """
        Compute confidence for a RANGING classification.

        Higher confidence when ADX is well below the threshold.
        ADX at threshold → 0 % confidence.
        ADX at 0          → 100 % confidence (perfectly flat market).
        """
        if threshold <= 0:
            return 0.0
        below = max(0.0, threshold - adx)
        return round(min(below / threshold, 1.0), 3)

    def _score_volatility_confidence(self, atr_ratio: float, high: bool) -> float:
        """
        Compute confidence for HIGH_VOLATILITY or LOW_VOLATILITY.

        high=True  → the more above the HIGH_VOL threshold, the higher the score
        high=False → the more below the LOW_VOL threshold, the higher the score
        """
        if high:
            excess = max(0.0, atr_ratio - self.HIGH_VOL_ATR_RATIO)
            return round(min(excess / self.HIGH_VOL_ATR_RATIO, 1.0), 3)
        else:
            deficit = max(0.0, self.LOW_VOL_ATR_RATIO - atr_ratio)
            return round(min(deficit / self.LOW_VOL_ATR_RATIO, 1.0), 3)

    # =========================================================================
    # REPR
    # =========================================================================

    def __repr__(self) -> str:
        return (
            f"MarketRegimeClassifier("
            f"adx_period={self.adx_period}, "
            f"adx_threshold={self.adx_threshold}, "
            f"atr_period={self.atr_period}, "
            f"ema={self.ema_fast}/{self.ema_slow}/{self.ema_macro}"
            f")"
        )
