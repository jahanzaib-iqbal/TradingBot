"""
strategy/liquidity_detection.py
================================
Liquidity sweep (stop-hunt / liquidity grab) detector for XAUUSD.

What is a liquidity sweep?
--------------------------
Liquidity pools form wherever large numbers of stop-loss orders cluster.
The most common pools are:

  Buy-side liquidity  — stop-losses from short sellers, sitting ABOVE
                        swing highs and equal highs.
  Sell-side liquidity — stop-losses from long buyers, sitting BELOW
                        swing lows and equal lows.

Smart Money (institutional traders) deliberately push price into these
pools to trigger the stops, absorb the resulting orders, and then
reverse.  The reversal after the sweep is the actual high-probability
trade entry.

Detection logic
---------------

Step 1 — Map swing levels
    Find all significant swing highs and swing lows within the lookback
    window.  A "swing high" is a bar whose high is the highest in a
    window of `swing_lookback` bars on each side.  Likewise for lows.

Step 2 — Detect the break
    A sweep has occurred on bar[i] if:
      • Sweep HIGH: bar[i].high > swing_high AND bar[i].close < swing_high
        (price pierced above the level but CLOSED BACK BELOW it)
      • Sweep LOW:  bar[i].low  < swing_low  AND bar[i].close > swing_low
        (price pierced below the level but CLOSED BACK ABOVE it)

    The key requirement — close inside the range — distinguishes a
    liquidity sweep from a genuine breakout.

Step 3 — Quality filters (each adds to the quality score)
    • Wick size: the spike beyond the swing level should be meaningful
      relative to ATR (wick >= MIN_WICK_ATR_RATIO × ATR)
    • Reversion body: the body of the sweep candle should lean back
      against the breakout direction (close < open for a high sweep)
    • Level age: a level built over more bars is more significant
    • Candle volume: high tick volume on the sweep bar is a stronger signal

Step 4 — Immediate reversal confirmation (optional filter)
    If REQUIRE_REVERSAL_CONFIRMATION = True, the bar AFTER the sweep
    must also close in the reversal direction.

Direction convention (SMC)
--------------------------
  Sweep of sell-side liquidity (previous LOW swept)  →  direction = "BUY"
    Rationale: bear stops below lows have been taken; institutional longs
    entered; price is now likely to rise.

  Sweep of buy-side liquidity (previous HIGH swept)  →  direction = "SELL"
    Rationale: bull stops above highs have been taken; institutional
    shorts entered; price is likely to fall.

Public API
----------
    detector = LiquiditySweepDetector(settings)
    sweeps   = detector.detect(df)                    # → list[LiquiditySweep]
    last     = detector.get_latest_sweep(df)          # → LiquiditySweep | None
    d        = detector.detect_as_signal_dicts(df)    # → list[dict]

Dependencies
------------
    pandas, numpy, config.settings.Settings, utils.logger
"""

from __future__ import annotations

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

class SweepType(str, Enum):
    """Which side of the market was swept."""
    HIGH = "HIGH"   # Buy-side liquidity taken -> SELL setup
    LOW  = "LOW"    # Sell-side liquidity taken -> BUY setup


class SweepQuality(str, Enum):
    """Human-readable quality band for a detected sweep."""
    HIGH    = "HIGH"    # score >= 0.70 — textbook sweep, high-probability
    MEDIUM  = "MEDIUM"  # score >= 0.40 — moderate confluence
    LOW     = "LOW"     # score <  0.40 — weak sweep, use with caution


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LiquiditySweep:
    """
    Represents a single detected liquidity sweep event.

    Attributes
    ----------
    type            : HIGH (buy-side taken) or LOW (sell-side taken)
    direction       : "BUY" or "SELL" — entry direction AFTER the sweep
    sweep_price     : price of the swept swing level
    sweep_bar_index : row index in the DataFrame where the sweep occurred
    sweep_bar_time  : timestamp of the sweep bar (if 'time' column exists)
    wick_size       : distance the wick extended beyond the swing level
    wick_atr_ratio  : wick_size / ATR at that bar
    level_age_bars  : how many bars old the swept swing level was
    quality_score   : composite quality score 0.0 – 1.0
    quality         : SweepQuality label
    confirmed       : True if next bar also closes in reversal direction
    notes           : plain-English description
    """
    type:            SweepType
    direction:       str              # "BUY" | "SELL"
    sweep_price:     float
    sweep_bar_index: int
    sweep_bar_time:  Optional[object] = None   # pd.Timestamp or None

    # Metric details
    wick_size:      float = 0.0
    wick_atr_ratio: float = 0.0
    level_age_bars: int   = 0
    quality_score:  float = 0.0
    quality:        SweepQuality = SweepQuality.LOW

    confirmed: bool = False
    notes:     str  = ""

    # ── Output helpers ────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """
        Return a standardised signal dictionary.

        Primary keys — used by SignalGenerator:
            type, direction, sweep_price, quality_score, quality, confirmed

        Example
        -------
            {
                "type":            "LIQUIDITY_SWEEP",
                "direction":       "BUY",
                "sweep_type":      "LOW",
                "sweep_price":     2287.50,
                "sweep_bar_index": 195,
                "wick_size":       4.30,
                "wick_atr_ratio":  0.62,
                "level_age_bars":  12,
                "quality_score":   0.78,
                "quality":         "HIGH",
                "confirmed":       True,
                "notes":           "Sell-side liquidity swept ..."
            }
        """
        return {
            "type":            "LIQUIDITY_SWEEP",
            "direction":       self.direction,
            "sweep_type":      self.type.value,
            "sweep_price":     round(self.sweep_price, 2),
            "sweep_bar_index": self.sweep_bar_index,
            "sweep_bar_time":  str(self.sweep_bar_time) if self.sweep_bar_time is not None else None,
            "wick_size":       round(self.wick_size, 2),
            "wick_atr_ratio":  round(self.wick_atr_ratio, 3),
            "level_age_bars":  self.level_age_bars,
            "quality_score":   round(self.quality_score, 3),
            "quality":         self.quality.value,
            "confirmed":       self.confirmed,
            "notes":           self.notes,
        }

    def __str__(self) -> str:
        conf = " [CONFIRMED]" if self.confirmed else ""
        return (
            f"LiquiditySweep("
            f"type={self.type.value}, "
            f"dir={self.direction}, "
            f"swept_level={self.sweep_price:.2f}, "
            f"wick={self.wick_size:.2f}pts({self.wick_atr_ratio:.2f}xATR), "
            f"age={self.level_age_bars}bars, "
            f"quality={self.quality.value}({self.quality_score:.0%})"
            f"{conf}"
            f")"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Detector
# ─────────────────────────────────────────────────────────────────────────────

class LiquiditySweepDetector:
    """
    Detects liquidity sweep (stop-hunt) events in OHLCV data.

    Detection requires columns: open, high, low, close.
    Optional column: tick_volume (improves quality scoring).

    Typical usage
    -------------
        detector = LiquiditySweepDetector(settings)
        df       = provider.get_candles("XAUUSD", "H1", 200)
        sweeps   = detector.detect(df)

        for s in sweeps:
            print(s)                 # human-readable
            print(s.to_dict())       # signal dict for downstream consumers

        latest = detector.get_latest_sweep(df)
        if latest and latest.quality == SweepQuality.HIGH:
            # Pass to signal generator
            ...
    """

    # ── Default thresholds ────────────────────────────────────────────────
    # Override at construction time or via Settings.

    # Bars on each side required to confirm a swing high/low.
    # swing_lookback=3 means the bar's high must be the max in a
    # window of 3 bars before + 3 bars after it.
    DEFAULT_SWING_LOOKBACK: int = 3

    # Maximum age (bars) of a swing level to still be considered "active".
    DEFAULT_MAX_LEVEL_AGE: int = 50

    # Minimum wick extension beyond the swept level, as a fraction of ATR.
    # wick >= MIN_WICK_ATR_RATIO × ATR
    DEFAULT_MIN_WICK_ATR_RATIO: float = 0.25

    # ATR period for volatility normalisation.
    DEFAULT_ATR_PERIOD: int = 14

    # Require confirmation: next bar must close in the reversal direction.
    DEFAULT_REQUIRE_CONFIRMATION: bool = False

    # Maximum number of recent sweeps that detect() will return.
    # Keeps the output list manageable (last N most recent sweeps only).
    MAX_RESULTS: int = 10

    def __init__(self, settings: Optional[Settings] = None) -> None:
        """
        Parameters
        ----------
        settings : Settings, optional
            If provided, ATR_PERIOD and OB_LOOKBACK_BARS (used as a proxy for
            max level age) are read from the settings object.
            All other thresholds use class-level defaults.
        """
        if settings is not None:
            self.atr_period       = settings.ATR_PERIOD
            self.max_level_age    = settings.OB_LOOKBACK_BARS
            self.swing_lookback   = self.DEFAULT_SWING_LOOKBACK
            self.min_wick_ratio   = self.DEFAULT_MIN_WICK_ATR_RATIO
            self.require_confirm  = self.DEFAULT_REQUIRE_CONFIRMATION
        else:
            self.atr_period       = self.DEFAULT_ATR_PERIOD
            self.max_level_age    = self.DEFAULT_MAX_LEVEL_AGE
            self.swing_lookback   = self.DEFAULT_SWING_LOOKBACK
            self.min_wick_ratio   = self.DEFAULT_MIN_WICK_ATR_RATIO
            self.require_confirm  = self.DEFAULT_REQUIRE_CONFIRMATION

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def detect(self, df: pd.DataFrame) -> list[LiquiditySweep]:
        """
        Run the full sweep detection pipeline on an OHLCV DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Columns required: open, high, low, close.
            Rows sorted oldest → newest (standard get_candles() output).

        Returns
        -------
        list[LiquiditySweep]
            All sweeps detected in the DataFrame, sorted newest → oldest.
            Limited to the last MAX_RESULTS entries.
        """
        required = {"open", "high", "low", "close"}
        missing  = required - set(df.columns)
        if missing:
            logger.error(f"detect() — missing columns: {missing}")
            return []

        # Need at least 2×swing_lookback + ATR_period bars
        min_bars = 2 * self.swing_lookback + self.atr_period + 2
        if len(df) < min_bars:
            logger.warning(
                f"detect() — only {len(df)} bars; need {min_bars}. "
                "Cannot detect swings. Returning empty list."
            )
            return []

        df = df.copy().reset_index(drop=True)

        # ── Pre-compute ATR series ─────────────────────────────────────────
        atr = self._compute_atr(df)

        # ── Find all swing highs and lows ─────────────────────────────────
        swing_highs = self._find_swing_highs(df)  # {bar_index: high_price}
        swing_lows  = self._find_swing_lows(df)   # {bar_index: low_price}

        # ── Scan every bar for a sweep event ─────────────────────────────
        sweeps: list[LiquiditySweep] = []

        # Start scanning from `swing_lookback`+1 to ensure swing context exists
        start = self.swing_lookback + 1

        for i in range(start, len(df)):
            bar_atr     = float(atr.iloc[i]) if not np.isnan(atr.iloc[i]) else 0.0
            if bar_atr == 0.0:
                continue  # skip bars where ATR hasn't warmed up yet

            # Check for a HIGH sweep on this bar
            high_sweep = self._check_high_sweep(df, i, swing_highs, bar_atr)
            if high_sweep is not None:
                # Attempt reversal confirmation (next bar)
                high_sweep.confirmed = self._check_confirmation(df, i, sweep_type=SweepType.HIGH)
                if not self.require_confirm or high_sweep.confirmed:
                    sweeps.append(high_sweep)

            # Check for a LOW sweep on this bar
            low_sweep = self._check_low_sweep(df, i, swing_lows, bar_atr)
            if low_sweep is not None:
                low_sweep.confirmed = self._check_confirmation(df, i, sweep_type=SweepType.LOW)
                if not self.require_confirm or low_sweep.confirmed:
                    sweeps.append(low_sweep)

        # Sort newest → oldest, return last MAX_RESULTS
        sweeps.sort(key=lambda s: s.sweep_bar_index, reverse=True)
        result = sweeps[:self.MAX_RESULTS]

        logger.info(
            f"detect() — {len(result)} sweep(s) found in {len(df)} bars  "
            f"({len(swing_highs)} swing highs, {len(swing_lows)} swing lows mapped)"
        )
        for s in result:
            logger.debug(f"  {s}")

        return result

    def detect_as_signal_dicts(self, df: pd.DataFrame) -> list[dict]:
        """
        Convenience wrapper — return all detected sweeps as plain dicts.

        Each dict has a "type": "LIQUIDITY_SWEEP" field so the downstream
        signal generator can route it correctly.
        """
        return [s.to_dict() for s in self.detect(df)]

    def get_latest_sweep(self, df: pd.DataFrame) -> Optional[LiquiditySweep]:
        """
        Return the most recently detected sweep, or None if none found.

        This is the primary method used by SignalGenerator on each cycle:
        if the most recent bar has a valid sweep the strategy can act on it.
        """
        sweeps = self.detect(df)
        return sweeps[0] if sweeps else None

    def get_latest_high_quality_sweep(
        self, df: pd.DataFrame, min_quality: SweepQuality = SweepQuality.MEDIUM
    ) -> Optional[LiquiditySweep]:
        """
        Return the most recent sweep that meets the minimum quality threshold.

        Parameters
        ----------
        min_quality : SweepQuality
            Only sweeps at this quality or better are returned.
            MEDIUM (default) filters out weak/noise sweeps.

        Returns
        -------
        LiquiditySweep | None
        """
        quality_rank = {SweepQuality.LOW: 0, SweepQuality.MEDIUM: 1, SweepQuality.HIGH: 2}
        min_rank     = quality_rank[min_quality]

        for sweep in self.detect(df):
            if quality_rank[sweep.quality] >= min_rank:
                return sweep
        return None

    # =========================================================================
    # SWING LEVEL DETECTION
    # =========================================================================

    def _find_swing_highs(self, df: pd.DataFrame) -> dict[int, float]:
        """
        Identify swing high bars within the DataFrame.

        A bar at index `i` is a swing high if its `high` is the maximum
        in the window [i - swing_lookback, i + swing_lookback].

        Returns
        -------
        dict[int, float]
            Mapping of {bar_index: swing_high_price}
        """
        swing_highs: dict[int, float] = {}
        n = len(df)
        lb = self.swing_lookback

        for i in range(lb, n - lb):
            window_start = i - lb
            window_end   = i + lb + 1                    # slice end is exclusive
            window_high  = float(df["high"].iloc[window_start:window_end].max())
            bar_high     = float(df["high"].iloc[i])

            if bar_high == window_high:
                swing_highs[i] = bar_high

        return swing_highs

    def _find_swing_lows(self, df: pd.DataFrame) -> dict[int, float]:
        """
        Identify swing low bars within the DataFrame.

        A bar at index `i` is a swing low if its `low` is the minimum
        in the window [i - swing_lookback, i + swing_lookback].

        Returns
        -------
        dict[int, float]
            Mapping of {bar_index: swing_low_price}
        """
        swing_lows: dict[int, float] = {}
        n = len(df)
        lb = self.swing_lookback

        for i in range(lb, n - lb):
            window_start = i - lb
            window_end   = i + lb + 1
            window_low   = float(df["low"].iloc[window_start:window_end].min())
            bar_low      = float(df["low"].iloc[i])

            if bar_low == window_low:
                swing_lows[i] = bar_low

        return swing_lows

    # =========================================================================
    # SWEEP CHECK METHODS
    # =========================================================================

    def _check_high_sweep(
        self,
        df: pd.DataFrame,
        i: int,
        swing_highs: dict[int, float],
        bar_atr: float,
    ) -> Optional[LiquiditySweep]:
        """
        Check whether bar[i] sweeps any previous swing high.

        Sweep HIGH requirements
        -----------------------
        1. bar[i].high > swing_high_price         (price pierced above the level)
        2. bar[i].close < swing_high_price         (body closed BACK below the level)
        3. wick_size >= min_wick_ratio × ATR       (wick is meaningful, not noise)
        4. swing level is within max_level_age bars (still "active" liquidity)

        If all pass → builds and returns a LiquiditySweep with direction "SELL".

        Returns
        -------
        LiquiditySweep | None
        """
        bar_high  = float(df["high"].iloc[i])
        bar_close = float(df["close"].iloc[i])

        # Inspect every known swing high that formed BEFORE this bar
        best_sweep: Optional[LiquiditySweep] = None
        best_score = -1.0

        for level_idx, level_price in swing_highs.items():

            if level_idx >= i:
                continue   # level must be in the past

            age = i - level_idx

            if age > self.max_level_age:
                continue   # level is too old — liquidity may have been invalidated

            # ── Condition 1: High must BREACH the swing high ─────────────
            if bar_high <= level_price:
                continue   # no breach

            # ── Condition 2: Close must RETURN below the swing high ───────
            # (This is the "sweep and reverse" signature)
            if bar_close >= level_price:
                continue   # price broke above and STAYED — genuine breakout, not a sweep

            # ── Condition 3: Wick size check ──────────────────────────────
            # wick = distance the HIGH extended beyond the swing level
            wick_size      = bar_high - level_price
            wick_atr_ratio = wick_size / bar_atr if bar_atr else 0.0

            if wick_atr_ratio < self.min_wick_ratio:
                continue   # wick too small — just noise

            # ── Score this sweep event ─────────────────────────────────────
            score = self._score_sweep(
                wick_atr_ratio = wick_atr_ratio,
                level_age      = age,
                bar_open       = float(df["open"].iloc[i]),
                bar_close      = bar_close,
                bar_high       = bar_high,
                bar_low        = float(df["low"].iloc[i]),
                sweep_type     = SweepType.HIGH,
                tick_volume    = float(df["tick_volume"].iloc[i]) if "tick_volume" in df.columns else None,
                df             = df,
                bar_index      = i,
            )

            if score > best_score:
                best_score = score
                quality    = self._score_to_quality(score)

                # Retrieve timestamp if available
                sweep_time = df["time"].iloc[i] if "time" in df.columns else None

                notes = (
                    f"Buy-side liquidity swept above {level_price:.2f} "
                    f"(swing high from {age} bars ago). "
                    f"Bar high={bar_high:.2f} pierced, then closed at {bar_close:.2f}. "
                    f"Wick={wick_size:.2f}pts ({wick_atr_ratio:.2f}x ATR). "
                    f"Expect bearish reversal -> SELL setup."
                )

                best_sweep = LiquiditySweep(
                    type            = SweepType.HIGH,
                    direction       = "SELL",          # sweep of highs → SELL
                    sweep_price     = level_price,
                    sweep_bar_index = i,
                    sweep_bar_time  = sweep_time,
                    wick_size       = round(wick_size, 3),
                    wick_atr_ratio  = round(wick_atr_ratio, 3),
                    level_age_bars  = age,
                    quality_score   = round(score, 3),
                    quality         = quality,
                    notes           = notes,
                )

        return best_sweep

    def _check_low_sweep(
        self,
        df: pd.DataFrame,
        i: int,
        swing_lows: dict[int, float],
        bar_atr: float,
    ) -> Optional[LiquiditySweep]:
        """
        Check whether bar[i] sweeps any previous swing low.

        Sweep LOW requirements
        ----------------------
        1. bar[i].low  < swing_low_price           (price pierced below the level)
        2. bar[i].close > swing_low_price           (body closed BACK above the level)
        3. wick_size >= min_wick_ratio × ATR        (wick is meaningful)
        4. swing level is within max_level_age bars

        If all pass → builds and returns a LiquiditySweep with direction "BUY".

        Returns
        -------
        LiquiditySweep | None
        """
        bar_low   = float(df["low"].iloc[i])
        bar_close = float(df["close"].iloc[i])

        best_sweep: Optional[LiquiditySweep] = None
        best_score = -1.0

        for level_idx, level_price in swing_lows.items():

            if level_idx >= i:
                continue

            age = i - level_idx

            if age > self.max_level_age:
                continue

            # ── Condition 1: Low must BREACH the swing low ────────────────
            if bar_low >= level_price:
                continue

            # ── Condition 2: Close must RETURN above the swing low ────────
            if bar_close <= level_price:
                continue   # stayed below — genuine breakdown, not a sweep

            # ── Condition 3: Wick size ────────────────────────────────────
            wick_size      = level_price - bar_low
            wick_atr_ratio = wick_size / bar_atr if bar_atr else 0.0

            if wick_atr_ratio < self.min_wick_ratio:
                continue

            # ── Score ─────────────────────────────────────────────────────
            score = self._score_sweep(
                wick_atr_ratio = wick_atr_ratio,
                level_age      = age,
                bar_open       = float(df["open"].iloc[i]),
                bar_close      = bar_close,
                bar_high       = float(df["high"].iloc[i]),
                bar_low        = bar_low,
                sweep_type     = SweepType.LOW,
                tick_volume    = float(df["tick_volume"].iloc[i]) if "tick_volume" in df.columns else None,
                df             = df,
                bar_index      = i,
            )

            if score > best_score:
                best_score = score
                quality    = self._score_to_quality(score)

                sweep_time = df["time"].iloc[i] if "time" in df.columns else None

                notes = (
                    f"Sell-side liquidity swept below {level_price:.2f} "
                    f"(swing low from {age} bars ago). "
                    f"Bar low={bar_low:.2f} pierced, then closed at {bar_close:.2f}. "
                    f"Wick={wick_size:.2f}pts ({wick_atr_ratio:.2f}x ATR). "
                    f"Expect bullish reversal -> BUY setup."
                )

                best_sweep = LiquiditySweep(
                    type            = SweepType.LOW,
                    direction       = "BUY",           # sweep of lows → BUY
                    sweep_price     = level_price,
                    sweep_bar_index = i,
                    sweep_bar_time  = sweep_time,
                    wick_size       = round(wick_size, 3),
                    wick_atr_ratio  = round(wick_atr_ratio, 3),
                    level_age_bars  = age,
                    quality_score   = round(score, 3),
                    quality         = quality,
                    notes           = notes,
                )

        return best_sweep

    # =========================================================================
    # REVERSAL CONFIRMATION
    # =========================================================================

    def _check_confirmation(
        self,
        df: pd.DataFrame,
        sweep_bar_index: int,
        sweep_type: SweepType,
    ) -> bool:
        """
        Check whether the bar IMMEDIATELY AFTER the sweep also closes
        in the reversal direction.

        Confirmation bar rules
        ----------------------
        After a HIGH sweep (SELL): next bar must close < sweep bar's close
        After a LOW  sweep (BUY):  next bar must close > sweep bar's close

        Returns
        -------
        bool — True if confirmed, False if next bar doesn't exist yet
               or fails the direction test.
        """
        next_idx = sweep_bar_index + 1
        if next_idx >= len(df):
            return False   # sweep is on the last (forming) bar — cannot confirm

        next_close  = float(df["close"].iloc[next_idx])
        sweep_close = float(df["close"].iloc[sweep_bar_index])

        if sweep_type == SweepType.HIGH:
            # After a high sweep → expect next close < sweep close (bearish)
            return next_close < sweep_close
        else:
            # After a low sweep → expect next close > sweep close (bullish)
            return next_close > sweep_close

    # =========================================================================
    # QUALITY SCORING
    # =========================================================================

    def _score_sweep(
        self,
        wick_atr_ratio: float,
        level_age:      int,
        bar_open:       float,
        bar_close:      float,
        bar_high:       float,
        bar_low:        float,
        sweep_type:     SweepType,
        tick_volume:    Optional[float],
        df:             pd.DataFrame,
        bar_index:      int,
    ) -> float:
        """
        Compute a composite quality score [0.0, 1.0] for a detected sweep.

        Sub-scores
        ----------
        A — Wick size (40%):
            wick >= 1.0 × ATR  →  score A = 1.0
            wick == 0.25 × ATR →  score A = 0.0   (just above minimum)
            Linearly interpolated.

        B — Reversal body (30%):
            The sweep candle's body should close AGAINST the sweep direction.
            For HIGH sweep: a bearish body (close < open) → full score.
            For LOW  sweep: a bullish body (close > open) → full score.
            Neutral candle (doji) → partial score.

        C — Level recency / significance (20%):
            Newer levels are more significant (active liquidity pool).
            age = 1 bar  → score C = 1.0
            age = max_level_age → score C = 0.0

        D — Volume surge (10%):
            High tick volume on the sweep bar = more institutional activity.
            If no volume data, D = 0.5 (neutral).
        """
        # ── Sub-score A: Wick size ─────────────────────────────────────────
        # Normalise wick against [MIN_WICK_RATIO, STRONG_WICK_RATIO = 1.0 ATR]
        STRONG_WICK = 1.0  # 1× ATR = "strong" wick
        score_a = min((wick_atr_ratio - self.min_wick_ratio) / (STRONG_WICK - self.min_wick_ratio), 1.0)
        score_a = max(score_a, 0.0)

        # ── Sub-score B: Reversal body ─────────────────────────────────────
        candle_range = bar_high - bar_low
        body_size    = abs(bar_close - bar_open)
        body_ratio   = body_size / candle_range if candle_range > 0 else 0.0

        if sweep_type == SweepType.HIGH:
            # Want bearish candle (close < open)
            is_reversal_body = bar_close < bar_open
        else:
            # Want bullish candle (close > open)
            is_reversal_body = bar_close > bar_open

        # Score: reversal body direction × body ratio (big body = high conviction)
        score_b = body_ratio if is_reversal_body else body_ratio * 0.2

        # ── Sub-score C: Level age ─────────────────────────────────────────
        # Recent levels = higher liquidity relevance
        # Exponential decay: score = (1 - age/max_age)^2
        age_fraction = level_age / max(self.max_level_age, 1)
        score_c      = max(1.0 - age_fraction, 0.0) ** 1.5   # slight decay curve

        # ── Sub-score D: Volume ────────────────────────────────────────────
        if tick_volume is not None and tick_volume > 0:
            # Compare this bar's volume to a rolling mean of recent bars
            lookback_vol = df["tick_volume"].iloc[max(0, bar_index - 20): bar_index]
            if len(lookback_vol) > 0:
                mean_vol = float(lookback_vol.mean())
                vol_ratio = tick_volume / mean_vol if mean_vol > 0 else 1.0
                # vol ≥ 2× average → score D = 1.0; vol = 1× → 0.5; vol < 1× → 0.0
                score_d = min((vol_ratio - 1.0) / 1.0, 1.0)
                score_d = max(score_d, 0.0)
            else:
                score_d = 0.5
        else:
            score_d = 0.5   # no volume data — neutral

        # ── Weighted composite ─────────────────────────────────────────────
        composite = (
            0.40 * score_a +
            0.30 * score_b +
            0.20 * score_c +
            0.10 * score_d
        )
        return min(max(composite, 0.0), 1.0)

    @staticmethod
    def _score_to_quality(score: float) -> SweepQuality:
        """Map a 0–1 score to a SweepQuality label."""
        if score >= 0.70:
            return SweepQuality.HIGH
        if score >= 0.40:
            return SweepQuality.MEDIUM
        return SweepQuality.LOW

    # =========================================================================
    # ATR CALCULATION
    # =========================================================================

    def _compute_atr(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute Wilder's ATR for the DataFrame.

        Uses the same Wilder's smoothing (alpha = 1/period) as the rest
        of the codebase for consistency.
        """
        high  = df["high"]
        low   = df["low"]
        close = df["close"]

        hl   = high - low
        h_pc = (high - close.shift(1)).abs()
        l_pc = (low  - close.shift(1)).abs()
        tr   = pd.concat([hl, h_pc, l_pc], axis=1).max(axis=1)

        return tr.ewm(
            alpha=1.0 / self.atr_period,
            min_periods=self.atr_period,
            adjust=False,
        ).mean()

    # =========================================================================
    # REPR
    # =========================================================================

    def __repr__(self) -> str:
        return (
            f"LiquiditySweepDetector("
            f"swing_lookback={self.swing_lookback}, "
            f"max_level_age={self.max_level_age}, "
            f"min_wick_ratio={self.min_wick_ratio}, "
            f"atr_period={self.atr_period}, "
            f"require_confirm={self.require_confirm}"
            f")"
        )
