"""
strategy/smart_money_strategy.py
==================================
Smart Money Concepts (SMC) strategy engine for XAUUSD intraday trading.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SMART MONEY CONCEPTS — EDUCATIONAL OVERVIEW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"Smart Money" refers to institutional participants (central banks, hedge
funds, market makers) whose order flow MOVES markets.  Retail traders
must identify WHERE institutions placed their orders and trade with them.

Core premise: institutions cannot fill large orders in one go.  They
enter in zones, create liquidity, manipulate price to collect stops, then
drive price to profit targets.  SMC maps this footprint.

COMPONENT OVERVIEW
------------------
1. Market Structure (MS)
   ├── Higher High (HH)  — uptrend: each swing high exceeds the last
   ├── Higher Low  (HL)  — uptrend: each swing low exceeds the last
   ├── Lower High  (LH)  — downtrend: each swing high is lower
   └── Lower Low   (LL)  — downtrend: each swing low is lower

2. Break of Structure (BOS)
   Continuation signal. Price closes beyond the most recent confirmed
   swing high (bullish BOS) or swing low (bearish BOS) in the same
   direction as the prevailing trend.

3. Change of Character (CHoCH)
   Reversal signal. Price closes beyond a swing point in the OPPOSITE
   direction to the prevailing trend.  First sign that institutions are
   switching sides.

4. Fair Value Gap (FVG) / Imbalance
   A 3-candle pattern.  If the gap between candle[i-2].high and
   candle[i].low is not zero, there is a BULLISH FVG (price skipped
   upward).  Price often returns to fill these gaps.

   Pattern:
       C[-2]    C[-1]       C[0]
       ┌───┐   ╔═══╗    ┌───┐
       │   │   ║   ║    │   │
       └───┘   ║   ║    └───┘
              FVG zone
       ( C[-2].high  <  C[0].low  →  gap exists )

5. Order Blocks (OB)
   The last candle of the OPPOSITE colour before a strong impulsive move.
   Represents the last area where institutions were actively placing orders
   before the move.

   Bullish OB: last BEARISH candle before a strong bullish impulse.
   Bearish OB: last BULLISH candle before a strong bearish impulse.

   Price almost always returns to the OB zone to rebalance before
   continuing in the impulse direction.

6. Liquidity Pools
   Stop orders cluster at obvious levels: swing highs/lows, equal highs
   (double tops), equal lows (double bottoms), and previous session extremes.
   Institutions deliberately hunt these stops before reversing.

7. Entry Signal Assembly
   A valid SMC setup requires CONFLUENCE — multiple SMC factors aligning:
   ┌─────────────────────────────────────────────────────────────────┐
   │  BUY SETUP                    SELL SETUP                        │
   │  ─────────────────────────    ──────────────────────────────    │
   │  1. Bullish market structure  1. Bearish market structure        │
   │  2. Swept previous low         2. Swept previous high            │
   │  3. Price enters bullish OB    3. Price enters bearish OB        │
   │     or bullish FVG                or bearish FVG                 │
   │  4. BOS or CHoCH confirms      4. BOS or CHoCH confirms          │
   │  5. Confirmation candle         5. Confirmation candle           │
   │  6. Target = prev liquidity    6. Target = prev liquidity        │
   └─────────────────────────────────────────────────────────────────┘

Public API
----------
    smc    = SmartMoneyStrategy(settings)
    setups = smc.scan(df)                # -> list[TradeIdea]
    best   = smc.get_best_setup(df)      # -> TradeIdea | None

    # Or step through individual detectors:
    ms     = smc.detect_market_structure(df)
    bos    = smc.detect_bos_choch(df)
    fvgs   = smc.detect_fvg(df)
    obs    = smc.detect_order_blocks(df)
    pools  = smc.detect_liquidity_pools(df)

Optimised for timeframes
------------------------
    M5   — scalp entries / trigger bars
    M15  — setup & confirmation timeframe (primary)
    H1   — structural context
    H4   — macro trend (used by MarketRegimeClassifier)

Dependencies
------------
    pandas, numpy, config.settings.Settings, utils.logger
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

class StructureLabel(str, Enum):
    """Market structure point classification (from swing sequence)."""
    HH = "HH"  # Higher High
    HL = "HL"  # Higher Low
    LH = "LH"  # Lower High
    LL = "LL"  # Lower Low
    EH = "EH"  # Equal High (potential equal-high liquidity pool)
    EL = "EL"  # Equal Low  (potential equal-low liquidity pool)


class BosType(str, Enum):
    """Break of Structure or Change of Character classification."""
    BULLISH_BOS   = "BULLISH_BOS"    # continuation — broke above prev SH
    BEARISH_BOS   = "BEARISH_BOS"    # continuation — broke below prev SL
    BULLISH_CHOCH = "BULLISH_CHOCH"  # reversal — broke above SH during downtrend
    BEARISH_CHOCH = "BEARISH_CHOCH"  # reversal — broke below SL during uptrend


class OBType(str, Enum):
    """Order block directional type."""
    BULLISH = "BULLISH"  # last bearish candle before bullish impulse
    BEARISH = "BEARISH"  # last bullish candle before bearish impulse


class FVGType(str, Enum):
    """Fair Value Gap directional type."""
    BULLISH = "BULLISH"  # gap between C[-2].high and C[0].low (price shot up)
    BEARISH = "BEARISH"  # gap between C[0].high and C[-2].low (price shot down)


# ─────────────────────────────────────────────────────────────────────────────
# Component dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SwingPoint:
    """A confirmed swing high or low in the price series."""
    index:     int
    price:     float
    is_high:   bool     # True = swing high, False = swing low
    label:     Optional[StructureLabel] = None   # set after context comparison
    time:      Optional[object] = None


@dataclass
class StructureEvent:
    """BOS or CHoCH event — a structural break beyond a previous swing."""
    bar_index:     int
    bos_type:      BosType
    broken_level:  float    # the swing price that was broken
    close_price:   float    # close that triggered the break
    displacement:  float    # how far beyond the level the close went
    time:          Optional[object] = None

    @property
    def is_bos(self) -> bool:
        return "BOS" in self.bos_type.value

    @property
    def is_choch(self) -> bool:
        return "CHOCH" in self.bos_type.value

    @property
    def is_bullish(self) -> bool:
        return "BULLISH" in self.bos_type.value


@dataclass
class FairValueGap:
    """
    A Fair Value Gap (FVG) — three-candle imbalance zone.

    Zone boundaries:
        BULLISH FVG: [gap_low, gap_high] = [candle[-2].high, candle[0].low]
        BEARISH FVG: [gap_low, gap_high] = [candle[0].high, candle[-2].low]
    """
    bar_index: int           # index of the THIRD (trigger) candle
    fvg_type:  FVGType
    gap_high:  float         # top of the unfilled gap
    gap_low:   float         # bottom of the unfilled gap
    gap_size:  float         # gap_high - gap_low
    filled:    bool = False  # True once price trades back into the zone
    time:      Optional[object] = None

    @property
    def midpoint(self) -> float:
        return (self.gap_high + self.gap_low) / 2.0

    def contains_price(self, price: float) -> bool:
        """Return True if price is within the FVG zone."""
        return self.gap_low <= price <= self.gap_high


@dataclass
class OrderBlock:
    """
    An Order Block — the last opposing candle before an impulsive move.

    Zone boundaries:
        BULLISH OB: [ob_low, ob_high] = [bearish candle low, bearish candle high]
        BEARISH OB: [ob_low, ob_high] = [bullish candle low, bullish candle high]

    The OB is invalidated if price closes BEYOND the opposite end of the zone.
    """
    bar_index:     int
    ob_type:       OBType
    ob_high:       float     # top of the OB zone
    ob_low:        float     # bottom of the OB zone
    ob_open:       float     # open of the OB candle
    ob_close:      float     # close of the OB candle
    impulse_size:  float     # size of the subsequent impulse in price
    valid:         bool = True   # False if OB is invalidated
    time:          Optional[object] = None

    @property
    def midpoint(self) -> float:
        return (self.ob_high + self.ob_low) / 2.0

    def contains_price(self, price: float) -> bool:
        return self.ob_low <= price <= self.ob_high

    def is_invalidated_by(self, price: float) -> bool:
        """
        An OB is invalidated when price closes through the
        opposite end — the institutional order pool is consumed.
        BULLISH OB invalidated if close < ob_low.
        BEARISH OB invalidated if close > ob_high.
        """
        if self.ob_type == OBType.BULLISH:
            return price < self.ob_low
        return price > self.ob_high


@dataclass
class LiquidityPool:
    """
    A price level where stop-loss orders are likely clustered.

    Types:
        HIGH  — stop-losses of short sellers above swing highs
        LOW   — stop-losses of long buyers below swing lows
        EQUAL_HIGH — two or more swing highs at approximately the same level
        EQUAL_LOW  — two or more swing lows at approximately the same level
    """
    price:        float
    pool_type:    str      # "HIGH" | "LOW" | "EQUAL_HIGH" | "EQUAL_LOW"
    bar_index:    int      # most recent bar this level was formed
    strength:     float    # 0.0–1.0; higher = more times price tested this level
    swept:        bool = False   # True once price trades beyond this level
    time:         Optional[object] = None


@dataclass
class TradeIdea:
    """
    A fully assembled SMC trade setup ready for the SignalGenerator.

    This is the PRIMARY OUTPUT of SmartMoneyStrategy.
    """
    direction:   str     # "BUY" | "SELL"
    entry:       float   # suggested entry price (OB / FVG midpoint or 50%)
    stop_loss:   float   # below OB low (BUY) or above OB high (SELL)
    take_profit: float   # primary target: swept liquidity or previous swing

    # Optional secondary target
    take_profit_2: Optional[float] = None

    # Confidence (0.0–1.0) = weighted confluence score
    confidence: float = 0.0

    # SMC evidence items contributing to this setup
    bos_event:    Optional[StructureEvent]  = None
    order_block:  Optional[OrderBlock]      = None
    fvg:          Optional[FairValueGap]    = None
    liquidity_pool: Optional[LiquidityPool] = None

    # Context
    bar_index:    int   = 0
    time:         Optional[object] = None
    notes:        str   = ""

    # Convenience flags checked by SignalGenerator
    has_ob:   bool = False
    has_fvg:  bool = False
    has_bos:  bool = False
    has_choch: bool = False
    has_liquidity_sweep: bool = False

    def to_dict(self) -> dict:
        """Serialise for Telegram / logging output."""
        return {
            "direction":   self.direction,
            "entry":       round(self.entry, 2),
            "stop_loss":   round(self.stop_loss, 2),
            "take_profit": round(self.take_profit, 2),
            "take_profit_2": round(self.take_profit_2, 2) if self.take_profit_2 else None,
            "confidence":  round(self.confidence, 3),
            "has_ob":      self.has_ob,
            "has_fvg":     self.has_fvg,
            "has_bos":     self.has_bos,
            "has_choch":   self.has_choch,
            "has_liquidity_sweep": self.has_liquidity_sweep,
            "notes":       self.notes,
        }

    def __str__(self) -> str:
        return (
            f"TradeIdea({self.direction} | "
            f"E={self.entry:.2f} SL={self.stop_loss:.2f} TP={self.take_profit:.2f} | "
            f"conf={self.confidence:.0%} | "
            f"OB={'Y' if self.has_ob else 'N'} "
            f"FVG={'Y' if self.has_fvg else 'N'} "
            f"BOS={'Y' if self.has_bos else 'N'} "
            f"CHoCH={'Y' if self.has_choch else 'N'} "
            f"LQsweep={'Y' if self.has_liquidity_sweep else 'N'})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main Strategy Engine
# ─────────────────────────────────────────────────────────────────────────────

class SmartMoneyStrategy:
    """
    SMC-based confluence strategy engine for XAUUSD.

    Each scan() call runs the full detection pipeline on ONE timeframe
    DataFrame (M5 or M15) and returns a list of ranked TradeIdea objects.

    Pipeline
    --------
    1. detect_market_structure()   → classify swing sequence HH/HL/LH/LL
    2. detect_bos_choch()          → identify structural breaks & reversals
    3. detect_order_blocks()       → locate institutional order zones
    4. detect_fvg()                → map imbalance / gap zones
    5. detect_liquidity_pools()    → mark stop-hunt target levels
    6. _assemble_setups()          → combine components into TradeIdea objects
    7. _score_confluence()         → rank setups by confidence score
    """

    # ── Detection parameters (can be overridden from Settings) ────────────

    # Number of bars on each side required to confirm a swing point.
    SWING_LOOKBACK: int = 3

    # Minimum impulse size (as multiple of ATR) for an OB to be valid.
    # The move away from the OB must be "impulsive" — not just a normal candle.
    OB_MIN_IMPULSE_ATR: float = 1.5

    # Minimum OB body-to-range ratio.  A large body = stronger institutional conviction.
    OB_MIN_BODY_RATIO: float = 0.4

    # Bars to look back for OB / FVG detection.
    OB_LOOKBACK: int = 50
    FVG_LOOKBACK: int = 30

    # Minimum FVG size as fraction of ATR (filters micro-gaps from noise).
    FVG_MIN_ATR_FRACTION: float = 0.3

    # Equal-high / equal-low tolerance in price units (for Gold: 0.5 = 50 cents).
    EQUAL_LEVEL_TOLERANCE: float = 0.5

    # Maximum OB lookback age for it to still be considered valid.
    OB_MAX_AGE_BARS: int = 50

    # Minimum confidence score for a setup to be returned by scan().
    MIN_CONFIDENCE: float = 0.35

    # ATR period for normalisation inside detectors.
    ATR_PERIOD: int = 14

    def __init__(self, settings: Optional[Settings] = None) -> None:
        if settings is not None:
            self.SWING_LOOKBACK        = 3
            self.OB_MIN_IMPULSE_ATR    = 1.5
            self.OB_MIN_BODY_RATIO     = settings.OB_MIN_BODY_RATIO
            self.OB_LOOKBACK           = settings.OB_LOOKBACK_BARS
            self.FVG_LOOKBACK          = settings.FVG_LOOKBACK_BARS
            self.FVG_MIN_ATR_FRACTION  = settings.FVG_MIN_ATR_FRACTION
            self.OB_MAX_AGE_BARS       = settings.OB_LOOKBACK_BARS
            self.MIN_CONFIDENCE        = settings.CONFIDENCE_THRESHOLD
            self.ATR_PERIOD            = settings.ATR_PERIOD

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def scan(self, df: pd.DataFrame) -> list[TradeIdea]:
        """
        Run the full SMC analysis pipeline on a single-timeframe OHLCV DataFrame.

        Designed for M5 and M15 data; at least 100 bars required for meaningful
        context (200+ recommended for full SMC structure mapping).

        Parameters
        ----------
        df : pd.DataFrame
            Columns required: open, high, low, close.
            Optional: tick_volume, time.

        Returns
        -------
        list[TradeIdea]
            All valid setups sorted by confidence (highest first).
            Returns empty list if no setup meets MIN_CONFIDENCE.
        """
        req = {"open", "high", "low", "close"}
        if not req.issubset(df.columns):
            logger.error(f"scan() — missing columns: {req - set(df.columns)}")
            return []

        if len(df) < 50:
            logger.warning(f"scan() — only {len(df)} bars; need ≥ 50.")
            return []

        df = df.copy().reset_index(drop=True)

        # ── Pre-compute ATR for the whole series ──────────────────────────
        atr = self._atr(df)

        # ── Run all detectors ─────────────────────────────────────────────
        swings   = self._find_swings(df)
        ms_seq   = self.detect_market_structure(df, swings)
        bos_list = self.detect_bos_choch(df, swings, ms_seq)
        obs      = self.detect_order_blocks(df, atr)
        fvgs     = self.detect_fvg(df, atr)
        pools    = self.detect_liquidity_pools(df, swings)

        # ── Assemble and score setups ──────────────────────────────────────
        setups = self._assemble_setups(df, ms_seq, bos_list, obs, fvgs, pools, atr)

        # Filter by minimum confidence and sort descending
        valid = sorted(
            [s for s in setups if s.confidence >= self.MIN_CONFIDENCE],
            key=lambda x: x.confidence,
            reverse=True,
        )

        logger.info(
            f"scan() — {len(df)} bars | "
            f"{len(swings)} swings | "
            f"{len(obs)} OBs | "
            f"{len(fvgs)} FVGs | "
            f"{len(pools)} liquidity pools | "
            f"{len(valid)} valid setup(s)"
        )

        for s in valid:
            logger.debug(f"  {s}")

        return valid

    def get_best_setup(self, df: pd.DataFrame) -> Optional[TradeIdea]:
        """Return only the highest-confidence setup, or None."""
        setups = self.scan(df)
        return setups[0] if setups else None

    # =========================================================================
    # 1. MARKET STRUCTURE
    # =========================================================================

    def detect_market_structure(
        self, df: pd.DataFrame, swings: list[SwingPoint]
    ) -> list[SwingPoint]:
        """
        Classify each swing point as HH / HL / LH / LL / EH / EL.

        Method
        ------
        We track the most recent confirmed swing high and swing low.
        Each new swing is compared to the PREVIOUS swing of the same type:

                Last SH     Current SH     Label
                ──────────  ─────────────  ──────
                100         110            HH  (Higher High — bullish)
                100         90             LH  (Lower High  — bearish)
                100         100 ± tol      EH  (Equal High  — pool)

                Last SL     Current SL     Label
                ──────────  ─────────────  ──────
                90          95             HL  (Higher Low  — bullish)
                90          80             LL  (Lower Low   — bearish)
                90          90 ± tol       EL  (Equal Low   — pool)

        Trend interpretation:
            Sequence HH → HL → HH → HL  = confirmed UPTREND
            Sequence LL → LH → LL → LH  = confirmed DOWNTREND
            Mixed sequence               = CONSOLIDATION / transition

        Parameters
        ----------
        df     : pd.DataFrame  — OHLCV data (used for equal-level tolerance)
        swings : list[SwingPoint]  — output of _find_swings()

        Returns
        -------
        list[SwingPoint]  — Same list with .label populated on each point.
        """
        if len(swings) < 2:
            return swings

        # Separate into highs and lows for comparison
        highs = [s for s in swings if s.is_high]
        lows  = [s for s in swings if not s.is_high]

        # ── Label swing highs ─────────────────────────────────────────────
        for i in range(1, len(highs)):
            prev_price = highs[i - 1].price
            curr_price = highs[i].price
            diff       = curr_price - prev_price

            if abs(diff) <= self.EQUAL_LEVEL_TOLERANCE:
                highs[i].label = StructureLabel.EH   # Equal High — liquidity pool!
            elif curr_price > prev_price:
                highs[i].label = StructureLabel.HH   # Higher High — bullish structure
            else:
                highs[i].label = StructureLabel.LH   # Lower High  — bearish structure

        # ── Label swing lows ──────────────────────────────────────────────
        for i in range(1, len(lows)):
            prev_price = lows[i - 1].price
            curr_price = lows[i].price
            diff       = curr_price - prev_price

            if abs(diff) <= self.EQUAL_LEVEL_TOLERANCE:
                lows[i].label = StructureLabel.EL   # Equal Low — liquidity pool!
            elif curr_price > prev_price:
                lows[i].label = StructureLabel.HL   # Higher Low  — bullish structure
            else:
                lows[i].label = StructureLabel.LL   # Lower Low   — bearish structure

        return swings

    def get_trend_bias(self, ms_seq: list[SwingPoint]) -> str:
        """
        Determine the current structural bias from the swing label sequence.

        Returns "BULLISH", "BEARISH", or "NEUTRAL" based on last 4 swings.
        """
        if not ms_seq:
            return "NEUTRAL"

        # Look at the last 6 labelled points
        recent = [s for s in ms_seq if s.label is not None][-6:]
        labels = [s.label for s in recent if s.label]

        bullish_labels = {StructureLabel.HH, StructureLabel.HL}
        bearish_labels = {StructureLabel.LH, StructureLabel.LL}

        bull_count = sum(1 for l in labels if l in bullish_labels)
        bear_count = sum(1 for l in labels if l in bearish_labels)

        if bull_count > bear_count and bull_count >= 2:
            return "BULLISH"
        if bear_count > bull_count and bear_count >= 2:
            return "BEARISH"
        return "NEUTRAL"

    # =========================================================================
    # 2. BREAK OF STRUCTURE (BOS) & CHANGE OF CHARACTER (CHoCH)
    # =========================================================================

    def detect_bos_choch(
        self,
        df: pd.DataFrame,
        swings: list[SwingPoint],
        ms_seq: list[SwingPoint],
    ) -> list[StructureEvent]:
        """
        Detect Break of Structure (BOS) and Change of Character (CHoCH) events.

        Break of Structure (BOS) — Trend Continuation
        ──────────────────────────────────────────────
        A BOS confirms the trend is still intact.

        In an UPTREND (sequence of HH/HL):
            Bullish BOS = current bar's close > previous swing HIGH
            This means the market made a new high, confirming continuation.

        In a DOWNTREND (sequence of LH/LL):
            Bearish BOS = current bar's close < previous swing LOW
            This means the market made a new low, confirming continuation.

        Change of Character (CHoCH) — Potential Reversal
        ──────────────────────────────────────────────────
        A CHoCH is the FIRST structural break in the OPPOSITE direction.
        It is an early warning sign that the trend may be reversing.

        In an UPTREND:
            Bearish CHoCH = close < the most recent swing LOW
            Smart money has broken the bullish structure — watch for reversal.

        In a DOWNTREND:
            Bullish CHoCH = close > the most recent swing HIGH
            Smart money has broken the bearish structure — watch for reversal.

        Parameters
        ----------
        df     : OHLCV DataFrame
        swings : list of confirmed swing points
        ms_seq : same list with .label populated by detect_market_structure()

        Returns
        -------
        list[StructureEvent]  — all BOS and CHoCH events found
        """
        events: list[StructureEvent] = []
        if len(swings) < 3:
            return events

        # We need at least: prev swing, current swing as reference level,
        # and then a bar that closes beyond the reference.
        trend_bias = self.get_trend_bias(ms_seq)

        # Build sorted lookup for most recent swing at each bar
        # For each bar i, what was the most recent swing HIGH and LOW?
        most_recent_sh: Optional[SwingPoint] = None
        most_recent_sl: Optional[SwingPoint] = None

        swing_by_idx = {s.index: s for s in swings}
        sorted_swing_indices = sorted(swing_by_idx.keys())

        # Track which swings have been used as reference levels
        # to avoid flagging the same break multiple times
        used_sh_indices: set[int] = set()
        used_sl_indices: set[int] = set()

        # Scan each bar
        n = len(df)
        swing_ptr = 0   # pointer into sorted_swing_indices

        for i in range(1, n):
            close = float(df["close"].iloc[i])

            # Update most recent swing pointers as we advance in time
            while (swing_ptr < len(sorted_swing_indices) and
                   sorted_swing_indices[swing_ptr] < i):
                idx = sorted_swing_indices[swing_ptr]
                sw  = swing_by_idx[idx]
                if sw.is_high:
                    most_recent_sh = sw
                else:
                    most_recent_sl = sw
                swing_ptr += 1

            time_val = df["time"].iloc[i] if "time" in df.columns else None

            # ── Check for BULLISH break (close above previous swing HIGH) ──
            if most_recent_sh is not None and most_recent_sh.index not in used_sh_indices:
                if close > most_recent_sh.price:
                    displacement = close - most_recent_sh.price

                    # BOS if we're in an uptrend (continuation)
                    # CHoCH if we're in a downtrend (reversal signal)
                    btype = (
                        BosType.BULLISH_BOS if trend_bias == "BULLISH"
                        else BosType.BULLISH_CHOCH
                    )
                    events.append(StructureEvent(
                        bar_index    = i,
                        bos_type     = btype,
                        broken_level = most_recent_sh.price,
                        close_price  = close,
                        displacement = displacement,
                        time         = time_val,
                    ))
                    used_sh_indices.add(most_recent_sh.index)

            # ── Check for BEARISH break (close below previous swing LOW) ───
            if most_recent_sl is not None and most_recent_sl.index not in used_sl_indices:
                if close < most_recent_sl.price:
                    displacement = most_recent_sl.price - close

                    btype = (
                        BosType.BEARISH_BOS if trend_bias == "BEARISH"
                        else BosType.BEARISH_CHOCH
                    )
                    events.append(StructureEvent(
                        bar_index    = i,
                        bos_type     = btype,
                        broken_level = most_recent_sl.price,
                        close_price  = close,
                        displacement = displacement,
                        time         = time_val,
                    ))
                    used_sl_indices.add(most_recent_sl.index)

        logger.debug(
            f"detect_bos_choch() — {len(events)} events  "
            f"(bias={trend_bias})"
        )
        return events

    # =========================================================================
    # 3. ORDER BLOCKS
    # =========================================================================

    def detect_order_blocks(
        self, df: pd.DataFrame, atr: pd.Series
    ) -> list[OrderBlock]:
        """
        Detect bullish and bearish Order Blocks.

        Definition
        ----------
        A BULLISH Order Block is the LAST BEARISH candle immediately before
        a significant upward impulse.  This is where institutional buyers
        placed their orders — creating the fuel for the subsequent rally.

        A BEARISH Order Block is the LAST BULLISH candle immediately before
        a significant downward impulse.  This is where institutional sellers
        accumulated short positions.

        Visual representation of a Bullish OB:
        ┌──────────────────────────────────────────┐
        │  BAR[-3]  BAR[-2]  BAR[-1]   BAR[0]     │
        │  ┌──┐    ┌──┐    ╔══╗(OB)  ┌──┐         │
        │  │  │    │  │    ║  ║      │  │ ↑        │
        │  └──┘    └──┘    ╚══╝      └──┘ ↑ Impulse│
        │                  BEAR      BULL  ↑        │
        └──────────────────────────────────────────┘
        The last BEARISH candle before the bullish impulse = Bullish OB.

        Detection criteria
        ------------------
        1. Scan for a bearish candle (close < open).
        2. The NEXT N candles after it form a bullish impulse:
           - Net price gain > OB_MIN_IMPULSE_ATR × ATR
           - All (or most) candles close higher
        3. The OB candle's body ratio >= OB_MIN_BODY_RATIO.
        4. Age of OB <= OB_MAX_AGE_BARS.

        Returns
        -------
        list[OrderBlock]  — sorted newest to oldest, valid OBs only.
        """
        obs: list[OrderBlock] = []
        n   = len(df)

        # Minimum impulse candles to qualify as "impulsive"
        MIN_IMPULSE_BARS = 3

        for i in range(1, n - MIN_IMPULSE_BARS):
            age = n - 1 - i         # bars since this OB formed

            if age > self.OB_MAX_AGE_BARS:
                continue

            bar_open  = float(df["open"].iloc[i])
            bar_close = float(df["close"].iloc[i])
            bar_high  = float(df["high"].iloc[i])
            bar_low   = float(df["low"].iloc[i])
            bar_range = bar_high - bar_low

            if bar_range == 0:
                continue

            body       = abs(bar_close - bar_open)
            body_ratio = body / bar_range

            # Body ratio filter — OB candle must have a substantial body
            if body_ratio < self.OB_MIN_BODY_RATIO:
                continue

            bar_atr = float(atr.iloc[i]) if not np.isnan(atr.iloc[i]) else 0.0
            if bar_atr == 0.0:
                continue

            # ── Check for BULLISH OB ──────────────────────────────────────
            # Condition: this is a BEARISH candle
            if bar_close < bar_open:
                # Measure the bullish impulse in the next MIN_IMPULSE_BARS bars
                next_slice = df.iloc[i + 1: i + 1 + MIN_IMPULSE_BARS]
                impulse    = float(next_slice["close"].iloc[-1]) - float(next_slice["open"].iloc[0])
                bullish_bars = int((next_slice["close"] > next_slice["open"]).sum())

                # Impulse must be strong (ATR multiple) and mostly bullish candles
                if (impulse >= self.OB_MIN_IMPULSE_ATR * bar_atr and
                        bullish_bars >= MIN_IMPULSE_BARS - 1):

                    time_val = df["time"].iloc[i] if "time" in df.columns else None

                    ob = OrderBlock(
                        bar_index    = i,
                        ob_type      = OBType.BULLISH,
                        ob_high      = bar_high,
                        ob_low       = bar_low,
                        ob_open      = bar_open,
                        ob_close     = bar_close,
                        impulse_size = impulse,
                        time         = time_val,
                    )

                    # A bullish OB is invalidated only if price closes BELOW ob_low
                    # for 2+ consecutive bars (persistent, not a spike / sweep).
                    subsequent_closes = df["close"].iloc[i + 1:]
                    if not subsequent_closes.empty:
                        below = subsequent_closes < bar_low
                        # Count max consecutive True values
                        consec = below.groupby((below != below.shift()).cumsum()).sum().max()
                        if consec >= 2:
                            ob.valid = False    # sustained break — OB consumed

                    if ob.valid:
                        obs.append(ob)

            # ── Check for BEARISH OB ──────────────────────────────────────
            # Condition: this is a BULLISH candle
            elif bar_close > bar_open:
                # Measure the bearish impulse in the next MIN_IMPULSE_BARS bars
                next_slice = df.iloc[i + 1: i + 1 + MIN_IMPULSE_BARS]
                impulse    = float(next_slice["open"].iloc[0]) - float(next_slice["close"].iloc[-1])
                bearish_bars = int((next_slice["close"] < next_slice["open"]).sum())

                # The OB must have been followed by at least MIN_IMPULSE_BARS-1
                # bearish candles (allowing one neutral or pullback candle)
                if (impulse >= self.OB_MIN_IMPULSE_ATR * bar_atr and
                        bearish_bars >= MIN_IMPULSE_BARS - 1):

                    time_val = df["time"].iloc[i] if "time" in df.columns else None

                    ob = OrderBlock(
                        bar_index    = i,
                        ob_type      = OBType.BEARISH,
                        ob_high      = bar_high,
                        ob_low       = bar_low,
                        ob_open      = bar_open,
                        ob_close     = bar_close,
                        impulse_size = impulse,
                        time         = time_val,
                    )

                    # A bearish OB is invalidated only if price closes ABOVE ob_high
                    # for 2+ consecutive bars (persistent reclaim, not a temporary spike).
                    # A single bar spike above ob_high (like a stop-hunt sweep) does NOT
                    # invalidate the OB — this is actually a SMC entry trigger.
                    subsequent_closes = df["close"].iloc[i + 1:]
                    if not subsequent_closes.empty:
                        above = subsequent_closes > bar_high
                        consec = above.groupby((above != above.shift()).cumsum()).sum().max()
                        if consec >= 2:
                            ob.valid = False

                    if ob.valid:
                        obs.append(ob)

        # Sort newest to oldest
        obs.sort(key=lambda x: x.bar_index, reverse=True)
        logger.debug(f"detect_order_blocks() — {len(obs)} valid OBs found")
        return obs

    # =========================================================================
    # 4. FAIR VALUE GAPS (FVG)
    # =========================================================================

    def detect_fvg(
        self, df: pd.DataFrame, atr: pd.Series
    ) -> list[FairValueGap]:
        """
        Detect Fair Value Gaps (FVG) — three-candle imbalances.

        Concept
        -------
        When price moves aggressively in one direction, it sometimes skips
        over a price range entirely.  This creates a "gap" between the high
        of the first candle and the low of the third candle (for bullish FVG).

        Bullish FVG Pattern:
        ────────────────────
            C[-2]        C[-1]         C[0]
        ┌────────┐    ┌────────┐    ┌────────┐
        │        │    │        │    │        │
        │  HIGH  │    │  BIG   │    │  LOW   │
        └────────┘    │  BULL  │    └────────┘
            ↑          │        │        ↑
         C[-2].high   └────────┘   C[0].low
            │                          │
            └──────── FVG ZONE ────────┘
              (price skipped over this area)

        Condition: C[-2].high < C[0].low  → gap exists between them

        Bearish FVG Pattern:
        ────────────────────
        Condition: C[-2].low > C[0].high  → downward gap

        FVG characteristics:
        - Price often RETURNS to the FVG zone to "fill" the imbalance
        - In trends, price respects the 50% of the FVG as entry
        - Strong FVGs stay unfilled and act as support/resistance

        Returns
        -------
        list[FairValueGap]  — all valid FVGs, newest first.
        """
        fvgs: list[FairValueGap] = []
        n    = len(df)

        for i in range(2, n):
            age = n - 1 - i
            if age > self.FVG_LOOKBACK:
                continue

            bar_atr = float(atr.iloc[i]) if not np.isnan(atr.iloc[i]) else 0.0
            if bar_atr == 0:
                continue

            c0_high = float(df["high"].iloc[i])
            c0_low  = float(df["low"].iloc[i])
            c2_high = float(df["high"].iloc[i - 2])
            c2_low  = float(df["low"].iloc[i - 2])

            time_val = df["time"].iloc[i] if "time" in df.columns else None

            # ── Bullish FVG ───────────────────────────────────────────────
            # Gap between C[-2].high and C[0].low:
            # C[-2].high < C[0].low → a gap exists and price moved UP through it
            if c2_high < c0_low:
                gap_size = c0_low - c2_high

                # Filter: gap must be at least FVG_MIN_ATR_FRACTION × ATR
                if gap_size >= self.FVG_MIN_ATR_FRACTION * bar_atr:
                    # Check if the FVG is still unfilled by subsequent price action
                    subsequent = df["low"].iloc[i + 1:] if i + 1 < n else pd.Series(dtype=float)
                    already_filled = not subsequent.empty and (subsequent < c2_high).any()

                    fvgs.append(FairValueGap(
                        bar_index = i,
                        fvg_type  = FVGType.BULLISH,
                        gap_high  = c0_low,    # TOP of the gap zone
                        gap_low   = c2_high,   # BOTTOM of the gap zone
                        gap_size  = gap_size,
                        filled    = already_filled,
                        time      = time_val,
                    ))

            # ── Bearish FVG ───────────────────────────────────────────────
            # Gap between C[0].high and C[-2].low:
            # C[-2].low > C[0].high → a gap exists and price moved DOWN through it
            elif c2_low > c0_high:
                gap_size = c2_low - c0_high

                if gap_size >= self.FVG_MIN_ATR_FRACTION * bar_atr:
                    subsequent = df["high"].iloc[i + 1:] if i + 1 < n else pd.Series(dtype=float)
                    already_filled = not subsequent.empty and (subsequent > c2_low).any()

                    fvgs.append(FairValueGap(
                        bar_index = i,
                        fvg_type  = FVGType.BEARISH,
                        gap_high  = c2_low,    # TOP of the gap zone
                        gap_low   = c0_high,   # BOTTOM of the gap zone
                        gap_size  = gap_size,
                        filled    = already_filled,
                        time      = time_val,
                    ))

        fvgs.sort(key=lambda x: x.bar_index, reverse=True)
        unfilled = [f for f in fvgs if not f.filled]
        logger.debug(
            f"detect_fvg() — {len(fvgs)} total FVGs ({len(unfilled)} unfilled)"
        )
        return fvgs

    # =========================================================================
    # 5. LIQUIDITY POOLS
    # =========================================================================

    def detect_liquidity_pools(
        self, df: pd.DataFrame, swings: list[SwingPoint]
    ) -> list[LiquidityPool]:
        """
        Map price levels where stop-loss orders are likely clustered.

        Liquidity pool types
        --------------------
        HIGH  — Buy-side liquidity.  Short sellers place their stop-losses
                ABOVE obvious swing highs.  Institutions hunt these stops.

        LOW   — Sell-side liquidity.  Long buyers place their stop-losses
                BELOW obvious swing lows.

        EQUAL_HIGH — Two or more swing highs at approximately the same price.
                     Double tops, triple tops.  The larger the cluster, the
                     bigger the stop-hunt when it's taken.

        EQUAL_LOW  — Two or more swing lows at approximately the same price.
                     Double / triple bottoms.

        Pool strength heuristic
        -----------------------
        The more times a level was tested without breaking, the more stops
        have accumulated there.  We count "touches" as a strength proxy.

        Returns
        -------
        list[LiquidityPool]  — all identified liquidity pools
        """
        pools: list[LiquidityPool] = []

        swing_highs = [(s.index, s.price) for s in swings if s.is_high]
        swing_lows  = [(s.index, s.price) for s in swings if not s.is_high]

        # ── Standard swing high / low pools ──────────────────────────────
        for idx, price in swing_highs:
            # Count how many other bars touched but didn't break this level
            touches = int((
                (df["high"] >= price - self.EQUAL_LEVEL_TOLERANCE) &
                (df["high"] <  price + self.EQUAL_LEVEL_TOLERANCE)
            ).sum())
            pools.append(LiquidityPool(
                price     = price,
                pool_type = "HIGH",
                bar_index = idx,
                strength  = min(touches / 5.0, 1.0),   # 5 touches = full strength
            ))

        for idx, price in swing_lows:
            touches = int((
                (df["low"] <= price + self.EQUAL_LEVEL_TOLERANCE) &
                (df["low"] >  price - self.EQUAL_LEVEL_TOLERANCE)
            ).sum())
            pools.append(LiquidityPool(
                price     = price,
                pool_type = "LOW",
                bar_index = idx,
                strength  = min(touches / 5.0, 1.0),
            ))

        # ── Equal highs / lows (strongest liquidity clusters) ────────────
        # Group swing highs that are within EQUAL_LEVEL_TOLERANCE of each other
        tol = self.EQUAL_LEVEL_TOLERANCE

        used: set[int] = set()
        for i, (idx_i, price_i) in enumerate(swing_highs):
            if i in used:
                continue
            cluster = [(idx_i, price_i)]
            for j, (idx_j, price_j) in enumerate(swing_highs):
                if j != i and j not in used and abs(price_j - price_i) <= tol:
                    cluster.append((idx_j, price_j))
                    used.add(j)
            if len(cluster) >= 2:
                # Equal highs found — mark as a pool with high strength
                avg_price = sum(p for _, p in cluster) / len(cluster)
                latest_idx = max(idx for idx, _ in cluster)
                pools.append(LiquidityPool(
                    price     = avg_price,
                    pool_type = "EQUAL_HIGH",
                    bar_index = latest_idx,
                    strength  = min(len(cluster) / 3.0, 1.0),
                ))

        used = set()
        for i, (idx_i, price_i) in enumerate(swing_lows):
            if i in used:
                continue
            cluster = [(idx_i, price_i)]
            for j, (idx_j, price_j) in enumerate(swing_lows):
                if j != i and j not in used and abs(price_j - price_i) <= tol:
                    cluster.append((idx_j, price_j))
                    used.add(j)
            if len(cluster) >= 2:
                avg_price = sum(p for _, p in cluster) / len(cluster)
                latest_idx = max(idx for idx, _ in cluster)
                pools.append(LiquidityPool(
                    price     = avg_price,
                    pool_type = "EQUAL_LOW",
                    bar_index = latest_idx,
                    strength  = min(len(cluster) / 3.0, 1.0),
                ))

        # Mark pools that have already been swept by price action.
        #
        # A HIGH / EQUAL_HIGH pool is swept when:
        #   price traded ABOVE the level at some point AND the final close is BELOW it
        #   (price went up, triggered stops, then reversed down)
        #
        # A LOW / EQUAL_LOW pool is swept when:
        #   price traded BELOW the level at some point AND the final close is ABOVE it
        #   (price went down, triggered stops, then reversed up — the SMC setup we want)
        #
        min_low   = float(df["low"].min())
        max_high  = float(df["high"].max())
        final_close = float(df["close"].iloc[-1])

        for pool in pools:
            if pool.pool_type in ("HIGH", "EQUAL_HIGH"):
                pool.swept = max_high > pool.price and final_close < pool.price
            else:
                pool.swept = min_low < pool.price and final_close > pool.price

        logger.debug(f"detect_liquidity_pools() — {len(pools)} pools mapped")
        return pools

    # =========================================================================
    # 6. SETUP ASSEMBLY & ENTRY LOGIC
    # =========================================================================

    def _assemble_setups(
        self,
        df:       pd.DataFrame,
        ms_seq:   list[SwingPoint],
        bos_list: list[StructureEvent],
        obs:      list[OrderBlock],
        fvgs:     list[FairValueGap],
        pools:    list[LiquidityPool],
        atr:      pd.Series,
    ) -> list[TradeIdea]:
        """
        Combine all SMC components to assemble complete TradeIdea setups.

        BUY Setup Logic
        ───────────────
        Pre-conditions:
          a) Market structure bias is BULLISH (recent HH/HL sequence), OR
          b) A bullish CHoCH just occurred (potential reversal after downtrend)

        Trigger:
          A sell-side liquidity pool has been swept (prev low taken then closed above)

        Entry zone (pick best available):
          Priority 1: Price is currently inside a BULLISH order block
          Priority 2: Price is currently inside an unfilled BULLISH FVG
          Priority 3: Use ATR-derived entry if only BOS/CHoCH confirmation

        Stop Loss:
          Below the sweep low (the lowest point of the spike that took the lows)
          with a small buffer (0.2 × ATR)

        Take Profit:
          Primary:   Previous swing HIGH (buy-side liquidity above)
          Secondary: Next higher swing HIGH if available

        SELL Setup Logic
        ────────────────
        Mirror of the BUY logic:
          - Bearish structural bias OR bearish CHoCH
          - Buy-side liquidity swept (prev high taken then closed below)
          - Price inside bearish OB or bearish FVG
          - TP = previous swing LOW below
        """
        setups: list[TradeIdea] = []

        if not ms_seq or not bos_list:
            return setups

        trend_bias = self.get_trend_bias(ms_seq)
        n          = len(df)
        current_close = float(df["close"].iloc[-1])
        current_low   = float(df["low"].iloc[-1])
        current_high  = float(df["high"].iloc[-1])
        current_atr   = float(atr.iloc[-1]) if not np.isnan(atr.iloc[-1]) else 1.0
        sl_buffer     = 0.2 * current_atr

        # ─────────────────────────────────────────────────────────────────
        # BUY SETUP
        # ─────────────────────────────────────────────────────────────────
        if trend_bias in ("BULLISH", "NEUTRAL"):

            # Look for a recent Bullish CHoCH or confirmed BOS (last 30 bars)
            recent_bull_events = [
                e for e in bos_list
                if e.is_bullish and (n - 1 - e.bar_index) <= 30
            ]

            # Find swept sell-side liquidity (low pool swept + close above it)
            # Use a generous lookback — a liquidity sweep could have happened 100 bars ago
            # and price may only now be retracing back to an OB in the same area.
            swept_low_pools = [
                p for p in pools
                if p.pool_type in ("LOW", "EQUAL_LOW")
                and p.swept
                and (n - 1 - p.bar_index) <= 100
            ]

            if swept_low_pools:
                # Find the most recently swept pool as the stop reference
                best_pool = max(swept_low_pools, key=lambda p: p.bar_index)

                # Stop loss: below the sweep low (lowest point of the wick)
                sweep_low = float(df["low"].iloc[
                    max(0, best_pool.bar_index - 2): best_pool.bar_index + 3
                ].min())
                stop_loss = sweep_low - sl_buffer

                # Entry zone: look for bullish OB or FVG that price is near
                matching_ob  = next(
                    (ob for ob in obs
                     if ob.ob_type == OBType.BULLISH
                     and ob.contains_price(current_close)), None
                )
                matching_fvg = next(
                    (fvg for fvg in fvgs
                     if fvg.fvg_type == FVGType.BULLISH
                     and not fvg.filled
                     and fvg.contains_price(current_close)), None
                )

                # Determine entry price
                if matching_ob is not None:
                    entry = matching_ob.midpoint
                elif matching_fvg is not None:
                    entry = matching_fvg.midpoint
                else:
                    # No OB/FVG at current price — use current close as entry
                    # (less precise; score will be lower without OB/FVG confluence)
                    entry = current_close

                # Safety: entry must be above stop_loss
                if entry <= stop_loss:
                    entry = stop_loss + current_atr

                # Take profit: find nearest unswept HIGH pool above entry
                tp_pools = sorted(
                    [p for p in pools if p.pool_type in ("HIGH", "EQUAL_HIGH")
                     and p.price > entry and not p.swept],
                    key=lambda p: p.price,
                )
                if tp_pools:
                    take_profit   = tp_pools[0].price
                    take_profit_2 = tp_pools[1].price if len(tp_pools) > 1 else None
                else:
                    # Fall back to nearest swing high above entry
                    highs_above = [s for s in ms_seq if s.is_high and s.price > entry]
                    if highs_above:
                        take_profit   = min(highs_above, key=lambda s: s.price).price
                        take_profit_2 = None
                    else:
                        take_profit   = entry + 2.0 * abs(entry - stop_loss)
                        take_profit_2 = entry + 4.0 * abs(entry - stop_loss)

                # Score and build the idea
                score = self._score_confluence(
                    direction         = "BUY",
                    trend_bias        = trend_bias,
                    has_ob            = matching_ob is not None,
                    has_fvg           = matching_fvg is not None,
                    has_bos           = any(e.is_bos for e in recent_bull_events),
                    has_choch         = any(e.is_choch for e in recent_bull_events),
                    has_liq_sweep     = bool(swept_low_pools),
                    entry             = entry,
                    stop_loss         = stop_loss,
                    take_profit       = take_profit,
                    min_rr            = 1.5,
                    pool_strength     = best_pool.strength,
                    ob_impulse        = matching_ob.impulse_size if matching_ob else 0.0,
                    fvg_size          = matching_fvg.gap_size if matching_fvg else 0.0,
                    current_atr       = current_atr,
                )

                has_bos_flag   = any(e.is_bos for e in recent_bull_events)
                has_choch_flag = any(e.is_choch for e in recent_bull_events)

                notes_parts = [
                    f"BUY setup | bias={trend_bias}",
                    f"entry={entry:.2f} SL={stop_loss:.2f} TP={take_profit:.2f}",
                    f"OB={'YES' if matching_ob else 'NO'} FVG={'YES' if matching_fvg else 'NO'}",
                    f"BOS={'YES' if has_bos_flag else 'NO'} CHoCH={'YES' if has_choch_flag else 'NO'}",
                    f"LiqSweep=YES ({best_pool.pool_type} @ {best_pool.price:.2f})",
                    f"confidence={score:.0%}",
                ]

                idea = TradeIdea(
                    direction            = "BUY",
                    entry                = round(entry, 2),
                    stop_loss            = round(stop_loss, 2),
                    take_profit          = round(take_profit, 2),
                    take_profit_2        = round(take_profit_2, 2) if take_profit_2 else None,
                    confidence           = round(score, 3),
                    bos_event            = recent_bull_events[0] if recent_bull_events else None,
                    order_block          = matching_ob,
                    fvg                  = matching_fvg,
                    liquidity_pool       = best_pool,
                    bar_index            = n - 1,
                    time                 = df["time"].iloc[-1] if "time" in df.columns else None,
                    has_ob               = matching_ob is not None,
                    has_fvg              = matching_fvg is not None,
                    has_bos              = has_bos_flag,
                    has_choch            = has_choch_flag,
                    has_liquidity_sweep  = True,
                    notes                = " | ".join(notes_parts),
                )
                setups.append(idea)

        # ─────────────────────────────────────────────────────────────────
        # SELL SETUP
        # ─────────────────────────────────────────────────────────────────
        if trend_bias in ("BEARISH", "NEUTRAL"):

            recent_bear_events = [
                e for e in bos_list
                if not e.is_bullish and (n - 1 - e.bar_index) <= 30
            ]

            # Buy-side liquidity swept: prev high taken then price closed below
            swept_high_pools = [
                p for p in pools
                if p.pool_type in ("HIGH", "EQUAL_HIGH")
                and p.swept
                and (n - 1 - p.bar_index) <= 100
            ]

            if swept_high_pools:
                best_pool = max(swept_high_pools, key=lambda p: p.bar_index)

                # Stop loss: above the sweep high + buffer
                sweep_high = float(df["high"].iloc[
                    max(0, best_pool.bar_index - 2): best_pool.bar_index + 3
                ].max())
                stop_loss = sweep_high + sl_buffer

                # Entry zone
                matching_ob = next(
                    (ob for ob in obs
                     if ob.ob_type == OBType.BEARISH
                     and ob.contains_price(current_close)), None
                )
                matching_fvg = next(
                    (fvg for fvg in fvgs
                     if fvg.fvg_type == FVGType.BEARISH
                     and not fvg.filled
                     and fvg.contains_price(current_close)), None
                )

                if matching_ob is not None:
                    entry = matching_ob.midpoint
                elif matching_fvg is not None:
                    entry = matching_fvg.midpoint
                else:
                    entry = current_close

                # Safety: entry must be below stop_loss
                if entry >= stop_loss:
                    entry = stop_loss - current_atr

                # Take profit: nearest unswept LOW pool below entry
                tp_pools = sorted(
                    [p for p in pools if p.pool_type in ("LOW", "EQUAL_LOW")
                     and p.price < entry and not p.swept],
                    key=lambda p: p.price,
                    reverse=True,
                )
                if tp_pools:
                    take_profit   = tp_pools[0].price
                    take_profit_2 = tp_pools[1].price if len(tp_pools) > 1 else None
                else:
                    lows_below = [s for s in ms_seq if not s.is_high and s.price < entry]
                    if lows_below:
                        take_profit   = max(lows_below, key=lambda s: s.price).price
                        take_profit_2 = None
                    else:
                        take_profit   = entry - 2.0 * abs(stop_loss - entry)
                        take_profit_2 = entry - 4.0 * abs(stop_loss - entry)

                score = self._score_confluence(
                    direction         = "SELL",
                    trend_bias        = trend_bias,
                    has_ob            = matching_ob is not None,
                    has_fvg           = matching_fvg is not None,
                    has_bos           = any(e.is_bos for e in recent_bear_events),
                    has_choch         = any(e.is_choch for e in recent_bear_events),
                    has_liq_sweep     = bool(swept_high_pools),
                    entry             = entry,
                    stop_loss         = stop_loss,
                    take_profit       = take_profit,
                    min_rr            = 1.5,
                    pool_strength     = best_pool.strength,
                    ob_impulse        = matching_ob.impulse_size if matching_ob else 0.0,
                    fvg_size          = matching_fvg.gap_size if matching_fvg else 0.0,
                    current_atr       = current_atr,
                )

                has_bos_flag   = any(e.is_bos for e in recent_bear_events)
                has_choch_flag = any(e.is_choch for e in recent_bear_events)

                notes_parts = [
                    f"SELL setup | bias={trend_bias}",
                    f"entry={entry:.2f} SL={stop_loss:.2f} TP={take_profit:.2f}",
                    f"OB={'YES' if matching_ob else 'NO'} FVG={'YES' if matching_fvg else 'NO'}",
                    f"BOS={'YES' if has_bos_flag else 'NO'} CHoCH={'YES' if has_choch_flag else 'NO'}",
                    f"LiqSweep=YES ({best_pool.pool_type} @ {best_pool.price:.2f})",
                    f"confidence={score:.0%}",
                ]

                idea = TradeIdea(
                    direction            = "SELL",
                    entry                = round(entry, 2),
                    stop_loss            = round(stop_loss, 2),
                    take_profit          = round(take_profit, 2),
                    take_profit_2        = round(take_profit_2, 2) if take_profit_2 else None,
                    confidence           = round(score, 3),
                    bos_event            = recent_bear_events[0] if recent_bear_events else None,
                    order_block          = matching_ob,
                    fvg                  = matching_fvg,
                    liquidity_pool       = best_pool,
                    bar_index            = n - 1,
                    time                 = df["time"].iloc[-1] if "time" in df.columns else None,
                    has_ob               = matching_ob is not None,
                    has_fvg              = matching_fvg is not None,
                    has_bos              = has_bos_flag,
                    has_choch            = has_choch_flag,
                    has_liquidity_sweep  = True,
                    notes                = " | ".join(notes_parts),
                )
                setups.append(idea)

        return setups

    # =========================================================================
    # 7. CONFLUENCE SCORER
    # =========================================================================

    def _score_confluence(
        self,
        direction:     str,
        trend_bias:    str,
        has_ob:        bool,
        has_fvg:       bool,
        has_bos:       bool,
        has_choch:     bool,
        has_liq_sweep: bool,
        entry:         float,
        stop_loss:     float,
        take_profit:   float,
        min_rr:        float,
        pool_strength: float,
        ob_impulse:    float,
        fvg_size:      float,
        current_atr:   float,
    ) -> float:
        """
        Compute a weighted confluence score [0.0, 1.0] for a trade setup.

        Weight allocation
        -----------------
        The weights reflect the relative importance of each SMC factor.
        A setup that scores all components gets confidence = 1.0.

        Component           Weight  Rationale
        ──────────────────  ──────  ──────────────────────────────────────
        Liquidity sweep      0.25   Core SMC concept — without it there's no
                                    "institutional signature" for the move.
        Order Block          0.20   OB is where institutions placed orders.
                                    Price returning to it = high-probability zone.
        Fair Value Gap       0.15   FVG overlap with OB increases precision.
        BOS confirmation     0.15   Structural break confirms the new move.
        CHoCH                0.10   Reversal warning — extra credit if present.
        Trend alignment      0.10   Trading with the higher-timeframe bias.
        R:R ratio            0.05   Minimum 1.5:1; bonus for 2+ R:R setups.

        Additional quality modifiers (sub-scores within each component):
        - Pool strength (how many times was the level tested?)
        - OB impulse size (larger impulse = stronger institutional move)
        - FVG size relative to ATR
        """
        score = 0.0

        # ── Liquidity sweep (25%) ──────────────────────────────────────────
        # Core requirement: must have a liquidity sweep to score here.
        if has_liq_sweep:
            # Bonus for pool strength (well-tested = more stops = bigger hunt)
            sweep_score = 0.15 + 0.10 * pool_strength
            score += sweep_score
        # No sweep → no score for this component

        # ── Order Block (20%) ─────────────────────────────────────────────
        if has_ob:
            # Larger impulse from the OB = stronger institutional conviction
            impulse_atr_ratio = ob_impulse / current_atr if current_atr > 0 else 0.0
            ob_score = 0.10 + 0.10 * min(impulse_atr_ratio / 3.0, 1.0)
            score += ob_score
        # Partial credit if only CHoCH/BOS without OB (maybe price is approaching)

        # ── Fair Value Gap (15%) ──────────────────────────────────────────
        if has_fvg:
            fvg_atr_ratio = fvg_size / current_atr if current_atr > 0 else 0.0
            fvg_score = 0.07 + 0.08 * min(fvg_atr_ratio / 2.0, 1.0)
            score += fvg_score

        # ── BOS confirmation (15%) ────────────────────────────────────────
        if has_bos:
            score += 0.15

        # ── CHoCH (10%) ───────────────────────────────────────────────────
        if has_choch:
            score += 0.10

        # ── Trend alignment (10%) ─────────────────────────────────────────
        trend_aligned = (
            (direction == "BUY"  and trend_bias == "BULLISH") or
            (direction == "SELL" and trend_bias == "BEARISH")
        )
        if trend_aligned:
            score += 0.10
        elif trend_bias == "NEUTRAL":
            score += 0.05   # partial credit — not counter-trend

        # ── R:R quality (5%) ──────────────────────────────────────────────
        sl_dist = abs(entry - stop_loss)
        if sl_dist > 0:
            tp_dist = abs(take_profit - entry)
            rr      = tp_dist / sl_dist
            if rr >= 2.0:
                score += 0.05          # full credit for 2:1+
            elif rr >= min_rr:
                score += 0.025         # partial for 1.5:1 to 2:1

        return min(max(score, 0.0), 1.0)

    # =========================================================================
    # INTERNAL UTILITIES
    # =========================================================================

    def _find_swings(self, df: pd.DataFrame) -> list[SwingPoint]:
        """
        Identify all confirmed swing highs and swing lows.

        A swing high at bar[i] requires:
            bar[i].high > bar[j].high  for all j in [i-lb, i+lb] except j==i

        A swing low at bar[i] requires:
            bar[i].low  < bar[j].low   for all j in [i-lb, i+lb] except j==i

        Uses vectorised rolling max/min for speed.
        """
        swings: list[SwingPoint] = []
        lb     = self.SWING_LOOKBACK
        n      = len(df)

        highs = df["high"]
        lows  = df["low"]

        # Rolling max/min over a window of 2*lb+1 bars centred on each bar.
        # Using min_periods=2*lb+1 ensures we only confirm fully-formed swings.
        window       = 2 * lb + 1
        roll_max     = highs.rolling(window=window, center=True, min_periods=window).max()
        roll_min     = lows.rolling(window=window, center=True, min_periods=window).min()

        for i in range(lb, n - lb):
            time_val = df["time"].iloc[i] if "time" in df.columns else None

            # Swing high: this bar's high == the rolling max centred on it
            if float(highs.iloc[i]) == float(roll_max.iloc[i]):
                swings.append(SwingPoint(
                    index   = i,
                    price   = float(highs.iloc[i]),
                    is_high = True,
                    time    = time_val,
                ))

            # Swing low: this bar's low == the rolling min centred on it
            if float(lows.iloc[i]) == float(roll_min.iloc[i]):
                swings.append(SwingPoint(
                    index   = i,
                    price   = float(lows.iloc[i]),
                    is_high = False,
                    time    = time_val,
                ))

        # Sort by bar index (chronological order)
        swings.sort(key=lambda s: s.index)
        return swings

    def _atr(self, df: pd.DataFrame) -> pd.Series:
        """Wilder's ATR series for the DataFrame."""
        high  = df["high"]
        low   = df["low"]
        close = df["close"]
        hl    = high - low
        h_pc  = (high - close.shift(1)).abs()
        l_pc  = (low  - close.shift(1)).abs()
        tr    = pd.concat([hl, h_pc, l_pc], axis=1).max(axis=1)
        return tr.ewm(
            alpha=1.0 / self.ATR_PERIOD,
            min_periods=self.ATR_PERIOD,
            adjust=False,
        ).mean()

    def __repr__(self) -> str:
        return (
            f"SmartMoneyStrategy("
            f"swing_lb={self.SWING_LOOKBACK}, "
            f"ob_lookback={self.OB_LOOKBACK}, "
            f"fvg_lookback={self.FVG_LOOKBACK}, "
            f"min_conf={self.MIN_CONFIDENCE}"
            f")"
        )
