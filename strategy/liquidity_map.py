"""
strategy/liquidity_map.py
===========================
Institutional Liquidity Zone Mapper for XAUUSD.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DISTINCTION FROM liquidity_detection.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  liquidity_detection.py  — detects liquidity SWEEP EVENTS
                             (a pool was already hit and price reversed)

  liquidity_map.py        — maps liquidity ZONES THAT STILL EXIST
                             (untouched pools where price is LIKELY to react)

This module answers the question:
  "Where are the clusters of stop-loss orders sitting RIGHT NOW?"

Strategies use this output to:
  1. TARGET take profits at the next liquidity zone above/below.
  2. AVOID entering trades where price is about to run straight into
     a thick liquidity pool (which would cause a reversal before TP).
  3. MARK entry zones near low-liquidity gaps (smoother path to TP).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ZONE TYPES DETECTED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  1. EQUAL HIGHS / EQUAL LOWS
       Two or more swing highs/lows within a tolerance band.
       → Retail stop clusters sitting at the same level.
       → Smart money magnets — high probability targets.

  2. PREVIOUS SESSION HIGHS / LOWS
       Yesterday's High and Low on the primary (H1) timeframe.
       → Intraday traders have stops just beyond these levels.
       → Classic ICT "daily range" liquidity.

  3. WEEKLY HIGHS / LOWS
       Current week's running High and Low.
       → Swing traders and position traders cluster stops here.
       → Key targets for larger institutional moves.

  4. ROUND NUMBERS (Psychological Levels)
       Prices ending in .00, .50 (and optionally .25/.75) increments.
       → Retail orders cluster at psychological levels.
       → Examples for Gold: 2300.00, 2325.00, 2350.00, 2375.00

  5. SWING HIGHS / LOWS  (significant untouched structure)
       Recent swing points that have NOT been swept yet.
       Higher strength = more bars confirmed the level.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  {
    "liquidity_above": [
      {
        "price":       2368.50,
        "type":        "EQUAL_HIGHS",
        "strength":    0.85,
        "touches":     3,
        "description": "Equal highs cluster at 2368.50 (3 touches)"
      },
      ...
    ],
    "liquidity_below": [...],

    # Convenience lookups
    "nearest_above":  { price, type, strength },
    "nearest_below":  { price, type, strength },
    "current_price":  2355.20,
    "summary":        "3 zones above | 4 zones below",
  }

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PUBLIC API
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  mapper = LiquidityMapper(settings)

  # Primary call — returns full LiquidityMap
  lmap = mapper.map(df_h1, df_m15=df_m15)

  # Convenience helpers
  above  = lmap.liquidity_above       # list[LiquidityZone], sorted nearest first
  below  = lmap.liquidity_below       # list[LiquidityZone], sorted nearest first
  near_a = lmap.nearest_above         # closest zone above current price
  near_b = lmap.nearest_below         # closest zone below current price

  # Check proximity before entry
  is_safe = mapper.is_entry_safe(entry_price=2355.0, direction="BUY", lmap=lmap)

  # Dict output (for JSON / Telegram / SignalGenerator)
  d = lmap.to_dict()
  # → {"liquidity_above": [...], "liquidity_below": [...], ...}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRATEGY INTEGRATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  # For a BUY signal entry at 2355, TP1 = 2368:
  lmap = mapper.map(df_h1)
  near_above = lmap.nearest_above

  if near_above and near_above.price < tp1:
      # Liquidity sits between entry and TP — consider adjusting TP
      # to target the liquidity zone itself, or filter out this signal
      adjusted_tp = near_above.price - 2 * atr

  # Or use is_entry_safe() which wraps this logic automatically
  if mapper.is_entry_safe(entry, "BUY", lmap):
      dispatch_signal(signal)
  else:
      log("Skipped: entry runs directly into liquidity zone")

Dependencies
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  pandas, numpy, config.settings.Settings, utils.logger
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import Settings
from utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Zone type enum
# ─────────────────────────────────────────────────────────────────────────────

class ZoneType(str, Enum):
    """Classification of a liquidity zone by origin."""
    EQUAL_HIGHS       = "EQUAL_HIGHS"
    EQUAL_LOWS        = "EQUAL_LOWS"
    SESSION_HIGH      = "SESSION_HIGH"
    SESSION_LOW       = "SESSION_LOW"
    WEEKLY_HIGH       = "WEEKLY_HIGH"
    WEEKLY_LOW        = "WEEKLY_LOW"
    ROUND_NUMBER      = "ROUND_NUMBER"
    HALF_NUMBER       = "HALF_NUMBER"        # e.g. 2350.50 (minor)
    SWING_HIGH        = "SWING_HIGH"
    SWING_LOW         = "SWING_LOW"


# Strength weights for each zone type (used in combined score + sorting)
_ZONE_TYPE_BASE_STRENGTH: dict[str, float] = {
    ZoneType.EQUAL_HIGHS:   0.90,
    ZoneType.EQUAL_LOWS:    0.90,
    ZoneType.WEEKLY_HIGH:   0.85,
    ZoneType.WEEKLY_LOW:    0.85,
    ZoneType.SESSION_HIGH:  0.80,
    ZoneType.SESSION_LOW:   0.80,
    ZoneType.ROUND_NUMBER:  0.75,
    ZoneType.SWING_HIGH:    0.65,
    ZoneType.SWING_LOW:     0.65,
    ZoneType.HALF_NUMBER:   0.50,
}


# ─────────────────────────────────────────────────────────────────────────────
# Single liquidity zone
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LiquidityZone:
    """
    Represents one identified liquidity concentration at or near a price level.

    Attributes
    ----------
    price       : The key price level (mid-point of the zone).
    zone_high   : Upper boundary of the band (price + tolerance).
    zone_low    : Lower boundary of the band (price - tolerance).
    zone_type   : What kind of liquidity cluster this is.
    strength    : 0.0 – 1.0 composite score (higher = more significant).
    touches     : Number of times price has tested this level.
    age_bars    : How many bars ago the zone was first established.
    description : Human-readable explanation.
    is_above    : True if zone is above current price (buy-side liquidity).
    is_below    : True if zone is below current price (sell-side liquidity).
    """
    price:       float
    zone_high:   float
    zone_low:    float
    zone_type:   ZoneType
    strength:    float
    touches:     int          = 1
    age_bars:    int          = 0
    description: str          = ""
    is_above:    bool         = True    # relative to current price at scan time

    @property
    def is_below(self) -> bool:
        return not self.is_above

    @property
    def midpoint(self) -> float:
        return round((self.zone_high + self.zone_low) / 2, 2)

    def to_dict(self) -> dict:
        return {
            "price":       round(self.price, 2),
            "zone_high":   round(self.zone_high, 2),
            "zone_low":    round(self.zone_low, 2),
            "type":        self.zone_type.value,
            "strength":    round(self.strength, 3),
            "touches":     self.touches,
            "age_bars":    self.age_bars,
            "description": self.description,
        }

    def __str__(self) -> str:
        side = "↑" if self.is_above else "↓"
        return (
            f"LiqZone({side} {self.zone_type.value} "
            f"@ {self.price:.2f}  "
            f"str={self.strength:.0%}  "
            f"touches={self.touches})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Full map result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LiquidityMap:
    """
    Complete snapshot of institutional liquidity zones above and below price.

    Attributes
    ----------
    liquidity_above : Price zones above current price (buy-side / resistance liquidity).
                      Sorted by price ascending (nearest zone first).
    liquidity_below : Price zones below current price (sell-side / support liquidity).
                      Sorted by price descending (nearest zone first).
    current_price   : Most recent close used as the reference price.
    scanned_bars    : Number of bars analysed.
    scan_time       : UTC timestamp when the scan was performed.
    """
    liquidity_above: list[LiquidityZone] = field(default_factory=list)
    liquidity_below: list[LiquidityZone] = field(default_factory=list)
    current_price:   float               = 0.0
    scanned_bars:    int                 = 0
    scan_time:       Optional[datetime]  = None

    # ── Convenience lookups ───────────────────────────────────────────────────

    @property
    def nearest_above(self) -> Optional[LiquidityZone]:
        """Closest liquidity zone above current price."""
        return self.liquidity_above[0] if self.liquidity_above else None

    @property
    def nearest_below(self) -> Optional[LiquidityZone]:
        """Closest liquidity zone below current price."""
        return self.liquidity_below[0] if self.liquidity_below else None

    @property
    def summary(self) -> str:
        return (
            f"{len(self.liquidity_above)} zones above | "
            f"{len(self.liquidity_below)} zones below  "
            f"(ref={self.current_price:.2f})"
        )

    def zones_between(self, price_low: float, price_high: float) -> list[LiquidityZone]:
        """
        Return all zones in the range [price_low, price_high].

        Useful for checking whether a TP target has liquidity obstruction.
        """
        all_zones = self.liquidity_above + self.liquidity_below
        return [
            z for z in all_zones
            if price_low <= z.price <= price_high
        ]

    def strongest_above(self, n: int = 3) -> list[LiquidityZone]:
        """Top N zones above, sorted by strength descending."""
        return sorted(self.liquidity_above, key=lambda z: z.strength, reverse=True)[:n]

    def strongest_below(self, n: int = 3) -> list[LiquidityZone]:
        """Top N zones below, sorted by strength descending."""
        return sorted(self.liquidity_below, key=lambda z: z.strength, reverse=True)[:n]

    def to_dict(self) -> dict:
        d = {
            "liquidity_above": [z.to_dict() for z in self.liquidity_above],
            "liquidity_below": [z.to_dict() for z in self.liquidity_below],
            "nearest_above":   self.nearest_above.to_dict() if self.nearest_above else None,
            "nearest_below":   self.nearest_below.to_dict() if self.nearest_below else None,
            "current_price":   round(self.current_price, 2),
            "scanned_bars":    self.scanned_bars,
            "summary":         self.summary,
        }
        if self.scan_time:
            d["scan_time"] = self.scan_time.isoformat()
        return d

    def print_summary(self) -> None:
        """Print a formatted liquidity map to stdout."""
        sep = "─" * 52
        print(f"\n{'═'*52}")
        print(f"  LIQUIDITY MAP   ref={self.current_price:.2f}")
        print(f"{'═'*52}")
        print(f"  {'Zones above (buy-side / resistance):':}")
        if self.liquidity_above:
            for z in self.liquidity_above[:8]:
                dist = z.price - self.current_price
                bar  = "█" * int(z.strength * 20)
                print(f"    +{dist:>6.2f}  {z.price:>8.2f}  "
                      f"{z.zone_type.value:<18}  str={z.strength:.0%}  {bar}")
        else:
            print("    (none detected)")
        print(f"{sep}")
        print(f"  ▶  Current: {self.current_price:.2f}")
        print(f"{sep}")
        print(f"  {'Zones below (sell-side / support):':}")
        if self.liquidity_below:
            for z in self.liquidity_below[:8]:
                dist = self.current_price - z.price
                bar  = "█" * int(z.strength * 20)
                print(f"    -{dist:>6.2f}  {z.price:>8.2f}  "
                      f"{z.zone_type.value:<18}  str={z.strength:.0%}  {bar}")
        else:
            print("    (none detected)")
        print(f"{'═'*52}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Mapper
# ─────────────────────────────────────────────────────────────────────────────

class LiquidityMapper:
    """
    Identifies institutional liquidity zones across multiple zone types.

    The mapper scans an OHLCV DataFrame (typically H1 or M15) and builds
    a full LiquidityMap that the signal generator and strategy layer use to:
      - Set smarter TP targets (aim for a zone, not beyond it)
      - Avoid entering trades that run into nearby heavy liquidity
      - Identify likely reversal price for stop placement

    Parameters
    ----------
    settings         : App config (for symbol point, ATR period, etc.)
    equal_tolerance  : % price tolerance for equal high/low clustering
                       (default 0.05 = within 0.05% of each other)
    swing_lookback   : Bars each side to confirm a swing high/low  (default 3)
    max_zone_age     : Levels older than this many bars are excluded (default 100)
    round_step       : Round-number interval in price points  (default 50)
    half_step        : Half-step round number interval     (default 25)
    max_zones_per_side: Maximum zones to keep per side     (default 10)
    """

    def __init__(
        self,
        settings:            Settings,
        equal_tolerance:     float = 0.05,   # %  (0.05 = within 0.05% → ~1.2pt on 2400)
        swing_lookback:      int   = 3,
        max_zone_age:        int   = 200,
        round_step:          float = 50.0,   # Gold: 2300, 2350, 2400 …
        half_step:           float = 25.0,   # Gold: 2325, 2375 …
        max_zones_per_side:  int   = 10,
        min_proximity_atr:   float = 0.5,    # block entry if zone within 0.5×ATR
    ) -> None:
        self.cfg               = settings
        self.equal_tol         = equal_tolerance / 100.0   # convert % to ratio
        self.swing_lookback    = swing_lookback
        self.max_zone_age      = max_zone_age
        self.round_step        = round_step
        self.half_step         = half_step
        self.max_zones_per_side = max_zones_per_side
        self.min_proximity_atr = min_proximity_atr

    # =========================================================================
    # PRIMARY PUBLIC API
    # =========================================================================

    def map(
        self,
        df_h1:  pd.DataFrame,
        df_m15: Optional[pd.DataFrame] = None,
        df_w:   Optional[pd.DataFrame] = None,
    ) -> LiquidityMap:
        """
        Build a complete LiquidityMap from OHLCV data.

        Parameters
        ----------
        df_h1   : H1 OHLCV — primary source for session / swing / equal levels
        df_m15  : M15 OHLCV — optional; used for fine-grained equal-level detection
        df_w    : Weekly OHLCV — optional; if None, weekly high/low are derived from df_h1

        Returns
        -------
        LiquidityMap with all zones classified, scored, and sorted.
        """
        if len(df_h1) < self.swing_lookback * 2 + 2:
            logger.warning(
                f"map(): only {len(df_h1)} H1 bars — insufficient for zone detection"
            )
            return LiquidityMap(current_price=self._current_price(df_h1))

        current_price = self._current_price(df_h1)
        atr           = self._compute_atr(df_h1)
        tol_pts       = current_price * self.equal_tol   # absolute tolerance in price points

        logger.info(
            f"LiquidityMapper.map(): price={current_price:.2f}  "
            f"ATR={atr:.2f}  tol={tol_pts:.2f}pts  "
            f"bars={len(df_h1)} H1"
        )

        # ── Collect all raw zones ─────────────────────────────────────────────
        all_zones: list[LiquidityZone] = []

        all_zones += self._detect_equal_highs(df_h1,  current_price, tol_pts)
        all_zones += self._detect_equal_lows(df_h1,   current_price, tol_pts)
        all_zones += self._detect_session_levels(df_h1, current_price)
        all_zones += self._detect_weekly_levels(df_h1,  current_price, df_w)
        all_zones += self._detect_round_numbers(current_price, atr)
        all_zones += self._detect_swing_levels(df_h1,  current_price, tol_pts)

        # Fine-grain equal levels from M15 (if provided)
        if df_m15 is not None and len(df_m15) > self.swing_lookback * 2 + 2:
            all_zones += self._detect_equal_highs(df_m15, current_price, tol_pts * 0.5)
            all_zones += self._detect_equal_lows(df_m15,  current_price, tol_pts * 0.5)

        # ── Merge overlapping zones ───────────────────────────────────────────
        merged = self._merge_nearby_zones(all_zones, tol_pts)

        # ── Split into above / below ──────────────────────────────────────────
        above = sorted(
            [z for z in merged if z.price > current_price],
            key=lambda z: z.price          # ascending: nearest first
        )[:self.max_zones_per_side]

        below = sorted(
            [z for z in merged if z.price < current_price],
            key=lambda z: z.price,
            reverse=True                   # descending: nearest first
        )[:self.max_zones_per_side]

        # Tag is_above / is_below
        for z in above: z.is_above = True
        for z in below: z.is_above = False

        result = LiquidityMap(
            liquidity_above = above,
            liquidity_below = below,
            current_price   = current_price,
            scanned_bars    = len(df_h1),
            scan_time       = datetime.now(timezone.utc),
        )

        logger.info(
            f"  → {len(above)} zones above, {len(below)} zones below"
        )
        for z in above[:5]:
            logger.debug(f"    {z}")
        for z in below[:5]:
            logger.debug(f"    {z}")

        return result

    def is_entry_safe(
        self,
        entry_price: float,
        direction:   str,            # "BUY" | "SELL"
        lmap:        LiquidityMap,
        atr:         float = 0.0,
        min_distance_pts: float = 5.0,
    ) -> tuple[bool, str]:
        """
        Check whether an entry price is "safe" relative to nearby liquidity zones.

        A BUY is unsafe if heavy liquidity sits ABOVE the entry within TP range
        (it would stop the move before TP is reached).

        A SELL is unsafe if heavy liquidity sits BELOW the entry within TP range.

        Parameters
        ----------
        entry_price      : Proposed entry price.
        direction        : "BUY" or "SELL".
        lmap             : LiquidityMap from map().
        atr              : Current ATR (used to scale the proximity threshold).
        min_distance_pts : Minimum points away from a strong zone to be "safe".
                           Default 5 pts (overridden by min_proximity_atr × ATR if larger).

        Returns
        -------
        (is_safe: bool, reason: str)
        """
        min_dist = max(min_distance_pts, atr * self.min_proximity_atr)

        if direction == "BUY":
            # Check zones immediately above the entry
            close_zones = [
                z for z in lmap.liquidity_above
                if z.price - entry_price < min_dist and z.strength >= 0.70
            ]
        else:
            # Check zones immediately below the entry
            close_zones = [
                z for z in lmap.liquidity_below
                if entry_price - z.price < min_dist and z.strength >= 0.70
            ]

        if close_zones:
            strongest = max(close_zones, key=lambda z: z.strength)
            return (
                False,
                f"Entry at {entry_price:.2f} is within {min_dist:.1f} pts of "
                f"{strongest.zone_type.value} @ {strongest.price:.2f} "
                f"(str={strongest.strength:.0%}) — trade likely to reverse"
            )

        return True, "No blocking liquidity zones detected"

    def get_nearest_tp_target(
        self,
        entry_price:  float,
        direction:    str,
        lmap:         LiquidityMap,
        min_rr:       float = 1.5,
        stop_distance: float = 10.0,
    ) -> Optional[float]:
        """
        Return the nearest high-strength liquidity zone as a TP target.

        The zone must be at least `min_rr` × stop_distance away from entry
        to ensure a minimum R:R.

        Parameters
        ----------
        entry_price   : Entry price.
        direction     : "BUY" or "SELL".
        lmap          : LiquidityMap.
        min_rr        : Minimum R:R required.
        stop_distance : Distance from entry to stop loss in price points.

        Returns
        -------
        float | None — Price of the best TP zone, or None if none qualify.
        """
        min_tp_dist = stop_distance * min_rr

        if direction == "BUY":
            candidates = [
                z for z in lmap.liquidity_above
                if z.price - entry_price >= min_tp_dist and z.strength >= 0.60
            ]
            return min(candidates, key=lambda z: z.price).price if candidates else None
        else:
            candidates = [
                z for z in lmap.liquidity_below
                if entry_price - z.price >= min_tp_dist and z.strength >= 0.60
            ]
            return max(candidates, key=lambda z: z.price).price if candidates else None

    # =========================================================================
    # ZONE DETECTORS
    # =========================================================================

    def _detect_equal_highs(
        self, df: pd.DataFrame, current_price: float, tol_pts: float
    ) -> list[LiquidityZone]:
        """
        Detect clusters of swing highs within a tolerance band.

        Equal highs form when retail longs place stop-loss orders ABOVE
        a resistance level that has been tested multiple times.
        They sit ABOVE current price (buy-side liquidity).

        Algorithm:
          1. Find all swing highs in df.
          2. Cluster highs within `tol_pts` of each other.
          3. Each cluster with 2+ members is an equal-high zone.
          4. Strength scales with number of touches and recency.
        """
        swing_highs = self._find_swings(df, "high", "max")
        zones: list[LiquidityZone] = []

        if not swing_highs:
            return zones

        # Group by proximity
        prices  = sorted(swing_highs.values())
        indices = sorted(swing_highs.keys())

        used = set()
        for i, (idx_i, p_i) in enumerate(zip(indices, prices)):
            if i in used:
                continue
            cluster_prices  = [p_i]
            cluster_indices = [idx_i]
            for j, (idx_j, p_j) in enumerate(zip(indices, prices)):
                if j == i or j in used:
                    continue
                if abs(p_j - p_i) <= tol_pts:
                    cluster_prices.append(p_j)
                    cluster_indices.append(idx_j)
                    used.add(j)
            used.add(i)

            if len(cluster_prices) < 2:
                continue   # single swing high — not "equal highs"

            level_price = round(float(np.mean(cluster_prices)), 2)
            touches     = len(cluster_prices)
            age_bars    = len(df) - 1 - min(cluster_indices)
            recency     = max(cluster_indices)   # most recent touch
            recency_age = len(df) - 1 - recency

            # Strength: base + bonus for more touches, penalty for age
            base_str       = _ZONE_TYPE_BASE_STRENGTH[ZoneType.EQUAL_HIGHS]
            touch_bonus    = min((touches - 2) * 0.05, 0.10)   # +5% per extra touch
            recency_factor = max(1.0 - recency_age / self.max_zone_age, 0.30)
            strength       = round(min(base_str + touch_bonus, 1.0) * recency_factor, 3)

            zones.append(LiquidityZone(
                price       = level_price,
                zone_high   = round(level_price + tol_pts, 2),
                zone_low    = round(level_price - tol_pts, 2),
                zone_type   = ZoneType.EQUAL_HIGHS,
                strength    = strength,
                touches     = touches,
                age_bars    = age_bars,
                description = (
                    f"Equal highs cluster at {level_price:.2f} "
                    f"({touches} touches, most recent {recency_age} bars ago)"
                ),
                is_above    = level_price > current_price,
            ))

        logger.debug(f"  equal_highs: {len(zones)} cluster(s)")
        return zones

    def _detect_equal_lows(
        self, df: pd.DataFrame, current_price: float, tol_pts: float
    ) -> list[LiquidityZone]:
        """
        Detect clusters of swing lows within a tolerance band.

        Equal lows sit BELOW current price (sell-side liquidity).
        Retail shorts and long stop-losses cluster just below these levels.
        """
        swing_lows = self._find_swings(df, "low", "min")
        zones: list[LiquidityZone] = []

        if not swing_lows:
            return zones

        prices  = sorted(swing_lows.values())
        indices = sorted(swing_lows.keys())

        used = set()
        for i, (idx_i, p_i) in enumerate(zip(indices, prices)):
            if i in used:
                continue
            cluster_prices  = [p_i]
            cluster_indices = [idx_i]
            for j, (idx_j, p_j) in enumerate(zip(indices, prices)):
                if j == i or j in used:
                    continue
                if abs(p_j - p_i) <= tol_pts:
                    cluster_prices.append(p_j)
                    cluster_indices.append(idx_j)
                    used.add(j)
            used.add(i)

            if len(cluster_prices) < 2:
                continue

            level_price = round(float(np.mean(cluster_prices)), 2)
            touches     = len(cluster_prices)
            recency_age = len(df) - 1 - max(cluster_indices)
            age_bars    = len(df) - 1 - min(cluster_indices)

            base_str       = _ZONE_TYPE_BASE_STRENGTH[ZoneType.EQUAL_LOWS]
            touch_bonus    = min((touches - 2) * 0.05, 0.10)
            recency_factor = max(1.0 - recency_age / self.max_zone_age, 0.30)
            strength       = round(min(base_str + touch_bonus, 1.0) * recency_factor, 3)

            zones.append(LiquidityZone(
                price       = level_price,
                zone_high   = round(level_price + tol_pts, 2),
                zone_low    = round(level_price - tol_pts, 2),
                zone_type   = ZoneType.EQUAL_LOWS,
                strength    = strength,
                touches     = touches,
                age_bars    = age_bars,
                description = (
                    f"Equal lows cluster at {level_price:.2f} "
                    f"({touches} touches, most recent {recency_age} bars ago)"
                ),
                is_above    = level_price > current_price,
            ))

        logger.debug(f"  equal_lows: {len(zones)} cluster(s)")
        return zones

    def _detect_session_levels(
        self, df: pd.DataFrame, current_price: float
    ) -> list[LiquidityZone]:
        """
        Detect previous session (yesterday's) high and low.

        Uses UTC 00:00 as the session boundary.  Works on H1 data with a
        'time' column.  Falls back to the last 24 bars if 'time' is absent.

        Session highs/lows typically have:
          - Strong retail stop clusters just beyond the level
          - Frequent ICT "sweep-and-reverse" setups
        """
        zones: list[LiquidityZone] = []

        if "time" not in df.columns:
            # Fallback: use last 24 bars as "yesterday"
            lookback = min(24, len(df) - 1)
            prev_df  = df.iloc[-lookback - 1 : -1]
        else:
            try:
                last_time  = pd.Timestamp(df["time"].iloc[-1])
                today_utc  = last_time.normalize()           # midnight UTC
                prev_start = today_utc - timedelta(days=1)
                prev_df    = df[
                    (df["time"] >= prev_start) &
                    (df["time"] <  today_utc)
                ]
            except Exception:
                lookback = min(24, len(df) - 1)
                prev_df  = df.iloc[-lookback - 1 : -1]

        if len(prev_df) < 2:
            return zones

        session_high = float(prev_df["high"].max())
        session_low  = float(prev_df["low"].min())
        tol          = current_price * self.equal_tol

        def _age(price_col, func):
            try:
                idx = df[price_col].iloc[-len(prev_df):].values.argmax() if func == "max" else \
                      df[price_col].iloc[-len(prev_df):].values.argmin()
                return len(df) - 1 - (len(df) - len(prev_df) + idx)
            except Exception:
                return len(prev_df)

        if session_high != session_low:  # sanity
            zones.append(LiquidityZone(
                price       = session_high,
                zone_high   = round(session_high + tol, 2),
                zone_low    = round(session_high - tol, 2),
                zone_type   = ZoneType.SESSION_HIGH,
                strength    = _ZONE_TYPE_BASE_STRENGTH[ZoneType.SESSION_HIGH],
                touches     = 1,
                age_bars    = _age("high", "max"),
                description = f"Previous session high at {session_high:.2f}",
                is_above    = session_high > current_price,
            ))

            zones.append(LiquidityZone(
                price       = session_low,
                zone_high   = round(session_low + tol, 2),
                zone_low    = round(session_low - tol, 2),
                zone_type   = ZoneType.SESSION_LOW,
                strength    = _ZONE_TYPE_BASE_STRENGTH[ZoneType.SESSION_LOW],
                touches     = 1,
                age_bars    = _age("low", "min"),
                description = f"Previous session low at {session_low:.2f}",
                is_above    = session_low > current_price,
            ))

        logger.debug(
            f"  session_levels: H={session_high:.2f}  L={session_low:.2f}"
        )
        return zones

    def _detect_weekly_levels(
        self,
        df:      pd.DataFrame,
        current_price: float,
        df_w:    Optional[pd.DataFrame] = None,
    ) -> list[LiquidityZone]:
        """
        Detect the current week's running High and Low (weekly OHLC levels).

        If a pre-built weekly DataFrame is provided (df_w), uses its last
        completed week's OHLC.  Otherwise, derives the weekly high/low from
        the current week's H1 bars (filtering by ISO week number).

        Weekly levels attract position-trader stop orders and are prime
        targets for Smart Money to hunt liquidity.
        """
        zones: list[LiquidityZone] = []
        tol   = current_price * self.equal_tol

        if df_w is not None and len(df_w) >= 2:
            # Use pre-built weekly data
            last_week   = df_w.iloc[-2]    # last COMPLETED week
            weekly_high = float(last_week["high"])
            weekly_low  = float(last_week["low"])
            age_bars_h  = 0
            age_bars_l  = 0
        elif "time" in df.columns:
            try:
                last_time    = pd.Timestamp(df["time"].iloc[-1])
                week_start   = last_time - timedelta(days=last_time.weekday())
                week_start   = week_start.normalize()
                # Previous week
                prev_w_start = week_start - timedelta(weeks=1)
                prev_w_df    = df[
                    (df["time"] >= prev_w_start) &
                    (df["time"] <  week_start)
                ]
                if len(prev_w_df) < 2:
                    return zones
                weekly_high = float(prev_w_df["high"].max())
                weekly_low  = float(prev_w_df["low"].min())
                age_bars_h  = len(df) - 1 - int(prev_w_df["high"].values.argmax())
                age_bars_l  = len(df) - 1 - int(prev_w_df["low"].values.argmin())
            except Exception as exc:
                logger.debug(f"  weekly_levels: failed ({exc}) — skipping")
                return zones
        else:
            # No time column — use last 120 H1 bars (≈ 5 trading days)
            week_df     = df.iloc[-120:]
            weekly_high = float(week_df["high"].max())
            weekly_low  = float(week_df["low"].min())
            age_bars_h  = len(week_df) - 1 - int(week_df["high"].values.argmax())
            age_bars_l  = len(week_df) - 1 - int(week_df["low"].values.argmin())

        recency_h = max(1.0 - age_bars_h / self.max_zone_age, 0.40)
        recency_l = max(1.0 - age_bars_l / self.max_zone_age, 0.40)

        zones.append(LiquidityZone(
            price       = weekly_high,
            zone_high   = round(weekly_high + tol, 2),
            zone_low    = round(weekly_high - tol, 2),
            zone_type   = ZoneType.WEEKLY_HIGH,
            strength    = round(
                _ZONE_TYPE_BASE_STRENGTH[ZoneType.WEEKLY_HIGH] * recency_h, 3
            ),
            touches     = 1,
            age_bars    = age_bars_h,
            description = f"Previous week high at {weekly_high:.2f}",
            is_above    = weekly_high > current_price,
        ))

        zones.append(LiquidityZone(
            price       = weekly_low,
            zone_high   = round(weekly_low + tol, 2),
            zone_low    = round(weekly_low - tol, 2),
            zone_type   = ZoneType.WEEKLY_LOW,
            strength    = round(
                _ZONE_TYPE_BASE_STRENGTH[ZoneType.WEEKLY_LOW] * recency_l, 3
            ),
            touches     = 1,
            age_bars    = age_bars_l,
            description = f"Previous week low at {weekly_low:.2f}",
            is_above    = weekly_low > current_price,
        ))

        logger.debug(
            f"  weekly_levels: H={weekly_high:.2f}  L={weekly_low:.2f}"
        )
        return zones

    def _detect_round_numbers(
        self, current_price: float, atr: float
    ) -> list[LiquidityZone]:
        """
        Generate round-number liquidity zones above and below current price.

        For Gold (XAUUSD):
          Major rounds  : 2300, 2350, 2400, 2450 …  (every $50)
          Minor rounds  : 2325, 2375 …               (every $25)

        Retail traders cluster limit-orders and stop-losses at predictable
        psychological levels.  Smart Money targets these pools.

        Only generates zones within `atr × 10` distance from current price
        to keep the map focused on actionable nearby levels.
        """
        zones: list[LiquidityZone] = []
        tol   = current_price * self.equal_tol
        range_above = atr * 10   # only look this far above
        range_below = atr * 10   # only look this far below

        # Snap to nearest multiple of `round_step` above and below
        nearest_round_above = (
            int(current_price / self.round_step) + 1
        ) * self.round_step
        nearest_round_below = (
            int(current_price / self.round_step)
        ) * self.round_step

        # Generate rounds above
        p = nearest_round_above
        while p <= current_price + range_above:
            dist = p - current_price
            # Stronger if further from current (more distant = more order pile-up)
            recency_factor = min(dist / range_above, 1.0)
            strength = round(
                _ZONE_TYPE_BASE_STRENGTH[ZoneType.ROUND_NUMBER]
                * (0.70 + 0.30 * (1 - recency_factor)),   # nearer = slightly stronger
                3,
            )
            zones.append(LiquidityZone(
                price       = p,
                zone_high   = round(p + tol, 2),
                zone_low    = round(p - tol, 2),
                zone_type   = ZoneType.ROUND_NUMBER,
                strength    = strength,
                touches     = 0,   # synthetic — not confirmed by past price action
                age_bars    = 0,
                description = f"Round number ${p:.0f}",
                is_above    = True,
            ))
            p += self.round_step

        # Generate rounds below
        p = nearest_round_below
        while p >= current_price - range_below:
            if p == current_price:
                p -= self.round_step
                continue
            dist = current_price - p
            recency_factor = min(dist / range_below, 1.0)
            strength = round(
                _ZONE_TYPE_BASE_STRENGTH[ZoneType.ROUND_NUMBER]
                * (0.70 + 0.30 * (1 - recency_factor)),
                3,
            )
            zones.append(LiquidityZone(
                price       = p,
                zone_high   = round(p + tol, 2),
                zone_low    = round(p - tol, 2),
                zone_type   = ZoneType.ROUND_NUMBER,
                strength    = strength,
                touches     = 0,
                age_bars    = 0,
                description = f"Round number ${p:.0f}",
                is_above    = False,
            ))
            p -= self.round_step

        # Half-step rounds (25-point increments for Gold)
        half_above = nearest_round_below + self.half_step
        if half_above > current_price:
            if abs(half_above % self.round_step) > 1e-6:  # not a major round
                zones.append(LiquidityZone(
                    price       = half_above,
                    zone_high   = round(half_above + tol, 2),
                    zone_low    = round(half_above - tol, 2),
                    zone_type   = ZoneType.HALF_NUMBER,
                    strength    = _ZONE_TYPE_BASE_STRENGTH[ZoneType.HALF_NUMBER],
                    touches     = 0,
                    age_bars    = 0,
                    description = f"Half-round number ${half_above:.0f}",
                    is_above    = True,
                ))

        half_below = nearest_round_above - self.half_step
        if half_below < current_price:
            if abs(half_below % self.round_step) > 1e-6:
                zones.append(LiquidityZone(
                    price       = half_below,
                    zone_high   = round(half_below + tol, 2),
                    zone_low    = round(half_below - tol, 2),
                    zone_type   = ZoneType.HALF_NUMBER,
                    strength    = _ZONE_TYPE_BASE_STRENGTH[ZoneType.HALF_NUMBER],
                    touches     = 0,
                    age_bars    = 0,
                    description = f"Half-round number ${half_below:.0f}",
                    is_above    = False,
                ))

        logger.debug(f"  round_numbers: {len(zones)} levels")
        return zones

    def _detect_swing_levels(
        self, df: pd.DataFrame, current_price: float, tol_pts: float
    ) -> list[LiquidityZone]:
        """
        Detect significant single swing highs and lows (not yet swept).

        These are significant structure points where stop-losses sit.
        Unlike equal-highs/lows, these are individual pivotal extremes.

        Strength is determined by:
          - How many bars confirmed the swing (lookback symmetry)
          - Recency (recent swings are more actively targeted)
          - Magnitude (how far the swing extends relative to ATR)
        """
        zones:      list[LiquidityZone] = []
        atr         = self._compute_atr(df)
        swing_highs = self._find_swings(df, "high", "max")
        swing_lows  = self._find_swings(df, "low",  "min")

        n = len(df)

        for idx, price in swing_highs.items():
            age      = n - 1 - idx
            if age > self.max_zone_age:
                continue
            # Magnitude relative to ATR
            window_low = float(df["low"].iloc[
                max(0, idx - self.swing_lookback) :
                min(n, idx + self.swing_lookback + 1)
            ].min())
            swing_magnitude = (price - window_low) / atr if atr > 0 else 1.0
            recency_factor  = max(1.0 - age / self.max_zone_age, 0.20)
            magnitude_bonus = min(swing_magnitude * 0.05, 0.15)
            strength = round(
                min((_ZONE_TYPE_BASE_STRENGTH[ZoneType.SWING_HIGH] + magnitude_bonus)
                    * recency_factor, 0.90),
                3,
            )

            zones.append(LiquidityZone(
                price       = price,
                zone_high   = round(price + tol_pts, 2),
                zone_low    = round(price - tol_pts, 2),
                zone_type   = ZoneType.SWING_HIGH,
                strength    = strength,
                touches     = 1,
                age_bars    = age,
                description = f"Swing high at {price:.2f} ({age} bars ago)",
                is_above    = price > current_price,
            ))

        for idx, price in swing_lows.items():
            age      = n - 1 - idx
            if age > self.max_zone_age:
                continue
            window_high = float(df["high"].iloc[
                max(0, idx - self.swing_lookback) :
                min(n, idx + self.swing_lookback + 1)
            ].max())
            swing_magnitude = (window_high - price) / atr if atr > 0 else 1.0
            recency_factor  = max(1.0 - age / self.max_zone_age, 0.20)
            magnitude_bonus = min(swing_magnitude * 0.05, 0.15)
            strength = round(
                min((_ZONE_TYPE_BASE_STRENGTH[ZoneType.SWING_LOW] + magnitude_bonus)
                    * recency_factor, 0.90),
                3,
            )

            zones.append(LiquidityZone(
                price       = price,
                zone_high   = round(price + tol_pts, 2),
                zone_low    = round(price - tol_pts, 2),
                zone_type   = ZoneType.SWING_LOW,
                strength    = strength,
                touches     = 1,
                age_bars    = age,
                description = f"Swing low at {price:.2f} ({age} bars ago)",
                is_above    = price > current_price,
            ))

        logger.debug(
            f"  swing_levels: {sum(1 for z in zones if z.zone_type == ZoneType.SWING_HIGH)} highs  "
            f"{sum(1 for z in zones if z.zone_type == ZoneType.SWING_LOW)} lows"
        )
        return zones

    # =========================================================================
    # ZONE MERGING
    # =========================================================================

    def _merge_nearby_zones(
        self, zones: list[LiquidityZone], tol_pts: float
    ) -> list[LiquidityZone]:
        """
        Merge overlapping or nearly overlapping zones.

        When multiple zone types coincide at the same price level
        (e.g., a round number that is ALSO an equal-high), they are merged
        into a single stronger zone by:
          1. Averaging the price (weighted by strength)
          2. Combining the descriptions
          3. Boosting strength for confluence

        Zones within `tol_pts * 2` of each other are candidates for merging.
        """
        if not zones:
            return []

        # Sort by price
        zones = sorted(zones, key=lambda z: z.price)

        merged:  list[LiquidityZone] = []
        visited: set[int]            = set()

        for i, z_i in enumerate(zones):
            if i in visited:
                continue
            group = [z_i]
            visited.add(i)

            for j, z_j in enumerate(zones):
                if j in visited or j == i:
                    continue
                if abs(z_j.price - z_i.price) <= tol_pts * 2:
                    group.append(z_j)
                    visited.add(j)

            if len(group) == 1:
                merged.append(z_i)
                continue

            # ── Merge group into one zone ─────────────────────────────────────
            total_strength = sum(z.strength for z in group)
            weighted_price = sum(
                z.price * z.strength for z in group
            ) / total_strength if total_strength else z_i.price

            # Confluence bonus: multiple independent zone types meeting = stronger
            unique_types  = {z.zone_type for z in group}
            type_bonus    = min(len(unique_types) * 0.05, 0.15)
            max_strength  = max(z.strength for z in group)
            merged_strength = round(min(max_strength + type_bonus, 1.0), 3)

            # Highest-priority zone type wins for classification
            priority_order = [
                ZoneType.EQUAL_HIGHS, ZoneType.EQUAL_LOWS,
                ZoneType.WEEKLY_HIGH, ZoneType.WEEKLY_LOW,
                ZoneType.SESSION_HIGH, ZoneType.SESSION_LOW,
                ZoneType.ROUND_NUMBER,
                ZoneType.SWING_HIGH, ZoneType.SWING_LOW,
                ZoneType.HALF_NUMBER,
            ]
            dominant_type = min(
                group, key=lambda z: priority_order.index(z.zone_type)
                if z.zone_type in priority_order else 99
            ).zone_type

            total_touches = sum(z.touches for z in group)
            min_age       = min(z.age_bars for z in group)
            desc_parts    = list({z.zone_type.value for z in group})
            combined_desc = (
                f"Confluence zone at {weighted_price:.2f}: "
                + " + ".join(desc_parts)
                + f"  (strength boost from {len(unique_types)} type(s))"
            )

            merged_zone = LiquidityZone(
                price       = round(weighted_price, 2),
                zone_high   = round(max(z.zone_high for z in group), 2),
                zone_low    = round(min(z.zone_low  for z in group), 2),
                zone_type   = dominant_type,
                strength    = merged_strength,
                touches     = total_touches,
                age_bars    = min_age,
                description = combined_desc,
                is_above    = z_i.is_above,
            )
            merged.append(merged_zone)

        return merged

    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================

    def _find_swings(
        self, df: pd.DataFrame, col: str, func: str
    ) -> dict[int, float]:
        """
        Generic swing-point finder for both highs and lows.

        Parameters
        ----------
        col  : "high" or "low"
        func : "max" for highs, "min" for lows
        """
        swings: dict[int, float] = {}
        lb = self.swing_lookback
        n  = len(df)

        values = df[col].values.astype(float)

        for i in range(lb, n - lb):
            window = values[i - lb : i + lb + 1]
            val    = values[i]
            pivot  = float(np.max(window)) if func == "max" else float(np.min(window))
            if abs(val - pivot) < 1e-8:   # val is the extreme of its window
                swings[i] = val

        return swings

    @staticmethod
    def _compute_atr(df: pd.DataFrame, period: int = 14) -> float:
        """Compute Wilder ATR from an OHLCV DataFrame. Returns 0.0 on error."""
        try:
            if len(df) < period + 2:
                return 0.0
            highs  = df["high"].values.astype(float)
            lows   = df["low"].values.astype(float)
            closes = df["close"].values.astype(float)
            trs    = np.maximum(
                highs[1:]  - lows[1:],
                np.maximum(abs(highs[1:] - closes[:-1]),
                           abs(lows[1:]  - closes[:-1]))
            )
            atr = float(np.mean(trs[:period]))
            for tr in trs[period:]:
                atr = (atr * (period - 1) + tr) / period
            return round(atr, 4)
        except Exception:
            return 0.0

    @staticmethod
    def _current_price(df: pd.DataFrame) -> float:
        """Last close price from a DataFrame."""
        if len(df) == 0:
            return 0.0
        return float(df["close"].iloc[-1])
