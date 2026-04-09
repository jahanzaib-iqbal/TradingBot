"""
risk/position_sizing.py
========================
Position sizing calculator for XAUUSD (Gold) trading.

Gold Contract Specification
----------------------------
Symbol      : XAUUSD  (may vary by broker: GOLD, XAUUSDm, XAUUSD.)
Contract    : 1 standard lot = 100 troy ounces
Point size  : 0.01 USD  (minimum price increment)
Point value : $1.00 USD per standard lot
              (100 oz × $0.01 = $1.00 per lot per 0.01 move)
Lot step    : 0.01 (micro-lots)  — most brokers
Min lot     : 0.01 standard lots (1 oz)
Max lot     : varies by broker / account type

Core Formula
------------
    stop_distance  = |entry_price − stop_loss|     (in USD, raw price units)
    risk_amount    = account_balance × risk_pct / 100
    lot_size       = risk_amount
                     ─────────────────────────────────────────────────
                     stop_distance × contract_size × point_value_per_oz

Simplified (using the $1/point/lot convention):
    lot_size       = risk_amount / (stop_distance_points × point_value_per_lot)

Where:
    stop_distance_points = stop_distance / point_size
    point_value_per_lot  = point_size × contract_size = 0.01 × 100 = $1.00

Therefore the formula reduces to:
    lot_size = risk_amount / (stop_distance × contract_size)
             = risk_amount / (stop_distance × 100)

Example
-------
    Balance = $10,000 | Risk = 1% | Entry = $2,400.00 | SL = $2,390.00
    risk_amount    = $10,000 × 1% = $100
    stop_distance  = $10.00
    lot_size       = $100 / ($10 × 100) = 0.10 lots
    monetary_risk  = 0.10 × 100 oz × $10 = $100  (verified)

Public API
----------
    sizer  = PositionSizer(settings)
    result = sizer.calculate(
        entry_price     = 2400.00,
        stop_loss       = 2390.00,
        take_profit_1   = 2420.00,
        take_profit_2   = 2440.00,
        account_balance = 10_000.00,
        risk_pct        = 1.0,
    )
    print(result)          # Human-readable summary
    print(result.to_dict()) # Signal-ready dict

Dependencies
------------
    config.settings.Settings, utils.logger
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from config.settings import Settings
from utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# XAUUSD contract constants
# ─────────────────────────────────────────────────────────────────────────────

# 1 standard lot of XAUUSD = 100 troy ounces
XAUUSD_CONTRACT_SIZE: float = 100.0

# Minimum tick / point size (broker-standard for Gold)
XAUUSD_POINT_SIZE: float = 0.01

# Dollar value per standard lot per 1-point move = 100 oz × $0.01 = $1.00
XAUUSD_POINT_VALUE_PER_LOT: float = XAUUSD_CONTRACT_SIZE * XAUUSD_POINT_SIZE  # $1.00

# Lot rounding step (0.01 = micro-lot precision, industry standard)
XAUUSD_LOT_STEP: float = 0.01


# ─────────────────────────────────────────────────────────────────────────────
# Custom exceptions
# ─────────────────────────────────────────────────────────────────────────────

class PositionSizingError(Exception):
    """Raised when inputs are invalid or the computed lot is outside safe bounds."""


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PositionSize:
    """
    Full position sizing result for a single trade.

    Attributes
    ----------
    lot_size            : recommended lot size (rounded to lot_step precision)
    risk_amount_usd     : dollar value at risk if SL is hit  (verified)
    actual_risk_pct     : actual risk % after lot rounding
    stop_distance_pts   : SL distance in price points
    stop_distance_usd   : SL distance in raw USD (price units)
    tp1_distance_usd    : TP1 distance in USD (None if not provided)
    tp2_distance_usd    : TP2 distance in USD (None if not provided)
    rr_ratio_tp1        : reward-to-risk ratio to TP1  (None if not provided)
    rr_ratio_tp2        : reward-to-risk ratio to TP2  (None if not provided)
    rr_ratio            : primary R:R (tp2 if available, else tp1, else None)
    lot_capped          : True if lot was clamped to MAX_LOT_SIZE
    lot_floored         : True if lot was raised to MIN_LOT_SIZE
    monetary_value_1pt  : dollar P&L for 1-point move at this lot size
    entry_price         : input entry price
    stop_loss           : input stop loss price
    take_profit_1       : input TP1 (or None)
    take_profit_2       : input TP2 (or None)
    account_balance     : input account balance
    risk_pct            : input risk percentage
    direction           : "BUY" or "SELL"
    """
    # Core outputs
    lot_size:           float
    risk_amount_usd:    float
    actual_risk_pct:    float
    stop_distance_pts:  float   # in price points (raw / point_size)
    stop_distance_usd:  float   # in price units

    # R:R
    tp1_distance_usd: Optional[float] = None
    tp2_distance_usd: Optional[float] = None
    rr_ratio_tp1:     Optional[float] = None
    rr_ratio_tp2:     Optional[float] = None
    rr_ratio:         Optional[float] = None   # primary (best available)

    # Lot limit flags
    lot_capped:   bool = False
    lot_floored:  bool = False

    # Per-point P&L at this lot size
    monetary_value_1pt: float = 0.0

    # Echo of inputs (for traceability)
    entry_price:     float = 0.0
    stop_loss:       float = 0.0
    take_profit_1:   Optional[float] = None
    take_profit_2:   Optional[float] = None
    account_balance: float = 0.0
    risk_pct:        float = 0.0
    direction:       str   = ""

    # ── Output helpers ────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """
        Return a plain dict suitable for embedding in signal messages.

        Example
        -------
            {
                "lot_size":          0.10,
                "risk_amount_usd":   100.00,
                "actual_risk_pct":   1.00,
                "stop_distance_usd": 10.00,
                "stop_distance_pts": 1000,
                "rr_ratio":          2.0,
                "rr_ratio_tp1":      1.0,
                "rr_ratio_tp2":      2.0,
                "monetary_value_1pt":0.10,
                "direction":         "BUY",
                "lot_capped":        False,
                "lot_floored":       False,
                ...
            }
        """
        def _r(v, decimals=2):
            return round(v, decimals) if v is not None else None

        return {
            # Core results
            "lot_size":           _r(self.lot_size, 2),
            "risk_amount_usd":    _r(self.risk_amount_usd, 2),
            "actual_risk_pct":    _r(self.actual_risk_pct, 4),
            "stop_distance_usd":  _r(self.stop_distance_usd, 2),
            "stop_distance_pts":  _r(self.stop_distance_pts, 1),
            # R:R
            "rr_ratio":           _r(self.rr_ratio, 2),
            "rr_ratio_tp1":       _r(self.rr_ratio_tp1, 2),
            "rr_ratio_tp2":       _r(self.rr_ratio_tp2, 2),
            # Trade parameters (echoed for downstream consumers)
            "entry_price":        _r(self.entry_price, 2),
            "stop_loss":          _r(self.stop_loss, 2),
            "take_profit_1":      _r(self.take_profit_1, 2),
            "take_profit_2":      _r(self.take_profit_2, 2),
            "direction":          self.direction,
            # Per-point value (useful for Telegram display)
            "monetary_value_1pt": _r(self.monetary_value_1pt, 2),
            # Quality / safety flags
            "lot_capped":   self.lot_capped,
            "lot_floored":  self.lot_floored,
            "account_balance": _r(self.account_balance, 2),
            "risk_pct":        _r(self.risk_pct, 2),
        }

    def is_valid_rr(self, min_rr: float = 2.0) -> bool:
        """Return True if the primary R:R ratio meets the minimum threshold."""
        if self.rr_ratio is None:
            return False
        return self.rr_ratio >= min_rr

    def __str__(self) -> str:
        rr_str = f"{self.rr_ratio:.2f}" if self.rr_ratio else "N/A"
        flags  = []
        if self.lot_capped:   flags.append("CAPPED")
        if self.lot_floored:  flags.append("FLOORED")
        flag_str = f"  [{', '.join(flags)}]" if flags else ""
        return (
            f"PositionSize("
            f"dir={self.direction}, "
            f"lots={self.lot_size:.2f}, "
            f"risk=${self.risk_amount_usd:.2f}({self.actual_risk_pct:.2f}%), "
            f"SL_dist=${self.stop_distance_usd:.2f}({self.stop_distance_pts:.0f}pts), "
            f"R:R={rr_str}, "
            f"$1pt=${self.monetary_value_1pt:.2f}"
            f"{flag_str}"
            f")"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Sizer class
# ─────────────────────────────────────────────────────────────────────────────

class PositionSizer:
    """
    Fixed-fractional position sizer calibrated for XAUUSD (Gold).

    Key formula
    -----------
        lot_size = risk_amount / (stop_distance × contract_size)

    Where:
        risk_amount    = balance × risk_pct / 100
        stop_distance  = |entry - stop_loss|  (in price units, e.g. $10)
        contract_size  = 100 (oz per standard lot)

    The result is then:
        1. Rounded DOWN to the nearest lot_step (0.01) — conservative
        2. Clamped to [MIN_LOT_SIZE, MAX_LOT_SIZE] from Settings

    Usage
    -----
        sizer  = PositionSizer(settings)
        result = sizer.calculate(
            entry_price=2400, stop_loss=2390,
            take_profit_1=2420, take_profit_2=2440,
        )

    Or with explicit overrides (bypasses Settings defaults):
        result = sizer.calculate(
            entry_price=2400, stop_loss=2390,
            account_balance=5000, risk_pct=0.5,
        )
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        """
        Parameters
        ----------
        settings : Settings, optional
            If provided, default risk, lot limits, and point size are read
            from Settings.  All can be overridden per-call.
            If None, uses class-level XAUUSD defaults.
        """
        if settings is not None:
            self.default_balance  = settings.ACCOUNT_BALANCE
            self.default_risk_pct = settings.RISK_PER_TRADE_PCT
            self.min_lot          = settings.MIN_LOT_SIZE
            self.max_lot          = settings.MAX_LOT_SIZE
            self.min_rr           = settings.MIN_RR_RATIO
            self.point_size       = settings.SYMBOL_POINT
            self.digits           = settings.SYMBOL_DIGITS
        else:
            self.default_balance  = 10_000.0
            self.default_risk_pct = 1.0
            self.min_lot          = 0.01
            self.max_lot          = 5.0
            self.min_rr           = 2.0
            self.point_size       = XAUUSD_POINT_SIZE
            self.digits           = 2

        # Gold contract constants (fixed — do not override)
        self.contract_size         = XAUUSD_CONTRACT_SIZE        # 100 oz/lot
        self.point_value_per_lot   = XAUUSD_POINT_VALUE_PER_LOT  # $1.00
        self.lot_step              = XAUUSD_LOT_STEP             # 0.01

    # =========================================================================
    # PRIMARY PUBLIC METHOD
    # =========================================================================

    def calculate(
        self,
        entry_price:     float,
        stop_loss:       float,
        take_profit_1:   Optional[float] = None,
        take_profit_2:   Optional[float] = None,
        account_balance: Optional[float] = None,
        risk_pct:        Optional[float] = None,
    ) -> PositionSize:
        """
        Compute lot size and full risk profile for a XAUUSD trade.

        Parameters
        ----------
        entry_price : float
            Intended entry price (e.g. 2400.00).
        stop_loss : float
            Stop-loss price.  May be above or below entry;
            the direction is inferred automatically.
        take_profit_1 : float, optional
            First take-profit target (partial close or TP1).
        take_profit_2 : float, optional
            Second / final take-profit target.
        account_balance : float, optional
            Override for Settings.ACCOUNT_BALANCE.
        risk_pct : float, optional
            Override for Settings.RISK_PER_TRADE_PCT (expressed as percent,
            e.g. 1.0 = 1%). If None, the Settings value is used.

        Returns
        -------
        PositionSize
            Full sizing result.  Access .to_dict() for signal emission.

        Raises
        ------
        PositionSizingError
            If inputs are invalid (e.g. entry == stop, negative balance).
        """
        # ── Resolve defaults ──────────────────────────────────────────────
        balance  = account_balance if account_balance is not None else self.default_balance
        risk_pct_val = risk_pct if risk_pct is not None else self.default_risk_pct

        # ── Validate inputs ───────────────────────────────────────────────
        self._validate_inputs(
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            balance=balance,
            risk_pct=risk_pct_val,
        )

        # ── Infer trade direction ─────────────────────────────────────────
        #
        # If stop_loss < entry_price  →  BUY  (we expect price to rise)
        # If stop_loss > entry_price  →  SELL (we expect price to fall)
        #
        direction = "BUY" if stop_loss < entry_price else "SELL"

        # ── Step 1 — Dollar amount at risk ────────────────────────────────
        #
        # Fixed-fractional rule: risk a fixed % of the account on each trade.
        # This keeps drawdown proportional and compounds with account growth.
        #
        risk_amount_usd = balance * (risk_pct_val / 100.0)

        # ── Step 2 — Stop-loss distance ───────────────────────────────────
        #
        # The distance in raw price units (same as USD for XAUUSD).
        # For BUY:  entry=2400, SL=2390 → distance = 10.00 USD
        # For SELL: entry=2400, SL=2410 → distance = 10.00 USD
        #
        stop_distance_usd = abs(entry_price - stop_loss)

        # Convert to "points" (price units / point_size)
        # For XAUUSD: 1 point = $0.01, so $10 distance = 1000 points
        stop_distance_pts = stop_distance_usd / self.point_size

        # ── Step 3 — Raw lot size ─────────────────────────────────────────
        #
        # Full derivation:
        #   monetary_risk_per_lot = stop_distance_usd × contract_size
        #   lot_size = risk_amount_usd / monetary_risk_per_lot
        #
        # For XAUUSD:
        #   monetary_risk_per_lot = stop_distance × 100
        #   lot_size = risk_amount / (stop_distance × 100)
        #
        monetary_risk_per_lot = stop_distance_usd * self.contract_size
        raw_lot_size          = risk_amount_usd / monetary_risk_per_lot

        # ── Step 4 — Round down to lot_step ──────────────────────────────
        #
        # Always round DOWN (floor) not nearest — to never exceed the
        # intended risk budget.  Round to the nearest lot_step (0.01).
        #
        lot_size = self._floor_to_step(raw_lot_size, self.lot_step)

        # ── Step 5 — Clamp to broker limits ──────────────────────────────
        lot_capped  = False
        lot_floored = False

        if lot_size > self.max_lot:
            lot_size   = self.max_lot
            lot_capped = True
            logger.warning(
                f"Lot size {raw_lot_size:.4f} capped to MAX_LOT {self.max_lot:.2f}. "
                f"Position risk is higher than intended."
            )

        if lot_size < self.min_lot:
            lot_size    = self.min_lot
            lot_floored = True
            logger.warning(
                f"Lot size {raw_lot_size:.4f} raised to MIN_LOT {self.min_lot:.2f}. "
                f"Risk is proportionally higher due to minimum constraint."
            )

        # ── Step 6 — Verify actual monetary risk after rounding ───────────
        #
        # Recalculate the actual risk with the rounded lot to catch any
        # discrepancy caused by flooring or clamping.
        #
        actual_risk_usd = lot_size * stop_distance_usd * self.contract_size
        actual_risk_pct = (actual_risk_usd / balance * 100.0) if balance > 0 else 0.0

        # ── Step 7 — Per-point monetary value ─────────────────────────────
        #
        # How much money the position earns/loses for each 1-point move.
        # For XAUUSD: 1 point = $0.01; 1 lot = 100 oz
        # monetary_value_1pt = lot_size × contract_size × point_size
        #                    = lot_size × 100 × 0.01
        #                    = lot_size × 1.00          ($1 per lot per point)
        #
        monetary_value_1pt = lot_size * self.contract_size * self.point_size

        # ── Step 8 — Reward-to-risk ratios ───────────────────────────────
        rr_tp1 = self._compute_rr(entry_price, stop_loss, take_profit_1)
        rr_tp2 = self._compute_rr(entry_price, stop_loss, take_profit_2)

        # Primary R:R = best available target (TP2 preferred, then TP1)
        primary_rr = rr_tp2 if rr_tp2 is not None else rr_tp1

        # ── Step 9 — TP distances ─────────────────────────────────────────
        tp1_dist = abs(take_profit_1 - entry_price) if take_profit_1 is not None else None
        tp2_dist = abs(take_profit_2 - entry_price) if take_profit_2 is not None else None

        # ── Build result ──────────────────────────────────────────────────
        result = PositionSize(
            lot_size           = lot_size,
            risk_amount_usd    = round(actual_risk_usd, 2),
            actual_risk_pct    = round(actual_risk_pct, 4),
            stop_distance_pts  = round(stop_distance_pts, 1),
            stop_distance_usd  = round(stop_distance_usd, 2),
            tp1_distance_usd   = round(tp1_dist, 2) if tp1_dist else None,
            tp2_distance_usd   = round(tp2_dist, 2) if tp2_dist else None,
            rr_ratio_tp1       = rr_tp1,
            rr_ratio_tp2       = rr_tp2,
            rr_ratio           = primary_rr,
            lot_capped         = lot_capped,
            lot_floored        = lot_floored,
            monetary_value_1pt = round(monetary_value_1pt, 2),
            entry_price        = entry_price,
            stop_loss          = stop_loss,
            take_profit_1      = take_profit_1,
            take_profit_2      = take_profit_2,
            account_balance    = balance,
            risk_pct           = risk_pct_val,
            direction          = direction,
        )

        logger.info(
            f"PositionSize calculated: {result}  "
            f"(raw_lot={raw_lot_size:.4f}, "
            f"risk_usd=${risk_amount_usd:.2f}, "
            f"stop=${stop_distance_usd:.2f})"
        )

        return result

    # =========================================================================
    # INDIVIDUAL UTILITY METHODS
    # =========================================================================

    def risk_amount(
        self,
        account_balance: Optional[float] = None,
        risk_pct:        Optional[float] = None,
    ) -> float:
        """
        Return the dollar amount at risk for a given balance and risk %.

        Parameters
        ----------
        account_balance : float, optional  — defaults to Settings.ACCOUNT_BALANCE
        risk_pct        : float, optional  — defaults to Settings.RISK_PER_TRADE_PCT

        Returns
        -------
        float — dollar risk amount
        """
        balance  = account_balance if account_balance is not None else self.default_balance
        risk     = risk_pct if risk_pct is not None else self.default_risk_pct
        return round(balance * risk / 100.0, 2)

    def lot_size_only(
        self,
        entry_price:     float,
        stop_loss:       float,
        account_balance: Optional[float] = None,
        risk_pct:        Optional[float] = None,
    ) -> float:
        """
        Quick helper — return only the lot size without the full PositionSize object.

        Useful when you just need the lot number for display in signal messages.
        """
        return self.calculate(
            entry_price     = entry_price,
            stop_loss       = stop_loss,
            account_balance = account_balance,
            risk_pct        = risk_pct,
        ).lot_size

    def rr_ratio(
        self,
        entry_price:  float,
        stop_loss:    float,
        take_profit:  float,
    ) -> float:
        """
        Standalone R:R ratio calculator.

        Parameters
        ----------
        entry_price  : trade entry price
        stop_loss    : stop-loss price
        take_profit  : take-profit price

        Returns
        -------
        float — reward / risk ratio  (e.g. 2.0 = 2:1 R:R)

        Raises
        ------
        PositionSizingError — if stop distance is zero
        """
        rr = self._compute_rr(entry_price, stop_loss, take_profit)
        if rr is None:
            raise PositionSizingError(
                "Cannot compute R:R — stop distance is zero "
                f"(entry={entry_price}, stop={stop_loss})"
            )
        return rr

    def monetary_value_per_point(self, lot_size: float) -> float:
        """
        Return the dollar P&L per 1-point move for a given lot size.

        For XAUUSD:
            value = lot_size × contract_size × point_size
                  = lot_size × 100 × 0.01
                  = lot_size × $1.00

        Parameters
        ----------
        lot_size : float

        Returns
        -------
        float — dollars per point
        """
        return round(lot_size * self.contract_size * self.point_size, 4)

    def max_lots_for_balance(
        self,
        entry_price:     float,
        stop_loss:       float,
        account_balance: Optional[float] = None,
        risk_pct:        Optional[float] = None,
    ) -> dict:
        """
        Return a breakdown of the maximum affordable lot size alongside
        safe recommendations at different risk levels.

        Returns
        -------
        dict
            {
                "conservative_0.5pct": float,
                "standard_1pct":       float,
                "aggressive_2pct":     float,
                "max_allowed":         float,   # Settings.MAX_LOT_SIZE
            }
        """
        def _lot(pct: float) -> float:
            try:
                return self.calculate(
                    entry_price     = entry_price,
                    stop_loss       = stop_loss,
                    account_balance = account_balance,
                    risk_pct        = pct,
                ).lot_size
            except PositionSizingError:
                return 0.0

        return {
            "conservative_0.5pct": _lot(0.5),
            "standard_1pct":       _lot(1.0),
            "aggressive_2pct":     _lot(2.0),
            "max_allowed":         self.max_lot,
        }

    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================

    def _compute_rr(
        self,
        entry: float,
        stop:  float,
        target: Optional[float],
    ) -> Optional[float]:
        """
        Compute the reward-to-risk ratio.

        R:R = |target − entry| / |stop − entry|

        Returns None if target is not provided or if stop distance is zero.
        """
        if target is None:
            return None

        stop_dist   = abs(entry - stop)
        reward_dist = abs(target - entry)

        if stop_dist == 0.0:
            return None

        return round(reward_dist / stop_dist, 2)

    @staticmethod
    def _floor_to_step(value: float, step: float) -> float:
        """
        Round a value DOWN to the nearest multiple of `step`.

        Examples:
            _floor_to_step(0.157, 0.01)  →  0.15
            _floor_to_step(0.999, 0.01)  →  0.99
            _floor_to_step(1.000, 0.01)  →  1.00
        """
        if step <= 0:
            return value
        # Use integer arithmetic to avoid floating-point drift
        factor = round(1.0 / step)
        return math.floor(value * factor) / factor

    def _validate_inputs(
        self,
        entry_price:  float,
        stop_loss:    float,
        take_profit_1: Optional[float],
        take_profit_2: Optional[float],
        balance:      float,
        risk_pct:     float,
    ) -> None:
        """
        Raise PositionSizingError with a descriptive message if any input
        is logically invalid.
        """
        errors: list[str] = []

        if entry_price <= 0:
            errors.append(f"entry_price must be positive; got {entry_price}")

        if stop_loss <= 0:
            errors.append(f"stop_loss must be positive; got {stop_loss}")

        if entry_price == stop_loss:
            errors.append(
                f"entry_price ({entry_price}) cannot equal stop_loss ({stop_loss}) "
                "— stop distance is zero, lot size would be infinite."
            )

        if balance <= 0:
            errors.append(f"account_balance must be positive; got {balance}")

        if risk_pct <= 0 or risk_pct > 100:
            errors.append(
                f"risk_pct must be in range (0, 100]; got {risk_pct}"
            )

        # TP direction validation — TP must be on the profit side
        direction = "BUY" if stop_loss < entry_price else "SELL"

        if take_profit_1 is not None:
            if direction == "BUY" and take_profit_1 <= entry_price:
                errors.append(
                    f"BUY trade: take_profit_1 ({take_profit_1}) must be above "
                    f"entry_price ({entry_price})"
                )
            if direction == "SELL" and take_profit_1 >= entry_price:
                errors.append(
                    f"SELL trade: take_profit_1 ({take_profit_1}) must be below "
                    f"entry_price ({entry_price})"
                )

        if take_profit_2 is not None:
            if direction == "BUY" and take_profit_2 <= entry_price:
                errors.append(
                    f"BUY trade: take_profit_2 ({take_profit_2}) must be above "
                    f"entry_price ({entry_price})"
                )
            if direction == "SELL" and take_profit_2 >= entry_price:
                errors.append(
                    f"SELL trade: take_profit_2 ({take_profit_2}) must be below "
                    f"entry_price ({entry_price})"
                )

        if errors:
            raise PositionSizingError(
                "PositionSizer input validation failed:\n  " + "\n  ".join(errors)
            )

    # =========================================================================
    # REPR
    # =========================================================================

    def __repr__(self) -> str:
        return (
            f"PositionSizer("
            f"balance=${self.default_balance:,.0f}, "
            f"risk={self.default_risk_pct}%, "
            f"lots=[{self.min_lot}, {self.max_lot}], "
            f"contract={self.contract_size}oz"
            f")"
        )
