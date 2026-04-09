"""
utils/helpers.py
----------------
General-purpose utility functions used across the bot.

Responsibilities:
- Time and timezone conversion helpers (UTC ↔ local, UTC ↔ broker time)
- Price formatting helpers (round to pip precision, format as string)
- Retry decorator for flaky network / API calls
- Safe division helper (avoid ZeroDivisionError)
- DataFrame validation utilities (check required columns exist)
- Simple in-memory rate limiter (calls per minute)

Dependencies:
    datetime, functools, time, pandas
"""

from __future__ import annotations
import functools
import time
from datetime import datetime, timezone
from typing import Any, Callable, Optional
import pandas as pd


# ── Price Utilities ───────────────────────────────────────────────────────────

def round_price(price: float, digits: int = 2) -> float:
    """Round a price to the given number of decimal places."""
    return round(price, digits)


def format_price(price: float, digits: int = 2) -> str:
    """Format a price as a string with consistent decimal places."""
    return f"{price:.{digits}f}"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide safely, returning `default` if denominator is zero."""
    return numerator / denominator if denominator != 0 else default


# ── Time Utilities ────────────────────────────────────────────────────────────

def utc_now() -> datetime:
    """Return the current time as a UTC-aware datetime object."""
    return datetime.now(timezone.utc)


def to_utc(dt: datetime) -> datetime:
    """Convert a naive or local datetime to UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


# ── DataFrame Validators ──────────────────────────────────────────────────────

def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Raise ValueError if any expected columns are missing from df."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")


# ── Retry Decorator ───────────────────────────────────────────────────────────

def retry(max_attempts: int = 3, delay_seconds: float = 1.0, exceptions: tuple = (Exception,)):
    """
    Decorator: retry a function up to `max_attempts` times on specified exceptions.

    Usage:
        @retry(max_attempts=3, delay_seconds=2.0, exceptions=(ConnectionError,))
        def fetch_data(): ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Optional[Exception] = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt < max_attempts:
                        time.sleep(delay_seconds * attempt)
            raise last_exc  # type: ignore[misc]
        return wrapper
    return decorator
