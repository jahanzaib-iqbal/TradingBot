"""data package — MT5 and news data providers."""
from .mt5_data import (
    MT5DataProvider,
    MT5Error,
    MT5NotAvailableError,
    MT5ConnectionError,
    MT5DataError,
    MT5SymbolError,
    TickPrice,
    AccountInfo,
)
from .news_data import NewsDataProvider, NewsEvent, NewsRiskAssessment

__all__ = [
    # Provider classes
    "MT5DataProvider",
    "NewsDataProvider",
    # News dataclasses
    "NewsEvent",
    "NewsRiskAssessment",
    # Dataclasses
    "TickPrice",
    "AccountInfo",
    # Exceptions
    "MT5Error",
    "MT5NotAvailableError",
    "MT5ConnectionError",
    "MT5DataError",
    "MT5SymbolError",
]
