"""config package — exposes Settings singleton for the entire bot."""
from .settings import Settings, get_settings

__all__ = ["Settings", "get_settings"]
