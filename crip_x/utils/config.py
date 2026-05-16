"""
CRIP-X Configuration Management

Single source of truth for all system configuration.
Reads from environment variables with sensible defaults.
Never import configuration values directly in other modules —
always import from here.
"""

from pathlib import Path
from pydantic_settings import BaseSettings
from functools import lru_cache


# ── Project Root ────────────────────────────────────────────
# Resolves to the crip-x/ directory regardless of where
# the code is run from. Use this for all file path operations.
ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "crip_x" / "models" / "saved"


# ── Settings ────────────────────────────────────────────────
class Settings(BaseSettings):
    """
    All settings are read from environment variables.
    Falls back to defaults if not set.
    Pydantic validates types automatically -if API_PORT
    is not a valid int, it fails loudly at startup, not
    silently at runtime.
    """

    # PhysioNet
    physionet_username: str = ""
    physionet_password: str = ""

    # Database
    database_url: str = f"sqlite:///{ROOT_DIR}/crip_x.db"

    # API
    api_host: str = "127.0.0.1"
    api_port: int = 8000

    # Environment
    environment: str = "development"
    debug: bool = True

    # Signal Processing
    # Default sampling frequency for PhysioNet Challenge 2015
    default_sampling_frequency: int = 250
    # Window size in seconds for signal analysis
    signal_window_seconds: int = 10
    # How much windows overlap -50% overlap is standard
    window_overlap: float = 0.5

    # Reliability Scoring
    # Below this threshold we flag as unreliable
    reliability_threshold: float = 50.0
    # Below this we consider it a critical failure
    critical_threshold: float = 25.0

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        # Allows PHYSIONET_USERNAME in .env to map to
        # physionet_username in the class
        case_sensitive = False


# ── Singleton Pattern ────────────────────────────────────────
# lru_cache means this is only instantiated once per
# application lifecycle. Every module gets the same
# Settings object -not a new one each time.
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


# ── Convenience Export ───────────────────────────────────────
settings = get_settings()