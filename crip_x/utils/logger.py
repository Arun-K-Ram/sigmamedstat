"""
CRIP-X Logging

Structured logging for the entire system.
Every module imports get_logger from here.
Never use print() anywhere in the codebase.

Why structured logging over print():
- Log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Timestamps on every message
- Module-aware — tells you exactly where a log came from
- Can be redirected to files, monitoring systems, dashboards
- Can be silenced in tests without touching application code
"""

import logging
import sys
from pathlib import Path
from crip_x.utils.config import settings, ROOT_DIR


# ── Log Directory ────────────────────────────────────────────
LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)


# ── Formatters ───────────────────────────────────────────────
# What each log line looks like
CONSOLE_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)

FILE_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | "
    "%(filename)s:%(lineno)d | %(message)s"
)

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# ── Log Level ─────────────────────────────────────────────────
# In development we want to see everything (DEBUG)
# In production we only want important messages (INFO)
LOG_LEVEL = logging.DEBUG if settings.debug else logging.INFO


# ── Root Logger Setup ────────────────────────────────────────
def _setup_root_logger() -> None:
    """
    Configure the root logger once at startup.
    Called automatically when this module is first imported.
    All child loggers (via get_logger) inherit this config.
    """
    root = logging.getLogger("crip_x")
    root.setLevel(LOG_LEVEL)

    # Prevent duplicate handlers if module is reimported
    if root.handlers:
        return

    # ── Console Handler ──────────────────────────────────────
    # Errors go to stderr, everything else to stdout
    # This matters when you redirect logs in production
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(LOG_LEVEL)
    console_handler.setFormatter(
        logging.Formatter(CONSOLE_FORMAT, datefmt=DATE_FORMAT)
    )

    # ── File Handler ─────────────────────────────────────────
    # Rotating file — max 5MB per file, keep last 3 files
    # Prevents logs from eating your disk
    from logging.handlers import RotatingFileHandler

    file_handler = RotatingFileHandler(
        LOG_DIR / "crip_x.log",
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(LOG_LEVEL)
    file_handler.setFormatter(
        logging.Formatter(FILE_FORMAT, datefmt=DATE_FORMAT)
    )

    root.addHandler(console_handler)
    root.addHandler(file_handler)


# ── Public Interface ─────────────────────────────────────────
def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger for a module.

    Usage in any module:
        from crip_x.utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Signal loaded")
        logger.warning("Low reliability score: 34")
        logger.error("Failed to connect to PhysioNet")

    The __name__ argument automatically uses the module's
    fully qualified name e.g. crip_x.signal.detectors.spike
    which tells you exactly where the log came from.
    """
    return logging.getLogger(f"crip_x.{name}")


# ── Initialize on import ─────────────────────────────────────
_setup_root_logger()