import numpy as np
from crip_x.utils.logger import get_logger
from crip_x.utils.validators import SignalValidator, SignalType
from crip_x.signal.detectors.base_detector import (
    BaseDetector, DetectionResult, ArtifactType
)

logger = get_logger(__name__)

# ── Validator Tests ──────────────────────────────────────────
validator = SignalValidator()

clean_spo2 = np.array([98.0, 97.5, 98.2, 97.8, 98.1])
result = validator.validate(clean_spo2, SignalType.SPO2)
logger.info(f"Test 1 - Clean SpO2: valid={result.is_valid}")

bad_spo2 = np.array([98.0, 450.0, 97.5, -10.0, 98.1])
result = validator.validate(bad_spo2, SignalType.SPO2)
logger.info(f"Test 2 - Bad SpO2: valid={result.is_valid} | {result.message}")

dropout_signal = np.array([98.0, np.nan, np.nan, np.nan, 97.5])
result = validator.validate(dropout_signal, SignalType.SPO2)
logger.info(f"Test 3 - Dropout: valid={result.is_valid} | {result.message}")

empty = np.array([])
result = validator.validate(empty, SignalType.HEART_RATE)
logger.info(f"Test 4 - Empty: valid={result.is_valid} | {result.message}")

# ── Base Detector Test ───────────────────────────────────────
try:
    detector = BaseDetector()
    logger.error("Should not reach here")
except TypeError as e:
    logger.info(f"Abstract class correctly blocked: {e}")

logger.info("All tests complete")