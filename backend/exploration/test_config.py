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

from crip_x.signal.detectors.flatline_detector import FlatlineDetector

detector = FlatlineDetector()

# Test 1 — obvious flatline
flatline = np.full(100, 98.0)  # 100 samples all exactly 98.0
result = detector.detect(flatline, SignalType.SPO2)
logger.info(
    f"Flatline Test 1 — detected={result.artifact_detected} | "
    f"confidence={result.confidence:.2f} | {result.message}"
)

# Test 2 — real signal with natural variance
real_signal = np.array([
    98.0, 97.8, 98.2, 97.9, 98.1,
    97.7, 98.3, 97.6, 98.4, 97.8,
    98.0, 97.9, 98.1, 97.8, 98.2,
    97.7, 98.3, 97.9, 98.0, 97.8,
] * 5)  # 100 samples of realistic SpO2
result = detector.detect(real_signal, SignalType.SPO2)
logger.info(
    f"Flatline Test 2 — detected={result.artifact_detected} | "
    f"confidence={result.confidence:.2f} | {result.message}"
)

# Test 3 — subtle flatline (nearly flat, not perfect)
subtle_flatline = np.full(100, 98.0)
subtle_flatline += np.random.normal(0, 0.001, 100)  # tiny noise
result = detector.detect(subtle_flatline, SignalType.SPO2)
logger.info(
    f"Flatline Test 3 — detected={result.artifact_detected} | "
    f"confidence={result.confidence:.2f} | {result.message}"
)

# Test 4 — high sensitivity catches subtle flatline
sensitive_detector = FlatlineDetector(sensitivity=2.0)
result = sensitive_detector.detect(subtle_flatline, SignalType.SPO2)
logger.info(
    f"Flatline Test 4 (high sensitivity) — "
    f"detected={result.artifact_detected} | "
    f"confidence={result.confidence:.2f}"
)