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
from crip_x.signal.detectors.spike_detector import SpikeDetector

spike_detector = SpikeDetector()

# Base signal — normal SpO2
base = np.array([98.0, 97.8, 98.2, 97.9, 98.1] * 20)  # 100 samples

# Test 1 — clean signal, no spikes
result = spike_detector.detect(base.copy(), SignalType.SPO2)
logger.info(
    f"Spike Test 1 — detected={result.artifact_detected} | "
    f"{result.message}"
)

# Test 2 — inject obvious spike
spiked = base.copy()
spiked[50] = 245.0   # impossible SpO2 value
result = spike_detector.detect(spiked, SignalType.SPO2)
logger.info(
    f"Spike Test 2 — detected={result.artifact_detected} | "
    f"confidence={result.confidence:.2f} | "
    f"{result.message}"
)

# Test 3 — multiple spikes
multi_spike = base.copy()
multi_spike[20] = 10.0
multi_spike[50] = 245.0
multi_spike[80] = 5.0
result = spike_detector.detect(multi_spike, SignalType.SPO2)
logger.info(
    f"Spike Test 3 — detected={result.artifact_detected} | "
    f"n_spikes={result.metadata['n_zscore_spikes']} | "
    f"confidence={result.confidence:.2f}"
)

# Test 4 — subtle spike
subtle_spike = base.copy()
subtle_spike[50] = 85.0  # low but not impossible
result = spike_detector.detect(subtle_spike, SignalType.SPO2)
logger.info(
    f"Spike Test 4 (subtle) — detected={result.artifact_detected} | "
    f"confidence={result.confidence:.2f}"
)
from crip_x.signal.detectors.dropout_detector import DropoutDetector

dropout_detector = DropoutDetector()
base = np.array([98.0, 97.8, 98.2, 97.9, 98.1] * 20)

# Test 1 — clean signal
result = dropout_detector.detect(base.copy(), SignalType.SPO2)
logger.info(
    f"Dropout Test 1 — detected={result.artifact_detected} | "
    f"{result.message}"
)

# Test 2 — NaN dropout
nan_signal = base.copy().astype(float)
nan_signal[40:50] = np.nan   # 10 consecutive NaNs
result = dropout_detector.detect(nan_signal, SignalType.SPO2)
logger.info(
    f"Dropout Test 2 — detected={result.artifact_detected} | "
    f"confidence={result.confidence:.2f} | "
    f"{result.message}"
)

# Test 3 — zero burst (device reports 0 instead of NaN)
zero_signal = base.copy()
zero_signal[60:68] = 0.0    # 8 consecutive zeros
result = dropout_detector.detect(zero_signal, SignalType.SPO2)
logger.info(
    f"Dropout Test 3 — detected={result.artifact_detected} | "
    f"confidence={result.confidence:.2f} | "
    f"{result.message}"
)

# Test 4 — sudden signal loss (second half flatlines)
sudden_loss = base.copy()
sudden_loss[50:] = 0.0      # second half goes dead
result = dropout_detector.detect(sudden_loss, SignalType.SPO2)
logger.info(
    f"Dropout Test 4 — detected={result.artifact_detected} | "
    f"confidence={result.confidence:.2f} | "
    f"{result.message}"
)