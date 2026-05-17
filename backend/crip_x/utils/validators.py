"""
CRIP-X Signal Validators

Physiological boundary validation for medical device signals.
This runs BEFORE any ML or statistical detection.

Why this layer exists:
    A spike detector shouldn't waste compute deciding whether
    an SpO2 of 450% is an artifact. That's not a spike -
    that's a completely invalid reading. Validators catch
    physically impossible values immediately.

    This is also your first line of regulatory defensibility.
    IEC 62304 requires software to handle invalid inputs
    gracefully. This module is that guarantee.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import numpy as np
from crip_x.utils.logger import get_logger

logger = get_logger(__name__)


# ── Signal Types ─────────────────────────────────────────────
class SignalType(Enum):
    """
    Supported medical device signal types.
    Extend this as CRIP-X supports more devices.
    """
    ECG = "ecg"
    SPO2 = "spo2"
    HEART_RATE = "heart_rate"
    RESPIRATORY_RATE = "respiratory_rate"
    ABP_SYSTOLIC = "abp_systolic"
    ABP_DIASTOLIC = "abp_diastolic"
    TEMPERATURE = "temperature"


# ── Physiological Boundaries ─────────────────────────────────
@dataclass(frozen=True)
class PhysiologicalBounds:
    """
    Valid physiological range for a signal type.
    Values outside these ranges are physically impossible
    in a living human - not just abnormal, impossible.

    frozen=True means these cannot be changed at runtime.
    Immutability is important for safety-critical constants.
    """
    signal_type: SignalType
    min_value: float
    max_value: float
    unit: str
    description: str


# ── Boundary Definitions ─────────────────────────────────────
# These are not clinical alarm thresholds.
# These are absolute physiological impossibility boundaries.
# Sources: AAMI standards, clinical physiology literature.
PHYSIOLOGICAL_BOUNDS: dict[SignalType, PhysiologicalBounds] = {
    SignalType.SPO2: PhysiologicalBounds(
        signal_type=SignalType.SPO2,
        min_value=0.0,
        max_value=100.0,
        unit="%",
        description="Oxygen saturation percentage"
    ),
    SignalType.HEART_RATE: PhysiologicalBounds(
        signal_type=SignalType.HEART_RATE,
        min_value=0.0,
        max_value=300.0,
        unit="bpm",
        description="Heart rate in beats per minute"
    ),
    SignalType.RESPIRATORY_RATE: PhysiologicalBounds(
        signal_type=SignalType.RESPIRATORY_RATE,
        min_value=0.0,
        max_value=60.0,
        unit="breaths/min",
        description="Respiratory rate"
    ),
    SignalType.ABP_SYSTOLIC: PhysiologicalBounds(
        signal_type=SignalType.ABP_SYSTOLIC,
        min_value=40.0,
        max_value=300.0,
        unit="mmHg",
        description="Arterial blood pressure systolic"
    ),
    SignalType.ABP_DIASTOLIC: PhysiologicalBounds(
        signal_type=SignalType.ABP_DIASTOLIC,
        min_value=20.0,
        max_value=200.0,
        unit="mmHg",
        description="Arterial blood pressure diastolic"
    ),
    SignalType.TEMPERATURE: PhysiologicalBounds(
        signal_type=SignalType.TEMPERATURE,
        min_value=25.0,
        max_value=45.0,
        unit="°C",
        description="Body temperature"
    ),
}


# ── Validation Result ─────────────────────────────────────────
@dataclass
class ValidationResult:
    """
    Result of validating a signal window.
    Always returned - never raises exceptions.

    Why not raise exceptions?
    In a real-time signal pipeline, one bad reading
    should not crash the system. It should be flagged
    and handled gracefully downstream.
    """
    is_valid: bool
    signal_type: SignalType
    n_samples: int
    n_invalid: int
    invalid_ratio: float
    n_nan: int
    nan_ratio: float
    out_of_bounds_min: int
    out_of_bounds_max: int
    message: str


# ── Validator ─────────────────────────────────────────────────
class SignalValidator:
    """
    Validates raw signal arrays before processing.

    Usage:
        validator = SignalValidator()
        result = validator.validate(signal_array, SignalType.SPO2)

        if not result.is_valid:
            logger.warning(result.message)
            # handle accordingly
    """

    # If more than this ratio of samples are invalid
    # the entire window is considered untrustworthy
    INVALID_RATIO_THRESHOLD = 0.20  # 20%
    NAN_RATIO_THRESHOLD = 0.10      # 10%

    def validate(
        self,
        signal: np.ndarray,
        signal_type: SignalType,
        sampling_frequency: Optional[int] = None,
    ) -> ValidationResult:
        """
        Validate a signal window against physiological bounds.

        Args:
            signal: Raw signal array
            signal_type: Type of signal being validated
            sampling_frequency: Optional - used for duration logging

        Returns:
            ValidationResult - never raises
        """
        n_samples = len(signal)

        if n_samples == 0:
            return ValidationResult(
                is_valid=False,
                signal_type=signal_type,
                n_samples=0,
                n_invalid=0,
                invalid_ratio=0.0,
                n_nan=0,
                nan_ratio=0.0,
                out_of_bounds_min=0,
                out_of_bounds_max=0,
                message="Empty signal array received"
            )

        # ── NaN Check ────────────────────────────────────────
        nan_mask = np.isnan(signal)
        n_nan = int(nan_mask.sum())
        nan_ratio = n_nan / n_samples

        # Work only on non-NaN values for bounds checking
        clean_signal = signal[~nan_mask]

        # ── Bounds Check ─────────────────────────────────────
        bounds = PHYSIOLOGICAL_BOUNDS.get(signal_type)
        out_of_bounds_min = 0
        out_of_bounds_max = 0

        if bounds and len(clean_signal) > 0:
            out_of_bounds_min = int(
                (clean_signal < bounds.min_value).sum()
            )
            out_of_bounds_max = int(
                (clean_signal > bounds.max_value).sum()
            )

        n_invalid = n_nan + out_of_bounds_min + out_of_bounds_max
        invalid_ratio = n_invalid / n_samples

        # ── Decision ─────────────────────────────────────────
        is_valid = (
            nan_ratio <= self.NAN_RATIO_THRESHOLD
            and invalid_ratio <= self.INVALID_RATIO_THRESHOLD
        )

        # ── Message ──────────────────────────────────────────
        if is_valid:
            message = (
                f"{signal_type.value} window valid - "
                f"{n_samples} samples, "
                f"{invalid_ratio:.1%} invalid"
            )
        else:
            reasons = []
            if nan_ratio > self.NAN_RATIO_THRESHOLD:
                reasons.append(
                    f"NaN ratio {nan_ratio:.1%} "
                    f"exceeds threshold {self.NAN_RATIO_THRESHOLD:.1%}"
                )
            if out_of_bounds_min > 0:
                reasons.append(
                    f"{out_of_bounds_min} samples below "
                    f"minimum {bounds.min_value}{bounds.unit}"
                )
            if out_of_bounds_max > 0:
                reasons.append(
                    f"{out_of_bounds_max} samples above "
                    f"maximum {bounds.max_value}{bounds.unit}"
                )
            message = (
                f"{signal_type.value} window INVALID - "
                + " | ".join(reasons)
            )

        logger.debug(message)

        return ValidationResult(
            is_valid=is_valid,
            signal_type=signal_type,
            n_samples=n_samples,
            n_invalid=n_invalid,
            invalid_ratio=invalid_ratio,
            n_nan=n_nan,
            nan_ratio=nan_ratio,
            out_of_bounds_min=out_of_bounds_min,
            out_of_bounds_max=out_of_bounds_max,
            message=message
        )