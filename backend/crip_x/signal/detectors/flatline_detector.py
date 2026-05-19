"""
CRIP-X Flatline Detector

Detects flatline artifacts in medical device signals.

A flatline occurs when a signal stops changing —
value stays constant or near-constant over time.

Common causes:
    - Sensor disconnection
    - Lead/probe detachment
    - Device freeze or malfunction
    - Signal clipping at min/max hardware limit

Why this matters clinically:
    A flatline SpO2 of 98% looks normal to a threshold
    alerter. CRIP-X recognizes it as a sensor artifact
    because real SpO2 never stays perfectly constant —
    it oscillates slightly with each heartbeat.
"""

import numpy as np
from crip_x.signal.detectors.base_detector import (
    BaseDetector,
    DetectionResult,
    ArtifactType,
)
from crip_x.utils.validators import SignalType
from crip_x.utils.logger import get_logger

logger = get_logger(__name__)


class FlatlineDetector(BaseDetector):
    """
    Detects flatline artifacts using two complementary methods:

    Method 1 — Standard Deviation
        A real physiological signal always has variance.
        If std dev over a window falls below a threshold,
        the signal is suspiciously flat.

    Method 2 — Unique Value Ratio
        Counts how many unique values exist in the window.
        A real signal has high uniqueness.
        A flatline has very few unique values.

    Both methods run and the results are combined.
    Requiring both to agree reduces false positives.
    """

    # Minimum std dev below which signal is considered flat
    # Scaled by sensitivity parameter
    STD_THRESHOLD = 0.05

    # If fewer than this fraction of samples are unique
    # signal is considered flat
    UNIQUE_RATIO_THRESHOLD = 0.05

    # Minimum window size to run detection
    MIN_SAMPLES = 10

    def _detect(
        self,
        signal: np.ndarray,
        signal_type: SignalType,
    ) -> DetectionResult:
        """
        Core flatline detection logic.

        Args:
            signal: Validated 1D signal array
            signal_type: Type of signal being analyzed

        Returns:
            DetectionResult with flatline assessment
        """
        n_samples = len(signal)

        # ── Minimum Length Check ─────────────────────────────
        if n_samples < self.MIN_SAMPLES:
            return DetectionResult(
                artifact_detected=False,
                artifact_type=ArtifactType.NONE,
                confidence=0.0,
                severity=0.0,
                affected_ratio=0.0,
                signal_type=signal_type,
                detector_name=self.name,
                message=(
                    f"Window too short for flatline detection "
                    f"({n_samples} < {self.MIN_SAMPLES} samples)"
                )
            )

        # ── Remove NaNs before analysis ───────────────────────
        clean = signal[~np.isnan(signal)]
        if len(clean) == 0:
            return DetectionResult(
                artifact_detected=False,
                artifact_type=ArtifactType.NONE,
                confidence=0.0,
                severity=0.0,
                affected_ratio=0.0,
                signal_type=signal_type,
                detector_name=self.name,
                message="All samples are NaN — skipping flatline detection"
            )

        # ── Method 1: Standard Deviation ─────────────────────
        std = float(np.std(clean))
        # Adjust threshold by sensitivity
        # Higher sensitivity = catches more subtle flatlines
        adjusted_std_threshold = self.STD_THRESHOLD / self.sensitivity
        std_flatline = std < adjusted_std_threshold

        # ── Method 2: Unique Value Ratio ──────────────────────
        n_unique = len(np.unique(np.round(clean, decimals=2)))
        unique_ratio = n_unique / len(clean)
        adjusted_unique_threshold = (
            self.UNIQUE_RATIO_THRESHOLD / self.sensitivity
        )
        unique_flatline = unique_ratio < adjusted_unique_threshold

        # ── Combined Decision ─────────────────────────────────
        # Both methods must agree for a flatline call
        # This reduces false positives on naturally
        # low-variance signals
        artifact_detected = std_flatline and unique_flatline

        # ── Confidence Calculation ────────────────────────────
        # Confidence is how far below threshold we are
        # The flatter the signal, the higher the confidence
        if artifact_detected:
            std_confidence = min(
                1.0,
                adjusted_std_threshold / (std + 1e-10)
            )
            unique_confidence = min(
                1.0,
                adjusted_unique_threshold / (unique_ratio + 1e-10)
            )
            confidence = float(
                (std_confidence + unique_confidence) / 2
            )
            confidence = min(0.99, confidence)
        else:
            confidence = 0.0

        # ── Severity Calculation ──────────────────────────────
        # A perfect flatline (std=0) is severity 1.0
        # Severity drops as variance increases
        severity = float(
            1.0 - min(1.0, std / (adjusted_std_threshold * 10))
        ) if artifact_detected else 0.0

        # ── Build Message ─────────────────────────────────────
        if artifact_detected:
            message = (
                f"Flatline detected — "
                f"std={std:.4f} (threshold={adjusted_std_threshold:.4f}) | "
                f"unique_ratio={unique_ratio:.3f} | "
                f"confidence={confidence:.2f}"
            )
        else:
            message = (
                f"No flatline — "
                f"std={std:.4f} | "
                f"unique_ratio={unique_ratio:.3f}"
            )

        return DetectionResult(
            artifact_detected=artifact_detected,
            artifact_type=(
                ArtifactType.FLATLINE
                if artifact_detected
                else ArtifactType.NONE
            ),
            confidence=confidence,
            severity=severity,
            affected_ratio=1.0 if artifact_detected else 0.0,
            signal_type=signal_type,
            detector_name=self.name,
            message=message,
            metadata={
                "std": std,
                "std_threshold": adjusted_std_threshold,
                "unique_ratio": unique_ratio,
                "unique_threshold": adjusted_unique_threshold,
                "std_flatline": std_flatline,
                "unique_flatline": unique_flatline,
            }
        )