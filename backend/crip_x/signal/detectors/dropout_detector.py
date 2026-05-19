"""
CRIP-X Dropout Detector

Detects signal dropout artifacts in medical device signals.

A dropout occurs when a signal goes missing entirely —
represented as NaN values, zeros, or sudden signal loss.

Different from a flatline:
    Flatline  → signal present but not changing (stuck value)
    Dropout   → signal absent entirely (missing data)

Common causes:
    - Probe/electrode physically disconnected
    - Cable damage or loose connection
    - Device power interruption
    - Wireless transmission loss (telemetry)
    - Buffer overflow in device firmware
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


class DropoutDetector(BaseDetector):
    """
    Detects signal dropout using three methods:

    Method 1 — NaN Detection
        Direct missing data. Device explicitly reports
        no signal. Clearest form of dropout.

    Method 2 — Zero Burst Detection
        Some devices report 0 instead of NaN during dropout.
        We detect sustained zero periods that are
        physiologically impossible for the signal type.

    Method 3 — Sudden Signal Loss
        Signal present, then abruptly drops to an
        implausible value and stays there.
        Detected via sudden large amplitude drop
        followed by sustained low variance.
    """

    # Fraction of NaN samples that triggers dropout flag
    NAN_THRESHOLD = 0.05          # 5% NaNs = dropout

    # Fraction of zero samples that triggers dropout flag
    ZERO_THRESHOLD = 0.05         # 5% zeros = dropout

    # Minimum consecutive NaN/zero samples to count
    # as a true dropout vs isolated missing sample
    MIN_CONSECUTIVE = 3

    MIN_SAMPLES = 10

    def _detect(
        self,
        signal: np.ndarray,
        signal_type: SignalType,
    ) -> DetectionResult:

        n_samples = len(signal)

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
                    f"Window too short for dropout detection "
                    f"({n_samples} < {self.MIN_SAMPLES} samples)"
                )
            )

        # ── Method 1: NaN Detection ───────────────────────────
        nan_mask = np.isnan(signal)
        n_nan = int(nan_mask.sum())
        nan_ratio = n_nan / n_samples

        # Find consecutive NaN runs
        max_consecutive_nan = self._max_consecutive(nan_mask)

        nan_dropout = (
            nan_ratio > (self.NAN_THRESHOLD / self.sensitivity)
            and max_consecutive_nan >= self.MIN_CONSECUTIVE
        )

        # ── Method 2: Zero Burst Detection ───────────────────
        # Only check non-NaN samples for zeros
        clean = signal.copy()
        clean[nan_mask] = np.nan
        zero_mask = np.isclose(clean, 0.0, atol=0.01)
        zero_mask[nan_mask] = False

        n_zeros = int(zero_mask.sum())
        zero_ratio = n_zeros / n_samples
        max_consecutive_zero = self._max_consecutive(zero_mask)

        zero_dropout = (
            zero_ratio > (self.ZERO_THRESHOLD / self.sensitivity)
            and max_consecutive_zero >= self.MIN_CONSECUTIVE
        )

        # ── Method 3: Sudden Signal Loss ──────────────────────
        # Split window in half — compare variance of each half
        # A sudden dropout causes the second half to have
        # dramatically lower variance than the first
        sudden_loss = False
        if n_samples >= 20:
            mid = n_samples // 2
            first_half = signal[:mid][~np.isnan(signal[:mid])]
            second_half = signal[mid:][~np.isnan(signal[mid:])]

            if len(first_half) > 5 and len(second_half) > 5:
                std_first = float(np.std(first_half))
                std_second = float(np.std(second_half))

                # Second half variance drops to near zero
                # while first half had real variance
                if (std_first > 0.1 and
                        std_second < std_first * 0.1):
                    sudden_loss = True

        # ── Combined Decision ─────────────────────────────────
        artifact_detected = nan_dropout or zero_dropout or sudden_loss

        # ── Affected Ratio ────────────────────────────────────
        affected_ratio = max(nan_ratio, zero_ratio)
        if sudden_loss and not artifact_detected:
            affected_ratio = 0.5  # approximate

        # ── Confidence ────────────────────────────────────────
        if artifact_detected:
            if nan_dropout:
                confidence = min(
                    0.99,
                    nan_ratio / (self.NAN_THRESHOLD / self.sensitivity)
                    * 0.8
                )
            elif zero_dropout:
                confidence = min(
                    0.99,
                    zero_ratio / (self.ZERO_THRESHOLD / self.sensitivity)
                    * 0.8
                )
            else:
                confidence = 0.75  # sudden loss is less certain
        else:
            confidence = 0.0

        # ── Severity ─────────────────────────────────────────
        severity = min(1.0, affected_ratio * 5) if artifact_detected else 0.0

        # ── Find dropout location ─────────────────────────────
        dropout_indices = np.where(nan_mask | zero_mask)[0]
        start_idx = int(dropout_indices[0]) if len(dropout_indices) > 0 else None
        end_idx = int(dropout_indices[-1]) if len(dropout_indices) > 0 else None

        # ── Message ───────────────────────────────────────────
        causes = []
        if nan_dropout:
            causes.append(
                f"NaN dropout ({nan_ratio:.1%} missing, "
                f"max {max_consecutive_nan} consecutive)"
            )
        if zero_dropout:
            causes.append(
                f"Zero burst ({zero_ratio:.1%} zeros, "
                f"max {max_consecutive_zero} consecutive)"
            )
        if sudden_loss:
            causes.append("Sudden signal loss detected")

        if artifact_detected:
            message = "Dropout detected — " + " | ".join(causes)
        else:
            message = (
                f"No dropout — "
                f"nan_ratio={nan_ratio:.1%} | "
                f"zero_ratio={zero_ratio:.1%}"
            )

        return DetectionResult(
            artifact_detected=artifact_detected,
            artifact_type=(
                ArtifactType.DROPOUT
                if artifact_detected
                else ArtifactType.NONE
            ),
            confidence=confidence,
            severity=severity,
            affected_ratio=affected_ratio,
            signal_type=signal_type,
            detector_name=self.name,
            message=message,
            start_idx=start_idx,
            end_idx=end_idx,
            metadata={
                "nan_ratio": nan_ratio,
                "zero_ratio": zero_ratio,
                "max_consecutive_nan": max_consecutive_nan,
                "max_consecutive_zero": max_consecutive_zero,
                "nan_dropout": nan_dropout,
                "zero_dropout": zero_dropout,
                "sudden_loss": sudden_loss,
            }
        )

    def _max_consecutive(self, mask: np.ndarray) -> int:
        """
        Find the longest consecutive run of True values in a mask.

        Example:
            [F, F, T, T, T, F, T, T] → 3
        """
        if not mask.any():
            return 0

        max_run = 0
        current_run = 0

        for val in mask:
            if val:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0

        return max_run