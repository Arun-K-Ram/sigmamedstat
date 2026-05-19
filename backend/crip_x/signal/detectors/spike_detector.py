"""
CRIP-X Spike Detector

Detects spike artifacts in medical device signals.

A spike is a sudden, extreme, instantaneous deviation
from the surrounding signal that disappears immediately.

Unlike a real physiological event which changes gradually
and is sustained over time, a spike appears in one or
two samples then vanishes.

Common causes:
    - Electrical interference / EMI
    - Patient movement causing brief electrode contact
    - ADC conversion errors in device hardware
    - Radio frequency interference from nearby equipment
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


class SpikeDetector(BaseDetector):
    """
    Detects spike artifacts using two methods:

    Method 1 — Z-Score
        Measures how many standard deviations a sample
        is from the window mean. Physiologically real
        values stay within a reasonable z-score range.
        Spikes produce extreme z-scores.

    Method 2 — Derivative (Rate of Change)
        Real signals change gradually.
        Spikes produce extreme first-order differences —
        the value jumps up AND comes back down immediately.
        We detect this by looking for large consecutive
        differences in opposite directions.
    """

    # Z-score threshold — samples beyond this are suspicious
    # 3.0 = 3 standard deviations from mean
    # Physiologically, real signals rarely exceed this
    ZSCORE_THRESHOLD = 3.0

    # Maximum physiologically plausible rate of change
    # per sample. Adjusted per signal type.
    # For SpO2: shouldn't change more than 5% per sample
    DERIVATIVE_THRESHOLD = 5.0

    # Minimum samples needed for reliable statistics
    MIN_SAMPLES = 10

    def _detect(
        self,
        signal: np.ndarray,
        signal_type: SignalType,
    ) -> DetectionResult:

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
                    f"Window too short for spike detection "
                    f"({n_samples} < {self.MIN_SAMPLES} samples)"
                )
            )

        # ── Remove NaNs ───────────────────────────────────────
        clean_mask = ~np.isnan(signal)
        clean = signal[clean_mask]

        if len(clean) < self.MIN_SAMPLES:
            return DetectionResult(
                artifact_detected=False,
                artifact_type=ArtifactType.NONE,
                confidence=0.0,
                severity=0.0,
                affected_ratio=0.0,
                signal_type=signal_type,
                detector_name=self.name,
                message="Insufficient clean samples for spike detection"
            )

        # ── Method 1: Z-Score ────────────────────────────────
        mean = np.mean(clean)
        std = np.std(clean)

        if std < 1e-10:
            # Flatline — no spikes possible, that's
            # the flatline detector's job
            return DetectionResult(
                artifact_detected=False,
                artifact_type=ArtifactType.NONE,
                confidence=0.0,
                severity=0.0,
                affected_ratio=0.0,
                signal_type=signal_type,
                detector_name=self.name,
                message="Signal has zero variance — deferring to flatline detector"
            )

        z_scores = np.abs((clean - mean) / std)
        adjusted_z_threshold = self.ZSCORE_THRESHOLD / self.sensitivity
        spike_mask_zscore = z_scores > adjusted_z_threshold
        n_zscore_spikes = int(spike_mask_zscore.sum())

        # ── Method 2: Derivative ──────────────────────────────
        # First difference — how much signal changes sample to sample
        diff = np.abs(np.diff(clean))
        adjusted_deriv_threshold = (
            self.DERIVATIVE_THRESHOLD / self.sensitivity
        )

        # A true spike shows large change UP then large change DOWN
        # We detect this by finding consecutive large differences
        spike_mask_deriv = np.zeros(len(clean), dtype=bool)
        for i in range(1, len(diff)):
            if (diff[i-1] > adjusted_deriv_threshold and
                    diff[i] > adjusted_deriv_threshold):
                # Large change in both directions — spike
                spike_mask_deriv[i] = True

        n_deriv_spikes = int(spike_mask_deriv.sum())

        # ── Combined Decision ─────────────────────────────────
        # Either method detecting spikes is enough
        # (unlike flatline where we require both)
        # Spikes are dangerous — better to over-detect
        n_spikes = max(n_zscore_spikes, n_deriv_spikes)
        artifact_detected = n_spikes > 0
        affected_ratio = n_spikes / n_samples

        # ── Confidence ────────────────────────────────────────
        if artifact_detected:
            # Confidence based on how extreme the z-scores are
            max_z = float(z_scores.max())
            confidence = min(
                0.99,
                (max_z - adjusted_z_threshold) /
                (adjusted_z_threshold * 2) + 0.5
            )
        else:
            confidence = 0.0

        # ── Severity ──────────────────────────────────────────
        severity = min(1.0, affected_ratio * 10) if artifact_detected else 0.0

        # ── Find first spike location ─────────────────────────
        spike_indices = np.where(spike_mask_zscore)[0]
        start_idx = int(spike_indices[0]) if len(spike_indices) > 0 else None
        end_idx = int(spike_indices[-1]) if len(spike_indices) > 0 else None

        # ── Message ───────────────────────────────────────────
        if artifact_detected:
            message = (
                f"Spike artifact detected — "
                f"{n_spikes} spike(s) | "
                f"max_z={float(z_scores.max()):.2f} | "
                f"affected={affected_ratio:.1%} | "
                f"confidence={confidence:.2f}"
            )
        else:
            message = (
                f"No spikes — "
                f"max_z={float(z_scores.max()):.2f} | "
                f"max_diff={float(diff.max()):.2f}"
            )

        return DetectionResult(
            artifact_detected=artifact_detected,
            artifact_type=(
                ArtifactType.SPIKE
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
                "n_zscore_spikes": n_zscore_spikes,
                "n_deriv_spikes": n_deriv_spikes,
                "max_z_score": float(z_scores.max()),
                "mean": float(mean),
                "std": float(std),
                "z_threshold": adjusted_z_threshold,
                "deriv_threshold": adjusted_deriv_threshold,
            }
        )