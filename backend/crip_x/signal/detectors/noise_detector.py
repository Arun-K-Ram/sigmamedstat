"""
CRIP-X Noise Detector

Detects excessive noise artifacts in medical device signals.

Noise is high-frequency random variation overlaid on top
of the real physiological signal. Unlike spikes which are
isolated extreme events, noise is sustained randomness
that corrupts the entire signal window.

Common causes:
    - Patient movement / motion artifact
    - Electrical interference (60Hz AC noise)
    - Poor electrode contact (high impedance)
    - Muscle artifact (EMG contamination)
    - Ambient electromagnetic interference

Why noise is different from the other artifacts:
    Flatline  → too little variation
    Spike     → isolated extreme value
    Dropout   → signal absent
    Noise     → too much variation, wrong frequency content
"""

import numpy as np
from scipy import signal as scipy_signal
from crip_x.signal.detectors.base_detector import (
    BaseDetector,
    DetectionResult,
    ArtifactType,
)
from crip_x.utils.validators import SignalType
from crip_x.utils.logger import get_logger

logger = get_logger(__name__)


class NoiseDetector(BaseDetector):
    """
    Detects noise artifacts using three methods:

    Method 1 — Signal-to-Noise Ratio (SNR)
        Separates signal into low-frequency physiological
        component and high-frequency noise component.
        Real signals have high SNR.
        Noisy signals have low SNR.

    Method 2 — Sample Entropy
        Measures signal complexity/unpredictability.
        Real physiological signals have moderate complexity.
        Pure noise has very high complexity (maximum unpredictability).
        Flatlines have zero complexity.
        Noise sits at the high end.

    Method 3 — Rolling Variance Instability
        Real signals have relatively stable local variance.
        Noisy signals show highly variable local variance —
        some windows very quiet, others very noisy.
    """

    # Change these three values
    SNR_THRESHOLD_DB = 15.0          # was 10.0
    ENTROPY_THRESHOLD = 0.5          # was 0.8
    ROLLING_VAR_CV_THRESHOLD = 0.3   # was 2.0

    # SNR below this threshold indicates excessive noise
    # In dB — lower means more noise relative to signal
    SNR_THRESHOLD_DB = 10.0

    # Sample entropy above this indicates excessive noise
    ENTROPY_THRESHOLD = 0.8

    # Rolling variance coefficient of variation threshold
    ROLLING_VAR_CV_THRESHOLD = 2.0

    # Window size for rolling variance calculation
    ROLLING_WINDOW = 10

    MIN_SAMPLES = 30

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
                    f"Window too short for noise detection "
                    f"({n_samples} < {self.MIN_SAMPLES} samples)"
                )
            )

        # ── Remove NaNs ───────────────────────────────────────
        clean = signal[~np.isnan(signal)]
        if len(clean) < self.MIN_SAMPLES:
            return DetectionResult(
                artifact_detected=False,
                artifact_type=ArtifactType.NONE,
                confidence=0.0,
                severity=0.0,
                affected_ratio=0.0,
                signal_type=signal_type,
                detector_name=self.name,
                message="Insufficient clean samples for noise detection"
            )

        # ── Method 1: SNR Estimation ──────────────────────────
        # Separate signal into low-freq (physiological)
        # and high-freq (noise) components using a
        # simple moving average as the "signal" estimate
        window = min(self.ROLLING_WINDOW, len(clean) // 4)
        window = max(3, window)

        # Low frequency component via moving average
        low_freq = np.convolve(
            clean,
            np.ones(window) / window,
            mode='same'
        )

        # High frequency component = original - smoothed
        high_freq = clean - low_freq

        # Power of each component
        signal_power = float(np.mean(low_freq ** 2))
        noise_power = float(np.mean(high_freq ** 2))

        if noise_power < 1e-10:
            snr_db = 100.0  # effectively infinite SNR
        else:
            snr_db = float(
                10 * np.log10(signal_power / noise_power + 1e-10)
            )

        adjusted_snr_threshold = (
            self.SNR_THRESHOLD_DB / self.sensitivity
        )
        snr_noisy = snr_db < adjusted_snr_threshold

        # ── Method 2: Sample Entropy ──────────────────────────
        entropy = self._sample_entropy(clean)
        adjusted_entropy_threshold = (
            self.ENTROPY_THRESHOLD * self.sensitivity
        )
        entropy_noisy = entropy > adjusted_entropy_threshold

        # ── Method 3: Rolling Variance Instability ────────────
        roll_vars = []
        step = max(1, self.ROLLING_WINDOW // 2)
        for i in range(0, len(clean) - self.ROLLING_WINDOW, step):
            window_slice = clean[i:i + self.ROLLING_WINDOW]
            roll_vars.append(float(np.var(window_slice)))

        if len(roll_vars) > 2:
            mean_var = float(np.mean(roll_vars))
            std_var = float(np.std(roll_vars))
            cv = std_var / (mean_var + 1e-10)
            adjusted_cv_threshold = (
                self.ROLLING_VAR_CV_THRESHOLD / self.sensitivity
            )
            variance_unstable = cv > adjusted_cv_threshold
        else:
            cv = 0.0
            variance_unstable = False

        # ── Combined Decision ─────────────────────────────────
        # Require at least 2 of 3 methods to flag noise
        # More conservative than spike (any 1 sufficient)
        # because noise boundaries are fuzzier
        votes = sum([snr_noisy, entropy_noisy, variance_unstable])
        artifact_detected = votes >= 1

        # ── Confidence ────────────────────────────────────────
        if artifact_detected:
            # More votes = higher confidence
            base_confidence = votes / 3
            # Boost if SNR is very low
            snr_boost = max(
                0.0,
                (adjusted_snr_threshold - snr_db) /
                adjusted_snr_threshold
            ) * 0.3
            confidence = min(0.99, base_confidence + snr_boost)
        else:
            confidence = 0.0

        # ── Severity ─────────────────────────────────────────
        if artifact_detected:
            # Lower SNR = higher severity
            severity = min(
                1.0,
                max(0.0, (adjusted_snr_threshold - snr_db) /
                    adjusted_snr_threshold)
            )
            severity = max(severity, 0.3)
        else:
            severity = 0.0

        # ── Message ───────────────────────────────────────────
        if artifact_detected:
            methods = []
            if snr_noisy:
                methods.append(f"Low SNR ({snr_db:.1f}dB)")
            if entropy_noisy:
                methods.append(f"High entropy ({entropy:.3f})")
            if variance_unstable:
                methods.append(f"Unstable variance (CV={cv:.2f})")
            message = (
                f"Noise detected ({votes}/3 methods) — "
                + " | ".join(methods)
            )
        else:
            message = (
                f"No excessive noise — "
                f"SNR={snr_db:.1f}dB | "
                f"entropy={entropy:.3f} | "
                f"var_cv={cv:.2f}"
            )

        return DetectionResult(
            artifact_detected=artifact_detected,
            artifact_type=(
                ArtifactType.NOISE
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
                "snr_db": snr_db,
                "entropy": entropy,
                "variance_cv": cv,
                "snr_noisy": snr_noisy,
                "entropy_noisy": entropy_noisy,
                "variance_unstable": variance_unstable,
                "votes": votes,
            }
        )

    def _sample_entropy(
        self,
        signal: np.ndarray,
        m: int = 2,
        r_factor: float = 0.2,
    ) -> float:
        """
        Compute sample entropy of a signal.

        Sample entropy measures signal unpredictability.
        Higher = more random = more noise-like.

        Args:
            signal: Input signal array
            m: Template length (default 2)
            r_factor: Tolerance as fraction of std dev

        Returns:
            Sample entropy value (0.0 to ~2.0)
        """
        n = len(signal)
        r = r_factor * float(np.std(signal))

        if r < 1e-10:
            return 0.0  # Flatline has zero entropy

        # Limit computation for long signals
        if n > 200:
            signal = signal[:200]
            n = 200

        def count_matches(template_len: int) -> int:
            count = 0
            for i in range(n - template_len):
                template = signal[i:i + template_len]
                for j in range(i + 1, n - template_len):
                    candidate = signal[j:j + template_len]
                    if np.max(np.abs(template - candidate)) <= r:
                        count += 1
            return count

        b = count_matches(m)
        a = count_matches(m + 1)

        if b == 0:
            return 2.0  # Maximum entropy

        return float(-np.log(a / b + 1e-10))