"""
CRIP-X Signal Quality Index

Combines all four detectors into a single trust score
per signal window.

This is the first real integration point - where individual
detector outputs become a unified reliability assessment.

The SQI (Signal Quality Index) answers:
    "How much should we trust this signal window?"

Output: 0-100 score where:
    90-100  → Excellent - high confidence reading
    70-89   → Good - minor issues, reading likely valid
    50-69   → Degraded - interpret with caution
    25-49   → Poor - reading unreliable
    0-24    → Critical - do not use this reading

Why 0-100 and not 0-1?
    Clinical systems use 0-100 for quality indices.
    It's the established convention in medical device
    literature. Matching convention makes CRIP-X
    immediately readable to domain experts.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from crip_x.signal.detectors.base_detector import (
    DetectionResult,
    ArtifactType,
)
from crip_x.signal.detectors.flatline_detector import FlatlineDetector
from crip_x.signal.detectors.spike_detector import SpikeDetector
from crip_x.signal.detectors.dropout_detector import DropoutDetector
from crip_x.signal.detectors.noise_detector import NoiseDetector
from crip_x.utils.validators import SignalType, SignalValidator
from crip_x.utils.config import settings
from crip_x.utils.logger import get_logger

logger = get_logger(__name__)


# ── SQI Grade ─────────────────────────────────────────────────
def get_sqi_grade(score: float) -> str:
    """Convert numeric SQI score to clinical grade label."""
    if score >= 90:
        return "EXCELLENT"
    elif score >= 70:
        return "GOOD"
    elif score >= 50:
        return "DEGRADED"
    elif score >= 25:
        return "POOR"
    else:
        return "CRITICAL"


# ── SQI Result ────────────────────────────────────────────────
@dataclass
class SQIResult:
    """
    Complete signal quality assessment for one signal window.

    This is what flows downstream to:
    - Context Correlation Layer
    - Reliability Scorer
    - Failure Attribution Engine
    - Dashboard
    """
    # Core output
    sqi_score: float              # 0-100
    grade: str                    # EXCELLENT/GOOD/DEGRADED/POOR/CRITICAL
    signal_type: SignalType
    n_samples: int

    # Individual detector results
    flatline_result: DetectionResult
    spike_result: DetectionResult
    dropout_result: DetectionResult
    noise_result: DetectionResult

    # Aggregated artifact info
    artifacts_detected: list[ArtifactType] = field(default_factory=list)
    dominant_artifact: Optional[ArtifactType] = None

    # Validation
    passed_validation: bool = True
    validation_message: str = ""

    # Human readable
    summary: str = ""

    def __post_init__(self) -> None:
        self.grade = get_sqi_grade(self.sqi_score)


# ── Signal Quality Index ───────────────────────────────────────
class SignalQualityIndex:
    """
    Orchestrates all four detectors and produces a unified
    signal quality score.

    Detector Weights:
        Not all artifacts are equally damaging to signal quality.
        A dropout is worse than mild noise.
        A flatline is worse than a single spike.

        Weights reflect clinical severity:
        - Dropout/Flatline → catastrophic (weight 1.0)
        - Spike            → serious but localized (weight 0.7)
        - Noise            → degrades but doesn't eliminate (weight 0.5)

    Scoring Logic:
        Start at 100.
        Each detected artifact subtracts points based on:
        - Artifact weight
        - Detection confidence
        - Severity of the artifact
        - Fraction of window affected

        Final score clipped to [0, 100].
    """

    # Penalty weights per artifact type
    # These reflect clinical severity, not just statistical severity
    ARTIFACT_WEIGHTS = {
        ArtifactType.FLATLINE:  1.0,   # complete signal failure
        ArtifactType.DROPOUT:   1.0,   # complete signal loss
        ArtifactType.SPIKE:     0.7,   # corrupts but briefly
        ArtifactType.NOISE:     0.5,   # degrades quality
    }

    # Maximum penalty each artifact type can contribute
    MAX_PENALTIES = {
        ArtifactType.FLATLINE:  60.0,
        ArtifactType.DROPOUT:   60.0,
        ArtifactType.SPIKE:     40.0,
        ArtifactType.NOISE:     30.0,
    }

    def __init__(self, sensitivity: float = 1.0) -> None:
        self.sensitivity = sensitivity
        self.validator = SignalValidator()

        # Initialize all four detectors
        self.flatline_detector = FlatlineDetector(sensitivity)
        self.spike_detector = SpikeDetector(sensitivity)
        self.dropout_detector = DropoutDetector(sensitivity)
        self.noise_detector = NoiseDetector(sensitivity)

        logger.debug(
            f"SignalQualityIndex initialized | "
            f"sensitivity={sensitivity}"
        )

    def compute(
        self,
        signal: np.ndarray,
        signal_type: SignalType,
    ) -> SQIResult:
        """
        Run all detectors and compute unified SQI score.

        Args:
            signal: Raw signal array
            signal_type: Type of signal

        Returns:
            SQIResult with score, grade, and full detector outputs
        """
        n_samples = len(signal)

        # ── Step 1: Validate Input ────────────────────────────
        validation = self.validator.validate(signal, signal_type)

        if not validation.is_valid:
            logger.warning(
                f"Signal failed validation - "
                f"returning minimum SQI | {validation.message}"
            )
            # Return minimum viable result
            empty_result = self._empty_detection_result(signal_type)
            return SQIResult(
                sqi_score=0.0,
                grade="CRITICAL",
                signal_type=signal_type,
                n_samples=n_samples,
                flatline_result=empty_result,
                spike_result=empty_result,
                dropout_result=empty_result,
                noise_result=empty_result,
                passed_validation=False,
                validation_message=validation.message,
                summary=f"CRITICAL: Signal failed validation - {validation.message}"
            )

        # ── Step 2: Run All Detectors ─────────────────────────
        flatline_result = self.flatline_detector.detect(
            signal, signal_type
        )
        spike_result = self.spike_detector.detect(
            signal, signal_type
        )
        dropout_result = self.dropout_detector.detect(
            signal, signal_type
        )
        noise_result = self.noise_detector.detect(
            signal, signal_type
        )

        detector_results = {
            ArtifactType.FLATLINE: flatline_result,
            ArtifactType.DROPOUT:  dropout_result,
            ArtifactType.SPIKE:    spike_result,
            ArtifactType.NOISE:    noise_result,
        }

        # ── Step 3: Compute Penalty Per Artifact ──────────────
        total_penalty = 0.0
        artifacts_detected = []

        for artifact_type, result in detector_results.items():
            if result.artifact_detected:
                artifacts_detected.append(artifact_type)

                weight = self.ARTIFACT_WEIGHTS[artifact_type]
                max_penalty = self.MAX_PENALTIES[artifact_type]

                # Penalty scales with confidence and severity
                penalty = (
                    weight
                    * result.confidence
                    * result.severity
                    * max_penalty
                )

                logger.debug(
                    f"{artifact_type.value} penalty: {penalty:.1f} | "
                    f"confidence={result.confidence:.2f} | "
                    f"severity={result.severity:.2f}"
                )

                total_penalty += penalty

        # ── Step 4: Compute Final Score ───────────────────────
        sqi_score = float(
            max(0.0, min(100.0, 100.0 - total_penalty))
        )

        # ── Step 5: Dominant Artifact ─────────────────────────
        # Which artifact contributed most to score reduction
        dominant_artifact = None
        if artifacts_detected:
            dominant_artifact = max(
                artifacts_detected,
                key=lambda a: self.ARTIFACT_WEIGHTS[a]
            )

        # ── Step 6: Summary Message ───────────────────────────
        grade = get_sqi_grade(sqi_score)

        if not artifacts_detected:
            summary = (
                f"{signal_type.value.upper()} | "
                f"SQI={sqi_score:.0f} | "
                f"{grade} | No artifacts detected"
            )
        else:
            artifact_names = [a.value for a in artifacts_detected]
            summary = (
                f"{signal_type.value.upper()} | "
                f"SQI={sqi_score:.0f} | "
                f"{grade} | "
                f"Artifacts: {', '.join(artifact_names)} | "
                f"Dominant: {dominant_artifact.value}"
            )

        logger.info(summary)

        return SQIResult(
            sqi_score=sqi_score,
            grade=grade,
            signal_type=signal_type,
            n_samples=n_samples,
            flatline_result=flatline_result,
            spike_result=spike_result,
            dropout_result=dropout_result,
            noise_result=noise_result,
            artifacts_detected=artifacts_detected,
            dominant_artifact=dominant_artifact,
            passed_validation=True,
            validation_message="",
            summary=summary,
        )

    def _empty_detection_result(
        self,
        signal_type: SignalType
    ) -> DetectionResult:
        """Returns a placeholder DetectionResult for failed validation."""
        from crip_x.signal.detectors.base_detector import DetectionResult
        return DetectionResult(
            artifact_detected=False,
            artifact_type=ArtifactType.NONE,
            confidence=0.0,
            severity=0.0,
            affected_ratio=0.0,
            signal_type=signal_type,
            detector_name="N/A",
            message="Signal failed validation - detection not run"
        )