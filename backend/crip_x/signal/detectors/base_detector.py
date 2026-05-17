"""
CRIP-X Base Detector

Abstract base class that every signal detector inherits from.

Why an abstract base class?
    Every detector - flatline, spike, dropout, noise -
    shares the same interface. This guarantees:

    1. Every detector accepts the same inputs
    2. Every detector returns the same output structure
    3. Adding a new detector means extending this class,
       not rewriting logic
    4. Tests written against this interface work for
       every detector automatically

    This is the Open/Closed principle - open for extension,
    closed for modification. Relevant in IEC 62304 because
    validated components shouldn't need to be rewritten
    to add new functionality.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np

from crip_x.utils.logger import get_logger
from crip_x.utils.validators import SignalType

logger = get_logger(__name__)


# ── Artifact Types ────────────────────────────────────────────
class ArtifactType(Enum):
    """
    Taxonomy of known signal artifact types.
    This is your failure vocabulary - what CRIP-X knows
    how to recognize and name.

    UNKNOWN is important - it's honest. If the system
    can't classify the artifact, it says so rather than
    guessing. In safety-critical systems, admitting
    uncertainty is better than false confidence.
    """
    FLATLINE = "flatline"
    SPIKE = "spike"
    DROPOUT = "dropout"
    NOISE = "noise"
    MOTION_ARTIFACT = "motion_artifact"
    CALIBRATION_DRIFT = "calibration_drift"
    UNKNOWN = "unknown"
    NONE = "none"            # No artifact detected


# ── Detection Result ──────────────────────────────────────────
@dataclass
class DetectionResult:
    """
    Standardized output from every detector.

    Every detector returns this exact structure.
    Downstream components - scorer, attribution engine,
    dashboard - never need to know which detector
    produced a result. They just consume DetectionResult.

    Fields:
        artifact_detected: Primary yes/no answer
        artifact_type:     What kind of artifact
        confidence:        How confident is the detector (0.0-1.0)
        severity:          How bad is it (0.0-1.0)
        affected_ratio:    Fraction of window affected
        start_idx:         Where artifact starts in window
        end_idx:           Where artifact ends in window
        signal_type:       Which signal was analyzed
        detector_name:     Which detector produced this
        metadata:          Detector-specific extra info
        message:           Human-readable explanation
    """
    artifact_detected: bool
    artifact_type: ArtifactType
    confidence: float           # 0.0 = no confidence, 1.0 = certain
    severity: float             # 0.0 = negligible, 1.0 = catastrophic
    affected_ratio: float       # fraction of window affected
    signal_type: SignalType
    detector_name: str
    message: str
    start_idx: Optional[int] = None
    end_idx: Optional[int] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate ranges after initialization."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"Confidence must be 0.0-1.0, got {self.confidence}"
            )
        if not 0.0 <= self.severity <= 1.0:
            raise ValueError(
                f"Severity must be 0.0-1.0, got {self.severity}"
            )
        if not 0.0 <= self.affected_ratio <= 1.0:
            raise ValueError(
                f"Affected ratio must be 0.0-1.0, got {self.affected_ratio}"
            )


# ── Base Detector ─────────────────────────────────────────────
class BaseDetector(ABC):
    """
    Abstract base class for all CRIP-X signal detectors.

    Subclasses must implement:
        _detect(signal, signal_type) -> DetectionResult

    Subclasses get for free:
        detect() - validates input, calls _detect, logs result
        name     - detector identifier
    """

    def __init__(self, sensitivity: float = 1.0) -> None:
        """
        Args:
            sensitivity: Multiplier for detection thresholds.
                        > 1.0 = more sensitive (more detections)
                        < 1.0 = less sensitive (fewer detections)
                        Default 1.0 = baseline calibration
        """
        if not 0.1 <= sensitivity <= 3.0:
            raise ValueError(
                f"Sensitivity must be 0.1-3.0, got {sensitivity}"
            )
        self.sensitivity = sensitivity
        logger.debug(
            f"{self.name} initialized with sensitivity={sensitivity}"
        )

    @property
    def name(self) -> str:
        """Detector name derived from class name."""
        return self.__class__.__name__

    @abstractmethod
    def _detect(
        self,
        signal: np.ndarray,
        signal_type: SignalType,
    ) -> DetectionResult:
        """
        Core detection logic. Implemented by each subclass.
        Receives clean, validated signal array.
        Must return a DetectionResult.
        """
        ...

    def detect(
        self,
        signal: np.ndarray,
        signal_type: SignalType,
    ) -> DetectionResult:
        """
        Public interface. Validates input then runs detection.

        This method is NOT overridden by subclasses.
        All input validation happens here once, not in
        every individual detector.
        """
        # ── Input Validation ─────────────────────────────────
        if not isinstance(signal, np.ndarray):
            signal = np.array(signal, dtype=np.float64)

        if signal.ndim != 1:
            raise ValueError(
                f"Signal must be 1D array, got shape {signal.shape}"
            )

        if len(signal) == 0:
            return DetectionResult(
                artifact_detected=False,
                artifact_type=ArtifactType.NONE,
                confidence=0.0,
                severity=0.0,
                affected_ratio=0.0,
                signal_type=signal_type,
                detector_name=self.name,
                message="Empty signal - no detection performed"
            )

        # ── Run Detection ─────────────────────────────────────
        result = self._detect(signal, signal_type)

        # ── Log Result ────────────────────────────────────────
        if result.artifact_detected:
            logger.warning(
                f"{self.name} | {signal_type.value} | "
                f"{result.artifact_type.value} detected | "
                f"confidence={result.confidence:.2f} | "
                f"severity={result.severity:.2f} | "
                f"{result.message}"
            )
        else:
            logger.debug(
                f"{self.name} | {signal_type.value} | "
                f"clean | {result.message}"
            )

        return result