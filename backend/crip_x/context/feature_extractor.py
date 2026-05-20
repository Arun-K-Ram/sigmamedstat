"""
CRIP-X Context Feature Extractor

Extracts contextual features surrounding a signal window.

This is CRIP-X's core differentiator.

Traditional signal quality systems ask:
    "Is this signal bad?"

CRIP-X asks:
    "Does this signal make sense given everything
     happening around it right now?"

Context features fall into four categories:

1. Multi-Signal Context
   What are neighboring signals doing simultaneously?
   If SpO2 is noisy AND ECG is noisy → likely patient movement
   If SpO2 is noisy but ECG is clean → likely probe issue

2. Temporal Context
   How long has this device been running?
   When was the last clean reading?
   Is reliability degrading over time?

3. Clinical Event Context
   Did something happen recently?
   Position change, medication, procedure?
   These explain transient signal artifacts.

4. Device Session Context
   How old is this monitoring session?
   Has this device been reliable historically?
   Is this a known problematic device?
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from crip_x.signal.signal_quality_index import SQIResult
from crip_x.utils.validators import SignalType
from crip_x.utils.logger import get_logger

logger = get_logger(__name__)


# ── Clinical Event Types ───────────────────────────────────────
class ClinicalEventType:
    """
    Known clinical events that explain signal artifacts.
    These are documented in the clinical literature as
    common causes of monitoring artifacts.
    """
    PATIENT_REPOSITIONING = "patient_repositioning"
    MEDICATION_ADMINISTRATION = "medication_administration"
    PROCEDURE_START = "procedure_start"
    PROCEDURE_END = "procedure_end"
    PATIENT_MOVEMENT = "patient_movement"
    DEVICE_RECONNECTION = "device_reconnection"
    CALIBRATION_EVENT = "calibration_event"
    STAFF_INTERVENTION = "staff_intervention"


# ── Clinical Event ─────────────────────────────────────────────
@dataclass
class ClinicalEvent:
    """
    A timestamped clinical event that may explain artifacts.

    In a real hospital system these come from:
    - Nurse call systems
    - EHR event logs
    - Device event logs
    - ADT (Admit/Discharge/Transfer) systems
    """
    event_type: str
    timestamp: float          # Unix timestamp or relative seconds
    duration_seconds: float = 0.0
    description: str = ""


# ── Context Input ──────────────────────────────────────────────
@dataclass
class ContextInput:
    """
    All contextual information available for a signal window.

    This is what gets passed to the feature extractor.
    In production this comes from multiple data sources
    assembled by the ingestion layer.

    Not all fields are always available — the extractor
    handles missing context gracefully.
    """
    # Primary signal being assessed
    primary_signal_type: SignalType
    primary_sqi: SQIResult
    current_timestamp: float

    # Neighboring signal SQI results (optional)
    # Key: SignalType, Value: SQIResult
    neighboring_sqis: dict = field(default_factory=dict)

    # Device session info
    session_start_timestamp: float = 0.0
    device_id: str = "unknown"

    # Historical SQI scores for this signal
    # Most recent first
    historical_sqis: list = field(default_factory=list)

    # Recent clinical events
    recent_events: list = field(default_factory=list)

    # Raw motion/accelerometer signal if available
    motion_signal: Optional[np.ndarray] = None


# ── Context Features ───────────────────────────────────────────
@dataclass
class ContextFeatures:
    """
    Extracted context features ready for the reliability scorer.

    These are the features the ML model will use alongside
    the SQI score to produce the final trust score.

    All values normalized to [0.0, 1.0] where possible.
    Higher = more problematic context.
    """

    # ── Multi-Signal Features ────────────────────────────────
    # Fraction of neighboring signals also degraded
    neighboring_degradation_ratio: float = 0.0

    # Average SQI of neighboring signals (0-100)
    neighboring_avg_sqi: float = 100.0

    # Whether ALL signals degraded simultaneously
    # (suggests environmental/patient cause vs device cause)
    multi_signal_simultaneous_degradation: bool = False

    # Correlation drop between primary and neighbors
    # High drop = signals diverging = device issue likely
    cross_signal_correlation: float = 1.0

    # ── Temporal Features ────────────────────────────────────
    # Seconds since last clean reading (SQI >= 70)
    seconds_since_clean_reading: float = 0.0

    # Device session age in seconds
    session_age_seconds: float = 0.0

    # Reliability trend — negative = degrading
    # Computed from historical SQI slope
    reliability_trend: float = 0.0

    # Fraction of recent windows that were clean
    recent_clean_ratio: float = 1.0

    # ── Clinical Event Features ──────────────────────────────
    # Whether a clinical event occurred recently
    recent_clinical_event: bool = False

    # Seconds since most recent clinical event
    seconds_since_event: float = float('inf')

    # Whether event type is known to cause artifacts
    event_known_artifact_cause: bool = False

    # ── Motion Features ──────────────────────────────────────
    # Motion index from accelerometer (0-1)
    motion_index: float = 0.0

    # Whether significant motion was detected
    significant_motion: bool = False

    # ── Composite Scores ─────────────────────────────────────
    # Pre-computed composite scores for the scorer
    context_artifact_probability: float = 0.0
    context_reliability_bonus: float = 0.0


# ── Feature Extractor ──────────────────────────────────────────
class ContextFeatureExtractor:
    """
    Extracts context features from surrounding signal environment.

    Design principle:
        Every feature has a fallback value when data is missing.
        Missing context = neutral assumption, not worst case.
        The system degrades gracefully with less information.
    """

    # Events that commonly cause monitoring artifacts
    ARTIFACT_CAUSING_EVENTS = {
        ClinicalEventType.PATIENT_REPOSITIONING,
        ClinicalEventType.PATIENT_MOVEMENT,
        ClinicalEventType.DEVICE_RECONNECTION,
        ClinicalEventType.STAFF_INTERVENTION,
        ClinicalEventType.PROCEDURE_START,
    }

    # Time window for "recent" events in seconds
    RECENT_EVENT_WINDOW = 60.0

    # SQI threshold for "clean" reading
    CLEAN_SQI_THRESHOLD = 70.0

    # Number of historical windows to consider
    HISTORY_WINDOW = 10

    def extract(self, context: ContextInput) -> ContextFeatures:
        """
        Extract all context features from a ContextInput.

        Args:
            context: All available contextual information

        Returns:
            ContextFeatures ready for reliability scorer
        """
        features = ContextFeatures()

        # Extract each feature category
        self._extract_multi_signal_features(context, features)
        self._extract_temporal_features(context, features)
        self._extract_clinical_event_features(context, features)
        self._extract_motion_features(context, features)
        self._compute_composite_scores(context, features)

        logger.debug(
            f"Context extracted | "
            f"neighboring_degradation={features.neighboring_degradation_ratio:.2f} | "
            f"reliability_trend={features.reliability_trend:.3f} | "
            f"recent_event={features.recent_clinical_event} | "
            f"motion={features.motion_index:.2f} | "
            f"artifact_probability={features.context_artifact_probability:.2f}"
        )

        return features

    def _extract_multi_signal_features(
        self,
        context: ContextInput,
        features: ContextFeatures,
    ) -> None:
        """
        Analyze neighboring signal behavior.

        Key insight: if multiple signals degrade simultaneously,
        it suggests a patient/environment cause rather than
        a device-specific issue. This is important for
        failure attribution downstream.
        """
        if not context.neighboring_sqis:
            # No neighboring signals — neutral assumption
            features.neighboring_degradation_ratio = 0.0
            features.neighboring_avg_sqi = 100.0
            return

        neighbor_scores = [
            sqi.sqi_score
            for sqi in context.neighboring_sqis.values()
        ]

        # Average SQI of neighbors
        features.neighboring_avg_sqi = float(np.mean(neighbor_scores))

        # How many neighbors are degraded
        degraded_neighbors = sum(
            1 for score in neighbor_scores
            if score < self.CLEAN_SQI_THRESHOLD
        )
        features.neighboring_degradation_ratio = (
            degraded_neighbors / len(neighbor_scores)
        )

        # All signals degraded simultaneously
        primary_degraded = (
            context.primary_sqi.sqi_score < self.CLEAN_SQI_THRESHOLD
        )
        features.multi_signal_simultaneous_degradation = (
            primary_degraded
            and features.neighboring_degradation_ratio > 0.5
        )

        # Cross-signal correlation
        # Lower = signals diverging = device-specific issue
        if features.neighboring_avg_sqi > 0:
            features.cross_signal_correlation = min(
                1.0,
                features.neighboring_avg_sqi / 100.0
            )

        logger.debug(
            f"Multi-signal: "
            f"neighbors={len(neighbor_scores)} | "
            f"degraded={degraded_neighbors} | "
            f"avg_sqi={features.neighboring_avg_sqi:.0f} | "
            f"simultaneous={features.multi_signal_simultaneous_degradation}"
        )

    def _extract_temporal_features(
        self,
        context: ContextInput,
        features: ContextFeatures,
    ) -> None:
        """
        Analyze time-based reliability patterns.

        Key insight: reliability degrading over a session suggests
        sensor drift or electrode drying — a temporal pattern
        a threshold alerter would completely miss.
        """
        # Session age
        features.session_age_seconds = max(
            0.0,
            context.current_timestamp - context.session_start_timestamp
        )

        if not context.historical_sqis:
            features.seconds_since_clean_reading = 0.0
            features.recent_clean_ratio = 1.0
            features.reliability_trend = 0.0
            return

        # Recent history window
        recent = context.historical_sqis[:self.HISTORY_WINDOW]

        # Time since last clean reading
        features.seconds_since_clean_reading = 0.0
        for i, sqi in enumerate(recent):
            if sqi >= self.CLEAN_SQI_THRESHOLD:
                break
            features.seconds_since_clean_reading += 10.0

        # Recent clean ratio
        clean_count = sum(
            1 for sqi in recent
            if sqi >= self.CLEAN_SQI_THRESHOLD
        )
        features.recent_clean_ratio = clean_count / len(recent)

        # Reliability trend using linear regression slope
        if len(recent) >= 3:
            x = np.arange(len(recent), dtype=float)
            y = np.array(recent, dtype=float)
            # Normalize slope to [-1, 1]
            slope = float(np.polyfit(x, y, 1)[0])
            features.reliability_trend = float(
                np.clip(slope / 10.0, -1.0, 1.0)
            )

        logger.debug(
            f"Temporal: "
            f"session_age={features.session_age_seconds:.0f}s | "
            f"since_clean={features.seconds_since_clean_reading:.0f}s | "
            f"clean_ratio={features.recent_clean_ratio:.2f} | "
            f"trend={features.reliability_trend:.3f}"
        )

    def _extract_clinical_event_features(
        self,
        context: ContextInput,
        features: ContextFeatures,
    ) -> None:
        """
        Analyze recent clinical events.

        Key insight: a SpO2 drop 3 seconds after a nurse
        repositioned the patient is almost certainly
        probe displacement, not desaturation.
        """
        if not context.recent_events:
            features.recent_clinical_event = False
            features.seconds_since_event = float('inf')
            return

        # Find most recent event
        recent_events = [
            event for event in context.recent_events
            if (context.current_timestamp - event.timestamp)
            <= self.RECENT_EVENT_WINDOW
        ]

        if not recent_events:
            features.recent_clinical_event = False
            features.seconds_since_event = float('inf')
            return

        # Most recent event
        latest_event = min(
            recent_events,
            key=lambda e: context.current_timestamp - e.timestamp
        )

        features.recent_clinical_event = True
        features.seconds_since_event = (
            context.current_timestamp - latest_event.timestamp
        )
        features.event_known_artifact_cause = (
            latest_event.event_type in self.ARTIFACT_CAUSING_EVENTS
        )

        logger.debug(
            f"Clinical event: "
            f"type={latest_event.event_type} | "
            f"seconds_ago={features.seconds_since_event:.0f} | "
            f"artifact_cause={features.event_known_artifact_cause}"
        )

    def _extract_motion_features(
        self,
        context: ContextInput,
        features: ContextFeatures,
    ) -> None:
        """
        Analyze motion/accelerometer signal if available.

        Motion is the single most common cause of false
        alarms in wearable and bedside monitoring.
        """
        if context.motion_signal is None:
            features.motion_index = 0.0
            features.significant_motion = False
            return

        motion = context.motion_signal
        motion_clean = motion[~np.isnan(motion)]

        if len(motion_clean) == 0:
            return

        # Motion index = normalized variance of accelerometer
        motion_var = float(np.var(motion_clean))
        features.motion_index = float(
            min(1.0, motion_var / 10.0)
        )
        features.significant_motion = features.motion_index > 0.3

        logger.debug(
            f"Motion: "
            f"index={features.motion_index:.2f} | "
            f"significant={features.significant_motion}"
        )

    def _compute_composite_scores(
        self,
        context: ContextInput,
        features: ContextFeatures,
    ) -> None:
        """
        Compute composite scores that summarize context.

        These give the reliability scorer a pre-digested
        view of context instead of raw features.
        """
        # ── Artifact Probability from Context ────────────────
        # How likely is it that the signal issue is an artifact
        # (vs a real clinical event) based purely on context?

        artifact_signals = []

        # Motion is a strong artifact indicator
        if features.significant_motion:
            artifact_signals.append(0.8)

        # Recent repositioning/movement event
        if (features.recent_clinical_event
                and features.event_known_artifact_cause
                and features.seconds_since_event < 30):
            artifact_signals.append(0.75)

        # Multiple signals degraded simultaneously
        # (patient cause, but still artifact)
        if features.multi_signal_simultaneous_degradation:
            artifact_signals.append(0.65)

        # High neighboring degradation
        if features.neighboring_degradation_ratio > 0.5:
            artifact_signals.append(0.6)

        features.context_artifact_probability = (
            float(np.mean(artifact_signals))
            if artifact_signals else 0.0
        )

        # ── Reliability Bonus ────────────────────────────────
        # Context can also increase confidence in a reading
        # If everything around it is clean and stable,
        # a borderline SQI is more trustworthy

        bonus_signals = []

        # All neighbors clean
        if features.neighboring_avg_sqi > 85:
            bonus_signals.append(0.2)

        # Positive reliability trend
        if features.reliability_trend > 0.1:
            bonus_signals.append(0.15)

        # High recent clean ratio
        if features.recent_clean_ratio > 0.8:
            bonus_signals.append(0.15)

        # No motion
        if features.motion_index < 0.1:
            bonus_signals.append(0.1)

        features.context_reliability_bonus = (
            float(np.sum(bonus_signals))
            if bonus_signals else 0.0
        )