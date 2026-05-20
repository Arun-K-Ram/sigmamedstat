"""
CRIP-X Failure Attribution Engine

Explains WHY a signal is unreliable in plain English.

This is what separates CRIP-X from a black box.

Most ML systems say:    "Trust score: 23 - LOW"
CRIP-X says:            "Likely sensor displacement -
                         SpO2 probe lost contact 8 seconds
                         after patient was repositioned.
                         ECG remains stable suggesting
                         patient condition is unchanged."

That explanation is what a nurse actually needs.
That explanation is what FDA explainability guidance requires.
That explanation is what makes CRIP-X clinically useful.

Attribution follows a priority hierarchy:
    1. Physical device failure (highest priority)
    2. Patient-caused artifact
    3. Environmental interference
    4. Device calibration/drift
    5. Unknown (honest fallback)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from crip_x.signal.signal_quality_index import SQIResult
from crip_x.signal.detectors.base_detector import ArtifactType
from crip_x.context.feature_extractor import ContextFeatures
from crip_x.scoring.reliability_scorer import ReliabilityScore
from crip_x.utils.validators import SignalType
from crip_x.utils.logger import get_logger

logger = get_logger(__name__)


# ── Failure Categories ────────────────────────────────────────
class FailureCategory(Enum):
    """
    High-level failure categories for clinical communication.
    Maps to actionable responses a clinician can take.
    """
    SENSOR_DISPLACEMENT  = "sensor_displacement"
    MOTION_ARTIFACT      = "motion_artifact"
    DEVICE_MALFUNCTION   = "device_malfunction"
    CALIBRATION_DRIFT    = "calibration_drift"
    ENVIRONMENTAL        = "environmental_interference"
    PATIENT_CONDITION    = "patient_condition"
    UNKNOWN              = "unknown"


# ── Attribution Result ────────────────────────────────────────
@dataclass
class AttributionResult:
    """
    Complete failure attribution for a signal reading.

    Designed to be directly displayable on the dashboard
    and returnable via the API.
    """
    # Primary attribution
    failure_category: FailureCategory
    confidence: float                    # 0-1

    # Human readable outputs
    primary_cause: str                   # one sentence
    supporting_evidence: list[str]       # bullet points
    recommended_action: str              # what to do
    clinical_context: str                # broader picture

    # Technical details
    contributing_artifacts: list[ArtifactType] = field(
        default_factory=list
    )
    context_factors: list[str] = field(default_factory=list)

    # Is this a false alarm?
    likely_false_alarm: bool = False
    false_alarm_confidence: float = 0.0


# ── Attribution Engine ────────────────────────────────────────
class AttributionEngine:
    """
    Infers the most probable cause of signal unreliability.

    Uses a rule-based expert system approach.
    Why not pure ML here?

    1. Explainability requirement - rules are auditable
    2. Data scarcity - labeled failure causes are rare
    3. Safety - deterministic rules are predictable
    4. IEC 62304 - rule logic is formally verifiable

    The rules encode clinical knowledge about device
    failure modes documented in medical device literature
    and FDA MAUDE adverse event database patterns.
    """

    def attribute(
        self,
        reliability_score: ReliabilityScore,
        sqi_result: Optional[SQIResult] = None,
        context_features: Optional[ContextFeatures] = None,
    ) -> AttributionResult:
        """
        Determine the most probable cause of signal unreliability.

        Args:
            reliability_score: Final trust score with recommendation
            sqi_result: Signal quality assessment
            context_features: Contextual features

        Returns:
            AttributionResult with plain English explanation
        """
        # Use embedded results if not provided separately
        if sqi_result is None:
            sqi_result = reliability_score.sqi_result
        if context_features is None:
            context_features = reliability_score.context_features

        # If signal is trustworthy - minimal attribution needed
        if reliability_score.is_trustworthy:
            return self._build_clean_attribution(
                reliability_score, sqi_result
            )

        # Run attribution hierarchy
        result = (
            self._check_sensor_displacement(
                reliability_score, sqi_result, context_features
            )
            or self._check_motion_artifact(
                reliability_score, sqi_result, context_features
            )
            or self._check_device_malfunction(
                reliability_score, sqi_result, context_features
            )
            or self._check_environmental(
                reliability_score, sqi_result, context_features
            )
            or self._check_calibration_drift(
                reliability_score, sqi_result, context_features
            )
            or self._build_unknown_attribution(
                reliability_score, sqi_result, context_features
            )
        )

        logger.info(
            f"Attribution | "
            f"{reliability_score.signal_type.value} | "
            f"category={result.failure_category.value} | "
            f"confidence={result.confidence:.2f} | "
            f"false_alarm={result.likely_false_alarm} | "
            f"{result.primary_cause}"
        )

        return result

    # ── Attribution Checks ────────────────────────────────────

    def _check_sensor_displacement(
        self,
        score: ReliabilityScore,
        sqi: Optional[SQIResult],
        ctx: Optional[ContextFeatures],
    ) -> Optional[AttributionResult]:
        """
        Sensor displacement: probe physically moved or detached.

        Signature:
        - Dropout or flatline detected
        - Recent clinical event (repositioning, staff intervention)
        - OR significant motion
        - Neighboring signals still clean (patient is fine)
        """
        if sqi is None:
            return None

        has_dropout = ArtifactType.DROPOUT in (
            sqi.artifacts_detected or []
        )
        has_flatline = ArtifactType.FLATLINE in (
            sqi.artifacts_detected or []
        )

        if not (has_dropout or has_flatline):
            return None

        evidence = []
        confidence = 0.5

        if has_dropout:
            evidence.append("Signal dropout detected - complete loss of sensor data")
            confidence += 0.15
        if has_flatline:
            evidence.append("Flatline detected - signal not responding to patient state")
            confidence += 0.1

        if ctx is not None:
            if ctx.recent_clinical_event and ctx.event_known_artifact_cause:
                evidence.append(
                    f"Clinical event {ctx.seconds_since_event:.0f}s ago "
                    f"(type known to cause sensor displacement)"
                )
                confidence += 0.2

            if ctx.significant_motion:
                evidence.append(
                    f"Significant patient motion detected "
                    f"(motion index={ctx.motion_index:.2f})"
                )
                confidence += 0.15

            # Neighboring signals clean = patient is stable
            # Strengthens displacement theory
            if ctx.neighboring_avg_sqi > 80:
                evidence.append(
                    "Neighboring signals remain stable - "
                    "suggests isolated sensor issue, not patient event"
                )
                confidence += 0.1

        if confidence < 0.6:
            return None

        likely_false_alarm = (
            ctx is not None and ctx.neighboring_avg_sqi > 80
        )

        return AttributionResult(
            failure_category=FailureCategory.SENSOR_DISPLACEMENT,
            confidence=min(0.95, confidence),
            primary_cause=(
                f"Likely sensor displacement - "
                f"{'probe detached' if has_dropout else 'probe contact lost'} "
                f"{'following patient movement' if ctx and ctx.significant_motion else ''}"
            ),
            supporting_evidence=evidence,
            recommended_action=(
                "Physically inspect and reseat the sensor. "
                "Verify probe position before interpreting readings."
            ),
            clinical_context=(
                "Neighboring signals suggest patient condition "
                "is unchanged. This appears to be a device issue."
                if likely_false_alarm
                else "Verify patient condition independently."
            ),
            contributing_artifacts=sqi.artifacts_detected or [],
            likely_false_alarm=likely_false_alarm,
            false_alarm_confidence=min(0.9, confidence),
        )

    def _check_motion_artifact(
        self,
        score: ReliabilityScore,
        sqi: Optional[SQIResult],
        ctx: Optional[ContextFeatures],
    ) -> Optional[AttributionResult]:
        """
        Motion artifact: patient movement corrupting signal.

        Signature:
        - Noise or spike detected
        - High motion index
        - Multiple signals affected simultaneously
        - Transient (brief) degradation
        """
        if ctx is None or not ctx.significant_motion:
            return None

        if sqi is None:
            return None

        has_noise = ArtifactType.NOISE in (sqi.artifacts_detected or [])
        has_spike = ArtifactType.SPIKE in (sqi.artifacts_detected or [])

        if not (has_noise or has_spike):
            return None

        evidence = []
        confidence = 0.55

        evidence.append(
            f"High motion index detected ({ctx.motion_index:.2f}/1.0)"
        )
        confidence += ctx.motion_index * 0.2

        if has_noise:
            evidence.append("High frequency noise consistent with movement artifact")
            confidence += 0.1
        if has_spike:
            evidence.append("Spike artifacts consistent with sudden movement")
            confidence += 0.1

        if ctx.multi_signal_simultaneous_degradation:
            evidence.append(
                "Multiple signals degraded simultaneously - "
                "consistent with whole-body movement"
            )
            confidence += 0.1

        if ctx.recent_clinical_event:
            evidence.append(
                f"Clinical event {ctx.seconds_since_event:.0f}s ago "
                f"correlates with artifact onset"
            )
            confidence += 0.1

        return AttributionResult(
            failure_category=FailureCategory.MOTION_ARTIFACT,
            confidence=min(0.92, confidence),
            primary_cause=(
                "Motion artifact - patient movement is corrupting "
                "the signal. Reading is likely not representative "
                "of true physiological state."
            ),
            supporting_evidence=evidence,
            recommended_action=(
                "Wait for patient to settle. "
                "Re-assess signal quality after movement stops. "
                "Do not act on current reading."
            ),
            clinical_context=(
                "Motion artifacts are transient. Signal quality "
                "typically recovers within 10-30 seconds after "
                "movement ceases."
            ),
            contributing_artifacts=sqi.artifacts_detected or [],
            likely_false_alarm=True,
            false_alarm_confidence=min(0.88, confidence),
        )

    def _check_device_malfunction(
        self,
        score: ReliabilityScore,
        sqi: Optional[SQIResult],
        ctx: Optional[ContextFeatures],
    ) -> Optional[AttributionResult]:
        """
        Device malfunction: hardware or firmware issue.

        Signature:
        - Flatline or dropout WITHOUT motion or clinical event
        - No neighboring signal degradation
        - Session is relatively new (drift unlikely)
        """
        if sqi is None:
            return None

        has_dropout = ArtifactType.DROPOUT in (sqi.artifacts_detected or [])
        has_flatline = ArtifactType.FLATLINE in (sqi.artifacts_detected or [])

        if not (has_dropout or has_flatline):
            return None

        # If motion or clinical event explains it, not malfunction
        if ctx is not None:
            if ctx.significant_motion or ctx.recent_clinical_event:
                return None

        evidence = []
        confidence = 0.5

        if has_dropout:
            evidence.append("Unexplained signal dropout - no environmental cause identified")
            confidence += 0.2
        if has_flatline:
            evidence.append("Signal flatlined without physiological explanation")
            confidence += 0.15

        if ctx is not None and ctx.neighboring_avg_sqi > 75:
            evidence.append(
                "All neighboring signals functioning normally - "
                "isolates issue to this specific device"
            )
            confidence += 0.15

        if confidence < 0.6:
            return None

        return AttributionResult(
            failure_category=FailureCategory.DEVICE_MALFUNCTION,
            confidence=min(0.88, confidence),
            primary_cause=(
                "Possible device malfunction - signal failure "
                "without identified environmental or patient cause."
            ),
            supporting_evidence=evidence,
            recommended_action=(
                "Replace or restart the monitoring device. "
                "Switch to backup monitoring if available. "
                "Document device behavior for biomedical engineering."
            ),
            clinical_context=(
                "Other monitoring signals are functioning normally. "
                "Patient status should be verified through "
                "alternative means."
            ),
            contributing_artifacts=sqi.artifacts_detected or [],
            likely_false_alarm=True,
            false_alarm_confidence=0.7,
        )

    def _check_environmental(
        self,
        score: ReliabilityScore,
        sqi: Optional[SQIResult],
        ctx: Optional[ContextFeatures],
    ) -> Optional[AttributionResult]:
        """
        Environmental interference: EMI, electrical noise.

        Signature:
        - Noise detected
        - Multiple signals affected
        - No motion
        - No clinical events
        """
        if sqi is None or ctx is None:
            return None

        has_noise = ArtifactType.NOISE in (sqi.artifacts_detected or [])
        if not has_noise:
            return None

        if ctx.significant_motion:
            return None

        if not ctx.multi_signal_simultaneous_degradation:
            return None

        evidence = [
            "Noise artifact without patient movement",
            "Multiple signals affected simultaneously - "
            "suggests common environmental source",
        ]
        confidence = 0.60

        if ctx.neighboring_degradation_ratio > 0.7:
            evidence.append(
                f"{ctx.neighboring_degradation_ratio:.0%} of "
                f"neighboring signals affected"
            )
            confidence += 0.15

        return AttributionResult(
            failure_category=FailureCategory.ENVIRONMENTAL,
            confidence=min(0.82, confidence),
            primary_cause=(
                "Environmental electrical interference - "
                "multiple signals affected by common noise source."
            ),
            supporting_evidence=evidence,
            recommended_action=(
                "Check for nearby electrical equipment. "
                "Inspect electrode connections and grounding. "
                "Consider moving patient away from interference source."
            ),
            clinical_context=(
                "Simultaneous multi-signal degradation without "
                "patient movement typically indicates environmental EMI "
                "rather than a clinical event."
            ),
            contributing_artifacts=sqi.artifacts_detected or [],
            likely_false_alarm=True,
            false_alarm_confidence=0.72,
        )

    def _check_calibration_drift(
        self,
        score: ReliabilityScore,
        sqi: Optional[SQIResult],
        ctx: Optional[ContextFeatures],
    ) -> Optional[AttributionResult]:
        """
        Calibration drift: gradual sensor degradation over time.

        Signature:
        - Negative reliability trend
        - Long session age
        - Gradual (not sudden) degradation
        - No acute events
        """
        if ctx is None:
            return None

        is_drifting = (
            ctx.reliability_trend < -0.2
            and ctx.session_age_seconds > 3600
            and ctx.recent_clean_ratio < 0.5
        )

        if not is_drifting:
            return None

        evidence = [
            f"Negative reliability trend ({ctx.reliability_trend:.3f}) - "
            f"signal quality degrading over time",
            f"Session age {ctx.session_age_seconds/3600:.1f} hours - "
            f"long monitoring duration",
            f"Only {ctx.recent_clean_ratio:.0%} of recent windows were clean",
        ]
        confidence = 0.65

        if ctx.session_age_seconds > 7200:
            evidence.append("Extended session duration increases drift probability")
            confidence += 0.1

        return AttributionResult(
            failure_category=FailureCategory.CALIBRATION_DRIFT,
            confidence=min(0.85, confidence),
            primary_cause=(
                "Calibration drift - sensor reliability has been "
                "gradually degrading over this monitoring session."
            ),
            supporting_evidence=evidence,
            recommended_action=(
                "Replace sensor or recalibrate device. "
                "Consider ending current session and "
                "starting fresh with new equipment."
            ),
            clinical_context=(
                "Gradual drift is common in long monitoring sessions. "
                "This is a device maintenance issue, not a "
                "patient clinical event."
            ),
            contributing_artifacts=sqi.artifacts_detected or [],
            likely_false_alarm=True,
            false_alarm_confidence=0.70,
        )

    def _build_unknown_attribution(
        self,
        score: ReliabilityScore,
        sqi: Optional[SQIResult],
        ctx: Optional[ContextFeatures],
    ) -> AttributionResult:
        """Fallback when no specific cause identified."""

        artifacts = sqi.artifacts_detected if sqi else []
        artifact_names = [a.value for a in artifacts]

        return AttributionResult(
            failure_category=FailureCategory.UNKNOWN,
            confidence=0.3,
            primary_cause=(
                f"Signal unreliability - cause undetermined. "
                f"Detected: {', '.join(artifact_names) if artifact_names else 'general degradation'}."
            ),
            supporting_evidence=[
                f"Trust score: {score.trust_score:.0f}/100",
                f"Artifacts detected: {', '.join(artifact_names) if artifact_names else 'none specific'}",
                "Insufficient context to determine specific cause",
            ],
            recommended_action=(
                "Verify signal manually. "
                "Check sensor placement and device status. "
                "Do not act on this reading without verification."
            ),
            clinical_context=(
                "CRIP-X cannot confidently attribute this failure. "
                "Manual assessment recommended."
            ),
            contributing_artifacts=artifacts,
            likely_false_alarm=False,
            false_alarm_confidence=0.3,
        )

    def _build_clean_attribution(
        self,
        score: ReliabilityScore,
        sqi: Optional[SQIResult],
    ) -> AttributionResult:
        """Attribution for trustworthy signals."""
        return AttributionResult(
            failure_category=FailureCategory.UNKNOWN,
            confidence=score.confidence,
            primary_cause="Signal is reliable - no failure attribution required.",
            supporting_evidence=[
                f"Trust score: {score.trust_score:.0f}/100",
                f"SQI: {score.sqi_score:.0f}/100",
                "No significant artifacts detected",
            ],
            recommended_action="No action required - signal is trustworthy.",
            clinical_context="Signal quality is within acceptable parameters.",
            contributing_artifacts=[],
            likely_false_alarm=False,
            false_alarm_confidence=0.0,
        )