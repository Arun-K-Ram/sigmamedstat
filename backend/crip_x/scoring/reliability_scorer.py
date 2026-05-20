"""
CRIP-X Reliability Scorer

Combines Signal Quality Index + Context Features into
a final trust score for a medical device signal reading.

This is where SQI and context meet.

SQI answers:     "Is the signal technically valid?"
Context answers: "Does the signal make sense right now?"
Scorer answers:  "Should we trust this reading?"

The final trust score is what the dashboard displays,
what the attribution engine explains, and what the
drift monitor tracks over time.

Output: ReliabilityScore with:
    trust_score:    0-100 final score
    confidence:     how certain CRIP-X is about the score
    interpretation: plain English explanation
    recommendation: what to do with this reading
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import numpy as np

from crip_x.signal.signal_quality_index import SQIResult
from crip_x.context.feature_extractor import ContextFeatures
from crip_x.utils.validators import SignalType
from crip_x.utils.config import settings
from crip_x.utils.logger import get_logger

logger = get_logger(__name__)


# ── Recommendation ────────────────────────────────────────────
class Recommendation(Enum):
    """
    Clinical recommendation based on trust score.

    These map directly to what a clinical system should do
    with this reading. Designed to align with alarm
    management best practices in IEC 60601-1-8.
    """
    TRUST = "trust"                   # Use this reading normally
    CAUTION = "caution"               # Use but verify with other signals
    VERIFY = "verify"                 # Do not act on alone - verify first
    DISCARD = "discard"               # Do not use this reading
    RESEAT_SENSOR = "reseat_sensor"   # Likely physical sensor issue


# ── Reliability Score ─────────────────────────────────────────
@dataclass
class ReliabilityScore:
    """
    Final output of the CRIP-X reliability pipeline.

    This is what flows to:
    - The dashboard (displayed as trust gauge)
    - The attribution engine (explained in plain English)
    - The drift monitor (tracked over time)
    - The API (returned to clients)
    """
    # Core scores
    trust_score: float          # 0-100 final trust score
    confidence: float           # 0-1 how certain we are
    sqi_score: float            # raw signal quality
    signal_type: SignalType

    # Decision
    recommendation: Recommendation
    is_trustworthy: bool        # simple boolean for downstream

    # Context influence
    context_adjusted: bool      # did context change the score?
    context_delta: float        # how much context changed score

    # Human readable
    interpretation: str         # one sentence explanation
    recommendation_reason: str  # why this recommendation

    # Raw inputs preserved for attribution
    sqi_result: Optional[SQIResult] = None
    context_features: Optional[ContextFeatures] = None


# ── Reliability Scorer ────────────────────────────────────────
class ReliabilityScorer:
    """
    Fuses SQI and context into a final trust score.

    Scoring Algorithm:

    Step 1 - Base score from SQI
        SQI is the technical signal quality.
        It becomes the starting point.

    Step 2 - Context adjustment
        Context can push the score up or down.

        Push DOWN (artifact likely):
            - High motion detected
            - Recent repositioning event
            - Multi-signal simultaneous degradation
            → These suggest the issue is an artifact,
              not a real clinical event. Score decreases
              because the reading is less trustworthy.

        Push UP (context supports signal):
            - All neighbors clean and stable
            - Positive reliability trend
            - No recent events or motion
            → These increase confidence in the reading.

    Step 3 - Confidence calculation
        How certain is CRIP-X about this score?
        Low confidence = borderline cases.
        High confidence = clear clean or clear artifact.

    Step 4 - Recommendation
        Maps final trust score to clinical action.
    """

    # Context adjustment limits
    MAX_CONTEXT_PENALTY = 20.0    # context can reduce score by max 20
    MAX_CONTEXT_BONUS = 10.0      # context can increase score by max 10

    # Trust score thresholds for recommendations
    TRUST_THRESHOLD = 80.0
    CAUTION_THRESHOLD = 60.0
    VERIFY_THRESHOLD = 40.0

    def score(
        self,
        sqi_result: SQIResult,
        context_features: Optional[ContextFeatures] = None,
    ) -> ReliabilityScore:
        """
        Compute final reliability score.

        Args:
            sqi_result: Output from SignalQualityIndex
            context_features: Output from ContextFeatureExtractor
                              (optional - degrades gracefully)

        Returns:
            ReliabilityScore - the final CRIP-X output
        """

        # ── Step 1: Base Score from SQI ───────────────────────
        base_score = sqi_result.sqi_score

        # ── Step 2: Context Adjustment ────────────────────────
        context_delta = 0.0
        context_adjusted = False

        if context_features is not None:
            context_delta = self._compute_context_delta(
                sqi_result, context_features
            )
            context_adjusted = abs(context_delta) > 0.5

        # Apply context adjustment
        adjusted_score = float(
            np.clip(base_score + context_delta, 0.0, 100.0)
        )

        # ── Step 3: Confidence ────────────────────────────────
        confidence = self._compute_confidence(
            adjusted_score, sqi_result, context_features
        )

        # ── Step 4: Recommendation ────────────────────────────
        recommendation, reason = self._get_recommendation(
            adjusted_score, sqi_result, context_features
        )

        is_trustworthy = adjusted_score >= self.CAUTION_THRESHOLD

        # ── Step 5: Interpretation ────────────────────────────
        interpretation = self._build_interpretation(
            adjusted_score, sqi_result, context_features, context_delta
        )

        result = ReliabilityScore(
            trust_score=round(adjusted_score, 1),
            confidence=round(confidence, 3),
            sqi_score=sqi_result.sqi_score,
            signal_type=sqi_result.signal_type,
            recommendation=recommendation,
            is_trustworthy=is_trustworthy,
            context_adjusted=context_adjusted,
            context_delta=round(context_delta, 1),
            interpretation=interpretation,
            recommendation_reason=reason,
            sqi_result=sqi_result,
            context_features=context_features,
        )

        log_level = "warning" if not is_trustworthy else "info"
        getattr(logger, log_level)(
            f"{sqi_result.signal_type.value.upper()} | "
            f"trust={adjusted_score:.0f} | "
            f"sqi={base_score:.0f} | "
            f"context_delta={context_delta:+.1f} | "
            f"confidence={confidence:.2f} | "
            f"rec={recommendation.value} | "
            f"{interpretation}"
        )

        return result

    def _compute_context_delta(
        self,
        sqi_result: SQIResult,
        features: ContextFeatures,
    ) -> float:
        """
        Compute how much context adjusts the base SQI score.

        Negative = context suggests reading is less trustworthy
        Positive = context supports reading trustworthiness
        """
        delta = 0.0

        # ── Penalties (context suggests artifact) ─────────────

        # High artifact probability from context
        # Strongest single context signal
        if features.context_artifact_probability > 0:
            penalty = (
                features.context_artifact_probability
                * self.MAX_CONTEXT_PENALTY
            )
            delta -= penalty
            logger.debug(
                f"Context penalty: artifact_prob "
                f"{features.context_artifact_probability:.2f} "
                f"→ -{penalty:.1f}"
            )

        # Degrading reliability trend
        if features.reliability_trend < -0.2:
            trend_penalty = abs(features.reliability_trend) * 5.0
            delta -= min(trend_penalty, 8.0)

        # ── Bonuses (context supports reading) ───────────────

        # Context reliability bonus
        if features.context_reliability_bonus > 0:
            bonus = (
                features.context_reliability_bonus
                * self.MAX_CONTEXT_BONUS
            )
            delta += bonus

        # Clip to max adjustment limits
        delta = float(
            np.clip(delta, -self.MAX_CONTEXT_PENALTY, self.MAX_CONTEXT_BONUS)
        )

        return delta

    def _compute_confidence(
        self,
        trust_score: float,
        sqi_result: SQIResult,
        features: Optional[ContextFeatures],
    ) -> float:
        """
        How confident is CRIP-X in this trust score?

        High confidence = score is clearly high or clearly low
        Low confidence = borderline cases
        """
        # Distance from the 50-point midpoint
        # Scores near 0 or 100 = high confidence
        # Scores near 50 = low confidence
        distance_from_midpoint = abs(trust_score - 50) / 50.0
        base_confidence = 0.5 + (distance_from_midpoint * 0.5)

        # Context availability increases confidence
        if features is not None:
            base_confidence = min(0.99, base_confidence + 0.1)

        # Multiple artifacts detected = more confident it's bad
        if sqi_result.artifacts_detected:
            n_artifacts = len(sqi_result.artifacts_detected)
            base_confidence = min(
                0.99,
                base_confidence + (n_artifacts * 0.05)
            )

        return float(np.clip(base_confidence, 0.1, 0.99))

    def _get_recommendation(
        self,
        trust_score: float,
        sqi_result: SQIResult,
        features: Optional[ContextFeatures],
    ) -> tuple[Recommendation, str]:
        """Map trust score to clinical recommendation."""

        # Special case: sensor likely physically displaced
        if (features is not None
                and features.significant_motion
                and trust_score < self.CAUTION_THRESHOLD):
            return (
                Recommendation.RESEAT_SENSOR,
                "High motion with low trust score suggests "
                "physical sensor displacement"
            )

        if trust_score >= self.TRUST_THRESHOLD:
            return (
                Recommendation.TRUST,
                f"Trust score {trust_score:.0f} exceeds "
                f"confidence threshold {self.TRUST_THRESHOLD:.0f}"
            )
        elif trust_score >= self.CAUTION_THRESHOLD:
            return (
                Recommendation.CAUTION,
                f"Trust score {trust_score:.0f} is acceptable "
                f"but verify with neighboring signals"
            )
        elif trust_score >= self.VERIFY_THRESHOLD:
            return (
                Recommendation.VERIFY,
                f"Trust score {trust_score:.0f} is too low "
                f"to act on without verification"
            )
        else:
            return (
                Recommendation.DISCARD,
                f"Trust score {trust_score:.0f} indicates "
                f"signal is unreliable - do not use"
            )

    def _build_interpretation(
        self,
        trust_score: float,
        sqi_result: SQIResult,
        features: Optional[ContextFeatures],
        context_delta: float,
    ) -> str:
        """Build a plain English interpretation."""

        parts = []

        # Signal quality part
        if not sqi_result.artifacts_detected:
            parts.append("Signal is clean")
        else:
            artifact_names = [
                a.value for a in sqi_result.artifacts_detected
            ]
            parts.append(
                f"Signal has {', '.join(artifact_names)}"
            )

        # Context part
        if features is not None:
            if features.significant_motion:
                parts.append("high patient motion detected")
            if (features.recent_clinical_event
                    and features.event_known_artifact_cause):
                parts.append(
                    f"clinical event {features.seconds_since_event:.0f}s ago"
                )
            if features.multi_signal_simultaneous_degradation:
                parts.append("multiple signals affected simultaneously")

        # Context adjustment part
        if abs(context_delta) > 1.0:
            direction = "reduced" if context_delta < 0 else "increased"
            parts.append(
                f"context {direction} trust by {abs(context_delta):.0f} points"
            )

        return " - ".join(parts) if parts else "No issues detected"