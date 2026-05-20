"""
CRIP-X Reliability Router

POST /analyze          → run full pipeline on signal window
POST /analyze/fixture  → run a fixture through the pipeline
"""

import time
import numpy as np
from fastapi import APIRouter, HTTPException
from typing import Optional

from crip_x.signal.signal_quality_index import SignalQualityIndex
from crip_x.context.feature_extractor import (
    ContextFeatureExtractor, ContextInput, ClinicalEvent
)
from crip_x.scoring.reliability_scorer import ReliabilityScorer
from crip_x.attribution.attribute_engine import AttributionEngine
from crip_x.drift.drift_tracker import DriftTracker
from crip_x.utils.validators import SignalType
from crip_x.ingestion.fixture_loader import (
    load_fixture, get_signal_array,
    get_neighboring_arrays, get_motion_array
)
from api.schemas.request import AnalyzeRequest
from api.schemas.response import (
    AnalyzeResponse, ArtifactDetail, AttributionDetail, DriftDetail
)
from crip_x.utils.logger import get_logger

router = APIRouter(prefix="/analyze", tags=["reliability"])
logger = get_logger(__name__)

# ── Pipeline Singletons ───────────────────────────────────────
# Initialized once at startup - not per request
sqi_engine = SignalQualityIndex()
context_extractor = ContextFeatureExtractor()
scorer = ReliabilityScorer()
attribution_engine = AttributionEngine()
drift_tracker = DriftTracker()


def _parse_signal_type(signal_type_str: str) -> SignalType:
    """Convert string to SignalType enum."""
    mapping = {
        "spo2": SignalType.SPO2,
        "heart_rate": SignalType.HEART_RATE,
        "ecg": SignalType.ECG,
        "respiratory_rate": SignalType.RESPIRATORY_RATE,
        "abp_systolic": SignalType.ABP_SYSTOLIC,
        "abp_diastolic": SignalType.ABP_DIASTOLIC,
        "temperature": SignalType.TEMPERATURE,
    }
    if signal_type_str not in mapping:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown signal type: {signal_type_str}"
        )
    return mapping[signal_type_str]


def _run_pipeline(
    signal: np.ndarray,
    signal_type: SignalType,
    session_id: str,
    neighboring_signals: dict = None,
    historical_sqi: list = None,
    clinical_events: list = None,
    motion_signal: np.ndarray = None,
    session_age_seconds: float = 0.0,
) -> AnalyzeResponse:
    """Run the full CRIP-X pipeline and return response."""
    start_time = time.time()

    # ── Step 1: SQI ───────────────────────────────────────────
    sqi_result = sqi_engine.compute(signal, signal_type)

    # ── Step 2: Neighboring SQIs ──────────────────────────────
    neighbor_sqis = {}
    if neighboring_signals:
        for sig_name, sig_values in neighboring_signals.items():
            try:
                neighbor_type = _parse_signal_type(sig_name)
                neighbor_array = np.array(
                    [np.nan if v is None else float(v)
                     for v in sig_values],
                    dtype=np.float64
                )
                neighbor_sqis[neighbor_type] = sqi_engine.compute(
                    neighbor_array, neighbor_type
                )
            except Exception as e:
                logger.warning(f"Could not process neighbor {sig_name}: {e}")

    # ── Step 3: Context ───────────────────────────────────────
    now = time.time()
    events = []
    if clinical_events:
        for event in clinical_events:
            events.append(ClinicalEvent(
                event_type=event.event_type
                if hasattr(event, 'event_type')
                else event.get('event_type', ''),
                timestamp=now + (
                    event.timestamp_offset_seconds
                    if hasattr(event, 'timestamp_offset_seconds')
                    else event.get('timestamp_offset_seconds', 0)
                ),
                description=event.description
                if hasattr(event, 'description')
                else event.get('description', ''),
            ))

    context_input = ContextInput(
        primary_signal_type=signal_type,
        primary_sqi=sqi_result,
        current_timestamp=now,
        neighboring_sqis=neighbor_sqis,
        session_start_timestamp=now - session_age_seconds,
        historical_sqis=historical_sqi or [],
        recent_events=events,
        motion_signal=motion_signal,
    )
    context_features = context_extractor.extract(context_input)

    # ── Step 4: Score ─────────────────────────────────────────
    reliability = scorer.score(sqi_result, context_features)

    # ── Step 5: Attribution ───────────────────────────────────
    attribution = attribution_engine.attribute(reliability)

    # ── Step 6: Drift ─────────────────────────────────────────
    drift_tracker.update(
        session_id=session_id,
        signal_type=signal_type,
        sqi_score=sqi_result.sqi_score,
    )
    drift_result = drift_tracker.assess(session_id, signal_type)

    # ── Build Response ────────────────────────────────────────
    processing_time_ms = (time.time() - start_time) * 1000

    # Artifact details
    artifact_details = {}
    for detector_result in [
        sqi_result.flatline_result,
        sqi_result.spike_result,
        sqi_result.dropout_result,
        sqi_result.noise_result,
    ]:
        if detector_result and detector_result.artifact_detected:
            artifact_details[detector_result.artifact_type.value] = (
                ArtifactDetail(
                    artifact_type=detector_result.artifact_type.value,
                    confidence=detector_result.confidence,
                    severity=detector_result.severity,
                    affected_ratio=detector_result.affected_ratio,
                    message=detector_result.message,
                )
            )

    drift_detail = None
    if drift_result:
        drift_detail = DriftDetail(
            drift_detected=drift_result.drift_detected,
            drift_severity=drift_result.drift_severity,
            trend_slope=drift_result.trend_slope,
            trend_direction=drift_result.trend_direction,
            session_age_seconds=drift_result.session_age_seconds,
            current_sqi=drift_result.current_sqi,
            baseline_sqi=drift_result.baseline_sqi,
            sqi_delta=drift_result.sqi_delta,
            estimated_minutes_to_critical=(
                drift_result.estimated_minutes_to_critical
            ),
        )

    return AnalyzeResponse(
        trust_score=reliability.trust_score,
        sqi_score=reliability.sqi_score,
        grade=sqi_result.grade,
        confidence=reliability.confidence,
        is_trustworthy=reliability.is_trustworthy,
        signal_type=signal_type.value,
        n_samples=len(signal),
        session_id=session_id,
        recommendation=reliability.recommendation.value,
        interpretation=reliability.interpretation,
        artifacts_detected=[
            a.value for a in sqi_result.artifacts_detected
        ],
        dominant_artifact=(
            sqi_result.dominant_artifact.value
            if sqi_result.dominant_artifact else None
        ),
        artifact_details=artifact_details,
        attribution=AttributionDetail(
            failure_category=attribution.failure_category.value,
            confidence=attribution.confidence,
            primary_cause=attribution.primary_cause,
            supporting_evidence=attribution.supporting_evidence,
            recommended_action=attribution.recommended_action,
            clinical_context=attribution.clinical_context,
            likely_false_alarm=attribution.likely_false_alarm,
            false_alarm_confidence=attribution.false_alarm_confidence,
        ),
        context_adjusted=reliability.context_adjusted,
        context_delta=reliability.context_delta,
        drift=drift_detail,
        processing_time_ms=round(processing_time_ms, 2),
    )


@router.post("", response_model=AnalyzeResponse)
async def analyze_signal(request: AnalyzeRequest):
    """Run full CRIP-X pipeline on a signal window."""
    signal_type = _parse_signal_type(request.signal_type)

    signal = np.array(
        [np.nan if v is None else float(v) for v in request.signal],
        dtype=np.float64
    )

    motion = None
    if request.motion_signal:
        motion = np.array(request.motion_signal, dtype=np.float64)

    return _run_pipeline(
        signal=signal,
        signal_type=signal_type,
        session_id=request.session_id,
        neighboring_signals=request.neighboring_signals,
        historical_sqi=request.historical_sqi,
        clinical_events=request.clinical_events,
        motion_signal=motion,
        session_age_seconds=request.session_age_seconds,
    )


@router.post("/fixture/{fixture_id}", response_model=AnalyzeResponse)
async def analyze_fixture(fixture_id: str):
    """Run a predefined fixture through the full pipeline."""
    from crip_x.ingestion.fixture_loader import list_fixtures

    fixture_files = list_fixtures()
    fixture = None

    for filename in fixture_files:
        try:
            f = load_fixture(filename)
            if f["fixture_id"] == fixture_id:
                fixture = f
                break
        except Exception:
            continue

    if not fixture:
        raise HTTPException(
            status_code=404,
            detail=f"Fixture '{fixture_id}' not found"
        )

    signal = get_signal_array(fixture)
    neighbors = get_neighboring_arrays(fixture)
    motion = get_motion_array(fixture)

    signal_type = _parse_signal_type(fixture["signal_type"])

    return _run_pipeline(
        signal=signal,
        signal_type=signal_type,
        session_id=fixture["clinical_context"]["session_id"],
        neighboring_signals={
            k: v.tolist() for k, v in neighbors.items()
        },
        historical_sqi=fixture.get("historical_sqi", []),
        clinical_events=fixture.get("clinical_events", []),
        motion_signal=motion,
        session_age_seconds=fixture["clinical_context"].get(
            "session_age_seconds", 0
        ),
    )