"""
CRIP-X API Response Schemas

Defines what the API returns.
These are what the React frontend consumes.
"""

from typing import Optional
from pydantic import BaseModel


class ArtifactDetail(BaseModel):
    artifact_type: str
    confidence: float
    severity: float
    affected_ratio: float
    message: str


class AttributionDetail(BaseModel):
    failure_category: str
    confidence: float
    primary_cause: str
    supporting_evidence: list[str]
    recommended_action: str
    clinical_context: str
    likely_false_alarm: bool
    false_alarm_confidence: float


class DriftDetail(BaseModel):
    drift_detected: bool
    drift_severity: str
    trend_slope: float
    trend_direction: str
    session_age_seconds: float
    current_sqi: float
    baseline_sqi: float
    sqi_delta: float
    estimated_minutes_to_critical: Optional[float]


class AnalyzeResponse(BaseModel):
    """
    Full pipeline output returned to React frontend.

    This is the single most important data structure
    in the entire system — everything the dashboard
    needs to render is in here.
    """
    # Core scores
    trust_score: float
    sqi_score: float
    grade: str
    confidence: float
    is_trustworthy: bool

    # Signal info
    signal_type: str
    n_samples: int
    session_id: str

    # Recommendation
    recommendation: str
    interpretation: str

    # Artifacts
    artifacts_detected: list[str]
    dominant_artifact: Optional[str]
    artifact_details: dict[str, ArtifactDetail]

    # Attribution
    attribution: AttributionDetail

    # Context
    context_adjusted: bool
    context_delta: float

    # Drift
    drift: Optional[DriftDetail]

    # Meta
    processing_time_ms: float


class FixtureSummary(BaseModel):
    fixture_id: str
    description: str
    signal_type: str
    n_samples: int
    has_clinical_events: bool
    has_motion_signal: bool
    expected_outcome: dict


class FixtureListResponse(BaseModel):
    fixtures: list[FixtureSummary]
    total: int


class HealthResponse(BaseModel):
    status: str
    version: str
    pipeline_ready: bool