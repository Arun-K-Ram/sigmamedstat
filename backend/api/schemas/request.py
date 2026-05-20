"""
CRIP-X API Request Schemas

Defines what the API accepts as input.
Pydantic validates incoming JSON automatically.
Invalid requests are rejected before touching the pipeline.
"""

from typing import Optional
from pydantic import BaseModel, Field, field_validator


class ClinicalEventRequest(BaseModel):
    event_type: str
    timestamp_offset_seconds: float = 0.0
    description: str = ""


class AnalyzeRequest(BaseModel):
    """
    Request body for POST /analyze

    The React frontend sends this when analyzing a signal window.
    """
    signal: list[Optional[float]] = Field(
        ...,
        description="Signal array — use null for missing samples",
        min_length=10,
        max_length=10000,
    )
    signal_type: str = Field(
        ...,
        description="Signal type: spo2, heart_rate, ecg, respiratory_rate"
    )
    session_id: str = Field(
        ...,
        description="Unique session identifier for drift tracking"
    )

    # Optional context
    neighboring_signals: dict[str, list[Optional[float]]] = Field(
        default_factory=dict,
        description="Other signals at the same timestamp"
    )
    historical_sqi: list[float] = Field(
        default_factory=list,
        description="Previous SQI scores for this session, most recent first"
    )
    clinical_events: list[ClinicalEventRequest] = Field(
        default_factory=list,
        description="Recent clinical events"
    )
    motion_signal: Optional[list[float]] = Field(
        default=None,
        description="Accelerometer signal if available"
    )
    session_age_seconds: float = Field(
        default=0.0,
        description="How long this monitoring session has been running"
    )

    @field_validator("signal_type")
    @classmethod
    def validate_signal_type(cls, v: str) -> str:
        valid = {"spo2", "heart_rate", "ecg", "respiratory_rate",
                 "abp_systolic", "abp_diastolic", "temperature"}
        if v not in valid:
            raise ValueError(f"signal_type must be one of {valid}")
        return v


class FixtureRunRequest(BaseModel):
    """Request to run a specific fixture through the pipeline."""
    fixture_id: str