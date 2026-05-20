"""
CRIP-X Temporal Drift Monitor

Tracks signal reliability degradation over time.

While all other components analyze a single window,
the drift tracker looks at the entire session history
and asks: "Is this device getting worse over time?"

This catches slow degradation that window-level analysis
misses entirely - a sensor that goes from SQI=95 to
SQI=60 over 2 hours looks fine at any single moment
but the trend tells the real story.
"""

from dataclasses import dataclass, field
from typing import Optional
from collections import deque
import numpy as np
import time

from crip_x.utils.validators import SignalType
from crip_x.utils.logger import get_logger

logger = get_logger(__name__)


# ── Drift Severity ────────────────────────────────────────────
class DriftSeverity:
    NONE     = "none"
    MILD     = "mild"
    MODERATE = "moderate"
    SEVERE   = "severe"


# ── Drift Result ──────────────────────────────────────────────
@dataclass
class DriftResult:
    """
    Current drift assessment for a signal session.
    """
    signal_type: SignalType
    session_id: str

    # Trend
    drift_detected: bool
    drift_severity: str
    trend_slope: float          # negative = degrading
    trend_direction: str        # "improving" / "stable" / "degrading"

    # History
    n_windows: int
    session_age_seconds: float
    current_sqi: float
    baseline_sqi: float         # first N windows average
    sqi_delta: float            # current - baseline

    # Prediction
    estimated_minutes_to_critical: Optional[float]

    # Human readable
    summary: str


# ── Session History ───────────────────────────────────────────
@dataclass
class SessionHistory:
    """Stores rolling SQI history for one signal session."""
    signal_type: SignalType
    session_id: str
    start_time: float = field(default_factory=time.time)
    sqi_history: deque = field(
        default_factory=lambda: deque(maxlen=100)
    )
    timestamp_history: deque = field(
        default_factory=lambda: deque(maxlen=100)
    )


# ── Drift Tracker ─────────────────────────────────────────────
class DriftTracker:
    """
    Tracks per-session reliability degradation over time.

    One DriftTracker instance per monitoring session.
    Call update() after each window analysis.
    Call assess() to get current drift status.
    """

    # Minimum windows before drift assessment is meaningful
    MIN_WINDOWS_FOR_ASSESSMENT = 5

    # Number of initial windows used to establish baseline
    BASELINE_WINDOWS = 5

    # Slope threshold for drift detection (SQI points per window)
    MILD_DRIFT_SLOPE     = -0.5
    MODERATE_DRIFT_SLOPE = -1.5
    SEVERE_DRIFT_SLOPE   = -3.0

    # SQI level considered critical
    CRITICAL_SQI = 40.0

    def __init__(self) -> None:
        # Key: session_id, Value: SessionHistory
        self._sessions: dict[str, SessionHistory] = {}

    def update(
        self,
        session_id: str,
        signal_type: SignalType,
        sqi_score: float,
        timestamp: Optional[float] = None,
    ) -> None:
        """
        Record a new SQI observation for a session.

        Call this after every window analysis.
        """
        if timestamp is None:
            timestamp = time.time()

        if session_id not in self._sessions:
            self._sessions[session_id] = SessionHistory(
                signal_type=signal_type,
                session_id=session_id,
                start_time=timestamp,
            )
            logger.debug(
                f"New session started | "
                f"id={session_id} | "
                f"signal={signal_type.value}"
            )

        session = self._sessions[session_id]
        session.sqi_history.append(sqi_score)
        session.timestamp_history.append(timestamp)

    def assess(
        self,
        session_id: str,
        signal_type: SignalType,
    ) -> Optional[DriftResult]:
        """
        Assess current drift status for a session.

        Returns None if insufficient data.
        """
        if session_id not in self._sessions:
            logger.debug(f"No session found: {session_id}")
            return None

        session = self._sessions[session_id]
        history = list(session.sqi_history)
        timestamps = list(session.timestamp_history)

        if len(history) < self.MIN_WINDOWS_FOR_ASSESSMENT:
            logger.debug(
                f"Insufficient windows for drift assessment | "
                f"session={session_id} | "
                f"n={len(history)} < {self.MIN_WINDOWS_FOR_ASSESSMENT}"
            )
            return None

        # ── Baseline ─────────────────────────────────────────
        baseline_sqi = float(
            np.mean(history[:self.BASELINE_WINDOWS])
        )
        current_sqi = float(history[-1])
        sqi_delta = current_sqi - baseline_sqi

        # ── Trend via Linear Regression ───────────────────────
        x = np.arange(len(history), dtype=float)
        y = np.array(history, dtype=float)
        slope = float(np.polyfit(x, y, 1)[0])

        # ── Session Age ───────────────────────────────────────
        session_age = timestamps[-1] - session.start_time

        # ── Drift Classification ──────────────────────────────
        if slope <= self.SEVERE_DRIFT_SLOPE:
            drift_detected = True
            drift_severity = DriftSeverity.SEVERE
        elif slope <= self.MODERATE_DRIFT_SLOPE:
            drift_detected = True
            drift_severity = DriftSeverity.MODERATE
        elif slope <= self.MILD_DRIFT_SLOPE:
            drift_detected = True
            drift_severity = DriftSeverity.MILD
        else:
            drift_detected = False
            drift_severity = DriftSeverity.NONE

        # ── Trend Direction ───────────────────────────────────
        if slope > 0.2:
            trend_direction = "improving"
        elif slope < -0.2:
            trend_direction = "degrading"
        else:
            trend_direction = "stable"

        # ── Time to Critical ──────────────────────────────────
        estimated_minutes = None
        if drift_detected and slope < 0 and current_sqi > self.CRITICAL_SQI:
            windows_to_critical = (
                current_sqi - self.CRITICAL_SQI
            ) / abs(slope)
            # Assume ~10 seconds per window
            estimated_minutes = (windows_to_critical * 10) / 60

        # ── Summary ───────────────────────────────────────────
        if not drift_detected:
            summary = (
                f"{signal_type.value.upper()} session stable | "
                f"SQI trend: {slope:+.2f}/window | "
                f"current={current_sqi:.0f} | "
                f"baseline={baseline_sqi:.0f}"
            )
        else:
            summary = (
                f"{signal_type.value.upper()} {drift_severity} drift | "
                f"SQI trend: {slope:+.2f}/window | "
                f"current={current_sqi:.0f} | "
                f"baseline={baseline_sqi:.0f} | "
                f"delta={sqi_delta:+.0f}"
            )
            if estimated_minutes:
                summary += (
                    f" | critical in ~{estimated_minutes:.0f}min"
                )

        result = DriftResult(
            signal_type=signal_type,
            session_id=session_id,
            drift_detected=drift_detected,
            drift_severity=drift_severity,
            trend_slope=round(slope, 3),
            trend_direction=trend_direction,
            n_windows=len(history),
            session_age_seconds=session_age,
            current_sqi=current_sqi,
            baseline_sqi=baseline_sqi,
            sqi_delta=round(sqi_delta, 1),
            estimated_minutes_to_critical=estimated_minutes,
            summary=summary,
        )

        log_level = "warning" if drift_detected else "info"
        getattr(logger, log_level)(summary)

        return result

    def get_history(
        self,
        session_id: str,
    ) -> list[float]:
        """Return full SQI history for a session."""
        if session_id not in self._sessions:
            return []
        return list(self._sessions[session_id].sqi_history)

    def end_session(self, session_id: str) -> None:
        """Clean up a completed session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info(f"Session ended | id={session_id}")