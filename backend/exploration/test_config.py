import numpy as np
from crip_x.utils.logger import get_logger
from crip_x.utils.validators import SignalValidator, SignalType
from crip_x.signal.detectors.base_detector import (
    BaseDetector, DetectionResult, ArtifactType
)

logger = get_logger(__name__)

# ── Validator Tests ──────────────────────────────────────────
validator = SignalValidator()

clean_spo2 = np.array([98.0, 97.5, 98.2, 97.8, 98.1])
result = validator.validate(clean_spo2, SignalType.SPO2)
logger.info(f"Test 1 - Clean SpO2: valid={result.is_valid}")

bad_spo2 = np.array([98.0, 450.0, 97.5, -10.0, 98.1])
result = validator.validate(bad_spo2, SignalType.SPO2)
logger.info(f"Test 2 - Bad SpO2: valid={result.is_valid} | {result.message}")

dropout_signal = np.array([98.0, np.nan, np.nan, np.nan, 97.5])
result = validator.validate(dropout_signal, SignalType.SPO2)
logger.info(f"Test 3 - Dropout: valid={result.is_valid} | {result.message}")

empty = np.array([])
result = validator.validate(empty, SignalType.HEART_RATE)
logger.info(f"Test 4 - Empty: valid={result.is_valid} | {result.message}")

# ── Base Detector Test ───────────────────────────────────────
try:
    detector = BaseDetector()
    logger.error("Should not reach here")
except TypeError as e:
    logger.info(f"Abstract class correctly blocked: {e}")

logger.info("All tests complete")

from crip_x.signal.detectors.flatline_detector import FlatlineDetector

detector = FlatlineDetector()

# Test 1 - obvious flatline
flatline = np.full(100, 98.0)  # 100 samples all exactly 98.0
result = detector.detect(flatline, SignalType.SPO2)
logger.info(
    f"Flatline Test 1 - detected={result.artifact_detected} | "
    f"confidence={result.confidence:.2f} | {result.message}"
)

# Test 2 - real signal with natural variance
real_signal = np.array([
    98.0, 97.8, 98.2, 97.9, 98.1,
    97.7, 98.3, 97.6, 98.4, 97.8,
    98.0, 97.9, 98.1, 97.8, 98.2,
    97.7, 98.3, 97.9, 98.0, 97.8,
] * 5)  # 100 samples of realistic SpO2
result = detector.detect(real_signal, SignalType.SPO2)
logger.info(
    f"Flatline Test 2 - detected={result.artifact_detected} | "
    f"confidence={result.confidence:.2f} | {result.message}"
)

# Test 3 - subtle flatline (nearly flat, not perfect)
subtle_flatline = np.full(100, 98.0)
subtle_flatline += np.random.normal(0, 0.001, 100)  # tiny noise
result = detector.detect(subtle_flatline, SignalType.SPO2)
logger.info(
    f"Flatline Test 3 - detected={result.artifact_detected} | "
    f"confidence={result.confidence:.2f} | {result.message}"
)

# Test 4 - high sensitivity catches subtle flatline
sensitive_detector = FlatlineDetector(sensitivity=2.0)
result = sensitive_detector.detect(subtle_flatline, SignalType.SPO2)
logger.info(
    f"Flatline Test 4 (high sensitivity) - "
    f"detected={result.artifact_detected} | "
    f"confidence={result.confidence:.2f}"
)
from crip_x.signal.detectors.spike_detector import SpikeDetector

spike_detector = SpikeDetector()

# Base signal - normal SpO2
base = np.array([98.0, 97.8, 98.2, 97.9, 98.1] * 20)  # 100 samples

# Test 1 - clean signal, no spikes
result = spike_detector.detect(base.copy(), SignalType.SPO2)
logger.info(
    f"Spike Test 1 - detected={result.artifact_detected} | "
    f"{result.message}"
)

# Test 2 - inject obvious spike
spiked = base.copy()
spiked[50] = 245.0   # impossible SpO2 value
result = spike_detector.detect(spiked, SignalType.SPO2)
logger.info(
    f"Spike Test 2 - detected={result.artifact_detected} | "
    f"confidence={result.confidence:.2f} | "
    f"{result.message}"
)

# Test 3 - multiple spikes
multi_spike = base.copy()
multi_spike[20] = 10.0
multi_spike[50] = 245.0
multi_spike[80] = 5.0
result = spike_detector.detect(multi_spike, SignalType.SPO2)
logger.info(
    f"Spike Test 3 - detected={result.artifact_detected} | "
    f"n_spikes={result.metadata['n_zscore_spikes']} | "
    f"confidence={result.confidence:.2f}"
)

# Test 4 - subtle spike
subtle_spike = base.copy()
subtle_spike[50] = 85.0  # low but not impossible
result = spike_detector.detect(subtle_spike, SignalType.SPO2)
logger.info(
    f"Spike Test 4 (subtle) - detected={result.artifact_detected} | "
    f"confidence={result.confidence:.2f}"
)
from crip_x.signal.detectors.dropout_detector import DropoutDetector

dropout_detector = DropoutDetector()
base = np.array([98.0, 97.8, 98.2, 97.9, 98.1] * 20)

# Test 1 - clean signal
result = dropout_detector.detect(base.copy(), SignalType.SPO2)
logger.info(
    f"Dropout Test 1 - detected={result.artifact_detected} | "
    f"{result.message}"
)

# Test 2 - NaN dropout
nan_signal = base.copy().astype(float)
nan_signal[40:50] = np.nan   # 10 consecutive NaNs
result = dropout_detector.detect(nan_signal, SignalType.SPO2)
logger.info(
    f"Dropout Test 2 - detected={result.artifact_detected} | "
    f"confidence={result.confidence:.2f} | "
    f"{result.message}"
)

# Test 3 - zero burst (device reports 0 instead of NaN)
zero_signal = base.copy()
zero_signal[60:68] = 0.0    # 8 consecutive zeros
result = dropout_detector.detect(zero_signal, SignalType.SPO2)
logger.info(
    f"Dropout Test 3 - detected={result.artifact_detected} | "
    f"confidence={result.confidence:.2f} | "
    f"{result.message}"
)

# Test 4 - sudden signal loss (second half flatlines)
sudden_loss = base.copy()
sudden_loss[50:] = 0.0      # second half goes dead
result = dropout_detector.detect(sudden_loss, SignalType.SPO2)
logger.info(
    f"Dropout Test 4 - detected={result.artifact_detected} | "
    f"confidence={result.confidence:.2f} | "
    f"{result.message}"
)
from crip_x.signal.detectors.noise_detector import NoiseDetector

noise_detector = NoiseDetector()
np.random.seed(42)

# Base clean signal
base = np.array([98.0, 97.8, 98.2, 97.9, 98.1] * 20)

# Test 1 - clean signal
result = noise_detector.detect(base.copy(), SignalType.SPO2)
logger.info(
    f"Noise Test 1 - detected={result.artifact_detected} | "
    f"{result.message}"
)

# Test 2 - heavy noise
noisy = base.copy() + np.random.normal(0, 5.0, 100)
result = noise_detector.detect(noisy, SignalType.SPO2)
logger.info(
    f"Noise Test 2 - detected={result.artifact_detected} | "
    f"confidence={result.confidence:.2f} | "
    f"{result.message}"
)

# Test 3 - pure random noise
pure_noise = np.random.normal(98.0, 8.0, 100)
result = noise_detector.detect(pure_noise, SignalType.SPO2)
logger.info(
    f"Noise Test 3 - detected={result.artifact_detected} | "
    f"confidence={result.confidence:.2f} | "
    f"{result.message}"
)

# Test 4 - mild noise (borderline)
mild_noise = base.copy() + np.random.normal(0, 1.0, 100)
result = noise_detector.detect(mild_noise, SignalType.SPO2)
logger.info(
    f"Noise Test 4 (mild) - detected={result.artifact_detected} | "
    f"SNR={result.metadata['snr_db']:.1f}dB"
)

from crip_x.signal.signal_quality_index import SignalQualityIndex

sqi = SignalQualityIndex()
np.random.seed(42)

# Base clean signal
base = np.array([98.0, 97.8, 98.2, 97.9, 98.1] * 20)

# Test 1 - clean signal
result = sqi.compute(base.copy(), SignalType.SPO2)
logger.info(
    f"SQI Test 1 - score={result.sqi_score:.0f} | "
    f"grade={result.grade} | {result.summary}"
)

# Test 2 - flatline
flatline = np.full(100, 98.0)
result = sqi.compute(flatline, SignalType.SPO2)
logger.info(
    f"SQI Test 2 - score={result.sqi_score:.0f} | "
    f"grade={result.grade} | artifacts={[a.value for a in result.artifacts_detected]}"
)

# Test 3 - spike injected
spiked = base.copy()
spiked[50] = 245.0
result = sqi.compute(spiked, SignalType.SPO2)
logger.info(
    f"SQI Test 3 - score={result.sqi_score:.0f} | "
    f"grade={result.grade} | dominant={result.dominant_artifact}"
)

# Test 4 - multiple artifacts
multi = base.copy().astype(float)
multi[40:50] = np.nan       # dropout
multi[80] = 245.0           # spike
result = sqi.compute(multi, SignalType.SPO2)
logger.info(
    f"SQI Test 4 - score={result.sqi_score:.0f} | "
    f"grade={result.grade} | "
    f"artifacts={[a.value for a in result.artifacts_detected]}"
)

# Test 5 - noisy signal
noisy = base.copy() + np.random.normal(0, 5.0, 100)
result = sqi.compute(noisy, SignalType.SPO2)
logger.info(
    f"SQI Test 5 - score={result.sqi_score:.0f} | "
    f"grade={result.grade}"
)
from crip_x.context.feature_extractor import (
    ContextFeatureExtractor,
    ContextInput,
    ClinicalEvent,
    ClinicalEventType,
)
from crip_x.signal.signal_quality_index import SignalQualityIndex
import time

extractor = ContextFeatureExtractor()
sqi_engine = SignalQualityIndex()
now = time.time()

# Build a primary signal SQI
base = np.array([98.0, 97.8, 98.2, 97.9, 98.1] * 20)
primary_sqi = sqi_engine.compute(base.copy(), SignalType.SPO2)

# Build a degraded neighbor SQI
noisy = base.copy() + np.random.normal(0, 5.0, 100)
neighbor_sqi = sqi_engine.compute(noisy, SignalType.HEART_RATE)

# ── Test 1: Clean context, no events, good neighbors
context1 = ContextInput(
    primary_signal_type=SignalType.SPO2,
    primary_sqi=primary_sqi,
    current_timestamp=now,
    neighboring_sqis={SignalType.HEART_RATE: primary_sqi},
    session_start_timestamp=now - 300,
    historical_sqis=[95, 96, 94, 97, 95, 96, 94, 95, 96, 97],
)
features1 = extractor.extract(context1)
logger.info(
    f"Context Test 1 - "
    f"artifact_prob={features1.context_artifact_probability:.2f} | "
    f"reliability_bonus={features1.context_reliability_bonus:.2f} | "
    f"trend={features1.reliability_trend:.3f}"
)

# ── Test 2: Patient repositioning event 10 seconds ago
context2 = ContextInput(
    primary_signal_type=SignalType.SPO2,
    primary_sqi=primary_sqi,
    current_timestamp=now,
    neighboring_sqis={SignalType.HEART_RATE: neighbor_sqi},
    session_start_timestamp=now - 1800,
    historical_sqis=[95, 90, 80, 65, 55, 50, 48, 52, 55, 60],
    recent_events=[
        ClinicalEvent(
            event_type=ClinicalEventType.PATIENT_REPOSITIONING,
            timestamp=now - 10,
            description="Nurse repositioned patient"
        )
    ],
    motion_signal=np.random.normal(0, 3.0, 100),
)
features2 = extractor.extract(context2)
logger.info(
    f"Context Test 2 - "
    f"artifact_prob={features2.context_artifact_probability:.2f} | "
    f"event={features2.recent_clinical_event} | "
    f"seconds_since={features2.seconds_since_event:.0f}s | "
    f"motion={features2.motion_index:.2f} | "
    f"trend={features2.reliability_trend:.3f}"
)

# ── Test 3: Multi-signal degradation
degraded_sqi = sqi_engine.compute(
    np.full(100, 98.0), SignalType.HEART_RATE
)
context3 = ContextInput(
    primary_signal_type=SignalType.SPO2,
    primary_sqi=degraded_sqi,
    current_timestamp=now,
    neighboring_sqis={
        SignalType.HEART_RATE: degraded_sqi,
        SignalType.RESPIRATORY_RATE: degraded_sqi,
    },
    session_start_timestamp=now - 600,
    historical_sqis=[90, 85, 75, 60, 50, 45, 42, 40, 38, 35],
)
features3 = extractor.extract(context3)
logger.info(
    f"Context Test 3 - "
    f"multi_signal_degradation={features3.multi_signal_simultaneous_degradation} | "
    f"neighboring_ratio={features3.neighboring_degradation_ratio:.2f} | "
    f"artifact_prob={features3.context_artifact_probability:.2f}"
)
from crip_x.scoring.reliability_scorer import ReliabilityScorer

scorer = ReliabilityScorer()

# ── Test 1: Clean signal, clean context
result = scorer.score(
    sqi_result=sqi_engine.compute(base.copy(), SignalType.SPO2),
    context_features=features1,
)
logger.info(
    f"Score Test 1 - "
    f"trust={result.trust_score} | "
    f"rec={result.recommendation.value} | "
    f"{result.interpretation}"
)

# ── Test 2: Degraded signal + motion artifact context
flatline_sqi = sqi_engine.compute(
    np.full(100, 98.0), SignalType.SPO2
)
result = scorer.score(
    sqi_result=flatline_sqi,
    context_features=features2,
)
logger.info(
    f"Score Test 2 - "
    f"trust={result.trust_score} | "
    f"confidence={result.confidence:.2f} | "
    f"rec={result.recommendation.value} | "
    f"context_delta={result.context_delta:+} | "
    f"{result.interpretation}"
)

# ── Test 3: No context provided
spike_sqi = sqi_engine.compute(spiked.copy(), SignalType.SPO2)
result = scorer.score(
    sqi_result=spike_sqi,
    context_features=None,
)
logger.info(
    f"Score Test 3 (no context) - "
    f"trust={result.trust_score} | "
    f"rec={result.recommendation.value} | "
    f"{result.interpretation}"
)

# ── Test 4: Multi-signal degradation context
result = scorer.score(
    sqi_result=flatline_sqi,
    context_features=features3,
)
logger.info(
    f"Score Test 4 - "
    f"trust={result.trust_score} | "
    f"rec={result.recommendation.value} | "
    f"context_delta={result.context_delta:+} | "
    f"{result.interpretation}"
)

from crip_x.attribution.attribute_engine import AttributionEngine

engine = AttributionEngine()

# ── Test 1: Clean signal
clean_score = scorer.score(
    sqi_result=sqi_engine.compute(base.copy(), SignalType.SPO2),
    context_features=features1,
)
attr = engine.attribute(clean_score)
logger.info(
    f"Attribution Test 1 - "
    f"category={attr.failure_category.value} | "
    f"{attr.primary_cause}"
)

# ── Test 2: Flatline + motion + repositioning
flatline_score = scorer.score(
    sqi_result=sqi_engine.compute(np.full(100, 98.0), SignalType.SPO2),
    context_features=features2,
)
attr = engine.attribute(flatline_score)
logger.info(
    f"Attribution Test 2 - "
    f"category={attr.failure_category.value} | "
    f"confidence={attr.confidence:.2f} | "
    f"false_alarm={attr.likely_false_alarm} | "
    f"{attr.primary_cause}"
)
for evidence in attr.supporting_evidence:
    logger.info(f"  Evidence: {evidence}")
logger.info(f"  Action: {attr.recommended_action}")

# ── Test 3: Multi-signal environmental degradation
multi_score = scorer.score(
    sqi_result=sqi_engine.compute(np.full(100, 98.0), SignalType.SPO2),
    context_features=features3,
)
attr = engine.attribute(multi_score)
logger.info(
    f"Attribution Test 3 - "
    f"category={attr.failure_category.value} | "
    f"confidence={attr.confidence:.2f} | "
    f"{attr.primary_cause}"
)
from crip_x.drift.drift_tracker import DriftTracker

tracker = DriftTracker()
session_id = "patient_001_spo2"
now = time.time()

# ── Test 1: Stable session
logger.info("--- Drift Test 1: Stable Session ---")
stable_sqis = [95, 96, 94, 97, 95, 96, 94, 95, 96, 97]
for i, sqi in enumerate(stable_sqis):
    tracker.update(session_id, SignalType.SPO2, sqi, now + i*10)

result = tracker.assess(session_id, SignalType.SPO2)
logger.info(
    f"Drift Test 1 - "
    f"detected={result.drift_detected} | "
    f"severity={result.drift_severity} | "
    f"slope={result.trend_slope:+.3f} | "
    f"{result.summary}"
)

# ── Test 2: Degrading session
session_id_2 = "patient_002_spo2"
logger.info("--- Drift Test 2: Degrading Session ---")
degrading_sqis = [95, 92, 88, 83, 78, 72, 65, 58, 50, 43]
for i, sqi in enumerate(degrading_sqis):
    tracker.update(session_id_2, SignalType.SPO2, sqi, now + i*10)

result = tracker.assess(session_id_2, SignalType.SPO2)
logger.info(
    f"Drift Test 2 - "
    f"detected={result.drift_detected} | "
    f"severity={result.drift_severity} | "
    f"slope={result.trend_slope:+.3f} | "
    f"time_to_critical={result.estimated_minutes_to_critical:.1f}min"
    if result.estimated_minutes_to_critical else
    f"Drift Test 2 - detected={result.drift_detected} | "
    f"severity={result.drift_severity} | "
    f"slope={result.trend_slope:+.3f}"
)

# ── Test 3: Recovering session
session_id_3 = "patient_003_spo2"
logger.info("--- Drift Test 3: Recovering Session ---")
recovering_sqis = [45, 50, 58, 65, 72, 78, 83, 88, 92, 95]
for i, sqi in enumerate(recovering_sqis):
    tracker.update(session_id_3, SignalType.SPO2, sqi, now + i*10)

result = tracker.assess(session_id_3, SignalType.SPO2)
logger.info(
    f"Drift Test 3 - "
    f"detected={result.drift_detected} | "
    f"direction={result.trend_direction} | "
    f"slope={result.trend_slope:+.3f} | "
    f"delta={result.sqi_delta:+.0f}"
)
from crip_x.ingestion.fixture_loader import (
    load_fixture, get_signal_array,
    get_neighboring_arrays, list_fixtures
)

logger.info("--- Fixture Loader Tests ---")
logger.info(f"Available fixtures: {list_fixtures()}")

# Load and run clean fixture through full pipeline
fixture = load_fixture("clean_spo2.json")
signal = get_signal_array(fixture)
neighbors = get_neighboring_arrays(fixture)

logger.info(
    f"Loaded: {fixture['fixture_id']} | "
    f"signal_length={len(signal)} | "
    f"neighbors={list(neighbors.keys())}"
)

# Run through full pipeline
sqi_result = sqi_engine.compute(signal, SignalType.SPO2)
reliability = scorer.score(sqi_result)
attribution = engine.attribute(reliability)

logger.info(
    f"Pipeline result - "
    f"trust={reliability.trust_score} | "
    f"rec={reliability.recommendation.value} | "
    f"attribution={attribution.failure_category.value}"
)

# Verify against expected outcome
expected = fixture["expected_outcome"]
passed = reliability.trust_score >= expected["trust_score_min"]
logger.info(
    f"Fixture validation - "
    f"expected_min={expected['trust_score_min']} | "
    f"actual={reliability.trust_score} | "
    f"{'PASS' if passed else 'FAIL'}"
)