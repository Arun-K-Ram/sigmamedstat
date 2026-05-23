from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FIXTURES = {
    "clean_spo2_001": {
        "trust_score": 100.0,
        "sqi_score": 100.0,
        "grade": "EXCELLENT",
        "confidence": 0.99,
        "is_trustworthy": True,
        "signal_type": "spo2",
        "n_samples": 100,
        "session_id": "session_001",
        "recommendation": "trust",
        "interpretation": "Signal is clean - context increased trust by 4 points",
        "artifacts_detected": [],
        "dominant_artifact": None,
        "artifact_details": {},
        "attribution": {
            "failure_category": "unknown",
            "confidence": 0.99,
            "primary_cause": "Signal is reliable - no failure attribution required.",
            "supporting_evidence": [
                "Trust score: 100/100",
                "SQI: 100/100",
                "No significant artifacts detected"
            ],
            "recommended_action": "No action required - signal is trustworthy.",
            "clinical_context": "Signal quality is within acceptable parameters.",
            "likely_false_alarm": False,
            "false_alarm_confidence": 0.0
        },
        "context_adjusted": True,
        "context_delta": 4.5,
        "drift": None,
        "processing_time_ms": 90.89
    },
    "flatline_spo2_001": {
        "trust_score": 38.9,
        "sqi_score": 40.6,
        "grade": "POOR",
        "confidence": 0.761,
        "is_trustworthy": False,
        "signal_type": "spo2",
        "n_samples": 100,
        "session_id": "session_002",
        "recommendation": "discard",
        "interpretation": "Signal has flatline - context reduced trust by 2 points",
        "artifacts_detected": ["flatline"],
        "dominant_artifact": "flatline",
        "artifact_details": {
            "flatline": {
                "artifact_type": "flatline",
                "confidence": 0.99,
                "severity": 1.0,
                "affected_ratio": 1.0,
                "message": "Flatline detected - std=0.0000 (threshold=0.0500)"
            }
        },
        "attribution": {
            "failure_category": "sensor_displacement",
            "confidence": 0.7,
            "primary_cause": "Likely sensor displacement - probe contact lost",
            "supporting_evidence": [
                "Flatline detected - signal not responding to patient state",
                "Neighboring signals remain stable - suggests isolated sensor issue, not patient event"
            ],
            "recommended_action": "Physically inspect and reseat the sensor. Verify probe position before interpreting readings.",
            "clinical_context": "Neighboring signals suggest patient condition is unchanged. This appears to be a device issue.",
            "likely_false_alarm": True,
            "false_alarm_confidence": 0.7
        },
        "context_adjusted": True,
        "context_delta": -1.7,
        "drift": None,
        "processing_time_ms": 37.88
    },
    "spike_spo2_001": {
        "trust_score": 87.5,
        "sqi_score": 83.02,
        "grade": "GOOD",
        "confidence": 0.99,
        "is_trustworthy": True,
        "signal_type": "spo2",
        "n_samples": 100,
        "session_id": "session_006",
        "recommendation": "trust",
        "interpretation": "Signal has spike - context increased trust by 4 points",
        "artifacts_detected": ["spike"],
        "dominant_artifact": "spike",
        "artifact_details": {
            "spike": {
                "artifact_type": "spike",
                "confidence": 0.758,
                "severity": 0.8,
                "affected_ratio": 0.08,
                "message": "Spike artifact detected - 8 spike(s) | max_z=4.55 | affected=8.0%"
            }
        },
        "attribution": {
            "failure_category": "unknown",
            "confidence": 0.99,
            "primary_cause": "Signal is reliable - no failure attribution required.",
            "supporting_evidence": [
                "Trust score: 88/100",
                "SQI: 83/100",
                "Spike artifacts present but within tolerable range"
            ],
            "recommended_action": "No action required - signal is trustworthy.",
            "clinical_context": "Signal quality is within acceptable parameters.",
            "likely_false_alarm": False,
            "false_alarm_confidence": 0.0
        },
        "context_adjusted": True,
        "context_delta": 4.5,
        "drift": None,
        "processing_time_ms": 38.57
    },
    "dropout_spo2_001": {
        "trust_score": 4.5,
        "sqi_score": 0.0,
        "grade": "CRITICAL",
        "confidence": 0.99,
        "is_trustworthy": False,
        "signal_type": "spo2",
        "n_samples": 100,
        "session_id": "session_004",
        "recommendation": "discard",
        "interpretation": "Signal dropout detected - sensor disconnected",
        "artifacts_detected": [],
        "dominant_artifact": None,
        "artifact_details": {},
        "attribution": {
            "failure_category": "unknown",
            "confidence": 0.3,
            "primary_cause": "Signal unreliability - cause undetermined. Detected: general degradation.",
            "supporting_evidence": [
                "Trust score: 4/100",
                "SQI: 0/100 - complete signal loss",
                "Insufficient context to determine specific cause"
            ],
            "recommended_action": "Verify signal manually. Check sensor placement and device status. Do not act on this reading without verification.",
            "clinical_context": "CRIP-X cannot confidently attribute this failure. Manual assessment recommended.",
            "likely_false_alarm": False,
            "false_alarm_confidence": 0.3
        },
        "context_adjusted": True,
        "context_delta": 4.5,
        "drift": None,
        "processing_time_ms": 45.1
    },
    "motion_artifact_001": {
        "trust_score": 24.8,
        "sqi_score": 40.6,
        "grade": "POOR",
        "confidence": 0.902,
        "is_trustworthy": False,
        "signal_type": "spo2",
        "n_samples": 100,
        "session_id": "session_005",
        "recommendation": "discard",
        "interpretation": "Signal has flatline - clinical event 10s ago - context reduced trust by 16 points",
        "artifacts_detected": ["flatline"],
        "dominant_artifact": "flatline",
        "artifact_details": {
            "flatline": {
                "artifact_type": "flatline",
                "confidence": 0.99,
                "severity": 1.0,
                "affected_ratio": 1.0,
                "message": "Flatline detected - std=0.0000 (threshold=0.0500)"
            }
        },
        "attribution": {
            "failure_category": "sensor_displacement",
            "confidence": 0.9,
            "primary_cause": "Likely sensor displacement - probe contact lost",
            "supporting_evidence": [
                "Flatline detected - signal not responding to patient state",
                "Clinical event 10s ago (type known to cause sensor displacement)",
                "Neighboring signals remain stable - suggests isolated sensor issue, not patient event"
            ],
            "recommended_action": "Physically inspect and reseat the sensor. Verify probe position before interpreting readings.",
            "clinical_context": "Neighboring signals suggest patient condition is unchanged. This appears to be a device issue.",
            "likely_false_alarm": True,
            "false_alarm_confidence": 0.9
        },
        "context_adjusted": True,
        "context_delta": -15.8,
        "drift": None,
        "processing_time_ms": 43.01
    },
    "degrading_session_001": {
        "trust_score": 100.0,
        "sqi_score": 100.0,
        "grade": "EXCELLENT",
        "confidence": 0.99,
        "is_trustworthy": True,
        "signal_type": "spo2",
        "n_samples": 100,
        "session_id": "session_006",
        "recommendation": "trust",
        "interpretation": "Signal is clean",
        "artifacts_detected": [],
        "dominant_artifact": None,
        "artifact_details": {},
        "attribution": {
            "failure_category": "unknown",
            "confidence": 0.99,
            "primary_cause": "Signal is reliable - no failure attribution required.",
            "supporting_evidence": [
                "Trust score: 100/100",
                "SQI: 100/100",
                "No significant artifacts detected"
            ],
            "recommended_action": "No action required - signal is trustworthy.",
            "clinical_context": "Signal quality is within acceptable parameters.",
            "likely_false_alarm": False,
            "false_alarm_confidence": 0.0
        },
        "context_adjusted": False,
        "context_delta": 0.5,
        "drift": None,
        "processing_time_ms": 80.44
    }
}

@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.2.0", "pipeline_ready": True}

@app.get("/")
async def root():
    return {"name": "SigmaMedStat API", "version": "0.2.0"}

@app.post("/analyze/fixture/{fixture_id}")
async def analyze_fixture(fixture_id: str):
    if fixture_id not in FIXTURES:
        return JSONResponse(
            status_code=404,
            content={"error": f"Fixture '{fixture_id}' not found",
                     "available": list(FIXTURES.keys())}
        )
    return FIXTURES[fixture_id]