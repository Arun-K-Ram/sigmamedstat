# SigmaMedStat

**Medical device signal reliability intelligence - built for real clinical environments.**

> Current systems ask: *"Is this reading abnormal?"*  
> SigmaMedStat asks: *"Should we trust this reading at all?"*

---

## What This Is

SigmaMedStat is a real-time signal reliability platform that evaluates the trustworthiness of medical device outputs before any clinical decision is made.

ICU alarm fatigue is a documented crisis. Studies show nurses ignore 85–99% of alarms - not from negligence, but because current systems fire hundreds of false alerts daily. A sensor slips. A patient moves. The monitor screams. Nobody questions whether the device itself can be trusted. SigmaMedStat does!

---

## The Core Idea

Most medical AI asks: **"What does this signal mean?"**

SigmaMedStat asks: **"Should we believe this signal?"**

It evaluates every reading against its full clinical context - neighboring signals, patient motion, recent events, device session history - and produces a trust score with a plain-English explanation of why a signal is or isn't reliable.

---

## System Architecture
Raw Device Signal
↓
Signal Reliability Engine     ← flatlines, spikes, dropouts, noise
↓
Context Correlation Layer     ← neighboring signals, motion, clinical events
↓
Reliability Scorer            ← 0–100 trust score + confidence
↓
Failure Attribution Engine    ← named cause + supporting evidence
↓
Temporal Drift Monitor        ← session degradation tracking
↓
FastAPI → React Dashboard

---

## Core Components

### Signal Reliability Engine
Four statistical detectors running on raw signal windows:

| Detector | Method | Detects |
|---|---|---|
| Flatline | Std deviation + unique value ratio | Sensor disconnection, probe loss |
| Spike | Z-score + derivative analysis | Electrical interference, ADC errors |
| Dropout | NaN detection + zero burst + sudden loss | Cable failure, transmission loss |
| Noise | SNR estimation + sample entropy + variance CV | Motion artifact, EMI, poor contact |

### Context Correlation Layer
The core differentiator. Extracts four feature categories:

- **Multi-signal** - are neighboring signals also degrading simultaneously?
- **Temporal** - is reliability trending down over this session?
- **Clinical events** - did a repositioning or procedure happen recently?
- **Motion** - is accelerometer data showing patient movement?

### Reliability Scorer
Fuses signal quality index (SQI) with context features into a final 0–100 trust score. Context can adjust the score up or down by up to 20 points.

### Failure Attribution Engine
Rule-based expert system that names the probable cause:
- Sensor displacement
- Motion artifact
- Device malfunction
- Environmental interference
- Calibration drift

Uses clinical knowledge encoded from medical device literature and FDA MAUDE adverse event patterns.

### Temporal Drift Monitor
Tracks per-session SQI history. Detects gradual degradation using linear regression slope. Predicts time to critical threshold.

---

## Example Output

```json
{
  "trust_score": 24.8,
  "grade": "CRITICAL",
  "recommendation": "reseat_sensor",
  "interpretation": "Signal has flatline - high patient motion - clinical event 10s ago - context reduced trust by 16 points",
  "attribution": {
    "failure_category": "sensor_displacement",
    "confidence": 0.95,
    "primary_cause": "Likely sensor displacement - probe contact lost following patient movement",
    "supporting_evidence": [
      "Flatline detected - signal not responding to patient state",
      "Clinical event 10s ago (repositioning - known artifact cause)",
      "Significant patient motion detected (index 1.00)",
      "Neighboring signals remain stable - isolated sensor issue"
    ],
    "recommended_action": "Physically inspect and reseat the sensor.",
    "likely_false_alarm": true
  },
  "processing_time_ms": 34.32
}
```

---

## Regulatory Awareness

Designed with regulatory traceability in mind:

- **IEC 62304** - Class B software safety classification. Modular architecture directly supports unit verification requirements in §5.5
- **ISO 14971** - Risk management documentation. Identified hazards, severity/probability assessment, and control measures documented
- **FDA AI/ML SaMD Action Plan** - Predetermined Change Control Plan for model updates. Algorithm transparency via XGBoost feature importance

This is a research project. Full FDA clearance requires clinical validation at a scale beyond a portfolio project. The architecture is designed to demonstrate what the path to clearance looks like.

---

## Data

Built on the **PhysioNet Challenge 2015** dataset - *"Reducing False Arrhythmia Alarms in the ICU"*.

Real ICU alarm events labeled as true or false alarms. Multi-signal (ECG, SpO₂, ABP, respiration). Ground truth labels built in - enabling direct evaluation against baseline threshold alerting.

No credentialing required. Freely available at: `physionet.org/content/challenge-2015`

---

## Tech Stack

### Backend
Python 3.12      Core language
FastAPI          REST API layer
Poetry           Dependency management
scikit-learn     ML models
XGBoost          Reliability ensemble
wfdb             PhysioNet signal loading
NumPy / SciPy    Signal processing
Pydantic         Request/response schemas
SQLite           Session storage

### Frontend
React + TypeScript   UI framework
Vite                 Build tool
Tailwind CSS         Styling
Recharts             Signal visualization
React Router         Navigation
Axios                API client

---

## Project Structure
crip-x/
├── backend/
│   ├── crip_x/
│   │   ├── signal/
│   │   │   ├── detectors/         # Flatline, Spike, Dropout, Noise
│   │   │   └── signal_quality_index.py
│   │   ├── context/
│   │   │   └── feature_extractor.py
│   │   ├── scoring/
│   │   │   └── reliability_scorer.py
│   │   ├── attribution/
│   │   │   └── attribute_engine.py
│   │   ├── drift/
│   │   │   └── drift_tracker.py
│   │   ├── ingestion/
│   │   │   └── fixture_loader.py
│   │   └── utils/
│   │       ├── config.py
│   │       ├── logger.py
│   │       └── validators.py
│   ├── api/
│   │   ├── main.py
│   │   ├── routers/
│   │   └── schemas/
│   ├── data/fixtures/             # 6 clinical test scenarios
│   ├── exploration/               # Development scripts
│   └── pyproject.toml
│
└── frontend/
├── src/
│   ├── pages/
│   │   ├── HomePage.tsx       # Landing page with hospital animation
│   │   └── DemoPage.tsx       # Live signal demo
│   └── components/
│       └── Navbar.tsx
└── vite.config.ts

---

## Setup

### Backend

```bash
git clone
cd crip-x/backend

# Install Poetry if needed
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Run API
poetry run uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

API docs available at `http://127.0.0.1:8000/docs`

### Frontend

```bash
cd crip-x/frontend
npm install
npm run dev
```

Open `http://localhost:5173`

---

## API Endpoints
GET  /health                          System health check
GET  /fixtures                        List all test scenarios
GET  /fixtures/{id}                   Get fixture details
POST /analyze                         Run pipeline on signal window
POST /analyze/fixture/{fixture_id}    Run a predefined scenario

---

## Clinical Scenarios Included

| Scenario | Signal Type | What It Tests |
|---|---|---|
| Clean SpO₂ | SpO₂ | Baseline - no artifacts |
| Flatline | SpO₂ | Sensor disconnection |
| Spike | SpO₂ | Electrical interference |
| Dropout | SpO₂ | Probe briefly disconnected |
| Motion Artifact | SpO₂ | Patient repositioning event |
| Degrading Session | SpO₂ | 4-hour calibration drift |

---

## Why This Project

Most medical AI projects predict diseases, classify images, or summarize records.

SigmaMedStat focuses on a different problem: **can the input data be trusted before any AI makes a decision?**

This aligns with real problems in:
- ICU alarm fatigue reduction
- Post-market surveillance of medical devices
- AI governance for Software as a Medical Device
- Signal integrity in wearable and remote monitoring


---

*SigmaMedStat is a research project.  
This project is Not FDA cleared. Not for clinical use.*