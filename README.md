# CRIP-X
### Contextual Reliability Intelligence Platform

> "Does this medical device signal make sense in the current 
> clinical context?" 

CRIP-X is a real-time reliability intelligence system that 
evaluates the trustworthiness of medical device signals using 
multi-signal reasoning, temporal analysis, and clinical context 
awareness.

---

## The Problem

ICU alarm fatigue is a documented clinical crisis.
Studies show nurses ignore 85-99% of alarms - not from 
negligence, but because current systems cry wolf constantly.

Current monitoring systems ask:
**"Is this signal abnormal?"**

CRIP-X asks:
**"Should we trust this signal at all?"**

---

## System Architecture

[Architecture diagram - coming soon]

---

## Core Components

| Component | Purpose |
|-----------|---------|
| Signal Reliability Engine | Detects flatlines, spikes, dropouts, noise |
| Context Correlation Layer | Cross-signal consistency + clinical event timing |
| Reliability Scoring System | Trust score 0-100 per signal per window |
| Failure Attribution Engine | Root cause - motion artifact, drift, disconnect |
| Temporal Drift Monitor | Long-term device reliability degradation |

---

## Data

Built on real ICU data from the PhysioNet Challenge 2015
"Reducing False Arrhythmia Alarms in the ICU" dataset.

---

## Regulatory Awareness

Designed with:
- IEC 62304 Class B software safety classification
- ISO 14971 risk management methodology  
- FDA AI/ML SaMD Action Plan alignment

---

## Tech Stack

Python 3.11 · FastAPI · Streamlit · scikit-learn · 
XGBoost · SHAP · wfdb · Poetry

---

## Setup

```bash
git clone
cd crip-x
poetry install
poetry install --with dev
```

---

## Status

🔨 Active development - Week 1 of 8