import { useState, useEffect } from "react"
import axios from "axios"
import Navbar from "../components/Navbar"

const API = "https://sigmamedstat-api.fly.dev"

const FIXTURES = [
  { id: "clean_spo2_001", label: "Clean SpO₂", tag: "Normal" },
  { id: "flatline_spo2_001", label: "Flatline", tag: "Sensor fail" },
  { id: "spike_spo2_001", label: "Spike", tag: "Interference" },
  { id: "dropout_spo2_001", label: "Dropout", tag: "Disconnected" },
  { id: "motion_artifact_001", label: "Motion artifact", tag: "Patient moved" },
  { id: "degrading_session_001", label: "Drift", tag: "Long session" },
]

function scoreColor(score: number) {
  if (score >= 90) return "#4a7a4a"
  if (score >= 70) return "#6a7a3a"
  if (score >= 50) return "#7a6a2a"
  if (score >= 25) return "#7a3a2a"
  return "#7a2a2a"
}

function scoreBg(score: number) {
  if (score >= 90) return "#0a1a0a"
  if (score >= 70) return "#0f1a0a"
  if (score >= 50) return "#1a140a"
  if (score >= 25) return "#1a0d0a"
  return "#1a0a0a"
}

function scoreLabel(score: number) {
  if (score >= 90) return "EXCELLENT"
  if (score >= 70) return "GOOD"
  if (score >= 50) return "DEGRADED"
  if (score >= 25) return "POOR"
  return "CRITICAL"
}

export default function DemoPage() {
  const [selected, setSelected] = useState(FIXTURES[0].id)
  const [result, setResult] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function runFixture(id: string) {
    setLoading(true)
    setError(null)
    try {
      const res = await axios.post(`${API}/analyze/fixture/${id}`)
      setResult(res.data)
    } catch {
      setError("Cannot connect to backend. Make sure the API is running on port 8000.")
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { runFixture(selected) }, [])

  function select(id: string) {
    setSelected(id)
    runFixture(id)
  }

  const color = result ? scoreColor(result.trust_score) : "#444"
  const bg = result ? scoreBg(result.trust_score) : "#111"

  return (
    <div style={{ background: "#0a0a0a", minHeight: "100vh", fontFamily: "'DM Sans', 'Helvetica Neue', sans-serif" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500&family=DM+Mono:wght@400;500&display=swap');
        * { box-sizing: border-box; }
        .mono { font-family: 'DM Mono', monospace; }
        .scene-btn { background: #111; border: 1px solid #1e1e1e; border-radius: 8px; padding: 14px 16px; cursor: pointer; text-align: left; transition: border-color 0.15s, background 0.15s; width: 100%; }
        .scene-btn:hover { border-color: #2e2e2e; background: #151515; }
        .scene-btn.active { border-color: #2e2e2e; background: #151515; }
        .panel { background: #111; border: 1px solid #1e1e1e; border-radius: 10px; padding: 28px; }
        .label { font-size: 11px; color: #444; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 16px; }
        .evidence-row { display: flex; gap: 10px; font-size: 13px; color: #555; padding: 8px 0; border-bottom: 1px solid #171717; }
        .evidence-row:last-child { border-bottom: none; }
        .spin { animation: spin 1s linear infinite; }
        @keyframes spin { to { transform: rotate(360deg); } }
      `}</style>

      <Navbar />

      <div style={{ maxWidth: 1100, margin: "0 auto", padding: "96px 40px 60px", display: "grid", gridTemplateColumns: "220px 1fr", gap: 24 }}>

        {/* Scenario list */}
        <div>
          <div className="label">Scenarios</div>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {FIXTURES.map(f => (
              <button key={f.id} className={`scene-btn ${selected === f.id ? "active" : ""}`} onClick={() => select(f.id)}>
                <div style={{ fontSize: 13, color: selected === f.id ? "#bbb" : "#666", fontWeight: 400, marginBottom: 3 }}>{f.label}</div>
                <div style={{ fontSize: 11, color: "#333" }}>{f.tag}</div>
              </button>
            ))}
          </div>

          {result && (
            <div style={{ marginTop: 24, padding: 16, background: "#0d0d0d", border: "1px solid #1a1a1a", borderRadius: 8 }}>
              <div style={{ fontSize: 11, color: "#333", marginBottom: 8 }}>Pipeline time</div>
              <div className="mono" style={{ fontSize: 20, color: "#444", fontWeight: 300 }}>{result.processing_time_ms.toFixed(0)}<span style={{ fontSize: 12 }}>ms</span></div>
            </div>
          )}
        </div>

        {/* Main panel */}
        <div>
          {loading && (
            <div style={{ display: "flex", alignItems: "center", gap: 12, padding: "60px 0", color: "#444", fontSize: 14 }}>
              <div className="spin" style={{ width: 16, height: 16, border: "1px solid #333", borderTopColor: "#666", borderRadius: "50%" }}></div>
              Analyzing signal...
            </div>
          )}

          {error && (
            <div style={{ background: "#1a0a0a", border: "1px solid #2a1010", borderRadius: 8, padding: 20, color: "#8a4a4a", fontSize: 14 }}>
              {error}
            </div>
          )}

          {result && !loading && (
            <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>

              {/* Top row */}
              <div style={{ display: "grid", gridTemplateColumns: "200px 1fr 1fr", gap: 16 }}>

                {/* Trust score */}
                <div className="panel" style={{ background: bg, borderColor: color + "40", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", padding: 28 }}>
                  <div className="mono" style={{ fontSize: 56, fontWeight: 300, color, lineHeight: 1 }}>{Math.round(result.trust_score)}</div>
                  <div style={{ fontSize: 11, color: "#444", marginTop: 4 }}>/ 100</div>
                  <div style={{ fontSize: 11, color, letterSpacing: "0.12em", marginTop: 12, textTransform: "uppercase" }}>{scoreLabel(result.trust_score)}</div>
                  <div style={{ fontSize: 11, color: "#333", marginTop: 6 }}>SQI {result.sqi_score.toFixed(0)} · {(result.confidence * 100).toFixed(0)}% confidence</div>
                </div>

                {/* Recommendation */}
                <div className="panel">
                  <div className="label">Recommendation</div>
                  <div style={{ fontSize: 18, fontWeight: 300, color: "#ccc", marginBottom: 12, textTransform: "capitalize" }}>
                    {result.recommendation.replace(/_/g, " ")}
                  </div>
                  <div style={{ fontSize: 13, color: "#555", lineHeight: 1.7 }}>{result.interpretation}</div>
                  {result.context_adjusted && (
                    <div style={{ marginTop: 16, fontSize: 12, color: "#333" }}>
                      Context adjusted score by {result.context_delta > 0 ? "+" : ""}{result.context_delta.toFixed(1)} pts
                    </div>
                  )}
                </div>

                {/* Artifacts */}
                <div className="panel">
                  <div className="label">Detected artifacts</div>
                  {result.artifacts_detected.length === 0 ? (
                    <div style={{ fontSize: 14, color: "#4a7a4a" }}>No artifacts — signal is clean</div>
                  ) : (
                    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                      {result.artifacts_detected.map((a: string) => (
                        <div key={a} style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                          <span style={{ fontSize: 14, color: "#8a5a5a", textTransform: "capitalize" }}>{a.replace(/_/g, " ")}</span>
                          {result.artifact_details[a] && (
                            <span className="mono" style={{ fontSize: 12, color: "#444" }}>
                              {(result.artifact_details[a].confidence * 100).toFixed(0)}%
                            </span>
                          )}
                        </div>
                      ))}
                    </div>
                  )}

                  {result.attribution.likely_false_alarm && (
                    <div style={{ marginTop: 16, paddingTop: 14, borderTop: "1px solid #1e1e1e", fontSize: 12, color: "#7a6a2a" }}>
                      ⚠ Probable false alarm
                    </div>
                  )}
                </div>
              </div>

              {/* Attribution */}
              <div className="panel">
                <div className="label">Failure attribution</div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 32 }}>
                  <div>
                    <div style={{ fontSize: 15, color: "#bbb", marginBottom: 16, fontWeight: 300 }}>
                      {result.attribution.primary_cause}
                    </div>
                    <div>
                      {result.attribution.supporting_evidence.map((e: string, i: number) => (
                        <div key={i} className="evidence-row">
                          <span style={{ color: "#3a5a3a", flexShrink: 0 }}>→</span>
                          <span>{e}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                  <div style={{ paddingLeft: 32, borderLeft: "1px solid #1a1a1a" }}>
                    <div style={{ fontSize: 11, color: "#444", marginBottom: 12, letterSpacing: "0.08em", textTransform: "uppercase" }}>Recommended action</div>
                    <div style={{ fontSize: 14, color: "#555", lineHeight: 1.7, marginBottom: 20 }}>
                      {result.attribution.recommended_action}
                    </div>
                    <div style={{ fontSize: 11, color: "#444", marginBottom: 12, letterSpacing: "0.08em", textTransform: "uppercase" }}>Clinical context</div>
                    <div style={{ fontSize: 14, color: "#444", lineHeight: 1.7 }}>
                      {result.attribution.clinical_context}
                    </div>
                  </div>
                </div>
              </div>

              {/* Drift */}
              {result.drift && (
                <div className="panel">
                  <div className="label">Session drift</div>
                  <div style={{ display: "flex", gap: 40, alignItems: "center" }}>
                    <div>
                      <div className="mono" style={{ fontSize: 28, fontWeight: 300, color: result.drift.drift_detected ? "#7a3a2a" : "#4a7a4a" }}>
                        {result.drift.drift_detected ? result.drift.drift_severity : "Stable"}
                      </div>
                      <div style={{ fontSize: 13, color: "#444", marginTop: 4 }}>{result.drift.trend_direction}</div>
                    </div>
                    <div style={{ width: 1, background: "#1e1e1e", alignSelf: "stretch" }}></div>
                    <div style={{ display: "flex", gap: 32 }}>
                      <div>
                        <div style={{ fontSize: 11, color: "#444", marginBottom: 4 }}>Slope</div>
                        <div className="mono" style={{ fontSize: 16, color: "#666" }}>{result.drift.trend_slope > 0 ? "+" : ""}{result.drift.trend_slope.toFixed(2)}/window</div>
                      </div>
                      <div>
                        <div style={{ fontSize: 11, color: "#444", marginBottom: 4 }}>SQI delta</div>
                        <div className="mono" style={{ fontSize: 16, color: "#666" }}>{result.drift.sqi_delta > 0 ? "+" : ""}{result.drift.sqi_delta.toFixed(0)} pts</div>
                      </div>
                      {result.drift.estimated_minutes_to_critical && (
                        <div>
                          <div style={{ fontSize: 11, color: "#444", marginBottom: 4 }}>Critical in</div>
                          <div className="mono" style={{ fontSize: 16, color: "#8a4a4a" }}>~{result.drift.estimated_minutes_to_critical.toFixed(0)}min</div>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )}

            </div>
          )}
        </div>
      </div>
    </div>
  )
}