import { useNavigate } from "react-router-dom"
import { useState, useEffect, useRef } from "react"

function MedicalBackground() {
  return (
    <div style={{ position:"fixed", inset:0, zIndex:0, pointerEvents:"none", overflow:"hidden", background:"#ffffff" }}>
      {/* Syringe pattern — red at low opacity */}
      <svg style={{ position:"absolute", inset:0, width:"100%", height:"100%", opacity:0.12 }} xmlns="http://www.w3.org/2000/svg">
        <defs>
          <pattern id="syringepat" x="0" y="0" width="120" height="120" patternUnits="userSpaceOnUse">
            <g transform="translate(60,60) rotate(45)">
              <rect x="-5" y="-36" width="10" height="48" rx="3" fill="#111111"/>
              <rect x="-10" y="-42" width="20" height="14" rx="3" fill="#111111"/>
              <rect x="-2.5" y="12" width="5" height="18" fill="#111111"/>
              <polygon points="0,34 -4,30 4,30" fill="#111111"/>
              <line x1="-5" y1="-20" x2="-10" y2="-20" stroke="#111111" strokeWidth="2"/>
              <line x1="-5" y1="-8"  x2="-10" y2="-8"  stroke="#111111" strokeWidth="2"/>
              <line x1="-5" y1="4"   x2="-10" y2="4"   stroke="#111111" strokeWidth="2"/>
            </g>
          </pattern>
          <pattern id="crosspat" x="60" y="60" width="120" height="120" patternUnits="userSpaceOnUse">
            <rect x="54" y="46" width="12" height="28" rx="2" fill="#c8222a"/>
            <rect x="46" y="54" width="28" height="12" rx="2" fill="#c8222a"/>
          </pattern>
        </defs>
        <rect width="100%" height="100%" fill="url(#syringepat)"/>
        <rect width="100%" height="100%" fill="url(#crosspat)" opacity="0.4"/>
      </svg>

      {/* Subtle red ECG lines scrolling */}
      {[
        {top:"10%", dur:"22s", del:"0s",   op:0.15, c:"#8b0000"},
        {top:"28%", dur:"30s", del:"-9s",  op:0.10, c:"#8b0000"},
        {top:"48%", dur:"26s", del:"-16s", op:0.18, c:"#8b0000"},
        {top:"68%", dur:"34s", del:"-6s",  op:0.10, c:"#8b0000"},
        {top:"86%", dur:"20s", del:"-20s", op:0.13, c:"#8b0000"},
      ].map((l,i) => (
        <div key={i} style={{ position:"absolute", top:l.top, width:"200%", animation:`ecgScroll ${l.dur} linear infinite`, animationDelay:l.del }}>
          <svg width="2400" height="44" viewBox="0 0 2400 44">
            {Array.from({length:12}).map((_,j) => (
              <path key={j}
                d={`M${j*200},22 L${j*200+70},22 L${j*200+76},22 L${j*200+80},6 L${j*200+84},38 L${j*200+88},22 L${j*200+96},22 L${j*200+100},28 L${j*200+104},22 L${j*200+200},22`}
                stroke={l.c} strokeWidth="1.5" fill="none" opacity={l.op}
              />
            ))}
          </svg>
        </div>
      ))}
    </div>
  )
}

function HospitalAnimation() {
  const [phase, setPhase] = useState<0|1|2|3>(0)
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const phaseRef   = useRef<number>(0)

  useEffect(() => {
    const timings = [5000, 4000, 4000, 5500]

    const tick = () => {
      phaseRef.current = (phaseRef.current + 1) % 4
      setPhase(phaseRef.current as 0|1|2|3)
      timeoutRef.current = setTimeout(tick, timings[phaseRef.current])
    }

    timeoutRef.current = setTimeout(tick, timings[0])
    return () => { if (timeoutRef.current) clearTimeout(timeoutRef.current) }
  }, [])

  const ecgNormal = "M0,26 L30,26 L35,26 L39,10 L43,42 L47,26 L55,26 L60,32 L65,26 L100,26 L105,26 L109,10 L113,42 L117,26 L125,26 L130,32 L135,26 L170,26 L175,26 L179,10 L183,42 L187,26 L195,26 L200,26"
  const ecgChaos  = "M0,26 L8,12 L16,40 L24,8 L32,44 L40,18 L48,38 L56,6 L64,44 L72,16 L80,40 L88,10 L96,44 L104,20 L112,36 L120,8 L128,44 L136,16 L144,38 L152,4 L160,44 L168,18 L176,40 L184,8 L192,44 L200,26"

  const isAlarm   = phase === 1
  const isAnalyze = phase === 2
  const isResult  = phase === 3

  const phaseLabel = ["Normal operation","⚠ Alarm triggered","Analyzing signal...","Analysis complete"][phase]
  const phaseLabelColor = isAlarm?"#b91c1c":isAnalyze?"#1d6a4a":isResult?"#166534":"#6b7280"

  return (
    <div>
      <div style={{ textAlign:"center", marginBottom:28 }}>
        <div style={{ display:"inline-flex", alignItems:"center", gap:8, padding:"6px 16px", borderRadius:100, background:isAlarm?"#fef2f2":isAnalyze?"#f0fdf4":isResult?"#f0fdf4":"#f9fafb", border:`1px solid ${isAlarm?"#fecaca":isAnalyze?"#bbf7d0":isResult?"#86efac":"#e5e7eb"}`, transition:"all 1s" }}>
          <div style={{ width:6, height:6, borderRadius:"50%", background:isAlarm?"#ef4444":isAnalyze?"#22c55e":isResult?"#16a34a":"#9ca3af", animation:isAlarm?"alarmBlink 1.4s ease infinite":isAnalyze?"pulseDot 1.4s ease infinite":"none", transition:"background 1s" }} />
          <span style={{ fontSize:12, color:phaseLabelColor, transition:"color 1s" }}>{phaseLabel}</span>
        </div>
      </div>

      <div style={{ display:"flex", alignItems:"center", justifyContent:"center" }}>

        {/* LEFT — traditional monitor */}
        <div style={{ width:260, background:"#ffffff", border:`1.5px solid ${isAlarm?"#fca5a5":"#e5e7eb"}`, borderRadius:12, padding:20, transition:"border-color 1s, box-shadow 1s", boxShadow:isAlarm?"0 0 40px rgba(220,38,38,0.12)":"0 1px 8px rgba(0,0,0,0.06)" }}>
          <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:14 }}>
            <div>
              <div style={{ fontSize:10, color:"#9ca3af", letterSpacing:"0.1em", textTransform:"uppercase" }}>ICU Monitor · Bed 4</div>
              <div style={{ fontSize:12, color:"#6b7280", marginTop:2 }}>SpO₂ · Session 2h 14m</div>
            </div>
            <div style={{ fontSize:9, padding:"3px 8px", borderRadius:4, letterSpacing:"0.08em", textTransform:"uppercase", fontFamily:"DM Mono", background:isAlarm?"#fef2f2":"#f0fdf4", border:`1px solid ${isAlarm?"#fca5a5":"#86efac"}`, color:isAlarm?"#dc2626":"#16a34a", animation:isAlarm?"alarmBlink 1.4s ease infinite":"none", transition:"all 1s" }}>
              {isAlarm ? "⚠ ALARM" : "● Live"}
            </div>
          </div>

          <div style={{ background:"#f8fafc", borderRadius:6, padding:"10px 12px", marginBottom:14, height:72, overflow:"hidden", position:"relative", border:"1px solid #f1f5f9" }}>
            <svg width="200" height="52" viewBox="0 0 200 52" style={{ display:"block" }}>
              <path d={isAlarm ? ecgChaos : ecgNormal} stroke={isAlarm?"#ef4444":"#16a34a"} strokeWidth="1.5" fill="none" style={{ transition:"stroke 1s" }}/>
            </svg>
            {isAlarm && <div style={{ position:"absolute", inset:0, background:"rgba(220,38,38,0.04)", animation:"alarmPulse 1.4s ease infinite" }} />}
          </div>

          <div style={{ display:"flex", justifyContent:"space-between" }}>
            {[{label:"SpO₂",val:isAlarm?"--":"98%",bad:isAlarm},{label:"HR",val:"72 bpm",bad:false},{label:"RR",val:"16/m",bad:false}].map((m,i) => (
              <div key={i} style={{ textAlign:"center" }}>
                <div style={{ fontSize:16, fontFamily:"DM Mono", color:m.bad?"#dc2626":"#374151", fontWeight:300, transition:"color 1s" }}>{m.val}</div>
                <div style={{ fontSize:10, color:"#9ca3af", marginTop:2 }}>{m.label}</div>
              </div>
            ))}
          </div>
          <div style={{ marginTop:16, fontSize:10, color:"#d1d5db", textAlign:"center" }}>Standard threshold monitor</div>
        </div>

        {/* CENTER */}
        <div style={{ display:"flex", flexDirection:"column", alignItems:"center", width:160, gap:12 }}>
          <div style={{ height:1, width:"100%", background:"#e5e7eb", position:"relative" }}>
            {isAnalyze && <div style={{ position:"absolute", top:-1, left:0, width:"28px", height:3, background:"#16a34a", borderRadius:2, animation:"scanRight 2.5s ease-in-out infinite" }} />}
          </div>
          <div style={{ background:isAnalyze||isResult?"#f0fdf4":"#f9fafb", border:`1px solid ${isAnalyze||isResult?"#86efac":"#e5e7eb"}`, borderRadius:8, padding:"12px 18px", textAlign:"center", transition:"all 1s", minWidth:120 }}>
            {phase===0 && <div style={{ fontSize:11, color:"#9ca3af" }}>Standby</div>}
            {phase===1 && <div style={{ fontSize:11, color:"#ef4444", animation:"alarmBlink 1.4s infinite" }}>Signal anomaly</div>}
            {phase===2 && (
              <div>
                <div style={{ fontSize:10, color:"#16a34a", letterSpacing:"0.08em", marginBottom:6 }}>ANALYZING</div>
                <div style={{ display:"flex", gap:4, justifyContent:"center" }}>
                  {[0,1,2].map(i => <div key={i} style={{ width:4, height:4, borderRadius:"50%", background:"#16a34a", animation:`dotBounce 1.4s ease ${i*0.25}s infinite` }} />)}
                </div>
              </div>
            )}
            {phase===3 && <div style={{ fontSize:11, color:"#16a34a" }}>✓ Complete</div>}
          </div>
          <div style={{ height:1, width:"100%", background:"#e5e7eb" }} />
          <div style={{ fontSize:9, color:"#9ca3af", letterSpacing:"0.08em", textTransform:"uppercase" }}>SigmaMedStat</div>
        </div>

        {/* RIGHT — result */}
        <div style={{ width:260, background:"#ffffff", border:`1.5px solid ${isResult?"#86efac":"#e5e7eb"}`, borderRadius:12, padding:20, opacity:isResult?1:0.2, transform:isResult?"translateY(0)":"translateY(10px)", transition:"opacity 1.2s, transform 1.2s, border-color 1.2s", boxShadow:isResult?"0 1px 8px rgba(0,0,0,0.06)":"none" }}>
          <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:14 }}>
            <div>
              <div style={{ fontSize:10, color:"#9ca3af", letterSpacing:"0.1em", textTransform:"uppercase" }}>SigmaMedStat · Bed 4</div>
              <div style={{ fontSize:12, color:"#6b7280", marginTop:2 }}>Signal Intelligence</div>
            </div>
            <div style={{ fontSize:9, padding:"3px 8px", borderRadius:4, background:"#f0fdf4", border:"1px solid #86efac", color:"#16a34a", letterSpacing:"0.08em", textTransform:"uppercase" }}>● Active</div>
          </div>

          <div style={{ background:"#fef2f2", borderRadius:6, padding:14, marginBottom:14, border:"1px solid #fecaca" }}>
            <div style={{ display:"flex", alignItems:"center", gap:14 }}>
              <div style={{ textAlign:"center" }}>
                <div style={{ fontSize:38, fontWeight:300, color:"#dc2626", fontFamily:"DM Mono", lineHeight:1 }}>24</div>
                <div style={{ fontSize:10, color:"#fca5a5" }}>/ 100</div>
                <div style={{ fontSize:9, color:"#dc2626", marginTop:4, letterSpacing:"0.08em" }}>CRITICAL</div>
              </div>
              <div>
                <div style={{ fontSize:12, color:"#b45309", marginBottom:8 }}>⚠ Probable false alarm</div>
                <div style={{ fontSize:11, color:"#166534" }}>Sensor displacement</div>
                <div style={{ fontSize:10, color:"#4ade80", marginTop:2 }}>Confidence 95%</div>
              </div>
            </div>
          </div>

          <div style={{ display:"flex", flexDirection:"column", gap:7 }}>
            {["Flatline — signal not changing","Patient repositioned 10s ago","ECG still normal — patient fine"].map((e,i) => (
              <div key={i} style={{ display:"flex", gap:8, fontSize:11, color:"#4b5563" }}>
                <span style={{ color:"#16a34a", flexShrink:0 }}>→</span><span>{e}</span>
              </div>
            ))}
          </div>
          <div style={{ marginTop:14, paddingTop:12, borderTop:"1px solid #f0fdf4", fontSize:11, color:"#16a34a" }}>
            Recommendation: Reseat sensor · Do not act on reading
          </div>
        </div>
      </div>
    </div>
  )
}

export default function HomePage() {
  const navigate = useNavigate()

  return (
    <div style={{ minHeight:"100vh", fontFamily:"'DM Sans','Helvetica Neue',sans-serif" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500&family=DM+Mono:wght@400;500&display=swap');
        *{box-sizing:border-box;margin:0;padding:0;}
        @keyframes ecgScroll{from{transform:translateX(0)}to{transform:translateX(-50%)}}
        @keyframes alarmBlink{0%,100%{opacity:0.4}50%{opacity:1}}
        @keyframes alarmPulse{0%,100%{opacity:0}50%{opacity:0.8}}
        @keyframes scanRight{0%{left:0;opacity:1}75%{left:calc(100% - 28px);opacity:0.3}100%{left:calc(100% - 28px);opacity:0}}
        @keyframes dotBounce{0%,100%{transform:translateY(0);opacity:0.3}50%{transform:translateY(-5px);opacity:1}}
        @keyframes fadeUp{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:translateY(0)}}
        @keyframes pulseDot{0%,100%{transform:scale(1)}50%{transform:scale(1.4)}}
        .fi{animation:fadeUp 0.8s ease forwards;opacity:0;}
        .d1{animation-delay:0.1s}.d2{animation-delay:0.3s}.d3{animation-delay:0.5s}.d4{animation-delay:0.7s}
        .btn{background:#111827;color:#ffffff;border:none;padding:14px 34px;border-radius:6px;font-size:15px;font-weight:400;cursor:pointer;font-family:inherit;transition:background 0.2s;}
        .btn:hover{background:#1f2937;}
        .card{background:#ffffff;border:1px solid #e5e7eb;border-radius:10px;padding:28px;box-shadow:0 1px 4px rgba(0,0,0,0.04);}
        .step{display:flex;gap:16px;padding:24px 0;border-bottom:1px solid #f3f4f6;}
        .step:last-child{border-bottom:none;}
        .stepnum{width:28px;height:28px;border-radius:50%;border:1px solid #e5e7eb;display:flex;align-items:center;justify-content:center;font-size:11px;color:#9ca3af;flex-shrink:0;font-family:DM Mono,monospace;margin-top:2px;}
      `}</style>

      <MedicalBackground />

      {/* Navbar needs white bg override */}
      <div style={{ position:"fixed", top:0, left:0, right:0, zIndex:100, background:"rgba(255,255,255,0.92)", borderBottom:"1px solid #f3f4f6", backdropFilter:"blur(8px)" }}>
        <div style={{ maxWidth:960, margin:"0 auto", padding:"0 40px", height:60, display:"flex", alignItems:"center", justifyContent:"space-between" }}>
          <div style={{ display:"flex", alignItems:"center", gap:10, cursor:"pointer" }} onClick={() => navigate("/")}>
            <div style={{ width:26, height:26, borderRadius:6, background:"#fef2f2", border:"1px solid #fecaca", display:"flex", alignItems:"center", justifyContent:"center", fontSize:13, color:"#dc2626", fontFamily:"DM Mono" }}>Σ</div>
            <span style={{ fontSize:15, color:"#374151", fontWeight:400 }}>SigmaMedStat</span>
          </div>
          <div style={{ display:"flex", alignItems:"center", gap:32 }}>
            <span onClick={() => navigate("/")} style={{ fontSize:14, color:"#9ca3af", cursor:"pointer" }}>Home</span>
            <span onClick={() => navigate("/demo")} style={{ fontSize:14, color:"#9ca3af", cursor:"pointer" }}>Demo</span>
            <button onClick={() => navigate("/demo")} style={{ background:"#111827", color:"#fff", border:"none", padding:"8px 20px", borderRadius:6, fontSize:13, cursor:"pointer", fontFamily:"inherit" }}>Try it →</button>
          </div>
        </div>
      </div>

      <div style={{ position:"relative", zIndex:1 }}>

        {/* Hero */}
        <section style={{ maxWidth:960, margin:"0 auto", padding:"130px 40px 60px", textAlign:"center" }}>
          <div className="fi d1" style={{ display:"inline-flex", alignItems:"center", gap:8, background:"#fef2f2", border:"1px solid #fecaca", borderRadius:100, padding:"5px 14px", marginBottom:36 }}>
            <div style={{ width:6, height:6, borderRadius:"50%", background:"#dc2626", animation:"pulseDot 2s ease infinite" }} />
            <span style={{ fontSize:11, color:"#dc2626", letterSpacing:"0.1em", textTransform:"uppercase" }}>Signal Intelligence Platform</span>
          </div>

          <h1 className="fi d2" style={{ fontSize:"clamp(34px, 5.5vw, 62px)", fontWeight:300, color:"#111827", lineHeight:1.1, letterSpacing:"-2px", marginBottom:22 }}>
            Should you trust what your<br />
            <span style={{ color:"#dc2626" }}>medical devices are reporting?</span>
          </h1>

          <p className="fi d3" style={{ fontSize:17, color:"#6b7280", maxWidth:480, margin:"0 auto 44px", lineHeight:1.8 }}>
            SigmaMedStat determines whether a device reading is trustworthy 
            before any clinical decision is made.
          </p>

          <div className="fi d4">
            <button className="btn" onClick={() => navigate("/demo")}>Open live demo →</button>
          </div>
        </section>

        {/* Animation */}
        <section style={{ maxWidth:960, margin:"0 auto", padding:"0 40px 100px" }}>
          <div style={{ fontSize:11, color:"#d1d5db", letterSpacing:"0.1em", textTransform:"uppercase", textAlign:"center", marginBottom:32 }}>Watch it in action</div>
          <HospitalAnimation />
          <div style={{ textAlign:"center", marginTop:20, fontSize:12, color:"#d1d5db" }}>
            Simulated — SpO₂ flatline after patient repositioning
          </div>
        </section>

        {/* Stats */}
        <section style={{ borderTop:"1px solid #f3f4f6", borderBottom:"1px solid #f3f4f6", background:"rgba(255,255,255,0.8)" }}>
          <div style={{ maxWidth:960, margin:"0 auto", padding:"0 40px", display:"flex" }}>
            {[
              {num:"85–99%", label:"of ICU alarms are false positives"},
              {num:"350+",   label:"alarms per patient per day"},
              {num:"34ms",   label:"full pipeline response time"},
              {num:"6",      label:"clinical failure modes detected"},
            ].map((s,i) => (
              <div key={i} style={{ flex:1, padding:"36px 0", ...(i>0?{paddingLeft:40, borderLeft:"1px solid #f3f4f6"}:{}), paddingRight:40 }}>
                <div style={{ fontSize:36, fontWeight:300, color:"#dc2626", fontFamily:"DM Mono, monospace", letterSpacing:"-1px" }}>{s.num}</div>
                <div style={{ fontSize:13, color:"#9ca3af", marginTop:6 }}>{s.label}</div>
              </div>
            ))}
          </div>
        </section>

        {/* Problem */}
        <section style={{ maxWidth:960, margin:"0 auto", padding:"80px 40px", background:"rgba(255,255,255,0.7)" }}>
          <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:60, alignItems:"start" }}>
            <div>
              <p style={{ fontSize:11, color:"#9ca3af", letterSpacing:"0.1em", textTransform:"uppercase", marginBottom:20 }}>The problem</p>
              <h2 style={{ fontSize:30, fontWeight:300, color:"#111827", lineHeight:1.3, letterSpacing:"-0.5px", marginBottom:20 }}>
                Alarm fatigue is a documented clinical crisis
              </h2>
              <p style={{ fontSize:14, color:"#6b7280", lineHeight:1.8, marginBottom:16 }}>
                A patient moves. A sensor slips. The monitor screams. Nurses — 
                conditioned to hundreds of false alarms daily — ignore it. 
                Sometimes that's the one that mattered.
              </p>
              <p style={{ fontSize:14, color:"#6b7280", lineHeight:1.8 }}>
                Current systems ask whether a reading is abnormal. 
                SigmaMedStat asks whether the reading should be trusted at all.
              </p>
            </div>
            <div className="card">
              <p style={{ fontSize:11, color:"#9ca3af", letterSpacing:"0.1em", textTransform:"uppercase", marginBottom:16 }}>What changes</p>
              {[
                {before:"Is this signal abnormal?",     after:"Should we trust this signal?"},
                {before:"Threshold alert fires",         after:"Context-aware trust score"},
                {before:"Nurse ignores alarm #347",      after:"False alarm suppressed with evidence"},
                {before:"Unknown failure cause",          after:"Named: sensor displacement, 95% confidence"},
              ].map((r,i) => (
                <div key={i} style={{ paddingBottom:14, marginBottom:14, borderBottom:i<3?"1px solid #f9fafb":"none" }}>
                  <div style={{ fontSize:12, color:"#d1d5db", marginBottom:4, textDecoration:"line-through" }}>{r.before}</div>
                  <div style={{ fontSize:12, color:"#111827" }}>→ {r.after}</div>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* How it works */}
        <section style={{ borderTop:"1px solid #f3f4f6", background:"rgba(249,250,251,0.8)" }}>
          <div style={{ maxWidth:960, margin:"0 auto", padding:"80px 40px" }}>
            <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:60 }}>
              <div>
                <p style={{ fontSize:11, color:"#9ca3af", letterSpacing:"0.1em", textTransform:"uppercase", marginBottom:24 }}>How it works</p>
                <div>
                  {[
                    {n:"01", title:"Signal analysis",      body:"Detects flatlines, spikes, dropouts, and noise — the four ways sensors produce untrustworthy data."},
                    {n:"02", title:"Context correlation",  body:"Cross-references neighboring signals, patient motion, and clinical events. Context changes everything."},
                    {n:"03", title:"Trust scoring",        body:"Fuses signal quality and context into a 0–100 trust score: Excellent, Good, Degraded, Poor, or Critical."},
                    {n:"04", title:"Failure attribution",  body:"Names the cause — sensor displacement, motion artifact, calibration drift — with supporting evidence."},
                  ].map(s => (
                    <div key={s.n} className="step">
                      <div className="stepnum">{s.n}</div>
                      <div>
                        <h3 style={{ fontSize:14, fontWeight:400, color:"#374151", marginBottom:6 }}>{s.title}</h3>
                        <p style={{ fontSize:13, color:"#6b7280", lineHeight:1.7 }}>{s.body}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
              <div>
                <p style={{ fontSize:11, color:"#9ca3af", letterSpacing:"0.1em", textTransform:"uppercase", marginBottom:21 }}>Capabilities</p>
                {[
                  {label:"Signal Reliability Engine",   sub:"Flatline · Spike · Dropout · Noise"},
                  {label:"Context Correlation Layer",   sub:"Motion · Events · Neighbors · History"},
                  {label:"0–100 Trust Scoring",         sub:"Graded · Confidence-weighted"},
                  {label:"Failure Attribution",          sub:"5 failure categories · Evidence-backed"},
                  {label:"Temporal Drift Monitor",       sub:"Session degradation · Prediction"},
                  {label:"IEC 62304 Aware",             sub:"ISO 14971 · FDA AI/ML SaMD aligned"},
                ].map((f,i) => (
                  <div key={i} style={{ padding:"14px 18px", background:"#ffffff", border:"1px solid #f3f4f6", borderRadius:8, marginBottom:4, boxShadow:"0 1px 2px rgba(0,0,0,0.03)" }}>
                    <div style={{ fontSize:13, color:"#374151", marginBottom:2 }}>{f.label}</div>
                    <div style={{ fontSize:11, color:"#9ca3af" }}>{f.sub}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>

        {/* CTA */}
        <section style={{ maxWidth:960, margin:"0 auto", padding:"60px 40px 80px" }}>
          <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between", padding:"44px 52px", border:"1px solid #e5e7eb", borderRadius:12, background:"#ffffff", boxShadow:"0 1px 4px rgba(0,0,0,0.04)" }}>
            <div>
              <h2 style={{ fontSize:24, fontWeight:300, color:"#111827", letterSpacing:"-0.5px", marginBottom:8 }}>See it running on real ICU data</h2>
              <p style={{ fontSize:14, color:"#9ca3af" }}>Six clinical scenarios. Full pipeline. Live results.</p>
            </div>
            <button className="btn" onClick={() => navigate("/demo")} style={{ flexShrink:0, marginLeft:40 }}>
              Open demo →
            </button>
          </div>
        </section>

        {/* Footer */}
        <footer style={{ borderTop:"1px solid #f3f4f6", maxWidth:960, margin:"0 auto", padding:"32px 40px", display:"flex", justifyContent:"space-between", alignItems:"center" }}>
          <div style={{ display:"flex", alignItems:"center", gap:10 }}>
            <div style={{ width:24, height:24, borderRadius:6, background:"#fef2f2", border:"1px solid #fecaca", display:"flex", alignItems:"center", justifyContent:"center", fontSize:12, color:"#dc2626", fontFamily:"DM Mono" }}>Σ</div>
            <span style={{ fontSize:14, color:"#6b7280" }}>SigmaMedStat</span>
            {/* Arunkumar Ramachandran */}
            <span style={{ fontSize:14, color:"#d1d5db" }}>· Built by Arunkumar Ramachandran</span>
          </div>
          <div style={{ fontSize:12, color:"#d1d5db" }}>IEC 62304 · ISO 14971 · FDA AI/ML SaMD</div>
        </footer>
      </div>
    </div>
  )
}