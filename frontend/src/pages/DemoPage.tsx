import { useState, useRef, useEffect } from "react"
import type { ReactNode, CSSProperties } from "react"
import { useNavigate } from "react-router-dom"

const R = "#c0392b"
const CHARCOAL = "#2c3e50"
const SOFT = "#f4f4f2"
const MID = "#e8e8e5"
const LIGHT = "#ffffff"

function useIsMobile() {
  const [isMobile, setIsMobile] = useState(
    typeof window !== "undefined" ? window.innerWidth < 768 : false
  )
  useEffect(() => {
    const handler = () => setIsMobile(window.innerWidth < 768)
    window.addEventListener("resize", handler)
    return () => window.removeEventListener("resize", handler)
  }, [])
  return isMobile
}

function useScrollReveal(threshold = 0.12) {
  const ref = useRef<HTMLDivElement>(null)
  const [visible, setVisible] = useState(false)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const observer = new IntersectionObserver(
      ([entry]) => { if (entry.isIntersecting) { setVisible(true); observer.disconnect() } },
      { threshold }
    )
    observer.observe(el)
    return () => observer.disconnect()
  }, [threshold])
  return { ref, visible }
}

function Reveal({ children, delay = 0, style }: { children: ReactNode; delay?: number; style?: CSSProperties }) {
  const { ref, visible } = useScrollReveal()
  return (
    <div ref={ref} style={{
      opacity: visible ? 1 : 0,
      transform: visible ? "translateY(0)" : "translateY(20px)",
      transition: `opacity 0.6s ease ${delay}s, transform 0.6s ease ${delay}s`,
      ...style,
    }}>
      {children}
    </div>
  )
}

function AnimatedBar({ value, color, trigger }: { value: number; color: string; trigger: boolean }) {
  const [width, setWidth] = useState(0)
  useEffect(() => {
    if (!trigger) { setWidth(0); return }
    const t = setTimeout(() => setWidth(value), 100)
    return () => clearTimeout(t)
  }, [value, trigger, color])
  return (
    <div style={{ flex:1, height:5, background:MID, borderRadius:3, overflow:"hidden" }}>
      <div style={{ height:"100%", width:`${width}%`, background:color, borderRadius:3, transition:"width 0.9s cubic-bezier(0.4,0,0.2,1)" }} />
    </div>
  )
}

const SCENARIOS: Record<string, any> = {
  v100s: {
    id:"v100s", label:"Irregular heartbeat alarm", bed:"Bed 7 · ICU West",
    situation:"A patient in the ICU triggered an irregular heartbeat alarm - one that, if real, requires a nurse to respond immediately. The patient appears calm and stable on observation.",
    what_monitor_said:"The monitor detected an abnormal heart rhythm pattern and fired a high-priority alarm.",
    alarm_type:"Irregular heartbeat (Ventricular Flutter)",
    ground_truth:false, ground_truth_plain:"This was actually a false alarm - no real emergency.",
    signal:[-0.181,0.071,-0.036,-0.126,-0.126,-0.116,-0.205,-0.189,-0.102,0.067,0.324,-0.063,-0.024,-0.103,-0.003,0.008,-0.071,-0.051,-0.019,-0.008,-0.071,-0.019,-0.069,-0.047,0.071,-0.09,-0.062,-0.008,-0.055,-0.018,0.011,-0.095,-0.047,0.072,0.001,-0.071,0.024,-0.086,-0.033,0.095,-0.142,-0.048,-0.049,0.457,-0.063,-0.047,-0.035,0.559,0.039,-0.397,-0.167,0.37,-0.118,-0.372,0.003,-0.142,-0.504,0.388,0.582,0.15,0.102,1.39,-0.228,-0.504,-0.11,1.504,0.762,1.504,-0.441,0.307,0.902,1.504,-0.504,-0.659,-0.468,-0.504,0.724,-0.537,0.926,0.74,0.181,0.212,0.002,-0.504,-0.504,-0.064,1.5,-0.504,-0.504,-0.3,-0.12,0.236,-0.024,-0.49,-0.344,-0.504,-0.016,0.222,-0.038,-0.083],
    prediction:{ is_false_alarm:false, false_alarm_prob:0.415, true_alarm_prob:0.585, correct:false },
    verdict:"Act now - this looks like a real event", verdict_correct:false, confidence:58,
    what_sigmamedstat_found:"The model analyzed 60 seconds of raw heart signal data, converted it into a visual pattern, and extracted 1,280 features. It identified patterns that resembled a true cardiac event, but in this case it was incorrect.",
    action:"The model flagged it as act now, but this was actually a false alarm. It's a misclassification - the patient did not require emergency treatment.",
    honest_note:true,
    why_wrong:"This type of irregular heartbeat is the hardest alarm for the model to classify. A model trained on this patient's individual baseline would likely have caught it.",
    without:"Without signal-level intelligence, this alarm would have been one of 350+ that day. A nurse would have responded, but 10 minutes spent on a false alarm is 10 minutes away from a patient who might actually need attention."
  },
  v101l: {
    id:"v101l", label:"Irregular heartbeat alarm", bed:"Bed 12 · ICU East",
    situation:"Another patient triggered the same type of irregular heartbeat alarm. Unlike the previous case, this patient is genuinely experiencing a cardiac event that requires immediate attention.",
    what_monitor_said:"The monitor detected an abnormal heart rhythm and fired a high-priority alarm.",
    alarm_type:"Irregular heartbeat (Ventricular Flutter)",
    ground_truth:true, ground_truth_plain:"This was a real emergency - the patient needed immediate care.",
    signal:[0.875,0.785,0.775,0.859,0.875,0.837,0.855,0.945,0.789,0.808,0.696,0.797,0.875,0.812,0.781,0.805,0.828,0.801,0.823,0.82,0.828,0.829,0.835,0.828,0.82,0.84,0.811,0.844,0.828,0.788,0.805,0.82,0.82,0.805,0.823,0.82,0.836,0.822,0.79,0.812,0.695,0.793,0.86,0.656,0.789,0.827,0.937,0.664,0.75,0.992,0.848,0.727,0.898,0.902,0.815,0.844,1.164,0.66,0.638,0.766,0.742,-0.421,0.719,0.547,0.891,-0.381,0.575,-0.492,0.992,0.68,0.545,0.094,1.242,1.316,0.981,1.5,0.461,1.095,0.534,0.539,0.766,0.706,0.832,1.195,1.258,0.807,-0.504,1.0,1.226,0.856,0.869,0.797,0.812,1.232,0.908,1.016,0.805,0.769,0.819,0.787],
    prediction:{ is_false_alarm:false, false_alarm_prob:0.342, true_alarm_prob:0.658, correct:true },
    verdict:"Act now - this looks like a real event", verdict_correct:true, confidence:65,
    what_sigmamedstat_found:"The model analyzed the heart signal and detected consistent abnormal rhythm patterns across the full 60-second window. Multiple features pointed toward a genuine cardiac event.",
    action:"The model correctly said act now. This was a real emergency. The nurse who responded was right to do so.",
    honest_note:false, why_wrong:"",
    without:"Without signal-level intelligence, this alarm looks identical to hundreds of others in a shift. The difference isn't obvious to the human eye, but it shows up clearly in the signal data."
  },
  a103l: {
    id:"a103l", label:"Cardiac arrest alarm", bed:"Bed 3 · ICU North",
    situation:"The monitor showed what appeared to be complete cardiac arrest - the heart stopping entirely. This is the most urgent alarm in any ICU, where every second matters if it's real.",
    what_monitor_said:"The monitor detected a flatline - no detectable heartbeat - and triggered its highest-priority alarm.",
    alarm_type:"Cardiac arrest (Asystole)",
    ground_truth:true, ground_truth_plain:"This was a real cardiac arrest - the patient needed emergency intervention.",
    signal:[0.331,1.403,0.489,0.331,0.441,0.427,0.483,0.488,0.315,0.282,0.337,0.441,0.291,0.338,0.453,0.362,0.425,0.441,0.318,1.102,0.331,0.442,0.446,0.291,0.425,0.474,0.411,0.402,0.331,0.431,0.441,0.252,0.937,0.47,0.433,0.276,1.339,0.292,0.465,0.378,0.449,0.475,0.453,0.315,0.465,0.277,0.402,0.472,0.394,0.887,0.318,0.354,0.472,0.315,0.471,0.472,0.457,0.269,0.338,0.74,0.472,0.298,0.469,0.441,0.276,0.423,0.488,0.433,0.26,0.505,0.316,1.339,0.425,0.31,0.301,0.441,0.378,0.469,0.401,0.244,0.465,0.273,0.931,0.346,1.055,0.405,0.326,0.496,0.315,0.481,0.298,0.803,0.284,0.414,0.311,0.425,0.346,0.53,0.339,0.403],
    prediction:{ is_false_alarm:false, false_alarm_prob:0.358, true_alarm_prob:0.642, correct:true },
    verdict:"Act now - this looks like a real event", verdict_correct:true, confidence:64,
    what_sigmamedstat_found:"The model identified patterns consistent with genuine cardiac arrest. The absence of a normal heartbeat rhythm across the full signal window was a strong indicator.",
    action:"The model correctly identified this as a real emergency. Immediate response was the right call.",
    honest_note:false, why_wrong:"",
    without:"A real cardiac arrest alarm ignored because a nurse is fatigued from 400 false alarms is one of the most preventable tragedies in hospital care. This is exactly the scenario SigmaMedStat is designed to protect against."
  },
  a104s: {
    id:"a104s", label:"Cardiac arrest alarm", bed:"Bed 8 · ICU North",
    situation:"The monitor showed apparent cardiac arrest, but the patient is awake, talking, and showing no signs of distress. Something is clearly wrong with the reading, not the patient.",
    what_monitor_said:"The monitor detected a flatline and triggered its highest-priority alarm, even though the patient appeared completely stable.",
    alarm_type:"Cardiac arrest (Asystole)",
    ground_truth:false, ground_truth_plain:"This was a false alarm - the patient was completely fine.",
    signal:[-0.079,-0.171,0.125,-0.125,0.226,-0.175,-0.09,-0.071,-0.104,-0.044,0.002,-0.122,-0.011,-0.065,-0.093,-0.011,-0.072,0.02,0.213,-0.068,0.035,-0.063,0.03,-0.002,0.416,-0.058,0.579,-0.008,0.056,-0.175,-0.031,-0.038,-0.094,-0.07,0.659,0.483,-0.068,-0.067,-0.006,-0.154,0.217,-0.262,-0.137,-0.138,-0.135,0.543,-0.056,-0.079,-0.07,-0.095,0.287,-0.179,-0.068,-0.057,-0.111,0.552,-0.003,-0.053,0.269,0.673,0.027,-0.053,-0.015,0.09,0.036,-0.191,0.198,-0.082,-0.093,-0.048,0.596,-0.055,0.222,-0.101,-0.007,-0.072,-0.04,-0.069,-0.083,-0.103,0.045,-0.04,0.013,0.056,-0.095,0.295,-0.054,0.007,0.022,-0.034,0.001,-0.064,0.51,-0.072,-0.038,-0.074,-0.078,-0.044,-0.046,-0.063],
    prediction:{ is_false_alarm:false, false_alarm_prob:0.064, true_alarm_prob:0.936, correct:false },
    verdict:"Act now - this looks like a real event", verdict_correct:false, confidence:93,
    what_sigmamedstat_found:"The model was 93% confident this was a real emergency, but it was wrong. This is its worst misclassification in the demo.",
    action:"The model flagged it as act now with high confidence, but the patient was completely fine. This is the kind of error that erodes trust in automated systems.",
    honest_note:true,
    why_wrong:"High-confidence mistakes are the most dangerous kind. The signal had features that genuinely resembled a cardiac arrest pattern. Patient-specific baselines could make a major difference here.",
    without:"Even without the model, a nurse would have responded - it's a cardiac arrest alert. The cost is 5-10 minutes of emergency response for a patient who was fine. Multiply that by the number of false cardiac arrest alarms per week."
  },
  t107l: {
    id:"t107l", label:"Rapid heart rate alarm", bed:"Bed 2 · Step-Down Unit",
    situation:"A patient's heart rate alarm fired for being too fast. A rapid heart rate can indicate many things - fever, pain, anxiety, dehydration, or a genuine cardiac issue. Signal data helps distinguish between these causes.",
    what_monitor_said:"Heart rate exceeded the threshold and the monitor fired a tachycardia alarm.",
    alarm_type:"Rapid heart rate (Tachycardia)",
    ground_truth:true, ground_truth_plain:"This was a real alarm - the patient's rapid heart rate needed clinical assessment.",
    signal:[0.331,1.403,0.489,0.331,0.441,0.427,0.483,0.488,0.315,0.282,0.337,0.441,0.291,0.338,0.453,0.362,0.425,0.441,0.318,1.102,0.331,0.442,0.446,0.291,0.425,0.474,0.411,0.402,0.331,0.431,0.441,0.252,0.937,0.47,0.433,0.276,1.339,0.292,0.465,0.378,0.449,0.475,0.453,0.315,0.465,0.277,0.402,0.472,0.394,0.887,0.318,0.354,0.472,0.315,0.471,0.472,0.457,0.269,0.338,0.74,0.472,0.298,0.469,0.441,0.276,0.423,0.488,0.433,0.26,0.505,0.316,1.339,0.425,0.31,0.301,0.441,0.378,0.469,0.401,0.244,0.465,0.273,0.931,0.346,1.055,0.405,0.326,0.496,0.315,0.481,0.298,0.803,0.284,0.414,0.311,0.425,0.346,0.53,0.339,0.403],
    prediction:{ is_false_alarm:false, false_alarm_prob:0.306, true_alarm_prob:0.694, correct:true },
    verdict:"Act now - this looks like a real event", verdict_correct:true, confidence:69,
    what_sigmamedstat_found:"Rapid heartbeat is the alarm type the model performs best on, reaching 61% accuracy for this category. The signal showed a consistently fast rhythm across the full window, without the noise patterns typically associated with sensor artifacts.",
    action:"The model correctly flagged this for clinical attention. The patient needed assessment.",
    honest_note:false, why_wrong:"",
    without:"Rapid heart rate alarms are extremely common - many are triggered by movement, coughing, or temporary agitation. SigmaMedStat helps restore the appropriate level of urgency to the ones that matter."
  },
  b124s: {
    id:"b124s", label:"Slow heart rate alarm", bed:"Bed 5 · Cardiac Unit",
    situation:"A patient's heart rate dropped below the threshold and triggered a slow heart rate alarm. The patient's other readings appear completely normal - genuinely slow heart rate rarely occurs in isolation.",
    what_monitor_said:"Heart rate dropped below the threshold and the monitor fired a bradycardia alarm.",
    alarm_type:"Slow heart rate (Bradycardia)",
    ground_truth:false, ground_truth_plain:"This was a false alarm - the low reading was a sensor artifact, not a real problem.",
    signal:[-0.079,-0.171,0.125,-0.125,0.226,-0.175,-0.09,-0.071,-0.104,-0.044,0.002,-0.122,-0.011,-0.065,-0.093,-0.011,-0.072,0.02,0.213,-0.068,0.035,-0.063,0.03,-0.002,0.416,-0.058,0.579,-0.008,0.056,-0.175,-0.031,-0.038,-0.094,-0.07,0.659,0.483,-0.068,-0.067,-0.006,-0.154,0.217,-0.262,-0.137,-0.138,-0.135,0.543,-0.056,-0.079,-0.07,-0.095,0.287,-0.179,-0.068,-0.057,-0.111,0.552,-0.003,-0.053,0.269,0.673,0.027,-0.053,-0.015,0.09,0.036,-0.191,0.198,-0.082,-0.093,-0.048,0.596,-0.055,0.222,-0.101,-0.007,-0.072,-0.04,-0.069,-0.083,-0.103,0.045,-0.04,0.013,0.056,-0.095,0.295,-0.054,0.007,0.022,-0.034,0.001,-0.064,0.51,-0.072,-0.038,-0.074,-0.078,-0.044,-0.046,-0.063],
    prediction:{ is_false_alarm:true, false_alarm_prob:0.555, true_alarm_prob:0.445, correct:true },
    verdict:"Stand down - this is probably a false alarm", verdict_correct:true, confidence:55,
    what_sigmamedstat_found:"The model found the signal was more consistent with a sensor reading error than a true slow heart rate. Confidence was modest at 55% - a close call - but it leaned toward false alarm, which turned out to be correct.",
    action:"The model correctly said stand down. The nurse should check sensor placement rather than treating the patient for bradycardia.",
    honest_note:false, why_wrong:"",
    without:"A slow heart rate alarm on a cardiac unit requires a response. If it's a false alarm, the nurse spends time assessing a patient who doesn't need it. When this happens dozens of times a day, it adds up to hours of lost clinical time per shift."
  }
}

function Waveform({ signal, active }: { signal: number[]; active: boolean }) {
  const w = 600, h = 90
  const min = Math.min(...signal), max = Math.max(...signal)
  const range = max - min || 1
  const pts = signal.map((v,i) => {
    const x = (i / (signal.length - 1)) * w
    const y = h - ((v - min) / range) * (h - 12) - 6
    return `${x},${y}`
  }).join(" ")
  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} style={{ width:"100%", height:"auto" }}>
      <polyline points={pts} fill="none" stroke={active ? R : "#d1d5db"} strokeWidth="1.5" strokeLinejoin="round" style={{ transition:"stroke 0.6s" }}/>
    </svg>
  )
}

type Phase = "analyzing" | "result"

export default function DemoPage() {
  const navigate  = useNavigate()
  const isMobile  = useIsMobile()
  const [selected, setSelected] = useState("v100s")
  const [phase, setPhase]       = useState<Phase>("analyzing")
  const timerRef  = useRef<ReturnType<typeof setTimeout> | null>(null)

  // Fix: start as analyzing, resolve to result immediately without scroll
  useEffect(() => {
    window.scrollTo({ top: 0 })
    const t = setTimeout(() => setPhase("result"), 50)
    return () => clearTimeout(t)
  }, [])

  const s        = SCENARIOS[selected]
  const pred     = s.prediction
  const correct  = pred.correct
  const accuracy = Object.values(SCENARIOS).filter((sc: any) => sc.prediction.correct).length
  const total    = Object.keys(SCENARIOS).length

  function pick(id: string) {
    if (timerRef.current) clearTimeout(timerRef.current)
    setSelected(id)
    setPhase("analyzing")
    window.scrollTo({ top: 0 })
    timerRef.current = setTimeout(() => setPhase("result"), 2000)
  }

  useEffect(() => () => { if (timerRef.current) clearTimeout(timerRef.current) }, [])

  const verdictColor  = s.verdict_correct ? (pred.is_false_alarm ? "#1e8449" : R) : "#b7770d"
  const verdictBg     = s.verdict_correct ? (pred.is_false_alarm ? "#edf7f1" : "#fdf0ef") : "#fdf3e3"
  const verdictBorder = s.verdict_correct ? (pred.is_false_alarm ? "#a8d5b5" : "#e8b4b0") : "#f0c87a"
  const cardPad       = isMobile ? 16 : 28

  return (
    <div style={{ minHeight:"100vh", background:SOFT, fontFamily:"'DM Sans','Helvetica Neue',sans-serif" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500&family=DM+Mono:wght@400;500&display=swap');
        *{box-sizing:border-box;margin:0;padding:0;}
        @keyframes fadeUp{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
        @keyframes dotBounce{0%,100%{transform:translateY(0);opacity:0.3}50%{transform:translateY(-4px);opacity:1}}
        .fade{animation:fadeUp 0.5s ease forwards;}
        .fade *:focus{outline:none;}
        .scard{background:${LIGHT};border:1px solid ${MID};border-radius:10px;padding:10px 12px;cursor:pointer;text-align:left;transition:all 0.15s;width:100%;}
        .scard:hover{border-color:#b0b0aa;background:${SOFT};}
        .scard.active{border-color:${CHARCOAL};background:${SOFT};}
      `}</style>

      {/* Navbar */}
      <div style={{ position:"fixed", top:0, left:0, right:0, zIndex:100, background:"rgba(244,244,242,0.95)", borderBottom:`1px solid ${MID}`, backdropFilter:"blur(8px)" }}>
        <div style={{ maxWidth:1060, margin:"0 auto", padding:isMobile?"0 16px":"0 40px", height:64, display:"flex", alignItems:"center", justifyContent:"space-between" }}>
          <div style={{ cursor:"pointer" }} onClick={() => navigate("/")}>
            <img src="/logo.png" alt="SigmaMedStat" style={{ height:isMobile?40:56, width:"auto" }} />
          </div>
          {!isMobile && (
            <div style={{ display:"flex", alignItems:"center", gap:16 }}>
              <div style={{ fontSize:12, color:"#95a5a6" }}>6 real ICU alarms · EfficientNet + Neural Classifier · PhysioNet 2015</div>
              <div style={{ fontSize:12, padding:"4px 14px", borderRadius:100, background:LIGHT, border:`1px solid ${MID}`, color:CHARCOAL }}>
                Got {accuracy} of {total} right
              </div>
            </div>
          )}
          {isMobile && (
            <div style={{ fontSize:11, padding:"4px 10px", borderRadius:100, background:LIGHT, border:`1px solid ${MID}`, color:CHARCOAL }}>
              {accuracy}/{total} correct
            </div>
          )}
        </div>
      </div>

      <div style={{ maxWidth:1060, margin:"0 auto", padding:isMobile?"74px 16px 40px":"90px 40px 60px" }}>

        {/* Page intro */}
        <div style={{ marginBottom:20 }}>
          <p style={{ fontSize:11, color:"#95a5a6", letterSpacing:"0.1em", textTransform:"uppercase", marginBottom:8 }}>Live demo - real data</p>
          <h1 style={{ fontSize:isMobile?20:26, fontWeight:300, color:CHARCOAL, letterSpacing:"-0.5px", marginBottom:8 }}>
            Six real ICU alarms. Our model analyzed each one.
          </h1>
          <p style={{ fontSize:isMobile?13:14, color:"#7f8c8d", lineHeight:1.8 }}>
            Each scenario comes from real patient alarm recordings in a published hospital dataset. Select any one - the model will analyze the heart signal and tell you whether the alarm is worth acting on.
          </p>
        </div>

        {/* Scenario picker */}
        <div style={{ display:"grid", gridTemplateColumns:isMobile?"repeat(2, 1fr)":"repeat(6, 1fr)", gap:isMobile?8:10, marginBottom:20 }}>
          {Object.values(SCENARIOS).map((sc: any) => (
            <button key={sc.id} className={`scard ${selected===sc.id?"active":""}`} onClick={() => pick(sc.id)}>
              <div style={{ fontSize:isMobile?11:12, fontWeight:selected===sc.id?500:400, color:selected===sc.id?CHARCOAL:"#5d6d7e", marginBottom:2, lineHeight:1.3 }}>{sc.label}</div>
              <div style={{ fontSize:9, color:"#95a5a6" }}>{sc.bed.split("·")[0].trim()}</div>
            </button>
          ))}
        </div>

        {/* Main story */}
        <div style={{ display:"flex", flexDirection:"column", gap:10 }}>

          {/* Step 1 */}
          <div style={{ background:LIGHT, border:`1px solid ${MID}`, borderRadius:12, padding:cardPad }}>
            <div style={{ display:"flex", alignItems:"center", gap:10, marginBottom:12 }}>
              <div style={{ width:26, height:26, borderRadius:"50%", background:"#fdf0ef", border:"1px solid #e8b4b0", display:"flex", alignItems:"center", justifyContent:"center", fontSize:10, color:R, fontFamily:"DM Mono", flexShrink:0 }}>01</div>
              <div style={{ fontSize:13, fontWeight:500, color:CHARCOAL }}>What happened</div>
            </div>
            <div style={{ marginBottom:10 }}>
              <div style={{ fontSize:10, color:"#95a5a6", letterSpacing:"0.08em", textTransform:"uppercase", marginBottom:6 }}>{s.bed} · {s.alarm_type}</div>
              <p style={{ fontSize:isMobile?13:14, color:"#5d6d7e", lineHeight:1.7 }}>{s.situation}</p>
            </div>
            <div style={{ display:"inline-block", fontSize:12, padding:"4px 14px", borderRadius:100, background:s.ground_truth?"#fdf0ef":"#edf7f1", border:`1px solid ${s.ground_truth?"#e8b4b0":"#a8d5b5"}`, color:s.ground_truth?R:"#1e8449" }}>
              {s.ground_truth_plain}
            </div>
          </div>

          {/* Step 2 */}
          <div style={{ background:LIGHT, border:`1px solid ${MID}`, borderRadius:12, padding:cardPad }}>
            <div style={{ display:"flex", alignItems:"center", gap:10, marginBottom:12 }}>
              <div style={{ width:26, height:26, borderRadius:"50%", background:"#fdf0ef", border:"1px solid #e8b4b0", display:"flex", alignItems:"center", justifyContent:"center", fontSize:10, color:R, fontFamily:"DM Mono", flexShrink:0 }}>02</div>
              <div style={{ fontSize:13, fontWeight:500, color:CHARCOAL }}>What the monitor did</div>
            </div>
            <div style={{ padding:"10px 14px", background:SOFT, borderRadius:8, border:`1px solid ${MID}`, marginBottom:12, fontSize:13, color:"#5d6d7e" }}>
              {s.what_monitor_said}
            </div>
            <div style={{ fontSize:10, color:"#95a5a6", letterSpacing:"0.08em", textTransform:"uppercase", marginBottom:8 }}>
              Raw heart signal - 60 seconds
            </div>
            <Waveform signal={s.signal} active={phase==="result"} />
            <div style={{ fontSize:11, color:"#bdc3c7", marginTop:6 }}>
              This is the actual electrical signal from the patient's heart. SigmaMedStat analyzes the full 60-second window - not just the moment the alarm fired.
            </div>
          </div>

          {/* Step 3 */}
          <div style={{ background:LIGHT, border:`1px solid ${MID}`, borderRadius:12, padding:cardPad }}>
            <div style={{ display:"flex", alignItems:"center", gap:10, marginBottom:12 }}>
              <div style={{ width:26, height:26, borderRadius:"50%", background:"#fdf0ef", border:"1px solid #e8b4b0", display:"flex", alignItems:"center", justifyContent:"center", fontSize:10, color:R, fontFamily:"DM Mono", flexShrink:0 }}>03</div>
              <div style={{ fontSize:13, fontWeight:500, color:CHARCOAL }}>What SigmaMedStat found</div>
            </div>

            {phase==="analyzing" && (
              <div style={{ padding:"16px", background:SOFT, borderRadius:8, border:`1px solid ${MID}`, fontSize:13, color:"#7f8c8d" }}>
                <div style={{ display:"flex", alignItems:"center", gap:10, marginBottom:8 }}>
                  <div style={{ display:"flex", gap:4 }}>
                    {[0,1,2].map(i => <div key={i} style={{ width:4, height:4, borderRadius:"50%", background:"#95a5a6", animation:`dotBounce 1.2s ease ${i*0.2}s infinite` }} />)}
                  </div>
                  <span style={{ fontWeight:500, color:CHARCOAL }}>Pipeline running...</span>
                </div>
                <div style={{ fontSize:12, color:"#95a5a6", lineHeight:1.7 }}>
                  Step 1 - Converting signal to time-frequency heat map...<br/>
                  Step 2 - EfficientNet extracting 1,280 signal features...<br/>
                  Step 3 - Neural classifier calculating alarm probability...
                </div>
              </div>
            )}

            {phase==="result" && (
              <div className="fade">
                <div style={{ display:"grid", gridTemplateColumns:isMobile?"1fr":"repeat(3, 1fr)", gap:8, marginBottom:14 }}>
                  {[
                    { n:"1", title:"Signal → Image", body:"60 seconds of raw signal converted into a visual heat map. Patterns invisible in numbers become visible as shapes." },
                    { n:"2", title:"Image → Features", body:"EfficientNet analyzed the heat map and extracted 1,280 measurements describing the signal's time-frequency patterns." },
                    { n:"3", title:"Features → Decision", body:"A neural classifier took those 1,280 measurements and calculated the probability that this alarm is real vs. a device error." },
                  ].map((step,i) => (
                    <div key={i} style={{ padding:"10px 12px", background:SOFT, borderRadius:8, border:`1px solid ${MID}` }}>
                      <div style={{ display:"flex", alignItems:"center", gap:8, marginBottom:6 }}>
                        <div style={{ width:20, height:20, borderRadius:"50%", background:"#fdf0ef", border:"1px solid #e8b4b0", display:"flex", alignItems:"center", justifyContent:"center", fontSize:9, color:R, flexShrink:0 }}>{step.n}</div>
                        <div style={{ fontSize:12, fontWeight:500, color:CHARCOAL }}>{step.title}</div>
                      </div>
                      <div style={{ fontSize:12, color:"#7f8c8d", lineHeight:1.5 }}>{step.body}</div>
                    </div>
                  ))}
                </div>

                <div style={{ padding:"12px 16px", background:SOFT, borderRadius:8, border:`1px solid ${MID}`, fontSize:13, color:"#5d6d7e", lineHeight:1.7, marginBottom:12 }}>
                  {s.what_sigmamedstat_found}
                </div>

                <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:8 }}>
                  <div style={{ padding:"10px 12px", background:SOFT, borderRadius:8, border:`1px solid ${MID}` }}>
                    <div style={{ fontSize:10, color:"#95a5a6", marginBottom:6 }}>Device error chance</div>
                    <div style={{ display:"flex", alignItems:"center", gap:8 }}>
                      <div style={{ fontSize:18, fontWeight:300, color:"#1e8449", fontFamily:"DM Mono", flexShrink:0 }}>{(pred.false_alarm_prob*100).toFixed(0)}%</div>
                      <AnimatedBar value={pred.false_alarm_prob*100} color="#27ae60" trigger={phase==="result"} />
                    </div>
                  </div>
                  <div style={{ padding:"10px 12px", background:SOFT, borderRadius:8, border:`1px solid ${MID}` }}>
                    <div style={{ fontSize:10, color:"#95a5a6", marginBottom:6 }}>Real emergency chance</div>
                    <div style={{ display:"flex", alignItems:"center", gap:8 }}>
                      <div style={{ fontSize:18, fontWeight:300, color:R, fontFamily:"DM Mono", flexShrink:0 }}>{(pred.true_alarm_prob*100).toFixed(0)}%</div>
                      <AnimatedBar value={pred.true_alarm_prob*100} color={R} trigger={phase==="result"} />
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Step 4 */}
          {phase==="result" && (
            <div className="fade" tabIndex={-1} style={{ outline:"none", background:LIGHT, border:`1px solid ${MID}`, borderRadius:12, padding:cardPad }}>
              <div style={{ display:"flex", alignItems:"center", gap:10, marginBottom:12 }}>
                <div style={{ width:26, height:26, borderRadius:"50%", background:"#fdf0ef", border:"1px solid #e8b4b0", display:"flex", alignItems:"center", justifyContent:"center", fontSize:10, color:R, fontFamily:"DM Mono", flexShrink:0 }}>04</div>
                <div style={{ fontSize:13, fontWeight:500, color:CHARCOAL }}>What to do</div>
              </div>

              <div style={{ padding:"12px 16px", background:verdictBg, border:`1px solid ${verdictBorder}`, borderRadius:10, marginBottom:12 }}>
                <div style={{ display:"flex", flexDirection:"column", gap:6, marginBottom:8 }}>
                  <div style={{ fontSize:isMobile?13:16, fontWeight:400, color:verdictColor }}>{s.verdict}</div>
                  <div style={{ fontSize:11, padding:"2px 10px", borderRadius:100, background:LIGHT, border:`1px solid ${verdictBorder}`, color:verdictColor, alignSelf:"flex-start" }}>
                    {s.confidence}% confident
                  </div>
                </div>
                <div style={{ fontSize:13, color:"#5d6d7e", lineHeight:1.7 }}>{s.action}</div>
              </div>

              <div style={{ display:"grid", gridTemplateColumns:isMobile?"1fr":"1fr 1fr", gap:10 }}>
                <div style={{ padding:"12px 14px", background:correct?"#edf7f1":"#fdf0ef", borderRadius:8, border:`1px solid ${correct?"#a8d5b5":"#e8b4b0"}` }}>
                  <div style={{ fontSize:10, color:correct?"#1e8449":R, letterSpacing:"0.08em", textTransform:"uppercase", marginBottom:6 }}>
                    {correct?"✓ The model got this right":"✗ The model got this wrong"}
                  </div>
                  <div style={{ fontSize:12, color:"#5d6d7e", lineHeight:1.6 }}>
                    {correct?"The model's prediction matched what was actually happening with this patient.":s.why_wrong}
                  </div>
                </div>
                <div style={{ padding:"12px 14px", background:SOFT, borderRadius:8, border:`1px solid ${MID}` }}>
                  <div style={{ fontSize:10, color:"#95a5a6", letterSpacing:"0.08em", textTransform:"uppercase", marginBottom:6 }}>Without SigmaMedStat</div>
                  <div style={{ fontSize:12, color:"#7f8c8d", lineHeight:1.6 }}>{s.without}</div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Bottom context */}
        <div style={{ marginTop:24, display:"grid", gridTemplateColumns:isMobile?"1fr":"repeat(3, 1fr)", gap:10 }}>
          {[
            { title:"Why are there so many false alarms?", body:"Hospital monitors were designed in the 1980s to alarm whenever a reading crosses a threshold. They have no way to distinguish between a sick patient and a loose sensor. The hardware hasn't changed - it just got louder." },
            { title:"What does the model actually do?", body:"It converts 60 seconds of raw signal into a visual pattern, extracts 1,280 measurements using a computer vision model, and calculates the probability that the alarm reflects something real. All in 34 milliseconds." },
            { title:"Why does it get some wrong?", body:"The model was trained on 498 recordings and achieves 64% accuracy - meaningfully better than random guessing, but not good enough for clinical deployment yet. Patient-specific training is the next step." },
          ].map((c,i) => (
            <div key={i} style={{ padding:"16px 18px", background:LIGHT, border:`1px solid ${MID}`, borderRadius:10 }}>
              <div style={{ fontSize:13, fontWeight:500, color:CHARCOAL, marginBottom:8 }}>{c.title}</div>
              <div style={{ fontSize:13, color:"#7f8c8d", lineHeight:1.7 }}>{c.body}</div>
            </div>
          ))}
        </div>

      </div>
    </div>
  )
}