import { useNavigate } from "react-router-dom"
import { useState, useEffect, useRef } from "react"

const R = "#c0392b"
const CHARCOAL = "#2c3e50"
const SOFT = "#f4f4f2"
const MID = "#e8e8e5"
const LIGHT = "#ffffff"

function HospitalAnimation() {
  const [phase, setPhase] = useState<0|1|2|3>(0)
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const phaseRef = useRef<number>(0)

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
  const phaseLabel = ["Everything looks fine","⚠ Something triggered an alarm","Checking if this alarm is real...","Here's what's actually happening"][phase]
  const phaseLabelColor = isAlarm?R:isAnalyze?"#2d6a4f":isResult?"#1b4332":"#7f8c8d"

  return (
    <div>
      <div style={{ textAlign:"center", marginBottom:28 }}>
        <div style={{ display:"inline-flex", alignItems:"center", gap:8, padding:"6px 16px", borderRadius:100, background:isAlarm?"#fdf0ef":isAnalyze?"#edf7f1":isResult?"#edf7f1":SOFT, border:`1px solid ${isAlarm?"#e8b4b0":isAnalyze?"#a8d5b5":isResult?"#7ec8a0":MID}`, transition:"all 1s" }}>
          <div style={{ width:6, height:6, borderRadius:"50%", background:isAlarm?R:isAnalyze?"#27ae60":isResult?"#1e8449":"#95a5a6", transition:"background 1s" }} />
          <span style={{ fontSize:12, color:phaseLabelColor, transition:"color 1s" }}>{phaseLabel}</span>
        </div>
      </div>
      <div style={{ display:"flex", alignItems:"center", justifyContent:"center" }}>
        <div style={{ width:260, background:LIGHT, border:`1.5px solid ${isAlarm?"#e8b4b0":MID}`, borderRadius:12, padding:20, transition:"border-color 1s, box-shadow 1s", boxShadow:isAlarm?"0 0 30px rgba(192,57,43,0.08)":"0 1px 8px rgba(0,0,0,0.06)" }}>
          <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:14 }}>
            <div>
              <div style={{ fontSize:10, color:"#95a5a6", letterSpacing:"0.1em", textTransform:"uppercase" }}>ICU Monitor · Bed 4</div>
              <div style={{ fontSize:12, color:"#7f8c8d", marginTop:2 }}>Blood Oxygen · 2h 14m</div>
            </div>
            <div style={{ fontSize:9, padding:"3px 8px", borderRadius:4, letterSpacing:"0.08em", textTransform:"uppercase", background:isAlarm?"#fdf0ef":"#edf7f1", border:`1px solid ${isAlarm?"#e8b4b0":"#a8d5b5"}`, color:isAlarm?R:"#1e8449", transition:"all 1s" }}>
              {isAlarm ? "⚠ ALARM" : "● All good"}
            </div>
          </div>
          <div style={{ background:SOFT, borderRadius:6, padding:"10px 12px", marginBottom:14, height:72, overflow:"hidden", border:`1px solid ${MID}` }}>
            <svg width="200" height="52" viewBox="0 0 200 52" style={{ display:"block" }}>
              <path d={isAlarm?ecgChaos:ecgNormal} stroke={isAlarm?R:"#27ae60"} strokeWidth="1.5" fill="none" style={{ transition:"stroke 1s" }}/>
            </svg>
          </div>
          <div style={{ display:"flex", justifyContent:"space-between" }}>
            {[{label:"Blood Oxygen",val:isAlarm?"--":"98%",bad:isAlarm},{label:"Heart Rate",val:"72 bpm",bad:false},{label:"Breathing",val:"16/min",bad:false}].map((m,i) => (
              <div key={i} style={{ textAlign:"center" }}>
                <div style={{ fontSize:16, fontFamily:"DM Mono", color:m.bad?R:CHARCOAL, fontWeight:300, transition:"color 1s" }}>{m.val}</div>
                <div style={{ fontSize:9, color:"#95a5a6", marginTop:2 }}>{m.label}</div>
              </div>
            ))}
          </div>
          <div style={{ marginTop:16, fontSize:10, color:"#bdc3c7", textAlign:"center" }}>Standard hospital monitor</div>
        </div>

        <div style={{ display:"flex", flexDirection:"column", alignItems:"center", width:160, gap:12 }}>
          <div style={{ height:1, width:"100%", background:MID, position:"relative" }}>
            {isAnalyze && <div style={{ position:"absolute", top:-1, left:0, width:"28px", height:3, background:"#27ae60", borderRadius:2, animation:"scanRight 2.5s ease-in-out infinite" }} />}
          </div>
          <div style={{ background:isAnalyze||isResult?"#edf7f1":SOFT, border:`1px solid ${isAnalyze||isResult?"#a8d5b5":MID}`, borderRadius:8, padding:"12px 18px", textAlign:"center", transition:"all 1s", minWidth:120 }}>
            {phase===0 && <div style={{ fontSize:11, color:"#95a5a6" }}>Watching...</div>}
            {phase===1 && <div style={{ fontSize:11, color:R }}>Alarm detected</div>}
            {phase===2 && (
              <div>
                <div style={{ fontSize:10, color:"#27ae60", letterSpacing:"0.08em", marginBottom:6 }}>CHECKING</div>
                <div style={{ display:"flex", gap:4, justifyContent:"center" }}>
                  {[0,1,2].map(i => <div key={i} style={{ width:4, height:4, borderRadius:"50%", background:"#27ae60", animation:`dotBounce 1.4s ease ${i*0.25}s infinite` }} />)}
                </div>
              </div>
            )}
            {phase===3 && <div style={{ fontSize:11, color:"#1e8449" }}>✓ Done</div>}
          </div>
          <div style={{ height:1, width:"100%", background:MID }} />
          <div style={{ fontSize:9, color:"#95a5a6", letterSpacing:"0.08em", textTransform:"uppercase" }}>SigmaMedStat</div>
        </div>

        <div style={{ width:260, background:LIGHT, border:`1.5px solid ${isResult?"#a8d5b5":MID}`, borderRadius:12, padding:20, opacity:isResult?1:0.2, transform:isResult?"translateY(0)":"translateY(10px)", transition:"opacity 1.2s, transform 1.2s, border-color 1.2s", boxShadow:isResult?"0 1px 8px rgba(0,0,0,0.06)":"none" }}>
          <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:14 }}>
            <div>
              <div style={{ fontSize:10, color:"#95a5a6", letterSpacing:"0.1em", textTransform:"uppercase" }}>SigmaMedStat · Bed 4</div>
              <div style={{ fontSize:12, color:"#7f8c8d", marginTop:2 }}>Signal Check</div>
            </div>
            <div style={{ fontSize:9, padding:"3px 8px", borderRadius:4, background:"#edf7f1", border:"1px solid #a8d5b5", color:"#1e8449", letterSpacing:"0.08em", textTransform:"uppercase" }}>● Active</div>
          </div>
          <div style={{ background:"#fdf0ef", borderRadius:6, padding:14, marginBottom:14, border:"1px solid #e8b4b0" }}>
            <div style={{ display:"flex", alignItems:"center", gap:14 }}>
              <div style={{ textAlign:"center" }}>
                <div style={{ fontSize:38, fontWeight:300, color:R, fontFamily:"DM Mono", lineHeight:1 }}>24</div>
                <div style={{ fontSize:10, color:"#e8b4b0" }}>/ 100</div>
                <div style={{ fontSize:9, color:R, marginTop:4, letterSpacing:"0.08em" }}>DON'T TRUST IT</div>
              </div>
              <div>
                <div style={{ fontSize:12, color:"#b7770d", marginBottom:8 }}>⚠ This alarm is probably fake</div>
                <div style={{ fontSize:11, color:"#1e8449" }}>Sensor slipped off</div>
                <div style={{ fontSize:10, color:"#1e8449", marginTop:2 }}>90% sure</div>
              </div>
            </div>
          </div>
          <div style={{ display:"flex", flexDirection:"column", gap:7 }}>
            {["Blood oxygen reading froze — sensor lost contact","Patient moved 10 seconds ago","Heart rhythm is completely normal"].map((e,i) => (
              <div key={i} style={{ display:"flex", gap:8, fontSize:11, color:"#5d6d7e" }}>
                <span style={{ color:"#27ae60", flexShrink:0 }}>→</span><span>{e}</span>
              </div>
            ))}
          </div>
          <div style={{ marginTop:14, paddingTop:12, borderTop:"1px solid #edf7f1", fontSize:11, color:"#1e8449" }}>
            What to do: Reattach the sensor. Don't treat the patient.
          </div>
        </div>
      </div>
    </div>
  )
}

function IndustrySection() {
  const companies = [
    {
      name:"Traditional bedside monitors",
      what:"The monitors found in every ICU worldwide are engineering marvels — they measure blood oxygen, heart rhythm, and breathing with remarkable precision. The hardware is excellent.",
      gap:"They were designed to compare a reading against a fixed threshold. If blood oxygen drops below 90%, the alarm fires. Always. Even if the sensor just slipped off the finger.",
      tag:"Hardware"
    },
    {
      name:"Threshold-based alerting systems",
      what:"Some newer systems let hospitals customize alert thresholds — a doctor can set alarm levels specific to a patient's condition rather than using generic defaults.",
      gap:"Customized thresholds still don't know whether a reading is trustworthy. A low blood oxygen alarm fires whether the patient is deteriorating or whether someone bumped the sensor.",
      tag:"Alerting"
    },
    {
      name:"Single-channel monitoring devices",
      what:"Modern devices measure individual signals with extraordinary accuracy — each sensor is calibrated, validated, and reliable on its own.",
      gap:"They analyze each signal in isolation. No device currently asks: if blood oxygen looks bad but heart rhythm is perfectly fine, should I trust the blood oxygen reading?",
      tag:"Devices"
    },
    {
      name:"Remote alarm management platforms",
      what:"Cloud-connected platforms let clinical staff view alarms from anywhere — a nurse can silence or escalate an alert from a phone rather than walking to the bedside.",
      gap:"Remote silencing is still manual. A human still has to decide, alarm by alarm, whether to act. There's no system that evaluates signal quality before the decision reaches the nurse.",
      tag:"Software"
    },
  ]

  return (
    <section style={{ borderTop:`1px solid ${MID}`, background:LIGHT }}>
      <div style={{ maxWidth:960, margin:"0 auto", padding:"80px 40px" }}>
        <div style={{ marginBottom:48 }}>
          <p style={{ fontSize:11, color:"#95a5a6", letterSpacing:"0.1em", textTransform:"uppercase", marginBottom:12 }}>Why this gap still exists</p>
          <h2 style={{ fontSize:32, fontWeight:300, color:CHARCOAL, letterSpacing:"-0.5px", marginBottom:16 }}>
            The industry is great at measuring signals.<br/>Nobody checks if they can be trusted.
          </h2>
          <p style={{ fontSize:15, color:"#7f8c8d", maxWidth:640, lineHeight:1.8 }}>
            The companies that build hospital monitors have spent decades perfecting the hardware. The sensors are accurate. The displays are clear. The alarms are loud. What nobody built was a layer that asks — before the alarm fires — is this reading actually telling us something real?
          </p>
        </div>

        <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:20, marginBottom:40 }}>
          {companies.map((c,i) => (
            <div key={i} style={{ background:SOFT, border:`1px solid ${MID}`, borderRadius:12, padding:28 }}>
              <div style={{ display:"flex", alignItems:"center", gap:10, marginBottom:16 }}>
                <div style={{ fontSize:10, padding:"2px 10px", borderRadius:100, background:LIGHT, border:`1px solid ${MID}`, color:"#7f8c8d" }}>{c.tag}</div>
                <div style={{ fontSize:14, fontWeight:500, color:CHARCOAL }}>{c.name}</div>
              </div>
              <div style={{ marginBottom:16 }}>
                <div style={{ fontSize:11, color:"#95a5a6", letterSpacing:"0.08em", textTransform:"uppercase", marginBottom:6 }}>What works well</div>
                <div style={{ fontSize:13, color:"#5d6d7e", lineHeight:1.7 }}>{c.what}</div>
              </div>
              <div style={{ paddingTop:16, borderTop:`1px solid ${MID}` }}>
                <div style={{ fontSize:11, color:R, letterSpacing:"0.08em", textTransform:"uppercase", marginBottom:6 }}>The gap</div>
                <div style={{ fontSize:13, color:"#7f8c8d", lineHeight:1.7 }}>{c.gap}</div>
              </div>
            </div>
          ))}
        </div>

        <div style={{ padding:"28px 32px", background:SOFT, borderRadius:12, border:`1px solid ${MID}` }}>
          <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:40, alignItems:"start" }}>
            <div>
              <div style={{ fontSize:15, fontWeight:400, color:CHARCOAL, marginBottom:12 }}>What SigmaMedStat adds — without replacing anything</div>
              <div style={{ fontSize:13, color:"#7f8c8d", lineHeight:1.8 }}>
                SigmaMedStat doesn't replace the monitor. It sits alongside it — reading the same signals, but asking a different question. Not "is this reading abnormal?" but "should anyone act on this reading at all?" That question has never had a systematic answer. Until now.
              </div>
            </div>
            <div style={{ display:"flex", flexDirection:"column", gap:12 }}>
              {[
                {them:"Alarm fires when reading crosses a threshold", us:"Signal is evaluated before the alarm reaches the nurse"},
                {them:"350 alarms per patient per day", us:"Each alarm comes with a confidence score and explanation"},
                {them:"Nurse decides on every single alarm", us:"Model pre-screens: act now, or stand down"},
                {them:"No explanation — just a beeping sound", us:"Plain English: what happened and what to do"},
              ].map((r,i) => (
                <div key={i} style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:12, paddingBottom:12, borderBottom:i<3?`1px solid ${MID}`:"none" }}>
                  <div style={{ fontSize:12, color:"#bdc3c7", textDecoration:"line-through", lineHeight:1.6 }}>{r.them}</div>
                  <div style={{ fontSize:12, color:CHARCOAL, lineHeight:1.6 }}>→ {r.us}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

function MLSection() {
  const experiments = [
    {
      num:"01", best:true,
      title:"Teaching a computer to see alarm patterns",
      subtitle:"We turned raw hospital signals into images and ran them through AI",
      detail:"We converted 60 seconds of heart monitor data into visual heat maps using a technique called Continuous Wavelet Transform. Then we fed those images into three different AI models originally trained to recognize everyday photos. The question: can an AI learn to spot a fake alarm the same way it learned to recognize a dog?",
      models:["ResNet18","ResNet50","EfficientNet"],
      results:[
        {name:"EfficientNet + Neural Classifier", auc:0.641, best:true},
        {name:"EfficientNet + Logistic Regression", auc:0.587, best:false},
        {name:"ResNet18 + SVM", auc:0.542, best:false},
      ],
      finding:"EfficientNet won. It correctly identified real vs fake alarms 64% of the time — better than any other approach we tried. The visual heat map approach captures signal patterns you simply can't see by looking at raw numbers.",
      findingColor:"#1b4332", findingBg:"#edf7f1", findingBorder:"#a8d5b5",
    },
    {
      num:"02", best:false,
      title:"What if we measure the signals the old-fashioned way?",
      subtitle:"We extracted 103 clinical measurements and let the algorithm decide",
      detail:"Instead of images, we measured everything we could about each signal directly — how noisy it is, its dominant frequency, how correlated the channels are, how much it varies over time. We extracted 103 measurements per recording, then ran a full hyperparameter sweep to find the best model settings.",
      models:["XGBoost","Random Forest","Gradient Boosting","SVM"],
      results:[
        {name:"SVM (best settings)", auc:0.539, best:true},
        {name:"XGBoost (tuned)", auc:0.517, best:false},
        {name:"Gradient Boosting", auc:0.465, best:false},
      ],
      finding:"This approach underperformed the image-based one. The measurements we could define by hand weren't as informative as the visual patterns the AI discovered on its own. The signal contains information that humans haven't yet figured out how to describe.",
      findingColor:"#7d4e00", findingBg:"#fdf3e3", findingBorder:"#f0c87a",
    },
    {
      num:"03", best:false,
      title:"Different alarms need different models",
      subtitle:"We trained a separate model for each type of heart alarm",
      detail:"Hospitals see four main alarm types — irregular heartbeat, stopped heart, too fast, too slow. Each looks completely different on a monitor. We built a beat detector to find individual heartbeats, measured their shape and timing, then trained a separate model for each alarm type.",
      models:["XGBoost per alarm type","Pan-Tompkins Beat Detector"],
      results:[
        {name:"Rapid heartbeat model", auc:0.612, best:true},
        {name:"Irregular heartbeat model", auc:0.528, best:false},
        {name:"Stopped heart model", auc:0.478, best:false},
      ],
      finding:"Rapid heartbeat alarms were easiest to classify correctly (61%). This makes clinical sense — that alarm type has a distinct, measurable pattern. One-size-fits-all models are the wrong approach for this problem.",
      findingColor:"#1a3a6b", findingBg:"#eaf0fb", findingBorder:"#9db8e8",
    },
  ]

  return (
    <section style={{ borderTop:`1px solid ${MID}`, background:SOFT }}>
      <div style={{ maxWidth:960, margin:"0 auto", padding:"80px 40px" }}>
        <div style={{ marginBottom:56 }}>
          <p style={{ fontSize:11, color:"#95a5a6", letterSpacing:"0.1em", textTransform:"uppercase", marginBottom:12 }}>The research behind it</p>
          <h2 style={{ fontSize:32, fontWeight:300, color:CHARCOAL, letterSpacing:"-0.5px", marginBottom:16 }}>
            We ran three experiments.<br/>Here's what we learned.
          </h2>
          <p style={{ fontSize:15, color:"#7f8c8d", maxWidth:600, lineHeight:1.8 }}>
            The first version of SigmaMedStat used hand-coded rules — if the signal flatlines, reduce the trust score. That works, but it's just logic we wrote ourselves. These experiments tested whether a machine learning model trained on 750 real hospital alarm recordings could do better.
          </p>
          <div style={{ display:"flex", gap:16, marginTop:24, flexWrap:"wrap" }}>
            {[
              {label:"Real hospital alarm recordings", value:"750"},
              {label:"Best accuracy achieved", value:"64%"},
              {label:"Signal measurements tested", value:"103+"},
              {label:"AI models compared", value:"15+"},
            ].map((s,i) => (
              <div key={i} style={{ padding:"16px 24px", background:LIGHT, border:`1px solid ${MID}`, borderRadius:8 }}>
                <div style={{ fontSize:24, fontWeight:300, color:R, fontFamily:"DM Mono" }}>{s.value}</div>
                <div style={{ fontSize:12, color:"#95a5a6", marginTop:4 }}>{s.label}</div>
              </div>
            ))}
          </div>
        </div>

        <img src="/model_comparison.png" alt="Model comparison" style={{ width:"100%", borderRadius:12, marginBottom:32, border:`1px solid ${MID}` }} />

        <div style={{ display:"flex", flexDirection:"column", gap:24 }}>
          {experiments.map((exp,idx) => (
            <div key={idx} style={{ background:LIGHT, border:`1px solid ${MID}`, borderRadius:12, padding:32 }}>
              <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:40 }}>
                <div>
                  <div style={{ display:"flex", alignItems:"center", gap:12, marginBottom:16 }}>
                    <div style={{ width:32, height:32, borderRadius:"50%", background:"#fdf0ef", border:"1px solid #e8b4b0", display:"flex", alignItems:"center", justifyContent:"center", fontSize:12, color:R, fontFamily:"DM Mono", flexShrink:0 }}>{exp.num}</div>
                    <div>
                      <div style={{ fontSize:15, fontWeight:400, color:CHARCOAL }}>{exp.title}</div>
                      <div style={{ fontSize:12, color:"#95a5a6", marginTop:2 }}>{exp.subtitle}</div>
                    </div>
                  </div>
                  <p style={{ fontSize:13, color:"#7f8c8d", lineHeight:1.8, marginBottom:16 }}>{exp.detail}</p>
                  <div style={{ display:"flex", flexWrap:"wrap", gap:6, marginBottom:16 }}>
                    {exp.models.map((m,i) => (
                      <span key={i} style={{ fontSize:11, padding:"3px 10px", borderRadius:100, background:SOFT, border:`1px solid ${MID}`, color:"#7f8c8d" }}>{m}</span>
                    ))}
                  </div>
                  <div style={{ padding:"12px 16px", borderRadius:8, background:exp.findingBg, border:`1px solid ${exp.findingBorder}` }}>
                    <div style={{ fontSize:10, color:exp.findingColor, letterSpacing:"0.08em", textTransform:"uppercase", marginBottom:6 }}>What we found</div>
                    <div style={{ fontSize:12, color:exp.findingColor, lineHeight:1.7 }}>{exp.finding}</div>
                  </div>
                </div>
                <div>
                  <div style={{ fontSize:11, color:"#95a5a6", letterSpacing:"0.1em", textTransform:"uppercase", marginBottom:16 }}>How accurate was each approach?</div>
                  <div style={{ display:"flex", flexDirection:"column", gap:12, marginBottom:20 }}>
                    {exp.results.map((r,i) => (
                      <div key={i}>
                        <div style={{ display:"flex", justifyContent:"space-between", marginBottom:4 }}>
                          <span style={{ fontSize:12, color:r.best?CHARCOAL:"#7f8c8d", fontWeight:r.best?500:400 }}>{r.name}</span>
                          <span style={{ fontSize:12, fontFamily:"DM Mono", color:r.best?R:"#95a5a6" }}>{(r.auc*100).toFixed(0)}% accurate</span>
                        </div>
                        <div style={{ height:6, background:SOFT, borderRadius:3, overflow:"hidden" }}>
                          <div style={{ height:"100%", width:`${r.auc*100}%`, background:r.best?R:"#bdc3c7", borderRadius:3 }} />
                        </div>
                      </div>
                    ))}
                  </div>
                  <div style={{ padding:"10px 14px", background:SOFT, borderRadius:6, border:`1px solid ${MID}` }}>
                    <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center" }}>
                      <span style={{ fontSize:11, color:"#95a5a6" }}>Flipping a coin (random guessing)</span>
                      <span style={{ fontSize:11, fontFamily:"DM Mono", color:"#95a5a6" }}>50% accurate</span>
                    </div>
                    <div style={{ height:4, background:MID, borderRadius:2, marginTop:6, overflow:"hidden" }}>
                      <div style={{ height:"100%", width:"50%", background:"#bdc3c7", borderRadius:2 }} />
                    </div>
                  </div>
                  {exp.best && (
                    <div style={{ marginTop:12, padding:"8px 12px", background:"#fdf0ef", borderRadius:6, border:"1px solid #e8b4b0", fontSize:11, color:R }}>
                      ★ Best result across all three experiments
                    </div>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Hyperparameter tuning section */}
        <div style={{ marginTop:40, background:LIGHT, border:`1px solid ${MID}`, borderRadius:12, padding:32 }}>
          <div style={{ marginBottom:24 }}>
            <p style={{ fontSize:11, color:"#95a5a6", letterSpacing:"0.1em", textTransform:"uppercase", marginBottom:12 }}>How we tuned the model</p>
            <h3 style={{ fontSize:22, fontWeight:300, color:CHARCOAL, letterSpacing:"-0.3px", marginBottom:12 }}>
              A different approach to hyperparameter tuning
            </h3>
            <p style={{ fontSize:14, color:"#7f8c8d", lineHeight:1.8, maxWidth:700 }}>
              Most people tune hyperparameters by instinct or trial and error — try a value, see if it improves, repeat. The problem with that is you're only ever testing one thing at a time and you have no idea how parameters interact with each other. Since Florida Tech, I've used a different method: define an explicit sweep grid, vary one parameter at a time while holding the others fixed at their best known value, and log every result. It takes longer but it gives you a real picture of what each decision actually costs you.
            </p>
          </div>

          <div style={{ display:"grid", gridTemplateColumns:"repeat(3, 1fr)", gap:20, marginBottom:28 }}>
            {[
              {
                param:"Dropout rate",
                values:"0.2 → 0.3 → 0.4 → 0.5",
                winner:"0.5",
                why:"With only 498 training samples, overfitting was our biggest risk. Dropout randomly switches off neurons during training, forcing the model not to rely on any single signal path. We tested four levels. Higher dropout won — the dataset was too small to afford anything less aggressive.",
              },
              {
                param:"Hidden layer size",
                values:"64 → 128 → 256 → 512",
                winner:"256",
                why:"The neural classifier sits on top of 1,280 EfficientNet features. Too small a hidden layer can't learn the patterns. Too large and it memorizes the training data instead of generalizing. 256 neurons hit the sweet spot — big enough to be expressive, small enough to stay honest.",
              },
              {
                param:"Learning rate",
                values:"0.01 → 0.001 → 0.0001 → 0.00001",
                winner:"0.0001",
                why:"Learning rate controls how aggressively the model updates its weights each step. Too high and it overshoots. Too low and it barely moves. The sweep confirmed 1e-4 — aggressive enough to learn, careful enough not to corrupt the pretrained EfficientNet features underneath.",
              },
            ].map((p,i) => (
              <div key={i} style={{ background:SOFT, borderRadius:10, padding:20, border:`1px solid ${MID}` }}>
                <div style={{ fontSize:12, fontWeight:500, color:CHARCOAL, marginBottom:8 }}>{p.param}</div>
                <div style={{ display:"flex", gap:4, marginBottom:12, flexWrap:"wrap" }}>
                  {p.values.split(" → ").map((v,j) => (
                    <span key={j} style={{ fontSize:11, padding:"2px 8px", borderRadius:4, background: v===p.winner?"#fdf0ef":LIGHT, border:`1px solid ${v===p.winner?"#e8b4b0":MID}`, color: v===p.winner?R:"#95a5a6", fontFamily:"DM Mono" }}>{v}</span>
                  ))}
                </div>
                <div style={{ fontSize:11, color:"#27ae60", marginBottom:8 }}>Winner: <span style={{ fontFamily:"DM Mono" }}>{p.winner}</span></div>
                <div style={{ fontSize:12, color:"#7f8c8d", lineHeight:1.7 }}>{p.why}</div>
              </div>
            ))}
          </div>

          <div style={{ padding:"20px 24px", background:SOFT, borderRadius:10, border:`1px solid ${MID}` }}>
            <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:32, alignItems:"center" }}>
              <div>
                <div style={{ fontSize:13, fontWeight:500, color:CHARCOAL, marginBottom:8 }}>What makes this different from most approaches</div>
                <div style={{ fontSize:13, color:"#7f8c8d", lineHeight:1.8 }}>
                  The standard approach in most tutorials is random search or manual guessing. What I use is a structured one-parameter-at-a-time sweep with all results logged — so I can look back and explain exactly why each decision was made. I've applied this method across every ML project since Florida Tech. It produces defensible results, not lucky ones.
                </div>
              </div>
              <div style={{ display:"flex", flexDirection:"column", gap:10 }}>
                <div style={{ padding:"10px 14px", background:LIGHT, borderRadius:8, border:`1px solid ${MID}` }}>
                  <div style={{ fontSize:11, color:"#95a5a6", marginBottom:4 }}>Sweep ran across</div>
                  <div style={{ fontSize:13, color:CHARCOAL }}>3 extractors × 3 parameters × 4 values = <span style={{ fontFamily:"DM Mono", color:R }}>36 training runs</span></div>
                </div>
                <div style={{ padding:"10px 14px", background:LIGHT, borderRadius:8, border:`1px solid ${MID}` }}>
                  <div style={{ fontSize:11, color:"#95a5a6", marginBottom:4 }}>Best configuration found</div>
                  <div style={{ fontSize:13, color:CHARCOAL, fontFamily:"DM Mono" }}>EfficientNet · dropout=0.5 · hidden=256 · lr=1e-4</div>
                </div>
                <div style={{ padding:"10px 14px", background:LIGHT, borderRadius:8, border:`1px solid ${MID}` }}>
                  <div style={{ fontSize:11, color:"#95a5a6", marginBottom:4 }}>Validated using</div>
                  <div style={{ fontSize:13, color:CHARCOAL }}>5-fold cross-validation · held-out test set · reproducible seed</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div style={{ marginTop:24, padding:"28px 32px", background:CHARCOAL, borderRadius:12 }}>
          <div style={{ fontSize:11, color:"#7f8c8d", letterSpacing:"0.1em", textTransform:"uppercase", marginBottom:12 }}>The bottom line</div>
          <p style={{ fontSize:15, color:"#ecf0f1", lineHeight:1.8 }}>
            The image-based approach beat everything else — <span style={{ color:R, fontFamily:"DM Mono" }}>64%</span> accuracy on real hospital alarms the model had never seen before. That's not good enough for clinical deployment yet, but it's a meaningful result on a genuinely hard problem. The next step is training on each patient's individual baseline — a model that learns what "normal" looks like for you specifically, not just for people in general.
          </p>
        </div>
      </div>
    </section>
  )
}

export default function HomePage() {
  const navigate = useNavigate()

  return (
    <div style={{ minHeight:"100vh", fontFamily:"'DM Sans','Helvetica Neue',sans-serif", background:SOFT }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500&family=DM+Mono:wght@400;500&display=swap');
        *{box-sizing:border-box;margin:0;padding:0;}
        @keyframes scanRight{0%{left:0;opacity:1}75%{left:calc(100% - 28px);opacity:0.3}100%{left:calc(100% - 28px);opacity:0}}
        @keyframes dotBounce{0%,100%{transform:translateY(0);opacity:0.3}50%{transform:translateY(-5px);opacity:1}}
        @keyframes fadeUp{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:translateY(0)}}
        @keyframes pulseDot{0%,100%{transform:scale(1)}50%{transform:scale(1.3)}}
        .fi{animation:fadeUp 0.8s ease forwards;opacity:0;}
        .d1{animation-delay:0.1s}.d2{animation-delay:0.3s}.d3{animation-delay:0.5s}.d4{animation-delay:0.7s}
        .btn{background:${CHARCOAL};color:#ffffff;border:none;padding:14px 34px;border-radius:6px;font-size:15px;font-weight:400;cursor:pointer;font-family:inherit;transition:background 0.2s;}
        .btn:hover{background:#1a252f;}
        .card{background:${LIGHT};border:1px solid ${MID};border-radius:10px;padding:28px;}
        .step{display:flex;gap:16px;padding:24px 0;border-bottom:1px solid ${MID};}
        .step:last-child{border-bottom:none;}
        .stepnum{width:28px;height:28px;border-radius:50%;border:1px solid ${MID};display:flex;align-items:center;justify-content:center;font-size:11px;color:#95a5a6;flex-shrink:0;font-family:DM Mono,monospace;margin-top:2px;}
        .footlink{font-size:13px;color:#7f8c8d;text-decoration:none;display:inline-flex;align-items:center;gap:6px;transition:color 0.15s;}
        .footlink:hover{color:${CHARCOAL};}
      `}</style>

      {/* Navbar */}
      <div style={{ position:"fixed", top:0, left:0, right:0, zIndex:100, background:"rgba(244,244,242,0.95)", borderBottom:`1px solid ${MID}`, backdropFilter:"blur(8px)" }}>
        <div style={{ maxWidth:960, margin:"0 auto", padding:"0 40px", height:64, display:"flex", alignItems:"center" }}>
          <div style={{ cursor:"pointer" }} onClick={() => navigate("/")}>
            <img src="/logo.png" alt="SigmaMedStat" style={{ height:56, width:"auto" }} />
          </div>
        </div>
      </div>

      <div style={{ position:"relative" }}>

        {/* Hero */}
        <section style={{ maxWidth:960, margin:"0 auto", padding:"140px 40px 60px", textAlign:"center" }}>
          <div className="fi d1" style={{ display:"inline-flex", alignItems:"center", gap:8, background:"#fdf0ef", border:"1px solid #e8b4b0", borderRadius:100, padding:"5px 14px", marginBottom:36 }}>
            <div style={{ width:6, height:6, borderRadius:"50%", background:R, animation:"pulseDot 2s ease infinite" }} />
            <span style={{ fontSize:11, color:R, letterSpacing:"0.1em", textTransform:"uppercase" }}>Built for ICU nurses and the patients they protect</span>
          </div>
          <h1 className="fi d2" style={{ fontSize:"clamp(34px,5.5vw,58px)", fontWeight:300, color:CHARCOAL, lineHeight:1.15, letterSpacing:"-2px", marginBottom:22 }}>
            Hospital monitors cry wolf<br/>
            <span style={{ color:R }}>hundreds of times a day.</span><br/>
            Most of it is noise.
          </h1>
          <p className="fi d3" style={{ fontSize:17, color:"#7f8c8d", maxWidth:520, margin:"0 auto 44px", lineHeight:1.8 }}>
            SigmaMedStat reads the same signals your hospital monitor does — and tells you, before the alarm even reaches the nurse, whether it's worth acting on.
          </p>
          <div className="fi d4" style={{ display:"flex", gap:12, justifyContent:"center", flexWrap:"wrap" }}>
          </div>
        </section>

        {/* Alert strip */}
        <section style={{ background:CHARCOAL, padding:"24px 40px", textAlign:"center" }}>
          <p style={{ fontSize:14, color:"#95a5a6", maxWidth:760, margin:"0 auto", lineHeight:1.8 }}>
            The Emergency Care Research Institute has listed alarm hazards as the <span style={{ color:"#ecf0f1" }}>#1 health technology danger</span> every single year for over a decade. Not because hospitals aren't trying — but because the monitors themselves have never gotten smarter.
          </p>
        </section>

        {/* Animation */}
        <section style={{ maxWidth:960, margin:"0 auto", padding:"60px 40px 80px" }}>
          <div style={{ fontSize:11, color:"#bdc3c7", letterSpacing:"0.1em", textTransform:"uppercase", textAlign:"center", marginBottom:32 }}>Here's what it looks like in practice</div>
          <HospitalAnimation />
          <div style={{ textAlign:"center", marginTop:20, fontSize:12, color:"#bdc3c7" }}>
            Simulated scenario — blood oxygen sensor loses contact after patient repositions
          </div>
        </section>

        {/* Stats */}
        <section style={{ borderTop:`1px solid ${MID}`, borderBottom:`1px solid ${MID}`, background:LIGHT }}>
          <div style={{ maxWidth:960, margin:"0 auto", padding:"0 40px", display:"flex" }}>
            {[
              {num:"99%",  label:"of ICU alarms are ignored — most are false positives"},
              {num:"350+", label:"alarms fired at a single patient every single day"},
              {num:"34ms", label:"how fast SigmaMedStat evaluates each alarm"},
              {num:"64%",  label:"accuracy on 750 real hospital alarm recordings"},
            ].map((s,i) => (
              <div key={i} style={{ flex:1, padding:"36px 0", ...(i>0?{paddingLeft:40, borderLeft:`1px solid ${MID}`}:{}), paddingRight:40 }}>
                <div style={{ fontSize:36, fontWeight:300, color:R, fontFamily:"DM Mono, monospace", letterSpacing:"-1px" }}>{s.num}</div>
                <div style={{ fontSize:13, color:"#95a5a6", marginTop:6 }}>{s.label}</div>
              </div>
            ))}
          </div>
        </section>

        {/* Problem */}
        <section style={{ maxWidth:960, margin:"0 auto", padding:"80px 40px" }}>
          <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:60, alignItems:"start" }}>
            <div>
              <p style={{ fontSize:11, color:"#95a5a6", letterSpacing:"0.1em", textTransform:"uppercase", marginBottom:20 }}>What's actually happening in hospitals</p>
              <h2 style={{ fontSize:30, fontWeight:300, color:CHARCOAL, lineHeight:1.3, letterSpacing:"-0.5px", marginBottom:20 }}>
                Nurses have learned to ignore alarms. That's the crisis.
              </h2>
              <p style={{ fontSize:14, color:"#7f8c8d", lineHeight:1.8, marginBottom:16 }}>
                It's not negligence. It's survival. When 99% of alarms mean nothing, your brain stops treating them as emergencies. That's called alarm fatigue — and it's the number one patient safety hazard identified by the Emergency Care Research Institute, year after year.
              </p>
              <p style={{ fontSize:14, color:"#7f8c8d", lineHeight:1.8 }}>
                The monitors aren't broken. They're doing exactly what they were designed to do — compare a reading to a fixed number and scream if it crosses the line. Nobody ever taught them to ask whether the reading itself should be trusted.
              </p>
            </div>
            <div className="card">
              <p style={{ fontSize:11, color:"#95a5a6", letterSpacing:"0.1em", textTransform:"uppercase", marginBottom:16 }}>The question no monitor currently asks</p>
              {[
                {before:"Is this reading outside the normal range?", after:"Should anyone trust this reading at all?"},
                {before:"Fire the alarm and let the nurse decide", after:"Check the signal first, then decide whether to alarm"},
                {before:"Nurse is on her 347th alarm of the shift", after:"Only the alarms worth acting on reach the nurse"},
                {before:"No explanation — just a beeping sound", after:"Plain English: what happened and what to do"},
              ].map((r,i) => (
                <div key={i} style={{ paddingBottom:14, marginBottom:14, borderBottom:i<3?`1px solid ${SOFT}`:"none" }}>
                  <div style={{ fontSize:12, color:"#bdc3c7", marginBottom:4, textDecoration:"line-through" }}>{r.before}</div>
                  <div style={{ fontSize:12, color:CHARCOAL }}>→ {r.after}</div>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* How it works */}
        <section style={{ borderTop:`1px solid ${MID}`, background:LIGHT }}>
          <div style={{ maxWidth:960, margin:"0 auto", padding:"80px 40px" }}>
            <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:60 }}>
              <div>
                <p style={{ fontSize:11, color:"#95a5a6", letterSpacing:"0.1em", textTransform:"uppercase", marginBottom:24 }}>How it works — in plain English</p>
                <div>
                  {[
                    {n:"01", title:"It reads the raw signal", body:"SigmaMedStat looks at the actual electrical data coming from the sensor — not just the final reading the monitor displays."},
                    {n:"02", title:"It checks all signals together", body:"If blood oxygen looks bad but heart rhythm is fine and the patient just moved, that tells a very different story than everything crashing at once."},
                    {n:"03", title:"It gives a confidence score", body:"Instead of just alarming, it says: 'We're 90% sure this is a sensor slipping off, not a real drop in oxygen.'"},
                    {n:"04", title:"It tells you what to do", body:"Not medical advice — just what the signal shows. Reattach the sensor. Don't treat the patient. Check back in 30 seconds."},
                  ].map(s => (
                    <div key={s.n} className="step">
                      <div className="stepnum">{s.n}</div>
                      <div>
                        <h3 style={{ fontSize:14, fontWeight:400, color:"#5d6d7e", marginBottom:6 }}>{s.title}</h3>
                        <p style={{ fontSize:13, color:"#7f8c8d", lineHeight:1.7 }}>{s.body}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
              <div>
                <p style={{ fontSize:11, color:"#95a5a6", letterSpacing:"0.1em", textTransform:"uppercase", marginBottom:21 }}>What it detects</p>
                {[
                  {label:"Sensor fell off or slipped", sub:"The most common cause of false alarms — motion, sweat, repositioning"},
                  {label:"Electrical interference", sub:"Nearby equipment can corrupt a reading without touching the patient"},
                  {label:"Signal dropped out completely", sub:"Total loss of data — often a cable issue, not a patient issue"},
                  {label:"Gradual signal drift", sub:"Long sessions cause sensors to drift — what looked normal at 8am may alarm by 4pm"},
                  {label:"Genuine emergency", sub:"When multiple signals all degrade together, that's usually real"},
                  {label:"Regulatory awareness", sub:"Built with IEC 62304 and FDA AI/ML guidance in mind"},
                ].map((f,i) => (
                  <div key={i} style={{ padding:"14px 18px", background:SOFT, border:`1px solid ${MID}`, borderRadius:8, marginBottom:4 }}>
                    <div style={{ fontSize:13, color:CHARCOAL, marginBottom:2 }}>{f.label}</div>
                    <div style={{ fontSize:11, color:"#95a5a6" }}>{f.sub}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>

        {/* Industry */}
        <IndustrySection />

        {/* ML Research */}
        <div id="ml-section">
          <MLSection />
        </div>

        {/* CTA */}
        <section style={{ maxWidth:960, margin:"0 auto", padding:"60px 40px 80px" }}>
          <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between", padding:"44px 52px", border:`1px solid ${MID}`, borderRadius:12, background:LIGHT }}>
            <div>
              <h2 style={{ fontSize:24, fontWeight:300, color:CHARCOAL, letterSpacing:"-0.5px", marginBottom:8 }}>Try it on real hospital data</h2>
              <p style={{ fontSize:14, color:"#95a5a6" }}>Six real ICU alarm events. Real model predictions. See where it gets it right — and where it doesn't.</p>
            </div>
            <button className="btn" onClick={() => navigate("/demo")} style={{ flexShrink:0, marginLeft:40 }}>Open the demo</button>
          </div>
        </section>

        {/* Footer */}
        <footer style={{ borderTop:`1px solid ${MID}`, background:LIGHT }}>
          <div style={{ maxWidth:960, margin:"0 auto", padding:"40px 40px", display:"grid", gridTemplateColumns:"1fr 1fr 1fr", gap:40, alignItems:"start" }}>
            <div>
              <img src="/logo.png" alt="SigmaMedStat" style={{ height:32, width:"auto", marginBottom:12 }} />
              <div style={{ fontSize:13, color:"#7f8c8d", lineHeight:1.7 }}>
                A signal intelligence platform for ICU alarm management. Research project — not FDA cleared.
              </div>
              <div style={{ fontSize:11, color:"#bdc3c7", marginTop:8 }}>IEC 62304 aware · ISO 14971</div>
            </div>
            <div>
              <div style={{ fontSize:11, color:"#95a5a6", letterSpacing:"0.1em", textTransform:"uppercase", marginBottom:16 }}>Built by</div>
              <div style={{ fontSize:14, color:CHARCOAL, marginBottom:4 }}>Arunkumar Ramachandran</div>
              <div style={{ fontSize:13, color:"#7f8c8d", marginBottom:16 }}>Florida Institute of Technology</div>
              <div style={{ display:"flex", flexDirection:"column", gap:8 }}>
                <a href="https://www.linkedin.com/in/arun-ramachandran-a2019a/" target="_blank" rel="noopener noreferrer" className="footlink">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.79-1.75-1.764s.784-1.764 1.75-1.764 1.75.79 1.75 1.764-.783 1.764-1.75 1.764zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z"/></svg>
                  LinkedIn
                </a>
                <a href="https://github.com/Arun-K-Ram" target="_blank" rel="noopener noreferrer" className="footlink">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/></svg>
                  GitHub
                </a>
              </div>
            </div>
            <div>
              <div style={{ fontSize:11, color:"#95a5a6", letterSpacing:"0.1em", textTransform:"uppercase", marginBottom:16 }}>Data & methods</div>
              <div style={{ display:"flex", flexDirection:"column", gap:8 }}>
                {[
                  "PhysioNet Challenge 2015 dataset",
                  "750 labeled ICU alarm recordings",
                  "EfficientNet-B0 feature extraction",
                  "Continuous Wavelet Transform",
                  "5-fold cross-validation",
                  "36-run hyperparameter sweep",
                ].map((t,i) => (
                  <div key={i} style={{ fontSize:12, color:"#7f8c8d", display:"flex", alignItems:"center", gap:8 }}>
                    <div style={{ width:3, height:3, borderRadius:"50%", background:"#bdc3c7", flexShrink:0 }} />
                    {t}
                  </div>
                ))}
              </div>
            </div>
          </div>
          <div style={{ borderTop:`1px solid ${MID}`, maxWidth:960, margin:"0 auto", padding:"16px 40px", display:"flex", justifyContent:"space-between", alignItems:"center" }}>
            <div style={{ fontSize:12, color:"#bdc3c7" }}>© 2026 Arunkumar Ramachandran · SigmaMedStat</div>
            <div style={{ fontSize:12, color:"#bdc3c7" }}>V1 deployed · ML validation complete · Seeking hospital partnerships</div>
          </div>
        </footer>

      </div>
    </div>
  )
}