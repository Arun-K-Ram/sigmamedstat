import { useEffect, useRef } from "react"

export default function LoadingScreen({ onDone }: { onDone: () => void }) {
  const pathRef = useRef<SVGPathElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const path = pathRef.current
    if (!path) return
    const length = path.getTotalLength()
    path.style.strokeDasharray = `${length}`
    path.style.strokeDashoffset = `${length}`

    let start: number | null = null
    const drawDuration = 1600

    const draw = (ts: number) => {
      if (!start) start = ts
      const elapsed = ts - start
      const progress = Math.min(elapsed / drawDuration, 1)
      path.style.strokeDashoffset = `${length * (1 - progress)}`
      if (progress < 1) {
        requestAnimationFrame(draw)
      } else {
        setTimeout(() => {
          if (containerRef.current) {
            containerRef.current.style.opacity = "0"
            containerRef.current.style.transition = "opacity 0.5s ease"
          }
          setTimeout(onDone, 500)
        }, 400)
      }
    }

    requestAnimationFrame(draw)
  }, [onDone])

  return (
    <div
      ref={containerRef}
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 9999,
        background: "#f4f4f2",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        gap: 24,
      }}
    >
      <style>{`
        @keyframes heartPulse {
          0%,100% { transform: scale(1); }
          15%      { transform: scale(1.08); }
          30%      { transform: scale(0.97); }
          45%      { transform: scale(1.05); }
          60%      { transform: scale(1); }
        }
        @keyframes fadeInUp {
          from { opacity:0; transform:translateY(6px); }
          to   { opacity:1; transform:translateY(0); }
        }
        .heart-wrap {
          animation: heartPulse 1.4s ease infinite, fadeInUp 0.4s ease forwards;
          transform-origin: center;
        }
        .sigma-label {
          animation: fadeInUp 0.6s ease 0.3s forwards;
          opacity: 0;
        }
      `}</style>

      {/* Anatomical heart SVG */}
      {/* Anatomical heart image */}
      <div className="heart-wrap">
        <img
          src="/heart.png"
          alt="Anatomical heart"
          style={{
            width: 130,
            height: "auto",
            display: "block",
          }}
        />
      </div>

      {/* ECG line */}
      <svg
        width="280"
        height="40"
        viewBox="0 0 280 40"
        style={{ overflow:"visible", display:"block" }}
      >
        <path
          ref={pathRef}
          d="M0,20 L50,20 L62,20 L70,2 L78,38 L86,20 L106,20 L114,26 L122,20 L160,20 L168,2 L176,38 L184,20 L204,20 L212,26 L220,20 L280,20"
          fill="none"
          stroke="#c0392b"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>

      {/* Label */}
      <div className="sigma-label" style={{
        fontSize: 11,
        color: "#95a5a6",
        letterSpacing: "0.2em",
        textTransform: "uppercase",
        fontFamily: "DM Sans, Helvetica Neue, sans-serif",
      }}>
        SigmaMedStat
      </div>
    </div>
  )
}