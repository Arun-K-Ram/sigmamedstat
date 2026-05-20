import { useNavigate, useLocation } from "react-router-dom"

export default function Navbar() {
  const navigate = useNavigate()
  const location = useLocation()
  const isHome = location.pathname === "/"

  return (
    <nav style={{
      position: "fixed", top: 0, left: 0, right: 0, zIndex: 50,
      display: "flex", alignItems: "center", justifyContent: "space-between",
      padding: "0 40px", height: 60,
      background: "rgba(10,10,10,0.95)",
      borderBottom: "1px solid #1a1a1a",
      backdropFilter: "blur(8px)",
      fontFamily: "'DM Sans', 'Helvetica Neue', sans-serif"
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, cursor: "pointer" }} onClick={() => navigate("/")}>
        <div style={{ width: 26, height: 26, borderRadius: 6, background: "#161616", border: "1px solid #2a2a2a", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 13, color: "#666", fontFamily: "DM Mono, monospace" }}>Σ</div>
        <span style={{ fontSize: 15, color: "#888", fontWeight: 400 }}>SigmaMedStat</span>
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: 32 }}>
        <span onClick={() => navigate("/")} style={{ fontSize: 14, color: isHome ? "#888" : "#444", cursor: "pointer", transition: "color 0.15s" }}>Home</span>
        <span onClick={() => navigate("/demo")} style={{ fontSize: 14, color: !isHome ? "#888" : "#444", cursor: "pointer", transition: "color 0.15s" }}>Demo</span>
        <button onClick={() => navigate("/demo")} style={{ background: "#1a1a1a", border: "1px solid #2a2a2a", color: "#888", padding: "8px 20px", borderRadius: 6, fontSize: 13, cursor: "pointer", fontFamily: "inherit" }}>
          Try it →
        </button>
      </div>
    </nav>
  )
}