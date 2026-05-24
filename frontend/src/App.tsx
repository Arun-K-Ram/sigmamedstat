import { BrowserRouter, Routes, Route, useLocation } from "react-router-dom"
import { useState, useEffect, useCallback } from "react"
import HomePage from "./pages/HomePage"
import DemoPage from "./pages/DemoPage"
import LoadingScreen from "./components/LoadingScreen"

function AnimatedRoutes() {
  const location = useLocation()
  const [loading, setLoading] = useState(true)
  const [currentPath, setCurrentPath] = useState(location.pathname)

  const handleDone = useCallback(() => {
    setLoading(false)
  }, [])

  useEffect(() => {
    if (location.pathname !== currentPath) {
      setLoading(true)
      setCurrentPath(location.pathname)
    }
  }, [location.pathname, currentPath])

  return (
    <>
      {loading && <LoadingScreen key={currentPath} onDone={handleDone} />}
      <div style={{ opacity: loading ? 0 : 1, transition: "opacity 0.3s ease" }}>
        <Routes location={location}>
          <Route path="/" element={<HomePage />} />
          <Route path="/demo" element={<DemoPage />} />
        </Routes>
      </div>
    </>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <AnimatedRoutes />
    </BrowserRouter>
  )
}