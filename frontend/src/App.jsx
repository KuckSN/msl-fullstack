import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import "./global.css";
import Navbar from "./components/Navbar";
import LandingPage from "./pages/LandingPage";
import UploadPage from "./pages/UploadPage";
import CameraPage from "./pages/CameraPage";
import LearnPage from "./pages/LearnPage";


function App() {
  return (
    <Router>
      <div style={styles.page}>
        <div style={styles.shell}>
          {/* Universiti Malaya style header */}
          <header style={styles.header}>
            <div style={styles.headerLeft}>
              <img
                src="/um_logo.png"
                alt="Universiti Malaya logo"
                style={styles.logo}
              />
              <div>
                <div style={styles.headerTitle}>Universiti Malaya</div>
                <div style={styles.headerSubtitle}>
                  WQF7006 · Computer Vision & Image Processing
                </div>
              </div>
            </div>
            <div style={styles.headerRight}>
              <span style={styles.tag}>MSL AI Translator</span>
            </div>
          </header>

          {/* Navigation bar */}
          <Navbar />

          {/* Main card containing pages */}
          <main style={styles.mainCard}>
            <Routes>
              <Route path="/" element={<LandingPage />} />
              <Route path="/upload" element={<UploadPage />} />
              <Route path="/camera" element={<CameraPage />} />
              <Route path="/learn" element={<LearnPage />} />
            </Routes>
          </main>
        </div>
      </div>
    </Router>
  );
}

const styles = {
  page: {
    minHeight: "100vh",
    background:
      "radial-gradient(circle at top, #ede9fe 0%, #e0f2fe 40%, #f9fafb 100%)",
    display: "flex",
    justifyContent: "center",
    alignItems: "flex-start",
    padding: "32px 16px",
    fontFamily:
      "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
  },
  shell: {
    width: "100%",
    maxWidth: "1040px",
  },
  header: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    background: "#ffffff",
    padding: "10px 18px",
    borderRadius: "999px",
    marginBottom: "14px",
    boxShadow: "0 6px 20px rgba(15,23,42,0.10)",
    border: "1px solid #e5e7eb",
  },
  headerLeft: {
    display: "flex",
    alignItems: "center",
    gap: "10px",
  },
  logo: {
    width: "40px",
    height: "40px",
    borderRadius: "8px",
    objectFit: "contain",
    backgroundColor: "#ffffff",
    boxShadow: "0 0 0 2px rgba(15,23,42,0.08)",
  },
  headerTitle: {
    fontSize: "1rem",
    fontWeight: 700,
    letterSpacing: "0.03em",
    textTransform: "uppercase",
    color: "#111827",
  },
  headerSubtitle: {
    fontSize: "0.8rem",
    color: "#6b7280",
  },
  headerRight: {
    display: "flex",
    alignItems: "center",
  },
  tag: {
    fontSize: "0.78rem",
    padding: "4px 10px",
    borderRadius: "999px",
    border: "1px solid #e5e7eb",
    background: "#f3e8ff",
    color: "#6d28d9",
    fontWeight: 600,
  },
  mainCard: {
    background: "#ffffff",
    borderRadius: "24px",
    padding: "22px 24px 20px",
    boxShadow: "0 18px 50px rgba(15,23,42,0.10)",
    border: "1px solid #e5e7eb",
  },
};

export default App;
