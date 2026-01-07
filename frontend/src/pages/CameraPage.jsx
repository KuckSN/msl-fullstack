import React, { useRef, useState, useEffect } from "react";

const DEV_MODE = false;
const SHOW_DEMO_FRAMES = true;
const DEMO_THUMBNAIL_URL = "/demo_frame.png";

const CAMERA_DEMO_SEQUENCE = [
  { gloss: "SAYA", translation: "I" },
  { gloss: "SUKA", translation: "love" },
  { gloss: "BELAJAR", translation: "learning" },
  { gloss: "WQF7002", translation: "WQF7002" },
  { gloss: "KOMPUTER_VISION", translation: "Computer Vision" },
  { gloss: "DAN", translation: "and" },
  { gloss: "PEMINPROSESAN_IMEJ", translation: "Image Processing" },
];

const GLOSS_ORDER = CAMERA_DEMO_SEQUENCE.map((t) => t.gloss);
const GLOSS_TO_TRANSLATION = CAMERA_DEMO_SEQUENCE.reduce((acc, t) => {
  acc[t.gloss] = t.translation;
  return acc;
}, {});

function CameraPage() {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const canvasRef = useRef(null);
  const detectTimerRef = useRef(null);
  const sessionIdRef = useRef(null);

  const [error, setError] = useState("");
  const [isCameraOn, setIsCameraOn] = useState(false);
  const [tokens, setTokens] = useState([]);
  const [sentenceParts, setSentenceParts] = useState([]); // one fragment per cycle
  const [showTokens, setShowTokens] = useState(true);

  // Capture current frame as JPEG blob
  const captureFrameBlob = () =>
    new Promise((resolve) => {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (!video || !canvas) {
        resolve(null);
        return;
      }

      if (video.readyState < 2) {
        resolve(null);
        return;
      }

      const width = video.videoWidth || 640;
      const height = video.videoHeight || 480;
      if (width === 0 || height === 0) {
        resolve(null);
        return;
      }

      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        resolve(null);
        return;
      }

      ctx.drawImage(video, 0, 0, width, height);
      canvas.toBlob(
        (blob) => {
          resolve(blob);
        },
        "image/jpeg",
        0.85
      );
    });

  const stopDetectionLoop = () => {
    if (detectTimerRef.current) {
      clearInterval(detectTimerRef.current);
      detectTimerRef.current = null;
    }
  };

  const stopStream = () => {
    stopDetectionLoop();
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsCameraOn(false);
  };

  const handleTokenFromBackend = (tokenFromBackend) => {
    if (!tokenFromBackend) return;

    const gloss = tokenFromBackend.gloss || "UNKNOWN";
    const translation =
      tokenFromBackend.translation ||
      GLOSS_TO_TRANSLATION[gloss] ||
      gloss;

    const confidence =
      typeof tokenFromBackend.confidence === "number"
        ? tokenFromBackend.confidence
        : 0.9;

    const thumbnailType =
      tokenFromBackend.thumbnail_type ||
      tokenFromBackend.thumbnailType ||
      "image";

    const thumbnailUrl =
      tokenFromBackend.thumbnail_url ||
      tokenFromBackend.thumbnailUrl ||
      null;

    // 1. Append token (never clear while camera is on)
    const newToken = {
      id:
        tokenFromBackend.id ??
        Date.now() + "-" + Math.random().toString(16).slice(2),
      gloss,
      translation,
      confidence,
      thumbnailType,
      thumbnailUrl,
    };

    setTokens((prev) => [...prev, newToken]);

    // 2. Update combined sentence fragments based on gloss order
    const idx = GLOSS_ORDER.indexOf(gloss);
    const phraseUpToIdx =
      idx >= 0
        ? GLOSS_ORDER.slice(0, idx + 1)
            .map((g) => GLOSS_TO_TRANSLATION[g] || g)
            .join(" ")
        : translation;

    setSentenceParts((prev) => {
      const next = [...prev];

      if (next.length === 0) {
        // First detection
        next.push(phraseUpToIdx);
      } else {
        if (idx === 0 && GLOSS_ORDER.length > 0 && gloss === GLOSS_ORDER[0]) {
          // Seeing the first gloss again -> start new sentence cycle
          next.push(phraseUpToIdx);
        } else if (idx >= 0) {
          // Extend current fragment
          next[next.length - 1] = phraseUpToIdx;
        } else {
          // Unknown gloss: just append word
          next[next.length - 1] =
            (next[next.length - 1] + " " + translation).trim();
        }
      }

      return next;
    });
  };

  const startDetectionLoop = () => {
    const tick = async () => {
      const blob = await captureFrameBlob();
      if (!blob) return;

      const sid = sessionIdRef.current || "default";

      const formData = new FormData();
      formData.append("frame", blob, "frame.jpg");
      formData.append("session_id", sid);

      try {
        const res = await fetch("http://localhost:8000/predict_camera", {
          method: "POST",
          body: formData,
        });

        if (!res.ok) {
          console.error("predict_camera returned non-OK:", res.status);
          return;
        }

        const data = await res.json();
        // Backend returns { token: {...} } or { token: null }
        if (!data || !data.token) {
          return; // no completed sign yet
        }

        const token = data.token;
        handleTokenFromBackend(token);
      } catch (err) {
        console.error("Error calling /predict_camera:", err);
      }
    };

    // fire every ~300ms for smoother temporal context
    detectTimerRef.current = setInterval(tick, 300);
  };


  const handleStartCamera = async () => {
    setError("");

    // Clear previous results whenever camera is started
    setTokens([]);
    setSentenceParts([]);
    setShowTokens(true);

    // ensure old stream + loop are stopped
    stopStream();

    const newSessionId =
      "sess-" + Date.now() + "-" + Math.random().toString(16).slice(2);
    sessionIdRef.current = newSessionId;

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
      });

      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      setIsCameraOn(true);

      // Start real-time detection loop
      startDetectionLoop();
    } catch (err) {
      console.error(err);

      if (
        err?.message?.toLowerCase().includes("reserved") ||
        err?.name === "NotReadableError"
      ) {
        setError(
          "It looks like another app is using your camera. Close apps like Zoom, Teams, or the Camera app, then try again."
        );
      } else if (err?.name === "NotAllowedError") {
        setError(
          "Camera permission was blocked. Please allow camera access in your browser and try again."
        );
      } else {
        setError(
          "We couldn't access your camera. Please check settings or try a different browser."
        );
      }

      stopStream();
    }
  };

  const handleStopCamera = () => {
    stopStream();
    // Keep tokens & sentenceParts frozen on screen
  };

  const handleDevSample = () => {
    const sampleTokens = [
      {
        id: "dev-0",
        gloss: "SAYA",
        translation: "I",
        confidence: 0.9,
        thumbnailType: "image",
        thumbnailUrl: DEMO_THUMBNAIL_URL,
      },
      {
        id: "dev-1",
        gloss: "SUKA",
        translation: "love",
        confidence: 0.91,
        thumbnailType: "image",
        thumbnailUrl: DEMO_THUMBNAIL_URL,
      },
      {
        id: "dev-2",
        gloss: "BELAJAR",
        translation: "learning",
        confidence: 0.92,
        thumbnailType: "gif",
        thumbnailUrl: DEMO_THUMBNAIL_URL,
      },
    ];
    setTokens((prev) => [...prev, ...sampleTokens]);
    setSentenceParts((prev) => [...prev, "I love learning"]);
    setShowTokens(true);
  };

  useEffect(() => {
    return () => {
      stopStream();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const renderedSentence =
    sentenceParts.length > 0
      ? sentenceParts.map((s) => s.trim()).join(". ") + "."
      : "";

  const hasResults = Boolean(renderedSentence) || tokens.length > 0;

  return (
    <div className="page-root">
      <div className="page-card" style={styles.container}>
        <h1 style={styles.title}>Real-time Camera Prototype</h1>
        <p style={styles.subtitle}>
          The webcam feeds live frames to the backend. For each frame, the server
          returns one detected sign and a thumbnail: static for pose-based signs,
          and a looping GIF for temporal signs.
        </p>

        <div style={styles.section}>
          <div className="button-row" style={styles.buttonRow}>
            <button
              onClick={handleStartCamera}
              style={{
                ...styles.buttonPrimary,
                ...(isCameraOn ? styles.buttonDisabled : {}),
              }}
              disabled={isCameraOn}
            >
              {isCameraOn ? "Camera On" : "Enable Camera"}
            </button>

            <button
              onClick={handleStopCamera}
              style={{
                ...styles.buttonGhost,
                ...(!isCameraOn ? styles.buttonDisabled : {}),
              }}
              disabled={!isCameraOn}
            >
              Disable Camera
            </button>

            {DEV_MODE && (
              <button onClick={handleDevSample} style={styles.devButton}>
                Dev: Sample Sequence
              </button>
            )}
          </div>
        </div>

        <div style={styles.section}>
          <div style={styles.videoFrame}>
            <div style={styles.videoWrapper}>
              <video
                ref={videoRef}
                style={styles.video}
                playsInline
                muted
                autoPlay
              />
              {!isCameraOn && (
                <div style={styles.placeholderOverlay}>
                  <span style={styles.placeholderText}>
                    Camera is off. Click “Enable Camera” to start preview.
                  </span>
                </div>
              )}
            </div>
          </div>
          {/* Hidden canvas for grabbing frames */}
          <canvas ref={canvasRef} style={{ display: "none" }} />
          {error && <p style={styles.error}>{error}</p>}
        </div>

        {/* RESULTS */}
        {hasResults && (
          <div style={styles.section}>
            <h2 style={styles.subheading}>Detected signs (demo)</h2>

            {renderedSentence && (
              <div style={styles.sentenceBox}>
                <span style={styles.sentenceLabel}>Combined sentence:</span>
                <span style={styles.sentenceText}>{renderedSentence}</span>
              </div>
            )}

            {tokens.length > 0 && (
              <div style={styles.tokensSection}>
                <button
                  type="button"
                  style={styles.toggleLink}
                  onClick={() => setShowTokens((prev) => !prev)}
                >
                  {showTokens
                    ? "▾ Hide sign-by-sign details"
                    : "▸ Show sign-by-sign details"}
                </button>

                {showTokens && (
                  <div className="horizontal-cards" style={styles.cardsScroller}>
                    {tokens.map((t, idx) => (
                      <div key={t.id ?? idx} style={styles.tokenCard}>
                        {SHOW_DEMO_FRAMES && (
                          t.thumbnailType === "gif" && t.thumbnailUrl ? (
                            <img
                              src={t.thumbnailUrl}
                              alt={`Animated preview for ${t.gloss}`}
                              style={styles.tokenImage}
                            />
                          ) : (t.thumbnailUrl || DEMO_THUMBNAIL_URL) ? (
                            <img
                              src={t.thumbnailUrl || DEMO_THUMBNAIL_URL}
                              alt={`Preview for ${t.gloss}`}
                              style={styles.tokenImage}
                            />
                          ) : null
                        )}

                        <div style={styles.tokenHeader}>
                          <span style={styles.tokenIndex}>#{idx + 1}</span>
                          {typeof t.confidence === "number" && (
                            <span style={styles.tokenChip}>
                              {(t.confidence * 100).toFixed(1)}%
                            </span>
                          )}
                        </div>
                        <div style={styles.tokenGloss}>{t.gloss}</div>
                        {t.translation && (
                          <div style={styles.tokenTranslation}>
                            {t.translation}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {!hasResults && (
          <div style={styles.section}>
            <h2 style={styles.subheading}>How real-time prediction works here</h2>
            <ul style={styles.list}>
              <li>Grab frames from the webcam at a fixed time interval.</li>
              <li>Send each frame to the backend via a REST API.</li>
              <li>
                Backend simulates the MSL model, returning one detected sign with
                either a GIF (temporal) or static frame (pose).
              </li>
              <li>
                The UI builds the running sentence and sign-by-sign cards you see
                above.
              </li>
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}

const styles = {
  container: { display: "flex", flexDirection: "column", gap: "14px" },
  title: { fontSize: "1.5rem", marginBottom: "2px", color: "#111827" },
  subtitle: {
    fontSize: "0.95rem",
    color: "#4b5563",
    maxWidth: "640px",
  },
  section: { marginTop: "8px" },
  buttonRow: { display: "flex", flexWrap: "wrap", gap: "10px" },
  buttonPrimary: {
    padding: "9px 18px",
    borderRadius: "999px",
    border: "none",
    cursor: "pointer",
    fontWeight: 600,
    fontSize: "0.95rem",
    background:
      "linear-gradient(120deg, #7c3aed 0%, #a855f7 40%, #facc15 100%)",
    color: "#111827",
    boxShadow: "0 10px 24px rgba(129,140,248,0.55)",
  },
  buttonGhost: {
    padding: "9px 18px",
    borderRadius: "999px",
    border: "1px solid #e5e7eb",
    cursor: "pointer",
    fontWeight: 500,
    fontSize: "0.9rem",
    background: "#ffffff",
    color: "#4b5563",
  },
  buttonDisabled: {
    opacity: 0.55,
    cursor: "not-allowed",
    boxShadow: "none",
  },
  devButton: {
    padding: "7px 14px",
    borderRadius: "999px",
    border: "1px dashed #e5e7eb",
    cursor: "pointer",
    fontWeight: 500,
    fontSize: "0.8rem",
    background: "#f3f4f6",
    color: "#6b7280",
  },
  videoFrame: {
    borderRadius: "16px",
    border: "1px solid #e5e7eb",
    background: "#f9fafb",
    padding: "8px",
    minHeight: "220px",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
  },
  videoWrapper: {
    position: "relative",
    width: "100%",
  },
  video: {
    width: "100%",
    borderRadius: "12px",
    backgroundColor: "#000000",
  },
  placeholderOverlay: {
    position: "absolute",
    inset: 0,
    borderRadius: "12px",
    border: "2px dashed #e5e7eb",
    background: "rgba(249,250,251,0.95)",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
  },
  placeholderText: {
    fontSize: "0.85rem",
    color: "#6b7280",
  },
  error: {
    marginTop: "6px",
    fontSize: "0.8rem",
    color: "#b91c1c",
  },
  subheading: {
    fontSize: "1.05rem",
    marginBottom: "6px",
    color: "#111827",
  },
  list: {
    paddingLeft: "18px",
    fontSize: "0.9rem",
    color: "#4b5563",
  },
  sentenceBox: {
    marginTop: "8px",
    marginBottom: "6px",
    padding: "8px 10px",
    borderRadius: "12px",
    background: "#eef2ff",
    border: "1px solid #c7d2fe",
    display: "flex",
    flexWrap: "wrap",
    gap: "4px",
  },
  sentenceLabel: {
    fontSize: "0.8rem",
    fontWeight: 600,
    color: "#4338ca",
  },
  sentenceText: {
    fontSize: "0.85rem",
    color: "#111827",
  },
  tokensSection: {
    marginTop: "4px",
  },
  toggleLink: {
    padding: 0,
    marginBottom: "6px",
    border: "none",
    background: "transparent",
    cursor: "pointer",
    fontSize: "0.8rem",
    color: "#4c4853ff",
    textDecoration: "none",
    fontWeight: 500,
  },
  cardsScroller: {
    display: "flex",
    gap: "10px",
    padding: "6px 2px 6px 0",
    overflowX: "auto",
    scrollbarWidth: "thin",
  },
  tokenCard: {
    minWidth: "140px",
    maxWidth: "160px",
    borderRadius: "14px",
    border: "1px solid #e5e7eb",
    background: "#f9fafb",
    padding: "8px 10px",
    flexShrink: 0,
  },
  tokenHeader: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: "4px",
  },
  tokenIndex: {
    fontSize: "0.75rem",
    color: "#6b7280",
  },
  tokenChip: {
    fontSize: "0.75rem",
    fontWeight: 600,
    padding: "1px 8px",
    borderRadius: "999px",
    background: "#ede9fe",
    color: "#4c1d95",
    border: "1px solid #a855f7",
  },
  tokenGloss: {
    fontSize: "0.9rem",
    fontWeight: 700,
    color: "#111827",
    marginBottom: "2px",
  },
  tokenTranslation: {
    fontSize: "0.8rem",
    color: "#4b5563",
  },
  tokenImage: {
    width: "100%",
    borderRadius: "10px",
    marginBottom: "6px",
    display: "block",
    objectFit: "cover",
  },
};

export default CameraPage;
