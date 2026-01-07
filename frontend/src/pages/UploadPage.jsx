import React, { useState, useRef } from "react";

const DEV_MODE = false; // keep dev helpers visible for now
const SHOW_DEMO_FRAMES = true; // toggle demo thumbnails
const USE_DEMO_ONLY = false; // <<< set to false later to call real backend

const DEMO_THUMBNAIL_URL = "/demo_frame.png"; // demo image in /public

function UploadPage() {
  const [videoFile, setVideoFile] = useState(null);
  const [tokens, setTokens] = useState([]); // each detected sign
  const [sentence, setSentence] = useState("");
  const [showTokens, setShowTokens] = useState(true);
  const [loading, setLoading] = useState(false);
  const videoRef = useRef(null);

  const resetPrediction = () => {
    setTokens([]);
    setSentence("");
    setShowTokens(true);
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setVideoFile(file);
    resetPrediction();
  };

  // Helper: normalise future real backend responses into our token format
  const normalizePrediction = (data) => {
    // Support both future multi-token and current single-token responses
    let seq =
      data.sequence ||
      data.tokens ||
      (Array.isArray(data) ? data : null);

    let tokenList = [];

    if (Array.isArray(seq) && seq.length > 0) {
      tokenList = seq.map((item, idx) => ({
        id: item.id ?? idx,
        gloss: item.gloss ?? `TOKEN_${idx + 1}`,
        translation: item.translation ?? "",
        confidence:
          typeof item.confidence === "number" ? item.confidence : null,
        thumbnailUrl: item.thumbnail_url ?? item.thumbnailUrl ?? null,
        thumbnailType: item.thumbnail_type ?? item.thumbnailType ?? "image",
      }));
    } else if (data.gloss || data.translation) {
      tokenList = [
        {
          id: 0,
          gloss: data.gloss ?? "UNKNOWN",
          translation: data.translation ?? "",
          confidence:
            typeof data.confidence === "number" ? data.confidence : null,
          thumbnailUrl: data.thumbnail_url ?? data.thumbnailUrl ?? null,
          thumbnailType: data.thumbnail_type ?? data.thumbnailType ?? "image",
          clipUrl: data.clip_url ?? data.clipUrl ?? null,
        },
      ];
    }

    const finalSentence =
      data.sentence ||
      tokenList
        .map((t) => t.translation || t.gloss)
        .filter(Boolean)
        .join(" ");

    return { tokenList, finalSentence };
  };

  // Demo-only prediction for now (e.g. 56-second clip with a few key signs)
  const applyDemoPrediction = () => {
    const sampleTokens = [
      {
        id: 0,
        gloss: "SALAAM",
        translation: "Hello",
        confidence: 0.96,
      },
      {
        id: 1,
        gloss: "SAYA",
        translation: "I",
        confidence: 0.93,
      },
      {
        id: 2,
        gloss: "PELAJAR",
        translation: "am a student",
        confidence: 0.9,
      },
      {
        id: 3,
        gloss: "UNIVERSITI_MALAYA",
        translation: "at Universiti Malaya",
        confidence: 0.92,
      },
    ];

    setTokens(sampleTokens);
    setSentence("Hello, I am a student at Universiti Malaya.");
    setShowTokens(true);
  };

  const handleTranslate = async () => {
    if (!videoFile) return;
    setLoading(true);
    resetPrediction();

    // --- DEMO PATH: no backend yet ---
    if (USE_DEMO_ONLY) {
      // Simulate a short processing delay so it feels real
      setTimeout(() => {
        applyDemoPrediction();
        setLoading(false);
      }, 600);
      return;
    }

    // --- REAL BACKEND PATH (future) ---
    try {
      const formData = new FormData();
      formData.append("file", videoFile);

      const res = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        throw new Error("Backend error");
      }

      const data = await res.json();
      const { tokenList, finalSentence } = normalizePrediction(data);
      setTokens(tokenList);
      setSentence(finalSentence);
      setShowTokens(true);
    } catch (err) {
      console.error(err);
      setTokens([
        {
          id: 0,
          gloss: "ERROR",
          translation: "Unable to get prediction. Check backend.",
          confidence: null,
        },
      ]);
      setSentence("");
      setShowTokens(true);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setVideoFile(null);
    resetPrediction();
    if (videoRef.current) {
      videoRef.current.pause();
      videoRef.current.currentTime = 0;
    }
  };

  // Dev-only: manual trigger for the same demo sequence
  const handleDevTestPrediction = () => {
    applyDemoPrediction();
  };

  const videoUrl = videoFile ? URL.createObjectURL(videoFile) : null;

  return (
    <div className="page-root">
      <div className="page-card" style={styles.container}>
        <h1 style={styles.title}>Upload & Translate MSL Video</h1>
        <p style={styles.subtitle}>
          Here you can upload a short Malaysian Sign Language (MSL) video and see
          how our system could turn the signs into simple text.
        </p>

        <div style={styles.section}>
          <label style={styles.label}>1. Upload video file</label>
          <div style={styles.uploadBox}>
            <input
              type="file"
              accept="video/*"
              onChange={handleFileChange}
              style={styles.fileInput}
            />
          </div>
          <p style={styles.helperText}>
            Tip: clear background, good lighting, and stable camera make the signs
            easier to recognise.
          </p>
        </div>

        {videoUrl && (
          <div style={styles.section}>
            <label style={styles.label}>2. Preview</label>
            <div style={styles.videoFrame}>
              <video
                ref={videoRef}
                src={videoUrl}
                controls
                style={styles.video}
              />
            </div>
          </div>
        )}

        <div className="button-row" style={{ ...styles.section, ...styles.actionsRow }}>
          <button
            onClick={handleTranslate}
            style={{
              ...styles.buttonPrimary,
              ...(loading || !videoFile ? styles.buttonDisabled : {}),
            }}
            disabled={loading || !videoFile}
          >
            {loading ? "Translating..." : "Translate Sign"}
          </button>

          {videoFile && (
            <button onClick={handleReset} style={styles.buttonGhost}>
              Reset
            </button>
          )}

          {DEV_MODE && (
            <button onClick={handleDevTestPrediction} style={styles.devButton}>
              Dev: Sample Sequence
            </button>
          )}
        </div>

        {tokens.length > 0 && (
          <div style={styles.section}>
            <h2 style={styles.resultTitle}>3. Detected signs (demo)</h2>

            {/* Sentence ABOVE cards */}
            {sentence && (
              <div style={styles.sentenceBox}>
                <span style={styles.sentenceLabel}>Combined sentence:</span>
                <span style={styles.sentenceText}>{sentence}</span>
              </div>
            )}

            {/* Optional cards with caret toggle */}
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
                          <img
                            src={t.thumbnailUrl || DEMO_THUMBNAIL_URL}
                            alt={`Preview for ${t.gloss}`}
                            style={styles.tokenMedia}
                          />
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
                          <div style={styles.tokenTranslation}>{t.translation}</div>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {!sentence && (
              <p style={styles.note}>
                Note: In a full system, these detected signs could be joined into
                an MSL-aware sentence or separated by full stops.
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

const styles = {
  container: {
    display: "flex",
    flexDirection: "column",
    gap: "14px",
  },
  title: {
    fontSize: "1.5rem",
    marginBottom: "2px",
    color: "#111827",
  },
  subtitle: {
    fontSize: "0.95rem",
    color: "#4b5563",
    maxWidth: "640px",
  },
  section: {
    marginTop: "8px",
    marginBottom: "4px",
  },
  label: {
    display: "block",
    marginBottom: "6px",
    fontWeight: 600,
    fontSize: "0.9rem",
    color: "#111827",
  },
  uploadBox: {
    borderRadius: "14px",
    border: "1px dashed #c4b5fd",
    background: "#f5f3ff",
    padding: "10px 12px",
  },
  fileInput: {
    width: "100%",
    color: "#4b5563",
  },
  helperText: {
    marginTop: "6px",
    fontSize: "0.8rem",
    color: "#6b7280",
  },
  videoFrame: {
    borderRadius: "16px",
    border: "1px solid #e5e7eb",
    background: "#f9fafb",
    padding: "8px",
  },
  video: {
    width: "100%",
    borderRadius: "12px",
    backgroundColor: "#000000",
  },
  actionsRow: {
    display: "flex",
    flexWrap: "wrap",
    alignItems: "center",
    gap: "10px",
  },
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
  buttonDisabled: {
    opacity: 0.55,
    cursor: "not-allowed",
    boxShadow: "none",
  },
  buttonGhost: {
    padding: "8px 16px",
    borderRadius: "999px",
    border: "1px solid #e5e7eb",
    cursor: "pointer",
    fontWeight: 500,
    fontSize: "0.9rem",
    background: "#ffffff",
    color: "#4b5563",
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
  resultTitle: {
    fontSize: "1.05rem",
    marginBottom: "8px",
    color: "#111827",
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
    color: "#4c4853ff", // your grey-purple
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
  tokenMedia: {
    width: "100%",
    borderRadius: "10px",
    marginBottom: "6px",
    display: "block",
    objectFit: "cover",
  },
  note: {
    marginTop: "8px",
    fontSize: "0.8rem",
    color: "#6b7280",
  },
};

export default UploadPage;
