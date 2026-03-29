import React, { useRef, useState, useEffect, useCallback } from "react";
import Webcam from "react-webcam";
import axios from "axios";

const API = "https://face-mask-detection-face-recognation.onrender.com";
const INTERVAL_MS = 2000;   // send frame every 2 seconds

const STATUS_STYLE = {
  no_face: { color: "#6B7280", bg: "rgba(107,114,128,.12)", border: "rgba(107,114,128,.3)", emoji: "👤" },
  masked: { color: "#F59E0B", bg: "rgba(245,158,11,.12)", border: "rgba(245,158,11,.3)", emoji: "😷" },
  identified: { color: "#10B981", bg: "rgba(16,185,129,.12)", border: "rgba(16,185,129,.3)", emoji: "✅" },
  unknown: { color: "#EF4444", bg: "rgba(239,68,68,.12)", border: "rgba(239,68,68,.3)", emoji: "❓" },
};

export default function Detection() {
  const cam = useRef(null);
  const timerRef = useRef(null);
  const [active, setActive] = useState(false);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [camReady, setCamReady] = useState(false);
  const [sending, setSending] = useState(false);

  const sendFrame = useCallback(async () => {
    if (!cam.current || sending) return;
    const img = cam.current.getScreenshot();
    if (!img) return;

    setSending(true);
    try {
      const fd = new FormData();
      fd.append("image", img);
      const r = await axios.post(`${API}/detect`, fd);
      const d = r.data;
      setResult(d);

      if (d.status === "identified") {
        setHistory(prev => [
          { name: d.name, department: d.department, confidence: d.confidence, time: new Date().toLocaleTimeString(), logged: d.logged },
          ...prev.slice(0, 14)
        ]);
      }
    } catch (e) {
      const detail = e.response?.data?.detail?.split("\n")[0] || "Server error";
      setResult({ status: "error", label: `⚠ ${detail}`, color: "#EF4444" });
    } finally {
      setSending(false);
    }
  }, [sending]);

  useEffect(() => {
    if (active) {
      timerRef.current = setInterval(sendFrame, INTERVAL_MS);
    } else {
      clearInterval(timerRef.current);
    }
    return () => clearInterval(timerRef.current);
  }, [active, sendFrame]);

  const style = result ? (STATUS_STYLE[result.status] || STATUS_STYLE.no_face) : null;

  return (
    <div>
      <div className="ph">
        <h1>Live <span style={{ color: "var(--blue)" }}>Detection</span></h1>
        <p>Camera scans every 2 seconds — mask check runs before identity check</p>
      </div>

      <div className="g2" style={{ alignItems: "start", gap: 24 }}>
        {/* ── Camera feed ── */}
        <div>
          <div className="card" style={{ marginBottom: 16 }}>
            {/* Status bar */}
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <div style={{
                  width: 9, height: 9, borderRadius: "50%",
                  background: active ? "#10B981" : "#6B7280",
                  boxShadow: active ? "0 0 8px #10B981" : "none",
                  animation: active ? "pulse 1.8s infinite" : "none"
                }} />
                <span style={{ fontSize: 12, fontFamily: "monospace", color: active ? "#10B981" : "#6B7280", fontWeight: 700 }}>
                  {active ? "SCANNING" : "IDLE"}
                </span>
              </div>
              {sending && <div className="spin" />}
            </div>

            {/* Webcam */}
            <div className="ww" style={{ marginBottom: 14 }}>
              <Webcam
                ref={cam}
                screenshotFormat="image/jpeg"
                screenshotQuality={0.9}
                width="100%"
                mirrored={true}
                onUserMedia={() => setCamReady(true)}
              />

              {/* Scan overlay */}
              {active && (
                <div style={{ position: "absolute", inset: 0, pointerEvents: "none" }}>
                  <div style={{ position: "absolute", left: 0, right: 0, height: 2, background: "linear-gradient(to right,transparent,#10B981,transparent)", animation: "scanline 2s linear infinite", opacity: .55 }} />
                  {/* Corner brackets */}
                  {[
                    { top: 12, left: 12, borderTop: "2px solid #10B981", borderLeft: "2px solid #10B981" },
                    { top: 12, right: 12, borderTop: "2px solid #10B981", borderRight: "2px solid #10B981" },
                    { bottom: 12, left: 12, borderBottom: "2px solid #10B981", borderLeft: "2px solid #10B981" },
                    { bottom: 12, right: 12, borderBottom: "2px solid #10B981", borderRight: "2px solid #10B981" },
                  ].map((s, i) => <div key={i} style={{ position: "absolute", width: 22, height: 22, ...s }} />)}
                </div>
              )}

              {/* Result overlay */}
              {result && style && (
                <div style={{
                  position: "absolute", bottom: 0, left: 0, right: 0,
                  padding: "14px 16px",
                  background: `linear-gradient(to top,${style.bg} 70%,transparent)`,
                  backdropFilter: "blur(3px)",
                }}>
                  <div style={{ fontSize: 22, fontWeight: 800, color: style.color }}>
                    {style.emoji} {result.label}
                  </div>
                  {result.department && (
                    <div style={{ fontSize: 12, color: "rgba(255,255,255,.55)", marginTop: 2 }}>{result.department}</div>
                  )}
                  {result.confidence != null && (
                    <div style={{ fontSize: 11, fontFamily: "monospace", color: style.color, opacity: .8, marginTop: 3 }}>
                      {result.confidence}% confidence
                    </div>
                  )}
                  {result.status === "identified" && (
                    <div style={{ fontSize: 11, color: result.logged ? "#10B981" : "#94A3B8", marginTop: 3 }}>
                      {result.logged ? "📝 Entry logged to Word doc" : "⏳ Cooldown — not re-logged"}
                    </div>
                  )}
                </div>
              )}
            </div>

            <button
              className={`btn btn-full ${active ? "btn-danger" : "btn-primary"}`}
              onClick={() => { setActive(a => !a); setResult(null); }}
              disabled={!camReady}
            >
              {active ? "⏹ Stop Detection" : "▶ Start Detection"}
            </button>
          </div>

          {/* Legend */}
          <div className="card">
            <div className="sl">What each status means</div>
            {[
              { e: "😷", l: "Mask ON", d: "Face found but mask detected → identity hidden, NOT logged", c: "#F59E0B" },
              { e: "✅", l: "Identified", d: "Known person, no mask → logged to Word doc", c: "#10B981" },
              { e: "❓", l: "Unknown", d: "No mask, face not in database", c: "#EF4444" },
              { e: "👤", l: "No Face", d: "No face visible in frame", c: "#6B7280" },
            ].map(x => (
              <div key={x.l} style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 12 }}>
                <div style={{ width: 36, height: 36, borderRadius: 8, background: `${x.c}18`, border: `1px solid ${x.c}30`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 16, flexShrink: 0 }}>{x.e}</div>
                <div>
                  <div style={{ fontWeight: 700, fontSize: 13, color: x.c }}>{x.l}</div>
                  <div style={{ fontSize: 12, color: "var(--t3)" }}>{x.d}</div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* ── History ── */}
        <div className="card">
          <div className="sl">Identified This Session ({history.length})</div>
          {history.length === 0 ? (
            <div style={{ textAlign: "center", padding: "48px 0", color: "var(--t3)" }}>
              <div style={{ fontSize: 36, marginBottom: 10 }}>📷</div>
              <div>No identifications yet</div>
              <div style={{ fontSize: 12, marginTop: 4 }}>Start detection and stand in front of the camera</div>
            </div>
          ) : (
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              {history.map((h, i) => (
                <div key={i} style={{
                  display: "flex", alignItems: "center", justifyContent: "space-between",
                  padding: "11px 14px", background: "var(--bg2)", borderRadius: 8, border: "1px solid var(--border)"
                }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                    <div style={{ width: 34, height: 34, borderRadius: "50%", background: "var(--green-d)", border: "1px solid rgba(16,185,129,.3)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 15 }}>
                      👤
                    </div>
                    <div>
                      <div style={{ fontWeight: 700, fontSize: 13 }}>{h.name}</div>
                      <div style={{ fontSize: 11, color: "var(--t3)" }}>{h.department || "—"}</div>
                    </div>
                  </div>
                  <div style={{ textAlign: "right" }}>
                    <div style={{ fontSize: 11, fontFamily: "monospace", color: "var(--green)" }}>{h.confidence}%</div>
                    <div style={{ fontSize: 11, color: "var(--t3)" }}>{h.time}</div>
                    <div style={{ fontSize: 10, color: h.logged ? "var(--green)" : "var(--t3)" }}>{h.logged ? "📝 logged" : "—"}</div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      <style>{`@keyframes scanline{0%{top:0}100%{top:100%}}`}</style>
    </div>
  );
}
