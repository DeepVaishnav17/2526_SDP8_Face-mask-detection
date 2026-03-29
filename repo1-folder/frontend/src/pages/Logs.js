import React, { useEffect, useState } from "react";
import axios from "axios";

const API = "https://face-mask-detection-face-recognation.onrender.com";

export default function Logs() {
  const [entries, setEntries] = useState([]);
  const [persons, setPersons] = useState([]);
  const [loading, setLoading] = useState(false);
  const [logReady, setLogReady] = useState(false);

  const [isAdmin, setIsAdmin] = useState(() => localStorage.getItem("faceguard_admin") === "true");
  const [password, setPassword] = useState("");

  const fetchAll = async () => {
    setLoading(true);
    try {
      const [e, p] = await Promise.all([
        axios.get(`${API}/logs/entries`),
        axios.get(`${API}/persons`),
      ]);
      setEntries(e.data.entries || []);
      setPersons(p.data.persons || []);
      axios.get(`${API}/logs/check`).then(res => setLogReady(res.data.exists)).catch(() => setLogReady(false));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (isAdmin) fetchAll();
  }, [isAdmin]);

  const avgConf = entries.length
    ? Math.round(entries.reduce((s, e) => s + (e.confidence || 0), 0) / entries.length)
    : 0;

  if (!isAdmin) {
    return (
      <div style={{ display: "flex", justifyContent: "center", marginTop: 100 }}>
        <div className="card" style={{ width: 380, textAlign: "center", padding: "40px 30px", borderRadius: 16, background: "linear-gradient(145deg, rgba(30,41,59,0.7) 0%, rgba(15,23,42,0.9) 100%)", boxShadow: "0 20px 40px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.1)", backdropFilter: "blur(10px)", border: "1px solid rgba(255,255,255,0.05)" }}>
          <div style={{ fontSize: 42, marginBottom: 16 }}>🛡️</div>
          <h2 style={{ color: "#fff", marginTop: 0, marginBottom: 8, fontSize: 24, fontWeight: 700, letterSpacing: "-0.5px" }}>Admin Access</h2>
          <p style={{ color: "#94A3B8", fontSize: 14, marginBottom: 32, lineHeight: 1.5 }}>Enter your administrative password to view system logs and manage users.</p>
          <form onSubmit={e => {
            e.preventDefault();
            if (password === "admin123") {
              setIsAdmin(true);
              localStorage.setItem("faceguard_admin", "true");
            }
            else alert("Incorrect password!");
          }}>
            <div style={{ position: "relative", marginBottom: 24 }}>
              <span style={{ position: "absolute", left: 14, top: 12, fontSize: 16, opacity: 0.6 }}>🔑</span>
              <input
                type="password"
                placeholder="Enter password..."
                value={password}
                onChange={e => setPassword(e.target.value)}
                style={{
                  width: "100%",
                  padding: "12px 14px 12px 42px",
                  borderRadius: 8,
                  border: "1px solid rgba(255,255,255,0.15)",
                  background: "rgba(0,0,0,0.3)",
                  color: "#fff",
                  fontSize: 15,
                  boxSizing: "border-box",
                  outline: "none",
                  transition: "border-color 0.2s ease"
                }}
              />
            </div>
            <button
              className="btn btn-primary"
              style={{
                width: "100%",
                padding: "12px",
                fontSize: 15,
                fontWeight: 600,
                borderRadius: 8,
                background: "linear-gradient(135deg, #10B981 0%, #059669 100%)",
                border: "none",
                color: "#fff",
                cursor: "pointer",
                boxShadow: "0 4px 12px rgba(16,185,129,0.3)",
                transition: "transform 0.1s ease"
              }}
            >
              🔓 Login to Dashboard
            </button>
          </form>
        </div>
      </div>
    );
  }

  return (
    <div>
      <div className="ph">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
          <div>
            <h1>Entry <span style={{ color: "var(--amber)" }}>Logs</span></h1>
            <p>All detected persons are logged here. Word doc saves to <code style={{ fontSize: 12, color: "var(--green)" }}>logs/entry_log.docx</code></p>
          </div>
          <div style={{ display: "flex", gap: 10 }}>
            <button className="btn btn-ghost" onClick={() => { setIsAdmin(false); localStorage.removeItem("faceguard_admin"); }}>🚪 Logout</button>
            <button className="btn btn-ghost" onClick={fetchAll}>🔄 Refresh</button>
            {logReady && (
              <a href={`${API}/logs/download`} download style={{ textDecoration: "none" }}>
                <button className="btn btn-primary">📥 Download Word Doc</button>
              </a>
            )}
          </div>
        </div>
      </div>

      {/* Stats */}
      <div className="g3" style={{ marginBottom: 24 }}>
        <div className="sc"><div className="val" style={{ color: "var(--green)" }}>{entries.length}</div><div className="lbl">Total Entries</div></div>
        <div className="sc"><div className="val" style={{ color: "var(--blue)" }}>{persons.length}</div><div className="lbl">Registered Persons</div></div>
        <div className="sc"><div className="val" style={{ color: "var(--amber)" }}>{avgConf}%</div><div className="lbl">Avg. Confidence</div></div>
      </div>

      {/* Entry log table */}
      <div className="card" style={{ marginBottom: 24 }}>
        <div className="sl">Detection Entries</div>
        {loading ? (
          <div style={{ display: "flex", justifyContent: "center", padding: 40 }}><div className="spin" /></div>
        ) : entries.length === 0 ? (
          <div style={{ textAlign: "center", padding: "40px 0", color: "var(--t3)" }}>
            <div style={{ fontSize: 36, marginBottom: 10 }}>📄</div>
            <div>No entries yet</div>
            <div style={{ fontSize: 12, marginTop: 4 }}>Run detection and get identified to create entries</div>
          </div>
        ) : (
          <table className="dt">
            <thead>
              <tr><th>#</th><th>Name</th><th>Department</th><th>Date & Time</th><th>Confidence</th></tr>
            </thead>
            <tbody>
              {[...entries].reverse().map((e, i) => (
                <tr key={i}>
                  <td style={{ fontFamily: "monospace", fontSize: 11, color: "var(--t3)" }}>{entries.length - i}</td>
                  <td style={{ color: "var(--t1)", fontWeight: 700 }}>{e.name}</td>
                  <td>{e.department || "—"}</td>
                  <td style={{ fontFamily: "monospace", fontSize: 12 }}>{new Date(e.timestamp).toLocaleString()}</td>
                  <td>
                    <span className={`badge ${e.confidence >= 80 ? "b-green" : e.confidence >= 60 ? "b-amber" : "b-red"}`}>
                      {(e.confidence || 0).toFixed(1)}%
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* Registered persons management */}
      <div className="card">
        <div className="sl">Manage Registered Persons</div>
        {persons.length === 0 ? (
          <div style={{ textAlign: "center", padding: "28px 0", color: "var(--t3)", fontSize: 13 }}>
            No persons registered. Go to Register tab to add people.
          </div>
        ) : (
          <table className="dt">
            <thead><tr><th>Name</th><th>Department</th><th>Registered</th><th></th></tr></thead>
            <tbody>
              {persons.map(p => (
                <tr key={p.id}>
                  <td style={{ color: "var(--t1)", fontWeight: 700 }}>{p.name}</td>
                  <td>{p.department || "—"}</td>
                  <td style={{ fontFamily: "monospace", fontSize: 12 }}>{new Date(p.registered_at).toLocaleDateString()}</td>
                  <td>
                    <button className="btn btn-danger" style={{ padding: "4px 12px", fontSize: 12 }}
                      onClick={async () => {
                        if (!window.confirm(`Remove "${p.name}"?`)) return;
                        await axios.delete(`${API}/persons/${p.id}`).catch(() => { });
                        fetchAll();
                      }}>
                      🗑 Remove
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
