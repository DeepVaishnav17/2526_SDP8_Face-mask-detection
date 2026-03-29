// import React, { useRef, useState, useCallback, useEffect } from "react";
// import Webcam from "react-webcam";
// import axios from "axios";

// const API = "http://localhost:8000";

// export default function Register() {
//   const cam = useRef(null);
//   const [captured, setCaptured]   = useState(null);
//   const [name, setName]           = useState("");
//   const [dept, setDept]           = useState("");
//   const [loading, setLoading]     = useState(false);
//   const [msg, setMsg]             = useState(null);     // {type, text}
//   const [persons, setPersons]     = useState([]);
//   const [camReady, setCamReady]   = useState(false);

//   const fetchPersons = () =>
//     axios.get(`${API}/persons`).then(r => setPersons(r.data.persons || [])).catch(()=>{});

//   useEffect(() => { fetchPersons(); }, []);

//   const capture = useCallback(() => {
//     const img = cam.current?.getScreenshot();
//     if (img) { setCaptured(img); setMsg(null); }
//   }, []);

//   const retake = () => { setCaptured(null); setMsg(null); };

//   const submit = async () => {
//     if (!name.trim()) return setMsg({type:"red", text:"Please enter a name."});
//     if (!captured)    return setMsg({type:"red", text:"Please capture a photo first."});
//     setLoading(true); setMsg(null);
//     try {
//       const fd = new FormData();
//       fd.append("name", name.trim());
//       fd.append("department", dept.trim());
//       fd.append("image", captured);
//       const r = await axios.post(`${API}/register`, fd);
//       setMsg({type:"green", text:`✅ ${r.data.name} registered successfully!`});
//       setName(""); setDept(""); setCaptured(null);
//       fetchPersons();
//     } catch(e) {
//       const detail = e.response?.data?.detail || "Registration failed.";
//       // Show only first line of traceback if server error
//       const short = detail.split("\n")[0];
//       setMsg({type:"red", text:`❌ ${short}`});
//     } finally {
//       setLoading(false);
//     }
//   };

//   const deletePerson = async (id, pname) => {
//     if (!window.confirm(`Remove "${pname}"?`)) return;
//     await axios.delete(`${API}/persons/${id}`).catch(()=>{});
//     fetchPersons();
//   };

//   return (
//     <div>
//       <div className="ph">
//         <h1>Register <span style={{color:"var(--green)"}}>Person</span></h1>
//         <p>Add someone to the face recognition system using your webcam</p>
//       </div>

//       <div className="g2" style={{alignItems:"start", gap:24}}>
//         {/* ── Camera ── */}
//         <div className="card">
//           <div className="sl">Step 1 — Capture Photo</div>

//           {!captured ? (
//             <>
//               <div className="ww" style={{marginBottom:14}}>
//                 <Webcam
//                   ref={cam}
//                   screenshotFormat="image/jpeg"
//                   screenshotQuality={0.92}
//                   width="100%"
//                   mirrored={true}
//                   onUserMedia={() => setCamReady(true)}
//                   onUserMediaError={() => setMsg({type:"red", text:"Camera access denied. Please allow camera in browser settings."})}
//                 />
//                 {/* Face guide box */}
//                 <div style={{position:"absolute",inset:0,display:"flex",alignItems:"center",justifyContent:"center",pointerEvents:"none"}}>
//                   <div style={{width:160,height:200,border:"2px solid rgba(16,185,129,.7)",borderRadius:8,boxShadow:"0 0 0 9999px rgba(0,0,0,.15)"}}/>
//                 </div>
//                 <div style={{position:"absolute",top:10,left:10,background:"rgba(0,0,0,.6)",padding:"3px 10px",borderRadius:6,fontSize:10,fontFamily:"monospace",color:"#10B981"}}>
//                   LIVE
//                 </div>
//               </div>
//               <button className="btn btn-primary btn-full" onClick={capture} disabled={!camReady}>
//                 📸 Capture Photo
//               </button>
//             </>
//           ) : (
//             <>
//               <div className="ww" style={{marginBottom:14}}>
//                 <img src={captured} alt="captured"/>
//                 <div style={{position:"absolute",top:10,left:10,background:"rgba(16,185,129,.85)",padding:"3px 10px",borderRadius:6,fontSize:10,fontFamily:"monospace",color:"#fff"}}>
//                   CAPTURED ✓
//                 </div>
//               </div>
//               <button className="btn btn-ghost btn-full" onClick={retake}>
//                 🔄 Retake
//               </button>
//             </>
//           )}
//         </div>

//         {/* ── Form ── */}
//         <div className="card">
//           <div className="sl">Step 2 — Enter Details</div>

//           <div className="ig">
//             <label>Full Name *</label>
//             <input placeholder="e.g. Rahul Sharma" value={name} onChange={e=>setName(e.target.value)} onKeyDown={e=>e.key==="Enter"&&submit()}/>
//           </div>
//           <div className="ig">
//             <label>Department (optional)</label>
//             <input placeholder="e.g. Engineering, HR" value={dept} onChange={e=>setDept(e.target.value)}/>
//           </div>

//           <div className="divider"/>

//           {/* Tips */}
//           <div className="alert a-green" style={{marginBottom:16}}>
//             <strong>📸 For best results:</strong>
//             <ul style={{paddingLeft:16,marginTop:6,lineHeight:1.8,fontSize:12}}>
//               <li>Face the camera directly, centered in the green box</li>
//               <li>Good lighting — no strong backlight</li>
//               <li><strong>Do NOT wear a mask</strong> while registering</li>
//               <li>Keep a neutral expression, eyes open</li>
//             </ul>
//           </div>

//           {msg && <div className={`alert a-${msg.type}`}>{msg.text}</div>}

//           <button className="btn btn-primary btn-full" onClick={submit} disabled={loading||!name.trim()||!captured}>
//             {loading ? <><div className="spin"/> Registering...</> : <>👤 Register Person</>}
//           </button>
//         </div>
//       </div>

//       {/* Registered persons list */}
//       {persons.length > 0 && (
//         <div className="card" style={{marginTop:24}}>
//           <div className="sl">Registered Persons ({persons.length})</div>
//           <table className="dt">
//             <thead><tr><th>Name</th><th>Department</th><th>Registered On</th><th></th></tr></thead>
//             <tbody>
//               {persons.map(p=>(
//                 <tr key={p.id}>
//                   <td style={{color:"var(--t1)",fontWeight:700}}>{p.name}</td>
//                   <td>{p.department||"—"}</td>
//                   <td style={{fontFamily:"monospace",fontSize:12}}>{new Date(p.registered_at).toLocaleDateString()}</td>
//                   <td>
//                     <button className="btn btn-danger" style={{padding:"4px 12px",fontSize:12}} onClick={()=>deletePerson(p.id,p.name)}>
//                       🗑 Remove
//                     </button>
//                   </td>
//                 </tr>
//               ))}
//             </tbody>
//           </table>
//         </div>
//       )}
//     </div>
//   );
// }

import React, { useRef, useState, useCallback, useEffect } from "react";
import Webcam from "react-webcam";
import axios from "axios";

const API = "https://face-mask-detection-face-recognation.onrender.com";

// Camera quality tips shown in real-time
const TIPS = [
  "👁️  Face the camera directly",
  "💡  Make sure your face is well-lit",
  "🚫  Do NOT wear a mask while registering",
  "📐  Keep your face inside the green box",
  "😐  Neutral expression, eyes open",
];

export default function Register() {
  const cam = useRef(null);
  const [captured, setCaptured] = useState(null);
  const [name, setName] = useState("");
  const [dept, setDept] = useState("");
  const [loading, setLoading] = useState(false);
  const [msg, setMsg] = useState(null);     // {type, text}
  const [persons, setPersons] = useState([]);
  const [camReady, setCamReady] = useState(false);
  const [tipIdx, setTipIdx] = useState(0);

  // Cycle tips every 3s
  useEffect(() => {
    const t = setInterval(() => setTipIdx(i => (i + 1) % TIPS.length), 3000);
    return () => clearInterval(t);
  }, []);

  const fetchPersons = () =>
    axios.get(`${API}/persons`).then(r => setPersons(r.data.persons || [])).catch(() => { });

  useEffect(() => { fetchPersons(); }, []);

  const capture = useCallback(() => {
    const img = cam.current?.getScreenshot();
    if (img) { setCaptured(img); setMsg(null); }
  }, []);

  const retake = () => { setCaptured(null); setMsg(null); };

  const submit = async () => {
    if (!name.trim()) return setMsg({ type: "red", text: "Please enter a name." });
    if (!captured) return setMsg({ type: "red", text: "Please capture a photo first." });
    setLoading(true); setMsg(null);
    try {
      const fd = new FormData();
      fd.append("name", name.trim());
      fd.append("department", dept.trim());
      fd.append("image", captured);
      const r = await axios.post(`${API}/register`, fd);
      setMsg({ type: "green", text: `✅ ${r.data.name} registered successfully!` });
      setName(""); setDept(""); setCaptured(null);
      fetchPersons();
    } catch (e) {
      // Show the exact reason from backend
      const detail = e.response?.data?.detail || "Registration failed. Please try again.";
      setMsg({ type: "red", text: `❌ ${detail}` });
    } finally {
      setLoading(false);
    }
  };

  const deletePerson = async (id, pname) => {
    if (!window.confirm(`Remove "${pname}"?`)) return;
    await axios.delete(`${API}/persons/${id}`).catch(() => { });
    fetchPersons();
  };

  return (
    <div>
      <div className="ph">
        <h1>Register <span style={{ color: "var(--green)" }}>Person</span></h1>
        <p>Add someone to the face recognition system using your webcam</p>
      </div>

      <div className="g2" style={{ alignItems: "start", gap: 24 }}>

        {/* ── Camera ── */}
        <div className="card">
          <div className="sl">Step 1 — Capture Photo</div>

          {/* Live tip bar */}
          <div style={{
            background: "rgba(16,185,129,.08)", border: "1px solid rgba(16,185,129,.2)",
            borderRadius: 8, padding: "8px 14px", marginBottom: 12,
            fontSize: 12, color: "var(--green)", fontWeight: 600,
            minHeight: 34, display: "flex", alignItems: "center",
            transition: "all .4s",
          }}>
            {TIPS[tipIdx]}
          </div>

          {!captured ? (
            <>
              <div className="ww" style={{ marginBottom: 14 }}>
                <Webcam
                  ref={cam}
                  screenshotFormat="image/jpeg"
                  screenshotQuality={0.95}
                  width="100%"
                  mirrored={true}
                  videoConstraints={{ width: 640, height: 480, facingMode: "user" }}
                  onUserMedia={() => setCamReady(true)}
                  onUserMediaError={() => setMsg({ type: "red", text: "❌ Camera access denied. Please allow camera in your browser settings." })}
                />
                {/* Face guide oval */}
                <div style={{
                  position: "absolute", inset: 0,
                  display: "flex", alignItems: "center", justifyContent: "center",
                  pointerEvents: "none",
                }}>
                  <div style={{
                    width: 150, height: 195,
                    border: "2.5px solid rgba(16,185,129,.8)",
                    borderRadius: "50%",
                    boxShadow: "0 0 0 9999px rgba(0,0,0,.18), inset 0 0 20px rgba(16,185,129,.08)",
                  }} />
                </div>
                <div style={{
                  position: "absolute", top: 10, left: 10,
                  background: "rgba(0,0,0,.65)", padding: "3px 10px",
                  borderRadius: 6, fontSize: 10, fontFamily: "monospace", color: "#10B981",
                }}>
                  LIVE FEED
                </div>
              </div>

              {!camReady && !msg && (
                <div className="alert a-amber" style={{ marginBottom: 12 }}>
                  ⏳ Waiting for camera... If it doesn't start, check browser permissions.
                </div>
              )}

              <button className="btn btn-primary btn-full" onClick={capture} disabled={!camReady}>
                📸 Capture Photo
              </button>
            </>
          ) : (
            <>
              <div className="ww" style={{ marginBottom: 14 }}>
                <img src={captured} alt="captured" />
                <div style={{
                  position: "absolute", top: 10, left: 10,
                  background: "rgba(16,185,129,.9)", padding: "3px 10px",
                  borderRadius: 6, fontSize: 10, fontFamily: "monospace", color: "#fff",
                }}>
                  CAPTURED ✓
                </div>
              </div>
              <button className="btn btn-ghost btn-full" onClick={retake}>
                🔄 Retake Photo
              </button>
            </>
          )}
        </div>

        {/* ── Form ── */}
        <div className="card">
          <div className="sl">Step 2 — Enter Details & Submit</div>

          <div className="ig">
            <label>Full Name *</label>
            <input
              placeholder="e.g. Rahul Sharma"
              value={name}
              onChange={e => setName(e.target.value)}
              onKeyDown={e => e.key === "Enter" && submit()}
            />
          </div>
          <div className="ig">
            <label>Department (optional)</label>
            <input
              placeholder="e.g. Engineering, HR, Security"
              value={dept}
              onChange={e => setDept(e.target.value)}
            />
          </div>

          <div className="divider" />

          {/* Checklist */}
          <div style={{ marginBottom: 16 }}>
            <div className="sl" style={{ marginBottom: 10 }}>Before you capture, confirm:</div>
            {[
              { label: "Face clearly visible & centered", done: !!captured },
              { label: "Good lighting (not too dark/bright)", done: !!captured },
              { label: "Not wearing a mask", done: true },
              { label: "Name entered", done: !!name.trim() },
            ].map(item => (
              <div key={item.label} style={{
                display: "flex", alignItems: "center", gap: 10,
                padding: "6px 0", fontSize: 13,
                color: item.done ? "var(--green)" : "var(--t2)",
              }}>
                <span style={{ fontSize: 16 }}>{item.done ? "✅" : "⬜"}</span>
                {item.label}
              </div>
            ))}
          </div>

          {/* Error / success message — shown prominently */}
          {msg && (
            <div className={`alert a-${msg.type}`} style={{
              marginBottom: 16,
              whiteSpace: "pre-wrap",
              lineHeight: 1.6,
              fontSize: 13,
            }}>
              {msg.text}
            </div>
          )}

          <button
            className="btn btn-primary btn-full"
            onClick={submit}
            disabled={loading || !name.trim() || !captured}
          >
            {loading
              ? <><div className="spin" /> Registering... (may take 5–10 sec first time)</>
              : <>👤 Register Person</>}
          </button>

          {loading && (
            <div style={{ textAlign: "center", fontSize: 12, color: "var(--t3)", marginTop: 10 }}>
              DeepFace is analysing the photo...
            </div>
          )}
        </div>
      </div>

    </div>
  );
}

