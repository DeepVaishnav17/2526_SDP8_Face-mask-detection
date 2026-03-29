import React from "react";
import { BrowserRouter, Routes, Route, Navigate, NavLink } from "react-router-dom";
import Register   from "./pages/Register";
import Detection  from "./pages/Detection";
import Logs       from "./pages/Logs";
import "./App.css";

function Nav() {
  const link = ({ isActive }) => ({
    display:"flex", alignItems:"center", gap:6,
    padding:"7px 16px", borderRadius:8,
    textDecoration:"none", fontWeight:600, fontSize:13,
    transition:"all .2s",
    background: isActive ? "rgba(16,185,129,.15)" : "transparent",
    color: isActive ? "#10B981" : "#94A3B8",
    border: isActive ? "1px solid rgba(16,185,129,.3)" : "1px solid transparent",
  });
  return (
    <nav className="navbar">
      <div className="nav-brand">
        <span className="nav-logo">🛡️</span>
        <span>Face<b style={{color:"#10B981"}}>Guard</b></span>
      </div>
      <div className="nav-links">
        <NavLink to="/register"  style={link}>📋 Register</NavLink>
        <NavLink to="/detection" style={link}>📷 Detection</NavLink>
        <NavLink to="/admin"     style={link}>⚙️ Admin</NavLink>
      </div>
      <div className="nav-status">
        <span className="dot-green"/>
        <span style={{fontSize:11,fontFamily:"monospace",color:"#10B981"}}>LIVE</span>
      </div>
    </nav>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <Nav />
      <main className="main">
        <Routes>
          <Route path="/" element={<Navigate to="/register" />} />
          <Route path="/register"  element={<Register />} />
          <Route path="/detection" element={<Detection />} />
          <Route path="/admin"     element={<Logs />} />
        </Routes>
      </main>
    </BrowserRouter>
  );
}
