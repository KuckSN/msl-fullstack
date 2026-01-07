import React from "react";
import { NavLink } from "react-router-dom";

const navItems = [
  { to: "/", label: "Overview" },
  { to: "/upload", label: "Upload & Translate" },
  { to: "/camera", label: "Real-time Camera" },
  { to: "/learn", label: "Learn" },
];

function Navbar() {
  return (
    <nav style={styles.nav}>
      {navItems.map((item) => (
        <NavLink
          key={item.to}
          to={item.to}
          className="top-nav-row"
          style={({ isActive }) => ({
            ...styles.link,
            ...(isActive ? styles.linkActive : {}),
          })}
          end={item.to === "/"}
        >
          {item.label}
        </NavLink>
      ))}
    </nav>
  );
}

const styles = {
  nav: {
    display: "flex",
    gap: "8px",
    padding: "8px 4px 10px",
    marginBottom: "8px",
    borderRadius: "999px",
    background: "rgba(255,255,255,0.9)",
    boxShadow: "0 4px 14px rgba(148,163,184,0.35)",
    border: "1px solid #e5e7eb",
  },
  link: {
    padding: "6px 14px",
    borderRadius: "999px",
    fontSize: "0.85rem",
    textDecoration: "none",
    color: "#4b5563",
    fontWeight: 500,
  },
  linkActive: {
    background:
      "linear-gradient(120deg, #7c3aed 0%, #a855f7 40%, #facc15 100%)",
    color: "#111827",
    boxShadow: "0 8px 18px rgba(129,140,248,0.55)",
  },
};

export default Navbar;
