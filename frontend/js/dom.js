// Safe DOM construction: all data goes through textContent / attributes, never innerHTML.
const SVGNS = "http://www.w3.org/2000/svg";

export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  setAttrs(el, attrs);
  append(el, children);
  return el;
}

export function s(tag, attrs = {}, ...children) {
  const el = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v);
  }
  append(el, children);
  return el;
}

function setAttrs(el, attrs) {
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "style" && typeof v === "object") Object.assign(el.style, v);
    else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
    else if (k === "dataset") Object.assign(el.dataset, v);
    else el.setAttribute(k, v === true ? "" : v);
  }
}

export function append(el, children) {
  for (const c of children.flat(Infinity)) {
    if (c === null || c === undefined || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

export function clear(el) { while (el.firstChild) el.removeChild(el.firstChild); return el; }

// --------------------------------------------------------------------------- icons (24px stroke set)
const P = {
  home: ["M3 10.5 12 3l9 7.5", "M5 9.5V21h14V9.5", "M9.5 21v-6h5v6"],
  database: ["M4 6c0-1.7 3.6-3 8-3s8 1.3 8 3-3.6 3-8 3-8-1.3-8-3Z", "M4 6v6c0 1.7 3.6 3 8 3s8-1.3 8-3V6", "M4 12v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"],
  play: ["M7 4.5v15l12-7.5Z"],
  search: ["M11 18a7 7 0 1 0 0-14 7 7 0 0 0 0 14Z", "m20 20-4-4"],
  link: ["M10 14a4.5 4.5 0 0 0 6.4 0l3-3a4.5 4.5 0 0 0-6.4-6.4l-1 1", "M14 10a4.5 4.5 0 0 0-6.4 0l-3 3a4.5 4.5 0 0 0 6.4 6.4l1-1"],
  nodes: ["M6 8a2.5 2.5 0 1 0 0-5 2.5 2.5 0 0 0 0 5Z", "M18 21a2.5 2.5 0 1 0 0-5 2.5 2.5 0 0 0 0 5Z", "M6 21a2.5 2.5 0 1 0 0-5 2.5 2.5 0 0 0 0 5Z", "M6 8v8", "M8.3 6.9 16 17"],
  chart: ["M4 20V10", "M10 20V4", "M16 20v-7", "M22 20H2"],
  flask: ["M9 3h6", "M10 3v6L4.5 18.5A1.7 1.7 0 0 0 6 21h12a1.7 1.7 0 0 0 1.5-2.5L14 9V3", "M7 15h10"],
  beaker: ["M5 3h14", "M7 3v14a4 4 0 0 0 4 4h2a4 4 0 0 0 4-4V3", "M7 11h10"],
  alert: ["M12 3 2 20h20Z", "M12 10v4", "M12 17.5v.5"],
  download: ["M12 4v11", "m7 10 5 5 5-5", "M4 20h16"],
  upload: ["M12 20V9", "m7 14 5-5 5 5", "M4 4h16"],
  pulse: ["M2 12h4l3-8 6 16 3-8h4"],
  bell: ["M6 16V11a6 6 0 1 1 12 0v5l2 2H4Z", "M10 21h4"],
  check: ["m5 12.5 4.5 4.5L19 7.5"],
  x: ["M6 6l12 12", "M18 6 6 18"],
  pin: ["M12 21s-7-6.2-7-11.5A7 7 0 0 1 19 9.5C19 14.8 12 21 12 21Z", "M12 12a2.5 2.5 0 1 0 0-5 2.5 2.5 0 0 0 0 5Z"],
  building: ["M4 21V5l8-2v18", "M12 8h8v13", "M2 21h20", "M7.5 8h1", "M7.5 12h1", "M7.5 16h1", "M15.5 12h1", "M15.5 16h1"],
  shield: ["M12 3 4.5 6v6c0 4.6 3.2 8 7.5 9 4.3-1 7.5-4.4 7.5-9V6Z", "m8.5 12 2.5 2.5 4.5-5"],
  user: ["M12 12a4 4 0 1 0 0-8 4 4 0 0 0 0 8Z", "M4 21a8 8 0 0 1 16 0"],
  target: ["M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18Z", "M12 16a4 4 0 1 0 0-8 4 4 0 0 0 0 8Z", "M12 12h.01"],
  zap: ["M13 2 4 14h7l-1 8 9-12h-7Z"],
  refresh: ["M20 11a8 8 0 0 0-14.7-4.4L4 8", "M4 3v5h5", "M4 13a8 8 0 0 0 14.7 4.4L20 16", "M20 21v-5h-5"],
  sun: ["M12 17a5 5 0 1 0 0-10 5 5 0 0 0 0 10Z", "M12 1.5v2", "M12 20.5v2", "M4.2 4.2l1.4 1.4", "M18.4 18.4l1.4 1.4", "M1.5 12h2", "M20.5 12h2", "M4.2 19.8l1.4-1.4", "M18.4 5.6l1.4-1.4"],
  moon: ["M20 14.5A8.5 8.5 0 1 1 9.5 4a7 7 0 0 0 10.5 10.5Z"],
  menu: ["M4 6h16", "M4 12h16", "M4 18h16"],
  file: ["M14 3H6v18h12V7Z", "M14 3v4h4", "M9 13h6", "M9 17h6"],
  cpu: ["M7 7h10v10H7Z", "M10 10h4v4h-4Z", "M9 2v3", "M15 2v3", "M9 19v3", "M15 19v3", "M2 9h3", "M2 15h3", "M19 9h3", "M19 15h3"],
  info: ["M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18Z", "M12 11v5", "M12 7.5v.5"],
  layers: ["m12 3 9 5-9 5-9-5Z", "m3 13 9 5 9-5"],
  arrowUp: ["M12 19V5", "m6 11 6-6 6 6"],
  arrowDown: ["M12 5v14", "m6 13 6 6 6-6"],
  arrowRight: ["M5 12h14", "m13 6 6 6-6 6"],
  arrowLeft: ["M19 12H5", "m11 6-6 6 6 6"],
  sparkle: ["M12 3v4", "M12 17v4", "M3 12h4", "M17 12h4", "m6 6 2.5 2.5", "m15.5 15.5 2.5 2.5", "m6 18 2.5-2.5", "m15.5 8.5 2.5-2.5"],
  compare: ["M8 3v18", "M16 3v18", "M3 8h5", "M16 16h5"],
  clock: ["M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18Z", "M12 7v5l3 2"],
  git: ["M6 3v12", "M18 9a3 3 0 1 0 0-6 3 3 0 0 0 0 6Z", "M6 21a3 3 0 1 0 0-6 3 3 0 0 0 0 6Z", "M18 9a9 9 0 0 1-9 9"],
  filter: ["M3 5h18l-7 8v6l-4 2v-8Z"],
  table: ["M3 5h18v14H3Z", "M3 10h18", "M3 15h18", "M9 5v14"],
  calendar: ["M4 6h16v15H4Z", "M4 10h16", "M8 3v4", "M16 3v4"],
};

export function icon(name, cls = "") {
  const el = s("svg", { viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", "stroke-width": "1.8",
    "stroke-linecap": "round", "stroke-linejoin": "round", "aria-hidden": "true", class: cls });
  for (const d of P[name] || P.info) el.append(s("path", { d }));
  return el;
}

export function brandMark() {
  // Two interlocking rings = two records resolving into one identity.
  const el = s("svg", { viewBox: "0 0 40 40", class: "brand-mark", "aria-hidden": "true" });
  const g = s("defs", {}, s("linearGradient", { id: "bm", x1: "0", y1: "0", x2: "1", y2: "1" },
    s("stop", { offset: "0", "stop-color": "#4f46e5" }), s("stop", { offset: "1", "stop-color": "#6366f1" })));
  el.append(g,
    s("rect", { x: "1", y: "1", width: "38", height: "38", rx: "11", fill: "url(#bm)" }),
    s("circle", { cx: "16", cy: "20", r: "7.5", fill: "none", stroke: "#fff", "stroke-width": "2.6" }),
    s("circle", { cx: "24", cy: "20", r: "7.5", fill: "none", stroke: "#fff", "stroke-opacity": ".75", "stroke-width": "2.6" }));
  return el;
}

// --------------------------------------------------------------------------- formatting
export const fmt = {
  int: (v) => (v === null || v === undefined || Number.isNaN(v)) ? "—" : Number(v).toLocaleString("en-US"),
  pct: (v, d = 1) => (v === null || v === undefined || Number.isNaN(v)) ? "—" : `${(Number(v) * 100).toFixed(d)}%`,
  dec: (v, d = 3) => (v === null || v === undefined || Number.isNaN(v)) ? "—" : Number(v).toFixed(d),
  secs: (v) => v === null || v === undefined ? "—" : v < 60 ? `${Number(v).toFixed(1)}s` : `${Math.floor(v / 60)}m ${Math.round(v % 60)}s`,
  date: (iso) => iso ? new Date(iso).toLocaleString("en-GB", { day: "2-digit", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit" }) : "—",
  compact: (v) => v === null || v === undefined ? "—" : Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 }).format(v),
};

export function badge(decision) {
  if (decision === "MATCH") return h("span", { class: "badge match" }, icon("check"), "Match");
  if (decision === "NO_MATCH" || decision === "NO MATCH") return h("span", { class: "badge nomatch" }, icon("x"), "No match");
  if (decision === "REVIEW") return h("span", { class: "badge review" }, icon("alert"), "Review");
  if (decision === "REJECTED") return h("span", { class: "badge nomatch" }, icon("x"), "No match");
  if (decision === "CANDIDATE") return h("span", { class: "badge cand" }, icon("nodes"), "Candidate");
  return h("span", { class: "badge nomatch" }, decision);
}

export function statusBadge(st) {
  const map = { COMPLETED: ["match", "check", "Completed"], RUNNING: ["run", "refresh", "Running"], QUEUED: ["run", "clock", "Queued"],
    FAILED: ["fail", "x", "Failed"], PASS: ["match", "check", "Pass"], FAIL: ["fail", "x", "Fail"], SKIPPED: ["nomatch", "info", "Skipped"] };
  const [c, i, l] = map[st] || ["nomatch", "info", st];
  return h("span", { class: `badge ${c}` }, icon(i), l);
}

export function toast(msg, err = false) {
  const t = h("div", { class: `toast${err ? " err" : ""}`, role: "status" }, msg);
  document.body.append(t);
  setTimeout(() => t.remove(), 4500);
}

export function empty(title, text, code, ic = "info") {
  return h("div", { class: "empty" }, h("div", { class: "ic" }, icon(ic)), h("h3", {}, title), text ? h("p", {}, text) : null, code ? h("pre", {}, code) : null);
}
