import { h, icon, brandMark, clear, fmt, toast } from "./dom.js";
import { get, state } from "./api.js";

const NAV = [
  ["dashboard", "Dashboard", "home"], ["datasets", "Datasets", "database"], ["runs", "Resolution Runs", "play"],
  ["entities", "Entity Explorer", "search"], ["match", "Match Investigation", "link"], ["candidates", "Candidate Explorer", "nodes"],
  ["analytics", "Analytics", "chart"], ["models", "Model Laboratory", "flask"], ["experiments", "Experiments", "beaker"],
  ["errors", "Error Analysis", "alert"], ["export", "Export Center", "download"], ["health", "System Health", "pulse"],
];
const PAGES = {
  dashboard: () => import("./pages/dashboard.js"), datasets: () => import("./pages/datasets.js"), runs: () => import("./pages/runs.js"),
  entities: () => import("./pages/entities.js"), entity: () => import("./pages/entity.js"), match: () => import("./pages/match.js"),
  candidates: () => import("./pages/candidates.js"), analytics: () => import("./pages/analytics.js"), models: () => import("./pages/models.js"),
  experiments: () => import("./pages/experiments.js"), errors: () => import("./pages/errors.js"), export: () => import("./pages/export.js"),
  health: () => import("./pages/health.js"),
};

const store = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch { /* storage unavailable */ } },
};

function applyTheme(t) { document.documentElement.dataset.theme = t; store.set("entiq-theme", t); }

function shell() {
  applyTheme(store.get("entiq-theme") || "dark");
  const nav = h("nav", { class: "nav", "aria-label": "Main" },
    NAV.map(([id, label, ic]) => h("a", { href: `#/${id}`, "data-route": id }, icon(ic), label)));
  const sidebar = h("aside", { class: "sidebar", id: "sidebar" },
    h("div", { class: "brand" }, brandMark(), h("div", {}, h("div", { class: "brand-name" }, "Business Identity", h("br"), "Intelligence"),
      h("div", { class: "brand-sub" }, "AI-Powered Entity Resolution Platform"))),
    nav,
    h("div", { class: "sidebar-foot" }, h("div", { class: "tag" }, "Smarter Data.", h("br"), "Stronger Connections.", h("br"), "Real World Impact."),
      h("div", { class: "ver" }, "EntiQ · Business Identity Intelligence v2.0")));
  const search = h("input", { type: "search", placeholder: "Search by business name, address, ID or keyword…", "aria-label": "Search entities", id: "global-search" });
  search.addEventListener("keydown", (e) => {
    if (e.key !== "Enter") return;
    const q = search.value.trim();
    if (!q) return;
    location.hash = /^S[123]-\S+$/i.test(q) ? `#/entity/${encodeURIComponent(q.toUpperCase())}` : `#/entities?q=${encodeURIComponent(q)}`;
  });
  document.addEventListener("keydown", (e) => { if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") { e.preventDefault(); search.focus(); } });
  const themeBtn = h("button", { class: "icon-btn", type: "button", "aria-label": "Toggle colour theme", onclick: () => {
    applyTheme(document.documentElement.dataset.theme === "light" ? "dark" : "light"); route(); } },
  icon("sun"));
  const statusPill = h("div", { class: "status-pill", id: "status-pill" }, h("span", { class: "dot warn" }), h("div", {}, "Connecting…", h("small", {}, "")));
  const topbar = h("header", { class: "topbar" },
    h("button", { class: "icon-btn menu-toggle", type: "button", "aria-label": "Open menu", onclick: () => sidebar.classList.toggle("open") }, icon("menu")),
    h("div", { class: "search" }, icon("search"), search, h("kbd", {}, "Ctrl K")),
    h("div", { class: "spacer" }), statusPill, themeBtn,
    h("div", { class: "who" }, h("div", { class: "avatar" }, "EQ"), h("div", {}, h("b", {}, "EntiQ Workspace"), h("small", { id: "ws-name" }, "—"))));
  const content = h("main", { class: "content", id: "content", tabindex: "-1" });
  const footer = h("div", { class: "footer-bar", id: "footer-bar" });
  document.getElementById("app").append(h("div", { class: "app" }, sidebar, h("div", { class: "main" }, topbar, content, footer)));
}

export async function refreshHealth() {
  try {
    const hl = await get("/health");
    state.health = hl;
    const pill = document.getElementById("status-pill");
    clear(pill).append(h("span", { class: `dot ${hl.status === "ok" ? "" : "bad"}` }),
      h("div", {}, hl.status === "ok" ? "System Online" : "System Degraded", h("small", {}, `Model ${hl.production_model || "not trained"} · ${hl.latest_run || "no runs yet"}`)));
    document.getElementById("ws-name").textContent = hl.demo_mode ? "Demo workspace (synthetic)" : hl.workspace === "default" ? "Competition workspace" : "Custom workspace";
    if (!state.runId) state.runId = hl.latest_run;
    footer(hl);
  } catch (e) {
    const pill = document.getElementById("status-pill");
    clear(pill).append(h("span", { class: "dot bad" }), h("div", {}, "API unreachable", h("small", {}, String(e.message || e))));
  }
}

function footer(hl) {
  const f = document.getElementById("footer-bar");
  clear(f).append(
    hl.latest_run ? h("span", { class: "ok" }, icon("check"), "Processing completed successfully") : h("span", {}, "No resolution run yet"),
    h("span", {}, `Database: ${hl.database_backend}`), h("span", {}, `Model: ${hl.production_model || "—"}`),
    h("span", {}, `Run ID: ${state.runId || "—"}`), h("span", { style: "margin-left:auto" }, `Uptime ${fmt.secs(hl.uptime_s)}`));
}

export function demoBanner() {
  if (!state.health?.demo_mode) return null;
  return h("div", { class: "demo-banner" }, icon("alert"),
    "Demo workspace: every number on this page comes from a real pipeline run on SYNTHETIC development data, not the competition dataset.");
}

export function parseHash() {
  const raw = location.hash.replace(/^#\/?/, "") || "dashboard";
  const [path, qs] = raw.split("?");
  const parts = path.split("/").map(decodeURIComponent);
  return { page: parts[0] || "dashboard", arg: parts[1], params: new URLSearchParams(qs || "") };
}

let routeSeq = 0;
export async function route() {
  const { page, arg, params } = parseHash();
  const seq = ++routeSeq;
  const navKey = page === "entity" ? "entities" : page;
  document.querySelectorAll(".nav a").forEach((a) => a.classList.toggle("active", a.dataset.route === navKey));
  document.getElementById("sidebar").classList.remove("open");
  const content = document.getElementById("content");
  content.classList.add("skeleton");
  try {
    const mod = await (PAGES[page] || PAGES.dashboard)();
    const view = h("div", { style: "display:contents" });
    await mod.render(view, { arg, params });
    if (seq !== routeSeq) return;
    clear(content).append(...[demoBanner(), view].filter(Boolean));
  } catch (e) {
    if (seq !== routeSeq) return;
    clear(content).append(h("div", { class: "card card-pad" }, h("h3", {}, "Something went wrong"), h("pre", { class: "pre" }, String(e.message || e))));
    toast(String(e.message || e), true);
  } finally {
    content.classList.remove("skeleton");
  }
}

shell();
await refreshHealth();
window.addEventListener("hashchange", route);
route();
setInterval(refreshHealth, 30000);
