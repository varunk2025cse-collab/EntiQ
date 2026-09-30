import { h, icon, brandMark, clear, fmt, toast } from "./dom.js";
import { get, state } from "./api.js";

const NAV = [
  ["Overview", [["dashboard", "Dashboard", "home"], ["datasets", "Datasets", "database"], ["runs", "Resolution Runs", "play"]]],
  ["Investigate", [["entities", "Entity Explorer", "search"], ["match", "Match Investigation", "link"], ["candidates", "Candidate Explorer", "nodes"]]],
  ["Evaluate", [["analytics", "Analytics", "chart"], ["models", "Model Laboratory", "flask"], ["experiments", "Experiments", "beaker"], ["errors", "Error Analysis", "alert"]]],
  ["Operate", [["export", "Export Center", "download"], ["health", "System Health", "pulse"]]],
];
const PAGES = {
  dashboard: () => import("./pages/dashboard.js"), datasets: () => import("./pages/datasets.js"), runs: () => import("./pages/runs.js"),
  entities: () => import("./pages/entities.js"), entity: () => import("./pages/entity.js"), match: () => import("./pages/match.js"),
  candidates: () => import("./pages/candidates.js"), analytics: () => import("./pages/analytics.js"), models: () => import("./pages/models.js"),
  experiments: () => import("./pages/experiments.js"), errors: () => import("./pages/errors.js"), export: () => import("./pages/export.js"),
  health: () => import("./pages/health.js"),
};

function shell() {
  document.documentElement.dataset.theme = "light";
  const nav = h("nav", { "aria-label": "Main" }, NAV.map(([group, items]) => h("div", {},
    h("div", { class: "nav-group" }, group),
    h("div", { class: "nav" }, items.map(([id, label, ic]) => h("a", { href: `#/${id}`, "data-route": id }, icon(ic), label))))));
  const sidebar = h("aside", { class: "sidebar", id: "sidebar" },
    h("div", { class: "brand" }, brandMark(), h("div", {}, h("div", { class: "brand-name" }, "ENTIQ"), h("div", { class: "brand-sub" }, "Business Identity Intelligence"))),
    nav,
    h("div", { class: "sidebar-foot", id: "sidebar-foot" }, h("b", {}, "Entity resolution"), "Source 1 reference entities linked to Source 2 / Source 3 records, scored for precision-first F0.5."));
  const search = h("input", { type: "search", placeholder: "Search entity ID, business name or address…", "aria-label": "Search entities", id: "global-search" });
  search.addEventListener("keydown", (e) => {
    if (e.key !== "Enter") return;
    const q = search.value.trim();
    if (!q) return;
    location.hash = /^S[123]-\S+$/i.test(q) ? `#/entity/${encodeURIComponent(q.toUpperCase())}` : `#/entities?q=${encodeURIComponent(q)}`;
  });
  document.addEventListener("keydown", (e) => { if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") { e.preventDefault(); search.focus(); } });
  const statusPill = h("div", { class: "status-pill", id: "status-pill" }, h("span", { class: "dot warn" }), h("div", {}, "Connecting…"));
  const topbar = h("header", { class: "topbar" },
    h("button", { class: "icon-btn menu-toggle", type: "button", "aria-label": "Open menu", onclick: () => sidebar.classList.toggle("open") }, icon("menu")),
    h("div", { class: "search" }, icon("search"), search, h("kbd", {}, "Ctrl K")),
    h("div", { class: "spacer" }), h("span", { id: "data-kind-chip" }), statusPill);
  const content = h("main", { class: "content", id: "content", tabindex: "-1" });
  const footer = h("div", { class: "footer-bar", id: "footer-bar" });
  document.getElementById("app").append(h("div", { class: "app" }, sidebar, h("div", { class: "main" }, topbar, content, footer)));
}

export const KIND = {
  competition: { label: "Real competition data", badge: "real", icon: "shield" },
  synthetic: { label: "Demo / synthetic data", badge: "demo", icon: "alert" },
  custom: { label: "Custom dataset", badge: "run", icon: "database" },
  unknown: { label: "Unknown data source", badge: "nomatch", icon: "info" },
};

export function kindBadge(kind) {
  const k = KIND[kind] || KIND.unknown;
  return h("span", { class: `badge ${k.badge}` }, icon(k.icon), k.label);
}

export async function refreshHealth() {
  try {
    const hl = await get("/health");
    state.health = hl;
    const pill = document.getElementById("status-pill");
    clear(pill).append(h("span", { class: `dot ${hl.status === "ok" ? "" : "bad"}` }),
      h("div", {}, hl.status === "ok" ? "System online" : "System degraded", h("small", {}, hl.production_model ? `Model ${hl.production_model}` : "No model trained")));
    clear(document.getElementById("data-kind-chip")).append(hl.data_kind ? kindBadge(hl.data_kind) : h("span"));
    if (!state.runId) state.runId = hl.latest_run;
    footer(hl);
  } catch (e) {
    const pill = document.getElementById("status-pill");
    clear(pill).append(h("span", { class: "dot bad" }), h("div", {}, "API unreachable", h("small", {}, String(e.message || e))));
  }
}

function footer(hl) {
  clear(document.getElementById("footer-bar")).append(
    hl.latest_run ? h("span", { class: "ok" }, icon("check"), `Last run ${hl.latest_run}`) : h("span", {}, "No resolution run yet"),
    h("span", {}, `Model: ${hl.production_model || "—"}`), h("span", {}, `Database: ${hl.database_backend}`),
    h("span", { style: "margin-left:auto" }, `API uptime ${fmt.secs(hl.uptime_s)}`));
}

/** Banner stating whose data the page shows. Derived from run metadata (never hard-coded). */
export function identityBanner(kind) {
  if (!kind) return null;
  if (kind === "synthetic") return h("div", { class: "identity demo", role: "note" }, icon("alert"),
    h("span", {}, h("b", {}, "Demo / synthetic data. "), "Every number here comes from a real pipeline run on generated development data — not the competition dataset, and not a competition result."));
  if (kind === "competition") return h("div", { class: "identity real", role: "note" }, icon("shield"),
    h("span", {}, h("b", {}, "Real competition data. "), "Figures come from runs on the official challenge files. Test-set runs have no labels; quality metrics come from held-out training entities."));
  return h("div", { class: "identity none", role: "note" }, icon("info"), h("span", {}, h("b", {}, `${(KIND[kind] || KIND.unknown).label}. `), "Not the official competition files."));
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
  const t = setTimeout(() => { if (seq === routeSeq && !content.firstChild) content.append(h("div", { class: "loading" }, h("span", { class: "spinner" }), "Loading…")); }, 150);
  content.classList.add("skeleton");
  try {
    const mod = await (PAGES[page] || PAGES.dashboard)();
    const view = h("div", { style: "display:contents" });
    await mod.render(view, { arg, params });
    if (seq !== routeSeq) return;
    clear(content).append(view);
    window.scrollTo(0, 0);
  } catch (e) {
    if (seq !== routeSeq) return;
    clear(content).append(h("div", { class: "card card-pad" }, h("div", { class: "callout err" }, icon("alert"),
      h("div", {}, h("b", {}, "This page could not be loaded. "), String(e.message || e)))));
    toast(String(e.message || e), true);
  } finally {
    clearTimeout(t);
    content.classList.remove("skeleton");
  }
}

shell();
await refreshHealth();
window.addEventListener("hashchange", route);
route();
setInterval(refreshHealth, 30000);
