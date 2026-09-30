import { h, icon, clear, fmt, badge, empty, toast } from "../dom.js";
import { get, post, state } from "../api.js";
import { barChart, donut, hbarChart, dotPlot, meters, withTable, legend, SEQ, SERIES } from "../charts.js";
import { identityBanner, kindBadge } from "../app.js";
import { card, pageHead, kpi, prettyModel, reasonLabel, validationMetrics, isModelExperiment } from "../ui.js";

export { card, prettyModel, reasonLabel, validationMetrics };

export async function render(root) {
  const runs = await get("/resolution/runs").catch(() => []);
  const completed = runs.filter((r) => r.status === "COMPLETED");
  if (!completed.length) {
    root.append(pageHead("ENTIQ · Business Identity Intelligence", "Dashboard", "Resolution results and model quality appear here after the first run."), onboarding());
    return;
  }
  const runId = completed.some((r) => r.id === state.runId) ? state.runId : completed[0].id;
  state.runId = runId;
  const [an, models, exps, ents] = await Promise.all([
    get(`/analytics/${runId}`), get("/models").catch(() => null), get("/experiments").catch(() => null),
    get("/entities", { run_id: runId, size: 8, sort: "-best" })]);
  const s = an.summary;
  const kind = s.dataset_kind || "unknown";
  const prod = models?.versions?.find((v) => v.version === s.model_version) || models?.versions?.[0];
  const val = validationMetrics(prod, exps);
  const c = s.counts;
  const b = an.blocking || {};

  root.append(identityBanner(kind), header(completed, runId, kind));

  // ---- resolution outputs ------------------------------------------------------
  const t = an.timings || {};
  const compute = (t.total_seconds || 0) + (t.run?.total_seconds || 0);
  root.append(h("div", { class: "kpis" },
    kpi({ icon: "database", label: "Source 1 entities", value: fmt.int(c.s1), foot: `${fmt.int(c.s2)} S2 · ${fmt.int(c.s3)} S3 records` }),
    kpi({ icon: "nodes", label: "Candidate pairs scored", value: fmt.int(c.candidates),
      foot: b.pre_pruning ? `${fmt.int(b.pre_pruning.candidates)} retrieved by blocking` : `${fmt.dec(b.union?.avg_candidates_per_s1, 1)} per entity` }),
    kpi({ icon: "link", tone: "green", label: "Final matches", value: fmt.int(c.links), foot: `${fmt.int(c.matched_entities)} entities linked` }),
    kpi({ icon: "user", tone: "slate", label: "No-match entities", value: fmt.int(c.singletons), foot: `${fmt.pct(c.singletons / Math.max(1, c.s1))} of Source 1` }),
    kpi({ icon: "clock", tone: "slate", label: "Pipeline compute", value: fmt.secs(compute), foot: `${fmt.int(Math.round(c.s1 / Math.max(compute, 1e-6)))} entities / s` })));

  // ---- quality (never a synthetic score presented as a competition result) --------
  root.append(qualityRow(kind, val));
  root.append(flowCard(c, b, s, an));

  // ---- charts ------------------------------------------------------------------
  const bins = an.confidence_histogram.bins;
  const lpe = an.links_per_entity || {};
  const noCand = b.union?.s1_without_candidates || 0;
  const single = Number(lpe["1"] || 0);
  const multi = Object.entries(lpe).filter(([k]) => Number(k) >= 2).reduce((a, [, v]) => a + v, 0);
  const noMatch = Math.max(0, c.singletons - noCand);
  const countries = Object.entries(an.country_distribution || {}).sort((x, y) => y[1] - x[1]);
  const totalC = countries.reduce((a, [, v]) => a + v, 0) || 1;
  root.append(h("div", { class: "grid g-3" },
    card("Decision outcomes", "per Source 1 entity", donut([
      { label: "Matched · 1 record", value: single, color: SERIES[0] }, { label: "Matched · 2+ records", value: multi, color: SERIES[6] },
      { label: "No match (rejected)", value: noMatch, color: "var(--text-muted)" }, { label: "No candidates", value: noCand, color: "var(--border-strong)" }],
    { center: fmt.int(c.s1), centerLabel: "entities" })),
    card("Confidence of accepted links", "calibrated match probability", withTable(barChart({ labels: bins.map((x) => x.split("-")[0]), values: an.confidence_histogram.counts, colors: SEQ,
      tipLabel: (i) => `links with confidence ${bins[i]}` }), ["Confidence", "Links"], bins.map((x, i) => [x, fmt.int(an.confidence_histogram.counts[i])]))),
    card("Countries", "share of entities · match rate", withTable(hbarChart(countries.slice(0, 6).map(([k, v], i) => ({ label: k, value: v / totalC, color: SERIES[i % 8],
      sub: `${fmt.int(v)} entities · ${fmt.pct(an.country_match_rate?.[k])} matched` })), { format: (v) => fmt.pct(v), max: 1 }),
    ["Country", "Entities", "Share", "Match rate"], countries.map(([k, v]) => [k, fmt.int(v), fmt.pct(v / totalC), fmt.pct(an.country_match_rate?.[k])])))));

  root.append(h("div", { class: "grid g-2" }, modelComparison(exps, kind), pipelineCard(an, compute)));
  root.append(recentEntities(runId, ents));
}

function header(completed, runId, kind) {
  const sel = h("select", { class: "select", "aria-label": "Resolution run", onchange: (e) => { state.runId = e.target.value; window.dispatchEvent(new HashChangeEvent("hashchange")); } },
    completed.map((r, i) => h("option", { value: r.id, selected: r.id === runId }, `${r.id} · ${r.split}${i === 0 ? " · latest" : ""}`)));
  const btn = h("button", { class: "btn primary", type: "button", onclick: async () => {
    btn.disabled = true;
    try {
      const ds = await get("/datasets");
      const d = ds.find((x) => x.data_kind === "competition" && x.splits.includes("test")) || ds.find((x) => x.splits.includes("test")) || ds[0];
      if (!d) throw new Error("No dataset available");
      const r = await post("/resolution/run", { dataset_id: d.id, split: d.splits.includes("test") ? "test" : d.splits[0] });
      toast(`Started ${r.run_id} on ${d.name}`);
      location.hash = "#/runs";
    } catch (e) { toast(e.message, true); } finally { btn.disabled = false; }
  } }, icon("play"), "Run resolution");
  const run = completed.find((r) => r.id === runId);
  return pageHead("ENTIQ · Business Identity Intelligence", "Dashboard",
    `What run ${runId} resolved, how the cascade got there, and how reliable the model is.`,
    [h("span", { class: "chip" }, icon("calendar"), fmt.date(run?.finished || run?.created)), sel, btn]);
}

function qualityRow(kind, val) {
  let na = null;
  let foot = "held-out training entities (never used for fitting)";
  if (val.f05 === null) na = kind === "competition" ? "Not evaluated on real competition data" : "Not evaluated";
  else if (kind === "competition" && val.kind !== "competition") na = "Not evaluated on real competition data";
  else if (val.kind === "synthetic") foot = "DEMO validation on synthetic data";
  const v = (x, f) => (na ? null : f(x));
  return h("div", { class: "kpis" },
    kpi({ icon: "target", label: "Validation F0.5", value: v(val.f05, (x) => fmt.dec(x, 4)), na, foot: na ? "official metric: macro F0.5 per S1 entity" : foot }),
    kpi({ icon: "shield", label: "Precision (macro)", value: v(val.precision, (x) => fmt.pct(x, 2)), na, foot: na ? "" : "wrong links are penalised 2× in F0.5" }),
    kpi({ icon: "search", label: "Recall (macro)", value: v(val.recall, (x) => fmt.pct(x, 2)), na }),
    kpi({ icon: "nodes", label: "Candidate recall", value: v(val.candidate_recall, (x) => fmt.pct(x, 2)), na, foot: na ? "" : "true links retrieved before matching" }),
    kpi({ icon: "user", label: "No-match accuracy", value: v(val.singleton, (x) => fmt.pct(x, 2)), na, foot: na ? "" : "true singletons left unmatched" }));
}

function flowCard(c, b, s, an) {
  const steps = [
    ["1", "Records", "normalised & indexed", fmt.int(c.s1 + c.s2 + c.s3)],
    ["2", "Blocking", "9 retrieval passes, unioned", fmt.int(b.pre_pruning?.candidates ?? b.union?.candidates)],
    ["3", "Cascade filter", "cheap model drops hopeless pairs", fmt.int(c.candidates)],
    ["4", "Matching", prettyModel(s.model_name), `${fmt.int(c.candidates)} scored`],
    ["5", "Decision", `threshold ${fmt.dec(s.decision?.threshold, 2)} · no-match gate`, `${fmt.int(c.links)} links`],
    ["6", "Output", "one row per S1 entity", `${fmt.int(c.matched_entities)} / ${fmt.int(c.singletons)}`],
  ];
  return card("How this run was resolved", "each stage narrows the search; nothing is brute-forced", h("div", { class: "flow" },
    steps.map(([n, t, d, v]) => h("div", {}, h("span", { class: "n" }, n), h("b", {}, t), h("span", {}, d), h("div", { class: "v" }, v)))),
  !s.validation ? null : s.validation.status === "PASS" ? h("span", { class: "badge match" }, icon("check"), "Submission valid")
    : h("span", { class: "badge fail" }, icon("x"), `Submission ${String(s.validation.status).toLowerCase()}`));
}

function onboarding() {
  return h("div", { class: "card" }, empty("No resolution run yet",
    "Train a model on the training split, then resolve the test split. Every figure on this dashboard is computed from real runs.",
    "python -m src.cli all          # profile → blocking → experiments → train → predict\npython -m backend.app.main       # then reload this page", "play"));
}

function modelComparison(exps, kind) {
  const res = (exps?.results || []).filter(isModelExperiment).filter((r) => !kind || !r.dataset_kind || r.dataset_kind === kind || kind === "unknown");
  const latest = new Map();
  for (const r of res) latest.set(r.name, r);
  const selected = exps?.selected?.overrides?.model?.primary;
  const rows = [...latest.values()].sort((a, b) => a.cv_f05 - b.cv_f05).map((r) => ({ label: prettyModel(r.name), value: r.cv_f05, name: r.name, sub: `holdout ${fmt.dec(r.holdout_f05, 4)}` }));
  const body = rows.length ? withTable(dotPlot(rows, { highlight: (r) => r.name === selected }), ["Model", "CV F0.5", "Holdout F0.5"],
    [...latest.values()].map((r) => [prettyModel(r.name), fmt.dec(r.cv_f05, 4), fmt.dec(r.holdout_f05, 4)]),
    legend([["Model family", SERIES[0]], ["Selected by validation", SERIES[6]]]))
    : empty("No model experiments for this data source", "Run the experiment suite to compare model families.", "python -m src.cli experiment", "flask");
  return card("Model comparison", "cross-validated F0.5 of each model family (A/B checks and tuning trials excluded)", body,
    selected ? h("span", { class: "chip blue" }, `Selected: ${prettyModel(selected)}`) : null);
}

function pipelineCard(an, compute) {
  const t = an.timings || {};
  const st = t.stages || {};
  const run = t.run?.stages || {};
  const rows = [["Normalisation & indexing", st.normalization_and_indexing?.seconds], ["Blocking", st.blocking?.seconds], ["Cascade filter", st.pruning?.seconds],
    ["Pair features", st.features?.seconds], ["Model scoring", run.inference?.seconds], ["Decision & output", (run.decision?.seconds || 0) + (run.outputs?.seconds || 0)]]
    .filter(([, v]) => v !== undefined && v !== null);
  const max = Math.max(...rows.map((r) => r[1]), 1e-6);
  return card("Pipeline runtime", `wall time per stage · total ${fmt.secs(compute)}`, h("div", {},
    meters(rows.map(([k, v]) => ({ label: k, value: v, max, display: fmt.secs(v), color: SERIES[0] }))),
    h("p", { class: "hint", style: "margin:12px 0 0" }, `Peak memory ${fmt.int(Math.round(t.run?.peak_rss_mb || t.peak_rss_mb || 0))} MB (process working set).`)));
}

function recentEntities(runId, ents) {
  const tabs = [["", "All"], ["MATCH", "Matched"], ["MULTI", "Multi-match"], ["NO_MATCH", "No match"], ["REVIEW", "Needs review"]];
  const body = h("tbody");
  const tabEl = h("div", { class: "tabs", role: "tablist" });
  const fill = (items) => {
    clear(body);
    if (!items.length) body.append(h("tr", {}, h("td", { colspan: 7, class: "muted" }, "No entities in this view.")));
    for (const e of items) {
      body.append(h("tr", { class: "click", onclick: () => { location.hash = `#/entity/${encodeURIComponent(e.s1_id)}`; } },
        h("td", {}, h("span", { class: "id-link" }, e.s1_id)),
        h("td", {}, h("div", { style: "font-weight:600" }, e.business_name), h("span", { class: "sub" }, e.business_address || "—")),
        h("td", {}, e.country || "—"), h("td", { class: "num" }, `${e.n_s2} / ${e.n_s3}`),
        h("td", { class: "num" }, fmt.dec(e.best, 3)),
        h("td", {}, e.decision === "NO_MATCH" && e.best >= 0.3 && e.gap < 0.15 ? badge("REVIEW") : badge(e.decision)),
        h("td", { class: "num r" }, fmt.pct(e.confidence))));
    }
  };
  for (const [k, label] of tabs) {
    const b = h("button", { type: "button", class: k === "" ? "on" : "", onclick: async () => {
      tabEl.querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
      fill((await get("/entities", { run_id: runId, size: 8, decision: k, sort: "-best" })).items);
    } }, label);
    tabEl.append(b);
  }
  fill(ents.items);
  return h("section", { class: "card" },
    h("div", { class: "card-head", style: "padding-bottom:12px;align-items:center" }, h("div", { class: "row", style: "gap:12px" }, h("h3", {}, "Entities"), tabEl),
      h("a", { href: "#/entities", class: "btn sm" }, "Open Entity Explorer", icon("arrowRight"))),
    h("div", { class: "table-wrap" }, h("table", { class: "t" },
      h("thead", {}, h("tr", {}, ["S1 ID", "Business", "Country", "Links S2 / S3", "Best score", "Decision", "Confidence"].map((x, i) => h("th", { class: i === 6 ? "r" : "" }, x)))), body)));
}
