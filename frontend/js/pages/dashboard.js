import { h, icon, clear, fmt, badge, empty } from "../dom.js";
import { get, post, state } from "../api.js";
import { barChart, donut, hbarChart, dotPlot, meters, withTable, legend, SEQ, SERIES } from "../charts.js";
import { toast } from "../dom.js";

const REASON_LABEL = {
  NAME_EXACT: "Exact name", NAME_HIGH_SIMILARITY: "Name variation", ADDRESS_HIGH_SIMILARITY: "Address agreement",
  COUNTRY_MATCH: "Country match", MULTI_FIELD_AGREEMENT: "Multi-field agreement", STRONG_SCORE_GAP: "Clear winner",
  MUTUAL_BEST: "Mutual best", POSTAL_MATCH: "Postal code", STREET_NUMBER_MATCH: "Street number", SEMANTIC_SUPPORT: "Semantic support",
  TRADE_NAME_MATCH: "Trade name / DBA", ACRONYM_MATCH: "Acronym", COMMON_NAME: "Common name",
};
export const reasonLabel = (c) => REASON_LABEL[c] || c.replace(/_/g, " ").toLowerCase();

export async function render(root) {
  const runs = await get("/resolution/runs").catch(() => []);
  const completed = runs.filter((r) => r.status === "COMPLETED");
  if (!completed.length) { root.append(header(null, completed), onboarding()); return; }
  const runId = completed.some((r) => r.id === state.runId) ? state.runId : completed[0].id;
  state.runId = runId;
  const run = completed.find((r) => r.id === runId);
  const prev = completed.find((r) => r.id !== runId && r.created < run.created && r.dataset_id === run.dataset_id && r.split === run.split);
  const [an, models, exps, ents] = await Promise.all([
    get(`/analytics/${runId}`), get("/models").catch(() => null), get("/experiments").catch(() => null),
    get("/entities", { run_id: runId, size: 8, sort: "-best" })]);
  const prod = models?.versions?.find((v) => v.version === an.summary.model_version) || models?.versions?.[0];
  const val = validationMetrics(prod, exps);
  const c = an.summary.counts;

  root.append(header(run, completed));
  const main = h("div", { class: "dash-main" });
  const inspector = h("aside", { class: "inspector", "aria-label": "Entity inspector" });
  root.append(h("div", { class: "dash" }, main, inspector));

  // KPI tiles -----------------------------------------------------------------
  const pc = prev?.summary?.counts;
  main.append(h("div", { class: "grid g-5" },
    kpi("c1", "database", "Total Entities (S1)", fmt.int(c.s1), delta(c.s1, pc?.s1), "Reference businesses processed"),
    kpi("c2", "link", "Candidate Pairs", fmt.int(c.candidates), delta(c.candidates, pc?.candidates), "From multi-pass blocking"),
    kpi("c3", "check", "Final Matches", fmt.int(c.links), delta(c.links, pc?.links), `Linked records · ${fmt.int(c.s2_links)} S2 / ${fmt.int(c.s3_links)} S3`),
    kpi("c4", "user", "Singletons (No Match)", fmt.int(c.singletons), delta(c.singletons, pc?.singletons, true), "No reliable match found"),
    kpi("c5", "shield", "Model Performance (F0.5)", val.f05 != null ? fmt.dec(val.f05, 3) : "—", null, val.label)));

  // charts row ----------------------------------------------------------------
  const bins = an.confidence_histogram.bins;
  const confChart = barChart({ labels: bins.map((b) => b.replace("-", "–")), values: an.confidence_histogram.counts, colors: SEQ,
    tipLabel: (i) => `accepted links with confidence ${bins[i]}` });
  const reasons = Object.entries(an.reason_codes || {}).sort((a, b) => b[1] - a[1]);
  const top = reasons.slice(0, 6).map(([k, v]) => ({ label: reasonLabel(k), value: v }));
  const rest = reasons.slice(6).reduce((a, [, v]) => a + v, 0);
  if (rest) top.push({ label: "Other", value: rest, color: "var(--text-muted)" });
  const countries = Object.entries(an.country_distribution || {}).sort((a, b) => b[1] - a[1]);
  const totalC = countries.reduce((a, [, v]) => a + v, 0) || 1;
  main.append(h("div", { class: "grid g-3" },
    card("Match Confidence Distribution", "calibrated score of accepted links",
      withTable(confChart, ["Confidence", "Links"], bins.map((b, i) => [b, fmt.int(an.confidence_histogram.counts[i])]))),
    card("Match Evidence", "reason codes behind accepted links",
      top.length ? donut(top, { center: fmt.int(c.matched_entities), centerLabel: "Matched entities" }) : empty("No matches", "")),
    card("Geographic Distribution", "S1 entities per country · match rate",
      withTable(hbarChart(countries.slice(0, 6).map(([k, v], i) => ({ label: k, value: v / totalC, color: SERIES[i % 8],
        sub: `${fmt.int(v)} entities · ${fmt.pct(an.country_match_rate?.[k])} matched` })), { format: (v) => fmt.pct(v), max: 1 }),
      ["Country", "Entities", "Share", "Match rate"], countries.map(([k, v]) => [k, fmt.int(v), fmt.pct(v / totalC), fmt.pct(an.country_match_rate?.[k])])))));

  // model comparison / pipeline / insights --------------------------------------
  main.append(h("div", { class: "grid g-3" }, modelComparison(exps), pipeline(an), insights(an, val, exps)));

  // recent entities -------------------------------------------------------------
  main.append(recentEntities(runId, ents, inspector));
  const first = ents.items.find((e) => e.decision === "MATCH") || ents.items[0];
  if (first) renderInspector(inspector, runId, first.s1_id);
}

function header(run, completed) {
  const sel = h("select", { class: "select", "aria-label": "Resolution run", onchange: (e) => { state.runId = e.target.value; window.dispatchEvent(new HashChangeEvent("hashchange")); } },
    completed.map((r, i) => h("option", { value: r.id, selected: run && r.id === run.id }, `${r.id}${i === 0 ? " · latest" : ""} · ${r.split}`)));
  const btn = h("button", { class: "btn primary", type: "button", onclick: async () => {
    btn.disabled = true;
    try {
      const ds = await get("/datasets");
      const d = ds.find((x) => x.kind === "competition" && x.splits.includes("test")) || ds.find((x) => x.splits.includes("test")) || ds[0];
      if (!d) throw new Error("No dataset available");
      const r = await post("/resolution/run", { dataset_id: d.id, split: d.splits.includes("test") ? "test" : d.splits[0] });
      toast(`Started ${r.run_id} on ${d.name}`);
      location.hash = "#/runs";
    } catch (e) { toast(e.message, true); } finally { btn.disabled = false; }
  } }, icon("play"), "Run Resolution");
  return h("div", { class: "card page-head" },
    h("div", {}, h("h1", {}, "Dashboard"), h("p", {}, "Monitor your entity resolution performance and data insights in real time.")),
    h("div", { class: "row" }, run ? h("span", { class: "chip" }, icon("calendar"), fmt.date(run.finished || run.created)) : null, completed.length ? sel : null, btn));
}

function onboarding() {
  return h("div", { class: "card" }, empty("No resolution run yet",
    "Train a model on the training split, then resolve the test split. Every figure on this dashboard is computed from real runs.",
    "python -m src.cli all          # profile → blocking → experiments → train → predict\npython -m backend.app.main       # then reload this page"));
}

function kpi(cls, ic, label, value, deltaEl, foot) {
  return h("div", { class: "card kpi" }, h("div", { class: `ico ${cls}` }, icon(ic)),
    h("div", { style: "min-width:0" }, h("div", { class: "lbl" }, label), h("div", { class: "val" }, value),
      deltaEl, h("div", { class: "foot" }, foot)));
}

function delta(cur, prev, lowerIsGood = false) {
  if (prev === undefined || prev === null || !prev) return null;
  const d = (cur - prev) / prev;
  if (!isFinite(d) || Math.abs(d) < 0.0005) return h("div", { class: "delta" }, "No change vs previous run");
  const up = d > 0;
  const good = lowerIsGood ? !up : up;
  return h("div", { class: `delta ${good ? "up" : "down"}` }, icon(up ? "arrowUp" : "arrowDown"), `${fmt.pct(Math.abs(d))} vs previous run`);
}

export function card(title, sub, body, action) {
  return h("section", { class: "card" }, h("div", { class: "card-head" }, h("h3", {}, title, sub ? h("span", { class: "sub" }, `(${sub})`) : null), action || null),
    h("div", { class: "card-body" }, body));
}

export function validationMetrics(prod, exps) {
  // Test data has no labels: model quality comes from held-out TRAINING entities (never seen in fitting).
  const res = exps?.results || [];
  const name = prod?.model_name || prod?.spec?.name;
  const rows = res.filter((r) => r.name === name && r.holdout_f05 !== null && r.holdout_f05 !== undefined);
  const row = rows[rows.length - 1];
  if (row) return { f05: row.holdout_f05, precision: row.holdout_precision, recall: row.holdout_recall, singleton: row.holdout_singleton_accuracy,
    candidate_recall: row.candidate_recall, label: "Holdout validation score", source: "holdout", cv: row.cv_f05 };
  if (prod?.cv_metrics) return { f05: prod.cv_metrics.f05, precision: prod.cv_metrics.precision, recall: prod.cv_metrics.recall,
    singleton: prod.cv_metrics.singleton_accuracy, label: "Cross-validated score", source: "cv" };
  return { f05: null, label: "No validation available" };
}

function modelComparison(exps) {
  // model families only: A/B checks (normalization, pruning, sibling) and tuning trials live on the Experiments page
  const res = (exps?.results || []).filter((r) => !["normalization", "pruning", "sibling", "hpo"].includes(r.suite));
  const latest = new Map();
  for (const r of res) latest.set(r.name, r);
  const rows = [...latest.values()].sort((a, b) => a.cv_f05 - b.cv_f05).map((r) => ({ label: prettyModel(r.name), value: r.cv_f05, name: r.name,
    sub: `holdout ${fmt.dec(r.holdout_f05, 4)}` }));
  const selected = exps?.selected?.overrides?.model?.primary;
  const body = rows.length ? withTable(dotPlot(rows, { highlight: (r) => r.name === selected }), ["Model", "CV F0.5", "Holdout F0.5"],
    [...latest.values()].map((r) => [prettyModel(r.name), fmt.dec(r.cv_f05, 4), fmt.dec(r.holdout_f05, 4)]),
    legend([["Evaluated model", SERIES[0]], ["Selected by validation", SERIES[6]]]))
    : empty("No experiments yet", "Run the experiment suite to compare models.", "python -m src.cli experiment");
  return card("Model Comparison", "validation F0.5", body, selected ? h("span", { class: "chip blue" }, `Best: ${prettyModel(selected)}`) : null);
}

export function prettyModel(n) {
  return ({ lr: "Logistic Regression", rf: "Random Forest", et: "Extra Trees", hgb: "Gradient Boosting", xgb: "XGBoost", lgbm: "LightGBM",
    fs: "Fellegi-Sunter", hgb_no_semantic: "GBM − semantic", hgb_no_context: "GBM − context", hgb_no_sibling: "GBM − siblings", lr_no_sibling: "LR − siblings", hgb_hnm: "GBM + hard negatives",
    cascade: "Semantic cascade", stack: "Stacked ensemble", stack_hnm: "Stack + hard neg." })[n] || n;
}

function pipeline(an) {
  const t = an.timings || {};
  const st = t.stages || {};
  const run = t.run?.stages || {};
  const rows = [["Data validation", run.data_validation?.seconds], ["Normalization & indexing", st.normalization_and_indexing?.seconds],
    ["Blocking & candidates", st.blocking?.seconds], ["Cascade pruning", st.pruning?.seconds], ["Feature engineering", st.features?.seconds],
    ["Model inference", run.inference?.seconds], ["Decision & post-processing", (run.decision?.seconds || 0) + (run.outputs?.seconds || 0)]]
    .filter(([, v]) => v !== undefined && v !== null);
  const total = rows.reduce((a, [, v]) => a + v, 0) || 1;
  return card("Pipeline Performance", "wall time per stage", h("div", {},
    meters(rows.map(([k, v]) => ({ label: k, value: v, max: Math.max(...rows.map((r) => r[1])), display: fmt.secs(v) }))),
    h("div", { class: "hr", style: "margin:12px 0" }),
    h("div", { class: "row", style: "justify-content:space-between;font-size:12.5px" }, h("span", { class: "secondary" }, "Total compute"), h("b", { class: "num" }, fmt.secs(total)))));
}

function insights(an, val, exps) {
  const b = an.summary.blocking?.union || {};
  const res = exps?.results || [];
  const selected = exps?.selected?.overrides?.model?.primary;
  const sel = [...res].reverse().find((r) => r.name === selected);
  const lr = [...res].reverse().find((r) => r.name === "lr");
  let call = null;
  if (sel && lr && sel.name !== "lr") {
    const d = sel.cv_f05 - lr.cv_f05;
    call = h("div", { class: `callout ${d >= 0 ? "" : "warn"}` }, h("div", { class: "ic" }, icon(d >= 0 ? "check" : "info")),
      h("div", {}, `${prettyModel(sel.name)} ${d >= 0 ? "improves" : "trails"} validation F0.5 by `, h("b", {}, `${(d * 100).toFixed(2)} pts`), " over the logistic-regression baseline."));
  } else if (sel) {
    call = h("div", { class: "callout info" }, h("div", { class: "ic" }, icon("info")), h("div", {}, `${prettyModel(sel.name)} was selected purely on validation F0.5.`));
  }
  const rows = [
    ["target", "Candidate recall (blocking)", val.candidate_recall ?? b.candidate_recall, val.candidate_recall != null ? "validation" : "no labels on this run"],
    ["user", "Singleton accuracy", val.singleton, val.source],
    ["shield", "Precision (macro)", val.precision, val.source]];
  return card("Key Insights", null, h("div", {}, rows.map(([ic, t, v, src]) => h("div", { class: "insight" }, h("div", { class: "ic" }, icon(ic)),
    h("div", {}, h("div", { class: "t" }, t, h("span", { class: "muted" }, src ? ` · ${src}` : "")), h("div", { class: "n" }, v != null ? fmt.pct(v) : "—")))),
  call));
}

function recentEntities(runId, ents, inspector) {
  const tabs = [["", "All"], ["MATCH", "Matched"], ["NO_MATCH", "Singletons"], ["REVIEW", "Needs Review"]];
  let cur = "";
  const body = h("tbody");
  const tabEl = h("div", { class: "tabs", role: "tablist" });
  const fill = (items) => {
    clear(body);
    if (!items.length) body.append(h("tr", {}, h("td", { colspan: 8, class: "muted" }, "No entities in this view.")));
    for (const e of items) {
      const tr = h("tr", { class: "click", onclick: () => { body.querySelectorAll("tr").forEach((x) => x.classList.remove("sel")); tr.classList.add("sel"); renderInspector(inspector, runId, e.s1_id); } },
        h("td", {}, h("a", { class: "id-link", href: `#/entity/${encodeURIComponent(e.s1_id)}`, onclick: (ev) => ev.stopPropagation() }, e.s1_id)),
        h("td", {}, e.business_name), h("td", {}, e.country || "—"),
        h("td", { class: "num" }, `${e.n_s2} / ${e.n_s3}`), h("td", { class: "num" }, e.decision === "MATCH" ? fmt.dec(e.best, 3) : "—"),
        h("td", {}, e.decision === "NO_MATCH" && e.best >= 0.3 && e.gap < 0.15 ? badge("REVIEW") : badge(e.decision)),
        h("td", { class: "num r" }, fmt.pct(e.confidence)),
        h("td", { class: "r" }, h("a", { class: "btn sm", href: `#/entity/${encodeURIComponent(e.s1_id)}` }, "View")));
      body.append(tr);
    }
  };
  for (const [k, label] of tabs) {
    const b = h("button", { type: "button", class: k === cur ? "on" : "", onclick: async () => {
      cur = k; tabEl.querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
      const r = await get("/entities", { run_id: runId, size: 8, decision: k, sort: k === "NO_MATCH" ? "-best" : "-best" });
      fill(r.items);
    } }, label);
    tabEl.append(b);
  }
  fill(ents.items);
  return h("section", { class: "card" },
    h("div", { class: "card-head" }, h("div", { class: "row" }, h("h3", {}, "Recent Entities"), tabEl), h("a", { href: "#/entities", class: "btn ghost sm" }, "View all", icon("arrowRight"))),
    h("div", { class: "table-wrap", style: "padding:6px 6px 8px" }, h("table", { class: "t" },
      h("thead", {}, h("tr", {}, ["S1 ID", "Business Name", "Country", "Matches (S2/S3)", "Best Score", "Decision", "Confidence", ""].map((x, i) => h("th", { class: i >= 6 ? "r" : "" }, x)))), body)));
}

export async function renderInspector(el, runId, s1) {
  el.classList.add("skeleton");
  const d = await get(`/entities/${encodeURIComponent(s1)}`, { run_id: runId });
  const e = d.entity;
  const matches = d.matches || [];
  const others = (d.candidates || []).filter((c) => !c.selected).slice(0, 3);
  const top = matches[0];
  const ex = top ? await get(`/matches/${encodeURIComponent(top.match_id)}/explanation`, { run_id: runId }).catch(() => null) : null;
  clear(el);
  el.classList.remove("skeleton");
  el.append(
    h("div", { class: "card" }, h("div", { class: "card-head" }, h("h3", {}, "Entity Explorer"), h("a", { class: "btn sm", href: `#/entity/${encodeURIComponent(s1)}` }, icon("compare"), "Open")),
      h("div", { class: "ent-head" }, h("div", { class: "ico" }, icon("building")),
        h("div", { style: "min-width:0" }, h("div", { class: "row" }, h("span", { class: "ent-id" }, s1), h("span", { class: "chip blue" }, "Reference")),
          h("div", { class: "ent-name" }, d.record.business_name), h("div", { class: "ent-loc" }, icon("pin"), d.record.business_address || "—", d.record.country ? ` · ${d.record.country}` : "")))),
    h("div", { class: "card" }, h("div", { class: "card-head", style: "padding-bottom:10px" }, h("h3", {}, `Matched Records (${matches.length})`), badge(e.decision)),
      h("div", { class: "rec-list" }, matches.length ? matches.map((m) => recRow(m, runId)) : h("div", { class: "rec", style: "cursor:default" },
        h("span"), h("div", { class: "muted" }, (e.reason_codes || []).includes("NO_CANDIDATES") ? "Blocking found no plausible candidate." : "No candidate cleared the precision-first decision gate."), h("span")))),
    others.length ? h("div", { class: "card" }, h("div", { class: "card-head", style: "padding-bottom:10px" }, h("h3", {}, "Non-matched Candidates"),
      h("a", { class: "link-btn", href: `#/candidates/${encodeURIComponent(s1)}` }, `View all (${d.n_candidates})`)),
    h("div", { class: "rec-list" }, others.map((m) => recRow(m, runId)))) : null,
    ex ? explanationCard(ex.explanation, runId, top.match_id) : null);
}

function recRow(m, runId) {
  const st = m.selected ? "MATCH" : m.p >= 0.3 ? "REVIEW" : "NO_MATCH";
  return h("a", { class: "rec", href: `#/match/${encodeURIComponent(m.match_id)}`, style: "text-decoration:none;color:inherit" },
    h("div", { class: `src ${m.selected ? m.target : "neg"}` }, m.target),
    h("div", { style: "min-width:0" }, h("div", { class: "rid" }, m.entity_id), h("div", { class: "rn" }, m.business_name), h("div", { class: "ra" }, m.business_address || "—")),
    h("div", { class: "score" }, fmt.pct(m.p), st === "NO_MATCH" ? h("span", { class: "badge nomatch" }, "No match") : badge(st)));
}

function explanationCard(ex, runId, matchId) {
  const bars = ex.evidence.filter((e) => e.kind !== "gap").map((e, i) => ({ label: e.label, value: Math.max(0, e.value), color: SERIES[[2, 0, 3, 6, 4, 5, 1][i % 7]] }));
  return h("div", { class: "card" }, h("div", { class: "card-head" }, h("h3", {}, "Match Explanation"),
    h("a", { class: "btn sm", href: `#/match/${encodeURIComponent(matchId)}` }, icon("sparkle"), "Investigate")),
  h("div", { class: "card-body" },
    h("div", { class: "grid g-2", style: "gap:8px;margin-bottom:12px" }, miniRec("S1 (Reference)", ex.s1), miniRec(`${ex.target} (${ex.decision === "MATCH" ? "Matched" : "Candidate"})`, ex.candidate)),
    meters(bars), h("div", { class: "hr", style: "margin:14px 0" }),
    h("div", { class: `verdict ${ex.decision === "MATCH" ? "" : "no"}` }, h("div", { class: "ring" }, icon(ex.decision === "MATCH" ? "check" : "x")),
      h("div", {}, h("div", { class: "muted", style: "font-size:12px" }, "Final confidence"), h("div", { class: "conf" }, fmt.pct(ex.confidence))),
      h("div", { style: "margin-left:auto;text-align:right" }, h("div", { class: "muted", style: "font-size:12px;margin-bottom:6px" }, "Decision"), badge(ex.decision === "MATCH" ? "MATCH" : "NO_MATCH"))),
    h("div", { style: "margin-top:12px" }, h("div", { class: "muted", style: "font-size:12px;margin-bottom:6px" }, "Reason codes"),
      h("div", { class: "row", style: "gap:6px" }, ex.reason_codes.map((c) => h("span", { class: "code-chip" }, c))))));
}

function miniRec(title, r) {
  return h("div", { class: "card", style: "padding:10px;box-shadow:none;background:var(--surface-2)" }, h("div", { class: "muted", style: "font-size:11px" }, title),
    h("div", { style: "font-weight:600;font-size:12.5px;margin-top:3px" }, r.business_name), h("div", { class: "muted", style: "font-size:11px" }, r.business_address || "—"));
}
