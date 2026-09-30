import { h, icon, clear, fmt, empty, toast } from "../dom.js";
import { get } from "../api.js";
import { dotPlot, hbarChart, lineChart, withTable, legend, SERIES } from "../charts.js";
import { kindBadge } from "../app.js";
import { card, pageHead, prettyModel, metricList } from "../ui.js";

const SUITE_LABEL = { normalization: "A/B · normalisation", pruning: "A/B · cascade filter", sibling: "A/B · sibling features", hpo: "Tuning",
  standard: "Model family", full: "Model family", baselines: "Model family", quick: "Model family" };

export async function render(root, { arg }) {
  const exps = await get("/experiments");
  root.append(pageHead("Experiments", "Experiment history",
    "Every design decision — normalisation, blocking, cascade filter, feature groups, model family, tuning — is kept only if it wins on held-out data. Selections below link to their evidence."));
  if (!exps.results.length && !exps.blocking) { root.append(h("div", { class: "card" }, empty("No experiments recorded", "Run the experiment suite.", "python -m src.cli blocking\npython -m src.cli experiment", "beaker"))); return; }
  const ev = exps.selected?.evidence || {};
  const ov = exps.selected?.overrides || {};
  const dataKinds = [...new Set(exps.results.map((r) => r.dataset_kind || "unknown"))];

  root.append(h("div", { class: "kpis" },
    decision("Normalisation", ov.normalization?.mode, ev.normalization ? Object.entries(ev.normalization).map(([k, v]) => `${k}: CV ${fmt.dec(v.cv_f05, 4)}`) : []),
    decision("Blocking rule set", ev.blocking?.selected?.rule_set?.replace(/_/g, " "), ev.blocking ? [`candidate recall ${fmt.pct(ev.blocking.selected.candidate_recall, 3)}`] : []),
    decision("Cascade filter", ev.pruning ? (ov.blocking?.pruning?.enabled ? "On" : "Off") : null, ev.pruning ? [`on: CV ${fmt.dec(ev.pruning.on?.cv_f05, 4)}`, `off: CV ${fmt.dec(ev.pruning.off?.cv_f05, 4)}`] : []),
    decision("Sibling features", ev.sibling_features ? (ev.sibling_features.kept ? "On" : "Off") : null, ev.sibling_features ? [`on: CV ${fmt.dec(ev.sibling_features.on?.cv_f05, 4)}`, `off: CV ${fmt.dec(ev.sibling_features.off?.cv_f05, 4)}`] : []),
    decision("Model", prettyModel(ev.model?.selected || ""), ev.model ? [`CV ${fmt.dec(ev.model.cv_f05, 4)} · holdout ${fmt.dec(ev.model.holdout_f05, 4)}`] : [])));
  if (dataKinds.length) root.append(h("div", { class: "row" }, h("span", { class: "hint" }, "Data sources in this history:"), ...dataKinds.map(kindBadge)));

  if (exps.blocking) {
    const t = exps.blocking.table;
    root.append(h("div", { class: "grid g-2" },
      card("Blocking rule sets", "true links retrieved vs. candidate volume (before the cascade filter)", withTable(dotPlot(t.map((r) => ({ label: r.rule_set.replace(/_/g, " "), value: r.candidate_recall, sub: `${fmt.int(r.candidates)} candidates`, name: r.rule_set })),
        { highlight: (r) => r.name === ev.blocking?.selected?.rule_set, format: (v) => fmt.pct(v, 2) }),
      ["Rule set", "Recall", "Candidates", "Per S1", "Seconds"], t.map((r) => [r.rule_set, fmt.pct(r.candidate_recall, 3), fmt.int(r.candidates), fmt.dec(r.avg_candidates_per_s1, 1), fmt.dec(r.seconds, 2)]),
      legend([["Rule set", SERIES[0]], ["Selected", SERIES[6]]]))),
      card("Leave-one-out", "true links lost if a single blocker is removed", withTable(hbarChart(exps.blocking.leave_one_out.map((r) => ({ label: r.removed, value: r.true_lost, sub: `saves ${fmt.int(r.candidates_saved)} candidates` }))),
        ["Removed blocker", "True links lost", "Candidates saved"], exps.blocking.leave_one_out.map((r) => [r.removed, fmt.int(r.true_lost), fmt.int(r.candidates_saved)])))));
  }

  // ---- history with comparison ---------------------------------------------------------
  const picked = new Set();
  const cmp = h("div", {});
  const cols = [["experiment_id", "ID"], ["timestamp", "Time"], ["dataset_kind", "Data"], ["suite", "Kind"], ["name", "Model"], ["cv_f05", "CV F0.5"],
    ["holdout_f05", "Holdout F0.5"], ["holdout_precision", "Precision"], ["holdout_recall", "Recall"], ["candidate_recall", "Cand. recall"], ["runtime_s", "Runtime"]];
  const tbody = h("tbody", {}, [...exps.results].reverse().map((r) => {
    const box = h("input", { type: "checkbox", "aria-label": `compare ${r.experiment_id}`, onclick: (e) => e.stopPropagation(), onchange: (e) => {
      if (e.target.checked) { if (picked.size >= 2) { e.target.checked = false; toast("Compare two experiments at a time"); return; } picked.add(r.experiment_id); } else picked.delete(r.experiment_id);
      drawCompare(cmp, exps.results.filter((x) => picked.has(x.experiment_id)));
    } });
    return h("tr", { class: "click", onclick: () => showExp(detail, r.experiment_id) }, h("td", {}, box), cols.map(([k]) => h("td", { class: typeof r[k] === "number" && k !== "experiment_id" ? "num" : "" }, cell(k, r[k], r))));
  }));
  const detail = h("div", {});
  root.append(card("History", `${exps.results.length} experiments · experiments/results.csv · tick two rows to compare`,
    h("div", { class: "table-wrap", style: "margin:0 -20px" }, h("table", { class: "t" }, h("thead", {}, h("tr", {}, h("th", {}, ""), cols.map(([, l]) => h("th", {}, l)))), tbody))), cmp, detail);
  if (arg) showExp(detail, arg);
}

function decision(title, value, lines) {
  return h("div", { class: "card kpi" }, h("div", { class: "top" }, h("div", { class: "ico" }, icon("beaker")), h("div", { class: "lbl" }, title)),
    value ? h("div", { class: "val", style: "font-size:19px" }, value) : h("div", { class: "val na" }, "Not decided yet"), lines.map((l) => h("div", { class: "foot" }, l)));
}

function cell(k, v, r) {
  if (v === null || v === undefined || v === "") return "—";
  if (k === "dataset_kind") return kindBadge(v);
  if (k === "suite") return SUITE_LABEL[v] || v;
  if (k === "name") return prettyModel(v);
  if (k === "timestamp") return fmt.date(v);
  if (k === "runtime_s") return fmt.secs(v);
  if (k === "experiment_id") return h("span", { class: "mono" }, v);
  if (["holdout_precision", "holdout_recall", "candidate_recall"].includes(k)) return fmt.pct(v, 2);
  if (typeof v === "number") return fmt.dec(v, 4);
  return String(v);
}

function drawCompare(el, rows) {
  clear(el);
  if (rows.length !== 2) return;
  const [a, b] = rows;
  const keys = [["Data source", "dataset_kind"], ["Kind", "suite"], ["Model", "name"], ["Features", "feature_set"], ["Normalisation", "normalization"],
    ["Threshold", "threshold"], ["CV F0.5", "cv_f05"], ["CV precision", "cv_precision"], ["CV recall", "cv_recall"], ["Holdout F0.5", "holdout_f05"],
    ["Holdout precision", "holdout_precision"], ["Holdout recall", "holdout_recall"], ["No-match accuracy", "holdout_singleton_accuracy"],
    ["Hard-case F0.5", "holdout_hard_f05"], ["Candidate recall", "candidate_recall"], ["Runtime", "runtime_s"]];
  el.append(card(`Compare ${a.experiment_id} vs ${b.experiment_id}`, a.dataset_kind !== b.dataset_kind ? "different data sources — not directly comparable" : "same data source",
    h("div", { class: "table-wrap", style: "margin:0 -20px" }, h("table", { class: "t" }, h("thead", {}, h("tr", {}, h("th", {}, "Metric"), h("th", {}, a.experiment_id), h("th", {}, b.experiment_id), h("th", {}, "Δ (B − A)"))),
      h("tbody", {}, keys.map(([l, k]) => {
        const d = typeof a[k] === "number" && typeof b[k] === "number" ? b[k] - a[k] : null;
        return h("tr", {}, h("td", {}, l), h("td", {}, cell(k, a[k], a)), h("td", {}, cell(k, b[k], b)),
          h("td", { class: "num", style: { color: d === null || k === "runtime_s" ? "" : d > 0 ? "var(--good-ink)" : d < 0 ? "var(--crit-ink)" : "" } }, d === null ? "" : k === "runtime_s" ? fmt.secs(d) : (d > 0 ? "+" : "") + d.toFixed(4)));
      }))))));
}

async function showExp(el, id) {
  const e = await get(`/experiments/${id}`);
  const a = e.artifacts || {};
  const imp = Object.entries(a.feature_importance || {}).slice(0, 12);
  const rel = a.reliability?.table || [];
  const sw = a.threshold_sweep || [];
  clear(el).append(h("div", { class: "section-title" }, `${id} · ${prettyModel(e.name)}`), h("div", { class: "grid g-3" },
    card("Result", e.notes || SUITE_LABEL[e.suite] || e.suite, metricList([
      ["Data source", kindBadge(e.dataset_kind || "unknown")], ["CV F0.5", fmt.dec(e.cv_f05, 4)], ["Holdout F0.5", fmt.dec(e.holdout_f05, 4)],
      ["Holdout precision", fmt.pct(e.holdout_precision, 2)], ["Holdout recall", fmt.pct(e.holdout_recall, 2)], ["No-match accuracy", fmt.pct(e.holdout_singleton_accuracy, 2)],
      ["Hard-case F0.5", fmt.dec(e.holdout_hard_f05, 4), `${fmt.int(e.holdout_hard_entities)} entities`],
      ["Calibration error (ECE)", e.calibration ? `${fmt.dec(e.calibration.holdout_raw?.ece, 4)} → ${fmt.dec(e.calibration.holdout_calibrated?.ece, 4)}` : null],
      ["Validation split", a.leakage_report ? `${fmt.int(a.leakage_report.dev_entities)} dev / ${fmt.int(a.leakage_report.holdout_entities)} holdout S1 · seed ${e.seed}` : null],
      ["Runtime", fmt.secs(e.runtime_s)]])),
    card("Threshold sweep (dev out-of-fold)", "tuning, not a model", sw.length ? lineChart(sw.map((r) => r.threshold), [
      { name: "F0.5", values: sw.map((r) => r.f05), color: SERIES[6] }, { name: "Precision", values: sw.map((r) => r.precision), color: SERIES[0] },
      { name: "Recall", values: sw.map((r) => r.recall), color: SERIES[1] }], { marker: e.threshold, markerLabel: `chosen ${fmt.dec(e.threshold, 2)}`, height: 220 })
      : rel.length ? h("div", { class: "table-wrap", style: "margin:0 -20px" }, h("table", { class: "t" }, h("thead", {}, h("tr", {}, ["Bin", "Pairs", "Mean conf.", "Observed"].map((x) => h("th", {}, x)))),
        h("tbody", {}, rel.map((r) => h("tr", {}, h("td", {}, r.bin), h("td", { class: "num" }, fmt.int(r.count)), h("td", { class: "num" }, fmt.dec(r.mean_confidence, 3)), h("td", { class: "num" }, fmt.dec(r.observed_match_rate, 3)))))))
        : empty("Not recorded", "")),
    card("Top inputs", null, imp.length ? hbarChart(imp.map(([k, v]) => ({ label: k, value: v })), { format: (v) => fmt.pct(v, 1) }) : empty("Not available", ""))));
  el.scrollIntoView({ behavior: "smooth", block: "start" });
}
