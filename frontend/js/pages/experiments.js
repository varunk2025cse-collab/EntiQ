import { h, icon, clear, fmt, empty } from "../dom.js";
import { get } from "../api.js";
import { dotPlot, hbarChart, withTable, legend, SERIES } from "../charts.js";
import { card, prettyModel } from "./dashboard.js";

export async function render(root, { arg }) {
  const exps = await get("/experiments");
  root.append(h("div", { class: "card page-head" }, h("div", {}, h("h1", {}, "Experiments"),
    h("p", {}, "Every design decision — normalization, blocking, model family, ensembling — is kept only if it wins on held-out data. Nothing here is hard-coded."))));
  if (!exps.results.length && !exps.blocking) { root.append(h("div", { class: "card" }, empty("No experiments recorded", "", "python -m src.cli blocking\npython -m src.cli experiment"))); return; }
  const sel = exps.selected?.evidence || {};

  // selection summary
  root.append(h("div", { class: "grid g-3" },
    decision("Normalization", exps.selected?.overrides?.normalization?.mode, sel.normalization ? Object.entries(sel.normalization).map(([k, v]) => `${k}: CV ${fmt.dec(v.cv_f05, 4)} · holdout ${fmt.dec(v.holdout_f05, 4)}`) : []),
    decision("Blocking rule set", sel.blocking?.selected?.rule_set, sel.blocking ? [`recall ${fmt.pct(sel.blocking.selected.candidate_recall, 3)}`, `${fmt.int(sel.blocking.selected.candidates)} candidates`] : []),
    decision("Matching model", prettyModel(sel.model?.selected || ""), sel.model ? [`CV F0.5 ${fmt.dec(sel.model.cv_f05, 4)}`, `holdout F0.5 ${fmt.dec(sel.model.holdout_f05, 4)}`] : [])));

  // blocking table
  if (exps.blocking) {
    const t = exps.blocking.table;
    root.append(h("div", { class: "grid g-2" },
      card("Blocking experiment", "candidate recall vs. volume", withTable(dotPlot(t.map((r) => ({ label: r.rule_set, value: r.candidate_recall, sub: `${fmt.int(r.candidates)} candidates`, name: r.rule_set })),
        { highlight: (r) => r.name === sel.blocking?.selected?.rule_set, format: (v) => fmt.pct(v, 2) }),
      ["Rule set", "Recall", "Candidates", "Per S1", "Seconds"], t.map((r) => [r.rule_set, fmt.pct(r.candidate_recall, 3), fmt.int(r.candidates), fmt.dec(r.avg_candidates_per_s1, 1), fmt.dec(r.seconds, 2)]),
      legend([["Rule set", SERIES[0]], ["Selected", SERIES[6]]]))),
      card("Leave-one-out", "true links lost if a blocker is removed", withTable(hbarChart(exps.blocking.leave_one_out.map((r) => ({ label: r.removed, value: r.true_lost, sub: `saves ${fmt.int(r.candidates_saved)} candidates` }))),
        ["Removed blocker", "True links lost", "Candidates saved"], exps.blocking.leave_one_out.map((r) => [r.removed, fmt.int(r.true_lost), fmt.int(r.candidates_saved)])))));
  }

  // results table
  const cols = [["experiment_id", "ID"], ["suite", "Suite"], ["name", "Model"], ["normalization", "Norm."], ["cv_f05", "CV F0.5"], ["cv_precision", "CV P"], ["cv_recall", "CV R"],
    ["holdout_f05", "Holdout F0.5"], ["holdout_precision", "Hold P"], ["holdout_recall", "Hold R"], ["holdout_singleton_accuracy", "Singleton acc."], ["holdout_hard_f05", "Hard-case F0.5"], ["threshold", "Thr."], ["runtime_s", "Time"]];
  const detail = h("div", {});
  const tbody = h("tbody", {}, [...exps.results].reverse().map((r) => h("tr", { class: "click", onclick: () => showExp(detail, r.experiment_id) },
    cols.map(([k]) => h("td", { class: typeof r[k] === "number" ? "num" : "" }, fmtCell(k, r[k]))))));
  root.append(card("All experiments", `${exps.results.length} runs · experiments/results.csv`, h("div", { class: "table-wrap" }, h("table", { class: "t" }, h("thead", {}, h("tr", {}, cols.map(([, l]) => h("th", {}, l)))), tbody))), detail);
  if (arg) showExp(detail, arg);
}

function decision(title, value, lines) {
  return h("div", { class: "card kpi" }, h("div", { class: "ico c5" }, icon("beaker")), h("div", {}, h("div", { class: "lbl" }, title), h("div", { class: "val", style: "font-size:22px" }, value || "—"),
    lines.map((l) => h("div", { class: "foot" }, l))));
}

function fmtCell(k, v) {
  if (v === null || v === undefined || v === "") return "—";
  if (k === "name") return prettyModel(v);
  if (k === "runtime_s") return fmt.secs(v);
  if (typeof v === "number") return k === "threshold" ? fmt.dec(v, 2) : fmt.dec(v, 4);
  return String(v);
}

async function showExp(el, id) {
  const e = await get(`/experiments/${id}`);
  const a = e.artifacts || {};
  const imp = Object.entries(a.feature_importance || {}).slice(0, 12);
  const rel = a.reliability?.table || [];
  clear(el).append(h("div", { class: "grid g-3" },
    card(`${id} · ${prettyModel(e.name)}`, e.notes || e.suite, h("dl", { class: "kv" }, [["CV F0.5", fmt.dec(e.cv_f05, 4)], ["Holdout F0.5", fmt.dec(e.holdout_f05, 4)], ["Holdout precision", fmt.dec(e.holdout_precision, 4)],
      ["Holdout recall", fmt.dec(e.holdout_recall, 4)], ["Singleton accuracy", fmt.pct(e.holdout_singleton_accuracy, 2)], ["Hard-case F0.5", fmt.dec(e.holdout_hard_f05, 4)],
      ["Hard-case entities", fmt.int(e.holdout_hard_entities)], ["Calibration ECE", `${fmt.dec(e.calibration?.holdout_raw?.ece, 4)} raw → ${fmt.dec(e.calibration?.holdout_calibrated?.ece, 4)} calibrated`],
      ["Decision", JSON.stringify(e.decision)], ["Runtime", fmt.secs(e.runtime_s)]].flatMap(([k, v]) => [h("dt", {}, k), h("dd", { class: k === "Decision" ? "mono" : "" }, v)]))),
    card("Reliability (holdout)", "does 0.9 mean 90%?", h("table", { class: "t" }, h("thead", {}, h("tr", {}, ["Bin", "Pairs", "Mean conf.", "Observed"].map((x) => h("th", {}, x)))),
      h("tbody", {}, rel.map((r) => h("tr", {}, h("td", {}, r.bin), h("td", { class: "num" }, fmt.int(r.count)), h("td", { class: "num" }, fmt.dec(r.mean_confidence, 3)), h("td", { class: "num" }, fmt.dec(r.observed_match_rate, 3))))))),
    card("Top features", null, imp.length ? hbarChart(imp.map(([k, v]) => ({ label: k, value: v })), { format: (v) => fmt.pct(v, 1) }) : h("p", { class: "muted" }, "n/a"))));
  el.scrollIntoView({ behavior: "smooth", block: "start" });
}
