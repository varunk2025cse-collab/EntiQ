import { h, icon, fmt, empty, toast, statusBadge } from "../dom.js";
import { get, post } from "../api.js";
import { hbarChart, lineChart, withTable, legend, SERIES } from "../charts.js";
import { kindBadge } from "../app.js";
import { card, pageHead, prettyModel, isModelExperiment, metricList } from "../ui.js";

export async function render(root) {
  const [m, exps, jobs] = await Promise.all([get("/models"), get("/experiments").catch(() => null), get("/models/jobs").catch(() => [])]);
  const specs = ["stack", "hgb", "lr", "rf", "et", "fs", "hgb_hnm", "cascade", "stack_hnm",
    ...Object.entries(m.available_families).filter(([k, v]) => v && ["xgb", "lgbm"].includes(k)).map(([k]) => k)];
  const selected = exps?.selected?.overrides?.model?.primary;
  const modelSel = h("select", { class: "select" }, specs.map((s) => h("option", { value: s, selected: s === selected }, `${prettyModel(s)}${s === selected ? " — validation choice" : ""}`)));
  const promote = h("input", { type: "checkbox", id: "promote" });
  const trainBtn = h("button", { class: "btn primary", onclick: async () => {
    trainBtn.disabled = true;
    try { const r = await post("/models/train", { model: modelSel.value, promote: promote.checked }); toast(`Training job ${r.job_id} queued — it creates a NEW version`); setTimeout(() => location.reload(), 1200); }
    catch (e) { toast(e.message, true); } finally { trainBtn.disabled = false; }
  } }, icon("zap"), "Train new version");

  root.append(pageHead("Model laboratory", "Models, threshold tuning and model families",
    "Three different things live here and are never mixed: trained model versions, threshold tuning of a model, and validation comparisons of model families. Training always creates a new version; competition outputs change only when a version is promoted.",
    [modelSel, h("label", { class: "row", style: "gap:6px;font-size:12.5px" }, promote, "Promote when done"), trainBtn]));

  if (jobs.length) {
    root.append(card("Training jobs", "background jobs started from this page", h("div", { class: "table-wrap", style: "margin:0 -20px" }, h("table", { class: "t" },
      h("thead", {}, h("tr", {}, ["Job", "Model", "Status", "Stage", "Version"].map((x) => h("th", {}, x)))),
      h("tbody", {}, jobs.map((j) => h("tr", {}, h("td", { class: "mono" }, j.id), h("td", {}, prettyModel(j.model)), h("td", {}, statusBadge(j.status)),
        h("td", { class: "secondary" }, j.error ? j.error.split("\n")[0] : j.stage), h("td", { class: "mono" }, j.version || "—"))))))));
  }
  if (!m.versions.length) {
    root.append(h("div", { class: "card" }, empty("No model trained yet", "Train one from here or with the CLI.", "python -m src.cli train", "flask")));
  } else {
    root.append(h("div", { class: "section-title" }, "1 · Trained model versions"));
    root.append(card("Model versions", "metrics: out-of-fold cross-validation on all training entities of the model's data source",
      h("div", { class: "table-wrap", style: "margin:0 -20px" }, h("table", { class: "t" },
        h("thead", {}, h("tr", {}, ["Version", "Model", "Data source", "CV F0.5", "Precision", "Recall", "No-match acc.", "Threshold", "Created", ""].map((x) => h("th", {}, x)))),
        h("tbody", {}, m.versions.map((v) => h("tr", {},
          h("td", {}, h("span", { class: "mono", style: "font-weight:600" }, v.version), v.production ? h("span", { class: "chip blue", style: "margin-left:6px" }, "Production") : null),
          h("td", {}, prettyModel(v.model_name)), h("td", {}, kindBadge(v.dataset_kind || "unknown")),
          h("td", { class: "num" }, fmt.dec(v.cv_f05, 4)), h("td", { class: "num" }, fmt.pct(v.cv_metrics?.precision, 2)), h("td", { class: "num" }, fmt.pct(v.cv_metrics?.recall, 2)),
          h("td", { class: "num" }, fmt.pct(v.cv_metrics?.singleton_accuracy, 2)), h("td", { class: "num" }, fmt.dec(v.decision?.threshold, 2)),
          h("td", { class: "secondary" }, fmt.date(v.created)),
          h("td", { class: "r" }, v.production ? null : h("button", { class: "btn sm", onclick: async () => { await post("/models/promote", { version: v.version }); toast(`${v.version} promoted`); location.reload(); } }, "Promote")))))))));

    const prod = m.versions.find((v) => v.production) || m.versions[0];
    root.append(h("div", { class: "section-title" }, `2 · Threshold tuning of ${prod.version}`));
    const sw = prod.threshold_sweep || [];
    root.append(h("div", { class: "grid g-2-1" },
      card("Precision / recall / F0.5 vs acceptance threshold", "out-of-fold scores on training entities · other decision parameters held at their tuned values",
        sw.length ? lineChart(sw.map((r) => r.threshold), [
          { name: "F0.5", values: sw.map((r) => r.f05), color: SERIES[6] }, { name: "Precision", values: sw.map((r) => r.precision), color: SERIES[0] },
          { name: "Recall", values: sw.map((r) => r.recall), color: SERIES[1] }], { marker: prod.decision?.threshold, markerLabel: `chosen ${fmt.dec(prod.decision?.threshold, 2)}` })
          : empty("No sweep stored", "Versions trained before this feature have no threshold sweep. Retrain to record one.")),
      card("Chosen decision policy", "grid-searched for macro F0.5 — not a separate model", metricList([
        ["Mode", prod.decision?.mode], ["Acceptance threshold", fmt.dec(prod.decision?.threshold, 2)],
        ["No-match gate (best score must reach)", fmt.dec(prod.decision?.singleton_threshold, 2)], ["Minimum score gap", fmt.dec(prod.decision?.min_gap, 2)],
        ["Links per source cap", prod.decision?.max_per_source ?? "unlimited (multi-match)"], ["Exclusive assignment", prod.decision?.exclusive_assignment ? "on" : "off"]]))));

    const imp = Object.entries(prod.feature_importance || {}).slice(0, 14);
    const fs = prod.fs_weights || [];
    root.append(h("div", { class: "grid g-2" },
      card(prod.spec?.kind === "stack" ? "Ensemble meta-weights" : "Feature importance", prod.spec?.kind === "stack" ? "how much the meta-classifier trusts each input" : prettyModel(prod.model_name),
        imp.length ? withTable(hbarChart(imp.map(([k, v]) => ({ label: k, value: v })), { format: (v) => fmt.pct(v, 1) }), ["Input", "Weight"], imp.map(([k, v]) => [k, fmt.pct(v, 2)]))
          : empty("Not available", "This model family has no native importance.")),
      card("Fellegi–Sunter match weights", "log2(m/u) per agreement level, learned from training labels", fsWeights(fs))));
  }

  // ---- model families (validation experiments only) ------------------------------------------
  const res = exps?.results || [];
  const fam = res.filter(isModelExperiment);
  const latest = new Map();
  for (const r of fam) latest.set(`${r.dataset_kind || "unknown"}|${r.name}`, r);
  const rows = [...latest.values()].sort((a, b) => b.cv_f05 - a.cv_f05);
  root.append(h("div", { class: "section-title" }, "3 · Model families compared on validation"));
  root.append(card("Model families", "identical candidates, features, folds and decision tuning · selection by cross-validated F0.5 · holdout scored once",
    rows.length ? h("div", { class: "table-wrap", style: "margin:0 -20px" }, h("table", { class: "t" },
      h("thead", {}, h("tr", {}, ["Model", "Data", "Features", "Blocking", "Threshold", "CV F0.5", "Holdout F0.5", "Precision", "Recall", "Cand. recall", "Runtime", "Status"].map((x) => h("th", {}, x)))),
      h("tbody", {}, rows.map((r) => h("tr", { class: "click", onclick: () => { location.hash = `#/experiments/${r.experiment_id}`; } },
        h("td", {}, h("b", {}, prettyModel(r.name))), h("td", {}, kindBadge(r.dataset_kind || "unknown")), h("td", { class: "secondary" }, r.feature_set || "all"),
        h("td", { class: "mono", title: r.blocking_version }, String(r.blocking_version || "").slice(0, 8) || "—"),
        h("td", { class: "num" }, fmt.dec(r.threshold, 2)), h("td", { class: "num" }, h("b", {}, fmt.dec(r.cv_f05, 4))), h("td", { class: "num" }, fmt.dec(r.holdout_f05, 4)),
        h("td", { class: "num" }, fmt.pct(r.holdout_precision, 2)), h("td", { class: "num" }, fmt.pct(r.holdout_recall, 2)), h("td", { class: "num" }, fmt.pct(r.candidate_recall, 2)),
        h("td", { class: "num" }, fmt.secs(r.runtime_s)),
        h("td", {}, r.name === selected ? h("span", { class: "badge match" }, icon("check"), "Selected") : h("span", { class: "badge nomatch" }, "Evaluated")))))))
      : empty("No model experiments", "Run the experiment suite.", "python -m src.cli experiment", "beaker")));

  const hpo = res.filter((r) => r.suite === "hpo");
  if (hpo.length) {
    root.append(card("Hyper-parameter search", "randomised search scored by out-of-fold macro F0.5 (tuning runs, not separate models)",
      h("div", { class: "table-wrap", style: "margin:0 -20px" }, h("table", { class: "t" },
        h("thead", {}, h("tr", {}, ["Run", "Model family", "Data", "Best CV F0.5", "Trials / best parameters"].map((x) => h("th", {}, x)))),
        h("tbody", {}, hpo.map((r) => h("tr", { class: "click", onclick: () => { location.hash = `#/experiments/${r.experiment_id}`; } },
          h("td", { class: "mono" }, r.experiment_id), h("td", {}, prettyModel(r.model)), h("td", {}, kindBadge(r.dataset_kind || "unknown")),
          h("td", { class: "num" }, fmt.dec(r.cv_f05, 4)), h("td", { class: "secondary", style: "font-size:12px" }, r.notes))))))));
  }
}

function fsWeights(fs) {
  if (!fs.length) return empty("Not available", "");
  const fields = [...new Set(fs.map((r) => r.field))];
  const max = Math.max(...fs.map((r) => Math.abs(r.weight_bits)));
  return h("div", { class: "table-wrap", style: "margin:0 -20px;max-height:420px;overflow:auto" }, h("table", { class: "t" },
    h("thead", {}, h("tr", {}, ["Field", "Level", "m", "u", "Weight"].map((x) => h("th", {}, x)))),
    h("tbody", {}, fields.flatMap((f) => fs.filter((r) => r.field === f).map((r, i) => h("tr", {}, h("td", {}, i === 0 ? f.replace(/_/g, " ") : ""), h("td", { class: "mono" }, r.level),
      h("td", { class: "num" }, fmt.dec(r.m, 3)), h("td", { class: "num" }, fmt.dec(r.u, 3)),
      h("td", { style: "min-width:150px" }, h("div", { class: "row", style: "gap:8px;flex-wrap:nowrap" }, h("span", { class: "num", style: "width:44px;text-align:right" }, `${r.weight_bits > 0 ? "+" : ""}${r.weight_bits.toFixed(1)}`),
        h("div", { style: { height: "8px", width: `${Math.abs(r.weight_bits) / max * 90}px`, borderRadius: "4px", background: r.weight_bits >= 0 ? SERIES[2] : SERIES[1] } })))))))));
}

export { legend };
