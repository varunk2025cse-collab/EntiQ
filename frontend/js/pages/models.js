import { h, icon, clear, fmt, empty, toast, statusBadge } from "../dom.js";
import { get, post } from "../api.js";
import { hbarChart, withTable, SERIES } from "../charts.js";
import { card, prettyModel } from "./dashboard.js";

export async function render(root) {
  const [m, exps, jobs] = await Promise.all([get("/models"), get("/experiments").catch(() => null), get("/models/jobs").catch(() => [])]);
  const specs = ["hgb", "lr", "rf", "et", "fs", "hgb_hnm", "cascade", "stack", "stack_hnm", ...Object.entries(m.available_families).filter(([k, v]) => v && ["xgb", "lgbm"].includes(k)).map(([k]) => k)];
  const selected = exps?.selected?.overrides?.model?.primary;
  const modelSel = h("select", { class: "select" }, specs.map((s) => h("option", { value: s, selected: s === selected }, `${prettyModel(s)}${s === selected ? " (validation choice)" : ""}`)));
  const promote = h("input", { type: "checkbox", id: "promote" });
  const trainBtn = h("button", { class: "btn primary", onclick: async () => {
    trainBtn.disabled = true;
    try { const r = await post("/models/train", { model: modelSel.value, promote: promote.checked }); toast(`Training job ${r.job_id} queued — creates a NEW version`); setTimeout(() => location.reload(), 1200); }
    catch (e) { toast(e.message, true); } finally { trainBtn.disabled = false; }
  } }, icon("zap"), "Train new version");

  root.append(h("div", { class: "card page-head" }, h("div", {}, h("h1", {}, "Model Laboratory"),
    h("p", {}, "Versioned models. Training always creates a new version; competition outputs change only when a version is explicitly promoted.")),
  h("div", { class: "row" }, modelSel, h("label", { class: "row", style: "gap:6px;font-size:12.5px" }, promote, "Promote when done"), trainBtn)));

  if (jobs.length) {
    root.append(card("Training jobs", null, h("table", { class: "t" }, h("thead", {}, h("tr", {}, ["Job", "Model", "Status", "Stage", "Version"].map((x) => h("th", {}, x)))),
      h("tbody", {}, jobs.map((j) => h("tr", {}, h("td", { class: "mono" }, j.id), h("td", {}, prettyModel(j.model)), h("td", {}, statusBadge(j.status)), h("td", { class: "secondary" }, j.error ? j.error.split("\n")[0] : j.stage), h("td", { class: "mono" }, j.version || "—")))))));
  }
  if (!m.versions.length) { root.append(h("div", { class: "card" }, empty("No model trained yet", "Train one from here or run the CLI.", "python -m src.cli train"))); return; }

  const table = h("table", { class: "t" }, h("thead", {}, h("tr", {}, ["Version", "Model", "CV F0.5", "Precision", "Recall", "Holdout F0.5", "Threshold", "Mode", "Created", ""].map((x) => h("th", {}, x)))),
    h("tbody", {}, m.versions.map((v) => h("tr", {},
      h("td", {}, h("span", { class: "mono", style: "font-weight:600" }, v.version), v.production ? h("span", { class: "chip blue", style: "margin-left:6px" }, "Production") : null),
      h("td", {}, prettyModel(v.model_name)), h("td", { class: "num" }, fmt.dec(v.cv_f05, 4)), h("td", { class: "num" }, fmt.dec(v.cv_metrics?.precision, 4)),
      h("td", { class: "num" }, fmt.dec(v.cv_metrics?.recall, 4)), h("td", { class: "num" }, fmt.dec(v.holdout_f05, 4)),
      h("td", { class: "num" }, fmt.dec(v.decision?.threshold, 2)), h("td", {}, v.decision?.mode), h("td", { class: "secondary" }, fmt.date(v.created)),
      h("td", {}, v.production ? null : h("button", { class: "btn sm", onclick: async () => { await post("/models/promote", { version: v.version }); toast(`${v.version} promoted`); location.reload(); } }, "Promote"))))));
  root.append(card("Model versions", "metrics = cross-validation on all training entities", h("div", { class: "table-wrap" }, table)));

  const prod = m.versions.find((v) => v.production) || m.versions[0];
  const imp = Object.entries(prod.feature_importance || {}).slice(0, 15);
  const fs = prod.fs_weights || [];
  root.append(h("div", { class: "grid g-2" },
    card(`${prod.spec?.kind === "stack" ? "Ensemble meta-weights" : "Feature importance"} · ${prod.version}`,
      prod.spec?.kind === "stack" ? "how much the meta-classifier trusts each input" : prettyModel(prod.model_name),
      imp.length ? withTable(hbarChart(imp.map(([k, v]) => ({ label: k, value: v })), { format: (v) => fmt.pct(v, 1) }), ["Feature", "Importance"], imp.map(([k, v]) => [k, fmt.pct(v, 2)]))
        : h("p", { class: "muted" }, "This model family has no native importance; see the Fellegi-Sunter weights.")),
    card("Fellegi-Sunter match weights", "log2(m/u) learned from training labels", fsWeights(fs))),
  h("div", { class: "grid g-2" },
    card("Decision policy", "tuned for macro F0.5 on out-of-fold scores", h("dl", { class: "kv" }, Object.entries(prod.decision || {}).flatMap(([k, v]) => [h("dt", {}, k.replace(/_/g, " ")), h("dd", { class: "mono" }, v === null ? "unlimited" : String(v))]))),
    card("Training data & blocking", null, h("dl", { class: "kv" },
      h("dt", {}, "Candidate recall"), h("dd", {}, fmt.pct(prod.blocking_report?.candidate_recall, 3)), h("dt", {}, "Training candidates"), h("dd", {}, fmt.int(prod.blocking_report?.candidates)),
      h("dt", {}, "Features"), h("dd", {}, fmt.int(prod.feature_count)), h("dt", {}, "Training time"), h("dd", {}, fmt.secs(prod.train_seconds)),
      h("dt", {}, "Dataset fingerprint"), h("dd", { class: "mono" }, prod.dataset_fingerprint || "—")))));
}

function fsWeights(fs) {
  if (!fs.length) return h("p", { class: "muted" }, "—");
  const fields = [...new Set(fs.map((r) => r.field))];
  const max = Math.max(...fs.map((r) => Math.abs(r.weight_bits)));
  return h("div", { class: "table-wrap" }, h("table", { class: "t" }, h("thead", {}, h("tr", {}, ["Field", "Level", "m", "u", "Weight"].map((x) => h("th", {}, x)))),
    h("tbody", {}, fields.flatMap((f) => fs.filter((r) => r.field === f).map((r, i) => h("tr", {}, h("td", {}, i === 0 ? f.replace(/_/g, " ") : ""), h("td", { class: "mono" }, r.level),
      h("td", { class: "num" }, fmt.dec(r.m, 3)), h("td", { class: "num" }, fmt.dec(r.u, 3)),
      h("td", { style: "min-width:140px" }, h("div", { class: "row", style: "gap:8px;flex-wrap:nowrap" }, h("span", { class: "num", style: "width:48px;text-align:right" }, `${r.weight_bits > 0 ? "+" : ""}${r.weight_bits.toFixed(1)}`),
        h("div", { style: { height: "8px", width: `${Math.abs(r.weight_bits) / max * 90}px`, borderRadius: "4px", background: r.weight_bits >= 0 ? SERIES[2] : SERIES[1] } })))))))));
}
