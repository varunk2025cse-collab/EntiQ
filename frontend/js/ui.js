// Shared page building blocks (one design language across all pages).
import { h, icon, fmt } from "./dom.js";

export function card(title, sub, body, action) {
  return h("section", { class: "card" },
    h("div", { class: "card-head" }, h("div", {}, h("h3", {}, title), sub ? h("span", { class: "sub" }, sub) : null), action || null),
    h("div", { class: "card-body" }, body));
}

export function pageHead(eyebrow, title, text, actions) {
  return h("div", { class: "page-head" }, h("div", {}, eyebrow ? h("div", { class: "eyebrow" }, eyebrow) : null, h("h1", {}, title), text ? h("p", {}, text) : null),
    actions ? h("div", { class: "row" }, actions) : null);
}

/** KPI tile. value === null renders an explicit "not available" message instead of a number. */
export function kpi({ icon: ic, tone = "", label, value, foot, na }) {
  return h("div", { class: "card kpi" },
    h("div", { class: "top" }, h("div", { class: `ico ${tone}` }, icon(ic)), h("div", { class: "lbl" }, label)),
    value === null || value === undefined ? h("div", { class: "val na" }, na || "Not measured") : h("div", { class: "val" }, value),
    foot ? h("div", { class: "foot" }, foot) : null);
}

export function metricList(rows) {
  return h("div", { class: "metric-list" }, rows.map(([k, v, note]) => h("div", {},
    h("span", {}, k, note ? h("span", { class: "hint" }, ` · ${note}`) : null),
    v === null || v === undefined ? h("b", { class: "na" }, "Not measured") : h("b", {}, v))));
}

const MODEL_NAMES = {
  lr: "Logistic regression", rf: "Random forest", et: "Extra trees", hgb: "Gradient boosting", xgb: "XGBoost", lgbm: "LightGBM",
  fs: "Fellegi–Sunter", hgb_no_semantic: "GBM − semantic features", hgb_no_context: "GBM − context features",
  hgb_no_sibling: "GBM − sibling features", hgb_with_sibling: "GBM + sibling features", lr_no_sibling: "LR − sibling features",
  hgb_hnm: "GBM + hard negatives", cascade: "Semantic cascade", stack: "Stacked ensemble", stack_hnm: "Stack + hard negatives",
};
export const prettyModel = (n) => MODEL_NAMES[n] || String(n || "").replace(/_/g, " ");

const REASONS = {
  NAME_EXACT: "Exact name", NAME_HIGH_SIMILARITY: "Name variation", ADDRESS_HIGH_SIMILARITY: "Address agreement",
  COUNTRY_MATCH: "Country match", MULTI_FIELD_AGREEMENT: "Multi-field agreement", STRONG_SCORE_GAP: "Clear winner",
  MUTUAL_BEST: "Mutual best", POSTAL_MATCH: "Postal code", STREET_NUMBER_MATCH: "Street number", SEMANTIC_SUPPORT: "Semantic support",
  TRADE_NAME_MATCH: "Trade name / DBA", ACRONYM_MATCH: "Acronym", COMMON_NAME: "Common name", NUMBER_CONFLICT: "Number conflict",
  POSTAL_CONFLICT: "Postal conflict", COUNTRY_CONFLICT: "Country conflict", LOW_CONFIDENCE_SINGLETON: "Low confidence",
  AMBIGUOUS_CANDIDATES: "Ambiguous", NO_CANDIDATES: "No candidates",
};
export const reasonLabel = (c) => REASONS[c] || String(c).replace(/_/g, " ").toLowerCase();

/** Experiment rows that are MODEL comparisons (not A/B checks, tuning trials or pipeline runs). */
export const MODEL_SUITES = new Set(["standard", "full", "baselines", "quick", ""]);
export const isModelExperiment = (r) => MODEL_SUITES.has(r.suite ?? "") && !String(r.name || "").includes("hpo");

/**
 * Validation quality of the production model. Test runs have no labels, so this comes from held-out TRAINING
 * entities of the same data source. Returns kind so the UI never shows a synthetic score as a competition result.
 */
export function validationMetrics(prodMeta, exps) {
  const res = exps?.results || [];
  const name = prodMeta?.model_name || prodMeta?.spec?.name;
  const rows = res.filter((r) => r.name === name && r.holdout_f05 !== null && r.holdout_f05 !== undefined && r.holdout_f05 !== "");
  const row = rows[rows.length - 1];
  if (!row) return { f05: null };
  return { f05: row.holdout_f05, precision: row.holdout_precision, recall: row.holdout_recall, singleton: row.holdout_singleton_accuracy,
    candidate_recall: row.candidate_recall, cv: row.cv_f05, kind: row.dataset_kind || prodMeta?.dataset_kind || "unknown", experiment: row.experiment_id };
}

export const pct = (v, d = 1) => (v === null || v === undefined || v === "" ? null : fmt.pct(Number(v), d));
export const dec = (v, d = 4) => (v === null || v === undefined || v === "" ? null : fmt.dec(Number(v), d));
