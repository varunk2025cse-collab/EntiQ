import { h, icon, clear, fmt, empty } from "../dom.js";
import { get, state } from "../api.js";
import { hbarChart, withTable, SERIES } from "../charts.js";
import { identityBanner, kindBadge } from "../app.js";
import { card, pageHead, reasonLabel } from "../ui.js";

const FILTERS = [["", "", "All errors"], ["FALSE_POSITIVE", "", "False positives"], ["FALSE_NEGATIVE", "", "False negatives"],
  ["FALSE_POSITIVE", "singleton_false_merge", "Singleton false merges"], ["FALSE_POSITIVE", "ambiguous_candidate", "Ambiguous"],
  ["FALSE_NEGATIVE", "blocking_failure", "Blocking failures"], ["FALSE_POSITIVE", "name_collision", "Name collision"],
  ["FALSE_POSITIVE", "address_collision", "Address collision"], ["FALSE_POSITIVE", "common_business_name", "Common names"],
  ["FALSE_NEGATIVE", "address_variation", "Weak address evidence"], ["FALSE_NEGATIVE", "lost_to_competitor", "Lost to competitor"]];

export async function render(root, { params }) {
  let d;
  try { d = await get("/errors", { run_id: state.runId, size: 1 }); } catch (e) {
    root.append(pageHead("Evaluate", "Error Analysis"), h("div", { class: "card" }, empty("No run", String(e.message), null, "alert"))); return;
  }
  if (!d.available) { await holdoutErrors(root); return; }
  root.append(pageHead("Evaluate", "Error Analysis", "Every wrong or missed link on this labelled run, categorised by its most likely cause. Select an error to inspect the full evidence."));
  if (d.note) root.append(h("div", { class: "callout warn" }, icon("info"), h("div", {}, `${d.note}. Use held-out experiment reports for unbiased error rates.`)));
  root.append(categoriesCard(d.categories, "this run"));
  const list = h("div", {});
  let cur = [params.get("type") || "", params.get("category") || ""];
  const load = async () => {
    const r = await get("/errors", { run_id: state.runId, type: cur[0], category: cur[1], size: 100 });
    clear(list).append(errorTable(r.items, `${fmt.int(r.total)} errors${r.total > 100 ? " · first 100 shown" : ""}`));
  };
  root.append(chips(FILTERS, cur, (t, c) => { cur = [t, c]; load(); }), list);
  load();
}

function chips(filters, cur, onPick) {
  const wrap = h("div", { class: "filters" });
  for (const [t, c, label] of filters) {
    const b = h("button", { class: `btn sm ${cur[0] === t && cur[1] === c ? "primary" : ""}`, type: "button", onclick: () => {
      wrap.querySelectorAll("button").forEach((x) => x.classList.toggle("primary", x === b)); onPick(t, c);
    } }, label);
    wrap.append(b);
  }
  return wrap;
}

function categoriesCard(cats, where) {
  if (!cats.length) return card("Error categories", where, empty("No errors", "Every labelled link was resolved correctly in this set.", null, "check"));
  return card("Error categories", where, withTable(hbarChart(cats.map((c) => ({ label: `${c.error_type === "FALSE_POSITIVE" ? "FP" : "FN"} · ${c.category.replace(/_/g, " ")}`,
    value: c.count, color: c.error_type === "FALSE_POSITIVE" ? SERIES[1] : SERIES[0] }))), ["Type", "Category", "Count"], cats.map((c) => [c.error_type, c.category, fmt.int(c.count)])));
}

function errorTable(items, title, linkable = true) {
  if (!items.length) return h("section", { class: "card" }, empty("No errors in this view", "Nothing matches the selected category.", null, "check"));
  return h("section", { class: "card" }, h("div", { class: "card-head", style: "padding-bottom:12px" }, h("div", {}, h("h3", {}, title))),
    h("div", { class: "table-wrap" }, h("table", { class: "t" },
      h("thead", {}, h("tr", {}, ["Type", "Cause", "Source 1", "Candidate", "Ground truth", "Prediction", "Score / threshold", "Reasons"].map((x) => h("th", {}, x)))),
      h("tbody", {}, items.map((e) => {
        const target = e.category === "blocking_failure" ? `#/entity/${encodeURIComponent(e.s1_id)}` : `#/match/${encodeURIComponent(`${e.s1_id}~${e.other_id}`)}`;
        const reasons = Array.isArray(e.reason_codes) ? e.reason_codes : String(e.reason_codes || "").replace(/[[\]']/g, "").split(",").map((x) => x.trim()).filter(Boolean);
        return h("tr", { class: linkable ? "click" : "", onclick: linkable ? () => { location.hash = target; } : null },
          h("td", {}, e.error_type === "FALSE_POSITIVE" ? h("span", { class: "badge fail" }, icon("x"), "False +") : h("span", { class: "badge review" }, icon("alert"), "Missed")),
          h("td", {}, h("span", { class: "code-chip" }, String(e.category).replace(/_/g, " "))),
          h("td", {}, h("span", { class: "id-link" }, e.s1_id), h("span", { class: "sub" }, e.s1_name), h("span", { class: "sub" }, e.s1_address || "")),
          h("td", {}, h("span", { class: "id-link" }, e.other_id), h("span", { class: "sub" }, e.other_name), h("span", { class: "sub" }, e.other_address || "")),
          h("td", {}, e.ground_truth || (e.error_type === "FALSE_POSITIVE" ? "different businesses" : "same business")),
          h("td", {}, e.prediction || (e.error_type === "FALSE_POSITIVE" ? "match" : "no match")),
          h("td", { class: "num" }, e.p === null || e.p === undefined || e.p === "" ? "not retrieved" : `${fmt.pct(Number(e.p))} / ${e.threshold !== undefined && e.threshold !== "" ? fmt.dec(Number(e.threshold), 2) : "—"}`),
          h("td", {}, h("div", { class: "row", style: "gap:4px" }, reasons.slice(0, 4).map((c) => h("span", { class: "code-chip", title: c }, reasonLabel(c))))));
      })))));
}

/** Unlabelled run (test): show the held-out validation errors of the selected model's experiment. */
async function holdoutErrors(root) {
  const exps = await get("/experiments").catch(() => null);
  const sel = exps?.selected?.overrides?.model?.primary;
  const row = [...(exps?.results || [])].reverse().find((r) => r.name === sel && r.suite !== "hpo") || [...(exps?.results || [])].reverse().find((r) => r.suite !== "hpo");
  root.append(pageHead("Evaluate", "Error Analysis", "Test data has no labels, so errors are measured on held-out training entities the model never saw during fitting."));
  if (!row) { root.append(h("div", { class: "card" }, empty("No labelled errors available", "No validation experiment exists yet.", "python -m src.cli experiment", "alert"))); return; }
  const exp = await get(`/experiments/${row.experiment_id}`);
  const errs = exp.artifacts.errors || [];
  const cats = exp.artifacts.error_summary?.by_category || [];
  root.append(identityBanner(exp.dataset_kind || "unknown"),
    h("div", { class: "callout info" }, icon("info"), h("div", {}, "Showing holdout errors from ", h("b", {}, `${row.experiment_id}`), ` (${row.name}). `, kindBadge(exp.dataset_kind || "unknown"))),
    categoriesCard(cats, `${exp.artifacts.error_summary?.false_positives ?? 0} false positives · ${exp.artifacts.error_summary?.false_negatives ?? 0} missed links (holdout)`));
  const list = h("div", {});
  const draw = (t, c) => list.replaceChildren(errorTable(errs.filter((e) => (!t || e.error_type === t) && (!c || e.category === c)).slice(0, 150),
    "Holdout errors (from the training split)", false));
  root.append(chips(FILTERS, ["", ""], draw), list);
  draw("", "");
}
