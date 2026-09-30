import { h, icon, clear, fmt, empty } from "../dom.js";
import { get, state } from "../api.js";
import { hbarChart, withTable, SERIES } from "../charts.js";
import { card } from "./dashboard.js";

const FILTERS = [["", "", "All errors"], ["FALSE_POSITIVE", "", "False positives"], ["FALSE_NEGATIVE", "", "False negatives"],
  ["FALSE_POSITIVE", "singleton_false_merge", "Singleton false merges"], ["FALSE_POSITIVE", "ambiguous_candidate", "Ambiguous"],
  ["FALSE_NEGATIVE", "blocking_failure", "Blocking failures"], ["FALSE_POSITIVE", "name_collision", "Name conflict"], ["FALSE_POSITIVE", "address_collision", "Address conflict"],
  ["FALSE_POSITIVE", "common_business_name", "Common names"]];

export async function render(root, { params }) {
  let d;
  try { d = await get("/errors", { run_id: state.runId, size: 1 }); } catch (e) {
    root.append(h("div", { class: "card" }, empty("No run", String(e.message)))); return;
  }
  root.append(h("div", { class: "card page-head" }, h("div", {}, h("h1", {}, "Error Analysis"),
    h("p", {}, "Every wrong or missed link, automatically categorised by its most likely cause. Click an error to see the complete evidence."))));
  if (!d.available) {
    await holdoutErrors(root);
    return;
  }
  if (d.note) root.append(h("div", { class: "callout warn" }, h("div", { class: "ic" }, icon("info")), h("div", {}, `Note: ${d.note}. Use the held-out experiment reports for unbiased error rates.`)));
  const cats = d.categories;
  root.append(card("Error categories", null, withTable(hbarChart(cats.map((c) => ({ label: `${c.error_type === "FALSE_POSITIVE" ? "FP" : "FN"} · ${c.category.replace(/_/g, " ")}`,
    value: c.count, color: c.error_type === "FALSE_POSITIVE" ? SERIES[1] : SERIES[0] }))), ["Type", "Category", "Count"], cats.map((c) => [c.error_type, c.category, fmt.int(c.count)]))));
  const chips = h("div", { class: "filters" });
  const list = h("div", {});
  let cur = [params.get("type") || "", params.get("category") || ""];
  const load = async () => {
    const r = await get("/errors", { run_id: state.runId, type: cur[0], category: cur[1], size: 100 });
    clear(list).append(h("section", { class: "card" }, h("div", { class: "card-head" }, h("h3", {}, `${fmt.int(r.total)} errors`), h("span", { class: "muted", style: "font-size:12px" }, "showing up to 100")),
      h("div", { class: "table-wrap" }, h("table", { class: "t" }, h("thead", {}, h("tr", {}, ["Type", "Cause", "S1", "Candidate", "Score", "Name sim.", "Addr sim."].map((x) => h("th", {}, x)))),
        h("tbody", {}, r.items.map((e) => h("tr", { class: "click", onclick: () => { location.hash = e.category === "blocking_failure" ? `#/entity/${encodeURIComponent(e.s1_id)}` : `#/match/${encodeURIComponent(e.match_id)}`; } },
          h("td", {}, e.error_type === "FALSE_POSITIVE" ? h("span", { class: "badge fail" }, icon("x"), "False +") : h("span", { class: "badge review" }, icon("alert"), "Missed")),
          h("td", {}, h("span", { class: "code-chip" }, e.category)),
          h("td", {}, h("div", { class: "id-link" }, e.s1_id), h("div", { style: "font-size:12px" }, e.s1_name)),
          h("td", {}, h("div", { class: "id-link" }, e.other_id), h("div", { style: "font-size:12px" }, e.other_name)),
          h("td", { class: "num" }, e.p === null || e.p === undefined ? "not retrieved" : fmt.pct(e.p)), h("td", { class: "num" }, fmt.pct(e.x_name_best, 0)), h("td", { class: "num" }, fmt.pct(e.x_addr_best, 0)))))))));
  };
  for (const [t, c, label] of FILTERS) {
    const b = h("button", { class: `btn sm ${cur[0] === t && cur[1] === c ? "primary" : ""}`, type: "button", onclick: () => { cur = [t, c]; chips.querySelectorAll("button").forEach((x) => x.classList.toggle("primary", x === b)); load(); } }, label);
    chips.append(b);
  }
  root.append(chips, list);
  load();
}

/** Unlabelled run (test): show the held-out validation errors of the selected model's experiment. */
async function holdoutErrors(root) {
  const exps = await get("/experiments").catch(() => null);
  const sel = exps?.selected?.overrides?.model?.primary;
  const row = [...(exps?.results || [])].reverse().find((r) => r.name === sel) || (exps?.results || []).slice(-1)[0];
  if (!row) {
    root.append(h("div", { class: "card" }, empty("No labelled errors available", "This run is unlabelled and no validation experiment exists yet.", "python -m src.cli experiment"))); return;
  }
  const exp = await get(`/experiments/${row.experiment_id}`);
  const errs = exp.artifacts.errors || [];
  const cats = exp.artifacts.error_summary?.by_category || [];
  root.append(h("div", { class: "callout info" }, h("div", { class: "ic" }, icon("info")),
    h("div", {}, "The current run is on unlabelled test data. Showing errors on ", h("b", {}, "held-out training entities"), ` from ${row.experiment_id} (${row.name}), which the model never saw during fitting.`)),
  card("Validation error categories", `${exp.artifacts.error_summary?.false_positives ?? 0} false positives · ${exp.artifacts.error_summary?.false_negatives ?? 0} missed links`,
    withTable(hbarChart(cats.map((c) => ({ label: `${c.error_type === "FALSE_POSITIVE" ? "FP" : "FN"} · ${c.category.replace(/_/g, " ")}`, value: c.count,
      color: c.error_type === "FALSE_POSITIVE" ? SERIES[1] : SERIES[0] }))), ["Type", "Category", "Count"], cats.map((c) => [c.error_type, c.category, fmt.int(c.count)]))));
  const types = ["", "FALSE_POSITIVE", "FALSE_NEGATIVE"];
  let cur = "";
  const list = h("div", {});
  const draw = () => {
    const rows = errs.filter((e) => !cur || e.error_type === cur || e.category === cur).slice(0, 150);
    clear(list).append(h("section", { class: "card" }, rows.length ? rows.map(errRow) : empty("No errors in this view", "")));
  };
  const catNames = [...new Set(cats.map((c) => c.category))];
  const chips = h("div", { class: "filters" }, [...types, ...catNames].map((t) => {
    const b = h("button", { class: `btn sm ${t === cur ? "primary" : ""}`, onclick: () => { cur = t; chips.querySelectorAll("button").forEach((x) => x.classList.toggle("primary", x === b)); draw(); } },
      t === "" ? "All" : t === "FALSE_POSITIVE" ? "False positives" : t === "FALSE_NEGATIVE" ? "False negatives" : t.replace(/_/g, " "));
    return b;
  }));
  root.append(chips, list);
  draw();
}

function errRow(e) {
  const detail = h("div", { hidden: true, style: "padding:0 16px 14px" }, h("div", { class: "grid g-2", style: "gap:10px" },
    h("div", { class: "card card-pad", style: "box-shadow:none;background:var(--surface-2)" }, h("div", { class: "muted", style: "font-size:11px" }, `S1 · ${e.s1_id}`),
      h("div", { style: "font-weight:600" }, e.s1_name), h("div", { class: "muted" }, e.s1_address || "—", " · ", e.s1_country)),
    h("div", { class: "card card-pad", style: "box-shadow:none;background:var(--surface-2)" }, h("div", { class: "muted", style: "font-size:11px" }, e.other_id),
      h("div", { style: "font-weight:600" }, e.other_name), h("div", { class: "muted" }, e.other_address || "—", " · ", e.other_country))),
  h("div", { class: "row", style: "margin-top:10px;gap:14px;font-size:12.5px" },
    ["x_name_best", "x_addr_best", "combo_lsa_cos", "ctx_margin_s1"].filter((k) => e[k] !== undefined && e[k] !== null && e[k] !== "")
      .map((k) => h("span", {}, h("span", { class: "muted" }, `${k}: `), h("b", { class: "num" }, fmt.dec(Number(e[k]), 3)))),
    h("span", {}, h("span", { class: "muted" }, "tags: "), String(e.tags || "").replace(/[[\]']/g, ""))));
  const head = h("div", { class: "rec", style: "grid-template-columns: 110px minmax(0,1fr) minmax(0,1fr) 90px" },
    e.error_type === "FALSE_POSITIVE" ? h("span", { class: "badge fail" }, icon("x"), "False +") : h("span", { class: "badge review" }, icon("alert"), "Missed"),
    h("div", { style: "min-width:0" }, h("div", { class: "rn" }, e.s1_name), h("div", { class: "ra" }, e.other_name)),
    h("span", { class: "code-chip" }, e.category), h("div", { class: "score" }, e.p === "" || e.p === null || e.p === undefined ? "—" : fmt.pct(Number(e.p))));
  head.addEventListener("click", () => { detail.hidden = !detail.hidden; });
  return h("div", {}, head, detail);
}
