import { h, icon, clear, fmt, badge, empty } from "../dom.js";
import { get, state } from "../api.js";
import { identityBanner } from "../app.js";
import { pageHead, reasonLabel } from "../ui.js";

export async function render(root, { params }) {
  const f = { q: params.get("q") || "", country: params.get("country") || "", decision: params.get("decision") || "", truth: params.get("truth") || "", page: 1, sort: "s1_id" };
  let first;
  try { first = await get("/entities", { run_id: state.runId, size: 25, ...f }); } catch (e) {
    root.append(pageHead("Investigate", "Entity Explorer"), h("div", { class: "card" }, empty("No resolution run to explore", String(e.message), "python -m src.cli predict", "search"))); return;
  }
  const an = await get(`/analytics/${first.run_id}`);
  const countries = Object.keys(an.country_distribution || {});
  const hasTruth = first.items.length && "f05" in first.items[0];

  const qIn = h("input", { class: "input", type: "search", placeholder: "Entity ID, business name or address (an S2-/S3- id finds its S1)", value: f.q, style: "flex:1;min-width:280px" });
  const cSel = h("select", { class: "select" }, h("option", { value: "" }, "All countries"), countries.map((c) => h("option", { value: c, selected: c === f.country }, c)));
  const dSel = h("select", { class: "select" }, [["", "All decisions"], ["MATCH", "Matched"], ["MULTI", "Multi-match (2+ links)"], ["NO_MATCH", "No match"], ["REVIEW", "Needs review (ambiguous)"]]
    .map(([v, l]) => h("option", { value: v, selected: v === f.decision }, l)));
  const tSel = hasTruth ? h("select", { class: "select" }, [["", "All outcomes"], ["wrong", "Imperfect (F0.5 < 1)"], ["right", "Perfect"]].map(([v, l]) => h("option", { value: v, selected: v === f.truth }, l))) : null;
  const body = h("tbody");
  const info = h("span", { class: "chip" });
  const pager = h("div", { class: "pager" });
  const tableCard = h("section", { class: "card" });

  const load = async () => {
    Object.assign(f, { q: qIn.value.trim(), country: cSel.value, decision: dSel.value, truth: tSel?.value || "" });
    tableCard.classList.add("skeleton");
    const r = await get("/entities", { run_id: state.runId, size: 25, ...f });
    tableCard.classList.remove("skeleton");
    clear(body);
    for (const e of r.items) {
      body.append(h("tr", { class: "click", onclick: () => { location.hash = `#/entity/${encodeURIComponent(e.s1_id)}`; } },
        h("td", {}, h("span", { class: "id-link" }, e.s1_id)),
        h("td", {}, h("div", { style: "font-weight:600" }, e.business_name), h("span", { class: "sub" }, e.business_address || "—")),
        h("td", {}, e.country || "—"), h("td", { class: "num" }, `${e.n_s2} / ${e.n_s3}`), h("td", { class: "num" }, fmt.int(e.n_candidates)),
        h("td", { class: "num" }, fmt.pct(e.best)), h("td", { class: "num" }, fmt.dec(e.gap, 3)), h("td", {}, badge(e.decision)),
        h("td", {}, h("div", { class: "row", style: "gap:4px" }, (e.reason_codes || []).slice(0, 3).map((c) => h("span", { class: "code-chip", title: c }, reasonLabel(c))))),
        hasTruth ? h("td", { class: "num r" }, fmt.dec(e.f05, 3)) : null));
    }
    if (!r.items.length) body.append(h("tr", {}, h("td", { colspan: 10 }, empty("No entities match these filters", "Try a broader search.", null, "search"))));
    info.textContent = `${fmt.int(r.total)} entities · ${r.run_id}`;
    const pages = Math.max(1, Math.ceil(r.total / 25));
    clear(pager).append(`Page ${f.page} of ${fmt.int(pages)}`,
      h("button", { class: "btn sm", disabled: f.page <= 1, onclick: () => { f.page--; load(); } }, icon("arrowLeft")),
      h("button", { class: "btn sm", disabled: f.page >= pages, onclick: () => { f.page++; load(); } }, icon("arrowRight")));
  };
  for (const el of [cSel, dSel, tSel].filter(Boolean)) el.addEventListener("change", () => { f.page = 1; load(); });
  qIn.addEventListener("keydown", (e) => { if (e.key === "Enter") { f.page = 1; load(); } });

  tableCard.append(h("div", { class: "table-wrap" }, h("table", { class: "t" },
    h("thead", {}, h("tr", {}, ["S1 ID", "Reference business", "Country", "Links S2 / S3", "Candidates", "Best score", "Gap", "Decision", "Evidence", hasTruth ? "F0.5*" : null]
      .filter(Boolean).map((x) => h("th", {}, x)))), body)), pager);
  root.append(identityBanner(an.summary.dataset_kind),
    pageHead("Investigate", "Entity Explorer", "Every Source 1 reference entity with its linked Source 2 / Source 3 records, the candidates it competed against and the evidence behind each decision.", [info]),
    h("div", { class: "card card-pad", style: "padding:14px 16px" }, h("div", { class: "filters" }, qIn, cSel, dSel, tSel, h("button", { class: "btn primary", onclick: () => { f.page = 1; load(); } }, icon("search"), "Search"))),
    tableCard,
    hasTruth ? h("div", { class: "hint" }, "* Entity-level F0.5 against this dataset's labels. On a training-split run this is in-sample, not a validation score.") : null);
  load();
}
