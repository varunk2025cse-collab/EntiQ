import { h, icon, fmt, badge, empty } from "../dom.js";
import { get, state } from "../api.js";
import { card, pageHead } from "../ui.js";
import { candState, evidenceChips } from "./entity.js";

export async function render(root, { arg }) {
  if (!arg) {
    try {
      const r = await get("/entities", { run_id: state.runId, size: 1, sort: "-n_candidates", decision: "MATCH" });
      if (r.items[0]) { location.replace(`#/candidates/${encodeURIComponent(r.items[0].s1_id)}`); return; }
    } catch { /* ignore */ }
    root.append(pageHead("Investigate", "Candidate Explorer"), h("div", { class: "card" }, empty("No run to explore", "Run a resolution first.", null, "nodes"))); return;
  }
  let d, ent;
  try {
    [d, ent] = await Promise.all([get(`/entities/${encodeURIComponent(arg)}/candidates`, { run_id: state.runId }), get(`/entities/${encodeURIComponent(arg)}`, { run_id: state.runId })]);
  } catch (e) {
    root.append(pageHead("Investigate", "Candidate Explorer"), h("div", { class: "card" }, empty("Entity not found", String(e.message), null, "nodes"))); return;
  }
  const thr = ent.decision?.threshold;
  const qIn = h("input", { class: "input", placeholder: "S1 entity ID", value: arg, style: "width:220px" });
  qIn.addEventListener("keydown", (e) => { if (e.key === "Enter") location.hash = `#/candidates/${encodeURIComponent(qIn.value.trim().toUpperCase())}`; });
  const counts = Object.fromEntries(d.blocker_catalog.map((b) => [b.key, d.candidates.filter((c) => c.blockers.some((x) => x.key === b.key)).length]));
  const accepted = d.candidates.filter((c) => c.selected).length;
  root.append(
    pageHead("Investigate", "Candidate Explorer", "The retrieval stage for one entity: every Source 2 / Source 3 record the blocking passes proposed, which passes proposed it, and what the matcher decided. Most candidates are rejected — the system does not match everything it retrieves.",
      [qIn, h("a", { class: "btn", href: `#/entity/${encodeURIComponent(arg)}` }, icon("building"), "Open entity")]),
    h("section", { class: "card card-pad" }, h("div", { class: "row", style: "justify-content:space-between;gap:16px" },
      h("div", { class: "ent-head" }, h("div", { class: "ico" }, icon("building")),
        h("div", {}, h("div", { class: "row" }, h("span", { class: "eyebrow", style: "margin:0" }, "Reference · Source 1"), h("span", { class: "ent-id" }, d.record.entity_id)),
          h("div", { class: "ent-name" }, d.record.business_name), h("div", { class: "ent-loc" }, icon("pin"), d.record.business_address || "—", " · ", d.record.country || "—"))),
      h("div", { class: "row", style: "gap:24px" }, stat("Candidates scored", fmt.int(d.candidates.length)), stat("Accepted", fmt.int(accepted)), stat("Rejected", fmt.int(d.candidates.length - accepted))))),
    card("Blocking passes", "candidates contributed by each pass (a candidate can come from several; results are unioned)", h("div", { class: "blk" },
      d.blocker_catalog.map((b) => h("div", { class: counts[b.key] ? "on" : "" }, icon(counts[b.key] ? "check" : "x"), b.label, h("b", { style: "margin-left:auto" }, String(counts[b.key])))))),
    h("section", { class: "card" }, h("div", { class: "card-head", style: "padding-bottom:12px" }, h("div", {}, h("h3", {}, "Candidates"), h("span", { class: "sub" }, "ranked by calibrated match probability"))),
      d.candidates.length ? d.candidates.map((c) => candRow(c, d.blocker_catalog, thr)) : empty("No candidates", "Blocking found nothing plausible for this entity; it is a no-match by construction.", null, "nodes")));
}

function candRow(c, catalog, thr) {
  const on = new Set(c.blockers.map((b) => b.key));
  return h("div", { class: "cand" },
    h("div", { class: "rank" }, `#${c.rank}`),
    h("div", { style: "min-width:0" }, h("div", { class: "row", style: "gap:8px" }, h("a", { class: "id-link", href: `#/match/${encodeURIComponent(c.match_id)}` }, c.entity_id),
      h("span", { class: "chip" }, `Source ${c.target.slice(1)}`), c.is_true ? h("span", { class: "badge match" }, icon("check"), "True link") : null),
    h("div", { style: "font-weight:600;margin-top:2px" }, c.business_name), h("div", { class: "hint" }, c.business_address || "—", " · ", c.country || "—"),
    h("div", { class: "row", style: "gap:4px;margin-top:4px" }, evidenceChips(c).map((x) => h("span", { class: "code-chip" }, x)))),
    h("div", { style: "font-size:12.5px" }, h("div", { class: "hint", style: "margin-bottom:4px" }, `Retrieved by ${on.size} pass${on.size === 1 ? "" : "es"}`),
      catalog.filter((b) => on.has(b.key)).map((b) => h("div", { class: "secondary", style: "display:flex;gap:6px;align-items:center" }, h("span", { style: "color:var(--good-ink);display:inline-flex;width:14px" }, icon("check")), b.label))),
    h("div", {}, h("div", { class: "row", style: "justify-content:space-between" }, h("b", { class: "num" }, fmt.pct(c.p)), badge(candState(c, thr))),
      h("div", { class: "pbar" }, h("i", { style: { width: `${c.p * 100}%` } })),
      h("div", { class: "hint", style: "margin-top:4px" }, `name ${fmt.pct(c.name_similarity, 0)} · address ${fmt.pct(c.address_similarity, 0)}`)));
}

function stat(k, v) { return h("div", {}, h("div", { class: "hint" }, k), h("div", { style: "font-weight:700;font-size:20px" }, v)); }
