import { h, icon, fmt, badge, empty } from "../dom.js";
import { get, state } from "../api.js";

export async function render(root, { arg }) {
  if (!arg) {
    try {
      const r = await get("/entities", { run_id: state.runId, size: 1, sort: "-n_candidates", decision: "MATCH" });
      if (r.items[0]) { location.replace(`#/candidates/${encodeURIComponent(r.items[0].s1_id)}`); return; }
    } catch { /* ignore */ }
    root.append(h("div", { class: "card" }, empty("No run to explore", "Run a resolution first."))); return;
  }
  let d;
  try { d = await get(`/entities/${encodeURIComponent(arg)}/candidates`, { run_id: state.runId }); } catch (e) {
    root.append(h("div", { class: "card" }, empty("Entity not found", String(e.message)))); return;
  }
  const qIn = h("input", { class: "input", placeholder: "S1 entity id", value: arg, style: "width:200px" });
  qIn.addEventListener("keydown", (e) => { if (e.key === "Enter") location.hash = `#/candidates/${encodeURIComponent(qIn.value.trim().toUpperCase())}`; });
  const counts = Object.fromEntries(d.blocker_catalog.map((b) => [b.key, d.candidates.filter((c) => c.blockers.some((x) => x.key === b.key)).length]));
  root.append(
    h("div", { class: "card page-head" }, h("div", {}, h("h1", {}, "Candidate Explorer"),
      h("p", {}, "Every S2/S3 record the retrieval layer considered for this entity, which blocking passes found it, and how the matcher scored it.")),
    h("div", { class: "row" }, qIn, h("a", { class: "btn sm", href: `#/entity/${encodeURIComponent(arg)}` }, icon("building"), "Entity"))),
    h("div", { class: "card card-pad" }, h("div", { class: "row", style: "justify-content:space-between" },
      h("div", {}, h("div", { class: "row" }, h("span", { class: "ent-id" }, d.record.entity_id), h("span", { class: "chip blue" }, "S1")),
        h("div", { class: "ent-name" }, d.record.business_name), h("div", { class: "muted" }, d.record.business_address, " · ", d.record.country)),
      h("div", { class: "row" }, stat("Candidates", fmt.int(d.candidates.length)), stat("Accepted", fmt.int(d.candidates.filter((c) => c.selected).length)),
        stat("Median blockers / candidate", median(d.candidates.map((c) => c.n_blockers)))))),
    h("div", { class: "card card-pad" }, h("div", { class: "muted", style: "font-size:12px;margin-bottom:8px" }, "Candidates contributed by each blocking pass (a candidate can come from several)"),
      h("div", { class: "blk" }, d.blocker_catalog.map((b) => h("div", { class: counts[b.key] ? "on" : "" }, icon(counts[b.key] ? "check" : "x"), `${b.label}`, h("b", { style: "margin-left:auto" }, String(counts[b.key])))))),
    h("section", { class: "card" }, d.candidates.length ? d.candidates.map((c) => candRow(c, d.blocker_catalog)) : empty("No candidates", "Blocking found nothing plausible for this entity; it is a singleton by construction.")));
}

function candRow(c, catalog) {
  const on = new Set(c.blockers.map((b) => b.key));
  return h("div", { class: "cand" },
    h("div", { class: "rank" }, `#${c.rank}`),
    h("div", { style: "min-width:0" }, h("div", { class: "row", style: "gap:8px" }, h("a", { class: "id-link", href: `#/match/${encodeURIComponent(c.match_id)}` }, c.entity_id), h("span", { class: "chip" }, c.target),
      c.is_true ? h("span", { class: "badge match" }, icon("check"), "True link") : null),
    h("div", { style: "font-weight:550;margin-top:2px" }, c.business_name), h("div", { class: "muted", style: "font-size:12px" }, c.business_address || "—")),
    h("div", { style: "font-size:12px" }, h("div", { class: "muted", style: "margin-bottom:4px" }, "Retrieved by"),
      h("div", {}, catalog.filter((b) => on.has(b.key)).map((b) => h("div", { class: "secondary", style: "display:flex;gap:6px;align-items:center" }, h("span", { style: "color:var(--good-text);display:inline-flex;width:14px" }, icon("check")), b.label)))),
    h("div", {}, h("div", { class: "row", style: "justify-content:space-between" }, h("b", { class: "num" }, fmt.pct(c.p)), badge(c.selected ? "MATCH" : "REJECTED")),
      h("div", { class: "pbar" }, h("i", { style: { width: `${c.p * 100}%` } })),
      h("div", { class: "muted", style: "font-size:11.5px;margin-top:4px" }, `name ${fmt.pct(c.name_similarity, 0)} · addr ${fmt.pct(c.address_similarity, 0)}`)));
}

function stat(k, v) { return h("div", { class: "card", style: "padding:8px 12px;box-shadow:none;background:var(--surface-2)" }, h("div", { class: "muted", style: "font-size:11px" }, k), h("div", { style: "font-weight:700" }, v)); }
function median(a) { if (!a.length) return "—"; const s = [...a].sort((x, y) => x - y); return String(s[Math.floor(s.length / 2)]); }
