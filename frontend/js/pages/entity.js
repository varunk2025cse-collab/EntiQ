import { h, icon, fmt, badge, empty } from "../dom.js";
import { get, state } from "../api.js";
import { card, reasonLabel } from "../ui.js";

/** Decision state of one candidate: accepted link, near-threshold review, scored-but-rejected, or merely retrieved. */
export function candState(c, threshold) {
  if (c.selected) return "MATCH";
  if (threshold !== undefined && threshold !== null && c.p >= Math.max(0.3, threshold - 0.25)) return "REVIEW";
  if (c.p >= 0.05) return "REJECTED";
  return "CANDIDATE";
}

export function evidenceChips(c) {
  const out = [];
  if (c.name_similarity >= 0.95) out.push("Name ≈ exact"); else if (c.name_similarity >= 0.8) out.push("Name strong"); else if (c.name_similarity < 0.5) out.push("Name weak");
  if (c.address_similarity >= 0.8) out.push("Address strong"); else if (c.address_similarity < 0.4) out.push("Address weak");
  return out;
}

export async function render(root, { arg }) {
  if (!arg) { location.hash = "#/entities"; return; }
  let d;
  try { d = await get(`/entities/${encodeURIComponent(arg)}`, { run_id: state.runId }); } catch (e) {
    root.append(h("div", { class: "card" }, empty("Entity not found", `${arg}: ${e.message}`, null, "search"))); return;
  }
  const r = d.record;
  root.append(h("div", { class: "row" }, h("a", { class: "btn ghost sm", href: "#/entities" }, icon("arrowLeft"), "Entity Explorer")));
  if (d.kind !== "s1") { renderLinkedRecord(root, d); return; }

  const e = d.entity;
  const thr = d.decision?.threshold;
  const s2 = d.matches.filter((m) => m.target === "S2"), s3 = d.matches.filter((m) => m.target === "S3");
  root.append(
    h("section", { class: "card card-pad" }, h("div", { class: "row", style: "justify-content:space-between;align-items:flex-start;gap:16px" },
      h("div", { class: "ent-head" }, h("div", { class: "ico" }, icon("building")),
        h("div", {}, h("div", { class: "row" }, h("span", { class: "eyebrow", style: "margin:0" }, "Reference entity · Source 1"), h("span", { class: "ent-id" }, r.entity_id)),
          h("div", { class: "ent-name" }, r.business_name), h("div", { class: "ent-loc" }, icon("pin"), r.business_address || "No address", " · ", r.country || "no country"))),
      h("div", { style: "text-align:right" }, badge(e.decision), h("div", { class: "hint", style: "margin-top:6px" }, `${d.matches.length} linked · ${d.n_candidates} candidates retrieved`)))),
    h("div", { class: "grid g-3" },
      card("Decision", "why the system linked (or did not link) this entity", h("div", {},
        h("p", { class: "secondary", style: "margin:0 0 12px" }, explainDecision(e, thr)),
        h("div", { class: "metric-list" },
          h("div", {}, h("span", {}, "Best candidate score"), h("b", {}, fmt.pct(e.best))),
          h("div", {}, h("span", {}, "Runner-up score"), h("b", {}, fmt.pct(e.second))),
          h("div", {}, h("span", {}, "Score gap"), h("b", {}, fmt.dec(e.gap, 3))),
          h("div", {}, h("span", {}, "Acceptance threshold"), h("b", {}, thr !== undefined && thr !== null ? fmt.dec(thr, 2) : "—"))),
        h("div", { class: "row", style: "gap:6px;margin-top:12px" }, (e.reason_codes || []).map((c) => h("span", { class: "code-chip", title: c }, reasonLabel(c)))),
        d.ground_truth ? h("div", { class: "callout info", style: "margin-top:12px" }, icon("info"),
          h("div", {}, "Labelled data — true links: ", d.ground_truth.length ? h("b", { class: "mono" }, d.ground_truth.join(", ")) : h("b", {}, "none (singleton)"), `. Entity F0.5 ${fmt.dec(e.f05, 3)}.`)) : null)),
      linkCard("Matched in Source 2", s2), linkCard("Matched in Source 3", s3)),
    card("Candidates", `${d.n_candidates} records retrieved by blocking, ranked by calibrated score`, candTable(d.candidates, thr),
      h("a", { class: "btn sm", href: `#/candidates/${encodeURIComponent(r.entity_id)}` }, icon("nodes"), "Retrieval details")),
    card("Normalised representation", "what the matcher compares (originals are never modified)", h("dl", { class: "kv" },
      h("dt", {}, "Name — original"), h("dd", {}, r.business_name), h("dt", {}, "Name — canonical"), h("dd", { class: "mono" }, r.name_normalized),
      h("dt", {}, "Name — core"), h("dd", { class: "mono" }, r.name_core), h("dt", {}, "Address — original"), h("dd", {}, r.business_address || "—"),
      h("dt", {}, "Address — canonical"), h("dd", { class: "mono" }, r.address_normalized || "—"), h("dt", {}, "Postal code"), h("dd", { class: "mono" }, r.postal || "—"))));
}

function renderLinkedRecord(root, d) {
  const r = d.record;
  root.append(h("section", { class: "card card-pad" }, h("div", { class: "ent-head" }, h("div", { class: "ico" }, icon("database")),
    h("div", {}, h("div", { class: "row" }, h("span", { class: "eyebrow", style: "margin:0" }, `Source ${r.source.slice(1)} record`), h("span", { class: "ent-id" }, r.entity_id)),
      h("div", { class: "ent-name" }, r.business_name), h("div", { class: "ent-loc" }, icon("pin"), r.business_address || "—", " · ", r.country || "—")))),
  card("Candidate of these Source 1 entities", `${d.linked_to.length} entities`, d.linked_to.length ? h("div", { class: "table-wrap", style: "margin:0 -20px" }, h("table", { class: "t" },
    h("thead", {}, h("tr", {}, ["S1 entity", "Business", "Score", "Decision"].map((x) => h("th", {}, x)))),
    h("tbody", {}, d.linked_to.map((x) => h("tr", { class: "click", onclick: () => { location.hash = `#/match/${encodeURIComponent(`${x.entity_id}~${r.entity_id}`)}`; } },
      h("td", {}, h("span", { class: "id-link" }, x.entity_id)), h("td", {}, x.business_name), h("td", { class: "num" }, fmt.pct(x.p)), h("td", {}, badge(x.selected ? "MATCH" : "REJECTED")))))))
    : empty("Never retrieved", "No Source 1 entity retrieved this record as a candidate.")));
}

function linkCard(title, list) {
  return card(title, `${list.length} record${list.length === 1 ? "" : "s"}`, list.length ? h("div", { class: "rec-list", style: "margin:0 -20px -20px" }, list.map((m) =>
    h("a", { class: "rec", href: `#/match/${encodeURIComponent(m.match_id)}` }, h("div", { class: `src ${m.target}` }, m.target),
      h("div", { style: "min-width:0" }, h("div", { class: "rid" }, m.entity_id), h("div", { class: "rn" }, m.business_name), h("div", { class: "ra" }, m.business_address || "—")),
      h("div", { class: "score" }, fmt.pct(m.p), badge("MATCH")))))
    : h("p", { class: "muted", style: "margin:0" }, "No record from this source passed the decision gate."));
}

function explainDecision(e, thr) {
  if (e.decision === "MATCH") return `Linked to ${e.n_matches} record${e.n_matches > 1 ? "s" : ""}. The weakest accepted link scores ${fmt.pct(e.confidence)}${thr !== undefined && thr !== null ? ` against a ${fmt.dec(thr, 2)} threshold` : ""}.`;
  if ((e.reason_codes || []).includes("NO_CANDIDATES")) return "No candidate was retrieved by any blocking pass, so the entity is a singleton by construction.";
  return `The best candidate scored ${fmt.pct(e.best)}, below the validated acceptance gate. A correct "no match" earns full credit on true singletons; a false merge scores zero.`;
}

export function candTable(cands, thr) {
  const hasTruth = cands.length && cands[0].is_true !== null && cands[0].is_true !== undefined;
  if (!cands.length) return empty("No candidates", "Blocking retrieved nothing plausible for this entity.");
  return h("div", { class: "table-wrap", style: "margin:0 -20px" }, h("table", { class: "t" },
    h("thead", {}, h("tr", {}, ["#", "Record", "Business / address", "Country", "Name", "Address", "Evidence", "Score", "Decision", hasTruth ? "Label" : null]
      .filter(Boolean).map((x, i) => h("th", { class: i >= 4 && i <= 5 ? "r" : "" }, x)))),
    h("tbody", {}, cands.map((c) => h("tr", { class: "click", onclick: () => { location.hash = `#/match/${encodeURIComponent(c.match_id)}`; } },
      h("td", { class: "num muted" }, c.rank), h("td", {}, h("span", { class: "id-link" }, c.entity_id), h("span", { class: "sub" }, `Source ${c.target.slice(1)}`)),
      h("td", {}, h("div", { style: "font-weight:550" }, c.business_name), h("span", { class: "sub" }, c.business_address || "—")),
      h("td", {}, c.country || "—"),
      h("td", { class: "num r" }, fmt.pct(c.name_similarity, 0)), h("td", { class: "num r" }, fmt.pct(c.address_similarity, 0)),
      h("td", {}, h("div", { class: "row", style: "gap:4px" }, evidenceChips(c).map((x) => h("span", { class: "code-chip" }, x)))),
      h("td", { class: "num", style: "min-width:90px" }, h("b", {}, fmt.pct(c.p)), h("div", { class: "pbar" }, h("i", { style: { width: `${c.p * 100}%` } }))),
      h("td", {}, badge(candState(c, thr))),
      hasTruth ? h("td", {}, c.is_true ? h("span", { class: "badge match" }, icon("check"), "True link") : h("span", { class: "muted" }, "—")) : null)))));
}
