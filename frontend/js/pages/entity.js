import { h, icon, fmt, badge, empty } from "../dom.js";
import { get, state } from "../api.js";
import { card, reasonLabel } from "./dashboard.js";

export async function render(root, { arg }) {
  if (!arg) { location.hash = "#/entities"; return; }
  let d;
  try { d = await get(`/entities/${encodeURIComponent(arg)}`, { run_id: state.runId }); } catch (e) {
    root.append(h("div", { class: "card" }, empty("Entity not found", `${arg}: ${e.message}`))); return;
  }
  const r = d.record;
  root.append(h("div", { class: "row" }, h("a", { class: "btn ghost sm", href: "#/entities" }, icon("arrowLeft"), "Back to results")));
  if (d.kind !== "s1") {
    root.append(h("div", { class: "card page-head" }, h("div", {}, h("div", { class: "row" }, h("span", { class: "ent-id" }, r.entity_id), h("span", { class: "chip" }, r.source)),
      h("h1", { style: "font-size:22px;margin-top:6px" }, r.business_name), h("p", {}, r.business_address, " · ", r.country))),
    card("Appears as a candidate for", `${d.linked_to.length} S1 entities`, h("table", { class: "t" }, h("thead", {}, h("tr", {}, ["S1", "Business", "Score", "Decision"].map((x) => h("th", {}, x)))),
      h("tbody", {}, d.linked_to.map((x) => h("tr", { class: "click", onclick: () => { location.hash = `#/match/${encodeURIComponent(`${x.entity_id}~${r.entity_id}`)}`; } },
        h("td", {}, h("span", { class: "id-link" }, x.entity_id)), h("td", {}, x.business_name), h("td", { class: "num" }, fmt.pct(x.p)), h("td", {}, badge(x.selected ? "MATCH" : "NO_MATCH"))))))));
    return;
  }
  const e = d.entity;
  const s2 = d.matches.filter((m) => m.target === "S2"), s3 = d.matches.filter((m) => m.target === "S3");
  root.append(
    h("div", { class: "card page-head" }, h("div", { class: "row", style: "gap:14px" },
      h("div", { class: "ent-head", style: "padding:0" }, h("div", { class: "ico" }, icon("building")),
        h("div", {}, h("div", { class: "row" }, h("span", { class: "ent-id" }, r.entity_id), h("span", { class: "chip blue" }, "Reference · S1"), badge(e.decision)),
          h("h1", { style: "font-size:22px;margin-top:4px" }, r.business_name), h("div", { class: "ent-loc" }, icon("pin"), r.business_address || "—", " · ", r.country || "—")))),
    h("div", { class: "row" }, stat("Confidence", fmt.pct(e.confidence)), stat("Best / 2nd", `${fmt.dec(e.best, 3)} / ${fmt.dec(e.second, 3)}`), stat("Gap", fmt.dec(e.gap, 3)), stat("Candidates", fmt.int(d.n_candidates)))),
    h("div", { class: "grid g-3" },
      card("Original vs normalized", null, h("dl", { class: "kv" }, h("dt", {}, "Name (original)"), h("dd", {}, r.business_name), h("dt", {}, "Name (canonical)"), h("dd", { class: "mono" }, r.name_normalized),
        h("dt", {}, "Name (core)"), h("dd", { class: "mono" }, r.name_core), h("dt", {}, "Address (original)"), h("dd", {}, r.business_address || "—"),
        h("dt", {}, "Address (canonical)"), h("dd", { class: "mono" }, r.address_normalized || "—"), h("dt", {}, "Postal code"), h("dd", { class: "mono" }, r.postal || "—"))),
      card("Decision", null, h("div", {}, h("p", { class: "secondary", style: "margin-top:0" }, explainDecision(e)),
        h("div", { class: "row", style: "gap:6px" }, (e.reason_codes || []).map((c) => h("span", { class: "code-chip", title: reasonLabel(c) }, c))),
        d.ground_truth ? h("div", { class: "callout info", style: "margin-top:12px" }, h("div", { class: "ic" }, icon("info")),
          h("div", {}, "Labelled data: true links are ", d.ground_truth.length ? h("b", { class: "mono" }, d.ground_truth.join(", ")) : h("b", {}, "none (singleton)"), `. Entity F0.5 = ${fmt.dec(e.f05, 3)}.`)) : null)),
      card("Linked records", `${s2.length} S2 · ${s3.length} S3`, d.matches.length ? h("div", { class: "rec-list", style: "margin:-12px -18px -16px" }, d.matches.map(recLink))
        : h("p", { class: "muted" }, "No link: precision-first decision keeps this entity as a singleton."))),
    card("All candidates", `${d.n_candidates} retrieved by blocking · ranked by calibrated score`, candTable(d.candidates), h("a", { class: "btn sm", href: `#/candidates/${encodeURIComponent(r.entity_id)}` }, icon("nodes"), "Retrieval details")));
}

function stat(k, v) { return h("div", { class: "card", style: "padding:10px 14px;box-shadow:none;background:var(--surface-2)" }, h("div", { class: "muted", style: "font-size:11.5px" }, k), h("div", { style: "font-weight:700;font-size:16px" }, v)); }

function explainDecision(e) {
  if (e.decision === "MATCH") return `Linked to ${e.n_matches} record${e.n_matches > 1 ? "s" : ""}. The weakest accepted link scores ${fmt.pct(e.confidence)}; the best candidate leads the runner-up by ${fmt.dec(e.gap, 3)}.`;
  if ((e.reason_codes || []).includes("NO_CANDIDATES")) return "No candidate was retrieved by any blocking pass — no record in S2/S3 shares enough name or address signal.";
  return `Best candidate scored ${fmt.pct(e.best)}, below the validated acceptance gate. Predicting no match is worth a full 1.0 on true singletons, while a false merge scores 0.`;
}

function recLink(m) {
  return h("a", { class: "rec", href: `#/match/${encodeURIComponent(m.match_id)}`, style: "text-decoration:none;color:inherit" },
    h("div", { class: `src ${m.target}` }, m.target), h("div", { style: "min-width:0" }, h("div", { class: "rid" }, m.entity_id), h("div", { class: "rn" }, m.business_name), h("div", { class: "ra" }, m.business_address || "—")),
    h("div", { class: "score" }, fmt.pct(m.p), badge("MATCH")));
}

export function candTable(cands) {
  const hasTruth = cands.length && cands[0].is_true !== null && cands[0].is_true !== undefined;
  return h("div", { class: "table-wrap", style: "margin:0 -6px" }, h("table", { class: "t" },
    h("thead", {}, h("tr", {}, ["#", "Record", "Business", "Name sim.", "Address sim.", "Blockers", "Score", "Decision", hasTruth ? "Label" : null].filter(Boolean).map((x) => h("th", {}, x)))),
    h("tbody", {}, cands.map((c) => h("tr", { class: "click", onclick: () => { location.hash = `#/match/${encodeURIComponent(c.match_id)}`; } },
      h("td", { class: "num muted" }, c.rank), h("td", {}, h("span", { class: "id-link" }, c.entity_id)),
      h("td", {}, h("div", {}, c.business_name), h("div", { class: "muted", style: "font-size:12px" }, c.business_address || "—")),
      h("td", { class: "num" }, fmt.pct(c.name_similarity, 0)), h("td", { class: "num" }, fmt.pct(c.address_similarity, 0)),
      h("td", { class: "num", title: c.blockers.map((b) => b.label).join(", ") }, `${c.n_blockers} / 9`),
      h("td", { class: "num" }, h("b", {}, fmt.pct(c.p)), h("div", { class: "pbar" }, h("i", { style: { width: `${c.p * 100}%` } }))),
      h("td", {}, badge(c.selected ? "MATCH" : "REJECTED")),
      hasTruth ? h("td", {}, c.is_true ? h("span", { class: "badge match" }, icon("check"), "True") : h("span", { class: "muted" }, "—")) : null)))));
}
