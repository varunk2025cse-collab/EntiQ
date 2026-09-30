import { h, icon, fmt, badge, empty } from "../dom.js";
import { get, state } from "../api.js";
import { meters, SERIES } from "../charts.js";
import { card, reasonLabel } from "../ui.js";

const LEVEL = { exact: "match", strong: "match", partial: "review", weak: "fail", match: "match", mismatch: "fail", unknown: "nomatch" };

function nameLevel(f) {
  if (f.name_core_eq >= 1) return "exact";
  const s = f.x_name_best ?? 0;
  return s >= 0.85 ? "strong" : s >= 0.6 ? "partial" : "weak";
}
function addrLevel(f) {
  const s = f.x_addr_best ?? 0;
  if (s >= 0.97) return "exact";
  return s >= 0.6 ? "partial" : "weak";
}

export async function render(root, { arg }) {
  if (!arg) {
    try {
      const r = await get("/entities", { run_id: state.runId, decision: "MATCH", size: 1, sort: "-gap" });
      const e = r.items[0];
      if (e) { const d = await get(`/entities/${encodeURIComponent(e.s1_id)}`, { run_id: r.run_id }); if (d.matches[0]) { location.replace(`#/match/${encodeURIComponent(d.matches[0].match_id)}`); return; } }
    } catch { /* fall through */ }
    root.append(h("div", { class: "card" }, empty("Pick a pair to investigate", "Open any entity and select one of its candidates.", null, "link"))); return;
  }
  let d;
  try { d = await get(`/matches/${encodeURIComponent(arg)}/explanation`, { run_id: state.runId }); } catch (e) {
    root.append(h("div", { class: "card" }, empty("Pair not found", `${arg} is not a candidate pair in this run (${e.message}).`, null, "link"))); return;
  }
  const ex = d.explanation;
  const f = ex.features;
  const isMatch = ex.decision === "MATCH";
  const thr = ex.decision_params?.threshold;
  const levels = [
    ["Business name", nameLevel(f), `${fmt.pct(f.x_name_best)} best name similarity`],
    ["Address", addrLevel(f), `${fmt.pct(f.x_addr_best)} best address similarity`],
    ["Country", f.country_conflict >= 1 ? "mismatch" : f.country_eq >= 1 ? "match" : "unknown", `${ex.s1.country || "—"} vs ${ex.candidate.country || "—"}`],
    ["Postal code", f.postal_conflict >= 1 ? "mismatch" : f.postal_eq >= 1 ? "match" : "unknown", ex.s1.postal || ex.candidate.postal ? `${ex.s1.postal || "—"} vs ${ex.candidate.postal || "—"}` : "not present"],
    ["Street number", f.num_first_conflict >= 1 ? "mismatch" : f.num_first_eq >= 1 ? "match" : "unknown", ""],
  ];
  const metrics = [
    ["Jaro–Winkler (core names)", fmt.dec(ex.string_metrics.name_jaro_winkler, 3)],
    ["Levenshtein distance (core names)", `${ex.string_metrics.name_levenshtein} edits · ${fmt.pct(f.name_lev_sim)} similar`],
    ["Token Jaccard (names)", fmt.pct(f.name_tok_jaccard)],
    ["TF-IDF cosine (name words)", fmt.pct(f.name_word_cos)],
    ["Character n-gram cosine (names)", fmt.pct(f.name_char_cos)],
    ["TF-IDF cosine (address words)", fmt.pct(f.addr_word_cos)],
    ["Character n-gram cosine (addresses)", fmt.pct(f.addr_char_cos)],
    ["Semantic similarity (LSA, name + address)", fmt.pct(f.combo_lsa_cos)],
  ];
  root.append(
    h("div", { class: "row" }, h("a", { class: "btn ghost sm", href: `#/entity/${encodeURIComponent(ex.s1.entity_id)}` }, icon("arrowLeft"), ex.s1.entity_id)),
    h("div", { class: "page-head" }, h("div", {}, h("div", { class: "eyebrow" }, "Match investigation"), h("h1", {}, isMatch ? "Why these records were linked" : "Why these records were not linked"),
      h("p", {}, ex.summary)),
    h("div", { class: "row" }, badge(isMatch ? "MATCH" : "REJECTED"),
      ex.ground_truth === null || ex.ground_truth === undefined ? null
        : ex.ground_truth ? h("span", { class: "badge match" }, icon("check"), "Ground truth: same business") : h("span", { class: "badge nomatch" }, icon("x"), "Ground truth: different"))),
    h("section", { class: "card compare" }, side("Reference · Source 1", ex.s1, ex.name_tokens.left, ex.address_tokens.left),
      h("div", { class: "vs" }, h("span", {}, "VS")), side(`Candidate · Source ${ex.target.slice(1)}`, ex.candidate, ex.name_tokens.right, ex.address_tokens.right)),
    h("div", { class: "grid g-3" },
      card("Evidence by field", "shared tokens are green, differing tokens orange", h("div", {}, levels.map(([k, lv, note]) =>
        h("div", { class: "strength" }, h("span", { class: "secondary" }, k), h("span", { class: "hint" }, note), h("span", { class: `badge ${LEVEL[lv]}` }, lv))))),
      card("Similarity metrics", "computed on normalised fields", h("div", { class: "metric-list" }, metrics.map(([k, v]) => h("div", {}, h("span", {}, k), h("b", {}, v))))),
      card("Decision", "calibrated model score vs validated threshold", h("div", {},
        h("div", { class: `verdict ${isMatch ? "" : "no"}` }, h("div", { class: "ring" }, icon(isMatch ? "check" : "x")),
          h("div", {}, h("div", { class: "hint" }, "Match probability"), h("div", { class: "conf" }, fmt.pct(ex.confidence))),
          h("div", { style: "margin-left:auto;text-align:right" }, h("div", { class: "hint" }, "Threshold"), h("b", { style: "font-size:18px" }, thr !== undefined ? fmt.dec(thr, 2) : "—"))),
        h("div", { class: "metric-list", style: "margin-top:8px" },
          h("div", {}, h("span", {}, "Rank among this entity's candidates"), h("b", {}, `#${ex.rank}`)),
          h("div", {}, h("span", {}, "Lead over next candidate"), h("b", {}, fmt.dec(f.ctx_margin_s1, 3)))),
        h("div", { class: "hint", style: "margin:10px 0 6px" }, "Reason codes"),
        h("div", { class: "row", style: "gap:6px" }, ex.reason_codes.length ? ex.reason_codes.map((c) => h("span", { class: "code-chip", title: c }, reasonLabel(c))) : h("span", { class: "muted" }, "none"))))),
    h("div", { class: "grid g-2" },
      card("Independent model votes", "only the calibrated primary model decides; the others are second opinions",
        meters(Object.entries(ex.model_votes).map(([k, v], i) => ({ label: k, value: v, color: SERIES[[0, 6, 2, 3][i % 4]] })))),
      ex.fellegi_sunter ? card("Probabilistic evidence (Fellegi–Sunter)", "log2(m/u) weight per field — positive supports a match", fsTable(ex.fellegi_sunter))
        : card("Probabilistic evidence", null, empty("Unavailable", "The model version has no Fellegi–Sunter component."))),
    card("Retrieved by", `${d.blockers.length} of 9 blocking passes found this candidate`, h("div", { class: "blk" }, ALL_BLOCKERS.map(([k, l]) => {
      const on = d.blockers.some((b) => b.key === k);
      return h("div", { class: on ? "on" : "" }, icon(on ? "check" : "x"), l);
    }))));
}

const ALL_BLOCKERS = [["exact_name", "Normalised name key"], ["acronym", "Acronym"], ["name_word_tfidf", "Name token TF-IDF"], ["name_char_tfidf", "Name n-gram TF-IDF"],
  ["name_char_global", "Cross-country name n-gram"], ["address_tfidf", "Address TF-IDF"], ["name_address_char", "Name + address n-gram"], ["postal_token", "Postal code + name token"], ["semantic_lsa", "Semantic embedding (LSA)"]];

function side(title, r, nameToks, addrToks) {
  return h("div", { class: "side" }, h("h4", {}, title, h("span", { class: "id-link", style: "letter-spacing:0;text-transform:none" }, r.entity_id)),
    h("div", { class: "big" }, r.business_name), h("div", { class: "secondary" }, r.business_address || "—"), h("div", { class: "hint" }, r.country || "—"),
    h("div", { style: "margin-top:12px" }, h("div", { class: "hint", style: "margin-bottom:4px" }, "Name tokens (normalised)"), toks(nameToks)),
    h("div", { style: "margin-top:8px" }, h("div", { class: "hint", style: "margin-bottom:4px" }, "Address tokens (normalised)"), toks(addrToks)),
    h("dl", { class: "kv" }, h("dt", {}, "Canonical name"), h("dd", { class: "mono" }, r.name_normalized), h("dt", {}, "Postal code"), h("dd", { class: "mono" }, r.postal || "—"),
      h("dt", {}, "Landmark"), h("dd", {}, r.landmark || "—")));
}

function toks(list) {
  if (!list.length) return h("span", { class: "muted" }, "—");
  return h("div", {}, list.map((t) => h("span", { class: `tok ${t.shared ? "same" : "diff"}`, title: t.shared ? "shared with the other record" : "only on this side" }, t.t)));
}

function fsTable(rows) {
  const max = Math.max(...rows.map((r) => Math.abs(r.weight_bits)), 1);
  return h("div", { class: "table-wrap", style: "margin:0 -20px" }, h("table", { class: "t" }, h("thead", {}, h("tr", {}, ["Field", "Agreement level", "Weight (bits)", ""].map((x) => h("th", {}, x)))),
    h("tbody", {}, rows.map((r) => h("tr", {}, h("td", {}, r.field.replace(/_/g, " ")), h("td", { class: "mono" }, r.level),
      h("td", { class: "num" }, `${r.weight_bits > 0 ? "+" : ""}${r.weight_bits.toFixed(2)}`),
      h("td", { style: "width:40%" }, h("div", { style: { height: "8px", width: `${Math.abs(r.weight_bits) / max * 100}%`, borderRadius: "4px",
        background: r.weight_bits >= 0 ? "var(--series-3)" : "var(--series-2)", marginLeft: r.weight_bits >= 0 ? "0" : "auto" } })))))));
}
