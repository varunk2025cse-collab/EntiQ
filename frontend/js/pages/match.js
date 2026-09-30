import { h, icon, fmt, badge, empty } from "../dom.js";
import { get, state } from "../api.js";
import { meters, SERIES } from "../charts.js";
import { card, reasonLabel } from "./dashboard.js";

export async function render(root, { arg }) {
  if (!arg) {
    // default: the top-scoring accepted link of the current run
    try {
      const r = await get("/entities", { run_id: state.runId, decision: "MATCH", size: 1, sort: "-gap" });
      const e = r.items[0];
      if (e) { const d = await get(`/entities/${encodeURIComponent(e.s1_id)}`, { run_id: r.run_id }); if (d.matches[0]) { location.replace(`#/match/${encodeURIComponent(d.matches[0].match_id)}`); return; } }
    } catch { /* fallthrough */ }
    root.append(h("div", { class: "card" }, empty("Pick a pair to investigate", "Open any entity and click one of its candidates."))); return;
  }
  let d;
  try { d = await get(`/matches/${encodeURIComponent(arg)}/explanation`, { run_id: state.runId }); } catch (e) {
    root.append(h("div", { class: "card" }, empty("Pair not found", `${arg} is not a candidate pair in this run (${e.message}).`))); return;
  }
  const ex = d.explanation;
  const isMatch = ex.decision === "MATCH";
  root.append(
    h("div", { class: "row" }, h("a", { class: "btn ghost sm", href: `#/entity/${encodeURIComponent(ex.s1.entity_id)}` }, icon("arrowLeft"), `Back to ${ex.s1.entity_id}`)),
    h("div", { class: "card page-head" }, h("div", {}, h("h1", {}, "Match Investigation"), h("p", {}, ex.summary)),
      ex.ground_truth === null ? null : ex.ground_truth ? h("span", { class: "badge match" }, icon("check"), "Labelled: same business") : h("span", { class: "badge nomatch" }, icon("x"), "Labelled: different businesses")),
    h("section", { class: "card compare" }, side("Source 1 · Reference", ex.s1, ex.name_tokens.left, ex.address_tokens.left),
      h("div", { class: "vs" }, h("span", {}, "VS")), side(`${ex.target} · Candidate`, ex.candidate, ex.name_tokens.right, ex.address_tokens.right)),
    h("div", { class: "grid g-3" },
      card("Evidence", "0–100% similarity; flags are yes/no", meters(ex.evidence.map((e, i) => ({ label: e.label, value: e.kind === "gap" ? Math.max(0, e.value) : e.value,
        display: e.kind === "flag" ? (e.value >= 1 ? "Yes" : "No") : e.kind === "gap" ? fmt.dec(e.value, 3) : fmt.pct(e.value), color: SERIES[i % 8], title: e.description })))),
      card("Model votes", "independent scorers", h("div", {}, meters(Object.entries(ex.model_votes).map(([k, v], i) => ({ label: k, value: v, color: SERIES[[0, 6, 2, 3][i % 4]] }))),
        h("p", { class: "hint" }, "Only the calibrated primary model decides; Fellegi-Sunter and logistic regression are shown as independent second opinions."))),
      card("Decision", null, h("div", {},
        h("div", { class: `verdict ${isMatch ? "" : "no"}` }, h("div", { class: "ring" }, icon(isMatch ? "check" : "x")),
          h("div", {}, h("div", { class: "muted", style: "font-size:12px" }, "Final confidence"), h("div", { class: "conf" }, fmt.pct(ex.confidence))),
          h("div", { style: "margin-left:auto" }, badge(isMatch ? "MATCH" : "NO_MATCH"))),
        h("dl", { class: "kv" }, h("dt", {}, "Rank among candidates"), h("dd", {}, `#${ex.rank}`),
          h("dt", {}, "Acceptance threshold"), h("dd", {}, ex.decision_params ? fmt.dec(ex.decision_params.threshold, 2) : "—"),
          h("dt", {}, "Country agreement"), h("dd", {}, ex.country_agreement ? "Yes" : "No"),
          h("dt", {}, "Edit distance (names)"), h("dd", {}, String(ex.string_metrics.name_levenshtein)),
          h("dt", {}, "Jaro-Winkler (names)"), h("dd", {}, fmt.dec(ex.string_metrics.name_jaro_winkler, 3))),
        h("div", { class: "row", style: "gap:6px;margin-top:10px" }, ex.reason_codes.map((c) => h("span", { class: "code-chip", title: reasonLabel(c) }, c)))))),
    h("div", { class: "grid g-2" },
      ex.fellegi_sunter ? card("Probabilistic evidence (Fellegi-Sunter)", "log2 match weight per field", fsTable(ex.fellegi_sunter)) : null,
      card("Retrieved by", `${d.blockers.length} of 9 blocking passes`, h("div", { class: "blk" }, ALL_BLOCKERS.map(([k, l]) => {
        const on = d.blockers.some((b) => b.key === k);
        return h("div", { class: on ? "on" : "" }, icon(on ? "check" : "x"), l);
      })))),
    card("All pair features", "exact values the model saw", h("div", { class: "table-wrap" }, h("table", { class: "t" }, h("tbody", {},
      Object.entries(ex.features).filter(([k]) => !["s1_code", "blk_mask", "rank_s1"].includes(k)).map(([k, v]) => h("tr", {}, h("td", { class: "mono" }, k), h("td", { class: "num r" }, fmt.dec(v, 4)))))))));
}

const ALL_BLOCKERS = [["exact_name", "Normalized name key"], ["acronym", "Acronym"], ["name_word_tfidf", "Name token TF-IDF"], ["name_char_tfidf", "Name n-gram TF-IDF"],
  ["name_char_global", "Cross-country name n-gram"], ["address_tfidf", "Address TF-IDF"], ["name_address_char", "Name+address n-gram"], ["postal_token", "Postal code + name token"], ["semantic_lsa", "Semantic embedding (LSA)"]];

function side(title, r, nameToks, addrToks) {
  return h("div", { class: "side" }, h("h4", {}, title, h("span", { class: "id-link" }, r.entity_id)),
    h("div", { class: "big" }, r.business_name), h("div", { class: "secondary" }, r.business_address || "—"), h("div", { class: "muted", style: "font-size:12px" }, r.country || "—"),
    h("div", { style: "margin-top:12px" }, h("div", { class: "muted", style: "font-size:11.5px;margin-bottom:4px" }, "Name tokens (normalized)"), toks(nameToks)),
    h("div", { style: "margin-top:8px" }, h("div", { class: "muted", style: "font-size:11.5px;margin-bottom:4px" }, "Address tokens (normalized)"), toks(addrToks)),
    h("dl", { class: "kv" }, h("dt", {}, "Canonical name"), h("dd", { class: "mono" }, r.name_normalized), h("dt", {}, "Postal"), h("dd", { class: "mono" }, r.postal || "—"),
      h("dt", {}, "Landmark"), h("dd", {}, r.landmark || "—")));
}

function toks(list) {
  if (!list.length) return h("span", { class: "muted" }, "—");
  return h("div", {}, list.map((t) => h("span", { class: `tok ${t.shared ? "same" : "diff"}`, title: t.shared ? "shared token" : "only on this side" }, t.t)));
}

function fsTable(rows) {
  const max = Math.max(...rows.map((r) => Math.abs(r.weight_bits)), 1);
  return h("table", { class: "t" }, h("thead", {}, h("tr", {}, ["Field", "Agreement level", "Weight (bits)", ""].map((x) => h("th", {}, x)))),
    h("tbody", {}, rows.map((r) => h("tr", {}, h("td", {}, r.field.replace(/_/g, " ")), h("td", { class: "mono" }, r.level),
      h("td", { class: "num" }, `${r.weight_bits > 0 ? "+" : ""}${r.weight_bits.toFixed(2)}`),
      h("td", { style: "width:40%" }, h("div", { style: { height: "8px", width: `${Math.abs(r.weight_bits) / max * 100}%`, borderRadius: "4px",
        background: r.weight_bits >= 0 ? "var(--series-3)" : "var(--series-2)", marginLeft: r.weight_bits >= 0 ? "0" : "auto" } }))))));
}
