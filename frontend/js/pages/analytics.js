import { h, fmt, empty } from "../dom.js";
import { get, state } from "../api.js";
import { barChart, hbarChart, donut, withTable, SEQ, SERIES, meters } from "../charts.js";
import { card, reasonLabel } from "./dashboard.js";

export async function render(root) {
  let an;
  try { an = await get(`/analytics/${state.runId || "latest"}`); } catch (e) {
    root.append(h("div", { class: "card" }, empty("No analytics yet", String(e.message)))); return;
  }
  const s = an.summary, c = s.counts, b = an.blocking || {};
  const blockers = Object.entries(b.blockers || {}).sort((x, y) => y[1].candidates - x[1].candidates);
  const hasRecall = blockers.length && blockers[0][1].recall !== undefined;
  const bins = an.confidence_histogram.bins;
  root.append(
    h("div", { class: "card page-head" }, h("div", {}, h("h1", {}, "Analytics"), h("p", {}, `${an.run_id} · ${s.dataset} split · model ${s.model_version || "—"}`))),
    h("div", { class: "grid g-4" },
      tile("Reduction ratio", fmt.pct(b.union?.reduction_ratio, 3), "of all S1×(S2∪S3) pairs never compared"),
      tile("Candidates per S1", fmt.dec(b.union?.avg_candidates_per_s1, 1), `${fmt.int(b.union?.s1_without_candidates)} entities with none`),
      tile("Multi-link entities", fmt.int(s.graph?.s1_with_multiple_links), `${fmt.int(s.graph?.s1_linked_to_both_sources)} linked to both S2 and S3`),
      b.pre_pruning ? tile("Cascade pruning", `−${fmt.pct(1 - b.union.candidates / b.pre_pruning.candidates, 0)}`,
        `${fmt.int(b.pre_pruning.candidates)} → ${fmt.int(b.union.candidates)} candidates before the full feature engine`)
        : tile("Records claimed twice", fmt.int(s.graph?.records_claimed_by_multiple_s1), "S2/S3 records linked to more than one S1")),
    h("div", { class: "grid g-2" },
      card("Candidate volume by blocking pass", hasRecall ? "with true-link recall (labelled data)" : "union is de-duplicated",
        withTable(hbarChart(blockers.map(([k, v]) => ({ label: v.label, value: v.candidates, sub: hasRecall ? `recall ${fmt.pct(v.recall)}` : "" }))),
          ["Blocker", "Candidates", "Recall", "Unique true links"], blockers.map(([, v]) => [v.label, fmt.int(v.candidates), v.recall !== undefined ? fmt.pct(v.recall, 2) : "—", v.unique_true_found ?? "—"]))),
      card("Best-candidate score per entity", "calibrated probability of each S1's top candidate",
        withTable(barChart({ labels: bins.map((x) => x.split("-")[0]), values: an.best_score_histogram, colors: SEQ, tipLabel: (i) => `entities whose best candidate scores ${bins[i]}` }),
          ["Best score", "Entities"], bins.map((x, i) => [x, fmt.int(an.best_score_histogram[i])])))),
    h("div", { class: "grid g-3" },
      card("Candidates per entity", null, withTable(barChart({ labels: an.candidates_per_entity_bins, values: an.candidates_per_entity, colors: [SERIES[0]] }),
        ["Candidates", "Entities"], an.candidates_per_entity_bins.map((x, i) => [x, fmt.int(an.candidates_per_entity[i])]))),
      card("Links per entity", null, withTable(barChart({ labels: Object.keys(an.links_per_entity || {}).map((k) => (k === "5" ? "5+" : k)), values: Object.values(an.links_per_entity || {}), colors: [SERIES[2]] }),
        ["Links", "Entities"], Object.entries(an.links_per_entity || {}).map(([k, v]) => [k, fmt.int(v)]))),
      card("Accepted links by source", null, donut([{ label: "Source 2", value: c.s2_links }, { label: "Source 3", value: c.s3_links }], { centerLabel: "Links" }))),
    h("div", { class: "grid g-2" },
      card("Score gap vs decision", "ambiguity drives precision-first abstention", withTable(barChart({ labels: an.gap_vs_decision.map((g) => g.gap),
        values: an.gap_vs_decision.map((g) => (g.total ? g.matched / g.total : 0)), colors: SEQ.slice(2), format: (v) => fmt.pct(v, 0) }),
      ["Best−second gap", "Entities", "Matched"], an.gap_vs_decision.map((g) => [g.gap, fmt.int(g.total), fmt.int(g.matched)]))),
      card("Reason codes", "across accepted links", meters(Object.entries(an.reason_codes || {}).sort((x, y) => y[1] - x[1]).slice(0, 10)
        .map(([k, v], i) => ({ label: reasonLabel(k), value: v, max: c.matched_entities || 1, display: fmt.int(v), color: SERIES[i % 8] }))))),
    an.error_categories ? card("Error categories", s.metrics_note || "labelled run", withTable(hbarChart(an.error_categories.map((e) => ({ label: `${e.error_type === "FALSE_POSITIVE" ? "FP" : "FN"} · ${e.category.replace(/_/g, " ")}`,
      value: e.count, color: e.error_type === "FALSE_POSITIVE" ? SERIES[1] : SERIES[0] }))), ["Type", "Category", "Count"], an.error_categories.map((e) => [e.error_type, e.category, fmt.int(e.count)]))) : null);
}

function tile(k, v, foot) {
  return h("div", { class: "card kpi" }, h("div", {}, h("div", { class: "lbl" }, k), h("div", { class: "val", style: "font-size:26px" }, v), h("div", { class: "foot" }, foot)));
}
