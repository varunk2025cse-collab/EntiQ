import { h, fmt, empty } from "../dom.js";
import { get, state } from "../api.js";
import { barChart, hbarChart, donut, withTable, SEQ, SERIES, meters } from "../charts.js";
import { identityBanner, kindBadge } from "../app.js";
import { card, pageHead, kpi, reasonLabel, metricList } from "../ui.js";

export async function render(root) {
  let an;
  try { an = await get(`/analytics/${state.runId || "latest"}`); } catch (e) {
    root.append(pageHead("Evaluate", "Analytics"), h("div", { class: "card" }, empty("No analytics yet", String(e.message), "python -m src.cli predict", "chart"))); return;
  }
  const s = an.summary, c = s.counts, b = an.blocking || {};
  const kind = s.dataset_kind || "unknown";
  const t = an.timings || {};
  const compute = (t.total_seconds || 0) + (t.run?.total_seconds || 0);
  const blockers = Object.entries(b.blockers || {}).sort((x, y) => y[1].candidates - x[1].candidates);
  const hasRecall = blockers.length && blockers[0][1].recall !== undefined;
  const bins = an.confidence_histogram.bins;
  const m = s.metrics;
  const countries = Object.entries(an.country_distribution || {}).sort((x, y) => y[1] - x[1]);

  root.append(identityBanner(kind), pageHead("Evaluate", "Analytics", `Computed from run ${an.run_id} (${s.dataset} split, model ${s.model_version || "—"}).`, [kindBadge(kind)]));

  root.append(h("div", { class: "kpis" },
    kpi({ icon: "nodes", label: "Reduction ratio", value: fmt.pct(b.union?.reduction_ratio, 3), foot: "of all S1 × (S2 ∪ S3) pairs never compared" }),
    kpi({ icon: "layers", label: "Candidates per entity", value: fmt.dec(b.union?.avg_candidates_per_s1, 1), foot: `${fmt.int(b.union?.s1_without_candidates)} entities with none` }),
    b.pre_pruning ? kpi({ icon: "filter", label: "Cascade filter", value: `−${fmt.pct(1 - b.union.candidates / Math.max(1, b.pre_pruning.candidates), 0)}`,
      foot: `${fmt.int(b.pre_pruning.candidates)} → ${fmt.int(b.union.candidates)} before feature engineering` }) : null,
    kpi({ icon: "zap", label: "Throughput", value: `${fmt.int(Math.round(c.s1 / Math.max(compute, 1e-6)))}/s`, foot: `S1 entities per second · ${fmt.secs(compute)} total` }),
    kpi({ icon: "link", tone: "green", label: "Multi-link entities", value: fmt.int(s.graph?.s1_with_multiple_links), foot: `${fmt.int(s.graph?.s1_linked_to_both_sources)} linked in both S2 and S3` })));

  if (m) {
    root.append(card("Quality on this run's labels", s.metrics_note || "labelled run", metricList([
      ["F0.5 (macro, official metric)", fmt.dec(m.f05, 4)], ["Precision (macro)", fmt.pct(m.precision, 2)], ["Recall (macro)", fmt.pct(m.recall, 2)],
      ["Candidate recall", fmt.pct(m.candidate_recall, 2)], ["No-match accuracy", fmt.pct(m.singleton_accuracy, 2)],
      ["True / false / missed links", `${fmt.int(m.true_matches)} / ${fmt.int(m.false_matches)} / ${fmt.int(m.missed_matches)}`]])));
  }

  root.append(h("div", { class: "grid g-2" },
    card("Candidate volume by blocking pass", hasRecall ? "with true-link recall of each pass (labelled data)" : "union is de-duplicated",
      withTable(hbarChart(blockers.map(([, v]) => ({ label: v.label, value: v.candidates, sub: hasRecall ? `recall ${fmt.pct(v.recall)}` : "" }))),
        ["Blocking pass", "Candidates", "Recall", "Unique true links"], blockers.map(([, v]) => [v.label, fmt.int(v.candidates), v.recall !== undefined ? fmt.pct(v.recall, 2) : "—", v.unique_true_found ?? "—"]))),
    card("Best-candidate score per entity", "calibrated probability of each entity's top candidate",
      withTable(barChart({ labels: bins.map((x) => x.split("-")[0]), values: an.best_score_histogram, colors: SEQ, tipLabel: (i) => `entities whose best candidate scores ${bins[i]}` }),
        ["Best score", "Entities"], bins.map((x, i) => [x, fmt.int(an.best_score_histogram[i])])))));
  root.append(h("div", { class: "grid g-3" },
    card("Candidates per entity", "retrieval volume after the cascade filter", withTable(barChart({ labels: an.candidates_per_entity_bins, values: an.candidates_per_entity, colors: [SERIES[0]] }),
      ["Candidates", "Entities"], an.candidates_per_entity_bins.map((x, i) => [x, fmt.int(an.candidates_per_entity[i])]))),
    card("Links per entity", "0 = no-match decision", withTable(barChart({ labels: Object.keys(an.links_per_entity || {}).map((k) => (k === "5" ? "5+" : k)), values: Object.values(an.links_per_entity || {}), colors: [SERIES[2]] }),
      ["Links", "Entities"], Object.entries(an.links_per_entity || {}).map(([k, v]) => [k, fmt.int(v)]))),
    card("Accepted links by source", null, donut([{ label: "Source 2", value: c.s2_links }, { label: "Source 3", value: c.s3_links }], { centerLabel: "links" }))));
  root.append(h("div", { class: "grid g-3" },
    card("Confidence of accepted links", "calibrated match probability", withTable(barChart({ labels: bins.map((x) => x.split("-")[0]), values: an.confidence_histogram.counts, colors: SEQ }),
      ["Confidence", "Links"], bins.map((x, i) => [x, fmt.int(an.confidence_histogram.counts[i])]))),
    card("Match rate by country", "share of S1 entities with at least one link", countries.length ? withTable(hbarChart(countries.map(([k, v], i) => ({ label: k, value: an.country_match_rate?.[k] ?? 0, color: SERIES[i % 8], sub: `${fmt.int(v)} entities` })), { format: (v) => fmt.pct(v), max: 1 }),
      ["Country", "Entities", "Match rate"], countries.map(([k, v]) => [k, fmt.int(v), fmt.pct(an.country_match_rate?.[k])])) : empty("No data", "")),
    card("Score gap vs decision", "share matched, by lead of the best candidate over the runner-up", withTable(barChart({ labels: an.gap_vs_decision.map((g) => g.gap),
      values: an.gap_vs_decision.map((g) => (g.total ? g.matched / g.total : 0)), colors: SEQ.slice(2), format: (v) => fmt.pct(v, 0) }),
    ["Best − second gap", "Entities", "Matched"], an.gap_vs_decision.map((g) => [g.gap, fmt.int(g.total), fmt.int(g.matched)])))));
  root.append(h("div", { class: "grid g-2" },
    card("Evidence behind accepted links", "reason codes of each entity's strongest link", meters(Object.entries(an.reason_codes || {}).sort((x, y) => y[1] - x[1]).slice(0, 10)
      .map(([k, v], i) => ({ label: reasonLabel(k), value: v, max: c.matched_entities || 1, display: fmt.int(v), color: SERIES[i % 8] })))),
    an.error_categories ? card("Error categories", s.metrics_note || "labelled run", withTable(hbarChart(an.error_categories.map((e) => ({ label: `${e.error_type === "FALSE_POSITIVE" ? "FP" : "FN"} · ${e.category.replace(/_/g, " ")}`,
      value: e.count, color: e.error_type === "FALSE_POSITIVE" ? SERIES[1] : SERIES[0] }))), ["Type", "Category", "Count"], an.error_categories.map((e) => [e.error_type, e.category, fmt.int(e.count)])))
      : card("Error categories", "requires labels", empty("Unlabelled run", "Test runs have no ground truth. Validation errors from held-out training entities are on the Error Analysis page.", null, "alert"))));
}
