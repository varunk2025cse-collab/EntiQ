import { h, icon, fmt, empty, statusBadge } from "../dom.js";
import { get, state } from "../api.js";
import { identityBanner, kindBadge } from "../app.js";
import { card, pageHead, metricList } from "../ui.js";

const bytes = (b) => (b >= 2 ** 20 ? `${(b / 2 ** 20).toFixed(1)} MB` : `${fmt.int(Math.round(b / 1024))} KB`);

export async function render(root) {
  const runs = (await get("/resolution/runs")).filter((r) => r.status === "COMPLETED");
  root.append(pageHead("Operate", "Export Center", "Submission files are produced by the pipeline and checked against every submission rule (and the official validator when present) before they are offered for download."));
  if (!runs.length) { root.append(h("div", { class: "card" }, empty("Nothing to export", "Complete a resolution run first.", null, "download"))); return; }
  const runId = runs.some((r) => r.id === state.runId) ? state.runId : runs[0].id;
  const run = runs.find((r) => r.id === runId);
  const sel = h("select", { class: "select", onchange: (e) => { state.runId = e.target.value; window.dispatchEvent(new HashChangeEvent("hashchange")); } },
    runs.map((r) => h("option", { value: r.id, selected: r.id === runId }, `${r.id} · ${r.split} · ${r.model_version || ""}`)));
  const res = await get(`/results/${runId}`);
  const v = res.validation || {};
  const off = res.official_validation;
  const kind = run.summary?.dataset_kind || "unknown";
  const ready = v.status === "PASS" && (!off || off.status === "PASS");
  const isTest = run.split === "test";

  root.append(identityBanner(kind), h("div", { class: "row" }, h("span", { class: "secondary" }, "Run"), sel));
  root.append(h("div", { class: "grid g-2-1" },
    card("Submission readiness", "built-in checker + official utils/validate_submission.py", h("div", {},
      h("div", { class: `callout ${ready && isTest && kind === "competition" ? "" : ready ? "info" : "err"}`, style: "margin-bottom:12px" }, icon(ready ? "check" : "x"),
        h("div", {}, ready ? (isTest && kind === "competition" ? h("b", {}, "Ready to submit. ") : h("b", {}, "Valid format. ")) : h("b", {}, "Not submittable. "),
          ready ? (isTest && kind === "competition" ? "These files come from a run on the official test split." : `This run is on ${kind} data (${run.split} split), not the competition test set.`) : "Fix the issues below.")),
      h("div", { class: "row", style: "margin-bottom:10px" }, h("span", { class: "hint" }, "Built-in"), statusBadge(v.status), off ? h("span", { class: "hint", style: "margin-left:12px" }, "Official") : null, off ? statusBadge(off.status) : h("span", { class: "hint" }, "official validator not found")),
      (v.issues || []).map((i) => h("div", { class: "row", style: "margin-bottom:6px" }, h("span", { class: "badge fail" }, icon("x"), "Issue"), h("span", {}, i))),
      (v.warnings || []).map((i) => h("div", { class: "row", style: "margin-bottom:6px" }, h("span", { class: "badge review" }, icon("alert"), "Warning"), h("span", {}, i))),
      off ? h("pre", { class: "pre", style: "margin-top:10px" }, off.stdout) : null)),
    card("Provenance", null, metricList([["Dataset", kindBadge(kind)], ["Split", run.split], ["Model version", run.model_version || "—"], ["Run", runId],
      ["Generated", fmt.date(run.finished || run.created)], ["Rows (S1 entities)", fmt.int(v.matching_rows)], ["Links", fmt.int(v.matched_links)], ["Candidates", fmt.int(v.candidate_links)]]))));
  const dl = (file, label, desc, ic = "file", primary = true) => h("div", { class: "card card-pad" }, h("div", { class: "row", style: "justify-content:space-between;flex-wrap:nowrap;gap:12px" },
    h("div", { class: "row", style: "flex-wrap:nowrap" }, h("div", { class: "kpi", style: "padding:0" }, h("div", { class: "ico" }, icon(ic))), h("div", {}, h("b", {}, label), h("div", { class: "hint" }, desc))),
    ready || !primary ? h("a", { class: `btn sm ${primary ? "primary" : ""}`, href: `/api/results/${runId}/download?file=${encodeURIComponent(file)}`, download: "" }, icon("download"), "Download")
      : h("span", { class: "badge fail" }, "Blocked")));
  root.append(h("div", { class: "grid g-3" },
    dl("matching_results.tsv", "matching_results.tsv", `Final matches · ${bytes(res.files["matching_results.tsv"].bytes)} · the leaderboard file`),
    dl("candidate_pairs.tsv", "candidate_pairs.tsv", `Candidates the model scored · ${bytes(res.files["candidate_pairs.tsv"].bytes)}`),
    dl("bundle", "Results bundle (.zip)", "Both TSVs + run diagnostics", "layers"),
    dl("diagnostics.json", "JSON diagnostics", "Counts, blocking, timings, validation", "cpu", false),
    dl("evaluation.json", "Evaluation report", "Metrics & error summary (labelled runs)", "target", false)),
  card("matching_results.tsv preview", "tab-separated, exactly as submitted", h("pre", { class: "pre" }, res.preview.join("\n"))));
}
