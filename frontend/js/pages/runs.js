import { h, icon, clear, fmt, empty, toast, statusBadge } from "../dom.js";
import { get, post, state } from "../api.js";
import { kindBadge } from "../app.js";
import { card, pageHead } from "../ui.js";

export async function render(root) {
  const [runs, ds, models] = await Promise.all([get("/resolution/runs"), get("/datasets"), get("/models").catch(() => ({ versions: [] }))]);
  const dsSel = h("select", { class: "select" }, ds.map((d) => h("option", { value: d.id }, `${d.name} · ${d.data_kind}`)));
  const spSel = h("select", { class: "select" }, h("option", { value: "test" }, "test split"), h("option", { value: "train" }, "train split (labelled, in-sample)"));
  const mSel = h("select", { class: "select" }, h("option", { value: "" }, "Production model"), models.versions.map((v) => h("option", { value: v.version }, v.version)));
  const go = h("button", { class: "btn primary", onclick: async () => {
    go.disabled = true;
    try { const r = await post("/resolution/run", { dataset_id: dsSel.value, split: spSel.value, model_version: mSel.value || null }); toast(`Queued ${r.run_id}`); poll(); }
    catch (e) { toast(e.message, true); } finally { go.disabled = false; }
  } }, icon("play"), "Start run");
  const body = h("tbody");
  const draw = (rs) => {
    clear(body);
    if (!rs.length) body.append(h("tr", {}, h("td", { colspan: 10 }, empty("No runs yet", "Start one above.", null, "play"))));
    for (const r of rs) {
      const c = r.summary?.counts || {};
      body.append(h("tr", { class: r.status === "COMPLETED" ? "click" : "", onclick: () => { if (r.status === "COMPLETED") { state.runId = r.id; location.hash = "#/dashboard"; } } },
        h("td", {}, h("span", { class: "mono", style: "font-weight:600" }, r.id)), h("td", {}, kindBadge(r.summary?.dataset_kind || "unknown")), h("td", {}, r.split),
        h("td", { class: "mono" }, r.model_version || "—"),
        h("td", {}, statusBadge(r.status), r.status === "RUNNING" || r.status === "QUEUED" ? h("div", { style: "margin-top:6px;min-width:160px" }, h("div", { class: "progress" }, h("i", { style: { width: `${r.progress || 0}%` } })), h("div", { class: "hint" }, r.stage)) : null,
          r.status === "FAILED" ? h("div", { class: "hint", style: "color:var(--crit-ink);max-width:320px" }, (r.error || "").split("\n")[0]) : null),
        h("td", { class: "num" }, fmt.int(c.s1)), h("td", { class: "num" }, fmt.int(c.candidates)), h("td", { class: "num" }, fmt.int(c.links)),
        h("td", {}, r.summary?.validation ? statusBadge(r.summary.validation.status) : "—"),
        h("td", { class: "num" }, fmt.secs(r.runtime_s)), h("td", { class: "secondary" }, fmt.date(r.finished || r.created))));
    }
  };
  let timer = null;
  const poll = async () => {
    const rs = await get("/resolution/runs");
    draw(rs);
    clearTimeout(timer);
    if (rs.some((r) => r.status === "RUNNING" || r.status === "QUEUED")) timer = setTimeout(() => { if (location.hash.startsWith("#/runs")) poll(); }, 1500);
  };
  root.append(pageHead("Overview", "Resolution runs", "A run resolves every Source 1 entity of a dataset split with a trained model and writes validated submission files. Runs execute in the background; this page follows them live.",
    [dsSel, spSel, mSel, go]),
  card("Run history", "select a completed run to open it on the dashboard", h("div", { class: "table-wrap", style: "margin:0 -20px" }, h("table", { class: "t" },
    h("thead", {}, h("tr", {}, ["Run", "Data", "Split", "Model", "Status", "S1 entities", "Candidates", "Links", "Submission", "Runtime", "Finished"].map((x) => h("th", {}, x)))), body))));
  draw(runs);
  poll();
}
