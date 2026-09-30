import { h, icon, clear, fmt, empty, toast } from "../dom.js";
import { get, post, upload } from "../api.js";
import { hbarChart, barChart, withTable, SERIES } from "../charts.js";
import { card } from "./dashboard.js";

export async function render(root, { params }) {
  const ds = await get("/datasets");
  root.append(h("div", { class: "card page-head" }, h("div", {}, h("h1", {}, "Dataset Management"),
    h("p", {}, "Source files are read with an explicit tab separator and validated; originals are never modified."))));
  const list = h("div", { class: "grid g-3" }, ds.map((d) => h("div", { class: "card card-pad" },
    h("div", { class: "row", style: "justify-content:space-between" }, h("b", {}, d.name), h("span", { class: `chip ${d.kind === "competition" ? "blue" : ""}` }, d.kind)),
    h("p", { class: "secondary", style: "font-size:12.5px" }, d.description), h("div", { class: "muted", style: "font-size:12px" }, `splits: ${d.splits.join(", ")}`),
    h("div", { class: "row", style: "margin-top:10px" },
      h("a", { class: "btn sm", href: `#/datasets?id=${encodeURIComponent(d.id)}` }, icon("chart"), "Profile"),
      d.splits.includes("test") || d.splits.includes("train") ? h("button", { class: "btn sm primary", onclick: async () => {
        try { const r = await post("/resolution/run", { dataset_id: d.id, split: d.splits.includes("test") ? "test" : "train" }); toast(`Started ${r.run_id}`); location.hash = "#/runs"; }
        catch (e) { toast(e.message, true); } } }, icon("play"), "Resolve") : null))));
  if (!ds.length) list.append(h("div", { class: "card span-2" }, empty("No dataset found", "Copy the competition files into ./dataset (train/ and test/ folders) or upload them below.")));
  root.append(list, uploader());
  const id = params.get("id") || ds.find((d) => d.kind === "synthetic")?.id || ds[0]?.id;
  if (id) root.append(await profileView(id));
}

function uploader() {
  const input = h("input", { type: "file", multiple: true, accept: ".tsv", hidden: true });
  const name = h("input", { class: "input", placeholder: "Dataset name", value: "Uploaded dataset", maxlength: 120 });
  const status = h("div", { class: "hint" }, "Expected names: train_source1.tsv … test_source3.tsv, train_ground_truth.tsv. Max 300 MB per file.");
  const drop = h("div", { class: "drop" }, icon("upload"), h("div", { style: "margin-top:8px" }, "Drop the .tsv files here or ", h("a", { href: "#", onclick: (e) => { e.preventDefault(); input.click(); } }, "browse")));
  const send = async (files) => {
    const fd = new FormData();
    fd.append("name", name.value);
    for (const f of files) fd.append("files", f, f.name);
    status.textContent = `Uploading ${files.length} files…`;
    try { const r = await upload("/datasets/upload", fd); toast(`Dataset ${r.id} registered`); location.hash = `#/datasets?id=${r.id}`; }
    catch (e) { clear(status).append(h("b", { style: "color:var(--critical-text)" }, "Rejected: "), ...(e.detail?.issues || [e.message]).map((i) => h("div", {}, i))); }
  };
  input.addEventListener("change", () => send([...input.files]));
  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", (e) => { e.preventDefault(); drop.classList.remove("over"); send([...e.dataTransfer.files]); });
  return card("Upload a dataset", "validated server-side: names, headers, encoding, row shape, size", h("div", {}, h("div", { class: "row", style: "margin-bottom:10px" }, name), drop, input, status));
}

async function profileView(id) {
  const wrap = h("div", { style: "display:flex;flex-direction:column;gap:16px" }, h("div", { class: "card card-pad muted" }, `Profiling ${id}…`));
  get(`/datasets/${encodeURIComponent(id)}/profile`).then((p) => {
    clear(wrap);
    for (const [split, d] of Object.entries(p)) {
      const pr = d.profile;
      const srcs = Object.entries(pr.sources);
      wrap.append(h("h2", { style: "margin:8px 0 0;font-size:18px" }, `${id} · ${split}`),
        h("div", { class: "grid g-4" }, srcs.map(([k, v]) => h("div", { class: "card kpi" }, h("div", {}, h("div", { class: "lbl" }, `Source ${k.slice(1)} records`), h("div", { class: "val" }, fmt.int(v.records)),
          h("div", { class: "foot" }, `missing name ${fmt.pct(v.missing_name_rate, 2)} · address ${fmt.pct(v.missing_address_rate, 2)} · dup ${fmt.pct(v.duplicate_record_rate, 2)}`)))),
        pr.ground_truth ? h("div", { class: "card kpi" }, h("div", {}, h("div", { class: "lbl" }, "Singleton rate"), h("div", { class: "val" }, fmt.pct(pr.ground_truth.singleton_rate)),
          h("div", { class: "foot" }, `${fmt.int(pr.ground_truth.true_pairs)} true links · ${fmt.dec(pr.ground_truth.mean_matches_per_s1, 2)} per S1`))) : null),
      h("div", { class: "grid g-3" },
        card("Countries", "all sources", withTable(hbarChart(Object.entries(pr.country_distribution).map(([k, v], i) => ({ label: k, value: v, color: SERIES[i % 8] }))),
          ["Country", "Records"], Object.entries(pr.country_distribution).map(([k, v]) => [k, fmt.int(v)]))),
        pr.ground_truth ? card("Match multiplicity", "true links per S1", withTable(barChart({ labels: Object.keys(pr.ground_truth.match_multiplicity), values: Object.values(pr.ground_truth.match_multiplicity), colors: [SERIES[0]] }),
          ["Links", "S1 entities"], Object.entries(pr.ground_truth.match_multiplicity).map(([k, v]) => [k, fmt.int(v)]))) : card("Name length", "Source 1 characters", nameHist(pr)),
        card("Data quality", d.validation.ok ? "no blocking errors" : "errors found", h("div", {},
          [...d.validation.errors.map((e) => ["fail", e]), ...d.validation.warnings.map((w) => ["review", w])].map(([c, t]) => h("div", { class: "row", style: "margin-bottom:6px;flex-wrap:nowrap;align-items:flex-start" },
            h("span", { class: `badge ${c}` }, icon(c === "fail" ? "x" : "alert"), c === "fail" ? "Error" : "Note"), h("span", { style: "font-size:12.5px" }, t))),
          !d.validation.errors.length && !d.validation.warnings.length ? h("p", { class: "muted" }, "All checks passed.") : null,
          pr.ground_truth ? h("dl", { class: "kv" }, h("dt", {}, "Records linked to >1 S1"), h("dd", {}, fmt.int(pr.ground_truth.records_linked_to_multiple_s1)),
            h("dt", {}, "True-pair country agreement"), h("dd", {}, fmt.pct(pr.ground_truth.true_pair_country_agreement, 2))) : null))));
    }
  }).catch((e) => { clear(wrap).append(h("div", { class: "card" }, empty("Profile failed", String(e.message)))); });
  return wrap;
}

function nameHist(pr) {
  const hst = pr.sources.S1.name_length.histogram;
  return barChart({ labels: hst.edges.slice(0, -1).map((e) => String(Math.round(e))), values: hst.counts, colors: [SERIES[0]], valueLabels: false });
}
