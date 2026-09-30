import { h, icon, clear, fmt, empty, toast } from "../dom.js";
import { get, post, upload } from "../api.js";
import { hbarChart, barChart, withTable, SERIES } from "../charts.js";
import { kindBadge } from "../app.js";
import { card, pageHead, metricList } from "../ui.js";

const bytes = (b) => (b >= 2 ** 30 ? `${(b / 2 ** 30).toFixed(2)} GB` : b >= 2 ** 20 ? `${(b / 2 ** 20).toFixed(1)} MB` : `${fmt.int(Math.round(b / 1024))} KB`);

export async function render(root, { params }) {
  const ds = await get("/datasets");
  const order = { competition: 0, custom: 1, synthetic: 2 };
  ds.sort((a, b) => (order[a.data_kind] ?? 3) - (order[b.data_kind] ?? 3));
  root.append(pageHead("Overview", "Datasets", "Source files are read with an explicit tab separator and validated; originals are never modified. Real competition data and demo data are kept separate and labelled everywhere."));
  if (!ds.length) root.append(h("div", { class: "card" }, empty("No dataset found", "Copy the competition files into ./dataset (train/ and test/) or upload them below.", null, "database")));
  for (const d of ds) root.append(datasetCard(d));
  root.append(uploader());
  const id = params.get("id");
  if (id) root.append(await profileView(id));
}

function datasetCard(d) {
  const splits = Object.entries(d.files || {});
  return h("section", { class: "card" },
    h("div", { class: "card-head" }, h("div", {}, h("div", { class: "row" }, h("h3", {}, d.name), kindBadge(d.data_kind),
      ...d.splits.map((s) => h("span", { class: "badge cand" }, s)), d.has_labels ? h("span", { class: "badge run" }, icon("check"), "Validation available") : null),
    h("span", { class: "sub" }, d.description)),
    h("div", { class: "row" }, h("a", { class: "btn sm", href: `#/datasets?id=${encodeURIComponent(d.id)}` }, icon("chart"), "Profile"),
      d.splits.includes("test") || d.splits.includes("train") ? h("button", { class: "btn sm primary", onclick: async () => {
        try { const r = await post("/resolution/run", { dataset_id: d.id, split: d.splits.includes("test") ? "test" : "train" }); toast(`Started ${r.run_id}`); location.hash = "#/runs"; }
        catch (e) { toast(e.message, true); } } }, icon("play"), "Resolve") : null)),
    h("div", { class: "card-body" }, h("div", { class: "table-wrap", style: "margin:0 -20px" }, h("table", { class: "t" },
      h("thead", {}, h("tr", {}, ["Split", "File", "Rows", "Size", "Columns"].map((x, i) => h("th", { class: i === 2 || i === 3 ? "r" : "" }, x)))),
      h("tbody", {}, splits.flatMap(([sp, files]) => files.map((f) => h("tr", {},
        h("td", {}, h("span", { class: "badge cand" }, sp)), h("td", { class: "mono" }, f.file), h("td", { class: "num r" }, fmt.int(f.rows)),
        h("td", { class: "num r" }, bytes(f.bytes)), h("td", { class: "mono", style: "font-size:11.5px;color:var(--text-muted)" }, (f.columns || []).join(" · ")))))))),
    d.data_kind === "synthetic" ? h("p", { class: "hint", style: "margin:10px 0 0" }, "Generated for development and tests only. Never used to train or select the competition model.") : null));
}

function uploader() {
  const input = h("input", { type: "file", multiple: true, accept: ".tsv", hidden: true });
  const name = h("input", { class: "input", placeholder: "Dataset name", value: "Uploaded dataset", maxlength: 120 });
  const status = h("div", { class: "hint", style: "margin-top:8px" }, "Expected names: train_source1.tsv … test_source3.tsv, train_ground_truth.tsv. Max 300 MB per file. Uploaded data is labelled \"custom\", never \"real competition\".");
  const drop = h("div", { class: "drop" }, icon("upload"), h("div", { style: "margin-top:8px" }, "Drop .tsv files here or ", h("a", { href: "#", onclick: (e) => { e.preventDefault(); input.click(); } }, "browse")));
  const send = async (files) => {
    const fd = new FormData();
    fd.append("name", name.value);
    for (const f of files) fd.append("files", f, f.name);
    status.textContent = `Uploading ${files.length} files…`;
    try { const r = await upload("/datasets/upload", fd); toast(`Dataset ${r.id} registered`); location.hash = `#/datasets?id=${r.id}`; }
    catch (e) { clear(status).append(h("b", { style: "color:var(--crit-ink)" }, "Rejected: "), ...(e.detail?.issues || [e.message]).map((i) => h("div", {}, i))); }
  };
  input.addEventListener("change", () => send([...input.files]));
  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", (e) => { e.preventDefault(); drop.classList.remove("over"); send([...e.dataTransfer.files]); });
  return card("Upload a dataset", "validated server-side: file names, headers, encoding, row shape, size", h("div", {}, h("div", { class: "row", style: "margin-bottom:10px" }, name), drop, input, status));
}

async function profileView(id) {
  const wrap = h("div", { style: "display:flex;flex-direction:column;gap:16px" }, h("div", { class: "loading" }, h("span", { class: "spinner" }), `Profiling ${id}…`));
  get(`/datasets/${encodeURIComponent(id)}/profile`).then((p) => {
    clear(wrap);
    for (const [split, d] of Object.entries(p)) {
      const pr = d.profile;
      const g = pr.ground_truth;
      wrap.append(h("div", { class: "section-title" }, `Profile · ${id} · ${split}`),
        h("div", { class: "grid g-3" },
          card("Countries", "records across all sources", withTable(hbarChart(Object.entries(pr.country_distribution).map(([k, v], i) => ({ label: k, value: v, color: SERIES[i % 8] }))),
            ["Country", "Records"], Object.entries(pr.country_distribution).map(([k, v]) => [k, fmt.int(v)]))),
          g ? card("Match multiplicity", "true links per S1 entity", withTable(barChart({ labels: Object.keys(g.match_multiplicity), values: Object.values(g.match_multiplicity), colors: [SERIES[0]] }),
            ["Links", "S1 entities"], Object.entries(g.match_multiplicity).map(([k, v]) => [k, fmt.int(v)])))
            : card("Source sizes", null, metricList(Object.entries(pr.sources).map(([k, v]) => [`Source ${k.slice(1)}`, fmt.int(v.records)]))),
          card("Data quality", d.validation.ok ? "no blocking errors" : "errors found", h("div", {},
            [...d.validation.errors.map((e) => ["fail", e]), ...d.validation.warnings.map((w) => ["review", w])].map(([c, t]) => h("div", { class: "row", style: "margin-bottom:6px;flex-wrap:nowrap;align-items:flex-start" },
              h("span", { class: `badge ${c}` }, icon(c === "fail" ? "x" : "alert"), c === "fail" ? "Error" : "Note"), h("span", { style: "font-size:12.5px" }, t))),
            !d.validation.errors.length && !d.validation.warnings.length ? h("p", { class: "muted", style: "margin:0" }, "All checks passed.") : null,
            g ? metricList([["Singleton rate", fmt.pct(g.singleton_rate, 2)], ["Mean links per S1", fmt.dec(g.mean_matches_per_s1, 2)],
              ["Records linked to more than one S1", fmt.int(g.records_linked_to_multiple_s1)], ["True-pair country agreement", fmt.pct(g.true_pair_country_agreement, 2)]]) : null))));
    }
  }).catch((e) => { clear(wrap).append(h("div", { class: "card" }, empty("Profile unavailable", String(e.message), null, "database"))); });
  return wrap;
}
