import { h, icon, fmt } from "../dom.js";
import { get } from "../api.js";
import { kindBadge } from "../app.js";
import { card, pageHead, prettyModel } from "../ui.js";

function row(label, value, ok = true) {
  return h("div", { class: "strength" }, h("span", { class: "secondary" }, label), h("span", { style: "font-weight:600;overflow-wrap:anywhere" }, value),
    h("span", { class: `badge ${ok === null ? "nomatch" : ok ? "match" : "fail"}` }, icon(ok === null ? "info" : ok ? "check" : "x"), ok === null ? "Info" : ok ? "OK" : "Attention"));
}

export async function render(root) {
  const hl = await get("/health");
  const cd = hl.competition_dataset || {};
  root.append(pageHead("Operate", "System health", "Live diagnostics of the service, data and models."),
    h("div", { class: "grid g-2" },
      card("Service", null, h("div", {}, row("API", hl.status === "ok" ? "Operational" : "Degraded", hl.status === "ok"),
        row("Database", `${hl.database} · ${hl.database_backend}`, hl.database === "ok"), row("Uptime", fmt.secs(hl.uptime_s), null),
        row("API process memory", `${fmt.int(Math.round(hl.process_rss_mb))} MB`, hl.process_rss_mb < 4000),
        row("Free disk", `${fmt.dec(hl.disk_free_gb, 1)} GB`, hl.disk_free_gb > 5), row("Host", `${hl.platform} · ${hl.cpu_count} CPUs · Python ${hl.python}`, null))),
      card("Data & pipeline", null, h("div", {},
        row("Competition dataset", cd.complete ? `complete · train ${cd.files?.train?.length || 0} files · test ${cd.files?.test?.length || 0} files` : "not installed in ./dataset", !!cd.complete),
        row("Data behind the latest run", hl.data_kind ? kindBadge(hl.data_kind) : "no run yet", hl.data_kind ? hl.data_kind === "competition" || null : null),
        row("Last successful run", hl.latest_run ? `${hl.latest_run} · ${fmt.date(hl.latest_run_finished)}` : "none", !!hl.latest_run),
        row("Last failed run", hl.last_failed_run ? `${hl.last_failed_run} — ${hl.last_failed_error || "see Runs"}` : "none", !hl.last_failed_run),
        row("Production model", hl.production_model ? `${hl.production_model} · ${prettyModel(hl.production_model_name)}` : "none — train one", !!hl.production_model),
        row("Model trained on", hl.production_model_data_kind ? kindBadge(hl.production_model_data_kind) : "—", hl.production_model_data_kind ? hl.production_model_data_kind === "competition" || null : null),
        row("Model families available", Object.entries(hl.available_model_families).filter(([, v]) => v).map(([k]) => prettyModel(k)).join(", "), null)))));
}
