import { h, icon, fmt } from "../dom.js";
import { get } from "../api.js";
import { card, prettyModel } from "./dashboard.js";

export async function render(root) {
  const hl = await get("/health");
  const item = (k, v, ok = true) => h("div", { class: "insight" }, h("div", { class: "ic" }, icon(ok ? "check" : "alert")), h("div", {}, h("div", { class: "t" }, k), h("div", { style: "font-weight:650" }, v)));
  root.append(h("div", { class: "card page-head" }, h("div", {}, h("h1", {}, "System Health"), h("p", {}, "Live service diagnostics."))),
    h("div", { class: "grid g-3" },
      card("Service", null, h("div", {}, item("API status", hl.status === "ok" ? "Operational" : "Degraded", hl.status === "ok"), item("Database", `${hl.database} · ${hl.database_backend}`, hl.database === "ok"),
        item("Uptime", fmt.secs(hl.uptime_s)), item("Workspace", hl.demo_mode ? "Demo (synthetic data)" : hl.workspace))),
      card("Models", null, h("div", {}, item("Production model", hl.production_model || "none — train one", !!hl.production_model), item("Registered versions", fmt.int(hl.model_versions)),
        item("Latest run", hl.latest_run || "none", !!hl.latest_run),
        item("Model families available", Object.entries(hl.available_model_families).filter(([, v]) => v).map(([k]) => prettyModel(k)).join(", ")))),
      card("Host", null, h("div", {}, item("Python", hl.python), item("Platform", hl.platform), item("CPU cores", String(hl.cpu_count)),
        item("API process memory", `${fmt.dec(hl.process_rss_mb, 0)} MB`), item("Free disk", `${fmt.dec(hl.disk_free_gb, 1)} GB`, hl.disk_free_gb > 2)))));
}
