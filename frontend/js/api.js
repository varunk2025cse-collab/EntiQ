// Thin JSON API client with in-flight de-duplication.
const inflight = new Map();

export class ApiError extends Error {
  constructor(status, detail) { super(typeof detail === "string" ? detail : JSON.stringify(detail)); this.status = status; this.detail = detail; }
}

export async function get(path, params) {
  const qs = params ? "?" + new URLSearchParams(Object.entries(params).filter(([, v]) => v !== undefined && v !== null && v !== "")).toString() : "";
  const url = `/api${path}${qs}`;
  if (inflight.has(url)) return inflight.get(url);
  const p = fetch(url, { headers: { Accept: "application/json" } }).then(handle).finally(() => inflight.delete(url));
  inflight.set(url, p);
  return p;
}

export async function post(path, body) {
  const res = await fetch(`/api${path}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });
  return handle(res);
}

export async function upload(path, form) {
  const res = await fetch(`/api${path}`, { method: "POST", body: form });
  return handle(res);
}

async function handle(res) {
  let data = null;
  try { data = await res.json(); } catch { /* empty body */ }
  if (!res.ok) throw new ApiError(res.status, data?.detail ?? res.statusText);
  return data;
}

export const state = { runId: null, health: null };
