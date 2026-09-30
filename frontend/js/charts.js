// Hand-rolled SVG charts. Thin marks, 4px rounded data ends, recessive grid, hover tooltips,
// every chart has a table view. Colours come from CSS role tokens only.
import { h, s, clear, fmt } from "./dom.js";

let tipEl;
function tip() {
  if (!tipEl) { tipEl = h("div", { class: "tooltip", role: "tooltip" }); document.body.append(tipEl); }
  return tipEl;
}
export function showTip(ev, value, label) {
  const t = tip();
  clear(t).append(h("b", {}, value), h("span", {}, label));
  const x = Math.min(ev.clientX + 14, window.innerWidth - 200);
  t.style.left = `${x}px`; t.style.top = `${ev.clientY + 14}px`; t.classList.add("on");
}
export function hideTip() { tip().classList.remove("on"); }

const cssVar = (name) => `var(${name})`;
export const SERIES = [1, 2, 3, 4, 5, 6, 7, 8].map((k) => cssVar(`--series-${k}`));
export const SEQ = [1, 2, 3, 4, 5, 6, 7].map((k) => cssVar(`--seq-${k}`));

function niceMax(v) {
  if (v <= 0) return 1;
  const p = Math.pow(10, Math.floor(Math.log10(v)));
  for (const m of [1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]) if (m * p >= v) return m * p;
  return 10 * p;
}

function topRoundedBar(x, y, w, hgt, r = 4) {
  if (hgt <= 0) return "";
  r = Math.min(r, w / 2, hgt);
  return `M${x},${y + hgt}V${y + r}Q${x},${y} ${x + r},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + hgt}Z`;
}
function rightRoundedBar(x, y, w, hgt, r = 4) {
  if (w <= 0) return "";
  r = Math.min(r, hgt / 2, w);
  return `M${x},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + hgt - r}Q${x + w},${y + hgt} ${x + w - r},${y + hgt}H${x}Z`;
}

/** Render `draw(width)` at the container's real pixel width (1 SVG unit = 1 CSS px, so text never shrinks). */
export function responsive(draw, minW = 260) {
  const host = h("div", { class: "chart" });
  let last = 0;
  const paint = (w) => {
    w = Math.max(minW, Math.round(w / 8) * 8);
    if (w === last) return;
    last = w;
    clear(host).append(draw(w));
  };
  const ro = new ResizeObserver((es) => { for (const e of es) if (e.contentRect.width) paint(e.contentRect.width); });
  ro.observe(host);
  paint(560);
  return host;
}

/** Wrap a chart with a "Table" toggle so values are reachable without hover. */
export function withTable(chartEl, headers, rows, legend) {
  const wrap = h("div", {});
  const table = h("div", { class: "table-wrap", hidden: true },
    h("table", { class: "t" }, h("thead", {}, h("tr", {}, headers.map((x, i) => h("th", { class: i ? "r" : "" }, x)))),
      h("tbody", {}, rows.map((r) => h("tr", {}, r.map((c, i) => h("td", { class: i ? "r num" : "" }, c)))))));
  const btn = h("button", { class: "link-btn", type: "button", onclick: () => {
    const showing = !table.hidden;
    table.hidden = showing; chartEl.hidden = !showing;
    btn.textContent = showing ? "View as table" : "View as chart";
  } }, "View as table");
  wrap.append(chartEl, table, h("div", { class: "chart-foot" }, legend || h("span"), btn));
  return wrap;
}

/** Vertical bars (zero baseline). opts: labels, values, colors?, format, height, title */
export function barChart(opts) { return responsive((W) => drawBars(opts, W)); }
function drawBars({ labels, values, colors, format = fmt.int, height = 220, valueLabels = true, tipLabel }, W) {
  const H = height, L = 40, R = 8, T = 18, B = 26;
  const max = niceMax(Math.max(...values, 0));
  const n = values.length || 1;
  const band = (W - L - R) / n, bw = Math.min(46, band * 0.62);
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" });
  for (let k = 0; k <= 4; k++) {
    const y = T + (H - T - B) * (1 - k / 4);
    svg.append(s("line", { x1: L, x2: W - R, y1: y, y2: y, class: "gridline" }),
      s("text", { x: L - 8, y: y + 4, "text-anchor": "end", class: "axis-label" }, fmt.compact(max * k / 4)));
  }
  values.forEach((v, i) => {
    const hh = (H - T - B) * (v / max);
    const x = L + band * i + (band - bw) / 2, y = H - B - hh;
    const fill = colors ? colors[i % colors.length] : SERIES[0];
    const p = s("path", { d: topRoundedBar(x, y, bw, Math.max(hh, v > 0 ? 1.5 : 0)), fill, class: "bar", tabindex: 0,
      "aria-label": `${labels[i]}: ${format(v)}` });
    const hit = s("rect", { x: L + band * i, y: T, width: band, height: H - T - B, fill: "transparent" });
    const on = (ev) => { p.classList.add("hl"); showTip(ev, format(v), tipLabel ? tipLabel(i) : labels[i]); };
    const off = () => { p.classList.remove("hl"); hideTip(); };
    for (const el of [p, hit]) { el.addEventListener("pointermove", on); el.addEventListener("pointerleave", off); }
    p.addEventListener("focus", (ev) => { const r = p.getBoundingClientRect(); on({ clientX: r.right, clientY: r.top }); });
    p.addEventListener("blur", off);
    const every = band < 34 ? 2 : 1;  // thin axis labels when bands get narrow
    if (i % every === 0) svg.append(s("text", { x: x + bw / 2, y: H - 8, "text-anchor": "middle", class: "axis-label" }, labels[i]));
    svg.append(hit, p);
    if (valueLabels && v > 0 && band >= 26) svg.append(s("text", { x: x + bw / 2, y: y - 6, "text-anchor": "middle", class: "val-label" }, format(v)));
  });
  svg.append(s("line", { x1: L, x2: W - R, y1: H - B, y2: H - B, stroke: "var(--axis)" }));
  return svg;
}

/** Horizontal bars. rows: [{label, value, sub?}] */
export function hbarChart(rows, opts = {}) { return responsive((W) => drawHbars(rows, opts, W)); }
function drawHbars(rows, { format = fmt.int, color = SERIES[0], max } = {}, W) {
  const rowH = 30, L = Math.min(170, Math.round(W * 0.34)), R = 58;
  const H = rows.length * rowH + 6;
  const mx = max ?? niceMax(Math.max(...rows.map((r) => r.value), 0));
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" });
  rows.forEach((r, i) => {
    const y = 4 + i * rowH, bh = 12, w = (W - L - R) * (r.value / mx);
    const p = s("path", { d: rightRoundedBar(L, y + 6, Math.max(w, r.value > 0 ? 1.5 : 0), bh), fill: r.color || color, class: "bar" });
    const hit = s("rect", { x: 0, y, width: W, height: rowH, fill: "transparent" });
    const on = (ev) => { p.classList.add("hl"); showTip(ev, format(r.value), r.sub ? `${r.label} · ${r.sub}` : r.label); };
    const off = () => { p.classList.remove("hl"); hideTip(); };
    hit.addEventListener("pointermove", on); hit.addEventListener("pointerleave", off);
    svg.append(s("rect", { x: L, y: y + 6, width: W - L - R, height: bh, rx: 4, fill: "var(--surface-3)" }), p,
      s("text", { x: L - 10, y: y + 16, "text-anchor": "end", class: "axis-label" }, truncate(r.label, Math.floor(L / 6.5))),
      s("text", { x: L + w + 8, y: y + 16, class: "val-label" }, format(r.value)), hit);
  });
  return svg;
}

/** Donut with a 2px surface gap between segments. segments: [{label, value}] (colors follow order). */
export function donut(segments, { center, centerLabel, format = fmt.int } = {}) {
  const total = segments.reduce((a, b) => a + b.value, 0) || 1;
  const size = 180, r = 70, sw = 22, c = size / 2;
  const svg = s("svg", { viewBox: `0 0 ${size} ${size}`, role: "img", style: "max-width:180px" });
  let a0 = -Math.PI / 2;
  const legend = h("div", { class: "donut-legend" });
  segments.forEach((seg, i) => {
    const frac = seg.value / total, a1 = a0 + frac * Math.PI * 2;
    const color = seg.color || SERIES[i % SERIES.length];
    if (frac > 0) {
      const gap = segments.length > 1 ? 0.018 : 0;
      const large = a1 - a0 - gap > Math.PI ? 1 : 0;
      const p1 = [c + r * Math.cos(a0 + gap / 2), c + r * Math.sin(a0 + gap / 2)];
      const p2 = [c + r * Math.cos(a1 - gap / 2), c + r * Math.sin(a1 - gap / 2)];
      const d = frac >= 0.9999 ? `M${c},${c - r}A${r},${r} 0 1 1 ${c - 0.01},${c - r}` : `M${p1[0]},${p1[1]}A${r},${r} 0 ${large} 1 ${p2[0]},${p2[1]}`;
      const path = s("path", { d, fill: "none", stroke: color, "stroke-width": sw, class: "bar" });
      path.addEventListener("pointermove", (ev) => showTip(ev, `${format(seg.value)} · ${fmt.pct(frac)}`, seg.label));
      path.addEventListener("pointerleave", hideTip);
      svg.append(path);
    }
    legend.append(h("div", {}, h("i", { style: { background: color } }), h("span", {}, seg.label), h("b", {}, fmt.pct(frac))));
    a0 = a1;
  });
  svg.append(s("text", { x: c, y: c + 2, "text-anchor": "middle", fill: "var(--text-primary)", "font-size": "22", "font-weight": "750" }, center ?? fmt.int(total)),
    s("text", { x: c, y: c + 20, "text-anchor": "middle", class: "axis-label" }, centerLabel || "Total"));
  return h("div", { class: "donut-wrap" }, h("div", { class: "chart", style: "max-width:180px" }, svg), legend);
}

/** Dot plot for values in a tight range (non-zero axis is legitimate for dots, never for bars). */
export function dotPlot(rows, opts = {}) { return responsive((W) => drawDots(rows, opts, W)); }
function drawDots(rows, { format = (v) => fmt.dec(v, 4), domain, highlight } = {}, W) {
  const L = Math.min(170, Math.round(W * 0.38)), R = 62, rowH = 28;
  const vals = rows.map((r) => r.value).filter((v) => v !== null && v !== undefined);
  let [lo, hi] = domain || [Math.min(...vals), Math.max(...vals)];
  const pad = Math.max((hi - lo) * 0.15, 0.005);
  lo = Math.max(0, lo - pad); hi = Math.min(1, hi + pad);
  const H = rows.length * rowH + 28;
  const x = (v) => L + (W - L - R) * ((v - lo) / (hi - lo || 1));
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" });
  const ticks = W < 420 ? 2 : 4;
  for (let k = 0; k <= ticks; k++) {
    const v = lo + (hi - lo) * k / ticks;
    svg.append(s("line", { x1: x(v), x2: x(v), y1: 4, y2: H - 22, class: "gridline" }),
      s("text", { x: x(v), y: H - 6, "text-anchor": "middle", class: "axis-label" }, v.toFixed(3)));
  }
  rows.forEach((r, i) => {
    const y = 4 + i * rowH + rowH / 2;
    const hl = highlight && highlight(r);
    svg.append(s("line", { x1: L, x2: x(r.value), y1: y, y2: y, stroke: hl ? SERIES[6] : SERIES[0], "stroke-width": 2, "stroke-opacity": .45 }));
    const dot = s("circle", { cx: x(r.value), cy: y, r: hl ? 7 : 5.5, fill: hl ? SERIES[6] : SERIES[0], stroke: "var(--surface-1)", "stroke-width": 2 });
    const hit = s("rect", { x: 0, y: y - rowH / 2, width: W, height: rowH, fill: "transparent" });
    hit.addEventListener("pointermove", (ev) => showTip(ev, format(r.value), r.sub ? `${r.label} · ${r.sub}` : r.label));
    hit.addEventListener("pointerleave", hideTip);
    svg.append(dot, s("text", { x: L - 10, y: y + 4, "text-anchor": "end", class: "axis-label", "font-weight": hl ? 700 : 400 }, truncate(r.label, Math.floor(L / 6.3))),
      s("text", { x: x(r.value) + 12, y: y + 4, class: "val-label" }, format(r.value)), hit);
  });
  return svg;
}

/** Paired bars per category (two series, shared zero baseline). */
export function pairedBars(labels, a, b, opts = {}) { return responsive((W) => drawPaired(labels, a, b, opts, W)); }
function drawPaired(labels, a, b, { names = ["A", "B"], format = fmt.int, height = 220 } = {}, W) {
  const H = height, L = 40, R = 8, T = 14, B = 26;
  const max = niceMax(Math.max(...a, ...b, 0));
  const band = (W - L - R) / (labels.length || 1), bw = Math.min(22, band * 0.32);
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" });
  for (let k = 0; k <= 4; k++) {
    const y = T + (H - T - B) * (1 - k / 4);
    svg.append(s("line", { x1: L, x2: W - R, y1: y, y2: y, class: "gridline" }),
      s("text", { x: L - 8, y: y + 4, "text-anchor": "end", class: "axis-label" }, fmt.compact(max * k / 4)));
  }
  labels.forEach((lab, i) => {
    [a[i], b[i]].forEach((v, k) => {
      const hh = (H - T - B) * (v / max), x = L + band * i + band / 2 - bw - 1 + k * (bw + 2), y = H - B - hh;
      const p = s("path", { d: topRoundedBar(x, y, bw, Math.max(hh, v > 0 ? 1.5 : 0)), fill: SERIES[k], class: "bar" });
      p.addEventListener("pointermove", (ev) => showTip(ev, format(v), `${names[k]} · ${lab}`));
      p.addEventListener("pointerleave", hideTip);
      svg.append(p);
    });
    svg.append(s("text", { x: L + band * i + band / 2, y: H - 8, "text-anchor": "middle", class: "axis-label" }, lab));
  });
  svg.append(s("line", { x1: L, x2: W - R, y1: H - B, y2: H - B, stroke: "var(--axis)" }));
  return svg;
}

export function legend(items) {
  return h("div", { class: "legend" }, items.map(([label, color]) => h("span", {}, h("i", { style: { background: color } }), label)));
}

export function meters(rows, { format = (v) => fmt.pct(v), max = 1 } = {}) {
  return h("div", { class: "meters" }, rows.map((r) => h("div", { class: "meter" },
    h("div", { class: "k" }, r.color ? h("i", { style: { background: r.color } }) : null, r.label),
    h("div", { class: "track", title: r.title || "" }, h("div", { class: "fill", style: { width: `${Math.max(0, Math.min(1, r.value / (r.max ?? max))) * 100}%`, background: r.color || SERIES[0] } })),
    h("div", { class: "v" }, r.display ?? format(r.value)))));
}

function truncate(t, n) { t = String(t ?? ""); return t.length > n ? t.slice(0, n - 1) + "…" : t; }
