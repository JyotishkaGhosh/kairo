// Kairo website: loads data.json (written by export.py) and renders every section.
// Plain JavaScript, no libraries. Charts are hand-built SVG so they follow one set of
// rules: thin marks, hover/focus tooltips, a table view for every chart, and colors
// taken from CSS variables so light/dark mode switch automatically.
"use strict";

(() => {
  // ---------- tiny DOM helpers (all text goes in via textContent, never innerHTML) ----------
  const $ = (id) => document.getElementById(id);

  function el(tag, props = {}, ...children) {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(props)) {
      if (v == null || v === false) continue;
      if (k === "class") n.className = v;
      else if (k === "text") n.textContent = v;
      else if (k === "style") Object.assign(n.style, v);
      else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
      else n.setAttribute(k, v === true ? "" : v);
    }
    for (const c of children.flat(Infinity)) {
      if (c == null || c === false) continue;
      n.append(c instanceof Node ? c : document.createTextNode(String(c)));
    }
    return n;
  }

  const SVG_NS = "http://www.w3.org/2000/svg";
  function svg(tag, attrs = {}, text) {
    const n = document.createElementNS(SVG_NS, tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null) continue;
      if (k === "style") Object.assign(n.style, v);
      else n.setAttribute(k, v);
    }
    if (text != null) n.textContent = text;
    return n;
  }

  // ---------- formatting ----------
  const trimZeros = (s) => s.replace(/\.0+$|(\.\d*[1-9])0+$/, "$1");
  function money(v) {
    if (v == null || Number.isNaN(v)) return "–";
    const a = Math.abs(v), sign = v < 0 ? "-" : "";
    if (a >= 1e6) return `${sign}$${trimZeros((a / 1e6).toFixed(a >= 1e7 ? 1 : 2))}M`;
    if (a >= 1e3) return `${sign}$${Math.round(a / 1e3)}K`;
    return `${sign}$${Math.round(a)}`;
  }
  const moneyFull = (v) => (v == null ? "–" : `$${Math.round(v).toLocaleString("en-US")}`);
  const pct = (v, digits = 0) => (v == null ? "–" : `${Number(v).toFixed(digits)}%`);
  const ratioPct = (v) => (v == null ? "–" : `${Math.round(v * 100)}%`);
  const parseDay = (s) => new Date(`${s}T00:00:00Z`);
  const fmtDate = (s) => parseDay(s).toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric", timeZone: "UTC" });
  const fmtDay = (s) => parseDay(s).toLocaleDateString("en-GB", { day: "numeric", month: "short", timeZone: "UTC" });
  const fmtMonth = (s) => parseDay(`${s}-01`).toLocaleDateString("en-GB", { month: "short", year: "2-digit", timeZone: "UTC" });
  const fmtMonthLong = (s) => parseDay(`${s}-01`).toLocaleDateString("en-GB", { month: "long", year: "numeric", timeZone: "UTC" });

  // Series colors: one meaning per color across the whole page
  const C = { kairo: "var(--s-kairo)", rep: "var(--s-rep)", actual: "var(--s-actual)", pos: "var(--pos)", neg: "var(--neg)" };

  // ---------- theme toggle ----------
  const root = document.documentElement;
  try {
    const saved = localStorage.getItem("kairo-theme");
    if (saved === "light" || saved === "dark") root.dataset.theme = saved;
  } catch (e) { /* storage unavailable: follow the OS setting */ }
  const activeTheme = () => root.dataset.theme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  const toggle = $("theme-toggle");
  const syncToggle = () => {
    const next = activeTheme() === "dark" ? "light" : "dark";
    toggle.textContent = next === "dark" ? "Dark" : "Light";
    toggle.setAttribute("aria-label", `Switch to ${next} theme`);
  };
  toggle.addEventListener("click", () => {
    root.dataset.theme = activeTheme() === "dark" ? "light" : "dark";
    try { localStorage.setItem("kairo-theme", root.dataset.theme); } catch (e) { /* ignore */ }
    syncToggle();
  });
  syncToggle();

  // ---------- tooltip ----------
  const tip = $("tooltip");
  function showTip(x, y, title, rows) {
    tip.replaceChildren(
      el("div", { class: "tt-title", text: title }),
      ...rows.map((r) => el("div", { class: "tt-row" },
        r.color ? el("span", { class: r.shape === "rect" ? "key" : "key-line", style: { background: r.color } }) : null,
        el("strong", { text: r.value }),
        r.label ? el("span", { class: "tt-label", text: r.label }) : null)));
    tip.hidden = false;
    const w = tip.offsetWidth, h = tip.offsetHeight;
    let left = x + 14;
    if (left + w > window.innerWidth - 8) left = Math.max(8, x - 14 - w);
    let top = y - h - 12;
    if (top < 8) top = y + 18;
    tip.style.left = `${left}px`;
    tip.style.top = `${top}px`;
  }
  const hideTip = () => { tip.hidden = true; };
  window.addEventListener("scroll", hideTip, { passive: true });

  // ---------- chart scaffolding ----------
  function niceScale(maxValue, ticks = 4) {
    if (!(maxValue > 0)) return { max: 1, step: 0.25 };
    const raw = maxValue / ticks;
    const exp = 10 ** Math.floor(Math.log10(raw));
    const f = raw / exp;
    const step = (f <= 1 ? 1 : f <= 2 ? 2 : f <= 2.5 ? 2.5 : f <= 5 ? 5 : 10) * exp;
    return { max: Math.ceil(maxValue / step) * step, step };
  }

  // A card with title, optional subtitle, the plot, and a "Show table" view of the same data
  function chartCard(host, { title, sub, legend, table }) {
    const plot = el("div", { class: "chart" });
    const tableBox = el("div", { class: "table-scroll", hidden: true });
    const btn = el("button", { class: "table-toggle", type: "button", "aria-expanded": "false", text: "Show table" });
    btn.addEventListener("click", () => {
      const open = tableBox.hidden;
      if (open && !tableBox.firstChild) tableBox.append(dataTable(table));
      tableBox.hidden = !open;
      btn.textContent = open ? "Hide table" : "Show table";
      btn.setAttribute("aria-expanded", String(open));
    });
    host.append(...[
      el("div", { class: "chart-head" }, el("h3", { text: title }), btn),
      sub ? el("p", { class: "chart-sub", text: sub }) : null,
      legend ? legendRow(legend) : null,
      plot, tableBox].filter(Boolean));
    return plot;
  }

  function dataTable({ columns, rows }) {
    return el("table", { class: "data-table" },
      el("thead", {}, el("tr", {}, columns.map((c) => el("th", { class: c.num ? "num" : null, scope: "col", text: c.label })))),
      el("tbody", {}, rows.map((r) => el("tr", {}, r.map((v, i) => el("td", { class: columns[i].num ? "num" : null, text: v }))))));
  }

  function legendRow(items) {
    return el("div", { class: "legend" }, items.map((it) =>
      el("span", {}, el("span", { class: it.shape === "line" ? "key-line" : "key", style: { background: it.color } }), it.label)));
  }

  // Redraw when the container width changes (phones, window resizes)
  function responsive(plot, draw) {
    let last = 0;
    new ResizeObserver(() => {
      const w = Math.round(plot.clientWidth);
      if (w && w !== last) { last = w; plot.replaceChildren(); draw(w); }
    }).observe(plot);
  }

  function svgRoot(w, h, label) {
    return svg("svg", { viewBox: `0 0 ${w} ${h}`, width: w, height: h, role: "img", "aria-label": label, tabindex: "0" });
  }

  // Map a pointer event to SVG user coordinates
  function pointerX(node, e, w) {
    const r = node.getBoundingClientRect();
    return (e.clientX - r.left) * (w / r.width);
  }

  // Column with 4px rounded top, square at the baseline
  function colPath(x, yTop, bw, yBase, r = 4) {
    r = Math.max(0, Math.min(r, bw / 2, yBase - yTop));
    return `M${x},${yBase}V${yTop + r}Q${x},${yTop} ${x + r},${yTop}H${x + bw - r}Q${x + bw},${yTop} ${x + bw},${yTop + r}V${yBase}Z`;
  }
  // Horizontal bar from x0 (baseline) to x1 (data end, rounded)
  function hBarPath(x0, x1, y, h, r = 4) {
    const dir = x1 >= x0 ? 1 : -1;
    r = Math.max(0, Math.min(r, h / 2, Math.abs(x1 - x0)));
    return `M${x0},${y}H${x1 - dir * r}Q${x1},${y} ${x1},${y + r}V${y + h - r}Q${x1},${y + h} ${x1 - dir * r},${y + h}H${x0}Z`;
  }

  // ---------- line chart with crosshair tooltip ----------
  function lineChart(plot, w, { dates, series, label, height = 240 }) {
    const showEnd = w >= 480;
    const m = { l: 56, r: showEnd ? 104 : 14, t: 14, b: 30 };
    const iw = w - m.l - m.r, ih = height - m.t - m.b;
    const t = dates.map((d) => parseDay(d).getTime());
    const t0 = t[0], t1 = t[t.length - 1];
    const x = (v) => m.l + (t1 === t0 ? iw / 2 : ((v - t0) / (t1 - t0)) * iw);
    const { max, step } = niceScale(Math.max(...series.flatMap((s) => s.values)));
    const y = (v) => m.t + ih - (v / max) * ih;
    const node = svgRoot(w, height, label);

    for (let v = 0; v <= max + 1e-9; v += step) {
      node.append(svg("line", { class: v === 0 ? "baseline" : "gridline", x1: m.l, x2: m.l + iw, y1: y(v), y2: y(v) }));
      node.append(svg("text", { class: "axis-text", x: m.l - 8, y: y(v) + 4, "text-anchor": "end" }, money(v)));
    }
    const nTicks = Math.max(2, Math.min(5, Math.floor(iw / 90)));
    for (let k = 0; k < nTicks; k++) {
      const i = Math.round((k * (dates.length - 1)) / (nTicks - 1));
      node.append(svg("text", { class: "axis-text", x: x(t[i]), y: height - 8, "text-anchor": k === 0 ? "start" : k === nTicks - 1 ? "end" : "middle" }, fmtDay(dates[i])));
    }
    for (const s of series) {
      const d = s.values.map((v, i) => `${i ? "L" : "M"}${x(t[i]).toFixed(1)},${y(v).toFixed(1)}`).join("");
      node.append(svg("path", { d, fill: "none", "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round", style: { stroke: s.color } }));
    }
    // End dots (with a surface ring) and selective end labels
    const ends = series.map((s) => ({ s, v: s.values[s.values.length - 1] })).map((e) => ({ ...e, yy: y(e.v) }));
    for (const e of ends) {
      node.append(svg("circle", { cx: x(t1), cy: e.yy, r: 4, "stroke-width": 2, style: { fill: e.s.color, stroke: "var(--surface)" } }));
    }
    if (showEnd) {
      const kept = [];
      for (const e of [...ends].sort((a, b) => a.yy - b.yy)) {
        if (kept.some((k) => Math.abs(k.yy - e.yy) < 15)) continue; // no nudging: legend + tooltip cover it
        kept.push(e);
        const txt = svg("text", { class: "label-text", x: x(t1) + 10, y: e.yy + 4 });
        txt.append(svg("tspan", { class: "value-text" }, money(e.v)), svg("tspan", { dx: 4 }, e.s.short || e.s.label));
        node.append(txt);
      }
    }

    // Hover layer: crosshair snaps to the nearest forecast date
    const cross = svg("line", { class: "crosshair", y1: m.t, y2: m.t + ih, visibility: "hidden" });
    const dots = series.map((s) => svg("circle", { r: 4, "stroke-width": 2, visibility: "hidden", style: { fill: s.color, stroke: "var(--surface)" } }));
    node.append(cross, ...dots);
    const overlay = svg("rect", { x: m.l, y: m.t, width: iw, height: ih, fill: "transparent" });
    node.append(overlay);
    let idx = dates.length - 1;
    function show(i, cx, cy) {
      idx = i;
      const px = x(t[i]);
      cross.setAttribute("x1", px); cross.setAttribute("x2", px); cross.setAttribute("visibility", "visible");
      dots.forEach((dot, k) => { dot.setAttribute("cx", px); dot.setAttribute("cy", y(series[k].values[i])); dot.setAttribute("visibility", "visible"); });
      showTip(cx, cy, `Forecast made ${fmtDate(dates[i])}`,
        series.map((s) => ({ color: s.color, value: money(s.values[i]), label: s.label })));
    }
    function hide() { cross.setAttribute("visibility", "hidden"); dots.forEach((d) => d.setAttribute("visibility", "hidden")); hideTip(); }
    const nearest = (px) => t.reduce((best, v, i) => (Math.abs(x(v) - px) < Math.abs(x(t[best]) - px) ? i : best), 0);
    overlay.addEventListener("pointermove", (e) => show(nearest(pointerX(node, e, w)), e.clientX, e.clientY));
    overlay.addEventListener("pointerleave", hide);
    const anchor = () => { const r = node.getBoundingClientRect(); return [r.left + (x(t[idx]) * r.width) / w, r.top + m.t]; };
    node.addEventListener("focus", () => show(idx, ...anchor()));
    node.addEventListener("blur", hide);
    node.addEventListener("keydown", (e) => {
      if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
      e.preventDefault();
      idx = Math.max(0, Math.min(dates.length - 1, idx + (e.key === "ArrowRight" ? 1 : -1)));
      show(idx, ...anchor());
    });
    plot.append(node);
  }

  // ---------- column chart (one series) ----------
  function columnChart(plot, w, { labels, values, color, label, tipTitle, tipNote, faded, height = 230 }) {
    const m = { l: 56, r: 8, t: 22, b: 30 };
    const iw = w - m.l - m.r, ih = height - m.t - m.b;
    const n = values.length, band = iw / n, bw = Math.min(24, band * 0.62);
    const { max, step } = niceScale(Math.max(...values));
    const y = (v) => m.t + ih - (v / max) * ih;
    const node = svgRoot(w, height, label);
    for (let v = 0; v <= max + 1e-9; v += step) {
      node.append(svg("line", { class: v === 0 ? "baseline" : "gridline", x1: m.l, x2: m.l + iw, y1: y(v), y2: y(v) }));
      node.append(svg("text", { class: "axis-text", x: m.l - 8, y: y(v) + 4, "text-anchor": "end" }, money(v)));
    }
    const every = Math.ceil(46 / band);
    const peak = values.indexOf(Math.max(...values));
    values.forEach((v, i) => {
      const bx = m.l + i * band + (band - bw) / 2;
      const hit = svg("rect", { x: m.l + i * band, y: m.t, width: band, height: ih, fill: "transparent", tabindex: "0", "aria-label": `${tipTitle(i)}: ${money(v)}` });
      const bar = svg("path", { d: colPath(bx, y(v), bw, y(0)), style: { fill: color, opacity: faded(i) ? 0.45 : 1 } });
      const on = (cx, cy) => { bar.style.opacity = faded(i) ? 0.3 : 0.8; showTip(cx, cy, tipTitle(i), [{ color, shape: "rect", value: money(v), label: tipNote(i) }]); };
      const off = () => { bar.style.opacity = faded(i) ? 0.45 : 1; hideTip(); };
      hit.addEventListener("pointermove", (e) => on(e.clientX, e.clientY));
      hit.addEventListener("pointerleave", off);
      hit.addEventListener("focus", () => { const r = hit.getBoundingClientRect(); on(r.left + r.width / 2, r.top + 20); });
      hit.addEventListener("blur", off);
      node.append(bar, hit);
      if (i % every === 0 || i === n - 1) {
        node.append(svg("text", { class: "axis-text", x: bx + bw / 2, y: height - 8, "text-anchor": "middle" }, labels[i]));
      }
      if (i === peak) node.append(svg("text", { class: "value-text", x: bx + bw / 2, y: y(v) - 6, "text-anchor": "middle" }, money(v)));
    });
    plot.append(node);
  }

  // ---------- horizontal grouped bars (several series per category) ----------
  function groupedBars(plot, w, { categories, series, label }) {
    const labelW = Math.min(104, w * 0.28);
    const m = { l: labelW, r: 58, t: 4, b: 26 };
    const barH = 12, gap = 2, groupGap = 16;
    const groupH = series.length * barH + (series.length - 1) * gap;
    const height = m.t + categories.length * (groupH + groupGap) + m.b;
    const iw = w - m.l - m.r;
    const { max, step } = niceScale(Math.max(...series.flatMap((s) => s.values)));
    const x = (v) => m.l + (v / max) * iw;
    const node = svgRoot(w, height, label);
    for (let v = 0; v <= max + 1e-9; v += step) {
      node.append(svg("line", { class: v === 0 ? "baseline" : "gridline", x1: x(v), x2: x(v), y1: m.t, y2: height - m.b }));
      if (w >= 360 || v === 0 || v >= max - 1e-9) {
        node.append(svg("text", { class: "axis-text", x: x(v), y: height - 8, "text-anchor": "middle" }, money(v)));
      }
    }
    categories.forEach((cat, ci) => {
      const gy = m.t + ci * (groupH + groupGap) + groupGap / 2;
      node.append(svg("text", { class: "label-text", x: m.l - 10, y: gy + groupH / 2 + 4, "text-anchor": "end" }, cat));
      const bars = series.map((s, si) => {
        const by = gy + si * (barH + gap);
        const bar = svg("path", { d: hBarPath(x(0), x(s.values[ci]), by, barH), style: { fill: s.color } });
        node.append(bar, svg("text", { class: "value-text", x: x(s.values[ci]) + 6, y: by + barH - 2 }, money(s.values[ci])));
        return bar;
      });
      const hit = svg("rect", { x: 0, y: gy - groupGap / 2, width: w, height: groupH + groupGap, fill: "transparent", tabindex: "0", "aria-label": `${cat}: ${series.map((s) => `${s.label} ${money(s.values[ci])}`).join(", ")}` });
      const on = (cx, cy) => { bars.forEach((b) => { b.style.opacity = 0.8; }); showTip(cx, cy, cat, series.map((s) => ({ color: s.color, shape: "rect", value: money(s.values[ci]), label: s.label }))); };
      const off = () => { bars.forEach((b) => { b.style.opacity = 1; }); hideTip(); };
      hit.addEventListener("pointermove", (e) => on(e.clientX, e.clientY));
      hit.addEventListener("pointerleave", off);
      hit.addEventListener("focus", () => { const r = hit.getBoundingClientRect(); on(r.left + r.width / 2, r.top); });
      hit.addEventListener("blur", off);
      node.append(hit);
    });
    plot.append(node);
  }

  // ---------- diverging bars (positive right, negative left) ----------
  function divergingBars(plot, w, { items, label }) {
    const labelW = Math.min(210, w * 0.46);
    const m = { l: labelW, r: 12, t: 4, b: 26 };
    const rowH = 26, barH = 14;
    const height = m.t + items.length * rowH + m.b;
    const iw = w - m.l - m.r;
    const maxAbs = niceScale(Math.max(...items.map((d) => Math.abs(d.value))), 2).max;
    const x = (v) => m.l + iw / 2 + (v / maxAbs) * (iw / 2 - 34);
    const node = svgRoot(w, height, label);
    node.append(svg("line", { class: "baseline", x1: x(0), x2: x(0), y1: m.t, y2: height - m.b }));
    node.append(svg("text", { class: "axis-text", x: x(0), y: height - 8, "text-anchor": "middle" }, "0"));
    const maxChars = Math.floor((labelW - 12) / 6.4);
    items.forEach((d, i) => {
      const cy = m.t + i * rowH + rowH / 2;
      const color = d.value >= 0 ? C.pos : C.neg;
      const short = d.label.length > maxChars ? `${d.label.slice(0, maxChars - 1)}…` : d.label;
      node.append(svg("text", { class: "label-text", x: m.l - 10, y: cy + 4, "text-anchor": "end" }, short));
      const bar = svg("path", { d: hBarPath(x(0), x(d.value), cy - barH / 2, barH), style: { fill: color } });
      const val = `${d.value >= 0 ? "+" : "−"}${Math.abs(d.value).toFixed(2)}`;
      node.append(bar, svg("text", { class: "value-text", x: x(d.value) + (d.value >= 0 ? 6 : -6), y: cy + 4, "text-anchor": d.value >= 0 ? "start" : "end" }, val));
      const hit = svg("rect", { x: 0, y: cy - rowH / 2, width: w, height: rowH, fill: "transparent", tabindex: "0", "aria-label": `${d.label}: ${val}` });
      const on = (cx, cyy) => { bar.style.opacity = 0.8; showTip(cx, cyy, d.label, [{ color, shape: "rect", value: val, label: d.value >= 0 ? "raises the score" : "lowers the score" }]); };
      const off = () => { bar.style.opacity = 1; hideTip(); };
      hit.addEventListener("pointermove", (e) => on(e.clientX, e.clientY));
      hit.addEventListener("pointerleave", off);
      hit.addEventListener("focus", () => { const r = hit.getBoundingClientRect(); on(r.left + r.width / 2, r.top); });
      hit.addEventListener("blur", off);
      node.append(hit);
    });
    plot.append(node);
  }

  // ---------- sections ----------
  function renderOverview(d) {
    const k = d.kpis;
    const f90 = d.forecast.today.find((f) => f.horizon_days === 90);
    const f30 = d.forecast.today.find((f) => f.horizon_days === 30);
    $("as-of").textContent = `data as of ${fmtDate(d.meta.as_of)}`;
    $("hero").append(
      el("p", { class: "hero-label", text: `Forecast revenue won in the next 90 days (to ${fmtDate(f90.window_end)})` }),
      el("div", { class: "hero-value", text: money(f90.forecast_usd) }),
      el("p", { class: "hero-sub", text: `Likely range ${money(f90.low_usd)} – ${money(f90.high_usd)}, based on how far past forecasts were off. The reps' own numbers suggest ${money(f90.rep_weighted_usd)}.` }));
    const tile = (label, value, foot, key) => el("div", { class: "tile" },
      el("p", { class: "tile-label" }, key ? el("span", { class: "key", style: { background: key } }) : null, label),
      el("p", { class: "tile-value", text: value }), el("p", { class: "tile-foot", text: foot }));
    $("tiles").append(
      tile("Open pipeline", money(k.pipeline_usd), `${k.open_deals} open deals`),
      tile("Weighted by Kairo model", money(k.model_weighted_usd), "Σ amount × model win chance", C.kairo),
      tile("Weighted by reps' estimates", money(k.rep_weighted_usd), "Σ amount × rep's probability", C.rep),
      tile("Forecast, next 30 days", money(f30.forecast_usd), `range ${money(f30.low_usd)} – ${money(f30.high_usd)}`),
      tile("Won, last 90 days", money(k.bookings_90d_usd), `${k.won_deals_90d} deals · win rate ${ratioPct(k.win_rate_90d)}`),
      tile("Hot leads", String(k.grade_a_leads), `grade A, of ${k.open_leads} open leads`));
  }

  function renderForecast(d) {
    const legend = [
      { label: "Actual revenue won", color: C.actual, shape: "line" },
      { label: "Kairo forecast", color: C.kairo, shape: "line" },
      { label: "Rep-weighted pipeline", color: C.rep, shape: "line" }];
    $("forecast-legend").append(legendRow(legend));
    for (const h of Object.keys(d.forecast.backtest).sort((a, b) => a - b)) {
      const bt = d.forecast.backtest[h];
      const card = el("div", { class: "card" });
      $("forecast-charts").append(card);
      const plot = chartCard(card, {
        title: `Next ${h} days`,
        sub: `Revenue won in the ${h} days after each forecast date`,
        table: {
          columns: [{ label: "Forecast date" }, { label: "Actual", num: true }, { label: "Kairo", num: true }, { label: "Rep-weighted", num: true }],
          rows: bt.map((r) => [fmtDate(r.date), moneyFull(r.actual_usd), moneyFull(r.kairo_usd), moneyFull(r.rep_weighted_usd)]),
        },
      });
      responsive(plot, (w) => lineChart(plot, w, {
        dates: bt.map((r) => r.date),
        label: `Backtest of the ${h}-day revenue forecast: actual revenue, Kairo forecast and rep-weighted pipeline for ${bt.length} past forecast dates`,
        series: [
          { label: "Actual", short: "Actual", color: C.actual, values: bt.map((r) => r.actual_usd) },
          { label: "Kairo forecast", short: "Kairo", color: C.kairo, values: bt.map((r) => r.kairo_usd) },
          { label: "Rep-weighted pipeline", short: "Reps", color: C.rep, values: bt.map((r) => r.rep_weighted_usd) }],
      }));
    }
    const acc = d.forecast.accuracy;
    const hs = Object.keys(acc).sort((a, b) => a - b);
    const methods = [["kairo", "Kairo forecast"], ["run_rate", "Same as the last period"], ["rep_weighted", "Rep-weighted pipeline (typical CRM)"]];
    $("forecast-accuracy").append(
      el("h3", { text: "Average error of past forecasts" }),
      el("div", { class: "table-scroll" }, dataTable({
        columns: [{ label: "Method" }, ...hs.map((h) => ({ label: `Next ${h} days`, num: true }))],
        rows: methods.map(([key, name]) => [name, ...hs.map((h) => ratioPct(acc[h][key]))]),
      })),
      el("p", { class: "chart-sub", style: { marginTop: "10px" }, text: `Average of |forecast − actual| ÷ actual over ${hs.map((h) => `${acc[h].forecasts} forecasts (${h} days)`).join(" and ")}. Lower is better.` }));
  }

  function renderPipeline(d) {
    const stages = d.pipeline_by_stage;
    const stagePlot = chartCard($("stage-chart"), {
      title: "Pipeline value by stage",
      sub: "Open deals, weighted by their chance of winning",
      legend: [{ label: "Kairo model", color: C.kairo }, { label: "Reps' estimate", color: C.rep }],
      table: {
        columns: [{ label: "Stage" }, { label: "Deals", num: true }, { label: "Pipeline", num: true }, { label: "Kairo-weighted", num: true }, { label: "Rep-weighted", num: true }],
        rows: stages.map((s) => [s.stage, String(s.deals), moneyFull(s.amount_usd), moneyFull(s.model_weighted_usd), moneyFull(s.rep_weighted_usd)]),
      },
    });
    responsive(stagePlot, (w) => groupedBars(stagePlot, w, {
      categories: stages.map((s) => s.stage),
      label: "Open pipeline by stage, weighted by the Kairo model and by reps' estimates",
      series: [
        { label: "Kairo model", color: C.kairo, values: stages.map((s) => s.model_weighted_usd) },
        { label: "Reps' estimate", color: C.rep, values: stages.map((s) => s.rep_weighted_usd) }],
    }));

    const months = d.bookings_monthly;
    const current = d.meta.as_of.slice(0, 7);
    const bookPlot = chartCard($("bookings-chart"), {
      title: "Revenue won per month",
      sub: `Closed-won deals since the CRM started${months.at(-1).month === current ? "; the last month is still in progress" : ""}`,
      table: {
        columns: [{ label: "Month" }, { label: "Revenue won", num: true }, { label: "Deals won", num: true }],
        rows: months.map((r) => [fmtMonthLong(r.month) + (r.month === current ? " (to date)" : ""), moneyFull(r.won_usd), String(r.won_deals)]),
      },
    });
    responsive(bookPlot, (w) => columnChart(bookPlot, w, {
      labels: months.map((r) => fmtMonth(r.month)),
      values: months.map((r) => r.won_usd),
      color: C.actual,
      label: "Revenue won per month",
      faded: (i) => months[i].month === current,
      tipTitle: (i) => fmtMonthLong(months[i].month) + (months[i].month === current ? " (to date)" : ""),
      tipNote: (i) => `${months[i].won_deals} deals won`,
    }));
  }

  function probRow(label, value, color) {
    return el("div", { class: "prob-row" },
      el("span", { text: label }),
      el("div", { class: "prob-track", role: "presentation" }, el("div", { class: "prob-fill", style: { width: `${Math.max(0, Math.min(100, value))}%`, background: color } })),
      el("span", { class: "prob-val", text: pct(value) }));
  }

  function renderDeals(d) {
    d.deals.forEach((deal, i) => {
      const b = deal.briefing;
      const briefingFoot = b && el("p", { class: "brief-foot", text: b.stale
        ? `Older AI briefing (written by ${b.model} on ${fmtDate(b.generated_on)}): the deal's facts have changed since, so some details may be out of date. ${b.stale_reason || ""}`
        : `AI briefing written by ${b.model} on ${fmtDate(b.generated_on)} from the facts shown here.` });
      const briefing = b && b.why
        ? el("div", { class: b.stale ? "briefing stale" : "briefing" },
          [["Situation", b.situation], ["Why", b.why], ["Action", b.action], ["Rep vs model", deal.rep_vs_model]]
            .filter(([, text]) => text)
            .map(([label, text]) => [el("p", { class: "brief-label", text: label }), el("p", { text })]),
          briefingFoot)
        : b // older briefing format (headline / risks / next steps)
          ? el("div", { class: "briefing" },
            el("h4", { text: b.headline }),
            el("p", { text: b.situation }),
            (b.risks || []).length ? [el("p", { class: "brief-label", text: "Risks" }), el("ul", {}, b.risks.map((r) => el("li", { text: r })))] : null,
            el("p", { class: "brief-label", text: "Next steps" }),
            el("ul", {}, (b.next_steps || []).map((s) => el("li", { text: s }))),
            briefingFoot)
          : el("div", { class: "briefing empty" },
          el("p", { text: "AI briefing unavailable for this deal right now. Briefings are written daily by Google Gemini for the top deals; when Gemini is busy, a deal keeps its last briefing or shows this note." }));
      const body = el("div", { class: "deal-body" },
        el("div", {},
          el("div", { class: "probs" },
            probRow("Rep's estimate", deal.rep_probability_pct, C.rep),
            probRow("Kairo model", deal.model_probability_pct, C.kairo),
            probRow("Won in 30 days", deal.p_won_30d_pct, C.kairo)),
          el("dl", { class: "facts", style: { marginTop: "14px" } },
            [["Owner", deal.owner_name], ["Deal type", deal.deal_type], ["Expected close", fmtDate(deal.expected_close_date)],
              ["Amount", moneyFull(deal.amount_usd)], ["Value at stake", moneyFull(deal.value_usd)]]
              .map(([k, v]) => el("div", {}, el("dt", { text: k }), el("dd", { text: v })))),
          deal.top_reasons && deal.top_reasons.length
            ? [el("p", { class: "brief-label", style: { marginTop: "14px" }, text: "Model's main reasons" }),
              el("ul", { class: "reasons" }, deal.top_reasons.map((r) =>
                el("li", { "data-sign": r.effect === "raises" ? "+" : "−", text: `${r.factor}: ${r.detail}` })))]
            : null,
          deal.risk_flags.length
            ? el("ul", { class: "flags", style: { marginTop: "12px" }, "aria-label": "Risk flags" },
              deal.risk_flags.map((f) => el("li", { class: "flag" }, el("span", { class: "flag-icon", "aria-hidden": "true", text: "!" }), `Risk: ${f}`)))
            : null),
        el("div", {},
          briefing,
          el("p", { class: "chart-sub", style: { marginTop: "10px" } }, el("strong", { text: `Why "${deal.action}": ` }), deal.reason)));
      const details = el("details", { class: "deal", open: i === 0 },
        el("summary", {},
          el("div", {}, el("span", { class: "deal-rank", text: `${i + 1}.` }), el("span", { class: "deal-name", text: deal.account_name })),
          el("div", { class: "deal-amount", text: money(deal.amount_usd) }),
          el("div", { class: "deal-meta" },
            el("span", { class: "action-chip", text: deal.action }),
            ` ${deal.stage} · closes ${fmtDay(deal.expected_close_date)} · ${deal.owner_name}`)),
        body);
      $("deal-list").append(details);
    });
  }

  function renderLeads(d) {
    const box = $("lead-list");
    box.append(el("h3", { text: `Top ${d.leads.top.length} open leads` }),
      el("p", { class: "chart-sub", text: d.leads.grades.map((g) => `${g.leads} grade ${g.grade}`).join(" · ") }));
    for (const lead of d.leads.top) {
      box.append(el("div", { class: "lead-row" },
        el("div", { class: "score", "aria-label": `Score ${Math.round(lead.score)}, grade ${lead.grade}` },
          el("div", {}, String(Math.round(lead.score)), el("small", { text: `grade ${lead.grade}` }))),
        el("div", {},
          el("div", { class: "lead-title", text: lead.account_name }),
          el("div", { class: "lead-meta", text: `${lead.source} · ${lead.industry} · ${lead.size_band} · ${lead.owner_name}` }),
          el("ul", { class: "reasons" }, lead.top_reasons.map((r) => {
            const sign = r.startsWith("- ") ? "−" : "+";
            return el("li", { "data-sign": sign, text: r.replace(/^[+-] /, "") });
          })))));
    }
    const drivers = d.leads.drivers;
    const plot = chartCard($("drivers-chart"), {
      title: "What moves a lead's score",
      sub: "Effect of each factor in the lead-scoring model (log-odds)",
      legend: [{ label: "Raises the score", color: C.pos }, { label: "Lowers the score", color: C.neg }],
      table: {
        columns: [{ label: "Factor" }, { label: "Effect", num: true }],
        rows: drivers.map((x) => [x.label, `${x.effect >= 0 ? "+" : "−"}${Math.abs(x.effect).toFixed(2)}`]),
      },
    });
    responsive(plot, (w) => divergingBars(plot, w, {
      items: drivers.map((x) => ({ label: x.label, value: x.effect })),
      label: "Strongest factors raising and lowering a lead's score",
    }));
  }

  function renderSegments(d) {
    for (const s of d.segments) {
      $("segments").append(el("div", { class: "card" },
        el("p", { class: "segment-name", text: s.segment_name }),
        el("p", { class: "segment-count", text: String(s.customers) }),
        el("p", { class: "segment-foot", text: "customers" }),
        el("dl", { class: "facts" },
          [["Median spend", money(s.revenue_median)], ["Total spend", money(s.revenue_total)],
            ["Last purchase", `${Math.round(s.days_since_last_win_median)} days ago`],
            ["Activities, 90 days", String(Math.round(s.activities_90d_median))],
            ["With an open deal", ratioPct(s.open_deals_share)]]
            .map(([k, v]) => el("div", {}, el("dt", { text: k }), el("dd", { text: v }))))));
    }
  }

  function renderActions(d) {
    const a = d.actions;
    $("action-summary").append(...[
      el("h3", { text: "All recommended actions" }),
      el("div", { class: "table-scroll" }, dataTable({
        columns: [{ label: "Action" }, { label: "Count", num: true }, { label: "Value at stake", num: true }],
        rows: a.summary.map((r) => [r.category === "hygiene" ? `${r.action} (hygiene)` : r.action, String(r.count), r.category === "hygiene" ? "–" : money(r.value_usd)]),
      })),
      a.owner_left_company
        ? el("p", { class: "chart-sub", style: { marginTop: "10px" }, text: `${a.owner_left_company} actions belong to accounts whose owner has left the company; they are flagged for reassignment.` })
        : null].filter(Boolean));
    $("action-top").append(
      el("h3", { text: `Top ${a.top.length} actions across the team` }),
      el("ol", { class: "action-list" }, a.top.map((r) => el("li", {},
        el("div", { class: "action-line" },
          el("span", {}, el("strong", { text: r.action }), ` — ${r.account_name}`),
          el("strong", { text: money(r.value_usd) })),
        el("div", { class: "action-who", text: `${r.owner_name} · ${r.object_type} · ${r.reason}` })))));
  }

  function renderModels(d) {
    const ls = d.models.lead_scoring || {}, wp = d.models.win_probability || {}, acc = d.forecast.accuracy;
    const card = (title, metric, metricLabel, ...lines) => el("div", { class: "card model-card" },
      el("h3", { text: title }), el("p", { class: "metric", text: metric }), el("p", { class: "metric-label", text: metricLabel }),
      lines.map((t) => el("p", { text: t })));
    const h = Object.keys(acc).sort((x, y) => y - x)[0];
    $("model-cards").append(
      card("Lead scoring", Number(ls.auc).toFixed(2), "ranking quality (AUC; 0.5 = random, 1 = perfect)",
        `Top-scored 20% of new leads converted ${ratioPct(ls.top20_conversion_rate)} of the time, against ${ratioPct(ls.avg_conversion_rate)} on average.`,
        `Backtest: built with only what was known on ${fmtDate(ls.backtest_cutoff)}, tested on ${ls.test_leads} later leads. Logistic regression, chosen over gradient boosting.`),
      card("Win probability", Number(wp.auc).toFixed(2), `ranking quality (AUC) — the reps' own estimates score ${Number(wp.rep_auc).toFixed(2)}`,
        `Predicted percentages match reality closely, while reps' high-confidence deals win far less often than they expect.`,
        `Backtest: model as of ${fmtDate(wp.backtest_cutoff)}, tested on ${wp.test_deals} later deals. Each deal counts once, however long it stays open.`),
      card("Revenue forecast", ratioPct(acc[h].kairo), `average error over ${h} days (${ratioPct(acc[h].rep_weighted)} for a rep-weighted pipeline)`,
        `Chance each open deal is won within the window × its expected amount, plus revenue from deals not created yet.`,
        `Walk-forward backtest: each past forecast rebuilt with only the data known that day.`));
  }

  // ---------- load data and render ----------
  fetch("data.json", { cache: "no-cache" })
    .then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
    .then((d) => {
      renderOverview(d);
      renderForecast(d);
      renderPipeline(d);
      renderDeals(d);
      renderLeads(d);
      renderSegments(d);
      renderActions(d);
      renderModels(d);
    })
    .catch((err) => {
      const box = $("load-error");
      box.hidden = false;
      box.textContent = `Couldn't load the data (${err.message}). If you opened index.html directly from disk, start a local web server instead: python -m http.server 8000 --directory site`;
    });
})();
