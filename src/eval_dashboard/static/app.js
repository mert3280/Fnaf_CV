/* Hand-tracking evaluation course — obstacle logic, timing, survey, submit.
 *
 * The hand tracker moves the real OS cursor and fires real clicks, so every
 * button here is driven by normal mouse events. We time three obstacles, then
 * (after /api/finish stops the tracker) collect a 1-5 survey and POST the run
 * to /api/results, which writes the doc file.
 */
"use strict";

const results = {
  startedAt: null, finishedAt: null,
  screen: { w: window.innerWidth, h: window.innerHeight },
  ob1: {}, ob2: {}, ob3: {}, survey: {},
};

const $ = (id) => document.getElementById(id);
const now = () => performance.now();

function showScreen(name) {
  document.querySelectorAll(".screen").forEach((s) => s.classList.remove("active"));
  $("screen-" + name).classList.add("active");
}

/* ---------- custom pointer (OS arrow is hidden) ---------- */
const handCursor = document.createElement("div");
handCursor.id = "handCursor";
document.body.appendChild(handCursor);
document.addEventListener("mousemove", (e) => {
  handCursor.style.transform = `translate(${e.clientX}px, ${e.clientY}px)`;
});
document.addEventListener("mousedown", () => {
  handCursor.classList.add("click");
  setTimeout(() => handCursor.classList.remove("click"), 140);
});

/* ================= MODEL PICKER ================= */
// Which checkpoint the next run loads; POSTed to /api/start. Populated from
// /api/models (models/*/best.pt on disk, enriched with DOCS/models test acc).
let selectedModel = null;
let modelOpts = [];

async function loadModels() {
  let data;
  try { data = await (await fetch("/api/models")).json(); }
  catch (_) { return; }                      // server not up
  modelOpts = data.models || [];
  if (!modelOpts.length) { renderModelCompare(modelOpts); return; }
  selectedModel = data.current || modelOpts[0].path;
  renderModelCompare(modelOpts);             // Overview tab's metrics table
  const sel = $("modelSelect");
  sel.innerHTML = modelOpts.map((m) =>
    `<option value="${m.path}">${m.id}${m.test_acc ? ` — ${(+m.test_acc * 100).toFixed(1)}% test acc` : ""}</option>`
  ).join("");
  sel.value = selectedModel;
  $("modelPicker").hidden = false;
  updateModelBlurb();
  updateStrategyVisibility();
}

function updateModelBlurb() {
  const m = modelOpts.find((x) => x.path === selectedModel);
  $("modelBlurb").textContent = m
    ? [m.classes && `${m.classes} classes`, m.crop_mode && `${m.crop_mode} crop`,
       m.test_acc && `${(+m.test_acc * 100).toFixed(1)}% test acc`].filter(Boolean).join(" · ")
    : "";
}

// Only fist-vs-rest checkpoints support the 3.2/3.2.1 click-FSM presets (AD-21).
function updateStrategyVisibility() {
  const m = modelOpts.find((x) => x.path === selectedModel);
  $("strategyPicker").hidden = !(m && m.fistvsrest && strategyOpts.length);
}

$("modelSelect").addEventListener("change", (e) => {
  selectedModel = e.target.value;
  updateModelBlurb();
  updateStrategyVisibility();
  renderModelCompare(modelOpts);
});

// Once the tracker is running the checkpoint is loaded — lock the pick.
function lockModelPicker() { $("modelSelect").disabled = true; }

/* ================= STRATEGY PICKER (fist-vs-rest: 3.2 vs 3.2.1) ================= */
// Picks which click-FSM preset (conf + K) the run uses; POSTed to /api/start.
// Visibility is gated by the selected model (updateStrategyVisibility above),
// not by the server default, so switching models updates it live.
let selectedStrategy = null;
let strategyOpts = [];

async function loadStrategies() {
  let data;
  try { data = await (await fetch("/api/strategies")).json(); }
  catch (_) { return; }                      // server not up / no tracker
  if (!(data.options || []).length) return;
  strategyOpts = data.options;
  selectedStrategy = data.current || data.options[0].id;
  const box = $("strategyOptions");
  box.innerHTML = "";
  data.options.forEach((o) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "sp-opt" + (o.id === selectedStrategy ? " sel" : "");
    b.textContent = o.label;
    b.dataset.id = o.id;
    b.addEventListener("click", () => selectStrategy(o.id));
    box.appendChild(b);
  });
  updateStrategyBlurb();
  updateStrategyVisibility();
}

function selectStrategy(id) {
  selectedStrategy = id;
  [...$("strategyOptions").children].forEach((c) => c.classList.toggle("sel", c.dataset.id === id));
  updateStrategyBlurb();
}

function updateStrategyBlurb() {
  const o = strategyOpts.find((x) => x.id === selectedStrategy);
  $("strategyBlurb").textContent = o ? o.blurb : "";
}

// Once the tracker is running the preset is baked into the loop — lock the pick.
function lockStrategyPicker() {
  [...$("strategyOptions").children].forEach((c) => (c.disabled = true));
}

/* ================= WELCOME ================= */
let trackerStarting = false;

$("startTrackerBtn").addEventListener("click", async () => {
  trackerStarting = true;
  const b = $("startTrackerBtn");
  b.disabled = true; b.textContent = "Starting camera…";
  $("trackerHint").textContent = "Loading the model and opening the camera…";
  try {
    const j = await (await fetch("/api/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ checkpoint: selectedModel, strategy: selectedStrategy }),
    })).json();
    if (!j.ok) throw new Error(j.error || "could not start tracker");
    lockModelPicker();
    lockStrategyPicker();
  } catch (e) {
    trackerStarting = false;
    b.disabled = false; b.textContent = "1 · Start hand tracker";
    $("trackerHint").textContent = "Could not start: " + e.message;
  }
});

$("startBtn").addEventListener("click", () => {
  results.startedAt = new Date().toISOString();
  startOb1();
});

/* ================= WELCOME TABS: Start / Overview / Help ================= */
function selectTab(name) {
  $("tabStart").classList.toggle("active", name === "start");
  $("tabOverview").classList.toggle("active", name === "overview");
  $("tabHelp").classList.toggle("active", name === "help");
  $("panelStart").classList.toggle("active", name === "start");
  $("panelOverview").classList.toggle("active", name === "overview");
  $("panelHelp").classList.toggle("active", name === "help");
  // charts + the full metric table need a wider card than the start panel
  $("screen-welcome").querySelector(".card").classList.toggle("card-xw", name !== "start");
  if (name === "overview") loadOverview();
}
$("tabStart").addEventListener("click", () => selectTab("start"));
$("tabOverview").addEventListener("click", () => selectTab("overview"));
$("tabHelp").addEventListener("click", () => selectTab("help"));

/* ---------- Overview: pull past runs and render scores ---------- */
const secs = (ms) => (typeof ms === "number" ? (ms / 1000).toFixed(1) + "s" : "—");
const accTxt = (a) => (typeof a === "number" ? a.toFixed(0) : "—");

function runDate(r) {
  const iso = r.finishedAt || r.startedAt;
  if (iso) { const d = new Date(iso); if (!isNaN(d)) return d.toLocaleString(); }
  const m = /(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})/.exec(r.id || "");
  return m ? `${m[1]}-${m[2]}-${m[3]} ${m[4]}:${m[5]}` : (r.id || "—");
}

// Numeric timestamp for a run (for date sorting); mirrors runDate's fallbacks.
function dateVal(r) {
  const iso = r.finishedAt || r.startedAt;
  if (iso) { const d = new Date(iso); if (!isNaN(d)) return d.getTime(); }
  const m = /(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})/.exec(r.id || "");
  if (m) return new Date(+m[1], +m[2] - 1, +m[3], +m[4], +m[5], +m[6]).getTime();
  return null;
}

function ovTile(val, label, sub) {
  return `<div class="ov-tile"><b>${val}</b><span>${label}</span>` +
         (sub ? `<small>${sub}</small>` : "") + `</div>`;
}

/* ---------- Overview: offline model-metrics table (from /api/models) ---------- */
function renderModelCompare(models) {
  const wrap = $("modelCompare");
  if (!models.length) { wrap.hidden = true; return; }
  wrap.hidden = false;
  const best = models.reduce((a, b) =>
    (+b.test_acc || -1) > (+(a && a.test_acc) || -1) ? b : a, null);
  const rows = models.map((m) => {
    const acc = m.test_acc ? (+m.test_acc * 100).toFixed(1) + "%" : "—";
    const isBest = best && m.id === best.id && m.test_acc;
    return `<tr${m.path === selectedModel ? ' class="mc-current"' : ""}>` +
      `<td>${m.id}${m.path === selectedModel ? ' <span class="mc-tag">selected</span>' : ""}</td>` +
      `<td>${m.classes || "—"}</td>` +
      `<td>${m.crop_mode || "—"}</td>` +
      `<td class="num">${acc}${isBest ? ' <span class="mc-tag mc-tag-best">best</span>' : ""}</td>` +
      `</tr>`;
  }).join("");
  $("modelCompareTable").innerHTML =
    `<table class="ov-table mc-table"><thead><tr>` +
    `<th>Model</th><th>Classes</th><th>Crop</th><th class="num">Test acc</th>` +
    `</tr></thead><tbody>${rows}</tbody></table>`;
}

/* ---------- Overview: filtering + sorting ----------
 * Metrics are read in the units shown in the table: times are converted from
 * the stored milliseconds to seconds, so "precision < 5" means 5 s, not 5 ms.
 * `better` marks which end of the scale is a good run — it powers the
 * "Best / Worst first" sort directions and the highlighted bar in the chart.
 * `sortOnly` entries can order the table but not feed a numeric filter/chart;
 * `str` entries sort alphabetically.
 */
const toSec = (ms) => (typeof ms === "number" ? ms / 1000 : null);
const asNum = (v) => (typeof v === "number" ? v : null);
const OV_METRICS = [
  { key: "date",     label: "Date",     sortOnly: true, better: "high", get: dateVal },
  { key: "model",    label: "Model",    sortOnly: true, str: true, get: (r) => r.model || null },
  { key: "strategy", label: "Strategy", sortOnly: true, str: true, get: (r) => r.strategy || null },
  { key: "ob1Ms", label: "Precision (s)", better: "low",
    get: (r) => toSec(r.ob1Ms), fmt: (v) => v.toFixed(2) + "s" },
  { key: "ob2Ms", label: "Rapid (s)", better: "low",
    get: (r) => toSec(r.ob2Ms), fmt: (v) => v.toFixed(2) + "s" },
  { key: "clickRate", label: "Click rate (/s)", better: "high",
    get: (r) => (typeof r.ob2Ms === "number" && r.ob2Ms > 0 && typeof r.ob2Clicks === "number"
                 ? r.ob2Clicks / (r.ob2Ms / 1000) : null),
    fmt: (v) => v.toFixed(2) + "/s" },
  { key: "ob3Accuracy", label: "Trace accuracy", better: "high",
    get: (r) => asNum(r.ob3Accuracy), fmt: (v) => v.toFixed(0) },
  { key: "ob3Coverage", label: "Coverage (%)", better: "high",
    get: (r) => (typeof r.ob3Coverage === "number" ? r.ob3Coverage * 100 : null),
    fmt: (v) => v.toFixed(0) + "%" },
  { key: "ob3MeanDevPx", label: "Deviation (px)", better: "low",
    get: (r) => asNum(r.ob3MeanDevPx), fmt: (v) => v.toFixed(1) },
  { key: "surveyAvg", label: "Survey (1-5)", better: "high",
    get: (r) => asNum(r.surveyAvg), fmt: (v) => v.toFixed(1) },
];
const ovMetric = (key) => OV_METRICS.find((m) => m.key === key);
const OV_NUMERIC = OV_METRICS.filter((m) => !m.sortOnly);   // chartable

// Fixed-order categorical palette (validated for CVD + contrast; see the
// dataviz skill). Assigned by identity (model/strategy name), never by rank,
// so a value keeps its color as filters change what's on screen.
const OV_PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"];
function buildOvColorMap(runs, key) {
  const uniq = [...new Set(runs.map((r) => r[key] || "—"))].sort();
  const map = new Map();
  uniq.forEach((v, i) => map.set(v, OV_PALETTE[i % OV_PALETTE.length]));
  return map;
}
function ovColorFor(r) {
  const key = ovState.colorBy === "model" ? (r.model || "—") : (r.strategy || "—");
  return ovState.colorMaps[ovState.colorBy].get(key) || "#94a3b8";
}

const ovState = {
  runs: [], strategy: "", model: "", search: "",
  sortKey: "date", sortDir: "best",  // sortDir: best | worst
  barMetric: "ob1Ms",                // metric charted per model/strategy pair
  barAgg: "mean",                    // "mean" | "count"
  colorBy: "strategy",               // which field the chart bars are colored by
  colorMaps: { strategy: new Map(), model: new Map() },
  wired: false,                      // controls get their listeners once
};

// Free-text search over the descriptive fields; every token must match somewhere.
function ovSearchHit(r) {
  const q = ovState.search.trim().toLowerCase();
  if (!q) return true;
  const hay = [r.model, r.strategy, r.id, runDate(r)].filter(Boolean).join(" ").toLowerCase();
  return q.split(/\s+/).every((tok) => hay.includes(tok));
}

function applyOvFilters(runs) {
  return runs.filter((r) => {
    if (ovState.strategy && (r.strategy || "") !== ovState.strategy) return false;
    if (ovState.model && (r.model || "") !== ovState.model) return false;
    if (!ovSearchHit(r)) return false;
    return true;
  });
}

// Resolve best/worst per metric: for times and deviation, LOWER is the good end.
function ovDir(m) {
  const best = m.str ? "asc" : (m.better === "low" ? "asc" : "desc");
  return ovState.sortDir === "worst" ? (best === "asc" ? "desc" : "asc") : best;
}

function sortOvRuns(runs) {
  const m = ovMetric(ovState.sortKey);
  const dir = ovDir(m) === "asc" ? 1 : -1;
  return [...runs].sort((a, b) => {
    const va = m.get(a), vb = m.get(b);
    const ha = va !== null && va !== undefined, hb = vb !== null && vb !== undefined;
    if (ha && hb)
      return (m.str ? String(va).localeCompare(String(vb)) : va - vb) * dir;
    return ha ? -1 : hb ? 1 : 0;      // runs missing the value sink to the bottom
  });
}

// Rebuild a select's options from the unique values present, keeping the pick.
function fillOvSelect(sel, values, current, allLabel) {
  sel.innerHTML = `<option value="">${allLabel}</option>` +
    values.map((v) => `<option value="${v}">${v}</option>`).join("");
  sel.value = values.includes(current) ? current : "";
}

function populateOvSelects() {
  const uniq = (key) => [...new Set(ovState.runs.map((r) => r[key]).filter(Boolean))].sort();
  fillOvSelect($("ovStrategy"), uniq("strategy"), ovState.strategy, "All strategies");
  fillOvSelect($("ovModel"), uniq("model"), ovState.model, "All models");
  ovState.strategy = $("ovStrategy").value;   // stay in sync if a value vanished
  ovState.model = $("ovModel").value;
}

function wireOvControls() {
  if (ovState.wired) return;
  ovState.wired = true;

  const opts = (list) => list.map((m) => `<option value="${m.key}">${m.label}</option>`).join("");
  $("ovSort").innerHTML = opts(OV_METRICS);
  $("ovSort").value = ovState.sortKey;
  $("ovSortDir").value = ovState.sortDir;
  $("ovBarMetric").innerHTML = opts(OV_NUMERIC);
  $("ovBarMetric").value = ovState.barMetric;
  $("ovBarAgg").value = ovState.barAgg;
  $("ovColorBy").value = ovState.colorBy;

  $("ovSearch").addEventListener("input", (e) => { ovState.search = e.target.value; renderOverview(); });
  $("ovStrategy").addEventListener("change", (e) => { ovState.strategy = e.target.value; renderOverview(); });
  $("ovModel").addEventListener("change", (e) => { ovState.model = e.target.value; renderOverview(); });
  $("ovSort").addEventListener("change", (e) => { ovState.sortKey = e.target.value; renderOverview(); });
  $("ovSortDir").addEventListener("change", (e) => { ovState.sortDir = e.target.value; renderOverview(); });
  $("ovBarMetric").addEventListener("change", (e) => { ovState.barMetric = e.target.value; renderOverview(); });
  $("ovBarAgg").addEventListener("change", (e) => { ovState.barAgg = e.target.value; renderOverview(); });
  $("ovColorBy").addEventListener("change", (e) => { ovState.colorBy = e.target.value; renderOverview(); });

  // Column headers sort too: first click puts the best runs on top, a second
  // click flips the direction. The "Sort by" select stays in sync.
  $("ovTable").addEventListener("click", (e) => {
    const th = e.target.closest("th[data-key]");
    if (!th) return;
    const key = th.dataset.key;
    ovState.sortDir = (ovState.sortKey === key && ovState.sortDir === "best") ? "worst" : "best";
    ovState.sortKey = key;
    $("ovSort").value = ovState.sortKey;
    $("ovSortDir").value = ovState.sortDir;
    renderOverview();
  });

  $("ovClear").addEventListener("click", () => {
    ovState.strategy = ""; ovState.model = ""; ovState.search = "";
    $("ovStrategy").value = ""; $("ovModel").value = ""; $("ovSearch").value = "";
    renderOverview();
  });

  // Charts are sized to the card, so re-render (debounced) when it changes.
  let resizeT = null;
  window.addEventListener("resize", () => {
    if (!$("panelOverview").classList.contains("active")) return;
    clearTimeout(resizeT);
    resizeT = setTimeout(renderOverview, 150);
  });
}

/* ---------- Overview: comparison charts (plain inline SVG, no chart lib) ----------
 * Both charts read the same filtered/sorted `shown` slice as the table, and
 * color every mark by whichever field (model/strategy) `ovColorBy` picks —
 * that's the "graphic comparison between runs" the model-metrics table can't do.
 */
function renderOvLegend(el, runs) {
  const key = ovState.colorBy;
  const map = ovState.colorMaps[key];
  const uniq = [...new Set(runs.map((r) => r[key] || "—"))].sort();
  if (uniq.length < 2) { el.hidden = true; el.innerHTML = ""; return; }  // one series needs no legend
  el.hidden = false;
  el.innerHTML = uniq.map((v) =>
    `<span class="ov-legend-item"><span class="ov-legend-swatch" style="background:${map.get(v)}"></span>${v}</span>`
  ).join("");
}

// A bar rounded only at its free (top) end, square at the baseline (mark spec).
function roundedTopRectPath(x, y, w, h, r) {
  r = Math.max(0, Math.min(r, w / 2, h));
  return `M${x},${y + h} L${x},${y + r} Q${x},${y} ${x + r},${y} ` +
         `L${x + w - r},${y} Q${x + w},${y} ${x + w},${y + r} L${x + w},${y + h} Z`;
}

// Minimal escaping for values (model/strategy names) interpolated into raw
// SVG text/attribute strings below.
function escapeXml(s) {
  return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

// One group per (model, strategy) pair — this is the axis the tester actually
// wants to compare ("which pairing wins"), not individual trials or time.
function buildOvGroups(runs) {
  const map = new Map();
  runs.forEach((r) => {
    const model = r.model || "—", strategy = r.strategy || "—";
    const key = model + "␟" + strategy;
    if (!map.has(key)) map.set(key, { model, strategy, label: `${model} · ${strategy}`, rows: [] });
    map.get(key).rows.push(r);
  });
  return [...map.values()];
}

function renderOvGroupChart(runs) {
  const body = $("ovBarBody"), note = $("ovBarNote");
  const metric = ovMetric(ovState.barMetric);
  const agg = ovState.barAgg;                          // "mean" | "count"
  const fmt = (v) => (metric.fmt ? metric.fmt(v) : v.toFixed(1));
  const groups = buildOvGroups(runs);

  const pts = groups
    .map((g) => {
      const vals = g.rows.map(metric.get).filter((v) => typeof v === "number");
      const y = agg === "count" ? vals.length : (vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null);
      return { g, y, n: vals.length };
    })
    .filter((p) => typeof p.y === "number" && (agg === "count" ? p.y > 0 : p.n > 0));
  renderOvLegend($("ovBarLegend"), runs);

  if (!pts.length) {
    body.innerHTML = ""; note.hidden = false;
    note.textContent = `No trials have "${metric.label}" recorded.`;
    return;
  }
  note.hidden = true;

  // Rank so the best-performing pair sits at the left (mean: metric's own
  // better direction; count: most trials first) — that's the "key in on the
  // winner" view this chart exists for.
  const asc = agg === "count" ? false : metric.better === "low";
  pts.sort((a, b) => (asc ? a.y - b.y : b.y - a.y));

  const n = pts.length;
  const slot = 110, barW = 48;
  const padL = 48, padR = 16, padT = 16, padB = 46;
  const innerW = slot * n;
  const W = Math.max(body.clientWidth || 0, padL + innerW + padR);
  const H = 280;
  const innerH = H - padT - padB;
  const yMax = Math.max(...pts.map((p) => p.y), 1e-6) * 1.15;   // headroom for the value label
  const yScale = (v) => padT + innerH - (v / yMax) * innerH;
  const baseY = padT + innerH;
  const valFmt = (v) => (agg === "count" ? String(v) : fmt(v));
  const aggLabel = agg === "count" ? `Count of ${metric.label}` : `Mean ${metric.label}`;

  // Each bar's model/strategy label is truncated to fit its own slot width —
  // horizontal, not rotated, so it can never run into a neighboring bar or the
  // chart's edges. The full untruncated label is still on the bar's tooltip.
  const measureCtx = (renderOvGroupChart._ctx ||= document.createElement("canvas").getContext("2d"));
  measureCtx.font = "600 10px 'Segoe UI', system-ui, sans-serif";   // matches .ov-xlabel weight
  const labelBudget = slot - 8;
  function fitLabel(str) {
    if (measureCtx.measureText(str).width <= labelBudget) return str;
    let lo = 0, hi = str.length;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (measureCtx.measureText(str.slice(0, mid) + "…").width <= labelBudget) lo = mid; else hi = mid - 1;
    }
    return str.slice(0, lo) + "…";
  }

  let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Model/strategy comparison: ${aggLabel}">`;
  const gridN = 4;
  for (let i = 0; i <= gridN; i++) {
    const v = (i / gridN) * yMax;
    const y = yScale(v);
    svg += `<line class="ov-grid-line" x1="${padL}" x2="${W - padR}" y1="${y.toFixed(1)}" y2="${y.toFixed(1)}"/>`;
    svg += `<text class="ov-tick-label" x="${padL - 6}" y="${(y + 3).toFixed(1)}" text-anchor="end">${agg === "count" ? Math.round(v) : fmt(v)}</text>`;
  }
  svg += `<line class="ov-axis-line" x1="${padL}" x2="${W - padR}" y1="${baseY}" y2="${baseY}"/>`;
  pts.forEach((p, i) => {
    const cx = padL + slot * i + slot / 2;
    const y = yScale(p.y);
    const h = Math.max(2, baseY - y);
    const color = ovColorFor(p.g.rows[0]);
    const path = roundedTopRectPath(cx - barW / 2, y, barW, h, 4);
    const title = `${p.g.label} — ${aggLabel} ${valFmt(p.y)} (n=${p.n})`;
    svg += `<path class="ov-bar" d="${path}" fill="${color}"><title>${escapeXml(title)}</title></path>`;
    svg += `<text class="ov-tick-label" x="${cx}" y="${(y - 6).toFixed(1)}" text-anchor="middle">${valFmt(p.y)}</text>`;
    svg += `<text class="ov-tick-label ov-xlabel" x="${cx}" y="${baseY + 18}" text-anchor="middle">` +
           `${escapeXml(fitLabel(p.g.label))}<title>${escapeXml(p.g.label)}</title></text>`;
  });
  svg += `</svg>`;
  body.innerHTML = svg;
}

async function loadOverview() {
  let runs = [];
  try { runs = (await (await fetch("/api/history")).json()).runs || []; }
  catch (_) { /* server not up */ }
  ovState.runs = runs;
  // Built from the FULL set (not the filtered slice) so a value's color stays
  // put as filters narrow what's on screen (color follows identity, not rank).
  ovState.colorMaps = { strategy: buildOvColorMap(runs, "strategy"), model: buildOvColorMap(runs, "model") };
  wireOvControls();
  populateOvSelects();
  renderOverview();
}

function renderOverview() {
  const agg = $("ovAgg"), tbl = $("ovTable"), empty = $("ovEmpty");
  const wrap = $("ovTableWrap"), filters = $("ovFilters"), charts = $("ovCharts");

  if (!ovState.runs.length) {                  // nothing recorded at all
    agg.innerHTML = ""; tbl.innerHTML = "";
    filters.hidden = true; wrap.hidden = true; charts.hidden = true;
    empty.hidden = false;
    empty.textContent = "No trials recorded yet — run the course to see results here.";
    return;
  }
  filters.hidden = false;

  const shown = sortOvRuns(applyOvFilters(ovState.runs));
  if (!shown.length) {                         // runs exist but the filter hides them all
    agg.innerHTML = ""; tbl.innerHTML = "";
    wrap.hidden = true; charts.hidden = true; empty.hidden = false;
    empty.textContent = "No trials match these filters.";
    return;
  }
  empty.hidden = true; wrap.hidden = false;

  // Charts need at least 2 trials to be a comparison; below that just show the table.
  charts.hidden = shown.length < 2;
  if (shown.length >= 2) {
    renderOvGroupChart(shown);
  }

  // Aggregate headline scores across the matching runs (best where lower/higher is better).
  const nums = (sel) => shown.map(sel).filter((v) => typeof v === "number");
  const t1 = nums((r) => r.ob1Ms);
  const accs = nums((r) => r.ob3Accuracy), surv = nums((r) => r.surveyAvg);
  const mean = (a) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : null);
  const bestLo = (a) => (a.length ? Math.min(...a) : null);
  const bestHi = (a) => (a.length ? Math.max(...a) : null);
  const countLabel = shown.length === ovState.runs.length
    ? "trials recorded" : `of ${ovState.runs.length} trials shown`;

  agg.innerHTML =
    ovTile(shown.length, countLabel, "") +
    ovTile(secs(bestLo(t1)), "best precision", t1.length ? `avg ${secs(mean(t1))}` : "") +
    ovTile(accTxt(bestHi(accs)), "best trace accuracy", accs.length ? `avg ${accTxt(mean(accs))}` : "") +
    ovTile(surv.length ? mean(surv).toFixed(1) + "/5" : "—", "avg survey", "1 poor · 5 great");

  const head =
    `<thead><tr><th>When</th><th>Model</th><th>Strategy</th>` +
    `<th class="num">Precision</th><th class="num">Rapid</th>` +
    `<th class="num">Trace</th><th class="num">Survey</th></tr></thead>`;
  const rows = shown.map((r) =>
    `<tr><td class="muted">${runDate(r)}</td>` +
    `<td>${r.model || "—"}</td>` +
    `<td class="strat">${r.strategy || "—"}</td>` +
    `<td class="num">${secs(r.ob1Ms)}</td>` +
    `<td class="num">${secs(r.ob2Ms)}</td>` +
    `<td class="num">${accTxt(r.ob3Accuracy)}</td>` +
    `<td class="num">${typeof r.surveyAvg === "number" ? r.surveyAvg.toFixed(1) : "—"}</td></tr>`
  ).join("");
  tbl.innerHTML = head + `<tbody>${rows}</tbody>`;
}

// Reflect tracker state on the welcome screen: gate the course button on a
// running tracker (or open it immediately in --no-tracker UI mode).
function updateWelcome(s) {
  if (!$("screen-welcome").classList.contains("active")) return;
  const startBtn = $("startBtn"), trackBtn = $("startTrackerBtn");
  const hint = $("trackerHint"), controls = $("trackerControls");
  if (!s.tracker) {                       // --no-tracker: use the normal mouse
    controls.hidden = true;
    startBtn.disabled = false;
    return;
  }
  if (s.error) {
    trackerStarting = false;
    trackBtn.disabled = false; trackBtn.textContent = "1 · Start hand tracker";
    hint.textContent = "Tracker error: " + s.error;
    startBtn.disabled = true;
  } else if (s.running && s.input_enabled) {
    trackerStarting = false;
    trackBtn.disabled = true; trackBtn.textContent = "Hand tracker on ✓";
    hint.textContent = "Camera on — use your hand. Fist = click, open palm = release.";
    startBtn.disabled = false;
  } else if (!trackerStarting) {
    trackBtn.disabled = false; trackBtn.textContent = "1 · Start hand tracker";
    hint.textContent = "Click this with your mouse to turn the camera on.";
    startBtn.disabled = true;
  }
}

/* ================= OBSTACLE 1: PRECISION ================= */
const OB1_COLORS = ["#ec4899", "#3b82f6", "#f97316", "#8b5cf6", "#22c55e"];
const ob1 = { n: 0, shownAt: 0, clickAt: [], shown: [], last: null };

function startOb1() {
  showScreen("ob1");
  ob1.n = 0; ob1.clickAt = []; ob1.shown = []; ob1.last = null;
  $("ob1Progress").textContent = "0 / 5";
  spawnNumButton(1);
}

function spawnNumButton(n) {
  const box = $("ob1Box").getBoundingClientRect();
  const size = 92, pad = size / 2 + 12;
  let x, y, tries = 0;
  do {
    x = pad + Math.random() * (box.width - 2 * pad);
    y = pad + Math.random() * (box.height - 2 * pad);
    tries++;
  } while (ob1.last && Math.hypot(x - ob1.last.x, y - ob1.last.y) < 220 && tries < 40);
  ob1.last = { x, y };

  const btn = document.createElement("button");
  btn.className = "num-btn";
  btn.textContent = n;
  btn.style.left = x + "px";
  btn.style.top = y + "px";
  btn.style.background = OB1_COLORS[(n - 1) % OB1_COLORS.length];
  btn.addEventListener("click", () => onNumClick(n, btn), { once: true });
  $("ob1Box").appendChild(btn);
  ob1.shown[n] = now();
}

function onNumClick(n, btn) {
  btn.remove();
  ob1.clickAt[n] = now();
  $("ob1Progress").textContent = `${n} / 5`;
  if (n < 5) {
    spawnNumButton(n + 1);
  } else {
    // Timed span is click-1 -> click-5 (per spec); also keep the fuller total.
    const splits = [];
    for (let i = 1; i <= 5; i++) splits.push(Math.round(ob1.clickAt[i] - ob1.shown[i]));
    results.ob1 = {
      timeMs: Math.round(ob1.clickAt[5] - ob1.clickAt[1]),
      totalMs: Math.round(ob1.clickAt[5] - ob1.shown[1]),
      splitsMs: splits,
    };
    startOb2();
  }
}

/* ================= OBSTACLE 2: RAPID CLICKS ================= */
const OB2_TARGET = 7;
const ob2 = { count: 0, firstAt: 0 };

function startOb2() {
  showScreen("ob2");
  ob2.count = 0; ob2.firstAt = 0;
  $("ob2Progress").textContent = `0 / ${OB2_TARGET}`;
  $("ob2Depth").style.height = "0%";
}

$("downBtn").addEventListener("click", () => {
  ob2.count++;
  if (ob2.count === 1) ob2.firstAt = now();
  $("ob2Progress").textContent = `${ob2.count} / ${OB2_TARGET}`;
  $("ob2Depth").style.height = (100 * ob2.count / OB2_TARGET) + "%";
  const b = $("downBtn"); b.classList.remove("flash"); void b.offsetWidth; b.classList.add("flash");
  if (ob2.count >= OB2_TARGET) {
    results.ob2 = { timeMs: Math.round(now() - ob2.firstAt), clicks: OB2_TARGET };
    setTimeout(startOb3, 250);
  }
});

/* ================= OBSTACLE 3: TRACER ================= */
const REACH_TOL = 52;   // px: how close counts as "covered this point"
const END_TOL = 60;     // px: cursor within this of the end dot to finish
const DEV_MAX = 70;     // px: deviation at which accuracy credit hits 0 (scoring only)
const COVER_MIN = 0.70; // fraction of the path that must be traced to finish
const ob3 = { active: false, startAt: 0, pts: [], reached: [], devs: [], trail: [], svgRect: null };

function startOb3() {
  showScreen("ob3");
  $("ob3Meta").textContent = "accuracy —";
  $("ob3Cursor").setAttribute("hidden", "");
  $("traceStartBtn").style.display = "";
  ob3.active = false;
  // Preview the path so the tester sees the line + start/finish before tracing.
  buildTracePath();                              // also positions the START button at pts[0]
  $("traceStartBtn").style.display = "";
}

$("traceStartBtn").addEventListener("click", () => {
  buildTracePath();                              // rebuild fresh (handles resize) + clear trail
  $("traceStartBtn").style.display = "none";     // reveals the yellow start dot underneath
  $("ob3Cursor").removeAttribute("hidden");
  ob3.active = true;
  ob3.startAt = now();
});

function buildTracePath() {
  const svg = $("ob3Svg");
  const r = svg.getBoundingClientRect();
  ob3.svgRect = r;
  const W = r.width, H = r.height;
  const mx = 110, midY = H / 2, amp = Math.min(H * 0.3, 260), cycles = 1.6;
  const N = 60, pts = [];
  for (let i = 0; i < N; i++) {
    const t = i / (N - 1);
    pts.push({ x: mx + t * (W - 2 * mx), y: midY + amp * Math.sin(t * cycles * 2 * Math.PI) });
  }
  ob3.pts = pts;
  ob3.reached = new Array(N).fill(false);
  ob3.devs = [];
  ob3.trail = [];
  $("ob3Trail").setAttribute("points", "");      // wipe any prior breadcrumb

  let d = `M ${pts[0].x} ${pts[0].y}`;
  for (let i = 1; i < N; i++) d += ` L ${pts[i].x} ${pts[i].y}`;
  $("ob3Path").setAttribute("d", d);
  setCircle($("ob3Start"), pts[0]);
  setCircle($("ob3End"), pts[N - 1]);
  setLabel($("ob3EndLabel"), pts[N - 1], -44);    // FINISH caption above the end dot

  // Park the START button on the head of the line (viewport coords: svg is offset).
  const b = $("traceStartBtn");
  b.style.left = (r.left + pts[0].x) + "px";
  b.style.top = (r.top + pts[0].y) + "px";
}

function setCircle(el, p) { el.setAttribute("cx", p.x); el.setAttribute("cy", p.y); }
function setLabel(el, p, dy) { el.setAttribute("x", p.x); el.setAttribute("y", p.y + dy); }

document.addEventListener("mousemove", (e) => {
  if (!ob3.active) return;
  const r = ob3.svgRect;
  const p = { x: e.clientX - r.left, y: e.clientY - r.top };
  setCircle($("ob3Cursor"), p);

  // Record where the hand actually went and redraw the breadcrumb trail.
  ob3.trail.push(p);
  $("ob3Trail").setAttribute("points", ob3.trail.map((q) => `${q.x.toFixed(1)},${q.y.toFixed(1)}`).join(" "));

  const dev = distToPolyline(p, ob3.pts);
  ob3.devs.push(dev);
  let reachedCount = 0;
  for (let i = 0; i < ob3.pts.length; i++) {
    if (!ob3.reached[i] && dist(p, ob3.pts[i]) < REACH_TOL) ob3.reached[i] = true;
    if (ob3.reached[i]) reachedCount++;
  }
  const coverage = reachedCount / ob3.pts.length;
  const acc = accuracyScore(ob3.devs);
  // Tell the tester when the finish gate is open, so hovering the dot is never a
  // silent no-op: below COVER_MIN the run *won't* finish no matter where you hover.
  $("ob3Meta").textContent = coverage >= COVER_MIN
    ? `accuracy ${acc.toFixed(0)}  ·  reach the green FINISH dot ✓`
    : `accuracy ${acc.toFixed(0)}  ·  ${(coverage * 100).toFixed(0)}% traced (need ${(COVER_MIN * 100).toFixed(0)}%)`;

  const end = ob3.pts[ob3.pts.length - 1];
  if (coverage >= COVER_MIN && dist(p, end) < END_TOL) finishTrace(coverage, acc);
});

function finishTrace(coverage, acc) {
  ob3.active = false;
  const meanDev = ob3.devs.reduce((a, b) => a + b, 0) / Math.max(ob3.devs.length, 1);
  results.ob3 = {
    timeMs: Math.round(now() - ob3.startAt),
    accuracy: +acc.toFixed(1),
    meanDevPx: +meanDev.toFixed(1),
    coverage: +coverage.toFixed(3),
  };
  showFinish();
}

function accuracyScore(devs) {
  if (!devs.length) return 0;
  let s = 0;
  for (const d of devs) s += Math.max(0, 1 - d / DEV_MAX);
  return 100 * s / devs.length;
}

const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

function distToPolyline(p, pts) {
  let best = Infinity;
  for (let i = 1; i < pts.length; i++) best = Math.min(best, distToSeg(p, pts[i - 1], pts[i]));
  return best;
}
function distToSeg(p, a, b) {
  const dx = b.x - a.x, dy = b.y - a.y;
  const len2 = dx * dx + dy * dy;
  let t = len2 ? ((p.x - a.x) * dx + (p.y - a.y) * dy) / len2 : 0;
  t = Math.max(0, Math.min(1, t));
  return Math.hypot(p.x - (a.x + t * dx), p.y - (a.y + t * dy));
}

/* ================= FINISH GATE ================= */
function showFinish() {
  const tile = (val, label) => `<div class="tile"><b>${val}</b><span>${label}</span></div>`;
  $("summary").innerHTML =
    tile((results.ob1.timeMs / 1000).toFixed(2) + "s", "Precision") +
    tile((results.ob2.timeMs / 1000).toFixed(2) + "s", "Rapid clicks") +
    tile(results.ob3.accuracy.toFixed(0), "Trace accuracy");
  showScreen("finish");
}

$("finishBtn").addEventListener("click", async () => {
  $("finishBtn").disabled = true;
  try { await fetch("/api/finish", { method: "POST" }); } catch (_) {}
  document.body.classList.add("show-cursor"); // tracker off -> normal mouse
  buildSurvey();
  showScreen("survey");
});

/* ================= SURVEY ================= */
const QUESTIONS = [
  "How well were you able to click?",
  "How well was your hand tracked?",
  "How easy was it to use overall?",
  "How easy were the controls to understand?",
  "How well did it do what you wanted?",
];

function buildSurvey() {
  const form = $("surveyForm");
  form.innerHTML = "";
  QUESTIONS.forEach((text, qi) => {
    const q = document.createElement("div");
    q.className = "q";
    q.innerHTML = `<div class="q-text">${qi + 1}. ${text}</div>`;
    const scale = document.createElement("div");
    scale.className = "scale";
    for (let v = 1; v <= 5; v++) {
      const b = document.createElement("button");
      b.type = "button"; b.textContent = v;
      b.addEventListener("click", () => {
        results.survey["q" + (qi + 1)] = v;
        [...scale.children].forEach((c) => c.className = "");
        b.className = "sel-" + v;
        if (Object.keys(results.survey).length === QUESTIONS.length) $("submitBtn").disabled = false;
      });
      scale.appendChild(b);
    }
    q.appendChild(scale);
    form.appendChild(q);
  });
  $("submitBtn").disabled = true;
}

$("submitBtn").addEventListener("click", async () => {
  $("submitBtn").disabled = true;
  results.finishedAt = new Date().toISOString();
  showScreen("done");
  $("doneMsg").textContent = "Saving your results…";
  try {
    const res = await fetch("/api/results", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(results),
    });
    const j = await res.json();
    $("doneMsg").textContent = j.ok
      ? `Results saved to ${j.filename}. You can close this window.`
      : `Could not save results: ${j.error || "unknown error"}`;
  } catch (e) {
    $("doneMsg").textContent = "Could not reach the server to save results: " + e;
  }
  $("restartBtn").removeAttribute("hidden");
});

$("restartBtn").addEventListener("click", () => location.reload());

/* ================= TRACKER STATUS CHIP ================= */
async function pollStatus() {
  try {
    const s = await (await fetch("/api/status")).json();
    updateWelcome(s);
    const chip = $("statusChip"), dot = $("statusDot"), txt = $("statusText");
    if (!s.tracker) { chip.setAttribute("hidden", ""); return; }
    chip.removeAttribute("hidden");
    if (s.error) { dot.className = "dot err"; txt.textContent = "tracker: " + s.error; }
    else if (!s.input_enabled) { dot.className = "dot warn"; txt.textContent = "tracker stopped"; }
    else {
      dot.className = s.hand_seen ? "dot ok" : "dot warn";
      const lbl = s.label ? `${s.label} ${(s.confidence * 100 | 0)}%` : "no hand";
      txt.textContent = `${lbl} · ${s.fps ? s.fps.toFixed(0) : 0} fps · ${s.clicks} clicks`;
    }
  } catch (_) { /* server not up yet */ }
}
setInterval(pollStatus, 700);
pollStatus();
loadModels().then(loadStrategies);
