"use strict";

// --- small helpers ---------------------------------------------------------
const $ = (id) => document.getElementById(id);
const num = (id) => parseFloat($(id).value);
const val = (id) => $(id).value.trim();

function setStatus(text, cls) {
  const el = $("status");
  el.textContent = text;
  el.className = "chip" + (cls ? " " + cls : "");
}

let mode = "real";

// Config blocks shared by both real and fictional runs.
function commonBlocks() {
  const overrides = {};
  val("mat_overrides").split(",").forEach((pair) => {
    const [k, v] = pair.split("=").map((s) => s && s.trim());
    if (k && v) overrides[k] = v;
  });
  const peaks = { min_prominence_m: num("peak_prom"), label_elevation: true };
  if (val("peak_max") !== "") peaks.max_count = parseInt(val("peak_max"), 10);
  return {
    machine: {
      profile: "glowforge",
      bed_mm: [num("bed_w"), num("bed_h")],
      margin_mm: num("margin_mm"),
      kerf_mm: num("kerf_mm"),
    },
    materials: {
      default: val("mat_default"),
      water: val("mat_water"),
      ...(Object.keys(overrides).length ? { overrides } : {}),
    },
    symbology: {
      peaks,
      rivers: { min_stream_order: parseInt(val("river_order"), 10) },
      lakes: { min_area_km2: num("lake_area"), mode: val("lake_mode") },
      labels: { cap_height_mm: num("cap_height") },
    },
    panelization: { seam_margin_mm: num("seam_margin"), seam_joint: val("seam_joint") },
  };
}

// Real-world config (bbox + a scale driver).
function readConfig() {
  const physical = {
    model_width_mm: num("model_width_mm"),
    ply_thickness_mm: num("ply_thickness_mm"),
  };
  physical[val("driver")] = num("driver_value");
  return {
    region: { bbox: [num("west"), num("south"), num("east"), num("north")], dem: val("dem") },
    physical,
    ...commonBlocks(),
  };
}

// Fictional config — no region (the adapter supplies the bundle); normalize_layers.
function fictionalConfig() {
  const physical = {
    model_width_mm: num("model_width_mm"),
    ply_thickness_mm: num("ply_thickness_mm"),
    normalize_layers: parseInt(val("fnorm"), 10),
  };
  const clip = val("fclip").split(",").map((s) => parseFloat(s.trim())).filter((n) => !isNaN(n));
  if (clip.length === 2) physical.clip_percentiles = clip;
  return { physical, ...commonBlocks() };
}

function readOptions() {
  return {
    dem_resolution_m: num("dem_res"),
    fetch_symbology: $("fetch_symbology").checked,
    panelize: $("opt_panelize").checked,
    nest: $("opt_nest").checked,
    use_cache: $("opt_cache").checked,
    project_name: val("project_name") || "map",
  };
}

// Serialize the config object to a minimal TOML string for the advanced panel.
function toToml(cfg) {
  const lines = [];
  const table = (name, obj) => {
    lines.push(`[${name}]`);
    for (const [k, v] of Object.entries(obj)) {
      if (v && typeof v === "object" && !Array.isArray(v)) continue;
      lines.push(`${k} = ${JSON.stringify(v)}`);
    }
    lines.push("");
  };
  table("region", cfg.region);
  table("physical", cfg.physical);
  table("machine", cfg.machine);
  table("materials", cfg.materials);
  lines.push("[symbology]");
  for (const [k, v] of Object.entries(cfg.symbology))
    lines.push(`${k} = ${JSON.stringify(v)}`);
  lines.push("");
  table("panelization", cfg.panelization);
  return lines.join("\n");
}

function requestBody() {
  const body = { options: readOptions() };
  if ($("use_toml").checked && val("toml")) body.toml = val("toml");
  else body.config = readConfig();
  const overrides = trailOverrides();
  if (overrides) body.feature_overrides = overrides;   // only when trails were loaded
  return body;
}

// --- individual trail selection -------------------------------------------
let loadedTrails = [];

async function loadTrails() {
  const btn = $("load-trails");
  btn.disabled = true;
  $("trails-status").textContent = "loading…";
  try {
    const r = await fetch("/api/features", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ config: readConfig() }),
    });
    const d = await r.json();
    if (d.error) {
      loadedTrails = [];
      renderTrails();
      $("trails-status").textContent = "couldn't load trails: " + d.error;
    } else {
      loadedTrails = d.features || [];
      renderTrails();
    }
  } catch (e) {
    loadedTrails = [];
    renderTrails();
    $("trails-status").textContent = "couldn't reach server";
  }
  btn.disabled = false;
}

function renderTrails() {
  const list = $("trails-list");
  if (!loadedTrails.length) {
    list.innerHTML = '<p class="muted" style="font-size:12px">no trails loaded</p>';
    $("trails-actions").classList.add("hidden");
    updateTrailCount();
    return;
  }
  list.innerHTML = loadedTrails.map((t, i) => {
    const badge = t.network ? `<span class="net-badge">${esc(t.network)}</span>` : "";
    const segs = t.count > 1 ? ` · ${t.count} segs` : "";
    const name = t.name
      ? esc(t.name)
      : `<i>unnamed paths (${t.count})</i>`;
    return `<label class="trail-row"><input type="checkbox" data-group="${i}" ${t.include ? "checked" : ""}>` +
      `<span class="trail-name">${name}${segs}</span>${badge}` +
      `<span class="muted trail-len">${t.length_km} km</span></label>`;
  }).join("");
  $("trails-actions").classList.remove("hidden");
  list.querySelectorAll("input[type=checkbox]").forEach((cb) =>
    cb.addEventListener("change", updateTrailCount));
  updateTrailCount();
}

function updateTrailCount() {
  const boxes = $("trails-list").querySelectorAll("input[type=checkbox]");
  if (!boxes.length) { $("trails-status").textContent = "no trails in this area"; return; }
  const sel = [...boxes].filter((b) => b.checked).length;
  $("trails-status").textContent = `${boxes.length} trails, ${sel} selected`;
}

// {osm_id: checked} for every OSM segment of every loaded trail group.
function trailOverrides() {
  const boxes = $("trails-list").querySelectorAll("input[type=checkbox]");
  if (!boxes.length) return null;
  const ov = {};
  boxes.forEach((b) => {
    const group = loadedTrails[parseInt(b.dataset.group, 10)];
    if (group) group.osm_ids.forEach((id) => { ov[id] = b.checked; });
  });
  return ov;
}

function toggleAllTrails(checked) {
  $("trails-list").querySelectorAll("input[type=checkbox]").forEach((b) => { b.checked = checked; });
  updateTrailCount();
}

async function uploadFile(inputId) {
  const f = $(inputId).files[0];
  if (!f) return null;
  const fd = new FormData();
  fd.append("file", f);
  const r = await fetch("/api/upload", { method: "POST", body: fd });
  if (!r.ok) throw new Error("upload failed");
  return (await r.json()).path;
}

// Build the {adapt, config, options} body for a fictional run (uploads first).
async function fictionalBody() {
  const tier = val("ftier");
  const extra = { cells_across: num("fcells"), normalize_layers: parseInt(val("fnorm"), 10) };
  let source_path;
  if (tier === "bundle") {
    source_path = val("fpath");
    if (!source_path) throw new Error("enter the bundle path");
  } else {
    appendLog("uploading source …");
    source_path = await uploadFile("ffile");
    if (!source_path) throw new Error("choose a source file");
    if (tier === "art") {
      extra.segmentation = val("fseg");
      const ov = await uploadFile("foverlay");
      if (ov) extra.class_overlay = ov;
    }
  }
  return { adapt: { tier, source_path, extra }, config: fictionalConfig(), options: readOptions() };
}

// --- Leaflet map with a draggable bbox rectangle --------------------------
let map, rect, drawing = false;

function initMap() {
  map = L.map("map").setView([36.575, -118.295], 11);
  L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 17, attribution: "© OpenStreetMap",
  }).addTo(map);
  drawRect(true);

  const btn = L.control({ position: "topright" });
  btn.onAdd = () => {
    const d = L.DomUtil.create("button", "");
    d.textContent = "▭ draw box";
    d.style.cssText = "padding:5px 9px;border-radius:6px;border:1px solid #ccc;background:#fff;cursor:pointer";
    L.DomEvent.on(d, "click", (e) => { L.DomEvent.stop(e); startDraw(); });
    return d;
  };
  btn.addTo(map);

  let origin = null;
  map.on("mousedown", (e) => {
    if (!drawing) return;
    origin = e.latlng; map.dragging.disable();
  });
  map.on("mousemove", (e) => {
    if (!drawing || !origin) return;
    setBboxFromLatLngs(origin, e.latlng);
  });
  map.on("mouseup", (e) => {
    if (!drawing || !origin) return;
    setBboxFromLatLngs(origin, e.latlng);
    origin = null; drawing = false; map.dragging.enable();
    $("map").style.cursor = "";
    scheduleScale();
  });
}

function startDraw() {
  drawing = true;
  $("map").style.cursor = "crosshair";
}

function setBboxFromLatLngs(a, b) {
  $("west").value = Math.min(a.lng, b.lng).toFixed(4);
  $("east").value = Math.max(a.lng, b.lng).toFixed(4);
  $("south").value = Math.min(a.lat, b.lat).toFixed(4);
  $("north").value = Math.max(a.lat, b.lat).toFixed(4);
  drawRect(false);
}

function drawRect(fit) {
  const b = [[num("south"), num("west")], [num("north"), num("east")]];
  if (rect) map.removeLayer(rect);
  rect = L.rectangle(b, { color: "#2d6cdf", weight: 2, fillOpacity: 0.08 }).addTo(map);
  if (fit) map.fitBounds(b, { padding: [20, 20] });
}

// --- live scale estimate ---------------------------------------------------
let scaleTimer = null;
function scheduleScale() { clearTimeout(scaleTimer); scaleTimer = setTimeout(liveScale, 350); }

async function liveScale() {
  let cfg;
  try { cfg = readConfig(); } catch { return; }
  try {
    const r = await fetch("/api/scale", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ config: cfg }),
    });
    const d = await r.json();
    if (!r.ok) { $("scale").innerHTML = `<div class="err">${d.detail || "invalid config"}</div>`; return; }
    const kv = (label, value) => `<div class="kv"><b>${value}</b><span>${label}</span></div>`;
    $("scale").innerHTML =
      kv("scale", d.scale_label) +
      kv("interval", d.interval_m == null ? "from terrain" : d.interval_m + " m") +
      kv("exaggeration", d.exaggeration == null ? "from terrain" : d.exaggeration + "×") +
      kv("layers", d.layer_count ?? "from terrain") +
      kv("real extent", d.extent_km.join(" × ") + " km") +
      kv("model size", d.model_size_mm.join(" × ") + " mm");
    if (d.warnings && d.warnings.length)
      $("scale").innerHTML += `<div class="warnbox" style="grid-column:1/-1">${d.warnings.map(esc).join("<br>")}</div>`;
  } catch (e) { /* transient */ }
}

// --- run + progress streaming ---------------------------------------------
async function run() {
  $("run").disabled = true;
  setStatus("starting…", "running");
  $("progresscard").classList.remove("hidden");
  $("resultscard").classList.add("hidden");
  $("log").textContent = "";

  let body;
  try {
    body = mode === "fictional" ? await fictionalBody() : requestBody();
  } catch (e) { return fail(e.message || "invalid input"); }

  let res;
  try {
    res = await fetch("/api/run", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch (e) { return fail("could not reach server"); }
  const data = await res.json();
  if (!res.ok) return fail(data.detail || "run failed to start");

  setStatus("running…", "running");
  const es = new EventSource(`/api/jobs/${data.job_id}/events`);
  es.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === "log") appendLog(msg.line);
    else if (msg.type === "done") { es.close(); finish(msg.result); }
    else if (msg.type === "error") { es.close(); fail(msg.error); }
  };
  es.onerror = () => { es.close(); $("run").disabled = false; };
}

function appendLog(line) {
  const el = $("log");
  el.textContent += line + "\n";
  el.scrollTop = el.scrollHeight;
}

function fail(err) {
  setStatus("error", "error");
  appendLog("✖ " + err);
  $("run").disabled = false;
}

function finish(result) {
  setStatus("done", "done");
  $("run").disabled = false;
  renderResults(result);
}

// --- results ---------------------------------------------------------------
let lastResult = null;

function renderResults(r) {
  lastResult = r;
  $("resultscard").classList.remove("hidden");
  const s = r.scale;
  const kv = (label, v) => `<div class="kv"><b>${v}</b><span>${label}</span></div>`;
  $("summary").innerHTML =
    kv("scale", s.scale_label ?? s.label) +
    kv("layers", s.layer_count) +
    kv("interval", s.interval_m + " m") +
    kv("exaggeration", s.exaggeration + "×") +
    kv("parts", r.total_parts) +
    kv("boards", r.total_boards) +
    kv("model", s.model_width_mm + "×" + s.model_height_mm + " mm");

  const dl = [];
  if (r.guide_url) dl.push(`<a href="${r.guide_url}" target="_blank">📄 assembly guide</a>`);
  if (r.zip_url) dl.push(`<a href="${r.zip_url}" download>⬇ download all (zip)</a>`);
  if (r.features_csv_url) dl.push(`<a href="${r.features_csv_url}" download>features.csv</a>`);
  $("warnings").innerHTML =
    (dl.length ? `<div class="downloads">${dl.join("")}</div>` : "") +
    (r.warnings && r.warnings.length
      ? `<details class="warnbox"><summary>${r.warnings.length} warning(s)</summary><ul>${
          r.warnings.map((w) => `<li>${esc(w)}</li>`).join("")}</ul></details>` : "");

  const thumb = (row, label, url, sub) =>
    `<button class="thumb" data-url="${url}"><b>${label}</b>${sub ? "<br>" + sub : ""}</button>`;

  // Boards tab: composite preview + nested boards.
  $("row-preview").innerHTML = r.preview_url
    ? thumb("preview", "composite preview", r.preview_url, "all layers") : "";
  $("row-boards").innerHTML = r.boards.map((b) =>
    thumb("b", b.material + " #" + b.index, b.url,
      b.parts + "p · " + b.utilization + "% · " + b.cut_m + " m")).join("");
  document.querySelectorAll("#tab-boards .thumb").forEach((t) =>
    t.addEventListener("click", () => showSvg(t)));
  $("stage").innerHTML = '<p class="muted">Select a board to preview.</p>';
  renderLegend(null);

  // Downloads tab.
  const link = (url, label) => `<a href="${url}" download>${esc(label)}</a>`;
  $("dl-main").innerHTML = dl.join("");
  $("dl-layers").innerHTML = r.layers.map((l) =>
    link(l.url, `layer ${l.index} (≥ ${l.threshold_m} m)`)).join("");
  $("dl-boards").innerHTML = r.boards.map((b) =>
    link(b.url, `${b.material} #${b.index}`)).join("");

  // Stack tab (the main view).
  showTab("stack");
  if (r.stack_url) loadStack(r.stack_url, r);
  else $("stk-stage").innerHTML = '<p class="muted">This run has no stack data.</p>';
}

function showTab(name) {
  document.querySelectorAll(".rtab").forEach((t) =>
    t.classList.toggle("active", t.dataset.tab === name));
  document.querySelectorAll(".rpane").forEach((p) =>
    p.classList.toggle("hidden", p.id !== "tab-" + name));
  if (name === "boards" && !$("stage").querySelector("svg")) {
    const first = document.querySelector("#tab-boards .thumb");
    if (first) showSvg(first);
  }
}

async function showSvg(btn) {
  document.querySelectorAll("#tab-boards .thumb").forEach((t) => t.classList.remove("active"));
  btn.classList.add("active");
  await showSvgInto($("stage"), btn.dataset.url, $("legend"));
}

// Load an SVG file into a stage element (with a download link) + its legend.
async function showSvgInto(stage, url, legendEl) {
  stage.innerHTML = '<p class="muted">loading…</p>';
  try {
    const txt = await (await fetch(url)).text();
    const name = url.split("/").pop();
    stage.innerHTML = txt +
      `<div class="downloads" style="position:absolute;right:14px;bottom:8px">` +
      `<a href="${url}" download="${name}">⬇ download ${name}</a></div>`;
    stage.style.position = "relative";
    renderLegend(stage.querySelector("svg"), legendEl);
  } catch (e) {
    stage.innerHTML = '<p class="err">could not load SVG</p>';
    renderLegend(null, legendEl);
  }
}

// --- stack viewer ------------------------------------------------------------
// Builds the map up layer by layer from stack.json: every layer is a filled
// group (elevation-graded, drop-shadowed) drawn in order, with its own
// symbology inside the group so hiding a layer hides its rivers/names too.
const STK = {
  data: null, result: null, height: 0, n: 0,
  svg: null, view: null, groups: [], ghost: null, hi: null,
  zoom: { s: 1, tx: 0, ty: 0 }, playTimer: null, isolated: null,
  toggles: { water: true, trails: true, names: true, seams: false, ghost: true, shadows: true },
};
const STK_KEY = "toposlicr.stack.toggles";
const SHADOW_VERTEX_BUDGET = 400000;   // above this the drop-shadow filter is too slow
// op → toggle group + which color key paints it
const STK_OPS = {
  score_hydro: ["water", "score_hydro", "line"],
  score_trail: ["trails", "score_trail", "line"],
  leader: ["names", "score_ids", "line"],
  engrave_fill: ["names", "engrave_fill", "fill"],
  icon: ["names", "engrave_fill", "fill"],
  seam: ["seams", "seam", "line"],
};

function stkShade(k, n) {
  // Same grading as preview.svg: light (base) → dark (summit).
  const v = Math.round(220 - 160 * (n > 1 ? k / (n - 1) : 0));
  return `rgb(${v},${v},${v})`;
}

function loadToggles() {
  try {
    const saved = JSON.parse(localStorage.getItem(STK_KEY) || "null");
    if (saved && typeof saved === "object") Object.assign(STK.toggles, saved);
  } catch (e) { /* private mode etc. */ }
  document.querySelectorAll("#tab-stack [data-toggle]").forEach((cb) => {
    cb.checked = !!STK.toggles[cb.dataset.toggle];
  });
}

function saveToggles() {
  try { localStorage.setItem(STK_KEY, JSON.stringify(STK.toggles)); } catch (e) { /* ignore */ }
}

async function loadStack(url, result) {
  const stage = $("stk-stage");
  stage.innerHTML = '<p class="muted">building stack…</p>';
  stopPlay();
  STK.isolated = null;
  try {
    STK.data = await (await fetch(url)).json();
  } catch (e) {
    stage.innerHTML = '<p class="err">could not load stack data</p>';
    return;
  }
  STK.result = result;
  STK.n = STK.data.layers.length;
  STK.height = STK.n;
  $("stk-slider").max = STK.n;
  $("stk-slider").value = STK.n;
  buildStackSvg();
  buildRail();
  applyHeight();
  resetZoom();
}

function buildStackSvg() {
  const d = STK.data, m = d.model, n = STK.n;
  const totalVerts = d.layers.reduce((a, l) => a + l.vertices, 0);
  STK.shadowsAllowed = totalVerts <= SHADOW_VERTEX_BUDGET;
  const ns = "http://www.w3.org/2000/svg";
  const el = (tag, attrs) => {
    const e = document.createElementNS(ns, tag);
    for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, v);
    return e;
  };
  const svg = el("svg", { viewBox: `0 0 ${m.width_mm} ${m.height_mm}`,
    preserveAspectRatio: "xMidYMid meet", class: "stacksvg" });
  const defs = el("defs");
  const filt = el("filter", { id: "stk-shadow", x: "-5%", y: "-5%", width: "110%", height: "110%" });
  // Offsets in mm: a 3 mm ply lit from the top-left casts a short soft shadow.
  filt.appendChild(el("feDropShadow", { dx: "0.5", dy: "0.7", stdDeviation: "0.6",
    "flood-color": "#000", "flood-opacity": "0.45" }));
  defs.appendChild(filt);
  svg.appendChild(defs);
  const view = el("g", { id: "stk-view" });
  view.appendChild(el("rect", { class: "frame", x: 0, y: 0, width: m.width_mm, height: m.height_mm }));
  STK.groups = [];
  d.layers.forEach((lyr, k) => {
    const g = el("g", { class: "stk-layer", "data-k": k });
    const cut = el("path", { class: "cut", d: lyr.cut, fill: stkShade(k, n),
      "fill-rule": "evenodd" });
    g.appendChild(cut);
    for (const [op, dd] of Object.entries(lyr.ops || {})) {
      const spec = STK_OPS[op];
      if (!spec || !dd) continue;
      const color = safeColor(d.colors[spec[1]] || "#000");
      const attrs = { class: `op op-${op} tg-${spec[0]}`, d: dd };
      if (spec[2] === "fill") { attrs.fill = color; attrs.stroke = "none"; }
      else { attrs.fill = "none"; attrs.stroke = color; }
      if (op === "seam") attrs["stroke-dasharray"] = "2 1.5";
      g.appendChild(el("path", attrs));
    }
    view.appendChild(g);
    STK.groups.push(g);
  });
  STK.ghost = el("path", { class: "ghost", d: "" });
  STK.hi = el("path", { class: "hi", d: "", "fill-rule": "evenodd" });
  view.appendChild(STK.ghost);
  view.appendChild(STK.hi);
  svg.appendChild(view);

  const stage = $("stk-stage");
  stage.classList.remove("isolate");
  stage.innerHTML = "";
  stage.appendChild(svg);
  const hint = document.createElement("div");
  hint.className = "hint";
  hint.textContent = `${m.width_mm} × ${m.height_mm} mm · ${m.scale_label} · scroll to zoom, drag to pan, ↑↓ to step`;
  stage.appendChild(hint);
  STK.svg = svg; STK.view = view;

  svg.addEventListener("mouseover", (e) => {
    const g = e.target.closest && e.target.closest(".stk-layer");
    setHover(g ? parseInt(g.dataset.k, 10) : null);
  });
  svg.addEventListener("mouseleave", () => setHover(null));
  applyToggles();
}

function buildRail() {
  const d = STK.data, n = STK.n;
  const rail = $("stk-rail");
  rail.innerHTML = d.layers.map((l, k) => {
    const warn = l.warnings.length
      ? `<span class="warn" title="${esc(l.warnings.join("\n"))}">⚠</span>` : "";
    const panels = l.panels > 1 ? `<span class="chip panels">${l.panels} panels</span>` : "";
    return `<button class="lrow" data-k="${k}" title="click to view this layer's cut file">` +
      `<span class="sw" style="background:${stkShade(k, n)}"></span>` +
      `<span><b>L${k}</b> <span class="elev">≥ ${l.threshold_m} m</span></span>${warn || "<span></span>"}` +
      `<span class="meta"><span class="chip" title="${esc(l.material)}">${esc(l.material)}</span>${panels}</span>` +
      `</button>`;
  }).join("");
  rail.querySelectorAll(".lrow").forEach((row) => {
    const k = parseInt(row.dataset.k, 10);
    row.addEventListener("mouseenter", () => setHover(k, true));
    row.addEventListener("mouseleave", () => setHover(null, true));
    row.addEventListener("click", () => isolateLayer(k));
  });
}

function setHeight(h, fromSlider) {
  STK.height = Math.max(0, Math.min(STK.n, h));
  if (!fromSlider) $("stk-slider").value = STK.height;
  applyHeight();
}

function applyHeight() {
  if (!STK.data) return;
  const h = STK.height, d = STK.data;
  STK.groups.forEach((g, k) => { g.style.display = k < h ? "" : "none"; });
  $("stk-rail").querySelectorAll(".lrow").forEach((row) => {
    row.classList.toggle("off", parseInt(row.dataset.k, 10) >= h);
  });
  // Ghost: the next layer's outline on top of the current stack.
  const showGhost = STK.toggles.ghost && h < STK.n && h > 0;
  STK.ghost.setAttribute("d", showGhost ? d.layers[h].cut : "");
  const top = h > 0 ? d.layers[h - 1] : null;
  $("stk-readout").innerHTML = top
    ? `<b>${h} / ${STK.n}</b> layers · top L${h - 1} ≥ ${top.threshold_m} m` +
      (h < STK.n ? ` · next L${h} ≥ ${d.layers[h].threshold_m} m` : " · complete")
    : `<b>0 / ${STK.n}</b> layers · empty frame`;
  if (STK.isolated !== null) showStackStage();
}

function setHover(k, fromRail) {
  if (!STK.data) return;
  const d = k !== null && k < STK.height ? STK.data.layers[k].cut : "";
  STK.hi.setAttribute("d", fromRail ? (k !== null ? STK.data.layers[k].cut : "") : d);
  $("stk-rail").querySelectorAll(".lrow").forEach((row) =>
    row.classList.toggle("hover", parseInt(row.dataset.k, 10) === k));
}

function applyToggles() {
  if (!STK.svg) return;
  for (const t of ["water", "trails", "names", "seams"]) {
    STK.svg.querySelectorAll(`.tg-${t}`).forEach((p) => {
      p.style.display = STK.toggles[t] ? "" : "none";
    });
  }
  const shadow = STK.toggles.shadows && STK.shadowsAllowed;
  STK.svg.querySelectorAll("path.cut").forEach((p) => {
    if (shadow) p.setAttribute("filter", "url(#stk-shadow)");
    else p.removeAttribute("filter");
  });
  const cb = document.querySelector('#tab-stack [data-toggle="shadows"]');
  if (cb) cb.title = STK.shadowsAllowed ? "" : "disabled: too many vertices for the shadow filter";
  applyHeight();
  renderStackLegend();
}

function renderStackLegend() {
  const el = $("stk-legend");
  if (!STK.data) { el.classList.add("hidden"); return; }
  const d = STK.data, m = d.model, n = STK.n;
  const stops = d.layers.map((_, k) =>
    `${stkShade(k, n)} ${(100 * k / n).toFixed(1)}% ${(100 * (k + 1) / n).toFixed(1)}%`);
  const items = [legendItem(
    `<span class="swatch bands" style="background:linear-gradient(90deg,${stops.join(",")})"></span>`,
    "Layers", `${n} × ${m.ply_thickness_mm} mm ply; light = base (${m.base_elev_m} m), dark = summit; one band = ${m.interval_m} m`)];
  const present = new Set();
  d.layers.forEach((l) => Object.keys(l.ops || {}).forEach((op) => present.add(op)));
  const rows = [
    ["score_hydro", "Water", "rivers and lake outlines"],
    ["score_trail", "Trail", "hiking paths and routes"],
    ["engrave_fill", "Names", "engraved place names, elevations, trail names"],
    ["icon", "Icon", "engraved feature icon"],
    ["leader", "Leader", "links a name to its feature"],
    ["seam", "Panel seam", "where an oversized layer is split (not cut)"],
  ];
  for (const [op, label, desc] of rows) {
    const spec = STK_OPS[op];
    if (!present.has(op) || !STK.toggles[spec[0]]) continue;
    const kind = spec[2] === "fill" ? "fill" : (op === "score_trail" || op === "seam" ? "dash" : "line");
    items.push(legendItem(swatch(kind, d.colors[spec[1]]), label, desc));
  }
  if (STK.toggles.ghost && STK.height < n)
    items.push(legendItem(swatch("dash", "#2d6cdf"), "Next layer", "where the next layer sits on the stack"));
  el.innerHTML = `<span class="legend-title">Legend · elevation stack</span>${items.join("")}`;
  el.classList.remove("hidden");
}

// -- isolate: one layer's real cut file on the stage, back button to the stack
async function isolateLayer(k) {
  const r = STK.result;
  if (!r || !r.layers[k]) return;
  stopPlay();
  STK.isolated = k;
  const l = STK.data.layers[k];
  const stage = $("stk-stage");
  stage.classList.add("isolate");
  stage.innerHTML = "";
  const bar = document.createElement("div");
  bar.className = "isolate-bar";
  bar.innerHTML = `<button class="small" id="stk-back">← back to stack</button>` +
    `<b>Layer ${k}</b><span class="muted">≥ ${l.threshold_m} m</span>` +
    `<span class="chip">${esc(l.material)}</span>` +
    (l.panels > 1 ? `<span class="muted">${l.panels} panels</span>` : "") +
    `<span class="muted">${(l.area_mm2 / 100).toFixed(0)} cm² · ${l.vertices} pts</span>`;
  const holder = document.createElement("div");
  holder.className = "stage isolate";
  holder.style.border = "0";
  stage.appendChild(bar);
  stage.appendChild(holder);
  bar.querySelector("#stk-back").addEventListener("click", showStackStage);
  $("stk-rail").querySelectorAll(".lrow").forEach((row) =>
    row.classList.toggle("active", parseInt(row.dataset.k, 10) === k));
  await showSvgInto(holder, r.layers[k].url, $("stk-legend"));
}

function showStackStage() {
  if (STK.isolated === null || !STK.data) return;
  STK.isolated = null;
  $("stk-rail").querySelectorAll(".lrow").forEach((row) => row.classList.remove("active"));
  buildStackSvg();
  applyZoom();
}

// -- play: add one layer every 400 ms
function togglePlay() {
  if (STK.playTimer) { stopPlay(); return; }
  if (!STK.data) return;
  if (STK.isolated !== null) showStackStage();
  if (STK.height >= STK.n) setHeight(0);
  $("stk-play").textContent = "❚❚";
  $("stk-play").classList.add("on");
  STK.playTimer = setInterval(() => {
    if (STK.height >= STK.n) { stopPlay(); return; }
    setHeight(STK.height + 1);
  }, 400);
}

function stopPlay() {
  if (STK.playTimer) clearInterval(STK.playTimer);
  STK.playTimer = null;
  $("stk-play").textContent = "▶";
  $("stk-play").classList.remove("on");
}

// -- zoom / pan: a transform on the view group, in SVG user units (mm)
function applyZoom() {
  if (!STK.view) return;
  const z = STK.zoom;
  STK.view.setAttribute("transform", `translate(${z.tx} ${z.ty}) scale(${z.s})`);
}

function resetZoom() { STK.zoom = { s: 1, tx: 0, ty: 0 }; applyZoom(); }

function svgPoint(evt) {
  const pt = STK.svg.createSVGPoint();
  pt.x = evt.clientX; pt.y = evt.clientY;
  return pt.matrixTransform(STK.svg.getScreenCTM().inverse());
}

function wireStackStage() {
  const stage = $("stk-stage");
  stage.addEventListener("wheel", (e) => {
    if (!STK.svg || STK.isolated !== null) return;
    e.preventDefault();
    const p = svgPoint(e);
    const z = STK.zoom;
    const f = Math.exp(-e.deltaY * 0.0015);
    const s = Math.max(1, Math.min(40, z.s * f));
    // Keep the point under the cursor fixed.
    z.tx = p.x - (p.x - z.tx) * (s / z.s);
    z.ty = p.y - (p.y - z.ty) * (s / z.s);
    z.s = s;
    if (s === 1) { z.tx = 0; z.ty = 0; }
    applyZoom();
  }, { passive: false });
  let drag = null;
  stage.addEventListener("mousedown", (e) => {
    if (!STK.svg || STK.isolated !== null || e.button !== 0) return;
    drag = { p: svgPoint(e), tx: STK.zoom.tx, ty: STK.zoom.ty, moved: false };
    stage.classList.add("dragging");
  });
  window.addEventListener("mousemove", (e) => {
    if (!drag) return;
    const p = svgPoint(e);
    STK.zoom.tx = drag.tx + (p.x - drag.p.x);
    STK.zoom.ty = drag.ty + (p.y - drag.p.y);
    drag.moved = true;
    applyZoom();
  });
  window.addEventListener("mouseup", () => { drag = null; stage.classList.remove("dragging"); });
  stage.addEventListener("dblclick", () => { if (STK.isolated === null) resetZoom(); });
  stage.addEventListener("keydown", (e) => {
    if (!STK.data) return;
    if (e.key === "ArrowUp" || e.key === "ArrowRight") { e.preventDefault(); stopPlay(); setHeight(STK.height + 1); }
    else if (e.key === "ArrowDown" || e.key === "ArrowLeft") { e.preventDefault(); stopPlay(); setHeight(STK.height - 1); }
    else if (e.key === " ") { e.preventDefault(); togglePlay(); }
    else if (e.key === "Escape" && STK.isolated !== null) showStackStage();
  });

  $("stk-slider").addEventListener("input", () => {
    stopPlay(); setHeight(parseInt($("stk-slider").value, 10), true);
  });
  $("stk-up").addEventListener("click", () => { stopPlay(); setHeight(STK.height + 1); });
  $("stk-down").addEventListener("click", () => { stopPlay(); setHeight(STK.height - 1); });
  $("stk-play").addEventListener("click", togglePlay);
  $("stk-reset").addEventListener("click", resetZoom);
  $("stk-png").addEventListener("click", exportStackPng);
  document.querySelectorAll("#tab-stack [data-toggle]").forEach((cb) =>
    cb.addEventListener("change", () => {
      STK.toggles[cb.dataset.toggle] = cb.checked;
      saveToggles();
      applyToggles();
    }));
  document.querySelectorAll(".rtab").forEach((t) =>
    t.addEventListener("click", () => showTab(t.dataset.tab)));
  loadToggles();
}

// -- PNG export of the whole map at the current stack height (ignores zoom)
function exportStackPng() {
  if (!STK.svg || STK.isolated !== null) return;
  const m = STK.data.model;
  const clone = STK.svg.cloneNode(true);
  clone.querySelector("#stk-view").removeAttribute("transform");
  clone.querySelector("path.hi").remove();
  // Inline the styles the page CSS provides so the standalone image matches.
  clone.querySelectorAll("path.cut").forEach((p) => {
    p.setAttribute("stroke", "rgba(0,0,0,.25)"); p.setAttribute("stroke-width", "0.12");
  });
  clone.querySelectorAll("path.op").forEach((p) => p.setAttribute("stroke-width", "0.3"));
  clone.querySelectorAll("path.op-engrave_fill, path.op-icon").forEach((p) => p.setAttribute("stroke", "none"));
  const ghost = clone.querySelector("path.ghost");
  ghost.setAttribute("fill", "none"); ghost.setAttribute("stroke", "#2d6cdf");
  ghost.setAttribute("stroke-width", "0.5"); ghost.setAttribute("stroke-dasharray", "2 1.5");
  const frame = clone.querySelector("rect.frame");
  frame.setAttribute("fill", "#fff"); frame.setAttribute("stroke", "#c9ced8");
  frame.setAttribute("stroke-width", "0.3");
  clone.querySelectorAll("[style]").forEach((e) => {
    if (e.style.display === "none") e.remove(); else e.removeAttribute("style");
  });
  const pxPerMm = Math.min(4, 4096 / Math.max(m.width_mm, m.height_mm));
  const w = Math.round(m.width_mm * pxPerMm), h = Math.round(m.height_mm * pxPerMm);
  clone.setAttribute("width", w); clone.setAttribute("height", h);
  clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  const blob = new Blob([new XMLSerializer().serializeToString(clone)], { type: "image/svg+xml" });
  const url = URL.createObjectURL(blob);
  const img = new Image();
  img.onload = () => {
    const canvas = document.createElement("canvas");
    canvas.width = w; canvas.height = h;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, w, h);
    ctx.drawImage(img, 0, 0, w, h);
    URL.revokeObjectURL(url);
    canvas.toBlob((png) => {
      const a = document.createElement("a");
      a.href = URL.createObjectURL(png);
      a.download = `${val("project_name") || "map"}_stack_${STK.height}of${STK.n}.png`;
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 5000);
    }, "image/png");
  };
  img.onerror = () => { URL.revokeObjectURL(url); alert("could not render the PNG"); };
  img.src = url;
}

// --- legend ----------------------------------------------------------------
// Every path the pipeline writes carries data-op = the laser operation it maps
// to (color = operation, the Glowforge convention). The legend is built from
// the ops actually present in the SVG on stage, and the swatch colors are read
// from the SVG itself, so [machine.colors] overrides show up automatically.
const OPS = [
  ["cut", "Cut", "through-cut outline of this layer / part", "line"],
  ["score_registration", "Registration", "outline of the layer above — glue the next layer inside this line", "line"],
  ["seam", "Panel seam", "where this oversized layer is split to fit the bed (preview only, not cut)", "line"],
  ["score_hydro", "Water", "rivers and lake outlines, scored", "line"],
  ["score_trail", "Trail", "hiking paths and routes, scored as dashes", "dash"],
  ["leader", "Label leader", "thin score linking a name to its feature", "line"],
  ["score_ids", "Part ID / leaders", "scored part number (matches the assembly guide) and label leaders", "line"],
  ["engrave_fill", "Engrave", "place names, elevations, trail names (filled = engraved)", "fill"],
  ["icon", "Icon", "engraved feature icon", "fill"],
  ["board_header", "Board header", "engraved project / material / board number", "fill"],
];

function safeColor(c) {
  return /^(#[0-9a-f]{3,8}|[a-z]{3,20})$/i.test(c || "") ? c : "#000";
}

function swatch(kind, color) {
  const c = safeColor(color);
  if (kind === "fill")
    return `<svg class="swatch" viewBox="0 0 30 12"><rect x="0" y="0" width="30" height="12" fill="${c}"/></svg>`;
  const dash = kind === "dash" ? ' stroke-dasharray="5 3"' : "";
  return `<svg class="swatch" viewBox="0 0 30 12"><line x1="0" y1="6" x2="30" y2="6" stroke="${c}" stroke-width="2"${dash}/></svg>`;
}

function legendItem(sw, label, desc) {
  return `<span class="item">${sw}<b>${esc(label)}</b><span class="desc">${esc(desc)}</span></span>`;
}

function renderLegend(svg, el = $("legend")) {
  const items = [];
  if (svg) {
    // First path per op → its color.
    const byOp = new Map();
    svg.querySelectorAll("path[data-op]").forEach((p) => {
      if (!byOp.has(p.dataset.op)) byOp.set(p.dataset.op, p);
    });
    // Composite preview: one filled band per layer, light (base) → dark (summit).
    const bands = [...byOp.keys()].filter((k) => /^layer_\d+$/.test(k))
      .sort((a, b) => parseInt(a.slice(6), 10) - parseInt(b.slice(6), 10));
    if (bands.length) {
      const n = bands.length;
      const stops = bands.map((k, i) => {
        const c = safeColor(byOp.get(k).getAttribute("fill"));
        return `${c} ${(100 * i / n).toFixed(1)}% ${(100 * (i + 1) / n).toFixed(1)}%`;
      });
      const s = lastResult && lastResult.scale;
      const per = s ? ` — each band is one ply, ${s.interval_m} m of elevation` : "";
      items.push(legendItem(
        `<span class="swatch bands" style="background:linear-gradient(90deg,${stops.join(",")})"></span>`,
        "Layer stack", `${n} layers stacked; light = base, dark = summit${per}`));
    }
    for (const [op, label, desc, kind] of OPS) {
      const p = byOp.get(op);
      if (!p) continue;
      const color = p.getAttribute(kind === "fill" ? "fill" : "stroke");
      items.push(legendItem(swatch(kind, color), label, desc));
    }
  }
  el.innerHTML = items.length
    ? `<span class="legend-title">Legend · color = laser operation</span>${items.join("")}` : "";
  el.classList.toggle("hidden", !items.length);
}

function esc(s) { return String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c])); }

function setMode(m) {
  mode = m;
  document.querySelectorAll(".tab").forEach((t) =>
    t.classList.toggle("active", t.dataset.mode === m));
  document.querySelectorAll(".mode-real").forEach((e) => e.classList.toggle("hidden", m !== "real"));
  document.querySelectorAll(".mode-fictional").forEach((e) => e.classList.toggle("hidden", m !== "fictional"));
  // Live scale only applies to real-world (fictional needs the bundle).
  $("scalecard").classList.toggle("hidden", m === "fictional");
  if (m === "real") liveScale();
}

// Relabel the value field + pick a sensible default when the scale driver changes.
const _DRIVER_META = {
  layer_count:  { label: "Number of layers", step: "1", def: "15" },
  exaggeration: { label: "Vertical exaggeration (×)", step: "0.1", def: "1.5" },
  interval_m:   { label: "Contour interval (m)", step: "10", def: "100" },
};
function syncDriverField() {
  const m = _DRIVER_META[val("driver")] || _DRIVER_META.layer_count;
  $("driver_value_label").childNodes[0].nodeValue = m.label + " ";
  $("driver_value").step = m.step;
  $("driver_value").value = m.def;
  scheduleScale();
}

function updateSourceFields() {
  const tier = val("ftier");
  $("ffile-wrap").classList.toggle("hidden", tier === "bundle");
  $("fpath-wrap").classList.toggle("hidden", tier !== "bundle");
  $("foverlay-wrap").classList.toggle("hidden", tier !== "art");
  $("fseg-wrap").classList.toggle("hidden", tier !== "art");
}

// --- wiring ----------------------------------------------------------------
function wire() {
  ["west", "south", "east", "north"].forEach((id) =>
    $(id).addEventListener("change", () => { drawRect(true); scheduleScale(); }));
  ["model_width_mm", "ply_thickness_mm", "driver", "driver_value", "dem"].forEach((id) =>
    $(id).addEventListener("input", scheduleScale));
  $("driver").addEventListener("change", syncDriverField);
  syncDriverField();
  $("run").addEventListener("click", run);
  $("toggle_toml").addEventListener("click", () => $("toml_wrap").classList.toggle("hidden"));
  $("from_form").addEventListener("click", () => { $("toml").value = toToml(readConfig()); });

  // Individual trail selection.
  $("load-trails").addEventListener("click", loadTrails);
  $("trails-all").addEventListener("click", () => toggleAllTrails(true));
  $("trails-none").addEventListener("click", () => toggleAllTrails(false));

  // Real / Fictional tab switch.
  document.querySelectorAll(".tab").forEach((t) =>
    t.addEventListener("click", () => setMode(t.dataset.mode)));
  $("ftier").addEventListener("change", updateSourceFields);
  updateSourceFields();
  wireStackStage();
  // The map is a convenience over the numeric fields; if Leaflet (CDN) fails to
  // load, keep the rest of the app fully functional.
  try {
    if (window.L) initMap();
    else $("map").innerHTML = '<p class="muted" style="padding:14px">Map unavailable (offline) — enter the bounding box numerically.</p>';
  } catch (e) {
    $("map").innerHTML = '<p class="muted" style="padding:14px">Map failed to load — use the numeric fields.</p>';
  }
  liveScale();
}
document.addEventListener("DOMContentLoaded", wire);
