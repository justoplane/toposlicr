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
      kv("interval", d.interval_m + " m") +
      kv("exaggeration", d.exaggeration + "×") +
      kv("layers", d.layer_count ?? "run to compute") +
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
      ? `<div class="warnbox"><b>${r.warnings.length} warning(s)</b><ul>${
          r.warnings.map((w) => `<li>${esc(w)}</li>`).join("")}</ul></div>` : "");

  const thumb = (row, label, url, sub) =>
    `<button class="thumb" data-url="${url}"><b>${label}</b>${sub ? "<br>" + sub : ""}</button>`;

  $("row-preview").innerHTML = r.preview_url
    ? thumb("preview", "composite preview", r.preview_url, "all layers") : "";
  $("row-layers").innerHTML = r.layers.map((l) =>
    thumb("l", "layer " + l.index, l.url, "≥ " + l.threshold_m + " m")).join("");
  $("row-boards").innerHTML = r.boards.map((b) =>
    thumb("b", b.material + " #" + b.index, b.url,
      b.parts + "p · " + b.utilization + "% · " + b.cut_m + " m")).join("");

  document.querySelectorAll(".thumb").forEach((t) =>
    t.addEventListener("click", () => showSvg(t)));
  const first = document.querySelector(".thumb");
  if (first) showSvg(first);
}

async function showSvg(btn) {
  document.querySelectorAll(".thumb").forEach((t) => t.classList.remove("active"));
  btn.classList.add("active");
  const url = btn.dataset.url;
  const stage = $("stage");
  stage.innerHTML = '<p class="muted">loading…</p>';
  try {
    const txt = await (await fetch(url)).text();
    const name = url.split("/").pop();
    stage.innerHTML = txt +
      `<div class="downloads" style="position:absolute;right:14px;bottom:8px">` +
      `<a href="${url}" download="${name}">⬇ download ${name}</a></div>`;
    stage.style.position = "relative";
    renderLegend(stage.querySelector("svg"));
  } catch (e) {
    stage.innerHTML = '<p class="err">could not load SVG</p>';
    renderLegend(null);
  }
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

function renderLegend(svg) {
  const el = $("legend");
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
