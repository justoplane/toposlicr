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

// --- config assembly from the form ----------------------------------------
function readConfig() {
  const bbox = [num("west"), num("south"), num("east"), num("north")];
  const physical = {
    model_width_mm: num("model_width_mm"),
    ply_thickness_mm: num("ply_thickness_mm"),
  };
  physical[val("driver")] = num("driver_value");

  const overrides = {};
  val("mat_overrides").split(",").forEach((pair) => {
    const [k, v] = pair.split("=").map((s) => s && s.trim());
    if (k && v) overrides[k] = v;
  });

  const peaks = { min_prominence_m: num("peak_prom"), label_elevation: true };
  if (val("peak_max") !== "") peaks.max_count = parseInt(val("peak_max"), 10);

  return {
    region: { bbox, dem: val("dem") },
    physical,
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
  return body;
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

  let res;
  try {
    res = await fetch("/api/run", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(requestBody()),
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
function renderResults(r) {
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
  } catch (e) { stage.innerHTML = '<p class="err">could not load SVG</p>'; }
}

function esc(s) { return String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c])); }

// --- wiring ----------------------------------------------------------------
function wire() {
  ["west", "south", "east", "north"].forEach((id) =>
    $(id).addEventListener("change", () => { drawRect(true); scheduleScale(); }));
  ["model_width_mm", "ply_thickness_mm", "driver", "driver_value", "dem"].forEach((id) =>
    $(id).addEventListener("input", scheduleScale));
  $("driver").addEventListener("change", () => {
    const d = $("driver").value;
    $("driver_value").step = d === "layer_count" ? "1" : "0.1";
  });
  $("run").addEventListener("click", run);
  $("toggle_toml").addEventListener("click", () => $("toml_wrap").classList.toggle("hidden"));
  $("from_form").addEventListener("click", () => { $("toml").value = toToml(readConfig()); });
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
