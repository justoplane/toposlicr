// 3D stack viewer: the same stack.json the 2D view draws, extruded into ply
// slabs with each layer's symbology on its top face. app.js owns the state
// (height, toggles, hover) and loads this module on demand; the module only
// renders what it is told, and renders on demand — no animation loop.
//
// Frames: the payload is model mm with SVG y-down. World is three.js Y-up with
// the map in the XZ plane: X = east, Z = svg y (south), Y = height. Shapes are
// built in svg (x, y) and rotated +90° about X, which maps svg y → +Z and the
// extrusion depth → −Y, so each slab hangs below its layer group's origin and
// the group sits at the layer's top surface.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { LineSegments2 } from "three/addons/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/addons/lines/LineSegmentsGeometry.js";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";

const ACCENT = 0x2d6cdf;
const LIFT = 0.04;                  // mm: symbology floats this far above its face
const SHADOW_VERTEX_BUDGET = 400000;  // same budget the 2D drop shadows use
// op → [toggle group, color key, kind, line px, dash mm]
const OPS = {
  score_hydro: ["water", "score_hydro", "line", 2],
  score_trail: ["trails", "score_trail", "line", 2],
  leader: ["names", "score_ids", "line", 1],
  engrave_fill: ["names", "engrave_fill", "fill"],
  icon: ["names", "engrave_fill", "fill"],
  seam: ["seams", "seam", "line", 2, [2, 1.5]],
};

// -- materials ----------------------------------------------------------------
// Material keys are free text (birch_3mm, walnut_3mm, blue_acrylic_3mm …), so
// they are matched by keyword with a neutral wood as the fallback.
const WOODS = [
  ["walnut", 0x7a5539], ["cherry", 0xa8674a], ["mahogany", 0x8a4b33],
  ["oak", 0xcaa36f], ["maple", 0xead6b2], ["birch", 0xe6cda4], ["poplar", 0xe0d3ac],
  ["bamboo", 0xdcc283], ["cork", 0xb8875a], ["mdf", 0xa88a68],
];
const ACRYLICS = [
  ["clear", 0xd8ecf5, 0.35], ["smoke", 0x4a4f55, 0.6], ["black", 0x1e1f22, 1],
  ["white", 0xf4f4f4, 1], ["green", 0x3aa66b, 0.7], ["red", 0xc8423b, 0.75],
  ["yellow", 0xe8c242, 0.75], ["blue", 0x3d8fd6, 0.72],
];

export function materialLook(name) {
  const s = String(name || "").toLowerCase();
  if (s.includes("acrylic")) {
    const [, color, opacity] = ACRYLICS.find(([k]) => s.includes(k)) || ACRYLICS.at(-1);
    return { kind: "acrylic", color, opacity };
  }
  const [, color] = WOODS.find(([k]) => s.includes(k)) || [null, 0xd8bd92];
  return { kind: "wood", color };
}

let grainTex = null;
function woodGrain() {
  // A light, tileable grain the wood color multiplies; UVs are mm.
  if (grainTex) return grainTex;
  const c = document.createElement("canvas");
  c.width = c.height = 512;
  const g = c.getContext("2d");
  g.fillStyle = "#fff";
  g.fillRect(0, 0, 512, 512);
  let seed = 7;
  const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  for (let i = 0; i < 90; i++) {
    const y0 = rnd() * 512, amp = 2 + rnd() * 6, f = 1 + Math.floor(rnd() * 3);
    g.strokeStyle = `rgba(110,70,30,${0.03 + rnd() * 0.07})`;
    g.lineWidth = 0.6 + rnd() * 2.2;
    for (const off of [-512, 0, 512]) {          // wrap vertically so it tiles
      g.beginPath();
      for (let x = 0; x <= 512; x += 8) {
        const y = y0 + off + amp * Math.sin((x / 512) * Math.PI * 2 * f + i);
        x ? g.lineTo(x, y) : g.moveTo(x, y);
      }
      g.stroke();
    }
  }
  grainTex = new THREE.CanvasTexture(c);
  grainTex.wrapS = grainTex.wrapT = THREE.RepeatWrapping;
  grainTex.repeat.set(1 / 140, 1 / 140);
  grainTex.colorSpace = THREE.SRGBColorSpace;
  grainTex.anisotropy = 4;
  return grainTex;
}

// Per-layer [top/bottom, sides] materials. Each layer gets its own pair so
// hover can tint one slab, with a tiny deterministic tone shift per layer
// (real sheets vary, and it helps tell neighbors apart).
function slabMaterials(name, k) {
  const look = materialLook(name);
  const base = new THREE.Color(look.color);
  base.offsetHSL(0, 0, ((k * 37) % 7 - 3) * 0.006);
  if (look.kind === "acrylic") {
    const m = new THREE.MeshStandardMaterial({
      color: base, roughness: 0.15, metalness: 0,
      transparent: look.opacity < 1, opacity: look.opacity });
    return [m, m.clone()];
  }
  const top = new THREE.MeshStandardMaterial({ color: base, map: woodGrain(), roughness: 0.85 });
  // Laser-cut edges are charred: a dark brown that keeps a hint of the wood.
  const side = new THREE.MeshStandardMaterial({
    color: base.clone().lerp(new THREE.Color(0x2e1d10), 0.72), roughness: 0.95 });
  return [top, side];
}

// -- path data → geometry -------------------------------------------------------
function parseRings(d) {
  // "M x,y L x,y … Z M …" → [{ pts: [[x, y], …], closed }]
  const out = [];
  let cur = null;
  const re = /([MLZ])([^MLZ]*)/g;
  let m;
  while ((m = re.exec(d || ""))) {
    if (m[1] === "Z") { if (cur) cur.closed = true; continue; }
    const comma = m[2].indexOf(",");
    const x = +m[2].slice(0, comma), y = +m[2].slice(comma + 1);
    if (m[1] === "M" || !cur) { cur = { pts: [], closed: false }; out.push(cur); }
    cur.pts.push([x, y]);
  }
  return out;
}

function signedArea(pts) {
  let a = 0;
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++)
    a += (pts[j][0] - pts[i][0]) * (pts[j][1] + pts[i][1]);
  return a;
}

// The backend winds every exterior one way and every hole the other, and
// lists each exterior before its holes — so winding alone groups the rings.
function toShapes(d) {
  const shapes = [];
  let exteriorSign = 0;
  for (const { pts } of parseRings(d)) {
    const last = pts.at(-1);
    if (pts.length > 1 && last[0] === pts[0][0] && last[1] === pts[0][1]) pts.pop();
    if (pts.length < 3) continue;
    const sign = Math.sign(signedArea(pts));
    if (!sign) continue;
    const v = pts.map(([x, y]) => new THREE.Vector2(x, y));
    if (!exteriorSign) exteriorSign = sign;
    if (sign === exteriorSign) shapes.push(new THREE.Shape(v));
    else if (shapes.length) shapes.at(-1).holes.push(new THREE.Path(v));
  }
  return shapes;
}

function extrude(d, depth) {
  const shapes = toShapes(d);
  if (!shapes.length) return null;
  const g = new THREE.ExtrudeGeometry(shapes, { depth, bevelEnabled: false, curveSegments: 1 });
  g.rotateX(Math.PI / 2);
  return g;
}

function flat(d) {
  const shapes = toShapes(d);
  if (!shapes.length) return null;
  const g = new THREE.ShapeGeometry(shapes, 1);
  g.rotateX(Math.PI / 2);
  return g;
}

// Polylines → one LineSegments2 per op per layer (one draw call), with dash
// distances accumulated along each polyline so dashes don't restart per vertex.
function lines(d, y, color, px, dash) {
  const pos = [], dist = [];
  for (const r of parseRings(d)) {
    const p = r.pts;
    if (r.closed && p.length > 2) p.push(p[0]);
    let s = 0;
    for (let i = 1; i < p.length; i++) {
      const [x0, z0] = p[i - 1], [x1, z1] = p[i];
      const len = Math.hypot(x1 - x0, z1 - z0);
      pos.push(x0, y, z0, x1, y, z1);
      dist.push(s, s + len);
      s += len;
    }
  }
  return segments(pos, dist, color, px, dash);
}

function segments(pos, dist, color, px, dash) {
  if (!pos.length) return null;
  const g = new LineSegmentsGeometry();
  g.setPositions(pos);
  // Polygon offset keeps lines lying on a face (side seams) from z-fighting it.
  const mat = new LineMaterial({ color, linewidth: px, worldUnits: false,
    polygonOffset: true, polygonOffsetFactor: -4, polygonOffsetUnits: -4 });
  if (dash) {
    const buf = new THREE.InstancedInterleavedBuffer(new Float32Array(dist), 2, 1);
    g.setAttribute("instanceDistanceStart", new THREE.InterleavedBufferAttribute(buf, 1, 0));
    g.setAttribute("instanceDistanceEnd", new THREE.InterleavedBufferAttribute(buf, 1, 1));
    mat.dashed = true;
    mat.dashSize = dash[0];
    mat.gapSize = dash[1];
  }
  return new LineSegments2(g, mat);
}

// Seams run down the slab edge wherever a seam line meets the outline: the
// endpoints of each seam polyline. Built at unit height so it scales with the
// slab (it lives inside the slab mesh).
function sideSeams(d, color) {
  const pos = [], dist = [];
  for (const { pts } of parseRings(d)) {
    for (const [x, z] of [pts[0], pts.at(-1)]) {
      pos.push(x, LIFT * 0.5, z, x, -1 - LIFT * 0.5, z);
      dist.push(0, 1);
    }
  }
  return segments(pos, dist, color, 2, null);
}

// -- viewer -------------------------------------------------------------------
export function createStack3D(container, { onHover } = {}) {
  const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.shadowMap.type = THREE.PCFShadowMap;
  renderer.domElement.className = "stack3d-canvas";
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf4f6fa);
  const camera = new THREE.PerspectiveCamera(35, 1, 1, 10000);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.maxPolarAngle = Math.PI * 0.49;     // stay above the table
  controls.screenSpacePanning = true;

  scene.add(new THREE.HemisphereLight(0xffffff, 0x8a7a66, 1.1));
  const sun = new THREE.DirectionalLight(0xffffff, 1.9);
  sun.shadow.mapSize.set(2048, 2048);
  sun.shadow.bias = -0.0004;
  sun.shadow.normalBias = 0.3;
  scene.add(sun, sun.target);

  const ground = new THREE.Mesh(new THREE.PlaneGeometry(1, 1),
    new THREE.ShadowMaterial({ opacity: 0.18 }));
  ground.rotation.x = -Math.PI / 2;
  ground.position.y = -0.01;
  ground.receiveShadow = true;
  scene.add(ground);

  const root = new THREE.Group();
  scene.add(root);
  const ghostMat = new THREE.MeshBasicMaterial({ color: ACCENT, transparent: true,
    opacity: 0.16, depthWrite: false });
  const ghost = new THREE.Mesh(undefined, ghostMat);
  const ghostEdge = new THREE.Group();
  ghost.visible = false;
  scene.add(ghost, ghostEdge);

  const S = {
    data: null, layers: [], height: 0, hover: null, explode: 0, stretch: 1,
    toggles: {}, shadowsAllowed: true, size: { w: 0, h: 0 },
  };
  let pending = false;
  const render = () => {
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => { pending = false; renderer.render(scene, camera); });
  };
  controls.addEventListener("change", render);

  // -- build
  function setData(data) {
    clear();
    S.data = data;
    const m = data.model, t = m.ply_thickness_mm, n = data.layers.length;
    const colors = data.colors || {};
    S.shadowsAllowed = 2 * data.layers.reduce((a, l) => a + l.vertices, 0) <= SHADOW_VERTEX_BUDGET;
    data.layers.forEach((lyr, k) => {
      const group = new THREE.Group();
      const geo = extrude(lyr.solid || lyr.cut, 1);
      const mats = slabMaterials(lyr.material, k);
      const slab = new THREE.Mesh(geo || undefined, mats);
      slab.castShadow = slab.receiveShadow = true;
      slab.userData.k = k;
      group.add(slab);
      const tagged = [];                        // [object, toggle group]
      for (const [op, dd] of Object.entries(lyr.ops || {})) {
        if (!dd) continue;
        if (op === "inset") {
          const g = extrude(dd, 1);
          if (!g) continue;
          const look = materialLook(m.water_material || "blue_acrylic");
          const mesh = new THREE.Mesh(g, new THREE.MeshStandardMaterial({
            color: look.color, roughness: 0.12, transparent: look.opacity < 1,
            opacity: look.opacity }));
          mesh.receiveShadow = true;
          mesh.userData.k = k;
          slab.add(mesh);                       // scales with the slab
          tagged.push([mesh, "water"]);
          continue;
        }
        const spec = OPS[op];
        if (!spec) continue;
        const color = new THREE.Color(cssColor(colors[spec[1]]));
        let obj;
        if (spec[2] === "fill") {
          const g = flat(dd);
          obj = g && new THREE.Mesh(g, new THREE.MeshBasicMaterial({
            color, side: THREE.DoubleSide, polygonOffset: true,
            polygonOffsetFactor: -2, polygonOffsetUnits: -2 }));
          if (obj) obj.position.y = LIFT;
        } else {
          obj = lines(dd, LIFT, color, spec[3], spec[4]);
        }
        if (!obj) continue;
        group.add(obj);
        tagged.push([obj, spec[0]]);
        if (op === "seam") {
          const side = sideSeams(dd, color);
          if (side) { slab.add(side); tagged.push([side, "seams"]); }
        }
      }
      root.add(group);
      S.layers.push({ group, slab, mats, geo, tagged, baseEmissive: mats.map((x) => x.emissive.clone()) });
    });

    // Table + shadow camera sized to the model.
    const w = m.width_mm, h = m.height_mm, top = n * t * 6;
    const span = Math.max(w, h) * 1.6;
    ground.scale.set(span * 2, span * 2, 1);
    ground.position.set(w / 2, -0.01, h / 2);
    sun.position.set(w / 2 - span * 0.5, span * 0.9, h / 2 - span * 0.35);
    sun.target.position.set(w / 2, 0, h / 2);
    const sc = sun.shadow.camera;
    sc.left = sc.bottom = -span * 0.75;
    sc.right = sc.top = span * 0.75;
    sc.near = 1; sc.far = span * 3 + top;
    sc.updateProjectionMatrix();

    layout();
    applyToggles();
    applyHeight();
    resize();          // the observer reports size asynchronously; fit needs the aspect now
    fit();
  }

  function clear() {
    for (const L of S.layers) {
      L.group.traverse((o) => {
        if (o.geometry) o.geometry.dispose();
        const mm = Array.isArray(o.material) ? o.material : [o.material];
        mm.forEach((x) => x && x.dispose());
      });
      root.remove(L.group);
    }
    clearGhostEdge();
    S.layers = [];
    S.hover = null;
  }

  // -- layout: z of every layer from thickness × stretch + explode gap
  function topOf(k) {
    const t = S.data.model.ply_thickness_mm;
    return (k + 1) * t * S.stretch + k * S.explode * t;
  }

  function layout() {
    if (!S.data) return;
    const t = S.data.model.ply_thickness_mm;
    S.layers.forEach((L, k) => {
      L.group.position.y = topOf(k);
      L.slab.scale.y = t * S.stretch;
    });
    placeGhost();
  }

  // -- state from app.js
  function setHeight(h) { S.height = h; applyHeight(); }

  function applyHeight() {
    S.layers.forEach((L, k) => { L.group.visible = k < S.height; });
    placeGhost();
    render();
  }

  function setToggles(t) { S.toggles = { ...t }; applyToggles(); }

  function applyToggles() {
    for (const L of S.layers)
      for (const [obj, tg] of L.tagged) obj.visible = !!S.toggles[tg];
    const shadows = !!S.toggles.shadows && S.shadowsAllowed;
    renderer.shadowMap.enabled = shadows;
    sun.castShadow = shadows;
    ground.visible = shadows;
    placeGhost();
    render();
  }

  // The ghost shows where a not-yet-stacked layer goes: the next layer when the
  // ghost toggle is on, or whichever hidden layer the rail is hovering.
  function clearGhostEdge() {
    for (const o of ghostEdge.children) { o.geometry.dispose(); o.material.dispose(); }
    ghostEdge.clear();
  }

  function placeGhost() {
    clearGhostEdge();
    ghost.visible = false;
    if (!S.data) return;
    const n = S.layers.length;
    let k = null;
    if (S.hover !== null && S.hover >= S.height) k = S.hover;
    else if (S.toggles.ghost && S.height < n && S.height > 0) k = S.height;
    if (k === null || !S.layers[k].geo) return;
    const t = S.data.model.ply_thickness_mm;
    ghost.geometry = S.layers[k].geo;
    ghost.position.y = topOf(k);
    ghost.scale.y = t * S.stretch;
    ghost.visible = true;
    const edge = lines(S.data.layers[k].cut, topOf(k) + LIFT, new THREE.Color(ACCENT), 1.5, [2, 1.5]);
    if (edge) ghostEdge.add(edge);
  }

  function setHover(k) {
    if (k === S.hover) return;
    if (S.hover !== null && S.layers[S.hover]) tint(S.layers[S.hover], false);
    S.hover = k;
    if (k !== null && S.layers[k] && k < S.height) tint(S.layers[k], true);
    placeGhost();
    render();
  }

  function tint(L, on) {
    L.mats.forEach((m, i) => {
      if (on) m.emissive.setHex(ACCENT).multiplyScalar(0.28);
      else m.emissive.copy(L.baseEmissive[i]);
    });
  }

  function setExplode(v) { S.explode = v; layout(); render(); }
  function setStretch(v) { S.stretch = v; layout(); render(); }

  // 3/4 view from the south-west, ~35° above the table, framing the full stack.
  function fit() {
    if (!S.data) return;
    const m = S.data.model;
    const n = S.layers.length;
    const box = new THREE.Box3(new THREE.Vector3(0, 0, 0),
      new THREE.Vector3(m.width_mm, n ? topOf(n - 1) : 0, m.height_mm));
    const center = box.getCenter(new THREE.Vector3());
    const el = THREE.MathUtils.degToRad(35), az = THREE.MathUtils.degToRad(-28);
    const dir = new THREE.Vector3(Math.sin(az) * Math.cos(el), Math.sin(el),
      Math.cos(az) * Math.cos(el));
    // Back off along dir until every box corner is inside the frustum.
    const right = new THREE.Vector3().crossVectors(new THREE.Vector3(0, 1, 0), dir).normalize();
    const up = new THREE.Vector3().crossVectors(dir, right);
    const tanV = Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2), tanH = tanV * camera.aspect;
    let dist = 0;
    const c = new THREE.Vector3();
    for (let i = 0; i < 8; i++) {
      c.set(i & 1 ? box.max.x : box.min.x, i & 2 ? box.max.y : box.min.y,
        i & 4 ? box.max.z : box.min.z).sub(center);
      const depth = c.dot(dir);
      dist = Math.max(dist, depth + Math.abs(c.dot(right)) / tanH,
        depth + Math.abs(c.dot(up)) / tanV);
    }
    dist *= 1.08;
    const radius = box.getBoundingSphere(new THREE.Sphere()).radius;
    controls.target.copy(center);
    camera.position.copy(center).addScaledVector(dir, dist);
    camera.near = Math.max(0.5, dist / 200);
    camera.far = dist * 20;
    camera.updateProjectionMatrix();
    controls.minDistance = radius * 0.05;
    controls.maxDistance = dist * 4;
    controls.update();
    render();
  }

  // -- size
  function resize() {
    const w = container.clientWidth, h = container.clientHeight;
    if (!w || !h || (w === S.size.w && h === S.size.h)) return;
    S.size = { w, h };
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    render();
  }
  const ro = new ResizeObserver(resize);
  ro.observe(container);

  // -- hover picking (throttled to one raycast per frame, skipped mid-drag)
  const ray = new THREE.Raycaster();
  const ndc = new THREE.Vector2();
  let dragging = false, pickQueued = null;
  controls.addEventListener("start", () => { dragging = true; });
  controls.addEventListener("end", () => { dragging = false; });
  renderer.domElement.addEventListener("pointermove", (e) => {
    if (dragging || !S.data) return;
    const first = pickQueued === null;
    pickQueued = e;
    if (first) requestAnimationFrame(() => { const ev = pickQueued; pickQueued = null; pick(ev); });
  });
  renderer.domElement.addEventListener("pointerleave", () => onHover && onHover(null));

  function pick(e) {
    const r = renderer.domElement.getBoundingClientRect();
    ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    ray.setFromCamera(ndc, camera);
    const slabs = S.layers.filter((L, k) => k < S.height).map((L) => L.slab);
    const hit = ray.intersectObjects(slabs, true)[0];
    let k = null;
    for (let o = hit && hit.object; o; o = o.parent)
      if (o.userData.k !== undefined) { k = o.userData.k; break; }
    if (onHover) onHover(k);
  }

  // -- PNG: render synchronously and read back before the buffer is cleared
  function snapshot() {
    const hover = S.hover;
    setHover(null);
    renderer.render(scene, camera);
    const url = renderer.domElement.toDataURL("image/png");
    setHover(hover);
    return url;
  }

  function dispose() {
    ro.disconnect();
    clear();
    controls.dispose();
    renderer.dispose();
    renderer.domElement.remove();
  }

  return { setData, setHeight, setToggles, setHover, setExplode, setStretch, fit, resize,
           snapshot, dispose };
}

function cssColor(c) {
  return /^#[0-9a-f]{3}([0-9a-f]{3})?$/i.test(c || "") ? c : "#000000";
}
