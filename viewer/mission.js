// Orbital mission control: lift off into orbit, pick a landing site on the globe, deorbit, then hand the
// landing over to the 6-DOF replay viewer in index.html. All physics runs on the server
// (rocket_sim/orbit.py via play.py); this module draws it. The look follows a live satellite tracker.
//
// Scene units: 1 = Earth radius. World axes map ECI as (x, y, z)_three = (x, z, -y)_eci, so the spin axis is
// +y and the globe turns by ω·T about it (T = mission time, seconds since liftoff).
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { mergeGeometries } from "three/addons/utils/BufferGeometryUtils.js";
import { Sky } from "three/addons/objects/Sky.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { ShaderPass } from "three/addons/postprocessing/ShaderPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";

const RE = 6371000, MU = 3.986004418e14, OMEGA = 7.2921159e-5, DEG = Math.PI / 180;
const TEX = "https://cdn.jsdelivr.net/gh/mrdoob/three.js@r160/examples/textures/planets/";
const Y = new THREE.Vector3(0, 1, 0);
const EVENT_LABEL = {
  liftoff: "Liftoff", max_q: "Max-Q", meco: "Main engine cutoff", stage_sep: "Stage separation",
  ses: "Stage 2 ignition", seco1: "SECO-1 · coasting", circ_burn: "Circularization burn", seco2: "SECO-2 · in orbit",
  deorbit: "Deorbit burn", entry_interface: "Entry interface", max_heating: "Peak heating",
  peak_decel: "Peak deceleration", braking_burn: "Braking burn", gate: "Hand-off to landing",
};
const WARPS = [1, 10, 60, 300, 1000];
const COUNTDOWN = 10;         // s of countdown before liftoff
const VIGNETTE = 17;          // s after liftoff shown from the pad camera

// ---------------------------------------------------------------- small helpers
const $ = id => document.getElementById(id);
const clamp = (v, a, b) => Math.min(Math.max(v, a), b);
const wrap180 = d => ((d + 540) % 360) - 180;
const smooth = v => { v = clamp(v, 0, 1); return v * v * (3 - 2 * v); };
const pad2 = n => String(Math.floor(n)).padStart(2, "0");
function fmtT(t) {
  const s = Math.abs(t), sign = t < 0 ? "T−" : "T+";
  return `${sign}${pad2(s / 3600)}:${pad2((s % 3600) / 60)}:${pad2(s % 60)}`;
}
function fmtDur(s) {
  if (s < 90) return `${Math.max(s, 0).toFixed(0)} s`;
  if (s < 5400) return `${(s / 60).toFixed(0)} min`;
  return `${Math.floor(s / 3600)} h ${pad2((s % 3600) / 60)} min`;
}
const km = m => `${(m / 1000).toLocaleString(undefined, { maximumFractionDigits: m < 1e5 ? 1 : 0 })} km`;
const fmtLat = v => `${Math.abs(v).toFixed(3)}° ${v >= 0 ? "N" : "S"}`;
const fmtLon = v => `${Math.abs(v).toFixed(3)}° ${v >= 0 ? "E" : "W"}`;
const fmtWind = w => {
  const sp = Math.hypot(w[0], w[2]);
  // landing-world axes: +x east, +z south; wind direction is reported as where it blows from
  const from = (Math.atan2(-w[0], w[2]) / DEG + 360) % 360;
  const names = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"];
  return `${sp.toFixed(1)} m/s from ${names[Math.round(from / 45) % 8]}`;
};
async function post(url, body) {
  const r = await fetch(url, { method: "POST", body: JSON.stringify(body) });
  const j = await r.json();
  if (j.error) throw new Error(j.error);
  return j;
}

function llaLocal(lat, lon, alt = 0, out = new THREE.Vector3()) {
  const la = lat * DEG, lo = lon * DEG, r = 1 + alt / RE;
  return out.set(Math.cos(la) * Math.cos(lo) * r, Math.sin(la) * r, -Math.cos(la) * Math.sin(lo) * r);
}
const eciWorld = (r, out = new THREE.Vector3()) => out.set(r[0] / RE, r[2] / RE, -r[1] / RE);
const fixedToWorld = (v, T) => v.applyAxisAngle(Y, OMEGA * T);

function propagate(o, t) {   // two-body, same as orbit.propagate
  const M = o.M0 + o.n * (t - o.epoch), e = o.e;
  let E = M;
  for (let i = 0; i < 10; i++) E -= (E - e * Math.sin(E) - M) / (1 - e * Math.cos(E));
  const c = Math.cos(E), s = Math.sin(E), b = Math.sqrt(1 - e * e), Ed = o.n / (1 - e * c);
  const x = o.a * (c - e), y = o.a * b * s, vx = -o.a * s * Ed, vy = o.a * b * c * Ed;
  return { r: [0, 1, 2].map(k => x * o.p[k] + y * o.q[k]), v: [0, 1, 2].map(k => vx * o.p[k] + vy * o.q[k]) };
}
function eciLatLon(r, t) {
  const n = Math.hypot(...r);
  return [Math.asin(r[2] / n) / DEG, wrap180((Math.atan2(r[1], r[0]) - OMEGA * t) / DEG)];
}

// Interpolate time-sorted rows ([t, lat, lon, ...]); `discrete` columns keep the earlier row's value.
function sampleRows(rows, t, discrete = []) {
  const n = rows.length;
  if (t <= rows[0][0]) return rows[0].slice();
  if (t >= rows[n - 1][0]) return rows[n - 1].slice();
  let lo = 0, hi = n - 1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (rows[m][0] <= t) lo = m; else hi = m; }
  const A = rows[lo], B = rows[hi], f = (t - A[0]) / Math.max(B[0] - A[0], 1e-9);
  return A.map((a, k) => discrete.includes(k) ? a : k === 2 ? wrap180(a + wrap180(B[k] - a) * f) : a + (B[k] - a) * f);
}
const idx = fields => Object.fromEntries(fields.map((f, i) => [f, i]));

function canvasTex(w, h, draw) {
  const cv = document.createElement("canvas"); cv.width = w; cv.height = h;
  draw(cv.getContext("2d"), w, h);
  const t = new THREE.CanvasTexture(cv); t.colorSpace = THREE.SRGBColorSpace;
  return t;
}
const glowTex = canvasTex(128, 128, (x, w) => {
  const g = x.createRadialGradient(w / 2, w / 2, 0, w / 2, w / 2, w / 2);
  g.addColorStop(0, "rgba(255,255,255,1)"); g.addColorStop(0.18, "rgba(255,255,255,.85)");
  g.addColorStop(0.45, "rgba(255,255,255,.18)"); g.addColorStop(1, "rgba(255,255,255,0)");
  x.fillStyle = g; x.fillRect(0, 0, w, w);
});
let seed = 11;
const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;

// ---------------------------------------------------------------- Earth imagery (also sampled for site picking)
const maps = {};   // name -> { tex, data, w, h }
function loadMap(name, file, srgb) {
  return new Promise(res => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      const tex = new THREE.Texture(img);
      if (srgb) tex.colorSpace = THREE.SRGBColorSpace;
      tex.anisotropy = 8; tex.needsUpdate = true;
      let data = null, w = img.width, h = img.height;
      try {   // pixel access for site classification
        const cv = document.createElement("canvas"); cv.width = w; cv.height = h;
        const cx = cv.getContext("2d", { willReadFrequently: true }); cx.drawImage(img, 0, 0);
        data = cx.getImageData(0, 0, w, h).data;
      } catch { /* tainted canvas: imagery shows, classification falls back */ }
      maps[name] = { tex, data, w, h };
      res(maps[name]);
    };
    img.onerror = () => res(null);
    img.src = TEX + file;
  });
}
function px(map, lat, lon, dx = 0, dy = 0) {
  const x = Math.floor(((lon + 180) / 360) * map.w + dx + map.w) % map.w;
  const y = clamp(Math.floor(((90 - lat) / 180) * map.h + dy), 0, map.h - 1);
  const i = (y * map.w + x) * 4;
  return [map.data[i], map.data[i + 1], map.data[i + 2]];
}
// Land/ocean/ice, ruggedness (0..1) from normal-map relief around the point, and the ground color.
function classify(lat, lon) {
  const spec = maps.spec?.data && maps.spec, nrm = maps.normal?.data && maps.normal, day = maps.day?.data && maps.day;
  let surface = "land", rug = 0.3, color = [0.42, 0.38, 0.3];
  if (spec && px(spec, lat, lon)[0] > 110) surface = "ocean";
  if (day) {
    let c = [0, 0, 0];
    for (let dx = -2; dx <= 2; dx++) for (let dy = -2; dy <= 2; dy++) px(day, lat, lon, dx, dy).forEach((v, k) => c[k] += v / 25 / 255);
    color = c;
    const bright = (c[0] + c[1] + c[2]) / 3, sat = Math.max(...c) - Math.min(...c);
    if (surface === "land" && bright > 0.72 && sat < 0.12) surface = "ice";
  }
  if (nrm) {
    let s = 0, n = 0;
    for (let dx = -6; dx <= 6; dx += 2) for (let dy = -6; dy <= 6; dy += 2) {
      const [r, g] = px(nrm, lat, lon, dx, dy); s += Math.hypot(r - 128, g - 128) / 128; n++;
    }
    rug = clamp((s / n - 0.03) * 3.2, 0, 1);
  }
  if (surface === "ocean") rug = 0;
  return { surface, ruggedness: rug, color };
}

// ---------------------------------------------------------------- globe scene
function buildGlobe() {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x020408);
  const camera = new THREE.PerspectiveCamera(40, 1, 0.005, 200);
  const sunDir = new THREE.Vector3(0.6, 0.25, 0.75).normalize();
  const sun = new THREE.DirectionalLight(0xffffff, 2.2); scene.add(sun, new THREE.AmbientLight(0x30405a, 0.6));

  const earth = new THREE.Group(); scene.add(earth);
  const blank = new THREE.DataTexture(new Uint8Array([22, 52, 92, 255]), 1, 1); blank.needsUpdate = true;
  const black = new THREE.DataTexture(new Uint8Array([0, 0, 0, 255]), 1, 1); black.needsUpdate = true;
  const uniforms = { day: { value: blank }, night: { value: black }, spec: { value: black }, sunDir: { value: sunDir } };
  const mat = new THREE.ShaderMaterial({
    uniforms,
    vertexShader: `varying vec2 vUv; varying vec3 vN; varying vec3 vW;
      void main() { vUv = uv; vN = normalize(mat3(modelMatrix) * normal); vec4 w = modelMatrix * vec4(position, 1.0);
        vW = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }`,
    fragmentShader: `uniform sampler2D day; uniform sampler2D night; uniform sampler2D spec; uniform vec3 sunDir;
      varying vec2 vUv; varying vec3 vN; varying vec3 vW;
      void main() {
        vec3 n = normalize(vN), v = normalize(cameraPosition - vW);
        float d = dot(n, sunDir), lit = smoothstep(-0.18, 0.25, d);
        vec3 dc = texture2D(day, vUv).rgb, nc = texture2D(night, vUv).rgb;
        vec3 col = dc * (0.16 + 1.15 * max(d, 0.0)) * lit + dc * 0.10 * (1.0 - lit) + nc * vec3(1.0, 0.78, 0.5) * 1.4 * (1.0 - lit);
        float s = texture2D(spec, vUv).r;
        col += s * pow(max(dot(n, normalize(sunDir + v)), 0.0), 140.0) * 0.18 * lit * vec3(1.0, 0.95, 0.85);
        float rim = 1.0 - max(dot(n, v), 0.0);
        col += vec3(0.25, 0.55, 1.0) * pow(rim, 3.0) * (0.25 + 0.6 * lit);
        gl_FragColor = vec4(col, 1.0);
        #include <colorspace_fragment>
      }`,
  });
  const globe = new THREE.Mesh(new THREE.SphereGeometry(1, 128, 64), mat);
  earth.add(globe);
  const clouds = new THREE.Mesh(new THREE.SphereGeometry(1.006, 96, 48),
    new THREE.MeshLambertMaterial({ transparent: true, opacity: 0.0, depthWrite: false }));
  earth.add(clouds);
  // atmosphere: back faces of a slightly larger sphere, glowing at the limb
  const atmo = new THREE.Mesh(new THREE.SphereGeometry(1.035, 64, 32), new THREE.ShaderMaterial({
    uniforms: { sunDir: { value: sunDir } },
    vertexShader: `varying vec3 vN; varying vec3 vW; void main() { vN = normalize(mat3(modelMatrix) * normal);
      vec4 w = modelMatrix * vec4(position, 1.0); vW = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }`,
    fragmentShader: `uniform vec3 sunDir; varying vec3 vN; varying vec3 vW;
      void main() { vec3 v = normalize(cameraPosition - vW); float i = pow(clamp(0.72 + dot(normalize(vN), v), 0.0, 1.0), 3.0);
        float lit = 0.35 + 0.65 * smoothstep(-0.3, 0.4, dot(normalize(vW), sunDir));
        gl_FragColor = vec4(vec3(0.3, 0.62, 1.0) * i * lit * 1.6, i * lit); }`,
    side: THREE.BackSide, blending: THREE.AdditiveBlending, transparent: true, depthWrite: false,
  }));
  scene.add(atmo);

  Promise.all([loadMap("day", "earth_atmos_2048.jpg", true), loadMap("night", "earth_lights_2048.png", true),
    loadMap("spec", "earth_specular_2048.jpg", false), loadMap("normal", "earth_normal_2048.jpg", false),
    loadMap("clouds", "earth_clouds_1024.png", true)]).then(([d, n, s, , c]) => {
    if (d) uniforms.day.value = d.tex;
    if (n) uniforms.night.value = n.tex;
    if (s) uniforms.spec.value = s.tex;
    if (c) { clouds.material.map = c.tex; clouds.material.opacity = 0.45; clouds.material.needsUpdate = true; }
  });

  // stars
  const sp = new Float32Array(4000 * 3), sc = new Float32Array(4000 * 3);
  seed = 5;
  for (let i = 0; i < 4000; i++) {
    const u = rnd() * 2 - 1, a = rnd() * Math.PI * 2, r = 90;
    sp.set([Math.sqrt(1 - u * u) * Math.cos(a) * r, u * r, Math.sqrt(1 - u * u) * Math.sin(a) * r], i * 3);
    const b = 0.35 + 0.65 * rnd() ** 3, warm = rnd();
    sc.set([b * (0.85 + 0.15 * warm), b * 0.92, b * (1.05 - 0.2 * warm)], i * 3);
  }
  const sg = new THREE.BufferGeometry();
  sg.setAttribute("position", new THREE.BufferAttribute(sp, 3)); sg.setAttribute("color", new THREE.BufferAttribute(sc, 3));
  scene.add(new THREE.Points(sg, new THREE.PointsMaterial({ size: 1.4, sizeAttenuation: false, vertexColors: true })));

  const sats = buildSatellites(); scene.add(sats.points);
  return { scene, camera, earth, globe, clouds, sun, sunDir, sats };
}

// Thousands of tracked objects in recognizable shells, moved on the GPU by mission time.
function buildSatellites() {
  const shells = [   // count, altitude (km), inclination (deg; <0 random), color
    [1100, 550, 53, 0xe9f1f7], [380, 570, 70, 0x9ad7ff], [420, 560, 97.6, 0xffb547], [300, 1200, 87.9, 0xa98bff],
    [250, 780, 86.4, 0x2fe39a], [520, -1, -1, 0xff6b6b], [150, 35786, 0.05, 0x5aa2ff], [110, 20200, 55, 0xffe17a],
  ];
  const n = shells.reduce((s, x) => s + x[0], 0);
  const A = { aR: 1, aInc: 1, aRaan: 1, aPh: 1, aN: 1, aCol: 3 };
  const buf = Object.fromEntries(Object.entries(A).map(([k, d]) => [k, new Float32Array(n * d)]));
  seed = 21;
  let i = 0;
  for (const [count, alt, inc, col] of shells) {
    const c = new THREE.Color(col), planes = Math.max(6, Math.round(Math.sqrt(count) * 1.4));
    for (let k = 0; k < count; k++, i++) {
      const h = (alt < 0 ? 350 + rnd() * 1300 : alt + (rnd() - 0.5) * 8) * 1000, r = 1 + h / RE;
      buf.aR[i] = r;
      buf.aInc[i] = (inc < 0 ? rnd() * 100 : inc) * DEG;
      buf.aRaan[i] = alt < 0 ? rnd() * Math.PI * 2 : (Math.floor(rnd() * planes) / planes) * Math.PI * 2;
      buf.aPh[i] = rnd() * Math.PI * 2;
      buf.aN[i] = Math.sqrt(MU / (r * RE) ** 3);
      buf.aCol.set([c.r, c.g, c.b], i * 3);
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(n * 3), 3));
  for (const [k, d] of Object.entries(A)) g.setAttribute(k, new THREE.BufferAttribute(buf[k], d));
  const mat = new THREE.ShaderMaterial({
    uniforms: { uTime: { value: 0 }, uSize: { value: 2.6 * Math.min(devicePixelRatio, 2) } },
    vertexShader: `attribute float aR; attribute float aInc; attribute float aRaan; attribute float aPh; attribute float aN;
      attribute vec3 aCol; uniform float uTime; uniform float uSize; varying vec3 vCol;
      void main() {
        float u = aPh + aN * uTime;
        vec3 p = vec3(cos(u), sin(u) * cos(aInc), sin(u) * sin(aInc));
        p = vec3(p.x * cos(aRaan) - p.y * sin(aRaan), p.x * sin(aRaan) + p.y * cos(aRaan), p.z);
        vec4 mv = modelViewMatrix * vec4(vec3(p.x, p.z, -p.y) * aR, 1.0);
        gl_Position = projectionMatrix * mv; gl_PointSize = uSize; vCol = aCol; }`,
    fragmentShader: `varying vec3 vCol; void main() { float d = length(gl_PointCoord - 0.5);
      if (d > 0.5) discard; gl_FragColor = vec4(vCol, smoothstep(0.5, 0.15, d)); }`,
    transparent: true, depthWrite: false,
  });
  const points = new THREE.Points(g, mat); points.frustumCulled = false;
  return { points, mat, count: n };
}

function makeLine(n, color, opacity = 0.9, dashed = false) {
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(n * 3), 3));
  g.setDrawRange(0, 0);
  const m = dashed
    ? new THREE.LineDashedMaterial({ color, transparent: true, opacity, dashSize: 0.012, gapSize: 0.008, depthWrite: false })
    : new THREE.LineBasicMaterial({ color, transparent: true, opacity, depthWrite: false });
  const l = new THREE.Line(g, m); l.frustumCulled = false;
  return l;
}
function setLine(line, pts) {
  const a = line.geometry.attributes.position;
  if (pts.length > a.count) {
    line.geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(pts.length * 3), 3));
    return setLine(line, pts);
  }
  pts.forEach((p, i) => a.setXYZ(i, p.x, p.y, p.z));
  a.needsUpdate = true;
  line.geometry.setDrawRange(0, pts.length);
  line.geometry.computeBoundingSphere();
  if (line.material.isLineDashedMaterial) line.computeLineDistances();
}
function glowSprite(color, size) {
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTex, color, transparent: true, depthWrite: false,
    blending: THREE.AdditiveBlending, sizeAttenuation: false }));
  s.scale.setScalar(size);
  return s;
}

// ---------------------------------------------------------------- hologram
function buildHolo(api, lander) {
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(32, 1, 0.5, 200);
  camera.position.set(0, 4, 36); camera.lookAt(0, -1.2, 0);
  const uniforms = { uTime: { value: 0 }, uColor: { value: new THREE.Color(0x5ee6ff) } };
  const holoMat = new THREE.ShaderMaterial({
    uniforms,
    vertexShader: `varying vec3 vN; varying vec3 vV; varying float vY;
      void main() { vec4 w = modelMatrix * vec4(position, 1.0); vY = w.y; vN = normalize(mat3(modelMatrix) * normal);
        vV = normalize(cameraPosition - w.xyz); gl_Position = projectionMatrix * viewMatrix * w; }`,
    fragmentShader: `uniform float uTime; uniform vec3 uColor; varying vec3 vN; varying vec3 vV; varying float vY;
      void main() {
        float f = pow(1.0 - abs(dot(normalize(vN), vV)), 2.2);
        float lines = 0.65 + 0.35 * sin(vY * 9.0 - uTime * 5.0);
        float sweep = smoothstep(0.9, 0.0, abs(vY - (mod(uTime * 4.0, 22.0) - 11.0)));
        float flick = 0.92 + 0.08 * sin(uTime * 41.0) * sin(uTime * 7.0);
        float a = ((0.10 + 0.85 * f) * lines + 0.5 * sweep) * flick;
        gl_FragColor = vec4(uColor * (0.55 + f + sweep), a); }`,
    transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide,
  });
  const edgeMat = new THREE.LineBasicMaterial({ color: 0x8ff0ff, transparent: true, opacity: 0.32,
    blending: THREE.AdditiveBlending, depthWrite: false });
  const model = api.buildRocket({ length: lander.length, radius: lander.radius, leg_radius: lander.leg_radius ?? 2.5,
    leg_drop: lander.leg_drop ?? 1, max_gimbal: lander.max_gimbal ?? 0.12 });
  model.plume.visible = false; model.light.intensity = 0;
  api.setLegs(model, 1);
  const meshes = [];
  model.root.traverse(o => { if (o.isMesh) meshes.push(o); });
  for (const m of meshes) {
    m.material = holoMat; m.castShadow = false;
    m.add(new THREE.LineSegments(new THREE.EdgesGeometry(m.geometry, 28), edgeMat));
  }
  scene.add(model.root);
  // projector pedestal
  const baseY = -lander.length / 2 - (lander.leg_drop ?? 1) - 0.3;
  const ringMat = new THREE.MeshBasicMaterial({ color: 0x5ee6ff, transparent: true, opacity: 0.55, side: THREE.DoubleSide,
    blending: THREE.AdditiveBlending, depthWrite: false });
  const rings = [3.6, 4.4, 5.6].map((r, i) => {
    const m = new THREE.Mesh(new THREE.RingGeometry(r - (i === 1 ? 0.5 : 0.08), r, 96, 1, 0, i === 1 ? Math.PI * 1.4 : Math.PI * 2), ringMat);
    m.rotation.x = -Math.PI / 2; m.position.y = baseY; scene.add(m); return m;
  });
  const beam = new THREE.Mesh(new THREE.CylinderGeometry(4.2, 5.2, lander.length + 3, 48, 1, true), new THREE.ShaderMaterial({
    uniforms,
    vertexShader: `varying float vH; void main() { vH = uv.y; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
    fragmentShader: `uniform vec3 uColor; varying float vH; void main() { gl_FragColor = vec4(uColor, 0.10 * pow(1.0 - vH, 2.0)); }`,
    transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide,
  }));
  beam.position.y = baseY + (lander.length + 3) / 2;
  scene.add(beam);
  return {
    scene, camera,
    update(dt, t) {
      uniforms.uTime.value = t;
      model.root.rotation.y += dt * 0.45;
      rings[1].rotation.z -= dt * 0.8;
    },
  };
}

// ---------------------------------------------------------------- lift-off vignette (cinematic pad cameras)
// A photoreal pad at golden hour: physical sky with image-based light and sun shadows, a Falcon-class stack (sooted
// booster, nine engines, stowed legs and grid fins), LOX venting, deluge, steam and a bloom + film-grain camera
// pipeline. During the count the view cuts between close shots of the vehicle; after liftoff a long-lens tracker.
const TEXROOT = TEX.replace("planets/", "");
const V3 = (x, y, z) => new THREE.Vector3(x, y, z);
function beam(a, b, th) {
  const d = new THREE.Vector3().subVectors(b, a), len = d.length();
  const g = new THREE.BoxGeometry(th, len, th);
  g.applyQuaternion(new THREE.Quaternion().setFromUnitVectors(Y, d.normalize()));
  return g.translate((a.x + b.x) / 2, (a.y + b.y) / 2, (a.z + b.z) / 2);
}
// square lattice tower (legs, rings, alternating X-bracing), optionally with a deck every `decks` levels
function lattice(H, w, step, leg, brace, decks = 0) {
  const h = w / 2, geos = [], cs = [[h, h], [h, -h], [-h, -h], [-h, h]];
  for (const [x, z] of cs) geos.push(beam(V3(x, 0, z), V3(x, H, z), leg));
  for (let y = 0, n = 0; y < H - 0.1; y += step, n++) {
    const y2 = Math.min(y + step, H);
    for (let k = 0; k < 4; k++) {
      const [x1, z1] = cs[k], [x2, z2] = cs[(k + 1) % 4];
      geos.push(beam(V3(x1, y2, z1), V3(x2, y2, z2), brace));
      geos.push(n % 2 ? beam(V3(x1, y, z1), V3(x2, y2, z2), brace * 0.7) : beam(V3(x2, y, z2), V3(x1, y2, z1), brace * 0.7));
    }
    if (decks && n % decks === decks - 1) geos.push(new THREE.BoxGeometry(w + 1.4, 0.25, w + 1.4).translate(0, y2, 0));
  }
  return geos;
}
const PLUME_VS = `varying vec2 vUv; varying vec3 vN; varying vec3 vV;
  void main() { vUv = uv; vec4 w = modelMatrix * vec4(position, 1.0); vN = normalize(mat3(modelMatrix) * normal);
    vV = normalize(cameraPosition - w.xyz); gl_Position = projectionMatrix * viewMatrix * w; }`;
const PLUME_FS = `uniform float uTime, uThr, uLen, uGreen, uGain, uSeed, uSoft; uniform vec3 uCore, uEdge;
  varying vec2 vUv; varying vec3 vN; varying vec3 vV;
  float hash(float n) { return fract(sin(n) * 43758.5453); }
  float noise(float x) { float i = floor(x), f = fract(x); return mix(hash(i + uSeed), hash(i + 1.0 + uSeed), f * f * (3.0 - 2.0 * f)); }
  void main() {
    float s = 1.0 - vUv.y, z = s * uLen;                       // 0 at the nozzle exit, 1 at the tail
    float face = pow(abs(dot(normalize(vN), normalize(vV))), uSoft);
    float flick = 0.75 + 0.25 * sin(vUv.x * 43.98 + z * 0.9 - uTime * 23.0) + 0.5 * (noise(z * 0.5 - uTime * 41.0) - 0.5);
    float shock = 1.0 + 0.8 * pow(0.5 + 0.5 * cos(z * 3.6), 10.0) * (1.0 - smoothstep(0.0, 0.3, s));
    float fade = 1.0 - smoothstep(0.3, 1.0, s);
    vec3 col = mix(uCore, uEdge, smoothstep(0.0, 0.6, s));
    col = mix(col, vec3(0.25, 1.0, 0.35), uGreen);            // TEA-TEB ignition flash
    gl_FragColor = vec4(col * uGain * face * flick * shock * fade * uThr, 1.0);
  }`;
const LensShader = {   // after tone mapping: chromatic fringe, vignette, film grain
  uniforms: { tDiffuse: { value: null }, uTime: { value: 0 }, uGrain: { value: 0.035 }, uVig: { value: 0.45 }, uCA: { value: 0.006 } },
  vertexShader: `varying vec2 vUv; void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
  fragmentShader: `uniform sampler2D tDiffuse; uniform float uTime, uGrain, uVig, uCA; varying vec2 vUv;
    float h(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
    void main() {
      vec2 d = vUv - 0.5; float r2 = dot(d, d); vec2 o = d * uCA * r2 * 4.0;
      vec3 c = vec3(texture2D(tDiffuse, vUv + o).r, texture2D(tDiffuse, vUv).g, texture2D(tDiffuse, vUv - o).b);
      c *= 1.0 - uVig * smoothstep(0.08, 0.5, r2);
      c += (h(vUv * 917.0 + fract(uTime * 7.31)) - 0.5) * uGrain;
      gl_FragColor = vec4(c, 1.0);
    }`,
};

// the stack: origin at the base of the booster's engine section, +y up
function buildStack(anis) {
  const root = new THREE.Group(), R1 = 1.83, segs = 72;
  const add = (geo, mat, parent = root) => { const m = new THREE.Mesh(geo, mat); m.castShadow = m.receiveShadow = true; parent.add(m); return m; };
  const tex = (w, h, draw) => { const t = canvasTex(w, h, draw); t.anisotropy = anis; return t; };
  const welds = (x, w, h, n, a = 0.32) => { x.fillStyle = `rgba(110,110,104,${a})`; for (let i = 1; i < n; i++) x.fillRect(0, h * i / n, w, 3); };

  // booster tank: white paint under a reflown booster's soot, heaviest at the base and streaked upward
  const boosterMap = tex(1024, 4096, (x, w, h) => {
    x.fillStyle = "#ecebe6"; x.fillRect(0, 0, w, h);
    for (let i = 0; i < 13; i++) { x.fillStyle = `rgba(0,0,0,${0.01 + rnd() * 0.025})`; x.fillRect(0, h * i / 13, w, h / 13); }
    welds(x, w, h, 13);
    x.fillStyle = "rgba(120,120,115,.25)"; for (const u of [0.07, 0.32, 0.57, 0.82]) x.fillRect(u * w, 0, 2, h);
    x.save(); x.translate(0.886 * w, h * 0.36); x.rotate(Math.PI / 2);
    x.font = "800 150px Helvetica, Arial, sans-serif"; x.textAlign = "center"; x.fillStyle = "#1c2230"; x.fillText("ROCKETSIM", 0, 52); x.restore();
    const g = x.createLinearGradient(0, h, 0, h * 0.2);
    g.addColorStop(0, "rgba(36,28,22,.94)"); g.addColorStop(0.12, "rgba(48,38,30,.72)"); g.addColorStop(0.4, "rgba(70,60,50,.22)");
    g.addColorStop(1, "rgba(80,70,60,0)");
    x.fillStyle = g; x.fillRect(0, 0, w, h);
    for (let i = 0; i < 1100; i++) {
      const sx = rnd() * w, len = h * (0.04 + rnd() ** 2 * 0.62), sw = 1 + rnd() * 9;
      const s = x.createLinearGradient(0, h, 0, h - len);
      s.addColorStop(0, `rgba(28,22,18,${0.08 + rnd() * 0.32})`); s.addColorStop(1, "rgba(28,22,18,0)");
      x.fillStyle = s; x.fillRect(sx, h - len, sw, len);
    }
  });
  const paint = map => new THREE.MeshStandardMaterial({ map, roughness: 0.42, metalness: 0.02 });
  add(new THREE.CylinderGeometry(R1, R1, 39.5, segs, 1, true).translate(0, 1 + 39.5 / 2, 0), paint(boosterMap));
  // interstage (carbon composite), second stage, adapter, fairing
  const carbon = new THREE.MeshStandardMaterial({ color: 0x17181a, roughness: 0.55, metalness: 0.15,
    map: tex(512, 512, (x, w) => { x.fillStyle = "#fff"; x.fillRect(0, 0, w, w);
      for (let i = 0; i < 9000; i++) { const g = 200 + rnd() * 55 | 0; x.fillStyle = `rgb(${g},${g},${g})`; x.fillRect(rnd() * w, rnd() * w, 2, 2); } }) });
  add(new THREE.CylinderGeometry(R1 * 1.004, R1 * 1.004, 6.5, segs, 1, true).translate(0, 40.5 + 3.25, 0), carbon);
  const s2Map = tex(1024, 1024, (x, w, h) => {
    x.fillStyle = "#efeeea"; x.fillRect(0, 0, w, h); welds(x, w, h, 4);
    x.fillStyle = "rgba(120,120,115,.25)"; x.fillRect(0.82 * w, 0, 2, h);
    x.save(); x.translate(0.886 * w, h * 0.5); x.rotate(Math.PI / 2);
    x.font = "700 64px Helvetica, Arial, sans-serif"; x.textAlign = "center"; x.fillStyle = "#1c2230"; x.fillText("LANDER-01", 0, 22); x.restore();
  });
  add(new THREE.CylinderGeometry(R1, R1, 11.5, segs, 1, true).translate(0, 47 + 5.75, 0), paint(s2Map));
  const white = new THREE.MeshStandardMaterial({ color: 0xefeeea, roughness: 0.4 });
  add(new THREE.CylinderGeometry(2.6, R1, 0.8, segs, 1, true).translate(0, 58.9, 0), white);
  const fairPts = [new THREE.Vector2(2.6, 0), new THREE.Vector2(2.6, 6.7)];
  for (let i = 1; i <= 20; i++) { const f = i / 20; fairPts.push(new THREE.Vector2(Math.max(2.6 * Math.pow(1 - f * f, 0.55), 0.02), 6.7 + 6.4 * f)); }
  const fairMap = tex(1024, 1024, (x, w, h) => {
    x.fillStyle = "#f1f0ec"; x.fillRect(0, 0, w, h);
    x.fillStyle = "rgba(90,90,86,.55)"; x.fillRect(0.25 * w - 1, 0, 3, h); x.fillRect(0.75 * w - 1, 0, 3, h);
    x.fillStyle = "rgba(110,110,104,.3)"; x.fillRect(0, h * (1 - 1 / 21), w, 3);
  });
  add(new THREE.LatheGeometry(fairPts, segs).translate(0, 59.3, 0), paint(fairMap));
  // engine section and heat shield
  const dark = new THREE.MeshStandardMaterial({ color: 0x2c2d2f, roughness: 0.65, metalness: 0.5 });
  add(new THREE.CylinderGeometry(R1, R1 * 0.985, 1, segs).translate(0, 0.5, 0), dark);
  // raceway along the tower side, stowed carbon legs, folded grid fins
  add(new THREE.BoxGeometry(0.34, 38.5, 0.18).translate(0, 1.6 + 38.5 / 2, R1 + 0.06).rotateY(-Math.PI / 2), white);
  const legShape = new THREE.Shape([new THREE.Vector2(-0.4, 0), new THREE.Vector2(0.4, 0), new THREE.Vector2(0.22, 9.6), new THREE.Vector2(-0.22, 9.6)]);
  const legGeo = new THREE.ExtrudeGeometry(legShape, { depth: 0.26, bevelEnabled: false }).translate(0, 0.35, R1);
  const legMat = new THREE.MeshStandardMaterial({ color: 0x151618, roughness: 0.45, metalness: 0.25 });
  for (let k = 0; k < 4; k++) {
    const a = Math.PI / 4 + k * Math.PI / 2;
    add(legGeo.clone().rotateY(a), legMat);
    add(new THREE.BoxGeometry(0.95, 0.3, 0.5).translate(0, 0.2, R1 + 0.25).rotateY(a), dark);
  }
  const finMap = canvasTex(256, 256, (x, w) => {
    x.fillStyle = "#000"; x.fillRect(0, 0, w, w); x.strokeStyle = "#fff"; x.lineWidth = 16; x.strokeRect(8, 8, w - 16, w - 16);
    x.lineWidth = 4;
    for (let i = -w; i < 2 * w; i += 24) { x.beginPath(); x.moveTo(i, 0); x.lineTo(i + w, w); x.moveTo(i, w); x.lineTo(i + w, 0); x.stroke(); }
  });
  finMap.colorSpace = THREE.NoColorSpace;
  const finMat = new THREE.MeshStandardMaterial({ color: 0x5a5d61, roughness: 0.42, metalness: 0.85, alphaMap: finMap,
    alphaTest: 0.5, side: THREE.DoubleSide });
  for (let k = 0; k < 4; k++) {
    const a = k * Math.PI / 2;
    add(new THREE.PlaneGeometry(1.25, 1.5).translate(0, 45.4, R1 + 0.2).rotateY(a), finMat);
    add(new THREE.BoxGeometry(0.5, 0.35, 0.3).translate(0, 44.5, R1 + 0.12).rotateY(a), dark);
  }
  // nine sea-level engines; their nozzles glow when firing
  const bellPts = [[0.46, -1.08], [0.43, -0.9], [0.36, -0.62], [0.27, -0.38], [0.18, -0.24], [0.2, -0.12], [0.27, -0.04], [0.27, 0]]
    .map(([r, y]) => new THREE.Vector2(r, y));
  const bellGeo = new THREE.LatheGeometry(bellPts, 36);
  const engineMat = new THREE.MeshStandardMaterial({ color: 0x5d544b, roughness: 0.3, metalness: 0.9, side: THREE.DoubleSide,
    emissive: 0xff5a18, emissiveIntensity: 0 });
  const enginePos = [[0, 0]];
  for (let k = 0; k < 8; k++) enginePos.push([1.22 * Math.sin(k * Math.PI / 4), 1.22 * Math.cos(k * Math.PI / 4)]);
  for (const [x, z] of enginePos) add(bellGeo, engineMat).position.set(x, 0, z);

  // plume: nine jets merging into one sea-level column, a hot core, and the light they throw
  const flame = new THREE.Group(); flame.position.y = -1.08; root.add(flame);
  const plumeMat = (core, edge, gain, soft, seed) => new THREE.ShaderMaterial({
    uniforms: { uTime: { value: 0 }, uThr: { value: 0 }, uLen: { value: 1 }, uGreen: { value: 0 }, uGain: { value: gain },
      uSeed: { value: seed }, uSoft: { value: soft }, uCore: { value: new THREE.Color(...core) }, uEdge: { value: new THREE.Color(...edge) } },
    vertexShader: PLUME_VS, fragmentShader: PLUME_FS, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    side: THREE.DoubleSide });
  const cone = (rt, rb, mat) => { const m = new THREE.Mesh(new THREE.CylinderGeometry(rt, rb, 1, 48, 1, true).translate(0, -0.5, 0), mat);
    m.frustumCulled = false; m.renderOrder = 2; flame.add(m); return m; };
  const outer = cone(1.9, 4.4, plumeMat([1, 0.66, 0.28], [0.95, 0.34, 0.08], 7, 1.1, 1.7));
  const core = cone(1.6, 1.2, plumeMat([1, 0.9, 0.68], [1, 0.6, 0.22], 9, 2.2, 7.3));
  const jetMat = plumeMat([1, 0.97, 0.9], [1, 0.7, 0.35], 7, 1.8, 3.1);
  const jets = enginePos.map(([x, z]) => { const j = cone(0.44, 0.7, jetMat); j.position.set(x, 0, z); return j; });
  const fireLight = new THREE.PointLight(0xffa04a, 0, 0, 2); fireLight.position.y = -6; flame.add(fireLight);
  return { root, flame, outer, core, jets, engineMat, fireLight };
}

// soft, top-lit cumulus puffs (a few variants so the cloud doesn't tile)
function puffTex() {
  return canvasTex(256, 256, (x, w) => {
    for (let i = 0; i < 16; i++) {
      const cx = w / 2 + (rnd() - 0.5) * w * 0.38, cy = w / 2 + (rnd() - 0.5) * w * 0.38, r = w * (0.1 + rnd() * 0.17);
      const g = x.createRadialGradient(cx, cy, 0, cx, cy, r);
      g.addColorStop(0, "rgba(255,255,255,.55)"); g.addColorStop(0.6, "rgba(255,255,255,.25)"); g.addColorStop(1, "rgba(255,255,255,0)");
      x.fillStyle = g; x.fillRect(0, 0, w, w);
    }
    x.globalCompositeOperation = "destination-in";   // round the silhouette off
    const m = x.createRadialGradient(w / 2, w / 2, w * 0.2, w / 2, w / 2, w / 2);
    m.addColorStop(0, "rgba(0,0,0,1)"); m.addColorStop(1, "rgba(0,0,0,0)");
    x.fillStyle = m; x.fillRect(0, 0, w, w);
    x.globalCompositeOperation = "source-atop";      // shade the underside
    const s = x.createLinearGradient(0, 0, 0, w);
    s.addColorStop(0, "rgba(255,255,255,0)"); s.addColorStop(1, "rgba(105,108,116,.55)");
    x.fillStyle = s; x.fillRect(0, 0, w, w);
  });
}

function buildPad(api) {
  const R = api.renderer, anis = R.capabilities.getMaxAnisotropy();
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(30, 1, 1, 40000);
  const DECK = 0, GROUND = -10, MOUNT = 7, BASE = MOUNT + 1, WIND = 2.5;   // m; the pad deck sits on a berm above the scrub
  const mesh = (geo, mat, cast = true, recv = true) => { const m = new THREE.Mesh(geo, mat); m.castShadow = cast; m.receiveShadow = recv; scene.add(m); return m; };
  const std = (color, roughness = 0.8, metalness = 0, extra = {}) => new THREE.MeshStandardMaterial({ color, roughness, metalness, ...extra });
  const repeatTex = (t, n, srgb = true) => { t.wrapS = t.wrapT = THREE.RepeatWrapping; t.repeat.set(n, n); t.anisotropy = anis;
    if (srgb) t.colorSpace = THREE.SRGBColorSpace; return t; };

  // sky, low morning sun over the ocean, image-based light from the same sky
  const sunDir = new THREE.Vector3().setFromSphericalCoords(1, (90 - 19) * DEG, 58 * DEG);
  const sky = new Sky(); sky.scale.setScalar(30000);
  const su = sky.material.uniforms;
  su.turbidity.value = 2.8; su.rayleigh.value = 1.8; su.mieCoefficient.value = 0.0035; su.mieDirectionalG.value = 0.82;
  su.sunPosition.value.copy(sunDir);
  const envScene = new THREE.Scene(); envScene.add(sky);
  const pmrem = new THREE.PMREMGenerator(R);
  scene.environment = pmrem.fromScene(envScene).texture; pmrem.dispose();
  scene.add(sky);
  scene.fog = new THREE.FogExp2(0xb9c5cf, 0.000035);
  const sun = new THREE.DirectionalLight(0xffe0bc, 2.6);
  sun.position.copy(sunDir).multiplyScalar(500); sun.castShadow = true;
  Object.assign(sun.shadow.camera, { left: -170, right: 170, top: 170, bottom: -170, near: 50, far: 1200 });
  sun.shadow.camera.updateProjectionMatrix();
  sun.shadow.mapSize.set(4096, 4096); sun.shadow.bias = -0.0004; sun.shadow.normalBias = 0.05;
  scene.add(sun, sun.target, new THREE.HemisphereLight(0xbcd2ff, 0x5a5040, 0.25));

  // land, beach, sea
  const loader = new THREE.TextureLoader();
  const grass = repeatTex(loader.load(TEXROOT + "terrain/grasslight-big.jpg"), 900);
  const scrub = std(0x7c7656, 1, 0, { map: grass });
  mesh(new THREE.PlaneGeometry(26000, 30000).rotateX(-Math.PI / 2).translate(1600 - 13000, GROUND, 0), scrub, false);
  mesh(new THREE.PlaneGeometry(230, 30000).rotateX(-Math.PI / 2).translate(1700, GROUND + 0.05, 0), std(0xd9cba6, 1), false);
  const waterN = repeatTex(loader.load(TEXROOT + "waternormals.jpg"), 260, false);
  mesh(new THREE.PlaneGeometry(30000, 30000).rotateX(-Math.PI / 2).translate(1790 + 15000, GROUND - 0.4, 0),
    std(0x0c2533, 0.1, 0, { normalMap: waterN, normalScale: new THREE.Vector2(0.4, 0.4) }), false);
  // bushes and palmetto scrub
  const bush = new THREE.InstancedMesh(new THREE.IcosahedronGeometry(1, 2), std(0x2c3520, 1), 5200);
  const dummy = new THREE.Object3D();
  for (let i = 0; i < bush.count; i++) {
    const a = rnd() * Math.PI * 2, r = 165 + rnd() ** 0.7 * 1400;
    const x = Math.cos(a) * r, z = Math.sin(a) * r;
    if (x > 1580 || Math.abs(z) < 7 || Math.abs(x + 4) < 6 || Math.hypot(x + 330, z - 310) < 70 || Math.hypot(x + 210, z - 230) < 45) { dummy.scale.setScalar(0); dummy.updateMatrix(); bush.setMatrixAt(i, dummy.matrix); continue; }
    const s = 0.8 + rnd() * 1.6;
    dummy.position.set(x, GROUND + s * 0.35, z); dummy.scale.set(s * (0.8 + rnd() * 0.5), s * (0.5 + rnd() * 0.5), s); dummy.rotation.y = rnd() * 6;
    dummy.updateMatrix(); bush.setMatrixAt(i, dummy.matrix);
  }
  bush.receiveShadow = true; scene.add(bush);
  const asphalt = std(0x2e2e2c, 0.95);
  mesh(new THREE.PlaneGeometry(2600, 9).rotateX(-Math.PI / 2).translate(-150 - 1300, GROUND + 0.04, 0), asphalt, false);
  mesh(new THREE.PlaneGeometry(9, 2600).rotateX(-Math.PI / 2).translate(-4, GROUND + 0.04, 150 + 1300), asphalt, false);

  // pad deck: jointed, stained concrete on its berm, scorched around the mount, flame trench running +z
  const concreteMap = tex => canvasTex(1024, 1024, (x, w) => {
    x.fillStyle = tex; x.fillRect(0, 0, w, w);
    for (let i = 0; i < 26000; i++) { const g = 120 + rnd() * 70 | 0; x.fillStyle = `rgba(${g},${g - 4},${g - 10},${0.12 + rnd() * 0.2})`;
      x.fillRect(rnd() * w, rnd() * w, 1 + rnd() * 3, 1 + rnd() * 3); }
    for (let i = 0; i < 40; i++) { const cx = rnd() * w, cy = rnd() * w, r = 30 + rnd() * 140, g = x.createRadialGradient(cx, cy, 0, cx, cy, r);
      g.addColorStop(0, `rgba(60,55,48,${0.06 + rnd() * 0.1})`); g.addColorStop(1, "rgba(60,55,48,0)"); x.fillStyle = g; x.fillRect(0, 0, w, w); }
    x.fillStyle = "rgba(58,56,52,.6)"; for (let k = 0; k <= 4; k++) { x.fillRect(k * w / 4 - 1, 0, 3, w); x.fillRect(0, k * w / 4 - 1, w, 3); }
  });
  const deckMap = repeatTex(concreteMap("#9d988f"), 11);
  mesh(new THREE.CircleGeometry(110, 96).rotateX(-Math.PI / 2), std(0xffffff, 0.92, 0, { map: deckMap }), false);
  const slope = repeatTex(loader.load(TEXROOT + "terrain/grasslight-big.jpg"), 1); slope.repeat.set(90, 1);
  mesh(new THREE.CylinderGeometry(110, 150, 10, 96, 1, true).translate(0, -5, 0), std(0x7c7656, 1, 0, { map: slope }), false);
  const scorch = canvasTex(512, 512, (x, w) => { const g = x.createRadialGradient(w / 2, w / 2, 0, w / 2, w / 2, w / 2);
    g.addColorStop(0, "rgba(22,18,16,.9)"); g.addColorStop(0.45, "rgba(30,26,22,.55)"); g.addColorStop(1, "rgba(30,26,22,0)");
    x.fillStyle = g; x.fillRect(0, 0, w, w); });
  mesh(new THREE.CircleGeometry(34, 64).rotateX(-Math.PI / 2).translate(0, 0.03, 0), std(0xffffff, 1, 0, { map: scorch, transparent: true,
    depthWrite: false, polygonOffset: true, polygonOffsetFactor: -2 }), false);
  mesh(new THREE.BoxGeometry(10, 0.1, 100).translate(0, 0.05, 58), std(0x1a1918, 1), false);
  const curb = std(0x8a867e, 0.9, 0, { map: repeatTex(concreteMap("#8a867e"), 2) });
  for (const s of [-1, 1]) mesh(new THREE.BoxGeometry(1, 1.5, 100).translate(5.5 * s, 0.75, 58), curb);
  // launch mount: a concrete table with an opening over the trench, steel hold-down clamps
  const mountMat = std(0xffffff, 0.9, 0, { map: repeatTex(concreteMap("#8f8a82"), 1) });
  for (const s of [-1, 1]) {
    mesh(new THREE.BoxGeometry(4.5, MOUNT, 18).translate(6.75 * s, MOUNT / 2, 0), mountMat);
  }
  mesh(new THREE.BoxGeometry(9, MOUNT, 4.5).translate(0, MOUNT / 2, -6.75), mountMat);   // open toward the trench
  const steel = std(0x4a4e53, 0.5, 0.8);
  const clamps = [0, 1, 2, 3].map(k => {
    const m = mesh(new THREE.BoxGeometry(0.55, 1.3, 1.1).translate(0, MOUNT + 0.65, 0), steel); m.userData.a = Math.PI / 4 + k * Math.PI / 2; return m;
  });
  // transporter-erector strongback (pivoted back a few degrees), fixed service structure with swung-away crew arm
  const towerMat = std(0x8e9399, 0.55, 0.65);
  const te = new THREE.Group(); te.position.set(-5.3, MOUNT, 0); scene.add(te);
  const teMesh = new THREE.Mesh(mergeGeometries(lattice(64, 3.2, 3.2, 0.42, 0.22)), std(0x9ea3a8, 0.5, 0.6));
  teMesh.castShadow = teMesh.receiveShadow = true; te.add(teMesh);
  const fss = new THREE.Mesh(mergeGeometries([...lattice(86, 8, 6, 0.7, 0.32, 2), new THREE.BoxGeometry(2.6, 86, 2.6).translate(0, 43, 0),
    new THREE.CylinderGeometry(0.22, 0.4, 22, 8).translate(0, 97, 0)]), towerMat);
  fss.position.set(-26, DECK, -6); fss.castShadow = fss.receiveShadow = true; scene.add(fss);
  const arm = new THREE.Group(); arm.position.set(-22, 66, -6); arm.rotation.y = 80 * DEG; scene.add(arm);
  arm.add(Object.assign(new THREE.Mesh(new THREE.BoxGeometry(15, 2.6, 2.4).translate(7.5, 0, 0), towerMat), { castShadow: true }),
    Object.assign(new THREE.Mesh(new THREE.BoxGeometry(3, 3.4, 3.4).translate(16, 0.2, 0), std(0xe6e6e2, 0.6)), { castShadow: true }));
  // lightning-protection masts with catenary wires
  const mastPos = [[78, 78], [-78, 78], [-78, -78], [78, -78]];
  const mastGeo = mergeGeometries([...lattice(118, 2.2, 4.4, 0.3, 0.14), new THREE.CylinderGeometry(0.12, 0.2, 14, 6).translate(0, 125, 0)]);
  for (const [x, z] of mastPos) { const m = mesh(mastGeo, towerMat, true, false); m.position.set(x, DECK, z); }
  const wireMat = new THREE.LineBasicMaterial({ color: 0x2d3034, transparent: true, opacity: 0.8 });
  const wire = (a, b, sag) => { const pts = []; for (let i = 0; i <= 32; i++) { const f = i / 32; pts.push(a.clone().lerp(b, f).add(V3(0, -sag * 4 * f * (1 - f), 0))); }
    scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), wireMat)); };
  for (let k = 0; k < 4; k++) {
    const [x1, z1] = mastPos[k], [x2, z2] = mastPos[(k + 1) % 4];
    wire(V3(x1, 132, z1), V3(x2, 132, z2), 18);
    wire(V3(x1, 132, z1), V3(x1 * 1.7, DECK + 1, z1 * 1.7), 6);
  }
  // water tower, propellant farm, the assembly building on the horizon, scattered hangars
  const tankWhite = std(0xe9e8e3, 0.55, 0.1);
  const wt = new THREE.Mesh(mergeGeometries([...[[-1, -1], [-1, 1], [1, -1], [1, 1]].map(([a, b]) =>
    beam(V3(7 * a, 0, 7 * b), V3(3 * a, 74, 3 * b), 1.1)), new THREE.SphereGeometry(11, 32, 20).translate(0, 82, 0)]), tankWhite);
  wt.position.set(-230, GROUND, -190); wt.castShadow = true; scene.add(wt);
  mesh(new THREE.SphereGeometry(9, 32, 20).translate(132, GROUND + 12, -96), tankWhite);
  for (let k = 0; k < 3; k++) mesh(new THREE.CylinderGeometry(3, 3, 26, 28).rotateZ(Math.PI / 2).translate(145, GROUND + 3.4, 64 + k * 8), tankWhite);
  mesh(new THREE.BoxGeometry(160, 160, 200).translate(-3400, GROUND + 80, -2900), std(0xc2c2bc, 0.85), false, false);
  mesh(new THREE.BoxGeometry(1, 140, 34).translate(-3319.5, GROUND + 70, -2900), std(0x5b5f63, 0.8), false, false);
  for (let i = 0; i < 26; i++) {
    const a = -0.4 - rnd() * 2.6, r = 500 + rnd() * 2600, w = 20 + rnd() * 60, h = 8 + rnd() * 22;
    mesh(new THREE.BoxGeometry(w, h, w * (0.6 + rnd())).translate(Math.cos(a) * r, GROUND + h / 2, Math.sin(a) * r),
      std(rnd() < 0.5 ? 0xd7d5cf : 0xa9aaa6, 0.85), false, false);
  }

  // the vehicle
  const stack = buildStack(anis); scene.add(stack.root);
  const groundLight = new THREE.PointLight(0xff9a40, 0, 0, 2); groundLight.position.set(0, 2, 0); scene.add(groundLight);
  const baseGlow = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTex, transparent: true, depthWrite: false,
    blending: THREE.AdditiveBlending })); baseGlow.position.set(0, 1.5, 0); baseGlow.scale.setScalar(16); scene.add(baseGlow);

  // smoke, steam and LOX vapour: a pool of lit billboards
  const ptex = [puffTex(), puffTex(), puffTex(), puffTex()];
  const puffs = Array.from({ length: 760 }, (_, i) => {
    const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: ptex[i % 4], transparent: true, depthWrite: false, fog: true }));
    s.visible = false; s.renderOrder = 1; scene.add(s);
    return { s, age: 0, life: 1, v: new THREE.Vector3(), size: 1, grow: 1, op: 1, buoy: 0, drag: 0.5, shade: 1 };
  });
  let next = 0;
  const emit = (pos, v, o) => {
    const p = puffs[next++ % puffs.length];
    p.s.position.copy(pos); p.v.copy(v); p.age = 0; Object.assign(p, o);
    p.s.material.rotation = (rnd() - 0.5) * 0.7; p.s.visible = true;
  };
  const acc = {};
  const spawn = (key, rate, dt, fn) => { acc[key] = (acc[key] ?? 0) + rate * dt; for (; acc[key] >= 1; acc[key]--) fn(); };
  // sound-suppression deluge: water thrown up over the mount from a ring of nozzles
  const ND = 2600, dPos = new Float32Array(ND * 3).fill(-1e5), dVel = new Float32Array(ND * 3);
  const dGeo = new THREE.BufferGeometry(); dGeo.setAttribute("position", new THREE.BufferAttribute(dPos, 3));
  const deluge = new THREE.Points(dGeo, new THREE.PointsMaterial({ map: glowTex, size: 0.09, transparent: true, opacity: 0.7,
    depthWrite: false, fog: true }));
  deluge.material.color.setRGB(1.7, 1.75, 1.8); deluge.frustumCulled = false; scene.add(deluge);
  let dNext = 0;

  // post: HDR render with MSAA, bloom on the flame, ACES in the output pass, then lens/grain
  const composer = new EffectComposer(R, new THREE.WebGLRenderTarget(1, 1, { type: THREE.HalfFloatType, samples: 4 }));
  composer.addPass(new RenderPass(scene, camera));
  const bloom = new UnrealBloomPass(new THREE.Vector2(256, 256), 0.5, 0.55, 3.5);
  composer.addPass(bloom);
  composer.addPass(new OutputPass());
  const lens = new ShaderPass(LensShader); composer.addPass(lens);
  const size = { w: 0, h: 0, pr: 0 };

  let lastT = null, shotNow = -1, expo = 0.5, trailAt = null;
  const lag = new THREE.Vector3(), tmp = new THREE.Vector3(), down = new THREE.Vector3();
  const SM = 2.3;   // smoke brightness, matched to the sunlit white paint
  return {
    scene, camera,
    update(T, frames, F) {
      let dt = lastT == null ? 0 : T - lastT;
      if (dt < 0 || dt > 1) {   // a new launch (or a jump): clear the air
        for (const p of puffs) p.s.visible = false;
        dPos.fill(-1e5); dt = 0; trailAt = null;
      }
      dt = Math.min(dt, 0.1); lastT = T;
      const s = T <= 0 ? null : sampleRows(frames, T, [F.stage]);
      const alt = s ? s[F.alt] : 0, tilt = s ? clamp(90 - s[F.pitch], 0, 60) * DEG : 0;
      stack.root.position.set(s ? s[F.downrange] * 1000 : 0, BASE + alt, 0);
      stack.root.rotation.set(0, 0, -tilt);
      stack.root.updateMatrixWorld();
      const nozzle = V3(0, -1.08, 0).applyMatrix4(stack.root.matrixWorld);
      down.set(0, -1, 0).applyQuaternion(stack.root.quaternion);

      // engines: ignition at T−2.6 with the green TEA-TEB flash, full thrust by the release at T−0
      const thr = T > -2.6 ? (T < 0 ? smooth((T + 2.6) / 1.2) : 1) : 0;
      const green = T > -2.6 && T < -2.15 ? 1 - smooth((T + 2.6) / 0.45) : 0;
      const vis = Math.max(thr, green * 0.5), fl = 0.92 + 0.08 * Math.sin(T * 61) * Math.sin(T * 17);
      const len = (40 + 42 * smooth(alt / 900)) * fl, spread = 1 + 0.9 * smooth(alt / 5000);
      stack.flame.visible = vis > 0.005;
      stack.outer.scale.set(spread, len * smooth(vis * 1.4), spread);
      stack.core.scale.set(1, 17 * fl * smooth(vis * 1.6), 1);
      for (const j of stack.jets) j.scale.set(1, 6 * fl, 1);
      for (const [m, l] of [[stack.outer, len], [stack.core, 17], [stack.jets[0], 6]]) {
        const u = m.material.uniforms; u.uTime.value = T; u.uThr.value = vis; u.uLen.value = l; u.uGreen.value = green;
      }
      stack.engineMat.emissiveIntensity = 0.7 * thr;
      stack.fireLight.intensity = 2500 * vis * fl;
      const onPad = 1 - smooth(alt / 60);
      groundLight.intensity = 900 * vis * fl * onPad;
      baseGlow.material.color.setRGB(2.6, 1.4, 0.55); baseGlow.material.opacity = 0.9 * vis * onPad;
      clamps.forEach(c => { const out = 2.35 + (T > 0 ? 0.9 * smooth(T / 0.6) : 0);
        c.position.set(Math.sin(c.userData.a) * out, 0, Math.cos(c.userData.a) * out); c.rotation.y = c.userData.a; });
      te.rotation.z = (2.5 + 4 * smooth(T / 4)) * DEG;
      waterN.offset.set(T * 0.004, T * 0.0025);

      // emitters
      if (T < 0.6) spawn("vent", 16 * (1 - smooth(T / 0.6)), dt, () => {
        const [y, a] = [[BASE + 57.2, -0.7], [BASE + 39.6, -0.95], [BASE + 1.4, -1.55]][Math.floor(rnd() * 3)];
        const out = V3(Math.sin(a), 0, Math.cos(a));
        emit(out.clone().multiplyScalar(1.95).setY(y), out.multiplyScalar(1.6 + rnd() * 1.6).setY(-0.5 - rnd() * 0.6),
          { size: 0.35, grow: 1.0, life: 3 + rnd() * 1.5, op: 0.32, buoy: -0.3, drag: 0.9, shade: SM * 1.05 });
      });
      const delugeOn = T > -5.5 && alt < 120;
      if (delugeOn) for (let n = Math.round(1600 * dt); n > 0; n--) {
        const i = dNext++ % ND, a = Math.floor(rnd() * 16) * Math.PI / 8 + (rnd() - 0.5) * 0.1, sp = 5 + rnd() * 4;
        dPos.set([Math.sin(a) * 9.4, MOUNT + 0.3, Math.cos(a) * 9.4], i * 3);
        dVel.set([-Math.sin(a) * sp + (rnd() - 0.5), 6 + rnd() * 5, -Math.cos(a) * sp + (rnd() - 0.5)], i * 3);
      }
      if (delugeOn) spawn("mist", 9, dt, () => { const a = rnd() * 6.283;
        emit(V3(Math.sin(a) * 7, MOUNT + 1, Math.cos(a) * 7), V3(0, 0.6, 0), { size: 1.5, grow: 1.4, life: 3.5, op: 0.12, buoy: 0.3, drag: 1, shade: SM }); });
      if (thr > 0.02 && alt < 160) spawn("trench", 75 * thr * (1 - smooth(alt / 160)), dt, () =>
        emit(V3((rnd() - 0.5) * 7, DECK + 1.5, 24 + rnd() * 10), V3((rnd() - 0.5) * 9, 2 + rnd() * 7, 26 + rnd() * 24),
          { size: 4 + rnd() * 3, grow: 6.5, life: 10 + rnd() * 5, op: 0.82, buoy: 1.3, drag: 0.42, shade: SM }));
      if (thr > 0.02 && alt < 70) spawn("spill", (alt > 4 ? 70 : 34) * thr * (1 - smooth(alt / 70)), dt, () => {
        const a = alt > 4 ? rnd() * 6.283 : Math.PI + (rnd() - 0.5) * 2, sp = 12 + rnd() * 16, r0 = alt > 4 ? 6 : 5;
        emit(V3(Math.sin(a) * r0, 1.5 + rnd() * (alt > 4 ? MOUNT : 3), Math.cos(a) * r0), V3(Math.sin(a) * sp, 2 + rnd() * 5, Math.cos(a) * sp),
          { size: alt > 4 ? 4 + rnd() * 2 : 2.5 + rnd() * 1.5, grow: 5.5, life: 9 + rnd() * 4, op: 0.78, buoy: 1.1, drag: 0.55, shade: SM * 0.97 });
      });
      const tail = nozzle.clone().addScaledVector(down, len * 0.8);
      if (alt > 12 && trailAt) for (let n = Math.min(Math.floor(trailAt.distanceTo(tail) / 1.4), 40); n > 0; n--) {
        trailAt.lerp(tail, 1 / n);
        emit(trailAt.clone().add(V3((rnd() - 0.5) * 3, (rnd() - 0.5) * 2, (rnd() - 0.5) * 3)),
          down.clone().multiplyScalar(6).add(V3((rnd() - 0.5) * 3, 0, (rnd() - 0.5) * 3)),
          { size: 3 + 4 * smooth(alt / 1500), grow: 3.2, life: 24, op: 0.5, buoy: 0.15, drag: 0.35, shade: SM * 0.92 });
      }
      if (alt <= 12 || !trailAt || trailAt.distanceTo(tail) > 80) trailAt = tail;

      // simulate
      const fireI = 3.5 * vis * fl;
      for (const p of puffs) {
        if (!p.s.visible) continue;
        p.age += dt;
        if (p.age > p.life) { p.s.visible = false; continue; }
        const P = p.s.position;
        p.v.multiplyScalar(Math.exp(-p.drag * dt)); p.v.y += p.buoy * dt;
        P.addScaledVector(p.v, dt); P.x += WIND * dt;
        const sz = p.size + p.grow * Math.sqrt(p.age);
        const floor = (Math.hypot(P.x, P.z) < 108 ? DECK : GROUND) + sz * 0.22;
        if (P.y < floor) { P.y = floor; p.v.y = Math.max(p.v.y, 0); }
        p.s.scale.setScalar(sz);
        const g = fireI * 500 / (500 + P.distanceToSquared(nozzle));
        p.s.material.color.setRGB(p.shade + g, p.shade * 0.97 + g * 0.5, p.shade * 0.94 + g * 0.17);
        const k = p.age / p.life;
        p.s.material.opacity = p.op * smooth(p.age / 0.35) * (1 - smooth((k - 0.5) / 0.5));
      }
      for (let i = 0; i < ND * 3; i += 3) {
        if (dPos[i + 1] < -1e4) continue;
        dVel[i + 1] -= 9.8 * dt;
        dPos[i] += dVel[i] * dt; dPos[i + 1] += dVel[i + 1] * dt; dPos[i + 2] += dVel[i + 2] * dt;
        if (dPos[i + 1] < DECK + 0.1) dPos[i + 1] = -1e5;
      }
      dGeo.attributes.position.needsUpdate = true;

      // camera: quick cuts through the count, then the pad and tower cameras, then the long-lens tracker
      const u = (a, b) => clamp((T - a) / (b - a), 0, 1);
      const rk = stack.root.position, mid = rk.clone().addScaledVector(down, -30);
      let c, i;
      if (T < -7) { i = 0; const k = u(-10, -7); c = { pos: V3(-360 + 60 * k, GROUND + 16, 330 - 25 * k), tgt: V3(0, 34, 0), fov: 24 - 2 * k, shake: 0.02, exp: 0.5 }; }
      else if (T < -5) { i = 1; c = { pos: V3(4.6, MOUNT + 0.9, 5.2), tgt: V3(0, BASE + 8 + 30 * smooth(u(-7, -5)), 0), fov: 58, shake: 0.02, exp: 0.55 }; }
      else if (T < -3.2) { i = 2; const k = u(-5, -3.2); c = { pos: V3(-44 + 4 * k, 54, 36 - 3 * k), tgt: V3(0, BASE + 41.5, 0), fov: 12 - 1.5 * k, shake: 0.03, exp: 0.5 }; }
      else if (T < -0.4) { i = 3; c = { pos: V3(3.4, 3.6, 15 - 0.8 * u(-3.2, -0.4)), tgt: V3(0, 5.6, 0), fov: 40, shake: 0.03 + 0.6 * thr, exp: 0.55 / (1 + 1.6 * thr) }; }
      else if (T < 3) { i = 4; c = { pos: V3(-64, DECK + 1.7, 82), tgt: rk.clone().add(V3(0, 30, 0)), fov: 44, shake: 0.08 + 0.4 * thr, exp: 0.5 / (1 + 0.5 * thr) }; }
      else if (T < 6.5) { i = 5; const pos = V3(-210, GROUND + 8, 230);
        c = { pos, aim: rk.clone().addScaledVector(down, -14), fov: 2 * Math.atan(62 / pos.distanceTo(mid)) / DEG, shake: 0.1, exp: 0.5 }; }
      else {
        i = 6;
        const pos = V3(-880, GROUND + 6, 1020), dist = pos.distanceTo(mid), k = smooth(u(6.5, 13));
        c = { pos, fov: clamp(2 * Math.atan((95 - 40 * k) / dist) / DEG, 1.2, 14), shake: 0.04, exp: 0.5 };
      }
      const aim = c.aim ?? mid;
      if (i !== shotNow) { shotNow = i; expo = c.exp; lag.copy(aim); }   // a cut
      lag.lerp(aim, 1 - Math.exp(-dt * 2.5));
      const tgt = c.tgt ?? lag;
      expo += (c.exp - expo) * Math.min(1, dt * 3);
      camera.position.copy(c.pos).add(tmp.set(Math.sin(T * 37.1) + Math.sin(T * 23.3), Math.sin(T * 41.7) + Math.sin(T * 19.9),
        Math.sin(T * 31.3)).multiplyScalar(c.shake * 0.12));
      camera.fov = c.fov; camera.updateProjectionMatrix();
      camera.lookAt(tgt);
      const sh = c.shake * 0.006;
      camera.rotateX((Math.sin(T * 29.3) + Math.sin(T * 13.1) * 0.6 + Math.sin(T * 0.7) * 0.4) * sh);
      camera.rotateY((Math.sin(T * 26.9) + Math.sin(T * 11.7) * 0.6 + Math.sin(T * 0.53) * 0.4) * sh);
      lens.uniforms.uTime.value = T;
    },
    render(R, w, h) {
      const pr = R.getPixelRatio();
      if (w !== size.w || h !== size.h || pr !== size.pr) { composer.setPixelRatio(pr); composer.setSize(w, h); Object.assign(size, { w, h, pr }); }
      const tm = R.toneMapping, ex = R.toneMappingExposure;
      R.toneMapping = THREE.ACESFilmicToneMapping; R.toneMappingExposure = expo;
      composer.render();
      R.toneMapping = tm; R.toneMappingExposure = ex;
    },
  };
}

// ---------------------------------------------------------------- mission control
export function initMission(api) {
  const st = api.serverState;
  const S = {
    active: false, phase: "setup", T: 0, warp: 10, follow: true, userHold: 0,
    ascent: null, AF: null, plan: null, PF: null, site: null, planBusy: false, landing: null, landErr: null,
    events: [], evDone: 0, pilot: "ppo", trainSteps: 3 * st.rollout_steps, launchKey: "cape", alt: 300, inc: 53,
    status: null, lastPanel: 0, time: 0,
  };
  window.__mission = S;   // handy for poking at it from the console (S.api drives the flow)

  // ------------------------------------------------ DOM
  const root = document.createElement("div");
  root.id = "mission";
  root.innerHTML = `
    <div class="mp" id="mTop">
      <div class="brand"><i>▲</i><span>Rocket<b>Sim</b></span></div>
      <div id="mTabs"><span data-p="setup">Pre-launch</span><span data-p="ascent">Lift-off</span><span data-p="orbit">Orbit</span>
        <span data-p="descent">Descent</span><span data-p="landing">Landing</span></div>
      <div class="spacer"></div>
      <span id="mClock" class="mono"></span><span class="mlive">LIVE</span>
      <button class="mbtn" id="mClassic" title="The original launch control: set landing conditions directly">Classic launch control</button>
    </div>
    <div class="mp" id="mLeft"></div>
    <div id="mRight"><div class="mp" id="mHolo"><div class="scan"></div>
        <div class="hd"><span class="lbl">◈ Holographic model</span><span class="lbl" id="mHoloTag">LANDER-01</span></div>
        <div class="stats" id="mHoloStats"></div></div>
      <div class="mp" id="mPanel"></div></div>
    <div class="mp" id="mCard"></div>
    <div class="mp" id="mWarp"></div>
    <div id="mBig"></div>
    <div id="mToast"></div>
    <div id="mTags"></div>`;
  document.body.appendChild(root);
  const fade = document.createElement("div"); fade.id = "mFade"; document.body.appendChild(fade);
  const ret = document.createElement("div"); ret.id = "mReturn"; ret.hidden = true; document.body.appendChild(ret);

  // ------------------------------------------------ scenes
  const G = buildGlobe();
  const holo = buildHolo(api, st.vehicle.lander);
  let pad = null;
  const controls = new OrbitControls(G.camera, api.renderer.domElement);
  Object.assign(controls, { enableDamping: true, dampingFactor: 0.08, enablePan: false, minDistance: 1.08, maxDistance: 14,
    rotateSpeed: 0.5, zoomSpeed: 0.7, enabled: false });
  controls.addEventListener("start", () => { S.userHold = 1e9; });
  controls.addEventListener("end", () => { S.userHold = S.time + 4; });
  G.camera.position.set(0, 0.8, 3.4);

  const rocket = glowSprite(0xffffff, 0.035); G.scene.add(rocket);
  const halo = glowSprite(0x5ee6ff, 0.09); G.scene.add(halo);
  const ascentTrail = makeLine(2000, 0xffb547, 0.95); G.earth.add(ascentTrail);
  const orbitLine = makeLine(361, 0x5ee6ff, 0.55); G.scene.add(orbitLine);
  const groundTrack = makeLine(241, 0x5ee6ff, 0.35, true); G.earth.add(groundTrack);
  const descentLine = makeLine(600, 0x2fe39a, 0.85); G.earth.add(descentLine);
  const burnMark = glowSprite(0xffb547, 0.05); burnMark.visible = false; G.scene.add(burnMark);
  const siteMark = new THREE.Group(); siteMark.visible = false; G.earth.add(siteMark);
  const siteRing = new THREE.Mesh(new THREE.RingGeometry(0.012, 0.016, 48), new THREE.MeshBasicMaterial({ color: 0x2fe39a,
    transparent: true, side: THREE.DoubleSide, depthWrite: false }));
  const siteBeam = new THREE.Mesh(new THREE.CylinderGeometry(0.0012, 0.0012, 0.08, 6).translate(0, 0.04, 0),
    new THREE.MeshBasicMaterial({ color: 0x2fe39a, transparent: true, opacity: 0.8 }));
  siteMark.add(siteRing, siteBeam, glowSprite(0x2fe39a, 0.03));
  const launchMark = glowSprite(0xffb547, 0.03); G.earth.add(launchMark);

  // HTML tags that follow 3D points
  const tags = {};
  function tag(key, text, color) {
    if (!tags[key]) { const el = document.createElement("div"); el.className = "mtag"; $("mTags").appendChild(el); tags[key] = el; }
    tags[key].textContent = text; tags[key].style.color = color; tags[key].style.borderColor = color + "66";
    return tags[key];
  }
  const _p = new THREE.Vector3();
  function occluded(world) {   // does the globe sit between the camera and this point?
    const c = G.camera.position, d = world.clone().sub(c);
    const a = d.dot(d), b = 2 * c.dot(d), disc = b * b - 4 * a * (c.dot(c) - 1);
    if (disc < 0) return false;
    const t = (-b - Math.sqrt(disc)) / (2 * a);
    return t > 0 && t < 0.998;
  }
  function placeTag(el, world, visible = true) {
    _p.copy(world).project(G.camera);
    const show = visible && _p.z < 1 && !occluded(world);
    el.style.display = show ? "block" : "none";
    if (show) { el.style.left = `${(_p.x + 1) / 2 * innerWidth}px`; el.style.top = `${(1 - _p.y) / 2 * innerHeight}px`; }
  }

  // ------------------------------------------------ state helpers
  const lander = st.vehicle.lander;
  const sites = st.launch_sites;
  function setPhase(p) {
    S.phase = p;
    document.querySelectorAll("#mTabs span").forEach(el => {
      const order = ["setup", "ascent", "orbit", "descent", "landing"];
      el.classList.toggle("on", el.dataset.p === p || (p === "liftoff" && el.dataset.p === "ascent"));
      el.classList.toggle("done", order.indexOf(el.dataset.p) < order.indexOf(p === "liftoff" ? "ascent" : p));
    });
    renderLeft(); renderPanel(); renderWarp();
  }
  function toast(html) {
    const el = $("mToast"); el.innerHTML = html; el.classList.add("show");
    clearTimeout(toast.t); toast.t = setTimeout(() => el.classList.remove("show"), 2800);
  }
  function big(l1, l2, l3) {
    $("mBig").innerHTML = l2 == null ? "" : `<div class="l1">${l1}</div><div class="l2">${l2}</div><div class="l3">${l3 ?? ""}</div>`;
  }
  const launchSite = () => sites[S.launchKey];

  // current rocket state: { pos (world), lat, lon, alt, speed, extra... }
  const _w = new THREE.Vector3();
  function rocketState() {
    const T = S.T;
    if (S.phase === "setup" || (S.phase === "liftoff" && T <= 0)) {
      const s = launchSite();
      return { pos: fixedToWorld(llaLocal(s.lat, s.lon, 0, _w), T), lat: s.lat, lon: s.lon, alt: 0, speed: OMEGA * RE * Math.cos(s.lat * DEG) };
    }
    const A = S.ascent;
    if ((S.phase === "liftoff" || S.phase === "ascent") && T < A.duration) {
      const f = sampleRows(A.frames, T, [S.AF.stage]), F = S.AF;
      return { pos: fixedToWorld(llaLocal(f[F.lat], f[F.lon], f[F.alt], _w), T), lat: f[F.lat], lon: f[F.lon], alt: f[F.alt],
        speed: f[F.speed], q: f[F.q], g: f[F.accel_g], thr: f[F.throttle], stage: f[F.stage], pitch: f[F.pitch], mass: f[F.mass] };
    }
    if (S.phase === "descent" && S.plan && T >= S.plan.t_burn) {
      const P = S.PF, f = sampleRows(S.plan.track, Math.min(T, S.plan.t_land), [P.phase]);
      return { pos: fixedToWorld(llaLocal(f[P.lat], f[P.lon], f[P.alt], _w), T), lat: f[P.lat], lon: f[P.lon], alt: f[P.alt],
        speed: f[P.speed], q: f[P.q], g: f[P.accel_g], heat: f[P.heat], fuel: f[P.fuel], dphase: f[P.phase] };
    }
    const { r, v } = propagate(A.orbit, T);
    const [lat, lon] = eciLatLon(r, T);
    return { pos: eciWorld(r, _w), lat, lon, alt: Math.hypot(...r) - RE, speed: Math.hypot(...v) };
  }

  // ------------------------------------------------ left sidebar
  function renderLeft() {
    const el = $("mLeft"), p = S.phase;
    const head = (title, count) => `<span class="mlive">LIVE TRACKING</span><h1>${title}</h1><div class="count">${count}</div>`;
    if (p === "setup") {
      const s = launchSite(), minInc = Math.ceil(Math.abs(s.lat));
      S.inc = Math.max(S.inc, minInc);
      const v = st.vehicle;
      el.innerHTML = head("Pre-launch", `<b>${G.sats.count.toLocaleString()}</b> objects in orbit`) + `
        <div class="mfield"><div class="lab"><span>Launch site</span></div><select id="mSite">${Object.entries(sites).map(([k, x]) =>
          `<option value="${k}" ${k === S.launchKey ? "selected" : ""}>${x.name}</option>`).join("")}</select></div>
        <div class="mfield"><div class="lab"><span>Target orbit</span><b id="mAltV"></b></div>
          <input type="range" id="mAlt" min="${st.orbit_limits.altitude[0]}" max="${st.orbit_limits.altitude[1]}" step="10" value="${S.alt}"></div>
        <div class="mfield"><div class="lab"><span>Inclination</span><b id="mIncV"></b></div>
          <input type="range" id="mInc" min="${minInc}" max="${st.orbit_limits.inclination[1]}" step="0.5" value="${S.inc}"></div>
        <div class="mnote" id="mIncNote"></div>
        <button class="mbtn go" id="mLift">▲ Lift off</button>
        <div class="lbl msec">Launch vehicle</div>
        ${v.stages.map((x, i) => `<div class="mrow"><div class="t"><b>${x.name}</b><small>Stage ${i + 1} · ${(x.thrust_vac / 1e3).toFixed(0)} kN · Isp ${x.isp_vac} s</small></div>
          <span class="mbadge" style="--c:${i ? "var(--m-violet)" : "var(--m-amber)"}">${(x.dv / 1000).toFixed(1)} km/s</span></div>`).join("")}
        <div class="mrow"><div class="t"><b>Lander</b><small>Payload · ${(lander.dry_mass / 1000).toFixed(1)} t + ${(lander.fuel / 1000).toFixed(1)} t fuel</small></div>
          <span class="mbadge" style="--c:var(--m-cyan)">${(lander.thrust / 1e3).toFixed(0)} kN</span></div>
        <div class="mrow"><div class="t"><b>Liftoff</b><small>${(v.liftoff_mass / 1000).toFixed(0)} t</small></div>
          <span class="mbadge" style="--c:var(--m-green)">TWR ${v.liftoff_twr.toFixed(2)}</span></div>`;
      const show = () => {
        S.alt = +$("mAlt").value; S.inc = +$("mInc").value;
        $("mAltV").textContent = `${S.alt} km`; $("mIncV").textContent = `${S.inc.toFixed(1)}°`;
        const lat = launchSite().lat;
        $("mIncNote").textContent = S.inc >= 90 ? "Polar / retrograde: passes over every latitude, so any landing site is reachable."
          : `Ground track covers ±${S.inc.toFixed(0)}° latitude; sites farther out need a costly plane change.`;
        if (Math.abs(S.inc - Math.abs(lat)) < 0.6) $("mIncNote").textContent += " Due-east launch: Earth's spin helps most.";
      };
      $("mAlt").oninput = $("mInc").oninput = show; show();
      $("mSite").onchange = e => { S.launchKey = e.target.value; renderLeft(); aimAtLaunchSite(); };
      $("mLift").onclick = liftOff;
      return;
    }
    S.leftDyn = true;
    el.innerHTML = `<div id="mLeftDyn"></div>`;
    updateLeft();
  }
  function updateLeft() {
    const el = $("mLeftDyn"); if (!el || !S.ascent) return;
    const A = S.ascent, o = A.orbit, T = S.T, p = S.phase;
    const title = { liftoff: "Lift-off", ascent: T < eventT("seco2") ? "Ascent" : "In Orbit", orbit: "In Orbit", descent: "Descent" }[p] ?? "In Orbit";
    const rev = Math.max(T - A.duration, 0) / o.period;
    const done = n => T >= eventT(n);
    const b1 = done("stage_sep") ? ["SEP", "var(--m-dim)"] : T >= 0 ? ["BURN", "var(--m-amber)"] : ["READY", "var(--m-green)"];
    const b2 = p === "descent" || done("seco2") ? [`${((S.plan && T >= S.plan.t_burn ? A.budget_dv - S.plan.dv : A.budget_dv) / 1000).toFixed(2)} km/s`, "var(--m-violet)"]
      : done("ses") ? ["BURN", "var(--m-amber)"] : ["STACKED", "var(--m-dim)"];
    const rs = rocketState();
    const fuel = rs.fuel ?? 1;
    const evs = [...A.events, ...(S.plan && (p === "descent" || S.plan.committed) ? S.plan.events : [])];
    el.innerHTML = `<span class="mlive">LIVE TRACKING</span><h1>${title}</h1>
      <div class="count"><b>${G.sats.count.toLocaleString()}</b> objects tracked${T > A.duration ? ` · rev <b>${rev.toFixed(2)}</b>` : ""}</div>
      <div class="mrow"><div class="t"><b>Booster</b><small>Stage 1 · ${sites[S.launchKey].name.split(",")[0]}</small></div>
        <span class="mbadge" style="--c:${b1[1]}">${b1[0]}</span></div>
      <div class="mrow"><div class="t"><b>Orbital stage</b><small>Stage 2 · Δv reserve</small></div>
        <span class="mbadge" style="--c:${b2[1]}">${b2[0]}</span></div>
      <div class="mrow"><div class="t"><b>Lander</b><small>LANDER-01 · fuel</small></div>
        <span class="mbadge" style="--c:var(--m-cyan)">${(fuel * 100).toFixed(0)}%</span></div>
      <div class="mrow"><div class="t"><b>Orbit</b><small>apoapsis × periapsis</small></div>
        <span class="mbadge" style="--c:var(--m-blue)">${(o.apoapsis / 1e3).toFixed(0)} × ${(o.periapsis / 1e3).toFixed(0)} km</span></div>
      <div class="mrow"><div class="t"><b>Inclination</b><small>period ${(o.period / 60).toFixed(1)} min</small></div>
        <span class="mbadge" style="--c:var(--m-green)">${o.inclination.toFixed(2)}°</span></div>
      <div class="lbl msec">Mission events</div>
      <div class="mev">${evs.map(e => {
        const d = T >= e.t, cls = d ? "done" + (e.name === "seco2" || e.name === "gate" ? " k" : "") : "";
        return `<span class="${cls}">${fmtT(e.t).slice(1)}</span><span class="${cls}">${EVENT_LABEL[e.name] ?? e.name}</span>`;
      }).join("")}</div>`;
  }
  const eventT = name => S.ascent?.events.find(e => e.name === name)?.t ?? Infinity;

  // ------------------------------------------------ right panel
  function renderPanel() {
    const el = $("mPanel"), p = S.phase;
    if (p === "setup") {
      el.innerHTML = `<h3>Mission profile</h3><div class="hint">From the pad to a landing anywhere on Earth.</div>
        <div class="kv"><span>1 · Lift-off</span><b>gravity turn, max-Q throttle bucket</b>
        <span>2 · Orbit</span><b>SECO-1, coast, circularize</b><span>3 · Landing site</span><b>rotate the Earth, click to pick</b>
        <span>4 · Deorbit</span><b>plane change + retro burn</b><span>5 · Train</span><b>policy practices that site</b>
        <span>6 · Land</span><b>entry, braking burn, 6-DOF touchdown</b></div>
        <div class="mnote">Ascent, orbit, entry and landing are simulated (point-mass orbital physics, then the 6-DOF landing sim with
        the site's terrain and wind).</div>`;
      return;
    }
    if (p === "liftoff" || p === "ascent") { el.innerHTML = `<h3>Ascent</h3><div class="hint">Guidance and loads</div><div id="mAscDyn"></div>`; return; }
    if (p === "orbit") {
      el.innerHTML = `<h3>Landing site</h3><div class="hint">Drag to rotate the Earth · scroll to zoom · click land to pick a site</div>
        <div id="mSiteDyn"></div>
        <div class="lbl msec">Landing pilot</div>
        <div class="mseg" id="mPilot"><button data-v="ppo">Neural net</button><button data-v="pd">Autopilot</button></div>
        <div class="mfield"><div class="lab"><span>Training before landing</span></div><select id="mTrain">${[0, 1, 3, 6, 15, 60].map(n =>
          `<option value="${n * st.rollout_steps}" ${n * st.rollout_steps === S.trainSteps ? "selected" : ""}>${n ? `${(n * st.rollout_steps / 1000).toFixed(0)}k steps at this site` : "none"}</option>`).join("")}</select></div>
        <button class="mbtn go amber" id="mLand" disabled>Deorbit &amp; land</button>
        <div class="mnote" id="mLandNote"></div>`;
      const seg = $("mPilot");
      const paint = () => { seg.querySelectorAll("button").forEach(b => b.classList.toggle("on", b.dataset.v === S.pilot)); $("mTrain").disabled = S.pilot !== "ppo"; };
      seg.onclick = e => { if (e.target.dataset.v) { S.pilot = e.target.dataset.v; paint(); } };
      $("mTrain").onchange = e => S.trainSteps = +e.target.value;
      $("mLand").onclick = commitLanding;
      paint(); renderSite();
      return;
    }
    if (p === "descent") { el.innerHTML = `<h3>Deorbit &amp; landing</h3><div class="hint" id="mDescHint"></div><div id="mDescDyn"></div>`; return; }
  }
  function renderSite() {
    const el = $("mSiteDyn"); if (!el) return;
    const s = S.site, P = S.plan, btn = $("mLand"), note = $("mLandNote");
    if (!s) { el.innerHTML = `<div class="mnote">No site selected yet.</div>`; btn.disabled = true; note.textContent = ""; return; }
    const rows = [
      ["Location", `${fmtLat(s.lat)} · ${fmtLon(s.lon)}`],
      ["Surface", { land: "Land", ocean: "Open ocean", ice: "Ice sheet" }[s.surface]],
    ];
    let html = `<div class="kv">${rows.map(([k, v]) => `<span>${k}</span><b>${v}</b>`).join("")}</div>`;
    if (s.surface !== "ocean") {
      const amp = 2 + 12 * s.ruggedness;
      html += `<div class="kv"><span>Terrain relief</span><b>±${amp.toFixed(0)} m · ${s.ruggedness < 0.25 ? "plains" : s.ruggedness < 0.6 ? "hills" : "mountains"}</b></div>
        <div class="mbar"><i style="width:${(s.ruggedness * 100).toFixed(0)}%;background:var(--m-amber)"></i></div>`;
      if (s.wind) html += `<div class="kv"><span>Prevailing wind</span><b>${fmtWind(s.wind)}</b></div>`;
    }
    if (P && s.surface !== "ocean") {
      const frac = Math.min(P.dv / P.budget_dv, 1);
      html += `<div class="lbl msec">Next opportunity</div><div class="kv">
        <span>Deorbit burn in</span><b>${fmtDur(P.t_burn - S.T)} · rev ${(Math.max(P.t_burn - S.ascent.duration, 0) / S.ascent.orbit.period).toFixed(1)}</b>
        <span>Δv needed / available</span><b>${P.dv.toFixed(0)} / ${P.budget_dv.toFixed(0)} m/s</b></div>
        <div class="mbar"><i style="width:${(frac * 100).toFixed(0)}%;background:${P.reachable ? "var(--m-green)" : "var(--m-red)"}"></i></div>
        <div class="kv"><span>Plane change</span><b>${P.plane_change.toFixed(2)}° (${(Math.abs(P.cross_track) / 1e3).toFixed(0)} km cross-track)</b>
        <span>Retro burn</span><b>${P.dv_retro.toFixed(0)} m/s</b>
        <span>Deorbit to touchdown</span><b>${fmtDur(P.tof)}</b>
        <span>Entry loads</span><b>${P.peak_g.toFixed(1)} g · ${(P.peak_heat / 1e4).toFixed(0)} W/cm²</b>
        <span>Hand-off gate</span><b>${P.gate.alt.toFixed(0)} m · ${P.gate.v_down.toFixed(0)} m/s · fuel ${(P.gate.fuel * 100).toFixed(0)}%</b></div>`;
    }
    el.innerHTML = html;
    const ok = s.surface !== "ocean" && P?.reachable && !S.planBusy;
    btn.disabled = !ok;
    note.className = "mnote" + (s.surface === "ocean" || (P && !P.reachable) ? " bad" : "");
    note.textContent = s.surface === "ocean" ? "Open ocean: the lander needs solid ground. Pick a spot on land."
      : S.planBusy ? "Computing deorbit opportunity…"
      : P && !P.reachable ? `Out of reach within 2 days: the best pass needs ${P.dv.toFixed(0)} m/s. Try a site closer to the ground track, or fly a higher-inclination orbit.`
      : S.landErr ? S.landErr : "";
    if (S.landErr) note.className = "mnote bad";
  }

  // ------------------------------------------------ warp + card + holo stats
  function renderWarp() {
    const el = $("mWarp"), p = S.phase;
    el.style.display = p === "setup" || p === "liftoff" ? "none" : "flex";
    el.innerHTML = `<button data-pause="1" title="Pause (Space)">${S.paused ? "▶" : "❚❚"}</button><span class="lbl">Time warp</span>` + WARPS.map(w => `<button data-w="${w}" class="${w === S.warp ? "on" : ""}">${w}×</button>`).join("")
      + `<button data-f="1" class="${S.follow ? "on" : ""}" title="Keep the camera on the rocket">Follow</button>`
      + (p === "ascent" ? `<button data-skip="1">Skip to orbit ▸▸</button>` : "")
      + (p === "descent" ? `<button data-skip="1">Skip to landing ▸▸</button>` : "");
    el.onclick = e => {
      const b = e.target.closest("button"); if (!b) return;
      if (b.dataset.w) S.warp = +b.dataset.w;
      if (b.dataset.f) S.follow = !S.follow;
      if (b.dataset.pause) S.paused = !S.paused;
      if (b.dataset.skip) {
        if (p === "ascent") S.T = Math.max(S.T, S.ascent.duration);
        if (p === "descent") S.T = Math.max(S.T, S.plan.t_land);
      }
      renderWarp();
    };
  }
  function renderCard(rs) {
    const p = S.phase;
    const name = { setup: "ON THE PAD", liftoff: S.T < 0 ? "TERMINAL COUNT" : "LIFT-OFF", ascent: "POWERED ASCENT", orbit: "ORBITING",
      descent: rs.dphase === 2 ? "BRAKING BURN" : rs.dphase === 1 ? "ATMOSPHERIC ENTRY" : "DEORBIT COAST" }[p] ?? "";
    const extra = rs.q != null && p !== "orbit"
      ? `<div><span class="lbl">Dyn. pressure</span><b class="num">${(rs.q / 1000).toFixed(1)} kPa</b></div>
         <div><span class="lbl">Acceleration</span><b class="num">${rs.g.toFixed(2)} g</b></div>` : "";
    $("mCard").innerHTML = `<div class="top"><span class="lbl">RocketSim · ${S.launchKey.toUpperCase()} · 2026-001A</span><span class="lbl">${name}</span></div>
      <h2>LANDER-01</h2><div class="sub">NORAD 99001 · ${p === "setup" || p === "liftoff" ? "Booster + orbital stage + lander" : p === "descent" ? "Lander (stage separated)" : "Orbital stage + lander"}</div>
      <div class="grid"><div><span class="lbl">Altitude</span><b class="num">${km(Math.max(rs.alt, 0))}</b></div>
        <div><span class="lbl">Velocity (inertial)</span><b class="num">${rs.speed >= 1000 ? (rs.speed / 1000).toFixed(2) + " km/s" : rs.speed.toFixed(0) + " m/s"}</b></div>
        <div><span class="lbl">Latitude</span><b class="num">${fmtLat(rs.lat)}</b></div>
        <div><span class="lbl">Longitude</span><b class="num">${fmtLon(rs.lon)}</b></div>${extra}</div>`;
  }
  function renderHoloStats(rs) {
    const fuel = rs.fuel ?? 1;
    const mass = lander.dry_mass + lander.fuel * fuel;
    $("mHoloStats").innerHTML = `<div>Mass<b>${(mass / 1000).toFixed(2)} t</b></div><div>Fuel<b>${(fuel * 100).toFixed(0)}%</b></div>
      <div>TWR (Earth)<b>${(lander.thrust / (mass * 9.81)).toFixed(2)}</b></div>`;
  }
  function renderAscentPanel(rs) {
    const el = $("mAscDyn"); if (!el) return;
    const T = S.T, A = S.ascent;
    const nextEv = A.events.find(e => e.t > T);
    const bar = (v, c) => `<div class="mbar"><i style="width:${clamp(v, 0, 1) * 100}%;background:${c}"></i></div>`;
    el.innerHTML = rs.q == null ? `<div class="mnote ok">In orbit · ${(A.orbit.apoapsis / 1e3).toFixed(0)} × ${(A.orbit.periapsis / 1e3).toFixed(0)} km</div>` : `
      <div class="kv"><span>Stage</span><b>${rs.stage ? "2 · orbital stage" : "1 · booster"}</b>
        <span>Pitch above horizon</span><b>${rs.pitch.toFixed(1)}°</b><span>Throttle</span><b>${(rs.thr * 100).toFixed(0)}%</b></div>${bar(rs.thr, "var(--m-amber)")}
      <div class="kv"><span>Dynamic pressure</span><b>${(rs.q / 1000).toFixed(1)} kPa</b></div>${bar(rs.q / 32000, "var(--m-red)")}
      <div class="kv"><span>Acceleration</span><b>${rs.g.toFixed(2)} g</b></div>${bar(rs.g / 8, "var(--m-violet)")}
      <div class="kv"><span>Mass</span><b>${(rs.mass / 1000).toFixed(1)} t</b><span>Next</span><b>${nextEv ? `${EVENT_LABEL[nextEv.name]} in ${fmtDur(nextEv.t - T)}` : "—"}</b></div>`;
  }
  function renderDescentPanel(rs) {
    const el = $("mDescDyn"); if (!el) return;
    const P = S.plan, T = S.T, s = S.site;
    $("mDescHint").textContent = `${fmtLat(s.lat)} · ${fmtLon(s.lon)} · ${S.pilot === "ppo" ? "neural-net pilot" : "autopilot"}`;
    const status = S.status?.phase === "training" ? `Training on this site… ${S.status.done.toLocaleString()} / ${S.status.total.toLocaleString()} steps`
      : S.landing ? (S.landing.replay.rockets[0].outcome === "landed" || S.landing.replay.rockets[0].outcome === "caught" ? "Landing flown · ready" : "Landing flown")
      : S.landErr ? S.landErr : "Flying the landing…";
    const frac = S.status?.phase === "training" ? S.status.done / Math.max(S.status.total, 1) : S.landing ? 1 : 0.1;
    const heatFrac = rs.heat != null ? rs.heat / P.peak_heat : 0;
    el.innerHTML = `<div class="kv">${T < P.t_burn ? `<span>Deorbit burn in</span><b>${fmtDur(P.t_burn - T)}</b>` : `<span>Touchdown gate in</span><b>${fmtDur(Math.max(P.t_land - T, 0))}</b>`}
      <span>Δv burned</span><b>${P.dv.toFixed(0)} m/s (${P.plane_change.toFixed(1)}° plane change)</b></div>
      ${rs.heat != null ? `<div class="kv"><span>Heating</span><b>${(rs.heat / 1e4).toFixed(0)} W/cm²</b></div>
        <div class="mbar"><i style="width:${(heatFrac * 100).toFixed(0)}%;background:var(--m-red)"></i></div>
        <div class="kv"><span>Deceleration</span><b>${rs.g.toFixed(2)} g</b><span>Lander fuel</span><b>${(rs.fuel * 100).toFixed(0)}%</b></div>` : ""}
      <div class="lbl msec">Landing policy</div><div class="mnote ${S.landErr ? "bad" : ""}">${status}</div>
      <div class="mbar"><i style="width:${(frac * 100).toFixed(0)}%"></i></div>
      ${S.landing ? `<div class="kv"><span>Terrain relief</span><b>±${S.landing.site.amplitude.toFixed(0)} m</b>
        <span>Wind</span><b>${fmtWind(S.landing.site.wind)}</b>
        <span>Pad offset at gate</span><b>${Math.hypot(S.landing.conditions.offset_x, S.landing.conditions.offset_z).toFixed(0)} m</b></div>` : ""}`;
  }

  // ------------------------------------------------ actions
  function aimAtLaunchSite() {
    const s = launchSite();
    const dir = fixedToWorld(llaLocal(s.lat + 12, s.lon), 0).normalize();
    G.camera.position.copy(dir.multiplyScalar(G.camera.position.length() || 3.2));
    launchMark.position.copy(llaLocal(s.lat, s.lon, 2e3));
    // a morning launch: the sun stands a little east of the pad
    G.sunDir.copy(llaLocal(s.lat * 0.4, s.lon + 35)).normalize();
  }

  async function liftOff() {
    const btn = $("mLift"); btn.disabled = true; btn.textContent = "Computing trajectory…";
    try {
      S.ascent = await post("/api/mission/ascent", { site: S.launchKey, altitude: S.alt, inclination: S.inc });
    } catch (e) {
      btn.disabled = false; btn.textContent = "▲ Lift off"; toast(`Ascent failed: ${e.message}`); return;
    }
    const A = S.ascent;
    S.AF = idx(A.fields);
    S.plan = null; S.site = null; S.landing = null; S.landErr = null; siteMark.visible = false;
    descentLine.geometry.setDrawRange(0, 0); burnMark.visible = false;
    setLine(ascentTrail, A.frames.map(f => llaLocal(f[S.AF.lat], f[S.AF.lon], f[S.AF.alt])));
    ascentTrail.geometry.setDrawRange(0, 0);
    const pts = [];
    for (let i = 0; i <= 360; i++) pts.push(eciWorld(propagate(A.orbit, A.duration + A.orbit.period * i / 360).r));
    setLine(orbitLine, pts); orbitLine.visible = false;
    pad = pad ?? buildPad(api);
    S.T = -COUNTDOWN; S.warp = 1; S.follow = true;
    setPhase("liftoff");
  }

  let planSeq = 0;
  async function selectSite(lat, lon) {
    if (S.phase !== "orbit" || S.plan?.committed) return;
    const c = classify(lat, lon);
    const a = Math.abs(lat), east = [-3, -4.5, 0, 5.5, 3, -2.5, -1], pole = [0, -1.5, 0, 1, 0, -1, 0];
    const interp = tbl => { const i = Math.min(Math.floor(a / 15), 5), f = (a - i * 15) / 15; return tbl[i] + (tbl[i + 1] - tbl[i]) * f; };
    S.site = { lat, lon, ...c, wind: c.surface === "ocean" ? null : [interp(east), 0, -interp(pole) * (lat >= 0 ? 1 : -1)] };
    S.plan = null; S.landErr = null;
    siteMark.visible = true;
    siteMark.position.copy(llaLocal(lat, lon));
    siteMark.quaternion.setFromUnitVectors(Y, siteMark.position.clone().normalize());
    siteRing.rotation.x = -Math.PI / 2;
    siteMark.children.forEach(m => m.material && (m.material.color.set(c.surface === "ocean" ? 0xff5d6c : 0x2fe39a)));
    descentLine.geometry.setDrawRange(0, 0); burnMark.visible = false;
    if (c.surface === "ocean") { renderSite(); return; }
    S.planBusy = true; renderSite();
    const seq = ++planSeq;
    try {
      const plan = await post("/api/mission/plan", { orbit: S.ascent.orbit, budget_dv: S.ascent.budget_dv, t_now: S.T, lat, lon });
      if (seq !== planSeq) return;
      showPlan(plan);
    } catch (e) {
      if (seq === planSeq) S.landErr = e.message;
    } finally {
      if (seq === planSeq) { S.planBusy = false; renderSite(); }
    }
  }
  function showPlan(plan) {
    S.plan = plan; S.PF = idx(plan.fields);
    const P = S.PF;
    setLine(descentLine, plan.track.map(f => llaLocal(f[P.lat], f[P.lon], f[P.alt])));
    descentLine.material.opacity = plan.reachable ? 0.85 : 0.35;
    descentLine.material.color.set(plan.reachable ? 0x2fe39a : 0xff5d6c);
    burnMark.visible = true;
  }

  async function commitLanding() {
    if (!S.site || !S.plan?.reachable) return;
    const site = S.site;
    // re-plan from now, so the burn is never in the past, then start the landing on the server
    let plan;
    try {
      plan = await post("/api/mission/plan", { orbit: S.ascent.orbit, budget_dv: S.ascent.budget_dv, t_now: S.T, lat: site.lat, lon: site.lon });
    } catch (e) { S.landErr = e.message; renderSite(); return; }
    if (!plan.reachable) { showPlan(plan); renderSite(); return; }
    showPlan(plan);
    plan.committed = true;
    S.landing = null; S.landErr = null; S.status = null;
    S.warp = Math.max(S.warp, Math.min(1000, Math.ceil((plan.t_burn - S.T) / 6)));
    S.follow = true;
    setPhase("descent");
    const poll = setInterval(async () => { try { S.status = (await api.fetchJSON(`/api/state?task=${api.task}`)).status; } catch {} }, 400);
    try {
      S.landing = await post("/api/mission/land", { task: api.task, orbit: S.ascent.orbit, budget_dv: S.ascent.budget_dv, t_now: plan.t_now,
        lat: site.lat, lon: site.lon, surface: site.surface, ruggedness: site.ruggedness, color: site.color,
        pilot: S.pilot, train_steps: S.pilot === "ppo" ? S.trainSteps : 0 });
    } catch (e) {
      S.landErr = `Landing failed: ${e.message}`;
    } finally {
      clearInterval(poll); S.status = null;
    }
  }

  function handOff() {
    S.phase = "landing";
    fade.classList.add("on");
    setTimeout(() => {
      const r = S.landing.replay, out = r.rockets[0].outcome.replace(/_/g, " ");
      exit();
      api.showReplay(r);
      document.body.classList.add("mission-landing");
      ret.hidden = false;
      ret.innerHTML = `<span>Mission · ${fmtLat(S.site.lat)}, ${fmtLon(S.site.lon)} · <b>${out}</b></span>
        <button id="mBack">◀ Back to orbit</button><button id="mNew">New mission</button>`;
      $("mBack").onclick = () => {   // fly another landing from a fresh copy of the same orbit
        S.T = S.ascent.duration; S.plan = null; S.site = null; S.landing = null; siteMark.visible = false;
        descentLine.geometry.setDrawRange(0, 0); burnMark.visible = false; S.warp = 10; S.follow = false; S.zoomTo = 2.7;
        enter(); setPhase("orbit");
      };
      $("mNew").onclick = () => { S.ascent = null; enter(); setPhase("setup"); aimAtLaunchSite(); };
      fade.classList.remove("on");
    }, 750);
  }

  // ------------------------------------------------ picking
  let downAt = null;
  const ray = new THREE.Raycaster();
  api.renderer.domElement.addEventListener("pointerdown", e => { if (S.active) downAt = [e.clientX, e.clientY]; });
  api.renderer.domElement.addEventListener("pointerup", e => {
    if (!S.active || !downAt || S.phase !== "orbit") return;
    const moved = Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]); downAt = null;
    if (moved > 5) return;
    ray.setFromCamera(new THREE.Vector2(e.clientX / innerWidth * 2 - 1, -(e.clientY / innerHeight) * 2 + 1), G.camera);
    const hit = ray.intersectObject(G.globe)[0];
    if (!hit) return;
    const p = G.earth.worldToLocal(hit.point.clone()).normalize();
    selectSite(Math.asin(p.y) / DEG, Math.atan2(-p.z, p.x) / DEG);
  });

  // ------------------------------------------------ enter / exit
  function enter() {
    S.active = true; root.hidden = false; ret.hidden = true;
    document.body.classList.add("mission-on"); document.body.classList.remove("mission-landing");
    api.setMainControls(false); controls.enabled = true;
  }
  function exit() {
    S.active = false; root.hidden = true;
    document.body.classList.remove("mission-on");
    api.setMainControls(true); controls.enabled = false;
  }
  addEventListener("keydown", e => {
    if (!S.active || e.code !== "Space" || e.target.tagName === "SELECT") return;
    e.preventDefault(); e.stopImmediatePropagation();
    S.paused = !S.paused; renderWarp();
  }, true);
  $("mClassic").onclick = () => { exit(); ret.hidden = true; document.body.classList.remove("mission-landing"); };
  api.onOpenMission(() => { enter(); if (S.phase === "landing") setPhase(S.ascent ? "orbit" : "setup"); });

  // ------------------------------------------------ frame
  const camDir = new THREE.Vector3();
  let lastUi = 0, lastTrack = -1e9;
  function frame(dtReal) {
    if (!S.active) return false;
    const R = api.renderer;
    S.time += dtReal;
    const p = S.phase, A = S.ascent;

    // advance mission time
    if (S.paused) {
      // hold
    } else if (p === "liftoff") {
      S.T += dtReal;
      const cd = -S.T;
      if (S.T < 0) big("Terminal count", `T−${Math.ceil(cd)}`, cd < 2.6 ? "Ignition" : `${launchSite().name} · ${S.alt} km × ${S.inc.toFixed(1)}°`);
      else if (S.T < 3) big("Liftoff", "LIFTOFF", `LANDER-01 · ${launchSite().name}`);
      else big();
      if (S.T >= VIGNETTE) {
        fade.classList.add("on");
        setTimeout(() => fade.classList.remove("on"), 600);
        big(); S.warp = 10; setPhase("ascent");
        const rs = rocketState();
        G.camera.position.copy(rs.pos.clone().normalize().multiplyScalar(1.8));
      }
    } else if (p === "ascent") {
      const coasting = S.T > eventT("seco1") && S.T < eventT("circ_burn") - 20;
      S.T += dtReal * (coasting ? Math.max(S.warp, 60) : S.warp);
      if (S.T >= A.duration) {
        S.T = A.duration; S.warp = 60; S.follow = false; S.zoomTo = 2.7;
        toast(`<b>Orbit achieved</b> · ${(A.orbit.apoapsis / 1e3).toFixed(0)} × ${(A.orbit.periapsis / 1e3).toFixed(0)} km · pick a landing site`);
        setPhase("orbit");
      }
    } else if (p === "orbit") {
      S.T += dtReal * S.warp;
    } else if (p === "descent") {
      const P = S.plan;
      if (S.T < P.t_burn) {
        S.T = Math.min(S.T + dtReal * S.warp, P.t_burn);
        if (S.T >= P.t_burn) { S.warp = 60; renderWarp(); }   // the wait was warped through; fly the descent watchably
      } else S.T += dtReal * S.warp;
      if (S.T >= P.t_land) {
        S.T = P.t_land;
        if (S.landing) handOff();
        else big("Hand-off gate", S.landErr ? "HOLD" : "STANDBY", S.landErr ?? "Training the landing policy for this site…");
      }
    }

    // events → toasts
    if (A) {
      const evs = [...A.events, ...(S.plan?.committed ? S.plan.events : [])].filter(e => e.name !== "liftoff");
      const passed = evs.filter(e => e.t <= S.T).length;
      if (passed > S.evDone && p !== "setup") {
        const e = evs.filter(e => e.t <= S.T).sort((a, b) => a.t - b.t).at(-1);
        if (S.T - e.t < 120 * Math.max(S.warp, 1)) toast(`<b>${fmtT(e.t)}</b> · ${EVENT_LABEL[e.name] ?? e.name}`);
      }
      S.evDone = passed;
    }

    // globe
    G.earth.rotation.y = OMEGA * S.T;
    G.clouds.rotation.y = S.T * 2e-6;
    G.sats.mat.uniforms.uTime.value = S.T;
    G.sun.position.copy(G.sunDir).multiplyScalar(10);
    const rs = rocketState();
    rocket.position.copy(rs.pos); halo.position.copy(rs.pos);
    const heat = rs.heat != null ? clamp(rs.heat / S.plan.peak_heat, 0, 1) : 0;
    halo.material.color.setRGB(0.37 + 0.63 * heat, 0.9 - 0.45 * heat, 1 - 0.85 * heat);
    halo.scale.setScalar(0.07 + 0.08 * heat + (rs.thr ? 0.03 * rs.thr : 0));
    rocket.visible = halo.visible = p !== "setup";
    launchMark.visible = !A || S.T < 120;
    if (A) {
      const n = A.frames.findIndex(f => f[0] > S.T);
      ascentTrail.geometry.setDrawRange(0, n < 0 ? A.frames.length : n);
      orbitLine.visible = S.T >= A.duration;
    }
    if (A && S.T >= A.duration && p !== "descent" && (S.time - lastTrack > 0.4)) {   // ground track, next revolution
      lastTrack = S.time;
      const pts = [];
      for (let i = 0; i <= 240; i++) {
        const t = S.T + A.orbit.period * i / 240;
        pts.push(eciWorld(propagate(A.orbit, t).r).applyAxisAngle(Y, -OMEGA * t).normalize().multiplyScalar(1.0015));
      }
      setLine(groundTrack, pts);
    }
    groundTrack.visible = A && S.T >= A.duration && p !== "descent";
    if (S.plan) {
      const P = S.PF, i0 = S.plan.track[0];
      burnMark.position.copy(fixedToWorld(llaLocal(i0[P.lat], i0[P.lon], i0[P.alt]), S.T));
      burnMark.visible = S.T <= S.plan.t_burn;
      if (p === "descent" && S.T >= S.plan.t_burn) {
        const n = S.plan.track.findIndex(f => f[0] > S.T);
        descentLine.geometry.setDrawRange(0, n < 0 ? S.plan.track.length : n);
      }
    }
    siteRing.scale.setScalar(1 + 0.25 * Math.sin(S.time * 4));

    // camera: follow the rocket unless the user is steering
    if (S.follow && S.time > S.userHold && p !== "setup") {
      camDir.copy(rs.pos).normalize();
      if (p === "descent" && S.T > S.plan.t_burn) camDir.lerp(siteMark.getWorldPosition(new THREE.Vector3()).normalize(), smooth((S.T - S.plan.t_burn) / S.plan.tof * 1.6));
      const want = p === "descent" ? 1.25 + 0.6 * (1 - smooth((S.T - S.plan.t_burn) / S.plan.tof)) : p === "ascent" ? 1.8 + 0.8 * smooth(S.T / A.duration * 3) : 2.6;
      const cur = G.camera.position.clone();
      const d = cur.length() + (want - cur.length()) * Math.min(dtReal * 1.2, 1);
      G.camera.position.copy(cur.normalize().lerp(camDir.clone().add(new THREE.Vector3(0, 0.25, 0)).normalize(), Math.min(dtReal * 1.5, 1)).normalize().multiplyScalar(d));
    }
    if (S.zoomTo) {   // one-shot zoom, abandoned as soon as the user takes the controls
      const d = G.camera.position.length(), nd = d + (S.zoomTo - d) * Math.min(dtReal * 1.5, 1);
      G.camera.position.setLength(nd);
      if (Math.abs(nd - S.zoomTo) < 0.01 || S.userHold > S.time) S.zoomTo = null;
    }
    controls.update();

    // HTML tags (not over the pad camera)
    const onGlobe = !(p === "liftoff" && S.T < VIGNETTE);
    $("mTags").style.display = onGlobe ? "" : "none";
    placeTag(tag("rk", "LANDER-01", "#5ee6ff"), rs.pos, p !== "setup");
    placeTag(tag("site", S.site ? `LANDING SITE · ${S.site.lat.toFixed(2)}°, ${S.site.lon.toFixed(2)}°` : "", "#2fe39a"),
      siteMark.getWorldPosition(new THREE.Vector3()), !!S.site);
    placeTag(tag("launch", launchSite().name.toUpperCase(), "#ffb547"), launchMark.getWorldPosition(new THREE.Vector3()), launchMark.visible);
    placeTag(tag("burn", "DEORBIT BURN", "#ffb547"), burnMark.position, burnMark.visible && !!S.plan?.reachable);

    // UI text, a few times a second
    if (S.time - lastUi > 0.15) {
      lastUi = S.time;
      $("mClock").innerHTML = `<b>${A || p === "liftoff" ? fmtT(S.T) : "T−00:00:00"}</b> · warp ${p === "liftoff" || p === "setup" ? 1 : S.warp}×`;
      renderCard(rs); renderHoloStats(rs);
      if (p !== "setup") updateLeft();
      if (p === "ascent" || p === "liftoff") renderAscentPanel(rs);
      if (p === "descent") renderDescentPanel(rs);
      if (p === "orbit" && S.plan && S.T > S.plan.t_burn - 5 && !S.planBusy && S.site) selectSite(S.site.lat, S.site.lon);   // missed the burn: re-plan
      else if (p === "orbit" && S.plan) renderSite();
    }

    // render: main view (pad camera during the vignette, globe otherwise), then the hologram in its panel
    const w = innerWidth, h = innerHeight;
    R.setScissorTest(false); R.setViewport(0, 0, w, h);
    if (p === "liftoff" && S.T < VIGNETTE) {
      pad.camera.aspect = w / h;
      pad.update(S.T, A.frames, S.AF);
      pad.render(R, w, h);
    } else {
      G.camera.aspect = w / h; G.camera.updateProjectionMatrix();
      R.render(G.scene, G.camera);
    }
    const rect = $("mHolo").getBoundingClientRect();
    if (rect.width > 0 && rect.height > 0) {
      holo.update(dtReal, S.time);
      holo.camera.aspect = rect.width / rect.height; holo.camera.updateProjectionMatrix();
      R.autoClear = false;
      R.setScissorTest(true);
      R.setViewport(rect.left, h - rect.bottom, rect.width, rect.height);
      R.setScissor(rect.left, h - rect.bottom, rect.width, rect.height);
      R.clearDepth();
      R.render(holo.scene, holo.camera);
      R.setScissorTest(false); R.setViewport(0, 0, w, h);
      R.autoClear = true;
    }
    return true;
  }
  api.setRenderHook(frame);

  S.api = { liftOff, selectSite, commitLanding, step: frame };
  enter();
  setPhase("setup");
  aimAtLaunchSite();
  return { enter, exit, state: S };
}
