import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { LineSegments2 } from 'three/addons/lines/LineSegments2.js';
import { LineSegmentsGeometry } from 'three/addons/lines/LineSegmentsGeometry.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';

const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];

const state = {
  scenes: [],
  scene: null,
  info: null,
  cloud: null,
  points: null,
  crackPoints: null,
  geometry: null,
  material: null,
  hist: null,
  diag: 1,
  analysis: null,
  runs: [],
  activeRun: null,
  threshold: 0.4,
  highlight: -1,
  jobId: null,
  jobKind: null,
  pendingScene: null,
  displayMode: 'points',
  showCracks: true,
  crackLines: null,
  lineWidth: 3,
  meshObject: null,
  meshMaterials: [],
  meshData: null,
  meshInfo: null,
  meshLevel: 'std',
  meshJobLevel: null,
  gallery: [],
  lbItem: null,
  lbMode: 'overlay',
  fsPath: null,
};

const fmt = (n) => Number(n).toLocaleString('zh-CN');
const fmtLen = (v) => (v >= 1 ? v.toFixed(3) : v.toFixed(4));

// =============================================================================
// Three.js 初始化
// =============================================================================
const canvas = $('#canvas');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.setClearColor(0x05070a, 1);

const view = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 1000);
camera.position.set(3, 2, 5);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.09;
controls.screenSpacePanning = true;
controls.rotateSpeed = 0.75;
controls.zoomSpeed = 1.2;
// 右键留给浏览器(右键菜单/触控板手势会和拖拽打架),平移改到中键
controls.mouseButtons = {
  LEFT: THREE.MOUSE.ROTATE,
  MIDDLE: THREE.MOUSE.PAN,
  RIGHT: null,
};

// 点云用自定义 shader 不吃灯光;网格用 Lambert,需要两盏灯
view.add(new THREE.HemisphereLight(0xffffff, 0x2a2f38, 1.05));
const meshDirLight = new THREE.DirectionalLight(0xffffff, 0.55);
meshDirLight.position.set(0.6, 1.0, 0.4);
view.add(meshDirLight);

const VERT = `
precision highp float;
uniform float uPointSize;
uniform float uDiag;
uniform float uFocal;
attribute vec3 aColor;
attribute float aScore;
attribute float aLabel;
attribute float aNvis;
varying vec3 vColor;
varying float vScore;
varying float vLabel;
varying float vNvis;
void main() {
  vColor = aColor;
  vScore = aScore;
  vLabel = aLabel;
  vNvis = aNvis;
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  // 点的世界尺寸随场景尺度走,近大远小由投影焦距换算
  float world = uPointSize * uDiag * 0.01;
  gl_PointSize = clamp(world * uFocal / max(0.001, -mv.z), 1.0, 24.0);
  gl_Position = projectionMatrix * mv;
}`;

const FRAG = `
precision highp float;
uniform float uThreshold;
uniform float uDim;
uniform bool uHeat;
uniform bool uOnly;
uniform bool uCrackShow;
uniform float uHighlight;
varying vec3 vColor;
varying float vScore;
varying float vLabel;
varying float vNvis;

vec3 heat(float t) {
  return clamp(vec3(
    1.5 - abs(4.0 * t - 3.0),
    1.5 - abs(4.0 * t - 2.0),
    1.5 - abs(4.0 * t - 1.0)
  ), 0.0, 1.0);
}

void main() {
  vec2 d = gl_PointCoord - 0.5;
  if (dot(d, d) > 0.25) discard;

  bool crack = uCrackShow && vScore >= uThreshold;
  bool hl = uCrackShow && uHighlight > -0.5 && abs(vLabel - uHighlight) < 0.5;
  // 关掉「显示裂隙」时忽略「只显示裂隙点」,否则整屏会被清空
  if (uCrackShow && uOnly && !crack && !hl) discard;

  vec3 c;
  if (hl) {
    c = vec3(0.25, 1.0, 0.95);
  } else if (crack) {
    if (uHeat) {
      c = heat(vScore);
    } else {
      float t = clamp((vScore - uThreshold) / max(1e-4, 1.0 - uThreshold), 0.0, 1.0);
      c = mix(vec3(1.0, 0.78, 0.12), vec3(1.0, 0.12, 0.05), t);
    }
    // 被很少视角看到的点稍微压暗,减少投影边缘的假阳性
    c *= mix(0.6, 1.0, clamp(vNvis / 3.0, 0.0, 1.0));
  } else {
    c = vColor * uDim;
  }
  gl_FragColor = vec4(c, 1.0);
}`;

state.material = new THREE.ShaderMaterial({
  uniforms: {
    uThreshold: { value: 0.4 },
    uPointSize: { value: 0.3 },
    uDiag: { value: 1.0 },
    uFocal: { value: 800.0 },
    uDim: { value: 0.55 },
    uHeat: { value: true },
    uOnly: { value: false },
    uCrackShow: { value: true },
    uHighlight: { value: -1 },
  },
  vertexShader: VERT,
  fragmentShader: FRAG,
  depthTest: true,
  depthWrite: true,
});

// 网格模式下单独叠一层「只有裂隙点」的点云:同一个 geometry,材质只改 uOnly
state.crackMaterial = state.material.clone();
state.crackMaterial.uniforms.uOnly.value = true;

function syncCrackUniforms() {
  const src = state.material.uniforms, dst = state.crackMaterial.uniforms;
  for (const k of ['uThreshold', 'uPointSize', 'uDiag', 'uFocal', 'uHeat',
                   'uCrackShow', 'uHighlight', 'uDim']) {
    dst[k].value = src[k].value;
  }
}

function meshShade() {
  return 0.35 + 0.5 * state.material.uniforms.uDim.value;
}

function resize() {
  const vp = $('#viewport');
  const w = vp.clientWidth, h = vp.clientHeight;
  if (w === 0 || h === 0) return;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  const focal = (h * renderer.getPixelRatio()) /
    (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2));
  if (state.material) state.material.uniforms.uFocal.value = focal;
  if (state.crackMaterial) state.crackMaterial.uniforms.uFocal.value = focal;
  // 粗线材质要知道视口大小,否则线宽会按错误的分辨率换算
  if (state.crackLines) state.crackLines.material.resolution.set(w, h);
}
new ResizeObserver(resize).observe($('#viewport'));
resize();

renderer.setAnimationLoop(() => {
  controls.update();
  renderer.render(view, camera);
});

// =============================================================================
// 点云
// =============================================================================
function disposeCloud() {
  if (state.crackPoints) {
    view.remove(state.crackPoints);
    state.crackPoints = null;
  }
  if (state.points) {
    view.remove(state.points);
    state.points.geometry.dispose();
    state.points = null;
  }
  state.geometry = null;
  state.cloud = null;
  state.hist = null;
}

async function loadCloud(version = '') {
  const scene = state.scene;
  if (!scene) return;
  const meta = await fetchJson(`/api/scene/${encodeURIComponent(scene)}/cloud/meta`);
  const q = version ? `?v=${encodeURIComponent(version)}` : '';
  const parts = ['positions', 'colors', 'scores', 'nvis', 'labels'];
  const bufs = await Promise.all(parts.map(p =>
    fetch(`/api/scene/${encodeURIComponent(scene)}/cloud/${p}${q}`).then(r => {
      if (!r.ok) throw new Error(`读取 ${p} 失败: ${r.status}`);
      return r.arrayBuffer();
    })));

  const positions = new Float32Array(bufs[0]);
  const colors = new Uint8Array(bufs[1]);
  const scores = new Uint8Array(bufs[2]);
  const nvis = new Uint8Array(bufs[3]);
  const labels = new Int16Array(bufs[4]);

  disposeCloud();
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geo.setAttribute('aColor', new THREE.BufferAttribute(colors, 3, true));
  geo.setAttribute('aScore', new THREE.BufferAttribute(scores, 1, true));
  geo.setAttribute('aNvis', new THREE.BufferAttribute(nvis, 1, false));
  geo.setAttribute('aLabel', new THREE.BufferAttribute(labels, 1, false));
  geo.computeBoundingSphere();
  state.geometry = geo;

  const pts = new THREE.Points(geo, state.material);
  pts.frustumCulled = true;
  view.add(pts);
  state.points = pts;

  // 网格模式下显示这一层:同一个 geometry,只保留裂隙/高亮点
  const cpts = new THREE.Points(geo, state.crackMaterial);
  cpts.frustumCulled = true;
  cpts.visible = false;
  view.add(cpts);
  state.crackPoints = cpts;

  state.cloud = { meta, positions, colors, scores, nvis, labels };
  syncCrackUniforms();

  const hist = new Uint32Array(256);
  for (let i = 0; i < scores.length; i++) hist[scores[i]]++;
  state.hist = hist;

  const bbmin = meta.bbox_min, bbmax = meta.bbox_max;
  state.diag = Math.hypot(bbmax[0] - bbmin[0], bbmax[1] - bbmin[1], bbmax[2] - bbmin[2]);
  state.material.uniforms.uDiag.value = state.diag;
  frameCamera(bbmin, bbmax);
  $('#emptyHint').classList.add('hidden');
  $('#statPoints').textContent = fmt(meta.n);
  $('#statScene').textContent = `${state.scene} · ${state.info?.n_images ?? '?'} 张`;
  updateHighlightCount();
  applyDisplayMode();
  logLine(`点云就绪:${fmt(meta.n)} 点${meta.kind === 'preview' ? '(仅预览,未检测)' : ''}`, 'ok');
}

function frameCamera(bbmin, bbmax) {
  const min = new THREE.Vector3(...bbmin);
  const max = new THREE.Vector3(...bbmax);
  const center = min.clone().add(max).multiplyScalar(0.5);
  const radius = Math.max(0.001, min.distanceTo(max) * 0.5);
  const dir = new THREE.Vector3(0.72, 0.34, 1).normalize();
  camera.position.copy(center).add(dir.multiplyScalar(radius * 2.15));
  camera.near = radius * 0.002;
  camera.far = radius * 60;
  camera.updateProjectionMatrix();
  controls.target.copy(center);
  controls.update();
}

function focusCrack(c) {
  if (!c?.centroid) return;
  const center = new THREE.Vector3(...c.centroid);
  const dir = camera.position.clone().sub(controls.target).normalize();
  const dist = Math.max(state.diag * 0.16, 0.01);
  camera.position.copy(center).add(dir.multiplyScalar(dist));
  controls.target.copy(center);
  controls.update();
}

// =============================================================================
// 网格(切换显示 / 下载原始 PLY)
// =============================================================================
const MESH_LEVEL_LABEL = { std: '标准', high: '精细', ultra: '超细', full: '原始', tex: '贴图' };

function disposeMesh() {
  if (state.meshObject) {
    view.remove(state.meshObject);
    if (state.meshObject.isGroup) {
      state.meshObject.traverse(o => {
        if (o.geometry) o.geometry.dispose();
        if (o.material) {
          if (o.material.map) o.material.map.dispose();
          o.material.dispose();
        }
      });
    } else {
      state.meshObject.geometry.dispose();
      state.meshObject.material.dispose();
    }
  }
  state.meshObject = null;
  state.meshMaterials = [];
  state.meshData = null;
}

function setDisplayControls(mode) {
  $$('#displaySeg button').forEach(b =>
    b.classList.toggle('on', b.dataset.value === mode));
}

function applyDisplayMode() {
  const meshMode = state.displayMode === 'mesh' && !!state.meshObject;
  if (state.points) state.points.visible = !meshMode;
  if (state.crackPoints) state.crackPoints.visible = meshMode && state.showCracks;
  if (state.meshObject) state.meshObject.visible = meshMode;
  if (meshMode) {
    $('#statPoints').textContent = fmt(state.meshData.n_faces);
    $('#statPointsLabel').textContent = '网格面';
  } else if (state.cloud) {
    $('#statPoints').textContent = fmt(state.cloud.meta.n);
    $('#statPointsLabel').textContent = '显示点';
  }
}

async function loadMesh() {
  if (state.meshLevel === 'tex') return loadTexturedMesh();
  const scene = state.scene;
  const level = state.meshLevel;
  const info = await fetchJson(
    `/api/scene/${encodeURIComponent(scene)}/mesh/info?level=${level}`);
  const m = info.meta;
  if (!m) throw new Error('网格缓存还没生成');
  state.meshInfo = info;
  const v = encodeURIComponent(m.created || Date.now());
  const parts = ['positions', 'colors', 'indices', 'normals'];
  const bufs = await Promise.all(parts.map(p =>
    fetch(`/api/scene/${encodeURIComponent(scene)}/mesh/${p}?level=${level}&v=${v}`)
      .then(r => r.ok ? r.arrayBuffer() : null)));
  if (state.scene !== scene || state.meshLevel !== level || !bufs[0]) return;

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(bufs[0]), 3));
  if (bufs[3]) {
    geo.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(bufs[3]), 3));
  }
  geo.setIndex(new THREE.BufferAttribute(new Uint32Array(bufs[2]), 1));

  let mat;
  if (bufs[1]) {
    geo.setAttribute('color', new THREE.BufferAttribute(new Uint8Array(bufs[1]), 3, true));
    mat = new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide });
  } else {
    mat = new THREE.MeshLambertMaterial({ color: 0x9aa0a6, side: THREE.DoubleSide });
  }
  if (!bufs[3]) geo.computeVertexNormals();
  mat.color.setScalar(meshShade());

  disposeMesh();
  const obj = new THREE.Mesh(geo, mat);
  view.add(obj);
  state.meshObject = obj;
  state.meshMaterials = [mat];
  state.meshData = m;
  applyDisplayMode();
  logLine(`网格就绪(${MESH_LEVEL_LABEL[level] || level}):${fmt(m.n_faces)} 面 / ` +
    `${fmt(m.n_verts)} 顶点` +
    (m.decimated ? `(原始 ${fmt(m.source_faces)} 面)` : ''), 'ok');
}

// 贴图模式:用 OpenMVS 的 scene_textured.ply(带 UV)+ 降采样后的贴图。
// 按 texnumber 分组,每组一个 mesh,共用同一份 position/normal 属性。
async function loadTexturedMesh() {
  const scene = state.scene;
  const info = await fetchJson(`/api/scene/${encodeURIComponent(scene)}/tex/info`);
  const m = info.meta;
  if (!m) throw new Error('贴图缓存还没生成');
  state.meshInfo = info;
  const v = encodeURIComponent(m.created || Date.now());
  const parts = ['positions', 'normals', 'indices', 'uv'];
  const bufs = await Promise.all(parts.map(p =>
    fetch(`/api/scene/${encodeURIComponent(scene)}/tex/${p}?v=${v}`)
      .then(r => r.ok ? r.arrayBuffer() : null)));
  if (state.scene !== scene || state.meshLevel !== 'tex' || !bufs[0] || !bufs[2]) return;

  // 每组是一段独立的顶点(按 (顶点,UV) 去重后组内索引从 0 开始),
  // 位置/法线/UV 都按组切片,索引也用组内号。
  const pos = new Float32Array(bufs[0]);
  const nor = bufs[1] ? new Float32Array(bufs[1]) : null;
  const idx = new Uint32Array(bufs[2]);
  const uv = new Float32Array(bufs[3]);
  const loader = new THREE.TextureLoader();
  const maxAniso = renderer.capabilities.getMaxAnisotropy();
  const group = new THREE.Group();
  const mats = [];

  for (const g of m.groups) {
    const geo = new THREE.BufferGeometry();
    const vs = g.vstart * 3, ve = (g.vstart + g.vcount) * 3;
    geo.setAttribute('position',
      new THREE.BufferAttribute(pos.subarray(vs, ve), 3));
    if (nor) geo.setAttribute('normal',
      new THREE.BufferAttribute(nor.subarray(vs, ve), 3));
    geo.setAttribute('uv', new THREE.BufferAttribute(
      uv.subarray(g.vstart * 2, (g.vstart + g.vcount) * 2), 2));
    geo.setIndex(new THREE.BufferAttribute(
      idx.subarray(g.istart, g.istart + g.icount), 1));

    const tex = loader.load(
      `/api/scene/${encodeURIComponent(scene)}/tex/img/${g.tex}?v=${v}`);
    tex.colorSpace = THREE.SRGBColorSpace;
    // OpenMVS 的 PLY 纹理坐标是 OpenGL 约定(v=0 是图像底边),而 Three.js 的
    // 贴图默认 flipY=true 正好对应这个约定,别关掉。
    tex.flipY = true;
    tex.anisotropy = maxAniso;
    const mat = new THREE.MeshLambertMaterial({ map: tex, side: THREE.DoubleSide });
    mat.color.setScalar(meshShade());
    mats.push(mat);
    group.add(new THREE.Mesh(geo, mat));
  }

  disposeMesh();
  view.add(group);
  state.meshObject = group;
  state.meshMaterials = mats;
  state.meshData = m;
  applyDisplayMode();
  logLine(`贴图网格就绪:${fmt(m.n_faces)} 面,${m.groups.length} 组贴图,` +
    `贴图 ${m.tex_size}²`, 'ok');
}

async function ensureMesh() {
  const scene = state.scene;
  const level = state.meshLevel;
  try {
    if (level === 'tex') {
      const info = await fetchJson(`/api/scene/${encodeURIComponent(scene)}/tex/info`);
      state.meshInfo = info;
      if (info.ready) {
        await loadMesh();
        return true;
      }
      const { job_id } = await fetchJson('/api/tex/prepare', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ scene }),
      });
      state.meshJobLevel = level;
      watchJob(job_id, 'tex');
      logLine('生成贴图网格缓存:读取 scene_textured.ply 并把 16 张 8192² 贴图降到 4096²,' +
        '首次约 2 分钟,完成后自动切过去…', '');
      return true;
    }
    const info = await fetchJson(
      `/api/scene/${encodeURIComponent(scene)}/mesh/info?level=${level}`);
    state.meshInfo = info;
    if (info.ready) {
      await loadMesh();
      return true;
    }
    const { job_id } = await fetchJson('/api/mesh/prepare', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ scene, level }),
    });
    state.meshJobLevel = level;
    watchJob(job_id, 'mesh');
    logLine(`生成网格显示缓存(${MESH_LEVEL_LABEL[level] || level}):` +
      `原始 ${fmt(info.source_faces)} 面,完成后自动切到网格…`, '');
    return true;
  } catch (e) {
    logLine(`网格不可用:${e.message}`, 'warn');
    return false;
  }
}

async function setDisplayMode(mode) {
  if (mode === 'mesh') {
    if (!state.info?.has_mesh) {
      logLine('这个场景没有网格文件(先跑重建)。', 'warn');
      setDisplayControls(state.displayMode);
      return;
    }
    if (!state.meshObject) {
      const ok = await ensureMesh();
      if (!ok) { setDisplayControls(state.displayMode); return; }
      if (!state.meshObject) {
        // 缓存任务已经启动,结束时会自动切过去
        state.displayMode = 'mesh';
        setDisplayControls('mesh');
        return;
      }
    }
  }
  state.displayMode = mode;
  setDisplayControls(mode);
  applyDisplayMode();
}

function renderMeshOutput(info) {
  const el = $('#meshOutputs');
  if (info?.has_mesh) {
    const mb = info.mesh_bytes ? ` (${(info.mesh_bytes / 1e6).toFixed(0)} MB)` : '';
    el.innerHTML = `<a href="/api/scene/${encodeURIComponent(info.name)}/mesh/download" download>原始网格 PLY${mb}</a>`;
  } else {
    el.innerHTML = '';
  }
}

// =============================================================================
// 场景
// =============================================================================
async function fetchJson(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) {
    let msg = r.statusText;
    try { const j = await r.json(); msg = j.detail || msg; } catch {}
    throw new Error(msg);
  }
  return r.json();
}

async function refreshScenes(preserve = true) {
  const old = preserve ? state.scene : null;
  const { scenes } = await fetchJson('/api/scenes');
  state.scenes = scenes;
  const sel = $('#sceneSelect');
  sel.innerHTML = '';
  for (const s of scenes) {
    const o = document.createElement('option');
    o.value = s.name;
    o.textContent = `${s.name}  (${s.n_images || 0} 图)${s.has_crack ? ' ✓检测' : s.has_reconstruction ? ' ·已重建' : ''}`;
    sel.appendChild(o);
  }
  let pick = old && scenes.some(s => s.name === old) ? old :
    (scenes.find(s => s.has_crack)?.name || scenes.find(s => s.has_reconstruction)?.name || scenes[0]?.name);
  if (pick) {
    sel.value = pick;
    await selectScene(pick);
  } else {
    $('#emptyHint').classList.remove('hidden');
  }
}

async function selectScene(name) {
  if (!name) return;
  state.scene = name;
  state.info = await fetchJson(`/api/scene/${encodeURIComponent(name)}`);
  const info = state.info;
  state.analysis = null;
  state.highlight = -1;
  state.material.uniforms.uHighlight.value = -1;
  if (!info.has_crack) {
    // 预览点云没有分数,别让「只显示裂隙」把整屏清空
    $('#onlyCracks').checked = false;
    state.material.uniforms.uOnly.value = false;
    $('#heatmap').checked = true;
    state.material.uniforms.uHeat.value = true;
  }
  clearTable();
  $('#hlId').textContent = '—';
  $('#highlightBar').classList.add('hidden');
  $('#outputs').innerHTML = '';
  disposeMesh();
  disposeCrackLines();
  $('#showLines').checked = false;
  state.displayMode = 'points';
  setDisplayControls('points');
  renderMeshOutput(info);
  renderGallery(info.gallery || []);
  $('#statScene').textContent = `${name} · ${info.n_images ?? '?'} 张`;
  await loadRuns();

  if (info.has_crack) {
    const thr = info.meta?.default_threshold ?? 0.4;
    setThreshold(thr);
    $('#threshold').value = String(thr);
    logLine(`场景 ${name}:${fmt(info.meta?.n_points || 0)} 点,已检测 ${info.meta?.n_images_used} 张`, 'ok');
    await loadCloud(info.meta?.created || Date.now());
    loadSavedAnalysis();
  } else if (info.has_preview) {
    setThreshold(0.4);
    logLine(`场景 ${name}:预览点云`, '');
    await loadCloud(info.preview_meta?.created || Date.now());
  } else if (info.has_reconstruction) {
    disposeCloud();
    $('#emptyHint').classList.remove('hidden');
    $('#emptyHint h3').textContent = '正在生成预览点云…';
    $('#emptyHint p').textContent = '第一次打开场景需要读取稠密点云并降采样,大约 10 秒。';
    startPreview();
  } else {
    disposeCloud();
    $('#emptyHint').classList.remove('hidden');
    $('#emptyHint h3').textContent = '这个场景还不完整';
    $('#emptyHint p').textContent = '先跑重建,或选择另一个场景。';
  }
}

async function startPreview() {
  try {
    const { job_id } = await fetchJson('/api/preview', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ scene: state.scene }),
    });
    watchJob(job_id, 'preview');
  } catch (e) {
    logLine(`预览失败:${e.message}`, 'err');
  }
}

// =============================================================================
// 阈值/显示
// =============================================================================
function setThreshold(v) {
  state.threshold = v;
  state.material.uniforms.uThreshold.value = v;
  state.crackMaterial.uniforms.uThreshold.value = v;
  $('#thrOut').textContent = v.toFixed(2);
  $('#threshold').value = String(v);
  updateHighlightCount();
}

function updateHighlightCount() {
  if (!state.hist) { $('#statCracks').textContent = '—'; return; }
  const bin = Math.round(state.threshold * 255);
  let n = 0;
  for (let i = bin; i < 256; i++) n += state.hist[i];
  $('#statCracks').textContent = fmt(n);
}

$('#threshold').addEventListener('input', e => setThreshold(parseFloat(e.target.value)));
$('#pointSize').addEventListener('input', e => {
  const v = parseFloat(e.target.value);
  state.material.uniforms.uPointSize.value = v;
  state.crackMaterial.uniforms.uPointSize.value = v;
  $('#sizeOut').textContent = v.toFixed(2);
});
$('#dim').addEventListener('input', e => {
  const v = parseFloat(e.target.value);
  state.material.uniforms.uDim.value = v;
  state.crackMaterial.uniforms.uDim.value = v;
  state.meshMaterials.forEach(m => m.color.setScalar(meshShade()));
  $('#dimOut').textContent = v.toFixed(2);
});
$('#heatmap').addEventListener('change', e => {
  state.material.uniforms.uHeat.value = e.target.checked;
  state.crackMaterial.uniforms.uHeat.value = e.target.checked;
});
$('#showCracks').addEventListener('change', e => {
  state.showCracks = e.target.checked;
  state.material.uniforms.uCrackShow.value = state.showCracks;
  state.crackMaterial.uniforms.uCrackShow.value = state.showCracks;
  applyDisplayMode();
});
$('#showLines').addEventListener('change', e => {
  if (e.target.checked && !state.crackLines) buildCrackLines();
  if (state.crackLines) state.crackLines.visible = e.target.checked;
});
$('#lineWidth').addEventListener('input', e => {
  state.lineWidth = parseFloat(e.target.value);
  $('#lineWOut').textContent = state.lineWidth.toFixed(1);
  if (state.crackLines) state.crackLines.material.linewidth = state.lineWidth;
});
$$('#displaySeg button').forEach(b =>
  b.addEventListener('click', () => setDisplayMode(b.dataset.value)));

// 网格细节档位:标准 45 万 / 精细 150 万 / 超细 400 万 / 原始
$$('#meshDetailSeg button').forEach(b => b.addEventListener('click', async () => {
  const level = b.dataset.value;
  if (level === state.meshLevel) return;
  if (!$('#btnStop').disabled) {
    logLine('有任务正在运行,等它结束再切换网格细节。', 'warn');
    return;
  }
  $$('#meshDetailSeg button').forEach(x => x.classList.toggle('on', x === b));
  state.meshLevel = level;
  if (state.displayMode !== 'mesh' || !state.scene) {
    logLine(`网格细节已切到「${MESH_LEVEL_LABEL[level] || level}」,切到网格时生效。`, '');
    return;
  }
  disposeMesh();
  applyDisplayMode();
  const ok = await ensureMesh();
  if (!ok) {
    state.displayMode = 'points';
    setDisplayControls('points');
    applyDisplayMode();
  }
}));
$('#onlyCracks').addEventListener('change', e => state.material.uniforms.uOnly.value = e.target.checked);
$('#btnResetView').addEventListener('click', () => {
  if (state.cloud) frameCamera(state.cloud.meta.bbox_min, state.cloud.meta.bbox_max);
});
$('#clearHighlight').addEventListener('click', () => setHighlight(-1));

function setHighlight(id) {
  state.highlight = id;
  state.material.uniforms.uHighlight.value = id;
  state.crackMaterial.uniforms.uHighlight.value = id;
  $('#highlightBar').classList.toggle('hidden', id < 0);
  if (id >= 0) {
    const crack = state.analysis?.cracks?.find(c => c.id === id);
    $('#hlId').textContent = `#${crack?.crack_id ?? id + 1}`;
    focusCrack(crack);
  }
  $$('#crackTable tbody tr').forEach(tr =>
    tr.classList.toggle('on', Number(tr.dataset.id) === id));
}

// =============================================================================
// 任务 / SSE
// =============================================================================
let es = null;

function setBusy(busy, kind) {
  const label = { detect: '检测中', analyze: '分析中', reconstruct: '重建中',
    preview: '生成预览', mesh: '生成网格', tex: '生成贴图' }[kind] || '运行中';
  const chip = $('#jobChip');
  chip.className = 'chip ' + (busy ? 'busy' : 'idle');
  chip.textContent = busy ? label : '空闲';
  $('#btnDetect').disabled = busy;
  $('#btnAnalyze').disabled = busy;
  $('#btnReconstruct').disabled = busy;
  $('#btnStop').disabled = !busy;
  $('#sceneSelect').disabled = busy;
  if (busy) {
    setProgress(0, '');
    $('#logTitle').textContent = `任务日志 · ${label}`;
  } else {
    $('#logTitle').textContent = '任务日志';
  }
}

function watchJob(jobId, kind) {
  if (es) es.close();
  state.jobId = jobId;
  state.jobKind = kind;
  setBusy(true, kind);
  es = new EventSource(`/api/jobs/${jobId}/events`);
  es.onmessage = (e) => {
    let ev;
    try { ev = JSON.parse(e.data); } catch { return; }
    handleEvent(ev, kind);
    if (ev.event === 'end') {
      es.close(); es = null;
      setBusy(false, kind);
      onJobEnd(ev, kind);
    }
  };
  es.onerror = () => {
    // 浏览器会自动重连;若任务已经结束,onmessage 会先收到 end
  };
}

function handleEvent(ev, kind) {
  if (ev.message) {
    const cls = ev.event === 'error' ? 'err' : ev.event === 'warn' ? 'warn' :
      ev.event === 'done' || ev.event === 'preview_done' ? 'ok' : '';
    logLine(ev.message, cls);
  }
  if (ev.event === 'progress' && ev.total) {
    const frac = Math.max(0, Math.min(1, (ev.done || 0) / ev.total));
    const pct = `${Math.round(frac * 100)}%`;
    let text = pct;
    if (ev.step_title) text = `${pct} · 步骤 ${ev.step}: ${ev.step_title}`;
    else if (ev.total !== 100) text = `${pct} · ${ev.done}/${ev.total}`;
    setProgress(frac, text);
  }
  if (ev.event === 'result' && ev.kind === 'analyze') {
    applyAnalysis(ev.result);
  }
  if (ev.event === 'preview_done') {
    if (ev.scene && ev.scene !== state.scene) return;
    loadCloud(Date.now()).then(() => {
      $('#emptyHint h3').textContent = '还没有可显示的点云';
      $('#emptyHint p').textContent = '在左侧从照片重建,或选择一个已完成的场景。';
    }).catch(e => logLine(e.message, 'err'));
  }
  if (ev.event === 'reconstruct_done') {
    state.pendingScene = ev.scene || null;
  }
  if (ev.event === 'error') {
    setProgress(0, '失败');
  }
}

async function onJobEnd(ev, kind) {
  if (ev.status === 'error') {
    $('#jobChip').className = 'chip';
    $('#jobChip').textContent = '失败';
    if (kind === 'mesh' || kind === 'tex') {
      state.displayMode = 'points';
      setDisplayControls('points');
      applyDisplayMode();
    }
    return;
  }
  if (kind === 'mesh' || kind === 'tex') {
    const jobLevel = state.meshJobLevel;
    state.meshJobLevel = null;
    if (jobLevel && jobLevel !== state.meshLevel) {
      logLine(`网格缓存(${MESH_LEVEL_LABEL[jobLevel] || jobLevel})已生成;` +
        `当前选的是「${MESH_LEVEL_LABEL[state.meshLevel] || state.meshLevel}」,切一下档位再加载。`, '');
      state.displayMode = 'points';
      setDisplayControls('points');
      applyDisplayMode();
      return;
    }
    try {
      await loadMesh();
      state.displayMode = 'mesh';
      setDisplayControls('mesh');
      applyDisplayMode();
    } catch (e) {
      logLine(`网格加载失败:${e.message}`, 'err');
      state.displayMode = 'points';
      setDisplayControls('points');
      applyDisplayMode();
    }
    return;
  }
  if (kind === 'detect') {
    state.analysis = null;
    clearTable();
    await refreshScenes(true);
    logLine('检测完成。点「分析并导出」生成裂隙列表和 PLY/CSV。', 'ok');
  }
  if (kind === 'reconstruct') {
    if (state.pendingScene) state.scene = state.pendingScene;
    state.pendingScene = null;
    await refreshScenes(true);
    logLine(`重建完成:${state.scene}`, 'ok');
  }
}

function setProgress(frac, text) {
  $('#progressBar').style.width = `${Math.max(0, Math.min(1, frac)) * 100}%`;
  $('#progressText').textContent = text || '';
}

let logCount = 0;
function logLine(msg, cls = '') {
  if (!msg) return;
  const el = $('#log');
  const line = document.createElement('div');
  if (cls) line.className = cls;
  const t = new Date().toLocaleTimeString('zh-CN', { hour12: false });
  line.textContent = `[${t}] ${msg}`;
  el.appendChild(line);
  logCount++;
  if (logCount > 2500) { el.removeChild(el.firstChild); logCount--; }
  el.scrollTop = el.scrollHeight;
}
$('#btnClearLog').addEventListener('click', () => { $('#log').innerHTML = ''; logCount = 0; });
$('#btnStop').addEventListener('click', async () => {
  if (!state.jobId) return;
  try { await fetchJson(`/api/jobs/${state.jobId}/stop`, { method: 'POST' }); }
  catch (e) { logLine(e.message, 'err'); }
});

// =============================================================================
// 检测 / 分析 / 重建
// =============================================================================
let detectMode = 'fast';
$$('#modeSeg button').forEach(b => b.addEventListener('click', () => {
  $$('#modeSeg button').forEach(x => x.classList.remove('on'));
  b.classList.add('on');
  detectMode = b.dataset.value;
}));

let detectDevice = 'gpu';
$$('#deviceSeg button').forEach(b => b.addEventListener('click', () => {
  $$('#deviceSeg button').forEach(x => x.classList.remove('on'));
  b.classList.add('on');
  detectDevice = b.dataset.value;
}));

function segValue(sel) {
  return $(`${sel} button.on`)?.dataset.value;
}
['#matcherSeg', '#relaxedSeg', '#meshSeg'].forEach(sel => {
  $$(`${sel} button`).forEach(b => b.addEventListener('click', () => {
    $$(`${sel} button`).forEach(x => x.classList.remove('on'));
    b.classList.add('on');
  }));
});

$('#btnDetect').addEventListener('click', async () => {
  if (!state.scene) return;
  const models = $$('#modelGrid input:checked').map(i => i.value);
  if (!models.length) { logLine('至少选一个模型。', 'warn'); return; }
  const limitRaw = $('#limitInput').value.trim();
  const body = {
    scene: state.scene,
    models,
    mode: detectMode,
    limit: limitRaw ? Math.max(1, parseInt(limitRaw, 10)) : null,
    gallery: $('#galleryCheck').checked,
    device: detectDevice,
  };
  const names = models.map(m => MODEL_SHORT[m] || m).join(' + ');
  try {
    logLine(`开始检测:${names} / ${body.mode} / ${body.limit ?? '全部'} 张` +
            (models.length > 1 ? `(共 ${models.length} 个模型)` : ''), '');
    const { job_id } = await fetchJson('/api/detect', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    watchJob(job_id, 'detect');
  } catch (e) {
    logLine(`检测启动失败:${e.message}`, 'err');
  }
});

$('#btnAnalyze').addEventListener('click', async () => {
  if (!state.scene) return;
  if (!state.info?.has_crack) { logLine('先做裂隙检测,再分析。', 'warn'); return; }
  const num = (sel) => {
    const v = $(sel).value.trim();
    return v === '' ? null : parseFloat(v);
  };
  const body = {
    scene: state.scene,
    threshold: state.threshold,
    min_length: num('#minLength'),
    gap: num('#gapInput'),
    scale: num('#scaleInput'),
    write_outputs: true,
  };
  try {
    logLine(`分析:阈值 ${state.threshold.toFixed(2)}`, '');
    const { job_id } = await fetchJson('/api/analyze', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    watchJob(job_id, 'analyze');
  } catch (e) {
    logLine(`分析启动失败:${e.message}`, 'err');
  }
});

$('#btnReconstruct').addEventListener('click', async () => {
  const photos = $('#photosDir').value.trim();
  let scene = $('#newScene').value.trim();
  if (!photos) { logLine('先填照片文件夹。', 'warn'); return; }
  if (!scene) {
    scene = photos.replace(/[\\/]+$/, '').split(/[\\/]/).pop().replace(/[^A-Za-z0-9_\-\u4e00-\u9fa5]/g, '_');
    $('#newScene').value = scene;
  }
  const resLevel = parseInt($('#resLevel').value, 10);
  const maxRes = parseInt($('#maxRes').value, 10);
  const body = {
    photos_dir: photos,
    scene,
    force: $('#forceRecon').checked,
    matcher: segValue('#matcherSeg') || 'exhaustive',
    colmap_relaxed: parseInt(segValue('#relaxedSeg') ?? '1', 10),
    resolution_level: Number.isFinite(resLevel) ? Math.max(0, Math.min(3, resLevel)) : 1,
    max_resolution: Number.isFinite(maxRes) ? Math.max(640, maxRes) : 2000,
    do_refine: $('#doRefine').checked ? 1 : 0,
    mesh_quality: segValue('#meshSeg') || 'standard',
    mesh_only: $('#meshOnly').checked,
  };
  const meshLabel = { standard: '标准', fine: '精细', ultra: '极细' }[body.mesh_quality] || body.mesh_quality;
  try {
    logLine(`开始重建:${photos} -> work/${scene}`, '');
    logLine(`参数: matcher=${body.matcher} COLMAP阈值=${body.colmap_relaxed ? '放宽' : '官方'} ` +
            `稠密层级=${body.resolution_level} 最大边长=${body.max_resolution} ` +
            `网格细化=${body.do_refine ? '开' : '关'} 网格精度=${meshLabel} ` +
            `只重做网格=${body.mesh_only ? '是' : '否'} 强制=${body.force ? '是' : '否'}`, '');
    const existing = (state.scenes || []).find(s => s.name === scene);
    if (existing && !body.force && !body.mesh_only) {
      logLine('提示:场景已存在且未勾选强制重跑,已完成步骤会被跳过,参数改动不会生效。', 'warn');
    } else if (existing && body.force) {
      const ok = window.confirm(
        `场景「${scene}」已存在。强制重跑会删除并重建 work\\${scene}\\sparse 和 dense\\,` +
        `以及 OpenMVS 输出,之前的检测/裂隙结果也会失效。\n\n确定继续?`);
      if (!ok) { logLine('已取消重建。', 'warn'); return; }
    } else if (existing && body.mesh_only) {
      const ok = window.confirm(
        `只重做 work\\${scene}\\mvs 下的网格和纹理(复用稀疏/稠密点云)。\n` +
        `旧的网格文件名不变,网页会按修改时间重新生成预览。\n\n确定继续?`);
      if (!ok) { logLine('已取消重建。', 'warn'); return; }
    }
    const { job_id } = await fetchJson('/api/reconstruct', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    watchJob(job_id, 'reconstruct');
  } catch (e) {
    logLine(`重建启动失败:${e.message}`, 'err');
  }
});

// =============================================================================
// 分析结果 / 表格
// =============================================================================
function clearTable() {
  $('#crackTable tbody').innerHTML = '';
  $('#crackCount').textContent = '0';
  $('#tableEmpty').classList.remove('hidden');
}

// =============================================================================
// 检测记录(每次检测一条,可改名、切换)
// =============================================================================
const MODEL_SHORT = {
  tom: 'Tomt996', tom202: 'Tomt996-s202', tom303: 'Tomt996-s303',
  tom404: 'Tomt996-s404', tom505: 'Tomt996-s505',
  dihao: 'CrackEvo', segformer: 'SegFormer-b0', ensemble: '集成',
};

async function loadRuns() {
  if (!state.scene) return;
  try {
    const res = await fetchJson(`/api/scene/${encodeURIComponent(state.scene)}/runs`);
    state.runs = res.runs || [];
    state.activeRun = res.active;
  } catch {
    state.runs = [];
    state.activeRun = null;
  }
  renderRuns();
}

function renderRuns() {
  const box = $('#runList');
  const runs = state.runs || [];
  if (!runs.length) {
    box.innerHTML = '<p class="note center">还没有检测记录</p>';
    return;
  }
  box.innerHTML = '';
  for (const r of runs) {
    const el = document.createElement('div');
    el.className = 'run-item' + (r.id === state.activeRun ? ' active' : '');
    const model = MODEL_SHORT[r.model] || r.model_label || '—';
    const bits = [model,
                  r.n_images_used != null ? `${r.n_images_used} 张` : null,
                  r.has_analysis ? '已分析' : null].filter(Boolean).join(' · ');
    const name = r.name || r.id;
    el.innerHTML = '<div class="run-top"><span class="run-name"></span>' +
      (r.id === state.activeRun ? '<span class="run-tag">当前</span>' : '') +
      '<button class="rename" title="重命名">改名</button></div><div class="run-sub"></div>';
    el.querySelector('.run-name').textContent = name;
    el.querySelector('.run-sub').textContent =
      `${r.created ? String(r.created).replace('T', ' ') : r.id} · ${bits}`;
    el.title = `记录 ID:${r.id}`;
    el.addEventListener('click', () => switchRun(r.id));
    el.querySelector('.rename').addEventListener('click', async (ev) => {
      ev.stopPropagation();
      const nn = prompt('重命名这条检测记录', name);
      if (nn == null) return;
      try {
        await fetchJson(
          `/api/scene/${encodeURIComponent(state.scene)}/runs/${encodeURIComponent(r.id)}/rename`,
          { method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: nn }) });
        await loadRuns();
        logLine(`检测记录已改名为「${nn.trim()}」`, 'ok');
      } catch (e) {
        logLine(`改名失败:${e.message}`, 'err');
      }
    });
    box.appendChild(el);
  }
}

async function switchRun(id) {
  if (!state.scene || id === state.activeRun) return;
  if (!$('#btnStop').disabled) {
    logLine('有任务正在运行,等结束再切换检测记录。', 'warn');
    return;
  }
  try {
    await fetchJson(
      `/api/scene/${encodeURIComponent(state.scene)}/runs/${encodeURIComponent(id)}/activate`,
      { method: 'POST' });
    logLine(`切换到检测记录 ${id}`, '');
    await selectScene(state.scene);
  } catch (e) {
    logLine(`切换检测记录失败:${e.message}`, 'err');
  }
}

async function loadSavedAnalysis() {
  const scene = state.scene;
  try {
    const res = await fetchJson(`/api/scene/${encodeURIComponent(scene)}/analysis`);
    if (state.scene !== scene || !res.cracks?.length) return;
    if (res.stale) {
      logLine('磁盘上有上次的分析结果,但之后重新检测过;点「分析并导出」更新列表。', 'warn');
      return;
    }
    applyAnalysis(res, true);
    logLine(`已载入上次分析结果:${res.cracks.length} 条裂隙(阈值 ${res.threshold})`, 'ok');
  } catch { /* 还没有分析过 */ }
}

function disposeCrackLines() {
  if (!state.crackLines) return;
  view.remove(state.crackLines);
  state.crackLines.geometry.dispose();
  state.crackLines.material.dispose();
  state.crackLines = null;
}

// 用分析结果里每条裂隙的折线(state.analysis.cracks[].polyline)画彩色线
function buildCrackLines() {
  disposeCrackLines();
  const cracks = state.analysis?.cracks || [];
  const pos = [], col = [];
  const c = new THREE.Color();
  for (const cr of cracks) {
    const pts = cr.polyline;
    if (!pts || pts.length < 2) continue;
    c.setHSL(((cr.id ?? 0) * 0.618034) % 1, 0.85, 0.62);
    for (let i = 0; i < pts.length - 1; i++) {
      pos.push(pts[i][0], pts[i][1], pts[i][2],
               pts[i + 1][0], pts[i + 1][1], pts[i + 1][2]);
      col.push(c.r, c.g, c.b, c.r, c.g, c.b);
    }
  }
  if (!pos.length) return;
  // 用 LineSegments2(粗线):WebGL 原生线宽大多数平台锁死在 1px,
  // 这个材质把每段线扩成屏幕空间里的四边形,线宽可调。
  const geo = new LineSegmentsGeometry();
  geo.setPositions(new Float32Array(pos));
  geo.setColors(new Float32Array(col));
  const vp = $('#viewport');
  // 点云是带点大小的 splat,线贴在同一层表面容易被盖住;当成标注层画在最上面
  const mat = new LineMaterial({
    vertexColors: true, linewidth: state.lineWidth,
    transparent: true, opacity: 0.95,
    depthWrite: false, depthTest: false,
  });
  mat.resolution.set(Math.max(vp.clientWidth, 1), Math.max(vp.clientHeight, 1));
  state.crackLines = new LineSegments2(geo, mat);
  state.crackLines.frustumCulled = false;
  state.crackLines.renderOrder = 3;
  state.crackLines.visible = $('#showLines').checked;
  view.add(state.crackLines);
}

function applyAnalysis(res, quiet = false) {
  state.analysis = res;
  const cracks = res.cracks || [];
  $('#crackCount').textContent = String(cracks.length);
  const tbody = $('#crackTable tbody');
  tbody.innerHTML = '';
  const scale = res.scale_m_per_unit;
  for (const c of cracks) {
    const tr = document.createElement('tr');
    tr.dataset.id = String(c.id);
    const len = scale ? `${c.length_m.toFixed(3)} m` : fmtLen(c.length);
    const wid = scale ? `${c.width_m.toFixed(3)} m` : fmtLen(c.width);
    tr.innerHTML = `<td>${c.crack_id ?? c.id + 1}</td><td>${len}</td><td>${wid}</td>
      <td>${c.azimuth_deg.toFixed(1)}</td><td>${fmt(c.n_points)}</td>
      <td>${c.mean_score.toFixed(2)}</td>`;
    tr.title = `中心 (${c.centroid.map(v => v.toFixed(2)).join(', ')})\n倾角 ${c.elevation_deg.toFixed(1)}°`;
    tr.addEventListener('click', () => setHighlight(c.id));
    tbody.appendChild(tr);
  }
  $('#tableEmpty').classList.toggle('hidden', cracks.length > 0);

  const out = res.outputs || {};
  const items = [['全点云 PLY', out.ply], ['纯裂隙 PLY', out.ply_only],
                 ['裂隙线 PLY', out.lines_ply], ['裂隙线 CSV', out.lines_csv],
                 ['CSV 属性', out.csv], ['JSON', out.json]]
    .filter(([, p]) => p);
  $('#outputs').innerHTML = items.map(([label, p]) =>
    `<a href="/api/download?path=${encodeURIComponent(p)}" download>${label}</a>`).join('');
  if (!quiet) {
    if (!cracks.length) logLine('当前参数下没有满足条件的裂隙。降低阈值或最短长度再试。', 'warn');
    else logLine(`分析完成:${cracks.length} 条裂隙,${fmt(res.n_crack_points)} 个点`, 'ok');
  }

  // 重建裂隙线几何(旧的分析结果没有 polyline 时会自动跳过)
  buildCrackLines();
  loadRuns();      // 更新列表里的「已分析」标记

  // 重新加载连通域标签,让点选/表格联动生效
  if (state.scene) {
    fetch(`/api/scene/${encodeURIComponent(state.scene)}/cloud/labels?v=${Date.now()}`)
      .then(r => r.ok ? r.arrayBuffer() : null)
      .then(buf => {
        if (!buf || !state.geometry) return;
        state.geometry.setAttribute('aLabel', new THREE.BufferAttribute(new Int16Array(buf), 1, false));
        logLine('连通域标签已更新', '');
      }).catch(() => {});
  }
}

// =============================================================================
// 2D 叠加图
// =============================================================================
function renderGallery(items) {
  state.gallery = items || [];
  $('#galleryCount').textContent = String(state.gallery.length);
  const grid = $('#galleryGrid');
  grid.innerHTML = '';
  for (const it of state.gallery) {
    const fig = document.createElement('figure');
    const img = document.createElement('img');
    img.loading = 'lazy';
    const fname = (it.overlay || '').split('/').pop();
    img.src = `/api/scene/${encodeURIComponent(state.scene)}/gallery/${encodeURIComponent(fname)}`;
    img.alt = it.stem;
    const cap = document.createElement('figcaption');
    cap.textContent = it.stem;
    fig.appendChild(img);
    fig.appendChild(cap);
    fig.addEventListener('click', () => openLightbox(it));
    grid.appendChild(fig);
  }
  $('#galleryEmpty').classList.toggle('hidden', state.gallery.length > 0);
}

function openLightbox(it) {
  state.lbItem = it;
  state.lbMode = 'overlay';
  $('#lightboxName').textContent = it.stem;
  $('#lbOverlay').classList.add('on');
  $('#lbProb').classList.remove('on');
  $('#lightboxImg').src = galleryUrl(it, 'overlay');
  $('#lightbox').classList.remove('hidden');
}
function galleryUrl(it, kind) {
  const p = kind === 'prob' ? it.heatmap : it.overlay;
  const fname = (p || '').split('/').pop();
  return `/api/scene/${encodeURIComponent(state.scene)}/gallery/${encodeURIComponent(fname)}`;
}
$('#lbOverlay').addEventListener('click', () => {
  if (!state.lbItem) return;
  $('#lbOverlay').classList.add('on'); $('#lbProb').classList.remove('on');
  $('#lightboxImg').src = galleryUrl(state.lbItem, 'overlay');
});
$('#lbProb').addEventListener('click', () => {
  if (!state.lbItem) return;
  $('#lbProb').classList.add('on'); $('#lbOverlay').classList.remove('on');
  $('#lightboxImg').src = galleryUrl(state.lbItem, 'prob');
});
$('#lightboxClose').addEventListener('click', () => $('#lightbox').classList.add('hidden'));
$('#lightbox').addEventListener('click', (e) => {
  if (e.target.id === 'lightbox') $('#lightbox').classList.add('hidden');
});

// =============================================================================
// 标签页
// =============================================================================
$('#tabCracks').addEventListener('click', () => switchTab('cracks'));
$('#tabGallery').addEventListener('click', () => switchTab('gallery'));
function switchTab(which) {
  const cracks = which === 'cracks';
  $('#tabCracks').classList.toggle('on', cracks);
  $('#tabGallery').classList.toggle('on', !cracks);
  $('#paneCracks').classList.toggle('hidden', !cracks);
  $('#paneGallery').classList.toggle('hidden', cracks);
}

// =============================================================================
// 文件夹浏览
// =============================================================================
$('#browsePhotos').addEventListener('click', () => {
  const start = $('#photosDir').value.trim();
  openFs(start);   // 留空时从用户主目录开始
});
async function openFs(path) {
  try {
    const r = await fetchJson(`/api/fs?path=${encodeURIComponent(path)}`);
    state.fsPath = r.path;
    $('#modalTitle').textContent = '选择照片文件夹';
    $('#modalPath').textContent = r.path;
    $('#modalUp').disabled = !r.parent;
    $('#modalUp').onclick = () => r.parent && openFs(r.parent);
    const list = $('#modalList');
    list.innerHTML = '';
    for (const d of r.dirs) {
      const b = document.createElement('button');
      b.textContent = '📁 ' + d;
      b.addEventListener('click', () => openFs(joinPath(r.path, d)));
      list.appendChild(b);
    }
    if (!r.dirs.length) {
      const p = document.createElement('div');
      p.className = 'note'; p.style.padding = '10px';
      p.textContent = '（没有子文件夹）';
      list.appendChild(p);
    }
    $('#modal').classList.remove('hidden');
  } catch (e) {
    logLine(`打不开目录:${e.message}`, 'err');
  }
}
function joinPath(a, b) {
  const sep = a.includes('\\') ? '\\' : '/';
  return a.replace(/[\\/]+$/, '') + sep + b;
}
$('#modalPick').addEventListener('click', () => {
  if (state.fsPath) $('#photosDir').value = state.fsPath;
  $('#modal').classList.add('hidden');
});
$('#modalClose').addEventListener('click', () => $('#modal').classList.add('hidden'));
$('#modal').addEventListener('click', (e) => {
  if (e.target.id === 'modal') $('#modal').classList.add('hidden');
});

// =============================================================================
// 启动
// =============================================================================
$('#sceneSelect').addEventListener('change', e => selectScene(e.target.value));
$('#refreshScenes').addEventListener('click', () => refreshScenes(true).catch(e => logLine(e.message, 'err')));

refreshScenes(false).catch(e => logLine(`加载场景失败:${e.message}`, 'err'));
logLine('就绪。选择场景,或从照片开始重建。', 'ok');

// 调试/自动化用,不影响界面
window.__crack = {
  state, camera, controls, view, renderer, selectScene, loadCloud,
  setDisplayMode, loadMesh,
};
