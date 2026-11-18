# -*- coding: utf-8 -*-
"""裂隙检测 + 投影到稠密点云。

一次遍历完成三件事,避免把 88 张全分辨率概率图落盘(约 5 GB):

1. 对每张 undistort 后的照片跑裂缝分割模型,得到 0~1 概率图;
2. 用 COLMAP 相机把点云投到该图,并做遮挡剔除,取可见点的概率;
3. 逐点累加 ``sum(prob)`` 和 ``n_visible``,最终 score = sum / n_visible。

产物(全部在 ``work/<scene>/crack/`` 下):

- ``scores.npz``   float16 score[N]、uint16 n_visible[N]、float16 max_prob[N]
- ``display.npz``  float32 positions[M,3]、uint8 colors[M,3]、uint8 scores[M]、uint8 n_visible[M]
- ``gallery/``     降采样后的 2D 叠加图/热力图
- ``meta.json``    运行参数、统计量、gallery 索引

命令行::

    python tools/crack_app/detect_project.py --scene rock --model tom --mode fast --limit 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

TOOLS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(TOOLS_DIR)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import mask_to_points as m2p          # noqa: E402
import bbox_to_mesh as b2m            # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

_MC = os.path.join(REPO_ROOT, 'models', 'crack')
MODEL_FILES = {
    'tom': os.path.join(_MC, 'tomt996_seed101.onnx'),
    'tom202': os.path.join(_MC, 'tomt996_seed202.onnx'),
    'tom303': os.path.join(_MC, 'tomt996_seed303.onnx'),
    'tom404': os.path.join(_MC, 'tomt996_seed404.onnx'),
    'tom505': os.path.join(_MC, 'tomt996_seed505.onnx'),
    'dihao': os.path.join(_MC, 'dihao', 'crackevo_unet_r34_A.onnx'),
    'segformer': os.path.join(_MC, 'varcoder', 'segformer_b0_crack.onnx'),
}
MODEL_LABELS = {
    'tom': 'Tomt996 UNet-R34 (seed101)',
    'tom202': 'Tomt996 UNet-R34 (seed202)',
    'tom303': 'Tomt996 UNet-R34 (seed303)',
    'tom404': 'Tomt996 UNet-R34 (seed404)',
    'tom505': 'Tomt996 UNet-R34 (seed505)',
    'dihao': 'CrackEvo UNet-R34',
    'segformer': 'SegFormer-b0 (CrackDataset)',
    'ensemble': '集成 (Tomt996 + CrackEvo)',
}
DEFAULT_THRESHOLD = {'tom': 0.50, 'tom202': 0.50, 'tom303': 0.50,
                     'tom404': 0.50, 'tom505': 0.50,
                     'dihao': 0.30, 'segformer': 0.35, 'ensemble': 0.40}

# 原生分辨率切块参数 (高, 宽, stride_h, stride_w)
MODEL_TILES = {
    'tom': (378, 672, 283, 504),
    'tom202': (378, 672, 283, 504),
    'tom303': (378, 672, 283, 504),
    'tom404': (378, 672, 283, 504),
    'tom505': (378, 672, 283, 504),
    'dihao': (512, 512, 384, 384),
    'segformer': (512, 512, 384, 384),
}


# =============================================================================
# 路径
# =============================================================================
def scene_paths(scene, root=None):
    """scene 可以是名字 (rock) 或 work 目录下的路径。"""
    root = root or os.path.join(REPO_ROOT, 'work')
    path = scene if os.path.isdir(scene) and os.path.sep in scene else os.path.join(root, scene)
    name = os.path.basename(os.path.normpath(path))
    return {
        'name': name,
        'root': path,
        'sparse': os.path.join(path, 'dense', 'sparse'),
        'images': os.path.join(path, 'dense', 'images'),
        'cloud': os.path.join(path, 'mvs', 'scene_dense.ply'),
        'mesh': os.path.join(path, 'mvs', 'scene_dense_mesh.ply'),
        'crack': os.path.join(path, 'crack'),
        'gallery': os.path.join(path, 'crack', 'gallery'),
    }


def check_scene(paths):
    missing = [p for k, p in paths.items()
               if k in ('sparse', 'images', 'cloud', 'mesh') and not os.path.exists(p)]
    if missing:
        raise SystemExit('场景不完整,缺少:\n  ' + '\n  '.join(missing) +
                         '\n先跑 run_pipeline.sh 或 GUI 里的「从照片重建」。')


# =============================================================================
# ONNX 模型
# =============================================================================
def _find_cuda_dll_dirs():
    """onnxruntime-gpu 需要 cuDNN 9 / CUDA 12 的 DLL。

    本机这些 DLL 由 torch 的 cu128 包提供(``torch/lib``)。按顺序找:
    环境变量 ``CRACK_CUDA_DLL_DIR`` → 当前环境的 torch → 其他 conda 环境里的 torch
    (例如 D 盘 face_rec)。只返回确实含 ``cudnn64_9.dll`` 的目录。
    """
    cands = [os.environ.get('CRACK_CUDA_DLL_DIR', ''),
             os.path.join(sys.prefix, 'Lib', 'site-packages', 'torch', 'lib')]
    for root in (r'D:\ProgramData\anaconda3\envs',
                 os.path.expanduser(r'~\.conda\envs')):
        try:
            for name in os.listdir(root):
                cands.append(os.path.join(
                    root, name, 'Lib', 'site-packages', 'torch', 'lib'))
        except OSError:
            pass
    found = []
    for d in cands:
        if d and d not in found and os.path.isfile(
                os.path.join(d, 'cudnn64_9.dll')):
            found.append(d)
    return found


def _session(path, threads=None, device='cpu'):
    import onnxruntime as ort
    if not os.path.isfile(path):
        raise SystemExit(f'模型文件不存在: {path}')
    providers = ['CPUExecutionProvider']
    if device in ('gpu', 'auto'):
        for d in _find_cuda_dll_dirs():
            os.environ['PATH'] = d + os.pathsep + os.environ.get('PATH', '')
            if hasattr(os, 'add_dll_directory'):
                try:
                    os.add_dll_directory(d)
                except OSError:
                    pass
        providers = [('CUDAExecutionProvider', {'device_id': 0}),
                     'CPUExecutionProvider']
    opts = ort.SessionOptions()
    if threads:
        opts.intra_op_num_threads = int(threads)
        opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, sess_options=opts, providers=providers)


def _norm(tile):
    """HxWx3 uint8 -> 1x3xHxW float32,ImageNet 归一化。"""
    x = tile.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def _positions(size, tile, stride):
    if size <= tile:
        return [0]
    pos = list(range(0, size - tile + 1, stride))
    if pos[-1] != size - tile:
        pos.append(size - tile)
    return pos


def _hann(h, w):
    wy = np.hanning(h + 2)[1:-1].astype(np.float32)
    wx = np.hanning(w + 2)[1:-1].astype(np.float32)
    return np.maximum(np.outer(wy, wx), 1e-3)


def _pad_reflect(rgb, H, W):
    ph, pw = max(0, H - rgb.shape[0]), max(0, W - rgb.shape[1])
    if ph or pw:
        rgb = np.pad(rgb, ((0, ph), (0, pw), (0, 0)), mode='reflect')
    return rgb


class Model:
    """薄封装:各模型输入尺寸/预处理略有差别,统一成 prob(rgb) -> float32 HxW。"""

    def __init__(self, kind, threads=None, device='cpu'):
        if kind not in MODEL_TILES:
            raise SystemExit(f'未知模型 {kind};可选:{", ".join(MODEL_TILES)}')
        self.kind = kind
        self.sess = _session(MODEL_FILES[kind], threads, device)
        self.providers = list(self.sess.get_providers())
        self.input_name = self.sess.get_inputs()[0].name
        th, tw, sh, sw = MODEL_TILES[kind]
        self.tile, self.stride = (th, tw), (sh, sw)
        # Tomt996 系列(5 个 seed)固定 384 高,要补 6 行 reflect
        self.reflect6 = kind.startswith('tom')
        self.w = _hann(*self.tile)

    # ---- 单块前向 ----
    def _forward(self, tile):
        if self.reflect6:
            # 模型固定 384 高:底下 6 行 reflect 补出来,推理后裁掉
            th, tw = self.tile
            inp = _pad_reflect(tile, th + 6, tw)
            logits = self.sess.run(None, {self.input_name: _norm(inp)})[0]
            logits = logits[0, 0, :th, :tw]
        else:
            logits = self.sess.run(None, {self.input_name: _norm(tile)})[0][0, 0]
        return 1.0 / (1.0 + np.exp(-logits.astype(np.float32)))

    # ---- 原生分辨率切块 ----
    def prob_tiled(self, rgb):
        th, tw = self.tile
        sh, sw = self.stride
        H, W = rgb.shape[:2]
        Hp, Wp = max(H, th), max(W, tw)
        rgb = _pad_reflect(rgb, Hp, Wp)
        acc = np.zeros((Hp, Wp), np.float32)
        wsum = np.zeros((Hp, Wp), np.float32)
        ys, xs = _positions(Hp, th, sh), _positions(Wp, tw, sw)
        n = 0
        for y in ys:
            for x in xs:
                p = self._forward(rgb[y:y + th, x:x + tw])
                acc[y:y + th, x:x + tw] += p * self.w
                wsum[y:y + th, x:x + tw] += self.w
                n += 1
        acc /= np.maximum(wsum, 1e-6)
        self.last_tiles = n
        return np.ascontiguousarray(acc[:H, :W])

    # ---- 快速档:整图缩到长边 1024 再推理,结果放大回原图 ----
    def prob_fast(self, rgb):
        from PIL import Image
        H, W = rgb.shape[:2]
        scale = 1024.0 / max(H, W)
        if scale >= 1.0:
            small = rgb
        else:
            small = np.asarray(Image.fromarray(rgb).resize(
                (max(1, int(round(W * scale))), max(1, int(round(H * scale)))),
                Image.BILINEAR))
        if self.reflect6:
            # Tomt996 直接吃 672x378;快速档不追求保真,整图缩放即可
            from PIL import Image as I
            th, tw = self.tile
            tile = np.asarray(I.fromarray(small).resize((tw, th), I.BILINEAR))
            p = self._forward(tile)
        else:
            p = self.prob_tiled(small)
        if p.shape != (H, W):
            p = np.asarray(Image.fromarray(p.astype(np.float32), mode='F').resize(
                (W, H), Image.BILINEAR))
        return p.astype(np.float32, copy=False)

    def prob(self, rgb, mode='fine'):
        if mode == 'fast':
            return self.prob_fast(rgb)
        return self.prob_tiled(rgb)


class Ensemble(Model):
    def __init__(self, threads=None, device='cpu'):
        self.kind = 'ensemble'
        self.models = [Model('tom', threads, device), Model('dihao', threads, device)]
        self.providers = list(self.models[0].providers)

    def prob(self, rgb, mode='fine'):
        ps = [m.prob(rgb, mode) for m in self.models]
        return (ps[0] + ps[1]) * 0.5


def load_model(name='tom', threads=None, device='cpu'):
    if name == 'ensemble':
        return Ensemble(threads, device)
    return Model(name, threads, device)


# =============================================================================
# 投影
# =============================================================================
def project_prob(P, cam, rc, prob, finite_pts, depth_tol=None):
    """把一个视角的概率图采样到点云上。

    返回 ``(vis_idx, prob_at, depth_tol, sigma)``;只处理投影落在图像内、
    且在相机前方的点,再对这些点打射线做遮挡剔除。rock 场景 12.5M 个候选
    点一次 raycast 约 5 s。
    """
    K, R, t, C = cam['K'], cam['R'], cam['t'], cam['C']
    H, W = prob.shape
    u, v, depth = b2m.project_points(P, K, R, t)
    u = u.astype(np.float32)
    v = v.astype(np.float32)
    depth = depth.astype(np.float32)

    cand = ((depth > 0) & (u >= -0.5) & (u <= W - 0.5) &
            (v >= -0.5) & (v <= H - 0.5) & finite_pts)
    idx = np.flatnonzero(cand)
    sigma = None
    if len(idx) == 0:
        return idx, np.empty(0, np.float32), depth_tol, sigma

    Pc = P[idx]
    d = Pc - C[None, :]
    dist = np.linalg.norm(d, axis=1)
    th = m2p.raycast_t_hit(rc, Pc, C)

    if depth_tol is None:
        depth_tol, sigma = m2p.auto_depth_tol(th, dist)
    vis = m2p.visibility_from(depth[idx], dist, th, depth_tol, finite_pts[idx])
    idx = idx[vis]
    if len(idx) == 0:
        return idx, np.empty(0, np.float32), depth_tol, sigma

    iu = np.rint(u[idx]).astype(np.int64)
    iv = np.rint(v[idx]).astype(np.int64)
    np.clip(iu, 0, W - 1, out=iu)
    np.clip(iv, 0, H - 1, out=iv)
    return idx, prob[iv, iu].astype(np.float32, copy=False), depth_tol, sigma


# =============================================================================
# 2D 预览图
# =============================================================================
def _colormap(p):
    """0~1 -> RGB uint8,近似 jet。"""
    r = np.clip(1.5 - np.abs(4.0 * p - 3.0), 0, 1)
    g = np.clip(1.5 - np.abs(4.0 * p - 2.0), 0, 1)
    b = np.clip(1.5 - np.abs(4.0 * p - 1.0), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def save_gallery(rgb, prob, out_dir, stem, threshold):
    """降采样后存叠加图和热力图,避免 88 张 9524x6338 PNG 把盘吃满。"""
    from PIL import Image
    os.makedirs(out_dir, exist_ok=True)
    H, W = prob.shape
    max_w = 1600
    if W > max_w:
        nh = int(round(H * max_w / W))
        base = np.asarray(Image.fromarray(rgb).resize((max_w, nh), Image.BILINEAR))
        small = np.asarray(Image.fromarray((prob * 255).astype(np.uint8)).resize(
            (max_w, nh), Image.BILINEAR))
    else:
        base, small = rgb, (prob * 255).astype(np.uint8)
    p = small.astype(np.float32) / 255.0
    mask = p > threshold

    overlay = base.astype(np.float32).copy()
    overlay[mask] = overlay[mask] * 0.35 + np.array([255, 40, 40], np.float32) * 0.65
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    heat = (base.astype(np.float32) * 0.35 + _colormap(p).astype(np.float32) * 0.65)
    heat = np.clip(heat, 0, 255).astype(np.uint8)

    op = os.path.join(out_dir, stem + '_overlay.jpg')
    hp = os.path.join(out_dir, stem + '_prob.jpg')
    Image.fromarray(overlay).save(op, quality=88)
    Image.fromarray(heat).save(hp, quality=88)
    return op, hp, float(mask.mean())


# =============================================================================
# 点云显示降采样
# =============================================================================
def _voxel_pick(P, voxel):
    q = np.floor(P / voxel).astype(np.int64)
    key = (q[:, 0] * 73856093) ^ (q[:, 1] * 19349663) ^ (q[:, 2] * 83492791)
    _, idx = np.unique(key, return_index=True)      # 每个 voxel 留第一个点
    return idx


def display_indices(P, target=1_200_000, voxel=0.012):
    """自适应 voxel,把 13M 点降到 ~1.2M 给浏览器。"""
    idx = _voxel_pick(P, voxel)
    tries = 0
    while len(idx) > int(target * 1.35) and tries < 6:
        voxel *= 1.3
        idx = _voxel_pick(P, voxel)
        tries += 1
    while len(idx) < target * 0.45 and voxel > 0.0008 and tries < 10:
        voxel *= 0.8
        idx = _voxel_pick(P, voxel)
        tries += 1
    return idx, float(voxel)


# =============================================================================
# 主流程
# =============================================================================
def run(paths, model='tom', mode='fine', limit=None, overwrite=False,
        gallery=True, threads=None, progress=None, should_stop=None,
        device='cpu'):
    """检测并投影。progress(dict) 用来给 GUI 推日志。"""
    check_scene(paths)
    os.makedirs(paths['crack'], exist_ok=True)
    scores_path = os.path.join(paths['crack'], 'scores.npz')
    display_path = os.path.join(paths['crack'], 'display.npz')
    meta_path = os.path.join(paths['crack'], 'meta.json')

    if not overwrite and os.path.isfile(scores_path) and os.path.isfile(display_path):
        raise RuntimeError('该场景已经检测过;勾选「重新检测」或加 --overwrite 覆盖。')

    def emit(**kw):
        kw.setdefault('scene', paths['name'])
        if progress:
            progress(kw)

    t_all = time.time()
    emit(event='stage', stage='load', message='读取相机与点云…')

    scene = b2m.load_scene(paths['sparse'])
    names = sorted(scene.keys())
    if limit:
        names = names[:int(limit)]
    if not names:
        raise RuntimeError('稀疏模型里没有相机。')

    P, cols = m2p.load_point_cloud(paths['cloud'])
    N = len(P)
    cols_u8 = None
    if cols is not None:
        cols_u8 = np.clip(cols * 255.0, 0, 255).astype(np.uint8)
        del cols

    V, F, _ = b2m.load_mesh(paths['mesh'])
    rc = m2p.build_raycast_scene(V, F)
    del V, F

    finite_pts = np.isfinite(P).all(axis=1)
    sum_prob = np.zeros(N, np.float32)
    n_vis = np.zeros(N, np.uint16)
    max_prob = np.zeros(N, np.float32)

    mdl = load_model(model, threads, device)
    providers = list(getattr(mdl, 'providers', []))
    dev_label = 'GPU (CUDA)' if providers[:1] == ['CUDAExecutionProvider'] else 'CPU'
    emit(event='log', message=f'推理设备: {dev_label}  providers={providers}')
    if device in ('gpu', 'auto') and providers[:1] != ['CUDAExecutionProvider']:
        emit(event='warn', message='GPU 不可用,已退回 CPU。检查 onnxruntime-gpu 和 '
                                   'cuDNN DLL(可用 CRACK_CUDA_DLL_DIR 指定)。')
    depth_tol = None
    sigma = None
    gallery_items = []
    per_image = []
    detected = 0

    emit(event='stage', stage='detect', message=f'开始检测 {len(names)} 张,模型 {MODEL_LABELS.get(model, model)},'
                                                 f'模式 {"快速" if mode == "fast" else "原生分块"}')

    for i, name in enumerate(names, 1):
        if should_stop and should_stop():
            raise RuntimeError('用户已停止任务。')
        img_path = os.path.join(paths['images'], name)
        if not os.path.isfile(img_path):
            emit(event='warn', message=f'缺图,跳过: {name}')
            continue
        t0 = time.time()
        from PIL import Image
        rgb = np.asarray(Image.open(img_path).convert('RGB'))
        cam = scene[name]
        if (cam['H'], cam['W']) != rgb.shape[:2]:
            emit(event='warn', message=f'{name} 尺寸 {rgb.shape[1]}x{rgb.shape[0]} '
                                       f'与相机 {cam["W"]}x{cam["H"]} 不一致,按相机参数投影。')

        prob = mdl.prob(rgb, mode)
        t_infer = time.time() - t0

        idx, pvals, depth_tol, s = project_prob(
            P, cam, rc, prob, finite_pts, depth_tol)
        if s is not None:
            sigma = s
        if len(idx):
            sum_prob[idx] += pvals
            n_vis[idx] += 1
            max_prob[idx] = np.maximum(max_prob[idx], pvals)

        stem = os.path.splitext(name)[0]
        frac = float((prob > DEFAULT_THRESHOLD.get(model, 0.4)).mean())
        if isinstance(mdl, Ensemble):
            tiles = [getattr(m, 'last_tiles', None) for m in mdl.models]
        else:
            tiles = getattr(mdl, 'last_tiles', None)
        item = {'image': name, 'stem': stem, 'visible_points': int(len(idx)),
                'mean_prob': round(float(prob.mean()), 5),
                'frac_above': round(frac, 6),
                'seconds': round(time.time() - t0, 2),
                'tiles': tiles}
        if gallery:
            thr = DEFAULT_THRESHOLD.get(model, 0.4)
            op, hp, mask_frac = save_gallery(rgb, prob, paths['gallery'], stem, thr)
            item['overlay'] = os.path.relpath(op, paths['crack']).replace('\\', '/')
            item['heatmap'] = os.path.relpath(hp, paths['crack']).replace('\\', '/')
            item['mask_frac_preview'] = round(mask_frac, 6)
        gallery_items.append(item)
        per_image.append(item)
        detected += 1

        del prob, rgb
        emit(event='progress', stage='detect', done=i, total=len(names),
             image=name, visible_points=int(len(idx)), seconds=item['seconds'],
             infer_seconds=round(t_infer, 2), mean_prob=item['mean_prob'],
             message=f'[{i}/{len(names)}] {stem}  可见 {len(idx):,} 点  '
                     f'{item["seconds"]:.1f}s')

    valid = n_vis > 0
    scores = np.zeros(N, np.float32)
    scores[valid] = sum_prob[valid] / n_vis[valid].astype(np.float32)
    del sum_prob

    emit(event='stage', stage='save', message='保存逐点分数与显示点云…')
    tmp = scores_path + '.tmp.npz'
    np.savez_compressed(
        tmp,
        score=scores.astype(np.float16),
        n_visible=n_vis,
        max_prob=max_prob.astype(np.float16),
    )
    os.replace(tmp, scores_path)

    t0 = time.time()
    didx, voxel = display_indices(P)
    pos = P[didx].astype(np.float32)
    if cols_u8 is not None:
        dcol = cols_u8[didx]
    else:
        dcol = np.full((len(didx), 3), 190, np.uint8)
    dscore = np.clip(scores[didx] * 255.0, 0, 255).astype(np.uint8)
    dnvis = np.clip(n_vis[didx], 0, 255).astype(np.uint8)
    tmp = display_path + '.tmp.npz'
    np.savez(tmp, positions=pos, colors=dcol, scores=dscore, n_visible=dnvis)
    os.replace(tmp, display_path)
    emit(event='stage', stage='save',
         message=f'显示点云 {len(didx):,} 点 (voxel {voxel:.4f}),{time.time()-t0:.1f}s')

    sv = scores[valid]
    meta = {
        'scene': paths['name'],
        'model': model,
        'model_label': MODEL_LABELS.get(model, model),
        'mode': mode,
        'device': dev_label,
        'providers': providers,
        'created': datetime.now().isoformat(timespec='seconds'),
        'n_points': int(N),
        'n_display': int(len(didx)),
        'voxel': voxel,
        'sparse_dir': os.path.relpath(paths['sparse'], REPO_ROOT).replace('\\', '/'),
        'cloud': os.path.relpath(paths['cloud'], REPO_ROOT).replace('\\', '/'),
        'n_images_total': len(scene),
        'n_images_used': detected,
        'depth_tol': depth_tol,
        'depth_sigma': sigma,
        'default_threshold': DEFAULT_THRESHOLD.get(model, 0.4),
        'score_percentiles': {str(q): round(float(np.percentile(sv, q)), 4)
                              for q in (50, 75, 90, 95, 99)} if len(sv) else {},
        'visible_point_fraction': round(float(valid.mean()), 4),
        'seconds_total': round(time.time() - t_all, 1),
        'bbox_min': P.min(axis=0).tolist(),
        'bbox_max': P.max(axis=0).tolist(),
        'gallery': gallery_items,
        'images': per_image,
        'units': 'model units (COLMAP/OpenMVS); set scale_m_per_unit to convert to metres',
        'scale_m_per_unit': None,
    }
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    emit(event='done', stage='done',
         message=f'完成:{detected} 张图,{int(valid.sum()):,} 个点被至少一个视角看到,'
                 f'总耗时 {meta["seconds_total"]}s',
         meta=meta)
    return meta


# =============================================================================
# CLI
# =============================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description='裂缝检测并投影到点云')
    ap.add_argument('--scene', required=True, help='work 下的场景名或路径,如 rock')
    ap.add_argument('--root', default=None, help='work 根目录')
    ap.add_argument('--model', default='tom', choices=['tom', 'dihao', 'ensemble'])
    ap.add_argument('--mode', default='fine', choices=['fine', 'fast'])
    ap.add_argument('--limit', type=int, default=None, help='只用前 N 张图')
    ap.add_argument('--threads', type=int, default=None, help='ONNX 线程数')
    ap.add_argument('--device', default='auto', choices=['auto', 'cpu', 'gpu'],
                    help='推理设备:auto=有 GPU 就用,cpu=强制 CPU')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--no-gallery', action='store_true')
    args = ap.parse_args(argv)

    paths = scene_paths(args.scene, args.root)

    def log(d):
        msg = d.get('message')
        if msg:
            print(msg, flush=True)

    meta = run(paths, model=args.model, mode=args.mode, limit=args.limit,
               overwrite=args.overwrite, gallery=not args.no_gallery,
               threads=args.threads, progress=log, device=args.device)
    print(json.dumps({k: meta[k] for k in
                      ('n_points', 'n_display', 'n_images_used', 'depth_tol',
                       'visible_point_fraction', 'seconds_total')},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
