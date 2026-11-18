# -*- coding: utf-8 -*-
"""阈值化、裂隙连通域分析与导出。

输入是 ``detect_project.py`` 生成的逐点分数,输出:

- ``out/<scene>_cracks.ply``       全点云,裂隙点按置信度染成黄→红
- ``out/<scene>_cracks_only.ply``  只含裂隙点,按连通域上色
- ``out/<scene>_cracks.csv``       每条裂隙的编号/长度/宽度/走向/中心/置信度
- ``out/<scene>_cracks.json``      同样的属性 + 每条裂隙的折线坐标 + 本次分析参数
- ``out/<scene>_cracks_lines.ply`` 每条裂隙的中心线折线(vertex + edge,按裂隙上色)
- ``out/<scene>_cracks_lines.json`` 折线坐标
- ``out/<scene>_cracks_lines.csv``  折线逐点坐标(crack_id, seq, x, y, z)

单位是重建的 model units。COLMAP/OpenMVS 没有绝对尺度,想换算成米就给
``scale_m_per_unit``(GUI 里也可以填)。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

TOOLS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(TOOLS_DIR)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import mask_to_points as m2p          # noqa: E402
from detect_project import scene_paths   # noqa: E402


# =============================================================================
# 连通域
# =============================================================================
def _voxel_keys(P, voxel):
    q = np.floor(P / voxel).astype(np.int64)
    q = np.ascontiguousarray(q)
    return q


def _as_void(q):
    return np.ascontiguousarray(q).view(
        np.dtype([('x', '<i8'), ('y', '<i8'), ('z', '<i8')])).ravel()


def voxel_components(P, voxel):
    """把裂隙点按体素 26 邻接聚成连通域。

    比 cKDTree.query_pairs 稳:裂隙带很密,pair 数量会爆;体素邻接只和
    占据的体素数成正比。返回 ``(comp_of_point, n_comp, uniq_voxels, comp_of_voxel)``。
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    q = _voxel_keys(P, voxel)
    vq = _as_void(q)
    uniq_void, inv = np.unique(vq, return_inverse=True)
    uniq = uniq_void.view(np.int64).reshape(-1, 3)
    m = len(uniq)
    if m == 0:
        return np.full(len(P), -1, np.int32), 0, uniq, np.empty(0, np.int32)

    lookup = {(int(x), int(y), int(z)): i
              for i, (x, y, z) in enumerate(uniq)}
    a, b = [], []
    for i, (x, y, z) in enumerate(uniq):
        x, y, z = int(x), int(y), int(z)
        for dx, dy, dz in ((1, 0, 0), (0, 1, 0), (0, 0, 1),
                           (1, 1, 0), (1, 0, 1), (0, 1, 1),
                           (1, -1, 0), (1, 0, -1), (0, 1, -1),
                           (1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1)):
            j = lookup.get((x + dx, y + dy, z + dz))
            if j is not None:
                a.append(i)
                b.append(j)
    if a:
        adj = coo_matrix((np.ones(len(a), np.int8), (a, b)), shape=(m, m))
        n_comp, comp_vox = connected_components(adj, directed=False)
    else:
        n_comp, comp_vox = m, np.arange(m, dtype=np.int32)
    return comp_vox[inv.astype(np.int64)].astype(np.int32), int(n_comp), uniq, comp_vox


def _robust_extent(x):
    if len(x) == 0:
        return 0.0
    return float(np.percentile(x, 98) - np.percentile(x, 2))


def component_attributes(Pc, n_vis_c, score_c, comp, n_comp, voxel):
    """逐连通域 PCA:长度、宽度、主方向、中心、置信度。"""
    rows = []
    for c in range(n_comp):
        sel = comp == c
        n = int(sel.sum())
        if n == 0:
            continue
        X = Pc[sel]
        center = X.mean(axis=0)
        Xc = X - center
        # 无偏协方差;点数太少时补零
        cov = (Xc.T @ Xc) / max(n - 1, 1)
        w, V = np.linalg.eigh(cov)
        order = np.argsort(w)[::-1]
        w, V = w[order], V[:, order]
        axis = V[:, 0]
        # 主方向取一个确定的符号,方便表格比较
        k = int(np.argmax(np.abs(axis)))
        if axis[k] < 0:
            axis = -axis
        t1 = Xc @ axis
        t2 = Xc @ V[:, 1]
        length = _robust_extent(t1)
        width = _robust_extent(t2)
        # 主方向可能只占很小比例;用体素跨度兜底
        if length <= 0:
            length = float(np.ptp(Xc, axis=0).max())
        az = float(np.degrees(np.arctan2(axis[1], axis[0])))
        el = float(np.degrees(np.arcsin(np.clip(axis[2], -1, 1))))
        rows.append({
            'comp': c,
            'n_points': n,
            'length': round(length, 6),
            'width': round(width, 6),
            'axis': [round(float(v), 6) for v in axis],
            'azimuth_deg': round(az, 2),
            'elevation_deg': round(el, 2),
            'centroid': [round(float(v), 5) for v in center],
            'mean_score': round(float(score_c[sel].mean()), 5),
            'max_score': round(float(score_c[sel].max()), 5),
            'extent_voxels': int(len(np.unique(_as_void(_voxel_keys(X, voxel))))),
        })
    rows.sort(key=lambda r: r['length'], reverse=True)
    for i, r in enumerate(rows):
        r['id'] = i          # 0 基,前端标签/着色用
        r['crack_id'] = i + 1  # 1 基,表格和 CSV 给用户看
    return rows


def crack_centerline(X, voxel, smooth=True):
    """把一条裂隙的点云压成一条折线(线几何)。

    做法:在和连通域相同的 gap 体素上建 26 邻接图,做两次最远点扫描
    (图的直径)取最长骨架路径,再把路径上的体素换成其中点的重心。
    返回 ``(M, 3)`` float64;点太少时返回单点或空数组。

    不用 PCA 直线而用图直径,是因为裂隙经常是弯的、分叉的,PCA 只能得到
    一条直线。
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import shortest_path

    X = np.asarray(X, np.float64)
    if len(X) == 0:
        return np.empty((0, 3))
    q = np.floor(X / voxel).astype(np.int64)
    uniq_void, inv = np.unique(_as_void(q), return_inverse=True)
    uniq = uniq_void.view(np.int64).reshape(-1, 3)
    m = len(uniq)
    if m == 1:
        return X.mean(axis=0, keepdims=True)

    lookup = {(int(x), int(y), int(z)): i
              for i, (x, y, z) in enumerate(uniq)}
    a, b = [], []
    for i, (x, y, z) in enumerate(uniq):
        x, y, z = int(x), int(y), int(z)
        for dx, dy, dz in ((1, 0, 0), (0, 1, 0), (0, 0, 1),
                           (1, 1, 0), (1, 0, 1), (0, 1, 1),
                           (1, -1, 0), (1, 0, -1), (0, 1, -1),
                           (1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1)):
            j = lookup.get((x + dx, y + dy, z + dz))
            if j is not None:
                a.append(i)
                b.append(j)
    if a:
        adj = coo_matrix((np.ones(len(a), np.float32), (a, b)),
                         shape=(m, m)).tocsr()
    else:
        return X.mean(axis=0, keepdims=True)

    # 两次最远点扫描近似图直径
    d0 = shortest_path(adj, directed=False, unweighted=True)
    v1 = int(np.argmax(d0[0]))
    d1, pred = shortest_path(adj, directed=False, unweighted=True,
                             indices=[v1], return_predecessors=True)
    d1, pred = d1[0], pred[0]
    v2 = int(np.argmax(d1))
    path = [v2]
    k = v2
    while k != v1:
        k = int(pred[k])
        if k < 0:
            break
        path.append(k)
    path.reverse()

    # 每个体素的重心(真实坐标,分辨率仍然是 gap)
    counts = np.maximum(np.bincount(inv, minlength=m), 1)
    centroids = np.empty((m, 3), np.float64)
    for c in range(3):
        centroids[:, c] = np.bincount(inv, weights=X[:, c], minlength=m) / counts
    line = centroids[np.array(path, np.int64)]

    # 轻微平滑,端点保持
    if smooth and len(line) >= 3:
        sm = line.copy()
        sm[1:-1] = (line[:-2] + 2.0 * line[1:-1] + line[2:]) / 4.0
        line = sm
    return line


def _line_length(line):
    if len(line) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(line, axis=0), axis=1).sum())


def write_lines_ply(path, lines, colors):
    """把每条裂隙的折线写成带 edge 元素的 ASCII PLY。

    ``lines`` 是折线数组列表,``colors`` 是 (K,3) uint8,每条线一种颜色。
    MeshLab / Open3D 能直接打开看线;顶点颜色方便区分不同裂隙。
    """
    verts, vcols, edges = [], [], []
    off = 0
    for pts, col in zip(lines, colors):
        pts = np.asarray(pts, np.float32)
        n = len(pts)
        if n == 0:
            continue
        verts.append(pts)
        vcols.append(np.repeat(np.asarray(col, np.uint8)[None, :], n, axis=0))
        if n >= 2:
            edges.append(np.stack([np.arange(off, off + n - 1),
                                   np.arange(off + 1, off + n)], axis=1))
        off += n
    if verts:
        V = np.concatenate(verts)
        C = np.concatenate(vcols)
        E = np.concatenate(edges) if edges else np.empty((0, 2), np.int64)
    else:
        V = np.empty((0, 3), np.float32)
        C = np.empty((0, 3), np.uint8)
        E = np.empty((0, 2), np.int64)
    header = (
        'ply\nformat ascii 1.0\n'
        f'comment crack centerlines: {len(lines)} polylines\n'
        f'element vertex {len(V)}\n'
        'property float x\nproperty float y\nproperty float z\n'
        'property uchar red\nproperty uchar green\nproperty uchar blue\n'
        f'element edge {len(E)}\n'
        'property int vertex1\nproperty int vertex2\n'
        'end_header\n'
    )
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='ascii', newline='\n') as f:
        f.write(header)
        for (x, y, z), (r, g, b) in zip(V.tolist(), C.tolist()):
            f.write(f'{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n')
        for i, j in E.tolist():
            f.write(f'{i} {j}\n')
    os.replace(tmp, path)


# =============================================================================
# PLY 写盘(手写二进制,比 open3d 快且不带法线)
# =============================================================================
def write_ply(path, P, colors):
    P = np.asarray(P, np.float32)
    colors = np.asarray(colors, np.uint8)
    n = len(P)
    header = (
        'ply\nformat binary_little_endian 1.0\n'
        f'element vertex {n}\n'
        'property float x\nproperty float y\nproperty float z\n'
        'property uchar red\nproperty uchar green\nproperty uchar blue\n'
        'end_header\n'
    )
    dt = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                   ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    arr = np.empty(n, dtype=dt)
    arr['x'], arr['y'], arr['z'] = P[:, 0], P[:, 1], P[:, 2]
    arr['red'], arr['green'], arr['blue'] = colors[:, 0], colors[:, 1], colors[:, 2]
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        f.write(header.encode('ascii'))
        arr.tofile(f)
    os.replace(tmp, path)


PALETTE = np.array([
    [255, 59, 48], [255, 149, 0], [255, 204, 0], [52, 199, 89],
    [0, 199, 190], [0, 122, 255], [88, 86, 214], [175, 82, 222],
    [255, 45, 85], [162, 132, 94], [48, 176, 199], [233, 103, 255],
], np.uint8)


# =============================================================================
# 分析
# =============================================================================
def analyze(paths, threshold=0.4, min_length=0.05, gap=None,
            scale=None, load_cloud=True, write_outputs=True, out_dir=None,
            min_points=12, progress=None, base_prefix=None):
    t0 = time.time()

    def _p(done, message=None, stage='analyze'):
        if progress:
            ev = {'event': 'progress', 'stage': stage, 'done': done, 'total': 100}
            if message:
                ev['message'] = message
            progress(ev)

    sfile = os.path.join(paths['crack'], 'scores.npz')
    dfile = os.path.join(paths['crack'], 'display.npz')
    mfile = os.path.join(paths['crack'], 'meta.json')
    if not os.path.isfile(sfile):
        raise RuntimeError('还没有检测结果,先运行裂缝检测。')

    _p(3, '读取逐点分数…')
    z = np.load(sfile)
    score = z['score'].astype(np.float32)
    n_vis = z['n_visible'].astype(np.int32)
    N = len(score)
    _p(8)

    meta = {}
    if os.path.isfile(mfile):
        with open(mfile, encoding='utf-8') as f:
            meta = json.load(f)
    if gap is None:
        bb = np.array(meta.get('bbox_max', [1, 1, 1])) - np.array(
            meta.get('bbox_min', [0, 0, 0]))
        diag = float(np.linalg.norm(bb)) if len(bb) == 3 else 1.0
        gap = float(np.clip(diag * 0.003, 0.005, 0.2))
    gap = float(gap)
    if scale is None:
        scale = meta.get('scale_m_per_unit')

    mask = (score >= threshold) & (n_vis > 0)
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        result = {'n_crack_points': 0, 'n_cracks': 0, 'cracks': [],
                  'threshold': threshold, 'min_length': min_length, 'gap': gap,
                  'message': '当前阈值下没有裂隙点,试试降低阈值。'}
        if write_outputs:
            _write_empty(paths, meta, result, out_dir, base_prefix)
        _p(100, '当前阈值下没有裂隙点。')
        return result

    # 聚类需要坐标:全点云只在导出时读,分析阶段优先用 display.npz 的坐标
    # 会漏掉不在显示子集里的点,所以这里仍读全点云。
    _p(12, f'读取点云 ({N:,} 点)…')
    P, cols = m2p.load_point_cloud(paths['cloud'])
    if len(P) != N:
        raise RuntimeError(f'点云点数 {len(P)} 与分数 {N} 不一致,重新检测。')
    Pc = P[idx]
    _p(28)

    _p(32, f'体素连通域 ({len(idx):,} 个阈值点)…')
    comp_p, n_comp, uniq_vox, comp_vox = voxel_components(Pc, gap)
    score_c = score[idx]
    nvis_c = n_vis[idx]
    _p(48, f'{n_comp} 个原始连通域,计算属性…')
    rows = component_attributes(Pc, nvis_c, score_c, comp_p, n_comp, gap)
    _p(58)

    # 过滤:长度和点数都要够
    keep = [r for r in rows if r['length'] >= min_length and r['n_points'] >= min_points]
    _p(62, f'保留 {len(keep)} 条裂隙')
    keep_ids = np.array([r['comp'] for r in keep], np.int64)
    remap = {int(c): i for i, c in enumerate(keep_ids)}
    if len(keep_ids):
        good = np.zeros(n_comp, bool)
        good[keep_ids] = True
        p_keep = good[comp_p]
        idx_good = idx[p_keep]
        comp_good = comp_p[p_keep]
        final_labels = np.full(len(idx_good), -1, np.int32)
        for j, c in enumerate(keep_ids):
            final_labels[comp_good == c] = j
    else:
        idx_good = np.empty(0, np.int64)
        final_labels = np.empty(0, np.int32)

    # 每条裂隙压成一条折线(线几何),坐标就是点云坐标
    _p(64, '提取裂隙中心线…')
    lines = []
    for i, r in enumerate(keep):
        line = crack_centerline(Pc[comp_p == r['comp']], gap)
        if scale and len(line):
            line_m = line * scale
        else:
            line_m = None
        r['line_length'] = round(_line_length(line), 6)
        r['line_n_points'] = int(len(line))
        r['polyline'] = [[round(float(v), 6) for v in p] for p in line]
        if line_m is not None:
            r['line_length_m'] = round(_line_length(line_m), 6)
        lines.append(line)

    # 重新编号按长度降序(keep 已经排好)
    for r in keep:
        r.pop('comp', None)
        if scale:
            r['length_m'] = round(r['length'] * scale, 6)
            r['width_m'] = round(r['width'] * scale, 6)
    result = {
        'scene': paths['name'],
        'threshold': threshold,
        'min_length': min_length,
        'gap': gap,
        'min_points': min_points,
        'scale_m_per_unit': scale,
        'n_points_total': int(N),
        'n_seen_points': int((n_vis > 0).sum()),
        'n_threshold_points': int(len(idx)),
        'n_crack_points': int(len(idx_good)),
        'n_raw_components': int(n_comp),
        'n_cracks': len(keep),
        'cracks': keep,
        'seconds': round(time.time() - t0, 2),
    }

    # ---- 显示点云的连通域标签,给前端点选/表格联动 ----
    _p(68, '生成显示点云标签…')
    if os.path.isfile(dfile):
        dz = np.load(dfile)
        dpos = dz['positions']
        dscore = dz['scores']
        lab_d = np.full(len(dpos), -1, np.int32)
        # display 的 score 是 uint8,阈值换算到 0~255
        dthr = int(np.ceil(threshold * 255.0))
        ok = dscore >= dthr
        if len(keep_ids) and ok.any():
            remap_arr = np.full(n_comp, -1, np.int32)
            for j, c in enumerate(keep_ids):
                remap_arr[c] = j
            vox_comp = {tuple(int(v) for v in row): int(comp_vox[i])
                        for i, row in enumerate(uniq_vox)}
            qd = _voxel_keys(dpos[ok], gap)
            vals = np.full(len(qd), -1, np.int32)
            for k, key in enumerate(map(tuple, qd.tolist())):
                c = vox_comp.get(key, -1)
                if c >= 0:
                    vals[k] = remap_arr[c]
            lab_d[ok] = vals
        np.save(os.path.join(paths['crack'], 'labels_display.npy'), lab_d)

    # ---- 导出 ----
    if write_outputs:
        _p(74, '写高亮 PLY (全点云,较大的文件)…')
        out_dir = out_dir or os.path.join(REPO_ROOT, 'out')
        os.makedirs(out_dir, exist_ok=True)
        base = base_prefix or os.path.join(out_dir, paths['name'] + '_cracks')

        # 全点云:背景保留原色(稍压暗),裂隙按分数黄→红
        bg = np.full((N, 3), 180, np.uint8) if cols is None else \
            np.clip(cols * 255.0, 0, 255).astype(np.uint8)
        bgc = (bg.astype(np.float32) * 0.78).astype(np.uint8)
        t = np.clip((score[idx_good] - threshold) / max(1e-6, 1.0 - threshold), 0, 1)
        hot = np.empty((len(idx_good), 3), np.uint8)
        hot[:, 0] = 255
        hot[:, 1] = (255 * (1.0 - t)).astype(np.uint8)
        hot[:, 2] = (40 * (1.0 - t)).astype(np.uint8)
        bgc[idx_good] = hot
        write_ply(base + '.ply', P, bgc)
        _p(88, '写纯裂隙 PLY…')

        # 只含裂隙:按连通域调色 + 置信度提亮
        if len(idx_good):
            ccols = PALETTE[np.arange(len(keep_ids)) % len(PALETTE)]
            col_only = np.empty((len(idx_good), 3), np.uint8)
            for j in range(len(keep_ids)):
                sel = final_labels == j
                c = ccols[j].astype(np.float32)
                b = 0.55 + 0.45 * t[sel][:, None]
                col_only[sel] = np.clip(c[None, :] * b, 0, 255).astype(np.uint8)
            write_ply(base + '_only.ply', P[idx_good], col_only)
        else:
            write_ply(base + '_only.ply', np.empty((0, 3), np.float32),
                      np.empty((0, 3), np.uint8))
        _p(94, '写 CSV / JSON…')

        # CSV (带 BOM,Excel 打开中文表头不乱码)
        cols_name = ['crack_id', 'n_points', 'length', 'width',
                     'orientation_x', 'orientation_y', 'orientation_z',
                     'azimuth_deg', 'elevation_deg',
                     'centroid_x', 'centroid_y', 'centroid_z',
                     'mean_score', 'max_score']
        if scale:
            cols_name += ['length_m', 'width_m']
        with open(base + '.csv', 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(cols_name)
            for r in keep:
                row = [r['crack_id'], r['n_points'], r['length'], r['width'],
                       r['axis'][0], r['axis'][1], r['axis'][2],
                       r['azimuth_deg'], r['elevation_deg'],
                       r['centroid'][0], r['centroid'][1], r['centroid'][2],
                       r['mean_score'], r['max_score']]
                if scale:
                    row += [r['length_m'], r['width_m']]
                w.writerow(row)

        # ---- 每条裂隙的线几何(中心线折线) ----
        _p(97, '写裂隙线几何…')
        lcols = PALETTE[np.arange(len(keep_ids)) % len(PALETTE)] if len(keep_ids) \
            else np.empty((0, 3), np.uint8)
        write_lines_ply(base + '_lines.ply', lines, lcols)
        lines_doc = {
            'scene': paths['name'],
            'threshold': threshold,
            'scale_m_per_unit': scale,
            'units': 'model units',
            'n_cracks': len(keep),
            'cracks': [{
                'crack_id': r['crack_id'],
                'n_points': r['n_points'],
                'length': r['length'],
                'line_length': r['line_length'],
                'points': r['polyline'],
            } for r in keep],
        }
        with open(base + '_lines.json', 'w', encoding='utf-8') as f:
            json.dump(lines_doc, f, ensure_ascii=False, indent=1)
        with open(base + '_lines.csv', 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(['crack_id', 'seq', 'x', 'y', 'z'])
            for r, line in zip(keep, lines):
                for k, p in enumerate(line):
                    w.writerow([r['crack_id'], k,
                                round(float(p[0]), 6), round(float(p[1]), 6),
                                round(float(p[2]), 6)])

        result['outputs'] = {
            'ply': os.path.relpath(base + '.ply', REPO_ROOT).replace('\\', '/'),
            'ply_only': os.path.relpath(base + '_only.ply', REPO_ROOT).replace('\\', '/'),
            'csv': os.path.relpath(base + '.csv', REPO_ROOT).replace('\\', '/'),
            'json': os.path.relpath(base + '.json', REPO_ROOT).replace('\\', '/'),
            'lines_ply': os.path.relpath(base + '_lines.ply', REPO_ROOT).replace('\\', '/'),
            'lines_json': os.path.relpath(base + '_lines.json', REPO_ROOT).replace('\\', '/'),
            'lines_csv': os.path.relpath(base + '_lines.csv', REPO_ROOT).replace('\\', '/'),
        }
        result['seconds'] = round(time.time() - t0, 2)
        # 主 JSON 最后写,这样刷新页面后重新载入也能拿到 outputs 链接
        with open(base + '.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    _p(100, f'分析完成:{len(keep)} 条裂隙,{time.time() - t0:.1f}s')
    return result


def _write_empty(paths, meta, result, out_dir, base_prefix=None):
    out_dir = out_dir or os.path.join(REPO_ROOT, 'out')
    os.makedirs(out_dir, exist_ok=True)
    base = base_prefix or os.path.join(out_dir, paths['name'] + '_cracks')
    write_ply(base + '.ply', np.empty((0, 3), np.float32), np.empty((0, 3), np.uint8))
    write_ply(base + '_only.ply', np.empty((0, 3), np.float32), np.empty((0, 3), np.uint8))
    write_lines_ply(base + '_lines.ply', [], np.empty((0, 3), np.uint8))
    result['outputs'] = {
        'ply': os.path.relpath(base + '.ply', REPO_ROOT).replace('\\', '/'),
        'ply_only': os.path.relpath(base + '_only.ply', REPO_ROOT).replace('\\', '/'),
        'lines_ply': os.path.relpath(base + '_lines.ply', REPO_ROOT).replace('\\', '/'),
    }
    with open(base + '.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


# =============================================================================
# CLI
# =============================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description='裂隙连通域分析与导出')
    ap.add_argument('--scene', required=True)
    ap.add_argument('--root', default=None)
    ap.add_argument('--threshold', type=float, default=None,
                    help='0~1,默认用 meta.json 里的 default_threshold')
    ap.add_argument('--min-length', type=float, default=None,
                    help='最短裂隙长度 (model units),默认 bbox 对角线的 0.3%%')
    ap.add_argument('--gap', type=float, default=None,
                    help='连通半径 (model units),默认 bbox 对角线的 0.3%%')
    ap.add_argument('--scale', type=float, default=None,
                    help='每 model unit 多少米')
    ap.add_argument('--out-dir', default=None)
    args = ap.parse_args(argv)

    paths = scene_paths(args.scene, args.root)
    meta = {}
    mfile = os.path.join(paths['crack'], 'meta.json')
    if os.path.isfile(mfile):
        with open(mfile, encoding='utf-8') as f:
            meta = json.load(f)
    thr = args.threshold if args.threshold is not None else meta.get('default_threshold', 0.4)
    bb = np.array(meta.get('bbox_max', [1, 1, 1])) - np.array(meta.get('bbox_min', [0, 0, 0]))
    diag = float(np.linalg.norm(bb)) if len(bb) == 3 else 1.0
    min_len = args.min_length if args.min_length is not None else max(0.005, diag * 0.003)

    def _log(ev):
        msg = ev.get('message')
        if msg:
            print(f'  [{ev.get("done", 0):3d}%] {msg}', flush=True)

    res = analyze(paths, threshold=thr, min_length=min_len, gap=args.gap,
                  scale=args.scale, out_dir=args.out_dir, progress=_log)
    print(json.dumps({k: res.get(k) for k in
                      ('threshold', 'min_length', 'gap', 'n_threshold_points',
                       'n_crack_points', 'n_raw_components', 'n_cracks',
                       'outputs', 'seconds')},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
