# -*- coding: utf-8 -*-
"""把 OpenMVS 的稠密网格转成浏览器能直接显示的轻量网格。

OpenMVS 的 ``scene_dense_mesh.ply`` 动辄几百万到上千万面(rock 有 1200 万面),
浏览器没法直接渲染。这里做三件事:

1. 体素聚类降采样(open3d 的 vertex clustering,秒级),目标 ~45 万面;
2. 从 ``display.npz`` 的显示点云把颜色搬到网格顶点上(体素查表 + cKDTree 兜底);
3. 算法线,存成 ``work/<scene>/crack/mesh_web.npz``(positions/normals/colors/indices)。

导出给用户的仍是原始 ``scene_dense_mesh.ply``,这个缓存只用于网页预览。
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime

import numpy as np

TARGET_FACES = 450_000
KEEP_FACES = 900_000          # 原始网格小于这个就直接用,不做简化
MIN_FACES = 180_000           # 简化过头就调小体素重来
MAX_TRIES = 3

# 网页预览细节档位:目标面数(0 = 不降采样,直接用原始网格)
LEVELS = {
    'std': 450_000,
    'high': 1_500_000,
    'ultra': 4_000_000,
    'full': 0,
}


def level_faces(level):
    if level not in LEVELS:
        raise ValueError(f'未知的网格细节档位: {level}(可选 {" / ".join(LEVELS)})')
    return LEVELS[level]


def cache_paths(paths, level='std'):
    """std 沿用旧文件名,新档位加后缀,互不覆盖。"""
    level_faces(level)
    suffix = '' if level == 'std' else f'_{level}'
    return (os.path.join(paths['crack'], f'mesh_web{suffix}.npz'),
            os.path.join(paths['crack'], f'mesh_web{suffix}.json'))


def ply_counts(path):
    """不读顶点数据,只解析 PLY 头,拿顶点/面数。"""
    nv = nf = None
    try:
        with open(path, 'rb') as f:
            for _ in range(400):
                line = f.readline()
                if not line:
                    break
                s = line.decode('ascii', 'replace').strip()
                if s.startswith('element vertex'):
                    nv = int(s.split()[-1])
                elif s.startswith('element face'):
                    nf = int(s.split()[-1])
                elif s == 'end_header':
                    break
    except OSError:
        pass
    return nv, nf


def cached_info(paths, level='std', display=None):
    """缓存有效则返回 meta,否则 None。

    display: 当前检测记录的 display.npz。换了检测记录时顶点颜色要重做,
    所以带 display 访问且 mtime 变了就让缓存失效。
    """
    npz, js = cache_paths(paths, level)
    if not (os.path.isfile(npz) and os.path.isfile(js)):
        return None
    src = paths['mesh']
    try:
        st = os.stat(src)
        with open(js, encoding='utf-8') as f:
            meta = json.load(f)
    except OSError:
        return None
    if meta.get('source_size') != st.st_size:
        return None
    if abs(float(meta.get('source_mtime', 0)) - st.st_mtime) > 1.0:
        return None
    if display and os.path.isfile(display):
        dm = meta.get('display_mtime')
        # 旧缓存没有记录 display 来源,按失效处理一次,保证换记录后颜色跟着变
        if dm is None or abs(float(dm) - os.path.getmtime(display)) > 1.0:
            return None
    return meta


def _emit(progress, done, message=None, stage='mesh'):
    if progress:
        ev = {'event': 'progress', 'stage': stage, 'done': done, 'total': 100}
        if message:
            ev['message'] = message
        progress(ev)


def _cluster(om, voxel):
    sm = om.simplify_vertex_clustering(voxel_size=float(voxel))
    sm.remove_degenerate_triangles()
    sm.remove_duplicated_vertices()
    sm.remove_unreferenced_vertices()
    return sm


def _display_source(paths, display=None):
    if display:
        return display
    disp = os.path.join(paths['crack'], 'display.npz')
    prev = os.path.join(paths['crack'], 'preview.npz')
    return disp if os.path.isfile(disp) else prev


def _colors_from_display(V, src, cell):
    """体素查表把显示点云颜色搬到网格顶点;返回 uint8[N,3] 或 None。"""
    if not src or not os.path.isfile(src):
        return None
    z = np.load(src)
    P = z['positions'].astype(np.float32)
    C = z['colors']
    if len(P) == 0 or len(C) != len(P) or C.ndim != 2 or C.shape[1] < 3:
        return None
    C = C[:, :3].astype(np.uint8)

    mn = P.min(axis=0)
    K = np.floor((P - mn) / cell).astype(np.int64)
    key = (K[:, 0] << 42) | (K[:, 1] << 21) | K[:, 2]
    uniq, inv = np.unique(key, return_inverse=True)
    acc = np.zeros((len(uniq), 3), np.float32)
    cnt = np.zeros(len(uniq), np.float32)
    np.add.at(acc, inv, C.astype(np.float32))
    np.add.at(cnt, inv, 1)
    avg = (acc / cnt[:, None]).astype(np.uint8)

    MK = np.floor((V - mn) / cell).astype(np.int64)
    mkey = (MK[:, 0] << 42) | (MK[:, 1] << 21) | MK[:, 2]
    idx = np.searchsorted(uniq, mkey)
    ok = idx < len(uniq)
    ok[ok] &= uniq[idx[ok]] == mkey[ok]

    out = np.full((len(V), 3), 165, np.uint8)
    out[ok] = avg[idx[ok]]
    miss = np.flatnonzero(~ok)
    if len(miss):
        from scipy.spatial import cKDTree
        tree = cKDTree(P)
        _, j = tree.query(V[miss], k=1, workers=-1)
        out[miss] = C[j]
    return out


def build(paths, progress=None, level='std', target_faces=None, display=None):
    """生成浏览器用网格缓存,返回 meta。

    level: std(45 万) / high(150 万) / ultra(400 万) / full(原始面数)。
    """
    import trimesh
    import open3d as o3d

    target_faces = level_faces(level) if target_faces is None else int(target_faces)
    src = paths['mesh']
    if not os.path.isfile(src):
        raise RuntimeError(f'没有网格文件: {src}')
    os.makedirs(paths['crack'], exist_ok=True)
    npz_path, js_path = cache_paths(paths, level)

    t0 = time.time()
    _emit(progress, 3, '读取网格 (%.0f MB)…' % (os.path.getsize(src) / 1e6))
    m = trimesh.load(src, process=False)
    if isinstance(m, trimesh.Scene):
        try:
            m = m.dump(concatenate=True)
        except TypeError:
            m = trimesh.util.concatenate(tuple(m.geometry.values()))
    V = np.asarray(m.vertices, np.float64)
    F = np.asarray(m.faces, np.int32)
    if len(F) == 0:
        raise RuntimeError('这个文件里没有三角形,不是网格:' + src)
    n_src_faces = int(len(F))
    _emit(progress, 12, f'网格 {n_src_faces:,} 面,{len(V):,} 顶点')

    lo, hi = V.min(axis=0), V.max(axis=0)
    diag = float(np.linalg.norm(hi - lo))
    # full(0)=不降采样;其余按目标面数决定(原始网格不超过 90 万面时也用原样)
    decimated = target_faces > 0 and n_src_faces > max(target_faces, KEEP_FACES)
    # 面数约与体素平方成反比;diag/600 实测约 45 万面,据此估算初值少试一轮
    voxel = diag / 600.0
    if decimated:
        voxel *= (float(TARGET_FACES) / float(target_faces)) ** 0.5

    if decimated:
        om = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F))
        sm = None
        for attempt in range(MAX_TRIES):
            _emit(progress, 20 + attempt * 15,
                  f'简化网格 (体素 {voxel:.4f})…')
            sm = _cluster(om, voxel)
            nf = len(sm.triangles)
            if MIN_FACES <= nf <= 2.5 * target_faces or attempt == MAX_TRIES - 1:
                break
            # 面数约与体素平方成反比
            voxel *= float(np.clip((nf / target_faces) ** 0.5, 0.3, 3.0))
        Vw = np.asarray(sm.vertices, np.float64)
        Fw = np.asarray(sm.triangles, np.int32)
        _emit(progress, 65, f'简化完成:{n_src_faces:,} → {len(Fw):,} 面')
        del om, V, F
    else:
        Vw, Fw = V, F
        sm = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(Vw), o3d.utility.Vector3iVector(Fw))
        _emit(progress, 65, f'网格 {len(Fw):,} 面,无需简化')

    _emit(progress, 70, '计算法线…')
    sm.compute_vertex_normals()
    N = np.asarray(sm.vertex_normals, np.float32)
    if len(N) != len(Vw):
        N = None

    _emit(progress, 78, '从点云转移顶点颜色…')
    dsrc = _display_source(paths, display)
    C = _colors_from_display(Vw.astype(np.float32), dsrc, diag / 600.0)

    _emit(progress, 92, '写入缓存…')
    pos = Vw.astype(np.float32)
    idx = Fw.astype(np.uint32).reshape(-1)
    arrays = {'positions': pos, 'indices': idx}
    if C is not None:
        arrays['colors'] = C
    if N is not None:
        arrays['normals'] = N.astype(np.float32)
    tmp = npz_path + '.tmp'
    with open(tmp, 'wb') as f:
        np.savez(f, **arrays)
    os.replace(tmp, npz_path)

    st = os.stat(src)
    meta = {
        'scene': paths['name'],
        'level': level,
        'target_faces': int(target_faces),
        'source': src,
        'source_size': st.st_size,
        'source_mtime': st.st_mtime,
        'source_faces': n_src_faces,
        'n_verts': int(len(pos)),
        'n_faces': int(len(idx) // 3),
        'decimated': bool(decimated),
        'voxel': round(float(voxel), 6),
        'has_colors': C is not None,
        'has_normals': N is not None,
        'display': dsrc if dsrc and os.path.isfile(dsrc) else None,
        'display_mtime': os.path.getmtime(dsrc) if dsrc and os.path.isfile(dsrc) else None,
        'bbox_min': lo.tolist(),
        'bbox_max': hi.tolist(),
        'created': datetime.now().isoformat(timespec='seconds'),
        'seconds': round(time.time() - t0, 2),
    }
    tmp = js_path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    os.replace(tmp, js_path)
    _emit(progress, 100, f'网格就绪:{meta["n_faces"]:,} 面 / '
                          f'{meta["n_verts"]:,} 顶点,{meta["seconds"]:.1f}s')
    return meta
