# -*- coding: utf-8 -*-
"""把 OpenMVS 的贴图网格(scene_textured.ply + scene_textured*.png)转成浏览器缓存。

OpenMVS 的多贴图 PLY 结构:

    顶点: x, y, z
    面:   vertex_indices(3 个 uint32) + texcoord(6 个 float,三个角的 UV)
          + texnumber(int32,指向哪个贴图文件)

贴图是 ``scene_textured0.png`` … ``scene_texturedN.png``,每张最多 8192²。

输出到 ``work/<scene>/crack/tex_web/``:

    arrays.npz   positions / normals / uv / indices
                 (按 texnumber 分组、组内按 (顶点, UV) 去重后连续存放,
                  分组边界在 meta 里)
    meta.json    分组、面数、贴图尺寸、源文件大小/修改时间
    tex_<i>.jpg  降采样后的贴图(默认 4096²,质量 90)

为什么不能直接用 PLY 的全局顶点号索引 UV:PLY 的 texcoord 是**逐角**的
(每个角 2 个 float),同一个顶点在不同贴图/接缝上可以有不同 UV。网页的
顶点属性只能逐顶点,所以每组按 (顶点, UV) 去重重新编号:组内顶点 0..m-1,
索引也改用组内号。直接拿全局顶点号去索引逐角 UV 数组会整片花掉。

坐标约定:OpenMVS 的 UV 是 OpenGL 约定(v=0 是图像底边)。Three.js 贴图
默认 ``flipY=true``,正好对上;用 numpy 按行采样时要取 ``1 - v``。
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime

import numpy as np

TEX_SIZE = 4096               # 网页用贴图边长(原始是 8192)
JPEG_QUALITY = 90
NORMAL_CHUNK = 1_000_000      # 算法线时每次处理的面数,控制内存峰值
# 缓存格式版本。2 = 每组按 (顶点, UV) 去重、组内局部索引(v1 是逐角 UV 直接
# 给全局顶点号索引,渲染是花的,必须让旧缓存失效)
FORMAT = 2


def source_ply(paths):
    return os.path.join(os.path.dirname(paths['mesh']), 'scene_textured.ply')


def cache_dir(paths):
    return os.path.join(paths['crack'], 'tex_web')


def array_paths(paths):
    d = cache_dir(paths)
    return os.path.join(d, 'arrays.npz'), os.path.join(d, 'meta.json')


def _emit(progress, done, message=None, stage='mesh'):
    if progress:
        ev = {'event': 'progress', 'stage': stage, 'done': done, 'total': 100}
        if message:
            ev['message'] = message
        progress(ev)


def _read_header(f):
    nv = nf = None
    texfiles = []
    while True:
        raw = f.readline()
        if not raw:
            raise RuntimeError('PLY 头不完整')
        s = raw.decode('ascii', 'replace').strip()
        if s.startswith('element vertex'):
            nv = int(s.split()[-1])
        elif s.startswith('element face'):
            nf = int(s.split()[-1])
        elif s.startswith('comment TextureFile'):
            parts = s.split(None, 2)
            if len(parts) == 3:
                texfiles.append(parts[2])
        elif s == 'end_header':
            break
    if not nv or not nf:
        raise RuntimeError('PLY 头里没有 vertex/face 数量')
    return nv, nf, texfiles


# 三角形 + 6 个 UV + texnumber 的定长记录(OpenMVS 只会写这种)
FACE_DTYPE = np.dtype([
    ('n', 'u1'), ('a', '<u4', (3,)),
    ('m', 'u1'), ('uv', '<f4', (6,)),
    ('tex', '<i4'),
])


def _vertex_normals(V, idx, nv):
    """面积加权的顶点法线(分块累加,避免一次性开几个 GB)。"""
    acc = np.zeros((nv, 3), np.float32)
    nf = len(idx)
    for s in range(0, nf, NORMAL_CHUNK):
        a = idx[s:s + NORMAL_CHUNK]
        p0 = V[a[:, 0]]
        p1 = V[a[:, 1]]
        p2 = V[a[:, 2]]
        fn = np.cross(p1 - p0, p2 - p0)
        for k in range(3):
            col = a[:, k]
            for c in range(3):
                acc[:, c] += np.bincount(col, weights=fn[:, c], minlength=nv)
    norm = np.linalg.norm(acc, axis=1, keepdims=True)
    acc /= np.maximum(norm, 1e-12)
    return acc


def _downscale_textures(ply_dir, texfiles, outdir, size, progress=None):
    """把 8192² 的贴图缩到 size²,存成 jpg;返回实际写出的文件名列表。"""
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None       # 8192² 超过 Pillow 的默认警告阈值
    written = []
    n = len(texfiles)
    for i, name in enumerate(texfiles):
        src = os.path.join(ply_dir, name)
        dst = os.path.join(outdir, f'tex_{i}.jpg')
        _emit(progress, 55 + int(30 * i / max(n, 1)),
              f'贴图 {i + 1}/{n}:{name}')
        if not os.path.isfile(src):
            continue
        with Image.open(src) as im:
            im = im.convert('RGB')
            if max(im.size) > size:
                im = im.resize((size, size), Image.LANCZOS)
            im.save(dst, 'JPEG', quality=JPEG_QUALITY, subsampling=0)
        written.append(os.path.basename(dst))
    return written


def build(paths, progress=None, tex_size=TEX_SIZE):
    """生成贴图预览缓存,返回 meta。"""
    src = source_ply(paths)
    if not os.path.isfile(src):
        raise RuntimeError(f'没有贴图网格:{src}(先跑 TextureMesh)')
    outdir = cache_dir(paths)
    os.makedirs(outdir, exist_ok=True)
    t0 = time.time()

    _emit(progress, 2, '读取贴图网格 (%.0f MB)…' % (os.path.getsize(src) / 1e6))
    with open(src, 'rb') as f:
        nv, nf, texfiles = _read_header(f)
        if not texfiles:
            raise RuntimeError('这个 PLY 没有 TextureFile 注释,不是 OpenMVS 的贴图网格')
        V = np.fromfile(f, dtype='<f4', count=nv * 3)
        if V.size != nv * 3:
            raise RuntimeError('顶点数据不完整')
        V = V.reshape(nv, 3)
        F = np.fromfile(f, dtype=FACE_DTYPE, count=nf)
    if len(F) != nf:
        raise RuntimeError(f'面数据不完整:{len(F)} / {nf}')
    if not np.all(F['n'] == 3) or not np.all(F['m'] == 6):
        raise RuntimeError('这个 PLY 不是三角形 + 6 个 UV 的结构,暂不支持')
    _emit(progress, 10, f'网格 {nf:,} 面 / {nv:,} 顶点,{len(texfiles)} 张贴图')

    idx_all = F['a']
    _emit(progress, 15, '计算顶点法线…')
    N = _vertex_normals(V, idx_all, nv)
    del idx_all

    _emit(progress, 30, '按贴图分组…')
    tex = F['tex']
    order = np.argsort(tex, kind='stable')
    # texnumber 可能为负或空缺,由排序结果直接算每组的真实起止
    uniq_t, first = np.unique(tex[order], return_index=True)
    group_sizes = np.diff(np.append(first, nf))
    A = F['a'][order]                       # 按贴图排序后的顶点索引
    UV = F['uv'][order].reshape(nf, 3, 2)   # 每个角的 UV
    del F, order, tex

    # 关键:PLY 的 UV 是「逐角」的,同一个顶点可能在不同贴图/接缝上有不同 UV。
    # 网页顶点属性只能逐顶点,所以每组按 (顶点, UV) 去重后重新编号,组内索引
    # 从 0 开始;不能拿全局顶点号直接索引一个逐角排布的 UV 数组。
    KEY_DTYPE = np.dtype([('v', '<u4'), ('u', '<f4'), ('w', '<f4')])
    pos_parts, nor_parts, uv_parts, idx_parts = [], [], [], []
    groups = []
    vbase = ibase = 0
    face_pos = 0
    for gi, (t, c) in enumerate(zip(uniq_t.tolist(), group_sizes.tolist())):
        sl = slice(face_pos, face_pos + c)
        a = A[sl]                           # (c, 3) 全局顶点号
        uvc = UV[sl]                        # (c, 3, 2)
        key = np.empty(c * 3, KEY_DTYPE)
        key['v'] = a.ravel()
        key['u'] = uvc[..., 0].ravel()
        key['w'] = uvc[..., 1].ravel()
        uniq_k, inv = np.unique(key, return_inverse=True)
        del key
        m = len(uniq_k)
        if 0 <= t < len(texfiles):
            pos_parts.append(V[uniq_k['v']])
            nor_parts.append(N[uniq_k['v']])
            uv_parts.append(np.stack([uniq_k['u'], uniq_k['w']], axis=1))
            idx_parts.append(inv.astype(np.uint32, copy=False))
            groups.append({'tex': int(t),
                           'vstart': int(vbase), 'vcount': int(m),
                           'istart': int(ibase), 'icount': int(c * 3)})
            vbase += m
            ibase += c * 3
        else:
            # 指向不存在贴图的面直接丢掉
            groups.append(None)
        face_pos += c
        if gi % 4 == 3:
            _emit(progress, 30 + int(15 * (gi + 1) / max(len(uniq_t), 1)),
                  f'分组 {gi + 1}/{len(uniq_t)}:{m:,} 顶点')
    groups = [g for g in groups if g is not None]
    del A, UV

    positions = np.concatenate(pos_parts)
    normals = np.concatenate(nor_parts)
    uv = np.concatenate(uv_parts)
    indices = np.concatenate(idx_parts)
    del pos_parts, nor_parts, uv_parts, idx_parts

    _emit(progress, 45, '写入数组缓存 (%.0f MB)…' %
          ((positions.nbytes + normals.nbytes + uv.nbytes + indices.nbytes) / 1e6))
    npz_path, js_path = array_paths(paths)
    tmp = npz_path + '.tmp'
    with open(tmp, 'wb') as f:
        np.savez(f, positions=positions, normals=normals, indices=indices, uv=uv)
    os.replace(tmp, npz_path)
    del positions, normals, uv, indices

    tex_written = _downscale_textures(os.path.dirname(src), texfiles, outdir,
                                      int(tex_size), progress)

    st = os.stat(src)
    meta = {
        'format': FORMAT,
        'scene': paths['name'],
        'source': src,
        'source_size': st.st_size,
        'source_mtime': st.st_mtime,
        'source_faces': int(nf),
        'n_verts': int(nv),
        'n_web_verts': int(vbase),
        'n_faces': int(nf),
        'n_corners': int(nf * 3),
        'tex_size': int(tex_size),
        'tex_files': tex_written,
        'groups': groups,
        'created': datetime.now().isoformat(timespec='seconds'),
        'seconds': round(time.time() - t0, 2),
    }
    tmp = js_path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    os.replace(tmp, js_path)
    _emit(progress, 100, f'贴图网格就绪:{nf:,} 面,{len(groups)} 组贴图,'
                          f'{meta["seconds"]:.1f}s')
    return meta


def cached_info(paths):
    """缓存有效则返回 meta,否则 None。"""
    npz, js = array_paths(paths)
    src = source_ply(paths)
    if not (os.path.isfile(npz) and os.path.isfile(js) and os.path.isfile(src)):
        return None
    try:
        st = os.stat(src)
        with open(js, encoding='utf-8') as f:
            meta = json.load(f)
    except (OSError, ValueError):
        return None
    if meta.get('format') != FORMAT:
        return None
    if meta.get('source_size') != st.st_size:
        return None
    if abs(float(meta.get('source_mtime', 0)) - st.st_mtime) > 1.0:
        return None
    for name in meta.get('tex_files', []):
        if not os.path.isfile(os.path.join(cache_dir(paths), name)):
            return None
    return meta
