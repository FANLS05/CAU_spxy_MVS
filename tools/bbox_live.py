# -*- coding: utf-8 -*-
"""
实时包围盒投影:左边画框,右边一个常驻的 3D 窗口即时跟着变。

和 bbox_gui.py 相比改了两件事:

  1. 投射从 ~20 秒降到 ~30 毫秒。
     慢的根源是 ray_triangle_hits:它拿 160 条射线对着全部 165 万个三角形
     硬碰。这里先在图像空间建一张 2D 网格(每个格子记下「投影会落到这里」
     的三角形),每条射线只跟一个格子里的候选面求交。
     剔除是**可证明安全**的:射线从光心 C 穿过像素 p 打到三角形上的点 X,
     X 必然投影回 p,所以该三角形的投影包围盒必然含 p —— 包围盒不含 p 的
     三角形根本打不到,裁掉不影响结果。

  2. 3D 窗口不再阻塞 Tk。
     b2m.visualize 里是 vis.run(),它开着的时候整个 Tk 界面是死的。这里改成
     常驻一个 Open3D 窗口,由 root.after(16ms) 泵 poll_events/update_renderer,
     所以拖动框的时候右边能实时刷,而且不影响在 3D 里旋转缩放。

正确性基准是 b2m.project_bbox —— --verify 会把两者的所有返回值逐位对比。
计算部分这里一个字都没改,连光线的构造顺序都照抄,就是为了让对比能过。

bbox_to_mesh.py 和 bbox_gui.py **只读导入,不改**。

用法(在 0916 目录下):
    python "D:\\code\\point_cloud\\0916\\tools\\bbox_live.py" --scene "D:\\code\\point_cloud\\0916\\work\\food\\dense\\sparse"

    # 只做正确性自检(逐位对比 old vs new + 测速),不开窗口
    python "D:\\code\\point_cloud\\0916\\tools\\bbox_live.py" --verify `
        --scene "D:\\code\\point_cloud\\0916\\work\\food\\dense\\sparse" `
        --mesh  "D:\\code\\point_cloud\\0916\\work\\food\\mvs\\scene_dense_mesh.ply"

--mesh / --images 不写会自动从 --scene 推(复用 bbox_gui 的推断逻辑)。
"""
import argparse
import os
import sys
import time
import traceback
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bbox_to_mesh as b2m                                   # noqa: E402
import bbox_gui as bg                                        # noqa: E402

try:
    from PIL import Image, ImageTk
except ImportError:
    raise SystemExit('缺 pillow,请先装:pip install pillow')


# =============================================================================
# 一、加速版投射
# =============================================================================
# 网格相关的缓存放这里,不往 b2m 上挂 —— 那个文件是只读导入,一个字节都不改。
_MESH = {'V': None, 'F': None}


def set_mesh(V, F):
    """缓存网格。F 统一成 ndarray,后面反复用。"""
    F = np.asarray(F)
    _MESH['V'] = V
    _MESH['F'] = F


def build_index(F, cam, cell=32, max_cells=64, verbose=True):
    """把「这张图的每个三角形会投到哪片像素」预先算好,做成 CSR 网格。

    图像相关的东西全部缓存,切图才重算;网格建好之后每次查询只花十几毫秒。

    返回的 dict 里:
      u, v, d     所有顶点投影后的像素坐标 + 相机系深度(和 project_bbox 同源)
      csr         (starts, counts, faces) —— 第 k 格的面是 faces[s:s+c]
      overflow    跨格太多 / 有顶点在相机背后的面,查询时永远参与
    """
    t0 = time.time()
    W, H = int(cam['W']), int(cam['H'])
    K, R, t = cam['K'], cam['R'], cam['t']

    u, v, d = b2m.project_points(_MESH['V'], K, R, t)
    t_proj = time.time()

    Fi = _MESH['F']
    uf, vf, df = u[Fi], v[Fi], d[Fi]

    # 每个三角形投影后的包围盒
    u0 = uf.min(axis=1)
    u1 = uf.max(axis=1)
    v0 = vf.min(axis=1)
    v1 = vf.max(axis=1)
    # 有顶点在相机背后的面,投影没意义,包围盒也不可信 —— 这些永不剔除,
    # 全丢进溢出桶。但「三个顶点**全都**在相机背后」的面是另一回事:
    # 深度在三角形上是仿射的,三个顶点都 <=0 就说明整个三角形上处处 <=0;
    # 而穿过像素的射线方向在相机系里 z 分量恒为 +1(见 ray_from_pixel),
    # 所以射线上任何 t>0 的点深度都 >0 —— 这样的面根本打不到,可以直接丢。
    # 不丢的话,相机万一在网格内部,溢出桶会大到每次查询都退化成暴力版。
    behind = (df <= 0).all(axis=1)
    straddle = ~(df > 0).all(axis=1) & ~behind     # 跨相机平面,不是全在背后
    t_bbox = time.time()

    # 格子编号。负数(画面外)直接夹到 0 —— 夹完只会多算不会少算:
    # 若某面投影包围盒含像素 p,夹过的 [c0,c1] 一定还含 p 所在的列。
    NC = int(W // cell) + 2
    NR = int(H // cell) + 2
    c0 = np.clip(np.floor(u0 / cell).astype(np.int64), 0, NC - 1)
    c1 = np.clip(np.floor(u1 / cell).astype(np.int64), 0, NC - 1)
    r0 = np.clip(np.floor(v0 / cell).astype(np.int64), 0, NR - 1)
    r1 = np.clip(np.floor(v1 / cell).astype(np.int64), 0, NR - 1)

    cw = c1 - c0 + 1
    rh = r1 - r0 + 1
    ncell = cw * rh
    in_grid = (~straddle) & (~behind) & (ncell <= max_cells)
    overflow = np.nonzero(straddle | (~behind & (ncell > max_cells)))[0]

    fi = np.nonzero(in_grid)[0]
    cw_i, rh_i = cw[fi], rh[fi]
    cnt = cw_i * rh_i
    total = int(cnt.sum())

    if total:
        # 把每个面的格子矩形摊平成一维,再按格子号稳定排序 —— 排完就是 CSR
        face_rep = np.repeat(fi, cnt)
        off = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        cw_rep = np.repeat(cw_i, cnt)
        rows = off // cw_rep
        cols = off - rows * cw_rep
        cells = (np.repeat(r0[fi], cnt) + rows) * NC + (np.repeat(c0[fi], cnt) + cols)
        order = np.argsort(cells, kind='stable')
        cells_s = cells[order]
        faces = face_rep[order].astype(np.int32)
        counts = np.bincount(cells_s, minlength=NC * NR).astype(np.int64)
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    else:
        faces = np.zeros(0, dtype=np.int32)
        counts = np.zeros(NC * NR, dtype=np.int64)
        starts = np.zeros(NC * NR, dtype=np.int64)

    idx = {
        'u': u, 'v': v, 'd': d,
        'W': W, 'H': H, 'NC': NC, 'NR': NR, 'CELL': cell,
        'csr': (starts, counts, faces), 'overflow': overflow,
        'n_faces': int(len(Fi)),
        'n_overflow': int(len(overflow)),
        'n_behind': int(behind.sum()),
        'n_scatter': total,
        't_proj': t_proj - t0,
        't_bbox': t_bbox - t_proj,
        't_grid': time.time() - t_bbox,
        't_all': time.time() - t0,
    }
    if verbose:
        print(f'  [索引] {W}x{H},格 {cell}px,{NC}x{NR} 格   '
              f'散布 {total} 项 / 溢出 {len(overflow)} 面 / '
              f'相机背后丢弃 {int(behind.sum())} 面   '
              f'投影 {idx["t_proj"]*1000:.0f}ms + 包围盒 {idx["t_bbox"]*1000:.0f}ms'
              f' + 建格 {idx["t_grid"]*1000:.0f}ms = {idx["t_all"]*1000:.0f}ms')
    return idx


def _rays(pix, K, R, C):
    """逐条算射线。特意写成循环 —— ray_from_pixel 一次算一条,
    和 b2m 里的写法完全一致,浮点结果才能对齐。"""
    o = np.repeat(C[None, :], len(pix), axis=0)
    d = np.stack([b2m.ray_from_pixel(u, v, K, R, C)[1] for u, v in pix])
    return o, d


def cast_rays(idx, origins, dirs, px, py):
    """网格索引裁剪,然后**直接调 b2m.ray_triangle_hits** 对候选面求交。

    这里刻意不自己重写一遍 Möller–Trumbore:候选面的个数每条射线都不一样,
    而 ray_triangle_hits 内部 `q @ d` 走的是 BLAS、`np.einsum` 的归约长度也
    跟候选数有关 —— 自己抄一份的话,浮点结果和暴力版是否逐位一致就成了一个
    要靠运气的问题。直接调原函数,这一层就「按构造保证一致」,--verify 也就
    只需要验真正新的东西:索引裁出来的候选集对不对。

    唯一的约定是候选面要按面号升序 —— ray_triangle_hits 里 np.argmin 取的是
    第一个最小值,两个面算出完全相同的 t 时,得和暴力版挑同一个。
    """
    V, F = _MESH['V'], _MESH['F']
    NC, NR, CELL = idx['NC'], idx['NR'], idx['CELL']
    starts, counts, faces = idx['csr']
    ovf = idx['overflow']

    n = len(origins)
    pts = np.full((n, 3), np.nan)
    hit = np.zeros(n, dtype=bool)

    col = np.clip(np.floor(px / CELL).astype(np.int64), 0, NC - 1)
    row = np.clip(np.floor(py / CELL).astype(np.int64), 0, NR - 1)
    cells = row * NC + col

    for i in range(n):
        k = int(cells[i])
        s, c = int(starts[k]), int(counts[k])
        cand = faces[s:s + c] if c else faces[:0]
        if len(ovf):
            cand = np.sort(np.concatenate([cand, ovf]))
        if len(cand) == 0:
            continue
        p1, h1 = b2m.ray_triangle_hits(origins[i][None, :], dirs[i][None, :],
                                       V[F[cand]])
        if h1[0]:
            pts[i] = p1[0]
            hit[i] = True
    return pts, hit


def _outline_pixels(corners_2d, n_edge=40):
    """沿框一圈的采样点。逐字照抄 bbox_to_mesh.py 里的构造:
    每条边取 s = i/n_edge(i 从 0 到 39),所以**不含远端角点**。
    千万别换成 np.linspace(a, b, 40) —— 那样两端都含,采样位置全变了。"""
    per = []
    for k in range(4):
        a, b = corners_2d[k], corners_2d[(k + 1) % 4]
        for i in range(n_edge):
            s = i / n_edge
            per.append(a * (1.0 - s) + b * s)
    return np.asarray(per, dtype=np.float64)


def fast_project(idx, V, F, cam, bbox):
    """和 b2m.project_bbox 同构,只是把光线求交换成了网格索引版。

    返回的 key 与 project_bbox 完全一致(额外多一个 frag_used,是碎片顶点
    在原网格里的下标,3D 那边拿它取法线用)。
    """
    xmin, ymin, xmax, ymax = bbox
    if xmin > xmax:
        xmin, xmax = xmax, xmin
    if ymin > ymax:
        ymin, ymax = ymax, ymin

    K, R, C = cam['K'], cam['R'], cam['C']
    u, v, d = idx['u'], idx['v'], idx['d']

    corners_2d = np.array([[xmin, ymin], [xmax, ymin],
                           [xmax, ymax], [xmin, ymax]], dtype=np.float64)

    # --- 4 角光线投射 ---
    o4, d4 = _rays(corners_2d, K, R, C)
    hp, hok = cast_rays(idx, o4, d4, corners_2d[:, 0], corners_2d[:, 1])
    hits = [hp[i] if hok[i] else None for i in range(4)]

    # --- 视锥裁剪(u,v,d 是预计算好的,这一段和原来逐字相同) ---
    inside = ((u >= xmin) & (u <= xmax) &
              (v >= ymin) & (v <= ymax) & (d > 0))
    face_inside = inside[F].any(axis=1)

    frag_v, frag_f, frag_used = None, None, None
    if face_inside.any():
        used = np.unique(F[face_inside])
        remap = -np.ones(len(V), dtype=np.int64)
        remap[used] = np.arange(len(used))
        frag_v = V[used]
        frag_f = remap[F[face_inside]]
        frag_used = used

    # --- 沿框的一整圈打密集射线,得到贴着网格表面的完整轮廓 ---
    per = _outline_pixels(corners_2d)

    ol_pts = np.zeros((len(per), 3), dtype=np.float64)
    ol_d = _rays(per, K, R, C)[1]
    op, ook = cast_rays(idx, np.repeat(C[None, :], len(per), axis=0), ol_d,
                        per[:, 0], per[:, 1])
    ol_pts[ook] = op[ook]
    ol_ok = ook

    n = len(per)
    seg = [(i, (i + 1) % n) for i in range(n) if ol_ok[i] and ol_ok[(i + 1) % n]]

    return {
        'bbox': (xmin, ymin, xmax, ymax),
        'corners_2d': corners_2d,
        'hits': hits,
        'n_hit': int(hok.sum()),
        'frag_v': frag_v,
        'frag_f': frag_f,
        'frag_used': frag_used,
        'face_mask': face_inside,
        'n_tri': int(face_inside.sum()),
        'n_faces_total': len(F),
        'outline_pts': ol_pts,
        'outline_seg': np.asarray(seg, dtype=np.int64).reshape(-1, 2),
        'n_outline_hit': int(ol_ok.sum()),
        'n_outline': n,
    }


# =============================================================================
# 二、正确性自检:--verify
# =============================================================================
def _same(a, b):
    """两个数组/标量是不是逐位一致。None 和 nan 都要能正确比较。"""
    if a is None or b is None:
        return a is None and b is None
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return False
    return bool(np.array_equal(a, b, equal_nan=True))


def _same_hits(a, b):
    """hits 是「4 个元素,每个是 None 或一个 3 维点」的列表。
    不能直接丢给 np.array_equal —— None 和 ndarray 混在一起会变成 object
    数组,逐个 == 又返回数组,比较结果不可靠。这里逐项手比。"""
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if (x is None) != (y is None):
            return False
        if x is not None and not np.array_equal(x, y):
            return False
    return True


def verify(cfg, cell=32, boxes=None):
    """跑 fast_project 和 b2m.project_bbox,把每个 key 逐位比一遍。"""
    try:
        sys.stdout.reconfigure(line_buffering=True)   # 每行都吐,别等跑完才出
    except Exception:
        pass
    print('=' * 72)
    print('正确性自检:fast_project  vs  bbox_to_mesh.project_bbox')
    print('=' * 72)

    t0 = time.time()
    V, F, vcols = b2m.load_mesh(cfg['mesh'])
    print(f'网格:{len(V)} 顶点 / {len(F)} 三角形   ({time.time()-t0:.1f}s)')
    set_mesh(V, F)

    scene = b2m.load_scene(cfg['scene'])
    names = [nm for nm in sorted(scene)
             if os.path.isfile(os.path.join(cfg['images'], nm))
             or os.path.isfile(os.path.join(cfg['images'], os.path.basename(nm)))]
    if not names:
        raise SystemExit(f'图像目录里一张都找不到:{cfg["images"]}')

    keys = ['bbox', 'corners_2d', 'hits', 'n_hit', 'frag_v', 'frag_f',
            'face_mask', 'n_tri', 'n_faces_total', 'outline_pts',
            'outline_seg', 'n_outline_hit', 'n_outline']

    n_img = min(2, len(names))
    all_ok = True
    t_old_tot = t_new_tot = t_pre = 0.0

    for nm in names[:n_img]:
        cam = scene[nm]
        W, H = int(cam['W']), int(cam['H'])

        t0 = time.time()
        idx = build_index(F, cam, cell=cell)
        t_pre += time.time() - t0

        if boxes is None:
            bs = [(0.25 * W, 0.25 * H, 0.75 * W, 0.75 * H),
                  (0.40 * W, 0.40 * H, 0.55 * W, 0.55 * H),
                  (0.05 * W, 0.05 * H, 0.35 * W, 0.30 * H),
                  (0.75 * W, 0.75 * H, 0.25 * W, 0.25 * H),   # 故意反着写
                  (0.50 * W, 0.50 * H, 0.50 * W + 1, 0.50 * H + 1)]
        else:
            bs = boxes

        for j, bb in enumerate(bs):
            t0 = time.time()
            ref = b2m.project_bbox(V, F, cam['K'], cam['R'], cam['t'], cam['C'], bb)
            t_old = time.time() - t0

            t0 = time.time()
            got = fast_project(idx, V, F, cam, bb)
            t_new = time.time() - t0

            t_old_tot += t_old
            t_new_tot += t_new

            bad = [k for k in keys
                   if not (_same_hits(ref[k], got[k]) if k == 'hits'
                           else _same(ref[k], got.get(k)))]
            tag = 'OK ' if not bad else 'FAIL'
            if bad:
                all_ok = False
            print(f'  [{tag}] {nm}  框{j} '
                  f'({bb[0]:.0f},{bb[1]:.0f})-({bb[2]:.0f},{bb[3]:.0f})   '
                  f'old {t_old*1000:8.1f} ms  ->  new {t_new*1000:7.1f} ms   '
                  f'面 {got["n_tri"]:>7d}  轮廓 {got["n_outline_hit"]}/{got["n_outline"]}')
            if bad:
                for k in bad:
                    a, b = ref[k], got.get(k)
                    print(f'        ✗ 键 {k} 不一致')
                    print(f'            old = {a!r}'[:200])
                    print(f'            new = {b!r}'[:200])

    print('-' * 72)
    print(f'每图预计算(投影+包围盒+建格):{t_pre/max(1,n_img)*1000:.0f} ms / 图')
    print(f'暴力版合计 {t_old_tot*1000:.1f} ms   加速版合计 {t_new_tot*1000:.1f} ms   '
          f'加速 {t_old_tot/max(1e-9, t_new_tot):.0f}x')
    print('-' * 72)
    if all_ok:
        print('✅ 全部逐位一致 —— 加速版和原版算出来的东西一模一样。')
    else:
        print('❌ 有不一致的键,别用,先修。')
    return all_ok


# =============================================================================
# 三、常驻 3D 窗口
# =============================================================================
class Live3D:
    """一个常驻的 Open3D 窗口。不调 vis.run(),由外面泵。"""

    def __init__(self, V, F, vcols, uv, tex_img, cam, res, cfg,
                 left=80, top=80, width=900, height=900):
        import open3d as o3d
        self.o3d = o3d
        # 相机状态(C/K/R/W/H)不在这里赋 —— 统一由 _rebuild_static 维护,
        # 它就在本函数末尾被调用。分两处赋过一次的教训见 _rebuild_static 的注释。
        self.closed = False

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name='3D —— 左键旋转 / 滚轮缩放 / 中键平移',
                               width=int(width), height=int(height),
                               left=int(left), top=int(top), visible=True)
        opt = self.vis.get_render_option()
        opt.background_color = np.asarray([0.8, 0.8, 1.0])
        opt.point_size = 1.0

        self.extent = float(np.linalg.norm(V.max(0) - V.min(0)))
        # eps 是**相对量**(无量纲):离相机越远的点要抬得越多才压得住深度
        # 冲突,因为真正决定胜负的是相对深度分辨率。所以用「以相机为中心的
        # 相似变换」—— C + (p-C)*(1-eps) —— 而不是给每个点加同一个位移;
        # 后者各点方向不同,整个碎片会被轻微剪切变形。
        #
        # ★ 符号别写反:要压过主网格,碎片必须离相机**更近**,也就是把
        #   (p-C) 乘一个**小于 1** 的系数。写成 (1+eps) 的话碎片会跑到主
        #   网格后面去,正面完全看不见,只能从背面透出来。
        self.eps = cfg['eps'] if cfg['eps'] is not None else 1e-4

        # ---------------- 主网格(带贴图),只加一次,之后永不重建 --------
        m = o3d.geometry.TriangleMesh()
        m.vertices = o3d.utility.Vector3dVector(np.asarray(V))
        m.triangles = o3d.utility.Vector3iVector(np.asarray(F))
        if uv is not None and tex_img is not None:
            m.triangle_uvs = o3d.utility.Vector2dVector(
                np.asarray(uv, dtype=np.float64).reshape(-1, 2))
            m.textures = [tex_img]
            m.triangle_material_ids = o3d.utility.IntVector(
                np.zeros(len(F), dtype=np.int32))
        elif vcols is not None:
            m.vertex_colors = o3d.utility.Vector3dVector(vcols)
        else:
            m.paint_uniform_color([0.72, 0.72, 0.75])
        m.compute_vertex_normals()
        self.main = m
        self.main_normals = np.asarray(m.vertex_normals).copy()
        self.vis.add_geometry(m)

        # ---------------- 被框中的碎片(青色)--------------------------
        # 不再像 visualize() 那样把选中面从主网格里挖掉 —— 实测每次重建
        # 165 万面的带贴图网格要 242ms(哪怕内容没变),拖动时会卡成幻灯片。
        # 这里主网格保持不动,碎片以相机为中心放大 (1+eps) 倍,靠深度测试赢出来。
        self.frag = o3d.geometry.TriangleMesh()
        self.frag.vertices = o3d.utility.Vector3dVector(np.zeros((3, 3)))
        self.frag.triangles = o3d.utility.Vector3iVector(
            np.zeros((1, 3), dtype=np.int32))     # 面积为零,不显示
        self.vis.add_geometry(self.frag)

        # ---------------- 各种线 + 光心球 -----------------------------
        self.ls_plane = self._new_lineset()      # 蓝:虚拟像平面
        self.ls_box2d = self._new_lineset()      # 黄:像平面上的 2D 框
        self.ls_frustum = self._new_lineset()    # 蓝:光心->像平面四角
        # 4 条角光线拆成两个 LineSet:Open3D 0.20 的 LineSet 没有 vertex_colors,
        # 没法给同一个 LineSet 里的线分别上色,只能一色一个。
        self.ls_rays_hit = self._new_lineset()   # 品红:打中网格的
        self.ls_rays_miss = self._new_lineset()  # 灰:没打中的
        self.ls_outline = self._new_lineset()    # 红:投到表面的 3D 轮廓

        self.sphere = o3d.geometry.TriangleMesh.create_sphere(
            radius=0.012 * self.extent)
        self.sphere.paint_uniform_color([0.0, 0.85, 0.1])
        self.sphere.compute_vertex_normals()
        self._sphere_c = np.zeros(3)
        self.vis.add_geometry(self.sphere)

        self._rebuild_static(cam)      # 同时完成 self.cam/K/R/W/H/C 的首次赋值
        self.update(res, cam)
        self.vis.reset_view_point(True)

    # ------------------------------------------------------------ 小工具
    def _new_lineset(self):
        ls = self.o3d.geometry.LineSet()
        # 先给一个点和一条退化线再 add_geometry,否则 Open3D 立刻会为空
        # 几何体打警告(原因见 _set_lineset)。所有 LineSet 都会在
        # reset_view_point 之前被填上真正的内容,不影响取景。
        ls.points = self.o3d.utility.Vector3dVector(np.zeros((1, 3)))
        ls.lines = self.o3d.utility.Vector2iVector(
            np.zeros((1, 2), dtype=np.int32))
        self.vis.add_geometry(ls)
        return ls

    def _set_lineset(self, ls, pts, lines, color):
        """给 LineSet 填内容。

        ★ 千万别让 points 或 lines 变成空的。
        Open3D 的 LineSet::IsEmpty() 定义是「没点 **或** 没线」,只要为空,
        **每一帧**渲染都会打三条:
            [Open3D WARNING] [SimpleShaderForLineSet] Binding failed ...
            ... Binding failed when preparing data.
            ... Something is wrong in compiling or binding.
        泵是每秒 30 帧在跑的,所以框一旦全打在模型外面(4 条角光线全落空),
        警告就会一直刷 —— 看着就像程序一直在报错。
        这里塞一条长度为零的退化线 [0,0] 把 IsEmpty() 顶掉:画不出任何东西,
        但 shader 能正常绑定。
        """
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
        lines = np.asarray(lines, dtype=np.int32).reshape(-1, 2)
        if len(pts) == 0:
            pts = np.zeros((1, 3))
        if len(lines) == 0:
            lines = np.zeros((1, 2), dtype=np.int32)     # [0,0],零长度
        ls.points = self.o3d.utility.Vector3dVector(pts)
        ls.lines = self.o3d.utility.Vector2iVector(lines)
        ls.paint_uniform_color(color)
        self.vis.update_geometry(ls)

    def _closed_lines(self, n):
        if n < 2:
            return np.zeros((0, 2), dtype=np.int32)
        idx = [[i, i + 1] for i in range(n - 1)]
        if n > 2:
            idx.append([n - 1, 0])
        return np.asarray(idx, dtype=np.int32)

    def _scale(self):
        """虚拟像平面的深度 —— 和 b2m.visualize 里的算法一致。

        不能缓存:深度是相机系的,换一张图 R/t 就变了。
        (只在换图时算一次,49ms,不值得为它引入按图缓存。)
        """
        _, _, d = b2m.project_points(_MESH['V'], self.K, self.R, self.cam['t'])
        dd = d[d > 0]
        med = float(np.median(dd)) if len(dd) else self.extent
        return 0.45 * med

    # ------------------------------------------------------- 静态几何重建
    def _rebuild_static(self, cam):
        """换图时重建:像平面、锥、光心球;并同步所有相机状态。

        K/R/t 必须在这里一起换掉,不能只在 __init__ 里赋一次 —— 像平面四角
        (point_at_depth)、黄色 2D 框、4 条角光线的方向全都由 K/R 决定,只更新
        光心 C 的话,换图后锥形和框还是指着上一张图的朝向。
        """
        self.cam = cam
        self.K, self.R = cam['K'], cam['R']
        self.W, self.H = int(cam['W']), int(cam['H'])

        C = np.asarray(cam['C'], dtype=np.float64)
        self.C = C
        scale = self._scale()

        pc = [b2m.point_at_depth(u, v, self.K, self.R, C, scale)
              for u, v in [(0, 0), (self.W, 0), (self.W, self.H), (0, self.H)]]
        self._plane_pts = np.asarray(pc, dtype=np.float64)
        self._scale_val = scale

        self._set_lineset(self.ls_plane, self._plane_pts,
                          self._closed_lines(4), [0.1, 0.35, 1.0])

        fr = np.empty((8, 3), dtype=np.float64)
        fr[0::2] = C
        fr[1::2] = self._plane_pts
        self._set_lineset(self.ls_frustum, fr,
                          np.asarray([[0, 1], [2, 3], [4, 5], [6, 7]], np.int32),
                          [0.1, 0.35, 1.0])

        dlt = C - self._sphere_c
        if np.any(dlt):
            self.sphere.translate(dlt)
            self._sphere_c = C.copy()
            self.vis.update_geometry(self.sphere)

    # --------------------------------------------------------------- 刷新
    def update(self, res, cam=None):
        if cam is not None and cam is not self.cam:
            self._rebuild_static(cam)      # 内部会同步 self.cam/K/R/W/H/C

        C = self.C
        corners = res['corners_2d']

        # 像平面上的 2D 框(黄)
        bp = np.asarray([b2m.point_at_depth(u, v, self.K, self.R, C,
                                            self._scale_val) for u, v in corners])
        self._set_lineset(self.ls_box2d, bp, self._closed_lines(4), [1.0, 0.9, 0.0])

        # 4 条角光线(品红=命中 / 灰=没打到网格)
        hits = res['hits']
        pts8 = np.empty((8, 3), dtype=np.float64)
        hl, ml = [], []
        for i in range(4):
            _, dv = b2m.ray_from_pixel(corners[i][0], corners[i][1], self.K, self.R, C)
            h = hits[i]
            pts8[2 * i] = C
            pts8[2 * i + 1] = C + dv * (self._scale_val * 4) if h is None else h
            (ml if h is None else hl).append([2 * i, 2 * i + 1])
        self._set_lineset(self.ls_rays_hit, pts8,
                          np.asarray(hl, np.int32).reshape(-1, 2), b2m.COL_RAY_HIT)
        self._set_lineset(self.ls_rays_miss, pts8,
                          np.asarray(ml, np.int32).reshape(-1, 2), b2m.COL_RAY_MISS)

        # 被框中的网格碎片(青色) —— 以相机为中心放大 (1+eps) 倍,压过主网格
        # 空的时候**不要**把顶点数设成 0:Open3D 每次 update_geometry 遇到
        # 0 个顶点都会打一行 "[Open3D WARNING] The number of points is 0 when
        # creating axis-aligned bounding box",拖动时一秒十几行,看着就像一直在
        # 报错。改成放一个面积为零的退化三角形(三个点都取网格的第 0 个顶点,
        # 保证在场景包围盒内部,不会把视图撑歪),既不显示东西也不报警告。
        fv, ff = res['frag_v'], res['frag_f']
        if fv is None or ff is None or len(ff) == 0:
            v0 = _MESH['V'][:1]
            self.frag.vertices = self.o3d.utility.Vector3dVector(
                np.repeat(v0, 3, axis=0))
            self.frag.triangles = self.o3d.utility.Vector3iVector(
                np.zeros((1, 3), dtype=np.int32))
            self.frag.vertex_normals = self.o3d.utility.Vector3dVector(
                np.array([[0.0, 0.0, 1.0]] * 3))
            self.frag.paint_uniform_color(b2m.COL_FRAGMENT)
        else:
            fvp = C[None, :] + (fv - C[None, :]) * (1.0 - self.eps)
            self.frag.vertices = self.o3d.utility.Vector3dVector(fvp)
            self.frag.triangles = self.o3d.utility.Vector3iVector(ff)
            used = res.get('frag_used')
            if used is not None and len(used) == len(fv):
                self.frag.vertex_normals = self.o3d.utility.Vector3dVector(
                    self.main_normals[used])
            self.frag.paint_uniform_color(b2m.COL_FRAGMENT)
        self.vis.update_geometry(self.frag)

        # 投到表面的 3D 轮廓(红)。
        # 这里必须抬得比碎片**更远**(3 倍),不能一样多:轮廓点是射线打在
        # 原始表面上算出来的,本来就贴在三角形上。如果和碎片同样放大,碎片
        # 就永远比轮廓离相机更近,深度测试会把这圈红线整个吃掉 —— 正好在
        # 你最想看清边界的那片区域里消失,只在射线没打中的地方留下断点。
        # (b2m.visualize 里碎片和轮廓完全共面,本来就有这个问题。)
        ol = res['outline_pts']
        seg = res['outline_seg']
        if ol is not None and len(ol) and seg is not None and len(seg):
            olp = C[None, :] + (ol - C[None, :]) * (1.0 - 3.0 * self.eps)
            self._set_lineset(self.ls_outline, olp, seg, b2m.COL_OUTLINE)
        else:
            # 同理:留 1 个点(而不是 0 个)来躲开 Open3D 的 AABB 警告。
            # 没有 lines 引用它,所以不会画出任何东西。
            self._set_lineset(self.ls_outline, _MESH['V'][:1],
                              np.zeros((0, 2), np.int32), b2m.COL_OUTLINE)

    # ---------------------------------------------------------------- 泵
    def tick(self):
        """返回 False 表示窗口被用户关掉了。"""
        if self.vis is None:
            return False
        if not self.vis.poll_events():
            try:
                self.vis.destroy_window()
            except Exception:
                pass
            self.vis = None
            self.closed = True
            return False
        self.vis.update_renderer()
        return True

    def close(self):
        if self.vis is not None:
            try:
                self.vis.destroy_window()
            except Exception:
                pass
            self.vis = None
        self.closed = True


# =============================================================================
# 四、Tk 界面
# =============================================================================
class LiveApp:
    MIN_ZOOM = 0.02
    MAX_ZOOM = 8.0
    MIN_DRAG = 6.0
    PUMP_MS = 33          # 带贴图的稠密网格一帧渲染中位数 ~15ms,16ms 会饿死 Tk
    LIVE_MS = 60

    def __init__(self, root, cfg):
        self.root = root
        self.cfg = cfg

        self.scene = {}
        self.mesh = None
        self.texture = (None, None)
        self.texture_tried = False
        self.pil_img = None
        self.img_name = None
        self.zoom = 1.0
        self.rect_bbox = None
        self.drag_start = None
        self._live = None
        self.rect_id = None
        self._photo = None
        self._proj_job = None
        self.last_result = None
        self.idx = None
        self.idx_name = None
        self.v3d = None
        self._last_exc = None
        self._exc_n = 0

        self._build_widgets()
        self._load_scene()
        self._bind_keys()

    # ------------------------------------------------------------------ 界面
    def _build_widgets(self):
        self.root.title('实时包围盒投影 —— 左画框 / 右即时 3D')

        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        tk_w = max(880, min(1120, sw // 2))
        tk_h = max(600, min(940, sh - 90))
        x0 = 20
        y0 = max(10, (sh - tk_h) // 2)
        self.root.geometry(f'{tk_w}x{tk_h}+{x0}+{y0}')
        self.tk_w, self.tk_h, self.tk_x, self.tk_y = tk_w, tk_h, x0, y0
        self.root.rowconfigure(1, weight=1)
        self.root.columnconfigure(0, weight=1)

        top = ttk.Frame(self.root, padding=(8, 6, 8, 0))
        top.grid(row=0, column=0, sticky='ew')
        btn = ttk.Frame(top)
        btn.pack(fill='x')

        ttk.Button(btn, text='清空框选 (Esc)',
                   command=self.clear_rect).pack(side='left')
        ttk.Button(btn, text='适应窗口 (F)',
                   command=self.fit).pack(side='left', padx=(6, 0))
        ttk.Button(btn, text='重算一次',
                   command=self.project).pack(side='left', padx=(6, 0))

        self.var_notexture = tk.BooleanVar(value=cfg_no_texture(self.cfg))
        ttk.Checkbutton(btn, text='纯灰(不贴图)', variable=self.var_notexture,
                        command=self._on_notexture).pack(side='left', padx=(14, 0))

        ttk.Button(btn, text='重开 3D 窗口',
                   command=self._reopen_3d).pack(side='right')

        ttk.Label(top, justify='left', foreground='#555',
                  text=f"场景: {self.cfg['scene']}\n"
                       f"图像: {self.cfg['images']}\n"
                       f"网格: {self.cfg['mesh']}").pack(fill='x', pady=(6, 0))

        mid = ttk.PanedWindow(self.root, orient='horizontal')
        mid.grid(row=1, column=0, sticky='nsew', padx=8, pady=6)

        left = ttk.Frame(mid)
        mid.add(left, weight=0)
        ttk.Label(left, text='图像').pack(anchor='w')
        self.listbox = tk.Listbox(left, width=26, exportselection=False)
        self.listbox.pack(fill='both', expand=True)
        self.listbox.bind('<<ListboxSelect>>', self._on_pick_image)

        right = ttk.Frame(mid)
        mid.add(right, weight=1)
        self.canvas = tk.Canvas(right, bg='#222', highlightthickness=0,
                                cursor='tcross')
        self.canvas.pack(fill='both', expand=True)
        self.canvas.bind('<ButtonPress-1>', self._on_press)
        self.canvas.bind('<B1-Motion>', self._on_drag)
        self.canvas.bind('<ButtonRelease-1>', self._on_release)
        self.canvas.bind('<MouseWheel>', self._on_wheel)
        self.canvas.bind('<Shift-MouseWheel>', self._on_wheel_h)
        self.canvas.bind('<Configure>', self._on_configure)
        self.canvas.bind('<Button-4>', lambda e: self._zoom_at(1.25, e.x, e.y))
        self.canvas.bind('<Button-5>', lambda e: self._zoom_at(0.8, e.x, e.y))

        logf = ttk.Frame(self.root, padding=(8, 0, 8, 0))
        logf.grid(row=2, column=0, sticky='ew')
        self.txt = tk.Text(logf, height=7, wrap='word', bg='#f6f6f8',
                           font=('Consolas', 9))
        sb = ttk.Scrollbar(logf, command=self.txt.yview)
        self.txt.configure(yscrollcommand=sb.set)
        sb.pack(side='right', fill='y')
        self.txt.pack(side='left', fill='both', expand=True)
        self.txt.configure(state='disabled')

        self.status = tk.StringVar(value='就绪')
        ttk.Label(self.root, textvariable=self.status, anchor='w',
                  relief='sunken', padding=(6, 3)).grid(
            row=3, column=0, sticky='ew', padx=8, pady=(0, 6))

    def _bind_keys(self):
        self.root.bind('<Escape>', lambda e: self.clear_rect())
        self.root.bind('<f>', lambda e: self.fit())
        self.root.bind('<F>', lambda e: self.fit())
        self.root.bind('<Control-Return>', lambda e: self.project())
        self.root.protocol('WM_DELETE_WINDOW', self.quit)

    def log(self, msg=''):
        self.txt.configure(state='normal')
        self.txt.insert('end', msg + '\n')
        self.txt.see('end')
        self.txt.configure(state='disabled')

    def log_err(self, msg):
        self.log('✗ ' + msg)

    def log_exc(self, prefix):
        """把完整 traceback 写进日志窗格。

        同一条错误只报一次:拖动的时候每秒会更新十几次,每次都刷一遍
        traceback 会把窗格淹掉,真正有用的第一条反而看不见了。错误变了
        或者中间成功过一次,就会重新报。"""
        msg = traceback.format_exc()
        if msg == self._last_exc:
            self._exc_n += 1
            self.status.set(f'{prefix} —— 同样的错误已重复 {self._exc_n} 次,'
                            f'细节见上方日志')
            return
        self._last_exc = msg
        self._exc_n = 1
        self.log_err(prefix)
        self.txt.configure(state='normal')
        self.txt.insert('end', msg + '\n')
        self.txt.see('end')
        self.txt.configure(state='disabled')

    def _clear_exc(self):
        self._last_exc = None
        self._exc_n = 0

    # -------------------------------------------------------------- 载入场景
    def _load_scene(self):
        try:
            self.scene = b2m.load_scene(self.cfg['scene'])
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror('读位姿失败', f'{self.cfg["scene"]}\n\n{e}')
            self.root.destroy()
            return

        names = []
        for nm in sorted(self.scene):
            if os.path.isfile(os.path.join(self.cfg['images'], nm)) or \
               os.path.isfile(os.path.join(self.cfg['images'], os.path.basename(nm))):
                names.append(nm)
        for nm in names:
            self.listbox.insert('end', nm)

        self.log(f'场景读入:{len(self.scene)} 张定位成功的图,可框选 {len(names)} 张')
        if not names:
            self.log_err(f'图像目录里一张都找不到:{self.cfg["images"]}')
            return
        self.log('操作:点左边选图 → 在图上拖一个框 → 松手右边即时出结果')
        self.log('      拖动过程中右侧就跟着变;框太小(<6px)会被忽略。')
        self.listbox.selection_set(0)
        self._on_pick_image()

    def _on_pick_image(self, event=None):
        sel = self.listbox.curselection()
        if not sel:
            return
        nm = self.listbox.get(sel[0])
        path = os.path.join(self.cfg['images'], nm)
        if not os.path.isfile(path):
            path = os.path.join(self.cfg['images'], os.path.basename(nm))
        if not os.path.isfile(path):
            self.log_err(f'磁盘上找不到 {nm}')
            return
        try:
            im = Image.open(path).convert('RGB')
        except Exception as e:
            self.log_err(f'打开 {path} 失败:{e}')
            return

        self.pil_img = im
        self.img_name = nm
        self.clear_rect(quiet=True)
        self.last_result = None

        W, H = im.size
        sc = self.scene[nm]
        if (W, H) != (sc['W'], sc['H']):
            self.log_err(f'⚠ 磁盘图 {W}x{H} 与位姿里的 {sc["W"]}x{sc["H"]} 不一致!'
                         f'框的坐标会错位 —— 确认这张图就是 COLMAP 用的那张。')
        self.log(f'已载入 {nm}  ({W} x {H} px)')
        self.fit()

        # 给一个居中的默认框,开局就不是空的,顺便当冒烟测试:
        # 第一次会读网格 + 开 3D 窗口(慢),之后换图只需要重建索引。
        # 用 after 推迟到主循环跑起来之后再算 —— 直接调 project 的话,
        # 里面 _ensure_mesh 的 root.update() 会把还没处理的事件(比如
        # selection_set 产生的 <<ListboxSelect>>)在当前栈里再跑一遍,递归。
        self.rect_bbox = (0.30 * W, 0.30 * H, 0.70 * W, 0.70 * H)
        self.redraw()
        self._schedule_project()

    # ------------------------------------------------------------------ 视图
    def _fit(self):
        if self.pil_img is None:
            return
        W, H = self.pil_img.size
        cw = max(self.canvas.winfo_width(), 100)
        ch = max(self.canvas.winfo_height(), 100)
        self.zoom = max(self.MIN_ZOOM,
                        min(self.MAX_ZOOM, min(cw / W, ch / H, 1.0)))
        self.redraw(reset_scroll=True)

    def fit(self):
        self._fit()

    def _zoom_at(self, factor, sx, sy):
        if self.pil_img is None:
            return
        z0 = self.zoom
        z1 = max(self.MIN_ZOOM, min(self.MAX_ZOOM, z0 * factor))
        if abs(z1 - z0) < 1e-9:
            return
        ix = self.canvas.canvasx(sx) / z0
        iy = self.canvas.canvasy(sy) / z0
        self.zoom = z1
        self.redraw()
        W, H = self.pil_img.size
        tw, th = W * z1, H * z1
        if tw > 1:
            self.canvas.xview_moveto(max(0.0, (ix * z1 - sx) / tw))
        if th > 1:
            self.canvas.yview_moveto(max(0.0, (iy * z1 - sy) / th))
        self.redraw()

    def _on_wheel(self, e):
        self._zoom_at(1.25 if e.delta > 0 else 0.8, e.x, e.y)

    def _on_wheel_h(self, e):
        self.canvas.xview_scroll(-1 if e.delta > 0 else 1, 'units')
        self.redraw()

    def _on_xview(self, *a):
        self.canvas.xview(*a)
        self.redraw()

    def _on_yview(self, *a):
        self.canvas.yview(*a)
        self.redraw()

    def _on_configure(self, e):
        if self.pil_img is not None:
            self.redraw()

    def redraw(self, reset_scroll=False):
        """只渲染视口那一块,缩放多少倍内存都不涨。"""
        if self.pil_img is None:
            return
        c = self.canvas
        W, H = self.pil_img.size
        z = self.zoom
        cw = max(c.winfo_width(), 1)
        ch = max(c.winfo_height(), 1)

        c.delete('all')
        self._photo = None
        c.config(scrollregion=(0, 0, W * z, H * z))
        if reset_scroll:
            c.xview_moveto(0.0)
            c.yview_moveto(0.0)

        ox, oy = c.canvasx(0), c.canvasy(0)
        ix0, iy0 = ox / z, oy / z
        cx0, cy0 = max(0.0, ix0), max(0.0, iy0)
        cx1, cy1 = min(float(W), ix0 + cw / z), min(float(H), iy0 + ch / z)

        if cx1 - cx0 > 0.5 and cy1 - cy0 > 0.5:
            tw = max(1, int(round((cx1 - cx0) * z)))
            th = max(1, int(round((cy1 - cy0) * z)))
            resample = Image.LANCZOS if z < 1 else Image.BICUBIC
            reg = self.pil_img.resize(
                (tw, th), resample,
                box=(int(cx0), int(cy0), int(round(cx1)), int(round(cy1))))
            self._photo = ImageTk.PhotoImage(reg)
            c.create_image(cx0 * z, cy0 * z, anchor='nw', image=self._photo)

        if self.rect_bbox:
            x0, y0, x1, y1 = self.rect_bbox
            c.create_rectangle(x0 * z, y0 * z, x1 * z, y1 * z,
                               outline='#ff2b2b', width=2)
        self.rect_id = None
        if self.drag_start is not None and self._live:
            x0, y0, x1, y1 = self._live
            self.rect_id = c.create_rectangle(x0 * z, y0 * z, x1 * z, y1 * z,
                                              outline='#ff2b2b', width=2,
                                              dash=(5, 3))

    # -------------------------------------------------------------- 鼠标画框
    def _canvas_to_image(self, sx, sy):
        return (self.canvas.canvasx(sx) / self.zoom,
                self.canvas.canvasy(sy) / self.zoom)

    def _on_press(self, e):
        if self.pil_img is None:
            return
        # 换图后延迟的那一次还没跑的话先撤掉,别让它插在拖动中间
        if self._proj_job is not None:
            self.root.after_cancel(self._proj_job)
            self._proj_job = None
        self.drag_start = self._canvas_to_image(e.x, e.y)
        self._live = None
        self.canvas.delete('live')
        self.rect_id = self.canvas.create_rectangle(
            e.x, e.y, e.x, e.y, outline='#ff2b2b', width=2, dash=(5, 3),
            tags='live')

    def _on_drag(self, e):
        if self.drag_start is None or self.rect_id is None:
            return
        x0, y0 = self.drag_start
        x1, y1 = self._canvas_to_image(e.x, e.y)
        W, H = self.pil_img.size
        x0c, x1c = max(0.0, min(x0, x1)), min(float(W), max(x0, x1))
        y0c, y1c = max(0.0, min(y0, y1)), min(float(H), max(y0, y1))
        z = self.zoom
        self.canvas.coords(self.rect_id, x0c * z, y0c * z, x1c * z, y1c * z)
        self._live = (x0c, y0c, x1c, y1c)
        self.status.set(f'框 ({x0c:.0f}, {y0c:.0f}) - ({x1c:.0f}, {y1c:.0f})   '
                        f'{x1c-x0c:.0f} x {y1c-y0c:.0f} px')
        # 拖动过程中持续重算,但不要每次鼠标移动都算 —— 攒到下一次空闲
        if self._proj_job is None:
            self._proj_job = self.root.after(self.LIVE_MS, self._live_project)

    def _live_project(self):
        """拖动过程中的定时重算。"""
        self._proj_job = None
        box = getattr(self, '_live', None)
        if box is None:
            return
        self.rect_bbox = box
        self.project(quiet=True)

    def _schedule_project(self):
        """换图之后延迟算一次(要读网格 / 建索引,别卡在事件处理里)。"""
        if self._proj_job is not None:
            self.root.after_cancel(self._proj_job)
        self._proj_job = self.root.after(60, self._auto_project)

    def _auto_project(self):
        self._proj_job = None
        self.project()

    def _on_release(self, e):
        if self.drag_start is None:
            return
        box = getattr(self, '_live', None)
        self.drag_start = None
        if box is None:
            return
        if self._proj_job is not None:
            self.root.after_cancel(self._proj_job)
            self._proj_job = None
        x0, y0, x1, y1 = box
        if (x1 - x0) < self.MIN_DRAG or (y1 - y0) < self.MIN_DRAG:
            self.log('框太小了(<6px),忽略。')
            self.clear_rect(quiet=True)
            return
        self.rect_bbox = box
        self._live = None
        self.rect_id = None
        self.canvas.delete('live')
        self.redraw()
        self.project()

    def clear_rect(self, quiet=False):
        if self._proj_job is not None:
            self.root.after_cancel(self._proj_job)
            self._proj_job = None
        self.rect_bbox = None
        self._live = None
        self.drag_start = None
        if self.rect_id is not None:
            self.canvas.delete(self.rect_id)
            self.rect_id = None
        self.canvas.delete('rect')
        self.canvas.delete('live')
        self.redraw()
        self._clear_3d()
        if not quiet:
            self.status.set('已清空框选')

    def _clear_3d(self):
        if self.v3d is not None and not self.v3d.closed:
            self.v3d.update({'corners_2d': np.zeros((4, 2)),
                             'hits': [None] * 4,
                             'frag_v': None, 'frag_f': None, 'frag_used': None,
                             'outline_pts': np.zeros((0, 3)),
                             'outline_seg': np.zeros((0, 2), np.int64)})

    # -------------------------------------------------------------- 载入资源
    def _ensure_mesh(self):
        if self.mesh is not None:
            return True
        self.status.set('正在读网格…')
        self.root.update()
        t0 = time.time()
        try:
            V, F, vcols = b2m.load_mesh(self.cfg['mesh'])
        except Exception as e:
            traceback.print_exc()
            self.log_err(f'读网格失败:{e}')
            messagebox.showerror('读网格失败', str(e))
            return False
        self.mesh = (V, F, vcols)
        set_mesh(V, F)
        self.log(f'网格读入:{len(V)} 顶点 / {len(F)} 三角形 / '
                 f'顶点色 {"有" if vcols is not None else "无"}   '
                 f'({time.time()-t0:.1f}s)')
        if not self.var_notexture.get():
            self._ensure_texture()
        self.status.set('就绪')
        return True

    def _ensure_texture(self):
        if self.texture[0] is not None or self.texture_tried:
            return
        self.texture_tried = True
        t0 = time.time()
        try:
            uv, tex, src = b2m.load_texture(self.cfg['mesh'],
                                            self.mesh[0], self.mesh[1])
        except Exception as e:
            uv, tex, src = None, None, None
            self.log_err(f'找贴图出错(不影响用):{e}')
        if uv is not None:
            self.texture = (uv, tex)
            self.log(f'贴图绑定好:{os.path.basename(src)}  ({time.time()-t0:.1f}s)')
        else:
            self.log('没找到同源的带贴图 ply —— 3D 里用顶点色/灰。')

    def _on_notexture(self):
        self.log(f'贴图:{"关(纯灰)" if self.var_notexture.get() else "开"}'
                 f' —— 重开 3D 窗口才生效')

    def _ensure_index(self):
        sc = self.scene[self.img_name]
        if self.idx is not None and self.idx_name == self.img_name:
            return True
        self.status.set('第一次用这张图,正在建索引…')
        self.root.update()
        try:
            self.idx = build_index(self.mesh[1], sc, cell=self.cfg['cell'],
                                   verbose=False)
        except Exception as e:
            traceback.print_exc()
            self.log_err(f'建索引失败:{e}')
            return False
        self.idx_name = self.img_name
        self.log(f'索引建好 [{self.img_name}]  '
                 f'{self.idx["W"]}x{self.idx["H"]}  格 {self.cfg["cell"]}px  '
                 f'散布 {self.idx["n_scatter"]} 项 / 溢出 {self.idx["n_overflow"]} 面  '
                 f'({self.idx["t_all"]*1000:.0f} ms,换图才重算)')
        return True

    # ---------------------------------------------------------------- 投射
    def project(self, quiet=False):
        if self.pil_img is None or self.rect_bbox is None:
            return
        if not self._ensure_mesh():
            return
        if not self._ensure_index():
            return

        sc = self.scene[self.img_name]
        t0 = time.time()
        try:
            res = fast_project(self.idx, self.mesh[0], self.mesh[1], sc,
                               self.rect_bbox)
        except Exception:
            self.log_exc('投射失败')
            self.status.set('投射失败')
            return
        dt = time.time() - t0

        self.last_result = res
        self._refresh_3d(res, sc)
        self.status.set(f'投影 {dt*1000:.0f} ms · 选中 {res["n_tri"]} 面 · '
                        f'角光线 {res["n_hit"]}/4 · '
                        f'轮廓 {res["n_outline_hit"]}/{res["n_outline"]}')
        if not quiet:
            x0, y0, x1, y1 = res['bbox']
            self.log('─' * 56)
            self.log(f'参考图  : {self.img_name}  {sc["W"]}x{sc["H"]}  {sc["model"]}')
            self.log(f'2D 框   : ({x0:.0f}, {y0:.0f}) - ({x1:.0f}, {y1:.0f})  '
                     f'{x1-x0:.0f} x {y1-y0:.0f} px')
            self.log(f'四角光线: 命中 {res["n_hit"]}/4,'
                     f'轮廓 {res["n_outline_hit"]}/{res["n_outline"]} 条射线打到网格')
            self.log(f'视锥裁剪: {res["n_tri"]}/{res["n_faces_total"]} 个三角形 '
                     f'({100.0*res["n_tri"]/max(1,res["n_faces_total"]):.1f}%)')
            if res['n_tri'] == 0:
                self.log('    ⚠ 这个框里一个三角形都没有 —— 框错地方了?')
            self.log(f'耗时    : {dt*1000:.1f} ms  (原版同样这个框要 ~20000 ms)')

    # ---------------------------------------------------------------- 3D
    def _open_3d(self, res, sc):
        if self.v3d is not None and not self.v3d.closed:
            return
        uv, tex = None, None
        if not self.var_notexture.get():
            self._ensure_texture()
            uv, tex = self.texture
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        w3 = max(560, min(sw - self.tk_w - self.tk_x - 24, 1100))
        h3 = self.tk_h
        left = min(self.tk_x + self.tk_w + 8, max(0, sw - w3))
        top = self.tk_y
        self.status.set('正在建 3D 场景(读贴图/算法线)…')
        self.root.update()
        t0 = time.time()
        self.v3d = Live3D(self.mesh[0], self.mesh[1], self.mesh[2], uv, tex,
                          sc, res, self.cfg, left=left, top=top,
                          width=w3, height=h3)
        self.log(f'3D 窗口已开在右边 ({w3}x{h3},贴图 '
                 f'{"有" if uv is not None else "无"})   {time.time()-t0:.1f}s')
        self._pump()

    def _refresh_3d(self, res, sc):
        if self.v3d is None or self.v3d.closed:
            try:
                self._open_3d(res, sc)
            except Exception:
                self.log_exc('3D 窗口打不开(左边还能继续画框,不影响)')
                self.v3d = None
            return
        try:
            self.v3d.update(res, sc)
            self._clear_exc()
        except Exception:
            self.log_exc('3D 刷新出错')

    def _reopen_3d(self):
        if self.v3d is not None and not self.v3d.closed:
            self.log('3D 窗口本来就开着。')
            return
        self.v3d = None
        if self.last_result is not None:
            self._refresh_3d(self.last_result, self.scene[self.img_name])
        else:
            self.log('先框一个区域,再开 3D。')

    def _pump(self):
        if self.v3d is None or self.v3d.closed:
            return
        if not self.v3d.tick():
            self.v3d = None
            self.status.set('3D 窗口已关闭 —— 点「重开 3D 窗口」可以再开')
            self.log('3D 窗口已关闭(不影响这边继续画框)。')
            return
        self.root.after(self.PUMP_MS, self._pump)

    # ---------------------------------------------------------------- 退出
    def quit(self):
        for attr in ('_proj_job',):
            j = getattr(self, attr, None)
            if j is not None:
                try:
                    self.root.after_cancel(j)
                except Exception:
                    pass
        if self.v3d is not None:
            self.v3d.close()
            self.v3d = None
        self.root.destroy()


def cfg_no_texture(cfg):
    return bool(cfg.get('no_texture'))


# =============================================================================
# 五、CLI
# =============================================================================
def main():
    ap = argparse.ArgumentParser(
        description='实时包围盒投影:左边画框,右边常驻 3D 窗口即时显示',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scene', required=True,
                    help='COLMAP sparse 目录,如 work\\food\\dense\\sparse')
    ap.add_argument('--images', default=None,
                    help='无畸变图像目录(默认从 --scene 推)')
    ap.add_argument('--mesh', default=None,
                    help='网格 .ply(默认从 --scene 推)')
    ap.add_argument('--no-texture', action='store_true',
                    help='3D 里不加载贴图,纯灰(快很多)')
    ap.add_argument('--cell', type=int, default=32,
                    help='网格索引的格边长(像素),默认 32')
    ap.add_argument('--eps', type=float, default=None,
                    help='碎片朝相机抬起的**相对**量(无量纲),默认 1e-4。'
                         '3D 里碎片和模型互相闪烁(深度冲突)就调大,'
                         '调到 3e-3 还闪就说明该开 --no-texture 看看了')
    ap.add_argument('--verify', action='store_true',
                    help='只跑正确性自检(逐位对比 old vs new)并退出')
    args = ap.parse_args()

    if not os.path.isdir(args.scene):
        raise SystemExit(f'--scene 目录不存在:{args.scene}')

    scene_dir = os.path.abspath(args.scene)
    images = args.images or bg.default_images_dir(scene_dir)
    if not os.path.isdir(images):
        raise SystemExit(f'图像目录不存在:{images}\n用 --images 指定。')
    mesh = args.mesh or bg.default_mesh(scene_dir)
    if mesh is None:
        raise SystemExit('推不出网格路径,用 --mesh 指定 .ply。')
    if not os.path.isfile(mesh):
        raise SystemExit(f'网格文件不存在:{mesh}')

    cfg = {'scene': scene_dir, 'images': os.path.abspath(images),
           'mesh': os.path.abspath(mesh), 'no_texture': args.no_texture,
           'cell': args.cell, 'eps': args.eps}

    if args.verify:
        ok = verify(cfg, cell=args.cell)
        raise SystemExit(0 if ok else 1)

    root = tk.Tk()
    app = LiveApp(root, cfg)                                  # noqa: F841
    root.mainloop()


if __name__ == '__main__':
    main()
