#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
把参考图上的 2D 检测框投影到 OpenMVS 重建的 3D 网格上,并做 3D 可视化。

原理:2D 框的四个角 + 相机光心 = 一个视锥。求视锥与网格的交集。

坐标系:实测确认 OpenMVS 场景的内部变换是 identity,
        因此 COLMAP 的世界坐标 == OpenMVS 网格的世界坐标,
        位姿可以直接从 COLMAP 的 sparse 模型读,不需要任何变换。

依赖: numpy trimesh open3d matplotlib Pillow

用法:
  # 交互式框选(弹窗里拖一个矩形,关掉窗口确认)
  python bbox_to_mesh.py --image IMG_2435.JPG

  # 直接给坐标(避开交互)
  python bbox_to_mesh.py --image IMG_2435.JPG --bbox 1200 900 2400 1900

  # 常用选项
  --scene   COLMAP sparse 目录(默认 work/<场景>/dense/sparse)
  --mesh    网格文件(默认 work/<场景>/mvs/scene_dense_mesh_refine.ply)
  --images  无畸变图像目录(默认 work/<场景>/dense/images)
  --scale   虚拟像平面到相机的距离(默认自动)
  --crop    把框内的网格碎片另存为 ply
  --save-json  把 3D 结果存成 json
"""

import argparse
import json
import os
import re
import struct
import sys

import numpy as np

# Windows 控制台中文输出
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass


# =============================================================================
# COLMAP 二进制模型解析
# =============================================================================
# model_id -> (名字, 参数个数)
COLMAP_MODELS = {
    0: ('SIMPLE_PINHOLE', 3),   1: ('PINHOLE', 4),
    2: ('SIMPLE_RADIAL', 4),    3: ('RADIAL', 5),
    4: ('OPENCV', 8),           5: ('OPENCV_FISHEYE', 8),
    6: ('FULL_OPENCV', 12),     7: ('FOV', 5),
    8: ('SIMPLE_RADIAL_FISHEYE', 4), 9: ('RADIAL_FISHEYE', 5),
    10: ('THIN_PRISM_FISHEYE', 12),
}


def read_cameras_bin(path):
    """返回 {camera_id: (model_name, width, height, params_tuple)}"""
    out = {}
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        for _ in range(n):
            cid, model_id, w, h = struct.unpack('<iiQQ', f.read(24))
            name, k = COLMAP_MODELS[model_id]
            params = struct.unpack('<%dd' % k, f.read(8 * k))
            out[cid] = (name, w, h, params)
    return out


def read_images_bin(path):
    """返回 [(image_id, qvec(4), tvec(3), camera_id, name)]"""
    out = []
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        for _ in range(n):
            iid = struct.unpack('<i', f.read(4))[0]
            q = np.array(struct.unpack('<4d', f.read(32)))
            t = np.array(struct.unpack('<3d', f.read(24)))
            cid = struct.unpack('<i', f.read(4))[0]
            buf = bytearray()
            while True:
                c = f.read(1)
                if c == b'\0':
                    break
                buf += c
            npts = struct.unpack('<Q', f.read(8))[0]
            f.read(24 * npts)                      # 跳过 2D 观测
            out.append((iid, q, t, cid, buf.decode('utf-8', 'replace')))
    return out


def qvec_to_rotmat(q):
    """COLMAP 四元数 (w,x,y,z) -> 3x3 旋转矩阵"""
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y],
    ])


def intrinsics_from_params(model_name, params, w, h):
    """取 fx, fy, cx, cy。去畸变后的模型应该是 SIMPLE_PINHOLE 或 PINHOLE。"""
    if model_name == 'SIMPLE_PINHOLE':
        f, cx, cy = params
        return f, f, cx, cy
    if model_name == 'PINHOLE':
        fx, fy, cx, cy = params[:4]
        return fx, fy, cx, cy
    raise SystemExit(
        f'相机模型是 {model_name},带畸变参数。\n'
        f'  请确认 --images 指向的是 image_undistorter 输出的无畸变图像目录,\n'
        f'  对应的 sparse 模型里相机应该已经变成 PINHOLE。'
    )


# =============================================================================
# 几何工具
# =============================================================================
def make_K(fx, fy, cx, cy):
    return np.array([[fx, 0.0, cx],
                     [0.0, fy, cy],
                     [0.0, 0.0, 1.0]])


def ray_from_pixel(u, v, K, R, C):
    """像素 (u,v) -> 世界坐标射线 (原点, 单位方向)"""
    d_cam = np.linalg.inv(K) @ np.array([u, v, 1.0])
    d_world = R.T @ d_cam                 # P_c = R(P_w - C) => d_w = R^T d_c
    return C.copy(), d_world / np.linalg.norm(d_world)


def project_points(Pw, K, R, t):
    """世界点 Nx3 -> (u, v, depth)。depth<=0 表示在相机背后。"""
    Pw = np.atleast_2d(Pw)
    Pc = (R @ Pw.T).T + t
    depth = Pc[:, 2]
    safe = np.where(np.abs(depth) < 1e-9, 1.0, depth)
    u = K[0, 0] * Pc[:, 0] / safe + K[0, 2]
    v = K[1, 1] * Pc[:, 1] / safe + K[1, 2]
    return u, v, depth


def point_at_depth(u, v, K, R, C, depth):
    """像素 (u,v) 在指定深度处的世界点(用来画虚拟像平面)"""
    _, d = ray_from_pixel(u, v, K, R, C)
    return C + d * depth


def ray_triangle_hits(origins, dirs, tris):
    """
    Möller–Trumbore 光线-三角形求交,纯 numpy 实现。
    比 trimesh 的 ray 模块少一个 rtree 依赖,对这种几千面的网格也更快。

    origins (RayN,3)  射线起点
    dirs    (RayN,3)  射线方向(单位向量)
    tris    (TriN,3,3) 三角形三个顶点
    返回: pts (RayN,3) 最近交点(未命中为 nan), hit (RayN,) 命中掩码
    """
    v0 = tris[:, 0]
    e1 = tris[:, 1] - v0
    e2 = tris[:, 2] - v0

    n_ray = len(origins)
    pts = np.full((n_ray, 3), np.nan)
    hit = np.zeros(n_ray, dtype=bool)

    for i in range(n_ray):
        o, d = origins[i], dirs[i]
        p = np.cross(d, e2)
        det = np.einsum('ij,ij->i', e1, p)
        ok = np.abs(det) > 1e-12                 # 剔除平行 / 退化三角形
        inv = 1.0 / np.where(ok, det, 1.0)       # 不满足 ok 的位置后面会被掩掉

        tvec = o - v0
        u = np.einsum('ij,ij->i', tvec, p) * inv
        q = np.cross(tvec, e1)                   # (TriN,3)
        v = (q @ d) * inv                        # d 是 (3,),广播成 (TriN,)
        t = np.einsum('ij,ij->i', e2, q) * inv

        ok &= (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1.0 + 1e-9) & (t > 1e-9)
        if ok.any():
            ti = int(np.argmin(np.where(ok, t, np.inf)))
            pts[i] = o + d * t[ti]
            hit[i] = True
    return pts, hit


# =============================================================================
# 可复用的核心接口(CLI 和 GUI 都用这几个)
# =============================================================================
def load_scene(scene_dir):
    """
    读 COLMAP 的 sparse 目录。
    返回 {图像名: {'K','R','t','C','W','H','fx','fy','cx','cy','model'}}
    """
    cams = read_cameras_bin(os.path.join(scene_dir, 'cameras.bin'))
    imgs = read_images_bin(os.path.join(scene_dir, 'images.bin'))
    out = {}
    for iid, qvec, tvec, cid, nm in imgs:
        model_name, W, H, params = cams[cid]
        fx, fy, cx, cy = intrinsics_from_params(model_name, params, W, H)
        R = qvec_to_rotmat(qvec)
        out[nm] = {
            'K': make_K(fx, fy, cx, cy), 'R': R, 't': tvec, 'C': -R.T @ tvec,
            'W': W, 'H': H, 'model': model_name,
            'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy,
        }
    return out


def load_mesh(path):
    """读网格文件,返回 (V, F, vcols)。vcols 为 None 表示没有顶点色。"""
    import trimesh
    m = trimesh.load(path, process=False)
    if isinstance(m, trimesh.Scene):
        try:
            m = m.dump(concatenate=True)                 # trimesh >= 4.0
        except TypeError:
            m = trimesh.util.concatenate(tuple(m.geometry.values()))
    V = np.asarray(m.vertices, dtype=np.float64)
    F = np.asarray(m.faces, dtype=np.int64)
    if len(F) == 0:
        raise ValueError(f'{path} 里没有三角形 —— 这是个点云,不是网格。\n'
                         f'光线投射需要网格,请用 scene_dense_mesh.ply 或 '
                         f'scene_dense_mesh_refine.ply。')
    # 只有网格真的带顶点色时才用它。trimesh 对没有颜色的 PLY 有时会填一个
    # 默认色(全白/全灰),那种情况画成灰色更好看。
    vcols = None
    try:
        raw = getattr(m.visual, 'vertex_colors', None)
        if raw is not None and len(raw) == len(V):
            arr = np.asarray(raw)[:, :3]
            if arr.max() > arr.min():
                vcols = arr / 255.0
    except Exception:
        vcols = None
    return V, F, vcols


# -----------------------------------------------------------------------------
# 贴图
# -----------------------------------------------------------------------------
# OpenMVS TextureMesh 输出的 ply 长这样(顶点 XYZ + 每面的 texcoord):
#     comment TextureFile scene_textured0.png
#     element vertex N / property float32 x,y,z
#     element face M / property list uint8 uint32 vertex_indices
#                    / property list uint8 float32 texcoord
# open3d 的 ply 读取器只认 texture_u / texture_v 这种属性名,读不了 OpenMVS
# 写的 texcoord,所以得自己解析 —— 实测 open3d.io.read_triangle_mesh 读这个
# 文件,triangle_uvs 是空的、textures 是 0 个。
_PLY_TYPES = {
    'char': 'i1', 'int8': 'i1', 'uchar': 'u1', 'uint8': 'u1',
    'short': '<i2', 'int16': '<i2', 'ushort': '<u2', 'uint16': '<u2',
    'int': '<i4', 'int32': '<i4', 'uint': '<u4', 'uint32': '<u4',
    'float': '<f4', 'float32': '<f4', 'double': '<f8', 'float64': '<f8',
}


def read_ply_uv(path):
    """读带贴图坐标的 ply。返回 (V, F, UV, texfile)。

    UV 形状是 (n_faces, 3, 2) —— 每个三角形的三个角各一组 (u, v)。
    texfile 是头里 TextureFile 那行写的文件名(相对路径)。
    """
    with open(path, 'rb') as f:
        hdr = b''
        while b'end_header' not in hdr:
            chunk = f.readline()
            if not chunk:
                raise ValueError(f'{path} 不是合法的 ply(找不到 end_header)')
            hdr += chunk
        text = hdr.decode('ascii', 'replace')

        cur, nv, nf, texfile = None, 0, 0, None
        vprops = []
        for ln in text.splitlines():
            w = ln.split()
            if not w:
                continue
            if w[0] == 'comment' and len(w) >= 3 and w[1] == 'TextureFile':
                texfile = w[2]
            elif w[0] == 'element':
                cur = w[1]
                if cur == 'vertex':
                    nv = int(w[2])
                elif cur == 'face':
                    nf = int(w[2])
            elif w[0] == 'property' and cur == 'vertex' and w[1] != 'list':
                vprops.append((w[-1], _PLY_TYPES[w[1]]))
        if texfile is None:
            raise ValueError(f'{path} 头里没有 TextureFile,不是带贴图的网格')
        if not vprops:
            raise ValueError(f'{path} 头里读不到 vertex 属性')

        vdt = np.dtype(vprops)
        raw = f.read()

    V = np.frombuffer(raw, dtype=vdt, count=nv)
    V = np.stack([V['x'], V['y'], V['z']], 1).astype(np.float64)

    off = nv * vdt.itemsize
    F = np.empty((nf, 3), dtype=np.int64)
    UV = np.empty((nf, 3, 2), dtype=np.float64)
    for i in range(nf):
        n = raw[off]
        off += 1
        if n != 3:
            raise ValueError(f'{path} 第 {i} 个面有 {n} 个顶点,只支持三角形网格')
        F[i] = np.frombuffer(raw[off:off + 4 * n], dtype='<u4', count=n).astype(np.int64)
        off += 4 * n
        m = raw[off]
        off += 1
        if m != 6:
            raise ValueError(f'{path} 第 {i} 个面的 texcoord 有 {m} 个数,期望 6')
        UV[i] = np.frombuffer(raw[off:off + 4 * m], dtype='<f4',
                              count=m).astype(np.float64).reshape(3, 2)
        off += 4 * m
    return V, F, UV, texfile


def _read_ply_header(path):
    """只读到 end_header 为止,返回头部字节;读不出来返回 None。"""
    try:
        with open(path, 'rb') as f:
            hdr = b''
            while b'end_header' not in hdr:
                chunk = f.readline()
                if not chunk:
                    return None
                hdr += chunk
    except OSError:
        return None
    return hdr


def ply_header_info(path):
    """只读 ply 头,返回 (顶点数, 面数, 贴图文件名或 None)。

    比 read_ply_uv 便宜得多(完全不碰二进制体),用来判断「哪份几何配得上
    那份带贴图的 ply」—— GUI 挑网格时需要的正是这个对应关系。
    """
    hdr = _read_ply_header(path)
    if hdr is None:
        raise ValueError('读不到 ply 头')
    nv = nf = 0
    tex = None
    for line in hdr.decode('ascii', 'replace').splitlines():
        f = line.split()
        if len(f) < 3:
            continue
        if f[0] == 'element' and f[1] == 'vertex':
            nv = int(f[2])
        elif f[0] == 'element' and f[1] == 'face':
            nf = int(f[2])
        elif f[0] == 'comment' and f[1] == 'TextureFile':
            tex = f[2]
    return nv, nf, tex


def find_textured_ply(mesh_path):
    """在网格同目录里找带贴图的 ply,返回路径列表(可能不止一个)。"""
    d = os.path.dirname(os.path.abspath(mesh_path))
    out = []
    for fn in sorted(os.listdir(d)):
        if not fn.lower().endswith('.ply'):
            continue
        p = os.path.join(d, fn)
        hdr = _read_ply_header(p)
        if hdr and b'TextureFile' in hdr:
            out.append(p)
    return out


def load_texture(mesh_path, V, F):
    """给网格配上贴图。

    返回 (UV, tex_image, 来源文件) —— 找不到就返回 (None, None, None)。
    UV 是按顶点数/面数/顶点坐标三重校验过的:只有几何完全一致,
    每面的 texcoord 才和我们的 F 对得上,不然贴图会整个错位。
    """
    import open3d as o3d

    for p in find_textured_ply(mesh_path):
        try:
            V2, F2, UV, texfile = read_ply_uv(p)
        except Exception as e:
            print(f'  [贴图] 跳过 {os.path.basename(p)}:{e}')
            continue
        if len(V2) != len(V) or len(F2) != len(F):
            print(f'  [贴图] 跳过 {os.path.basename(p)}:'
                  f'几何不同({len(V2)}顶点/{len(F2)}面 vs {len(V)}/{len(F)})')
            continue
        if not np.allclose(V2, V, atol=1e-6):
            print(f'  [贴图] 跳过 {os.path.basename(p)}:顶点坐标对不上')
            continue
        tex = os.path.join(os.path.dirname(p), texfile)
        if not os.path.isfile(tex):
            print(f'  [贴图] 跳过 {os.path.basename(p)}:缺贴图文件 {texfile}')
            continue
        img = o3d.io.read_image(tex)
        if not np.asarray(img).size:
            print(f'  [贴图] 跳过 {os.path.basename(p)}:贴图读不出来 {tex}')
            continue
        ar = np.asarray(img)
        # OpenMVS 的 texcoord 用的是 OpenGL 约定:v=0 在贴图「底边」。
        # 而 Open3D 采样时按 v*H 当行号(从顶边数)。不翻的话整个模型会
        # 采样到图集右下角那片空白 —— OpenMVS 给没用到的地方填的是
        # 橙色 (255,127,39),占这张 8192x8192 图集的 55%,模型就会整个
        # 变橙。验证方式:把 UV 三角形栅格化到图集上,和「网格实际占的
        # 区域」求交并比 —— (u, 1-v) 是 0.942,不翻只有 0.017。
        UV = np.stack([UV[..., 0], 1.0 - UV[..., 1]], axis=-1)
        print(f'  [贴图] {os.path.basename(p)} + {texfile}  '
              f'{ar.shape[1]}x{ar.shape[0]}')
        return UV, img, p

    return None, None, None


def project_bbox(V, F, K, R, t, C, bbox):
    """
    核心计算:把 2D 框投影到网格上。

    V,F    网格顶点 (N,3) 和三角面 (M,3)
    K,R,t  相机内参和外参(世界->相机)
    C      相机光心(世界坐标)
    bbox   (xmin, ymin, xmax, ymax) 像素坐标

    返回 dict,键见下方 return。
    """
    xmin, ymin, xmax, ymax = bbox
    if xmin > xmax:
        xmin, xmax = xmax, xmin
    if ymin > ymax:
        ymin, ymax = ymax, ymin

    corners_2d = np.array([[xmin, ymin], [xmax, ymin],
                           [xmax, ymax], [xmin, ymax]], dtype=np.float64)

    # --- 4 角光线投射 ---
    ray_o = np.repeat(C[None, :], 4, axis=0)
    ray_d = np.stack([ray_from_pixel(u, v, K, R, C)[1] for u, v in corners_2d])
    hit_pts, hit_ok = ray_triangle_hits(ray_o, ray_d, V[F])
    hits = [hit_pts[i] if hit_ok[i] else None for i in range(4)]

    # --- 视锥裁剪 ---
    u_all, v_all, dep_all = project_points(V, K, R, t)
    inside = ((u_all >= xmin) & (u_all <= xmax) &
              (v_all >= ymin) & (v_all <= ymax) & (dep_all > 0))
    face_inside = inside[F].any(axis=1)

    frag_v, frag_f = None, None
    if face_inside.any():
        used = np.unique(F[face_inside])
        remap = -np.ones(len(V), dtype=np.int64)
        remap[used] = np.arange(len(used))
        frag_v = V[used]
        frag_f = remap[F[face_inside]]

    # --- 沿框的一整圈打密集射线,得到贴着网格表面的完整轮廓 ---
    # 只打 4 个角是不够的:角一落到背景/空洞上,轮廓就断成几截。
    n_edge = 40
    per = []
    for k in range(4):
        a, b = corners_2d[k], corners_2d[(k + 1) % 4]
        for i in range(n_edge):
            s = i / n_edge
            per.append(a * (1.0 - s) + b * s)
    per = np.asarray(per, dtype=np.float64)

    ol_pts = np.zeros((len(per), 3), dtype=np.float64)
    ol_ok = np.zeros(len(per), dtype=bool)
    ol_d = np.stack([ray_from_pixel(u, v, K, R, C)[1] for u, v in per])
    op, ook = ray_triangle_hits(np.repeat(C[None, :], len(per), axis=0), ol_d, V[F])
    ol_pts[ook] = op[ook]
    ol_ok = ook

    n = len(per)
    seg = [(i, (i + 1) % n) for i in range(n) if ol_ok[i] and ol_ok[(i + 1) % n]]

    return {
        'bbox': (xmin, ymin, xmax, ymax),
        'corners_2d': corners_2d,
        'hits': hits,
        'n_hit': int(hit_ok.sum()),
        'frag_v': frag_v,
        'frag_f': frag_f,
        'face_mask': face_inside,
        'n_tri': int(face_inside.sum()),
        'n_faces_total': len(F),
        'outline_pts': ol_pts,
        'outline_seg': np.asarray(seg, dtype=np.int64).reshape(-1, 2),
        'n_outline_hit': int(ol_ok.sum()),
        'n_outline': n,
    }


# =============================================================================
# 主流程
# =============================================================================
def main():
    ap = argparse.ArgumentParser(
        description='把 2D 框投影到 OpenMVS 3D 网格上并可视化',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scene', required=True,
                    help='COLMAP sparse 目录(含 cameras.bin / images.bin)')
    ap.add_argument('--mesh', required=True, help='网格文件 .ply')
    ap.add_argument('--images', required=True, help='无畸变图像目录')
    ap.add_argument('--image', required=True, help='参考图文件名,如 IMG_2435.JPG')
    ap.add_argument('--bbox', type=float, nargs=4, default=None,
                    metavar=('XMIN', 'YMIN', 'XMAX', 'YMAX'),
                    help='2D 框;不给则弹窗交互框选')
    ap.add_argument('--scale', type=float, default=None,
                    help='虚拟像平面到相机的距离(默认自动)')
    ap.add_argument('--crop', default=None,
                    help='把框内的网格碎片另存为 ply 的路径')
    ap.add_argument('--save-json', dest='save_json', default=None,
                    help='把结果存成 json 的路径')
    ap.add_argument('--no-texture', action='store_true',
                    help='可视化时不加载贴图/顶点色,只用灰色网格')
    ap.add_argument('--texture', default=None,
                    help='手工指定带贴图的 ply(默认在 --mesh 同目录里自动找)')
    args = ap.parse_args()

    import trimesh

    # ---------- 1. 读相机位姿 ----------
    scene = load_scene(args.scene)
    name = None
    for nm in scene:
        if nm == args.image or os.path.basename(nm) == args.image:
            name = nm
            break
    if name is None:
        names = '\n    '.join(scene)
        raise SystemExit(f'在模型里找不到图像 "{args.image}"。模型里有:\n    {names}')

    cam = scene[name]
    K, R, t, C = cam['K'], cam['R'], cam['t'], cam['C']
    W, H = cam['W'], cam['H']
    model_name = cam['model']
    fx, fy, cx, cy = cam['fx'], cam['fy'], cam['cx'], cam['cy']

    print('=' * 66)
    print(f'  参考图    : {name}')
    print(f'  图像尺寸  : {W} x {H}')
    print(f'  相机模型  : {model_name}  fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}')
    print(f'  光心(世界): [{C[0]:.4f}, {C[1]:.4f}, {C[2]:.4f}]')
    print('=' * 66)

    # ---------- 2. 读网格 ----------
    try:
        V, F, vcols = load_mesh(args.mesh)
    except ValueError as e:
        raise SystemExit(str(e))
    print(f'\n网格: {len(V)} 顶点 / {len(F)} 三角形')
    print(f'  包围盒 min {np.round(V.min(0), 3)}  max {np.round(V.max(0), 3)}')
    print(f'  顶点色: {"有" if vcols is not None else "无"}')

    # 贴图:默认自动找同目录里带 TextureFile 的 ply(TextureMesh 的产物)
    uv = tex_img = None
    if args.no_texture:
        print('  贴图: 已用 --no-texture 关掉')
    elif args.texture:
        import open3d as o3d
        V2, F2, uv2, texfile = read_ply_uv(args.texture)
        tex_path = os.path.join(os.path.dirname(args.texture), texfile)
        uv, tex_img = uv2, o3d.io.read_image(tex_path)
        ar = np.asarray(tex_img)
        print(f'  贴图: {os.path.basename(args.texture)} + {texfile}  '
              f'{ar.shape[1]}x{ar.shape[0]}')
    else:
        uv, tex_img, src = load_texture(args.mesh, V, F)
        if uv is None:
            print('  贴图: 无(同目录里没有几何一致的带贴图 ply,'
                  '将退回顶点色/灰色)')

    # 网格是否都在相机前方(坐标系对齐的健全性检查)
    _, _, depth_all = project_points(V, K, R, t)
    front = (depth_all > 0)
    print(f'  在相机前方的顶点: {front.sum()}/{len(V)} ({100*front.mean():.1f}%)')
    if front.mean() < 0.3:
        print('  ⚠️  大部分顶点在相机背后 —— 坐标系可能不匹配,结果会不对!')
        print('     请确认 --scene / --mesh 来自同一次重建,且 mesh 没有被人为变换过。')

    # ---------- 3. 拿 2D 框 ----------
    if args.bbox is None:
        bbox = pick_bbox_interactive(os.path.join(args.images, name))
    else:
        bbox = tuple(args.bbox)

    # ---------- 4/5. 投射:4 角光线 + 视锥裁剪 ----------
    res = project_bbox(V, F, K, R, t, C, bbox)
    xmin, ymin, xmax, ymax = res['bbox']
    corners_2d = res['corners_2d']
    hits = res['hits']
    frag_v, frag_f = res['frag_v'], res['frag_f']

    print(f'\n2D 框: ({xmin:.0f}, {ymin:.0f}) - ({xmax:.0f}, {ymax:.0f})'
          f'  = {xmax-xmin:.0f} x {ymax-ymin:.0f} px')
    print(f'\n4 条角光线命中网格: {res["n_hit"]}/4')
    for i, h in enumerate(hits):
        if h is None:
            print(f'  角 {i} ({corners_2d[i][0]:.0f},{corners_2d[i][1]:.0f}) 未命中'
                  f'  —— 可能落在背景/空洞上')
        else:
            print(f'  角 {i} ({corners_2d[i][0]:.0f},{corners_2d[i][1]:.0f}) -> '
                  f'[{h[0]:.4f}, {h[1]:.4f}, {h[2]:.4f}]')

    print(f'\n投影轮廓: 框一圈 {res["n_outline"]} 条射线,'
          f'命中网格 {res["n_outline_hit"]} 条,'
          f'连成 {len(res["outline_seg"])} 段线')
    if res['n_outline_hit'] == 0:
        print('  ⚠️  一条都没打中 —— 框整个落在背景上?')

    n_sel = res['n_tri']
    print(f'\n视锥裁剪: {n_sel}/{len(F)} 个三角形至少一个顶点落在框内 '
          f'({100*n_sel/max(len(F),1):.1f}%)')
    if n_sel == 0:
        print('  ⚠️  框内没有选中任何三角形 —— 框是不是画到空白处了?')
    if frag_v is not None:
        print(f'  碎片包围盒 min {np.round(frag_v.min(0), 3)}  '
              f'max {np.round(frag_v.max(0), 3)}')

    # ---------- 6. 存文件 ----------
    if args.crop and n_sel > 0:
        trimesh.Trimesh(vertices=frag_v, faces=frag_f, process=False).export(args.crop)
        print(f'\n碎片已保存: {args.crop}')

    if args.save_json:
        out = {
            'image': name,
            'image_size': [W, H],
            'intrinsics': {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy},
            'camera_center_world': C.tolist(),
            'rotation_world_to_cam': R.tolist(),
            'translation': t.tolist(),
            'bbox_2d': [xmin, ymin, xmax, ymax],
            'bbox_3d_corners_on_mesh': [None if h is None else h.tolist() for h in hits],
            'n_corners_hit': res['n_hit'],
            'n_outline_rays': res['n_outline'],
            'n_outline_hit': res['n_outline_hit'],
            'n_triangles_in_frustum': n_sel,
            'fragment_bbox_min': None if frag_v is None else frag_v.min(0).tolist(),
            'fragment_bbox_max': None if frag_v is None else frag_v.max(0).tolist(),
        }
        with open(args.save_json, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f'结果已保存: {args.save_json}')

    # ---------- 7. 可视化 ----------
    visualize(V, F, vcols, res, cam, scale=args.scale, uv=uv, tex_img=tex_img)


def pick_bbox_interactive(image_path):
    """弹窗,拖一个矩形,关掉窗口确认"""
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.widgets import RectangleSelector
    from PIL import Image

    img = np.asarray(Image.open(image_path).convert('RGB'))
    h, w = img.shape[:2]

    state = {'bbox': None}
    fig, ax = plt.subplots(figsize=(13, 9))
    ax.imshow(img)
    ax.set_title(f'拖一个矩形框选目标,然后关闭窗口确认\n{os.path.basename(image_path)}'
                 f'  ({w}x{h})', fontsize=11)
    ax.set_xlabel('u (px)')
    ax.set_ylabel('v (px)')

    def onselect(eclick, erelease):
        state['bbox'] = (eclick.xdata, eclick.ydata, erelease.xdata, erelease.ydata)
        print(f'  框选: ({eclick.xdata:.0f},{eclick.ydata:.0f}) -> '
              f'({erelease.xdata:.0f},{erelease.ydata:.0f})')

    sel = RectangleSelector(ax, onselect, useblit=False, button=[1],
                            minspanx=5, minspany=5, spancoords='pixels',
                            interactive=True)
    sel.set_active(True)
    plt.tight_layout()
    plt.show()

    if state['bbox'] is None:
        raise SystemExit('没有框选任何矩形。重跑一次,或者在命令行用 --bbox 指定坐标。')
    return state['bbox']


# 配色。碎片**别用橙色**:真实照片贴图本身就充满暖色(食物、砖墙、木头),
# 橙色碎片贴上去和贴图糊成一片,根本看不出哪块被选中了。用青色,冷暖对比最强。
COL_FRAGMENT = (0.0, 0.90, 1.00)       # 青色 —— 被框中的那部分网格
COL_OUTLINE = (1.0, 0.05, 0.05)        # 红色 —— 投影到表面的 3D 框轮廓
COL_RAY_HIT = (0.85, 0.10, 0.85)       # 品红 —— 打中网格的角光线
COL_RAY_MISS = (0.55, 0.55, 0.55)      # 灰色 —— 没打中的角光线


def visualize(V, F, vcols, res, cam, scale=None, uv=None, tex_img=None):
    """用 open3d 把网格 / 相机 / 像平面 / 2D框 / 投影框 全画出来。

    V,F,vcols  网格(load_mesh 的输出)
    res        project_bbox 的返回值
    cam        load_scene 里对应那张图的 dict(K,R,t,C,W,H)
    uv,tex_img 贴图坐标(每个三角形 3 组 uv)和贴图图像;给了就渲染贴图
    """
    import open3d as o3d

    K, R, t = cam['K'], cam['R'], cam['t']
    C, W, H = cam['C'], cam['W'], cam['H']
    frag_v, frag_f = res['frag_v'], res['frag_f']
    hits, corners_2d = res['hits'], res['corners_2d']
    face_mask = res.get('face_mask')

    def lineset(pts, color, closed=False, width=1.0):
        pts = np.asarray(pts, dtype=np.float64)
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(pts)
        n = len(pts)
        idx = [[i, i + 1] for i in range(n - 1)]
        if closed and n > 2:
            idx.append([n - 1, 0])
        ls.lines = o3d.utility.Vector2iVector(np.array(idx, dtype=np.int32))
        ls.paint_uniform_color(color)
        return ls

    geoms = []

    # --- 主网格 ---
    # 关键:把框内的那些三角形从主网格里挖掉,交给下面的橙色碎片去画。
    # 不挖的话两片三角形完全重合,深度测试打平,谁先画谁赢 —— 橙色永远出不来。
    keep = ~face_mask if face_mask is not None else np.ones(len(F), dtype=bool)
    F_main = np.asarray(F)[keep]

    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(V))
    o3d_mesh.triangles = o3d.utility.Vector3iVector(F_main)
    if uv is not None and tex_img is not None:
        o3d_mesh.triangle_uvs = o3d.utility.Vector2dVector(
            np.asarray(uv, dtype=np.float64)[keep].reshape(-1, 2))
        o3d_mesh.textures = [tex_img]
        o3d_mesh.triangle_material_ids = o3d.utility.IntVector(
            np.zeros(len(F_main), dtype=np.int32))
    elif vcols is not None:
        o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(vcols)
    else:
        o3d_mesh.paint_uniform_color([0.72, 0.72, 0.75])
    o3d_mesh.compute_vertex_normals()
    geoms.append(o3d_mesh)

    # --- 框内的网格碎片(高亮) ---
    if frag_v is not None and len(frag_f) > 0:
        fm = o3d.geometry.TriangleMesh()
        fm.vertices = o3d.utility.Vector3dVector(frag_v)
        fm.triangles = o3d.utility.Vector3iVector(frag_f)
        fm.paint_uniform_color(COL_FRAGMENT)
        fm.compute_vertex_normals()
        geoms.append(fm)

    # --- 相机光心 ---
    scene_extent = float(np.linalg.norm(V.max(0) - V.min(0)))
    spr = 0.012 * scene_extent
    cam_sph = o3d.geometry.TriangleMesh.create_sphere(radius=spr)
    cam_sph.translate(C)
    cam_sph.paint_uniform_color([0.0, 0.85, 0.1])
    cam_sph.compute_vertex_normals()
    geoms.append(cam_sph)

    # --- 虚拟像平面深度:默认放在相机到场景中位深度之间 ---
    if scale is None:
        _, _, d = project_points(V, K, R, t)
        d = d[d > 0]
        med = float(np.median(d)) if len(d) else scene_extent
        scale = 0.45 * med

    # 像平面四角
    img_corners_px = [(0, 0), (W, 0), (W, H), (0, H)]
    plane_pts = [point_at_depth(u, v, K, R, C, scale) for u, v in img_corners_px]
    geoms.append(lineset(plane_pts, [0.1, 0.35, 1.0], closed=True))

    # 像平面上的 2D 框
    bbox_pts = [point_at_depth(u, v, K, R, C, scale) for u, v in corners_2d]
    geoms.append(lineset(bbox_pts, [1.0, 0.9, 0.0], closed=True))

    # 从光心到像平面四角的连线(画出相机的"锥")
    for p in plane_pts:
        geoms.append(lineset([C, p], [0.1, 0.35, 1.0]))

    # --- 4 条角光线:相机 -> 网格 ---
    for i, (u, v) in enumerate(corners_2d):
        _, dvec = ray_from_pixel(u, v, K, R, C)
        end = hits[i] if hits[i] is not None else C + dvec * (scale * 4)
        col = COL_RAY_HIT if hits[i] is not None else COL_RAY_MISS
        geoms.append(lineset([C, end], col))

    # --- 网格表面上的投影框轮廓 ---
    # 用沿框一圈打出来的密集射线(默认 160 条)连成线,而不是只连 4 个角 ——
    # 4 个角只要漏一个,整条轮廓就少一大段。
    seg = res.get('outline_seg')
    ol_pts = res.get('outline_pts')
    if seg is not None and ol_pts is not None and len(seg):
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(ol_pts)
        ls.lines = o3d.utility.Vector2iVector(seg.astype(np.int32))
        ls.paint_uniform_color(COL_OUTLINE)
        geoms.append(ls)

    has_tex = uv is not None and tex_img is not None
    print('\n图例:')
    print('  🟢 绿球      = 相机光心')
    print('  🔵 蓝框      = 虚拟像平面(参考图所在的平面)')
    print('  🟡 黄框      = 你在图上画的 2D 框')
    print('  🟣 品红线    = 光心穿过框 4 个角的射线')
    print(f'  🔴 红轮廓    = 投影到网格表面的 3D 框'
          f'({int(res.get("n_outline_hit", 0))}/{int(res.get("n_outline", 0))} '
          f'条射线打到了网格)')
    print('  🟦 青色网格  = 被框中的那部分网格(视锥裁剪结果)')
    print('  ⬜ 其余网格  = ' + ('带贴图的模型' if has_tex else '完整模型'))
    print('\n鼠标操作: 左键旋转 / 滚轮缩放 / 中键或 Shift+左键平移 / 按 Q 退出')

    '''
    o3d.visualization.draw_geometries(
        geoms, window_name='2D BBox -> OpenMVS 3D Mesh',
        width=1400, height=900)

    '''
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name='2D BBox -> OpenMVS 3D Mesh',
        width=1400,
        height=900
    )

    for geom in geoms:
        vis.add_geometry(geom)

    # 3D窗口背景颜色
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.8, 0.8, 1.0])  # 白色

    vis.run()
    vis.destroy_window()

if __name__ == '__main__':
    main()
