# -*- coding: utf-8 -*-
"""
把「图生 3D」生成的模型配准到同一场景重建出来的点云上,叠加显示。

    混元模型(.glb)                     点云(OpenMVS scene_dense.ply)
         │                                      │ SAM3 mask → mask_to_points
         │                                      ▼
         │                              labels.npy(1 = building)
         └──────────────┬───────────────────────┘
                        ▼
        FPFH + RANSAC + ICP(全局配准,指标用于对照/兜底)
                        │
                        ▼
        宽度定尺度 + ±20° yaw + 包围盒居中(最终位姿)
                        │
                        ▼
        轮廓 F1 选 yaw + 深度对齐(模型立面贴到点云立面)
                        │
                        ▼
          一个 Open3D 窗口:点云(建筑=青色)+ 模型(带贴图)
                        │
                        ▼
              out\\sceaux_registered.ply(合并成一个点云)

为什么最终位姿不是 RANSAC 给的:
    这条点云几乎只有**正立面**,是一个近似平面。3D 配准沿立面方向
    是病态的:ICP/RANSAC 可以把模型沿墙面滑几十厘米甚至几米,
    fitness 依然很高。实测 RANSAC(+ICP)的位姿往往整体滑向一侧。

    所以这里反过来利用已知条件:
      - 点云和模型都是 Z-up(坐标系见下);
      - 模型主立面和点云拍摄面都朝 -Y,只差一个绕 Z 的小角度。
    用「立面宽度定尺度 + 搜 ±20° yaw + 包围盒居中」得到一个稳定
    的位姿,再用立面投影(X-Z)的轮廓 F1 选 yaw,最后把模型正面
    沿 Y 对齐到点云立面。RANSAC/ICP 仍然会跑,指标打印出来做对照,
    需要时加 --use-icp 让它参与选位姿。

为什么要先缩放:
    混元输出的模型是一个「单位尺度」的物体(实测包围盒约 0.8);
    OpenMVS 重建的点云是米级坐标(建筑宽约 10)。Open3D 的
    RANSAC 只估计旋转+平移,不估计尺度,所以必须先把模型缩到
    和点云同一个尺度。

    尺度用**立面宽度比**(X 方向),不用高度/进深:生成模型只有三张
    正面图,进深是编出来的,和真实建筑差很多;拿进深比做尺度会把
    模型缩得过小。

坐标轴:
    COLMAP/OpenMVS 的坐标系是 Y 轴朝下(相机 up = -Y)——从
    images.bin 的位姿算出来的。模型是 Y 轴朝上(glTF 惯例)。
    脚本先把点云绕 X 轴转 -90°,统一成 Z 轴朝上再配准和显示;
    合并后的 PLY 也是这个 Z-up 坐标系。

    cup 这种放在桌面上的物体例外:点云里真正朝上的是桌面法向,
    不是 -Y。加 --up-axis auto 让脚本 RANSAC 拟合最大平面,把桌面
    法向转到 +Z;配准本身也要用 --mode object(360° yaw + Chamfer)。

用法(在 0916 目录下,conda basic_env):

    $py = "C:\\Users\\FANLS\\.conda\\envs\\basic_env\\python.exe"
    $root = "D:\\code\\point_cloud\\0916"

    # 先给点云打标签(mask_to_points.py,见 docs\\mask点云分割.md)
    & $py "$root\\tools\\mask_to_points.py" `
      --scene "$root\\work\\sceaux\\dense\\sparse" `
      --mask-dir "$root\\masks_sceaux" --mask-mode index `
      --out "$root\\out\\sceaux_masked.ply" `
      --out-all "$root\\out\\sceaux_all_labeled.ply" `
      --labels "$root\\out\\sceaux_labels.npy" `
      --save-npz "$root\\out\\sceaux_full.npz"

    # 再配准 + 显示
    & $py "$root\\tools\\register_model_to_cloud.py" `
      --cloud "$root\\work\\sceaux\\mvs\\scene_dense.ply" `
      --labels "$root\\out\\sceaux_labels.npy" `
      --npz "$root\\out\\sceaux_full.npz" `
      --mesh "$root\\work\\img2_3d\\<JobId>\\xxx.glb" `
      --out-merged "$root\\out\\sceaux_registered.ply"

    # 不弹窗,只截图
    & $py ... --no-show --screenshot "$root\\work\\register.png"

    # 手动指定缩放(默认按立面宽度比,Sceaux 约 10.78)
    & $py ... --scale 10.78
"""
import argparse
import copy
import json
import os
import re
import sys
import time

import numpy as np
import open3d as o3d

# Windows 的 PowerShell/Git Bash 默认 stdout 是 GBK,遇到 ⚠ 这类字符会崩
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

# 类别颜色:类别 1 沿用仓库统一的青色(和 bbox_to_mesh.COL_FRAGMENT 一致),
# 多物体时类别 2/3/4 用对比色,保证同一个窗口里分得清。
CLASS_COLORS = {
    1: np.array([0.0, 0.90, 1.00]),   # 青
    2: np.array([1.0, 0.55, 0.10]),   # 橙
    3: np.array([0.85, 0.20, 0.90]),  # 品红
    4: np.array([0.30, 1.00, 0.35]),  # 绿
    5: np.array([1.0, 0.90, 0.10]),   # 黄
}
COL_CLASS = CLASS_COLORS[1]
COL_OTHER = np.array([0.55, 0.55, 0.55])
CAM_COLOR = np.array([1.0, 0.80, 0.10])


def class_color(class_id):
    """类别 id → 显示颜色;超过 5 个用黄金角散列,保证彼此分开。"""
    cid = int(class_id)
    if cid in CLASS_COLORS:
        return CLASS_COLORS[cid]
    import colorsys
    h = ((cid - 1) * 0.61803398875) % 1.0
    return np.array(colorsys.hsv_to_rgb(h, 0.85, 1.0))

# COLMAP/OpenMVS 是 Y 轴朝下;绕 X 转 -90° 后 -Y → +Z
ROT_UP = o3d.geometry.get_rotation_matrix_from_xyz((-np.pi / 2, 0.0, 0.0))


def log(msg):
    print(msg, flush=True)


def robust_bounds(points, lo=1.0, hi=99.0):
    """用分位数去掉离群点后的包围盒,返回 (min, max)。"""
    return (np.percentile(points, lo, axis=0),
            np.percentile(points, hi, axis=0))


def preprocess(pcd, voxel):
    """降采样 + 法线 + FPFH,给 RANSAC 用。"""
    down = pcd.voxel_down_sample(voxel)
    down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 4, max_nn=60)
    )
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 6, max_nn=100),
    )
    return down, fpfh


def global_then_icp(src, tgt, src_fpfh, tgt_fpfh, voxel):
    """RANSAC 全局配准,再用 ICP 点到面精配。

    初值(缩放+居中)在调用前已经通过变换 src 本身给进去了 ——
    RANSAC 这个接口不接受 initial 变换。
    """
    thr = voxel * 2.0
    ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src, tgt, src_fpfh, tgt_fpfh, True, thr,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(thr),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(400000, 0.999),
    )
    icp = o3d.pipelines.registration.registration_icp(
        src, tgt, thr, ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=80),
    )
    # 反方向覆盖率:只看 src→tgt 会偏向「模型缩得很小、全落在点云里面」的解,
    # 再量一遍 tgt→src,两边都覆盖得好的才是对的位置
    rev = o3d.pipelines.registration.evaluate_registration(
        tgt, src, thr, np.linalg.inv(icp.transformation))
    return ransac, icp, rev


def icp_with_init(src, tgt, init, thr):
    """从给定初值直接 ICP(不跑 RANSAC),用于「已知大致朝向」的快路径。

    生成模型和点云都是 Z-up、正面都朝 -Y,只需要搜 scale + yaw,
    比 RANSAC 的 6 自由度搜索稳定得多。
    """
    icp = o3d.pipelines.registration.registration_icp(
        src, tgt, thr, init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
    )
    rev = o3d.pipelines.registration.evaluate_registration(
        tgt, src, thr, np.linalg.inv(icp.transformation))
    return icp, rev


def balanced_fitness(f1, f2):
    return 2 * f1 * f2 / (f1 + f2) if f1 + f2 > 0 else 0.0


def silhouette_f1(model_pts, cloud_pts, bins=120):
    """把两边投影到立面平面(X-Z),算轮廓占据格子的 F1。

    不用普通 IoU:模型缩小后整个躲在点云轮廓里,IoU 只被 union 稀释,
    反而偏向「越小越好」。F1 同时看 precision(模型被点云覆盖的比例)
    和 recall(点云被模型覆盖的比例),尺度过小会被 recall 拉下来。

    这条点云几乎只有正立面,3D 距离对「沿立面滑动」不敏感;
    看轮廓重叠能更准地判断位置对不对。
    """
    m = model_pts[:, [0, 2]]
    c = cloud_pts[:, [0, 2]]
    lo = np.minimum(m.min(0), c.min(0))
    hi = np.maximum(m.max(0), c.max(0))
    xe = np.linspace(lo[0], hi[0], bins + 1)
    ze = np.linspace(lo[1], hi[1], bins + 1)
    hm, _, _ = np.histogram2d(m[:, 0], m[:, 1], bins=[xe, ze])
    hc, _, _ = np.histogram2d(c[:, 0], c[:, 1], bins=[xe, ze])
    a, b = hm > 0, hc > 0
    inter = float((a & b).sum())
    na, nb = float(a.sum()), float(b.sum())
    if inter == 0 or na == 0 or nb == 0:
        return 0.0
    precision, recall = inter / na, inter / nb
    return 2 * precision * recall / (precision + recall)


def make_scale_transform(scale, mesh_center, target_center, yaw=0.0):
    """缩放 + 绕 Z 轴 yaw + 平移到目标中心(作用在原始模型上)。"""
    R = o3d.geometry.get_rotation_matrix_from_xyz((0.0, 0.0, yaw))
    T = np.eye(4)
    T[:3, :3] = scale * R
    T[:3, 3] = target_center - T[:3, :3] @ mesh_center
    return T


def make_align_transform(scale, mesh_center, target_center, yaw=0.0):
    """只旋转+平移(作用在**已经按 scale 缩放好**的点上)。"""
    R = o3d.geometry.get_rotation_matrix_from_xyz((0.0, 0.0, yaw))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = target_center - R @ (scale * mesh_center)
    return T


def rotation_from_up(up):
    """把 up 方向转到 +Z 的旋转矩阵(点云坐标 → Z-up)。"""
    n = np.asarray(up, dtype=float)
    n = n / np.linalg.norm(n)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(n, z)
    c = float(np.dot(n, z))
    if c < -1.0 + 1e-9:  # up 正好是 -Z:绕任意垂直轴转 180°
        axis = np.array([1.0, 0.0, 0.0])
        if abs(n[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        axis = axis - n * np.dot(axis, n)
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    K = np.array([[0.0, -v[2], v[1]],
                  [v[2], 0.0, -v[0]],
                  [-v[1], v[0], 0.0]])
    return np.eye(3) + K + K @ K / (1.0 + c)


def fit_plane_up(points, target_center):
    """用最大平面(桌面/地面)的法向当 up,朝向目标中心一侧。

    返回 (单位法向, 内点比例, 距离阈值)。桌面场景里这条平面一般
    占 95% 以上的点,cup 实测 98.5%,法向很准。
    """
    pts = np.asarray(points)
    if len(pts) > 200000:
        idx = np.random.default_rng(0).choice(len(pts), 200000, replace=False)
        pts = pts[idx]
    lo, hi = robust_bounds(pts, 1.0, 99.0)
    thr = max(1e-3, 0.0015 * float(np.linalg.norm(hi - lo)))
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    model, inliers = pcd.segment_plane(thr, 3, 1000)
    a, b, c, d = model
    n = np.array([a, b, c], dtype=float)
    n /= np.linalg.norm(n)
    # 平面方程 a x + b y + c z + d = 0;让 n 从桌面指向物体
    if np.dot(target_center, n) + d < 0:
        n = -n
    return n, len(inliers) / max(1, len(pts)), thr


def resolve_up(axis_arg, cloud_points, target_center):
    """把 --up-axis 解析成旋转矩阵 + 说明文字。"""
    text = str(axis_arg).strip().lower()
    if text in ("y-down", "ydown", "camera"):
        return ROT_UP, "y-down(相机 up = -Y,COLMAP/OpenMVS 惯例)"
    if text in ("z", "z-up", "zup"):
        return np.eye(3), "点云已经是 Z-up"
    if text == "auto":
        n, ratio, thr = fit_plane_up(cloud_points, target_center)
        R = rotation_from_up(n)
        return R, (f"最大平面法向 {np.round(n, 4)} "
                   f"(内点 {ratio:.1%}, 阈值 {thr:.4f}) → +Z")
    vals = [float(v) for v in text.replace(";", ",").split(",")]
    if len(vals) != 3:
        raise SystemExit("--up-axis 只接受 y-down / auto / z / x,y,z")
    return rotation_from_up(vals), f"自定义 up {np.round(vals, 4)} → +Z"


def mean_nn_distance(src_pts, tree):
    """src 每个点到 tree 的最近邻距离均值(Chamfer 的单向项)。"""
    vals = np.empty(len(src_pts), dtype=float)
    for i, p in enumerate(src_pts):
        _, idx, d2 = tree.search_knn_vector_3d(p, 1)
        vals[i] = np.sqrt(d2[0]) if idx else np.nan
    return float(np.nanmean(vals))


def symmetric_chamfer(model_pts, cloud_pts, cloud_tree):
    """模型↔目标点的对称最近邻距离,越小越好。

    独立物体(杯子)没有「正立面」可用,轮廓 F1 也就没有投影平面,
    所以改用 3D Chamfer。ICP 是刚体变换,不会靠缩小模型来作弊。
    """
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(model_pts))
    model_tree = o3d.geometry.KDTreeFlann(pc)
    d_mc = mean_nn_distance(model_pts, cloud_tree)
    d_cm = mean_nn_distance(cloud_pts, model_tree)
    return d_mc + d_cm


def colored_points(cloud, labels, class_colors, keep_colors=True):
    """每点颜色:目标类别按 class_colors 染色;其余保留原始照片色或灰色。"""
    pts = np.asarray(cloud.points)
    if keep_colors and cloud.has_colors():
        cols = np.asarray(cloud.colors, dtype=float).copy()
    else:
        cols = np.tile(COL_OTHER, (len(pts), 1))
    for cid, col in class_colors.items():
        cols[labels == cid] = col
    return cols


def colorize_cloud(cloud, labels, class_colors, keep_colors=True):
    """显示用点云:目标类别各自染色,其余按 keep_colors 保留照片色或灰色。"""
    out = o3d.geometry.PointCloud(cloud)
    out.colors = o3d.utility.Vector3dVector(
        colored_points(cloud, labels, class_colors, keep_colors))
    return out


def build_merged_ply(cloud, labels, meshes, class_colors, n_merge_points,
                     out_path, merge_mesh=True, keep_colors=True):
    """点云 + 一个或多个模型采样点合并成一个 ply。"""
    pts = np.asarray(cloud.points)
    cols = colored_points(cloud, labels, class_colors, keep_colors)
    all_pts = [pts]
    all_cols = [cols]

    if merge_mesh:
        for class_id, mesh in meshes:
            sampled = mesh.sample_points_uniformly(number_of_points=n_merge_points)
            mp = np.asarray(sampled.points)
            if sampled.has_colors():
                mc = np.asarray(sampled.colors)
            else:  # 没有顶点色就用暖白,至少和点云分得开
                mc = np.tile(np.array([0.85, 0.72, 0.55]), (len(mp), 1))
            all_pts.append(mp)
            all_cols.append(mc)
            log(f"  类别 {class_id} 模型采样 {len(mp)} 点")

    merged = o3d.geometry.PointCloud()
    merged.points = o3d.utility.Vector3dVector(np.vstack(all_pts))
    merged.colors = o3d.utility.Vector3dVector(np.vstack(all_cols))
    o3d.io.write_point_cloud(out_path, merged, write_ascii=False)
    log(f"合并点云已保存:{out_path} ({len(merged.points)} 点)")
    return merged


def find_colmap_model(cloud_path):
    """从 <场景>/mvs/scene_dense.ply 推出 <场景>/dense/sparse。"""
    scene = os.path.dirname(os.path.dirname(os.path.abspath(cloud_path)))
    for rel in (("dense", "sparse"), ("sparse", "0"), ("sparse",)):
        p = os.path.join(scene, *rel)
        if os.path.isfile(os.path.join(p, "images.bin")):
            return p
    return None


def load_cameras(model_dir, R_up):
    """读 COLMAP 位姿,并把它们旋转到和点云相同的 Z-up 坐标系。"""
    try:
        from colmap_views import read_poses
    except ImportError:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from colmap_views import read_poses

    cams = []
    for p in read_poses(model_dir):
        Rc = np.asarray(p["R"], dtype=float)
        cams.append({
            "name": p["name"],
            "C": R_up @ np.asarray(p["C"], dtype=float),
            "forward": R_up @ (Rc.T @ np.array([0.0, 0.0, 1.0])),
            "up": R_up @ (Rc.T @ np.array([0.0, -1.0, 0.0])),
            "right": R_up @ (Rc.T @ np.array([1.0, 0.0, 0.0])),
        })
    return cams


def camera_lineset(cameras, center, diag, scale=0.05):
    """每个相机画一个四棱锥:顶点=相机中心,底面=朝向场景的取景框。

    只用于显示,不写进合并 ply。锥体大小按场景对角线缩放。
    """
    if not cameras:
        return None
    d = max(diag * scale, 1e-6)
    s = d * 0.55
    pts, lines = [], []
    for cam in cameras:
        C = cam["C"]
        f = cam["forward"] / max(np.linalg.norm(cam["forward"]), 1e-12)
        u = cam["up"] / max(np.linalg.norm(cam["up"]), 1e-12)
        r = cam["right"] / max(np.linalg.norm(cam["right"]), 1e-12)
        # 让锥体朝向场景中心(COLMAP 的 +Z 是视线方向)
        if np.dot(center - C, f) < 0:
            f = -f
        base = C + f * d
        apex = len(pts)
        pts.append(C)
        corner0 = len(pts)
        pts.extend([
            base + r * s + u * s,
            base - r * s + u * s,
            base - r * s - u * s,
            base + r * s - u * s,
        ])
        for k in range(4):
            lines.append([apex, corner0 + k])
            lines.append([corner0 + k, corner0 + (k + 1) % 4])
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.array(pts)),
        lines=o3d.utility.Vector2iVector(np.array(lines)),
    )
    ls.paint_uniform_color(CAM_COLOR)
    return ls


def show(cloud, meshes, screenshot=None, show_window=True, point_offset=0.012,
         display_voxel=0.0, cameras=None, camera_scale=0.05):
    """旧版 OpenGL Visualizer:大网格 + 大点云都能正常画。

    point_offset:把点云沿视线方向朝相机挪一点点(场景对角线的比例)。
    不挪的话,配准好的模型表面会把点云完全挡住,窗口里只能看见模型。
    这只影响显示,合并的 ply 不动。

    cameras:load_cameras() 的结果;给了就在窗口里画出每个相机的四棱锥。

    注意 Open3D 的 set_front 传的是「从物体指向相机」的方向(eye 方向),
    不是相机的视线方向。
    """
    meshes = list(meshes)
    obj_pts = [np.asarray(cloud.points)]
    for _, m in meshes:
        obj_pts.append(np.asarray(m.vertices))
    obj_P = np.vstack(obj_pts)
    obj_center = (obj_P.min(axis=0) + obj_P.max(axis=0)) / 2.0

    all_pts = [obj_P]
    if cameras:
        all_pts.append(np.array([c["C"] for c in cameras]))
    P = np.vstack(all_pts)
    lo, hi = P.min(axis=0), P.max(axis=0)
    center = (lo + hi) / 2.0
    diag = float(np.linalg.norm(hi - lo))
    # 相机在 -Y 一侧(照片拍摄的位置),略高一点
    front = np.array([-0.15, -1.0, 0.12])
    front /= np.linalg.norm(front)

    if display_voxel > 0:
        # 46 万点在 1280×800 里是每像素两个点,会把模型整个盖住;
        # 显示时降采样,合并的 ply 仍然是全量。
        cloud = cloud.voxel_down_sample(display_voxel)
        log(f"  显示用点云降采样:{len(cloud.points)} 点(体素 {display_voxel})")

    if point_offset:
        cloud = copy.deepcopy(cloud)
        cloud.translate(front * diag * point_offset)  # 朝相机挪

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="生成模型 × 重建点云 - 配准叠加",
                      width=1280, height=800, visible=show_window)
    vis.add_geometry(cloud)
    for _, m in meshes:
        vis.add_geometry(m)
    cam_geo = camera_lineset(cameras, obj_center, diag, camera_scale) if cameras else None
    if cam_geo is not None:
        vis.add_geometry(cam_geo)
    opt = vis.get_render_option()
    opt.background_color = np.array([0.05, 0.05, 0.06])
    opt.point_size = 2.5
    opt.line_width = 2.0
    opt.mesh_show_back_face = True
    opt.light_on = True

    vc = vis.get_view_control()
    vc.set_lookat(center)
    vc.set_up([0.0, 0.0, 1.0])
    vc.set_front(front)
    vc.set_zoom(0.55)
    for _ in range(3):
        vis.poll_events()
        vis.update_renderer()

    if screenshot:
        try:
            vis.capture_screen_image(screenshot, do_render=True)
            log(f"截图已保存:{screenshot}")
        except Exception as e:  # noqa: BLE001
            log(f"截图失败({e})")

    if show_window:
        log("Open3D 窗口已打开;左键旋转,滚轮缩放,Q 退出。")
        vis.run()
    vis.destroy_window()


def parse_mesh_args(items, default_class=1):
    """--mesh 接受「路径」或「类id=路径」,可重复;返回 [(class_id, path)]。"""
    out = []
    for item in items:
        text = str(item).strip()
        m = re.match(r"^(\d+)\s*[=:]\s*(.+)$", text)
        if m and not os.path.exists(text):
            cid, path = int(m.group(1)), m.group(2).strip()
        else:
            cid, path = int(default_class), text
        out.append((cid, path))
    ids = [c for c, _ in out]
    if len(ids) != len(set(ids)):
        raise SystemExit("--mesh 里同一个类别 id 出现了多次;一个物体只配一个模型")
    return out


def register_one(cloud, labels, conf, class_id, mesh_path, args):
    """把一个模型配准到 labels == class_id 的点上,返回变换后的模型和指标。

    cloud 必须已经转到 Z-up。多个物体时,每个物体调一次。
    """
    tag = f"[类别 {class_id}]"

    def out(msg):
        log(f"{tag} {msg}")

    sel = labels == class_id
    if conf is not None and args.min_conf > 0:
        sel &= conf >= args.min_conf
    n_sel = int(sel.sum())
    out(f"配准目标:{n_sel} 点(conf ≥ {args.min_conf})")
    if n_sel < 100:
        raise SystemExit(
            f"类别 {class_id} 的配准目标点太少({n_sel});"
            "检查 --labels / 类别 id / --min-conf"
        )

    pts = np.asarray(cloud.points)
    target = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts[sel]))
    # 杯子这种深色小物在稠密点云里只有两三百个点,统计滤波会把
    # 稀疏的杯身/把手当成离群点删掉(实测 205 → 188,高度从 1.23
    # 塌成 0.46)。点少的时候直接保留全部。
    if n_sel >= 1000:
        target, _ = target.remove_statistical_outlier(nb_neighbors=20,
                                                      std_ratio=2.0)
        out(f"  统计滤波后:{len(target.points)} 点")
    else:
        out(f"  目标点较少({n_sel}),跳过统计滤波")

    tgt_min, tgt_max = robust_bounds(np.asarray(target.points), 1.0, 99.0)
    tgt_center = (tgt_min + tgt_max) / 2.0
    tgt_extent = tgt_max - tgt_min
    out(f"  目标包围盒(1-99%):{np.round(tgt_extent, 3)} "
        f"中心 {np.round(tgt_center, 3)}")

    try:
        from img_to_3d import load_mesh_with_colors
    except ImportError:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from img_to_3d import load_mesh_with_colors

    out(f"读模型:{mesh_path}")
    mesh = load_mesh_with_colors(mesh_path, y_up=True)
    mesh_bbox = mesh.get_axis_aligned_bounding_box()
    mesh_center = np.asarray(mesh_bbox.get_center())
    mesh_extent = np.asarray(mesh_bbox.get_extent())
    out(f"  模型包围盒:{np.round(mesh_extent, 4)}")

    voxel = args.voxel or float(tgt_extent.max()) / 80.0
    out(f"  体素:{voxel:.4f}")

    tgt_down, tgt_fpfh = preprocess(target, voxel)
    out(f"  目标降采样:{len(tgt_down.points)} 点")

    # 模型先均匀采样一次;不同尺度候选只是套不同的缩放矩阵
    src_sampled = mesh.sample_points_uniformly(number_of_points=200000)
    out(f"  模型采样:{len(src_sampled.points)} 点")

    if args.scale:
        candidates = [args.scale]
        scale_desc = "手动指定"
    else:
        # facade:用立面宽度定尺度。高度/进深的比值不可靠:生成模型是从
        #   三张正面图重建的,进深和真实建筑差很多,拿它做尺度会把模型
        #   缩得过小或过大。轮廓 F1 也会偏向「模型缩在点云轮廓里」的解,
        #   所以这里不搜尺度。
        # object:独立物体没有单一立面,用最大边长的比值。
        axis = args.scale_axis or ("max" if args.mode == "object" else "x")
        if axis == "max":
            scale = float(tgt_extent.max() / mesh_extent.max())
            scale_desc = "最大边长比"
        else:
            i = "xyz".index(axis)
            scale = float(tgt_extent[i] / mesh_extent[i])
            scale_desc = f"{axis.upper()} 方向包围盒比"
        candidates = [scale]
    out(f"  尺度:{candidates[0]:.3f}({scale_desc})")

    best = None
    V = np.asarray(mesh.vertices)
    cloud_pts = np.asarray(target.points)
    cloud_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cloud_pts))
    cloud_tree = o3d.geometry.KDTreeFlann(cloud_pc)
    cloud_front = float(np.percentile(cloud_pts[:, 1], 5))
    if args.mode == "facade":
        out(f"  点云立面(朝相机)前沿 y={cloud_front:.2f}")

    def consider(what, T, cov_fwd, cov_rev, rmse, mode, scale):
        nonlocal best
        Vt = (T[:3, :3] @ V.T).T + T[:3, 3]
        balanced = balanced_fitness(cov_fwd, cov_rev)
        if args.mode == "object":
            # object 模式没有正立面,轮廓 F1 没有意义,用 3D Chamfer
            step = max(1, len(Vt) // 8000)
            chamfer = symmetric_chamfer(Vt[::step], cloud_pts, cloud_tree)
            out(f"  {what:22s} chamfer={chamfer:.4f} 覆盖={balanced:.3f} "
                f"rmse={rmse:.4f}")
            score = (-chamfer, balanced, -rmse)
            info = {"f1": None, "depth_err": 0.0, "chamfer": chamfer}
        else:
            f1 = silhouette_f1(Vt, cloud_pts)
            model_front = float(np.percentile(Vt[:, 1], 5))
            depth_err = abs(model_front - cloud_front)
            out(f"  {what:22s} F1={f1:.3f} 深度差={depth_err:5.2f} "
                f"覆盖={balanced:.3f} rmse={rmse:.4f}")
            # 轮廓 F1 为主;深度反正最后会对齐,不参与打分,
            # 免得把正确朝向的位姿罚掉
            score = (f1, balanced, -rmse)
            info = {"f1": f1, "depth_err": depth_err, "chamfer": None}
        if best is None or score > best["score"]:
            best = {"score": score, "T": T, "fitness": cov_fwd,
                    "rev_fitness": cov_rev, "balanced": balanced,
                    "rmse": rmse, "mode": mode, "scale": scale, **info}

    # facade:两边都是 Z-up,模型主立面和点云拍摄面都朝 -Y,只差一个绕 Z
    #   的小角度;搜 ±20° 直接 ICP,再跑一次 RANSAC 兜底。
    # object:物体的「正面」朝哪是未知的,绕 Z 搜一整圈。
    yaw_range = args.yaw_range
    yaw_step = args.yaw_step
    if yaw_range is None:
        yaw_range = 20.0 if args.mode == "facade" else 360.0
    if yaw_step is None:
        yaw_step = 5.0 if args.mode == "facade" else 15.0
    if yaw_range >= 359.0:
        n_yaw = max(1, int(round(360.0 / yaw_step)))
        yaw_deg = [i * 360.0 / n_yaw for i in range(n_yaw)]
    else:
        n_half = max(1, int(round(yaw_range / yaw_step)))
        yaw_deg = [i * yaw_step for i in range(-n_half, n_half + 1)]
    yaws = [np.deg2rad(a) for a in yaw_deg]
    out(f"  yaw 搜索:±{yaw_range:g}° / 步长 {yaw_step:g}°({len(yaws)} 个)")
    thr_icp = voxel * 3.0

    for s in candidates:
        out(f"  scale={s:.3f}")
        src_scaled = copy.deepcopy(src_sampled)
        src_scaled.scale(s, center=(0.0, 0.0, 0.0))
        src_down, src_fpfh = preprocess(src_scaled, voxel)

        # src_down 已经按 s 缩放过,icp 的结果作用在缩放后的点上;
        # 补一个 S 把「原始模型」先缩放到同一尺度,再交给 icp 的结果
        S = np.eye(4)
        S[0, 0] = S[1, 1] = S[2, 2] = s
        for yaw in yaws:
            if args.mode == "facade":
                # 直接位姿:只做缩放 + yaw + 包围盒居中,不跑 ICP。
                # 正立面几乎是平面,ICP 会沿着立面滑动,这个位姿反而是
                # 最稳的基准。
                T_dir = make_scale_transform(s, mesh_center, tgt_center, yaw)
                consider(f"直接 yaw={np.degrees(yaw):+5.0f}°", T_dir,
                         0.0, 0.0, 0.0, "direct", s)

            T_align = make_align_transform(s, mesh_center, tgt_center, yaw)
            icp, rev = icp_with_init(src_down, tgt_down, T_align, thr_icp)
            if args.use_icp or args.mode == "object":
                consider(f"ICP yaw={np.degrees(yaw):+5.0f}°",
                         icp.transformation @ S,
                         icp.fitness, rev.fitness, icp.inlier_rmse, "icp", s)
            else:
                out(f"    ICP yaw={np.degrees(yaw):+5.0f}°  "
                    f"正向={icp.fitness:.3f} 反向={rev.fitness:.3f}"
                    f"(仅参考,不参与选位姿)")

        if args.mode == "facade":
            T0 = make_scale_transform(s, mesh_center, tgt_center)
            src_r = copy.deepcopy(src_down)
            src_r.transform(T0)
            ransac, icp, rev = global_then_icp(
                src_r, tgt_down, src_fpfh, tgt_fpfh, voxel)
            if args.use_icp:
                consider("RANSAC", icp.transformation @ T0,
                         icp.fitness, rev.fitness, icp.inlier_rmse, "ransac", s)
            else:
                out(f"    RANSAC   fitness={icp.fitness:.3f} "
                    f"rmse={icp.inlier_rmse:.4f}(仅参考,不参与选位姿)")

    if best is None:
        raise SystemExit(f"类别 {class_id}:没有可用的配准位姿")
    T_final = best["T"].copy()
    if args.mode == "facade" and not args.no_depth_align:
        Vt = (T_final[:3, :3] @ V.T).T + T_final[:3, 3]
        # front = 点云最靠近相机的那层(5%);median = 点云立面的主深度。
        # 默认用 median:这条点云只有正立面,用最前沿对齐会把模型整块推到
        # 95% 的点前面,窗口里就只剩模型了。
        ref = float(np.percentile(
            cloud_pts[:, 1], 5 if args.depth_ref == "front" else 50))
        shift = ref - float(np.percentile(Vt[:, 1], 5))
        T_final[1, 3] += shift
        out(f"  深度对齐:沿 y 平移 {shift:+.2f}"
            f"(参考 {args.depth_ref}, y={ref:.2f})")

    if args.mode == "object":
        out(f"选用 {best['mode']}  scale={best['scale']:.4f}  "
            f"chamfer={best['chamfer']:.4f} 覆盖={best['balanced']:.3f} "
            f"rmse={best['rmse']:.4f}")
        out("  object 模式不假设拍摄方向;要更细可减小 --yaw-step 再跑一次")
    else:
        out(f"选用 {best['mode']}  scale={best['scale']:.4f}  "
            f"F1={best['f1']:.3f} 深度差={best['depth_err']:.2f}  "
            f"覆盖={best['balanced']:.3f} rmse={best['rmse']:.4f}")
        if best["f1"] < 0.4:
            out("  ⚠ 轮廓 F1 很低,配准可能不可靠;可用 --scale 手动指定再试")

    mesh.transform(T_final)
    return {
        "class_id": int(class_id),
        "mesh_path": os.path.abspath(mesh_path),
        "mesh": mesh,
        "transform_4x4": T_final,
        "scale": best["scale"],
        "pose_mode": best["mode"],
        "silhouette_f1": best["f1"],
        "chamfer": best["chamfer"],
        "fitness_src_to_tgt": best["fitness"],
        "fitness_tgt_to_src": best["rev_fitness"],
        "balanced_fitness": best["balanced"],
        "inlier_rmse": best["rmse"],
        "tgt_extent": tgt_extent,
        "n_target_points": n_sel,
    }


def main():
    ap = argparse.ArgumentParser(
        description="把生成模型配准到重建点云并叠加显示(支持多个物体)")
    ap.add_argument("--cloud", required=True, help="点云 ply(OpenMVS scene_dense.ply)")
    ap.add_argument("--mesh", action="append", required=True,
                    help="生成的模型 glb/obj,可重复;多物体写成 类别id=路径,"
                         "例如 --mesh 1=cup.glb --mesh 2=bottle.glb")
    ap.add_argument("--labels", default=None, help="mask_to_points 的 labels.npy")
    ap.add_argument("--npz", default=None, help="mask_to_points 的 --save-npz 结果")
    ap.add_argument("--class-id", type=int, default=1,
                    help="不给 类别id=路径 时,--mesh 使用的类别 id,默认 1")
    ap.add_argument("--min-conf", type=float, default=0.9,
                    help="取目标点时要求的最低置信度,默认 0.9;0=不过滤")
    ap.add_argument("--scale", type=float, default=None,
                    help="手动指定缩放;默认用包围盒比值(见 --scale-axis)")
    ap.add_argument("--mode", choices=["facade", "object"], default="facade",
                    help="facade=建筑立面(默认,只搜 ±20°);"
                         "object=独立物体/桌面小物(360° yaw + Chamfer)")
    ap.add_argument("--up-axis", dest="up_axis", default="y-down",
                    help="点云的上方向:y-down(默认,相机 up=-Y)/auto(拟合最大平面)"
                         "/z(已是 Z-up)/x,y,z 自定义向量")
    ap.add_argument("--yaw-range", dest="yaw_range", type=float, default=None,
                    help="yaw 搜索半角(度);默认 facade=20、object=360")
    ap.add_argument("--yaw-step", dest="yaw_step", type=float, default=None,
                    help="yaw 搜索步长(度);默认 facade=5、object=15")
    ap.add_argument("--scale-axis", dest="scale_axis",
                    choices=["x", "y", "z", "max"], default=None,
                    help="尺度用哪个包围盒方向:默认 facade=x(立面宽)、object=max")
    ap.add_argument("--voxel", type=float, default=None,
                    help="降采样体素,默认建筑最大边长的 1/80")
    ap.add_argument("--merge-points", type=int, default=200000,
                    help="合并 ply 时每个模型采样点数,默认 20 万")
    ap.add_argument("--out-merged", dest="out_merged", default=None,
                    help="合并后的点云 ply")
    ap.add_argument("--save-transform", dest="save_transform", default=None,
                    help="可选:把 4x4 变换矩阵存成 json")
    ap.add_argument("--no-depth-align", dest="no_depth_align",
                    action="store_true",
                    help="不做最后一步「模型正面贴到点云立面」的深度平移")
    ap.add_argument("--depth-ref", dest="depth_ref",
                    choices=["front", "median"], default="median",
                    help="深度对齐参考:front=点云最靠近相机的 5%%,"
                         "median=立面主深度(默认)")
    ap.add_argument("--use-icp", dest="use_icp", action="store_true",
                    help="让 RANSAC/ICP 的结果参与选位姿(默认只打印指标)。"
                         "这条点云几乎只有正立面,ICP 容易沿立面滑走,一般不要开")
    ap.add_argument("--display-voxel", dest="display_voxel", type=float,
                    default=0.0,
                    help="窗口里点云的降采样体素,默认 0=不降采样(全量显示);"
                         ">0=指定体素;-1=按目标最大边长的 1/100 自动降采样"
                         "(点太多盖住模型时用)")
    ap.add_argument("--display-offset", dest="display_offset", type=float,
                    default=0.0,
                    help="显示时点云朝相机挪动的比例(场景对角线),默认 0=严格原位;"
                         "模型把点云挡住时可以试 0.012")
    ap.add_argument("--colmap-model", dest="colmap_model", default=None,
                    help="COLMAP sparse 目录(含 images.bin);窗口里画出相机位姿。"
                         "默认从 --cloud 自动找 <场景>/dense/sparse")
    ap.add_argument("--no-cameras", dest="no_cameras", action="store_true",
                    help="不画相机位姿")
    ap.add_argument("--camera-scale", dest="camera_scale", type=float,
                    default=0.05,
                    help="相机四棱锥大小(场景对角线的比例),默认 0.05")
    ap.add_argument("--screenshot", default=None, help="叠加结果截图")
    ap.add_argument("--no-show", action="store_true", help="不弹窗")
    ap.add_argument("--no-merge-mesh", action="store_true",
                    help="合并 ply 时不要模型采样点,只存点云")
    ap.add_argument("--cloud-colors", dest="cloud_colors",
                    choices=["keep", "gray"], default="keep",
                    help="场景点云颜色:keep=保留照片色(默认),目标类别仍染色;"
                         "gray=除目标外全部灰色")
    args = ap.parse_args()

    meshes = parse_mesh_args(args.mesh, args.class_id)
    class_ids = [cid for cid, _ in meshes]
    class_colors = {cid: class_color(cid) for cid in class_ids}
    if len(meshes) > 1:
        log(f"多物体配准:{len(meshes)} 个(类别 {class_ids})")

    # RANSAC 是随机的;固定种子,保证同一个输入每次结果一致
    o3d.utility.random.seed(0)
    t0 = time.time()

    # ------------------------------------------------------------------
    # 1. 读点云 + 标签
    # ------------------------------------------------------------------
    cloud = o3d.io.read_point_cloud(args.cloud)
    n = len(cloud.points)
    log(f"点云:{n} 点")
    if args.labels:
        labels = np.load(args.labels)
        labels = np.asarray(labels).reshape(-1)
        if len(labels) != n:
            raise SystemExit(f"labels 数量 {len(labels)} 和点云 {n} 对不上")
    else:
        log("⚠ 没给 --labels,所有点都当背景,配准目标 = 整份点云")
        labels = np.zeros(n, dtype=np.int32)

    conf = None
    if args.npz and os.path.isfile(args.npz):
        conf = np.load(args.npz)["confidence"]
        if len(conf) != n:
            raise SystemExit(f"npz 里的 confidence 数量 {len(conf)} 和点云 {n} 对不上")

    # 点云转到 Z-up(和模型一致)。默认沿用 Sceaux 的假设 Y-down;
    # cup 这种放在桌面上的物体用 --up-axis auto,由桌面法向决定 up。
    sel_any = np.isin(labels, class_ids)
    if conf is not None and args.min_conf > 0:
        sel_any &= conf >= args.min_conf
    pts_raw = np.asarray(cloud.points)
    if int(sel_any.sum()) >= 10:
        center_guess = pts_raw[sel_any].mean(axis=0)
    else:
        lo, hi = robust_bounds(pts_raw, 1.0, 99.0)
        center_guess = (lo + hi) / 2.0
    R_up, up_desc = resolve_up(args.up_axis, pts_raw, center_guess)
    cloud.rotate(R_up, center=(0.0, 0.0, 0.0))
    log(f"Z-up 转换:{up_desc}")

    # ------------------------------------------------------------------
    # 2. 相机位姿(只用于显示)
    # ------------------------------------------------------------------
    cameras = None
    cam_model = args.colmap_model
    if not args.no_cameras:
        if cam_model is None:
            cam_model = find_colmap_model(args.cloud)
            if cam_model:
                log(f"相机位姿:自动找到 {cam_model}")
        if cam_model:
            if os.path.isfile(os.path.join(cam_model, "images.bin")):
                try:
                    cameras = load_cameras(cam_model, R_up)
                    log(f"相机位姿:{len(cameras)} 个")
                except Exception as e:  # noqa: BLE001
                    log(f"⚠ 读相机位姿失败({e}),不画相机")
            else:
                log(f"⚠ {cam_model} 里没有 images.bin,不画相机")
        else:
            log("没找到 COLMAP 位姿目录,不画相机(可用 --colmap-model 指定)")

    # ------------------------------------------------------------------
    # 3. 逐物体配准
    # ------------------------------------------------------------------
    results = []
    for cid, path in meshes:
        results.append(register_one(cloud, labels, conf, cid, path, args))

    # ------------------------------------------------------------------
    # 4. 输出
    # ------------------------------------------------------------------
    if args.out_merged:
        out_dir = os.path.dirname(os.path.abspath(args.out_merged))
        os.makedirs(out_dir, exist_ok=True)
        build_merged_ply(
            cloud, labels,
            [(r["class_id"], r["mesh"]) for r in results],
            class_colors, args.merge_points, args.out_merged,
            merge_mesh=not args.no_merge_mesh,
            keep_colors=args.cloud_colors == "keep",
        )

    if args.save_transform:
        objects = []
        for r in results:
            objects.append({
                "class_id": r["class_id"],
                "mesh": r["mesh_path"],
                "scale": r["scale"],
                "pose_mode": r["pose_mode"],
                "silhouette_f1": r["silhouette_f1"],
                "chamfer": r["chamfer"],
                "transform_4x4": r["transform_4x4"].tolist(),
                "fitness_src_to_tgt": r["fitness_src_to_tgt"],
                "fitness_tgt_to_src": r["fitness_tgt_to_src"],
                "balanced_fitness": r["balanced_fitness"],
                "inlier_rmse": r["inlier_rmse"],
                "n_target_points": r["n_target_points"],
            })
        data = {
            "registration_mode": args.mode,
            "up_axis": up_desc,
            "objects": objects,
            "note": "点云已按 up_axis 转成 Z-up;矩阵把原始模型变换到该坐标系",
        }
        if len(objects) == 1:
            # 单物体保留原来的顶层字段,老脚本/文档还能直接读
            o = objects[0]
            data.update({k: o[k] for k in (
                "scale", "pose_mode", "silhouette_f1", "chamfer",
                "transform_4x4", "fitness_src_to_tgt", "fitness_tgt_to_src",
                "balanced_fitness", "inlier_rmse")})
        if cameras:
            data["cameras"] = [
                {"name": c["name"], "center": [float(x) for x in c["C"]]}
                for c in cameras
            ]
        os.makedirs(os.path.dirname(os.path.abspath(args.save_transform)),
                    exist_ok=True)
        with open(args.save_transform, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        log(f"变换矩阵已保存:{args.save_transform}")

    log(f"总耗时 {time.time() - t0:.1f}s")
    if args.cloud_colors == "keep" and not cloud.has_colors():
        log("⚠ 点云没有颜色属性,退回灰色显示")
    display_cloud = colorize_cloud(cloud, labels, class_colors,
                                   keep_colors=args.cloud_colors == "keep")
    display_voxel = args.display_voxel
    if display_voxel < 0:
        max_extent = max(float(r["tgt_extent"].max()) for r in results)
        display_voxel = max_extent / 100.0
    show(display_cloud, [(r["class_id"], r["mesh"]) for r in results],
         args.screenshot, show_window=not args.no_show,
         point_offset=args.display_offset, display_voxel=display_voxel,
         cameras=cameras, camera_scale=args.camera_scale)


if __name__ == "__main__":
    main()
