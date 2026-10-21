#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从 COLMAP 相机位姿里挑出最接近 front / left / right / back 的图片。

混元多视角要求每张图带一个固定 `ViewType`(left/right/back,3.1 还支持
left_front/right_front/top/bottom),主图就是 front。这里不去猜标签,
而是用重建出来的相机位姿算方位角:

    相机中心 C = -R^T t
    物体中心 O(默认取 labels.npy 里目标类别的质心)
    方位角 = 把 C-O 投影到水平面后,相对前视图的夹角

然后穷举「哪张图当 front、其余怎么分配」,选角度误差最小的一组。

用法(在 0916 目录下):

    # 只看选择结果,输出 json + 表格
    python tools/colmap_views.py --model work\\cup\\dense\\sparse

    # 指定 front,其余自动配
    python tools\\colmap_views.py --model work\\cup\\dense\\sparse `
      --front-image 7af1443a8dd993b75028471bb434d14d.jpg

    # 打印可直接复制的 img_to_3d.py 命令
    python tools\\colmap_views.py --model work\\cup\\dense\\sparse --print-command

默认路径:
    cutouts  out\\<场景>_cutouts
    labels   out\\<场景>_labels.npy
    cloud    work\\<场景>\\mvs\\scene_dense.ply

坐标约定(和 bbox_to_mesh.py 一致):OpenMVS 场景变换是 identity,
COLMAP 世界坐标 == 点云坐标。`up` 默认用点云最大平面(桌面/地面)的法向,
符号由物体中心决定;没有点云时退回相机 up 的平均。
"""

import argparse
import itertools
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bbox_to_mesh as b2m                                   # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# API 支持的水平视角。角度是相对 front 的方位角:
#   +90 = 物体左侧(相机在物体左边), -90 = 右侧, 180 = 背面。
# 这个约定和「相机绕 up 轴逆时针转 +90°」一致。
VIEW_ANGLES = {
    "front": 0.0,
    "left_front": 45.0,
    "left": 90.0,
    "back": 180.0,
    "right": -90.0,
    "right_front": -45.0,
}
TOP_BOTTOM = ("top", "bottom")
API_VIEW_TYPES = tuple(VIEW_ANGLES) + TOP_BOTTOM
MODEL_31_ONLY = {"top", "bottom", "left_front", "right_front"}
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")


# -----------------------------------------------------------------------------
# 路径推断
# -----------------------------------------------------------------------------
def scene_root(model_dir):
    """<场景>/dense/sparse 或 <场景>/sparse/0 -> <场景>"""
    return os.path.dirname(os.path.dirname(os.path.abspath(model_dir)))


def default_paths(model_dir):
    root = scene_root(model_dir)
    scene = os.path.basename(root)
    out = os.path.join(PROJECT_ROOT, "out")
    return {
        "scene": scene,
        "cloud": os.path.join(root, "mvs", "scene_dense.ply"),
        "cutouts": os.path.join(out, f"{scene}_cutouts"),
        "labels": os.path.join(out, f"{scene}_labels.npy"),
    }


def default_images_dir(model_dir):
    """去畸变图目录。--cutouts 不存在时退回这里。"""
    par = os.path.dirname(os.path.abspath(model_dir))
    for nm in ("images", os.path.join("dense", "images")):
        p = os.path.join(par, nm)
        if os.path.isdir(p):
            return p
    return os.path.join(par, "images")


# -----------------------------------------------------------------------------
# 读 COLMAP 位姿
# -----------------------------------------------------------------------------
def read_poses(model_dir):
    """返回 [{id, name, R, C, forward, up}, ...]"""
    path = os.path.join(model_dir, "images.bin")
    if not os.path.isfile(path):
        raise SystemExit(f"找不到 {path};--model 要指向 COLMAP sparse 目录")
    out = []
    for iid, qvec, tvec, _cid, name in b2m.read_images_bin(path):
        R = b2m.qvec_to_rotmat(qvec)
        out.append({
            "id": iid,
            "name": name,
            "R": R,
            "C": -R.T @ tvec,
            "forward": R.T @ np.array([0.0, 0.0, 1.0]),
            "up": R.T @ np.array([0.0, -1.0, 0.0]),
        })
    if not out:
        raise SystemExit(f"{path} 里没有注册图像")
    return out


def camera_up(poses):
    """相机 up 的平均;返回 (单位向量, 一致度 0~1)。"""
    us = np.array([p["up"] / np.linalg.norm(p["up"]) for p in poses])
    mean = us.mean(axis=0)
    consistency = float(np.linalg.norm(mean))
    if consistency < 1e-6:
        return np.array([0.0, 0.0, 1.0]), 0.0
    return mean / consistency, consistency


# -----------------------------------------------------------------------------
# 点云 / 平面 / 物体中心
# -----------------------------------------------------------------------------
def _read_points(path, max_points=None):
    import open3d as o3d

    pc = o3d.io.read_point_cloud(path)
    P = np.asarray(pc.points, dtype=np.float64)
    if len(P) == 0:
        raise SystemExit(f"点云是空的:{path}")
    if max_points and len(P) > max_points:
        rng = np.random.default_rng(0)
        P = P[rng.choice(len(P), max_points, replace=False)]
    return P


def plane_up(cloud_path, center=None, threshold=None):
    """RANSAC 找最大平面,法向当 up。返回 (up, inlier_ratio, threshold, n_points)。"""
    import open3d as o3d

    P = _read_points(cloud_path)
    diag = float(np.linalg.norm(P.max(axis=0) - P.min(axis=0)))
    th = float(threshold) if threshold else max(1e-3, 0.0015 * diag)
    if len(P) > 400000:
        rng = np.random.default_rng(0)
        P_fit = P[rng.choice(len(P), 400000, replace=False)]
    else:
        P_fit = P
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P_fit))
    try:
        o3d.utility.random.seed(0)               # RANSAC 结果固定,便于复现
    except Exception:  # noqa: BLE001
        pass
    model, inliers = cloud.segment_plane(
        distance_threshold=th, ransac_n=3, num_iterations=3000
    )
    n = np.asarray(model[:3], dtype=np.float64)
    n /= np.linalg.norm(n)
    d = float(model[3])
    ratio = len(inliers) / len(P_fit)

    # 法向符号:让物体中心落在平面正侧
    if center is not None:
        if np.dot(n, center) + d < 0:
            n = -n
    else:
        rest = np.delete(P_fit, np.asarray(inliers, dtype=np.int64), axis=0)
        if len(rest) and np.dot(n, rest.mean(axis=0)) + d < 0:
            n = -n
    return n, ratio, th, len(P)


def label_center(cloud_path, labels_path, class_id=1, min_conf=None, npz_path=None):
    """labels.npy 里类别 class_id 的点的质心。支持 --npz 里的 confidence。"""
    P = _read_points(cloud_path)
    labels = np.load(labels_path)
    labels = np.asarray(labels).reshape(-1)
    if labels.shape[0] != P.shape[0]:
        raise SystemExit(
            f"labels 长度 {labels.shape[0]} 和点云 {P.shape[0]} 对不上;"
            "labels 必须用同一个 cloud 跑 mask_to_points.py 生成"
        )
    keep = labels == class_id
    if npz_path:
        z = np.load(npz_path)
        keys = set(z.files)
        if "confidence" in keys:
            conf = np.asarray(z["confidence"]).reshape(-1)
            if conf.shape[0] == P.shape[0] and min_conf is not None:
                keep &= conf >= float(min_conf)
        elif "conf" in keys:
            conf = np.asarray(z["conf"]).reshape(-1)
            if conf.shape[0] == P.shape[0] and min_conf is not None:
                keep &= conf >= float(min_conf)
    pts = P[keep]
    if len(pts) == 0:
        raise SystemExit(
            f"labels 里类别 {class_id} 没有点;检查 --class-id 或重新生成 labels"
        )
    return pts.mean(axis=0), len(pts), P[keep]


def ray_center(poses):
    """所有相机光轴的最小二乘交点。没有 labels 时的兜底。"""
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for p in poses:
        d = p["forward"] / np.linalg.norm(p["forward"])
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ p["C"]
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.mean([p["C"] for p in poses], axis=0)


# -----------------------------------------------------------------------------
# 方位角
# -----------------------------------------------------------------------------
def _unit(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


def _horizontal_ref(up):
    """水平面内一个固定参考方向(只用来定义绝对 0°,不参与最终选择)。"""
    for cand in (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])):
        h = cand - np.dot(cand, up) * up
        if np.linalg.norm(h) > 0.1:
            return _unit(h)
    return _unit(np.array([0.0, 0.0, 1.0]))


def add_angles(poses, center, up):
    """给每个 pose 加 az(度)、elev(度)、dist。相机在物体正上方时 az=None。"""
    ref = _horizontal_ref(up)
    for p in poses:
        d = p["C"] - center
        dist = float(np.linalg.norm(d))
        p["dist"] = dist
        if dist < 1e-9:
            p["az"] = None
            p["elev"] = None
            continue
        p["elev"] = math.degrees(math.asin(float(np.clip(np.dot(d / dist, up), -1, 1))))
        h = d - np.dot(d, up) * up
        if np.linalg.norm(h) < 1e-6:
            p["az"] = None
            continue
        h = _unit(h)
        p["az"] = math.degrees(
            math.atan2(float(np.dot(np.cross(ref, h), up)), float(np.dot(ref, h)))
        )
    return poses


def _wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


# -----------------------------------------------------------------------------
# 选图
# -----------------------------------------------------------------------------
def _find_front(poses, front_image):
    base = os.path.basename(str(front_image))
    stem = os.path.splitext(base)[0]
    for p in poses:
        if p["name"] == base or os.path.splitext(os.path.basename(p["name"]))[0] == stem:
            return p
    names = ", ".join(os.path.splitext(os.path.basename(p["name"]))[0] for p in poses)
    raise SystemExit(f"--front-image 不在模型里:{front_image}\n可选:{names}")


def select_views(poses, view_types=("front", "left", "right", "back"),
                 front_image=None, max_error=45.0, swap_left_right=False,
                 strict=False, min_coverage=0.0, size_weight=15.0,
                 border_penalty=0.5):
    """穷举 front 候选 + 其余视角的排列,选「角度准 + 图大且完整」的一组。

    打分不再只看方位误差,而是

        cost = 方位 RMS(度) + size_weight × (1 - 平均图片质量)

    图片质量 q ∈ [0, 1]:
      - 目标在原始照片里的像素占比 / 本组最大值(越"大"越好);
      - 目标贴到画面边缘(touches_border)时再乘 (1 - border_penalty),
        因为被裁掉一部分的照片"不完整"。
    size_weight 的单位是度:默认 15 表示「面积翻倍 ≈ 抵 7.5° 方位误差」。
    没有 report.json(比如直接拿原图)时所有 q 都取 0.5,退化成纯角度选择。

    返回 (front_pose, assignment, info)
      assignment: {view_type: pose}
      info: rms/max error、cost、平均质量、warning
    """
    types = [t.strip() for t in view_types if t.strip()]
    if "front" not in types:
        types.insert(0, "front")
    types = ["front"] + [t for t in types if t != "front"]
    unknown = [t for t in types if t not in API_VIEW_TYPES]
    if unknown:
        raise SystemExit(
            f"不认识的视角 {unknown};可选:{', '.join(API_VIEW_TYPES)}"
        )

    cands = [
        p for p in poses
        if p.get("az") is not None
        # 没有 report.json 时(直接拿原图)不知道覆盖率,不能让 --min-coverage
        # 把所有图都筛掉
        and (p.get("coverage", 0.0) >= min_coverage or not p.get("has_report"))
        and p.get("path")
    ]
    max_cov = max((p.get("coverage", 0.0) for p in cands), default=0.0)

    def quality(p):
        if not p.get("has_report"):
            return 0.5
        q = (p.get("coverage", 0.0) / max_cov) if max_cov > 0 else 0.0
        if p.get("touches_border"):
            q *= max(0.0, 1.0 - border_penalty)
        return q

    if len(cands) < len(types):
        raise SystemExit(
            f"可用图片 {len(cands)} 张,不够 {len(types)} 个视角;"
            "降低 --min-coverage 或减少 --views"
        )

    fronts = [_find_front(cands, front_image)] if front_image else list(cands)
    horiz_types = [t for t in types if t in VIEW_ANGLES and t != "front"]
    tb_types = [t for t in types if t in TOP_BOTTOM]

    best = None
    for f in fronts:
        rest = [p for p in cands if p is not f]
        if len(horiz_types) > len(rest):
            continue
        for perm in itertools.permutations(rest, len(horiz_types)):
            assign = {t: p for t, p in zip(horiz_types, perm)}
            errs = []
            for t in horiz_types:
                target = VIEW_ANGLES[t]
                if swap_left_right and t in ("left", "left_front", "right", "right_front"):
                    target = -target
                errs.append(abs(_wrap_deg(assign[t]["az"] - f["az"] - target)))
            # front 的误差恒为 0,也算进 RMS,和打印的 4 个视角保持一致
            rms = math.sqrt(sum(e * e for e in errs) / (len(errs) + 1)) if errs else 0.0
            mx = max(errs) if errs else 0.0

            # top/bottom 不走方位角,按俯仰角挑剩余的图
            used = set(id(p) for p in assign.values()) | {id(f)}
            remain = [p for p in cands if id(p) not in used]
            tb_assign = {}
            tb_errors = {}
            ok = True
            for t in tb_types:
                pool = [p for p in remain if id(p) not in set(id(x) for x in tb_assign.values())]
                if not pool:
                    ok = False
                    break
                if t == "top":
                    p = max(pool, key=lambda x: x.get("elev", -90.0))
                    tb_errors[t] = abs(90.0 - p.get("elev", 0.0))
                else:
                    p = min(pool, key=lambda x: x.get("elev", 90.0))
                    tb_errors[t] = abs(-90.0 - p.get("elev", 0.0))
                tb_assign[t] = p
            if not ok:
                continue

            chosen = [f] + list(assign.values()) + list(tb_assign.values())
            cov = sum(p.get("coverage", 0.0) for p in chosen)
            mean_q = sum(quality(p) for p in chosen) / len(chosen)
            # 角度(度)和图片质量折算到同一个量纲:q 每低 1,罚 size_weight 度
            cost = rms + size_weight * (1.0 - mean_q)
            score = (round(cost, 6), round(rms, 6), -round(cov, 9))
            if best is None or score < best["score"]:
                best = {
                    "score": score,
                    "front": f,
                    "assign": assign,
                    "tb": tb_assign,
                    "tb_errors": tb_errors,
                    "rms": rms,
                    "max": mx,
                    "cost": cost,
                    "mean_quality": mean_q,
                }

    if best is None:
        raise SystemExit("凑不出要求的视角组合")

    all_errs = [abs(_wrap_deg(best["assign"][t]["az"] - best["front"]["az"] - VIEW_ANGLES[t]))
                for t in horiz_types]
    if swap_left_right:
        all_errs = []
        for t in horiz_types:
            target = VIEW_ANGLES[t]
            if t in ("left", "left_front", "right", "right_front"):
                target = -target
            all_errs.append(abs(_wrap_deg(best["assign"][t]["az"] - best["front"]["az"] - target)))
    best["errors"] = {t: e for t, e in zip(horiz_types, all_errs)}
    warning = None
    if best["max"] > max_error:
        warning = (
            f"最佳组合的最大方位误差 {best['max']:.1f}° 超过 --max-error {max_error:.1f}°;"
            "照片本身没有更接近的视角,仍按最佳组合提交"
        )
        if strict:
            raise SystemExit(warning)
    best["warning"] = warning
    for p in poses:
        p["quality"] = quality(p)
    return best["front"], {"front": best["front"], **best["assign"], **best["tb"]}, {
        "rms": best["rms"],
        "max": best["max"],
        "cost": best["cost"],
        "mean_quality": best["mean_quality"],
        "errors": best["errors"],
        "tb_errors": best["tb_errors"],
        "warning": warning,
    }


# -----------------------------------------------------------------------------
# 输出 / 路径
# -----------------------------------------------------------------------------
def _find_image(cutouts_dir, images_dir, stem):
    """返回 (path, source)。优先抠图,其次原去畸变图。"""
    for d, src in ((cutouts_dir, "cutout"), (images_dir, "raw")):
        if not d or not os.path.isdir(d):
            continue
        for ext in IMAGE_EXTS:
            p = os.path.join(d, stem + ext)
            if os.path.isfile(p):
                # 抠图目录不存在时 resolve() 会把它指到原图目录,别误标成 cutout
                if src == "cutout" and images_dir and \
                        os.path.abspath(d) == os.path.abspath(images_dir):
                    src = "raw"
                return p, src
    return None, None


def _load_report(cutouts_dir):
    """读 mask_cutout.py 的 report.json:每张图的占比 + 是否贴边。

    返回 {stem: {"coverage": float, "touches_border": bool}}。
    没有 report.json(例如直接拿原图)时返回空,选择器会退化成纯角度。
    """
    if not cutouts_dir:
        return {}
    p = os.path.join(cutouts_dir, "report.json")
    if not os.path.isfile(p):
        return {}
    try:
        with open(p, encoding="utf-8") as f:
            rep = json.load(f)
        return {
            o["stem"]: {
                "coverage": float(o.get("coverage", 0.0)),
                "touches_border": bool(o.get("touches_border", False)),
            }
            for o in rep.get("outputs", [])
        }
    except Exception:  # noqa: BLE001
        return {}


def _parse_center(text):
    if not text:
        return None
    parts = [x for x in str(text).replace(" ", "").split(",") if x]
    if len(parts) != 3:
        raise SystemExit("--center 要写成 x,y,z")
    return np.array([float(x) for x in parts], dtype=np.float64)


def resolve(model_dir, cutouts=None, labels=None, cloud=None, center=None,
            class_id=1, min_conf=None, npz=None, up="auto", plane_threshold=None,
            view_types=("front", "left", "right", "back"), front_image=None,
            max_error=45.0, swap_left_right=False, strict=False,
            min_coverage=0.0, allow_raw=True, quiet=False,
            size_weight=15.0, border_penalty=0.5):
    """主入口:返回 selection 字典,里面的 pose 已带 path/source。"""
    model_dir = os.path.abspath(model_dir)
    dft = default_paths(model_dir)
    images_dir = default_images_dir(model_dir)
    cutouts = cutouts or dft["cutouts"]
    labels = labels or dft["labels"]
    cloud = cloud or dft["cloud"]

    poses = read_poses(model_dir)
    cutouts_is_raw = False
    if not os.path.isdir(cutouts):
        if allow_raw and os.path.isdir(images_dir):
            print(f"⚠️  抠图目录不存在:{cutouts};退回原图 {images_dir}(背景会进模型)")
            cutouts = images_dir
            cutouts_is_raw = True
        else:
            raise SystemExit(
                f"抠图目录不存在:{cutouts}\n"
                "先跑 mask_cutout.py,或用 --cutouts 指定目录"
            )

    report = _load_report(cutouts)
    missing = []
    for p in poses:
        stem = os.path.splitext(os.path.basename(p["name"]))[0]
        # 抠图目录存在时,某张图没有该物体的抠图 = 这张图里没检测到它,
        # 不能退回原图顶替(那会把背景和别的物体带进模型)。
        raw_lookup = images_dir if (allow_raw and cutouts_is_raw) else None
        path, src = _find_image(cutouts, raw_lookup, stem)
        if path is None:
            missing.append(p["name"])
            continue
        p["path"] = path
        p["source"] = src
        info = report.get(stem)
        p["coverage"] = info["coverage"] if info else 0.0
        p["touches_border"] = info["touches_border"] if info else False
        p["has_report"] = info is not None
    if missing:
        print(f"⚠️  这些图找不到对应文件,不参与选择:{', '.join(missing)}")

    # 物体中心
    center_arr = _parse_center(center)
    center_source = "cli"
    if center_arr is None and labels and os.path.isfile(labels):
        if not os.path.isfile(cloud):
            raise SystemExit(f"有 labels 但找不到点云:{cloud};用 --cloud 指定")
        center_arr, n_obj, obj_pts = label_center(
            cloud, labels, class_id=class_id, min_conf=min_conf, npz_path=npz
        )
        center_source = f"labels(class {class_id}, {n_obj} pts)"
    if center_arr is None:
        center_arr = ray_center(poses)
        center_source = "camera-axes"
        print("⚠️  没给 --center / labels,用相机光轴交点当物体中心")

    # up
    up_arg = str(up or "auto").strip()
    up_source = ""
    if up_arg == "camera":
        up_vec, cons = camera_up(poses)
        up_source = f"camera(一致度 {cons:.2f})"
    elif up_arg == "plane":
        if not os.path.isfile(cloud):
            raise SystemExit(f"--up plane 需要点云:{cloud}")
        up_vec, ratio, th, npts = plane_up(cloud, center_arr, plane_threshold)
        up_source = f"plane({ratio * 100:.1f}% inliers, {npts} pts)"
    elif up_arg == "auto":
        if os.path.isfile(cloud):
            up_vec, ratio, th, npts = plane_up(cloud, center_arr, plane_threshold)
            up_source = f"plane({ratio * 100:.1f}% inliers, {npts} pts)"
            if ratio < 0.2:
                print(f"⚠️  最大平面只有 {ratio * 100:.1f}% 的内点,up 可能不可靠;可改 --up camera")
        else:
            up_vec, cons = camera_up(poses)
            up_source = f"camera(一致度 {cons:.2f})"
    else:
        parts = [x for x in up_arg.replace(" ", "").split(",") if x]
        if len(parts) != 3:
            raise SystemExit("--up 只接受 auto / camera / plane / x,y,z")
        up_vec = _unit(np.array([float(x) for x in parts]))
        up_source = "cli"

    add_angles(poses, center_arr, up_vec)
    front, assign, info = select_views(
        poses, view_types=view_types, front_image=front_image,
        max_error=max_error, swap_left_right=swap_left_right,
        strict=strict, min_coverage=min_coverage,
        size_weight=size_weight, border_penalty=border_penalty,
    )

    out_views = []
    for t in ["front"] + [x for x in view_types if x != "front"]:
        p = assign[t]
        if t == "front":
            rel = 0.0
            target = 0.0
            err = 0.0
        elif t in TOP_BOTTOM:
            rel = None
            target = None
            err = info["tb_errors"].get(t)
        else:
            target = VIEW_ANGLES[t]
            if swap_left_right and t in ("left", "left_front", "right", "right_front"):
                target = -target
            rel = _wrap_deg(p["az"] - front["az"])
            err = info["errors"].get(t)
        out_views.append({
            "view_type": t,
            "image": p["name"],
            "path": p["path"],
            "source": p.get("source"),
            "azimuth_deg": None if p.get("az") is None else float(p["az"]),
            "relative_to_front_deg": None if rel is None else float(rel),
            "target_deg": target,
            "error_deg": None if err is None else float(err),
            "elevation_deg": None if p.get("elev") is None else float(p["elev"]),
            "coverage": float(p.get("coverage", 0.0)),
            "touches_border": bool(p.get("touches_border", False)),
            "quality": round(float(p.get("quality", 0.0)), 4),
        })

    selection = {
        "model": model_dir,
        "scene": dft["scene"],
        "images_dir": images_dir,
        "cutouts": cutouts,
        "cloud": cloud,
        "labels": labels,
        "up": [float(x) for x in up_vec],
        "up_source": up_source,
        "center": [float(x) for x in center_arr],
        "center_source": center_source,
        "front_azimuth_deg": float(front["az"]),
        "rms_error_deg": float(info["rms"]),
        "max_error_deg": float(info["max"]),
        "cost": float(info["cost"]),
        "mean_quality": float(info["mean_quality"]),
        "size_weight": float(size_weight),
        "border_penalty": float(border_penalty),
        "warning": info["warning"],
        "views": out_views,
    }
    return selection


def print_selection(sel, stream=sys.stdout):
    print(f"模型       {sel['model']}", file=stream)
    print(f"up         {np.round(sel['up'], 4)}  [{sel['up_source']}]", file=stream)
    print(f"物体中心   {np.round(sel['center'], 4)}  [{sel['center_source']}]", file=stream)
    print(f"前视方位   {sel['front_azimuth_deg']:.1f}°", file=stream)
    print(f"误差       RMS {sel['rms_error_deg']:.1f}°, max {sel['max_error_deg']:.1f}°"
          f"  | 综合代价 {sel.get('cost', 0.0):.1f}"
          f"(图片质量 {sel.get('mean_quality', 0.0):.2f})",
          file=stream)
    if sel.get("warning"):
        print(f"⚠️  {sel['warning']}", file=stream)
    print("", file=stream)
    print(f"{'视角':<11}{'相对front':>10}{'误差':>8}{'俯仰':>8}{'覆盖率':>8}{'质量':>7}  图片",
          file=stream)
    for v in sel["views"]:
        rel = "-" if v["relative_to_front_deg"] is None else f"{v['relative_to_front_deg']:+.1f}°"
        err = "-" if v["error_deg"] is None else f"{v['error_deg']:.1f}°"
        elev = "-" if v["elevation_deg"] is None else f"{v['elevation_deg']:.1f}°"
        tag = " *贴边" if v.get("touches_border") else ""
        print(
            f"{v['view_type']:<11}{rel:>10}{err:>8}{elev:>8}"
            f"{100 * v['coverage']:>7.2f}%{v.get('quality', 0.0):>7.2f}  "
            f"{os.path.basename(v['path'])}{tag}",
            file=stream,
        )


def save_preview(sel, path, width=420):
    """把选中的几张图拼成一张预览图,提交前先肉眼确认。"""
    from PIL import Image, ImageDraw

    views = sel["views"]
    if not views:
        return
    im0 = Image.open(views[0]["path"])
    ratio = im0.height / max(im0.width, 1)
    height = int(width * ratio) if ratio > 0 else width
    sheet = Image.new("RGB", (width * len(views), height + 34), (30, 30, 30))
    draw = ImageDraw.Draw(sheet)
    for i, v in enumerate(views):
        im = Image.open(v["path"]).convert("RGB").resize((width, height), Image.LANCZOS)
        sheet.paste(im, (i * width, 34))
        rel = v.get("relative_to_front_deg")
        err = v.get("error_deg")
        txt = v["view_type"]
        if rel is not None:
            txt += f"  rel={rel:+.1f}"
        if err is not None:
            txt += f"  err={err:.1f}"
        draw.text((i * width + 8, 10), txt, fill=(255, 255, 255))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    sheet.save(path, quality=92)
    print(f"预览图 -> {os.path.abspath(path)}")


def build_submit_args(sel, python_exe="python", script=None, model_version=None):
    """生成 img_to_3d.py 的参数列表(不含 python/脚本)。"""
    if model_version is None:
        model_version = "3.1" if any(
            v["view_type"] in MODEL_31_ONLY for v in sel["views"]
        ) else "3.0"
    args = ["--pro", "--model-version", model_version,
            "--image", sel["views"][0]["path"]]
    for v in sel["views"][1:]:
        args += ["--view", f"{v['view_type']}={v['path']}"]
    return args


def print_command(sel, python_exe="python", script=None):
    script = script or os.path.join(PROJECT_ROOT, "tools", "img_to_3d.py")
    args = build_submit_args(sel, python_exe=python_exe, script=script)
    print("\n可直接运行:")
    parts = [f'& "{python_exe}" "{script}"']
    for a in args:
        parts.append(f'"{a}"' if " " in a else a)
    print(" ".join(parts))


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_view_types(text):
    if not text:
        return ("front", "left", "right", "back")
    return tuple(x.strip() for x in str(text).split(",") if x.strip())


def _fix_center_argv(argv):
    """argparse 会把 '-0.3,2,3' 当成选项;改写成 --center=-0.3,2,3。"""
    out = []
    i = 0
    while i < len(argv):
        if argv[i] == "--center" and i + 1 < len(argv) and argv[i + 1].startswith("-"):
            out.append("--center=" + argv[i + 1])
            i += 2
        else:
            out.append(argv[i])
            i += 1
    return out


def main():
    ap = argparse.ArgumentParser(
        description="按 COLMAP 相机位姿挑 front/left/right/back 图片",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--model", required=True, help="COLMAP sparse 目录(含 images.bin)")
    ap.add_argument("--cutouts", default=None, help="抠图目录;默认 out/<场景>_cutouts")
    ap.add_argument("--labels", default=None, help="每点类别 .npy;默认 out/<场景>_labels.npy")
    ap.add_argument("--cloud", default=None, help="点云 .ply;默认 <场景>/mvs/scene_dense.ply")
    ap.add_argument("--center", default=None, help="物体中心 x,y,z;不给就按 labels/光轴交点")
    ap.add_argument("--class-id", type=int, default=1, help="目标类别 id,默认 1")
    ap.add_argument("--npz", default=None, help="mask_to_points 的 npz(可选,用于 confidence)")
    ap.add_argument("--min-conf", type=float, default=None, help="配合 --npz 的最低置信度")
    ap.add_argument("--up", default="auto", help="auto / camera / plane / x,y,z")
    ap.add_argument("--plane-threshold", type=float, default=None,
                    help="RANSAC 平面距离阈值;默认点云对角线的 0.15%%")
    ap.add_argument("--views", default="front,left,right,back",
                    help="要选的视角,逗号分隔;front 总会包含")
    ap.add_argument("--front-image", default=None, help="指定前视图文件名(可带/不带扩展名)")
    ap.add_argument("--max-error", type=float, default=45.0,
                    help="最大可接受方位误差(度);超过只警告,除非 --strict")
    ap.add_argument("--strict", action="store_true", help="误差超过 --max-error 直接失败")
    ap.add_argument("--swap-left-right", action="store_true",
                    help="交换 left/right(不同数据集的左右约定可能相反)")
    ap.add_argument("--min-coverage", type=float, default=0.0,
                    help="mask 覆盖率低于此值的图不参与;0.01 = 1%%")
    ap.add_argument("--size-weight", type=float, default=15.0,
                    help="图片大小的权重(折合成方位角,单位:度),默认 15;"
                         "0=只看角度,越大越偏向目标占比大、完整的照片")
    ap.add_argument("--border-penalty", type=float, default=0.5,
                    help="目标贴到画面边缘时质量打折的比例,默认 0.5(打五折)")
    ap.add_argument("--no-raw-fallback", action="store_true",
                    help="找不到抠图就不要退回原图")
    ap.add_argument("--out", default=None, help="选择结果 json 路径")
    ap.add_argument("--preview", default=None, help="把选中的图拼成一张预览图")
    ap.add_argument("--print-command", action="store_true", help="打印 img_to_3d.py 命令")
    ap.add_argument("--python", default=sys.executable, help="--print-command 用的解释器")
    args = ap.parse_args(_fix_center_argv(sys.argv[1:]))

    sel = resolve(
        args.model, cutouts=args.cutouts, labels=args.labels, cloud=args.cloud,
        center=args.center, class_id=args.class_id, min_conf=args.min_conf,
        npz=args.npz, up=args.up, plane_threshold=args.plane_threshold,
        view_types=parse_view_types(args.views), front_image=args.front_image,
        max_error=args.max_error, swap_left_right=args.swap_left_right,
        strict=args.strict, min_coverage=args.min_coverage,
        allow_raw=not args.no_raw_fallback,
        size_weight=args.size_weight, border_penalty=args.border_penalty,
    )
    print_selection(sel)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(sel, f, ensure_ascii=False, indent=2)
        print(f"\n选择结果 -> {os.path.abspath(args.out)}")
    if args.preview:
        save_preview(sel, args.preview)
    if args.print_command:
        print_command(sel, python_exe=args.python)


if __name__ == "__main__":
    main()
