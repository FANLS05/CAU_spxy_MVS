# -*- coding: utf-8 -*-
"""直接显示已经配准好的结果,不重新跑配准。

输入是 register_model_to_cloud.py 的两个产物:

    out\\<场景>_registered.ply       --out-merged 合并后的点云(点云 + 模型采样点)
    out\\<场景>_registration.json    --save-transform 的变换和相机位姿(可选)

registered.ply 已经是 Z-up,生成模型的采样点也按纹理着色合并进去了,
所以这个脚本只做「读文件 + 画窗口」,不会再碰 FPFH/ICP。

用法(0916 目录下,conda basic_env):

    $py = "C:\\Users\\FANLS\\.conda\\envs\\basic_env\\python.exe"

    # 直接看
    & $py "$root\\tools\\show_registered.py" `
      --merged "$root\\out\\cup2_registered.ply" `
      --registration "$root\\out\\cup2_registration.json"

    # 只截图,不弹窗
    & $py "$root\\tools\\show_registered.py" `
      --merged "$root\\out\\cup2_registered.ply" `
      --registration "$root\\out\\cup2_registration.json" `
      --no-show --screenshot "$root\\out\\cup2_registered.png"

相机位姿:
    JSON 里存了 forward/up/right 时直接画真实四棱锥;
    老 JSON 只有 center 时,会自动找 work\\<场景>\\dense\\sparse 重新读位姿;
    两者都没有就只画黄色小球标出拍摄位置。
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import open3d as o3d

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from register_model_to_cloud import (  # noqa: E402
    CAM_COLOR,
    ROT_UP,
    camera_lineset,
    class_color,
    colored_points,
    load_cameras,
    rotation_from_up,
)


def log(msg):
    print(msg, flush=True)


def robust_bounds(points, lo=1.0, hi=99.0):
    return (np.percentile(points, lo, axis=0),
            np.percentile(points, hi, axis=0))


def read_json(path):
    if not path:
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:  # noqa: BLE001
        log(f"⚠ 读 registration JSON 失败({e}),只显示点云")
        return {}


def json_rot_up(data):
    """从 JSON 恢复「点云 → Z-up」的旋转。

    新 JSON 直接有 rot_up_4x4;老 JSON 只有 up_axis 文字,里面带着
    拟合出来的平面法向,从文字里把向量抠出来再重建同一个旋转。
    """
    m = data.get("rot_up_4x4")
    if m:
        return np.asarray(m, dtype=float)[:3, :3]

    desc = str(data.get("up_axis") or "")
    hit = re.search(r"\[([^\]]+)\]", desc)
    if hit:
        vals = [float(v) for v in re.findall(
            r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", hit.group(1))]
        if len(vals) == 3:
            return rotation_from_up(vals)
    if "y-down" in desc or "ydown" in desc:
        return ROT_UP
    if "z-up" in desc.lower() or "已经是" in desc:
        return np.eye(3)
    return None


def find_colmap_model(merged, data):
    """优先用 JSON 里记的路径,其次按 <场景>_registered.ply 猜 work\\<场景>\\dense\\sparse。"""
    cand = data.get("colmap_model")
    if cand and os.path.isfile(os.path.join(cand, "images.bin")):
        return cand

    scene = re.sub(r"_registered(\.ply)?$", "", os.path.basename(merged),
                   flags=re.IGNORECASE)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for p in (os.path.join(root, "work", scene, "dense", "sparse"),
              os.path.join(root, "work", scene, "sparse"),
              os.path.join(root, "work", scene, "sparse", "0")):
        if os.path.isfile(os.path.join(p, "images.bin")):
            return p
    return None


def scene_name(path):
    return re.sub(r"_registered(\.ply)?$", "", os.path.basename(path),
                  flags=re.IGNORECASE)


def find_original_cloud(merged, data):
    """找配准前的原始点云,用来重现「点云 + 模型」的叠加效果。

    新 JSON 里记了 --cloud;老 JSON 按场景名猜 work\\<场景>\\mvs\\scene_dense.ply。
    """
    p = data.get("cloud")
    if p and os.path.isfile(p):
        return p
    scene = scene_name(merged)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for cand in (os.path.join(root, "work", scene, "mvs", "scene_dense.ply"),
                 os.path.join(root, "work", scene, "dense", "scene_dense.ply")):
        if os.path.isfile(cand):
            return cand
    return None


def find_labels(merged, data):
    p = data.get("labels")
    if p and os.path.isfile(p):
        return p
    cand = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "out", f"{scene_name(merged)}_labels.npy")
    return cand if os.path.isfile(cand) else None


def load_registered_meshes(data):
    """按 JSON 里的 transform_4x4 把模型直接摆到已配准的位置。"""
    try:
        from img_to_3d import load_mesh_with_colors
    except ImportError:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from img_to_3d import load_mesh_with_colors

    meshes = []
    for o in data.get("objects") or []:
        cid = int(o.get("class_id", 1))
        path = o.get("mesh")
        T = o.get("transform_4x4")
        if not path or not os.path.isfile(path) or T is None:
            log(f"⚠ 类别 {cid} 的模型文件不存在,跳过:{path}")
            continue
        mesh = load_mesh_with_colors(path, y_up=True)
        mesh.transform(np.asarray(T, dtype=float))
        meshes.append((cid, mesh))
        log(f"模型已按保存的位姿摆好:类别 {cid}  {os.path.basename(path)}")
    return meshes


def load_scene(merged, data):
    """返回 (显示点云, 模型列表, 是否用了原始点云)。"""
    original = find_original_cloud(merged, data)
    if original:
        cloud = o3d.io.read_point_cloud(original)
        if not cloud.has_points():
            log(f"⚠ 原始点云读不到:{original}")
        else:
            r_up = json_rot_up(data)
            if r_up is not None:
                cloud.rotate(r_up, center=(0.0, 0.0, 0.0))
            else:
                log("⚠ JSON 里没有 up 旋转,原始点云按原样显示(可能不是 Z-up)")

            labels_path = find_labels(merged, data)
            if labels_path:
                labels = np.load(labels_path).reshape(-1)
                if len(labels) == len(cloud.points):
                    class_colors = {
                        int(o.get("class_id", 1)): class_color(int(o.get("class_id", 1)))
                        for o in (data.get("objects") or [])
                    }
                    cols = colored_points(cloud, labels, class_colors,
                                          keep_colors=True)
                    cloud.colors = o3d.utility.Vector3dVector(cols)
                else:
                    log(f"⚠ labels 数量 {len(labels)} 和点云 "
                        f"{len(cloud.points)} 对不上,不染色")
            else:
                log("⚠ 没找到 labels.npy,点云保留原色")

            meshes = load_registered_meshes(data)
            log(f"原始点云:{original}({len(cloud.points)} 点)")
            return cloud, meshes, True

    cloud = o3d.io.read_point_cloud(merged)
    log(f"没找到配准前的原始点云,直接显示合并 ply:{merged}")
    return cloud, [], False


def cameras_from_json(data):
    """返回 (相机列表, 是否带真实朝向)。"""
    raw = data.get("cameras") or []
    if not raw:
        return [], False
    has_dir = all(("forward" in c and "up" in c and "right" in c) for c in raw)
    cams = []
    for c in raw:
        item = {"name": c.get("name", ""),
                "C": np.asarray(c["center"], dtype=float)}
        if has_dir:
            item["forward"] = np.asarray(c["forward"], dtype=float)
            item["up"] = np.asarray(c["up"], dtype=float)
            item["right"] = np.asarray(c["right"], dtype=float)
        cams.append(item)
    return cams, has_dir


def centers_geometry(cameras, radius):
    """没有朝向时,只画黄色小球表示每个拍摄位置。"""
    meshes = []
    for c in cameras:
        s = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=8)
        s.translate(c["C"])
        s.paint_uniform_color(CAM_COLOR)
        meshes.append(s)
    if not meshes:
        return None
    out = meshes[0]
    for s in meshes[1:]:
        out += s
    return out


def show(cloud, meshes, camera_geo, offset_diag, center, front,
         zoom, screenshot, show_window=True, point_size=2.5,
         display_offset=0.0, width=1280, height=800):
    if meshes and display_offset:
        # 和 register_model_to_cloud.py 一样:点云朝相机挪一点,
        # 否则配准好的模型表面会把点云完全盖住。
        # 这里必须用「场景对角线」(含相机),不能用被配准物体的对角线:
        # 杯子只有 1.5,算出来只挪 1.8 cm,点云仍然埋在模型里。
        import copy
        cloud = copy.deepcopy(cloud)
        f = front / np.linalg.norm(front)
        cloud.translate(f * offset_diag * float(display_offset))

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="配准结果", width=width, height=height,
                      visible=show_window)
    vis.add_geometry(cloud)
    for _, m in meshes:
        vis.add_geometry(m)
    if camera_geo is not None:
        vis.add_geometry(camera_geo)

    opt = vis.get_render_option()
    opt.background_color = np.array([0.05, 0.05, 0.06])
    opt.point_size = float(point_size)
    opt.line_width = 2.0
    opt.mesh_show_back_face = True
    opt.light_on = True

    vc = vis.get_view_control()
    vc.set_lookat(center)
    vc.set_up([0.0, 0.0, 1.0])
    vc.set_front(front / np.linalg.norm(front))
    vc.set_zoom(float(zoom))
    for _ in range(3):
        vis.poll_events()
        vis.update_renderer()

    if screenshot:
        parent = os.path.dirname(os.path.abspath(screenshot))
        if parent:
            os.makedirs(parent, exist_ok=True)
        try:
            vis.capture_screen_image(screenshot, do_render=True)
            log(f"截图已保存:{screenshot}")
        except Exception as e:  # noqa: BLE001
            log(f"截图失败({e})")

    if show_window:
        log("窗口已打开;左键旋转,滚轮缩放,Q 退出。")
        vis.run()
    vis.destroy_window()


def main():
    ap = argparse.ArgumentParser(
        description="直接显示 register_model_to_cloud.py 已配准好的结果")
    ap.add_argument("--merged", required=True,
                    help="--out-merged 的 ply,例如 out/cup2_registered.ply")
    ap.add_argument("--registration", default=None,
                    help="--save-transform 的 json;不给就只显示点云")
    ap.add_argument("--colmap-model", dest="colmap_model", default=None,
                    help="COLMAP sparse 目录;JSON 里没有朝向时用来补画四棱锥")
    ap.add_argument("--no-cameras", dest="no_cameras", action="store_true",
                    help="不画相机")
    ap.add_argument("--camera-scale", dest="camera_scale", type=float,
                    default=0.05, help="四棱锥大小(点云对角线的比例),默认 0.05")
    ap.add_argument("--voxel", type=float, default=0.0,
                    help="显示前降采样体素,默认 0=全量;点太多卡顿时设 0.002 之类")
    ap.add_argument("--point-size", dest="point_size", type=float, default=2.5,
                    help="点大小,默认 2.5")
    ap.add_argument("--display-offset", dest="display_offset", type=float,
                    default=0.0,
                    help="有模型时点云朝相机挪动的比例(场景对角线),默认 0=严格原位,"
                         "和 register_model_to_cloud.py 的窗口一致;"
                         "只有模型把点云完全挡住时才试 0.012(纯显示,不改文件)")
    ap.add_argument("--focus", choices=["all", "objects"], default="all",
                    help="取景:all=点云+相机(默认);objects=只看配准物体")
    ap.add_argument("--zoom", type=float, default=None,
                    help="初始缩放,越小离物体越近;默认 all=0.55、objects=0.18")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=800)
    ap.add_argument("--screenshot", default=None, help="截图路径")
    ap.add_argument("--no-show", action="store_true", help="不弹窗,只截图")
    args = ap.parse_args()

    if not os.path.isfile(args.merged):
        raise SystemExit(f"找不到 {args.merged}")

    data = read_json(args.registration)
    cloud, meshes, used_original = load_scene(args.merged, data)
    if not cloud.has_points():
        raise SystemExit(f"读不到点:{args.merged}")
    log(f"显示点云:{len(cloud.points)} 点,颜色={cloud.has_colors()},"
        f"模型={len(meshes)} 个")
    if args.voxel > 0:
        cloud = cloud.voxel_down_sample(args.voxel)
        log(f"显示用降采样:{len(cloud.points)} 点(体素 {args.voxel})")

    # 配准物体自己的范围:决定相机锥大小和「点云朝相机挪」的幅度
    obj_pts = [np.asarray(cloud.points)]
    for _, m in meshes:
        obj_pts.append(np.asarray(m.vertices))
    obj_lo, obj_hi = robust_bounds(np.vstack(obj_pts))
    obj_center = (obj_lo + obj_hi) / 2.0
    obj_diag = float(np.linalg.norm(obj_hi - obj_lo))

    # ---------------- 相机 ----------------
    cameras = []
    camera_geo = None
    if not args.no_cameras:
        cams, has_dir = cameras_from_json(data)
        if cams and not has_dir:
            r_up = json_rot_up(data)
            model = args.colmap_model or find_colmap_model(args.merged, data)
            if r_up is not None and model:
                try:
                    cams = load_cameras(model, r_up)
                    has_dir = True
                    log(f"从 {model} 补全相机朝向:{len(cams)} 个")
                except Exception as e:  # noqa: BLE001
                    log(f"⚠ 读 COLMAP 位姿失败({e}),只画位置")
            else:
                if r_up is None:
                    log("⚠ JSON 里没有 up 旋转,无法恢复相机朝向")
                if not model:
                    log("⚠ 没找到 COLMAP 模型,只画相机位置")
        elif not cams:
            model = args.colmap_model or find_colmap_model(args.merged, data)
            r_up = json_rot_up(data) or ROT_UP
            if model:
                try:
                    cams = load_cameras(model, r_up)
                    has_dir = True
                    log(f"相机位姿:{len(cams)} 个(来自 {model})")
                except Exception as e:  # noqa: BLE001
                    log(f"⚠ 读 COLMAP 位姿失败({e})")
            else:
                log("没有相机信息(可加 --colmap-model work/<场景>/dense/sparse)")

        if cams:
            cameras = cams
            if has_dir:
                camera_geo = camera_lineset(cams, obj_center, obj_diag,
                                            args.camera_scale)
            else:
                camera_geo = centers_geometry(
                    cams, max(obj_diag * 0.006, 1e-4))
                log(f"相机位置:{len(cams)} 个(没有朝向,画成黄色小球)")

    # ---------------- 取景 ----------------
    # 相机环绕物体时「平均方向」会相互抵消,这里固定用 3/4 侧前方;
    # 和 register_model_to_cloud.py 的默认视角一致。
    front = np.array([-0.15, -1.0, 0.12])
    # 用 1-99% 分位防止重建离群点把视野拉飞;相机中心再单独并进来
    lo, hi = obj_lo.copy(), obj_hi.copy()
    if cameras:
        cc = np.asarray([c["C"] for c in cameras])
        lo = np.minimum(lo, cc.min(axis=0))
        hi = np.maximum(hi, cc.max(axis=0))
    all_center = (lo + hi) / 2.0
    offset_diag = float(np.linalg.norm(hi - lo))
    center = obj_center if args.focus == "objects" else all_center
    zoom = (args.zoom if args.zoom is not None
            else (0.18 if args.focus == "objects" else 0.55))

    show(cloud, meshes, camera_geo, offset_diag, center, front,
         zoom, args.screenshot,
         show_window=not args.no_show, point_size=args.point_size,
         display_offset=args.display_offset,
         width=args.width, height=args.height)


if __name__ == "__main__":
    main()
