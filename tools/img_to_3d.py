# -*- coding: utf-8 -*-
"""
单图生 3D:调腾讯云「混元生3D极速版」API,下载结果并用 Open3D 显示。

接口(腾讯云 API 3.0,产品 ai3d,版本 2025-05-13):
  提交: https://cloud.tencent.com/document/api/1804/123463  SubmitHunyuanTo3DRapidJob
  查询: https://cloud.tencent.com/document/api/1804/123464  QueryHunyuanTo3DRapidJob
  域名: ai3d.tencentcloudapi.com

密钥从项目根目录的 SecretKey.csv 读(两列:SecretId,SecretKey,首行是表头)。
该文件已在 .gitignore 里,不会进仓库。

用法(在 0916 目录下,用 conda basic_env 的 python):

    # 单图 → 3D,自动轮询,完成后 Open3D 打开
    python tools\\img_to_3d.py --image data\\food\\IMG_2430.JPG

    # 文生 3D
    python tools\\img_to_3d.py --prompt "一只小猫"

    # 专业版多视角:--image 是前视图,其余用 --view 补
    python tools\\img_to_3d.py --pro --model-version 3.0 `
      --image out\\sceaux_cutouts\\100_7104.jpg `
      --view left=out\\sceaux_cutouts\\100_7100.jpg `
      --view right=out\\sceaux_cutouts\\100_7110.jpg

    # 按 COLMAP 相机位姿自动挑 front/left/right/back 再提交
    python tools\\img_to_3d.py --pro --colmap-model work\\cup\\dense\\sparse

    # 先看会挑哪几张、不花钱
    python tools\\img_to_3d.py --pro --colmap-model work\\cup\\dense\\sparse --dry-run

    # 只查询已有任务(24 小时内有效);专业版要加 --pro
    python tools\\img_to_3d.py --job-id 1335141824121323520 --pro

    # 不弹窗,只下载 + 存一张渲染图
    python tools\\img_to_3d.py --image ... --no-show --screenshot out.png

产物:
    work\\img2_3d\\<JobId>\\response_submit.json
    work\\img2_3d\\<JobId>\\response_query.json
    work\\img2_3d\\<JobId>\\<模型文件或解压目录>
"""

import argparse
import csv
import hashlib
import hmac
import json
import os
import sys
import time
import zipfile
from datetime import datetime, timezone

import requests

HOST = "ai3d.tencentcloudapi.com"
SERVICE = "ai3d"
VERSION = "2025-05-13"
REGION = "ap-guangzhou"          # 文档示例里的 COS 都在 ap-guangzhou

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_KEY_FILE = os.path.join(PROJECT_ROOT, "SecretKey.csv")
DEFAULT_OUT_ROOT = os.path.join(PROJECT_ROOT, "work", "img2_3d")


# -----------------------------------------------------------------------------
# TC3-HMAC-SHA256 签名(不依赖腾讯云 SDK)
# -----------------------------------------------------------------------------
def load_credentials(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.reader(f) if any(c.strip() for c in r)]
    if len(rows) < 2:
        raise SystemExit(f"密钥文件格式不对:{path}(需要表头 + 一行 SecretId,SecretKey)")
    secret_id, secret_key = rows[1][0].strip(), rows[1][1].strip()
    if not secret_id or not secret_key:
        raise SystemExit(f"密钥文件里 SecretId/SecretKey 为空:{path}")
    return secret_id, secret_key


def _hmac(key, msg):
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def call_api(secret_id, secret_key, action, payload):
    """按 TC3-HMAC-SHA256 签名 POST 到 ai3d.tencentcloudapi.com,返回 Response 字典。"""
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    timestamp = int(time.time())
    date = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%d")
    content_type = "application/json; charset=utf-8"

    canonical_headers = f"content-type:{content_type}\nhost:{HOST}\n"
    signed_headers = "content-type;host"
    hashed_payload = hashlib.sha256(body.encode("utf-8")).hexdigest()
    canonical_request = "\n".join(
        ["POST", "/", "", canonical_headers, signed_headers, hashed_payload]
    )

    credential_scope = f"{date}/{SERVICE}/tc3_request"
    string_to_sign = "\n".join(
        [
            "TC3-HMAC-SHA256",
            str(timestamp),
            credential_scope,
            hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
        ]
    )

    secret_date = _hmac(("TC3" + secret_key).encode("utf-8"), date)
    secret_service = _hmac(secret_date, SERVICE)
    secret_signing = _hmac(secret_service, "tc3_request")
    signature = hmac.new(
        secret_signing, string_to_sign.encode("utf-8"), hashlib.sha256
    ).hexdigest()

    headers = {
        "Authorization": (
            f"TC3-HMAC-SHA256 Credential={secret_id}/{credential_scope}, "
            f"SignedHeaders={signed_headers}, Signature={signature}"
        ),
        "Content-Type": content_type,
        "Host": HOST,
        "X-TC-Action": action,
        "X-TC-Version": VERSION,
        "X-TC-Timestamp": str(timestamp),
        "X-TC-Region": REGION,
    }

    resp = requests.post(
        f"https://{HOST}/", headers=headers, data=body.encode("utf-8"), timeout=60
    )
    resp.raise_for_status()
    data = resp.json()
    result = data.get("Response", {})
    if "Error" in result:
        err = result["Error"]
        raise SystemExit(
            f"API 调用失败:{action}\n"
            f"  Code: {err.get('Code')}\n  Message: {err.get('Message')}\n"
            f"  RequestId: {result.get('RequestId')}"
        )
    return result


# -----------------------------------------------------------------------------
# 提交 / 轮询 / 下载
# -----------------------------------------------------------------------------
def encode_image(path):
    import base64

    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise SystemExit(f"找不到输入图:{path}")
    with open(path, "rb") as f:
        raw = f.read()
    b64 = base64.b64encode(raw).decode("ascii")
    return b64, len(raw), len(b64)


def parse_views(items):
    views = []
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--view 要写成 类型=路径,例如 left=out\\left.jpg;收到:{item}")
        view_type, path = item.split("=", 1)
        view_type = view_type.strip()
        if view_type not in (
            "left", "right", "back", "top", "bottom", "left_front", "right_front"
        ):
            raise SystemExit(f"不认识的视角类型:{view_type}")
        views.append((view_type, path.strip()))
    return views


def resolve_auto_views(args):
    """供 --colmap-model 用:按相机位姿挑图,写回 args.image / args.view。"""
    import colmap_views

    sel = colmap_views.resolve(
        args.colmap_model,
        cutouts=args.cutouts,
        labels=args.labels,
        cloud=args.cloud,
        center=args.center,
        class_id=args.class_id,
        up=args.up,
        view_types=colmap_views.parse_view_types(args.auto_view_types),
        front_image=args.front_image,
        max_error=args.max_view_error,
        swap_left_right=args.swap_left_right,
        strict=args.strict_views,
        min_coverage=args.min_coverage,
    )
    colmap_views.print_selection(sel)
    if args.views_json:
        path = os.path.abspath(args.views_json)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(sel, f, ensure_ascii=False, indent=2)
        print(f"\n选择结果 -> {path}")
    if args.views_preview:
        colmap_views.save_preview(sel, args.views_preview)

    args.image = sel["views"][0]["path"]
    args.view = [f"{v['view_type']}={v['path']}" for v in sel["views"][1:]]
    if not args.pro:
        args.pro = True
        print("\n自动多视角 -> 使用专业版(SubmitHunyuanTo3DProJob)")


def submit_job(args, secret_id, secret_key):
    if getattr(args, "colmap_model", None):
        resolve_auto_views(args)

    pro = args.pro
    action = "SubmitHunyuanTo3DProJob" if pro else "SubmitHunyuanTo3DRapidJob"
    payload = {}

    if pro:
        payload["Model"] = args.model_version
        if args.geometry:
            payload["GenerateType"] = "Geometry"
        if args.format:
            payload["ResultFormat"] = args.format   # pro 只接受 STL/USDZ/FBX
    else:
        payload["ResultFormat"] = args.format or "GLB"
        if args.geometry:
            payload["EnableGeometry"] = True
            payload["ResultFormat"] = "GLB"        # 白模不支持 OBJ

    if args.pbr and not (pro and args.geometry):
        payload["EnablePBR"] = True

    main_b64 = None
    if args.image:
        main_b64, raw_size, b64_size = encode_image(args.image)
        payload["ImageBase64"] = main_b64
        print(f"主视图:{os.path.abspath(args.image)}  "
              f"({raw_size / 1e6:.2f} MB, base64 {b64_size / 1e6:.2f} MB)")
    elif args.prompt:
        payload["Prompt"] = args.prompt
        print(f"提示词:{args.prompt}")
    else:
        raise SystemExit("需要 --image 或 --prompt 之一")

    views = parse_views(args.view)
    if views and not pro:
        raise SystemExit("多视角只有专业版支持,加 --pro")
    if views:
        if args.model_version != "3.1":
            extra = {"top", "bottom", "left_front", "right_front"}
            bad = [t for t, _ in views if t in extra]
            if bad:
                raise SystemExit(f"{bad} 只有 Model=3.1 支持,加 --model-version 3.1")
        total_b64 = len(main_b64 or "")
        multiview = []
        for view_type, path in views:
            b64, raw_size, b64_size = encode_image(path)
            total_b64 += b64_size
            multiview.append({"ViewType": view_type, "ViewImageBase64": b64})
            print(f"  {view_type:11s}:{os.path.abspath(path)}  "
                  f"({raw_size / 1e6:.2f} MB, base64 {b64_size / 1e6:.2f} MB)")
        if total_b64 > 6 * 1024 * 1024:
            raise SystemExit(
                f"所有图片 base64 合计 {total_b64 / 1e6:.2f} MB,超过专业版 6 MB 上限;"
                "把 --size 调小或提高 JPEG 压缩"
            )
        payload["MultiViewImages"] = multiview
        print(f"多视角合计 base64:{total_b64 / 1e6:.2f} MB")

    if getattr(args, "dry_run", False):
        print(f"dry-run:不提交。Action={action},主图 1 张 + 附加视角 "
              f"{len(views)} 个,payload 键:{', '.join(payload)}")
        return None, None

    print(f"提交任务 {action} ...")
    result = call_api(secret_id, secret_key, action, payload)
    return result["JobId"], result


def query_job(job_id, secret_id, secret_key, pro=False):
    action = "QueryHunyuanTo3DProJob" if pro else "QueryHunyuanTo3DRapidJob"
    return call_api(secret_id, secret_key, action, {"JobId": job_id})


def wait_job(job_id, secret_id, secret_key, timeout, interval, pro=False):
    if timeout <= 0:
        return None
    started = time.time()
    last = None
    while True:
        result = query_job(job_id, secret_id, secret_key, pro=pro)
        status = result.get("Status", "?")
        if status != last:
            print(f"[{time.strftime('%H:%M:%S')}] 状态:{status}")
            last = status
        if status == "DONE":
            return result
        if status == "FAIL":
            raise SystemExit(
                f"任务失败:{result.get('ErrorCode')} {result.get('ErrorMessage')}"
            )
        if time.time() - started > timeout:
            raise SystemExit(f"等待超时({timeout}s),任务仍在 {status}。JobId={job_id}")
        time.sleep(interval)


def download_results(result, out_dir):
    """下载 ResultFile3Ds;zip 自动解压,返回下载后的本地模型文件列表。"""
    os.makedirs(out_dir, exist_ok=True)
    local_files = []
    for i, item in enumerate(result.get("ResultFile3Ds") or []):
        url = item.get("Url")
        if not url:
            continue
        name = os.path.basename(url.split("?")[0]) or f"result_{i}"
        path = os.path.join(out_dir, name)
        print(f"下载 {item.get('Type', '?')}:{url}")
        with requests.get(url, stream=True, timeout=300) as r:
            r.raise_for_status()
            with open(path, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        print(f"  -> {path}  ({os.path.getsize(path) / 1e6:.2f} MB)")
        local_files.append(path)

        if zipfile.is_zipfile(path):
            extract_dir = os.path.splitext(path)[0]
            os.makedirs(extract_dir, exist_ok=True)
            with zipfile.ZipFile(path) as z:
                z.extractall(extract_dir)
            print(f"  解压到 {extract_dir}")
            for root, _, names in os.walk(extract_dir):
                for n in names:
                    if n.lower().endswith((".glb", ".gltf", ".obj", ".stl", ".ply")):
                        local_files.append(os.path.join(root, n))
    return local_files


# -----------------------------------------------------------------------------
# Open3D 显示(贴图采样成顶点色,Open3D 才看得见)
# -----------------------------------------------------------------------------
def load_mesh_with_colors(path, y_up=True):
    import numpy as np
    import open3d as o3d
    import trimesh

    print(f"加载模型:{path}")
    mesh = None
    try:
        loaded = trimesh.load(path, force="scene")
        if isinstance(loaded, trimesh.Scene):
            if hasattr(loaded, "to_geometry"):
                mesh = loaded.to_geometry()
            else:  # trimesh < 4.0
                mesh = loaded.dump(concatenate=True)
        else:
            mesh = loaded
    except Exception as e:  # noqa: BLE001
        print(f"  trimesh 加载失败({e}),改用 Open3D 直接读")
        fallback = o3d.io.read_triangle_mesh(path)
        if y_up:
            fallback.rotate(
                o3d.geometry.get_rotation_matrix_from_xyz((np.pi / 2, 0.0, 0.0)),
                center=(0.0, 0.0, 0.0),
            )
        fallback.compute_vertex_normals()
        return fallback

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    print(f"  顶点 {len(verts)},三角形 {len(faces)}")

    colors = None
    try:
        visual = mesh.visual
        # TextureVisuals -> 采样贴图;ColorVisuals -> 直接用
        if getattr(visual, "kind", None) == "texture":
            visual = visual.to_color()
        vc = getattr(visual, "vertex_colors", None)
        if vc is not None and len(vc) == len(verts):
            candidate = np.asarray(vc)[:, :3].astype(np.float64) / 255.0
            # 有些模型的贴图按面存,逐顶点采样会全黑;全黑就不如白模
            if candidate.mean() > 0.03:
                colors = candidate
            else:
                print("  顶点色几乎全黑(贴图采样失败),改用白模显示")
    except Exception as e:  # noqa: BLE001
        print(f"  取顶点色失败({e}),显示纯白模型")

    o3d_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(verts),
        triangles=o3d.utility.Vector3iVector(faces),
    )
    if colors is not None:
        o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    else:
        o3d_mesh.paint_uniform_color([0.8, 0.8, 0.8])
    if y_up:
        # 混元的结果是 Y 轴朝上(glTF 惯例),转到 Open3D 习惯的 Z 轴朝上
        o3d_mesh.rotate(
            o3d.geometry.get_rotation_matrix_from_xyz((np.pi / 2, 0.0, 0.0)),
            center=(0.0, 0.0, 0.0),
        )
    o3d_mesh.compute_vertex_normals()
    return o3d_mesh


def show(mesh, screenshot=None, show_window=True):
    """用旧版 Visualizer(OpenGL)显示/截图。

    不用 Filament 的 OffscreenRenderer:实测这张 47 万面的混元模型会让它整帧
    输出全黑(连背景都没了),换成 OpenGL 的 Visualizer 就正常。
    """
    import numpy as np
    import open3d as o3d

    mesh.compute_vertex_normals()
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="混元生3D - 单图生3D",
        width=1280, height=800,
        visible=show_window,
    )
    vis.add_geometry(mesh)

    opt = vis.get_render_option()
    opt.background_color = np.array([0.08, 0.09, 0.1])
    opt.mesh_show_back_face = True
    opt.light_on = True

    # 相机:从上、右前方斜看
    bbox = mesh.get_axis_aligned_bounding_box()
    center = np.asarray(bbox.get_center())
    direction = np.array([0.9, -1.0, 0.6])
    direction /= np.linalg.norm(direction)
    vc = vis.get_view_control()
    vc.set_lookat(center)
    vc.set_up([0.0, 0.0, 1.0])
    vc.set_front(-direction)
    vc.set_zoom(0.7)

    for _ in range(3):
        vis.poll_events()
        vis.update_renderer()

    if screenshot:
        try:
            vis.capture_screen_image(screenshot, do_render=True)
            print(f"渲染图已保存:{screenshot}")
        except Exception as e:  # noqa: BLE001
            print(f"截图失败({e})")

    if show_window:
        print("Open3D 窗口已打开;鼠标左键拖动旋转,滚轮缩放,按 Q 退出。")
        vis.run()
    vis.destroy_window()


# -----------------------------------------------------------------------------
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
    parser = argparse.ArgumentParser(description="腾讯混元单图/文生 3D + Open3D 显示")
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--image", help="输入图片路径(jpg/png/jpeg/webp,单边 128~5000,≤6MB)")
    src.add_argument("--prompt", help="文生 3D 提示词(中文,≤200 字)")
    src.add_argument("--job-id", help="查询已有任务,不再提交")
    src.add_argument("--open", dest="mesh_file", help="不调 API,直接打开本地 glb/obj 看效果")
    src.add_argument("--colmap-model",
                     help="COLMAP sparse 目录;按相机位姿自动选 front/left/right/back 后提交")
    parser.add_argument("--pro", action="store_true",
                        help="用专业版(SubmitHunyuanTo3DProJob),支持 --view 多视角")
    parser.add_argument("--model-version", default="3.0", choices=["3.0", "3.1"],
                        help="专业版模型版本,默认 3.0;3.1 才支持 top/bottom/斜前视角")
    parser.add_argument("--view", action="append", metavar="TYPE=PATH",
                        help="专业版附加视角,可重复:left/right/back(3.1 还支持 "
                             "top/bottom/left_front/right_front)。--image 是主(前)视图")
    # ---- --colmap-model 自动选视角 ----
    parser.add_argument("--cutouts", help="抠图目录;默认 out\\<场景>_cutouts")
    parser.add_argument("--labels", help="每点类别 .npy;默认 out\\<场景>_labels.npy")
    parser.add_argument("--cloud", help="点云 .ply;默认 <场景>\\mvs\\scene_dense.ply")
    parser.add_argument("--center", help="物体中心 x,y,z;默认按 labels 质心")
    parser.add_argument("--class-id", type=int, default=1, help="目标类别 id,默认 1")
    parser.add_argument("--up", default="auto", help="auto / camera / plane / x,y,z")
    parser.add_argument("--auto-view-types", default="front,left,right,back",
                        help="自动选视角的类型,逗号分隔,front 总会包含")
    parser.add_argument("--front-image", help="指定前视图文件名;默认自动搜最优")
    parser.add_argument("--max-view-error", type=float, default=45.0,
                        help="最大可接受方位误差(度),默认 45")
    parser.add_argument("--strict-views", action="store_true",
                        help="方位误差超过 --max-view-error 时直接失败")
    parser.add_argument("--swap-left-right", action="store_true",
                        help="交换 left/right(左右约定相反时用)")
    parser.add_argument("--min-coverage", type=float, default=0.0,
                        help="mask 覆盖率低于此值的图不参与,0.01 = 1%%")
    parser.add_argument("--views-json", help="把自动选图结果写到 json")
    parser.add_argument("--views-preview", help="把选中的图拼成一张预览图")
    parser.add_argument("--dry-run", action="store_true",
                        help="只选图 + 编码,不调用 API")
    parser.add_argument("--format", default=None,
                        choices=["OBJ", "GLB", "STL", "USDZ", "FBX", "MP4"],
                        help="极速版默认 GLB;专业版默认返回 obj+glb,只有 STL/USDZ/FBX 需要显式指定")
    parser.add_argument("--pbr", action="store_true", help="开启 PBR 材质")
    parser.add_argument("--geometry", action="store_true", help="只要白模(无纹理)")
    parser.add_argument("--key-file", default=DEFAULT_KEY_FILE, help="SecretKey.csv 路径")
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT, help="产物根目录")
    parser.add_argument("--timeout", type=int, default=900, help="轮询超时秒数,0=不等待")
    parser.add_argument("--interval", type=int, default=5, help="轮询间隔秒数")
    parser.add_argument("--no-show", action="store_true", help="不弹 Open3D 窗口")
    parser.add_argument("--screenshot", help="保存一张离屏渲染图到指定路径")
    args = parser.parse_args(_fix_center_argv(sys.argv[1:]))

    if args.mesh_file:
        mesh = load_mesh_with_colors(args.mesh_file)
        show(mesh, args.screenshot, show_window=not args.no_show)
        return

    if args.dry_run:
        secret_id, secret_key = None, None
    else:
        secret_id, secret_key = load_credentials(args.key_file)

    if args.job_id:
        job_id = args.job_id
        submit_result = None
        print(f"查询已有任务:{job_id}")
    else:
        job_id, submit_result = submit_job(args, secret_id, secret_key)
        if job_id is None:
            print("dry-run 结束,没有提交任务。")
            return
        print(f"任务已提交,JobId = {job_id}")

    out_dir = os.path.join(args.out_root, job_id)
    os.makedirs(out_dir, exist_ok=True)
    if submit_result is not None:
        with open(os.path.join(out_dir, "response_submit.json"), "w", encoding="utf-8") as f:
            json.dump(submit_result, f, ensure_ascii=False, indent=2)

    result = wait_job(job_id, secret_id, secret_key, args.timeout, args.interval, pro=args.pro)
    if result is None:
        print("已提交但不等待;之后用 --job-id 查询。")
        return

    with open(os.path.join(out_dir, "response_query.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    files = download_results(result, out_dir)
    if not files:
        raise SystemExit("任务成功但没有可下载/可显示的模型文件")
    # 优先 GLB:专业版会同时给 obj + glb,zip 解压后列表顺序里 OBJ 常常在前
    def _rank(p):
        low = p.lower()
        for i, ext in enumerate((".glb", ".gltf", ".obj", ".ply", ".stl")):
            if low.endswith(ext):
                return i
        return 99

    preferred = sorted((p for p in files if _rank(p) < 99), key=_rank)
    mesh_path = (preferred or files)[0]
    print(f"用于显示:{mesh_path}")

    mesh = load_mesh_with_colors(mesh_path)
    show(mesh, args.screenshot, show_window=not args.no_show)


if __name__ == "__main__":
    main()
