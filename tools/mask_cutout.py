# -*- coding: utf-8 -*-
"""
把 SAM3 的索引 mask 变成「白底抠图」,喂给混元生3D。

    work\\sceaux\\dense\\images\\100_7104.JPG   (原图)
    masks_sceaux\\100_7104.png                  (SAM3 索引图:0=背景,1=building)
                    │
                    ▼
    out\\sceaux_cutouts\\100_7104.jpg            (白底、裁到物体、补成正方形、统一尺寸)

为什么要做这一步:
  - 混元生3D 希望输入是「一个完整物体 + 干净背景」,不是整张带天空、
    地面、树木的照片;
  - 多视角输入要求各图里**物体的占比接近**,所以统一裁到 mask 外接框,
    再留同样的边距、补成正方形、缩到同一尺寸;
  - 背景用纯白(也可 --bg 改),比透明 PNG 稳,文档只承诺 jpg/png。

用法(0916 目录下,basic_env):

    # 抠出所有 mask 对应的图
    python tools\\mask_cutout.py `
      --images work\\sceaux\\dense\\images `
      --masks masks_sceaux `
      --out out\\sceaux_cutouts

    # 只要某几张
    python tools\\mask_cutout.py --images ... --masks ... --out ... `
      --only 100_7100 100_7104 100_7110

    # 改尺寸 / 背景 / 边距
    python tools\\mask_cutout.py ... --size 1024 --pad 0.08 --bg 255,255,255

输出:
    <out>\\<stem>.jpg        白底抠图
    <out>\\<stem>_mask.png   同一裁剪下的 mask(检查用)
    <out>\\report.json       每张图的 bbox、像素占比、是否贴边
"""

import argparse
import json
import os

import numpy as np
from PIL import Image

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def find_image(images_dir, stem):
    for ext in IMG_EXTS:
        p = os.path.join(images_dir, stem + ext)
        if os.path.isfile(p):
            return p
    # 大小写不敏感地再找一遍(Windows 上一般不需要,但数据集扩展名不统一)
    low = stem.lower()
    for name in os.listdir(images_dir):
        if os.path.splitext(name)[0].lower() == low and name.lower().endswith(IMG_EXTS):
            return os.path.join(images_dir, name)
    return None


def parse_bg(text):
    parts = [int(v) for v in text.split(",")]
    if len(parts) != 3 or any(v < 0 or v > 255 for v in parts):
        raise SystemExit("--bg 要写成 r,g,b,例如 255,255,255")
    return tuple(parts)


def make_square(img, bg):
    h, w = img.shape[:2]
    side = max(h, w)
    canvas = np.full((side, side, 3), bg, dtype=np.uint8)
    oy, ox = (side - h) // 2, (side - w) // 2
    canvas[oy:oy + h, ox:ox + w] = img
    return canvas


def process_one(image_path, mask_path, class_id, size, pad, bg):
    rgb = np.array(Image.open(image_path).convert("RGB"))
    mask = np.array(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[:, :, 0]

    if rgb.shape[:2] != mask.shape[:2]:
        raise SystemExit(
            f"尺寸对不上:{image_path} {rgb.shape[:2]} vs {mask_path} {mask.shape[:2]}"
        )

    sel = mask == class_id
    n = int(sel.sum())
    if n == 0:
        return None

    ys, xs = np.where(sel)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    h, w = y1 - y0 + 1, x1 - x0 + 1

    # 按物体外接框的边长留边距,再裁
    margin = int(round(pad * max(h, w)))
    cy0 = max(0, y0 - margin)
    cy1 = min(rgb.shape[0], y1 + 1 + margin)
    cx0 = max(0, x0 - margin)
    cx1 = min(rgb.shape[1], x1 + 1 + margin)

    crop = rgb[cy0:cy1, cx0:cx1]
    crop_mask = sel[cy0:cy1, cx0:cx1]

    # 只保留 mask 内的像素,其余涂成背景色
    out = np.full_like(crop, bg)
    out[crop_mask] = crop[crop_mask]
    out = make_square(out, bg)

    pil = Image.fromarray(out).resize((size, size), Image.LANCZOS)

    mask_vis = np.zeros((crop_mask.shape[0], crop_mask.shape[1], 3), dtype=np.uint8)
    mask_vis[crop_mask] = (0, 255, 255)
    mask_sq = make_square(mask_vis, (0, 0, 0))
    mask_pil = Image.fromarray(mask_sq).resize((size, size), Image.NEAREST)

    return {
        "pil": pil,
        "mask_pil": mask_pil,
        "bbox": [int(cx0), int(cy0), int(cx1), int(cy1)],
        "coverage": n / (rgb.shape[0] * rgb.shape[1]),
        "touches_border": bool(x0 == 0 or y0 == 0 or x1 == rgb.shape[1] - 1 or y1 == rgb.shape[0] - 1),
    }


def main():
    ap = argparse.ArgumentParser(description="SAM3 mask -> 白底抠图(混元生3D 输入)")
    ap.add_argument("--images", required=True, help="原图目录")
    ap.add_argument("--masks", required=True, help="SAM3 索引 mask 目录")
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--class-id", type=int, default=1, help="要抠的类别 id,默认 1")
    ap.add_argument("--size", type=int, default=1024, help="输出正方形边长,默认 1024")
    ap.add_argument("--pad", type=float, default=0.08, help="外接框四周留白比例,默认 0.08")
    ap.add_argument("--bg", default="255,255,255", help="背景色 r,g,b,默认纯白")
    ap.add_argument("--quality", type=int, default=92, help="JPEG 质量,默认 92")
    ap.add_argument("--only", nargs="*", help="只处理这些 stem")
    ap.add_argument("--overwrite", action="store_true", help="覆盖已有输出")
    args = ap.parse_args()

    bg = parse_bg(args.bg)
    os.makedirs(args.out, exist_ok=True)

    masks = sorted(f for f in os.listdir(args.masks) if f.lower().endswith(".png"))
    if args.only:
        wanted = {s.lower() for s in args.only}
        masks = [f for f in masks if os.path.splitext(f)[0].lower() in wanted]

    report = []
    skipped = []
    for name in masks:
        stem = os.path.splitext(name)[0]
        out_jpg = os.path.join(args.out, stem + ".jpg")
        if os.path.exists(out_jpg) and not args.overwrite:
            print(f"跳过(已存在):{out_jpg}")
            continue

        image_path = find_image(args.images, stem)
        if image_path is None:
            skipped.append({"stem": stem, "reason": "找不到原图"})
            print(f"⚠ 找不到原图:{stem}")
            continue

        item = process_one(
            image_path, os.path.join(args.masks, name),
            args.class_id, args.size, args.pad, bg,
        )
        if item is None:
            skipped.append({"stem": stem, "reason": f"mask 里没有类别 {args.class_id}"})
            print(f"⚠ {stem}:mask 里没有类别 {args.class_id},跳过")
            continue

        item["pil"].save(out_jpg, quality=args.quality, subsampling=0)
        item["mask_pil"].save(os.path.join(args.out, stem + "_mask.png"))
        size_kb = os.path.getsize(out_jpg) / 1024

        report.append({
            "stem": stem,
            "image": os.path.abspath(image_path),
            "mask": os.path.abspath(os.path.join(args.masks, name)),
            "output": os.path.abspath(out_jpg),
            "bbox": item["bbox"],
            "coverage": round(item["coverage"], 5),
            "touches_border": item["touches_border"],
            "output_kb": round(size_kb, 1),
        })
        flag = "  ← 贴边,物体可能被裁掉" if item["touches_border"] else ""
        print(f"{stem:16s} bbox={item['bbox']} 占比={item['coverage']:.1%} "
              f"{size_kb:6.1f} KB{flag}")

    with open(os.path.join(args.out, "report.json"), "w", encoding="utf-8") as f:
        json.dump({"outputs": report, "skipped": skipped}, f, ensure_ascii=False, indent=2)

    print(f"\n完成 {len(report)} 张 → {os.path.abspath(args.out)}")
    if skipped:
        print(f"跳过 {len(skipped)} 张,详见 report.json")


if __name__ == "__main__":
    main()
