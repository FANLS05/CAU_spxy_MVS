# -*- coding: utf-8 -*-
import os
import numpy as np
from PIL import Image
from ultralytics import YOLO

ROOT = r"D:\code\point_cloud\0916"
model_path = os.path.join(ROOT, "models", "crack", "levanell", "yolov8n-seg-cracks-joints.pt")
img_path = os.path.join(ROOT, "data", "rock", "Riebo000001.JPG")
outdir = os.path.join(ROOT, "out", "crack_test")
os.makedirs(outdir, exist_ok=True)

model = YOLO(model_path)
print("names:", model.names)

def run(rgb, tag):
    r = model.predict(rgb, imgsz=1024, conf=0.15, retina_masks=True, verbose=False)[0]
    n = 0 if r.masks is None else len(r.masks.data)
    print(f"  {tag}: boxes={len(r.boxes)} masks={n}")
    if r.boxes is not None and len(r.boxes):
        print("    conf:", [round(float(x), 3) for x in r.boxes.conf])
    plot = r.plot()
    Image.fromarray(plot[..., ::-1]).save(os.path.join(outdir, f"yolo_{tag}_overlay.png"))
    if r.masks is not None and len(r.masks.data):
        m = r.masks.data.cpu().numpy().max(axis=0) > 0.5
        over = np.asarray(rgb, np.float32).copy()
        over[m] = over[m] * 0.25 + np.array([255, 20, 20], np.float32) * 0.75
        Image.fromarray(over.astype(np.uint8)).save(os.path.join(outdir, f"yolo_{tag}_mask.png"))
        print(f"    mask frac={m.mean():.4f}")

im = Image.open(img_path).convert("RGB")
W, H = im.size
print("image:", W, H)
run(np.asarray(im.resize((1024, 681), Image.BILINEAR)), "global")
cx, cy = W // 2, H // 2
box = (cx - 512, cy - 340, cx + 512, cy + 340)
run(np.asarray(im.crop(box)), "crop")
print("done")
