# -*- coding: utf-8 -*-
import os
import numpy as np
from PIL import Image
import onnxruntime as ort

ROOT = r"D:\code\point_cloud\0916"
model = os.path.join(ROOT, "models", "crack", "tomt996_seed101.onnx")
img_path = os.path.join(ROOT, "data", "rock", "Riebo000001.JPG")
outdir = os.path.join(ROOT, "out", "crack_test")
os.makedirs(outdir, exist_ok=True)

sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
print("inputs:", [(i.name, i.shape, i.type) for i in sess.get_inputs()])
print("outputs:", [(o.name, o.shape, o.type) for o in sess.get_outputs()])
iname = sess.get_inputs()[0].name

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

def infer(rgb):
    im = Image.fromarray(rgb).resize((672, 378), Image.BILINEAR)
    x = np.asarray(im, np.float32) / 255.0
    x = (x - MEAN) / STD
    x = np.transpose(x, (2, 0, 1))[None]
    x = np.pad(x, ((0, 0), (0, 0), (0, 384 - 378), (0, 0)), mode="reflect")
    logits = sess.run(None, {iname: x})[0][:, :, :378]
    return 1.0 / (1.0 + np.exp(-logits))[0, 0]

def save(prob, rgb, tag):
    h, w = prob.shape
    Image.fromarray((np.clip(prob, 0, 1) * 255).astype(np.uint8)).save(
        os.path.join(outdir, f"tomt996_{tag}_prob.png"))
    over = np.asarray(rgb, np.float32).copy()
    m = prob > 0.5
    over[m] = over[m] * 0.25 + np.array([255, 20, 20], np.float32) * 0.75
    Image.fromarray(over.astype(np.uint8)).save(
        os.path.join(outdir, f"tomt996_{tag}_overlay.png"))
    print(f"  {tag}: shape={prob.shape} max={prob.max():.3f} mean={prob.mean():.4f} frac>0.5={m.mean():.4f}")

im = Image.open(img_path).convert("RGB")
W, H = im.size
print("image:", W, H)

# 1) 整图缩到 672 宽，看全局能不能检出
g = np.asarray(im.resize((672, 378), Image.BILINEAR))
p = infer(g)
save(p, g, "global")

# 2) 原分辨率中心裁 672x378，看细缝
cx, cy = W // 2, H // 2
box = (cx - 336, cy - 189, cx + 336, cy + 189)
c = np.asarray(im.crop(box))
p2 = infer(c)
save(p2, c, "crop")

print("saved to", outdir)
