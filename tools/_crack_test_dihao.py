# -*- coding: utf-8 -*-
import os
import numpy as np
from PIL import Image
import onnxruntime as ort

ROOT = r"D:\code\point_cloud\0916"
model = os.path.join(ROOT, "models", "crack", "dihao", "crackevo_unet_r34_A.onnx")
img_path = os.path.join(ROOT, "data", "rock", "Riebo000001.JPG")
outdir = os.path.join(ROOT, "out", "crack_test")
os.makedirs(outdir, exist_ok=True)

sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
print("inputs:", [(i.name, i.shape, i.type) for i in sess.get_inputs()])
print("outputs:", [(o.name, o.shape, o.type) for o in sess.get_outputs()])
iname = sess.get_inputs()[0].name
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

def infer_tile(rgb):
    x = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
    x = x.transpose(2, 0, 1)[None].astype(np.float32)
    return sess.run(None, {iname: x})[0][:, 0]

def predict(rgb, tile=512, stride=384):
    h, w = rgb.shape[:2]
    acc = np.zeros((h, w), np.float32)
    wt = np.zeros((h, w), np.float32)
    ys = list(range(0, max(h - tile, 0) + 1, stride))
    xs = list(range(0, max(w - tile, 0) + 1, stride))
    if ys[-1] != h - tile: ys.append(h - tile)
    if xs[-1] != w - tile: xs.append(w - tile)
    for y in ys:
        for x in xs:
            yy, xx = max(y, 0), max(x, 0)
            patch = rgb[yy:yy+tile, xx:xx+tile]
            ph, pw = patch.shape[:2]
            if ph < tile or pw < tile:
                patch = np.pad(patch, ((0, tile-ph), (0, tile-pw), (0, 0)), mode="reflect")
            logit = infer_tile(patch)[0]
            acc[yy:yy+ph, xx:xx+pw] += logit[:ph, :pw]
            wt[yy:yy+ph, xx:xx+pw] += 1.0
    return 1.0 / (1.0 + np.exp(-acc / np.maximum(wt, 1e-6)))

def save(prob, rgb, tag):
    print(f"  {tag}: max={prob.max():.3f} mean={prob.mean():.4f} frac>0.3={(prob>0.3).mean():.4f}")
    Image.fromarray((np.clip(prob,0,1)*255).astype(np.uint8)).save(os.path.join(outdir, f"dihao_{tag}_prob.png"))
    over = np.asarray(rgb, np.float32).copy(); m = prob > 0.3
    over[m] = over[m]*0.25 + np.array([255,20,20], np.float32)*0.75
    Image.fromarray(over.astype(np.uint8)).save(os.path.join(outdir, f"dihao_{tag}_overlay.png"))

im = Image.open(img_path).convert("RGB")
W, H = im.size
print("image:", W, H)
run_global = np.asarray(im.resize((1024, 681), Image.BILINEAR))
save(predict(run_global), run_global, "global")
cx, cy = W//2, H//2
box = (cx-512, cy-340, cx+512, cy+340)
crop = np.asarray(im.crop(box))
save(predict(crop), crop, "crop")
print("done")
