# -*- coding: utf-8 -*-
import os
import numpy as np
from PIL import Image, ImageDraw
import onnxruntime as ort

ROOT = r"D:\code\point_cloud\0916"
outdir = os.path.join(ROOT, "out", "crack_test")
os.makedirs(outdir, exist_ok=True)
imgs = ["Riebo000001.JPG", "Riebo000020.JPG", "Riebo000045.JPG", "Riebo000070.JPG"]

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

tom = ort.InferenceSession(os.path.join(ROOT, "models", "crack", "tomt996_seed101.onnx"),
                           providers=["CPUExecutionProvider"])
dih = ort.InferenceSession(os.path.join(ROOT, "models", "crack", "dihao", "crackevo_unet_r34_A.onnx"),
                           providers=["CPUExecutionProvider"])
T_IN = tom.get_inputs()[0].name
D_IN = dih.get_inputs()[0].name

def tom_prob(rgb):
    im = Image.fromarray(rgb).resize((672, 378), Image.BILINEAR)
    x = (np.asarray(im, np.float32) / 255.0 - MEAN) / STD
    x = np.pad(x.transpose(2, 0, 1)[None], ((0,0),(0,0),(0,6),(0,0)), mode="reflect")
    logit = tom.run(None, {T_IN: x})[0][:, :, :378]
    return (1.0 / (1.0 + np.exp(-logit)))[0, 0]

def dih_prob(rgb, tile=512, stride=384):
    h, w = rgb.shape[:2]
    acc = np.zeros((h, w), np.float32); wt = np.zeros((h, w), np.float32)
    ys = list(range(0, max(h-tile,0)+1, stride)); xs = list(range(0, max(w-tile,0)+1, stride))
    if ys[-1] != h-tile: ys.append(h-tile)
    if xs[-1] != w-tile: xs.append(w-tile)
    for y in ys:
        for x in xs:
            p = rgb[y:y+tile, x:x+tile]
            ph, pw = p.shape[:2]
            if ph < tile or pw < tile:
                p = np.pad(p, ((0,tile-ph),(0,tile-pw),(0,0)), mode="reflect")
            xx = (p.astype(np.float32)/255.0 - MEAN)/STD
            logit = dih.run(None, {D_IN: xx.transpose(2,0,1)[None].astype(np.float32)})[0][0,0]
            acc[y:y+ph, x:x+pw] += logit[:ph,:pw]; wt[y:y+ph, x:x+pw] += 1
    return 1.0/(1.0+np.exp(-acc/np.maximum(wt,1e-6)))

def overlay(rgb, prob, thr, color):
    over = rgb.astype(np.float32).copy(); m = prob > thr
    over[m] = over[m]*0.25 + np.array(color, np.float32)*0.75
    return over.astype(np.uint8), m.mean()

rows = []
for name in imgs:
    im = Image.open(os.path.join(ROOT, "data", "rock", name)).convert("RGB")
    g = np.asarray(im.resize((1024, 681), Image.BILINEAR))
    pt, pd = tom_prob(g), dih_prob(g)
    pt = np.asarray(Image.fromarray(pt).resize((pd.shape[1], pd.shape[0]), Image.BILINEAR))
    pe = (pt + pd) / 2
    o1, f1 = overlay(g, pt, 0.5, [255,20,20])
    o2, f2 = overlay(g, pd, 0.3, [255,20,20])
    o3, f3 = overlay(g, pe, 0.4, [255,20,20])
    row = np.concatenate([o1, o2, o3], axis=1)
    d = ImageDraw.Draw(Image.fromarray(row))
    rows.append(row)
    print(f"{name}: tom={f1:.4f} dih={f2:.4f} ens={f3:.4f}")

grid = np.concatenate(rows, axis=0)
Image.fromarray(grid).resize((grid.shape[1]//2, grid.shape[0]//2), Image.LANCZOS).save(
    os.path.join(outdir, "compare_grid.jpg"), quality=90)
print("saved compare_grid.jpg  (列: Tomt996 | Dihao | 平均)")
