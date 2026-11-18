# -*- coding: utf-8 -*-
import os, sys, time
import numpy as np
from PIL import Image
import onnxruntime as ort

ROOT = r"D:\code\point_cloud\0916"
which = sys.argv[1] if len(sys.argv) > 1 else "tom"
img_path = os.path.join(ROOT, "work", "rock", "dense", "images", "Riebo000001.JPG")
outdir = os.path.join(ROOT, "out", "crack_test")
os.makedirs(outdir, exist_ok=True)

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

if which == "tom":
    model = os.path.join(ROOT, "models", "crack", "tomt996_seed101.onnx")
    tile_h, tile_w, stride = 378, 672, 504
    sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    def infer(p):
        im = Image.fromarray(p).resize((672, 378), Image.BILINEAR)
        x = (np.asarray(im, np.float32)/255.0 - MEAN)/STD
        x = np.pad(x.transpose(2,0,1)[None], ((0,0),(0,0),(0,6),(0,0)), mode="reflect")
        logit = sess.run(None, {iname: x})[0][:, :, :378]
        return (1.0/(1.0+np.exp(-logit)))[0,0]
else:
    model = os.path.join(ROOT, "models", "crack", "dihao", "crackevo_unet_r34_A.onnx")
    tile_h = tile_w = 512; stride = 384
    sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    def infer(p):
        x = (p.astype(np.float32)/255.0 - MEAN)/STD
        logit = sess.run(None, {iname: x.transpose(2,0,1)[None].astype(np.float32)})[0][0,0]
        return 1.0/(1.0+np.exp(-logit))

def hann2d(h, w):
    y = np.hanning(h+2)[1:-1].astype(np.float32)
    x = np.hanning(w+2)[1:-1].astype(np.float32)
    return np.maximum(np.outer(y, x), 1e-3)

rgb = np.asarray(Image.open(img_path).convert("RGB"))
H, W = rgb.shape[:2]
print(f"{which}: image {W}x{H}, tile {tile_w}x{tile_h}, stride {stride}")

ys = list(range(0, max(H-tile_h,0)+1, stride))
xs = list(range(0, max(W-tile_w,0)+1, stride))
if ys[-1] != H-tile_h: ys.append(H-tile_h)
if xs[-1] != W-tile_w: xs.append(W-tile_w)
print(f"tiles: {len(xs)} x {len(ys)} = {len(xs)*len(ys)}")

acc = np.zeros((H, W), np.float32); wsum = np.zeros((H, W), np.float32)
wgt = hann2d(tile_h, tile_w)
t0 = time.time()
for iy, y in enumerate(ys):
    for x in xs:
        p = rgb[y:y+tile_h, x:x+tile_w]
        prob = infer(p)
        acc[y:y+tile_h, x:x+tile_w] += prob * wgt
        wsum[y:y+tile_h, x:x+tile_w] += wgt
    print(f"  row {iy+1}/{len(ys)} elapsed {time.time()-t0:.1f}s", end="\r", flush=True)
prob = acc / np.maximum(wsum, 1e-6)
print(f"\ndone in {time.time()-t0:.1f}s  max={prob.max():.3f} mean={prob.mean():.4f} frac>0.5={(prob>0.5).mean():.4f}")
Image.fromarray((np.clip(prob,0,1)*255).astype(np.uint8)).save(os.path.join(outdir, f"full_{which}_prob.png"))
np.save(os.path.join(outdir, f"full_{which}_prob_sample.npy"), prob[3000:3500, 4500:5000])

# center crop overlay for visual inspection
cx, cy = W//2, H//2
box = (cx-512, cy-340, cx+512, cy+340)
c = rgb[cy-340:cy+340, cx-512:cx+512]
pc = prob[cy-340:cy+340, cx-512:cx+512]
over = c.astype(np.float32).copy(); m = pc > 0.5
over[m] = over[m]*0.25 + np.array([255,20,20], np.float32)*0.75
Image.fromarray(over.astype(np.uint8)).save(os.path.join(outdir, f"full_{which}_crop_overlay.png"))
print("saved")
