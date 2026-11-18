# -*- coding: utf-8 -*-
import os, sys, time
import numpy as np
from PIL import Image

ROOT = r"D:\code\point_cloud\0916"
mdir = os.path.join(ROOT, "models", "crack", "thaki")
sys.path.insert(0, mdir)
from crackseg import CrackSeg

img_path = os.path.join(ROOT, "data", "rock", "Riebo000001.JPG")
outdir = os.path.join(ROOT, "out", "crack_test")
os.makedirs(outdir, exist_ok=True)

m = CrackSeg(mdir)
print("device:", m.dev)

def run(rgb, tag, thr):
    t0 = time.time()
    prob = m.prob(rgb)
    msk = prob > thr
    print(f"  {tag}: {time.time()-t0:.1f}s max={prob.max():.3f} mean={prob.mean():.4f} frac>{thr}={msk.mean():.4f}")
    Image.fromarray((np.clip(prob,0,1)*255).astype(np.uint8)).save(os.path.join(outdir, f"thaki_{tag}_prob.png"))
    over = np.asarray(rgb, np.float32).copy()
    over[msk] = over[msk]*0.25 + np.array([255,20,20], np.float32)*0.75
    Image.fromarray(over.astype(np.uint8)).save(os.path.join(outdir, f"thaki_{tag}_overlay.png"))

im = Image.open(img_path).convert("RGB")
W, H = im.size
print("image:", W, H)
run(np.asarray(im.resize((1024, 681), Image.BILINEAR)), "global", 0.6)
cx, cy = W//2, H//2
box = (cx-512, cy-340, cx+512, cy+340)
run(np.asarray(im.crop(box)), "crop", 0.6)
print("done")
