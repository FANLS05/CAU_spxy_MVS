# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

ROOT = r"D:\code\point_cloud\0916"
mdir = os.path.join(ROOT, "models", "crack", "varcoder")
img_path = os.path.join(ROOT, "data", "rock", "Riebo000001.JPG")
outdir = os.path.join(ROOT, "out", "crack_test")
os.makedirs(outdir, exist_ok=True)

model = SegformerForSemanticSegmentation.from_pretrained(mdir)
model.eval()
print("config labels:", model.config.id2label)

proc = SegformerImageProcessor(
    do_resize=True, size={"height": 512, "width": 512},
    do_normalize=True,
    image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225],
)

def run(rgb, tag):
    h, w = rgb.shape[:2]
    inputs = proc(images=Image.fromarray(rgb), return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    prob = torch.softmax(logits, dim=1)[0, 1]
    prob = F.interpolate(prob[None, None], size=(h, w), mode="bilinear", align_corners=False)[0, 0].numpy()
    print(f"  {tag}: max={prob.max():.3f} mean={prob.mean():.4f} frac>0.5={(prob>0.5).mean():.4f}")
    Image.fromarray((np.clip(prob,0,1)*255).astype(np.uint8)).save(os.path.join(outdir, f"varcoder_{tag}_prob.png"))
    over = np.asarray(rgb, np.float32).copy(); m = prob > 0.5
    over[m] = over[m]*0.25 + np.array([255,20,20], np.float32)*0.75
    Image.fromarray(over.astype(np.uint8)).save(os.path.join(outdir, f"varcoder_{tag}_overlay.png"))

im = Image.open(img_path).convert("RGB")
W, H = im.size
print("image:", W, H)
run(np.asarray(im.resize((1024, 681), Image.BILINEAR)), "global")
cx, cy = W//2, H//2
box = (cx-512, cy-340, cx+512, cy+340)
run(np.asarray(im.crop(box)), "crop")
print("done")
