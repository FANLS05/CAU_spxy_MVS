# -*- coding: utf-8 -*-
import os, sys, time
import numpy as np
ROOT = r"D:\code\point_cloud\0916"
sys.path.insert(0, os.path.join(ROOT, "tools"))
import bbox_to_mesh as b2m
import mask_to_points as m2p

scene_dir = os.path.join(ROOT, "work", "rock", "dense", "sparse")
cloud = os.path.join(ROOT, "work", "rock", "mvs", "scene_dense.ply")
mesh = os.path.join(ROOT, "work", "rock", "mvs", "scene_dense_mesh.ply")
prob_path = os.path.join(ROOT, "out", "crack_test", "full_tom_prob.png")

t0 = time.time(); scene = b2m.load_scene(scene_dir); print("scene", len(scene), f"{time.time()-t0:.1f}s")
t0 = time.time(); P, cols = m2p.load_point_cloud(cloud); print("cloud", len(P), f"{time.time()-t0:.1f}s")
t0 = time.time(); V, F, _ = b2m.load_mesh(mesh); print("mesh", len(V), len(F), f"{time.time()-t0:.1f}s")
t0 = time.time(); rc = m2p.build_raycast_scene(V, F); print("raycast", f"{time.time()-t0:.1f}s")
del V, F
from PIL import Image
prob = np.asarray(Image.open(prob_path))
print("prob", prob.shape, prob.dtype)

name = next(iter(scene))
cam = scene[name]
print("camera", name, cam['W'], cam['H'], "K", cam['K'].shape)
finite = np.isfinite(P).all(axis=1)
t0 = time.time(); depth, dist, th = m2p.view_geometry(P, cam, rc); print("view_geometry", f"{time.time()-t0:.2f}s")
tol, sig = m2p.auto_depth_tol(th, dist); print("tol", tol, sig)
t0 = time.time(); vis = m2p.visibility_from(depth, dist, th, tol, finite); print("visibility", f"{time.time()-t0:.2f}s", "visible", int(vis.sum()))
t0 = time.time()
u, v, _ = b2m.project_points(P, cam['K'], cam['R'], cam['t'])
iu = np.rint(u).astype(np.int64); iv = np.rint(v).astype(np.int64)
ok = vis & (iu>=0)&(iu<prob.shape[1])&(iv>=0)&(iv<prob.shape[0])
s = np.zeros(len(P), np.float32); s[ok] = prob[iv[ok], iu[ok]]
print("lookup", f"{time.time()-t0:.2f}s", "mean", float(s[ok].mean()), "frac>128", float((s[ok]>128).mean()))
