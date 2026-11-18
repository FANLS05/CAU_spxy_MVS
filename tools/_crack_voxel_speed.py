# -*- coding: utf-8 -*-
import os, sys, time
import numpy as np
ROOT = r"D:\code\point_cloud\0916"
sys.path.insert(0, os.path.join(ROOT, "tools"))
import open3d as o3d
pc = o3d.io.read_point_cloud(os.path.join(ROOT, "work", "rock", "mvs", "scene_dense.ply"))
P = np.asarray(pc.points)
print("points", len(P), "bbox", P.min(0), P.max(0), "extent", P.max(0)-P.min(0))
for vox in (0.005, 0.008, 0.012, 0.016, 0.02):
    t0 = time.time()
    q = np.floor(P / vox).astype(np.int64)
    key = (q[:,0]*73856093) ^ (q[:,1]*19349663) ^ (q[:,2]*83492791)
    _, idx = np.unique(key, return_index=True)
    print(f"voxel {vox}: {len(idx)} pts  {time.time()-t0:.2f}s")
