# -*- coding: utf-8 -*-
"""岩壁裂隙检测 —— 本地 Web 应用后端。

启动::

    python tools/crack_app/server.py
    # 浏览器打开 http://127.0.0.1:8765

职责:
- 列出 work/ 下的场景、提供 Three.js 需要的点云二进制数组;
- 后台任务: 重建(run_pipeline.sh)、裂缝检测、连通域分析/导出;
- SSE 推送日志与进度。
"""

from __future__ import annotations

import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from datetime import datetime

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

APP_DIR = os.path.dirname(os.path.abspath(__file__))
TOOLS_DIR = os.path.dirname(APP_DIR)
REPO_ROOT = os.path.dirname(TOOLS_DIR)
STATIC_DIR = os.path.join(APP_DIR, 'static')
WORK_ROOT = os.path.join(REPO_ROOT, 'work')
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import detect_project as dp          # noqa: E402
import analyze as ana                # noqa: E402
import mesh_web as mw                # noqa: E402
import mesh_tex as mt                # noqa: E402
import mask_to_points as m2p         # noqa: E402
import runs as rn                    # noqa: E402

app = FastAPI(title='岩壁裂隙检测', version='1.0')


@app.middleware('http')
async def _static_no_cache(request: Request, call_next):
    """静态文件每次都向服务器确认,避免改了 JS/CSS 后浏览器还在用旧缓存。"""
    resp = await call_next(request)
    if request.url.path.startswith('/static/'):
        resp.headers['Cache-Control'] = 'no-cache'
    return resp


# =============================================================================
# 场景
# =============================================================================
def _scene_dir(name: str) -> str:
    if not re.fullmatch(r'[A-Za-z0-9_\-\u4e00-\u9fa5.]+', name or ''):
        raise HTTPException(400, '非法场景名')
    return os.path.join(WORK_ROOT, name)


def _image_count(paths) -> int:
    try:
        return len([f for f in os.listdir(paths['images'])
                    if os.path.splitext(f)[1].lower() in
                    ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')])
    except OSError:
        return 0


def _active(scene: str):
    """返回 (场景 paths, 当前记录 id, 记录 paths);没有检测记录时后两个为 None。"""
    paths = dp.scene_paths(_scene_dir(scene))
    rid = rn.active_run(paths)
    rp = rn.run_paths(paths, rid) if rid else None
    return paths, rid, rp


def _read_meta(scene: str):
    paths, rid, rp = _active(scene)
    if not rp:
        return None
    p = os.path.join(rp['crack'], 'meta.json')
    if not os.path.isfile(p):
        return None
    try:
        with open(p, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def _scene_info(name: str):
    d = os.path.join(WORK_ROOT, name)
    paths = dp.scene_paths(d)
    has = lambda p: os.path.exists(p)
    has_recon = all(has(paths[k]) for k in ('sparse', 'images', 'cloud', 'mesh'))
    rid, rp = None, None
    try:
        rid = rn.active_run(paths)
        rp = rn.run_paths(paths, rid) if rid else None
    except Exception:
        pass
    meta = None
    if rp and os.path.isfile(os.path.join(rp['crack'], 'meta.json')):
        try:
            with open(os.path.join(rp['crack'], 'meta.json'), encoding='utf-8') as f:
                meta = json.load(f)
        except Exception:
            meta = None
    has_crack = bool(rp) and os.path.isfile(os.path.join(rp['crack'], 'display.npz')) \
        and os.path.isfile(os.path.join(rp['crack'], 'scores.npz'))
    has_preview = os.path.isfile(os.path.join(paths['crack'], 'preview.npz'))
    preview_meta = None
    pm = os.path.join(paths['crack'], 'preview_meta.json')
    if os.path.isfile(pm):
        try:
            with open(pm, encoding='utf-8') as f:
                preview_meta = json.load(f)
        except Exception:
            preview_meta = None
    has_mesh = os.path.isfile(paths['mesh'])
    mesh_faces = None
    mesh_bytes = None
    if has_mesh:
        try:
            _, mesh_faces = mw.ply_counts(paths['mesh'])
            mesh_bytes = os.path.getsize(paths['mesh'])
        except OSError:
            pass
    info = {
        'name': name,
        'has_reconstruction': has_recon,
        'has_crack': has_crack,
        'has_preview': has_preview,
        'has_mesh': has_mesh,
        'mesh_faces': mesh_faces,
        'mesh_bytes': mesh_bytes,
        'mesh_ready': bool(has_mesh and mw.cached_info(paths)),
        'n_images': _image_count(paths),
        'meta': meta,
        'preview_meta': preview_meta,
        'gallery': (meta or {}).get('gallery', []),
        'active_run': rid,
    }
    if has_crack and meta:
        info['n_display'] = meta.get('n_display')
        info['bbox'] = [meta.get('bbox_min'), meta.get('bbox_max')]
        info['default_threshold'] = meta.get('default_threshold', 0.4)
    elif preview_meta:
        info['n_display'] = preview_meta.get('n_display')
        info['bbox'] = [preview_meta.get('bbox_min'), preview_meta.get('bbox_max')]
        info['default_threshold'] = 0.4
    return info


@app.get('/api/scenes')
def api_scenes():
    if not os.path.isdir(WORK_ROOT):
        return {'scenes': []}
    names = []
    for n in sorted(os.listdir(WORK_ROOT)):
        if not os.path.isdir(os.path.join(WORK_ROOT, n)):
            continue
        if os.path.isdir(os.path.join(WORK_ROOT, n, 'dense')) or \
           os.path.isdir(os.path.join(WORK_ROOT, n, 'mvs')):
            names.append(n)
    out = []
    for n in names:
        try:
            out.append(_scene_info(n))
        except Exception as e:
            out.append({'name': n, 'error': str(e)})
    return {'scenes': out}


@app.get('/api/scene/{scene}')
def api_scene(scene: str):
    if not os.path.isdir(_scene_dir(scene)):
        raise HTTPException(404, '场景不存在')
    return _scene_info(scene)


# =============================================================================
# 点云二进制
# =============================================================================
_CLOUD = {}
_CLOUD_LOCK = threading.Lock()


def _cloud_arrays(scene: str):
    paths, rid, rp = _active(scene)
    disp = os.path.join(rp['crack'], 'display.npz') if rp else ''
    prev = os.path.join(paths['crack'], 'preview.npz')
    src = disp if disp and os.path.isfile(disp) else prev
    if not os.path.isfile(src):
        raise HTTPException(404, '这个场景还没有可显示的点云:先检测或生成预览。')
    mtime = os.path.getmtime(src)
    key = (scene, src, mtime)
    with _CLOUD_LOCK:
        hit = _CLOUD.get(key)
        if hit:
            return hit
        z = np.load(src)
        n = len(z['positions'])
        arrs = {
            'positions': z['positions'].astype('<f4', copy=False),
            'colors': z['colors'].astype('<u1', copy=False),
            'scores': z['scores'].astype('<u1', copy=False),
            'nvis': z['n_visible'].astype('<u1', copy=False),
            'labels': np.full(n, -1, np.int16),
            'kind': 'crack' if src == disp else 'preview',
            'n': n,
        }
        lab = os.path.join(rp['crack'], 'labels_display.npy') if rp else ''
        if lab and os.path.isfile(lab) and os.path.getsize(lab) > 0:
            la = np.load(lab)
            if len(la) == n:
                arrs['labels'] = la.astype('<i2', copy=False)
        _CLOUD.clear()
        _CLOUD[key] = arrs
        return arrs


@app.get('/api/scene/{scene}/cloud/meta')
def api_cloud_meta(scene: str):
    a = _cloud_arrays(scene)
    p = a['positions']
    return {
        'n': int(a['n']),
        'kind': a['kind'],
        'bbox_min': p.min(axis=0).tolist(),
        'bbox_max': p.max(axis=0).tolist(),
        'has_scores': bool(a['kind'] == 'crack'),
        'has_labels': bool((a['labels'] >= 0).any()),
    }


@app.get('/api/scene/{scene}/cloud/{part}')
def api_cloud_part(scene: str, part: str):
    a = _cloud_arrays(scene)
    if part == 'labels':
        # labels 在 analyze 之后才生成/更新,不跟随 display.npz 缓存
        _, _, rp = _active(scene)
        p = os.path.join(rp['crack'], 'labels_display.npy') if rp else ''
        if p and os.path.isfile(p) and os.path.getsize(p) > 0:
            la = np.load(p)
            if len(la) == a['n']:
                return Response(content=la.astype('<i2', copy=False).tobytes(),
                                media_type='application/octet-stream',
                                headers={'Cache-Control': 'no-cache'})
        arr = a['labels']
        return Response(content=arr.tobytes(), media_type='application/octet-stream',
                        headers={'Cache-Control': 'no-cache'})
    if part not in ('positions', 'colors', 'scores', 'nvis'):
        raise HTTPException(404, '未知数组')
    arr = a[part]
    return Response(content=arr.tobytes(), media_type='application/octet-stream',
                    headers={'Cache-Control': 'public, max-age=300'})


@app.get('/api/scene/{scene}/analysis')
def api_analysis(scene: str):
    """当前检测记录的分析结果(out/<scene>/<run>/cracks.json)。"""
    paths, rid, rp = _active(scene)
    if not rp:
        raise HTTPException(404, '还没有检测记录')
    base = rn.out_base(scene, rid)
    j = base + '.json'
    if not os.path.isfile(j):
        raise HTTPException(404, '这条检测记录还没有分析结果')
    try:
        with open(j, encoding='utf-8') as f:
            res = json.load(f)
    except Exception as e:
        raise HTTPException(500, f'读取分析结果失败: {e}')
    outs = {'ply': base + '.ply', 'ply_only': base + '_only.ply',
            'csv': base + '.csv', 'json': j,
            'lines_ply': base + '_lines.ply',
            'lines_csv': base + '_lines.csv',
            'lines_json': base + '_lines.json'}
    res['outputs'] = {k: os.path.relpath(v, REPO_ROOT).replace('\\', '/')
                      for k, v in outs.items() if os.path.isfile(v)}
    res['run_id'] = rid
    try:
        res['stale'] = os.path.getmtime(j) < os.path.getmtime(
            os.path.join(rp['crack'], 'meta.json'))
    except OSError:
        res['stale'] = False
    return res


# ---- 检测记录 ----
class RunRenameReq(BaseModel):
    name: str


class RunActivateReq(BaseModel):
    run_id: str


@app.get('/api/scene/{scene}/runs')
def api_runs(scene: str):
    paths = dp.scene_paths(_scene_dir(scene))
    rid = rn.active_run(paths)
    return {'scene': scene, 'active': rid, 'runs': rn.list_runs(paths)}


@app.post('/api/scene/{scene}/runs/{run_id}/activate')
def api_run_activate(scene: str, run_id: str):
    paths = dp.scene_paths(_scene_dir(scene))
    try:
        d = rn.run_dir(paths, run_id)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if not os.path.isfile(os.path.join(d, rn.RUN_FILE)):
        raise HTTPException(404, '检测记录不存在')
    rn.set_active(paths, run_id)
    _CLOUD.clear()
    return {'ok': True, 'active': run_id}


@app.post('/api/scene/{scene}/runs/{run_id}/rename')
def api_run_rename(scene: str, run_id: str, req: RunRenameReq):
    paths = dp.scene_paths(_scene_dir(scene))
    try:
        run = rn.rename(paths, run_id, req.name)
    except FileNotFoundError:
        raise HTTPException(404, '检测记录不存在')
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {'ok': True, 'run': run}


@app.get('/api/scene/{scene}/gallery/{fname}')
def api_gallery(scene: str, fname: str):
    _, _, rp = _active(scene)
    base = os.path.abspath(rp['gallery'] if rp else
                           os.path.join(_scene_dir(scene), 'crack', 'gallery'))
    p = os.path.abspath(os.path.join(base, fname))
    if not p.startswith(base + os.sep) or not os.path.isfile(p):
        raise HTTPException(404, '图片不存在')
    return FileResponse(p, headers={'Cache-Control': 'public, max-age=3600'})


# =============================================================================
# 网格:网页预览缓存 + 原始 PLY 导出
# =============================================================================
_MESH = {}
_MESH_LOCK = threading.Lock()


def _mesh_paths(scene: str):
    paths = dp.scene_paths(_scene_dir(scene))
    if not os.path.isfile(paths['mesh']):
        raise HTTPException(404, '这个场景没有网格文件(先跑重建)。')
    return paths


def _check_mesh_level(level: str) -> str:
    if level not in mw.LEVELS:
        raise HTTPException(400, f'未知的网格细节档位:{level}')
    return level


def _display_for_mesh(scene: str):
    """当前检测记录的 display.npz;没有记录时返回 None(用预览点云上色)。"""
    _, _, rp = _active(scene)
    p = os.path.join(rp['crack'], 'display.npz') if rp else ''
    return p if p and os.path.isfile(p) else None


def _mesh_arrays(scene: str, level: str = 'std'):
    paths = _mesh_paths(scene)
    meta = mw.cached_info(paths, level, display=_display_for_mesh(scene))
    if not meta:
        raise HTTPException(409, '网格还没生成预览缓存,先调用 /api/mesh/prepare。')
    npz_path, _ = mw.cache_paths(paths, level)
    mtime = os.path.getmtime(npz_path)
    key = (scene, npz_path, mtime)
    with _MESH_LOCK:
        hit = _MESH.get(key)
        if hit:
            return hit, meta
        z = np.load(npz_path)
        arrs = {k: z[k] for k in z.files}
        n = len(arrs['positions'])
        if 'colors' not in arrs:
            arrs['colors'] = np.full((n, 3), 165, np.uint8)
        _MESH.clear()
        _MESH[key] = arrs
        return arrs, meta


@app.get('/api/scene/{scene}/mesh/info')
def api_mesh_info(scene: str, level: str = 'std'):
    _check_mesh_level(level)
    paths = _mesh_paths(scene)
    nv, nf = mw.ply_counts(paths['mesh'])
    display = _display_for_mesh(scene)
    meta = mw.cached_info(paths, level, display=display)
    levels = {}
    for name in mw.LEVELS:
        lm = mw.cached_info(paths, name, display=display)
        if lm:
            levels[name] = {'faces': lm.get('n_faces'),
                            'seconds': lm.get('seconds')}
    return {
        'scene': scene,
        'has_mesh': True,
        'level': level,
        'source_faces': nf,
        'source_verts': nv,
        'source_bytes': os.path.getsize(paths['mesh']),
        'ready': bool(meta),
        'meta': meta,
        'levels': levels,
    }


@app.get('/api/scene/{scene}/mesh/download')
def api_mesh_download(scene: str):
    paths = _mesh_paths(scene)
    return FileResponse(paths['mesh'], filename=f'{scene}_mesh.ply',
                        media_type='application/octet-stream')


@app.get('/api/scene/{scene}/mesh/{part}')
def api_mesh_part(scene: str, part: str, level: str = 'std'):
    if part not in ('positions', 'normals', 'colors', 'indices'):
        raise HTTPException(404, '未知数组')
    _check_mesh_level(level)
    arrs, meta = _mesh_arrays(scene, level)
    arr = arrs.get(part)
    if arr is None:
        raise HTTPException(404, '缓存里没有这个数组')
    if part in ('positions', 'normals'):
        arr = arr.astype('<f4', copy=False)
    elif part == 'colors':
        arr = arr.astype('<u1', copy=False)
    else:
        arr = arr.astype('<u4', copy=False)
    return Response(content=arr.tobytes(), media_type='application/octet-stream',
                    headers={'Cache-Control': 'public, max-age=300'})


class MeshReq(BaseModel):
    scene: str
    level: str = 'std'        # std 45万 / high 150万 / ultra 400万 / full 原始


def _mesh_job(job: Job, scene: str, level: str = 'std'):
    paths = dp.scene_paths(_scene_dir(scene))
    job.emit({'event': 'stage', 'stage': 'mesh',
              'message': f'准备网页网格(档位 {level})…'})
    meta = mw.build(paths, progress=job.emit, level=level,
                    display=_display_for_mesh(scene))
    job.emit({'event': 'mesh_done', 'scene': scene, 'level': level, 'meta': meta,
              'message': f'网格就绪:{meta["n_faces"]:,} 面'})


# =============================================================================
# 任务
# =============================================================================
class Job:
    def __init__(self, kind: str):
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind
        self.status = 'running'
        self.events = []
        self.stop_flag = False
        self.created = datetime.now().isoformat(timespec='seconds')
        self.lock = threading.Lock()

    def emit(self, ev: dict):
        ev = dict(ev)
        ev.setdefault('ts', round(time.time(), 3))
        with self.lock:
            self.events.append(ev)

    def finish(self, status: str, message: str = ''):
        self.status = status
        self.emit({'event': 'end', 'status': status, 'message': message})


JOBS = {}
_JOB_LOCK = threading.Lock()
_ACTIVE = {'id': None}


def start_job(kind: str, fn, *args, **kwargs) -> Job:
    with _JOB_LOCK:
        cur = _ACTIVE['id']
        if cur and cur in JOBS and JOBS[cur].status == 'running':
            raise HTTPException(409, f'已有任务在跑 ({JOBS[cur].kind}),等它结束或先停止。')
        job = Job(kind)
        JOBS[job.id] = job
        _ACTIVE['id'] = job.id
        # 只保留最近 20 个任务
        while len(JOBS) > 20:
            oldest = min(JOBS.values(), key=lambda j: j.created)
            if oldest.status == 'running':
                break
            JOBS.pop(oldest.id, None)

    def runner():
        try:
            fn(job, *args, **kwargs)
            if job.status == 'running':
                job.finish('done')
        except Exception as e:                       # noqa: BLE001
            import traceback
            traceback.print_exc()
            job.emit({'event': 'error', 'message': f'{type(e).__name__}: {e}'})
            job.finish('error', str(e))

    threading.Thread(target=runner, daemon=True).start()
    return job


@app.get('/api/jobs/{jid}/events')
async def api_job_events(jid: str):
    job = JOBS.get(jid)
    if not job:
        raise HTTPException(404, '任务不存在')

    async def gen():
        sent = 0
        while True:
            with job.lock:
                evs = job.events[sent:]
            for ev in evs:
                sent += 1
                yield 'data: ' + json.dumps(ev, ensure_ascii=False) + '\n\n'
            if job.status != 'running' and sent >= len(job.events):
                break
            import asyncio
            await asyncio.sleep(0.25)

    return StreamingResponse(gen(), media_type='text/event-stream',
                             headers={'Cache-Control': 'no-cache',
                                      'X-Accel-Buffering': 'no'})


@app.post('/api/jobs/{jid}/stop')
def api_job_stop(jid: str):
    job = JOBS.get(jid)
    if not job:
        raise HTTPException(404, '任务不存在')
    job.stop_flag = True
    job.emit({'event': 'log', 'message': '收到停止请求…'})
    return {'ok': True}


# ---- 预览点云(还没检测时也能看) ----
def _preview_job(job: Job, scene: str):
    paths = dp.scene_paths(_scene_dir(scene))
    dp.check_scene(paths)
    os.makedirs(paths['crack'], exist_ok=True)
    job.emit({'event': 'progress', 'stage': 'preview', 'done': 5, 'total': 100,
              'message': '读取点云…'})
    P, cols = m2p.load_point_cloud(paths['cloud'])
    job.emit({'event': 'progress', 'stage': 'preview', 'done': 50, 'total': 100,
              'message': f'{len(P):,} 点,降采样…'})
    idx, voxel = dp.display_indices(P)
    pos = P[idx].astype(np.float32)
    if cols is not None:
        col = np.clip(cols[idx] * 255, 0, 255).astype(np.uint8)
    else:
        col = np.full((len(idx), 3), 190, np.uint8)
    np.savez(os.path.join(paths['crack'], 'preview.npz'),
             positions=pos, colors=col,
             scores=np.zeros(len(idx), np.uint8),
             n_visible=np.zeros(len(idx), np.uint8))
    meta = {
        'scene': scene, 'n_points': int(len(P)), 'n_display': int(len(idx)),
        'voxel': voxel,
        'bbox_min': P.min(axis=0).tolist(), 'bbox_max': P.max(axis=0).tolist(),
        'created': datetime.now().isoformat(timespec='seconds'),
    }
    with open(os.path.join(paths['crack'], 'preview_meta.json'), 'w',
              encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    job.emit({'event': 'progress', 'stage': 'preview', 'done': 100, 'total': 100})
    job.emit({'event': 'preview_done', 'scene': scene, 'meta': meta,
              'message': f'预览点云 {len(idx):,} 点'})


class PreviewReq(BaseModel):
    scene: str


@app.post('/api/preview')
def api_preview(req: PreviewReq):
    if not os.path.isdir(_scene_dir(req.scene)):
        raise HTTPException(404, '场景不存在')
    job = start_job('preview', _preview_job, req.scene)
    return {'job_id': job.id}


@app.post('/api/mesh/prepare')
def api_mesh_prepare(req: MeshReq):
    _check_mesh_level(req.level)
    _mesh_paths(req.scene)
    job = start_job('mesh', _mesh_job, req.scene, req.level)
    return {'job_id': job.id}


# =============================================================================
# 贴图预览(OpenMVS 的 scene_textured.ply + 16 张贴图)
# =============================================================================
_TEX = {}
_TEX_LOCK = threading.Lock()


def _tex_paths(scene: str):
    paths = dp.scene_paths(_scene_dir(scene))
    if not os.path.isfile(mt.source_ply(paths)):
        raise HTTPException(404, '这个场景没有贴图网格 scene_textured.ply(先重建出纹理)。')
    return paths


def _tex_arrays(scene: str):
    paths = _tex_paths(scene)
    meta = mt.cached_info(paths)
    if not meta:
        raise HTTPException(409, '贴图缓存还没生成,先调用 /api/tex/prepare。')
    npz_path, _ = mt.array_paths(paths)
    key = (scene, npz_path, os.path.getmtime(npz_path))
    with _TEX_LOCK:
        hit = _TEX.get(key)
        if hit:
            return hit, meta
        z = np.load(npz_path)
        arrs = {k: z[k] for k in z.files}
        _TEX.clear()
        _TEX[key] = arrs
        return arrs, meta


class TexReq(BaseModel):
    scene: str
    tex_size: int = mt.TEX_SIZE


def _tex_job(job: Job, scene: str, tex_size: int = mt.TEX_SIZE):
    paths = dp.scene_paths(_scene_dir(scene))
    job.emit({'event': 'stage', 'stage': 'mesh', 'message': '准备贴图网格…'})
    meta = mt.build(paths, progress=job.emit, tex_size=tex_size)
    job.emit({'event': 'tex_done', 'scene': scene, 'meta': meta,
              'message': f'贴图网格就绪:{meta["n_faces"]:,} 面'})


@app.post('/api/tex/prepare')
def api_tex_prepare(req: TexReq):
    _tex_paths(req.scene)
    if not 512 <= req.tex_size <= 8192:
        raise HTTPException(400, '贴图边长应在 512~8192 之间')
    job = start_job('tex', _tex_job, req.scene, req.tex_size)
    return {'job_id': job.id}


@app.get('/api/scene/{scene}/tex/info')
def api_tex_info(scene: str):
    paths = _tex_paths(scene)
    meta = mt.cached_info(paths)
    return {'scene': scene, 'ready': bool(meta), 'meta': meta}


@app.get('/api/scene/{scene}/tex/img/{index}')
def api_tex_img(scene: str, index: int):
    paths = _tex_paths(scene)
    f = os.path.join(mt.cache_dir(paths), f'tex_{index}.jpg')
    if not os.path.isfile(f):
        raise HTTPException(404, '没有这张贴图')
    return FileResponse(f, media_type='image/jpeg',
                        headers={'Cache-Control': 'public, max-age=3600'})


@app.get('/api/scene/{scene}/tex/{part}')
def api_tex_part(scene: str, part: str):
    if part not in ('positions', 'normals', 'indices', 'uv'):
        raise HTTPException(404, '未知数组')
    arrs, meta = _tex_arrays(scene)
    arr = arrs.get(part)
    if arr is None:
        raise HTTPException(404, '缓存里没有这个数组')
    arr = arr.astype('<u4' if part == 'indices' else '<f4', copy=False)
    return Response(content=arr.tobytes(), media_type='application/octet-stream',
                    headers={'Cache-Control': 'public, max-age=300'})


# ---- 裂缝检测 ----
MODEL_KEYS = ('tom', 'tom202', 'tom303', 'tom404', 'tom505',
              'dihao', 'segformer', 'ensemble')
MODEL_SHORT = {'tom': 'seed101', 'tom202': 'seed202', 'tom303': 'seed303',
               'tom404': 'seed404', 'tom505': 'seed505', 'dihao': 'CrackEvo',
               'segformer': 'SegFormer', 'ensemble': '集成'}


class DetectReq(BaseModel):
    scene: str
    model: str = Field('tom', pattern='^(tom|tom202|tom303|tom404|tom505|'
                                          'dihao|segformer|ensemble)$')
    models: list[str] | None = None      # 多选;每个模型各存一条检测记录
    mode: str = Field('fine', pattern='^(fast|fine)$')
    limit: int | None = None
    overwrite: bool = False
    gallery: bool = True
    threads: int | None = None
    device: str = 'gpu'          # gpu | cpu | auto


def _req_models(req: DetectReq):
    raw = req.models or [req.model]
    out = []
    for m in raw:
        if m not in MODEL_KEYS:
            raise HTTPException(400, f'未知模型 {m}')
        if m not in out:
            out.append(m)
    return out or ['tom']


def _detect_job(job: Job, req: DetectReq, models: list[str]):
    paths = dp.scene_paths(_scene_dir(req.scene))
    stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    failures = []
    done = []
    for i, model in enumerate(models):
        short = MODEL_SHORT.get(model, model)
        name = stamp if len(models) == 1 else f'{stamp} · {short}'
        rid = rn.create_run(paths, model=model, mode=req.mode,
                            device=req.device, limit=req.limit, name=name)
        rp = rn.run_paths(paths, rid)
        job.emit({'event': 'stage', 'stage': 'detect',
                  'message': f'[{i + 1}/{len(models)}] {short} → 新记录 {rid}',
                  'run_id': rid, 'model': model})
        try:
            dp.run(rp, model=model, mode=req.mode, limit=req.limit,
                   overwrite=req.overwrite, gallery=req.gallery,
                   threads=req.threads, progress=job.emit,
                   should_stop=lambda: job.stop_flag, device=req.device)
        except Exception as e:                  # noqa: BLE001
            # 没跑完的记录没有 scores,直接删掉,别在列表里留半成品
            try:
                if not os.path.isfile(os.path.join(rp['crack'], 'scores.npz')):
                    shutil.rmtree(rp['crack'], ignore_errors=True)
            except OSError:
                pass
            failures.append((short, str(e)))
            job.emit({'event': 'error',
                      'message': f'{short} 检测失败:{type(e).__name__}: {e}'})
            if job.stop_flag:
                raise RuntimeError('用户已停止任务') from e
            continue
        rn.set_active(paths, rid)
        done.append(rid)
        job.emit({'event': 'run_done', 'run_id': rid, 'scene': req.scene,
                  'model': model,
                  'message': f'{short} 完成,记录 {rid}'})
    if not done:
        raise RuntimeError('所有模型都失败了:' +
                           '; '.join(f'{n}: {e}' for n, e in failures))
    job.emit({'event': 'log',
              'message': f'本次共完成 {len(done)} 个模型'
                         + (f',失败 {len(failures)} 个' if failures else '')
                         + f';当前记录 {done[-1]}'})


@app.post('/api/detect')
def api_detect(req: DetectReq):
    if not os.path.isdir(_scene_dir(req.scene)):
        raise HTTPException(404, '场景不存在')
    if req.limit is not None and req.limit < 1:
        raise HTTPException(400, '图片数量必须 >= 1')
    if req.device not in ('gpu', 'cpu', 'auto'):
        raise HTTPException(400, 'device 只能是 gpu / cpu / auto')
    models = _req_models(req)
    job = start_job('detect', _detect_job, req, models)
    return {'job_id': job.id, 'models': models}


# ---- 分析 + 导出 ----
class AnalyzeReq(BaseModel):
    scene: str
    threshold: float | None = Field(None, ge=0.0, le=1.0)
    min_length: float | None = Field(None, ge=0.0)
    gap: float | None = Field(None, ge=0.0)
    scale: float | None = Field(None, gt=0.0)
    write_outputs: bool = True


def _analyze_job(job: Job, req: AnalyzeReq):
    paths, rid, rp = _active(req.scene)
    if not rp:
        raise RuntimeError('还没有检测记录,先做一次检测。')
    meta = _read_meta(req.scene) or {}
    thr = req.threshold if req.threshold is not None else meta.get('default_threshold', 0.4)
    bb = np.array(meta.get('bbox_max', [1, 1, 1])) - np.array(meta.get('bbox_min', [0, 0, 0]))
    diag = float(np.linalg.norm(bb)) if len(bb) == 3 else 1.0
    min_len = req.min_length if req.min_length is not None else max(0.005, diag * 0.003)
    job.emit({'event': 'log',
              'message': f'记录 {rid}:阈值 {thr:.2f},最短长度 {min_len:.4f},'
                         f'连接半径 {req.gap if req.gap else "自动"},'
                         f'导出 {req.write_outputs}'})
    out_dir = rn.out_dir(req.scene, rid)
    res = ana.analyze(rp, threshold=thr, min_length=min_len, gap=req.gap,
                      scale=req.scale, write_outputs=req.write_outputs,
                      progress=job.emit, out_dir=out_dir,
                      base_prefix=os.path.join(out_dir, 'cracks'))
    res['run_id'] = rid
    job.emit({'event': 'result', 'kind': 'analyze', 'result': res,
              'message': f'找到 {res.get("n_cracks", 0)} 条裂隙,'
                         f'{res.get("n_crack_points", 0):,} 个裂隙点'})


@app.post('/api/analyze')
def api_analyze(req: AnalyzeReq):
    if not os.path.isdir(_scene_dir(req.scene)):
        raise HTTPException(404, '场景不存在')
    _, _, rp = _active(req.scene)
    if not rp or not os.path.isfile(os.path.join(rp['crack'], 'scores.npz')):
        raise HTTPException(400, '还没有检测记录,先做一次检测。')
    job = start_job('analyze', _analyze_job, req)
    return {'job_id': job.id}


# ---- 从照片重建 ----
class ReconstructReq(BaseModel):
    photos_dir: str
    scene: str
    force: bool = False
    matcher: str = 'exhaustive'      # exhaustive | sequential
    colmap_relaxed: int = 1          # 1=放宽容忍度, 0=COLMAP 官方
    resolution_level: int = 1        # OpenMVS 稠密降采样, 0=全分辨率
    max_resolution: int = 2000       # 特征提取/稠密的最大图像边长
    do_refine: int = 1               # RefineMesh
    mesh_quality: str = 'standard'   # standard | fine | ultra
    mesh_only: bool = False          # 只重跑网格/纹理(MESH_FORCE=1)


# 网格精度预设 → run_pipeline.sh 的环境变量
MESH_PRESETS = {
    'standard': {},
    'fine': {
        'MESH_MIN_POINT_DISTANCE': '0.75',
        'REFINE_MAX_FACE_AREA': '8',
    },
    'ultra': {
        'MESH_MIN_POINT_DISTANCE': '0.5',
        'REFINE_MAX_FACE_AREA': '4',
        'REFINE_SCALES': '2',
    },
}


def _msys_path(p: str) -> str:
    p = os.path.abspath(p).replace('\\', '/')
    if len(p) > 1 and p[1] == ':':
        p = '/' + p[0].lower() + p[2:]
    return p


def _bash() -> str:
    for c in (r'C:\Program Files\Git\bin\bash.exe',
              r'C:\Program Files (x86)\Git\bin\bash.exe',
              shutil.which('bash')):
        if c and os.path.isfile(c):
            return c
    raise RuntimeError('找不到 Git Bash (bash.exe)')


def _reconstruct_job(job: Job, req: ReconstructReq):
    if not os.path.isdir(req.photos_dir):
        raise RuntimeError(f'照片目录不存在: {req.photos_dir}')
    if not re.fullmatch(r'[A-Za-z0-9_\-\u4e00-\u9fa5]+', req.scene or ''):
        raise RuntimeError('场景名只能用字母/数字/下划线/短横线/中文')
    # 只数顶层图片(run_pipeline.sh 和 COLMAP 也只读顶层)。
    # 少于 8 张时重建质量很差,勾了强制重跑还会覆盖掉已有的好场景。
    img_exts = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
    try:
        n_img = sum(1 for f in os.listdir(req.photos_dir)
                    if f.lower().endswith(img_exts)
                    and os.path.isfile(os.path.join(req.photos_dir, f)))
    except OSError as e:
        raise RuntimeError(f'读不了照片目录: {e}')
    if n_img < 8:
        raise RuntimeError(
            f'照片文件夹顶层只有 {n_img} 张图片,不足以稳定重建。请选 '
            r'data\<场景>(原始照片)或 work\<场景>\dense\images(去畸变图);'
            f'当前选的是 {req.photos_dir}')
    script = os.path.join(REPO_ROOT, 'run_pipeline.sh')
    if not os.path.isfile(script):
        raise RuntimeError('找不到 run_pipeline.sh')
    if req.matcher not in ('exhaustive', 'sequential'):
        raise RuntimeError('特征匹配只能是 exhaustive 或 sequential')
    if req.colmap_relaxed not in (0, 1):
        raise RuntimeError('COLMAP 阈值只能是 0(官方) 或 1(放宽)')
    if not 0 <= req.resolution_level <= 3:
        raise RuntimeError('稠密层级只能是 0~3')
    if not 640 <= req.max_resolution <= 20000:
        raise RuntimeError('最大边长应在 640~20000 之间')
    if req.do_refine not in (0, 1):
        raise RuntimeError('网格细化只能是 0 或 1')
    if req.mesh_quality not in MESH_PRESETS:
        raise RuntimeError('网格精度只能是 standard / fine / ultra')

    # 不要设 MSYS_NO_PATHCONV:脚本里传给 colmap/OpenMVS 的是 /d/... 形式,
    # 需要 Git Bash 自动转换成 D:\...,否则 Windows 程序会把 /d/... 当成
    # 当前盘符下的相对根路径,报 database_parent_path 不存在。
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    params = {
        'MATCHER': req.matcher,
        'COLMAP_RELAXED': str(req.colmap_relaxed),
        'RESOLUTION_LEVEL': str(req.resolution_level),
        'MAX_RESOLUTION': str(req.max_resolution),
        'DO_REFINE': str(req.do_refine),
        'MESH_QUALITY': req.mesh_quality,
    }
    params.update(MESH_PRESETS[req.mesh_quality])
    env.update(params)
    if req.mesh_only:
        env['MESH_FORCE'] = '1'
        if not os.path.isfile(os.path.join(REPO_ROOT, 'work', req.scene,
                                           'mvs', 'scene_dense.ply')):
            job.emit({'event': 'log', 'message':
                      '注意:还没有稠密点云 scene_dense.ply,「只重做网格」会退化成从头重建。'})
    if req.force:
        env['FORCE'] = '1'
        if os.path.isdir(os.path.join(REPO_ROOT, 'work', req.scene)):
            job.emit({'event': 'log', 'message':
                      f'注意:场景「{req.scene}」已存在,强制重跑会删除并重建 sparse/ 与 dense/。'})
    cmd = [_bash(), script, _msys_path(req.photos_dir), req.scene]
    job.emit({'event': 'log', 'message': '$ ' + ' '.join(cmd)})
    job.emit({'event': 'log', 'message': '参数: ' +
              ' '.join(f'{k}={v}' for k, v in params.items()) +
              f" FORCE={1 if req.force else 0} MESH_FORCE={1 if req.mesh_only else 0}"})
    p = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                         errors='replace', bufsize=1)
    try:
        for line in p.stdout:
            if job.stop_flag:
                break
            line = line.rstrip()
            if not line:
                continue
            job.emit({'event': 'log', 'message': line})
            # run_pipeline.sh 用 "  [3] COLMAP 增量式 SfM" 标记步骤
            m = re.match(r'^\s*\[(\d+)\]\s+(.+)$', line)
            if m:
                n = int(m.group(1))
                job.emit({'event': 'progress', 'stage': 'reconstruct',
                          'step': n, 'step_title': m.group(2).strip(),
                          'done': min(96, n * 10), 'total': 100})
            elif '全部完成' in line:
                job.emit({'event': 'progress', 'stage': 'reconstruct',
                          'done': 100, 'total': 100})
    finally:
        if job.stop_flag and p.poll() is None:
            job.emit({'event': 'log', 'message': '终止重建进程…'})
            subprocess.run(['taskkill', '/F', '/T', '/PID', str(p.pid)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        rc = p.wait()
    if rc != 0:
        raise RuntimeError(f'run_pipeline.sh 退出码 {rc}')
    job.emit({'event': 'reconstruct_done', 'scene': req.scene,
              'message': '重建完成'})


@app.post('/api/reconstruct')
def api_reconstruct(req: ReconstructReq):
    job = start_job('reconstruct', _reconstruct_job, req)
    return {'job_id': job.id}


# =============================================================================
# 文件与静态资源
# =============================================================================
@app.get('/api/download')
def api_download(path: str):
    out_root = os.path.abspath(os.path.join(REPO_ROOT, 'out'))
    p = os.path.abspath(os.path.join(REPO_ROOT, path))
    if not (p + os.sep).startswith(out_root + os.sep) or not os.path.isfile(p):
        raise HTTPException(404, '文件不存在')
    return FileResponse(p, filename=os.path.basename(p))


@app.get('/api/fs')
def api_fs(path: str | None = None):
    """极简目录浏览,方便选照片文件夹。"""
    p = os.path.abspath(path or os.path.expanduser('~'))
    if not os.path.isdir(p):
        raise HTTPException(404, '目录不存在')
    parent = os.path.dirname(p)
    dirs = []
    try:
        for n in sorted(os.listdir(p)):
            fp = os.path.join(p, n)
            if os.path.isdir(fp) and not n.startswith('.'):
                dirs.append(n)
    except PermissionError:
        raise HTTPException(403, '没有权限')
    return {'path': p, 'parent': parent if parent != p else None,
            'dirs': dirs[:500]}


app.mount('/static', StaticFiles(directory=STATIC_DIR), name='static')


@app.get('/')
def index():
    return FileResponse(os.path.join(STATIC_DIR, 'index.html'))


def main(argv=None):
    import argparse
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--no-browser', action='store_true')
    args = ap.parse_args(argv)
    url = f'http://{args.host}:{args.port}'
    print(f'\n  岩壁裂隙检测  ->  {url}\n')
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level='info')


if __name__ == '__main__':
    main()
