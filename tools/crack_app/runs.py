# -*- coding: utf-8 -*-
"""检测记录:每次「开始检测」生成一份独立结果,可命名、切换、重命名。

目录结构(场景级缓存仍在 ``crack/`` 下,和检测记录无关)::

    work/<scene>/crack/
        runs/<run_id>/            一次检测
            run.json              记录元信息:名字、开始时间、模型、模式…
            scores.npz            逐点分数
            display.npz           显示点云
            meta.json             检测参数与统计
            gallery/              2D 叠加图
            labels_display.npy    分析后的连通域标签
        mesh_web*.npz/json        网格预览缓存(场景级)
        tex_web/                  贴图缓存(场景级)
        preview.npz               未检测场景的预览点云
        active_run.json           当前选中的检测记录

兼容旧版本:第一次访问时,如果 ``crack/`` 根目录下还有旧的
``scores.npz/display.npz/meta.json/gallery``,会整体挪进 ``runs/<时间>/``,
``out/<scene>_cracks.*`` 也会挪到 ``out/<scene>/<时间>/``。
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime

RUN_FILE = 'run.json'
ACTIVE_FILE = 'active_run.json'
RUN_FILES = ('scores.npz', 'display.npz', 'meta.json', 'labels_display.npy')
RUN_DIRS = ('gallery',)
MODEL_LABELS = {
    'tom': 'Tomt996 UNet-R34 (seed101)',
    'tom202': 'Tomt996 UNet-R34 (seed202)',
    'tom303': 'Tomt996 UNet-R34 (seed303)',
    'tom404': 'Tomt996 UNet-R34 (seed404)',
    'tom505': 'Tomt996 UNet-R34 (seed505)',
    'dihao': 'CrackEvo UNet-R34',
    'segformer': 'SegFormer-b0 (CrackDataset)',
    'ensemble': '集成 (Tomt996 + CrackEvo)',
}

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))


def runs_root(paths):
    return os.path.join(paths['crack'], 'runs')


def run_dir(paths, run_id):
    if not run_id or os.path.basename(str(run_id)) != str(run_id) or run_id in ('.', '..'):
        raise ValueError(f'非法检测记录:{run_id}')
    return os.path.join(runs_root(paths), str(run_id))


def run_paths(paths, run_id):
    """把 paths 里的 crack/gallery 指到某条记录,其他键保持不变。"""
    d = run_dir(paths, run_id)
    out = dict(paths)
    out['crack'] = d
    out['gallery'] = os.path.join(d, 'gallery')
    return out


def _read_json(p, default=None):
    try:
        with open(p, encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def _write_json(p, obj):
    tmp = p + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, p)


def _id_from_created(created, fallback_ts):
    try:
        return datetime.fromisoformat(str(created)).strftime('%Y%m%d-%H%M%S')
    except (TypeError, ValueError):
        return datetime.fromtimestamp(fallback_ts).strftime('%Y%m%d-%H%M%S')


def _default_name(dt=None):
    return (dt or datetime.now()).strftime('%Y-%m-%d %H:%M:%S')


# =============================================================================
# 兼容旧布局
# =============================================================================
def ensure_migrated(paths):
    """旧版本的检测结果没有 runs/ 目录:第一次访问时整体迁进来。"""
    crack = paths['crack']
    root = runs_root(paths)
    if os.path.isdir(root) and any(
            os.path.isfile(os.path.join(root, d, RUN_FILE))
            for d in os.listdir(root)):
        return
    legacy_scores = os.path.join(crack, 'scores.npz')
    legacy_display = os.path.join(crack, 'display.npz')
    if not (os.path.isfile(legacy_scores) and os.path.isfile(legacy_display)):
        return
    meta = _read_json(os.path.join(crack, 'meta.json'), {}) or {}
    try:
        ts = os.path.getmtime(legacy_display)
    except OSError:
        ts = 0.0
    rid = _id_from_created(meta.get('created'), ts)
    d = run_dir(paths, rid)
    k = 1
    while os.path.exists(d):
        k += 1
        d = run_dir(paths, f'{rid}-{k}')
    rid = os.path.basename(d)
    os.makedirs(d, exist_ok=True)
    for name in RUN_FILES:
        src = os.path.join(crack, name)
        if os.path.isfile(src):
            shutil.move(src, os.path.join(d, name))
    for name in RUN_DIRS:
        src = os.path.join(crack, name)
        if os.path.isdir(src):
            shutil.move(src, os.path.join(d, name))
    created = meta.get('created') or datetime.fromtimestamp(ts).isoformat(
        timespec='seconds')
    _write_json(os.path.join(d, RUN_FILE), {
        'id': rid,
        'name': created.replace('T', ' '),
        'created': created,
        'model': meta.get('model'),
        'model_label': meta.get('model_label') or MODEL_LABELS.get(
            meta.get('model'), meta.get('model')),
        'mode': meta.get('mode'),
        'device': meta.get('device'),
        'migrated': True,
    })
    _migrate_legacy_outputs(paths['name'], rid)
    set_active(paths, rid)


def _migrate_legacy_outputs(scene, rid):
    """旧的 out/<scene>_cracks.* 挪到 out/<scene>/<rid>/cracks.*"""
    src_base = os.path.join(REPO_ROOT, 'out', f'{scene}_cracks')
    if not os.path.isfile(src_base + '.json'):
        return
    dst_dir = os.path.join(REPO_ROOT, 'out', scene, rid)
    os.makedirs(dst_dir, exist_ok=True)
    pairs = [('_cracks.ply', 'cracks.ply'),
             ('_cracks_only.ply', 'cracks_only.ply'),
             ('_cracks.csv', 'cracks.csv'),
             ('_cracks.json', 'cracks.json'),
             ('_cracks_lines.ply', 'cracks_lines.ply'),
             ('_cracks_lines.json', 'cracks_lines.json'),
             ('_cracks_lines.csv', 'cracks_lines.csv')]
    for suffix, name in pairs:
        src = os.path.join(REPO_ROOT, 'out', scene + suffix)
        dst = os.path.join(dst_dir, name)
        if os.path.isfile(src) and not os.path.exists(dst):
            try:
                shutil.move(src, dst)
            except OSError:
                pass


# =============================================================================
# 增删查改
# =============================================================================
def create_run(paths, **info):
    """新建一条检测记录,返回 run_id。名字默认是开始时间。"""
    root = runs_root(paths)
    os.makedirs(root, exist_ok=True)
    now = datetime.now()
    rid = now.strftime('%Y%m%d-%H%M%S')
    d = os.path.join(root, rid)
    k = 1
    while os.path.exists(d):
        k += 1
        d = os.path.join(root, f'{rid}-{k}')
    rid = os.path.basename(d)
    os.makedirs(d)
    model = info.get('model')
    _write_json(os.path.join(d, RUN_FILE), {
        'id': rid,
        'name': info.get('name') or _default_name(now),
        'created': now.isoformat(timespec='seconds'),
        'model': model,
        'model_label': MODEL_LABELS.get(model, model),
        'mode': info.get('mode'),
        'device': info.get('device'),
        'limit': info.get('limit'),
    })
    return rid


def list_runs(paths):
    """按开始时间倒序返回 [{id, name, created, model, …, has_analysis}]。"""
    ensure_migrated(paths)
    root = runs_root(paths)
    if not os.path.isdir(root):
        return []
    out = []
    for rid in os.listdir(root):
        d = os.path.join(root, rid)
        run = _read_json(os.path.join(d, RUN_FILE))
        if not run:
            continue
        meta = _read_json(os.path.join(d, 'meta.json'), {}) or {}
        run = dict(run)
        run['id'] = rid
        run.setdefault('name', rid)
        run.setdefault('created', meta.get('created'))
        run.setdefault('model', meta.get('model'))
        run.setdefault('model_label', meta.get('model_label')
                       or MODEL_LABELS.get(run.get('model'), run.get('model')))
        run['mode'] = run.get('mode') or meta.get('mode')
        run['device'] = run.get('device') or meta.get('device')
        run['n_images_used'] = meta.get('n_images_used')
        run['n_points'] = meta.get('n_points')
        run['n_display'] = meta.get('n_display')
        run['default_threshold'] = meta.get('default_threshold', 0.4)
        run['has_scores'] = os.path.isfile(os.path.join(d, 'scores.npz')) and \
            os.path.isfile(os.path.join(d, 'display.npz'))
        run['has_analysis'] = os.path.isfile(os.path.join(
            out_dir(paths['name'], rid), 'cracks.json'))
        out.append(run)
    out.sort(key=lambda r: str(r.get('created') or ''), reverse=True)
    return out


def out_dir(scene, run_id):
    return os.path.join(REPO_ROOT, 'out', scene, run_id)


def out_base(scene, run_id):
    return os.path.join(out_dir(scene, run_id), 'cracks')


def set_active(paths, run_id):
    run_paths(paths, run_id)          # 校验
    os.makedirs(paths['crack'], exist_ok=True)
    _write_json(os.path.join(paths['crack'], ACTIVE_FILE),
                {'run_id': str(run_id), 'updated': datetime.now().isoformat(
                    timespec='seconds')})


def active_run(paths):
    """当前选中的记录 id;没有记录返回 None。记录失效时回退到最新一条。"""
    ensure_migrated(paths)
    runs = list_runs(paths)
    if not runs:
        return None
    cur = (_read_json(os.path.join(paths['crack'], ACTIVE_FILE)) or {}).get('run_id')
    ids = {r['id'] for r in runs}
    if cur in ids:
        return cur
    cur = runs[0]['id']
    try:
        set_active(paths, cur)
    except OSError:
        pass
    return cur


def rename(paths, run_id, name):
    name = ' '.join(str(name or '').split())
    if not name:
        raise ValueError('名字不能为空')
    if len(name) > 80:
        raise ValueError('名字最多 80 个字符')
    p = os.path.join(run_dir(paths, run_id), RUN_FILE)
    run = _read_json(p)
    if not run:
        raise FileNotFoundError(run_id)
    run['name'] = name
    _write_json(p, run)
    return run
