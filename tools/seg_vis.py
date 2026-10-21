#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""seg_vis.py —— 一条命令:场景 + 关键词 → 直接弹 3D 可视化窗口。

    $py = "C:\\Users\\FANLS\\.conda\\envs\\basic_env\\python.exe"
    $root = "D:\\code\\point_cloud\\0916"

    # 看窗户
    & $py "$root\\tools\\seg_vis.py" --scene food --text window

    # 多个关键词,顺序就是类别 id 的顺序
    & $py "$root\\tools\\seg_vis.py" --scene food --text window --text door

它是个**编排器,自己不实现任何算法**:

    1. 由场景名推出 sparse / images / 点云 / 网格 的路径
    2. mask 缓存不齐 → 子进程调 sam3 环境跑 sam3_masks.py 出 mask
    3. 只有加 --overlay 时,才把 mask 半透明叠到去畸变图上,存 out\\<场景>\\overlay\\
    4. 子进程调本环境的 mask_to_points.py,--vis 弹窗口

**为什么非得跨进程**:`basic_env` 没 torch(跑不了 SAM 3),`sam3` 环境没
open3d(做不了遮挡剔除,也开不了窗口)。**没有任何单个 Python 进程能跑完这条
链路** —— 这是「两个环境分开」的直接后果。两个子调用各自就是现成的 CLI,
而且它们本来就是靠 PNG 文件交接的,拆成进程毫无损耗。

**为什么不在这里重写投影/投票**:那套逻辑在 mask_to_points.py 里,已经过
`--verify` 与 bbox_live 逐位对拍。重写一遍必然漂移,以后两边不一致会极难查。

mask 是**中间产物**(复用就靠它),缓存在 `<仓库>\\masks\\<场景>\\<概念>\\`。
想重跑加 `--remask`。
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

# Windows 控制台中文输出;stderr 也要设,报错走的是它。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8')
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mask_to_points as m                                    # noqa: E402
import bbox_to_mesh as b2m                                    # noqa: E402
import sam3_masks as s3                                       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def find_sam_python(explicit=None):
    """找跑 SAM 3 的解释器。**必须和当前环境分开** —— 当前是 basic_env,没 torch。"""
    if explicit:
        if not os.path.isfile(explicit):
            raise SystemExit(f'--sam-python 指向的文件不存在: {explicit}')
        return explicit

    cands = [
        r'D:\ProgramData\anaconda3\envs\sam3\python.exe',
        os.path.expanduser(r'~\anaconda3\envs\sam3\python.exe'),
        os.path.expanduser(r'~\.conda\envs\sam3\python.exe'),
    ]
    # 当前 env 的兄弟目录(conda 的 envs 通常都放一起)
    prefix = os.environ.get('CONDA_PREFIX')
    if prefix:
        cands.append(os.path.join(os.path.dirname(prefix), 'sam3', 'python.exe'))

    for c in cands:
        if os.path.isfile(c):
            return c

    # 最后问 conda 自己
    try:
        out = subprocess.run(['conda', 'env', 'list'], capture_output=True,
                             text=True, timeout=30).stdout
        for line in out.splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            name = line.split()[0]
            if os.path.basename(name).lower() == 'sam3':
                py = os.path.join(name, 'python.exe')
                if os.path.isfile(py):
                    return py
    except Exception:
        pass

    raise SystemExit(
        '找不到跑 SAM 3 的那个 conda 环境的解释器(sam3)。\n'
        '    试过:\n' + '\n'.join(f'      {c}' for c in cands) + '\n'
        '加上 conda env list 里叫 sam3 的环境,都没有。\n'
        '    用 --sam-python 明确指定,比如:\n'
        '      --sam-python D:\\ProgramData\\anaconda3\\envs\\sam3\\python.exe\n'
        '    还没建的话见 docs\\SAM3生成mask.md 第三节。')


def resolve_scene(scene_arg):
    """--scene 可以给场景名(food),也可以给 sparse 目录的完整路径。"""
    if os.sep in scene_arg or '/' in scene_arg:
        sparse = os.path.abspath(scene_arg)
        if not os.path.isdir(sparse):
            raise SystemExit(f'--scene 指的目录不存在: {sparse}')
        # ...\<场景>\dense\sparse
        scene_dir = os.path.dirname(os.path.dirname(sparse))
        name = os.path.basename(scene_dir)
    else:
        name = scene_arg
        sparse = os.path.join(ROOT, 'work', name, 'dense', 'sparse')
        if not os.path.isdir(sparse):
            raise SystemExit(
                f'场景 {name} 的 sparse 目录不存在: {sparse}\n'
                f'    按约定场景应该在 <仓库>\\work\\<场景>\\dense\\sparse。\n'
                f'    目录布局不同的话,--scene 直接给 sparse 的完整路径。')
    return name, sparse


def check_masks(mask_dir, stems):
    """返回缺哪些图的 mask。空的 mask(全 0 像素)也算数 —— 那是「这张图没有」,
    不是「还没跑」,不该逼着重跑 SAM 3。"""
    missing = []
    for s in stems:
        p = os.path.join(mask_dir, s + '.png')
        if not os.path.isfile(p) or os.path.getsize(p) == 0:
            missing.append(s)
    return missing


def overlay_dir_for(name):
    """2D 叠加图目录:<仓库>/out/<场景>/overlay。"""
    return os.path.join(ROOT, 'out', name, 'overlay')


def _images_by_stem(images_dir):
    by = {}
    for fn in sorted(os.listdir(images_dir)):
        stem, ext = os.path.splitext(fn)
        if ext.lower() in s3.IMG_EXTS:
            by[stem] = os.path.join(images_dir, fn)
    return by


def write_overlays(mask_dir, images_dir, stems, texts, out_dir, force=False):
    """把索引 mask 半透明叠到去畸变图上,供肉眼验收。

    已存在的跳过;--remask 或概念集合变了就整体重写 —— 否则同一个
    out/<场景>/overlay 下留着上一次概念的旧图,颜色和类别都对不上。
    """
    from PIL import Image

    os.makedirs(out_dir, exist_ok=True)
    manifest = os.path.join(out_dir, '.concepts.json')
    want = {'texts': list(texts)}
    try:
        with open(manifest, encoding='utf-8') as f:
            old = json.load(f)
    except Exception:
        old = None
    if old != want and (old is not None or any(
            os.path.isfile(os.path.join(out_dir, s + '.png')) for s in stems)):
        force = True

    by_stem = _images_by_stem(images_dir)
    n_w = n_s = n_missing = 0
    for i, stem in enumerate(stems, 1):
        mask_p = os.path.join(mask_dir, stem + '.png')
        img_p = by_stem.get(stem)
        if not os.path.isfile(mask_p) or not img_p:
            n_missing += 1
            continue
        ov_p = os.path.join(out_dir, stem + '.png')
        if os.path.isfile(ov_p) and not force:
            n_s += 1
            continue
        with Image.open(mask_p) as im:
            idx = np.asarray(im)
        if idx.ndim != 2:
            raise SystemExit(
                f'索引 mask 不是单通道: {mask_p} shape={idx.shape}\n'
                f'    这个目录里的 png 必须是 sam3_masks.py 生成的 uint8 索引图。')
        s3.save_overlay(ov_p, s3.make_overlay(img_p, idx, len(texts)))
        n_w += 1
        print(f'  叠加图 [{i}/{len(stems)}] {stem}')

    with open(manifest, 'w', encoding='utf-8') as f:
        json.dump(want, f, ensure_ascii=False, indent=2)
    print(f'  叠加图: 写 {n_w} / 跳过 {n_s} / 缺 mask 或原图 {n_missing}')
    print(f'  目录  : {out_dir}')


def run(cmd, what):
    print(f'\n{"=" * 70}\n{what}\n{"=" * 70}', flush=True)
    print('  ' + ' '.join(f'"{c}"' if ' ' in str(c) else str(c) for c in cmd), flush=True)
    print(flush=True)
    rc = subprocess.call([str(c) for c in cmd])
    if rc != 0:
        raise SystemExit(f'\n{what} 失败,退出码 {rc}。上面的输出里有原因。')


def build_parser():
    ap = argparse.ArgumentParser(
        description='场景 + 关键词 → 直接弹 3D 窗口。',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='例:\n'
               '  seg_vis.py --scene food --text window\n'
               '  seg_vis.py --scene food --text window --text door\n')
    ap.add_argument('--scene', required=True,
                    help='场景名(如 food,自动推 work\\<场景>\\dense\\sparse),'
                         '或 sparse 目录的完整路径')
    ap.add_argument('--text', action='append', default=None, required=True,
                    help='识别关键词,可重复。**顺序就是类别 id 的顺序**,'
                         '也是窗口里颜色的顺序')
    ap.add_argument('--min-conf', dest='min_conf', type=float, default=0.9,
                    help='一致比例 n_vote/n_visible 下限(默认 0.9)。'
                         '默认的并集会把 mask 边界抖动擦进来的点全带上,'
                         '点云上看起来「胖一圈」')
    ap.add_argument('--remask', action='store_true',
                    help='强制重跑 SAM 3 出 mask(默认已有的直接复用)')
    ap.add_argument('--overlay', action='store_true',
                    help='保存 2D 叠加图到 <仓库>\\out\\<场景>\\overlay(默认不写)')
    ap.add_argument('--overlay-dir', dest='overlay_dir', default=None,
                    help='自定义叠加图目录(隐含 --overlay)')
    ap.add_argument('--sam-python', dest='sam_python', default=None,
                    help='sam3 环境的 python.exe;默认自动找')
    ap.add_argument('--conf', type=float, default=0.25, help='SAM 3 置信度阈值(默认 0.25)')
    ap.add_argument('--imgsz', type=int, default=1008, help='SAM 3 输入尺寸(默认 1008)')
    ap.add_argument('--device', default=None, help='传给 SAM 3,如 0 / cpu;默认自动')
    ap.add_argument('--priority', choices=['first', 'last'], default='first',
                    help='多概念重叠时谁赢:first=先列出的(默认)')
    return ap


def main():
    args = build_parser().parse_args()
    texts = [t.strip() for t in args.text if t.strip()]
    if not texts:
        raise SystemExit('--text 全是空白,没概念可认。')

    name, sparse = resolve_scene(args.scene)
    images = m.default_images_dir(sparse)
    cloud = m.default_cloud(sparse)
    mesh = m.default_occlusion_mesh(sparse)
    overlay_dir = args.overlay_dir or (overlay_dir_for(name) if args.overlay else None)

    # ---------- 路径体检:缺哪个就明确说哪个,别拖到子进程里才炸 ----------
    print('=' * 70)
    print(f'  场景    : {name}')
    print(f'  sparse  : {sparse}')
    print(f'  去畸变图: {images}')
    print(f'  点云    : {cloud}')
    print(f'  遮挡网格: {mesh}')
    print(f'  叠加图  : {overlay_dir or "(不写)"}')
    print(f'  概念    : ' + ', '.join(f'{i+1}={t}' for i, t in enumerate(texts)))
    print('=' * 70)

    for p, what, kind, hint in (
            (images, '去畸变图目录', 'dir', '见 docs\\2D框投影到3D模型.md'),
            (cloud, '稠密点云', 'file', '被 .gitignore 排除了,要重跑 DensifyPointCloud'),
            (mesh, '遮挡网格', 'file', '同上,要跑 ReconstructMesh'),
    ):
        ok = os.path.isdir(p) if kind == 'dir' else os.path.isfile(p)
        if not ok:
            raise SystemExit(f'{what}不存在: {p}\n    {hint}')

    scene = b2m.load_scene(sparse)
    if not scene:
        raise SystemExit(f'{sparse} 里没读到任何相机,COLMAP 结果不完整?')
    first = next(iter(scene.values()))
    W, H = int(first['W']), int(first['H'])
    stems = sorted(os.path.splitext(n)[0] for n in scene)
    print(f'\n  相机 {len(scene)} 个视角,图像尺寸 {W} x {H}')

    # ---------- mask 缓存 ----------
    mask_dir = os.path.join(ROOT, 'masks', name, '+'.join(texts))
    missing = check_masks(mask_dir, stems)

    if args.remask:
        print(f'\n  --remask:重跑 SAM 3({len(stems)} 张)')
    elif missing:
        print(f'\n  mask 缓存 {mask_dir}')
        print(f'  缺 {len(missing)}/{len(stems)} 张{missing[:5]}'
              + (' …' if len(missing) > 5 else '') + ' → 跑 SAM 3')
    else:
        print(f'\n  mask 缓存齐全({len(stems)}/{len(stems)} 张),跳过 SAM 3')
        print(f'  {mask_dir}')

    if args.remask or missing:
        sam = find_sam_python(args.sam_python)
        cmd = [sam, os.path.join(HERE, 'sam3_masks.py'),
               '--images', images, '--out', mask_dir,
               '--expect-size', W, H,
               '--conf', args.conf, '--imgsz', args.imgsz,
               '--priority', args.priority,
               # 叠加图统一由本工具写(路径按场景,而不是概念组合名);
               # sam3_masks 只管出 mask。
               '--no-overlay']
        for t in texts:
            cmd += ['--text', t]
        if args.device:
            cmd += ['--device', args.device]
        if args.remask:
            cmd.append('--overwrite')
        run(cmd, '第 1/3 步:出 mask(SAM 3,独立环境)')

    # ---------- 2D 叠加图(从缓存的 mask 直接画,不用再过一遍模型) ----------
    if overlay_dir:
        print(f'\n  叠加图目录: {overlay_dir}')
        write_overlays(mask_dir, images, stems, texts, overlay_dir,
                       force=args.remask)

    # ---------- 融合 + 弹窗 ----------
    cmd = [sys.executable, os.path.join(HERE, 'mask_to_points.py'),
           '--scene', sparse,
           '--mask-dir', mask_dir,
           '--mask-mode', 'index',
           '--class-names', ','.join(texts),
           '--min-conf', args.min_conf,
           # 一个输出参数都不给 —— 工具就真的什么都不写,只弹窗
           '--vis']
    run(cmd, '第 3/3 步:回灌点云 + 弹窗(basic_env)')
    if overlay_dir:
        print(f'\n叠加图: {overlay_dir}')
    print('窗口关掉就结束了。要重跑 SAM 3 加 --remask。')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
