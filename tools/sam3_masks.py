# -*- coding: utf-8 -*-
"""
SAM 3 出 mask:把去畸变图 + 文字概念,变成 mask_to_points.py 能直接吃的索引图。

这是「2D 识图模型 → 3D 点云分割」链路的最上游那一步:

    data\\food\\            (原图 3648x2736)
        │  COLMAP 去畸变
        ▼
    work\\food\\dense\\images\\IMG_xxxx.JPG   (3781x2788)   ← SAM 3 跑这里
        │  tools\\sam3_masks.py --text "window"
        ▼
    masks\\IMG_xxxx.png    uint8 单通道,0=背景,k=第 k 个 --text
        │  tools\\mask_to_points.py --mask-dir masks --mask-mode index
        ▼
    点云逐点类别标签

> 场景名叫 `food`,但**它是食品学院的楼,不是食物**。首批概念是 `window`。

四件事值得先说清楚:

1. **一个概念 = 一个类别 id = 一张图里该类所有实例的并集**。一栋楼有几十扇窗,
   它们是同一个类别 1。下游 mask_to_points 的 `--class-names` 是按**类别 id**
   取名字的(`names[val-1]`),所以 id 必须是「概念」而不是「实例」,而且**跨图一致**。
   别试图去区分单个窗户 —— 多视角投票是按类投的,实例 id 跨图会漂移,只会打碎投票。

2. **重叠像素的归属由 `--priority` 定,默认「先列出的概念优先」**。用户天然会按
   「具体 → 宽泛」写,`--text window wall` 读起来就是对的;反过来「后列优先」会让
   墙一口吃掉所有窗户。重叠像素数会打印出来 —— 光看索引图看不出有没有重叠。

3. **输出必须是单通道 uint8 PNG**。下游 `load_label_mask(mode='index')` 用
   `np.asarray(Image.open(p))` 读,读到 3 维直接报错退出。尺寸必须严格等于去畸变图,
   **不能缩放**(把 `--images` 指到原图目录 `data\\food` 是最典型的误用,`--expect-size`
   会在第一张图上就拦住)。

4. **只 import numpy 和 PIL,不碰 bbox_to_mesh / open3d / trimesh**。这个工具跑在
   独立的 `sam3` conda 环境里(要 torch+ultralytics+GPU),而 COLMAP/open3d 那条线
   在 `basic_env`。两个环境靠 PNG 文件交接,不要互相 import。

用法(先 `conda activate sam3`,然后):

    $sam  = "D:\\ProgramData\\anaconda3\\envs\\sam3\\python.exe"
    $root = "D:\\code\\point_cloud\\0916"

    # 自检:纯合成,不需要模型和 GPU
    & $sam "$root\\tools\\sam3_masks.py" --selftest

    # 单张试 + 叠加图肉眼验收(先跑这个!)
    & $sam "$root\\tools\\sam3_masks.py" `
        --images "$root\\work\\food\\dense\\images" --out "$root\\masks" `
        --text "window" --expect-size 3781 2788 `
        --limit 1 --overlay-dir "$root\\out\\sam_overlay"

    # 批量(默认跳过已经生成过的,断了可以接着跑)
    & $sam "$root\\tools\\sam3_masks.py" `
        --images "$root\\work\\food\\dense\\images" --out "$root\\masks" `
        --text "window" --expect-size 3781 2788 `
        --overlay-dir "$root\\out\\sam_overlay" --save-json "$root\\out\\sam_report.json"

--model / --bpe 不写会按候选路径找。sam3.pt **不会自动下载**,找不到会直接报错。
"""
import argparse
import gc
import glob
import json
import os
import sys
import time

import numpy as np

# Windows 控制台中文输出。stderr 也要设 —— 报错信息(raise SystemExit)走的是
# stderr,它默认用 GBK,遇到 `——`、`⚠️` 这类字符会直接崩或者变成乱码。
# 仓库里 bbox_to_mesh.py 只设了 stdout,所以它的报错信息其实是会乱码的。
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
try:
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass


# =============================================================================
# 一、配色
# =============================================================================
# 和 mask_to_points.PALETTE **手工同步** —— 故意不 import 它,那个模块会连带
# 拖进 open3d/trimesh,而这里跑在只有 torch 的 sam3 环境里。
# 类别 1 是青色(仓库「青色 = 被选中的那部分」的颜色语言),后面是不同的色相。
# 两边颜色一致,2D 叠加图和点云图例就能直接对上,人脑不用做映射。
PALETTE = [
    (0.00, 0.90, 1.00),      # 青  ← 类别 1
    (1.00, 0.45, 0.10),      # 橙
    (0.45, 0.90, 0.25),      # 黄绿
    (0.95, 0.20, 0.75),      # 品红
    (0.99, 0.85, 0.15),      # 黄
    (0.55, 0.35, 0.95),      # 紫
    (0.10, 0.80, 0.45),      # 绿
    (0.95, 0.30, 0.30),      # 红
    (0.35, 0.65, 0.95),      # 蓝
    (0.70, 0.70, 0.30),      # 橄榄
]

# uint8 索引图装不下超过 255 个类别
MAX_CLASSES = 255

# 掩膜的原始分辨率比去畸变图小这么多倍以上时警告 —— 说明模型在低分辨率出图,
# 窗户这种小目标会被抹掉。retina_masks 应该能避免,但值得盯一下。
WARN_UPSCALE = 4.0

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')


# =============================================================================
# 二、路径
# =============================================================================
def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _default_images_dir(scene_dir):
    """scene 是 <...>/<场景>/dense/sparse,去畸变图在 <场景>/dense/images。"""
    root = os.path.dirname(os.path.dirname(os.path.abspath(scene_dir)))
    return os.path.join(root, 'dense', 'images')


def _weight_candidates(name):
    """sam3.pt / bpe 的候选路径。权重有 3.45 GB,默认放仓库外,别进 git。"""
    home = os.path.expanduser('~')
    out = [
        os.path.join('D:\\models\\sam3', name),
        os.path.join(_repo_root(), 'weights', name),
        os.path.join(home, '.cache', 'sam3', name),
    ]
    out += glob.glob(os.path.join(
        home, '.cache', 'huggingface', 'hub', 'models--facebook--sam3',
        'snapshots', '*', name))
    return out


def find_weight(cli_path, name, what, hint):
    """--model/--bpe 没给就按候选路径找;找不到列出找过哪些地方。"""
    if cli_path:
        if not os.path.isfile(cli_path):
            raise SystemExit(f'{what} 不存在: {cli_path}')
        return cli_path
    cands = _weight_candidates(name)
    for c in cands:
        if os.path.isfile(c):
            return c
    raise SystemExit(
        f'找不到 {what}。找过这些地方:\n'
        + ''.join(f'    {c}\n' for c in cands)
        + f'  {what} 不会自动下载,得手动放到位。{hint}')


def list_images(d):
    if not os.path.isdir(d):
        raise SystemExit(
            f'图像目录不存在: {d}\n'
            f'    这里要的是**去畸变图**目录 <场景>\\dense\\images\\。\n'
            f'    常见误用是指到原图目录 data\\<场景>\\(尺寸不一样,去畸变是非线性形变)。')
    names = [fn for fn in sorted(os.listdir(d))
             if os.path.splitext(fn)[1].lower() in IMG_EXTS]
    if not names:
        raise SystemExit(f'{d} 里一张图都没有(只认 {", ".join(IMG_EXTS)})。')
    return [os.path.join(d, fn) for fn in names]


def read_image_size(path):
    from PIL import Image
    with Image.open(path) as im:
        return int(im.size[0]), int(im.size[1]), im.mode


# =============================================================================
# 三、光栅化
# =============================================================================
class Inst:
    """一个 SAM 检测实例。cls 是**概念下标**(0 起),不是类别 id。"""

    __slots__ = ('cls', 'mask', 'conf', 'box')

    def __init__(self, cls, mask, conf, box):
        self.cls = int(cls)
        self.mask = mask
        self.conf = float(conf)
        self.box = box


def to_size(mask, H, W):
    """把掩膜搬到 (H, W)。**只用最近邻** —— 掩膜是布尔的,双线性会把边界糊成
    半透明灰阶,再被 `>0.5` 二值化时边界漂移,小目标(窗户)会缩掉或糊掉。"""
    m = np.asarray(mask)
    if m.dtype != np.bool_:
        m = m > 0.5
    if m.shape == (H, W):
        return m
    from PIL import Image
    up = (m * 255).astype(np.uint8)
    up = np.asarray(Image.fromarray(up).resize((W, H), Image.NEAREST))
    return up > 127


def rasterize(insts, H, W, n_classes, priority):
    """把实例光栅化成 int 索引图:0 = 背景,k+1 = 第 k 个概念。

    同一概念的多个实例取**并集**(一栋楼的窗都是同一个类别)。

    重叠走「只填还没被占的像素」:按 order 依次画,谁先画谁占住。
    priority='first' → order 是 0,1,2...(先列出的概念先画 → 先列出的赢)
    priority='last'  → order 反过来(后列出的概念先画 → 后列出的赢)
    """
    idx = np.zeros((H, W), np.uint8)
    touched = np.zeros((H, W), np.uint8)      # 每个像素被几个概念碰到,用来数重叠
    order = range(n_classes) if priority == 'first' else range(n_classes - 1, -1, -1)

    inst_count = {}
    for k in order:
        m = None
        n = 0
        for inst in insts:
            if inst.cls != k:
                continue
            mm = to_size(inst.mask, H, W)
            m = mm if m is None else (m | mm)
            n += 1
        inst_count[k] = n
        if m is None or not m.any():
            continue
        np.copyto(idx, k + 1, where=(m & (idx == 0)))
        touched[m] += 1

    src = None
    if insts:
        src = tuple(int(v) for v in np.asarray(insts[0].mask).shape[-2:])
    stats = {
        'n_instances': {k: inst_count.get(k, 0) for k in range(n_classes)},
        'n_overlap_px': int((touched > 1).sum()),
        'mask_src_size': src,
    }
    return idx, stats


# =============================================================================
# 四、输出
# =============================================================================
def save_index_png(path, idx):
    from PIL import Image
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    Image.fromarray(np.ascontiguousarray(idx, dtype=np.uint8), mode='L').save(path)


def make_overlay(image_path, idx, n_classes, alpha=0.5):
    """把索引图半透明叠到原图上,肉眼验收用。"""
    from PIL import Image
    with Image.open(image_path) as im:
        rgb = np.asarray(im.convert('RGB')).astype(np.float32)
    out = rgb.copy()
    for k in range(n_classes):
        m = idx == (k + 1)
        if not m.any():
            continue
        col = np.asarray(PALETTE[k % len(PALETTE)], dtype=np.float32) * 255.0
        out[m] = (1.0 - alpha) * out[m] + alpha * col
    return out.astype(np.uint8)


def save_overlay(path, img_u8):
    from PIL import Image
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    Image.fromarray(img_u8).save(path)


# =============================================================================
# 五、模型
# =============================================================================
def _torch():
    import torch
    return torch


def build_predictor(args, model, bpe, imgsz):
    """建一次,复用所有图 —— 加载 3.45 GB 权重是这一整步里最贵的操作。"""
    from ultralytics.models.sam import SAM3SemanticPredictor
    overrides = dict(model=model, task='segment', mode='predict',
                     conf=args.conf, imgsz=imgsz, verbose=False,
                     # 要**原尺寸**的掩膜:不然掩膜是模型内部 1/4 分辨率的小图,
                     # 放大 3800 多像素时窗户会糊成一团。
                     retina_masks=True,
                     # 我们只要内存里的 Results,不要它往磁盘上画图。
                     # 开着会在仓库里堆出 runs\segment\predict\ 一堆没用的预览图。
                     save=False)
    if args.half:
        overrides['half'] = True
    if args.device:
        overrides['device'] = args.device
    try:
        return SAM3SemanticPredictor(overrides=overrides, bpe_path=bpe)
    except TypeError:
        # 有些版本不接 bpe_path 参数(自己会去找),退回默认构造
        return SAM3SemanticPredictor(overrides=overrides)


def predict(predictor, img_path, texts):
    """跑一张图。set_image + __call__ 是官方文档的写法,source= 是兜底。"""
    try:
        predictor.set_image(img_path)
        return predictor(text=list(texts))
    except (TypeError, AttributeError):
        return predictor(source=img_path, text=list(texts))


def names_to_dict(raw):
    """把 `r.names` 归一成 {id: 名字}。

    **SAM3 的 `r.names` 是 list 不是 dict** —— `sam3_image.set_classes()` 里写的是
    `self.names = text`,存的就是传进去的概念列表,顺序一字不差。所以这里 `enumerate`
    出来的下标,正好就是 `boxes.cls` 的值,也就是 `--text` 的下标。
    (直接 `dict(r.names)` 会炸:`dict(['window'])` 报「元素长度 6,需要 2」。)
    """
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return {int(k): str(v) for k, v in raw.items()}
    if isinstance(raw, (list, tuple)):
        return dict(enumerate(str(v) for v in raw))
    if hasattr(raw, 'tolist'):                      # numpy 数组
        return dict(enumerate(str(v) for v in raw.tolist()))
    return {}


def extract_instances(res):
    """从 Results 里抠出实例列表。masks.data / boxes.cls 是一一对应的。"""
    r = res[0] if isinstance(res, (list, tuple)) else res
    if r is None or getattr(r, 'masks', None) is None or getattr(r, 'boxes', None) is None:
        return [], {}
    boxes = r.boxes
    if len(boxes) == 0:
        return [], names_to_dict(getattr(r, 'names', None))

    data = r.masks.data
    cls = boxes.cls
    conf = boxes.conf
    if hasattr(data, 'cpu'):
        data = data.cpu().numpy()
    if hasattr(cls, 'cpu'):
        cls = cls.cpu().numpy()
    if hasattr(conf, 'cpu'):
        conf = conf.cpu().numpy()
    data = np.asarray(data)
    cls = np.asarray(cls).astype(int).ravel()
    conf = np.asarray(conf).astype(float).ravel()

    boxes_xyxy = boxes.xyxy
    if hasattr(boxes_xyxy, 'cpu'):
        boxes_xyxy = boxes_xyxy.cpu().numpy()
    boxes_xyxy = np.asarray(boxes_xyxy).astype(float)

    insts = []
    for i in range(len(cls)):
        insts.append(Inst(cls[i], data[i], conf[i],
                          tuple(float(v) for v in boxes_xyxy[i])))
    return insts, names_to_dict(getattr(r, 'names', None))


def predict_with_oom_retry(predictor, img_path, texts, args, model, bpe, state):
    """显存不够时降 imgsz 重试。一张一张处理、不累积结果,是省显存的第一原则。"""
    torch = _torch()
    while True:
        try:
            return predict(predictor, img_path, texts), predictor
        except Exception as e:                     # noqa: BLE001
            msg = str(e).lower()
            oom = 'out of memory' in msg or type(e).__name__ == 'OutOfMemoryError'
            if not oom or state['tries'] >= 2:
                raise
            state['tries'] += 1
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            state['imgsz'] = max(256, int(state['imgsz'] * 0.75))
            print(f'  ⚠️  显存不够,把 imgsz 降到 {state["imgsz"]} 重试'
                  f'(第 {state["tries"]} 次)')
            predictor = build_predictor(args, model, bpe, state['imgsz'])
            # 重建之后要重新量一次最大显存占用
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()


# =============================================================================
# 六、自检(纯合成,不需要模型和 GPU)
# =============================================================================
def selftest():
    """钉死两条最容易**静默**出错的不变量。打 ✅/❌,退出码 0/1。"""
    ok = bad = 0

    def check(cond, msg):
        nonlocal ok, bad
        if cond:
            ok += 1
            print(f'  ✅ {msg}')
        else:
            bad += 1
            print(f'  ❌ {msg}')

    def square(r0, r1, c0, c1, H, W):
        m = np.zeros((H, W), np.bool_)
        m[r0:r1, c0:c1] = True
        return m

    H = W = 200
    # 概念 0 是个大块;概念 1 是个小块,**完全嵌在**它里面 —— 重叠 50x50 = 2500 px
    big = square(0, 100, 0, 100, H, W)
    nested = square(50, 100, 50, 100, H, W)
    insts = [Inst(0, big, 0.9, (0, 0, 100, 100)),
             Inst(1, nested, 0.8, (50, 50, 100, 100))]

    print('\n一、重叠裁决(概念 0 大块 + 概念 1 小块嵌在它里面,重叠 2500 px)')
    idx_f, st_f = rasterize(insts, H, W, 2, 'first')
    check(st_f['n_overlap_px'] == 2500,
          f"--priority first 数出重叠 2500 px(实际 {st_f['n_overlap_px']})")
    check(bool((idx_f[50:100, 50:100] == 1).all()),
          '--priority first:重叠区归**先列出的**概念 0')
    # 嵌套的小块在 first 下会被整个吃掉 —— 这是规则的直接后果,不是 bug。
    # 报告里会显示「实例数 > 0 但像素数 = 0」,就是这种情况的信号。
    check(int((idx_f == 2).sum()) == 0,
          '--priority first:被完全盖住的概念 1 剩 0 px(实例数仍会报 1 个)')

    idx_l, st_l = rasterize(insts, H, W, 2, 'last')
    check(st_l['n_overlap_px'] == 2500,
          f"--priority last 数出重叠 2500 px(实际 {st_l['n_overlap_px']})")
    check(bool((idx_l[50:100, 50:100] == 2).all()),
          '--priority last:重叠区归**后列出的**概念 1')

    print('\n二、掩膜放大不产生新 id')
    # 掩膜在 50x50 上,目标 200x200 —— 走 to_size 的最近邻那条路
    insts_small = [
        Inst(0, square(0, 25, 0, 25, 50, 50), 0.9, (0, 0, 25, 25)),
        Inst(1, square(12, 25, 12, 25, 50, 50), 0.8, (12, 12, 25, 25)),
    ]
    idx_s, _ = rasterize(insts_small, H, W, 2, 'first')
    vals = set(np.unique(idx_s).tolist())
    check(vals <= {0, 1, 2}, f'输出值集 {sorted(vals)} ⊆ {{0,1,2}}(没有插值造出的中间值)')
    check(idx_s.ndim == 2, f'索引图是 2 维(实际 {idx_s.ndim} 维 —— 下游读到 3 维直接报错)')
    check(idx_s.dtype == np.uint8, f'dtype 是 uint8(实际 {idx_s.dtype})')
    check(idx_s.shape == (H, W), f'尺寸是 ({H}, {W})(实际 {idx_s.shape})')

    print('\n三、边界情况')
    idx_empty, st_e = rasterize([], H, W, 2, 'first')
    check(int(idx_empty.max()) == 0, '一个实例都没有 → 全 0 索引图,不崩')
    check(st_e['n_overlap_px'] == 0, '没有实例 → 重叠数 0')

    # 给 3 个概念但第 3 个没实例;前两个**部分重叠**,保证两个都能留下像素
    partial = [Inst(0, square(0, 100, 0, 100, H, W), 0.9, (0, 0, 100, 100)),
               Inst(1, square(60, 160, 60, 160, H, W), 0.8, (60, 60, 160, 160))]
    idx_third, st_third = rasterize(partial, H, W, 3, 'first')
    check(int(idx_third.max()) == 2, '给了 3 个概念但第 3 个没实例 → 最大值是 2')
    check(st_third['n_instances'][2] == 0, '第 3 个概念实例数报 0')
    check(st_third['n_overlap_px'] == 1600,
          f"部分重叠数出 1600 px(实际 {st_third['n_overlap_px']})")

    print(f'\n通过 {ok} 项,失败 {bad} 项')
    if bad == 0:
        print('✅ 自检通过')
    else:
        print('❌ 自检失败')
    return bad == 0


# =============================================================================
# 七、主流程
# =============================================================================
def parse_texts(text_args, text_file):
    """收集概念,顺序即类别 id。

    `--text` 既可以重复写(`--text a --text b`),也可以用逗号分隔
    (`--text "a,b"`)。两种写法等价 —— 文档里一直写的是逗号形式,
    但 argparse 的 action='append' 不会自动拆,所以在这里统一拆。
    """
    raw = [t for t in (text_args or [])]
    if text_file:
        if not os.path.isfile(text_file):
            raise SystemExit(f'--text-file 不存在: {text_file}')
        with open(text_file, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    raw.append(line)
    out = []
    for item in raw:
        # 同时接受英文逗号和中文逗号
        for part in item.replace('，', ',').split(','):
            part = part.strip()
            if part:
                out.append(part)
    return out


def build_parser():
    ap = argparse.ArgumentParser(
        description='SAM 3 出 mask:去畸变图 + 文字概念 → 单通道索引图(喂给 mask_to_points.py)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='概念 id:第 1 个 --text 是类别 1,第 2 个是 2……\n'
               '下游要对应上:--class-names 按同样的顺序写。')

    ap.add_argument('--images', default=None, help='去畸变图目录(不写则从 --scene 推)')
    ap.add_argument('--scene', default=None,
                    help='COLMAP sparse 目录,只用来推 --images')
    ap.add_argument('--out', default=None, help='mask 输出目录(必填)')

    ap.add_argument('--text', action='append', default=None,
                    help='文字概念,**可重复**,也可用逗号分隔。顺序即类别 id 顺序')
    ap.add_argument('--text-file', dest='text_file', default=None,
                    help='每行一个概念,# 开头是注释')

    ap.add_argument('--model', default=None, help='sam3.pt 路径')
    ap.add_argument('--bpe', default=None, help='bpe_simple_vocab_16e6.txt.gz 路径')
    ap.add_argument('--conf', type=float, default=0.25, help='置信度阈值(默认 0.25)')
    ap.add_argument('--imgsz', type=int, default=1008, help='推理分辨率(默认 1008)')
    ap.add_argument('--half', action='store_true', help='半精度推,更快更省显存')
    ap.add_argument('--device', default=None, help='cuda / cpu / 0(默认自动)')

    ap.add_argument('--expect-size', dest='expect_size', type=int, nargs=2,
                    default=None, metavar=('W', 'H'),
                    help='期望的图像尺寸,如 3781 2788。第一张图就硬校验')
    ap.add_argument('--priority', choices=['first', 'last'], default='first',
                    help='重叠像素归谁。first = 先列出的概念优先(默认)')

    ap.add_argument('--limit', type=int, default=None, help='只处理前 N 张')
    ap.add_argument('--overwrite', action='store_true',
                    help='重跑已经生成过的 mask(默认跳过,方便断点续跑)')
    ap.add_argument('--no-write-empty', dest='no_write_empty', action='store_true',
                    help='一个实例都没检测到的图不写 png(默认照写全 0 的)')

    ap.add_argument('--overlay-dir', dest='overlay_dir', default=None,
                    help='半透明叠加图存这个目录(可选;默认不写)')
    ap.add_argument('--no-overlay', dest='no_overlay', action='store_true',
                    help='不写叠加图(默认就不写,保留此开关兼容旧命令)')
    ap.add_argument('--save-json', dest='save_json', default=None, help='统计存 json')
    ap.add_argument('--selftest', action='store_true',
                    help='纯合成自检,不需要模型和 GPU')
    return ap


def main():
    args = build_parser().parse_args()

    if args.selftest:
        raise SystemExit(0 if selftest() else 1)

    if not args.out:
        raise SystemExit('--out 必填 —— 得给个目录放生成的 mask。')
    if not args.images and not args.scene:
        raise SystemExit('要给 --images 去畸变图目录,或者给 --scene 让我推。')

    texts = parse_texts(args.text, args.text_file)
    if not texts:
        raise SystemExit('至少给一个 --text 概念。比如 --text "window"')
    if len(texts) > MAX_CLASSES:
        raise SystemExit(
            f'概念有 {len(texts)} 个,超过 {MAX_CLASSES} —— uint8 索引图装不下。\n'
            f'    (下游 mask_to_points 的 WARN_N_CLASSES 是 64,真到那么多类投票表也会很大)')

    images_dir = args.images or _default_images_dir(args.scene)

    # 默认只出 mask,不写叠加图;显式 --overlay-dir 才写。
    if args.no_overlay:
        args.overlay_dir = None

    model = find_weight(args.model, 'sam3.pt', 'SAM 3 权重',
                        'ModelScope: modelscope download --model facebook/sam3 '
                        '--local_dir D:\\models\\sam3')
    bpe = find_weight(args.bpe, 'bpe_simple_vocab_16e6.txt.gz', 'BPE 词表',
                      '从 clip 包里拿:python -c "import clip,os,shutil;'
                      'shutil.copy(os.path.join(os.path.dirname(clip.__file__),'
                      '\'bpe_simple_vocab_16e6.txt.gz\'), r\'D:\\models\\sam3\')"')

    files = list_images(images_dir)
    if args.limit:
        files = files[:args.limit]

    print('=' * 70)
    print(f'  去畸变图: {images_dir}')
    print(f'  输出    : {args.out}')
    print(f'  叠加图  : {args.overlay_dir or "(不写)"}')
    print(f'  权重    : {model}')
    print(f'  BPE     : {bpe}')
    print(f'  概念    : {", ".join(f"{i + 1}={t}" for i, t in enumerate(texts))}')
    print(f'  重叠优先级: --priority {args.priority}')
    print(f'  待处理  : {len(files)} 张')
    print('=' * 70)

    # ---------- 第一张图就校尺寸,别一路跑到下游才炸 ----------
    W0, H0, mode0 = read_image_size(files[0])
    print(f'\n图像尺寸: {W0} x {H0} ({mode0})  ← {os.path.basename(files[0])}')
    if args.expect_size:
        ew, eh = args.expect_size
        if (W0, H0) != (ew, eh):
            raise SystemExit(
                f'图像尺寸对不上:\n'
                f'    {files[0]}\n'
                f'    读出来 {W0}x{H0},但 --expect-size 要的是 {ew}x{eh}。\n'
                f'    这里要的是**去畸变图**目录 <场景>\\dense\\images\\,'
                f'不是原图目录 data\\<场景>\\(原图是 3648x2736)。')
    for f in files:
        w, h, _ = read_image_size(f)
        if (w, h) != (W0, H0):
            raise SystemExit(
                f'{f} 的尺寸是 {w}x{h},和第一张的 {W0}x{H0} 不一样。\n'
                f'    同一批图必须同尺寸,否则下游的 (H,W) 校验会炸。')

    # ---------- 建模型(最贵的一步) ----------
    print(f'\n加载 SAM 3 权重({os.path.getsize(model) / 2**30:.2f} GB)……')
    t0 = time.time()
    state = {'imgsz': args.imgsz, 'tries': 0}
    predictor = build_predictor(args, model, bpe, state['imgsz'])
    print(f'  加载耗时 {time.time() - t0:.1f}s')

    # ---------- 逐张推 ----------
    print(f'\n{"=" * 70}\n逐张推理\n{"=" * 70}')
    report = {'texts': texts, 'priority': args.priority, 'model': model, 'bpe': bpe,
              'conf': args.conf, 'imgsz': args.imgsz,
              'size': [W0, H0], 'images': {}, 'skipped': []}
    class_px_total = {k: 0 for k in range(len(texts))}
    class_inst_total = {k: 0 for k in range(len(texts))}
    names_checked = False

    for i, path in enumerate(files, 1):
        stem = os.path.splitext(os.path.basename(path))[0]
        out_png = os.path.join(args.out, stem + '.png')

        if os.path.isfile(out_png) and not args.overwrite:
            print(f'  [{i}/{len(files)}] {stem}  [skip] 已存在')
            report['skipped'].append(stem)
            # 断点续跑时补写缺失的叠加图(不重跑推理)
            if args.overlay_dir:
                ov_path = os.path.join(args.overlay_dir, stem + '.png')
                if not os.path.isfile(ov_path):
                    from PIL import Image
                    with Image.open(out_png) as im:
                        idx_old = np.asarray(im)
                    save_overlay(ov_path, make_overlay(path, idx_old, len(texts)))
            continue

        t0 = time.time()
        res, predictor = predict_with_oom_retry(
            predictor, path, texts, args, model, bpe, state)
        t_inf = time.time() - t0

        insts, names = extract_instances(res)

        # r.names 的顺序应该和 --text 顺序一致。不一致的话下游按 id 取名字会
        # 整体错位,而且极难查 —— 所以第一张就断言并打印实际映射。
        if not names_checked and names:
            got = [names.get(k) for k in range(len(names))]
            if got != texts:
                print(f'  ⚠️  模型报的类名 {got} 和 --text {texts} 不一致。'
                      f'以 --text 的顺序为准,但请核对一下。')
            else:
                print(f'  类别映射: {names}')
            names_checked = True

        idx, st = rasterize(insts, H0, W0, len(texts), args.priority)
        # 尺寸是硬契约,而**转置是静默的**:下游只检查「单通道 uint8」,700x950 的图
        # 照样读得进去,只是所有像素都错位 —— 到点云上才发现,极难归因。钉死它。
        assert idx.shape == (H0, W0), \
            f'索引图尺寸 {idx.shape} 和图像 (H,W)=({H0},{W0}) 不符'
        n_px = int((idx > 0).sum())

        src = st['mask_src_size']
        upscale = ''
        if src and src != (H0, W0):
            r = max(H0 / src[0], W0 / src[1]) if src[0] and src[1] else 0
            upscale = f'  掩膜原尺寸 {src[0]}x{src[1]}(放大 {r:.1f}x)'
            if r > WARN_UPSCALE:
                upscale += ' ⚠️ 放得太狠,窗户这种小目标会被抹掉,试试 retina_masks'

        if idx.max() == 0 and args.no_write_empty:
            print(f'  [{i}/{len(files)}] {stem}  什么都没检测到,跳过不写({t_inf:.1f}s)')
            report['images'][stem] = {'empty': True, 'sec': round(t_inf, 2)}
            continue

        save_index_png(out_png, idx)
        if args.overlay_dir:
            ov = make_overlay(path, idx, len(texts))
            save_overlay(os.path.join(args.overlay_dir, stem + '.png'), ov)

        per_cls = {}
        for k in range(len(texts)):
            px = int((idx == k + 1).sum())
            per_cls[k] = px
            class_px_total[k] += px
            class_inst_total[k] += st['n_instances'][k]
        parts = []
        for k in range(len(texts)):
            n = st['n_instances'][k]
            s = f'{texts[k]}={per_cls[k]}px({n}个)'
            if per_cls[k] == 0 and n > 0:
                # 检测到了却被别的概念整个盖住 —— 光看像素数会以为「没检测到」
                s += '←被盖住'
            parts.append(s)
        detail = '  '.join(parts)
        print(f'  [{i}/{len(files)}] {stem}  {detail}  '
              f'总前景 {n_px}px  重叠 {st["n_overlap_px"]}px  {t_inf:.1f}s{upscale}')

        report['images'][stem] = {
            'sec': round(t_inf, 2),
            'pixels': {texts[k]: per_cls[k] for k in range(len(texts))},
            'instances': {texts[k]: st['n_instances'][k] for k in range(len(texts))},
            'n_foreground_px': n_px,
            'n_overlap_px': st['n_overlap_px'],
            'mask_src_size': list(src) if src else None,
            'empty_classes': [texts[k] for k in range(len(texts)) if per_cls[k] == 0],
        }

        del res, insts, idx
        gc.collect()

    # ---------- 汇总 ----------
    print(f'\n{"=" * 70}\n汇总\n{"=" * 70}')
    rep_tot = {}
    for k, name in enumerate(texts):
        rep_tot[name] = {'pixels': class_px_total[k], 'instances': class_inst_total[k]}
        print(f'  类别 {k + 1} = {name:<16s} 共 {class_px_total[k]} px,'
              f'{class_inst_total[k]} 个实例')
    # 某个概念全场一个像素都没有 → 大概率 prompt 词不对
    dead = [texts[k] for k in range(len(texts)) if class_px_total[k] == 0]
    if dead:
        print(f'\n  ⚠️  这些概念在所有图里都是 0 像素: {dead}\n'
              f'     大概率是提示词不对(SAM 3 是英文模型),换个说法再试。')
    if report['skipped']:
        print(f'  跳过 {len(report["skipped"])} 张(已存在)')
    print(f'\n  mask 输出目录: {args.out}')
    print(f'  下游接法: --mask-dir "{args.out}" --mask-mode index '
          f'--class-names "{",".join(texts)}"')
    if args.overlay_dir:
        print(f'  叠加图(先看这个!): {args.overlay_dir}')

    report['totals'] = rep_tot
    report['dead_concepts'] = dead
    if args.save_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_json)), exist_ok=True)
        with open(args.save_json, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f'  统计: {args.save_json}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
