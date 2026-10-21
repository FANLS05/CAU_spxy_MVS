# -*- coding: utf-8 -*-
"""
多视角 mask 融合到点云:每张(去畸变)图配一个 mask,给点云里每个点打类别标签。

这是「2D 识图模型 → 3D 点云分割」的那一步:

    data\\food\\  ──COLMAP 去畸变──►  work\\food\\dense\\images\\   (3781x2788)
                                             │
                                             │  识图模型逐张出 mask
                                             ▼
                                       masks\\IMG_xxxx.png
                                             │
    work\\food\\dense\\sparse\\   (位姿)  ───┤
    work\\food\\mvs\\scene_dense.ply      ───┤  (点云,要被标注的东西)
    work\\food\\mvs\\scene_dense_mesh.ply ───┘  (只用来做遮挡)
                                             ▼
                                       每个点一个类别标签

和 bbox_to_mesh.py 的区别:那边是「一个矩形框」,这边是一组不规则 mask,
而且是**多视角融合** —— 每个点取它在各个可见视角里被标成的类别,再投票。

四件事值得先说清楚:

1. **遮挡剔除用 Open3D 的 RaycastingScene**(CPU embree),不用 bbox_to_mesh 里的
   ray_triangle_hits。后者是逐条射线的 Python 循环(见 bbox_live.py 的 cast_rays),
   110 万个点根本跑不动 —— 实测 RaycastingScene 打 82 万条射线只要 0.61 秒。
   判据是「从光心连到点的这条射线上,有没有在更近的地方撞到网格」,撞到就是被挡了。

2. **mask 分辨率必须等于去畸变图的 3781x2788**,不是原图的 3648x2736。
   去畸变是非线性形变,缩放是错的。尺寸对不上直接报错,不做任何猜测。

3. **背景默认不投票**。一个表面点常常在某个视角落在 mask 外(模型标小了),
   如果背景也投票,可能反压过真类别。不投票的话这个点仍能从别的视角拿到类别,
   分歧记在 confidence 里。

4. **遮挡网格必须用稠密网格** scene_dense_mesh.ply(82 万顶点),不能用
   scene_dense_mesh_refine.ply(food 这份只有 7622 顶点,到处是洞,
   有洞的地方射线打不中,会被判成「可见」,遮挡就白做了)。
   所以这里自己写了个 default_occlusion_mesh,没用 bbox_gui.default_mesh ——
   后者偏好 refine 网格,那是给「配贴图」用的,不是给遮挡用的。

bbox_to_mesh.py 只读导入,一个字节都不改;它和 bbox_live.py 的逐位对比契约
(bbox_live.py --verify)因此不受影响。

用法(在 0916 目录下,用 conda basic_env 的 python):

    $py = "C:\\Users\\FANLS\\.conda\\envs\\basic_env\\python.exe"
    $root = "D:\\code\\point_cloud\\0916"

    # 自检:RaycastingScene 的可见性判定 vs 暴力求交,打印 ✅/❌,退出码 0/1
    & $py "$root\\tools\\mask_to_points.py" --verify `
        --scene "$root\\work\\food\\dense\\sparse" `
        --mesh  "$root\\work\\food\\mvs\\scene_dense_mesh.ply"

    # 单张:显式给 (图, mask) 对。注意是两个 token,不要写成 IMG:mask ——
    # Windows 盘符 D:\\ 自带冒号,拿冒号当分隔符必错
    & $py "$root\\tools\\mask_to_points.py" `
        --scene "$root\\work\\food\\dense\\sparse" `
        --pair "IMG_2435.JPG" "$root\\masks\\IMG_2435.png" `
        --out "$root\\out\\masked.ply" --labels "$root\\out\\labels.npy"

    # 批量:按文件名 stem 自动匹配场景里的所有图
    & $py "$root\\tools\\mask_to_points.py" `
        --scene "$root\\work\\food\\dense\\sparse" `
        --mask-dir "$root\\masks" --min-conf 0.9 `
        --out "$root\\out\\masked.ply" --save-npz "$root\\out\\full.npz"

--cloud / --mesh / --images 不写会自动从 --scene 推。

**压误标用 --min-conf 而不是 --min-votes。** 默认 --min-votes 1 取的是跨视角
**并集**,mask 边界在多视角间抖动擦进来的点全在里面。--min-votes 数的是绝对票数,
不看这个点被几个视角看到 —— 立面边缘只有少数相机看得见,票数天然少却完全正确,
会被它误杀。详见 docs\\mask点云分割.md 第三节。
"""
import argparse
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bbox_to_mesh as b2m                                   # noqa: E402


# =============================================================================
# 一、配色
# =============================================================================
# 类别 1 用 b2m.COL_FRAGMENT 的青色 —— 沿用仓库「青色 = 被选中的那部分」的
# 颜色语言(见 bbox_to_mesh.visualize 里碎片的颜色)。后面几类是不同的色相。
PALETTE = [
    b2m.COL_FRAGMENT,        # 0.00, 0.90, 1.00  青
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
COL_UNLABELED = (0.60, 0.60, 0.60)      # 没拿到标签的点,恒为灰

# 类别数超过这个值就提醒 —— counts 矩阵是 N x L,类别多了内存会涨
WARN_N_CLASSES = 64


def class_color(val):
    """类别 id -> 颜色。1 号是青色,之后按 PALETTE 循环。"""
    return PALETTE[(int(val) - 1) % len(PALETTE)]


# =============================================================================
# 二、路径推断
# =============================================================================
def _scene_root(scene_dir):
    """scene 是 <...>/<场景>/dense/sparse,场景根是往上两级。"""
    return os.path.dirname(os.path.dirname(os.path.abspath(scene_dir)))


def default_cloud(scene_dir):
    """稠密点云:**不是**网格,是 DensifyPointCloud 的产物。"""
    return os.path.join(_scene_root(scene_dir), 'mvs', 'scene_dense.ply')


def default_occlusion_mesh(scene_dir):
    """遮挡用的网格,**优先稠密网格**。

    和 bbox_gui.default_mesh 的取舍刚好相反:那边要的是「有配套贴图的那份」,
    所以偏好 refine;这边要的是「面最多、洞最少的那份」,所以偏好 dense。
    food 这份 refine 只有 7622 顶点,拿来遮挡等于没有遮挡。
    """
    mvs = os.path.join(_scene_root(scene_dir), 'mvs')
    for nm in ('scene_dense_mesh.ply', 'scene_dense_mesh_refine.ply'):
        p = os.path.join(mvs, nm)
        if os.path.isfile(p):
            return p
    return os.path.join(mvs, 'scene_dense_mesh.ply')


def default_images_dir(scene_dir):
    """去畸变图目录。注意**不是** data\\<场景>\\ 里的原图。"""
    par = os.path.dirname(os.path.abspath(scene_dir))
    for nm in ('images', os.path.join('dense', 'images')):
        p = os.path.join(par, nm)
        if os.path.isdir(p):
            return p
    return os.path.join(par, 'images')


# =============================================================================
# 三、读点云
# =============================================================================
def load_point_cloud(path):
    """读点云 ply,返回 (P (N,3) float64, cols (N,3) float64 或 None)。

    **不能用 trimesh**:OpenMVS 的 scene_dense.ply 带了自定义属性
    view_indices / view_weights,trimesh 读它直接抛
    `ValueError: PLY is unexpected length!`。open3d 能读。
    """
    import open3d as o3d
    if not os.path.isfile(path):
        raise SystemExit(f'点云文件不存在: {path}\n'
                         f'稠密点云被 .gitignore 排除了,需要重跑 DensifyPointCloud 才有。')
    t0 = time.time()
    pc = o3d.io.read_point_cloud(path)
    P = np.asarray(pc.points, dtype=np.float64)
    if len(P) == 0:
        raise SystemExit(f'{path} 里一个点都没有。')
    cols = np.asarray(pc.colors, dtype=np.float64)
    if cols.shape != P.shape:
        cols = None
    print(f'  点云: {len(P)} 点,顶点色{"有" if cols is not None else "无"}  '
          f'({time.time()-t0:.2f}s)')
    finite = np.isfinite(P).all(axis=1)
    if not finite.all():
        print(f'  ⚠️  有 {int((~finite).sum())} 个点的坐标不是有限值(NaN/Inf),'
              f'这些点会被强制标成背景。')
    return P, cols


# =============================================================================
# 四、读 mask
# =============================================================================
class MaskShapeError(ValueError):
    """mask 尺寸和相机对不上。"""


def otsu_threshold(gray_u8):
    """Otsu 阈值(0-255)。直方图双峰时分得最好,单峰时退化成随便一个值。"""
    g = np.asarray(gray_u8).reshape(-1)
    hist = np.bincount(g.astype(np.uint8), minlength=256).astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 128
    omega = np.cumsum(hist)                      # 前 k 级的像素数
    mu = np.cumsum(hist * np.arange(256))        # 前 k 级的灰度和
    mu_t = mu[-1]
    denom = omega * (total - omega)
    with np.errstate(divide='ignore', invalid='ignore'):
        sigma_b = np.where(denom > 0,
                           (mu_t * omega - total * mu) ** 2 / np.where(denom > 0, denom, 1.0),
                           -1.0)
    return int(np.argmax(sigma_b))


def _dilate(mask, n):
    """4 邻域膨胀 n 次。不依赖 scipy,纯 numpy 移位。"""
    for _ in range(int(n)):
        m = mask.copy()
        m[1:, :] |= mask[:-1, :]
        m[:-1, :] |= mask[1:, :]
        m[:, 1:] |= mask[:, :-1]
        m[:, :-1] |= mask[:, 1:]
        mask = m
    return mask


def load_label_mask(path, W, H, mode='auto', threshold=None, invert=False, dilate=0):
    """读一张 mask,统一成 int32 标签图 (H, W):**0 = 背景,>0 = 类别 id**。

    二值和多类别从此走同一条路。返回 (labels, info)。

    mode:
      'auto'   有能用的 alpha 用 alpha,否则按灰度二值化
      'alpha'  强制用 alpha 通道
      'binary' 强制二值化
      'index'  像素值**原样**当类别 id(多类别索引图)

    特意不猜「彩色就是索引图」—— 抗锯齿的二值 mask 也是彩色的,猜必错。
    多类别请显式传 --mask-mode index。
    """
    from PIL import Image
    if not os.path.isfile(path):
        d = os.path.dirname(os.path.abspath(path))
        if not os.path.isdir(d):
            raise SystemExit(
                f'mask 文件不存在: {path}\n'
                f'    连它所在的目录都不存在: {d}\n'
                f'    mask 是**识图模型**的产物 —— 这一步之前得先生成出来。\n'
                f'    模型要跑在去畸变图目录 <场景>\\dense\\images\\ 上(不是原图目录),'
                f'输出的\n'
                f'    PNG 文件名和原图一致即可(如 IMG_2435.png),放哪个目录都行。\n'
                f'    详见 docs\\mask点云分割.md 第五节。')
        raise SystemExit(f'mask 文件不存在: {path}')

    im = Image.open(path)
    info = {'path': path, 'pixel_mode': im.mode, 'file_size': list(im.size),
            'size': [W, H]}

    # ---- 判断有没有能用的 alpha 通道 ----
    alpha = None
    if 'A' in im.getbands():
        alpha = np.asarray(im.getchannel('A'))
    elif im.mode == 'P' and 'transparency' in im.info:
        alpha = np.asarray(im.convert('RGBA').getchannel('A'))
    # alpha 全一样 = 不是当 mask 用的(比如整张不透明),不能采纳
    if alpha is not None and int(alpha.max()) == int(alpha.min()):
        alpha = None

    use_alpha = (mode == 'alpha') or (mode == 'auto' and alpha is not None)
    if use_alpha and alpha is None:
        raise SystemExit(f'{path}: 要求用 alpha 通道当 mask,但这张图没有可用的 alpha。')

    if mode == 'index':
        arr = np.asarray(im)
        if arr.ndim == 3:
            raise SystemExit(
                f'{path}: --mask-mode index 要求单通道(灰度/索引)图,'
                f'但读出来是 {arr.shape[2]} 通道。\n'
                f'多类别标签图请存成 8/16 位单通道 PNG。')
        labels = arr.astype(np.int32)
        info['source'] = 'index'
    elif use_alpha:
        thr = 128 if threshold is None else int(threshold)
        fg = alpha > thr
        if invert:
            fg = ~fg
        info['source'] = 'alpha'
        info['threshold'] = thr
        labels = fg.astype(np.int32)
    else:
        gray = np.asarray(im.convert('L'))
        thr = otsu_threshold(gray) if threshold is None else int(threshold)
        fg = gray > thr
        if invert:
            fg = ~fg
        info['source'] = 'binary'
        info['threshold'] = thr
        info['otsu'] = otsu_threshold(gray)
        labels = fg.astype(np.int32)

    # ---- 膨胀(SAM 的 mask 常常比物体小一圈)----
    # 只支持二值:多类别图膨胀会「串类」(两个类别的边缘撞上时谁吃掉谁说不清),
    # 与其给一个含糊的结果,不如直接拒绝。
    if dilate and dilate > 0:
        if int(labels.max()) > 1:
            raise SystemExit(
                f'{path}: --mask-dilate 只支持二值 mask(前景值都是 1)。\n'
                f'    这张图里有类别值 {int(labels.max())},多类别膨胀会串类。\n'
                f'    请在外部先膨胀好再传进来。')
        labels = _dilate(labels > 0, dilate).astype(np.int32)
        info['dilate'] = int(dilate)

    if labels.shape != (H, W):
        raise MaskShapeError(_shape_message(path, labels.shape, W, H))
    info['class_values'] = [int(v) for v in np.unique(labels) if v != 0]
    info['n_foreground'] = int((labels > 0).sum())
    return labels, info


def _shape_message(path, got, W, H):
    h, w = got
    msg = (f'mask 尺寸对不上:\n'
           f'    {path}\n'
           f'    读出来 {w}x{h},但相机(去畸变)模型是 {W}x{H}。\n')
    if (w, h) == (3648, 2736):
        msg += ('    3648x2736 是**原图**尺寸。去畸变图被拉大到了 '
                f'{W}x{H},而且这个过程是非线性形变 ——\n'
                '    按比例缩放 mask 是错的,坐标对不上。\n')
    msg += (f'    正确做法:把识图模型直接跑在去畸变图目录'
            f'({os.path.join("<场景>", "dense", "images")})上,再导出 mask。')
    return msg


# =============================================================================
# 五、遮挡剔除(RaycastingScene)
# =============================================================================
def build_raycast_scene(V, F):
    """把网格塞进 Open3D 的 CPU 光线投射引擎。

    注意 dtype:顶点必须 Float32、面索引必须 UInt32。
    """
    import open3d as o3d
    rc = o3d.t.geometry.RaycastingScene()
    rc.add_triangles(
        o3d.core.Tensor(np.ascontiguousarray(V, dtype=np.float32)),
        o3d.core.Tensor(np.ascontiguousarray(F, dtype=np.uint32)))
    return rc


def raycast_t_hit(rc, P, C, chunk=500_000):
    """从光心 C 朝每个点打一条射线,返回命中的最近距离 t_hit(没打中是 inf)。

    分块只是内存保险:110 万条射线 x 6 个 float32 约 27 MB,其实一次塞得下。
    """
    import open3d as o3d
    n = len(P)
    out = np.empty(n, dtype=np.float64)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        d = P[s:e] - C[None, :]
        dist = np.linalg.norm(d, axis=1, keepdims=True)
        d = d / np.where(dist < 1e-12, 1.0, dist)
        o = np.broadcast_to(C[None, :], (e - s, 3))
        rays = np.hstack([o, d]).astype(np.float32)
        ans = rc.cast_rays(o3d.core.Tensor(rays))
        out[s:e] = ans['t_hit'].numpy().astype(np.float64)
    return out


def view_geometry(P, cam, rc):
    """一个视角下,每个点的「投影深度 + 到光心距离 + 射线最近命中距离」。

    这三样和 depth_tol 无关,先算好 —— 自适应容差要用它,正式判定也要用它,
    算一次就够,别打两遍射线。
    """
    K, R, t, C = cam['K'], cam['R'], cam['t'], cam['C']
    _, _, depth = b2m.project_points(P, K, R, t)
    dist = np.linalg.norm(P - C[None, :], axis=1)
    return depth, dist, raycast_t_hit(rc, P, C)


def visibility_from(depth, dist, th, depth_tol, finite_pts):
    """可见 = 在相机前方 且 (射线没打到任何三角形 或 最近命中 >= |P-C|*(1-tol))。

    「射线没打到」(t_hit = inf)要算**可见**:网格有洞(重建空洞)时射线穿过去
    打不中,判成不可见的话洞后面的点会被静默丢掉。

    相机背后的点必须排除:它们的投影会除以一个负深度,算出个毫无意义的像素位置。
    """
    ok = np.ones(len(th), dtype=bool)         # inf -> 保持 True
    fin = np.isfinite(th)
    ok[fin] = th[fin] >= dist[fin] * (1.0 - depth_tol)
    return (depth > 0) & ok & finite_pts


def auto_depth_tol(th, dist, lo=0.005, hi=0.10):
    """自适应遮挡容差:拿「明显在表面上」的那批点估噪声,取 4 sigma。

    点云是网格表面附近的带噪采样,t_hit 和 dist 不会严格相等。容差太小会把
    表面点误判成被遮挡(标签丢一片),太大又会让真正被挡住的点复活。
    """
    fin = np.isfinite(th) & (dist > 0)
    if fin.sum() < 1000:
        return lo, None
    r = th[fin] / dist[fin]
    seed = r[np.abs(r - 1.0) < 0.05]          # 确信在表面上的那批
    if len(seed) < 1000:
        return lo, None
    sigma = 1.4826 * float(np.median(np.abs(seed - 1.0)))     # MAD -> sigma
    return float(np.clip(max(lo, 4.0 * sigma), lo, hi)), sigma


def print_depth_hist(th, dist, depth_tol, label='全部视角'):
    """打印 t_hit/dist 的直方图,让用户自己看容差该取多少。"""
    fin = np.isfinite(th) & (dist > 0)
    if fin.sum() == 0:
        return
    r = th[fin] / dist[fin]
    qs = np.percentile(r, [0, 10, 35, 62, 87, 100])
    print(f'  [深度直方图] {label}  t_hit/dist 分位 '
          f'[{qs[0]:.2f}, {qs[1]:.2f}, {qs[2]:.2f}, {qs[3]:.2f}, {qs[4]:.2f}, {qs[5]:.2f}]'
          f'   (当前 --depth-tol={depth_tol:.4f})')
    lo, hi, nb = 0.90, 1.02, 12
    h, _ = np.histogram(r, bins=np.linspace(lo, hi, nb + 1))
    scale = max(1, h.max())
    for i in range(nb):
        a = lo + (hi - lo) * i / nb
        b = lo + (hi - lo) * (i + 1) / nb
        bar = '#' * int(round(40 * h[i] / scale))
        thr = '  <- 容差线' if a <= 1.0 - depth_tol < b else ''
        print(f'    {a:.2f}-{b:.2f} |{bar:<40s}{h[i]:>9d}{thr}')


# =============================================================================
# 六、投影 + 查标签
# =============================================================================
def lookup_labels(P, cam, label_img):
    """把点投到图上,查它落在哪个像素、拿那个像素的标签。

    像素索引是 [行=v, 列=u] —— 图像数组是 [y, x]。查之前必须 rint + 边界检查,
    不然负索引会静默绕到最后一行/列去。
    """
    K, R, t = cam['K'], cam['R'], cam['t']
    u, v, depth = b2m.project_points(P, K, R, t)
    H, W = label_img.shape
    iu = np.rint(u).astype(np.int64)
    iv = np.rint(v).astype(np.int64)
    inb = (depth > 0) & (iu >= 0) & (iu < W) & (iv >= 0) & (iv < H)
    if not np.isfinite(P).all():
        inb &= np.isfinite(P).all(axis=1)
    out = np.zeros(len(P), dtype=np.int32)
    out[inb] = label_img[iv[inb], iu[inb]]
    return out, inb


# =============================================================================
# 七、多视角融合
# =============================================================================
def fuse(per_view, n_points, min_votes=1, min_visible=1, min_conf=None,
         include_background=False):
    """逐视角投票,给每个点定一个类别。

    per_view 是 [{'name', 'labels', 'visible'}, ...],labels/visible 都是长度 N。

    min_conf 是**一致比例**下限(n_vote / n_visible),None 表示不过滤。
    这是唯一能压掉「并集噪声」的旋钮:默认的 min_votes=1 取的是**跨视角并集**,
    一个点只要在任意一个视角被标上就进来。窗户是漫反射的墙面构件,真窗点在能看到
    它的视角里应该几乎全部落在 mask 内(conf≈1),所以低 conf 的那一撮是 mask 边界
    在多视角间抖动擦过来的,不是窗户。实测 food 场景:conf>=0.9 保留 46%,
    其中 0 个是「各视角一致却被误杀」的,而 --min-votes 5 会误杀 4437 个 conf=1.000
    的真窗户(它们只是可见视角少,不是不一致)。

    返回 dict:labels int32, n_visible uint16, n_vote uint16, confidence float32,
               class_values int32(排过序,决定并列时谁赢)

    并列时取**最小的类别 id** —— class_values 排序后 np.argmax 天然就是这个行为,
    所以结果是确定性的、可复现的。
    """
    n_visible = np.zeros(n_points, dtype=np.uint16)
    vote = {}                                  # 类别 id -> uint16[N] 得票数

    for view in per_view:
        vis = view['visible']
        n_visible[vis] += 1
        lab = view['labels']
        for val in np.unique(lab[vis]):
            val = int(val)
            if val == 0 and not include_background:
                continue                       # 背景默认不投票
            col = vote.get(val)
            if col is None:
                col = np.zeros(n_points, dtype=np.uint16)
                vote[val] = col
            col[vis & (lab == val)] += 1

    if not vote:
        z = np.zeros(n_points, dtype=np.int32)
        return {'labels': z, 'n_visible': n_visible,
                'n_vote': np.zeros(n_points, dtype=np.uint16),
                'confidence': np.zeros(n_points, dtype=np.float32),
                'class_values': np.zeros(0, dtype=np.int32)}

    vals = np.array(sorted(vote), dtype=np.int32)
    M = np.stack([vote[int(v)] for v in vals], axis=1)          # (N, L) uint16
    wi = np.argmax(M, axis=1)
    n_vote = M[np.arange(n_points), wi]
    labels = vals[wi]

    ok = (n_vote >= min_votes) & (n_visible >= min_visible)
    # conf 是胜出类别的得票率。必须在 ok 里用上,所以先算再过滤。
    conf = n_vote.astype(np.float32) / np.maximum(n_visible, 1).astype(np.float32)
    if min_conf is not None:
        ok &= (conf >= float(min_conf))
    labels = np.where(ok, labels, 0).astype(np.int32)

    return {'labels': labels, 'n_visible': n_visible, 'n_vote': n_vote,
            'confidence': conf, 'class_values': vals}


# =============================================================================
# 八、配 (图, mask) 对
# =============================================================================
_MASK_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def match_masks(mask_dir, scene_names):
    """按文件名 stem 把 mask 目录和场景里的图对上。

    返回 (pairs, missing, orphans):pairs 是 [(图像名, mask 路径)],missing 是
    场景里有、但没找到 mask 的图,orphans 是 mask 目录里有、但场景里没有的。
    """
    if not os.path.isdir(mask_dir):
        raise SystemExit(
            f'--mask-dir 不是一个目录: {mask_dir}\n'
            f'    mask 是**识图模型**的产物 —— 这一步之前得先生成出来。\n'
            f'    把生成的 PNG 放进这个目录,文件名和去畸变图一致(如 IMG_2435.png),\n'
            f'    工具按文件名 stem 自动配对。详见 docs\\mask点云分割.md 第五节。')
    index = {}
    for fn in sorted(os.listdir(mask_dir)):
        stem, ext = os.path.splitext(fn)
        if ext.lower() in _MASK_EXTS:
            index.setdefault(stem, []).append(os.path.join(mask_dir, fn))

    pairs, missing, used = [], [], set()
    for nm in scene_names:
        stem = os.path.splitext(os.path.basename(nm))[0]
        cands = index.get(stem)
        if cands:
            pairs.append((nm, cands[0]))
            used.add(stem)
        else:
            missing.append(nm)
    orphans = [p for s, ps in index.items() if s not in used for p in ps]
    return pairs, missing, orphans


def parse_pairs(pair_args, pairs_file):
    """--pair IMG MASK(可重复)+ --pairs-file 每行 `IMG MASK`。

    特意用**两个 token**,不用 `IMG:mask` 那种写法:Windows 的盘符 D:\\ 自带冒号,
    拿冒号当分隔符在绝对路径上必错(相对路径又没冒号),怎么切都是错的。
    """
    out = []
    for a in (pair_args or []):
        out.append((a[0], a[1]))
    if pairs_file:
        if not os.path.isfile(pairs_file):
            raise SystemExit(f'--pairs-file 不存在: {pairs_file}')
        with open(pairs_file, encoding='utf-8') as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) != 2:
                    raise SystemExit(
                        f'{pairs_file} 第 {ln} 行不是 `图像名 mask路径` 两个字段:\n'
                        f'    {line}\n'
                        f'(路径里有空格的话用 --pair "IMG" "路径 with space.png" 代替)')
                out.append((parts[0], parts[1]))
    return out


def resolve_image(scene, name):
    """把用户给的图像名对上模型里的名字。"""
    for nm in scene:
        if nm == name or os.path.basename(nm) == os.path.basename(name):
            return nm
    names = '\n    '.join(scene)
    raise SystemExit(f'在模型里找不到图像 "{name}"。模型里有:\n    {names}')


# =============================================================================
# 九、自检:RaycastingScene vs 暴力求交
# =============================================================================
def verify(args, V, F, cam, n_samples):
    """拿 b2m.ray_triangle_hits(暴力,float64)对拍 RaycastingScene(float32)。

    只抽**网格顶点**:顶点严格落在表面上,两种实现应该算出同一个最近三角形、
    同一个 t。点云的点在表面附近带噪,没法拿来做硬断言。

    两边浮点精度不同,不可能逐位一致,所以比的是几何容差。射线打在顶点上时
    会出现「一个打中、一个擦边没打中」,这种不算错 —— 但如果一边命中在明显
    更近的地方(真被挡住了),那就是真分歧,判失败。
    """
    import open3d as o3d
    C = cam['C']
    rng = np.random.default_rng(args.seed)
    n = min(n_samples, len(V))
    idx = np.sort(rng.choice(len(V), size=n, replace=False))
    tris = V[F]

    dist_all = np.linalg.norm(V[idx] - C[None, :], axis=1)
    tol = max(1e-5, 1e-4 * float(np.median(dist_all)))

    rc = build_raycast_scene(V, F)
    print(f'  抽样 {n} 个网格顶点,容差 tol_t = {tol:.3e}')
    print(f'  参考实现 b2m.ray_triangle_hits(暴力 {len(F)} 个三角形,float64)')
    print(f'  被检实现 RaycastingScene(CPU embree,float32)\n')

    n_both, n_miss, n_grazing, n_bad = 0, 0, 0, 0
    worst, worst_i = 0.0, -1
    t_brute = t_fast_all = 0.0

    for k, i in enumerate(idx):
        p = V[i]
        v = p - C
        dd = float(np.linalg.norm(v))
        if dd < 1e-9:
            continue
        d = v / dd

        t0 = time.time()
        pts, hit = b2m.ray_triangle_hits(C[None, :], d[None, :], tris)
        t_brute += time.time() - t0
        t_ref = float(np.linalg.norm(pts[0] - C)) if hit[0] else np.inf

        t0 = time.time()
        rays = np.array([[C[0], C[1], C[2], d[0], d[1], d[2]]], dtype=np.float32)
        th = float(rc.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy()[0])
        t_fast_all += time.time() - t0
        t_fast = th if np.isfinite(th) else np.inf

        ref_hit, fast_hit = np.isfinite(t_ref), np.isfinite(t_fast)
        if ref_hit and fast_hit:
            dt = abs(t_ref - t_fast)
            if dt > worst:
                worst, worst_i = dt, int(i)
            if dt <= tol:
                n_both += 1
            else:
                n_bad += 1
                if n_bad <= 5:
                    print(f'  ❌ 顶点 {i}: t 差 {dt:.3e} > tol  '
                          f'(暴力 {t_ref:.6f} / 快速 {t_fast:.6f})')
        elif ref_hit != fast_hit:
            t_one = t_ref if ref_hit else t_fast
            if abs(t_one - dd) <= max(tol, 1e-3 * dd):
                n_grazing += 1                 # 打在目标顶点本身,另一个擦边
            else:
                n_bad += 1
                if n_bad <= 5:
                    print(f'  ❌ 顶点 {i}: 命中与否不一致  '
                          f'(暴力 {"hit" if ref_hit else "miss"} / '
                          f'快速 {"hit" if fast_hit else "miss"},'
                          f' |P-C|={dd:.4f})')
        else:
            n_miss += 1

        if (k + 1) % 50 == 0:
            print(f'    ... {k+1}/{n}')

    # 量批量吞吐。上面那圈是**逐条**调用,每次都要现建 tensor,开销全在调用上 ——
    # 拿它去比速度会严重低估 RaycastingScene(正式跑用的是批量)。
    # 固定打 10 万条来量:n 小的时候(比如 --verify 20)一批只要几十微秒,
    # 计时精度根本不够,量出来会是 0.0us。样本不够就循环铺满。
    P_s = V[idx]
    d_s = P_s - C[None, :]
    d_s = d_s / np.linalg.norm(d_s, axis=1, keepdims=True)
    rays_all = np.hstack([np.broadcast_to(C[None, :], (n, 3)), d_s]).astype(np.float32)
    m = 100_000
    rays_big = np.tile(rays_all, (int(np.ceil(m / max(n, 1))), 1))[:m]
    t0 = time.time()
    rc.cast_rays(o3d.core.Tensor(rays_big))
    t_batch = time.time() - t0

    # 顺带验一下方向构造:(P-C)/|P-C| 和 ray_from_pixel(u,v) 应该指向同一个方向
    K, R = cam['K'], cam['R']
    uu, vv, _ = b2m.project_points(V[idx], K, R, cam['t'])
    ang = 0.0
    for u, v, i in zip(uu, vv, idx):
        _, d2 = b2m.ray_from_pixel(float(u), float(v), K, R, C)
        d1 = (V[i] - C) / np.linalg.norm(V[i] - C)
        ang = max(ang, float(np.degrees(np.arccos(np.clip(abs(d1 @ d2), 0, 1)))))

    print(f'\n  两者都命中且一致 : {n_both}')
    print(f'  两者都没命中     : {n_miss}')
    print(f'  擦边(打在顶点上) : {n_grazing}   —— 允许,不算错')
    print(f'  真分歧           : {n_bad}')
    print(f'  最大 |Δt|        : {worst:.3e}' +
          (f'  (顶点 {worst_i})' if worst_i >= 0 else ''))
    print(f'  方向一致性       : 投影反投影方向最大夹角 {ang:.4e} 度')
    print(f'  耗时             : 暴力逐条 {t_brute:.2f}s ({1e3*t_brute/max(n,1):.0f}ms/条)'
          f' / RaycastingScene 逐条 {t_fast_all:.3f}s')
    print(f'                     批量打 {m} 条 {1e3*t_batch:.1f}ms '
          f'({m/max(t_batch,1e-9)/1e6:.1f} 百万条/秒) —— 正式跑走的是批量这条路')
    ok = (n_bad == 0) and (n_both + n_miss + n_grazing > 0) and ang < 1e-2
    print('\n' + ('✅ 自检通过' if ok else '❌ 自检失败'))
    return ok


# =============================================================================
# 十、可视化
# =============================================================================
def visualize(P, cols, labels, class_values, show_all=True):
    """打过标签的点按类别上色;底色用点云自带的顶点色(照片色),没顶点色时退回灰。

    **只能用一个 Geometry,逐点填色。** 曾经的写法是「灰底一份 + 每个类别再放一份
    只含该类点的点云」,但那些点的坐标和灰底那份**完全重合** —— Open3D 默认深度测试
    是 GL_LESS,同深度的后画者直接被丢弃,于是彩色那份一个像素都画不出来,
    整个窗口全是灰的。写文件的 write_labeled_ply() 一直是单几何体,所以没这个毛病。
    """
    import open3d as o3d

    if show_all and cols is not None:
        c = np.asarray(cols, dtype=np.float64)[:len(P)].copy()
    else:
        c = np.tile(np.asarray(COL_UNLABELED, dtype=np.float64)[None, :], (len(P), 1))
    for val in class_values:
        c[labels == int(val)] = class_color(int(val))

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(P)
    pc.colors = o3d.utility.Vector3dVector(c)

    print('\n  图例:')
    for val in class_values:
        val = int(val)
        col = class_color(val)
        print(f'    类别 {val:>3d}  {int((labels == val).sum()):>8d} 点   '
              f'RGB({col[0]:.2f},{col[1]:.2f},{col[2]:.2f})')
    n0 = int((labels == 0).sum())
    base_txt = '照片色' if (show_all and cols is not None) else \
        f'RGB{tuple(round(float(v), 2) for v in COL_UNLABELED)}'
    print(f'    未标注    {n0:>8d} 点   {base_txt}')

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Mask -> Point Cloud Labels', width=1400, height=900)
    vis.add_geometry(pc)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.10, 0.10, 0.12])
    opt.point_size = 3.0
    vis.run()
    vis.destroy_window()


# =============================================================================
# 十一、输出
# =============================================================================
def write_labeled_ply(path, P, labels, class_values, keep_only_labeled=True):
    """把点云写出去:打过标签的按类别上色,没标签的用灰(keep_only_labeled 时丢掉)。"""
    import open3d as o3d
    sel = labels > 0 if keep_only_labeled else np.ones(len(P), dtype=bool)
    c = np.tile(np.asarray(COL_UNLABELED)[None, :], (len(P), 1))
    for val in class_values:
        c[labels == int(val)] = class_color(int(val))
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(P[sel])
    pc.colors = o3d.utility.Vector3dVector(c[sel])
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    o3d.io.write_point_cloud(path, pc)
    print(f'  已写出 {path}  ({int(sel.sum())} 点)')


def class_report(P, labels, class_values, names):
    """每个类别的点数 / 质心 / 包围盒。

    名字按 **类别 id** 取(--class-names 的第 1 项对应 id 1),不按出现顺序 ——
    否则某个类别这次一个点都没中标,后面的名字会整体错位。
    """
    rep = []
    for val in class_values:
        val = int(val)
        sel = labels == val
        n = int(sel.sum())
        item = {'id': val,
                'name': (names[val - 1] if names and 1 <= val <= len(names)
                         else f'class{val}'),
                'n_points': n}
        if n:
            item['centroid'] = P[sel].mean(axis=0).tolist()
            item['bbox_min'] = P[sel].min(axis=0).tolist()
            item['bbox_max'] = P[sel].max(axis=0).tolist()
        rep.append(item)
    return rep


# =============================================================================
# 十二、主流程
# =============================================================================
def main():
    ap = argparse.ArgumentParser(
        description='多视角 mask 融合到点云:每点一个类别标签',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scene', required=True, help='COLMAP sparse 目录(含 cameras.bin/images.bin)')
    ap.add_argument('--cloud', default=None, help='点云 .ply;默认 <场景>/mvs/scene_dense.ply')
    ap.add_argument('--mesh', default=None,
                    help='遮挡用的网格 .ply;默认 <场景>/mvs/scene_dense_mesh.ply(稠密的那份)')
    ap.add_argument('--images', default=None, help='去畸变图目录(只用于名字匹配和分辨率校对)')

    ap.add_argument('--pair', action='append', nargs=2, default=None,
                    metavar=('IMG', 'MASK'),
                    help='图像名 + mask 路径,可重复。注意是两个参数,不要写 IMG:mask')
    ap.add_argument('--pairs-file', dest='pairs_file', default=None,
                    help='批量配对文件,每行 `图像名 mask路径`')
    ap.add_argument('--mask-dir', dest='mask_dir', default=None,
                    help='mask 目录,按文件名 stem 自动匹配场景里的图')

    ap.add_argument('--mask-mode', dest='mask_mode', default='auto',
                    choices=['auto', 'binary', 'index', 'alpha'],
                    help='auto=有 alpha 用 alpha,否则灰度二值化;index=像素值原样当类别 id')
    ap.add_argument('--threshold', type=int, default=None, help='二值/alpha 阈值;默认 Otsu')
    ap.add_argument('--invert', action='store_true', help='翻转极性(前景是暗的那边)')
    ap.add_argument('--mask-dilate', dest='mask_dilate', type=int, default=0,
                    help='把 mask 膨胀 N 像素(SAM 常标小一圈)')

    ap.add_argument('--depth-tol', dest='depth_tol', type=float, default=None,
                    help='遮挡容差,|P-C| 的比例;默认自适应')
    ap.add_argument('--depth-tol-hist', dest='depth_tol_hist', action='store_true',
                    help='打印 t_hit/dist 直方图,方便手调 --depth-tol')
    ap.add_argument('--min-votes', dest='min_votes', type=int, default=1,
                    help='至少几个视角投给同一类才采纳(默认 1;多视角建议 2-3)')
    ap.add_argument('--min-visible', dest='min_visible', type=int, default=1,
                    help='至少被几个视角看到才采纳(默认 1)')
    ap.add_argument('--min-conf', dest='min_conf', type=float, default=None,
                    help='窗口内一致比例(n_vote/n_visible)下限,默认不过滤。'
                         '默认 min-votes=1 取的是跨视角**并集**,mask 边界抖动会带进'
                         '大量低置信点;0.9 左右能干净切掉(实测两端 0 误判)。')
    ap.add_argument('--include-background', dest='include_background', action='store_true',
                    help='让 label 0(背景)也当普通类别投票')

    ap.add_argument('--class-names', dest='class_names', default=None,
                    help='类别名字,逗号分隔,对应类别 id 1,2,3...')

    ap.add_argument('--out', default=None, help='只含已标注点的 ply')
    ap.add_argument('--out-all', dest='out_all', default=None, help='含全部点的 ply')
    ap.add_argument('--labels', default=None, help='标签存 npy(int32,与输入点云同序)')
    ap.add_argument('--save-npz', dest='save_npz', default=None,
                    help='labels + n_visible + n_vote + confidence 存 npz')
    ap.add_argument('--save-json', dest='save_json', default=None, help='统计存 json')
    ap.add_argument('--vis', action='store_true',
                    help='开 open3d 窗口看结果:底色是点云自带的照片色,'
                         '已标注点按类别上色盖在上面')

    ap.add_argument('--verify', type=int, nargs='?', const=200, default=None,
                    metavar='N', help='自检:RaycastingScene vs 暴力求交,抽 N 个顶点(默认 200)')
    ap.add_argument('--seed', type=int, default=0, help='--verify 抽样的随机种子')
    args = ap.parse_args()

    if not any([args.pair, args.pairs_file, args.mask_dir, args.verify]):
        raise SystemExit('要用 --pair / --pairs-file / --mask-dir 至少给一组 (图, mask),'
                         '或者用 --verify 只跑自检。')

    cloud = args.cloud or default_cloud(args.scene)
    mesh = args.mesh or default_occlusion_mesh(args.scene)
    images = args.images or default_images_dir(args.scene)

    print('=' * 70)
    print(f'  场景    : {args.scene}')
    print(f'  点云    : {cloud}')
    print(f'  遮挡网格: {mesh}')
    print(f'  去畸变图: {images}')
    print('=' * 70)

    # ---------- 位姿 ----------
    scene = b2m.load_scene(args.scene)
    print(f'\n相机模型: {len(scene)} 个视角')

    # ---------- 自检模式:不需要点云 ----------
    if args.verify is not None:
        try:
            V, F, _ = b2m.load_mesh(mesh)
        except ValueError as e:
            raise SystemExit(str(e))
        print(f'  网格: {len(V)} 顶点 / {len(F)} 三角形')
        if len(F) < 1000:
            print(f'  ⚠️  网格只有 {len(F)} 个面 —— 这份网格多半是 refine 出来的粗网格,'
                  f'洞很多,遮挡会不准。自检仍然有参考价值。')
        first = next(iter(scene.values()))
        print(f'\n用 {list(scene)[0]} 这个视角来对拍\n')
        ok = verify(args, V, F, first, args.verify)
        raise SystemExit(0 if ok else 1)

    # ---------- 凑 (图, mask) 对 ----------
    pairs = parse_pairs(args.pair, args.pairs_file)
    if args.mask_dir:
        auto, missing, orphans = match_masks(args.mask_dir, list(scene))
        pairs += auto
        print(f'\n--mask-dir 匹配: {len(auto)} 张对上,{len(missing)} 张缺 mask')
        if missing:
            print('  缺 mask 的图:' + ', '.join(missing[:10]) +
                  (f' (还有 {len(missing)-10} 张)' if len(missing) > 10 else ''))
        if orphans:
            print('  mask 目录里多余的:' + ', '.join(os.path.basename(p) for p in orphans[:10]))

    if not pairs:
        raise SystemExit('一组 (图, mask) 都没有,没法定标签。')

    # ---------- 逐个读 mask,先把类别集合摸清楚 ----------
    print(f'\n{"=" * 70}\n读 mask\n{"=" * 70}')
    H = W = None
    resolved = []
    all_vals = set()
    for img_name, mask_path in pairs:
        nm = resolve_image(scene, img_name)
        cam = scene[nm]
        if W is None:
            W, H = int(cam['W']), int(cam['H'])
            print(f'  图像尺寸(去畸变): {W} x {H}')
        try:
            lab, info = load_label_mask(mask_path, W, H, mode=args.mask_mode,
                                        threshold=args.threshold, invert=args.invert,
                                        dilate=args.mask_dilate)
        except MaskShapeError as e:
            raise SystemExit(str(e))
        all_vals |= set(info['class_values'])
        resolved.append((nm, mask_path, lab))
        fg = info['n_foreground']
        print(f'  {nm:<16s} <- {os.path.basename(mask_path):<24s} '
              f'{info["source"]:<6s} 前景 {fg} px  类别 {info["class_values"]}')

    class_values = np.array(sorted(all_vals), dtype=np.int32)
    if len(class_values) and class_values.min() == 0 and not args.include_background:
        class_values = class_values[class_values > 0]
    print(f'\n  类别共 {len(class_values)} 个: {class_values.tolist()}')
    if len(class_values) == 0:
        print('  ⚠️  所有 mask 都是空的(前景 0 像素),什么也标不出来。')
    if len(class_values) > WARN_N_CLASSES:
        print(f'  ⚠️  类别数 {len(class_values)} 偏多,投票表是 N x L 的,内存会涨。')
    class_names = ([s.strip() for s in args.class_names.split(',')]
                   if args.class_names else None)

    # ---------- 读点云 ----------
    print(f'\n{"=" * 70}\n读点云\n{"=" * 70}')
    P, cols = load_point_cloud(cloud)
    N = len(P)
    print(f'  包围盒 min {np.round(P.min(0), 3)}  max {np.round(P.max(0), 3)}')

    # ---------- 遮挡网格 ----------
    print(f'\n{"=" * 70}\n遮挡网格\n{"=" * 70}')
    try:
        V, F, _ = b2m.load_mesh(mesh)
    except ValueError as e:
        raise SystemExit(str(e))
    print(f'  {os.path.basename(mesh)}: {len(V)} 顶点 / {len(F)} 三角形')
    if len(F) < 20000:
        print(f'  ⚠️  网格只有 {len(F)} 个面,很可能是 refine 出来的粗网格。'
              f'洞多的地方射线打不中,会被判成「可见」,遮挡基本失效。\n'
              f'     请用 --mesh 指定稠密的 scene_dense_mesh.ply。')
    t0 = time.time()
    rc = build_raycast_scene(V, F)
    print(f'  光线投射引擎就绪 ({time.time()-t0:.3f}s)')

    # ---------- 逐视角:可见性 + 标签 ----------
    print(f'\n{"=" * 70}\n逐视角投影 + 遮挡剔除\n{"=" * 70}')
    depth_tol = args.depth_tol
    finite_pts = np.isfinite(P).all(axis=1)
    per_view = []
    all_th, all_dist = [], []
    t_start = time.time()
    for k, (nm, mask_path, lab) in enumerate(resolved):
        cam = scene[nm]
        t0 = time.time()
        depth, dist, th = view_geometry(P, cam, rc)
        if depth_tol is None:
            # 容差只定一次:拿第一个视角估出来的噪声,后面所有视角都用它。
            # 各视角的噪声量级是同一个东西(重建精度),没必要逐视角重估。
            tol_auto, sigma = auto_depth_tol(th, dist)
            depth_tol = tol_auto
            print(f'  --depth-tol 自适应: {depth_tol:.4f}'
                  + (f'  (sigma={sigma:.5f}, 4σ)' if sigma is not None else '  (样本不足,用下限)'))
        visible = visibility_from(depth, dist, th, depth_tol, finite_pts)
        labels_pt, _ = lookup_labels(P, cam, lab)
        per_view.append({'name': nm, 'labels': labels_pt, 'visible': visible})
        all_th.append(th)
        all_dist.append(dist)
        nvis = int(visible.sum())
        nhit = int((visible & (labels_pt > 0)).sum())
        print(f'  [{k+1:>2d}/{len(resolved)}] {nm:<16s} 可见 {nvis:>8d}  '
              f'其中落在 mask 内 {nhit:>8d}  ({100.0*nhit/max(nvis,1):.1f}%)  '
              f'{time.time()-t0:.2f}s')
        del lab
    print(f'\n  遮挡 + 投影总耗时 {time.time()-t_start:.1f}s')

    if args.depth_tol_hist:
        print()
        print_depth_hist(np.concatenate(all_th), np.concatenate(all_dist), depth_tol)
    del all_th, all_dist

    # ---------- 融合 ----------
    print(f'\n{"=" * 70}\n多视角投票\n{"=" * 70}')
    t0 = time.time()
    res = fuse(per_view, N, min_votes=args.min_votes,
               min_visible=args.min_visible,
               min_conf=args.min_conf,
               include_background=args.include_background)
    labels, n_visible, n_vote, conf = (res['labels'], res['n_visible'],
                                       res['n_vote'], res['confidence'])
    print(f'  投票完成 ({time.time()-t0:.2f}s)')
    print(f'  可见视角数: 中位 {int(np.median(n_visible))}  '
          f'最多 {int(n_visible.max())}  一个都没看到 {int((n_visible == 0).sum())} 点')
    n_lab = int((labels > 0).sum())
    print(f'  打了标签: {n_lab}/{N} ({100.0*n_lab/max(N,1):.1f}%)')
    if n_lab == 0:
        print('  ⚠️  一个点都没标上 —— 检查 mask 是不是全空,或者 --min-votes 是不是定太高。')
    else:
        print(f'  置信度(可见视角中的一致比例)分位 '
              f'{np.round(np.percentile(conf[labels > 0], [5, 25, 50, 75, 95]), 3).tolist()}')

    # ---------- 输出 ----------
    print(f'\n{"=" * 70}\n输出\n{"=" * 70}')
    used_vals = np.array(sorted({int(v) for v in np.unique(labels) if v > 0}),
                         dtype=np.int32)
    if args.out:
        write_labeled_ply(args.out, P, labels, used_vals, keep_only_labeled=True)
    if args.out_all:
        write_labeled_ply(args.out_all, P, labels, used_vals, keep_only_labeled=False)
    if args.labels:
        d = os.path.dirname(os.path.abspath(args.labels))
        if d:
            os.makedirs(d, exist_ok=True)
        np.save(args.labels, labels.astype(np.int32))
        print(f'  已写出 {args.labels}  (int32, 与输入点云同序)')
    if args.save_npz:
        d = os.path.dirname(os.path.abspath(args.save_npz))
        if d:
            os.makedirs(d, exist_ok=True)
        np.savez_compressed(args.save_npz, labels=labels.astype(np.int32),
                            n_visible=n_visible, n_vote=n_vote,
                            confidence=conf, class_values=used_vals)
        print(f'  已写出 {args.save_npz}')

    if args.save_json:
        rep = class_report(P, labels, used_vals, class_names)
        out = {
            'scene': os.path.abspath(args.scene),
            'cloud': os.path.abspath(cloud),
            'occlusion_mesh': os.path.abspath(mesh),
            'n_points': int(N),
            'image_size': [W, H],
            'views_used': [v['name'] for v in per_view],
            'depth_tol': float(depth_tol),
            'min_votes': args.min_votes,
            'min_visible': args.min_visible,
            'min_conf': args.min_conf,
            'include_background': bool(args.include_background),
            'n_labeled': n_lab,
            'n_unlabeled': int(N - n_lab),
            'n_never_visible': int((n_visible == 0).sum()),
            'per_class': rep,
        }
        d = os.path.dirname(os.path.abspath(args.save_json))
        if d:
            os.makedirs(d, exist_ok=True)
        with open(args.save_json, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f'  已写出 {args.save_json}')
        print('\n  每类:')
        for it in rep:
            print(f'    {it["name"]:<12s} {it["n_points"]:>8d} 点'
                  + (f'   质心 {np.round(it["centroid"], 3).tolist()}' if it['n_points'] else ''))

    # ---------- 看图 ----------
    if args.vis:
        visualize(P, cols, labels, used_vals, show_all=True)


if __name__ == '__main__':
    main()
