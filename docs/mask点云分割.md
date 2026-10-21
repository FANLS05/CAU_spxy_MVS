# 多视角 mask → 点云分割

**场景**:`work\food`(13 张图,1112606 点点云)
**工具**:`tools\mask_to_points.py`
**实测日期**:2026-10-14

---

## 一、这一步是干什么的

给「2D 识图模型 → 3D 点云分割」搭的那座桥。

```
data\food\  (原图 3648x2736)
    │
    │  COLMAP 去畸变
    ▼
work\food\dense\images\   (3781x2788)  ←★ 识图模型跑在这一层
    │
    │  识图模型逐张输出 mask
    ▼
masks\IMG_2430.png  IMG_2432.png  ...   ←★ 工具的输入之一
    │
    │  ┌─ work\food\dense\sparse\          (相机位姿)
    │  ├─ work\food\mvs\scene_dense.ply     (点云,要被标注的东西)
    │  └─ work\food\mvs\scene_dense_mesh.ply(网格,只用来做遮挡)
    ▼
tools\mask_to_points.py
    ▼
每个点一个类别标签 + 置信度
```

**和 `2D框投影到3D模型.md` 的区别**:那边是**一个轴对齐矩形**,这边是一组
**不规则 mask**,而且是**多视角融合**。同一个点会被好几个视角看到,每个视角
给它一个类别,再投票定夺 —— 这才是「类似点云分割」的关键。

---

## 二、原理

对每张图 `i`、每个点 `p`,回答两个问题:

**1. 这个点在视角 `i` 里看得见吗?**

```
可见 = 在相机前方 且 (从光心到点的射线上,没在更近的地方撞到网格)
```

射线求交用 Open3D 的 `RaycastingScene`(CPU embree),**不是** `bbox_to_mesh.py`
里的 `ray_triangle_hits`。后者是逐条射线的 Python 循环(`bbox_live.py` 的
`cast_rays` 也是),110 万条射线根本跑不动。

> 两个细节:
> - **射线没打到任何三角形,要算「可见」。** 网格有洞(重建空洞)时射线会穿过去,
>   判成不可见的话,洞后面的点会被静默丢掉。
> - **相机背后的点必须排除。** 它们的投影会除以一个负深度,算出毫无意义的像素位置。

**2. 看得见的话,它落在 mask 的哪一类?**

```
(u, v) = project(p)        # 投影
标签 = mask[round(v), round(u)]     # 注意是 [行=v, 列=u]
```

**3. 投票**

```
每个点: 在它可见的所有视角里,统计各非零类别各得几票
        → 票最多的类别胜出(并列取最小类别 id,保证可复现)
        → 票数 < --min-votes、可见视角数 < --min-visible、或
          confidence < --min-conf 的点,标成背景 0
confidence = 胜出票数 / 可见视角数
```

**默认 `--min-votes 1` 取的是跨视角「并集」** —— 一个点只要在**任意一个**视角被标上
就进来。这是最容易踩的坑:mask 边界在不同视角间抖动几十个像素,擦过边界的点各视角
意见不一,于是并集里混进一大撮低置信度的点。`--min-conf` 就是治它的旋钮,见第三节。

**背景(label 0)默认不投票。** 一个表面点常常在某个视角落在 mask 外(模型标小了),
如果背景也投票,可能反压过真类别。不投票的话这个点仍能从别的视角拿到类别,
分歧体现在 `confidence` 上。要改这个行为用 `--include-background`。

⚠️ `confidence` 是「可见视角中意见一致的比例」,**不是**「这个点属于该类别的概率」。
一个点被 5 个视角看到、3 个标它、2 个标成背景 → `confidence = 0.6`。

---

## 三、实测参数与性能

数据集 `food`,N = **1112606** 点,遮挡网格 **824422 顶点 / 1648306 面**,
13 个视角(全部 PINHOLE 3781×2788)。

### 耗时

| 步骤 | 耗时 |
|---|---|
| 读点云(`o3d.io.read_point_cloud`) | 0.35 s |
| 读网格 | 1.5 s |
| `RaycastingScene` 建索引 | 0.012 s |
| 单个视角(投影 + 110 万条射线 + 查标签) | **首次 3.4 s,之后 0.19 s** |
| 多视角投票 | 0.06 s |
| **批量吞吐(射线)** | **约 2300 万条/秒** |

> 「首次 3.4 s / 之后 0.19 s」是冷热缓存之差 —— 第一次 `cast_rays` 要把 BVH
> 建起来、页表踩热,后面就快了。13 个视角总耗时约 **6 秒**。

### 遮挡判据的实测分布

单个视角(IMG_2435,110 万点)的 `t_hit / |P-C|` 直方图:

```
   0.95-0.96 |#                                            7664
   0.96-0.97 |#                                            7154
   0.97-0.98 |#                                           10915
   0.98-0.99 |##                                          21855
   0.99-1.00 |########################################   511687   <- 容差线
   1.00-1.01 |####################################       461798
   1.01-1.02 |#                                           10650
```

**双峰 + 左侧尾巴**,正是预期的形状:

- 0.99~1.01 那一大坨 = 点在表面上(`t_hit ≈ |P-C|`)
- 左侧尾巴 = 被遮挡的点(网格在更近的地方挡住了)
- 判定可见 **976957/1112606 = 87.8%**

### 自检(`--verify`)

| 项 | 值 |
|---|---|
| 两者都命中且一致 | 19 / 20 |
| 擦边(射线正打在顶点上,一个命中一个没命中) | 1 —— 允许,不算错 |
| **真分歧** | **0** |
| 最大 \|Δt\| | **1.5e-06** |
| 方向一致性(投影→反投影,夹角) | **1.5e-06 度** |
| 耗时 | 暴力逐条 164 ms/条;RaycastingScene 批量 100000 条 4.4 ms |

---

## 四、环境

**必须用 conda `basic_env` 的解释器** —— `PATH` 上的 `python`(C:\Python314)和
anaconda base 里都**没有** trimesh / open3d。

```powershell
$py = "C:\Users\FANLS\.conda\envs\basic_env\python.exe"
& $py -c "import numpy,trimesh,open3d,PIL;print(open3d.__version__)"
# 应打印 0.20.0
```

| 包 | 版本 | 用途 |
|---|---|---|
| numpy | 1.26.4 | |
| open3d | 0.20.0 | 读点云、`RaycastingScene`、写 ply、可视化 |
| Pillow | 12.2.0 | 读 mask |
| trimesh | 5.1.0 | 读**网格**(读不了点云,见坑 2) |

> ⚠️ `docs\2D框投影到3D模型.md` 和 `README.md` 里写的「用你默认的 Python 3.14」
> **是过时的**。真正能跑的是 `basic_env`(Python 3.10.20)。

---

## 五、mask 怎么准备

> **这一步现在有具体工具了**:[`tools\sam3_masks.py`](../tools/sam3_masks.py) —— 用
> SAM 3 对每张去畸变图按**文字概念**出 mask,直接产出下面这种索引图。
> 用法见 [`SAM3生成mask.md`](SAM3生成mask.md)。下面这段是通用约定,用别的模型也行。

识图模型直接跑在**去畸变图目录** `work\<场景>\dense\images\` 上,输出 mask 存成
和原图同名的 PNG 到任意一个目录:

```
masks\IMG_2430.png
masks\IMG_2432.png
...
```

**尺寸必须严格是 3781×2788**(= 相机模型的尺寸),不是原图的 3648×2736。
去畸变是非线性形变,**缩放 mask 是错的**,工具会直接报错拒绝:

```
mask 尺寸对不上:
    masks\IMG_2435.png
    读出来 3648x2736,但相机(去畸变)模型是 3781x2788。
    3648x2736 是**原图**尺寸。去畸变图被拉大到了 3781x2788,而且这个过程是非线性形变 ——
    按比例缩放 mask 是错的,坐标对不上。
    正确做法:把识图模型直接跑在去畸变图目录(<场景>\dense\images)上,再导出 mask。
```

### 支持哪些 mask 格式

工具统一把 mask 读成 `int32[H,W]`,**0 = 背景,>0 = 类别 id**。二值和多类别
从此走同一条路。

| 你的 mask 长什么样 | `--mask-mode` | 说明 |
|---|---|---|
| RGBA + alpha 当掩膜(SAM 常见) | `auto` 自动认 | alpha 全相同 = 不是在当掩膜用,会退回灰度 |
| 黑白 PNG(白=前景) | `auto` 自动认 | Otsu 自动定阈值 |
| 黑白 PNG(黑=前景) | `--invert` | |
| 多类别索引图(像素值 = 类别 id) | **`--mask-mode index`** | 必须显式指定,见坑 4 |

**特意不猜「彩色就是索引图」** —— 抗锯齿的二值 mask 也是彩色的,猜必错。
多类别请显式传 `--mask-mode index`。

类别 id 的语义:`1, 2, 3...` 分别对应 `--class-names "盘子,食物"` 的第 1、2 项。
**按 id 取,不按出现顺序** —— 否则某个类别这次一个点都没中标,后面的名字会整体错位。

---

## 六、逐步命令

```powershell
$py    = "C:\Users\FANLS\.conda\envs\basic_env\python.exe"
$root  = "D:\code\point_cloud\0916"
$scene = "$root\work\food\dense\sparse"
```

### 第 1 步:自检(先跑这个)

```powershell
& $py "$root\tools\mask_to_points.py" --verify 20 --scene $scene
```

拿暴力 Möller–Trumbore(float64)对拍 `RaycastingScene`(float32),抽 20 个网格顶点。
打印 ✅/❌,退出码 0/1。

`--verify` 不带数字时默认抽 200 个(约 35 秒 —— 暴力版是 164 ms/条)。
**这个自检不需要点云**,只要有 `--scene` 和网格。

### 第 2 步:单张试

> ⚠️ **这一步需要先有一个真的 mask。** 识图模型还没接的话,`masks\` 目录是不存在的,
> 会直接报 `mask 文件不存在`。怎么产出 mask 见上面**第五节**;只想先验证通路能不能
> 跑通,可以先跳过第 2 步,等模型能出图了再回来。

```powershell
& $py "$root\tools\mask_to_points.py" --scene $scene `
  --pair "IMG_2435.JPG" "$root\masks\IMG_2435.png" `
  --depth-tol-hist `
  --out "$root\out\masked.ply" --labels "$root\out\labels.npy" --save-json "$root\out\stats.json"
```

`--pair` 是**两个参数**,不是 `IMG_2435.JPG:D:\...\mask.png` —— Windows 的盘符
`D:\` 自带冒号,拿冒号当分隔符必错(见坑 1)。

`--depth-tol-hist` 会把 `t_hit/dist` 直方图打出来,方便你看容差该定多少。

### 第 3 步:批量(识图模型产出一堆 mask 之后)

```powershell
& $py "$root\tools\mask_to_points.py" --scene $scene `
  --mask-dir "$root\masks" --min-conf 0.9 `
  --out "$root\out\masked.ply" --save-npz "$root\out\full.npz" `
  --save-json "$root\out\stats.json" --vis
```

`--mask-dir` 按**文件名 stem** 把 mask 和场景里的图对上,会自动报告
「哪些图缺 mask」和「mask 目录里多余的」。

**压误标要用 `--min-conf`,不是 `--min-votes`。** 两者看着都像「要求多视角一致」,
但 `--min-votes` 数的是**绝对票数**,完全不看这个点被几个视角看到:

| | `--min-votes 5` | `--min-conf 0.9` |
|---|---|---|
| 误杀真标签 | **4,437** 个(conf 均值 **1.000**,只是平均仅 3.5 个视角看得见) | 0 |
| 放进噪声 | **9,246** 个(conf < 0.5) | 0 |

劣视角下的立面边缘、掠射角区域,本来就只有少数相机看得见,票数天然少却完全正确 ——
`--min-votes` 会把这些真标签一起砍掉。实测(food,13 视角,window):
并集 19.95 万点 → `--min-conf 0.9` 剩 9.18 万点,留下的点里 95% 以上 conf 正好 1.0。

> **先跑一次不带 `--min-conf` 的,把 `--save-npz` 存下来看 confidence 分布再定阈值。**
> 分布是双峰(一个峰在 1.0,一条低置信长尾)才说明有并集噪声可压;如果是单峰,
> 说明 mask 本身干净,不用过滤。别不看分布就拍一个阈值。

### 第 4 步:看结果

```powershell
& $py -c "import numpy as np,collections; a=np.load(r'$root\out\labels.npy'); print(a.shape, collections.Counter(a.tolist()))"
```

或者用 `--vis` 开 Open3D 窗口,也可以直接把 `out\masked.ply` 拖进 MeshLab / CloudCompare。

---

## 七、参数表

| 参数 | 默认 | 说明 |
|---|---|---|
| `--scene` | **必填** | COLMAP sparse 目录 |
| `--cloud` | 自动推 | 点云 `.ply`,默认 `<场景>\mvs\scene_dense.ply` |
| `--mesh` | 自动推 | **只用来做遮挡**,默认 `<场景>\mvs\scene_dense_mesh.ply` |
| `--images` | 自动推 | 去畸变图目录(只用于名字匹配和尺寸校对) |
| `--pair IMG MASK` | — | 可重复。**两个参数** |
| `--pairs-file PATH` | — | 每行 `图像名 mask路径`,`#` 开头是注释 |
| `--mask-dir DIR` | — | 按文件名 stem 自动匹配场景里的所有图 |
| `--mask-mode` | `auto` | `auto` / `binary` / `alpha` / `index` |
| `--threshold` | Otsu | 二值/alpha 阈值 |
| `--invert` | 关 | 翻转极性 |
| `--mask-dilate N` | 0 | 膨胀 N 像素(**只支持二值**) |
| `--depth-tol` | 自适应 | 遮挡容差,`\|P-C\|` 的比例 |
| `--depth-tol-hist` | 关 | 打印直方图 |
| `--min-votes` | 1 | 至少几个视角投同一类才采纳 |
| `--min-visible` | 1 | 至少被几个视角看到才采纳 |
| `--min-conf` | 不过滤 | **一致比例** `n_vote/n_visible` 下限。压掉并集噪声的正牌旋钮,通常 `0.9` |
| `--include-background` | 关 | 让背景也当普通类别投票 |
| `--class-names` | — | 逗号分隔,对应类别 id 1,2,3... |
| `--out` | — | **只含已标注点**的 ply,按类别上色 |
| `--out-all` | — | 含全部点的 ply |
| `--labels` | — | `npy`,int32,与输入点云**同序** |
| `--save-npz` | — | `labels + n_visible + n_vote + confidence` |
| `--save-json` | — | 统计(每类点数/质心/包围盒) |
| `--vis` | 关 | 开 Open3D 窗口 |

**必须至少给一组 (图, mask) 或 `--verify`**,否则报错退出。

### 颜色

沿用仓库的颜色语言:**类别 1 = 青色**(`bbox_to_mesh.COL_FRAGMENT`,和 2D 框工具里
「被选中的那部分网格」是同一个颜色),之后橙、黄绿、品红……;未标注恒为灰。

---

## 八、坑

| # | 现象 | 原因 | 解决 |
|---|---|---|---|
| 1 | `--pair "IMG.JPG:D:\masks\a.png"` 解析错乱 | Windows 盘符 `D:\` 自带冒号,和 `:` 分隔符撞车;相对路径又没冒号,怎么切都是错的 | `--pair` 是**两个参数**:`--pair "IMG.JPG" "D:\masks\a.png"` |
| 2 | `trimesh.load` 读 `scene_dense.ply` 报 `ValueError: PLY is unexpected length!` | OpenMVS 的点云带自定义属性 `view_indices` / `view_weights`,trimesh 的 ply 读取器处理不了 | 点云走 `o3d.io.read_point_cloud`(工具里已如此);`trimesh` 只用来读网格 |
| 3 | 遮挡完全失效,可见率接近 100% | 遮挡网格用了 `scene_dense_mesh_refine.ply` —— food 这份只有 **7622 顶点 / 14992 面**,到处是洞,射线打不中就被判成「可见」 | 用稠密的 `scene_dense_mesh.ply`(824422 顶点)。工具默认就偏好这份,并在面数 < 2 万时警告 |
| 4 | 多类别 mask 被当成单类别 | `auto` 模式**不猜**「彩色就是索引图」(抗锯齿的二值 mask 也是彩色的) | 显式传 `--mask-mode index` |
| 5 | 框在原图上做的 mask,投影全偏 | 原图 3648×2736 ≠ 去畸变图 3781×2788,且形变非线性 | 识图模型跑在 `dense\images\` 上。工具会直接报错拒绝,不会默默算错 |
| 6 | 洞后面的点被静默丢掉 | 把「射线没打到」当成不可见 | 工具里 `t_hit = inf` 算**可见** |
| 7 | 相机背后的点被标了一堆标签 | 投影要除以深度,负深度算出无意义的像素坐标 | 工具已把 `depth <= 0` 强制排除 |
| 8 | 表面点成片丢标签 | `--depth-tol` 太小,点云噪声被误判成「被遮挡」 | 看 `--depth-tol-hist` 直方图调;默认自适应取 4σ(food 实测 0.0050) |
| 9 | `--mask-dilate` 在多类别 mask 上报错 | 多类别膨胀会「串类」(两类边缘撞上时谁吃掉谁说不清) | 在外部膨胀好再传进来 |
| 10 | 报错信息在 PowerShell 里是乱码 | stderr 默认用 GBK,遇到 `——` `⚠️` 这类字符就崩 | 工具已把 stdout **和** stderr 都 `reconfigure(encoding='utf-8')` |
| 11 | 点云文件不存在 | `work\*\mvs\scene_dense.ply` 被 `.gitignore` 排除(80 MB) | 重跑 `DensifyPointCloud`,见 `docs\OpenMVS操作手册.md` |
| 12 | `--min-votes 2` 之后点数骤降 | 这是**预期行为** —— 要求至少 2 个视角意见一致 | 3 视角实测 30.1 万 → 7.5 万;按需要调 |
| 13 | 点云上标签「胖一圈」,边缘毛糙、糊在窗框外 | 默认 `--min-votes 1` 是跨视角**并集**,mask 边界抖动擦进来的点全在里面 | 加 `--min-conf 0.9`(先 `--save-npz` 看分布确认是双峰,见第三节) |
| 14 | 想压误标但 `--min-votes` 一调就砍掉整片真标签 | 它数**绝对票数**,不看这个点被几个视角看到;劣视角区域的点票数天然少但完全正确 | 用 `--min-conf`。实测 `--min-votes 5` 误杀 4437 个 conf=1.000 的真标签,`--min-conf 0.9` 两端 0 误判 |

---

## 九、输出文件

| 文件 | 内容 |
|---|---|
| `masked.ply` | **只含已标注的点**,按类别上色。主产物,任何点云查看器都能开 |
| `labels.npy` | int32 `(N,)`,和**输入点云同序** —— 下游处理用这个 |
| `full.npz` | `labels` + `n_visible`(可见视角数) + `n_vote`(胜出票数) + `confidence` + `class_values` |
| `stats.json` | 每类的点数 / 质心 / 包围盒,以及深度容差、用到的视角、缺失 mask 的视角等 |

> 标签为什么不写进 ply:open3d 没法把自定义的逐点属性往返写进 PLY。所以
> **几何走 ply,标签走 npy/npz**,两份靠「点序一致」对应 —— 别对点云做过任何
> 裁剪、排序、去重,否则顺序就对不上了。

---

## 十、和主流程的关系

```
data\food\  (原图 3648x2736)
    │  ①-⑤ COLMAP
    ▼
work\food\dense\
    ├── images\   去畸变图 3781x2788   ←★ 识图模型跑这、2D 框画这
    └── sparse\   PINHOLE + 位姿       ←★ mask_to_points.py 读这
    │
    │  ⑥-⑦ InterfaceCOLMAP → DensifyPointCloud
    ▼
work\food\mvs\scene_dense.ply       (1112606 点)         ←★ 点云
    │  ⑧ ReconstructMesh
    ▼
work\food\mvs\scene_dense_mesh.ply  (824422v / 1648306f) ←★ 遮挡网格
    │
    ▼
tools\mask_to_points.py  →  每点一个类别标签
```

**三个输入(位姿 / 点云 / 遮挡网格)必须来自同一次重建**,坐标系才天然一致
—— 这是整个方案成立的前提,和 `2D框投影到3D模型.md` 里实测过的那条是同一个。

`bbox_to_mesh.py` 在这个工具里是**只读导入**(复用 `load_scene` / `project_points` /
`load_mesh` / `ray_triangle_hits` / `COL_FRAGMENT`),一个字节都没改 ——
所以 `bbox_live.py --verify` 那套逐位对比契约不受影响。
