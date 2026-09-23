# food 全流程 README —— 图片 → 2D 框 → 3D 区域

从 `data\food\` 里的 13 张照片,一路走到「在照片上框一块,模型上对应的那块被高亮出来」。

**本文档的数字全部来自本机实测**(数据集 `data\food`,13 张 `.JPG`,3648×2736,
2026-09-17 重建跑通,2026-09-22 补上实时投影工具)。全程 **CPU**,Windows + PowerShell。

---

## 仓库里有什么

| 目录 | 内容 | 备注 |
|---|---|---|
| `README.md` | 本文档 | |
| `docs/` | 其余手册:OpenMVS 通用手册、food 详细流程、2D→3D 专题、流程与产物对照表 | |
| `tools/` | 三个投影脚本 + 几个批处理外壳 | **核心代码** |
| `run_pipeline.sh` | ①-⑩ 一键重建脚本(Git Bash) | |
| `data/` | 三组原始照片:`food` 13 张、`building1` 41 张、`sceaux` 11 张 | |
| `work/*/mvs/` | 三个场景的**最终模型**:`scene_dense_mesh.ply`、`scene_textured.ply` + `.png`、`.glb` + `_0.png` | 可直接查看 |
| `work/*/dense/sparse/` | 去畸变后的 COLMAP 位姿 + 内参 | **投影工具的输入** |
| `work/*/sparse/0/` | SfM 稀疏模型(相机内参、位姿、稀疏点云) | 重跑去畸变的输入,只有几 MB |
| `building/OBJ_export/` | 另一个场景的 OBJ 导出 | |

**仓库里没有**(都被 `.gitignore` 排除,体积太大且可重生):

| 排除的东西 | 怎么拿回来 |
|---|---|
| `OpenMVS_Windows_x64/`、`tools/colmap/`、`tools/ffmpeg/` | 自己下载,见 [docs/OpenMVS操作手册.md](docs/OpenMVS操作手册.md) 第一节 |
| `work/*/dense/images/` | 去畸变图,重跑步骤 5 `colmap image_undistorter` 即可 |
| `work/*/mvs/depth*.dmap`、`scene_dense.ply` | 深度图与稠密点云,重跑步骤 7 `DensifyPointCloud` |
| `work/*/db.db`、`work/*/sparse/`、`*.log` | COLMAP 数据库与日志 |

> **想直接跑投影工具(第二部分)**,你还需要 `work/<场景>/dense/images/` 那一层
> —— 那是画 2D 框要用的去畸变图。仓库里没有它,补一条命令就有:
>
> ```powershell
> colmap image_undistorter --image_path "D:\code\point_cloud\0916\data\food" --input_path "D:\code\point_cloud\0916\work\food\sparse\0" --output_path "D:\code\point_cloud\0916\work\food\dense" --output_type COLMAP
> ```
>
> 输入 `work\food\sparse\0\` 和原图都在仓库里,所以这条命令能直接跑。
> 它是纯几何变换(按相机畸变模型重映射像素),同样的输入和参数产出同样的图,
> 尺寸会是文档里那个 3781×2788。
>
> **注意它会覆盖 `work\food\dense\`** —— 仓库里那份 `dense\sparse\` 会被
> 同内容覆盖,不会有影响。

---

## 0. 总览

### 一句话流程

```
data\food\  (13 张 JPG, 3648×2736)
    │
    │  ①-⑤  COLMAP:SIFT → 匹配 → SfM → 去畸变
    ▼
work\food\dense\
    ├── images\   去畸变图 3781×2788   ┐
    └── sparse\   PINHOLE 内参 + 位姿  ┘  ← ★ 投影工具的两个输入
    │
    │  ⑥  InterfaceCOLMAP    转成 OpenMVS 工程
    ▼
work\food\mvs\scene.mvs
    │
    │  ⑦  DensifyPointCloud  稠密点云(最慢)
    │  ⑧  ReconstructMesh    点云 → 网格
    │  ⑨  RefineMesh         网格细化(可选)
    │  ⑩  TextureMesh        贴图
    ▼
work\food\mvs\scene_dense_mesh.ply    纯几何网格   ← ★ 投影工具的第三个输入
work\food\mvs\scene_textured.ply+.png 带贴图       ← ★ 投影工具的显示贴图
    │
    ▼
tools\bbox_live.py  →  框中的 3D 网格碎片 + 实时可视化
```

### 两段的分界

| 段 | 干什么 | 产物 | 谁在用 |
|---|---|---|---|
| **第一部分:点云重建** | 图像 → 带纹理三维模型 | `scene_textured.ply` + `.png`、`.glb` | 交付、浏览 |
| **第二部分:2D→3D 投影** | 在图上框一块 → 找出网格上对应的区域 | 3D 碎片(`.ply`)、参数(`.json`)、可视化 | 框选、标注、量测 |

第二部分**不重跑重建**,它只消费第一部分的产物。所以重建跑一次就够,
之后换框、换图随便试 —— 投影一次 30 毫秒。

### 关键前提:两段必须来自同一次重建

投影工具需要三样东西,它们**必须是同一次重建的产物**:

1. 参考图 —— `work\food\dense\images\` 里的一张
2. 相机位姿 —— `work\food\dense\sparse\`
3. 网格 —— `work\food\mvs\`

**三者坐标系天然一致,不需要任何坐标变换。** 这不是推测,是实测:
OpenMVS 的 `TransformScene` 导出的内部变换 `xform.txt` 是**单位矩阵**:

```
1 0 0 0
0 1 0 0
0 0 1 0
0 0 0 1
```

即 OpenMVS 场景没有做任何归一化,**COLMAP 的世界系就是网格的世界系**。
把网格顶点用 COLMAP 位姿投到 13 张图上,100% 在相机前方、86.2% 落在画面内。
详见第二部分「坐标系对齐实测」。

### 目录约定

```
D:\code\point_cloud\0916\
├── data\food\              原图 13 张(直接躺在这里,没有 images\ 子目录)
├── work\food\              ← 每个场景一份,互不干扰
│   ├── db.db               COLMAP 数据库(特征 + 匹配)
│   ├── sparse\0\           稀疏重建结果
│   ├── dense\              去畸变图像 + 位姿   ← ★ 投影工具的输入
│   └── mvs\                ← ★ 模型成品和投影工具的网格都在这
├── tools\
│   ├── bbox_to_mesh.py     投影核心(算法 + 命令行)
│   ├── bbox_gui.py         投影 GUI(点选版)
│   └── bbox_live.py        投影 GUI(实时版,最新)
├── run_pipeline.sh         ①-⑩ 的一键脚本(Git Bash)
└── OpenMVS_Windows_x64\    预编译的 OpenMVS v2.4.0
```

**换数据集只改两个变量:`$W`(工作目录)和 `$IMG`(图像目录)。** 场景名不同
就自动隔离,旧结果原封不动保留。

### 环境清单

| 组件 | 位置 | 版本 |
|---|---|---|
| OpenMVS | `OpenMVS_Windows_x64\vc17\x64\Release\` | v2.4.0 |
| COLMAP | `tools\colmap\` | 4.2.0 (nocuda) |
| ffmpeg | `tools\ffmpeg\bin\` | 仅视频输入才需要 |
| Python | 任意装了 numpy 的环境 | 投影工具用 |

**为什么全是 CPU 版:** 显卡是 RTX 5070 Ti Laptop(Blackwell / sm_120),
而 OpenMVS 的 CUDA 预编译包只编了 `sm_89` 且没嵌 PTX:

```
cuModuleLoad(sm_89 cubin) -> FAIL code=209
(no kernel image is available for execution on the device)
```

所以 GPU 路径用不了,只能 CPU。本机 CPU 是 AMD Ryzen 9 9955HX(32 核),
13 张图的稠密重建约 2 分钟,可以接受。

---

# 第一部分:图像 → 带纹理三维模型

## 步骤 0:设置 PATH

**每开一个新终端都要做一次。**

```powershell
$env:PATH = "D:\code\point_cloud\0916\OpenMVS_Windows_x64\vc17\x64\Release;D:\code\point_cloud\0916\tools\colmap;D:\code\point_cloud\0916\tools\ffmpeg\bin;" + $env:PATH
```

**验证:** `colmap -h` 有输出即成功。

---

## 步骤 0.5:检查图像尺寸 ⚠️ 别跳过

```powershell
Get-ChildItem "D:\code\point_cloud\0916\data\food\*.JPG" | ForEach-Object {
    $d = & "D:\code\point_cloud\0916\tools\ffmpeg\bin\ffprobe.exe" -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 $_.FullName
    "{0}  {1}" -f $d, $_.Name
} | Sort-Object
```

**产物:** 屏幕输出,一行一张图,格式 `宽,高 文件名`。

**怎么看:** 排序后同类尺寸会聚在一起。**所有行尺寸必须完全一致**(food 这组全是
`3648,2736`)。

**为什么:** 步骤 2 用了 `--ImageReader.single_camera 1`(所有图共用一台相机),
它要求尺寸完全相同。混进一张竖拍图就会报:

```
W... feature_extraction.cc:277] IMG_xxxx.jpg CAMERA_SINGLE_DIM_ERROR:
     Single camera specified, but images have different dimensions.
```

**处理办法(二选一):**

```powershell
# 办法 A(推荐):把尺寸不同的那张移出去
New-Item -ItemType Directory -Force -Path "D:\code\point_cloud\0916\data\food\_excluded" | Out-Null
Move-Item "D:\code\point_cloud\0916\data\food\IMG_XXXX.JPG" "D:\code\point_cloud\0916\data\food\_excluded\"
```

```powershell
# 办法 B:去掉 --ImageReader.single_camera 1,让每张图各自建相机(约束弱一些)
```

---

## 步骤 1:定义变量 + 建目录

```powershell
$W   = "D:\code\point_cloud\0916\work\food"
$IMG = "D:\code\point_cloud\0916\data\food"
New-Item -ItemType Directory -Force -Path "$W\sparse","$W\dense","$W\mvs" | Out-Null
cd $W
```

**注意:** `$IMG` 填**图像所在的那一层**。food 的图直接躺在 `data\food\` 下,就填
`data\food`,**不要再加 `\images`** —— 多写一层会报
`Check failed: ExistsDir(*image_path)`。

> `$W` 只在**你敲过赋值那一行的那个终端窗口**里有效。换个窗口就是空的,
> `"$W\mvs\scene.mvs"` 会变成 `"\mvs\scene.mvs"`。所以下面所有**查看/投影命令
> 一律写绝对路径**。

---

## 步骤 2:COLMAP 特征提取

```powershell
colmap feature_extractor --database_path "$W\db.db" --image_path "$IMG" --ImageReader.single_camera 1 --FeatureExtraction.use_gpu 0 --FeatureExtraction.max_image_size 2000
```

**产物:** `$W\db.db`(含每张图的 SIFT 特征)

**怎么看:** 终端逐张打印 `Processing file [n/m]`。留意有没有 `CAMERA_SINGLE_DIM_ERROR`。

**注意:**

- `--FeatureExtraction.use_gpu 0` **必须显式传**。nocuda 版没编 GPU 代码,
  但默认值仍是 `1`,不传会直接报错。
- 参数名是 COLMAP 4.2 的新名字。旧教程里的 `--SiftExtraction.max_image_size` 已失效。
- `--max_image_size 2000` = 超过 2000px 的图先缩小,提速明显。

---

## 步骤 3:特征匹配

```powershell
colmap exhaustive_matcher --database_path "$W\db.db" --FeatureMatching.use_gpu 0
```

**产物:** 写回 `$W\db.db`(匹配关系)

**怎么看:** 终端打印 `Matching block [n/m]`。13 张图几分钟内跑完。

**注意:**

- 图像少(<200 张)用 `exhaustive_matcher`,最准。
- 图像多(>200 张)或视频抽帧(天然按时间有序)改
  `colmap sequential_matcher --database_path "$W\db.db" --FeatureMatching.use_gpu 0 --SequentialMatching.overlap 10`,
  快很多。
- `--FeatureMatching.use_gpu 0` 同样必须显式传。

---

## 步骤 4:稀疏重建(SfM)

```powershell
colmap mapper --database_path "$W\db.db" --image_path "$IMG" --output_path "$W\sparse"
```

**产物:** `$W\sparse\0\`(food 实测):

| 文件 | 大小 | 内容 |
|---|---|---|
| `cameras.bin` | 64 B | 相机内参 |
| `images.bin` | 3.4 MB | **每张图的位姿**(四元数 + 平移) |
| `points3D.bin` | 474 KB | 稀疏点云(**6015 点**) |
| `project.ini` | 2.3 KB | 工程配置,记事本可读 |

**怎么看:**

```powershell
colmap model_analyzer --path "$W\sparse\0"
```

输出里 `Registered images` 就是成功注册的图像数。**13 张图期望注册 13 张。**

**注意:** 如果 `$W\sparse\` 下出现了 `0` `1` `2` 多个子目录 → 场景断成了多块。
用注册图像最多的那个,后面所有 `sparse\0` 改成对应编号。

---

## 步骤 5:图像去畸变 ★ 这一步会改变图像尺寸

```powershell
colmap image_undistorter --image_path "$IMG" --input_path "$W\sparse\0" --output_path "$W\dense" --output_type COLMAP
```

**产物:** `$W\dense\`

| 文件 | 内容 |
|---|---|
| `images\` | **去畸变后的图像**(原图副本,**尺寸会变**) |
| `sparse\` | 对应的位姿(相机模型已变成 `PINHOLE`) |
| `stereo\` | 立体匹配配置 |

**⚠️ 这是整个流程里最需要记住的一句话:**

```
原图      data\food\IMG_2430.JPG               3648 × 2736
去畸变图  work\food\dense\images\IMG_2430.JPG  3781 × 2788
```

去畸变会把图像**拉大**(为了容纳矫正后完整的视场),而且这个过程是**非线性形变**。
第二部分画 2D 框时,必须在**去畸变图**上画,不能按 `3781/3648` 简单缩放原图坐标。

**注意:** 文件大小翻倍(每张图存了一份副本),磁盘要留够。

---

## 步骤 6:转成 OpenMVS 工程 ★ 关键人工检查点

```powershell
InterfaceCOLMAP -i "$W\dense" -o "$W\mvs\scene.mvs" --image-folder "$W\dense\images" -w "$W\mvs"
```

**产物:** `work\food\mvs\scene.mvs`(实测 313 KB)

**怎么看:**

```powershell
& "D:\code\point_cloud\0916\OpenMVS_Windows_x64\vc17\x64\Release\Viewer.exe" "D:\code\point_cloud\0916\work\food\mvs\scene.mvs"
```

打开后能看到**一堆小金字塔(相机视锥)+ 稀疏点云**。

**⚠️ 相机应该围着物体排成一圈 / 一圈半。** 如果有相机乱飞、只有两三台相机、
或者点云散成一团 —— **别往下跑了**,稠密重建只会更糟,先回头查图像重叠度。

**注意:** OpenMVS 的日志写在同目录,文件名 `InterfaceCOLMAP-<时间戳>.log`。
**终端不打印任何东西是正常的。**

---

## 步骤 7:稠密点云 ← 最慢的一步

```powershell
cd $W\mvs
DensifyPointCloud scene.mvs --resolution-level 1 --max-resolution 2000 --number-views 8
```

**产物**(food 实测):

| 文件 | 实测大小 | 内容 |
|---|---|---|
| `scene_dense.mvs` | 316 KB | 稠密工程文件 |
| `scene_dense.ply` | 76.08 MB | **稠密点云,1,112,606 点** |
| `depth0001.dmap` … `depth0013.dmap` | 各 50.25 MB | 每张图的深度图(共 653 MB) |

**怎么看:**

```powershell
& "D:\code\point_cloud\0916\OpenMVS_Windows_x64\vc17\x64\Release\Viewer.exe" "D:\code\point_cloud\0916\work\food\mvs\scene_dense.mvs"
```

**注意:**

- **深度图最占空间**,跑完确认没问题就可以删(见「产物与清理」)。
- `--resolution-level` 每 +1 快约 4 倍,代价是糊。慢得受不了就调成 2。
- `--number-views` 不能超过图像总数,图像少于 8 张时要调小。
- 日志:`DensifyPointCloud-<时间戳>.log`。

---

## 步骤 8:网格重建 ★ 这里有个大坑

```powershell
ReconstructMesh scene_dense.mvs -p scene_dense.ply -o scene_dense_mesh.ply
```

**产物:** `scene_dense_mesh.ply`,**实测 29.87 MB,824,422 顶点 / 1,648,306 面**,
耗时约 24.5 秒。

日志关键行(正确的样子):

```
[PointCld] Point-cloud 'scene_dense.ply' loaded: 1112606 points (276ms)
[Mesh    ] Delaunay tetrahedralization completed: 1112606 points -> ... (4s323ms)
[ScnRecnt] Delaunay tetrahedras graph-cut completed (1.25978e+06 flow): 824684 vertices, 1649657 faces (11s824ms)
[Mesh    ] Mesh 'scene_dense_mesh.ply' saved: 824422 vertices, 1648306 faces
```

### 坑:`-p` 是**输入点云**,不是输出网格

```
-p [ --pointcloud-file ] arg   dense point-cloud with views file name to reconstruct
                               (overwrite existing point-cloud)
-o [ --output-file ] arg       output filename for storing the mesh
```

**`-p` 传的是步骤 7 那份稠密点云 `scene_dense.ply`。** 写成
`-p scene_dense_mesh.ply`(传了个网格文件名进去)**不会报错**,而是**静默退回**
`scene_dense.mvs` 里那 6015 个稀疏点:

```
[Delaunay] Delaunay tetrahedralization completed: 6015 points     ← 少了三个数量级
```

结果只有 **2621 顶点 / 5132 面**,网格粗糙到没法贴图。

**为什么这个 bug 藏得住:** 输出文件名 `scene_dense_mesh.ply` 恰好和 OpenMVS
的默认命名一致,产物名字看起来完全正常,只有打开看或翻日志才发现面数不对。

**自检:** 日志里 `Point-cloud '...' loaded:` 那行的点数应该是**百万级**;
`Delaunay tetrahedralization completed:` 后面跟着的也应该是百万级。是 6015 就是踩坑了。

**永远显式写 `-p` 和 `-o` 两个参数。**

**怎么看:**

```powershell
& "D:\code\point_cloud\0916\OpenMVS_Windows_x64\vc17\x64\Release\Viewer.exe" "D:\code\point_cloud\0916\work\food\mvs\scene_dense_mesh.ply"
```

打开应该是**一个密度很高的完整模型**,不是几千面的粗模。

---

## 步骤 9:网格细化(可选)

```powershell
RefineMesh scene_dense.mvs -m scene_dense_mesh.ply -o scene_dense_mesh_refine.ply --scales 1 --max-face-area 16
```

**什么时候需要它:**

| 步骤 8 的网格面数 | 要不要 refine |
|---|---|
| 几千面(踩了坑) | **必须** —— 否则贴图完全出不来 |
| **百万面(正常)** | **通常不需要** —— 已经够细,refine 又慢又占内存 |

> food 目录里那份 `scene_dense_mesh_refine.ply`(286 KB / 7622 顶点)是**修复前的
> 稀疏网格**留下的残留,不是现在这份稠密网格的细化结果,**谁都配不上,可以删**。

---

## 步骤 10:纹理贴图 ★ 必须关掉 seam leveling

```powershell
TextureMesh scene_dense.mvs -m scene_dense_mesh.ply -o scene_textured.ply --global-seam-leveling 0 --local-seam-leveling 0
```

**`--global-seam-leveling 0 --local-seam-leveling 0` 这两个参数不能省。**

**产物:** `scene_textured.ply`(69.17 MB)+ `scene_textured0.png`(62.95 MB)

日志关键行:

```
[Mesh    ] Mesh 'scene_dense_mesh.ply' loaded: 824422 vertices, 1648306 faces
[ScnTextr] Assigning the best view to each face completed: 1648306 faces, 12605 patches (26s163ms)
[ScnTextr] Generating texture atlas and image completed: 7780 patches, 8192 image size, 1 textures (710ms)
[Mesh    ] Mesh 'scene_textured.ply' saved: 824422 vertices, 1648306 faces
```

### 为什么必须关:不关的话模型是橙黑两色

默认 `--global-seam-leveling 1 --local-seam-leveling 1` 在本机这个 v2.4.0
构建上,会把**面自己的 texel** 整片涂成黑/红/绿标记色,只留下向外的膨胀
(dilation)光晕 —— 而 UV 恰好只覆盖面自己的 texel,于是采样全黑。

同一份场景、同一份网格,只改这两个开关:

| 选项 | 图集纯黑 | 图集真内容 | 网格 UV 落在真照片上 | UV 处中位亮度 |
|---|---|---|---|---|
| 默认 | 24.3% | 20.4% | 2.9% | 0.0 |
| 只关 `--global-seam-leveling 0` | 24.5% | 20.2% | 0.8% | 0.0 |
| 只关 `--local-seam-leveling 0` | 5.1% | 39.6% | 83.0% | 94.0 |
| **两个都关** | **0.0%** | **64.7%** | **100.0%** | **115.3** |

**主犯是 local**,global 也贡献一小部分,一起关最干净。

**最终验收**(关掉后重跑):

| 指标 | 值 |
|---|---|
| 图集纯黑 | **0.0%** |
| 图集真内容 | 64.7% |
| 网格 UV 采样点落在真照片上 | **100.0%** |
| **逐面色差:贴图色 vs 该面某视角的照片色** | **中位 1.00 / 平均 1.36,99.7% 的面 < 15** |

最后一行才是真正的验收 —— 色差 1/255 就是 JPEG 噪声级别,说明贴图确实
**逐面复现了源照片**,不只是"有颜色了"。

**代价:** 关掉 seam leveling 就少了 patch 之间的颜色匀色,接缝处可能看得出色差。
但比整片黑强得多,而且同一组照片是同一台相机,色差本来就小。

### 出 glb(交付格式)

```powershell
TextureMesh scene_dense.mvs -m scene_dense_mesh.ply --export-type glb --global-seam-leveling 0 --local-seam-leveling 0
```

没传 `-o` 时输出名自动取「输入场景名 + `_texture`」:

| 文件 | 实测大小 | 内容 |
|---|---|---|
| `scene_dense_texture.glb` | 36.74 MB | 几何数据(内嵌) |
| `scene_dense_texture_0.png` | **63.06 MB** | 纹理贴图(**外链**) |

**⚠️ glb 不是单文件。** 它内部是这样的:

```json
"images": [ { "name": "scene_dense_texture_0", "uri": "scene_dense_texture_0.png" } ]
```

几何嵌在 glb 里,纹理指向旁边的 png。**这两个文件必须一起拷贝**,
单独把 glb 拿走会得到个没贴图的白模。

### 关于 `-o`

```
-o [ --output-file ] arg    output filename for storing the **mesh**
--export-type arg (=ply)    file type used to export the 3D scene   ← 格式由它决定
```

**`-o` 只是网格的文件名,格式由 `--export-type` 决定。TextureMesh 不写 `.mvs`
工程文件** —— 传 `-o scene_textured.mvs` 只会得到一个叫这名字的 `.ply`。

---

## 步骤 11:查看成品

```powershell
& "D:\code\point_cloud\0916\OpenMVS_Windows_x64\vc17\x64\Release\Viewer.exe" "D:\code\point_cloud\0916\work\food\mvs\scene_textured.ply"
```

**⚠️ Viewer 打不开 glb。** 打开列表里虽然有 `GLB / GLTF`,但实测报错:

```
[Scene   ] error: invalid project
[App     ] error: can not open scene '...scene_dense_texture.glb'
```

| 方式 | 说明 |
|---|---|
| **Viewer** | 看 `.mvs` / `.ply`;**不能看 glb** |
| Blender | glb 拖进去即可,**会自动按相对路径找到同目录的 png** |
| https://gltf-viewer.donmccurdy.com/ | 能看,但网页读不到本地同级文件,模型会是白的 |
| Unity / three.js | 按各自的资源导入流程,**注意 png 要一起给** |

Viewer 里还带一个 Workflow 菜单,可以不回终端直接在界面里跑
Densify / Reconstruct / Refine / Texture,调参数试效果时比敲命令行方便。

---

# 第二部分:2D 框 → 3D 区域

## 任务定义

### 输入

| 输入 | 说明 | food 对应的东西 |
|---|---|---|
| 参考图 I | 数据集里的一张**去畸变**图 | `work\food\dense\images\IMG_2435.JPG` |
| 相机位姿 | I 的内参 K + 外参 R,t | `work\food\dense\sparse\`(COLMAP 二进制模型) |
| 3D 模型 M | OpenMVS 重建的**纯几何**网格 | `work\food\mvs\scene_dense_mesh.ply` |
| 贴图(可选) | 让 3D 窗口里显示真实材质 | `work\food\mvs\scene_textured.ply` + `.png` |
| 2D 框 | 在 I 上框选目标的矩形 | 鼠标拖出来,或检测模型输出 |

### 输出

**3D 可视化**,画面里同时有:完整网格、相机位姿(光心 + 虚拟像平面)、
像平面上的 2D 框、投影到网格表面的 3D 框轮廓、被框中的网格碎片(高亮)。
另外可以导出**框中的网格碎片 `.ply`** 和**参数 `.json`**。

### 核心原理

**2D 框的 4 个角 + 相机光心 = 一个视锥(frustum)。求视锥与网格的交集。**

```
        2D 框                   3D 视锥
    ┌─────────┐
    │  (u,v)  │  ──反投影──►  从光心射出的射线
    └─────────┘                      │
                                     ▼
                          与网格求交 → 网格上的一块区域
```

两种做法,**本工具两个都做**:

| 方法 | 做法 | 用途 |
|---|---|---|
| **A. 光线投射** | 从光心穿过框的边界发射线,求与网格的交点,连成贴着表面的 3D 轮廓 | 画**红色轮廓** |
| **B. 视锥裁剪** | 把网格每个顶点投到图上,落在框内且深度为正的留下 | 取**青色碎片**(你要的 3D 区域) |

A 早期只在框的 **4 个角**打射线,角一落到背景/空洞上轮廓就断成几截。现在改成
**沿框一圈打 160 条**(每边 40 条,且不含重复的角点),打中的相邻射线连成线段 ——
断掉的只是少数几段。

## ⚠️⚠️ 框必须画在**去畸变图**上

这是整个第二部分最容易错的地方:

```
原图      data\food\IMG_2430.JPG               3648 × 2736
去畸变图  work\food\dense\images\IMG_2430.JPG  3781 × 2788
```

**坐标不能互用**,也不能简单按 `3781/3648` 缩放 —— 去畸变是非线性形变,
得走畸变模型重新映射。

- 如果 2D 框是在**原图**上标出来的(比如检测模型跑在 `data\food\` 上),
  **不能直接喂给这些工具**。
- **最简单的正确做法:让检测模型直接跑在 `work\food\dense\images\` 上。**
  这些图已经去畸变,和 COLMAP 模型里的 PINHOLE 内参严格对应,坐标拿来就能用。

程序检测到尺寸对不上会在日志里红字警告。

## 坐标系对齐实测

**位姿可以直接从 COLMAP 的 `dense\sparse\` 里读,网格直接从 `mvs\` 里读,
两者天然对齐,不需要任何坐标变换。** 三条证据:

**证据 1 —— `xform.txt` 是单位矩阵(见上文)。** OpenMVS 的 `TransformScene`
能把场景的内部变换导出:

```powershell
TransformScene -i scene_dense.mvs -o scene_text.mvs --archive-type 0 --normalize-coordinates 3 -t xform.txt
```

`--archive-type 0` = 文本格式,`--normalize-coordinates 3` = 导出内部变换的逆。
结果是单位矩阵 → **OpenMVS 场景没有做任何内部变换 → COLMAP 世界系 == 网格世界系。**

**证据 2 —— 投影检验。** 把网格顶点用 COLMAP 位姿投到每张图上,统计落在画面内的比例。
13 张图全部 74%~100% 落在画面内,深度全为正:

| 图像 | 画面内 | 深度范围 | 图像 | 画面内 | 深度范围 |
|---|---|---|---|---|---|
| IMG_2430 | 100.0% | 3.77 ~ 10.66 | IMG_2437 | 74.4% | 1.41 ~ 6.45 |
| IMG_2432 | 97.0% | 3.03 ~ 9.66 | IMG_2438 | 75.9% | 1.72 ~ 6.93 |
| IMG_2433 | 96.3% | 2.35 ~ 8.95 | IMG_2439 | 78.2% | 1.97 ~ 7.53 |
| IMG_2435 | 77.4% | 1.46 ~ 7.34 | IMG_2440 | 86.6% | 2.36 ~ 8.36 |
| IMG_2434 | 89.5% | 1.85 ~ 8.13 | IMG_2441 | 93.8% | 3.04 ~ 9.16 |
| IMG_2436 | 74.5% | 1.33 ~ 6.68 | IMG_2442 | 95.2% | 3.56 ~ 9.74 |
| | | | IMG_2443 | 99.8% | 4.05 ~ 10.75 |

> 上表用 7622 顶点的 refine 网格测得;换成 824422 顶点的稠密网格,
> IMG_2435 是 **100% 在相机前方、86.2% 落在画面内、深度 0.77~7.61**,结论一致。

**证据 3 —— 反投影往返。** 取相机前方 2/5/9 三个深度的点,投影再反投影,
射线到原点的垂距最大 `8.9e-16`(机器精度)。

### 实测参数

| 项 | 值 |
|---|---|
| 相机模型 | `PINHOLE` |
| fx = fy | 3050.236 |
| cx, cy | 1890.5, 1394.0 |
| 去畸变图尺寸 | **3781 × 2788** |
| 原图尺寸 | **3648 × 2736** |
| 网格(稠密) | 824422 顶点 / 1648306 面 |
| 稀疏点云 | 6015 点 |

### 为什么不用 `.mvs` / `pyOpenMVS`

| 来源 | 能不能用 | 原因 |
|---|---|---|
| `.mvs` | ❌ | 自定义二进制格式(`MVSI` 魔数 + 长度前缀字符串),**不是 JSON**,没有公开解析器 |
| `pymvs-parser` 包 | ❌ | PyPI 上不存在 |
| `pyOpenMVS` | ❌ | 虽然装了能 import,但 `Scene` 只暴露 `load` / `reconstruct_mesh` / `texture_mesh` 这类高层操作,**没有任何访问 views / cameras / 位姿的接口** |
| **COLMAP `dense\sparse\`** | ✅ | 世界坐标系与网格一致(上面三条证据) |

## 步骤 1:装依赖

```powershell
python -m pip install numpy trimesh open3d matplotlib Pillow
```

**验证:**

```powershell
python -c "import numpy, trimesh, open3d, matplotlib, PIL; print('全部就绪')"
```

| 包 | 用途 |
|---|---|
| `numpy` | 矩阵运算、光线求交 |
| `trimesh` | 读网格文件、导出碎片 ply |
| `open3d` | 3D 可视化 |
| `matplotlib` | 命令行版的拖框选择器 |
| `Pillow` | 读 JPG、给 Tk 界面显示图 |

> **不需要 `rtree`。** trimesh 自带的光线投射模块依赖 `rtree`(C 库
> libspatialindex 的绑定),conda 环境里经常装不上。脚本改用**自己实现的
> Möller–Trumbore 算法**(纯 numpy),彻底去掉这个依赖。
>
> **不需要 pyOpenMVS,也不需要特定 Python 版本。**

## 步骤 2:选哪个工具

三个脚本,同一套算法,`bbox_to_mesh.py` 是唯一的算法来源,另外两个只做界面:

| 脚本 | 形态 | 投影耗时 | 能导出 | 适合 |
|---|---|---|---|---|
| **`bbox_live.py`** | GUI,**实时** | **~30 ms** | ❌ | **日常首选** —— 拖框时 3D 跟着变 |
| `bbox_gui.py` | GUI,点选后按 Ctrl+Enter | ~20 s | ✅ 碎片刻 `.ply` | 不需要实时,想用按钮导出 |
| `bbox_to_mesh.py` | 核心 + 命令行 | ~20 s | ✅ 碎片 + JSON | 批处理、脚本调用、`--bbox` 给坐标 |

**依赖关系(都在同目录,CWD 无所谓):**

```
bbox_live.py ──► bbox_gui.py ──► bbox_to_mesh.py
  (实时界面)      (点选界面)       (算法核心,只读导入,一个字节都没改)
```

`bbox_live.py` 会复用 `bbox_gui.py` 里的路径推断函数,所以**三个文件必须放在
一起**,不能只拷一个走。

---

## 步骤 3:实时版(推荐)

```powershell
python "D:\code\point_cloud\0916\tools\bbox_live.py" --scene "D:\code\point_cloud\0916\work\food\dense\sparse"
```

`--images` / `--mesh` 不写会自动推:

- `--images` ← `<scene>\..\images`
- `--mesh`   ← `<scene>\..\..\mvs\` 下**和带贴图 ply 几何一致**的那份
  (food 现在是 `scene_dense_mesh.ply`)

**界面:** 左边是 Tk 窗口(图片列表 + 画布 + 日志 + 状态栏),右边独立一个
**常驻 Open3D 3D 窗口**。窗口位置是自动摆的:左边靠屏幕左侧,3D 窗口摆在它右边。

**操作:**

| 动作 | 怎么弄 |
|---|---|
| 换图 | 点左边的图名(会自动装一个居中的默认框 `0.30W,0.30H,0.70W,0.70H` 并立即投一次) |
| 框选 | 在图上按住左键拖矩形 —— **拖的过程中 3D 就跟着变**(60 ms 节流) |
| 缩放 | 滚轮(以鼠标位置为锚点) |
| 横向滚动 | `Shift` + 滚轮 |
| 清掉框 | `Esc` |
| 适应窗口 | `F` |
| 重算一次 | 按钮「重算一次」 |
| 关掉贴图 | 勾上「纯灰(不贴图)」—— 快很多 |
| 重开 3D 窗口 | 按钮「重开 3D 窗口」(关掉 3D 窗口不影响继续画框) |

**为什么快:** `bbox_to_mesh.py` 的 `ray_triangle_hits` 是拿 160 条射线对全部
1648306 个三角形暴力求交 —— 一次 20 秒。实时版在建索引时先把所有顶点投影一次,
给每个三角形记下**「投影落在图像上的哪个格子」**,求交时每条射线只查它所在格子
的候选面。剔除规则是**可以证明安全的**:射线从光心 C 穿过像素 p 打在三角形 X 上,
那么 X 必然投影到 p,所以 X 的投影包围盒必然包含 p —— 包围盒不含 p 的三角形
根本不可能被打到。

**正确性:** 计算部分一个字没改,连光线的构造顺序都照抄,而且**仍然调用原来的
`b2m.ray_triangle_hits`**,所以浮点结果逐位相同。自带自检:

```powershell
python "D:\code\point_cloud\0916\tools\bbox_live.py" --verify `
  --scene "D:\code\point_cloud\0916\work\food\dense\sparse" `
  --mesh  "D:\code\point_cloud\0916\work\food\mvs\scene_dense_mesh.ply"
```

它会用 5 个框(含一个故意反写的框和一个 1×1 像素的退化框)对比新旧实现
的 13 个返回值,打印每框耗时、总加速比,最后给结论 ——
**应该是 `✅ 全部逐位一致`**;如果出现 `❌ 有不一致的键,别用,先修。` 就别信这个工具。

**碎片闪烁怎么办:** 实时版不把选中的三角形从主网格里挖掉(百万面重建一次要
242 ms,拖起来就是幻灯片),改成把碎片**朝相机挪一点点**赢下深度测试。
默认挪 `1e-4` 的相对量,闪就加大:

```powershell
python "D:\code\point_cloud\0916\tools\bbox_live.py" --scene "..." --eps 3e-3
```

调到 `3e-3` 还闪,就说明该开 `--no-texture` 看看了。红色轮廓比碎片多挪 3 倍,
以免深度测试正好在你要看的那条边界上把它吃掉。

**其它可调项:**

| 参数 | 默认 | 说明 |
|---|---|---|
| `--cell` | `32` | 图像空间网格索引的格边长(像素)。调小更精确但索引更大 |
| `--eps` | `1e-4` | 碎片朝相机抬起的**相对**量,见上 |
| `--no-texture` | 关 | 3D 里不加载贴图,纯灰,快很多 |

**注意:**

- **实时版不能导出文件。** 要碎片 ply / JSON 用下面两个。
- 网格只在第一次投射时读一次,之后常驻内存。中途换 `--mesh` 要重启程序。
- 界面上显示的图是**去畸变后**的图,**不要**改成 `data\food\` 里的原图。

---

## 步骤 4:点选版

```powershell
python "D:\code\point_cloud\0916\tools\bbox_gui.py" --scene "D:\code\point_cloud\0916\work\food\dense\sparse"
```

只有 `--scene` 是必填,`--images` / `--mesh` 同样自动推。

**操作:** 框选 → 按 `Ctrl+Enter`(或点「投射到 3D」)→ **弹出一个阻塞的
Open3D 窗口**,关掉它才回到主界面。所以它的节奏是「框完直接看」,
不能一边拖一边看。

**比实时版多的东西:** 「导出框内网格…」按钮(存成 `.ply`,文件名默认
`<图名>_crop.ply`)。food 里现成的一个产物就是这样来的:
`work\food\mvs\IMG_2437_crop.ply`(15.4 MB)。

---

## 步骤 5:命令行 / 批处理

```powershell
python "D:\code\point_cloud\0916\tools\bbox_to_mesh.py" `
  --scene  "D:\code\point_cloud\0916\work\food\dense\sparse" `
  --images "D:\code\point_cloud\0916\work\food\dense\images" `
  --mesh   "D:\code\point_cloud\0916\work\food\mvs\scene_dense_mesh.ply" `
  --image  "IMG_2435.JPG"
```

**这个脚本四个参数全是必填**,它自己不做任何路径推断:

| 参数 | 必填 | 说明 |
|---|---|---|
| `--scene` | ✅ | COLMAP sparse 目录(含 `cameras.bin` / `images.bin`) |
| `--mesh` | ✅ | 网格文件 `.ply`(**三角形网格**,点云会直接报错) |
| `--images` | ✅ | **去畸变**图像目录 |
| `--image` | ✅ | 参考图文件名,如 `IMG_2435.JPG`(可给全路径,按 basename 匹配) |
| `--bbox` | | `XMIN YMIN XMAX YMAX`;不给则弹窗交互 |
| `--scale` | | 虚拟像平面到相机的距离;默认自动(网格正深度中位数的 0.45 倍) |
| `--crop` | | 把框中的网格碎片另存为 ply |
| `--save-json` | | 结果存 json |
| `--texture` | | 手工指定带贴图的 ply;默认在 `--mesh` 同目录自动找 |
| `--no-texture` | | 可视化时不加载贴图,只用灰色网格 |

**不给 `--bbox` 时会发生什么:**

1. 弹出参考图(matplotlib),**按住鼠标左键拖一个矩形**框住你要的目标
2. **关闭这个窗口**(右上角 ×)确认 —— 不框就关会报 `没有框选任何矩形`
3. 终端打印诊断信息(角命中情况、轮廓命中多少条、裁剪了多少三角形)
4. 弹出阻塞的 Open3D 3D 窗口

**给了 `--bbox` 就跳过弹窗**,适合批处理。

---

## 步骤 6:导出结果

```powershell
# 把框中的网格碎片存成 ply
python "D:\code\point_cloud\0916\tools\bbox_to_mesh.py" `
  --scene  "D:\code\point_cloud\0916\work\food\dense\sparse" `
  --images "D:\code\point_cloud\0916\work\food\dense\images" `
  --mesh   "D:\code\point_cloud\0916\work\food\mvs\scene_dense_mesh.ply" `
  --image  "IMG_2435.JPG" --bbox 1200 900 2400 1900 `
  --crop   "D:\code\point_cloud\0916\work\food\mvs\cropped.ply"

# 把相机参数 / 3D 角点 / 碎片包围盒存成 json
python "D:\code\point_cloud\0916\tools\bbox_to_mesh.py" ... --save-json "result.json"
```

**用 `--bbox 1200 900 2400 1900` + IMG_2435 在稠密网格上实测:**

```
4 角命中        4/4
沿框 160 条命中 160/160   → 轮廓段数 160(完整闭合)
视锥裁剪        184238/1648306 面 (11.18%)
碎片            92763 顶点 / 184238 面
碎片包围盒      min [-0.486 -0.915  3.998] / max [2.244 0.974 5.428]
耗时            19.6 秒   ← 用 bbox_live.py 同样这个框是 ~30 ms
```

> 碎片包围盒的单位**不是米** —— 重建本身是任意尺度(见「已知限制」)。

### JSON 输出格式

```json
{
  "image": "IMG_2435.JPG",
  "image_size": [3781, 2788],
  "intrinsics": {"fx": 3050.236, "fy": 3050.236, "cx": 1890.5, "cy": 1394.0},
  "camera_center_world": [...],
  "rotation_world_to_cam": [[...], [...], [...]],
  "translation": [...],
  "bbox_2d": [1200, 900, 2400, 1900],
  "bbox_3d_corners_on_mesh": [[...], [...], null, [...]],
  "n_corners_hit": 3,
  "n_outline_rays": 160,
  "n_outline_hit": 158,
  "n_triangles_in_frustum": 1234,
  "fragment_bbox_min": [...],
  "fragment_bbox_max": [...]
}
```

`bbox_3d_corners_on_mesh` 里那个 `null` 表示**这个角的射线没打到网格**
(落在背景/空洞上),不是出错。

---

## 步骤 7:3D 窗口怎么看

| 元素 | 颜色 | 含义 |
|---|---|---|
| 🟢 球 | 绿色 | 相机光心 |
| 🔵 框 | 蓝色 | 虚拟像平面(参考图所在的平面) |
| 🟡 框 | 黄色 | 你在图上画的 2D 框 |
| 🟣 线 | 品红 | 光心穿过框边界的射线(**打中了**网格) |
| ⚪ 线 | 灰色 | 同上,但**没打中**(画到 `scale×4` 就停) |
| 🔴 轮廓 | 红色 | 投影到网格表面的 3D 框 |
| 🟦 网格 | **青色** | **被框中的那部分网格(这就是你要的 3D 区域)** |
| ⬜ 网格 | 贴图 | 模型的其余部分 |

**鼠标:** 左键旋转 / 滚轮缩放 / 中键(或 Shift+左键)平移 / `Q` 退出
(`bbox_to_mesh.py` 和 `bbox_gui.py` 的窗口)。实时版的 3D 窗口不阻塞,
关掉它左边还能继续画框。

**为什么碎片用青色:** 一开始用的是橙色,结果和模型贴图(暖色)糊成一片,
完全看不出选了哪块。换成青色冷暖对比最强,任何照片贴图上都看得清。

**为什么点选版要把碎片从主网格里挖掉:** 两片三角形完全重合时深度测试打平,
先画的赢 —— 碎片会被主网格整个盖住,一点都看不见。所以主网格里被框中的那些
三角形会被挖掉,那块地方交给碎片去画。(实时版不挖,改用 `--eps` 微抬。)

---

## 内置的健壮性检查

脚本跑起来会自己报告这些问题:

```
✅ 在相机前方的顶点: 824422/824422 (100.0%)   ← 低于 30% 会警告坐标系不匹配
✅ 4 条角光线命中网格: 4/4                     ← 命中不足会指出哪个角、落在哪
✅ 视锥裁剪: 184238/1648306 个三角形 (11.2%)   ← 0 会警告"框画到空白处了"
```

---

# 第三部分:原理与公式

## 约定

OpenMVS / COLMAP 都是**计算机视觉约定**:图像坐标 y 轴**向下**,
相机坐标系 z 轴**向前**(光轴)。

```
世界点 P_w  --(R, t)-->  相机点 P_c = R · P_w + t
相机点 P_c  --(K)----->  像素   (u, v) = (fx·X/Z + cx,  fy·Y/Z + cy)
```

**相机光心**(世界坐标):

```
C = -Rᵀ · t          (因为 P_c = R·P_w + t,令 P_c = 0 得 P_w = -Rᵀt)
```

## 反投影:像素 → 射线

```python
d_cam   = K⁻¹ · [u, v, 1]ᵀ      # 相机坐标系下的方向
d_world = Rᵀ · d_cam            # 转到世界坐标系
射线:  P(s) = C + s · d_world,  s > 0
```

> 注意方向变换是 `Rᵀ` 不是 `R`。因为 `R` 把世界方向转到相机,反过来就是转置。

## 视锥裁剪

对每个网格顶点做投影,保留满足下式的:

```
xmin ≤ u ≤ xmax   AND   ymin ≤ v ≤ ymax   AND   depth > 0
```

然后取「**至少一个顶点**被选中」的三角形(`face_inside = inside[F].any(axis=1)`)。

**这是纯投影测试:不做遮挡判断、不做 z-buffer。** 所以「被框中的 3D 区域」是
**沿该视线方向、框内所有可见或不可见的表面**的并集。对「框选一块区域」这个
需求是对的(背面本来也被框住了),但**不能用它算体积 _除非_ 你知道物体是单层壳**。

**为什么它比 4 角光线可靠:** 4 角光线要求每个角都打在网格实体上。如果框的
一部分落在背景或重建空洞上(实测很常见),那个角就没有交点,3D 轮廓会缺一段。
顶点投影法完全不受影响。

## 光线-三角形求交(Möller–Trumbore)

`ray_triangle_hits()` 是这个算法的纯 numpy 实现,不依赖 trimesh 的 ray 模块。

对三角形 `(v0, e1, e2)` 和射线 `P(t) = o + t·d`:

```
p   = d × e2
det = e1 · p                     # 行列式;|det| < 1e-12 视为平行,剔除
inv = 1 / det
tv  = o - v0
u   = (tv · p) · inv             # 重心坐标 u
q   = tv × e1
v   = (q · d) · inv              # 重心坐标 v
t   = (e2 · q) · inv             # 距离
```

命中条件:

```
|det| > 1e-12   AND   u ≥ -1e-9   AND   v ≥ -1e-9   AND   u + v ≤ 1+1e-9   AND   t > 1e-9
```

取 `t` 最小的那个三角形就是最近交点。

**验证:** 用人造三角面测了 5 个用例(正中直射 / 框内斜射 / 打到三角外 / 反向射 /
平行于面),全部符合预期。真实数据上把命中点反投影回图像,和原框角的
**像素误差 ~1e-13**。

## 相机位姿从哪读

COLMAP 二进制格式(**只支持 `.bin`,不支持 `.txt` 文本模型**):

```
cameras.bin:  u64 数量,然后每个: i32 id, i32 model, u64 w, u64 h, f64 params[参数个数]
images.bin:   u64 数量,然后每个: i32 id, f64 qvec[4], f64 tvec[3], i32 cam_id,
                               字符串 name(\0 结尾), u64 npts, 然后 npts 个 (f64,f64,i64)
```

四元数 `(w,x,y,z)` 转旋转矩阵:

```python
R = [[1-2y²-2z², 2xy-2zw,   2xz+2yw],
     [2xy+2zw,   1-2x²-2z², 2yz-2xw],
     [2xz-2yw,   2yz+2xw,   1-2x²-2y²]]
```

`images.bin` 里那 2D 观测点**会被跳过不读**(`f.read(24 * npts)`),`points3D.bin`
从头到尾没被读过 —— 投影只需要内参和位姿。

**相机模型只认 `PINHOLE` 和 `SIMPLE_PINHOLE`。** 遇到 `OPENCV` / `RADIAL` 等
带畸变参数的模型会直接报错退出,提示你把 `--images` 指向 `image_undistorter`
的输出 —— 因为**畸变系数从头到尾没有被应用过**。这也是「框必须画在去畸变图上」
的代码级原因。

## OpenMVS 带贴图的 ply 格式

`scene_textured.ply` 的头:

```
ply
format binary_little_endian 1.0
comment TextureFile scene_textured0.png          ← 贴图文件名(相对路径)
element vertex 824422
property float32 x / y / z
element face 1648306
property list uint8 uint32 vertex_indices        ← 每个面的 3 个顶点下标
property list uint8 float32 texcoord             ← 每个面的 3 组 (u,v),共 6 个 float
end_header
```

二进制体:

```
[vertex 段]  824422 × 3 × float32
[face  段]  每个面:
    uint8  顶点个数 = 3
    uint32 ×3        顶点下标
    uint8  texcoord 个数 = 6
    float32 ×6       u0 v0 u1 v1 u2 v2    ← 顺序和上面 3 个顶点一一对应
```

**UV 是挂在面上、不是挂在顶点上的。** 同一个顶点在两个面里可以有不同 UV
(接缝处就是这样),所以不能按顶点索引存。这也解释了为什么必须做几何一致性校验。

**v 的方向是 OpenGL 约定:v = 0 在贴图的「底边」。** 而 open3d 采样时拿
`v * H` 当行号(从顶边数),所以**交给 open3d 之前必须翻成 `1 - v`**,
不翻的话整片贴图会采样到图集右下角的空白区。空白区是 OpenMVS 用
`--empty-color`(默认 `0xFF7F27`)填的橙色 `(255,127,39)` —— 8192×8192 的图集里
**约 55% 是这种橙色填充**,所以不翻的典型症状就是:**模型整个变成橙色**。

怎么证实的:把 UV 三角形栅格化到图集上,和「网格实际占的那块区域」求交并比 ——

| 映射 | 与网格占区的 IoU |
|---|---|
| `(u, v)` 不翻 | 0.017 |
| `(u, 1-v)` | **0.942** |
| `(1-u, v)` | 0.008 |
| `(1-u, 1-v)` | 0.327 |

绑到 open3d 上要转成**平铺的顶点对**:

```python
mesh.triangles = F                      # (M,3)
mesh.triangle_uvs = UV.reshape(-1, 2)   # (3M,2),第 i 个面的 3 组
mesh.textures = [o3d.io.read_image(png)]
mesh.triangle_material_ids = zeros(M)   # 全指到 0 号材质
```

翻 v 在 `load_texture()` 里做(而不是在 `read_ply_uv()` 里),这样
`read_ply_uv()` 保持忠实于文件内容,方向约定只出现在「喂给渲染器」这一步。

注意 `triangle_uvs` 的长度必须是 `3 × 三角形数`;如果挖掉了一部分三角形
(点选版会把框内的挖给碎片),**UV 必须用同一个掩码一起挖**,否则长度对不上,
open3d 会直接抛异常。

## 内置的路径推断:网格和贴图是怎么配上的

`default_mesh()` 不硬性偏好 refine 网格,而是挑**有配套贴图的那份**:
扫同目录所有 `.ply` 的头部找 `comment TextureFile`,把带贴图那份的
(顶点数, 面数)记下来,再拿候选网格去对 —— 对得上的就是它。

**为什么不能硬偏好 refine:** 跑没跑过 RefineMesh 不一定,而且最后一次
`TextureMesh` 贴的是谁取决于流程。硬偏好 refine 会配上几何不一致的贴图,
3D 里直接退回灰模。

`load_texture()` 找到贴图的流程:扫同目录 `.ply` 头里的 `TextureFile` →
**顶点数、面数、顶点坐标三重比对**(`atol=1e-6`)→ 读 png 绑上去。
第三步的校验不能省:`texcoord` 是**按面**存的,几何只要差一点,UV 就整个错位。

**为什么不能直接 `o3d.io.read_triangle_mesh` 读这个 ply:** open3d 的 ply 读取器
只认 `texture_u` / `texture_v` 这种属性名,OpenMVS 写的是 `texcoord`,读出来
`triangle_uvs` 是空的、`textures` 是 0 个 —— 于是模型显示成白的/灰的。
脚本自己解析了 ply(`read_ply_uv`)。

## 性能

| 操作 | 规模 | 实测 |
|---|---|---|
| `load_scene` | — | 0.00 s |
| `load_mesh` | 824422 顶点 | 0.94 s |
| `project_points` | 824422 顶点 × 13 图 | 0.03 s / 图 |
| **`project_bbox`(原始版,含 160 条射线暴力求交)** | 1648306 面 | **19.6 s** |
| **`fast_project`(实时版,带索引)** | 1648306 面 | **~30 ms** |

**原始版的瓶颈是射线求交**,它是对全部 1648306 个三角形暴力做的。
老文档里写的 109 ms 是 14992 面的 refine 网格时代的数字;网格大了 110 倍,
这一步就慢了 180 倍。

实时版的加速是**可以证明安全**的(见步骤 3),并且调用同一个
`ray_triangle_hits`,所以结果逐位相同 —— 且自带 `--verify` 让你能自己验。

---

# 第四部分:产物清单与清理

## 最终成果

| 文件 | 实测大小 | 用途 |
|---|---|---|
| `scene_dense_mesh.ply` | 29.87 MB | **纯几何网格** —— Viewer 看形状;**投影工具的 `--mesh` 输入** |
| `scene_textured.ply` + `scene_textured0.png` | 69.17 MB + 62.95 MB | **带纹理网格**,Viewer 能直接看;投影工具自动找到它当贴图 |
| `scene_dense_texture.glb` + `scene_dense_texture_0.png` | 36.74 MB + 63.06 MB | 交付格式,网页/Unity/Blender 通用,**两个文件要一起给** |
| `IMG_2437_crop.ply` | 15.4 MB | 框选导出的碎片(一次框选的结果) |

## 中间产物(确认无误后可删)

| 文件 | 实测大小 | 说明 |
|---|---|---|
| `depth0001.dmap` … `depth0013.dmap` | 各 50.25 MB | **最占空间**,13 张 = 653 MB |
| `scene_dense.ply` | 76.08 MB | 稠密点云 |
| `scene.mvs` / `scene_dense.mvs` | 0.30 / 0.31 MB | 工程文件 |
| `scene_dense_mesh_refine.ply` | 0.27 MB | 修复前稀疏网格的残留,**谁都配不上** |
| `db.db` | — | COLMAP 数据库(特征 + 匹配) |
| `dense\` | — | 去畸变图像(原图副本) |
| `scene_text.mvs` / `xform.txt` | 很小 | 只用来验证坐标系,验完可删 |

## 只保留成品

```powershell
cd $W\mvs
Remove-Item depth*.dmap, scene_dense.ply, scene.mvs, scene_dense.mvs -ErrorAction SilentlyContinue
# 顺手删掉修复前那份稀疏网格的残留(现在谁都配不上)
Remove-Item scene_dense_mesh_refine.ply, scene_text.mvs, xform.txt -ErrorAction SilentlyContinue
cd $W
Remove-Item db.db -Force -ErrorAction SilentlyContinue
Remove-Item sparse, dense -Recurse -Force -ErrorAction SilentlyContinue
```

**⚠️ 别删的:**

| 文件 | 为什么留着 |
|---|---|
| `scene_dense_mesh.ply` | 投影工具的 `--mesh` 输入 |
| `scene_textured.ply` + `scene_textured0.png` | 带贴图浏览;投影工具的贴图来源 |
| `scene_dense_texture.glb` + `scene_dense_texture_0.png` | 交付用 |
| `dense\sparse\` | 投影工具的位姿来源 |
| `dense\images\` | **投影工具画框的那一层** |

**注意:** 删掉 `depth*.dmap` / `scene_dense.ply` / `scene_dense.mvs` 之后,
**重跑 `DensifyPointCloud` 就必须从头再来**。确认不再改重建参数再删。

## 13 张图的总占用

| 阶段 | 占用 |
|---|---|
| 重建中间产物(depth + 点云 + 工程) | ≈ 730 MB |
| 成品(ply + png + glb) | ≈ 172 MB |

## 日志文件

OpenMVS 的每个程序都会在同目录留下一个日志:

```
InterfaceCOLMAP-2609172115000273D9.log
DensifyPointCloud-2609172116040339A9.log
ReconstructMesh-2609172214510259AD.log
RefineMesh-26091721195902ADB5.log
TextureMesh-2609172227490246AD.log
```

**OpenMVS 命令跑完终端没有任何输出是正常的** —— 日志都写在这里。
出问题时先翻这个,尤其是 `Delaunay tetrahedralization completed:` 的点数。

## 视频输入

输入不是图像目录而是视频的话,在步骤 2 之前加一步抽帧:

```powershell
ffmpeg -hide_banner -i "input.mp4" -vf "fps=2,scale='min(2000,iw)':-2" -q:v 2 "$W\images\frame_%05d.jpg"
```

然后 `$IMG` 指向抽帧输出目录。

---

# 第五部分:坑

## 主流程

| # | 现象 | 原因 | 解决 |
|---|---|---|---|
| 1 | `unrecognised option '--SiftExtraction.xxx'` | COLMAP 4.2 把顶层参数改名了 | 改用 `--FeatureExtraction.xxx` / `--FeatureMatching.xxx` |
| 2 | `use_gpu` 相关报错 | nocuda 版没编 GPU 代码,但默认值仍是 1 | 必须显式传 `--FeatureExtraction.use_gpu 0` 和 `--FeatureMatching.use_gpu 0` |
| 3 | `Check failed: ExistsDir(*image_path)` | `$IMG` 多写了一层 `\images` | 填图像**所在地那一层** |
| 4 | `CAMERA_SINGLE_DIM_ERROR` | `single_camera 1` 要求尺寸一致,混进了竖拍图 | 移走尺寸不同的图,或去掉 `single_camera 1` |
| 5 | **`ReconstructMesh` 日志里只有 6015 个点,网格只有几千面** | `-p` 是**输入稠密点云**,不是输出网格;把网格名传给 `-p` 会**静默退回**稀疏点 | `ReconstructMesh scene_dense.mvs -p scene_dense.ply -o scene_dense_mesh.ply`。自检看日志的 `Point-cloud '...' loaded:` 点数 |
| 6 | **贴图模型整片橙黑两色** | `TextureMesh` 默认 `--global-seam-leveling 1 --local-seam-leveling 1` 在本机 v2.4.0 构建上会把面的 texel 涂成标记色 | 加 `--global-seam-leveling 0 --local-seam-leveling 0` |
| 7 | `--ignore-mask-label -2` 直接崩溃 | 本机构建上 mask 那条代码路径有 bug(段错误) | **别用这个参数**,默认 `-1` 就好;mask 也不是黑块的元凶 |
| 8 | `-o scene_textured.mvs` 得到的是 `.ply` | `-o` 是网格文件名,格式由 `--export-type` 定;TextureMesh 不写 `.mvs` | 要 glb 就用 `--export-type glb` |
| 9 | Viewer 打不开 glb:`invalid project` | Viewer 不能导入 glTF | 用 `.mvs` / `.ply` 看;glb 用 Blender 看 |
| 10 | 交付的 glb 在别人机器上是白模 | glb 的贴图是**外链**的 png | glb 和 `*_0.png` 一起给 |
| 11 | `no kernel image is available` | CUDA 架构不匹配(sm_89 vs sm_120) | 用 CPU 版 |
| 12 | mapper 之后 `sparse\` 下有 `0` `1` 多个目录 | 场景断成多个子模型 | 用注册图像最多的那个 |
| 13 | OpenMVS 命令跑完终端没输出 | 正常,日志写在同目录 `.log` | 看 `*-<时间戳>.log` |
| 14 | 模型形状不对 / 缺一大块 | 图像重叠不足 | 提高抽帧率、降低 `max_image_size`、或改用 `exhaustive_matcher` |
| 15 | 没有绝对尺度和真实坐标 | 没有 GPS EXIF,重建结果是任意尺度 | 需要真实尺寸就在场景里放已知大小的标定物 |

## 投影工具

| # | 现象 | 原因 | 解决 |
|---|---|---|---|
| 16 | `json.loads` 读 `.mvs` 崩溃 | `.mvs` 是二进制,不是 JSON | 别读 `.mvs`,用 COLMAP 的 sparse 模型 |
| 17 | `pip install pymvs-parser` 失败 | 这个包不存在 | 同上 |
| 18 | `import pyOpenMVS` 报"找不到指定的模块" | Python 3.8+ 不再自动搜 `.pyd` 所在目录 | 本流程**不需要** pyOpenMVS,直接不装 |
| 19 | **框画在原图上,投影全偏** | 原图 3648×2736 ≠ 去畸变图 3781×2788,且形变非线性 | **框画在 `work\food\dense\images\` 上** |
| 20 | 报"相机模型是 OPENCV,带畸变参数" | 位姿是从**未去畸变**的 `sparse\` 里读的 | 用 `dense\sparse\`(经 `image_undistorter` 后相机已变 PINHOLE) |
| 21 | 3D 轮廓缺一段 | 边界射线打在背景/空洞上,没交点 | 已改成沿框打 160 条;断的只是少数几段。看青色碎片(视锥裁剪),它完全不受影响 |
| 22 | 像平面跑到模型里面 / 相机背后 | `--scale` 不合适 | 用 `--scale` 手动指定,或调大调小试 |
| 23 | **顶点大量在相机背后** | `--scene` 和 `--mesh` 不是同一次重建 | 确认两者来自同一个 `work\food\` |
| 24 | 模型没有真实尺度 | 重建本身是任意尺度 | 投影数学不受影响,只是单位不是米 |
| 25 | `mesh.ray.intersects_location` 报 `No module named 'rtree'` | trimesh 的光线投射要 rtree | 本脚本已自己实现求交,不依赖 trimesh 的 ray 模块 |
| 26 | 界面里框的坐标和实际差几个像素 | 图被缩小显示了,鼠标精度 = 1/zoom 像素 | 滚轮放大后再框,或框完看日志里的坐标 |
| 27 | 3D 里模型是白的/灰的,没贴图 | open3d 的 ply 读取器不认 OpenMVS 写的 `texcoord` | 脚本自己解析 ply(`read_ply_uv`) |
| 28 | 贴图整个花掉 / 错位 | UV 是按面存的,配到了几何不一致的 ply 上 | 三重校验(顶点数/面数/顶点坐标),对不上会打印跳过原因 |
| 29 | **把带贴图的 ply 直接传给 `--mesh`,贴图不加载** | `load_mesh` 走 trimesh,trimesh 会把「按面角存的 UV」转成「按顶点存」,**顶点数被重编**(824422 → 937775,面数不变);几何一致性检查是精确比顶点数,就判"几何不同",退回灰模 | **别传带贴图的 ply。传纯几何那份 `scene_dense_mesh.ply`,程序自己会去同目录找配套贴图** |
| 30 | 模型整个变成橙色 | UV 的 v 没翻成 `1-v`,采样到图集的橙色空白区 | 已在 `load_texture()` 里翻 |
| 31 | 碎片一片都看不见 | 碎片和主网格画在同一个位置,深度打平被盖住 | 点选版:主网格里把框中的三角形挖掉。实时版:`--eps` 微抬 |
| 32 | 碎片和模型分不清 | 碎片用了橙色,而贴图本身偏暖 | 碎片改成青色(`COL_FRAGMENT`) |
| 33 | 点「投射到 3D」后主界面卡住不动(点选版) | 3D 窗口是阻塞的,是故意的 —— 免得两个窗口抢鼠标 | 关掉 3D 窗口,或直接改用 `bbox_live.py` |
| 34 | 实时版里碎片和模型互相闪烁 | 深度冲突 | 加大 `--eps`(默认 1e-4,可试 1e-3 / 3e-3);还闪就 `--no-texture` |
| 35 | 3D 窗口关掉后左边还能画框,但没 3D 看 | 设计如此,非致命 | 点「重开 3D 窗口」 |

---

# 第六部分:换一组数据

**结果自动保留,不用做任何事** —— 工作目录按场景名隔离:

```
work\
├── food\        ← 各自独立一份
└── <新场景名>\
```

**只需改步骤 1 的两个变量**,然后从步骤 1 往下跑:

```powershell
$W   = "D:\code\point_cloud\0916\work\mynewdata"          # 新场景名
$IMG = "D:\code\point_cloud\0916\data\mynewdata"          # 新图像目录
New-Item -ItemType Directory -Force -Path "$W\sparse","$W\dense","$W\mvs" | Out-Null
cd $W
```

投影工具换成新的 `--scene` 即可,不用改任何代码:

```powershell
python "D:\code\point_cloud\0916\tools\bbox_live.py" --scene "D:\code\point_cloud\0916\work\mynewdata\dense\sparse"
```

**四条注意:**

1. **绝不要复用同一个场景名** —— `$W` 指向已有目录时,步骤 4 `mapper` 会因为
   `sparse\0` 已存在而报错或产出混乱。
2. **路径不要有中文和空格** —— COLMAP 对非 ASCII 路径支持不好。
3. **图像要够、重叠要足** —— 单航带直线飞行会失败(见第七部分)。
4. **磁盘留够** —— 总占用 ≈ 原图总量 × 2(去畸变副本)+ 图像数 × 50 MB(深度图)。

## 一键脚本

`run_pipeline.sh`(Git Bash)把第一部分的 ①-⑩ 全串好了,包含本文档的全部修复:

```bash
cd /d/code/point_cloud/0916
./run_pipeline.sh data/food               # 图像目录输入
./run_pipeline.sh data/drone.mp4 myscene  # 视频输入(自动抽帧)
FORCE=1 ./run_pipeline.sh data/food mydata # 强制重跑
```

可调环境变量:`FPS` / `MAX_WIDTH` / `MATCHER` / `RESOLUTION_LEVEL` /
`MAX_RESOLUTION` / `MAX_THREADS` / `DO_REFINE` / `FORCE`。

**⚠️ 它只跑重建,不跑投影。** 另外它的 `DO_REFINE` **默认是 1**,会贴
`scene_dense_mesh_refine.ply` 而不是本文档主线的 `scene_dense_mesh.ply`。
想和本文档完全一致就加 `DO_REFINE=0`:

```bash
DO_REFINE=0 ./run_pipeline.sh data/food
```

---

# 第七部分:已知限制

1. **无 GPU 加速**,稠密重建慢。数据量大时优先调 `--resolution-level`(每 +1 快约 4 倍)。
2. **没有绝对尺度** —— 重建是任意尺度,碎片包围盒的单位不是米。
   需要真实尺寸就在场景里放已知大小的标定物。
3. **`project_bbox` 原始版在百万面网格上要 20 秒**(暴力射线求交)。
   实时版 `bbox_live.py` 用图像空间索引降到 ~30 ms,结果逐位一致。
4. **投影工具只认 `PINHOLE` / `SIMPLE_PINHOLE`** —— 必须用去畸变图 + 去畸变后的 sparse 模型。
5. **投影工具只读 COLMAP 二进制模型**(`cameras.bin` / `images.bin`),不支持 `.txt` 文本模型。
6. **`--mesh` 必须是三角形网格**,点云会直接报错。
7. **`--mesh` 要给纯几何那份**,带贴图的 ply 会被 trimesh 重编顶点数(见坑 29)。
8. **视锥裁剪不做遮挡判断** —— 得到的是沿视线方向框内所有表面的并集(前后两面都在),
   想算体积要自己再筛。
9. **Viewer 不能导入 glTF** —— 看模型只能用 `.mvs` / `.ply`。
10. **`TextureMesh` 必须关 seam leveling**(见坑 6)—— 本机构建上关掉才出正确贴图,
    代价是少了 patch 间的颜色匀色。
11. **实时版不能导出文件** —— 要碎片 ply / JSON 用 `bbox_to_mesh.py`。
12. **进程内没有 `CreateStructure` / `ExtractKeyframes`** —— 这两个是 OpenMVS
    `develop` 分支的新功能,不在 v2.4.0 预编译包里,要自己编源码才能用。
    所以 SfM 必须外接 COLMAP,视频抽帧必须外接 ffmpeg。
13. **单航带直线飞行会失败** —— 无人机只沿一条直线飞一趟时,相机中心共线,
    无法约束绕航迹的滚转角。需要往复式(割草机式)航线、环绕拍摄或高度变化。

---

## 相关文档

| 文件 | 内容 |
|---|---|
| [`docs/OpenMVS操作手册.md`](docs/OpenMVS操作手册.md) | OpenMVS 通用操作手册(Sceaux Castle 案例) |
| [`docs/food完整流程.md`](docs/food完整流程.md) | food 流程的详细版(含逐条实测数据与推导) |
| [`docs/2D框投影到3D模型.md`](docs/2D框投影到3D模型.md) | 2D→3D 投影的专题记录(含对豆包原始方案的逐条纠错) |
| [`docs/流程与产物.md`](docs/流程与产物.md) | 全仓库的流程与产物对照表 |
