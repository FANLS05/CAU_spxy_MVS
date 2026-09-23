# food 完整流程 —— 点云重建 + 2D 框投影到 3D

从 13 张照片到「能框选的一块 3D 区域」,一条龙。

**本文档所有命令和数字都来自实测**(数据集 `data\food`,13 张 `.JPG`,
2026-09-17 跑通;最后更新于当天修完贴图和网格两处流程 bug 之后)。

---

## 0. 总览

### 一句话流程

```
data\food\  (13 张 JPG, 3648x2736)
    │
    │  ①-⑤  COLMAP:SIFT → 匹配 → SfM → 去畸变
    ▼
work\food\dense\
    ├── images\   去畸变图 3781x2788   ┐
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
work\food\mvs\scene_textured.ply + scene_textured0.png   ← ★ 投影工具的第三个输入
    │
    ▼
tools\bbox_gui.py  →  框中的 3D 网格碎片 + 可视化
```

### 两段的分界

| 段                             | 干什么                               | 产物                              | 谁在用           |
| ------------------------------ | ------------------------------------ | --------------------------------- | ---------------- |
| **第一部分:点云重建**    | 图像 → 带纹理的三维模型             | `scene_textured.ply` + `.png` | 交付、浏览       |
| **第二部分:2D→3D 投影** | 在图上框一块 → 找出网格上对应的区域 | 3D 碎片、可视化、JSON             | 框选、标注、量测 |

第二部分**不重跑重建**,它只是消费第一部分的产物。所以重建跑一次就够,
之后换框、换图随便试。

### 关键前提:两段必须来自同一次重建

投影工具需要三样东西,它们**必须是同一次重建的产物**:

1. 参考图 —— `work\<场景>\dense\images\` 里的一张
2. 相机位姿 —— `work\<场景>\dense\sparse\`
3. 网格 —— `work\<场景>\mvs\`

**三者坐标系天然一致,不需要任何坐标变换。** 这不是推测,是实测:
OpenMVS 的 `TransformScene` 导出的内部变换 `xform.txt` 是**单位矩阵**,
即 OpenMVS 场景没有做任何归一化,COLMAP 的世界系就是网格的世界系。
把网格顶点用 COLMAP 位姿投到 13 张图上,**100% 在相机前方、86.2% 落在画面内**,
完全对得上。

> 详细验证过程和数字见第二部分「坐标系对齐实测」。

### 目录约定

```
D:\code\point_cloud\0916\
├── data\<数据集>\          ← 原图放这里
├── work\<场景名>\          ← 每个场景一份,互不干扰
│   ├── db.db               COLMAP 数据库(特征 + 匹配)
│   ├── sparse\0\           稀疏重建结果
│   ├── dense\              去畸变图像 + 位姿   ← ★ 投影工具的输入
│   └── mvs\                ← ★ 模型成品在这
├── tools\
│   ├── bbox_gui.py         投影工具(图形界面)
│   └── bbox_to_mesh.py     投影工具(核心计算 + 命令行)
├── run_pipeline.sh         上面 ①-⑩ 的一键脚本(bash)
└── OpenMVS_Windows_x64\    预编译的 OpenMVS
```

**换数据集只改两个变量:`$W`(工作目录)和 `$IMG`(图像目录)。** 场景名不同
就自动隔离,旧结果原封不动保留。

### 环境清单

| 组件    | 位置                                      | 版本                           |
| ------- | ----------------------------------------- | ------------------------------ |
| OpenMVS | `OpenMVS_Windows_x64\vc17\x64\Release\` | v2.4.0(build date Jan 20 2026) |
| COLMAP  | `tools\colmap\`                         | 4.2.0 (nocuda)                 |
| ffmpeg  | `tools\ffmpeg\bin\`                     | 仅视频输入才需要               |
| Python  | 任意装了 numpy 的环境                     | 投影工具用                     |

**为什么全是 CPU 版:** 显卡是 RTX 5070 Ti Laptop(Blackwell / sm_120),
而 OpenMVS 的 CUDA 预编译包只编了 `sm_89` 且没嵌 PTX:

```
cuModuleLoad(sm_89 cubin) -> FAIL code=209
(no kernel image is available for execution on the device)
```

所以 GPU 路径用不了,只能 CPU。本机 CPU 是 AMD Ryzen 9 9955HX(32 核),
13 张图的稠密重建约 2 分钟,可以接受。

---

# 第一部分:点云重建

> 每一步:命令 → 产物 → 怎么看 → 注意。全程 PowerShell。

## 第 0 步:设置 PATH

**每开一个新终端都要做一次。**

```powershell
$env:PATH = "D:\code\point_cloud\0916\OpenMVS_Windows_x64\vc17\x64\Release;D:\code\point_cloud\0916\tools\colmap;D:\code\point_cloud\0916\tools\ffmpeg\bin;" + $env:PATH
```

**验证:** `colmap -h` 有输出即成功。

---

## 第 0.5 步:检查图像尺寸 ⚠️ 别跳过

```powershell
Get-ChildItem "D:\code\point_cloud\0916\data\food\*.JPG" | ForEach-Object {
    $d = & "D:\code\point_cloud\0916\tools\ffmpeg\bin\ffprobe.exe" -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 $_.FullName
    "{0}  {1}" -f $d, $_.Name
} | Sort-Object
```

**产物:** 屏幕输出,一行一张图,格式 `宽,高 文件名`。

**怎么看:** 排序后同类尺寸会聚在一起。**所有行尺寸必须完全一致。**

**注意:** 第 2 步用了 `--ImageReader.single_camera 1`(所有图共用一台相机),
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

## 第 1 步:定义变量 + 建目录

```powershell
$W   = "D:\code\point_cloud\0916\work\food"
$IMG = "D:\code\point_cloud\0916\data\food"
New-Item -ItemType Directory -Force -Path "$W\sparse","$W\dense","$W\mvs" | Out-Null
cd $W
```

**注意:** `$IMG` 填**图像所在的那一层**。图像直接躺在 `data\food\` 下就填
`data\food`,不要再加 `\images` —— 多写一层会报 `Check failed: ExistsDir(*image_path)`。

---

## 第 2 步:COLMAP 特征提取

```powershell
colmap feature_extractor --database_path "$W\db.db" --image_path "$IMG" --ImageReader.single_camera 1 --FeatureExtraction.use_gpu 0 --FeatureExtraction.max_image_size 2000
```

**产物:** `$W\db.db`(含每张图的 SIFT 特征)

**怎么看:** 终端逐张打印 `Processing file [n/m]`。**留意有没有
`CAMERA_SINGLE_DIM_ERROR` 警告。**

**注意:**

- `--FeatureExtraction.use_gpu 0` **必须显式传**。nocuda 版没编 GPU 代码,
  但默认值仍是 `1`,不传会直接报错。
- 参数名是 COLMAP 4.2 的新名字。旧教程里的 `--SiftExtraction.max_image_size` 已失效。
- `--max_image_size 2000` = 超过 2000px 的图先缩小,提速明显。

---

## 第 3 步:特征匹配

```powershell
colmap exhaustive_matcher --database_path "$W\db.db" --FeatureMatching.use_gpu 0
```

**产物:** 写回 `$W\db.db`(匹配关系)

**怎么看:** 终端打印 `Matching block [n/m]`。13 张图几分钟内跑完。

**注意:**

- 图像少(<200 张)用 `exhaustive_matcher`,最准。
- 图像多(>200 张)改
  `colmap sequential_matcher --database_path "$W\db.db" --FeatureMatching.use_gpu 0 --SequentialMatching.overlap 10`,
  快很多。视频抽帧的场景本来就按时间有序,`sequential` 很合适。
- `--FeatureMatching.use_gpu 0` 同样必须显式传。

---

## 第 4 步:稀疏重建(SfM)

```powershell
colmap mapper --database_path "$W\db.db" --image_path "$IMG" --output_path "$W\sparse"
```

**产物:** `$W\sparse\0\`,四个文件:

| 文件             | 内容                                |
| ---------------- | ----------------------------------- |
| `cameras.bin`  | 相机内参                            |
| `images.bin`   | **每张图的位姿**(位置 + 朝向) |
| `points3D.bin` | 稀疏点云(实测 6015 点)              |
| `project.ini`  | 工程配置                            |

**怎么看:**

```powershell
colmap model_analyzer --path "$W\sparse\0"
```

输出里 `Registered images` 就是成功注册的图像数。**13 张图期望注册 13 张。**

**注意:**

- 如果 `$W\sparse\` 下出现了 `0` `1` `2` 多个子目录 → 场景断成了多块。
  用注册图像最多的那个,后面所有 `sparse\0` 改成对应编号。
- `project.ini` 可以直接用记事本打开,里面有每张图的位姿和重投影误差。

---

## 第 5 步:图像去畸变

```powershell
colmap image_undistorter --image_path "$IMG" --input_path "$W\sparse\0" --output_path "$W\dense" --output_type COLMAP
```

**产物:** `$W\dense\`

| 文件        | 内容                                                    |
| ----------- | ------------------------------------------------------- |
| `images\` | **去畸变后的图像**(原图的副本,**尺寸会变**) |
| `sparse\` | 对应的位姿                                              |
| `stereo\` | 立体匹配配置                                            |

**怎么看:** 打开 `$W\dense\images\`,和原图对比 —— 直线应该更直。

**⚠️ 这一步会改变图像尺寸,而这一点在第二部分很重要:**

```
原图      data\food\IMG_2430.JPG               3648 x 2736
去畸变图  work\food\dense\images\IMG_2430.JPG  3781 x 2788
```

**注意:** 文件大小翻倍(每张图存了一份副本),磁盘要留够。

---

## 第 6 步:转成 OpenMVS 工程

```powershell
InterfaceCOLMAP -i "$W\dense" -o "$W\mvs\scene.mvs" --image-folder "$W\dense\images" -w "$W\mvs"
```

**产物:** `$W\mvs\scene.mvs`(实测 13 图 = 313 KB)

**怎么看:**

```powershell
& "D:\code\point_cloud\0916\OpenMVS_Windows_x64\vc17\x64\Release\Viewer.exe" "D:\code\point_cloud\0916\work\food\mvs\scene.mvs"
```

> **这里的查看命令一律写绝对路径**,不用 `$W`。`$W` 只在**你敲过赋值那一行的那个
> 终端窗口**里有效;换个窗口就是空的,`"$W\mvs\scene.mvs"` 会变成
> `"\mvs\scene.mvs"` = 当前盘根目录下的 `mvs\scene.mvs`,Viewer 直接报
> `error: can not open scene`。绝对路径在任何终端、任何 shell 里都一样。

打开后能看到**一堆小金字塔(相机视锥)+ 稀疏点云**。

**⚠️ 这是最关键的人工检查点。** 相机应该围着物体排成一圈 / 一圈半;
如果有相机乱飞、只有两三台相机、或者点云散成一团 —— **别往下跑了**,
稠密重建只会更糟。先回头查图像重叠度。

**注意:** OpenMVS 的日志写在同目录,文件名 `InterfaceCOLMAP-<时间戳>.log`。
**终端不打印任何东西是正常的。**

---

## 第 7 步:稠密点云 ← 最慢的一步

```powershell
cd $W\mvs
DensifyPointCloud scene.mvs --resolution-level 1 --max-resolution 2000 --number-views 8
```

**产物:**

| 文件                                     | 实测大小(13 图) | 内容                            |
| ---------------------------------------- | --------------- | ------------------------------- |
| `scene_dense.mvs`                      | 316 KB          | 稠密工程文件                    |
| `scene_dense.ply`                      | 76.08 MB        | **稠密点云,1,112,606 点** |
| `depth0001.dmap` … `depth0013.dmap` | 各 50.25 MB     | 每张图的深度图                  |

**怎么看:**

```powershell
& "D:\code\point_cloud\0916\OpenMVS_Windows_x64\vc17\x64\Release\Viewer.exe" "D:\code\point_cloud\0916\work\food\mvs\scene_dense.mvs"
```

**注意:**

- **深度图最占空间**,13 张 = 653 MB。跑完确认没问题就可以删(见「产物与清理」)。
- `--resolution-level` 每 +1 快约 4 倍,代价是糊。慢得受不了就调成 2。
- `--number-views` 不能超过图像总数,图像少于 8 张时要调小。
- 日志:`DensifyPointCloud-<时间戳>.log`。

---

## 第 8 步:网格重建 ★ 这里有两个大坑

```powershell
ReconstructMesh scene_dense.mvs -p scene_dense.ply -o scene_dense_mesh.ply
```

**产物:** `scene_dense_mesh.ply`,**实测 29.87 MB,824,422 顶点 / 1,648,306 面**,
耗时约 24.5 秒。

日志关键行:

```
[PointCld] Point-cloud 'scene_dense.ply' loaded: 1112606 points (276ms)
[Mesh    ] Delaunay tetrahedralization completed: 1112606 points -> ... (4s323ms)
[ScnRecnt] Delaunay tetrahedras graph-cut completed (1.25978e+06 flow): 824684 vertices, 1649657 faces (11s824ms)
[Mesh    ] Mesh 'scene_dense_mesh.ply' saved: 824422 vertices, 1648306 faces
```

### 坑 1:`-p` 是**输入点云**,不是输出网格

这是本节最重要的一句话:

```
-p [ --pointcloud-file ] arg   dense point-cloud with views file name to reconstruct
                               (overwrite existing point-cloud)
-o [ --output-file ] arg       output filename for storing the mesh
```

**`-p` 传的是第 7 步那份稠密点云 `scene_dense.ply`。** 写成
`-p scene_dense_mesh.ply`(传了个网格文件名进去)不会报错,而是**静默退回**
`scene_dense.mvs` 里那 6015 个稀疏点:

```
[Delaunay] Delaunay tetrahedralization completed: 6015 points     ← 少了三个数量级
```

结果只有 **2621 顶点 / 5132 面**,网格粗糙到没法贴图(见第 10 步)。

**为什么这个 bug 藏得住:** 输出文件名 `scene_dense_mesh.ply` 恰好和 OpenMVS
的默认命名一致,所以产物名字看起来完全正常,只有打开看或翻日志才发现面数不对。

**自检:** 日志里 `Point-cloud '...' loaded:` 那行的点数应该是**百万级**;
`Delaunay tetrahedralization completed:` 后面跟着的也应该是百万级。是 6015
就是踩了这个坑。

### 坑 2:和默认命名撞车

`ReconstructMesh` 默认输出名就是 `<输入场景名>_mesh.ply`。所以如果**同时**
指定 `-o` 又不小心复用 `-p` 的位置,很容易得到"看着对、实际是稀疏网格"的结果。
**永远显式写 `-p` 和 `-o` 两个参数。**

**怎么看:**

```powershell
& "D:\code\point_cloud\0916\OpenMVS_Windows_x64\vc17\x64\Release\Viewer.exe" "D:\code\point_cloud\0916\work\food\mvs\scene_dense_mesh.ply"
```

打开应该是**一个密度很高的完整模型**,不是几千面的粗模。

---

## 第 9 步:网格细化(可选)

```powershell
RefineMesh scene_dense.mvs -m scene_dense_mesh.ply -o scene_dense_mesh_refine.ply --scales 1 --max-face-area 16
```

**产物:** `scene_dense_mesh_refine.ply`

**什么时候需要它:**

| 第 8 步的网格面数 | 要不要 refine                                          |
| ----------------- | ------------------------------------------------------ |
| 几千面(踩了坑 1)  | **必须** —— 否则贴图完全出不来                 |
| 百万面(正常)      | **通常不需要** —— 已经够细,refine 又慢又占内存 |

**注意:** 本目录里现在那份 `scene_dense_mesh_refine.ply`(0.27 MB / 7622 顶点)
是**修复前的稀疏网格**留下的残留,不是现在这份稠密网格的细化结果。它谁都配不上,
可以删(见「产物与清理」)。

---

## 第 10 步:纹理贴图 ★ 必须关掉 seam leveling

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

| 选项                             | 图集纯黑       | 图集真内容      | 网格 UV 落在真照片上 | UV 处中位亮度   |
| -------------------------------- | -------------- | --------------- | -------------------- | --------------- |
| 默认                             | 24.3%          | 20.4%           | 2.9%                 | 0.0             |
| 只关`--global-seam-leveling 0` | 24.5%          | 20.2%           | 0.8%                 | 0.0             |
| 只关`--local-seam-leveling 0`  | 5.1%           | 39.6%           | 83.0%                | 94.0            |
| **两个都关**               | **0.0%** | **64.7%** | **100.0%**     | **115.3** |

**主犯是 local**,global 也贡献一小部分,一起关最干净。

另一个旁证:开 seam leveling 时 1648306 个面被切成 **112897 个 patch**
(平均 15 面/patch),关掉后只剩 **12605 个**(平均 131 面/patch)。
差 9 倍,说明它当时就是在面级别乱切。

**最终验收**(关掉后重跑):

| 指标                                            | 值                                              |
| ----------------------------------------------- | ----------------------------------------------- |
| 图集纯黑                                        | **0.0%**                                  |
| 图集真内容                                      | 64.7%                                           |
| 网格 UV 采样点落在真照片上                      | **100.0%**                                |
| **逐面色差:贴图色 vs 该面某视角的照片色** | **中位 1.00 / 平均 1.36,99.7% 的面 < 15** |

最后一行才是真正的验收 —— 色差 1/255 就是 JPEG 噪声级别,说明贴图确实
逐面复现了源照片,不只是"有颜色了"。

**代价:** 关掉 seam leveling 就少了 patch 之间的颜色匀色,接缝处可能看得出
色差。但比整片黑强得多,而且同一次拍摄的图色差本来就小。

### 出 glb(交付格式)

```powershell
TextureMesh scene_dense.mvs -m scene_dense_mesh.ply --export-type glb --global-seam-leveling 0 --local-seam-leveling 0
```

没传 `-o` 时输出名自动取「输入场景名 + `_texture`」,所以产物是:

| 文件                          | 实测大小           | 内容                     |
| ----------------------------- | ------------------ | ------------------------ |
| `scene_dense_texture.glb`   | 36.74 MB           | 几何数据(内嵌)           |
| `scene_dense_texture_0.png` | **63.06 MB** | 纹理贴图(**外链**) |

**⚠️ glb 不是单文件!** glb 内部是这样的:

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

## 第 11 步:查看成品

```powershell
& "D:\code\point_cloud\0916\OpenMVS_Windows_x64\vc17\x64\Release\Viewer.exe" "D:\code\point_cloud\0916\work\food\mvs\scene_textured.ply"
```

**这是唯一的成品查看入口** —— 打开就是第 10 步贴好图的模型。

> **为什么绝对路径:** 见第 6 步那个说明 —— `$W` 只在你敲过赋值的那一个终端窗口里
> 有效。查看成品通常是你**新开一个窗口、或者隔天再回来看**的时候,这时候 `$W`
> 早就没了。绝对路径直接粘就能跑。
>
> 换场景的话把中间那两段换掉即可:
> `...\work\<场景名>\mvs\<文件名>`。

**⚠️ Viewer 打不开 glb。** 打开列表里虽然有 `GLB / GLTF`,但实测报错:

```
[Scene   ] error: invalid project
[App     ] error: can not open scene '...scene_dense_texture.glb'
```

**看模型请用 `.mvs` 或 `.ply`。**

| 方式                                | 说明                                                      |
| ----------------------------------- | --------------------------------------------------------- |
| Blender                             | glb 拖进去即可,**会自动按相对路径找到同目录的 png** |
| https://gltf-viewer.donmccurdy.com/ | 能看,但网页读不到本地同级文件,模型会是白的                |
| Unity / three.js                    | 按各自的资源导入流程,注意 png 要一起                      |

Viewer 里还带一个 Workflow 菜单,可以不回终端直接在界面里跑
Densify / Reconstruct / Refine / Texture,调参数试效果时比敲命令行方便。

---

## 主流程产物总表

### 最终成果

| 文件                                                        | 实测大小            | 用途                                                        |
| ----------------------------------------------------------- | ------------------- | ----------------------------------------------------------- |
| `scene_textured.ply` + `scene_textured0.png`            | 69.17 MB + 62.95 MB | **带纹理网格,Viewer 能直接看**;投影工具也用它         |
| `scene_dense_texture.glb` + `scene_dense_texture_0.png` | 36.74 MB + 63.06 MB | 交付格式,网页/Unity/Blender 通用,**两个文件要一起给** |
| `scene_dense_mesh.ply`                                    | 29.87 MB            | **纯几何网格**(无纹理),看形状、做投影用               |

### 中间产物(占空间,确认无误后可删)

| 文件                                     | 实测大小       | 说明                              |
| ---------------------------------------- | -------------- | --------------------------------- |
| `depth0001.dmap` … `depth0013.dmap` | 各 50.25 MB    | **最占空间**,13 张 = 653 MB |
| `scene_dense.ply`                      | 76.08 MB       | 稠密点云                          |
| `scene.mvs` / `scene_dense.mvs`      | 0.30 / 0.31 MB | 工程文件                          |
| `db.db`                                | —             | COLMAP 数据库(特征 + 匹配)        |
| `dense\`                               | —             | 去畸变图像(原图副本)              |

### 日志文件

OpenMVS 的每个程序都会在同目录留下一个日志:

```
DensifyPointCloud-2609172116040339A9.log
InterfaceCOLMAP-2609172115000273D9.log
ReconstructMesh-2609172214510259AD.log
RefineMesh-26091721195902ADB5.log
TextureMesh-2609172706031C4D.log
```

**OpenMVS 命令跑完终端没有任何输出是正常的** —— 日志都写在这里。
出问题时先翻这个,尤其是 `Delaunay tetrahedralization completed:` 的点数
(见第 8 步坑 1)。

---

# 第二部分:2D 框投影到 3D

## 任务是什么

### 输入

| 输入      | 说明                  | food 对应的东西                         |
| --------- | --------------------- | --------------------------------------- |
| 参考图 I  | 数据集里的一张图      | `work\food\dense\images\IMG_2435.JPG` |
| 相机位姿  | I 的内参 K + 外参 R,t | `work\food\dense\sparse\`             |
| 3D 模型 M | OpenMVS 重建的网格    | `work\food\mvs\scene_dense_mesh.ply`  |
| 2D 框     | 在 I 上框选目标的矩形 | 鼠标拖出来,或检测模型输出               |

### 输出

**3D 可视化**,画面里同时有:完整网格、相机位姿(光心 + 虚拟像平面)、
像平面上的 2D 框、投影到网格表面的 3D 框轮廓、被框中的网格碎片(高亮)。

### 核心原理

**2D 框的 4 个角 + 相机光心 = 一个视锥(frustum)。求视锥与网格的交集。**

```
        2D 框                   3D 视锥
    ┌─────────┐
    │  (u,v)  │  ──反投影──►  从光心射出的 4 条射线
    └─────────┘                      │
                                     ▼
                          与网格求交 → 网格上的一块区域
```

两种做法:

| 方法                  | 做法                                                           | 适用                            |
| --------------------- | -------------------------------------------------------------- | ------------------------------- |
| **A. 光线投射** | 从光心穿过框的边界发射线,求与网格的交点,连成贴着表面的 3D 轮廓 | 要轮廓;射线打到的必须是网格实体 |
| **B. 视锥裁剪** | 把网格每个顶点投到图上,落在框内且深度为正的留下                | 要完整的 3D 区域,不怕空洞       |

**本工具两个都做**:B 用来取网格碎片(青色),A 用来画表面的 3D 框轮廓(红色)。

A 早期只在框的 **4 个角**打射线,角一落到背景/空洞上轮廓就断成几截。现在改成
**沿框一圈打 160 条**(每边 40 条),打中的相邻射线连成线段 —— 断掉的只是少数几段。

## ⚠️⚠️ 框必须画在**去畸变图**上

这是整个第二部分最容易错的地方:

```
原图      data\food\IMG_2430.JPG               3648 x 2736
去畸变图  work\food\dense\images\IMG_2430.JPG  3781 x 2788
```

**去畸变会把图像"拉大"**(为了容纳矫正后完整的视场),而且这个过程是
**非线性形变**。所以:

- 如果 2D 框是在**原图**上标出来的(比如检测模型跑在 `data\food\` 上),
  **坐标不能直接喂给这个工具**,也不能简单按 `3781/3648` 缩放 —— 得走畸变
  模型重新映射。
- **最简单的正确做法:让检测模型直接跑在 `work\<场景>\dense\images\` 上。**
  这些图已经去畸变,和 COLMAP 模型里的 PINHOLE 内参严格对应,坐标拿来就能用。

程序检测到尺寸对不上会红字警告。

## 坐标系对齐实测

**位姿可以直接从 COLMAP 的 `dense\sparse\` 里读,网格直接从 `mvs\` 里读,
两者天然对齐,不需要任何坐标变换。** 三条证据:

**证据 1 —— `xform.txt` 是单位矩阵。** OpenMVS 的 `TransformScene` 能把场景的
内部变换导出:

```powershell
TransformScene -i scene_dense.mvs -o scene_text.mvs --archive-type 0 --normalize-coordinates 3 -t xform.txt
```

`--archive-type 0` = 文本格式,`--normalize-coordinates 3` = 导出内部变换的逆。
结果:

```
1 0 0 0
0 1 0 0
0 0 1 0
0 0 0 1
```

**单位矩阵 → OpenMVS 场景没有做任何内部变换 → COLMAP 世界系 == OpenMVS 网格世界系。**

**证据 2 —— 投影检验。** 把网格顶点用 COLMAP 位姿投到每张图上,统计落在画面内的比例。
13 张图全部 74%~100% 落在画面内,深度全为正。(下表用 7622 顶点的 refine 网格测得;
换成 824422 顶点的稠密网格,IMG_2435 是 100% 在相机前方、86.2% 落在画面内、
深度 0.77~7.61,结论一致。)

| 图像     | 画面内 | 深度范围     |  | 图像     | 画面内 | 深度范围     |
| -------- | ------ | ------------ | - | -------- | ------ | ------------ |
| IMG_2430 | 100.0% | 3.77 ~ 10.66 |  | IMG_2437 | 74.4%  | 1.41 ~ 6.45  |
| IMG_2432 | 97.0%  | 3.03 ~ 9.66  |  | IMG_2438 | 75.9%  | 1.72 ~ 6.93  |
| IMG_2433 | 96.3%  | 2.35 ~ 8.95  |  | IMG_2439 | 78.2%  | 1.97 ~ 7.53  |
| IMG_2435 | 77.4%  | 1.46 ~ 7.34  |  | IMG_2440 | 86.6%  | 2.36 ~ 8.36  |
| IMG_2434 | 89.5%  | 1.85 ~ 8.13  |  | IMG_2441 | 93.8%  | 3.04 ~ 9.16  |
| IMG_2436 | 74.5%  | 1.33 ~ 6.68  |  | IMG_2442 | 95.2%  | 3.56 ~ 9.74  |
|          |        |              |  | IMG_2443 | 99.8%  | 4.05 ~ 10.75 |

**证据 3 —— 反投影往返。 取相机前方 2/5/9 三个深度的点,投影再反投影,
射线到原点的垂距最大 `8.9e-16`(机器精度)。

### 实测参数

| 项           | 值                       |
| ------------ | ------------------------ |
| 相机模型     | `PINHOLE`              |
| fx = fy      | 3050.236                 |
| cx, cy       | 1890.5, 1394.0           |
| 去畸变图尺寸 | **3781 × 2788**   |
| 原图尺寸     | **3648 × 2736**   |
| 网格(稠密)   | 824422 顶点 / 1648306 面 |

### 为什么不用 `.mvs` / `pyOpenMVS`

| 来源                               | 能不能用 | 原因                                                                                                                                                     |
| ---------------------------------- | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `.mvs`                           | ❌       | 自定义二进制格式(`MVSI` 魔数 + 长度前缀字符串),**不是 JSON**,没有公开解析器                                                                      |
| `pymvs-parser` 包                | ❌       | PyPI 上不存在                                                                                                                                            |
| `pyOpenMVS`                      | ❌       | 虽然装了能 import,但`Scene` 只暴露 `load` / `reconstruct_mesh` / `texture_mesh` 这类高层操作,**没有任何访问 views / cameras / 位姿的接口** |
| **COLMAP `dense\sparse\`** | ✅       | 世界坐标系与网格一致(上面的三条证据)                                                                                                                     |

## 第 1 步:装依赖

```powershell
python -m pip install numpy trimesh open3d matplotlib Pillow
```

**验证:**

```powershell
python -c "import numpy, trimesh, open3d, matplotlib, PIL; print('全部就绪')"
```

| 包             | 用途             |
| -------------- | ---------------- |
| `numpy`      | 矩阵运算         |
| `trimesh`    | 读网格文件       |
| `open3d`     | 3D 可视化        |
| `matplotlib` | 弹窗里拖框选目标 |
| `Pillow`     | 读 JPG           |

> **不需要 `rtree`。** trimesh 自带的光线投射模块依赖 `rtree`(C 库
> libspatialindex 的绑定),conda 环境里经常装不上。本脚本改用**自己实现的
> Möller–Trumbore 算法**(纯 numpy),彻底去掉这个依赖。
>
> 代价是暴力求交,网格大了会慢 —— 见「性能」一节。

> **不需要 pyOpenMVS,也不需要特定 Python 版本。**

## 第 2 步:图形界面版(推荐)

```powershell
python "D:\code\point_cloud\0916\tools\bbox_gui.py" --scene "D:\code\point_cloud\0916\work\food\dense\sparse"
```

`--mesh` 和 `--images` 不写会自动推:

- `--images` ← `<scene>\..\images`
- `--mesh`   ← `<scene>\..\..\mvs\` 下**和带贴图 ply 几何一致**的那份
  (food 现在是 `scene_dense_mesh.ply`)

推不出来再手动指定:

```powershell
python "D:\code\point_cloud\0916\tools\bbox_gui.py" `
  --scene  "D:\code\point_cloud\0916\work\building1\dense\sparse" `
  --images "D:\code\point_cloud\0916\work\building1\dense\images" `
  --mesh   "D:\code\point_cloud\0916\work\building1\mvs\scene_dense_mesh.ply"
```

**界面长什么样:**

```
┌──────────────────────────────────────────────────────────────┐
│ [投射到3D] [清空框选] [适应窗口] [导出框内网格] □纯灰 [打开3D窗口] │
│ 场景: work\food\dense\sparse                                  │
│ 图像: work\food\dense\images                                  │
│ 网格: work\food\mvs\scene_dense_mesh.ply                      │
├───────────┬──────────────────────────────────────────────────┤
│ IMG_2430  │                                                  │
│ IMG_2432  │           参考图(可滚轮缩放)                     │
│ IMG_2433  │        ┌────────────┐                            │
│ IMG_2435  │        │  拖出来的框 │                            │
│ ...       │        └────────────┘                            │
│ (13 张)   │                                                  │
├───────────┴──────────────────────────────────────────────────┤
│ 日志: 2D框 / 角命中 4/4 / 轮廓 160/160 / 视锥裁剪 184238/1648306 │
├──────────────────────────────────────────────────────────────┤
│ 状态: 框选完成,按 Ctrl+Enter 投射到 3D                        │
└──────────────────────────────────────────────────────────────┘
```

**操作:**

| 动作          | 怎么弄                                 |
| ------------- | -------------------------------------- |
| 换图          | 点左边的图名                           |
| 框选          | 在图上按住左键拖一个矩形               |
| 缩放          | 滚轮(以鼠标位置为锚点),`Ctrl` 不用按 |
| 横向滚动      | `Shift` + 滚轮                       |
| 上下/左右滚动 | 拖右边/下边的滚动条                    |
| 投射到 3D     | `Ctrl+Enter` 或点按钮                |
| 清掉框        | `Esc`                                |
| 适应窗口      | `F`                                  |
| 导出框内网格  | 「导出框内网格…」按钮,存成`.ply`    |
| 关掉贴图      | 勾上「网格显示成纯灰」                 |

**和命令行版比,多了什么:**

- 不用每次重开脚本换图 —— 13 张图直接点着换
- 框选后**先看日志再决定要不要开 3D 窗口** —— 命令行版是框完直接弹窗
- 「导出框内网格」按钮,不用再记 `--crop` 参数
- 窗口标题和日志里都写着当前图名,不会框错图
- 贴图只在第一次投射时读一次(63 MB 的 8192×8192 png,约几秒),
  之后每次开 3D 窗口都是瞬开

**注意:**

- 3D 窗口是**阻塞**的 —— 它开着的时候主界面点不动,关掉它才回来。这是故意的,
  免得两个窗口抢鼠标。
- 网格只在第一次投射时读一次,之后常驻内存。中途换 `--mesh` 要重启程序。
- 界面上显示的图是**去畸变后**的图,**不要**改成 `data\<场景>\` 里的原图。

## 第 3 步:命令行版(交互式框选)

```powershell
python "D:\code\point_cloud\0916\tools\bbox_to_mesh.py" `
  --scene  "D:\code\point_cloud\0916\work\food\dense\sparse" `
  --images "D:\code\point_cloud\0916\work\food\dense\images" `
  --mesh   "D:\code\point_cloud\0916\work\food\mvs\scene_dense_mesh.ply" `
  --image  "IMG_2435.JPG"
```

**会发生什么:**

1. 弹出参考图,**按住鼠标左键拖一个矩形**框住你要的目标
2. **关闭这个窗口**(右上角 ×)确认
3. 终端打印诊断信息(角命中情况、轮廓命中多少条、裁剪了多少三角形)
4. 弹出 Open3D 3D 窗口,所有元素都画在里面

## 第 4 步:看结果

在 Open3D 窗口里:

| 元素    | 颜色           | 含义                                               |
| ------- | -------------- | -------------------------------------------------- |
| 🟢 球   | 绿色           | 相机光心                                           |
| 🔵 框   | 蓝色           | 虚拟像平面(参考图所在的平面)                       |
| 🟡 框   | 黄色           | 你在图上画的 2D 框                                 |
| 🟣 线   | 品红           | 光心穿过框边界的射线(打中的)                       |
| ⚪ 线   | 灰色           | 同上,但没打中网格的                                |
| 🔴 轮廓 | 红色           | 投影到网格表面的 3D 框                             |
| 🟦 网格 | **青色** | 被框中的那部分网格(**这就是你要的 3D 区域**) |
| ⬜ 网格 | 贴图           | 模型的其余部分                                     |

**鼠标:** 左键旋转 / 滚轮缩放 / 中键(或 Shift+左键)平移 / `Q` 退出。

**为什么碎片用青色不用橙色:** 一开始用的是橙色,结果和模型贴图(暖色)糊成一片,
完全看不出选了哪块。换成青色冷暖对比最强,任何照片贴图上都看得清。

> **碎片和主网格不能画在同一个位置。** 两片三角形完全重合时深度测试打平,
> 先画的赢 —— 碎片会被主网格整个盖住,一点都看不见。所以主网格里
> **被框中的那些三角形会被挖掉**,那块地方交给碎片去画。

**模型是带贴图的:** 只要 `--mesh` 那个目录里有 `TextureMesh` 产出的
`scene_textured.ply`,3D 窗口里显示的就是贴图模型,不是灰模。程序自动找、自动校验。

## 第 5 步(可选):导出结果

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

`--bbox XMIN YMIN XMAX YMAX` 直接给坐标就**跳过交互弹窗**,适合批处理。

**用 `--bbox 1200 900 2400 1900` + IMG_2435 在稠密网格上实测:**

```
4 角命中        4/4
沿框 160 条命中 160/160   → 轮廓段数 160(完整闭合)
视锥裁剪        184238/1648306 面 (11.18%)
碎片            92763 顶点 / 184238 面
碎片包围盒      min [-0.486 -0.915  3.998] / max [2.244 0.974 5.428]
耗时            19.6 秒  ← 见「性能」
```

## 工具脚本说明

| 文件                      | 干什么                         |
| ------------------------- | ------------------------------ |
| `tools\bbox_gui.py`     | 图形界面:选图 + 框选 + 3D 浏览 |
| `tools\bbox_to_mesh.py` | 核心计算 + 命令行入口          |

`bbox_gui.py` 里的界面代码**不含任何数学** —— 位姿读取、光线投射、可视化全部从
`bbox_to_mesh.py` 导入(`load_scene` / `load_mesh` / `project_bbox` / `visualize`)。
改算法只需要改一个文件。

### `bbox_gui.py` 参数

| 参数             | 必填 | 说明                                        |
| ---------------- | ---- | ------------------------------------------- |
| `--scene`      | ✅   | COLMAP sparse 目录                          |
| `--images`     |      | 去畸变图目录,默认`<scene>\..\images`      |
| `--mesh`       |      | 网格`.ply`,默认自动挑「有配套贴图的那份」 |
| `--no-texture` |      | 3D 里网格显示成纯灰                         |

### `bbox_to_mesh.py` 参数

| 参数             | 必填 | 说明                                                    |
| ---------------- | ---- | ------------------------------------------------------- |
| `--scene`      | ✅   | COLMAP sparse 目录(含`cameras.bin` / `images.bin`)  |
| `--mesh`       | ✅   | 网格文件`.ply`                                        |
| `--images`     | ✅   | **去畸变**图像目录                                |
| `--image`      | ✅   | 参考图文件名                                            |
| `--bbox`       |      | `XMIN YMIN XMAX YMAX`;不给则弹窗交互                  |
| `--scale`      |      | 虚拟像平面到相机的距离;默认自动(网格中位深度的 0.45 倍) |
| `--crop`       |      | 把框中的网格碎片另存为 ply                              |
| `--save-json`  |      | 结果存 json                                             |
| `--texture`    |      | 手工指定带贴图的 ply;默认在`--mesh` 同目录自动找      |
| `--no-texture` |      | 可视化时不加载贴图,只用灰色网格                         |

### 网格是怎么挑的

`default_mesh()` 不硬性偏好 refine 网格,而是挑**有配套贴图的那份**:
扫同目录所有 `.ply` 的头部找 `comment TextureFile`,把带贴图那份的
(顶点数, 面数)记下来,再拿候选网格去对 —— 对得上的就是它。

**为什么不能硬偏好 refine:** 跑没跑过 RefineMesh 不一定,而且最后一次
`TextureMesh` 贴的是谁取决于流程。硬偏好 refine 会配上几何不一致的贴图,
3D 里直接退回灰模。

### 贴图是怎么找到的

`TextureMesh` 的产物是 `scene_textured.ply` + `scene_textured0.png` 两个文件,
顶点数/面数和被贴的那份网格**完全一样** —— 只是多带了每面的 `texcoord`:

1. 扫 `--mesh` 同目录下所有 `.ply`,看头里有没有 `comment TextureFile ...`
2. 逐个解析,**顶点数、面数、顶点坐标三重比对**,和 `--mesh` 对不上就跳过
3. 读 `TextureFile` 指向的 png,绑到网格上

第三步的校验不能省:`texcoord` 是**按面**存的,几何只要差一点,UV 就整个错位。

**为什么不能直接 `o3d.io.read_triangle_mesh` 读这个 ply:** open3d 的 ply 读取器
只认 `texture_u` / `texture_v` 这种属性名,OpenMVS 写的是 `texcoord`,读出来
`triangle_uvs` 是空的、`textures` 是 0 个 —— 于是模型显示成白的/灰的。
本脚本自己解析了 ply(见 `read_ply_uv`)。

### 内置的健壮性检查

脚本跑起来会自己报告这些问题:

```
✅ 在相机前方的顶点: 824422/824422 (100.0%)   ← 低于 30% 会警告坐标系不匹配
✅ 4 条角光线命中网格: 4/4                     ← 命中不足会指出哪个角、落在哪
✅ 视锥裁剪: 184238/1648306 个三角形 (11.2%)   ← 0 会警告"框画到空白处了"
```

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
  "n_triangles_in_frustum": 1234,
  "fragment_bbox_min": [...],
  "fragment_bbox_max": [...]
}
```

## 原理与公式

### 约定

OpenMVS / COLMAP 都是**计算机视觉约定**:图像坐标 y 轴**向下**,
相机坐标系 z 轴**向前**(光轴)。

```
世界点 P_w  --(R, t)-->  相机点 P_c = R · P_w + t
相机点 P_c  --(K)----->  像素   (u, v) = (fx·X/Z + cx,  fy·Y/Z + cy)
```

**相机光心**(世界坐标):

```
C = -Rᵀ · t
```

(因为 `P_c = R·P_w + t`,令 `P_c = 0` 得 `P_w = -Rᵀt`)

### 反投影:像素 → 射线

```python
d_cam  = K⁻¹ · [u, v, 1]ᵀ        # 相机坐标系下的方向
d_world = Rᵀ · d_cam            # 转到世界坐标系
射线:  P(s) = C + s · d_world,  s > 0
```

> 注意方向变换是 `Rᵀ` 不是 `R`。因为 `R` 把世界方向转到相机,反过来就是转置。

### 视锥裁剪

对每个网格顶点做投影,保留满足下式的:

```
xmin ≤ u ≤ xmax   AND   ymin ≤ v ≤ ymax   AND   depth > 0
```

然后取"至少一个顶点被选中"的三角形。

**为什么这个比 4 角光线更可靠:** 4 角光线要求每个角都打在网格实体上。如果框的
一部分落在背景或重建空洞上(实测很常见),那个角就没有交点,3D 轮廓会缺一段。
顶点投影法不受影响。

### 光线-三角形求交(Möller–Trumbore)

脚本里 `ray_triangle_hits()` 是这个算法的纯 numpy 实现,不依赖 trimesh 的 ray 模块。

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
|det| > 1e-12   AND   u ≥ 0   AND   v ≥ 0   AND   u + v ≤ 1   AND   t > 0
```

取 `t` 最小的那个三角形就是最近交点。

**验证:** 用人造三角面测了 5 个用例(正中直射 / 框内斜射 / 打到三角外 / 反向射 /
平行于面),全部符合预期。真实数据上把命中点反投影回图像,和原框角的
**像素误差 ~1e-13**。

### 相机位姿从哪读

COLMAP 二进制格式(本脚本已实现):

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

### OpenMVS 带贴图的 ply 格式

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
(接缝处就是这样),所以不能按顶点索引存。这也解释了为什么必须做几何一致性校验:
换了任何一份几何,第 i 个面的 texcoord 就对不上了。

**v 的方向是 OpenGL 约定:v = 0 在贴图的「底边」。** 而 open3d 采样时拿
`v * H` 当行号(从顶边数),所以**交给 open3d 之前必须翻成 `1 - v`**,
不翻的话整片贴图会采样到图集右下角的空白区。

空白区是 OpenMVS 用 `--empty-color`(默认 `0xFF7F27`)填的橙色 `(255,127,39)`。
不翻的典型症状就是:**模型整个变成橙色**。

怎么证实的:把 UV 三角形栅格化到图集上,和「网格实际占的那块区域」求交并比 ——

| 映射            | 与网格占区的 IoU |
| --------------- | ---------------- |
| `(u, v)` 不翻 | 0.017            |
| `(u, 1-v)`    | **0.942**  |
| `(1-u, v)`    | 0.008            |
| `(1-u, 1-v)`  | 0.327            |

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
(比如把框内的挖给碎片),**UV 必须用同一个掩码一起挖**,否则长度对不上,
open3d 会直接抛异常。

## 性能

| 操作                                          | 网格                   | 实测             |
| --------------------------------------------- | ---------------------- | ---------------- |
| `load_scene`                                | —                     | 0.00 s           |
| `load_mesh`                                 | 824422 顶点            | 0.94 s           |
| `project_points`(824422 顶点 × 13 图)      | 824422                 | 0.03 s / 图      |
| **`project_bbox`(含 160 条射线求交)** | 1648306 面             | **19.6 s** |
| 其中射线求交单独                              | 1648306 面 × 160 射线 | 19.5 s           |

**射线求交是唯一的瓶颈,而且它现在是对全部 1648306 个三角形暴力做的。**

老文档里写的 109 ms 是 14992 面的 refine 网格时代的数字。网格大了 110 倍,
这一步就慢了 180 倍。**点一次「投射到3D」要等 20 秒。**

**可优化且能证明正确:** 从相机出发、穿过 2D 框的射线,只可能命中
**投影与框有重叠**的三角形(因为射线上的点必然投影到框内)。所以可以先用
`project_points` 算出每个三角形三个顶点的投影包围盒,只对和框重叠的那些做求交。
按上面那次实测,1648306 → 约 20 万,能省掉 ~9 倍。

---

# 第三部分:坑

## 主流程

| #  | 现象                                                              | 原因                                                                                                                    | 解决                                                                                                                            |
| -- | ----------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| 1  | `unrecognised option '--SiftExtraction.xxx'`                    | COLMAP 4.2 把顶层参数改名了                                                                                             | 改用`--FeatureExtraction.xxx` / `--FeatureMatching.xxx`                                                                     |
| 2  | `use_gpu` 相关报错                                              | nocuda 版没编 GPU 代码,但默认值仍是 1                                                                                   | 必须显式传`--FeatureExtraction.use_gpu 0` 和 `--FeatureMatching.use_gpu 0`                                                  |
| 3  | `Check failed: ExistsDir(*image_path)`                          | `$IMG` 多写了一层 `\images`                                                                                         | 填图像**所在地那一层**                                                                                                    |
| 4  | `CAMERA_SINGLE_DIM_ERROR`                                       | `single_camera 1` 要求尺寸一致,混进了竖拍图                                                                           | 移走尺寸不同的图,或去掉`single_camera 1`                                                                                      |
| 5  | **`ReconstructMesh` 日志里只有 6015 个点,网格只有几千面** | `-p` 是**输入稠密点云**,不是输出网格;把网格名传给 `-p` 会**静默退回**稀疏点                             | `ReconstructMesh scene_dense.mvs -p scene_dense.ply -o scene_dense_mesh.ply`。自检看日志的 `Point-cloud '...' loaded:` 点数 |
| 6  | **贴图模型整片橙黑两色**                                    | `TextureMesh` 默认 `--global-seam-leveling 1 --local-seam-leveling 1` 在本机 v2.4.0 构建上会把面的 texel 涂成标记色 | 加`--global-seam-leveling 0 --local-seam-leveling 0`                                                                          |
| 7  | `--ignore-mask-label -2` 直接崩溃                               | 本机构建上 mask 那条代码路径有 bug(段错误)                                                                              | **别用这个参数**,默认 `-1` 就好;mask 也不是黑块的元凶                                                                   |
| 8  | `-o scene_textured.mvs` 得到的是 `.ply`                       | `-o` 是网格文件名,格式由 `--export-type` 定;TextureMesh 不写 `.mvs`                                               | 要 glb 就用`--export-type glb`                                                                                                |
| 9  | Viewer 打不开 glb:`invalid project`                             | Viewer 不能导入 glTF                                                                                                    | 用`.mvs` / `.ply` 看;glb 用 Blender 看                                                                                      |
| 10 | 交付的 glb 在别人机器上是白模                                     | glb 的贴图是**外链**的 png                                                                                        | glb 和`*_0.png` 一起给                                                                                                        |
| 11 | `no kernel image is available`                                  | CUDA 架构不匹配(sm_89 vs sm_120)                                                                                        | 用 CPU 版                                                                                                                       |
| 12 | mapper 之后`sparse\` 下有 `0` `1` 多个目录                  | 场景断成多个子模型                                                                                                      | 用注册图像最多的那个                                                                                                            |
| 13 | OpenMVS 命令跑完终端没输出                                        | 正常,日志写在同目录`.log`                                                                                             | 看`*-<时间戳>.log`                                                                                                            |
| 14 | 模型形状不对 / 缺一大块                                           | 图像重叠不足                                                                                                            | 提高抽帧率、降低`max_image_size`、或改用 `exhaustive_matcher`                                                               |
| 15 | 没有绝对尺度和真实坐标                                            | 没有 GPS EXIF,重建结果是任意尺度                                                                                        | 需要真实尺寸就在场景里放已知大小的标定物                                                                                        |
| 16 | 单航带直线飞行重建失败                                            | 相机中心共线,滚转角无法约束                                                                                             | 用往复式(割草机式)航线、环绕拍摄或高度变化                                                                                      |

## 投影工具

| #  | 现象                                                            | 原因                                                                                                                                                                                             | 解决                                                                                            |
| -- | --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------- |
| 17 | `json.loads` 读 `.mvs` 崩溃                                 | `.mvs` 是二进制,不是 JSON                                                                                                                                                                      | 别读`.mvs`,用 COLMAP 的 sparse 模型                                                           |
| 18 | `pip install pymvs-parser` 失败                               | 这个包不存在                                                                                                                                                                                     | 同上                                                                                            |
| 19 | `import pyOpenMVS` 报"找不到指定的模块"                       | Python 3.8+ 不再自动搜`.pyd` 所在目录                                                                                                                                                          | `os.add_dll_directory(Release目录)` + Python 3.12                                             |
| 20 | 框画在原图上,投影全偏                                           | 原图 3648×2736 ≠ 去畸变图 3781×2788,且形变非线性                                                                                                                                              | **框画在 `work\<场景>\dense\images\` 上**                                               |
| 21 | 3D 轮廓缺一段                                                   | 边界射线打在背景/空洞上,没交点                                                                                                                                                                   | 已改成沿框打 160 条;断的只是少数几段。看青色碎片(视锥裁剪),它完全不受影响                       |
| 22 | 像平面跑到模型里面 / 相机背后                                   | `--scale` 不合适                                                                                                                                                                               | 用`--scale` 手动指定,或调大调小试                                                             |
| 23 | 顶点大量在相机背后                                              | `--scene` 和 `--mesh` 不是同一次重建                                                                                                                                                         | 确认两者来自同一个`work\<场景>\`                                                              |
| 24 | 模型没有真实尺度                                                | 重建本身是任意尺度                                                                                                                                                                               | 投影数学不受影响,只是单位不是米                                                                 |
| 25 | `mesh.ray.intersects_location` 报 `No module named 'rtree'` | trimesh 的光线投射要 rtree                                                                                                                                                                       | 本脚本已自己实现求交,不依赖 trimesh 的 ray 模块                                                 |
| 26 | 点「投射到 3D」后主界面卡住不动                                 | 3D 窗口是阻塞的,是故意的                                                                                                                                                                         | 关掉 3D 窗口就回来了                                                                            |
| 27 | 界面里框的坐标和实际差几个像素                                  | 图被缩小显示了,鼠标精度 = 1/zoom 像素                                                                                                                                                            | 滚轮放大后再框,或框完看日志里的坐标                                                             |
| 28 | 3D 里模型是白的/灰的,没贴图                                     | open3d 的 ply 读取器不认 OpenMVS 写的`texcoord`                                                                                                                                                | 本脚本自己解析 ply(`read_ply_uv`)                                                             |
| 29 | 贴图整个花掉 / 错位                                             | UV 是按面存的,配到了几何不一致的 ply 上                                                                                                                                                          | 三重校验(顶点数/面数/顶点坐标),对不上会打印跳过原因                                             |
| 30 | **把带贴图的 ply 直接传给 `--mesh`,贴图不加载**         | `load_mesh` 走 trimesh,trimesh 会把「按面角存的 UV」转成「按顶点存」,**顶点数被重编**(824422 → 937775,面数不变);`load_texture` 的几何一致性检查是精确比顶点数,就判"几何不同",退回灰模 | 别传带贴图的 ply。传**纯几何**那份(`scene_dense_mesh.ply`),程序自己会去同目录找配套贴图 |
| 31 | 碎片一片都看不见                                                | 碎片和主网格画在同一个位置,深度打平被盖住                                                                                                                                                        | 主网格里把框中的三角形挖掉了,不会重合                                                           |
| 32 | 碎片和模型分不清                                                | 碎片用了橙色,而贴图本身偏暖                                                                                                                                                                      | 碎片改成青色(`COL_FRAGMENT`)                                                                  |

---

# 第四部分:产物与清理

## 只保留成品

```powershell
cd $W\mvs
Remove-Item depth*.dmap, scene_dense.ply, scene.mvs, scene_dense.mvs -ErrorAction SilentlyContinue
# 顺手删掉修复前那份稀疏网格的残留(现在谁都配不上)
Remove-Item scene_dense_mesh_refine.ply -ErrorAction SilentlyContinue
cd $W
Remove-Item db.db -Force -ErrorAction SilentlyContinue
Remove-Item sparse, dense -Recurse -Force -ErrorAction SilentlyContinue
```

**别删的:**

| 文件                                                        | 为什么留着                    |
| ----------------------------------------------------------- | ----------------------------- |
| `scene_dense_mesh.ply`                                    | 投影工具的`--mesh` 输入     |
| `scene_textured.ply` + `scene_textured0.png`            | 带贴图浏览;投影工具的贴图来源 |
| `scene_dense_texture.glb` + `scene_dense_texture_0.png` | 交付用                        |
| `dense\sparse\`                                           | 投影工具的位姿来源            |
| `dense\images\`                                           | 投影工具画框的那一层          |

**注意:** 删掉 `depth*.dmap` / `scene_dense.ply` / `scene_dense.mvs` 之后,
**重跑 `DensifyPointCloud` 就必须从头再来**。确认不再改重建参数再删。

## 13 张图的总占用

| 阶段                              | 占用      |
| --------------------------------- | --------- |
| 重建中间产物(depth + 点云 + 工程) | ≈ 730 MB |
| 成品(ply + png + glb)             | ≈ 172 MB |

## 视频输入

输入不是图像目录而是视频的话,在第 2 步之前加一步抽帧:

```powershell
ffmpeg -hide_banner -i "input.mp4" -vf "fps=2,scale='min(2000,iw)':-2" -q:v 2 "$W\images\frame_%05d.jpg"
```

然后 `$IMG` 指向抽帧输出目录。

---

# 第五部分:换一组数据

**结果自动保留,不用做任何事** —— 工作目录按场景名隔离:

```
work\
├── food\        ← 各自独立一份
└── <新场景名>\
```

**只需改第 1 步的两个变量**,然后从第 1 步往下跑:

```powershell
$W   = "D:\code\point_cloud\0916\work\mynewdata"          # 新场景名
$IMG = "D:\code\point_cloud\0916\data\mynewdata"          # 新图像目录
New-Item -ItemType Directory -Force -Path "$W\sparse","$W\dense","$W\mvs" | Out-Null
cd $W
```

**四条注意:**

1. **绝不要复用同一个场景名** —— `$W` 指向已有目录时,第 4 步 `mapper` 会因为
   `sparse\0` 已存在而报错或产出混乱。
2. **路径不要有中文和空格** —— COLMAP 对非 ASCII 路径支持不好。
3. **图像要够、重叠要足** —— 单航带直线飞行会失败(见坑 16)。
4. **磁盘留够** —— 总占用 ≈ 原图总量 × 2(去畸变副本)+ 图像数 × 50 MB(深度图)。

## 一键脚本

`run_pipeline.sh`(Git Bash)把 ①-⑩ 全串好了,包含本文档的全部修复:

```bash
./run_pipeline.sh data/drone.mp4            # 视频输入:自动抽帧
./run_pipeline.sh data/sceyeaux/images      # 图像目录输入
FORCE=1 ./run_pipeline.sh data/food  mydata # 强制重跑
```

可调环境变量:`FPS` / `MAX_WIDTH` / `MATCHER` / `RESOLUTION_LEVEL` /
`MAX_RESOLUTION` / `DO_REFINE` / `FORCE`。

---

# 第六部分:原理速览与已知限制

## 原理速览

```
图像
   │ ① COLMAP feature_extractor   每张图找 SIFT 特征
   │ ② COLMAP exhaustive_matcher  找图与图之间的同名点
   │ ③ COLMAP mapper              增量式 SfM → 相机位姿 + 稀疏点云
   │ ④ COLMAP image_undistorter   去镜头畸变
   ▼
去畸变图像 + 位姿
   │ ⑤ InterfaceCOLMAP           → OpenMVS .mvs 工程   ← 两者之间的桥
   ▼
scene.mvs
   │ ⑥ DensifyPointCloud         PatchMatch 逐视图算深度图 → 融合成稠密点云
   ▼
scene_dense.ply
   │ ⑦ ReconstructMesh           点云 → 三角网格(Delaunay + 图割)
   │ ⑧ RefineMesh                优化网格、恢复细节(可选)
   │ ⑨ TextureMesh               从原图给网格上色(选最佳视角 + 打包图集)
   ▼
scene_textured.ply + png
   │ ⑩ bbox_gui.py              2D 框 → 视锥 → 网格上的一块区域
   ▼
框中的 3D 碎片 + 可视化
```

**OpenMVS 不做 SfM。** 相机位姿和稀疏点云必须由上游 COLMAP 提供,
`InterfaceCOLMAP` 是两者之间的桥。

`TextureMesh` 内部三步:`Assigning the best view to each face`(给每个面挑一张
最佳视角图)→ `Generating texture atlas`(把用到的图像块打包进一张 8192×8192 图集)
→ 写 ply + png。图集里没用到的区域填 `--empty-color`(橙色)。

## 已知限制

1. **无 GPU 加速**,稠密重建慢。数据量大时优先调 `--resolution-level`(每 +1 快约 4 倍)。
2. **无真实尺度**(见坑 15)。
3. **进程内没有 `CreateStructure` / `ExtractKeyframes`** —— 这两个是 OpenMVS
   `develop` 分支的新功能,不在 v2.4.0 预编译包里,要自己编源码才能用。
   所以 SfM 必须外接 COLMAP,视频抽帧必须外接 ffmpeg。
4. **Viewer 不能导入 glTF** —— 看模型只能用 `.mvs` / `.ply`。
5. **`TextureMesh` 必须关 seam leveling**(见坑 6)—— 本机构建上关掉才出正确贴图,
   代价是少了 patch 间的颜色匀色。
6. **`project_bbox` 在百万面网格上要 20 秒**(见「性能」)—— 暴力射线求交,
   还没做空间加速。
