# OpenMVS 三维重建 — 操作手册

图像 / 视频 → 带纹理三维模型。全程 **CPU**,Windows + PowerShell。

**已验证跑通的案例**:Sceaux Castle 数据集,11 张图,2832×2128。

---

## 一、环境清单

| 组件     | 版本          | 位置                                      | 说明                               |
| -------- | ------------- | ----------------------------------------- | ---------------------------------- |
| OpenMVS  | v2.4.0        | `OpenMVS_Windows_x64\vc17\x64\Release\` | CPU 版,11 个 exe                   |
| COLMAP   | 4.2.0         | `tools\colmap\`                         | **nocuda** 版,`colmap.exe` |
| ffmpeg   | N-126574      | `tools\ffmpeg\bin\`                     | 仅视频输入时用于抽帧               |
| 示例数据 | Sceaux Castle | `data\sceaux\images\`                   | 11 张 JPG                          |

### 为什么都是 CPU 版

你的显卡是 **RTX 5070 Ti Laptop(Blackwell,sm_120)**,而 OpenMVS v2.4.0 的 CUDA 预编译包只编了 `sm_89` 且没嵌入 PTX。实测:

```
cuModuleLoad(sm_89 cubin) -> FAIL code=209
(no kernel image is available for execution on the device)
```

所以 GPU 用不了,只能 CPU。COLMAP 同理装了 nocuda 版保持一致。

---

## 二、两个必须知道的坑

### 坑 1:COLMAP 4.2 改了参数名

```
旧:--SiftExtraction.max_image_size
新:--FeatureExtraction.max_image_size
```

带 `Sift` 前缀的只剩 `--SiftExtraction.max_num_features` 这类底层 SIFT 参数,顶层的都改成了 `FeatureExtraction` / `FeatureMatching`。

### 坑 2:`use_gpu` 默认是 1

nocuda 版没编 GPU 代码,但 `--FeatureExtraction.use_gpu` 和 `--FeatureMatching.use_gpu` **默认值仍是 1**,不显式传 `0` 会直接报错。下面所有命令都已带 `0`。

---

## 三、逐步命令(复制粘贴即可)

### 第 0 步:设置 PATH

**每开一个新终端都要做一次。**

```powershell
$env:PATH = "D:\code\point_cloud\0916\OpenMVS_Windows_x64\vc17\x64\Release;D:\code\point_cloud\0916\tools\colmap;D:\code\point_cloud\0916\tools\ffmpeg\bin;" + $env:PATH
```

### 第 1 步:定义变量 + 建目录

```powershell
$W   = "D:\code\point_cloud\0916\work\sceaux"
$IMG = "D:\code\point_cloud\0916\data\sceaux\images"
New-Item -ItemType Directory -Force -Path "$W\sparse","$W\dense","$W\mvs" | Out-Null
cd $W
```

> **换数据集改这里的 `$W` 和 `$IMG` 两个变量即可**,详见第五节。

### 第 2 步:COLMAP 特征提取

```powershell
colmap feature_extractor --database_path "$W\db.db" --image_path "$IMG" --ImageReader.single_camera 1 --FeatureExtraction.use_gpu 0 --FeatureExtraction.max_image_size 2000
```

- `single_camera 1` = 所有图同一台相机(单相机采集必开,能显著提升稳定性)
- `max_image_size 2000` = 超过 2000px 的图先缩放,提速

### 第 3 步:特征匹配

```powershell
colmap exhaustive_matcher --database_path "$W\db.db" --FeatureMatching.use_gpu 0
```

> 图像数很多(>200 张)时改 `sequential_matcher --SequentialMatching.overlap 10` 会快很多。

### 第 4 步:稀疏重建(SfM)

```powershell
colmap mapper --database_path "$W\db.db" --image_path "$IMG" --output_path "$W\sparse"
```

**检查** —— 应该注册全部图像:

```powershell
colmap model_analyzer --path "$W\sparse\0"
```

### 第 5 步:图像去畸变

```powershell
colmap image_undistorter --image_path "$IMG" --input_path "$W\sparse\0" --output_path "$W\dense" --output_type COLMAP
```

### 第 6 步:转成 OpenMVS 工程

```powershell
InterfaceCOLMAP -i "$W\dense" -o "$W\mvs\scene.mvs" --image-folder "$W\dense\images" -w "$W\mvs"
```

### 第 7 步:稠密点云 ← 最慢的一步

```powershell
cd $W\mvs
DensifyPointCloud scene.mvs --resolution-level 1 --max-resolution 2000 --number-views 8
```

产出 `scene_dense.mvs` + `scene_dense.ply` + 每张图的 `depthXXXX.dmap`。

### 第 8 步:网格重建

```powershell
ReconstructMesh scene_dense.mvs -p scene_dense.ply -o scene_dense_mesh.ply
```

**⚠️ `-p` 是「输入稠密点云」,不是「输出网格」。** 写成 `-p scene_dense_mesh.ply`
不会报错,而是**静默退回** `scene_dense.mvs` 里那 6015 个稀疏点,只得到几千面的粗模:

```
[Delaunay] Delaunay tetrahedralization completed: 6015 points     ← 少了三个数量级
```

**自检:** 日志里 `Point-cloud '...' loaded:` 那行的点数应该是**百万级**。
正确结果实测 **29.87 MB / 824422 顶点 / 1648306 面**,耗时约 24.5 秒。

### 第 9 步:网格细化(可选)

```powershell
RefineMesh scene_dense.mvs -m scene_dense_mesh.ply -o scene_dense_mesh_refine.ply --scales 1 --max-face-area 16
```

**注意:** 第 8 步得到的是百万面网格时通常**不需要** refine(又慢又占内存)。
只有当第 8 步踩了上面那个 `-p` 的坑、只剩几千面时才必须补做。

### 第 10 步:纹理贴图(最终成品的两种出法)

TextureMesh 的参数设计容易误解,先看清楚:

```
-o [ --output-file ] arg    output filename for storing the **mesh**
--export-type arg (=ply)    file type used to export the 3D scene   ← 格式由它决定
```

**关键:`-o` 只是网格的文件名,格式由 `--export-type` 决定(默认 `ply`)。TextureMesh 不写 `.mvs` 工程文件。** 所以 `-o scene_textured.mvs` 得到的是 `scene_textured.ply`,不是 `.mvs`。

**⚠️ `--global-seam-leveling 0 --local-seam-leveling 0` 两个参数不能省。** 默认全开时,
本机这个 v2.4.0 构建会把**面自己的 texel** 整片涂成黑/红/绿标记色,只留下向外的膨胀光晕,
而 UV 恰好只覆盖面自己的 texel —— 打开模型只见橙黑两色。实测同一份数据:

| 选项 | 图集纯黑 | 网格 UV 落在真照片上 |
|---|---|---|
| 默认 | 24.3% | 2.9% |
| 两个都关 | **0.0%** | **100.0%** |

**出法 A —— 只要带纹理网格(ply + 贴图 png):**

```powershell
TextureMesh scene_dense.mvs -m scene_dense_mesh.ply -o scene_textured.ply --global-seam-leveling 0 --local-seam-leveling 0
```

产出 `scene_textured.ply` + `scene_textured0.png`(实测 69.17 MB + 62.95 MB)。

**出法 B —— 直接出 glb(推荐交付格式):**

```powershell
TextureMesh scene_dense.mvs -m scene_dense_mesh.ply --export-type glb --global-seam-leveling 0 --local-seam-leveling 0
```

输出名自动取输入场景名,生成 **`scene_dense_texture.glb`** + `scene_dense_texture_0.png`
(实测 36.74 MB + 63.06 MB)。**glb 的贴图是外链的,两个文件要一起给。**

换 `--export-type obj` 可出 OBJ(含 mtl + 贴图)。

> 想改输出文件名就加 `-o 你想要的名字.glb`。
>
> `--ignore-mask-label -2`(想关掉 mask)**在本机构建上会段错误**,别用。它也不是黑块的元凶。

### 查看结果

```powershell
Viewer.exe scene_dense_texture.glb
```

或者看中间过程(相机位姿 + 稠密点云):

```powershell
Viewer.exe scene_dense.mvs
```

`Viewer` 还支持拖拽文件进去。左侧 `Workflow` 菜单可以直接在界面里跑 Densify / Reconstruct / Refine / Texture。

---

## 四、产物说明

### 最终成果

| 文件                        | 内容                       | 用途                                           |
| --------------------------- | -------------------------- | ---------------------------------------------- |
| `scene_dense_texture.glb` | **单文件带纹理模型** | **推荐交付格式**,网页/Unity/Blender 通用 |
| `scene_textured.ply`      | 带纹理网格                 | 大部分三维软件可读                             |
| `scene_textured0.png`     | 纹理贴图                   | 配合 ply/obj                                   |

### 中间产物(占空间,可清理)

| 文件                            | 大小(11 图案例) | 说明                       |
| ------------------------------- | --------------- | -------------------------- |
| `db.db`                       | 23 MB           | COLMAP 数据库(特征 + 匹配) |
| `sparse/0/`                   | 小              | 稀疏点云 + 相机位姿        |
| `dense/`                      | 大              | 去畸变图像(原图副本)       |
| `depthXXXX.dmap`              | 30 MB × N      | 深度图,**最占空间**  |
| `scene_dense.ply`             | 38 MB           | 稠密点云                   |
| `scene_dense_mesh.ply`        | 0.2 MB          | 粗糙网格                   |
| `scene_dense_mesh_refine.ply` | 0.6 MB          | 细化网格                   |

**只保留最终模型、清理中间产物:**

```powershell
cd $W\mvs
Remove-Item depth*.dmap, scene_dense.ply, scene.mvs -ErrorAction SilentlyContinue
cd $W
Remove-Item db.db -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item sparse, dense -Recurse -Force -ErrorAction SilentlyContinue
```

---

## 五、换一组数据(重点)

**结论:结果自动保留,不用做任何事。** 因为工作目录是按**场景名**隔离的:

```
work/
├── sceaux/          ← 第一组数据(保留着)
│   ├── db.db
│   ├── sparse/dense/
│   └── mvs/         ← 模型在这
└── <新场景名>/       ← 新数据自动独立一份
```

### 操作步骤

**只需改第 1 步的两个变量**,然后从第 1 步重新往下跑:

```powershell
$W   = "D:\code\point_cloud\0916\work\mynewdata"      # 改这里:新场景名
$IMG = "D:\code\point_cloud\0916\data\mynewdata\images"  # 改这里:新图像目录
New-Item -ItemType Directory -Force -Path "$W\sparse","$W\dense","$W\mvs" | Out-Null
cd $W
```

`work\sceaux` 原封不动保留。想回看旧结果:

```powershell
Viewer.exe "D:\code\point_cloud\0916\work\sceaux\mvs\scene_dense_texture.glb"
```

### 三条注意事项

1. **绝对不要复用同一个场景名**。如果 `$W` 指向已有目录,第 4 步 `mapper` 会因为 `sparse/0` 已存在而报错或产生混乱。要么换名,要么先删掉旧目录。
2. **图像目录名要被 COLMAP 接受**。路径、文件名**不要有中文和空格**,COLMAP 对非 ASCII 路径支持不好。
3. **磁盘要留够**。粗略估算:总占用 ≈ 原图总量 × 2(去畸变副本) + 图像数 × 30 MB(深度图)。

### 一键脚本(可选)

如果不想每次粘贴 11 条命令,`run_pipeline.sh` 把整个流程串好了,支持视频和图像目录两种输入,已完成步骤会自动跳过:

```bash
cd /d/code/point_cloud/0916
./run_pipeline.sh data/sceaux/images sceaux          # 图像目录
./run_pipeline.sh data/drone.mp4 myscene             # 视频(自动抽帧)
```

可调参数用环境变量覆盖:

```bash
FPS=2 MAX_WIDTH=2000 RESOLUTION_LEVEL=1 DO_REFINE=1 ./run_pipeline.sh data/drone.mp4
```

---

## 六、常见问题

| 现象                                             | 原因 / 解决                                                                            |
| ------------------------------------------------ | -------------------------------------------------------------------------------------- |
| `unrecognised option '--SiftExtraction.xxx'`   | COLMAP 4.2 改名了,用`--FeatureExtraction.xxx`                                        |
| `use_gpu` 相关报错                             | nocuda 版必须传`--FeatureExtraction.use_gpu 0` 和 `--FeatureMatching.use_gpu 0`    |
| OpenMVS 命令跑完终端没输出                       | 正常,它把日志写进同目录`.log` 文件。看退出码和产物                                   |
| `no kernel image is available`                 | CUDA 架构不匹配,用 CPU 版                                                              |
| mapper 之后`sparse/` 下有 `0` `1` 多个目录 | 场景断成了多个子模型。用注册图像最多的那个(改`$W\sparse\0` 里的 `0`)               |
| 模型形状不对 / 缺一大块                          | 图像重叠不足。提高抽帧率、降低`max_image_size`、或改用 `exhaustive_matcher`        |
| **没有绝对尺度和真实坐标**                 | 视频抽帧的图没有 GPS EXIF,重建结果是任意尺度。需要真实尺寸就在场景里放已知大小的标定物 |

---

## 七、流程原理速览

```
图像/视频
   │
   │ ① ffmpeg 抽帧(视频才需要)
   ▼
图像序列
   │ ② COLMAP feature_extractor   每张图找 SIFT 特征
   │ ③ COLMAP exhaustive_matcher  找图与图之间的同名点
   │ ④ COLMAP mapper              增量式 SfM → 相机位姿 + 稀疏点云
   │ ⑤ COLMAP image_undistorter   去镜头畸变
   ▼
去畸变图像 + 位姿
   │ ⑥ InterfaceCOLMAP           → OpenMVS .mvs 工程
   ▼
scene.mvs
   │ ⑦ DensifyPointCloud         PatchMatch 逐视图算深度图 → 融合成稠密点云
   ▼
scene_dense.ply
   │ ⑧ ReconstructMesh           点云 → 三角网格
   │ ⑨ RefineMesh                优化网格、恢复细节
   │ ⑩ TextureMesh               从原图给网格上色
   ▼
带纹理模型 → ⑪ TransformScene 导出 glb/obj
```

**OpenMVS 在链条里的位置**:它**不做 SfM**,只负责稠密重建之后的部分。相机位姿和稀疏点云必须由上游(这里用 COLMAP)提供——`InterfaceCOLMAP` 就是这两者之间的桥梁。

---

## 八、已知限制

1. **无 GPU 加速**,稠密重建慢。数据量大时优先调 `--resolution-level`(每 +1 快约 4 倍)。
2. **无真实尺度**(见第六节倒数第二行)。
3. **没有 `CreateStructure` / `ExtractKeyframes`**。这两个是 OpenMVS `develop` 分支的新功能,不在 v2.4.0 预编译包里,需要自己编译源码才能用。所以 SfM 必须外接 COLMAP,视频抽帧必须外接 ffmpeg。
4. **单航带直线飞行会失败**。无人机只沿一条直线飞一趟时,相机中心共线,无法约束绕航迹的滚转角。需要往复式(割草机式)航线、环绕拍摄或高度变化。
