# 倾斜摄影三维重建 + 点云语义分割

从一组照片出发：

```
照片 → COLMAP 稀疏重建 → OpenMVS 稠密点云/网格 → SAM 3 文本概念 mask
     → 多视角投票 → 点云类别标签 → 3D 可视化
```

实测环境：Windows 11 + PowerShell；COLMAP / OpenMVS 走 CPU；SAM 3 单独一个 conda 环境，走 CUDA。

> `food` 是食品学院的楼，不是食物。这个场景用来演示窗户、植物、空调外机等建筑构件。

---

## 全流程：先看 [docs/全流程.md](docs/全流程.md)

从一组照片一路做到「生成的 3D 模型配准回重建点云」，完整命令、参数和踩坑都在这份文档里：

```
照片
 │ COLMAP 特征 → 匹配 → SfM → 去畸变
 ▼
work\<场景>\dense\images       去畸变图
 │ OpenMVS 稠密点云 / 网格
 ▼
work\<场景>\mvs\scene_dense.ply
 │ SAM 3 多提示词（一个概念 = 一个类别 id）
 ▼
masks_<场景>\*.png             索引 mask
 ├──────────────┐
 ▼              ▼
点云类别标签      每个物体白底抠图
 │              │ 按相机位姿自动选 front/left/right/back
 │              ▼
 │         混元多视角图生 3D（付费 API）
 │              │
 └──────────────┴──► 多物体配准 → Open3D 叠加显示
                      + 黄色四棱锥标出相机位置
```

文档里包含：

- **一次定义场景变量**（`$root` / `$objs` / `$scene` / `$W` / `$cutouts`），后面所有命令直接复用；
- **多物体**：`$objs = @("black cup", "white cup")`，SAM 3 分割、抠图、选视角、提交 API、配准全部按类别 id 一一对应；
- **选视角**：同时考虑方位角和目标在画面里的大小/完整性，不会为了几度的角度差选一张物体很小或被裁掉的照片；
- **配准显示**：支持一次配准多个模型，窗口里用黄色四棱锥画出每张照片的相机位姿；
- 每步的产物、可调参数和常见报错。

只做「点云语义分割」、不生成模型时，按下面「快速开始」跑即可；
需要图生 3D、配准回点云或在窗口里看相机位置，直接按全流程文档执行。

> 不想手工逐条跑重建，可以先用 `bash ./run_pipeline.sh data/<场景> <场景>`
> 完成 COLMAP + OpenMVS（产物在 `work\<场景>\`），再从全流程文档第 4 节
> 接 SAM 3。脚本的注册阈值比 COLMAP 官方默认宽松；想恢复用
> `COLMAP_RELAXED=0`。

---

## 仓库结构

```
data\<场景>\                 原始照片
work\<场景>\                 COLMAP / OpenMVS 工作目录（中间产物，不入库）
  dense\images\              去畸变图 ← SAM 3 的输入
  dense\sparse\              相机内参 + 位姿 ← mask_to_points / seg_vis 的输入
  mvs\scene_dense.ply        稠密点云
  mvs\scene_dense_mesh.ply   网格（遮挡剔除用）
masks\<场景>\<概念组合>\      SAM 3 生成的索引 mask（可重建，不入库）
out\<场景>\                  分割结果、叠加图（可重建，不入库）
tools\                       重建、投影、分割工具
docs\                        详细手册
```

**仓库只发布代码和文档。** `data/rock/`、`work/rock/`、`masks/`、`out/` 都不上传；它们体积大，且能重新生成。

---

## 环境

| 用途 | 环境 / 路径 | 关键依赖 |
|---|---|---|
| 重建 | `tools\colmap\colmap.exe`、`OpenMVS_Windows_x64\` | COLMAP 4.2、OpenMVS v2.4.0（CPU） |
| 投影与点云 | `basic_env`（conda） | numpy、open3d、trimesh、Pillow |
| SAM 3 | `sam3`（conda，独立另建） | torch cu128、ultralytics、GPU |

```powershell
$root = "D:\code\point_cloud\0916"
$py   = "C:\Users\FANLS\.conda\envs\basic_env\python.exe"
$sam  = "D:\ProgramData\anaconda3\envs\sam3\python.exe"
```

两个 Python 环境靠 PNG 文件交接：`sam3` 出 mask，`basic_env` 读 mask 投点云。没有任何单个进程能跨这两个环境。

OpenMVS 和 COLMAP 的安装、SAM 3 环境和权重下载见 [docs/OpenMVS操作手册.md](docs/OpenMVS操作手册.md)、[docs/SAM3生成mask.md](docs/SAM3生成mask.md)。

---

## 快速开始

下面以 `food` 场景为例。换场景只改 `$scene`。

### 0. 变量和目录

```powershell
$scene = "food"
$W     = "$root\work\$scene"
$IMG   = "$root\data\$scene"
$mvs   = "$root\OpenMVS_Windows_x64\vc17\x64\Release"

New-Item -ItemType Directory -Force "$W\sparse","$W\dense","$W\mvs" | Out-Null
```

### 1. 三维重建

#### COLMAP：特征 → 匹配 → SfM → 去畸变

```powershell
$colmap = "$root\tools\colmap\colmap.exe"

& $colmap feature_extractor `
  --database_path "$W\db.db" --image_path $IMG `
  --ImageReader.single_camera 1 `
  --FeatureExtraction.use_gpu 0 `
  --FeatureExtraction.max_image_size 2000

& $colmap exhaustive_matcher `
  --database_path "$W\db.db" --FeatureMatching.use_gpu 0

& $colmap mapper `
  --database_path "$W\db.db" --image_path $IMG --output_path "$W\sparse"

& $colmap image_undistorter `
  --image_path $IMG --input_path "$W\sparse\0" `
  --output_path "$W\dense" --output_type COLMAP
```

> `image_undistorter` 会改变图像尺寸（food：3648×2736 → 3781×2788）。后面的 mask、投影、SAM 3 都必须用 `work\<场景>\dense\images\`，不能用原图目录。

#### OpenMVS：稠密点云 → 网格 → 贴图

```powershell
& "$mvs\InterfaceCOLMAP.exe" `
  -i "$W\dense" -o "$W\mvs\scene.mvs" `
  --image-folder "$W\dense\images" -w "$W\mvs"

Push-Location "$W\mvs"

& "$mvs\DensifyPointCloud.exe" scene.mvs `
  --resolution-level 1 --max-resolution 2000 --number-views 8 --max-threads 0

& "$mvs\ReconstructMesh.exe" scene_dense.mvs `
  -p scene_dense.ply -o scene_dense_mesh.ply

& "$mvs\TextureMesh.exe" scene_dense.mvs `
  -m scene_dense_mesh.ply -o scene_textured.ply `
  --global-seam-leveling 0 --local-seam-leveling 0

Pop-Location
```

注意 `ReconstructMesh` 的 `-p` 是**输入稠密点云**，`-o` 才是输出网格。

### 2. 生成 mask

SAM 3 一次可以给多个概念：**`--text` 的顺序就是类别 id 的顺序**，后续所有工具都要保持一致。

```powershell
$mask = "$root\masks\$scene\window+plant+airconditioner"

# 自检：不加载模型，验证索引图不变量
& $sam "$root\tools\sam3_masks.py" --selftest

# 单张先看叠加图（可选，确认概念选得对不对）
& $sam "$root\tools\sam3_masks.py" `
  --images "$W\dense\images" --out $mask `
  --text window --text plant --text airconditioner `
  --expect-size 3781 2788 --limit 1 `
  --overlay-dir "$root\out\sam_overlay"

# 批量：默认跳过已生成的 mask，断了可以直接重跑
& $sam "$root\tools\sam3_masks.py" `
  --images "$W\dense\images" --out $mask `
  --text window --text plant --text airconditioner `
  --expect-size 3781 2788
```

- 默认**只写 mask，不写叠加图**。需要肉眼验收时显式加 `--overlay-dir`。
- `--expect-size` 会硬校验去畸变图尺寸，防止误把原图目录传进来。
- 输出是单通道 uint8 索引图：`0` = 背景，`1/2/3` = 第 1/2/3 个概念。

### 3. mask 回灌点云

```powershell
& $py "$root\tools\mask_to_points.py" `
  --scene "$W\dense\sparse" `
  --mask-dir $mask --mask-mode index `
  --class-names "window,plant,airconditioner" `
  --min-conf 0.9 `
  --out "$root\out\$scene\segmented.ply" `
  --labels "$root\out\$scene\labels.npy" `
  --save-json "$root\out\$scene\stats.json"
```

`--class-names` 的顺序必须和 `--text` 一致。`--min-conf 0.9` 压掉跨视角并集噪声；不要用 `--min-votes` 代替，它会误杀只有少数视角看得见的正确点。

### 4. 一条命令看 3D

```powershell
& $py "$root\tools\seg_vis.py" `
  --scene food `
  --text window --text plant --text airconditioner
```

它会自动：

1. 检查 `masks\food\window+plant+airconditioner\`，缺 mask 时调用 SAM 3 补；
2. 用 `work\food\dense\sparse`、`scene_dense.ply`、`scene_dense_mesh.ply` 做多视角投票和遮挡剔除；
3. 弹出带类别颜色的 Open3D 点云窗口。

默认**不写叠加图**。需要保存 2D 叠加图时加：

```powershell
& $py "$root\tools\seg_vis.py" `
  --scene food `
  --text window --text plant --text airconditioner `
  --overlay
```

叠加图会写到 `out\food\overlay\`；也可用 `--overlay-dir` 指定目录。

---

## 点云分割约定

- **一张图一个概念 = 一个类别 id**。同类多个实例取并集，不区分单个窗户。
- **概念顺序必须跨工具一致**：`sam3_masks.py --text`、`mask_to_points.py --class-names`、`seg_vis.py --text` 三处顺序要相同。
- **mask 必须来自去畸变图**：原图和去畸变图尺寸、畸变都不同，混用会整体错位。
- **背景默认不投票**；一个点只有在多数可见视角中落在同一类别，才会被采纳。
- 多概念重叠时由 `--priority first` 决定：先列出的概念优先，建议按「具体 → 宽泛」排列。
- 标签不写进 PLY 顶点属性，而是单独存 `labels.npy`；它和输入点云**同序**，不要对点云做裁剪、排序或去重。

### 常用参数

| 参数 | 作用 | 建议 |
|---|---|---|
| `--min-conf 0.9` | 一致比例 `n_vote / n_visible` 下限 | 压 mask 边界噪声，通常 0.9 |
| `--min-votes N` | 至少 N 个视角投同一类 | 少用，容易误杀劣视角真标签 |
| `--min-visible N` | 至少被 N 个视角看到 | 默认 1 |
| `--depth-tol` | 遮挡容差 | 默认自适应；调试时加 `--depth-tol-hist` 看分布 |
| `--conf` | SAM 3 置信度阈值 | 默认 0.25，漏检多就调低 |
| `--imgsz` | SAM 3 推理分辨率 | 默认 1008，显存不够调小 |
| `--priority first/last` | 多概念重叠归谁 | 默认 `first` |

---

## 产物与清理

| 路径 | 内容 | 能否删除 |
|---|---|---|
| `work\<场景>\mvs\scene_dense.ply` | 稠密点云，分割的几何输入 | 重跑 `DensifyPointCloud` 可重建 |
| `work\<场景>\mvs\scene_dense_mesh.ply` | 网格，遮挡剔除用 | 重跑 `ReconstructMesh` 可重建 |
| `work\<场景>\dense\images\` | 去畸变图，SAM 3 的输入 | 重跑 `image_undistorter` 可重建 |
| `masks\<场景>\<概念组合>\` | 索引 mask | 重跑 `sam3_masks.py` 可重建 |
| `out\<场景>\` | 分割点云、标签、叠加图、统计 | 重跑第 3/4 步可重建 |

`masks/` 和 `out/` 默认不入库，也不随仓库发布。

---

## 已知限制

- **SAM 3 的文本编码器是英文 CLIP**：中文提示基本无效，请用英文概念。
- **细粒度结构词不可靠**：`crevice`、`crack`、`joint` 这类岩缝/节理提示词实测几乎检不出；`rock` 会返回整面岩壁。需要岩缝时得另做 2D 黑帽/脊线或点云几何凹陷检测。
- **OpenMVS 的 CUDA 预编译包只编了 sm_89**，RTX 5070 Ti（sm_120）加载会失败，所以重建走 CPU。
- **单航带直线飞行会失败**：相机中心共线，无法约束滚转角。需要往复式航线、环绕拍摄或高度变化。

---

## 相关文档

| 文件 | 内容 |
|---|---|
| **[docs/全流程.md](docs/全流程.md)** | **推荐入口**：照片 → 重建 → 多物体分割 → 图生 3D → 配准 → 相机显示的完整命令 |
| [docs/food完整流程.md](docs/food完整流程.md) | 从照片到带纹理模型的逐条实测流程与参数 |
| [docs/SAM3生成mask.md](docs/SAM3生成mask.md) | SAM 3 环境、权重、mask 生成与排错 |
| [docs/mask点云分割.md](docs/mask点云分割.md) | 多视角投票、遮挡剔除、参数与实测分布 |
| [docs/换标注概念.md](docs/换标注概念.md) | 换概念时的检查清单 |
| [docs/2D框投影到3D模型.md](docs/2D框投影到3D模型.md) | 2D 框 → 3D 网格碎片的另一条链路 |
| [docs/OpenMVS操作手册.md](docs/OpenMVS操作手册.md) | OpenMVS 通用操作与安装 |
| [docs/流程与产物.md](docs/流程与产物.md) | 全仓库流程与产物对照表 |
