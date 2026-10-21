# SAM 3 生成 mask

**场景**:`work\food`(13 张去畸变图,3781×2788)
**工具**:`tools\sam3_masks.py`
**环境**:独立的 conda `sam3`(GPU),不是跑 SfM 的 `basic_env`
**实测日期**:2026-10-14

---

## 一、这一步是干什么的

`mask点云分割.md` 那条链路的上游。那边已经能从一堆 mask 算出点云的逐点标签,
但**mask 从哪来**一直空着 —— 就是这个工具。

```
work\food\dense\images\IMG_2430.JPG … IMG_2443.JPG   (3781x2788,去畸变图)
        │
        │  SAM 3:一张图 + 一串文字概念 → 每个概念的实例掩膜
        ▼
   tools\sam3_masks.py
        │
        ▼
masks\IMG_2430.png …    ← uint8 单通道,0=背景,1/2/3=第 1/2/3 个概念
        │
        │  tools\mask_to_points.py --mask-dir masks --mask-mode index
        ▼
点云每个点一个类别标签
```

⚠️ **场景名叫 `food`,但它是食品学院的楼,不是食物。** 首批概念是 `window`。
按这个名字去猜「盘子、食物」会一路猜错。

---

## 二、原理

### 概念 id = 类别 id,实例取并集

SAM 3 是**概念分割**(PCS):给它 `"window"`,它把图里**所有**窗户都找出来,
每个窗户一个实例。但下游要的是「类别」不是「实例」,所以:

```
类别 1 = "window" = 这一张图里所有窗户掩膜的并集
类别 2 = "wall"   = ……以此类推
```

**类别 id 跨图必须一致** —— 下游 `mask_to_points` 的 `--class-names` 是按
**类别 id** 取名字的(`names[val - 1]`),id 1 永远是第 1 个 `--text`。

> 别去区分单个窗户(哪个窗是哪扇)。多视角投票是**按类**投的,实例 id 一跨图就漂,
> 只会把投票打碎。工具里没有这个选项,是故意的。

### 重叠像素归谁:`--priority`

概念之间会重叠。给 `--text window wall`,墙的掩膜显然把窗户整个包在里面。
一个像素只能有一个类别 id,所以必须有个确定的裁决:

| `--priority` | 含义 | 绘制顺序 |
|---|---|---|
| **`first`(默认)** | **先列出的概念优先** | 第 1 个先占,后面的只填没被占的像素 |
| `last` | 后列出的概念优先 | 反过来 |

默认选 `first`,因为用户天然按「具体 → 宽泛」写:`--text window wall` 读起来就是
「窗,以及墙」。反过来(`last`)会让 `wall` 一口吃掉所有窗户。

**工具会把重叠像素数打出来**(`重叠 Npx`)。光看索引图看不出有没有重叠、规则有没有
生效,这个数字是唯一的客观指标。

> ⚠️ 一个直接后果:如果两个概念**完全嵌套**(窗户整个在墙里面),`first` 下被盖住的
> 那个概念会**一个像素都不剩**。这时报告里那行会显示 `window=0px(37个)←被盖住`
> —— 括号里实例数是 37 说明**检测到了**,只是全被盖住了。看到这个标记就调
> `--priority` 或者换 `--text` 顺序。

---

## 三、环境(独立的 conda env)

**绝不装进 `basic_env`。** `basic_env` 跑 COLMAP/open3d 那条线,依赖是 numpy/open3d/
trimesh;这边要 torch+ultralytics+GPU,两边的 torch 版本需求完全不同,混在一起迟早出事。
两个环境**只靠 PNG 文件交接**,`sam3_masks.py` 一个 `import` 都不碰 `bbox_to_mesh`。

```powershell
conda create -n sam3 python=3.12 -y
conda activate sam3
python -m pip install --upgrade pip

# cu128 是**唯一**能给 5070 Ti(sm_120/Blackwell)出 kernel 的构建
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install "ultralytics>=8.3.237"
# timm 是 SAM 3 的隐式依赖。不预装的话第一次推理时 ultralytics 会自己
# AutoUpdate(约 46 秒,还附一句「重启运行时才生效」的吓人提示),预装省事。
pip install timm

# 文字提示的 tokenizer 坑,先修掉(见坑 1)
pip uninstall clip -y
pip install git+https://github.com/ultralytics/CLIP.git
```

> **⚠️ 本机 `github.com` 直连不通(HTTP 000),上面最后一条会一直卡住。**
> 走 ghproxy 镜像,依赖包走清华源:
>
> ```powershell
> pip install -i https://pypi.tuna.tsinghua.edu.cn/simple `
>   "git+https://ghproxy.net/https://github.com/ultralytics/CLIP.git"
> ```
>
> 实测装出来的就是 ultralytics 的 fork:`clip-1.0` + `ftfy` `regex` `colorama`
> `tqdm` `wcwidth`。装完必须验一下 **`SimpleTokenizer` 是可调用的类**(坑 1 的根因):
>
> ```powershell
> python -c "import clip; T=clip.simple_tokenizer.SimpleTokenizer(); print(type(T), 'ok')"
> ```
>
> 顺带,文字提示要的 BPE 词表就在这个包里,第四节不用另找地方下。

验证 —— **`arch list` 里必须有 `sm_120`**,没有就是装错版本了:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_arch_list())"
# 期望: 2.x.x+cu128 True ... 且列表里有 'sm_120'
```

**环境位置**:本机 conda 自己有 envs 目录,建出来的是
`D:\ProgramData\anaconda3\envs\sam3`,**不是** `C:\Users\FANLS\.conda\envs\sam3`。
下面各节用的解释器路径以 `conda env list` 实际打印的为准:

```powershell
conda env list          # 抄 sam3 那一行的路径,下面所有 $sam 都用它
```

> 顺带一提:**GPU 在这台机器上是能用的**。之前「GPU 用不了」只针对 OpenMVS 的官方
> 预编译包(只编了 `sm_89`、没有 PTX 回退,见 `issue_blackwell.md`),跟 torch 无关。

---

## 四、权重怎么下

`sam3.pt` 约 **3.45 GB**,放**仓库外**(`D:\models\sam3`),别进 git。

```powershell
pip install modelscope
# 只下 sam3.pt —— 整个仓库 6.42 GB,其中一半是下面的 safetensors,这条路用不上
modelscope download --model facebook/sam3 sam3.pt --local_dir D:\models\sam3
```

仓库里两个大文件,别下错:

| 文件 | 大小 | 要吗 |
|---|---|---|
| `sam3.pt` | 3,450,062,241 字节(3.45 GB) | **要**,Ultralytics 这条路读它 |
| `model.safetensors` | 3,439,938,512 字节(3.44 GB) | 不要,那是 transformers 格式 |

`modelscope download` 不带文件名会把整个仓库拉下来(6.42 GB);**带上 `sam3.pt` 就只下
这一个**,省 3.4 GB。下完 `ls D:\models\sam3` 确认大小对得上。

不想装 modelscope 也可以直接 curl(实测约 9.5 MB/s,六分钟):

```powershell
curl -L -C - -o D:\models\sam3\sam3.pt `
  "https://modelscope.cn/models/facebook/sam3/resolve/master/sam3.pt"
```

`-C -` 是断点续传,断了重跑同一条命令接着下。

> **为什么不从 HuggingFace 下**:`facebook/sam3` 是 **gated** 仓库,要申请并等审批,
> 而且大陆申请经常被拒。ModelScope 有同一个模型的镜像,直连、不需要审批。

文字提示还需要 **BPE 词表** `bpe_simple_vocab_16e6.txt.gz`(约 1.4 MB),从上面装好的
CLIP 包里直接拿:

```powershell
python -c "import clip,os,shutil; shutil.copy(os.path.join(os.path.dirname(clip.__file__),'bpe_simple_vocab_16e6.txt.gz'), r'D:\models\sam3')"
```

`--model` / `--bpe` 不写的话,工具按这个顺序找:

1. `D:\models\sam3\<文件名>`
2. `<仓库>\weights\<文件名>`
3. `~\.cache\sam3\<文件名>`
4. `~\.cache\huggingface\hub\models--facebook--sam3\snapshots\*\`

**`sam3.pt` 不会自动下载**,找不到会直接把上面四个路径列出来报错。

---

## 五、逐步命令

```powershell
$sam  = "D:\ProgramData\anaconda3\envs\sam3\python.exe"
$root = "D:\code\point_cloud\0916"
$imgs = "$root\work\food\dense\images"
```

### 第 1 步:自检(先跑这个,不碰模型也不碰 GPU)

```powershell
& $sam "$root\tools\sam3_masks.py" --selftest
```

纯合成数据,钉死两条**最容易静默出错**的不变量:重叠裁决(谁能盖住谁)和掩膜放大
不产生新类别 id(混进一个 id=1.5 这种值,下游不会报错,但整个标签就错了)。
打印 ✅/❌,退出码 0/1。

### 第 2 步:单张 + 叠加图(★ 一定要看叠加图)

```powershell
& $sam "$root\tools\sam3_masks.py" `
  --images $imgs --out "$root\masks" `
  --text "window" --expect-size 3781 2788 `
  --limit 1 --overlay-dir "$root\out\sam_overlay"
```

**先看 `out\sam_overlay\IMG_2430.png`**。mask 错了在点云上极难归因(是概念顺序反了?
尺寸错了?掩膜糊了?),而叠加图一眼就能看穿。这一步不能省。

`--expect-size 3781 2788` 会在第一张图上硬校验。把 `--images` 指到原图目录
`data\food`(3648×2736)是最典型的误用,不加这个会一路跑到下游才炸。

### 第 3 步:批量

```powershell
& $sam "$root\tools\sam3_masks.py" `
  --images $imgs --out "$root\masks" `
  --text "window" --expect-size 3781 2788 `
  --overlay-dir "$root\out\sam_overlay" --save-json "$root\out\sam_report.json"
```

默认**跳过已经生成过的 mask**,所以跑一半断了直接重跑,会从断点接着走。
想重来加 `--overwrite`。

### 第 4 步:回灌点云

```powershell
$py = "C:\Users\FANLS\.conda\envs\basic_env\python.exe"
& $py "$root\tools\mask_to_points.py" `
  --scene "$root\work\food\dense\sparse" --mask-dir "$root\masks" `
  --mask-mode index --class-names "window" --depth-tol-hist --min-conf 0.9 `
  --out "$root\out\window.ply" --labels "$root\out\window.npy"
```

`--class-names` 的顺序**必须和 `--text` 一致**(都是按 id 1,2,3…取名字)。
细节见 `mask点云分割.md`。

> **`--min-conf 0.9` 别省。** 不加的话融合结果是**跨视角并集** —— 一个点只要在
> 任意一个视角被标上就进来,而 SAM 的 mask 边界在不同图之间抖动几十像素,擦过边界的
> 点会大量混进来。实测 food 13 视角:不加 19.95 万点(17.9%,点云上标签「胖一圈」),
> 加 `0.9` 剩 9.18 万点(8.3%),和 2D mask 逐视角覆盖的 9.2~13.5% 正好自洽。
>
> 注意**别用 `--min-votes` 代替** —— 它数绝对票数,会误杀 4437 个 conf=1.000 的真窗户
> (立面边缘本来就只有少数相机看得见)。详见 `mask点云分割.md` 第三节。

---

## 六、参数表

| 参数 | 默认 | 说明 |
|---|---|---|
| `--images DIR` | 从 `--scene` 推 | **去畸变图**目录(不是原图目录) |
| `--scene DIR` | — | COLMAP sparse 目录,只用来推 `--images` |
| `--out DIR` | **必填** | mask 输出目录 |
| `--text C` | — | 文字概念,**可重复给**。顺序即类别 id 顺序 |
| `--text-file PATH` | — | 每行一个概念,`#` 开头是注释 |
| `--model` / `--bpe` | 按候选路径找 | `sam3.pt` / BPE 词表 |
| `--conf` | 0.25 | 置信度阈值,漏检多就调低 |
| `--imgsz` | 1008 | 推理分辨率,显存不够调小 |
| `--half` | 关 | 半精度,更快更省显存 |
| `--device` | 自动 | `cuda` / `cpu` / `0` |
| `--expect-size W H` | — | 期望图像尺寸,如 `3781 2788`。**强烈建议给** |
| `--priority` | `first` | 重叠归谁,`first` / `last` |
| `--limit N` | — | 只处理前 N 张 |
| `--overwrite` | 关 | 重跑已存在的(默认跳过,断点续跑) |
| `--no-write-empty` | 关 | 什么都没检测到的图不写 png(默认照写全 0 的) |
| `--overlay-dir DIR` | — | 半透明叠加图,肉眼验收 |
| `--save-json PATH` | — | 统计(每图每概念的像素数/实例数/重叠数) |
| `--selftest` | 关 | 纯合成自检,不需要模型和 GPU |

### 颜色

沿用仓库的颜色语言:**类别 1 = 青色**,和点云工具里「被选中的那部分」同一个颜色。
叠加图和点云图例的颜色是对得上的,不用做映射。

---

## 七、性能(待实测)

> ⚠️ **这一节是空的,因为我没法替你跑。** 下面只是量级预估,第一次跑完把实测值填进来。

硬件:RTX 5070 Ti Laptop,12 GB 显存,驱动 596.49。

| 项 | 预估 | 实测 |
|---|---|---|
| 加载 `sam3.pt`(3.45 GB) | 5~15 s(一次性) | |
| 单张推理(3781×2788) | 1~4 s | |
| 13 张总计 | 30 s ~ 2 min | |
| 峰值显存 | 4~8 GB(`--half` 更省) | |

`--limit 1` 那一步会打印分段耗时,直接抄进来。
显存不够的话工具会**自动把 `--imgsz` 降到 0.75 倍重试**(最多两档),同时打警告;
再不够就 `OOM`,加 `--half` 或手动调小 `--imgsz`。

---

## 八、坑

| # | 现象 | 原因 | 解决 |
|---|---|---|---|
| 1 | `TypeError: 'SimpleTokenizer' object is not callable` | 装了 PyPI 上的 `clip` 包(和 fork **同为 1.0 版**,`pip install clip` 会装错的那个),接口不一样 | `pip uninstall clip -y`,再装 ultralytics 的 fork(本机得走 ghproxy,见坑 13) |
| 2 | 找不到 `sam3.pt` | **它不会自动下载**,得手动放 | 见第四节;`--model` 显式给路径 |
| 3 | HuggingFace 下不动 / 401 | `facebook/sam3` 是 gated 仓库,大陆申请常被拒 | 走 ModelScope 镜像(第四节) |
| 4 | 图像尺寸 3648×2736 对不上 | `--images` 指到了**原图**目录 `data\food` | 指到 `work\<场景>\dense\images\`。给 `--expect-size` 让它第一张就拦住 |
| 5 | 掩膜放大后类别 id 混进奇怪的值 | 用了双线性插值 —— id 是**标号**不是强度,插值会在 1 和 2 之间造出 1.5 | 工具全程最近邻,`--selftest` 断言值集不扩张。**别自己在外面对索引图做缩放** |
| 6 | 某张图上一个概念都没标出来 | 该图确实没有;或提示词不对 | 报告里会列 `empty_classes`。**所有图**都是 0 才算真问题(会显著警告) |
| 7 | 某个概念显示 `0px(N个)←被盖住` | 检测到了,但被优先级更高的概念整个盖住 | 调 `--priority`,或把更具体的概念放到 `--text` 前面 |
| 8 | 窗户大量漏检 | 窗户在图里太小,或 `--conf` 太高 | 先 `--imgsz` 调大 / 降 `--conf`;还不够就得切块推理(本次没做) |
| 9 | 显存不够 OOM | 12 GB 笔记本卡,`retina_masks` 出原尺寸掩膜很吃显存 | 工具会自动降 `--imgsz` 重试;再不够加 `--half` |
| 10 | `masks\` 里缺了几张图的 mask | 用 `--no-write-empty` 跳过了全空的图 | 去掉这个开关(默认就会写全 0 的 png,下游不会报 missing) |
| 11 | 下游报「mask 尺寸对不上」 | 生成的 png 被外部工具重新存过(比如画图软件另存) | 别用图形软件碰 `masks\` 里的文件 |
| 12 | 断点续跑时旧的坏 mask 没被覆盖 | 默认跳过已存在的文件 | 删掉那个 png,或整个加 `--overwrite` 重跑 |
| 13 | `pip install git+https://github.com/...` **卡住不动** | 本机 `github.com` 直连不通(HTTP 000),不是网慢 | 用 ghproxy:`pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "git+https://ghproxy.net/https://github.com/ultralytics/CLIP.git"` |
| 14 | `C:\Users\FANLS\.conda\envs\sam3\python.exe` 不存在 | 本机 conda 装在 `D:\ProgramData\anaconda3`,环境建在它自己的 `envs` 下 | 以 `conda env list` 打印的为准,本机是 `D:\ProgramData\anaconda3\envs\sam3\python.exe` |

---

## 九、输出文件

| 文件 | 内容 |
|---|---|
| `masks\<stem>.png` | **主产物**。uint8 单通道索引图,0=背景,1/2/3=第 1/2/3 个概念。文件名 stem 和去畸变图一致 |
| `<overlay-dir>\<stem>.png` | 半透明叠加图,肉眼验收用 |
| `<save-json>` | 每张图的:每概念像素数/实例数、重叠像素数、掩膜原始分辨率、空概念列表;以及全场汇总 |

`--overlay-dir` 和 `--save-json` 都是可选的,但**第一次跑别省**。

---

## 十、和主流程的关系

```
data\food\  (原图 3648x2736,食品学院的楼)
    │  ①-⑤ COLMAP
    ▼
work\food\dense\
    ├── images\   去畸变图 3781x2788   ←★ SAM 3 跑这
    └── sparse\   PINHOLE + 位姿
    │  ⑥-⑦ InterfaceCOLMAP → DensifyPointCloud
    ▼
work\food\mvs\scene_dense.ply          ←★ 点云
    │  ⑧ ReconstructMesh
    ▼
work\food\mvs\scene_dense_mesh.ply     ←★ 遮挡网格
    │
    ▼
tools\sam3_masks.py  →  masks\*.png  →  tools\mask_to_points.py  →  每点一个类别标签
```

**两个环境,一条文件链**:

| 环境 | 跑什么 | 依赖 |
|---|---|---|
| `sam3` | `sam3_masks.py` | torch(cu128)、ultralytics、GPU |
| `basic_env` | `mask_to_points.py`、SfM 那条线 | numpy、open3d、trimesh |

`sam3_masks.py` 是**完全独立**的:不 import `bbox_to_mesh`,也不 import 任何 SfM 那边
的东西 —— 唯一的接口就是 `masks\` 目录里的 PNG。所以 `bbox_live.py --verify` 那套
逐位对比契约不受任何影响。
