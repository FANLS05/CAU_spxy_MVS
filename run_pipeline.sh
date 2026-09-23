#!/usr/bin/env bash
# =============================================================================
# 无人机视频 → 带纹理场景模型  一键流程脚本
#
# 流程:视频 → 抽帧 → COLMAP(SfM) → InterfaceCOLMAP → OpenMVS(稠密/网格/纹理)
#
# 用法:
#   ./run_pipeline.sh <视频文件|图像目录> [场景名]
#
# 例子:
#   ./run_pipeline.sh data/drone.mp4            # 视频输入:自动抽帧
#   ./run_pipeline.sh data/drone.mp4 castle     # 视频输入 + 指定场景名
#   ./run_pipeline.sh data/sceaux/images        # 图像目录输入:跳过抽帧
#
# 中断后重跑:已完成的步骤会自动跳过(检测输出文件是否存在)。
# 想强制全部重跑,加环境变量 FORCE=1
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENMVS_BIN="$ROOT/OpenMVS_Windows_x64/vc17/x64/Release"
FFMPEG_BIN="$ROOT/tools/ffmpeg/bin"
COLMAP_BIN="$ROOT/tools/colmap"

# ---------------------------------------------------------------------------
# 可调参数(用环境变量覆盖,例如: FPS=4 ./run_pipeline.sh ...)
# ---------------------------------------------------------------------------
FPS="${FPS:-2}"                  # 每秒抽多少帧
MAX_WIDTH="${MAX_WIDTH:-2000}"   # 抽帧后图像最大宽度(压低可提速)
JPEG_QUALITY="${JPEG_QUALITY:-2}" # ffmpeg -q:v,越小质量越高(2~5 合适)

# COLMAP 匹配策略:exhaustive(帧少,最准) / sequential / vocabtree
MATCHER="${MATCHER:-exhaustive}"
SEQUENTIAL_OVERLAP="${SEQUENTIAL_OVERLAP:-10}"

# OpenMVS 稠密重建
RESOLUTION_LEVEL="${RESOLUTION_LEVEL:-1}"  # 越大越快越糊
MAX_RESOLUTION="${MAX_RESOLUTION:-2000}"
MAX_THREADS="${MAX_THREADS:-0}"            # 0 = 全部核心

# 是否做网格细化(慢,但细节更好)
DO_REFINE="${DO_REFINE:-1}"

# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------
if [[ $# -lt 1 ]]; then
    echo "用法: $0 <视频文件|图像目录> [场景名]"
    echo
    echo "可调环境变量:"
    echo "  FPS=$FPS                 每秒抽帧数(仅视频输入)"
    echo "  MAX_WIDTH=$MAX_WIDTH           图像最大宽度(仅视频输入)"
    echo "  MATCHER=$MATCHER        exhaustive | sequential"
    echo "  RESOLUTION_LEVEL=$RESOLUTION_LEVEL      稠密重建降采样层级(0=全分辨率,最大=最快)"
    echo "  DO_REFINE=$DO_REFINE              1=做网格细化, 0=跳过"
    echo "  FORCE=1                  强制重跑已完成步骤"
    exit 1
fi

INPUT="$1"

if [[ -d "$INPUT" ]]; then
    MODE="images"
    SCENE="${2:-$(basename "$INPUT" | tr -cd '[:alnum:]_-')}"
elif [[ -f "$INPUT" ]]; then
    MODE="video"
    SCENE="${2:-$(basename "${INPUT%.*}" | tr -cd '[:alnum:]_-')}"
else
    echo "错误:找不到 $INPUT"; exit 1
fi

VIDEO="$INPUT"   # 仅 MODE=video 时使用

FORCE="${FORCE:-0}"
WORK="$ROOT/work/$SCENE"
IMG="$WORK/images"
SPARSE="$WORK/sparse"
DENSE="$WORK/dense"
MVS="$WORK/mvs"

# 图像目录输入:直接使用源目录(不复制),跳过抽帧
if [[ "$MODE" == "images" ]]; then
    IMG="$(cd "$INPUT" && pwd)"
fi

export PATH="$FFMPEG_BIN:$COLMAP_BIN:$OPENMVS_BIN:$PATH"

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
step_no=0
step() {
    step_no=$((step_no+1))
    echo
    echo "=============================================================="
    echo "  [$step_no] $*"
    echo "=============================================================="
}

# done_marker <文件>  -> 已完成则返回 0
done_marker() { [[ "$FORCE" != "1" && -e "$1" ]]; }

require() {
    command -v "$1" >/dev/null 2>&1 || {
        echo "错误:找不到 $1"
        echo "  期望位置: $FFMPEG_BIN / $COLMAP_BIN / $OPENMVS_BIN"
        exit 1
    }
}

timer_start() { T0=$(date +%s); }
timer_end() {
    local d=$(( $(date +%s) - T0 ))
    printf '  ⏱  耗时 %02d:%02d:%02d\n' $((d/3600)) $(((d%3600)/60)) $((d%60))
}

# ---------------------------------------------------------------------------
# 0. 环境检查
# ---------------------------------------------------------------------------
echo "=============================================================="
echo "  图像 / 视频 → 场景模型"
echo "=============================================================="
if [[ "$MODE" == "images" ]]; then
    echo "  输入:     图像目录 $IMG"
else
    echo "  输入:     视频 $VIDEO"
fi
echo "  场景名:   $SCENE"
echo "  工作目录: $WORK"
echo
[[ "$MODE" == "video" ]] && require ffmpeg
require colmap
require InterfaceCOLMAP
require DensifyPointCloud
require ReconstructMesh
require TextureMesh
echo "  工具检查: 全部就绪 ✓"

mkdir -p "$IMG" "$SPARSE" "$DENSE" "$MVS"

# ---------------------------------------------------------------------------
# 1. 准备图像(视频抽帧 / 图像目录直接使用)
# ---------------------------------------------------------------------------
if [[ "$MODE" == "images" ]]; then
    step "图像目录输入 — 跳过抽帧"
    n=$(find "$IMG" -maxdepth 1 -type f \
          \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \
             -o -iname '*.tif' -o -iname '*.tiff' \) | wc -l)
    [[ "$n" -gt 0 ]] || { echo "错误:$IMG 里没找到图像"; exit 1; }
    echo "  找到 $n 张图像"
else
    step "ffmpeg 抽帧 (${FPS} fps, 最大宽度 ${MAX_WIDTH})"
    if done_marker "$WORK/.frames_done"; then
        echo "  已跳过。想重跑请设 FORCE=1"
    else
        timer_start
        ffmpeg -hide_banner -loglevel warning -stats \
            -i "$VIDEO" \
            -vf "fps=${FPS},scale='min(${MAX_WIDTH},iw)':-2" \
            -q:v "$JPEG_QUALITY" \
            "$IMG/frame_%05d.jpg"
        timer_end
        n=$(find "$IMG" -maxdepth 1 -name 'frame_*.jpg' -type f | wc -l)
        [[ "$n" -gt 0 ]] || { echo "错误:没有抽到任何帧"; exit 1; }
        touch "$WORK/.frames_done"
        echo "  抽出 $n 帧"
    fi
fi

# 图像数量检查:少于 8 张稠密重建会不稳
if [[ "$n" -lt 8 ]]; then
    echo "  警告:只有 $n 张图像,重建质量可能不佳"
fi

# OpenMVS 的 --number-views 不能超过图像总数
N_VIEWS=$(( n < 8 ? n : 8 ))

# ---------------------------------------------------------------------------
# 2. COLMAP 稀疏重建
# ---------------------------------------------------------------------------
step "COLMAP 特征提取"
if done_marker "$WORK/db.db"; then
    echo "  已跳过"
else
    timer_start
    THREAD_ARG=()
    [[ "$MAX_THREADS" -gt 0 ]] && THREAD_ARG=(--FeatureExtraction.num_threads "$MAX_THREADS")
    colmap feature_extractor \
        --database_path "$WORK/db.db" \
        --image_path "$IMG" \
        --ImageReader.single_camera 1 \
        --FeatureExtraction.use_gpu 0 \
        --FeatureExtraction.max_image_size "$MAX_RESOLUTION" \
        "${THREAD_ARG[@]}"
    timer_end
fi

step "COLMAP 特征匹配 ($MATCHER)"
if done_marker "$WORK/.matched"; then
    echo "  已跳过"
else
    timer_start
    case "$MATCHER" in
        exhaustive)  colmap exhaustive_matcher \
                         --database_path "$WORK/db.db" \
                         --FeatureMatching.use_gpu 0 ;;
        sequential)  colmap sequential_matcher --database_path "$WORK/db.db" \
                         --FeatureMatching.use_gpu 0 \
                         --SequentialMatching.overlap "$SEQUENTIAL_OVERLAP" ;;
        vocabtree)   echo "错误:vocabtree 模式需要额外的 vocab tree 文件,请改用 exhaustive 或 sequential"
                     exit 1 ;;
        *)           echo "错误:未知 MATCHER=$MATCHER"; exit 1 ;;
    esac
    timer_end
    touch "$WORK/.matched"
fi

step "COLMAP 增量式 SfM (mapper)"
if done_marker "$SPARSE/0/cameras.bin"; then
    echo "  已跳过"
else
    timer_start
    rm -rf "$SPARSE"
    mkdir -p "$SPARSE"
    colmap mapper \
        --database_path "$WORK/db.db" \
        --image_path "$IMG" \
        --output_path "$SPARSE"
    timer_end
fi

# 取注册图像最多的那个子模型
if [[ -d "$SPARSE/1" ]]; then
    best=$(for d in "$SPARSE"/*/; do
             n=$(python -c "
import struct,sys,os
p=os.path.join('$d','images.bin')
try:
    f=open(p,'rb'); f.read(8); print(struct.unpack('<Q',f.read(8))[0])
except Exception: print(0)
" 2>/dev/null || echo 0)
             echo "$n $d"
           done | sort -rn | head -1 | awk '{print $2}')
    if [[ "$best" != "$SPARSE/0/" && -n "$best" ]]; then
        echo "  注意:检测到多个子模型,选用图像最多的: $best"
        echo "  (其余子模型保留在 $SPARSE 下)"
    fi
fi

step "COLMAP 去畸变 (image_undistorter)"
if done_marker "$DENSE/images/.done"; then
    echo "  已跳过"
else
    timer_start
    rm -rf "$DENSE"
    mkdir -p "$DENSE"
    colmap image_undistorter \
        --image_path "$IMG" \
        --input_path "$SPARSE/0" \
        --output_path "$DENSE" \
        --output_type COLMAP
    timer_end
    touch "$DENSE/images/.done"
fi

# ---------------------------------------------------------------------------
# 3. 转成 OpenMVS 工程
# ---------------------------------------------------------------------------
step "InterfaceCOLMAP 转换场景"
if done_marker "$MVS/scene.mvs"; then
    echo "  已跳过"
else
    timer_start
    mkdir -p "$MVS"
    InterfaceCOLMAP \
        -i "$DENSE" \
        -o "$MVS/scene.mvs" \
        --image-folder "$DENSE/images" \
        -w "$MVS"
    timer_end
fi

cd "$MVS"

# ---------------------------------------------------------------------------
# 4. 稠密点云
# ---------------------------------------------------------------------------
step "DensifyPointCloud 稠密点云重建  ← 最慢的一步"
if done_marker "$MVS/scene_dense.mvs"; then
    echo "  已跳过"
else
    timer_start
    DensifyPointCloud scene.mvs \
        --resolution-level "$RESOLUTION_LEVEL" \
        --max-resolution "$MAX_RESOLUTION" \
        --number-views "$N_VIEWS" \
        --max-threads "$MAX_THREADS"
    timer_end
fi

# ---------------------------------------------------------------------------
# 5. 网格重建
# ---------------------------------------------------------------------------
# 注意:ReconstructMesh 的 -p 是「输入稠密点云」,不是输出网格!
#   -p scene_dense_mesh.ply  是错的 —— 那份是网格,根本不是点云,会被
#     当成读不到而静默退回 scene_dense.mvs 里的稀疏点(6015 个),
#     于是只得到 5 千面的粗模。正确写法是下面这样。
step "ReconstructMesh 网格重建"
if done_marker "$MVS/scene_dense_mesh.ply"; then
    echo "  已跳过"
else
    timer_start
    ReconstructMesh scene_dense.mvs \
        -p scene_dense.ply \
        -o scene_dense_mesh.ply
    timer_end
fi

# ---------------------------------------------------------------------------
# 6. 网格细化(可选)
# ---------------------------------------------------------------------------
if [[ "$DO_REFINE" == "1" ]]; then
    step "RefineMesh 网格细化"
    if done_marker "$MVS/scene_dense_mesh_refine.ply"; then
        echo "  已跳过"
    else
        timer_start
        RefineMesh scene_dense.mvs \
            -m scene_dense_mesh.ply \
            -o scene_dense_mesh_refine.ply \
            --scales 1 --max-face-area 16
        timer_end
    fi
    MESH_FOR_TEXTURE="scene_dense_mesh_refine.ply"
else
    echo
    echo "  (已按 DO_REFINE=0 跳过网格细化)"
    MESH_FOR_TEXTURE="scene_dense_mesh.ply"
fi

# ---------------------------------------------------------------------------
# 7. 纹理
# ---------------------------------------------------------------------------
# 注意:TextureMesh 的 -o 只是「网格文件名」,输出格式由 --export-type 决定(默认 ply)。
#       TextureMesh 不写 .mvs 工程文件,别再传 -o xxx.mvs,那样只会得到 .ply。
#
# ★★ 必须关掉 seam leveling ★★
#   默认 --global-seam-leveling 1 --local-seam-leveling 1 在本机这个
#   v2.4.0 构建上会把「面自己的 texel」整片涂黑(只剩膨胀出来的光晕),
#   而 UV 恰好就指向那些黑块 —— 打开模型只见橙黑两色。
#   实测(14992 面小网格 / 1648306 面稠密网格结论一致):
#       默认                    黑 24.3%   网格 UV 落在真照片上  2.9%
#       只关 global             黑 24.5%                       0.8%
#       只关 local              黑  5.1%                      83.0%
#       两个都关                黑  0.0%                     100.0%
#   所以 main culprit 是 local,global 也顺带关掉。
#   另一个坑:--ignore-mask-label -2(想关掉 mask)会直接段错误,别用。
step "TextureMesh 纹理贴图"
if done_marker "$MVS/scene_dense_texture.glb"; then
    echo "  已跳过"
else
    timer_start
    SEAM_OFF=(--global-seam-leveling 0 --local-seam-leveling 0)
    # 出法 A:ply 网格 + png 贴图
    TextureMesh scene_dense.mvs \
        -m "$MESH_FOR_TEXTURE" \
        -o scene_textured.ply \
        "${SEAM_OFF[@]}"
    # 出法 B:单文件 glb(输出名自动取输入场景名 → scene_dense_texture.glb)
    TextureMesh scene_dense.mvs \
        -m "$MESH_FOR_TEXTURE" \
        --export-type glb \
        "${SEAM_OFF[@]}"
    timer_end
fi

echo
echo "=============================================================="
echo "  ✓ 全部完成"
echo "=============================================================="
echo "  工作目录: $MVS"
echo
echo "  最终模型:"
for f in "$MVS"/scene_dense_texture.glb "$MVS"/scene_textured.ply "$MVS"/scene_textured0.png; do
    if [[ -f "$f" ]]; then
        ls -la "$f" | awk '{printf "    %-50s %8.2f MB\n", $9, $5/1048576}'
    fi
done
echo
echo "  中间产物:"
for f in scene_dense.ply scene_dense_mesh.ply scene_dense_mesh_refine.ply; do
    if [[ -f "$MVS/$f" ]]; then
        ls -la "$MVS/$f" | awk '{printf "    %-50s %8.2f MB\n", $9, $5/1048576}'
    fi
done
echo
echo "  查看:"
echo "    \"$OPENMVS_BIN/Viewer.exe\" \"$MVS/scene_dense_texture.glb\""
echo
