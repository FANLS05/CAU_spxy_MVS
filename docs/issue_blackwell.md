# Issue 草稿:Windows CUDA 预编译包缺少 sm_120(Blackwell)

> 提交地址:https://github.com/cdcseacave/openMVS/issues/new
> 建议标签:`bug` / `build`

---

## 标题(复制这一行)

```
Windows CUDA release binaries only contain the build host's SM arch (CMAKE_CUDA_ARCHITECTURES=native), failing on Blackwell/sm_120 with CUDA error 209
```

---

## 正文(复制以下全部)

### Summary

The pre-built `OpenMVS_Windows_x64_CUDA` release binary cannot execute CUDA kernels on Blackwell GPUs (compute capability 12.0, e.g. RTX 50-series). CUDA device initialization *appears* to succeed, but any kernel launch fails with `cudaErrorNoKernelImageForDevice` (error 209).

The binary is compiled for a single architecture — `sm_89` (Ada) — with no PTX embedded, so there is no JIT fallback. This affects **every** GPU that is not Ada-generation, not just Blackwell.

This is a *build configuration* issue, not a missing-feature issue: the source already handles sm_120 (see #1248 / #1249). The problem is that the released binary is built with the default `CMAKE_CUDA_ARCHITECTURES=native`, which targets only the build machine's GPU.

### Environment

| | |
|---|---|
| OpenMVS | v2.4.0 (`OpenMVS_Windows_x64_CUDA.7z` from the release page) |
| OS | Windows 11 (10.0.26200) |
| GPU | NVIDIA GeForce RTX 5070 Ti Laptop, compute capability **12.0**, 11.94 GB |
| Driver | 596.49 |
| CUDA toolkit | 12.8 (V12.8.61) |

### Steps to reproduce

```powershell
DensifyPointCloud.exe -i scene.mvs --cuda-device 0 -v 3
```

The log misleadingly reports success at startup:

```
[App] CUDA device 0 initialized: NVIDIA GeForce RTX 5070 Ti Laptop GPU (compute capability 12.0; memory 11.94GB)
```

…but the kernels cannot be loaded.

### Evidence

**1. Only one cubin is embedded, and it is sm_89. No PTX is present for forward JIT:**

```
$ cuobjdump --list-elf DensifyPointCloud.exe
ELF file    1: DensifyPointCloud.1.sm_89.cubin

$ cuobjdump --list-ptx DensifyPointCloud.exe
(empty)

$ cuobjdump --list-elf Viewer.exe
ELF file    1: Viewer.1.sm_89.cubin
```

This is reproducible across all shipped CUDA binaries: `DensifyPointCloud.exe`, `Tests.exe`, `Viewer.exe`, `pyOpenMVS.dll` — all only `sm_89`, none with PTX.

**2. Loading the embedded cubin on the device fails (CUDA Driver API, independent of OpenMVS):**

```
cuInit                           -> OK
cuDeviceGet(0)                   -> NVIDIA GeForce RTX 5070 Ti Laptop GPU
compute capability               -> 12.0
cuCtxCreate                      -> OK
cuModuleLoad(sm_89 cubin)        -> FAIL  code=209
                                    (no kernel image is available for execution on the device)
```

Note that `cuCtxCreate` succeeds while `cuModuleLoad` fails — this is exactly the confusing behaviour seen in the OpenMVS log, where the "device initialized" line is printed but nothing can actually run.

### Root cause

`CMakeLists.txt` (develop, line ~227):

```cmake
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # `native` (CMake >= 3.24) tells nvcc to detect and target the build
    # host's GPU; the project's CMAKE_MINIMUM_REQUIRED guarantees support.
    SET(CMAKE_CUDA_ARCHITECTURES "native")
endif()
```

and line ~41:

```cmake
OPTION(OpenMVS_MAX_CUDA_COMPATIBILITY "Build for maximum CUDA device compatibility" OFF)
```

`native` resolves to whatever GPU is present on the **build** machine. The release runner evidently has an Ada GPU, so the published binaries contain `sm_89` only. Since `OpenMVS_MAX_CUDA_COMPATIBILITY` defaults to `OFF`, the `--list-gpu-arch` loop that would emit one `-gencode` per supported architecture is never taken.

Anyone building a distributed binary from a machine with GPU X effectively ships a binary that only runs on GPU X.

### Suggested fix

Build the released CUDA binaries with maximum compatibility for distribution:

```bash
cmake -DOpenMVS_USE_CUDA=ON -DOpenMVS_MAX_CUDA_COMPATIBILITY=ON ...
```

or set the architecture list explicitly:

```bash
cmake -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;120" ...
```

(the newer CMake `all-major` value is also an option if the minimum CMake version permits it)

Compiling for all supported architectures increases build time and binary size, but a released binary that silently cannot use a modern GPU is worse — the failure is non-obvious because device initialization succeeds.

### Minor: the startup log is misleading

```
CUDA device 0 initialized: ... (compute capability 12.0; memory 11.94GB)
```

This line is printed after context creation, which always succeeds. It would be much clearer to probe an actual kernel load at startup and downgrade to CPU with an explicit warning (e.g. `CUDA device 0 cannot execute this binary's kernels (no kernel image for device); falling back to CPU`) rather than letting the user discover the mismatch minutes later when kernel launch fails, or worse, run a long reconstruction only to fail.

### Related

- #1248 — "Add CUDA SM 12.0 (Blackwell/RTX 5090) support to MapSMtoCores" (closed)
- #1249 — "Add CUDA SM 8.7 (Jetson Orin) and SM 12.0 (Blackwell) GPU support" (closed)

Those addressed the `nGpuArchCoresPerSM` mapping used for a runtime *warning* ("MapSMtoCores for SM 12.0 is undefined; default to use 64 cores/SM"). That is a separate concern from the compiled architecture list, and the released binaries still contain no sm_120 code. This issue is therefore not a duplicate of either.

### Workaround

Use the CPU build (`OpenMVS_Windows_x64.zip`), or build from source with `-DOpenMVS_MAX_CUDA_COMPATIBILITY=ON`.
