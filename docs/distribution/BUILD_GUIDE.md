# dgl-ascend Wheel 构建操作指导

> 面向构建者：如何从源码编出可分发的 dgl-ascend wheel。已验证路径（torch 2.7.1 + CANN 8.5 + manylinux_2_28_aarch64，910B3 上全测试通过）。
>
> 关联文档：分发方案 `PYPI_PUBLISH_PLAN.md`、用户安装 `USER_GUIDE.md`、manylinux 分析 `MANYLINUX_PLAN.md`。

---

## 目录

1. [产物与版本矩阵](#1-产物与版本矩阵)
2. [构建前提](#2-构建前提)
3. [主流程：manylinux_2_28 容器构建（路径 A）](#3-主流程manylinux_2_28-容器构建路径-a)
4. [关键约束与坑（必读）](#4-关键约束与坑必读)
5. [版本矩阵扩展](#5-版本矩阵扩展)
6. [备选：宿主机直建（linux_aarch64，快）](#6-备选宿主机直建linux_aarch64快)
7. [验证](#7-验证)
8. [故障排查](#8-故障排查)

---

## 1. 产物与版本矩阵

### 1.1 产物命名

```
dgl_ascend-{dgl版本}+cann{CANN版本}-{cpython}-{abi}-{平台}.whl
例: dgl_ascend-2.5+cann8.5-cp311-cp311-manylinux_2_28_aarch64.whl
```

- `2.5`：上游 DGL 版本（`python/dgl/_ffi/libinfo.py` `__version__`）。
- `+cann8.5`：构建所用 CANN 版本（对标 DGL CUDA 的 `+cu118`），由 `DGL_VERSION_SUFFIX` 注入。
- `cp311-cp311`：Python 标签 + ABI 标签（因含编译型 cython 扩展，绑定具体 Python）。
- `manylinux_2_28_aarch64`：平台标签（glibc≥2.28 通用）。

### 1.2 版本矩阵（对标 DGL CUDA 的 cuXXX × cpXXX）

矩阵是 **2D：(CANN/torch_npu 版本) × (Python 版本)**：

| 维度 | 取值 | 说明 |
|------|------|------|
| CANN / torch_npu | CANN 8.5 + torch_npu 2.7.1.post2（当前） | torch、torch_npu、CANN 三者版本耦合，由 torch_npu 版本统一决定。换 CANN 版本要换对应 torch_npu + CANN toolkit |
| Python | cp310 / cp311 / cp312 | 对标 DGL CUDA 的 Python 矩阵 |
| 架构 | aarch64（x86_64 需 x86 Ascend 机器） | — |

每个 (CANN, Python) 组合出一个 wheel，全列进 `repo.html`，pip 按用户机自动选。

> **只出 manylinux_2_28，不出 linux_aarch64**：manylinux_2_28（glibc≥2.28）是 linux_aarch64（glibc≥2.38）的超集，覆盖更广，linux_aarch64 冗余。

---

## 2. 构建前提

### 2.1 构建环境（容器）

需在一个**已具备 NPU + CANN 的 Ascend 容器**内执行（构建用 podman 在其中再起 manylinux 子容器）。容器拉起要点：

```bash
docker run -itd --name <name> --privileged --net=host --shm-size=32g \
  --device /dev/davinci0 ... --device /dev/davinci7 \
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v <repo>:/data <image> bash
```

### 2.2 自检

| 项 | 要求 | 检查 |
|----|------|------|
| NPU 可见 | 至少 1 张空闲卡 | `npu-smi info` |
| CANN toolkit | 含 `ascendc.cmake` | `ls $ASCEND_TOOLKIT_HOME/compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake` |
| `import torch_npu` 干净 | NPU 在场时 stderr 无驱动报错 | `python3 -c "import torch_npu" 2>&1 \| head`（不应有 `DrvMngGetConsoleLogLevel`/`ATRACE`） |
| podman | 已装 | `podman --version`（无则 `dnf install podman patchelf`） |
| 子模块 | 已初始化 | `ls third_party/dmlc-core/` 非空 |

> ⚠️ `import torch_npu` 必须干净——否则其驱动日志会污染 cmake 探测的路径变量，导致 makefile `target pattern contains no '%'`。NPU 设备穿透是根因解法（见 §4）。

---

## 3. 主流程：manylinux_2_28 容器构建（路径 A）

> 已验证：torch 2.7.1 + CANN 8.5 + cp311，910B3 上 8 单进程 + 5 分布式测试全过。

### 3.1 准备 podman（容器内 overlay → vfs）

宿主机若是 overlay 文件系统，podman 的 overlay 驱动会卡死，必须用 vfs：

```bash
cat > /tmp/podman-vfs.conf <<'EOF'
[storage]
driver = "vfs"
graphroot = "/var/lib/containers/storage-vfs"
EOF
# quay.io CDN 国内不可达，用 DaoCloud 代理
CONTAINERS_STORAGE_CONF=/tmp/podman-vfs.conf podman pull quay.m.daocloud.io/pypa/manylinux_2_28_aarch64:latest
# 注：vfs 解包慢，首次拉取约 10-15 分钟；若卡在收尾，重跑同条命令即可完成提交
```

### 3.2 构建镜像（装 torch/torch_npu + 工具）

`docker/Dockerfile.manylinux_cann_aarch64` 关键内容（CANN 不装进镜像，运行时只读挂载）：
```dockerfile
FROM quay.m.daocloud.io/pypa/manylinux_2_28_aarch64:latest
ARG PYVER=cp311-cp311
ENV PYBIN=/opt/python/${PYVER}/bin
ENV PATH=${PYBIN}:${PATH}
ENV ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-8.5.0
ENV LD_LIBRARY_PATH=${ASCEND_TOOLKIT_HOME}/lib64:...
# pip 用华为云镜像；装构建工具 + torch + torch_npu
RUN ${PYBIN}/pip install --upgrade pip setuptools wheel build "Cython>=0.29" auditwheel cmake patchelf
RUN ${PYBIN}/pip install torch==2.7.1
RUN ${PYBIN}/pip install torch_npu==2.7.1.post2
```

构建镜像（上下文用 `docker/` 避免传整个仓库）：
```bash
cd /data/dgl-ascend-install/dgl-ascend
CONTAINERS_STORAGE_CONF=/tmp/podman-vfs.conf podman build -t dgl-ascend-build:aarch64 \
    -f docker/Dockerfile.manylinux_cann_aarch64 docker/
```

### 3.3 构建 wheel（NPU 穿透 + CANN 只读挂载）

```bash
CONTAINERS_STORAGE_CONF=/tmp/podman-vfs.conf podman run --rm --privileged --cpus=32 --memory=128g \
  --device /dev/davinci0 --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /usr/local/Ascend:/usr/local/Ascend:ro -v "$PWD":/work -w /work \
  dgl-ascend-build:aarch64 bash -c '
set -e
export PATH=/opt/python/cp311-cp311/bin:$PATH
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-8.5.0
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:${ASCEND_TOOLKIT_HOME}/lib64:$LD_LIBRARY_PATH
export SOC_VERSION=Ascend910B3 TORCH_DEVICE_BACKEND_AUTOLOAD=0

# 运行时补依赖（镜像里 Cython 可能是 3.x，强制 <3；torch_npu 导入需 numpy/pyyaml）
pip install --quiet numpy pyyaml "cmake<4" "Cython<3"
# gfortran 库在 gcc-toolset 非标准路径，cmake find_library 找不到，建符号链接
GCC_LIB=$(dirname "$(gcc -print-file-name=libgfortran.so)")
ln -sf "$GCC_LIB/libgfortran.so" /usr/lib64/libgfortran.so
ln -sf "$GCC_LIB/libgomp.so" /usr/lib64/libgomp.so

cd /work
DGL_VERSION_SUFFIX=+cann8.5 python python/update_version.py   # 注入 +cann8.5 到 libinfo.py/c_runtime_api.h/meta.yaml

# 编译（必须 -c 干净构建，AscendC kernel 不支持增量）
bash script/build_dgl_ascend.sh -t release -c -j 32

# 打 wheel
cd python && rm -rf build dist *.egg-info dgl/_ffi/_cy3
DGL_PACKAGE_SUFFIX=-ascend python -m build --wheel --no-isolation
'
# 产物: python/dist/dgl_ascend-2.5+cann8.5-cp311-cp311-linux_aarch64.whl
```

### 3.4 重标签为 manylinux_2_28（用 `wheel tags`，不用 auditwheel）

manylinux 容器编的 libdgl.so 符号 ≤2.28（合规），无需 bundle 任何库，只需改平台标签：

```bash
mkdir -p repaired_wheels && cp python/dist/dgl_ascend-*-linux_aarch64.whl repaired_wheels/
cd repaired_wheels
python -m wheel tags --platform-tag manylinux_2_28_aarch64 --remove dgl_ascend-*-linux_aarch64.whl
# → dgl_ascend-2.5+cann8.5-cp311-cp311-manylinux_2_28_aarch64.whl
```

合规性自检（wheel 内每个 .so 的 GLIBC 都应 ≤2.28）：
```bash
for so in dgl/libdgl.so dgl/_ffi/_cy3/core.*.so dgl/dgl_sparse/*.so dgl/graphbolt/*.so; do
  echo "$(readelf -V $so 2>/dev/null | grep -oE 'GLIBC_[0-9.]+' | sort -u -V | tail -1) <- $so"
done
```

### 3.5 生成 repo.html + 托管

```bash
python tools/gen_repo_html.py --wheel-dir <wheel目录> --base-url <托管URL>/ --out repo.html
```

---

## 4. 关键约束与坑（必读）

| # | 约束 | 原因 | 不遵守的后果 |
|---|------|------|------------|
| 1 | **Cython 必须 `<3`**（0.29.37） | `dgl/_ffi/_cython/function.pxi` 用 Py2 `long` | `CompileError: undeclared name: long` |
| 2 | **cmake 必须 `<4`** | cmake 4 FindPython3 行为变 | find_package 异常 |
| 3 | **CMakeLists 用 `Development.Module`**（不是 Development） | manylinux cp311 无 libpython.so/.a | `find_package(Python3 Development)` 失败 |
| 4 | **gfortran 库符号链接**到 `/usr/lib64` | libgfortran.so 在 gcc-toolset 非标准路径 | `Could not find GFORTRAN_LIB` |
| 5 | **manylinux 子容器必须穿透 NPU 设备**（`--device /dev/davinci*`） | 无设备时 `import torch_npu` 驱动报错污染 cmake 变量 | `target pattern contains no '%'` |
| 6 | **重标签用 `wheel tags`，不用 `auditwheel repair`** | auditwheel 会把 libdgl.so 重定位到 .libs/，DGL 的 ctypes.CDLL 双加载 → 重复注册崩溃 | `PackedFunc already registered` |
| 7 | **改 c_runtime_api.h 后必须 `-c` 干净重建** | AscendC kernel 用 ExternalProject，增量留陈旧 .o | `ld.lld: unknown file type` |
| 8 | **`SOC_VERSION` 必须匹配硬件** | 默认 910B4，硬件 910B3 时 kernel 不匹配 | kernel 运行异常 |
| 9 | **`TORCH_DEVICE_BACKEND_AUTOLOAD=0`** | 防 torch 自动加载 torch_npu 干扰 cmake 探测 | cmake 探测路径被污染 |
| 10 | **装 numpy/pyyaml** | torch_npu 导入时需要（未声明在 wheel deps） | `No module named 'yaml'` |
| 11 | **`--no-isolation`** | `python -m build` 默认隔离会去 PyPI 拉 build 依赖（pypi.org 不通） | 构建失败 |
| 12 | **torch_npu 版本须匹配 CANN** | libdgl.so 按 torch C++ ABI 编译 | 2.8.0（无 post，配 CANN 8.3）跑在 CANN 8.5 上 → `free(): invalid pointer` 崩溃；8.5 用 2.7.1.post2 |

---

## 5. 版本矩阵扩展

### 5.1 多 Python 版本（cp310 / cp311 / cp312）

一个 manylinux 镜像里有 `/opt/python/cp310-cp310`、`cp311-cp311`、`cp312-cp312`。对每个 Python：

1. 镜像里装该 Python 的 torch/torch_npu（改 Dockerfile 的 `ARG PYVER` 重建镜像，或运行时 `/opt/python/<PYVER>/bin/pip install torch==2.7.1 torch_npu==2.7.1.post2`）。
2. §3.3 构建命令里把 `PATH=/opt/python/cp311-cp311/bin` 换成目标 `cp310-cp310` / `cp312-cp312`。
3. 产出对应 `cp310-cp310` / `cp312-cp312` 的 wheel。

3 个 wheel 都放进 `repo.html`，pip 按用户 Python 自动选。

### 5.2 多 CANN 版本（如 +cann8.1）

每个 CANN 版本需独立构建环境（CANN toolkit + 匹配 torch_npu）：

1. 改 `setup.py` 的 `install_requires`：`torch==<对应版本>` / `torch_npu==<对应版本>`。
2. Dockerfile 里 `pip install torch==<对应>` / `torch_npu==<对应>`。
3. CANN toolkit 换版本：只读挂载对应 CANN 目录（`ASCEND_TOOLKIT_HOME` 指向它），或临时容器里装该 CANN 的 `.run`。
4. `DGL_VERSION_SUFFIX=+cann8.1` 注入版本后缀。
5. 产出的 wheel 名带 `+cann8.1`，与 `+cann8.5` 区分。

> CANN 一般向后兼容：`+cann8.5` wheel 只能在 CANN≥8.5 跑；要覆盖 CANN 8.1 用户须专门编 `+cann8.1`。

### 5.3 x86_64 架构

需在 x86 Ascend 机器上执行（CANN/torch 无法交叉编译）：换 `manylinux_2_28_x86_64` 镜像 + x86 CANN toolkit + x86 torch/torch_npu，流程同上。

---

## 6. 备选：宿主机直建（linux_aarch64，快）

若仅用于验证、不需 manylinux 广兼容（用户都在 glibc≥2.38 的 openEuler 24.03+），可在宿主机直接建，省掉 podman/镜像（~5-8 分钟）：

```bash
cd /data/dgl-ascend-install/dgl-ascend
# 用系统 python（已有 torch 2.7.1）建 venv，--system-site-packages 继承 torch
python3 -m venv --system-site-packages /tmp/dgl-venv
/tmp/dgl-venv/bin/pip install "Cython<3" build auditwheel wheel

export PATH=/tmp/dgl-venv/bin:$PATH SOC_VERSION=Ascend910B3 TORCH_DEVICE_BACKEND_AUTOLOAD=0
bash script/build_dgl_ascend.sh -t release -c -j 32
cd python && DGL_PACKAGE_SUFFIX=-ascend python -m build --wheel --no-isolation
# → dgl_ascend-2.5+cann8.5-cp311-cp311-linux_aarch64.whl (glibc≥2.38)
```

> 产物是 `linux_aarch64`（非 manylinux），只能在 glibc≥2.38 机器装。正式分发用 §3 的 manylinux 版。

---

## 7. 验证

### 7.1 安装 + 基础验证

```bash
pip install --no-deps --force-reinstall <wheel>
python -c "
import torch, torch_npu, dgl
print(dgl.__version__, torch.npu.is_available(), torch.npu.get_device_name(0))
"
```

### 7.2 跑 tests/ascend/ 脚本（不修改脚本）

```bash
cd tests/ascend
# 单进程（用空闲卡，默认 --device-id 6 可能被占）
python test_spmm_sum_npu.py --device-id 0 --reduces sum
# 分布式（需 HCCL_IF_IP 设为本机 IP）
export HCCL_IF_IP=<本机IP>
python -m torch.distributed.run --nproc_per_node=4 test_dist_optim_npu.py
```

预期：8 个单进程测试 + 5 个分布式用例全 PASS，无 `free(): invalid pointer`。

### 7.3 端到端 `-f` 分发验证

```bash
# 托管目录起 http 服务
cd wheels && python -m http.server 18080 --bind 0.0.0.0
# 干净 venv 安装
pip install dgl-ascend -f http://<本机IP>:18080/repo.html
```

---

## 8. 故障排查

| 症状 | 原因 | 解决 |
|------|------|------|
| `Cython.Compiler.Errors.CompileError: core.pyx` / `undeclared name: long` | Cython 3.x | `pip install "Cython<3"` |
| `ld.lld: ... unknown file type`（AscendC kernel） | 增量构建留陈旧 .o | `build_dgl_ascend.sh -c` 干净重建 |
| `find_package(Python3 ... Development)` 失败 | manylinux 无 libpython | CMakeLists 用 `Development.Module` |
| `Could not find GFORTRAN_LIB` | gfortran 库在非标准路径 | `ln -sf $(gcc -print-file-name=libgfortran.so) /usr/lib64/libgfortran.so` |
| `target pattern contains no '%'` | 容器无 NPU，`import torch_npu` 驱动日志污染 cmake | 子容器加 `--device /dev/davinci*` 穿透 |
| `PackedFunc ... already registered` 崩溃 | auditwheel 重定位 libdgl.so 导致双加载 | 用 `wheel tags` 重标签，不用 auditwheel |
| `free(): invalid pointer` 崩溃 | torch_npu 版本与 CANN 不匹配（如 2.8.0 跑在 CANN 8.5） | 用匹配 CANN 的 torch_npu 版本（8.5 → 2.7.1.post2） |
| `Int64/Float precision not supported on Ascend` | 用了 int64/float32 | `idtype=torch.int32` + 特征 `.to(torch.float16)` |
| `HCCL_SOCKET_IFNAME ... not found in nic list` / `HcclGetRootInfo error 6` | `HCCL_IF_IP` 指了别的机器 IP | `export HCCL_IF_IP=<本机IP>` |
| `No matching distribution found` | Python 版本/架构不匹配 | repo.html 须含对应 cpXXX + manylinux_2_28_aarch64 的 wheel |
| podman pull 卡死 | overlay 根上 overlay 驱动 | 用 vfs（`/tmp/podman-vfs.conf`） |
| podman pull 超时 | quay.io CDN 不可达 | 用 DaoCloud 代理 `quay.m.daocloud.io` |

---

## 附录 A：manylinux 踩坑记录

> 从 MANYLINUX_PLAN.md 提取的两个关键踩坑，供后续构建者参考。

### A.1 NPU 缺失导致 torch_npu 日志污染 cmake 变量

**现象**：manylinux 容器（无 NPU 设备）里构建时，cmake 探测的 `LIBTORCH_NPU_PATH` 等变量里混入 CANN/驱动日志文本（`DrvMngGetConsoleLogLevel failed` / `[INFO] RUNTIME ...`），导致 makefile `target pattern contains no '%'`，dmlc/GKlib/METIS 三个子目标全挂。

**根因**：容器里没有 NPU 设备 → `import torch_npu` 触发驱动初始化失败 → 驱动日志直写 stdout（fd 1）→ cmake `execute_process` 的 `OUTPUT_VARIABLE` 捕获全部 stdout（不只是 `print(...)` 那行）→ 日志文本污染变量 → 写进 `flags.make` 的 `CXX_INCLUDES`。

**解法**：manylinux 子容器穿透 NPU 设备（`--device /dev/davinci0` 等）。设备在场 → 驱动正常初始化 → 无日志污染。环境变量 `ASCEND_GLOBAL_LOG_LEVEL=3` / `ASCEND_SLOG_PRINT_TO_STDOUT=0` 对驱动级日志无效（驱动不走 CANN slog 体系）。

### A.2 auditwheel repair 重定位 libdgl.so 导致双加载崩溃

**现象**：`auditwheel repair` 后 wheel 装上跑 `import dgl` → `Check failed: override: Global PackedFunc ndarray._CAPI_DGLSparseMatrixGetFormat is already registered` 崩溃。

**根因**：auditwheel 把 wheel 内的 `dgl/libdgl.so` 重定位到 `dgl_ascend.libs/libdgl-<hash>.so`（加哈希名），并改写 cython 扩展的 `NEEDED` 指向它。但 DGL 的 `libinfo.py/find_lib_path` 仍按 `libdgl.so` 名字搜索，找到 `dgl/libdgl.so`（auditwheel 留了一份原件），`ctypes.CDLL` 第二次加载它。两份不同的 libdgl.so 各自注册一遍 PackedFunc → 重复注册崩溃。

**解法**：manylinux 容器编的 libdgl.so 符号 ≤2.28（`readelf -V` 实测最高 GLIBC_2.27 / GLIBCXX_3.4.22），**不需要 auditwheel bundle 任何库**。用 `wheel tags --platform-tag manylinux_2_28_aarch64` 纯重标签（只改 Tag 字段 + 文件名 + RECORD，不动 .so 位置），绕开 auditwheel 的重定位冲突。

```bash
python -m wheel tags --platform-tag manylinux_2_28_aarch64 --remove dgl_ascend-*-linux_aarch64.whl
```

---

> **文档版本**: v1.1
> **基于**: torch 2.7.1 + CANN 8.5 + manylinux_2_28_aarch64，910B3 全测试通过
> **最后更新**: 2026-08-03
