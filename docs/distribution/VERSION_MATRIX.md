# dgl-ascend 版本矩阵构建方案

> 面向构建者：什么时候要出哪些 wheel、怎么出。照着本文执行即可。
>
> 关联：构建操作细节见 `BUILD_GUIDE.md`，本文只管"出哪些版本 + 矩阵扩展"。

---

## 1. 版本对应关系（官方 COMPATIBILITY.md）

### 1.1 torch_npu ↔ torch ↔ CANN

来源：https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.md

| CANN 版本 | torch_npu 可选版本 | torch 版本 |
|----------|-------------------|-----------|
| **9.1.0**（最新推荐） | 2.12.0 / 2.11.0 / 2.10.0.post4 / 2.9.0.post6 / 2.7.1.post8 | 2.12.0 / 2.11.0 / 2.10.0 / 2.9.0 / 2.7.1 |
| 9.0.0 | 2.9.0.post2 / 2.8.0.post4 / 2.7.1.post4 | 2.9.0 / 2.8.0 / 2.7.1 |
| **8.5.0**（本机当前） | 2.9.0 / 2.8.0.post2 / **2.7.1.post2** ← 我们用的 / 2.6.0.post5 | 2.9.0 / 2.8.0 / **2.7.1** / 2.6.0 |
| 8.3.RC1 | 2.8.0 / 2.7.1 / 2.6.0.post3 | 2.8.0 / 2.7.1 / 2.6.0 |
| 8.2.RC1 | 2.6.0 / 2.5.1.post1 | 2.6.0 / 2.5.1 |
| 8.1.RC1 | 2.5.1 / 2.4.0.post4 / 2.3.1.post6 | 2.5.1 / 2.4.0 / 2.3.1 |
| 8.0.0 | 2.4.0.post2 / 2.3.1.post4 | 2.4.0 / 2.3.1 |

> **一个 CANN 版本可配多个 torch_npu 版本**。CANN 8.5.0 可用 torch_npu 2.6/2.7.1/2.8.0.post2/2.9.0。我们选 **2.7.1.post2**（已验证全测试通过）。

### 1.2 torch ↔ Python

| torch 版本 | Python 版本 |
|-----------|------------|
| 2.12.0 | 3.10 / 3.11 / 3.12 / 3.13 / 3.14 |
| 2.10.0 | 3.10 / 3.11 / 3.12 / 3.13 |
| 2.9.0 | 3.10 / 3.11 / 3.12 / 3.13 |
| 2.8.0 | 3.9 / 3.10 / 3.11 / 3.12 / 3.13 |
| **2.7.1** | **3.9 / 3.10 / 3.11 / 3.12 / 3.13** |
| 2.6.0 | 3.9 / 3.10 / 3.11 |

### 1.3 硬件支持

Ascend 910 系列（含 910B3）✅。其他：950DT、A3、A2、310P、310B、310。

---

## 2. 当前已构建的版本

| CANN | torch_npu | torch | Python | wheel | 状态 |
|------|-----------|-------|--------|-------|------|
| 8.5.0 | 2.7.1.post2 | 2.7.1 | **cp311** | `dgl_ascend-2.5+cann8.5-cp311-cp311-manylinux_2_28_aarch64.whl` | ✅ 已构建 + 全测试通过 |

---

## 3. 版本矩阵计划

### 3.1 近期：CANN 8.5 × cp310/311/312（本机可直接编）

| CANN | torch_npu | torch | Python | wheel 状态 |
|------|-----------|-------|--------|-----------|
| 8.5.0 | 2.7.1.post2 | 2.7.1 | cp310 | ⬜ 待构建 |
| 8.5.0 | 2.7.1.post2 | 2.7.1 | cp311 | ✅ 已有 |
| 8.5.0 | 2.7.1.post2 | 2.7.1 | cp312 | ⬜ 待构建 |

> torch 2.7.1 支持 Python 3.9-3.13。当前只编 cp310/311/312（对标 DGL CUDA 的 Python 矩阵）。cp313/cp39 后续可扩展。

**预检结果**（已确认华为云镜像有对应 wheel）：
```
torch-2.7.1-cp310-cp310-manylinux_2_28_aarch64.whl          ✅
torch-2.7.1-cp312-cp312-manylinux_2_28_aarch64.whl          ✅
torch_npu-2.7.1.post2-cp310-cp310-manylinux_2_28_aarch64.whl ✅
torch_npu-2.7.1.post2-cp312-cp312-manylinux_2_28_aarch64.whl ✅
```

### 3.2 中期：多 CANN 版本（需额外 CANN toolkit）

| CANN | torch_npu | torch | 需要什么 |
|------|-----------|-------|---------|
| 8.5.0 | 2.7.1.post2 | 2.7.1 | 本机已有 ✅ |
| 9.1.0 | 2.12.0 | 2.12.0 | 下载 CANN 9.1.0 toolkit + 临时容器 |
| 9.0.0 | 2.8.0.post4 | 2.8.0 | 下载 CANN 9.0.0 toolkit + 临时容器 |

> 每个 CANN 版本需独立构建环境（CANN toolkit + 对应 torch_npu），因为 libdgl.so 链接 CANN 和 torch 的 C++ 库。构建方式：临时 manylinux 容器内装该 CANN toolkit + torch_npu，编完容器销毁（`--rm`），不影响本机。

---

## 4. 构建步骤（CANN 8.5 × cp310/cp312）

### 4.1 前提

- 本容器已有 NPU + CANN 8.5.0 + podman。
- `dgl-ascend-build:aarch64` 镜像（torch 2.7.1 for cp311）已构建或需重建。
- 仓库 `third_party/` 子模块已初始化。

### 4.2 重建 podman 环境（如已清理）

```bash
cat > /tmp/podman-vfs.conf <<'EOF'
[storage]
driver = "vfs"
graphroot = "/var/lib/containers/storage-vfs"
EOF
CONTAINERS_STORAGE_CONF=/tmp/podman-vfs.conf podman pull quay.m.daocloud.io/pypa/manylinux_2_28_aarch64:latest
# 若卡在收尾，重跑同条命令完成提交

cd /data/dgl-ascend-install/dgl-ascend
CONTAINERS_STORAGE_CONF=/tmp/podman-vfs.conf podman build -t dgl-ascend-build:aarch64 \
    -f docker/Dockerfile.manylinux_cann_aarch64 docker/
```

### 4.3 构建 cp310 + cp312（一个容器跑两个）

```bash
cd /data/dgl-ascend-install/dgl-ascend
CONTAINERS_STORAGE_CONF=/tmp/podman-vfs.conf podman run --rm --privileged --cpus=32 --memory=128g \
  --device /dev/davinci0 --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /usr/local/Ascend:/usr/local/Ascend:ro -v "$PWD":/work -w /work \
  dgl-ascend-build:aarch64 bash -c '
set -e
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-8.5.0
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:${ASCEND_TOOLKIT_HOME}/lib64:$LD_LIBRARY_PATH
export SOC_VERSION=Ascend910B3 TORCH_DEVICE_BACKEND_AUTOLOAD=0

GCC_LIB=$(dirname "$(gcc -print-file-name=libgfortran.so)")
ln -sf "$GCC_LIB/libgfortran.so" /usr/lib64/libgfortran.so
ln -sf "$GCC_LIB/libgomp.so" /usr/lib64/libgomp.so
cd /work
DGL_VERSION_SUFFIX=+cann8.5 python python/update_version.py >/dev/null

for PYVER in cp310-cp310 cp312-cp312; do
  echo "======== Building $PYVER ========"
  export PATH=/opt/python/${PYVER}/bin:$PATH
  pip install --quiet numpy pyyaml "cmake<4" "Cython<3" torch==2.7.1 torch_npu==2.7.1.post2

  # 干净编译（AscendC 不支持增量）
  bash script/build_dgl_ascend.sh -t release -c -j 32

  # 打 wheel
  cd python && rm -rf build dist *.egg-info dgl/_ffi/_cy3
  DGL_PACKAGE_SUFFIX=-ascend python -m build --wheel --no-isolation
  cd /work

  # 重标签 manylinux_2_28（用 wheel tags，不用 auditwheel）
  WHL=$(ls python/dist/dgl_ascend-*-linux_aarch64.whl)
  python -m wheel tags --platform-tag manylinux_2_28_aarch64 "$WHL" -w /work/matrix_wheels/
  echo "Done: $(ls /work/matrix_wheels/dgl_ascend-*${PYVER%%-*}*.whl)"
done
echo "======== All done ========"
ls -lh /work/matrix_wheels/
'
```

> 产物：`matrix_wheels/dgl_ascend-2.5+cann8.5-cp310-cp310-manylinux_2_28_aarch64.whl` 和 `dgl_ascend-2.5+cann8.5-cp312-cp312-manylinux_2_28_aarch64.whl`。

### 4.4 汇总 + 更新分发

```bash
cd /data/dgl-ascend-install

# 把 cp311 的也放进 matrix_wheels（已有）
cp wheels/dgl_ascend-2.5+cann8.5-cp311-cp311-manylinux_2_28_aarch64.whl matrix_wheels/

# 3 个 wheel 统一放 wheels/ + 重生成 repo.html
cp matrix_wheels/*.whl wheels/
python dgl-ascend/tools/gen_repo_html.py --wheel-dir wheels --out wheels/repo.html

# 更新 start.html 矩阵表（cp310 ✅ / cp311 ✅ / cp312 ✅）
# 手动编辑 wheels/start.html 的 <tbody>，把 cp310/cp312 列从 — 改成 ✅
```

### 4.5 验证

```bash
# 在 manylinux 容器内验证 cp310/cp312
CONTAINERS_STORAGE_CONF=/tmp/podman-vfs.conf podman run --rm \
  --device /dev/davinci0 --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /usr/local/Ascend:/usr/local/Ascend:ro -v /data/dgl-ascend-install/wheels:/wheels:ro \
  dgl-ascend-build:aarch64 bash -c '
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/cann-8.5.0/lib64:$LD_LIBRARY_PATH
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
for PYVER in cp310-cp310 cp312-cp312; do
  /opt/python/${PYVER}/bin/pip install --quiet numpy pyyaml torch==2.7.1 torch_npu==2.7.1.post2
  /opt/python/${PYVER}/bin/pip install --no-deps /wheels/dgl_ascend-*${PYVER%%-*}*.whl
  /opt/python/${PYVER}/bin/python -c "
import torch,torch_npu,dgl
print(\"$PYVER:\", dgl.__version__, torch.npu.is_available())
g=dgl.graph(([0,1,2],[1,2,3]),idtype=torch.int32).to(\"npu:0\")
g.update_all(dgl.function.copy_u(\"x\",\"m\"),dgl.function.sum(\"m\",\"h\"))
print(\"  SpMM OK\")
" 2>&1 | grep -vE "^\[|PROFILING|RUNTIME|DrvMng|ATRACE|WARN|UserWarning"
done
'
```

### 4.6 清理

```bash
CONTAINERS_STORAGE_CONF=/tmp/podman-vfs.conf podman rmi -a -f
rm -rf /var/lib/containers/storage-vfs /tmp/podman-vfs.conf
```

---

## 5. 扩展到其他 CANN 版本

以 CANN 9.1.0（torch_npu 2.12.0 / torch 2.12.0）为例：

### 5.1 需要的东西

- **CANN 9.1.0 toolkit 安装包**（`.run` 文件，aarch64）—— 从华为 Ascend 官网下载。
- **torch_npu 2.12.0 + torch 2.12.0** —— pip 安装（华为云镜像）。

### 5.2 构建

```bash
# Dockerfile 改 torch 版本
# 或运行时 pip install torch==2.12.0 torch_npu==2.12.0

# 临时容器：装 CANN 9.1.0 toolkit + torch 2.12.0 + 编译
CONTAINERS_STORAGE_CONF=/tmp/podman-vfs.conf podman run --rm --privileged \
  --device /dev/davinci0 ... \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v "$PWD":/work -w /work \
  manylinux_2_28_aarch64 bash -c '
# 装 CANN 9.1.0 toolkit（从 .run 文件）
COPY Ascend-cann-toolkit_9.1.0_linux-aarch64.run /tmp/cann.run
/tmp/cann.run --install --quiet --install-path=/usr/local/Ascend
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest

# 装 torch 2.12.0 + torch_npu 2.12.0
pip install torch==2.12.0 torch_npu==2.12.0 ...

# 版本后缀改 +cann9.1
DGL_VERSION_SUFFIX=+cann9.1 python python/update_version.py

# 编译 + 打包 + 重标签（同 §4.3）
bash script/build_dgl_ascend.sh -t release -c -j 32
cd python && DGL_PACKAGE_SUFFIX=-ascend python -m build --wheel --no-isolation
python -m wheel tags --platform-tag manylinux_2_28_aarch64 dist/*.whl -w /work/matrix_wheels/
'
# 容器销毁（--rm），本机 CANN 8.5 不受影响
```

> **注意**：CANN 9.1.0 的 `ascendc.cmake` 路径可能与 8.5.0 不同（`compiler/tikcpp/` vs `tools/tikcpp/`），CMakeLists.txt 已兼容两者。但需确认 CANN 9.1.0 的 `.run` 能在 manylinux（AlmaLinux 8）上安装。

### 5.3 其他 CANN 版本同理

换 torch_npu/torch 版本 + CANN toolkit + `DGL_VERSION_SUFFIX`，其余步骤不变。

---

## 6. 矩阵产出汇总

最终 `repo.html` 应包含（按需逐步补齐）：

```
# CANN 8.5 × torch 2.7.1
dgl_ascend-2.5+cann8.5-cp310-cp310-manylinux_2_28_aarch64.whl
dgl_ascend-2.5+cann8.5-cp311-cp311-manylinux_2_28_aarch64.whl
dgl_ascend-2.5+cann8.5-cp312-cp312-manylinux_2_28_aarch64.whl

# CANN 9.1 × torch 2.12.0（未来）
dgl_ascend-2.5+cann9.1-cp310-cp310-manylinux_2_28_aarch64.whl
dgl_ascend-2.5+cann9.1-cp311-cp311-manylinux_2_28_aarch64.whl
dgl_ascend-2.5+cann9.1-cp312-cp312-manylinux_2_28_aarch64.whl
```

用户 `pip install dgl-ascend -f <URL>/repo.html` 时，pip 按 (CANN 后缀 + Python 版本 + 架构) 自动选最匹配的 wheel。

---

## 7. 注意事项

| 事项 | 说明 |
|------|------|
| **构建=运行版本** | libdgl.so 按 torch C++ ABI 编译，构建用的 torch 版本必须和用户运行时一致（`install_requires` 钉死） |
| **torch_npu 须匹配 CANN** | 2.8.0（无 post，配 CANN 8.3）跑在 CANN 8.5 上 → `free(): invalid pointer` 崩溃。CANN 8.5 用 **2.7.1.post2**；2.8.0.post2/post4 未验证 |
| **only manylinux_2_28** | 不出 linux_aarch64（manylinux_2_28 是超集，覆盖更广） |
| **CANN 向后兼容** | `+cann8.5` wheel 只能在 CANN≥8.5 跑；要覆盖 CANN 8.1 用户须专门编 `+cann8.1` |
| **包名固定 `dgl-ascend`** | 不用 `dgl`（与上游 DGL 不同团队，避免包名冲突） |

---

> **文档版本**: v1.0
> **基于**: torch_npu 2.7.1.post2 + CANN 8.5.0 + manylinux_2_28_aarch64，910B3 全测试通过
> **最后更新**: 2026-07-29
> **数据来源**: Ascend/pytorch 官方 COMPATIBILITY.md
