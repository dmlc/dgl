# dgl-ascend 用户使用指南

> 面向终端用户：如何安装并验证 dgl-ascend。源码构建方式见 `AscendInstallation.md`。

---

## 1. 安装前提（需自行备好，pip 装不了）

| 准备项 | 要求 | 检查命令 |
|--------|------|---------|
| NPU 硬件 + 驱动 | Ascend 910B 系列 | `npu-smi info` 能看到卡 |
| CANN toolkit | ≥ 8.5.0（wheel 名 `+cann8.5` 即指此版本） | `ls $ASCEND_TOOLKIT_HOME` |
| 操作系统 | aarch64 Linux，glibc ≥ 2.28 | `ldd --version` |
| Python | **3.11**（目前仅 cp311） | `python3 --version` |
| 网络 | 能访问 PyPI | — |

> Python 3.10 / 3.12 暂无对应 wheel，会报 `No matching distribution found`。

### 关键：CANN 环境必须先 source

CANN 提供运行时库（`libhccl`/`libascendcl`/`libacl`），不 source 会导致 `import dgl` 失败：

```bash
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
# 或手动确保 LD_LIBRARY_PATH 含:
#   $ASCEND_TOOLKIT_HOME/lib64
#   /usr/local/Ascend/driver/lib64
```

---

## 2. 安装（一条命令）

```bash
pip install dgl-ascend
```

> 若机器上已装了别的 torch 版本，先卸载：`pip uninstall -y torch torch_npu`。
> dgl-ascend 钉死 `torch==2.8.0` / `torch_npu==2.8.0.post2`，版本不匹配会运行时 segfault。

---

## 3. 安装时发生了什么（自动完成，无需干预）

1. pip 从 **PyPI** 下载 dgl-ascend wheel（cp311 + aarch64）。
2. wheel 内含 libdgl.so + cython 扩展 + 子组件 .so + Python 代码。
3. 解析依赖，从 **PyPI** 自动拉：`torch==2.8.0`、`torch_npu==2.8.0.post2`、numpy、scipy、pandas、psutil、pydantic、pyyaml、requests、tqdm、networkx、packaging。
4. 全部装入当前 Python 环境。

> 国内用户建议配 PyPI 镜像加速依赖下载（如华为云 `mirrors.huaweicloud.com/repository/pypi/simple`），写入 `~/.pip/pip.conf` 即可。

---

## 4. 安装后验证

### ⚠️ import 顺序：`torch` → `torch_npu` → `dgl`

`import dgl` 会立即加载 libdgl.so，需 torch/torch_npu 已先把符号载入全局命名空间，否则报 `ImportError: libtorch_npu.so`。

```bash
python -c "
import torch
import torch_npu
import dgl
print('dgl       ', dgl.__version__)         # 0.0.1
print('torch     ', torch.__version__)        # 2.8.0+cpu
print('torch_npu ', torch_npu.__version__)    # 2.8.0.post2
print('npu avail ', torch.npu.is_available()) # True
print('device    ', torch.npu.get_device_name(0))  # Ascend910B3
"
```

### 功能测试：SpMM

> Ascend kernel 的 dtype 限制：图索引必须 `int32`，特征必须 `float16`。用 int64/float32 会报 `not fully supported on Ascend yet`。

```bash
python -c "
import torch, torch_npu, dgl
g = dgl.graph(([0,1,2],[1,2,3]), idtype=torch.int32)          # 必须 int32
g = dgl.add_self_loop(g).to('npu:0')
h = torch.randn(g.num_nodes(), 16, device='npu:0').to(torch.float16)  # 必须 float16
g.ndata['h'] = h
g.update_all(dgl.function.copy_u('h','m'), dgl.function.sum('m','h'))
print('SpMM OK', tuple(g.ndata['h'].shape))   # (4, 16)
"
```

看到 `SpMM OK (4, 16)` 即安装成功。

---

## 5. 常见问题

| 现象 | 原因 | 解决 |
|------|------|------|
| `No matching distribution found` | Python 非 3.11，或架构非 aarch64 | 用 Python 3.11；x86_64/其他 Python 版本暂无 wheel |
| `ImportError: libhccl.so` / `libascend_hal.so` | CANN/驱动环境未 source | `source setenv.bash` 或补 `LD_LIBRARY_PATH` |
| `ImportError: libtorch_npu.so` | import 顺序错（dgl 先于 torch_npu） | 严格按 `torch`→`torch_npu`→`dgl` |
| 安装成功但首次 kernel 调用 segfault | torch/torch_npu 版本非 2.8.0 | `pip install torch==2.8.0 torch_npu==2.8.0.post2` |
| `Int64 precision not fully supported` | 图索引用了 int64 | `dgl.graph(..., idtype=torch.int32)` |
| `Float precision not fully supported` | 特征用了 float32 | 特征 `.to(torch.float16)` |
| `pip` 卡在下载 torch | 走了公网 PyPI | 配国内 PyPI 镜像（见第 3 节） |

---

## 6. 示例：完整最小可用脚本

```python
import torch
import torch_npu
import dgl

device = torch.device("npu:0")

# 建图（int32 索引）
src = torch.tensor([0, 1, 2], dtype=torch.int32)
dst = torch.tensor([1, 2, 3], dtype=torch.int32)
g = dgl.graph((src, dst))
g = dgl.add_self_loop(g).to(device)

# 特征（float16）
h = torch.randn(g.num_nodes(), 16, device=device).to(torch.float16)
g.ndata["h"] = h

# 消息传递
g.update_all(dgl.function.copy_u("h", "m"), dgl.function.sum("m", "h"))
print("result shape:", g.ndata["h"].shape)
```

---

## 7. 版本说明

- `dgl-ascend 0.0.1`：基于上游 DGL + Ascend NPU 支持。
  - `0.0.1` = DGL-Ascend 版本
  - 构建所用 CANN ≥ 8.5（对标 DGL CUDA 的 `+cu118`），用户需 CANN ≥ 8.5。
- 强制依赖：`torch==2.8.0`、`torch_npu==2.8.0.post2`（精确版本，不可混用）。

---

> 遇到文档未覆盖的问题，可参考 `AscendInstallation.md` 的运行时说明，或反馈给维护者。
