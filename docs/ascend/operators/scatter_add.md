# ScatterAdd

> Ascend NPU 算子文档。

## 功能描述

ScatterAdd 是沿第一维的散射累加算子，数学语义为：

```
对每个 i ∈ [0, N):
    out[idx[i], *] += feat[i, *]
```

即把 `feat` 的每一行按 `idx` 指定的目标行号累加到 `out` 对应行上。当多个 `i` 映射到同一目标行时，所有命中的 `feat[i,*]` 全部累加到该 `out` 行（跨核写冲突由 GM 原子累加保证正确性）。

该算子为 DGL Ascend 后端的原生 kernel，替换原先的 CPU 回退路径（D2H + `cpu::ScatterAdd` + H2D），消除 host-device 间数据搬运开销。累加策略采用 Strategy A：`SetAtomicAdd<float>()` 开启 GM 原子 read-modify-write，通过 `DataCopyPad` UB→GM 实现原子加，语义与 CUDA `atomicAdd` 的 `+=` 逐字一致。**kernel 不清零 `out`，仅做 `+=`；`out` 由调用方预清零**（与 CUDA/CPU 一致，DGL Python 调用层 `F.zeros` 已保证）。

## 接口原型

```cpp
// host 侧 NDArray 接口（src/array/kernel_decl.h 声明，与 CUDA/CPU 同构）
template <int XPU, typename IdType, typename DType>
void ScatterAdd(NDArray feat, NDArray idx, NDArray out);

// Ascend 特化（本次新增）
template <>
void ScatterAdd<kDGLAscend, IdType, DType>(NDArray feat, NDArray idx, NDArray out);

// device kernel 入口（scatter_add_kernel.cpp）
extern "C" __global__ __aicore__ void kernel_scatter_add(
    GM_ADDR feat, GM_ADDR idx, GM_ADDR output, GM_ADDR tiling_ptr);

// host launcher（scatter_add.cc，ACLRT_LAUNCH_KERNEL 宏展开）
extern "C" uint32_t aclrtlaunch_kernel_scatter_add(
    uint32_t blockDim, aclrtStream stream,
    void* feat, void* idx, void* out, void* tiling);
```

> 上层 Python 通路：`dgl.backend.scatter_add(x, idx, m)` → `_CAPI_DGLKernelScatterAdd`（`_sparse_ops.py`），host 侧 `F.zeros` 预清零 `out` 后调用本算子。

## 参数说明

| 参数 | 内存位置 | 方向 | 类型 | 说明 |
|------|----------|------|------|------|
| feat | Device (GM) | 输入 | DType* | 特征张量，shape `[N, *]`，行优先；`N = feat->shape[0]`，`*` 为第 1..k 维乘积记为 `dim`。按行散射，每行被唯一核读取。 |
| idx | Device (GM) | 输入 | IdType* | 目标行号张量，shape `[N]`，元素须满足 `0 <= idx[i] < M`；越界行为由上游保证（kernel 不显式校验，与 CUDA 一致）。int64 时 host 侧经 `AsNumBits(32)` 位宽转换后送 kernel。 |
| out | Device (GM) | 输出 | DType* | 输出张量，shape `[M, *]`，`M = out->shape[0]`。**调用前必须由调用方清零**；kernel 仅做 `+=`，不清零。 |
| tiling | Device (GM) | 输入（host 下发） | `ScatterAddTilingData*` | 形状/切分参数 `{ N, M, featDim }`，由 host `aclrtMalloc` + `aclrtMemcpy` 下发，kernel `Init` 时读取。 |

### Tiling 结构体

```cpp
struct ScatterAddTilingData {
    uint32_t N;        // feat 行数（= idx 长度）
    uint32_t M;        // out 行数
    uint32_t featDim;  // 每行元素数 dim = ∏(out->shape[1..ndim])
};
```

## 支持数据类型

| 数据类型（IdType × DType） | 支持状态 | 说明 |
|----------|------|------|
| int32 × float32 | 支持 | 原生 kernel 直走，首版落地。 |
| int64 × float32 | 支持 | host 侧 `idx.CopyTo(CPU) → AsNumBits(32) → CopyTo(NPU)` 位宽转换后走 int32 内核（与 `segment_reduce.cc` 同范式，`AsNumBits` 为位宽保持转换非截断）。 |
| int32/int64 × half (float16) | 不支持 | `LOG(FATAL)` 占位，提示「Current Ascend scatter_add kernel only supports float features」。 |
| int32/int64 × bfloat16 | 不支持 | 非 CUDA 构建下由 `ATEN_FLOAT_TYPE_SWITCH_16BITS` 宏在到达特化前 `LOG(FATAL)`，与 `segment_reduce.cc` 一致。 |
| int32/int64 × double (float64) | 不支持 | `LOG(FATAL)` 占位。 |
| 其他 dtype (int8/uint8/...) | 不支持 | dispatch 走 `ATEN_FLOAT_TYPE_SWITCH_16BITS`，不进入本算子。 |

## 约束说明

### Shape 约束

- `feat->shape[0] == idx->shape[0] == N`。
- `out->ndim == feat->ndim`，且 `out->shape[1..ndim]` 与 `feat->shape[1..ndim]` 逐维相等（即 `dim` 相同）。
- `M = out->shape[0]`，`idx[i]` 须满足 `0 <= idx[i] < M`（上游保证，kernel 不校验）。
- `dim = ∏(out->shape[1..ndim])`，允许 `dim % 8 != 0`（非对齐），`DataCopyPad` 自动处理行尾对齐填充。
- `N`、`M`、`dim` 受 GM/HBM 容量约束，无固定上限；多核切分 `blockDim = min(N, 40)`，`N=0` 时 `blockDim=1`、kernel 提前返回（空操作）。

### dtype 约束

- 仅支持 `IdType ∈ {int32, int64}` × `DType = float32`，其余组合 `LOG(FATAL)`（见上表）。

### 清零契约

- **`out` 必须由调用方预清零**。kernel 仅执行 `+=`（`SetAtomicAdd` 原子累加），不清零也不初始化 `out`。DGL Python 调用层 `_sparse_ops.py` 已在调用 `_CAPI_DGLKernelScatterAdd` 前 `out = F.zeros(out_shp, dtype, ctx)`。若调用方未清零，原子加会在旧值上累加，结果错误——此行为与 CUDA/CPU 实现一致。

### 平台约束

- 目标芯片：Ascend 910B3（dav-2201，`--npu-arch=ascend910b`）。
- 代码架构：SIMD（实现载体 MemBase，`kernel_operator.h` + `GlobalTensor` + `TPipe`/`TQue`），纯 Vector 任务（`KERNEL_TYPE_AIV_ONLY`）。
- `SetAtomicAdd<float>` 要求芯片支持 float 类型 GM 原子加，dav-2201 支持。

### 确定性约束

- 累加顺序非确定：`SetAtomicAdd` 的原子 RMW 顺序由硬件调度，同一输入多次执行累加顺序可能不同（尤其跨核冲突时）。此为预期行为，与 CUDA `atomicAdd` 同范式，不保证 bit-exact，精度由混合容差覆盖。

### 调用前置条件

- host 侧 `ScatterAddDispatch`（`src/array/kernel.cc`）经 `ATEN_XPU_SWITCH_CUDA_ASCEND` + `ATEN_ID_TYPE_SWITCH` + `ATEN_FLOAT_TYPE_SWITCH_16BITS` 三级分派进入特化。调用前 host 侧执行 `aclrtSynchronizeDevice()` 排空 PyTorch NPU op，确保 `feat/idx/out` 数据就绪。

## 返回值 / 错误码

| 返回值 | 含义 |
|--------|------|
| （host 接口 `void`） | 无返回值。kernel launch 失败时 host 侧 `LOG(FATAL)` 终止（释放 tiling 后报错码）。 |
| `aclrtlaunch_kernel_scatter_add` 返回 `ACL_SUCCESS` | kernel launch 成功。 |
| `aclrtlaunch_kernel_scatter_add` 返回非 `ACL_SUCCESS` | launch 失败，host `aclrtFree(tiling)` 后 `LOG(FATAL)` 输出错误码。 |
| `aclrtMalloc` / `aclrtMemcpy` / `aclrtSynchronizeStream` / `aclrtFree` 失败 | `CHECK` 宏终止，输出 `Ascend Error, code: <e>`。 |
| 不支持 dtype | `LOG(FATAL)`："Current Ascend scatter_add kernel only supports float features." |
| 未编译 Ascend 支持 | `LOG(FATAL)`："Ascend support is not compiled. Please compile with -DUSE_ASCEND=ON"。 |

## 调用示例

### C++ 直接调用（host 侧）

```cpp
#include <dgl/array.h>
#include <dgl/runtime/device_api.h>

using dgl::aten::ScatterAdd;
using dgl::kDGLAscend;

// feat: NDArray shape [N, dim], dtype float32, ctx=NPU
// idx : NDArray shape [N],       dtype int32,   ctx=NPU
// out : NDArray shape [M, dim], dtype float32, ctx=NPU
//
// 调用方必须预清零 out（与 CUDA/CPU 一致，kernel 仅 +=）。
// dgl::aten::FillZeros(out);  // 或上层 F.zeros

ScatterAdd<kDGLAscend, int32_t, float>(feat, idx, out);

// int64 idx 同样支持，host 侧自动转 int32：
// ScatterAdd<kDGLAscend, int64_t, float>(feat, idx_int64, out);
```

### Python 上层调用（DGL 通路）

```python
import torch
import dgl
from dgl import backend as F

N, M, dim = 1024, 1024, 64
feat = torch.randn(N, dim, device="npu", dtype=torch.float32)
idx  = torch.randint(0, M, (N,), device="npu", dtype=torch.int32)

# DGL 内部 F.zeros 预清零 out，再调用 _CAPI_DGLKernelScatterAdd
out = F.zeros((M, dim), feat.dtype, feat.ctx)
dgl.backend.scatter_add(feat, idx, M, out=out)
# out[i] == sum(feat[j] for j where idx[j]==i), 累加顺序非确定
```

## 实现要点

> 本节为使用者提供实现概览，便于理解性能与确定性特征。

- **累加策略（Strategy A）**：`SetAtomicAdd<float>()` 开启 GM 原子累加 → 分批 `DataCopyPad` 搬 `feat` 行到 UB → `DataCopyPad` UB→GM（在原子模式下执行 atomic_add）→ `SetAtomicNone()` 复位。与 CUDA `atomicAdd` 语义逐字一致。
- **多核切分**：按 `N`（feat 行数）切分到各核，`blockDim = min(N, 40)`，核 `c` 负责 feat 行 `[c×ceil(N/blockDim), min((c+1)×ceil(N/blockDim), N))`。各核写出地址可能重叠（同一目标行被多行累加），由 `SetAtomicAdd` 保证原子正确性。
- **idx 数据流**：idx 一次性预搬 UB（`DataCopyPad` + `TBuf`），`PipeBarrier<PIPE_ALL>` 显式同步 MTE2→Scalar 后，主循环内 `LocalTensor.GetValue` 标量读 idx 值（生产代码禁用 `GlobalTensor::GetValue` 红线 API，改用 UB 预搬 + LocalTensor 读）。
- **流水线同步**：feat 采用 Double Buffer（`BUFFER_NUM=2`）。每批 `FreeTensor` 前插入 `PipeBarrier<PIPE_ALL>`（多队列汇聚点）排空在途 MTE3 写出，防止 Double Buffer 复用覆盖在途原子加源数据；feat `DeQue` 后用 `SetFlag/WaitFlag<HardEvent::MTE2_MTE3>` 事件同步跨 pipe 依赖（MTE2 搬入完成才能 MTE3 原子写出）。
- **原子态复位**：`Process()` 末尾 `PipeBarrier<PIPE_ALL>` 排空本核 MTE3 后调用 `SetAtomicNone()`，复位设备原子加模式寄存器，杜绝跨 launch 泄漏。

## 精度标准

| 项目 | 内容 |
|------|------|
| 输出 dtype | float32 |
| 判定方式 | 混合容差（非逐比特），逐元素 `|actual - golden| ≤ atol + rtol × |golden|` |
| rtol | 2^-10 ≈ 9.77e-4 |
| atol | 2^-16 ≈ 1.53e-5 |
| required_matched_ratio | ≥ 0.99 |
| max_abs_error_limit | 1e-2 |
| golden 来源 | CPU `cpu::ScatterAdd`（`src/array/cpu/segment_reduce.h`），float32 `+=` 累加 |
| 精度差异来源 | 累加顺序（硬件原子调度 vs CPU OpenMP 线程调度），ULP 级，远小于容差阈值 |
| 验收结论 | 42/42 精度用例 PASS（含多核 + 单核，确定性验证通过）。不要求 bit-exact。 |

## 性能特征

| 项目 | 内容 |
|------|------|
| 对比基线 | CPU 回退路径（D2H + `cpu::ScatterAdd` + H2D）端到端时延 |
| 加速比范围 | **17x – 44x**（代表性 shape，中位数口径，详见 `tests/ascend/perf_output/scatter_add/scatter_add_perf.md`） |
| 典型 case | int32×float32，N=4096/dim=128/M=4096：原生 472 µs vs CPU 回退 20833 µs → 44.1x；N=1024/dim=16/M=1：511 µs vs 8878 µs → 17.4x |
| 瓶颈维度 | host dispatch/launch（scalar 占比 ~90-99%），device kernel 计算占比 < 10%。AICore 利用率 64%-97%。 |
| 验收结论 | 性能验收通过（17x-44x 加速比，全部用例不劣于 CPU 回退）。 |

## 支持芯片

| 项目 | 内容 |
|------|------|
| 芯片型号 | Ascend 910B3（dav-2201，c220 微架构） |
| `--npu-arch` | `ascend910b` |
| `NpuArch` | 2201 |
| 仓库默认 SOC_VERSION | Ascend910B4（`CMakeLists.txt`），运行时检测为 910B3 / dav-2201 |
| CANN 版本 | 9.0.0（Ascend-cann-toolkit 9.0.0） |
| 代码架构 | SIMD / MemBase（`kernel_operator.h`），不支持 SIMT / RegBase / Cube（dav-2201 无这些能力，且本算子非 Matmul 类） |

## 修订记录

| 版本 | 日期 | 修改内容 |
|------|------|----------|
| v1.0 | 2026-08-26 | 首版算子文档。接口、参数、约束、dtype、精度、性能、示例齐全。 |
