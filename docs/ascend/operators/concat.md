# concat 算子需求规范与设计文档（Ascend）

> **算子名称**: concat
> **算子分类**: custom（数据搬移型）
> **需求规范确认时间**: 2026-09-23
> **设计文档更新时间**: 2026-09-29
> **说明**: 本文档由需求规范（SPEC）与设计文档（DESIGN）两部分合并而成。第一部分描述算子需求，第二部分描述已落地的 C++ / AscendC 实现（`src/array/ascend/concat*.{h,cc,cpp}` + `src/array/array.cc`）。

---

# 第一部分 需求规范（SPEC）

## 1. 基础信息
- **算子名称**: concat
- **算子分类**: custom  <!-- element-wise / reduction / matmul / attention / custom -->
- **数学公式**: $out[d_0, ..., d_{dim} = \sum_i len_i, ..., d_{n-1}]$，$out[offset_i : offset_i + len_i] = inputs[i]$
- **功能描述**: 沿指定维度 `dim` 拼接一组张量，其余维度 shape 必须一致，各输入 dtype 必须相同。主要用于图分区间特征拼接。

## 2. 关键特性
<!-- 简单拼接算子，无复杂流程性逻辑，省略算法描述 -->

## 3. 算法描述
<!-- 数据搬移型算子，无需分块/在线更新等算法描述 -->

## 4. 数据流图

```
    输入 inputs (N 个)                       输出 y
┌────────────────────┐                 ┌────────────────────┐
│ t0: [d0, x1, x2]   │                 │ [Σd0, x1, x2]     │
│ t1: [d1, x1, x2]   │ ────concat────▶ │  (dim=0)           │
│ ...                │   沿 dim 拼接    │                    │
└────────────────────┘                 └────────────────────┘

参数: dim（拼接轴，支持负数索引）
约束:
  - N ≥ 1
  - 各输入除 dim 外 shape 一致
  - 各输入 dtype / device 一致
```

## 5. 数据规格

**输入规格**:

| 变量 | Shape | Dtype | 动态轴 | 说明 |
|------|-------|-------|--------|------|
| inputs[i] | [..., dim_len_i, ...] | float32 / float16 / 其他 | 所有轴 | 待拼接张量列表，i = 0..N-1；ndim ≥ 1，须连续（封装层 `.contiguous()`） |

**输出规格**:

| 变量 | Shape | Dtype | 动态轴 | 说明 |
|------|-------|-------|--------|------|
| y | [..., Σ dim_len_i, ...] | 同输入 | 所有轴 | 拼接结果，新分配内存 |

## 6. 数据类型支持

| Dtype | AscendC kernel | 通用回退路径 | 精度要求 | 备注 |
|-------|----------------|--------------|----------|------|
| float32 | ✓ | ✓ | 逐位一致 | 默认路径 |
| float16 | ✓ | ✓ | 逐位一致 | host 侧以 `uint16_t` 传递，device 侧解释为 `half` |
| int32 / int64 | ✗ | ✓ | 逐位一致 | 走 `DeviceAPI::CopyDataFromTo` |
| float64 / uint8 / 其他 | ✗ | ✓ | 逐位一致 | 同上 |
| bfloat16 | ✗ | — | — | 非 CUDA 构建下由 `ATEN_FLOAT_TYPE_SWITCH_16BITS` 提前 `LOG(FATAL)` |

## 7. 精度要求
- **atol**: 0
- **rtol**: 0
- **判定方式**: 逐位一致（bit-exact）。concat 是纯数据搬移，不含任何算术运算，NaN/Inf/整型极值均须原样透传，因此不接受混合容差。

## 8. 动态轴说明
- **动态轴**: 所有轴
- **轴含义**: 除 dim 外各轴维度相同，dim 轴维度可变；任一轴允许为 0
- **取值范围**: 输出总元素数 < 2^32（device 侧 `GlobalTensor::operator[]` 的元素偏移为 `uint32_t`，fp32 下即 16 GB）；输入个数 N 无上限

## 9. 边界条件处理
- **零值**: 正常计算
- **极值**: 正常计算
- **NaN/Inf**: 正常透传
- **拼接轴长度为 0**: 输出该轴仍为 0，不发起 kernel
- **非拼接轴长度为 0**: 输出保留该 0 维，段数为 0，不发起 kernel
- **N=1**: 退化为整块拷贝

## 10. 性能要求
- **性能目标**: 首跑精度成功性能的 2 倍

## 11. 参考信息
- **参考实现**: `torch.cat(inputs, dim)`（见 `tests/python/ascend/concat_golden.py`）
- **论文**: 无
- **类似算子**: `torch.stack`（新增维度拼接）
- **仓内参考实现**: `src/array/ascend/spmm.cc`（host launcher 范式）、`sddmm_copy_lhs_kernel.cpp`（纯搬运同步范式）、`scatter_add_kernel.cpp`（UB 预算与多核切分范式）

## 12. 应用场景
- **目标模型**: 通用
- **使用位置**: 特征/结果拼接，图分区间特征拼接

**典型配置**:

| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 说明 |
|----------|------|--------|------|------------|------------|------|
| 功能_P0 | 功能 | P0 | dim=0 | 2×(256, 1024) | (512, 1024) | 核心功能验证 |
| 性能_P0 | 性能 | P0 | dim=-1 | 2×(1024, 512) | (1024, 1024) | 核心性能场景 |
| 多输入_P1 | 功能 | P1 | dim=0, N=4 | 4×(128, 64) | (512, 64) | 多输入拼接 |

---
*确认状态: 已确认（§6/§7/§8 于 2026-09-29 随 C++ 实现落地同步修订）*

---

# 第二部分 设计文档（DESIGN）

> **算子名称**: concat
> **算子分类**: custom（数据搬移型）
> **基于**: 本文档第一部分（需求规范）
> **实现载体**: C++ / AscendC（`kernel_operator.h` + `GlobalTensor` + `TPipe`/`TQue`），非 PyPTO

---

## 1. 概述

### 1.1 功能描述

沿指定维度 `dim` 拼接一组张量，语义与 `torch.cat(arrays, dim)` 逐位一致。

### 1.2 数学公式

$out[..., \sum_{j<i} len_j : \sum_{j \le i} len_j, ...] = inputs[i]$

### 1.3 分段模型：把 N 维拼接降解为一维分段连续搬运

任意 `dim` 的拼接都可以写成「`outer` 个段，每段内把 `len_i * inner` 个连续元素搬到输出的固定位置」：

```
outer      = Π shape[d]   (d < dim)      拼接轴之前的维度乘积 = 段数
len_i      = shape_i[dim]                第 i 个输入在拼接轴上的长度
inner      = Π shape[d]   (d > dim)      拼接轴之后的维度乘积

segElems_i = len_i * inner               第 i 个输入每段的元素数
dstStride  = (Σ_j len_j) * inner         输出中相邻段的步长
dstBase_i  = (Σ_{j<i} len_j) * inner     第 i 个输入在输出段内的起点
```

第 `i` 个输入的第 `seg` 段、第 `k` 个 chunk：

```
源偏移   = seg * segElems_i + k * chunkElems
目的偏移 = seg * dstStride + dstBase_i + k * chunkElems
```

两个退化端点：`dim=0` 时 `outer=1`（整块单段搬运）；`dim=ndim-1` 时 `inner=1`（段最短、段数最多）。

### 1.4 数据流图

```
host (array.cc)                      device (concat_kernel.cpp)
───────────────                      ─────────────────────────
校验 shape/dtype/ctx/dim
  │
NDArray::Empty(out_shape)
  │
  ├─ kDGLAscend + fp32/fp16 + 32B 对齐
  │     │
  │     └─ impl::ConcatND<kDGLAscend, T>  ──┐
  │        (concat.cc)                      │  每个输入一次 launch：
  │        计算 N 份 ConcatTilingData       │    MTE2: src GM → UB (DataCopyPad)
  │        1× aclrtMalloc + 1× H2D memcpy ──┼─▶  MTE3: UB → dst GM (DataCopyPad)
  │        N× aclrtlaunch_concat_kernel_*   │    核 b 处理 chunk [b*cpc, ...)
  │        1× aclrtSynchronizeStream        │
  │        1× aclrtFree                   ──┘
  │
  └─ 其他 device / dtype / 非对齐
        └─ outer × N 次 DeviceAPI::CopyDataFromTo（按字节偏移）
```

---

## 2. 接口原型与调用链

### 2.1 对外 C++ API

```cpp
// include/dgl/aten/array_ops.h:258
// 命名为 ConcatND 而非重载 Concat：types.h 中 `typedef NDArray IdArray`，
// 重载 Concat(const std::vector<IdArray>&) 会构成重复定义。
NDArray ConcatND(const std::vector<NDArray>& arrays, int64_t dim);
```

### 2.2 内部模板声明与 Ascend 特化

```cpp
// src/array/kernel_decl.h:144（纯数据搬移，无 IdType 维度）
template <int XPU, typename DType>
void ConcatND(const std::vector<NDArray>& arrays, NDArray out, int64_t dim);

// src/array/ascend/concat.cc
template <> void ConcatND<kDGLAscend, float>(
    const std::vector<NDArray>& arrays, NDArray out, int64_t dim);
template <> void ConcatND<kDGLAscend, uint16_t>(   // fp16
    const std::vector<NDArray>& arrays, NDArray out, int64_t dim);
```

### 2.3 device kernel 入口

```cpp
// src/array/ascend/concat_kernel.cpp
extern "C" __global__ __aicore__ void concat_kernel_f32(
    GM_ADDR src, GM_ADDR dst, GM_ADDR tiling);
extern "C" __global__ __aicore__ void concat_kernel_f16(
    GM_ADDR src, GM_ADDR dst, GM_ADDR tiling);
```

### 2.4 host launcher 符号（`ACLRT_LAUNCH_KERNEL` 宏展开）

```cpp
extern "C" uint32_t aclrtlaunch_concat_kernel_f32(
    uint32_t blockDim, aclrtStream stream,
    void* src, void* dst, void* tiling);
extern "C" uint32_t aclrtlaunch_concat_kernel_f16(
    uint32_t blockDim, aclrtStream stream,
    void* src, void* dst, void* tiling);
```

### 2.5 FFI 与 Python 通路

```cpp
// src/array/array.cc:1808
DGL_REGISTER_GLOBAL("ndarray._CAPI_DGLConcatND")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      List<Value> arrays = args[0];
      const int64_t dim = args[1];
      *rv = ConcatND(ListValueToVector<NDArray>(arrays), dim);
    });
```

```python
# tests/python/ascend/concat_impl.py
from dgl.backend import F
from dgl.ndarray import _CAPI_DGLConcatND

def concat_impl(tensors, dim=0):
    nds = [F.zerocopy_to_dgl_ndarray(t) for t in tensors]
    return F.zerocopy_from_dgl_ndarray(_CAPI_DGLConcatND(nds, dim))
```

torch_npu 张量经 DLPack 导出为 `kDLExtDev = 12`，`src/runtime/dlpack_convert.cc:29` 的 `static_cast<DGLDeviceType>` 恰好落到 `kDGLAscend = 12`；DLPack 的 `byte_offset` 恒为 0，故 `NDArray->data` 就是切片真实基址。

---

## 3. 参数说明

| 参数 | 内存位置 | 方向 | 类型 | 说明 |
|------|----------|------|------|------|
| src | Device (GM) | 输入 | `T*` | 单个输入张量的连续数据区，元素数 `segCount * segElems`。每次 launch 只处理一个输入。 |
| dst | Device (GM) | 输出 | `T*` | 输出张量数据区，由 host 侧 `NDArray::Empty(out_shape)` 预先分配；Ascend 分配器返回 512B 对齐地址。 |
| tiling | Device (GM) | 输入（host 下发） | `ConcatTilingData*` | 分段/切分参数，host 侧一次 `aclrtMalloc` + 一次 `aclrtMemcpy(HOST_TO_DEVICE)` 下发整个数组，第 i 次 launch 传 `base + i * sizeof(ConcatTilingData)`。 |

### 3.1 Tiling 结构体

```cpp
// src/array/ascend/concat_tiling.h —— host (.cc) 与 kernel (.cpp) 共用，纯 C/C++ 语法
struct ConcatTilingData {
  uint32_t segCount;      // outer：本输入被切成的连续段数
  uint32_t segElems;      // len_i * inner：每段元素数
  uint32_t dstStride;     // (Σ_j len_j) * inner：输出中相邻段的步长
  uint32_t dstBase;       // (Σ_{j<i} len_j) * inner：本输入在输出段内的起点
  uint32_t chunkElems;    // 单次 MTE2/MTE3 搬运的元素数
  uint32_t chunksPerSeg;  // ceil(segElems / chunkElems)
  uint32_t totalChunks;   // segCount * chunksPerSeg
  uint32_t chunksPerCore; // ceil(totalChunks / blockDim)
  uint32_t blockDim;      // host 侧记录本次 launch 的核数，kernel 不使用
};
```

### 3.2 UB 预算常量

```cpp
constexpr uint32_t CONCAT_UB_TOTAL_SIZE = 192 * 1024;   // 对齐 scatter_add_kernel.cpp
constexpr uint32_t CONCAT_UB_RESERVED   = 2 * 1024;     // 栈帧开销预留
constexpr uint32_t CONCAT_BUFFER_NUM    = 2;            // 双缓冲
constexpr uint32_t CONCAT_ALIGN_BYTES   = 32;           // DataCopyPad 对齐要求
// ((192K - 2K) / 2) 向下对齐到 32B = 97280 B
constexpr uint32_t CONCAT_CHUNK_BYTES   = 97280;
```

`chunkElems = CONCAT_CHUNK_BYTES / sizeof(T)`：fp32 为 24320，fp16 为 48640。UB 容量由 host 侧推导后经 tiling 下发，kernel 不硬编码 `UB_SIZE`（与 `sddmm_copy_lhs_tiling.h` 的约定一致）。

---

## 4. Tiling 与多核切分策略

### 4.1 算子类型

- **类型**: Vector / 纯数据搬移（`KERNEL_TYPE_AIV_ONLY`），无 Cube、无 V pipeline 算术。
- **判断依据**: 输出每个元素直接来自某个输入元素，不做任何计算，瓶颈在 MTE2/MTE3 带宽。

### 4.2 以 chunk 为粒度全局切分

```
totalChunks   = segCount * chunksPerSeg
blockDim      = max(1, min(totalChunks, vectorCoreNum))
chunksPerCore = ceil(totalChunks / blockDim)
核 b 处理 chunk 区间 [b * chunksPerCore, min((b+1) * chunksPerCore, totalChunks))
空闲核早退守卫：startChunk >= endChunk 时直接 return
```

`vectorCoreNum` 由 `aclrtGetDeviceInfo(device_id, ACL_DEV_ATTR_VECTOR_CORE_NUM, &n)` 运行时查询（A2/A3 均为 48），查询失败或返回 0 时回落到默认值 48。

**为什么按 chunk 而不是按 seg 切分**：`dim=0` 时 `segCount = 1`，按 seg 切分只能用 1 个核；按 chunk 切分则单段也会被拆到所有核上，保证 功能_P0（dim=0，单段 262144 元素 → 11 chunk → 11 核）同样吃满算力。反之 性能_P0（dim=-1，1024 段 × 1 chunk → `blockDim=48`、`chunksPerCore=22`）走的是每核串行多段的分支。

### 4.3 launch 策略

AscendC kernel 形参是定长 `GM_ADDR` 列表，device 侧不便持有指针数组，故**每个输入一次 launch**。为避免 N 次同步开销，host 侧：

1. 计算 N 份 `ConcatTilingData`（`std::vector`）；
2. **一次** `aclrtMalloc(ACL_MEM_MALLOC_HUGE_FIRST)` + **一次** `aclrtMemcpy(HOST_TO_DEVICE)` 下发整个数组；
3. N 次 launch，`totalChunks == 0` 的输入直接 `continue`（空输入不发起 kernel）；
4. **一次** `aclrtSynchronizeStream(stream)` + **一次** `aclrtFree`。

launch 失败时先 `aclrtFree(tiling_dev)` 再 `LOG(FATAL)`，避免 GM 泄漏。

### 4.4 流序

stream 取自 `dgl::runtime::getCurrentAscendStream()`（与 `index_select.cc` / `binary_elewise.cc` / `as_num_bits.cc` 一致），使 kernel 与同一 stream 上待执行的 torch_npu 算子保持顺序，无需 `aclrtSynchronizeDevice()` 全设备排空。device 侧先 `aclrtSetDevice(ctx.device_id)`。

---

## 5. Kernel 流水线与同步设计

### 5.1 数据流

```
for c in [startChunk, endChunk):
    (seg, k) = divmod(c, chunksPerSeg)
    len   = min(segElems - k*chunkElems, chunkElems)      # 尾块自然变短
    bytes = len * sizeof(T)
    srcOff = seg*segElems  + k*chunkElems                 # uint64 计算后窄化
    dstOff = seg*dstStride + dstBase + k*chunkElems

    if c > startChunk:                                    # buffer 复用安全
        SetFlag<MTE3_MTE2>(evt); WaitFlag<MTE3_MTE2>(evt)

    buf = bufQue.AllocTensor<T>()
    DataCopyPad(buf, srcGm[srcOff], {1, bytes, 0, 0, 0}, padIn)   # MTE2
    bufQue.EnQue(buf)

    ready = bufQue.DeQue<T>()
    SetFlag<MTE2_MTE3>(evt); WaitFlag<MTE2_MTE3>(evt)             # 数据就绪

    DataCopyPad(dstGm[dstOff], ready, {1, bytes, 0, 0, 0})        # MTE3
    bufQue.FreeTensor(ready)
```

### 5.2 同步要点

| 要点 | 做法 | 依据 |
|------|------|------|
| buffer 轮转 | `TQue<TPosition::VECIN, 2>` + `AllocTensor/EnQue/DeQue/FreeTensor` | MTE 缓冲区用 TQue 管理（AscendC API 最佳实践） |
| MTE2 → MTE3 数据就绪 | `SetFlag/WaitFlag<HardEvent::MTE2_MTE3>`，immediate Set→Wait | `TQue` 的 EnQue/DeQue 只同步 MTE2→V，**不覆盖** MTE2→MTE3 直接依赖 |
| MTE3 → MTE2 buffer 复用 | `SetFlag/WaitFlag<HardEvent::MTE3_MTE2>`，immediate Set→Wait | MTE3 顺序执行，等到上一轮即保证更早各轮均已完成 |
| Event ID | `pipe.AllocEventID<HardEvent::X>()` / `ReleaseEventID` | 唯一、持久，避免硬编码 ID 冲突 |
| 全局屏障 | **不使用** `PipeBarrier<PIPE_ALL>` | 仅调试用，性能差（对齐 `sddmm_copy_lhs_kernel.cpp`） |
| 尾块与非对齐 | `DataCopyPad` 统一处理 | 仓内既有先例：`binary_elewise_kernel.cpp:65`、`csr_get_row_nnz_kernel.cpp:59`、`coo2csr_kernel.cpp:139` 均以非 32B 整数倍的 `blockLen` 下发 MTE3 |

### 5.3 API 黑名单合规

不使用 `GlobalTensor::GetValue/SetValue` 承载生产数据流量，所有数据走 UB + `DataCopyPad`。

---

## 6. dtype 支持与回退路径

`ConcatND` 在**所有** device / dtype 上都正确，AscendC kernel 只是快路径。分派逻辑（`src/array/array.cc:430-465`）：

```cpp
#ifdef DGL_USE_ASCEND
  if (first->ctx.device_type == kDGLAscend && AscendDataCopyAligned(arrays, ret)) {
    if (IS_FLOAT32(first)) { impl::ConcatND<kDGLAscend, float>(arrays, ret, axis); return ret; }
    if (first->dtype.code == kDGLFloat && first->dtype.bits == 16) {
      impl::ConcatND<kDGLAscend, uint16_t>(arrays, ret, axis); return ret;   // fp16
    }
  }
#endif
  // 通用路径：每个 (输入, outer 段) 一次按字节偏移的 device copy
```

| 条件 | 路径 | 说明 |
|------|------|------|
| Ascend + fp32 + 全部指针 32B 对齐 | AscendC kernel | `concat_kernel_f32` |
| Ascend + fp16 + 全部指针 32B 对齐 | AscendC kernel | `concat_kernel_f16`，host 侧以 `uint16_t` 传递（`ATEN_FLOAT_TYPE_SWITCH_16BITS` 在非 CUDA 构建下把 Ascend+16bit 映射为 `uint16_t`），kernel 内解释为 `half`，同为 2 字节 |
| Ascend + int32/int64/float64/其他 | 通用路径 | `DeviceAPI::CopyDataFromTo`，`outer × N` 次 |
| Ascend + 任一指针非 32B 对齐 | 通用路径 | 见 §6.1 |
| CPU / CUDA | 通用路径 | 同一份代码覆盖所有后端 |
| 输出拼接轴长度为 0 | 直接返回 | `NDArray::Empty` 后 `return ret`，不发起任何拷贝 |

### 6.1 32B 对齐守卫

`DataCopyPad` 要求 GM 地址 32B 对齐。DGL 自身分配的 buffer 恒为 512B 对齐（`ascend_device_api.cc:118-149` 的 `aclrtMalloc(nbytes + 512)` 返回 `base + 512`），但 **DLPack 导入的张量可能是更大分配的一个切片**——`ATen/dlpack.h` 明确说明各库并不遵守 256B 对齐约定，因此导入指针可能只有元素级对齐。

```cpp
// src/array/array.cc:383
bool AscendDataCopyAligned(const std::vector<NDArray>& arrays, const NDArray& out) {
  auto aligned = [](const void* p) {
    return (reinterpret_cast<uintptr_t>(p) & (CONCAT_ALIGN_BYTES - 1)) == 0;
  };
  if (!aligned(out->data)) return false;
  for (const NDArray& a : arrays) if (!aligned(a->data)) return false;
  return true;
}
```

不满足时整条输入列表回退通用路径（`CopyDataFromTo` 无对齐要求）。

---

## 7. 约束与边界条件

### 7.1 Shape 约束

- `arrays` 非空（`CHECK(!arrays.empty())`）。
- `ndim >= 1`；`dim` 支持负数索引，归一化后须落在 `[0, ndim)`。
- 各输入 `ndim` 相同（`CHECK_NDIM`），除 `dim` 外逐维相等（`CHECK_EQ`），`dim` 轴长度累加进 `out_shape[axis]`。
- 各输入 `dtype` 相同（`CHECK_SAME_DTYPE`）、`ctx` 相同（`CHECK_SAME_CONTEXT`）。
- 输出总元素数 `< 2^32`（`CHECK_LT`），受 `GlobalTensor::operator[]` 的 `uint32_t` 偏移限制。
- **不校验空张量**：`IsNullArray` 的语义是 `shape[0] == 0`（判空而非判空指针），若沿用会错误拒绝 `(0,64) + (0,64)` 沿 `dim=1` 拼接这类合法调用。空输入由三处守卫处理：`out_shape[axis] == 0` 提前返回、通用路径 `seg_bytes > 0`、kernel 路径 `totalChunks == 0 → continue`。

### 7.2 平台约束

| 项目 | 内容 |
|------|------|
| 芯片型号 | Ascend 910B（dav-2201，c220 微架构） |
| 仓库默认 SOC_VERSION | `ascend910b4`（`CMakeLists.txt`） |
| 代码架构 | SIMD / MemBase（`kernel_operator.h`），纯 Vector 任务，不涉及 Cube / SIMT / RegBase |
| 编译开关 | `USE_ASCEND=ON` → 定义 `DGL_USE_ASCEND` |
| CANN 版本 | ≥ 8.5.0（配 torch==2.8.0 / torch_npu==2.8.0.post2） |

---

## 8. 验证方案

### 8.1 Golden

```python
# tests/python/ascend/concat_golden.py —— 纯 torch，禁止引入 pypto
def concat_golden(tensors, dim=0):
    return torch.cat(tensors, dim=dim)
```

### 8.2 判定标准

`torch.testing.assert_close(got.cpu(), want.cpu(), rtol=0, atol=0, equal_nan=True)`，即**逐位一致**；`equal_nan=True` 使 NaN 透传用例也可比较。同时校验 `shape` / `dtype` / `device.type`。

### 8.3 测试覆盖矩阵（`tests/python/ascend/test_concat.py`）

| 测试类 | 覆盖点 |
|--------|--------|
| `TestConcatTypical` | 规格典型配置 功能_P0 / 性能_P0 / 多输入_P1 |
| `TestConcatDims` | 1D~4D、各 dim（含负数）、中间轴非等长、`inner=1` 短段多段 |
| `TestConcatInputCount` | N = 1 / 2 / 3 / 48（等于核数）/ 128 |
| `TestConcatChunking` | 单段跨多 chunk（fp32/fp16）、恰为 chunk 边界（无尾块）、chunk+1（尾块 1 元素）、段数 4096 ≫ 48 核、`totalChunks=1 → blockDim=1` |
| `TestConcatDtype` | fp32/fp16 走 kernel；int32/int64/float64 走通用路径；fp16 与 fp32 同数据一致性 |
| `TestConcatValues` | 大值 1e30、小值 1e-30、零值、NaN/Inf 逐位透传（2 个 NaN + 4 个 Inf）、int64 极值 |
| `TestConcatEdgeCases` | 拼接轴为 0、空与非空混合、非拼接轴为 0、`(1,1,1)` 最小形状、非 32B 对齐切片（触发回退）、非连续张量、32B 对齐连续切片（触发 kernel） |
| `TestConcatCpu` | CPU 设备走通用路径，与 NPU 结果一致 |
| `TestConcatStability` | 重复调用无漂移、与其他 NPU 算子交替、流序（前置异步写可见）、输出内存独立于输入 |
| `TestConcatErrors` | 空列表、dim 越界（正/负）、shape 不匹配、dtype 不匹配、device 不匹配 → `dgl.DGLError` |

对齐前提由用例自身断言：`randn(64,100)` fp32 行 400 B ⇒ `full[1:5].data_ptr() % 32 == 16`（走回退）；`randn(64,128)` fp32 行 512 B ⇒ `full[16:32].data_ptr() % 32 == 0`（走 kernel）。

### 8.4 运行环境

本机（Windows x86_64 / CPU torch）**无法构建与运行**，须在远端 Linux aarch64 + Ascend 910B + CANN ≥ 8.5.0 环境执行：

```bash
pytest tests/python/ascend/test_concat.py -v
```

`tests/python/ascend/` 下无 `conftest.py` / `__init__.py`，pytest 的 prepend 导入模式使 `from concat_golden import concat_golden`、`from concat_impl import concat_impl` 可直接解析。

---

## 9. 风险点与注意事项

| 风险 / 错误 | 触发场景 | 影响 / 原因 | 规避方法 |
|-------------|----------|-------------|----------|
| GM 地址非 32B 对齐 | DLPack 导入的切片张量 | `DataCopyPad` 静默错搬 | `AscendDataCopyAligned` 守卫，不满足即回退通用路径（§6.1） |
| 元素偏移溢出 `uint32_t` | 输出 ≥ 2^32 元素（fp32 ≥ 16 GB） | `GlobalTensor::operator[]` 截断 | host 侧 `CHECK_LT(out_numel, 1LL << 32)` |
| 缺少 MTE2↔MTE3 同步 | 只依赖 `TQue` 的 EnQue/DeQue | EnQue/DeQue 只同步 MTE2→V，MTE3 可能读到未就绪数据 | 显式 `SetFlag/WaitFlag<MTE2_MTE3>` + `<MTE3_MTE2>`（§5.2） |
| 尾块向上取整到 32B | 想把非对齐尾块凑整 | Ascend 分配器的 512B 余量在返回指针**之前**，凑整会越界写 | 依赖 `DataCopyPad` 的非整块 `blockLen` 语义（仓内既有先例） |
| 用 `IsNullArray` 判空 | 想拒绝"空"输入 | 其语义是 `shape[0] == 0`，会误拒 `dim=1` 拼接 `(0,64)` 的合法调用 | 不做空张量校验，靠三处 0 长度守卫（§7.1） |
| 重载 `aten::Concat` | 想加 `dim` 参数版本 | `typedef NDArray IdArray` ⇒ 与既有 `Concat(const std::vector<IdArray>&)` 重复定义 | 新 API 命名为 `ConcatND` |
| N 次 `aclrtSynchronizeStream` | 每个输入各同步一次 | host 侧同步开销随 N 线性放大（N=128 时显著） | 一次 malloc/memcpy + N 次 launch + 一次 sync + 一次 free（§4.3） |
| 新建 `src/array/ascend/concat/` 子目录 | 想按算子分目录 | CMake 两处 glob 均非递归，子目录内文件不会被编译 | 保持扁平：`concat.cc` / `concat_kernel.cpp` / `concat_tiling.h` |
| bfloat16 | Ascend + bf16 输入 | 非 CUDA 构建下 `ATEN_FLOAT_TYPE_SWITCH_16BITS` 提前 `LOG(FATAL)` | 无需 bf16 特化，与 `scatter_add.cc` 一致 |
