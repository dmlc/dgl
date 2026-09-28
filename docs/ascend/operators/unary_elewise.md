# UnaryElewise（Neg）

> Ascend NPU 算子文档。

## 功能描述

UnaryElewise 是 DGL Ascend 后端的一元逐元素（elementwise）算子族，首个实例为 `arith::Neg`（即 aten 层 `Neg`），数学语义为：

```
对每个 i ∈ [0, n):
    y[i] = -x[i]
```

即对 1-D 整型数组的每个元素取负（补码语义）。输入 `x` 与输出 `y` 均为 `IdArray`（1-D，shape `[n]`），dtype 同为 int32 或 int64，位于同一设备上下文。

该算子为 **DGL C++ aten 内部算子**（上游 DGL 惯例，无 Python / C API 对外暴露），经 `operator-(const NDArray&)` 运算符重载被图算法内部使用。本次开发补齐 `kDGLAscend` 设备路径（原先 Ascend 设备调用 `Neg` 直接 `LOG(FATAL)`）；CPU / CUDA 既有实现与路径不变。host 侧分发位于 `src/array/array_arith.cc`（`UNARY_ELEMENT_OP` 宏：`ctx == kDGLAscend` 时进入 Ascend 特化 `impl::UnaryElewise<kDGLAscend, IdType, arith::Neg>`，其余设备走 `ATEN_XPU_SWITCH_CUDA` 原有分支）。

## 接口原型

```cpp
// aten 公开层（include/dgl/aten/array_ops.h:128，上游既有声明）
IdArray Neg(IdArray array);

// impl 模板层（src/array/array_op.h:39，上游既有声明；本次新增 kDGLAscend 特化）
template <DGLDeviceType XPU, typename IdType, typename Op>
IdArray UnaryElewise(IdArray lhs);
// 显式实例化：UnaryElewise<kDGLAscend, int32_t, arith::Neg>
//            UnaryElewise<kDGLAscend, int64_t, arith::Neg>

// C++ 运算符重载（src/array/array_arith.cc:237，上游既有）
NDArray operator-(const NDArray& array);  // → dgl::aten::Neg(array)

// device kernel 入口（src/array/ascend/unary_elewise_kernel.cpp）
extern "C" __global__ __aicore__ void unary_neg_i32(
    GM_ADDR x, GM_ADDR y, GM_ADDR tiling_ptr);

// device kernel 入口（src/array/ascend/unary_elewise_i64_kernel.cpp）
extern "C" __global__ __aicore__ void unary_neg_i64(
    GM_ADDR x, GM_ADDR y, GM_ADDR tiling_ptr);

// host launcher（src/array/ascend/unary_elewise.cc，ACLRT_LAUNCH_KERNEL 宏展开）
extern "C" uint32_t aclrtlaunch_unary_neg_i32(
    uint32_t blockDim, aclrtStream stream, void* x, void* y, void* tiling);
extern "C" uint32_t aclrtlaunch_unary_neg_i64(
    uint32_t blockDim, aclrtStream stream, void* x, void* y, void* tiling);
```

> 无 Python 上层通路：全仓 `aten::Neg` 仅经 `operator-(NDArray)` 被 C++ 内部调用（分发宏 `UNARY_ELEMENT_OP(Neg, Neg)`，src/array/array_arith.cc:186）。

## 参数说明

| 参数 | 内存位置 | 方向 | 类型 | 说明 |
|------|----------|------|------|------|
| x（array） | Device (GM)（随输入 ctx） | 输入 | IdArray | 输入数组，1-D，shape `[n]`，dtype int32/int64；`n = x->shape[0]`。 |
| y（返回值） | Device (GM)（同 x 的 ctx） | 输出 | IdArray | 输出数组，1-D，shape `[n]`，dtype 与 ctx 同 x；逐元素 `y[i] = -x[i]`（补码语义，与 CPU golden `arith::Neg::Call` 逐位一致）。 |
| tiling | Device (GM) | 输入（host 下发） | `TilingData*` | 形状参数，仅 `{ uint32_t n }`；由 host `aclrtMalloc` + `aclrtMemcpy` 下发，kernel 侧读取，其余切分参数在核内由 `n` 与 `GetBlockNum()` 确定性推导（见「实现要点」）。 |

### Tiling 结构体

```cpp
struct TilingData {
    uint32_t n;  // 元素数 = x->shape[0]（host 侧 CHECK_LE(n, UINT32_MAX) 保证不截断）
};
```

## 支持数据类型

| 数据类型 | 支持状态 | 说明 |
|----------|----------|------|
| int32 | 支持 | 向量化路径 kernel `unary_neg_i32`（`Muls(-1)`）。 |
| int64 | 支持 | 位级向量化路径 kernel `unary_neg_i64`（2×int32 lane 视角 + Rare-tile 精确向量化路径）。 |
| 其他 dtype（float/int16/uint8/...） | 不支持 | 分发宏 `ATEN_ID_TYPE_SWITCH` 仅接受 int32/int64，其余 dtype 到达分发即 `LOG(FATAL)`，不进入本算子（与 BinaryElewise Ascend 分发一致）。 |

## 约束说明

### Shape 约束

- 输入必须为 1-D（IdArray 语义），`n = x->shape[0]` 任意非负整数。
- **`n ≤ UINT32_MAX`**：tiling 元素计数为 uint32，host 侧 `CHECK_LE(len, UINT32_MAX)` 显式拒绝并报错 `"UnaryElewise: element count exceeds the uint32 tiling limit"`（src/array/ascend/unary_elewise.cc:120）。超界输入当前不支持。
- **`n = 0`**：host 侧提前返回同 shape 空 IdArray，不发射 kernel（空操作，语义正确）。

### dtype 约束

- 仅支持 int32 / int64（见「支持数据类型」表）；整型取负为精确整数运算，无舍入链。

### 数值语义约束（bit-exact）

- **int32**：`Muls(y, x, -1, cnt)` 为模 2^32 整型乘法，`x × (-1) ≡ -x (mod 2^32)` 全域成立，含 `INT32_MIN`（(-2^31)×(-1) wrap 回 -2^31），与 CPU 补码 `-t1` 逐位一致。
- **int64**：DAV_2201 无 int64 向量算术（`Muls` static_assert 编译期硬报错），kernel 采用 2×int32 lane 位级补码取负；`lo == 0` 等触发 Rare-tile 的片（含 `INT64_MIN` 所在片）走 rare-tile 精确向量化重算路径（见「实现要点」）。**所有 int64 值（含 INT64_MIN）结果与 CPU golden 逐位一致**——位级路径与 rare-tile 路径均为精确整数运算，bit-exact 保持。

### 平台约束

- 目标芯片：Ascend 910B3（dav-2201，`--npu-arch=ascend910b`）。
- 代码架构：SIMD（实现载体 MemBase，`kernel_operator.h` + `GlobalTensor` + `TPipe`/`TQue`），纯 Vector 任务（`KERNEL_TYPE_AIV_ONLY`）。
- 双 kernel 分置两个翻译单元（i32 / i64 各一个 `*_kernel.cpp`）：CANN 8.5 工具链对设置任务类型的多入口 device blob 存在 `.ascend.meta` 注册编号错位问题（整包注册被拒），单入口 blob 注册正常，故两入口不可合并。

## 返回值 / 错误码

| 返回值 | 含义 |
|--------|------|
| `IdArray`（成功路径） | 与输入同 shape / dtype / ctx 的输出数组，逐元素 `y[i] = -x[i]`。 |
| `n > UINT32_MAX` | `CHECK_LE` 失败，`LOG(FATAL)`：`UnaryElewise: element count exceeds the uint32 tiling limit`。 |
| dtype 非 int32/int64 | 分发宏 `ATEN_ID_TYPE_SWITCH` `LOG(FATAL)`（未进入本算子）。 |
| ctx 非 Ascend 设备 | `CHECK` 失败：`Expected Ascend device context`。 |
| `aclrtSetDevice` / `aclrtMalloc` / `aclrtMemcpy` / `aclrtSynchronizeStream` / `aclrtFree` 失败 | `ASCEND_CALL` 的 `CHECK` 宏终止，输出 `Ascend Error, code: <e>`。 |
| `aclrtlaunch_unary_neg_*` 返回非 `ACL_SUCCESS` | `ASCEND_CALL` 的 `CHECK` 宏终止，输出 `Ascend Error, code: <e>`。 |
| 未编译 Ascend 支持 | `LOG(FATAL)`：`Ascend support is not compiled.`。 |

## 调用示例

### C++ 直接调用（aten 层）

```cpp
#include <dgl/array.h>

using dgl::runtime::NDArray;
using dgl::aten::Neg;

// x: IdArray shape [n], dtype int32/int64, ctx = kDGLAscend（NPU）
dgl::aten::IdArray y = Neg(x);          // y[i] = -x[i]，bit-exact
// 或经运算符重载（内部同路径）：
// dgl::runtime::NDArray y = -x_ndarray;
```

### C++ 内部使用（DGL 图算法通路）

```cpp
// 上游 DGL 惯例：aten 内部算子经 operator- 被图算法间接使用，无 Python 暴露。
// 例（NDArray 层）：
dgl::runtime::NDArray x = ...;  // int32/int64, NPU ctx
dgl::runtime::NDArray y = -x;   // → dgl::aten::Neg → impl::UnaryElewise<kDGLAscend, ...>
```

## 实现要点

> 本节为使用者提供实现概览，便于理解性能与数值语义特征。实现细节以 kernel 源码注释为权威。

- **Tiling 与多核切分**：host tiling 仅 `{uint32 n}`；kernel 侧 `UnaryBlockRange`（src/array/ascend/unary_elewise_tiling.h）以 `n` + `GetBlockNum()` 推导 512 元素对齐的 `blockFormer`（512×4B=2048B，保证向量化路径 GM 段首 32B 对齐），前核等量、尾核取余，`start ≥ n` 的核空转——**正确性与发射核数解耦**。host 侧 `CalcBlockDim` 查询 `ACL_DEV_ATTR_VECTOR_CORE_NUM`（失败回退 8，range.cc 先例），按每核 ≥1024 元素门控发射 `min(ceil(n/1024), cores)` 个核。
- **int32 向量化路径**（unary_neg_i32）：GM→UB（`DataCopy` 主块 / `DataCopyPad` 尾块）→ `Muls(dst, src, -1, cnt)` → UB→GM，`TQue<VECIN/VECOUT, 2>` 双缓冲流水；tile 8192 元素（32KB），总 UB 4×32KB=128KB ≤ DAV_2201 UB 192KB。
- **int64 位级向量化路径**（unary_neg_i64）：GM int64 流按 `GlobalTensor<int32_t>` 2n lane 视角寻址（小端：lane 2i = lo_i，lane 2i+1 = hi_i）。静态构建 `[0,-1,0,-1,...]` 奇偶 borrow 模式（`CreateVecIndex → Duplicate(1) → 两次 1024-lane And 分块 → Muls(-1)`；**分块规避 dav_c220 Level-2 `And` 硬编码 int16-lane 计数的工具链陷阱**——其他 kernel 引入位运算时同样需分块或显式 Level-0 调用）。热路径每片一条 `Sub(t, borrow, s)` 同时完成 `lo' = -lo`、`hi' = -hi - 1`；每片先做输入探针（Cast f32 + Abs + ReduceMin），`min|lane| == 0` 判片内含 0 lane（随机数据概率 ≈2⁻³²/元素），探针结果经 `SetFlag/WaitFlag<HardEvent::V_S>` 定向事件对同步到标量侧读取（仅同步「reduce 结果 → 标量读」这一条依赖，不排空整个 V pipe）。**仅命中片走 rare-tile 精确向量化重算**（全程 blockCount=1 连续 MTE 拷贝、无标量 GM 访问；rare 分支内以 `MTE2_V` / `V_MTE3` / `MTE3_MTE2` 定向事件对衔接搬入→计算→搬出及 WAR 守卫）：元素 0 两条单 lane（4B）拷入，V-pipe 精确计算 `lo' = -lo`、`b = [lo≠0]`（对含 `INT32_MIN` 的全部 int32 精确）、`hi' = -hi - b` 后单 lane 拷出；子 tile（元素 1..cnt-1）两条连续拷贝搬入（tile 本体 + 自 GM lane 1 的一 lane 前移视图 `tmpRSub[k] = s[k+1]`），构造校正项 `corr = [tmpRSub == 0] × parityOdd`（仅 hi lane 生效：`lo == 0` 时 `-hi - 1 + 1 = -hi`）后一条连续拷出。UB 标量修补分支已实测触发 ACL 507035 vector core exception，弃用；2D strided `DataCopyPad` 在 dav_c220 实测静默丢块，亦弃用。i64 路径 UB 预算 ~104KB（rare-tile 新增 rareBuf 16KB + subBuf 24KB）。
- **host 侧资源与设备状态守卫**：`AscendUnaryResourceGuard`（RAII）覆盖 `UnaryElewise` 所有退出路径——构造时保存线程当前 Ascend 设备、析构时恢复，`aclrtSetDevice` 的设备切换不外泄（多卡/多线程安全）；`tiling_dev` 在异常退出路径（`ASCEND_CALL` 的 `CHECK` 抛出）由析构兜底释放（正常路径仍带 CHECK 释放并置空守卫指针，避免双释放）；kernel 启动返回值经 `ASCEND_CALL` 校验，launch 失败即 `CHECK` 报错。守卫清理均为 best-effort、析构不抛异常，不遮蔽主流程错误。
- **同步语义**：host 侧 launch 后 `aclrtSynchronizeStream` 同步再释放 tiling（与 binary_elewise.cc 同款直调范式）。

## 精度标准

| 项目 | 内容 |
|------|------|
| 输出 dtype | int32 / int64 |
| 判定方式 | **bit-exact**（整型类别标准）：逐元素逐位相等，不一致数 = 0 |
| 容差 | 无（整型精确运算，无舍入链，不适用混合容差） |
| golden 来源 | CPU `arith::Neg::Call`（src/array/cpu/array_op_impl.cc:229-238，`-t1` 补码取负） |
| 边界闭环 | INT32_MIN / INT64_MIN wrap（L2-03/L2-04 上板用例）、全零输入触发 i64 rare-tile 路径（L1-13，0/2048 探针命中）均 bit-exact |
| 验收结论 | gtest 59/59 PASSED（bit-exact mismatch=0）。 |

## 性能特征

> 需求无硬性能门槛，按「基线对照 + 瓶颈维度分解 + 优化路径穷尽」评估，结论：**通过**。

| 项目 | 内容 |
|------|------|
| 对比基线 | 未向量化/标量实现，host e2e 中位数口径 |
| i32 典型 case | n=2^22：e2e 103.48µs / 324.26GB/s，设备侧 kernel 34.02µs（较基线 e2e 带宽 +48%） |
| i64 典型 case | n=2^20：e2e 113.02µs / 148.44GB/s（基线 1861.5µs / 9.01GB/s，**e2e 16.5×**），设备侧 kernel 19.5~19.7µs（**>100×**）；n=2^22 设备侧 88.3~102.38µs（基线 ~8.4ms，**~82×**） |
| 瓶颈维度 | 大 n 实测锁定 **MTE 搬运主导**（i32 2^22：搬入 56.3% / 搬出 60.0%，计算仅 13.7%）；访存受限型 |
| 小 n 特征 | n ≤ 4096 时处于 ~100µs host 启动地板（dispatch/tiling/launch 固定开销，与基线同量级），非 kernel 效应 |
| ⚠ 平台级提示 | **host 分配悬崖**：ACL 驱动在输出 >~16MB 处分配/释放成本悬崖（32MB 分配+释放 ≈23ms，`AllocDataSpace` 无缓存直通 `aclrtMalloc/Free`），与算子实现无关，影响所有大输出算子的 e2e 口径（例：i64 2^22 e2e 24.6ms 而设备侧 kernel 仅 ~89-102µs）。**大 n 性能评估请以 msprof 设备侧 kernel 时长为权威口径**；分配器优化（缓存/内存池）属运行时平台层事项 |

## 测试

| 项目 | 内容 |
|------|------|
| 测试目录 | `tests/cpp/`：`test_unary_elewise.cc`（黑盒 + FatalMsg + Cross）、`test_unary_elewise_whitebox.cc`（白盒）、`unary_elewise_test_util.h`（golden/校验/注入工具） |
| 用例规模 | **59 条** = 黑盒 20（L0 基线 2 + L1 边界/大 n 13 + L2 空/极值/压力 5）+ FatalMsg 3（float32/float64/int16 dtype 拒绝路径）+ 白盒 12（WB-01..12：多核切分/空闲核/余数核/尾块分支）+ Cross 24（dtype × n 阶梯参数化补全覆盖） |
| 运行命令 | `./build/runUnitTests --gtest_filter='*UnaryElewise*'`（过滤式全量，一次命中全部 4 套件）→ 59/59 PASSED |
| 断言口径 | **bit-exact 权威断言**：输出与 CPU golden 逐元素逐位相等，mismatch = 0（含 INT32_MIN / INT64_MIN、非对齐尾块、多核余数、全零 rare-tile 路径）；FatalMsg 用例校验不支持 dtype 的 `LOG(FATAL)` 文案 |
| 构建 | `bash build.sh`（USE_ASCEND=ON；kernel 变更后仅支持全量重建，增量 merge 非幂等） |

## 支持芯片

| 项目 | 内容 |
|------|------|
| 芯片型号 | Ascend 910B3（dav-2201，c220 微架构） |
| `--npu-arch` | `ascend910b` |
| `NpuArch` | 2201（DAV_2201） |
| 仓库默认 SOC_VERSION | Ascend910B4（构建脚本自动回退，与 910B3 同属 DAV_2201、内核代码一致，兼容） |
| CANN 版本 | 8.5.0（Ascend-cann-toolkit，aarch64-linux） |
| 代码架构 | SIMD / MemBase（`kernel_operator.h`），纯 Vector（AIV_ONLY）；不支持 SIMT / RegBase（dav-2201 无此能力），不涉及 Cube（非 Matmul 类） |
