// ============================================================================
// sddmm_binary Tiling 结构体 - kernel 和 host 共用
// ============================================================================
//
// 对应 DESIGN.md §2.5 Tiling 结构 + §2.1 多核切分策略 + §2.2 UB 切分策略。
//
// 与 sddmm_copy_lhs 的差异（DESIGN.md §5.1）：
//   - 新增 op 字段（0=add, 1=sub, 2=mul, 3=div）用于运行时分发
//   - 新增 numNodesRhs 字段（rhs 节点数，与 numNodesLhs 独立）
//   - 新增 featDimAlignedF32 字段（FP16 路径 float 中间 buffer 对齐长度）
//
// 多核切分（DESIGN.md §2.1）：
//   - 沿 edge（边）维度切分，每边独立 gather+compute
//   - blockDim = min(nnz, coreNum)，Host 侧动态获取 Vector Core 数量
//   - 每核处理 [blockIdx * edgesPerCore, min((blockIdx+1) * edgesPerCore, nnz))
//   - 空闲核早退守卫：startEdge >= endEdge 时直接返回
//
// UB 切分（DESIGN.md §1.5、§2.2）：
//   - 索引分批加载（每批 batchSize 个 int32，idxLhs + idxRhs 两个队列）
//   - 逐边 Gather lhs + rhs → binary compute → store out
//   - lhsQueue/rhsQueue/outQueue 均 Double Buffer (num=2)
//   - FP16 路径额外 3 个 TBuf<VECCALC> float 中间 buffer
//
// 同步方案（DESIGN.md §1.3）：
//   - TQue<VECIN> EnQue/DeQue 自动同步 MTE2→V
//   - TQue<VECOUT> EnQue/DeQue 自动同步 V→MTE3
//   - 无需手动 SetFlag/WaitFlag（V 是天然中间消费者/生产者）
// ============================================================================
#pragma once
#include <cstdint>
// ============================================================================
// 常量定义
// ============================================================================
// UB 预留空间（用于栈帧等系统开销）
constexpr uint32_t BINARY_UB_RESERVED = 2 * 1024;      // 预留 2 KB
constexpr uint32_t BINARY_MAX_BATCH = 4095;
constexpr uint32_t BINARY_DTYPE_FP32 = 0;
constexpr uint32_t BINARY_DTYPE_FP16 = 1;
// 数据类型标识
// 二元运算类型标识（DESIGN.md §2.3）
constexpr uint32_t BINARY_OP_ADD = 0;
constexpr uint32_t BINARY_OP_SUB = 1;
constexpr uint32_t BINARY_OP_MUL = 2;
constexpr uint32_t BINARY_OP_DIV = 3;
constexpr uint32_t BINARY_ALIGN_BYTES = 32;
constexpr uint32_t BINARY_HALF_SIZE = 2;
// FP16 类型大小为 2 字节（host 侧不使用 half 类型，用 sizeof(int16_t) 替代）
// ============================================================================
// Tiling 数据结构 - 向核函数传递的运行时参数
// 对应 DESIGN.md §2.5
// ============================================================================
struct SddmmBinaryTilingData {
    uint32_t numNodesLhs;       // lhs 节点数 = lhsFeat 行数
    uint32_t numNodesRhs;       // rhs 节点数 = rhsFeat 行数
    uint32_t nnz;               // 边数 = indexLhs/indexRhs 长度 = out 行数
    uint32_t featDim;           // 特征维度
    uint32_t blockDim;          // 实际使用的核数 = min(nnz, coreNum)
    uint32_t edgesPerCore;      // 每核处理的边数 = ceil(nnz / blockDim)
    uint32_t batchSize;         // UB 批量加载索引数
    uint32_t dtype;             // 0=FP32, 1=FP16
    uint32_t op;                // 0=add, 1=sub, 2=mul, 3=div
    uint32_t featDimAligned;    // featDim 向上对齐到 32B（以 native dtype T 元素计）
    uint32_t featDimAlignedF32; // featDim 向上对齐到 32B（以 float 元素计，FP16 路径专用）
    uint32_t ubSize;            // UB 容量（字节），Host 侧动态获取后传入
};
