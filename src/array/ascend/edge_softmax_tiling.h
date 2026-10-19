// ============================================================================
// Edge_softmax Tiling 结构体 - kernel 和 host 共用
// ============================================================================
//
// 纯 C/C++ 语法，不含 __aicore__、__gm__ 等 ASC 关键字。
// 对应 DESIGN.md §1.5 Buffer 规划 + §2.1 多核切分策略。
//
// 多核切分（DESIGN.md §2.1）：
//   - 沿目标节点（indptr 行）维度切分，每段独立计算 softmax
//   - blockDim = min(num_nodes, coreNum)，Host 侧动态获取 Vector Core 数量
//   - 每核处理 [blockIdx * rowsPerCore, min((blockIdx+1) * rowsPerCore, num_nodes))
//   - 尾核 indptr 按 actualRows = num_nodes - blockIdx * rowsPerCore 加载，避免越界读
//
// UB 切分（DESIGN.md §1.5、§2.2）：
//   - 逐段处理，每段内分批加载（每批 maxBatch 行）
//   - maxBatch 由 UB 剩余空间动态计算，上限 255（Pattern::Reduce::RA repeatTimes ≤ 255）
//   - FullLoad（degree ≤ maxBatch）：整段驻留 UB，3-pass in-place
//   - RowSplit（degree > maxBatch）：分批加载，3-pass 每遍重新加载
// ============================================================================

#pragma once

#include <cstdint>

// ============================================================================
// 常量定义
// ============================================================================

// UB 预留空间（用于栈帧等系统开销）
constexpr uint32_t UB_RESERVED = 2 * 1024;      // 预留 2 KB

// Pattern::Reduce::RA repeatTimes 上限（uint8_t）— ARA 模式 (num_heads>1) 上限
constexpr uint32_t MAX_BATCH = 255;

// AR 模式 (num_heads==1) 上限：Level 2 Reduce 无 255 限制，按 UB 容量取较大值
constexpr uint32_t MAX_BATCH_AR = 4095;

// Reduce API tmpBuf 保守预留大小（参考 api-reduce.md 示例 32KB）
// Pattern::Reduce::RA 实际需求由 GetReduceMaxMaxMinTmpSize 查询，32KB 足够覆盖
// Level 2 Reduce tmpBuf 类型为 LocalTensor<T>，复用此空间
constexpr uint32_t TMP_BUF_SIZE = 32 * 1024;

// 数据类型标识
constexpr uint32_t DTYPE_FP32 = 0;
constexpr uint32_t DTYPE_FP16 = 1;

// 模式标识
constexpr uint32_t MODE_FORWARD = 0;
constexpr uint32_t MODE_BACKWARD = 1;

// 32 字节对齐辅助（DESIGN.md §1.5 对齐计算）
constexpr uint32_t ALIGN_BYTES = 32;

// FP16 类型大小为 2 字节（host 侧不使用 half 类型，用 sizeof(int16_t) 替代）
constexpr uint32_t HALF_SIZE = 2;

// ============================================================================
// Tiling 数据结构 - 向核函数传递的运行时参数
// ============================================================================
struct EdgeSoftmaxTilingData {
    uint32_t numNodes;           // 目标节点数 = indptr 长度 - 1
    uint32_t numEdges;           // 边总数 = indptr[num_nodes] = efeat 行数
    uint32_t numHeads;           // 注意力头数（num_heads=1 时退化为 1D）
    uint32_t mode;               // 0=forward, 1=backward
    uint32_t dtype;              // 0=FP32, 1=FP16
    uint32_t blockDim;           // 实际使用的核数 = min(num_nodes, coreNum)
    uint32_t rowsPerCore;        // 每核处理的节点数 = ceil(num_nodes / blockDim)
    uint32_t maxBatch;           // UB 单批加载上限（FullLoad/RowSplit 判定阈值）
    uint32_t numHeadsAlignedF;   // num_heads 向上对齐到 32B（以 FP32 元素计，即 8 的倍数）
    uint32_t numHeadsAlignedH;   // num_heads 向上对齐到 32B（以 FP16 元素计，即 16 的倍数）
    uint32_t ubSize;             // UB 容量（字节），Host 侧动态获取后传入
};
