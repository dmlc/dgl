// ============================================================================
// spmm_unified Tiling 结构体 - kernel 和 host 共用
// ============================================================================
//
// 纯 C/C++ 语法，不含 __aicore__、__gm__ 等 ASC 关键字。
// 对应 DESIGN.md §1.5 Buffer 规划 + §2.1 多核切分策略。
//
// 多核切分（DESIGN.md §2.1）：
//   - sum 路径: AIC+AIV, blockDim=20 (cubeCoreCount), VectorCore=40
//   - max/min 路径: AIV_ONLY, blockDim=40 (vectorCoreCount)
// ============================================================================

#pragma once

#include <cstdint>

// ============================================================================
// 常量定义
// ============================================================================

// UB 预留空间（用于栈帧等系统开销）
constexpr uint32_t UB_RESERVED = 2 * 1024;      // 预留 2 KB

// Cube 相关常量
constexpr uint32_t CUBE_BLOCK_LENGTH = 16;       // M/N 方向分形宽度
constexpr uint32_t CUBE_L0A_BUFFER_BYTES = 32 * 1024;
constexpr uint32_t CUBE_L0B_BUFFER_BYTES = 32 * 1024;
constexpr uint32_t CUBE_L0C_BUFFER_BYTES = 64 * 1024;
constexpr uint32_t CUBE_BUFFER_NUM = 2;          // 双缓冲

// Vector 相关常量
constexpr uint32_t VECTOR_CUBE_BLOCK = 16;       // 窗口行数
constexpr uint32_t BUFFER_NUM = 2;               // 双缓冲
constexpr uint32_t UB_TOTAL_SIZE = 192 * 1024;
constexpr uint32_t ALIGN_BYTES = 32;
constexpr uint32_t UINT32_PER_ALIGN = ALIGN_BYTES / sizeof(uint32_t);
constexpr uint32_t UB_AVAILABLE = UB_TOTAL_SIZE - UB_RESERVED;

static constexpr uint32_t DTYPE_FP32 = 0;
static constexpr uint32_t DTYPE_FP16 = 1;
static constexpr uint32_t REDUCE_SUM = 0;
static constexpr uint32_t REDUCE_MAX = 1;
static constexpr uint32_t REDUCE_MIN = 2;
static constexpr uint32_t HALF_SIZE = 2;
static constexpr uint32_t CUBE_CORE_COUNT = 20;
static constexpr uint32_t VECTOR_CORE_COUNT = 40;

// 数据类型标识

// reduce 操作标识

// FP16 类型大小为 2 字节

// 核心数（Ascend 910B3 实测）

// ============================================================================
// 模板辅助: 获取 c0Size (L0A/L0B 分形 C0 维度大小)
// ============================================================================
// FP16 (half, 2B): c0Size = 16 (32B / 2B = 16)
// FP32 (float, 4B): c0Size = 8  (32B / 4B = 8)
// ============================================================================

// Host 侧辅助函数（非 __aicore__）：根据 dtype 获取 c0Size
inline uint32_t GetC0Size(uint32_t dtype)
{
    return (dtype == DTYPE_FP32) ? 8 : 16;
}

// ============================================================================
// Tiling 数据结构 - sum 路径 (向核函数传递的运行时参数)
// ============================================================================
// sum 路径使用 AIC+AIV 模式, blockDim=20
// 参数通过 aclrtlaunch_spmm_unified_sum 传递
// ============================================================================
struct SpmmUnifiedSumTilingData {
    uint32_t numDstRows;             // 目标节点数 = CSR 行数
    uint32_t numSrcRows;             // 源节点数 = ufeat 行数
    uint32_t featureDim;             // 特征维度
    uint32_t nonZeroCount;           // 非零元素总数
    uint32_t totalTcBlocks;          // 稠密块总 TC block 数
    uint32_t vectorWindowCount;      // 稀疏窗口数
    uint32_t cubeWindowCount;        // 稠密窗口数
    uint32_t columnToEdgeLength;     // column_to_edge 数组长度
    uint32_t batchCount;             // batch 数 (2D=1, 3D>1)
    uint32_t dtype;                  // 0=FP32, 1=FP16
};

// ============================================================================
// Tiling 数据结构 - max/min 路径
// ============================================================================
// max/min 路径使用 AIV_ONLY 模式, blockDim=40
// 参数通过 aclrtlaunch_spmm_unified_max / aclrtlaunch_spmm_unified_min 传递
// ============================================================================
struct SpmmUnifiedMaxMinTilingData {
    uint32_t numDstRows;             // 目标节点数
    uint32_t numSrcRows;             // 源节点数
    uint32_t featureDim;             // 特征维度
    uint32_t nonZeroCount;           // 非零元素总数
    uint32_t batchCount;             // batch 数
    uint32_t dtype;                  // 0=FP32, 1=FP16
};

// ============================================================================
// Tiling 数据结构 - aiv 路径 (sum 纯 Vector 备选)
// ============================================================================
struct SpmmUnifiedAivTilingData {
    uint32_t numDstRows;
    uint32_t numSrcRows;
    uint32_t featureDim;
    uint32_t nonZeroCount;
    uint32_t batchCount;
    uint32_t dtype;
    uint32_t isCopyRhs;  // 0=copy_lhs (gather feature[indices[e]]), 1=copy_rhs (sequential read feature[e])
};
