// ============================================================================
// sddmm_copy_lhs Tiling 结构体 - kernel 和 host 共用
// ============================================================================
//
// 纯 C/C++ 语法，不含 __aicore__、__gm__ 等 ASC 关键字。
// 对应 DESIGN.md §1.5 Buffer 规划 + §2.1 多核切分策略。
//
// 多核切分（DESIGN.md §2.1）：
//   - 沿 edge（边）维度切分，每边独立 gather
//   - blockDim = min(nnz, coreNum)，Host 侧动态获取 Vector Core 数量
//   - 每核处理 [blockIdx * edgesPerCore, min((blockIdx+1) * edgesPerCore, nnz))
//   - 空闲核早退守卫：startEdge >= endEdge 时直接返回
//
// UB 切分（DESIGN.md §1.5、§2.2）：
//   - 索引分批加载（每批 batchSize 个 int32）
//   - 逐边 Gather，TQue<VECIN> Double Buffer MTE2/MTE3 流水（featQueue, num=2）
//   - feat_dim ≤ 256 时整行加载，无需分 chunk
//   - UB 容量通过 Host 侧编译期映射后传入
//
// 同步方案（修复 Round 1：TQue + immediate Set/Wait）:
//   - 索引加载：TQue EnQue/DeQue（MTE2→S 自动同步）
//   - 特征搬运：TQue<VECIN> Double Buffer (num=2) 管理 buffer 轮转
//     + SetFlag/WaitFlag<MTE2_MTE3> immediate Set→Wait 同步 MTE2→MTE3（数据就绪）
//     + SetFlag/WaitFlag<MTE3_MTE2> immediate Set→Wait 同步 MTE3→MTE2（buffer 复用安全）
//     参考 Ascend C 内部 dav_c220/kernel_operator_sync_impl.h 的 immediate Set/Wait 模式。
//     关键发现：跨迭代分离的 Set/Wait 在 910B3 上不工作，immediate Set/Wait 零误差通过。
// ============================================================================
#pragma once
#include <cstdint>
// ============================================================================
// 常量定义
// ============================================================================
// UB 预留空间（用于栈帧等系统开销）
constexpr uint32_t UB_RESERVED = 2 * 1024;      // 预留 2 KB
// 批量索引加载上限（DESIGN.md §1.5 MAX_BATCH，自设上限控制标量循环开销）
constexpr uint32_t MAX_BATCH = 4095;
// 数据类型标识
constexpr uint32_t DTYPE_FP32 = 0;
constexpr uint32_t DTYPE_FP16 = 1;
// 32 字节对齐辅助（DESIGN.md §1.5 对齐计算）
constexpr uint32_t ALIGN_BYTES = 32;
// FP16 类型大小为 2 字节（host 侧不使用 half 类型，用 sizeof(int16_t) 替代）
constexpr uint32_t HALF_SIZE = 2;
// ============================================================================
// Tiling 数据结构 - 向核函数传递的运行时参数
// ============================================================================
struct SddmmCopyLhsTilingData {
    uint32_t numNodes;          // 节点数 = feat 行数
    uint32_t nnz;               // 边数 = index 长度 = out 行数
    uint32_t featDim;           // 特征维度
    uint32_t blockDim;          // 实际使用的核数 = min(nnz, coreNum)
    uint32_t edgesPerCore;      // 每核处理的边数 = ceil(nnz / blockDim)
    uint32_t batchSize;         // UB 批量加载索引数
    uint32_t dtype;             // 0=FP32, 1=FP16
    uint32_t featDimAligned;    // featDim 向上对齐到 32B（以 T 元素计）
    uint32_t ubSize;            // UB 容量（字节），Host 侧动态获取后传入
};
