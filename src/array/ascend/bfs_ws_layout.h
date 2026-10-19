// =============================================================================
// BFSNodesFrontiers / BFSEdgesFrontiers 共享布局头（host .cc 与 kernel 共用）
//
// 供host 侧（g++ 编译进 libdgl）与 kernel 侧（bisheng 编译）共同 include，
// 防止 workspace 布局 / tiling 算式在 host 与 device 间漂移。
//
// 平移原则：
//   · 算式与独立工程严格同式，勿单边修改（HsWorkspaceLayout 与 kernel 侧
//     BfsWorkspaceLayout 保持同式）。
//   · BfsTilingData 为主仓直启形态新增的 tiling 结构（POD，int64/uint32，
//     无 padding 风险：全部 8 字节字段在前、uint32 收尾，kernel 经
//     GlobalTensor<int64_t> / <uint32_t> 读取与 host 侧填充布局一致）。
// =============================================================================
#ifndef DGL_ARRAY_ASCEND_BFS_WS_LAYOUT_H_
#define DGL_ARRAY_ASCEND_BFS_WS_LAYOUT_H_

#include <cstdint>

namespace bfscore {

// ---------------- 常量段（独立工程 bfs_core.h L26-45 原样） ----------------
constexpr static int32_t INT32_MAXV = 0x7fffffff;
constexpr static int64_t UB_KEY0_BUDGET = 96 * 1024;   // 方案 §7：Key0 UB 工作集预算
constexpr static int64_t KEY1_MIN_ROWS = 8192;         // 方案 §4：深图启发式阈值
constexpr static int32_t D0_ROW_MTE = 16;              // 方案 O2：行段 MTE 搬运度数阈值
constexpr static int32_t TINY_F = 256;                 // 方案 O1：微层父节点数阈值
constexpr static int32_t TINY_E = 2048;                // 方案 O1：微层边数阈值
// [FIX-mc3] 保底开关（Key2 multi-chunk 多核路径与 O1 微层分支默认关闭。
// 上板实测跨核 MTE 读可见性未达确定性（WB-K2-01~04 输出错、05~06 hang）→ 关闭时 key=2
// 在 kernel 内退避到已验证正确的 Key1 串行（GM 标量）路径。置 true 可重新启用多核路径
// 说明：此为 kernel 运行时保底开关，不改 host 侧 tilingKey 判定（tiling 仍算 key=2，
// kernel 内部退避），tiling 观测与 wb_tiling expected_key 的差异由 PM 协调 developer-test 调口径。
constexpr static bool BFS_ENABLE_KEY2_MULTICORE = false;
constexpr static bool BFS_ENABLE_O1_MICROLAYER = false;
constexpr static int32_t REC_INTS = 4;                 // 候选/确认记录 = 16B（v,pos,valLo,valHi）
constexpr static int32_t SRC_WIN = 2048;               // 源层扫描分窗元素数（三处源层循环共用）
constexpr static int32_t WS_ALIGN = 512;

// ---------------------------------------------------------------------------
// workspace 布局（host/plugin 与 kernel 共用同一算式，避免漂移）
//   visTag    : N×4                层号标记（0=未发现，否则=发现层+1）
//   syncGm    : C×32               SoftSyncAll workspace
//   errflag   : C×32               校验错误旗标（每核 32B 块，lane0=类别码）
//   segByCore : C×32               P0 度数段总计
//   cntDom    : numChunksMax×rowStrideint32   [j][d] P1 候选域直方图（chunk owner 写行）
//   cntConfD  : numChunksMax×rowStrideint32   [d][j] P2 确认计数（域 owner 写行）
//   cntConfT  : numChunksMax×rowStrideint32   [j][d] P2.5 转置（core0 写）
//   candRec   : nnz×16B            候选记录（chunk 分片，pos 序，全记录）
//   confRec   : (N+C)×16B          确认记录（(j,d) 分片，容量=domHist 精确前缀）
//   frNodes   : N×4                edges 版下一层父序（nodes 版免此 buffer）
//   stgIds    : numChunksMax×CH×es chunk 槽位化 ids staging（T 型，位置对齐）
//   stgFr     : numChunksMax×CH×4  chunk 槽位化 frontier staging（int32，仅 edges 用）
//   fragIds   : 32                 ids 流 pending 尾扇区 frag 槽（单槽）
//   fragFr    : 32                 frontier 流 pending 尾扇区 frag 槽（单槽）
//   （主仓集成扩展）计数槽：host 在 ws 尾部追加 16B（wsBytes..wsBytes+16），
//   kernel 经 tiling.countOff 写入 idsLen / secLen，host launch 后回读做精确长度截断。
// ---------------------------------------------------------------------------
struct BfsWs {
    int64_t visTag;
    int64_t syncGm;
    int64_t errflag;
    int64_t segByCore;
    int64_t cntDom;
    int64_t cntConfD;
    int64_t cntConfT;
    int64_t candRec;
    int64_t confRec;
    int64_t frNodes;
    int64_t stgIds;
    int64_t stgFr;
    int64_t fragIds;
    int64_t fragFr;
    int64_t total;
};

// ---------------- host 侧算式（g++ 可见；与 device 侧 __aicore__ 版严格同式，勿单边修改） ----------------
inline int64_t HsAlign(int64_t v, int64_t a) { return (v + a - 1) / a * a; }
inline int32_t HsRowStrideInt32(int32_t cores)
{
    return (int32_t)HsAlign((int64_t)cores * 4, 32) / 4;
}

// cntConfD[d][j] 的 d 维行宽：d 行有 numChunksMax 列（j 维），须 32B 对齐。
// （不能复用基于 cores 的 rowStride：当 numChunksMax > rowStride 时 d*rowStride+j 会与 (d+1) 行重叠）
inline int32_t HsRowStrideJ(int32_t numChunksMax)
{
    return (int32_t)HsAlign((int64_t)numChunksMax * 4, 32) / 4;
}

inline int32_t HsChunkElems(int64_t elemSize) { return (elemSize == 4) ? 1024 : 512; }

inline int64_t HsConfRecCap(int64_t numRows, int64_t nnz, int32_t numChunksMax, int32_t cores)
{
    // confRec (j,d) 分片按候选数 cntDom[j][d] 对齐预留槽位（跨核无冲突写需要按候选数定位，
    // 因域 owner 写片时不知道其他域实际确认数）。总槽位 = Σ_j Σ_d align2(cntDom[j][d])
    // ≤ Σ(cap+1) = E + numChunks*C ≤ nnz + numChunksMax*C。此前的 N+C 容量在 E>N（大 frontier
    // 层 / avg_deg>1）时越界写 GM，破坏后续 workspace（stgIds/frNodes），是 Key2 数值错乱根因。
    return nnz + (int64_t)numChunksMax * cores;
}

inline BfsWs HsWorkspaceLayout(int64_t numRows, int64_t nnz, int32_t cores, int32_t numChunksMax,
                               int64_t elemSize)
{
    BfsWs w;
    int32_t rs = HsRowStrideInt32(cores);
    int64_t off = 0;
    w.visTag = off;  off += HsAlign(numRows * 4, WS_ALIGN);
    w.syncGm = off;  off += HsAlign((int64_t)cores * 32, WS_ALIGN);
    w.errflag = off; off += HsAlign((int64_t)cores * 32, WS_ALIGN);
    w.segByCore = off; off += HsAlign((int64_t)cores * 32, WS_ALIGN);
    w.cntDom = off;  off += HsAlign((int64_t)numChunksMax * rs * 4, WS_ALIGN);
    w.cntConfD = off; off += HsAlign((int64_t)cores * HsRowStrideJ(numChunksMax) * 4, WS_ALIGN);
    w.cntConfT = off; off += HsAlign((int64_t)numChunksMax * rs * 4, WS_ALIGN);
    w.candRec = off; off += HsAlign(nnz * REC_INTS * 4, WS_ALIGN);
    w.confRec = off; off += HsAlign(HsConfRecCap(numRows, nnz, numChunksMax, cores) * REC_INTS * 4, WS_ALIGN);
    w.frNodes = off; off += HsAlign(numRows * 4, WS_ALIGN);
    w.stgIds = off;  off += HsAlign((int64_t)numChunksMax * HsChunkElems(elemSize) * elemSize, WS_ALIGN);
    w.stgFr = off;   off += HsAlign((int64_t)numChunksMax * HsChunkElems(elemSize) * 4, WS_ALIGN);
    w.fragIds = off; off += HsAlign(64, WS_ALIGN);   // 两槽 fragIds[parity]（头读/尾写分槽，避免同层并发竞争）
    w.fragFr = off;  off += HsAlign(64, WS_ALIGN);   // 两槽 fragFr[parity]
    w.total = off;
    return w;
}

// ---------------- 主仓直启 tiling 结构（host/kernel 共用同式，POD） ----------------
// 字段顺序/位宽即 GM 内存布局：kernel 经 GlobalTensor<int64_t>（10 槽）读前 10 个
// int64 字段、GlobalTensor<uint32_t> 第 20 槽读 dtype_bytes；host 侧同构填充。
struct BfsTilingData {
    int64_t numRows;      // [0]  CSR 行数
    int64_t nnz;          // [1]  CSR 非零元素数
    int64_t S;            // [2]  源节点数
    int64_t hasEid;       // [3]  edges 版 eid 是否存在（0/1）
    int64_t tilingKey;    // [4]  0/1/2
    int64_t numBlocks;    // [5]  launch block_dim
    int64_t numChunksMax; // [6]  chunk 数上界
    int64_t idsLenUb;     // [7]  ids 输出缓冲上界（host 分配长度，仅 host 校验用）
    int64_t secLenUb;     // [8]  sections 输出缓冲上界（仅 host 校验用）
    int64_t countOff;     // [9]  ws 内计数槽偏移（kernel 写 idsLen/secLen）
    uint32_t dtype_bytes; // [20] sizeof(IdType)：4 / 8
};

}  // namespace bfscore

#endif  // DGL_ARRAY_ASCEND_BFS_WS_LAYOUT_H_
