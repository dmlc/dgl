// BFSNodesFrontiers kernel 直调入口
//
//   · 由 C++ 包装 + `<<<>>>` 调用改为直调形态 `extern "C" __global__ __aicore__`
//     （与 `range_kernel.cpp` 同款）。
//   · 单入口 + dtype_bytes 分支：规避 CANN 9.0.0 下同文件多 `extern "C" __global__`
//     AIV 函数 meta 段编号不匹配导致注册失败 107000（range_kernel.cpp 注释）。
//   · 标量参数（numRows/nnz/S/hasEid/tilingKey/numBlocks/numChunksMax/countOff）
//     打包进 `bfs_ws_layout.h::BfsTilingData`（POD），经 tiling GM 缓冲传入。
//   · 算法本体在 `bfs_core.h::KernelBfs<T, false>`，原样复用。

#include "bfs_core.h"
#include "bfs_ws_layout.h"

using namespace bfscore;

extern "C" __global__ __aicore__ void bfs_nodes_frontiers(
    GM_ADDR indptr, GM_ADDR indices, GM_ADDR eid, GM_ADDR source,
    GM_ADDR ids, GM_ADDR sections, GM_ADDR ws, GM_ADDR tiling) {
  // Single entry + dtype_bytes branch: avoids the K_TYPE_AIV multi-function
  // meta-section numbering bug (CANN 9.0.0: 2nd function's meta section _0
  // vs symbol _1 mismatch -> registration 107000).
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

  // tiling 字段读取（与 host 侧 BfsTilingData 布局一致）：
  // 前 10 个 int64 字段经 int64 视图读；dtype_bytes 在 uint32 视图第 20 槽。
  AscendC::GlobalTensor<int64_t> t64;
  t64.SetGlobalBuffer((__gm__ int64_t *)tiling, 10);
  AscendC::GlobalTensor<uint32_t> t32;
  t32.SetGlobalBuffer((__gm__ uint32_t *)tiling, 21);
  const int64_t numRows = t64.GetValue(0);
  const int64_t nnz = t64.GetValue(1);
  const int64_t S = t64.GetValue(2);
  const int64_t hasEid = t64.GetValue(3);
  const int64_t tilingKey = t64.GetValue(4);
  const int64_t numBlocks = t64.GetValue(5);
  const int64_t numChunksMax = t64.GetValue(6);
  const int64_t countOff = t64.GetValue(9);
  const uint32_t dtype_bytes = t32.GetValue(20);

  if (dtype_bytes == 4) {
    KernelBfs<int32_t, false> op;
    op.Init(indptr, indices, eid, source, ids, sections, ws, countOff,
            numRows, nnz, S, hasEid, tilingKey, numBlocks, numChunksMax);
    op.Process();
  } else {
    KernelBfs<int64_t, false> op;
    op.Init(indptr, indices, eid, source, ids, sections, ws, countOff,
            numRows, nnz, S, hasEid, tilingKey, numBlocks, numChunksMax);
    op.Process();
  }
}
