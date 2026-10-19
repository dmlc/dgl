// UnaryElewise int32 kernel (arith::Neg, y = -x): vectorized GM->UB -> Muls(-1) -> UB->GM.
// Keep one kernel entry per translation unit: CANN 8.5 mis-registers multi-entry
// blobs that set a task type, so the int64 entry lives in unary_elewise_i64_kernel.cpp.
#include "kernel_operator.h"

#include "unary_elewise_tiling.h"

using namespace AscendC;

constexpr uint32_t kUnaryTileElems = 8192;  // UB tile: 8192 x 4B = 32KB

// int32 vectorized path: y = -x via Muls(dst, src, -1, count).
// int32 multiply wraps modulo 2^32, so x * (-1) == -x bitwise for every
// int32 value, including INT32_MIN ((-2^31) * (-1) wraps back to -2^31):
// bit-exact with the CPU golden (arith::Neg::Call -> -t1).
extern "C" __global__ __aicore__ void unary_neg_i32(
    GM_ADDR x, GM_ADDR y, GM_ADDR tiling_ptr) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

  GlobalTensor<uint32_t> tilingGm;
  tilingGm.SetGlobalBuffer((__gm__ uint32_t*)tiling_ptr, 1);
  uint32_t n = tilingGm.GetValue(0);

  uint32_t start = 0;
  uint32_t end = 0;
  UnaryBlockRange(n, start, end);  // last core carries remainder; idle if start >= n

  GlobalTensor<int32_t> xGm;
  xGm.SetGlobalBuffer((__gm__ int32_t*)x, n);
  GlobalTensor<int32_t> yGm;
  yGm.SetGlobalBuffer((__gm__ int32_t*)y, n);

  AscendC::TPipe pipe;
  TQue<TPosition::VECIN, 2> xQueue;
  TQue<TPosition::VECOUT, 2> yQueue;
  pipe.InitBuffer(xQueue, 2, kUnaryTileElems * sizeof(int32_t));
  pipe.InitBuffer(yQueue, 2, kUnaryTileElems * sizeof(int32_t));

  for (uint32_t off = start; off < end;) {
    uint32_t cnt = end - off;
    if (cnt > kUnaryTileElems) cnt = kUnaryTileElems;

    LocalTensor<int32_t> xLocal = xQueue.AllocTensor<int32_t>();
    if (cnt == kUnaryTileElems) {
      DataCopy(xLocal, xGm[off], kUnaryTileElems);
    } else {
      DataCopyExtParams inParams{1,
          static_cast<uint32_t>(cnt * sizeof(int32_t)), 0, 0, 0};
      DataCopyPadExtParams<int32_t> inPad{false, 0, 0, 0};
      DataCopyPad(xLocal, xGm[off], inParams, inPad);
    }
    xQueue.EnQue(xLocal);

    LocalTensor<int32_t> xDeq = xQueue.DeQue<int32_t>();
    LocalTensor<int32_t> yLocal = yQueue.AllocTensor<int32_t>();
    Muls(yLocal, xDeq, static_cast<int32_t>(-1), static_cast<int32_t>(cnt));
    yQueue.EnQue(yLocal);
    xQueue.FreeTensor(xDeq);

    LocalTensor<int32_t> yDeq = yQueue.DeQue<int32_t>();
    if (cnt == kUnaryTileElems) {
      DataCopy(yGm[off], yDeq, kUnaryTileElems);
    } else {
      DataCopyExtParams outParams{1,
          static_cast<uint32_t>(cnt * sizeof(int32_t)), 0, 0, 0};
      DataCopyPad(yGm[off], yDeq, outParams);  // only cnt valid elems to GM
    }
    yQueue.FreeTensor(yDeq);

    off += cnt;
  }
}
