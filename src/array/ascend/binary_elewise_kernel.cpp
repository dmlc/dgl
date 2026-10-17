// Binary elementwise (array vs scalar) kernels.
//
// IMPORTANT (Ascend 950PR): scalar GetValue/SetValue on GlobalTensor (GM)
// is a production-blacklist API on this SoC (per-core DCache write-back
// coherence risk, see coo2csr_kernel.cpp ADR-0007).  All GM traffic below
// therefore goes through UB with DMA (DataCopyPad); the per-element
// compare runs on UB-resident values with scalar UB access only.  The
// tiling block is read through a typed __gm__ pointer.

#include "kernel_operator.h"

using namespace AscendC;

namespace {

struct BinaryLScalarTiling {
  uint32_t n;
  uint32_t scalar_hi;
  uint32_t scalar_lo;
};

constexpr uint32_t kChunk = 4096;  // elements per UB chunk (32 KB @ int64)

// Generic L-variant scalar compare driver: dst[i] = (scalar op lhs[i]).
// Op is a functor with `static bool cmp(T lhs, T rhs)` (rhs = scalar).
template <typename T, typename Op>
__aicore__ inline void RunBinaryL(
    GM_ADDR lhs, GM_ADDR dst, GM_ADDR tiling_ptr) {
  const __gm__ BinaryLScalarTiling* t =
      (const __gm__ BinaryLScalarTiling*)tiling_ptr;
  const uint32_t n = t->n;
  const int64_t raw = (static_cast<int64_t>(t->scalar_hi) << 32) |
                      static_cast<int64_t>(t->scalar_lo);
  const T scalar = static_cast<T>(raw);

  GlobalTensor<T> lhsGm, dstGm;
  lhsGm.SetGlobalBuffer((__gm__ T*)lhs);
  dstGm.SetGlobalBuffer((__gm__ T*)dst);

  TPipe pipe;
  TQue<QuePosition::VECIN, 1> inQue;
  TQue<QuePosition::VECOUT, 1> outQue;
  pipe.InitBuffer(inQue, 1, kChunk * sizeof(T));
  pipe.InitBuffer(outQue, 1, kChunk * sizeof(T));

  for (uint32_t off = 0; off < n; off += kChunk) {
    const uint32_t len = (n - off < kChunk) ? (n - off) : kChunk;
    const uint32_t bytes = len * sizeof(T);

    LocalTensor<T> in = inQue.AllocTensor<T>();
    DataCopyExtParams cp{1, bytes, 0, 0, 0};
    DataCopyPadExtParams<T> pad{false, 0, 0, 0};
    DataCopyPad(in, lhsGm[off], cp, pad);
    inQue.EnQue(in);
    LocalTensor<T> inDeq = inQue.DeQue<T>();

    LocalTensor<T> out = outQue.AllocTensor<T>();
    for (uint32_t i = 0; i < len; i++) {
      out.SetValue(i, static_cast<T>(Op::cmp(inDeq.GetValue(i), scalar)));
    }
    inQue.FreeTensor(inDeq);

    outQue.EnQue(out);
    LocalTensor<T> outDeq = outQue.DeQue<T>();
    DataCopyPad(dstGm[off], outDeq, cp);
    outQue.FreeTensor(outDeq);
  }
}

#define DEFINE_CMP(name, expr)                            \
  struct Cmp##name {                                      \
    template <typename T>                                 \
    __aicore__ static inline bool cmp(T l, T r) {         \
      return (expr);                                      \
    }                                                     \
  };

DEFINE_CMP(LT, l < r)
DEFINE_CMP(GT, l > r)
DEFINE_CMP(LE, l <= r)
DEFINE_CMP(GE, l >= r)
DEFINE_CMP(EQ, l == r)
DEFINE_CMP(NE, l != r)

}  // namespace

// int32 kernels
extern "C" __global__ __aicore__ void binary_l_lt_i32(
    GM_ADDR lhs, GM_ADDR dst, GM_ADDR tiling_ptr) {
  RunBinaryL<int32_t, CmpLT>(lhs, dst, tiling_ptr);
}
extern "C" __global__ __aicore__ void binary_l_gt_i32(
    GM_ADDR lhs, GM_ADDR dst, GM_ADDR tiling_ptr) {
  RunBinaryL<int32_t, CmpGT>(lhs, dst, tiling_ptr);
}
extern "C" __global__ __aicore__ void binary_l_le_i32(
    GM_ADDR lhs, GM_ADDR dst, GM_ADDR tiling_ptr) {
  RunBinaryL<int32_t, CmpLE>(lhs, dst, tiling_ptr);
}
extern "C" __global__ __aicore__ void binary_l_ge_i32(
    GM_ADDR lhs, GM_ADDR dst, GM_ADDR tiling_ptr) {
  RunBinaryL<int32_t, CmpGE>(lhs, dst, tiling_ptr);
}
extern "C" __global__ __aicore__ void binary_l_eq_i32(
    GM_ADDR lhs, GM_ADDR dst, GM_ADDR tiling_ptr) {
  RunBinaryL<int32_t, CmpEQ>(lhs, dst, tiling_ptr);
}
extern "C" __global__ __aicore__ void binary_l_ne_i32(
    GM_ADDR lhs, GM_ADDR dst, GM_ADDR tiling_ptr) {
  RunBinaryL<int32_t, CmpNE>(lhs, dst, tiling_ptr);
}

// int64 kernels
extern "C" __global__ __aicore__ void binary_l_lt_i64(
    GM_ADDR lhs, GM_ADDR dst, GM_ADDR tiling_ptr) {
  RunBinaryL<int64_t, CmpLT>(lhs, dst, tiling_ptr);
}
extern "C" __global__ __aicore__ void binary_l_gt_i64(
    GM_ADDR lhs, GM_ADDR dst, GM_ADDR tiling_ptr) {
  RunBinaryL<int64_t, CmpGT>(lhs, dst, tiling_ptr);
}
extern "C" __global__ __aicore__ void binary_l_le_i64(
    GM_ADDR lhs, GM_ADDR dst, GM_ADDR tiling_ptr) {
  RunBinaryL<int64_t, CmpLE>(lhs, dst, tiling_ptr);
}
extern "C" __global__ __aicore__ void binary_l_ge_i64(
    GM_ADDR lhs, GM_ADDR dst, GM_ADDR tiling_ptr) {
  RunBinaryL<int64_t, CmpGE>(lhs, dst, tiling_ptr);
}
extern "C" __global__ __aicore__ void binary_l_eq_i64(
    GM_ADDR lhs, GM_ADDR dst, GM_ADDR tiling_ptr) {
  RunBinaryL<int64_t, CmpEQ>(lhs, dst, tiling_ptr);
}
extern "C" __global__ __aicore__ void binary_l_ne_i64(
    GM_ADDR lhs, GM_ADDR dst, GM_ADDR tiling_ptr) {
  RunBinaryL<int64_t, CmpNE>(lhs, dst, tiling_ptr);
}
