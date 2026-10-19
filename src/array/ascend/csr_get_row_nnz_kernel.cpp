#include "kernel_operator.h"

using namespace AscendC;

template <typename IdxT>
class KernelCsrGetRowNNZ {
 public:
  __aicore__ inline KernelCsrGetRowNNZ() {}

  __aicore__ inline void Init(
      GM_ADDR indptr, GM_ADDR rows, GM_ADDR out, uint32_t n,
      uint32_t orig_rows, TPipe* pipe) {
    // Typed __gm__ pointers for reads; outputs go through a UB staging
    // buffer flushed with DataCopy — per-word GM stores (GlobalTensor
    // SetValue AND raw __gm__ pointer stores) are NOT reliably visible
    // when written from multiple blocks (writes from non-zero blocks
    // silently dropped, verified on the real machine; the same class of
    // failure the biased operator documented). MTE copies carry the
    // cache-coherent path.
    indptr_p_ = (const __gm__ IdxT *)indptr;
    rows_p_ = (const __gm__ IdxT *)rows;
    out_gm_.SetGlobalBuffer((__gm__ IdxT *)out, n);
    n_ = n;
    orig_rows_ = orig_rows;
    pipe->InitBuffer(out_buf_, kFlushElems * sizeof(IdxT));
  }

  __aicore__ inline void Process() {
    const uint32_t block_id = GetBlockIdx();
    const uint32_t block_num = GetBlockNum();
    const uint32_t chunk = (n_ + block_num - 1) / block_num;
    const uint32_t start = block_id * chunk;
    const uint32_t end = (start + chunk > n_) ? n_ : start + chunk;
    LocalTensor<IdxT> staging = out_buf_.Get<IdxT>();
    uint32_t staged = 0;
    uint32_t flushed = 0;
    for (uint32_t i = start; i < end; i++) {
      const IdxT row = rows_p_[i];
      IdxT deg = 0;
      if (static_cast<uint32_t>(row) < orig_rows_) {
        deg = indptr_p_[row + 1] - indptr_p_[row];
      }
      staging.SetValue(staged, deg);
      ++staged;
      if (staged == kFlushElems) {
        FlushOut(staging, start + flushed, staged);
        flushed += staged;
        staged = 0;
      }
    }
    if (staged > 0) FlushOut(staging, start + flushed, staged);
  }

 private:
  __aicore__ inline void FlushOut(
      LocalTensor<IdxT>& staging, uint32_t out_index, uint32_t count) {
    const uint32_t copy_bytes = count * sizeof(IdxT);
    DataCopyExtParams cp{1, copy_bytes, 0, 0, 0};
    DataCopyPad(out_gm_[out_index], staging, cp);
  }

  static constexpr uint32_t kFlushElems = 1024;

  const __gm__ IdxT *indptr_p_ = nullptr;
  const __gm__ IdxT *rows_p_ = nullptr;
  GlobalTensor<IdxT> out_gm_;
  uint32_t n_;
  uint32_t orig_rows_;
  TBuf<TPosition::VECCALC> out_buf_;
};

extern "C" __global__ __aicore__ void csr_get_row_nnz_int32(
    GM_ADDR indptr, GM_ADDR rows, GM_ADDR out, GM_ADDR tiling_ptr) {
  // Struct-pointer tiling access: per-word GetValue on a shared
  // GlobalTensor is not consistently visible across blocks.
  const __gm__ uint32_t *tiling = (const __gm__ uint32_t *)tiling_ptr;
  const uint32_t n = tiling[0];
  const uint32_t orig_rows = tiling[1];
  KernelCsrGetRowNNZ<int32_t> op;
  TPipe pipe;
  op.Init(indptr, rows, out, n, orig_rows, &pipe);
  op.Process();
}

extern "C" __global__ __aicore__ void csr_get_row_nnz_int64(
    GM_ADDR indptr, GM_ADDR rows, GM_ADDR out, GM_ADDR tiling_ptr) {
  const __gm__ uint32_t *tiling = (const __gm__ uint32_t *)tiling_ptr;
  const uint32_t n = tiling[0];
  const uint32_t orig_rows = tiling[1];
  KernelCsrGetRowNNZ<int64_t> op;
  TPipe pipe;
  op.Init(indptr, rows, out, n, orig_rows, &pipe);
  op.Process();
}
