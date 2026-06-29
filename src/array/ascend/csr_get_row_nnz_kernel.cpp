#include "kernel_operator.h"

using namespace AscendC;

template <typename IdxT>
class KernelCsrGetRowNNZ {
 public:
  __aicore__ inline KernelCsrGetRowNNZ() {}

  __aicore__ inline void Init(
      GM_ADDR indptr, GM_ADDR rows, GM_ADDR out,
      uint32_t n, uint32_t orig_rows) {
    indptr_gm.SetGlobalBuffer((__gm__ IdxT *)indptr, orig_rows + 1);
    rows_gm.SetGlobalBuffer((__gm__ IdxT *)rows, n);
    out_gm.SetGlobalBuffer((__gm__ IdxT *)out, n);
    n_ = n;
    orig_rows_ = orig_rows;
  }

  __aicore__ inline void Process() {
    uint32_t block_id = GetBlockIdx();
    uint32_t block_num = GetBlockNum();
    uint32_t chunk = (n_ + block_num - 1) / block_num;
    uint32_t start = block_id * chunk;
    uint32_t end = (start + chunk > n_) ? n_ : start + chunk;
    for (uint32_t i = start; i < end; i++) {
      IdxT row = rows_gm.GetValue(i);
      if (static_cast<uint32_t>(row) >= orig_rows_) {
        out_gm.SetValue(i, 0);
        continue;
      }
      IdxT start_i = indptr_gm.GetValue(static_cast<uint32_t>(row));
      IdxT end_i = indptr_gm.GetValue(static_cast<uint32_t>(row) + 1);
      out_gm.SetValue(i, end_i - start_i);
    }
  }

 private:
  GlobalTensor<IdxT> indptr_gm;
  GlobalTensor<IdxT> rows_gm;
  GlobalTensor<IdxT> out_gm;
  uint32_t n_;
  uint32_t orig_rows_;
};

extern "C" __global__ __aicore__ void csr_get_row_nnz_int32(
    GM_ADDR indptr, GM_ADDR rows, GM_ADDR out,
    GM_ADDR tiling_ptr) {
  GlobalTensor<uint32_t> tilingGm;
  tilingGm.SetGlobalBuffer((__gm__ uint32_t *)tiling_ptr, 2);
  uint32_t n = tilingGm.GetValue(0);
  uint32_t orig_rows = tilingGm.GetValue(1);
  KernelCsrGetRowNNZ<int32_t> op;
  op.Init(indptr, rows, out, n, orig_rows);
  op.Process();
}

extern "C" __global__ __aicore__ void csr_get_row_nnz_int64(
    GM_ADDR indptr, GM_ADDR rows, GM_ADDR out,
    GM_ADDR tiling_ptr) {
  GlobalTensor<uint32_t> tilingGm;
  tilingGm.SetGlobalBuffer((__gm__ uint32_t *)tiling_ptr, 2);
  uint32_t n = tilingGm.GetValue(0);
  uint32_t orig_rows = tilingGm.GetValue(1);
  KernelCsrGetRowNNZ<int64_t> op;
  op.Init(indptr, rows, out, n, orig_rows);
  op.Process();
}

