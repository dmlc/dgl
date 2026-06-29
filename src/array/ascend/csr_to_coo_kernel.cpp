#include "kernel_operator.h"

using namespace AscendC;

template <typename IdxT>
class KernelCsrToCOO {
 public:
  __aicore__ inline KernelCsrToCOO() {}

  __aicore__ inline void Init(
      GM_ADDR indptr, GM_ADDR ret_row,
      uint32_t num_rows, uint32_t nnz) {
    indptr_gm.SetGlobalBuffer((__gm__ IdxT *)indptr, num_rows + 1);
    ret_row_gm.SetGlobalBuffer((__gm__ IdxT *)ret_row, nnz);
    num_rows_ = num_rows;
    nnz_ = nnz;
  }

  __aicore__ inline void Process() {
    uint32_t block_id = GetBlockIdx();
    uint32_t block_num = GetBlockNum();
    uint32_t chunk = (num_rows_ + block_num - 1) / block_num;
    uint32_t start_row = block_id * chunk;
    uint32_t end_row = (start_row + chunk > num_rows_) ? num_rows_ : start_row + chunk;
    for (uint32_t row = start_row; row < end_row; row++) {
      IdxT start = indptr_gm.GetValue(row);
      IdxT end = indptr_gm.GetValue(row + 1);
      for (IdxT pos = start; pos < end; pos++) {
        ret_row_gm.SetValue(static_cast<uint32_t>(pos), static_cast<IdxT>(row));
      }
    }
  }

 private:
  GlobalTensor<IdxT> indptr_gm;
  GlobalTensor<IdxT> ret_row_gm;
  uint32_t num_rows_;
  uint32_t nnz_;
};

extern "C" __global__ __aicore__ void csr_to_coo_int32(
    GM_ADDR indptr, GM_ADDR ret_row, GM_ADDR tiling_ptr) {
  GlobalTensor<uint32_t> tilingGm;
  tilingGm.SetGlobalBuffer((__gm__ uint32_t *)tiling_ptr, 2);
  uint32_t num_rows = tilingGm.GetValue(0);
  uint32_t nnz = tilingGm.GetValue(1);
  KernelCsrToCOO<int32_t> op;
  op.Init(indptr, ret_row, num_rows, nnz);
  op.Process();
}

extern "C" __global__ __aicore__ void csr_to_coo_int64(
    GM_ADDR indptr, GM_ADDR ret_row, GM_ADDR tiling_ptr) {
  GlobalTensor<uint32_t> tilingGm;
  tilingGm.SetGlobalBuffer((__gm__ uint32_t *)tiling_ptr, 2);
  uint32_t num_rows = tilingGm.GetValue(0);
  uint32_t nnz = tilingGm.GetValue(1);
  KernelCsrToCOO<int64_t> op;
  op.Init(indptr, ret_row, num_rows, nnz);
  op.Process();
}

