#ifndef COO_TO_CSR_TILING_H
#define COO_TO_CSR_TILING_H
#include <cstdint>

struct CooToCsrIndptrTiling {
  uint32_t nnz;
  uint32_t num_rows;
  uint32_t block_idx;
  uint32_t block_num;
};
#endif

#include "kernel_operator.h"

using namespace AscendC;

template <typename T>
class KernelCooToCsrIndptr {
 public:
  __aicore__ inline KernelCooToCsrIndptr() {}

  __aicore__ inline void Init(
      GM_ADDR rows, GM_ADDR indptr, uint32_t nnz, uint32_t num_rows) {
    rows_gm.SetGlobalBuffer((__gm__ T *)rows, nnz);
    indptr_gm.SetGlobalBuffer((__gm__ T *)indptr, num_rows + 1);
    nnz_ = nnz;
    num_rows_ = num_rows;
  }

  __aicore__ inline void Process(uint32_t blockIdx, uint32_t blockNum) {
    uint32_t rowsPerBlock = (num_rows_ + blockNum - 1) / blockNum;
    uint32_t rowStart = blockIdx * rowsPerBlock;
    uint32_t rowEnd = rowStart + rowsPerBlock > num_rows_
                          ? num_rows_
                          : rowStart + rowsPerBlock;

    for (uint32_t rowIdx = rowStart; rowIdx < rowEnd; rowIdx++) {
      T lo = 0;
      T hi = static_cast<T>(nnz_);
      while (lo < hi) {
        T mid = (lo + hi) >> 1;
        T val = rows_gm.GetValue(static_cast<uint32_t>(mid));
        if (val <= static_cast<T>(rowIdx)) {
          lo = mid + 1;
        } else {
          hi = mid;
        }
      }
      indptr_gm.SetValue(rowIdx + 1, lo);
    }
  }

 private:
  GlobalTensor<T> rows_gm;
  GlobalTensor<T> indptr_gm;
  uint32_t nnz_;
  uint32_t num_rows_;
};

extern "C" __global__ __aicore__ void coo_to_csr_indptr(
    GM_ADDR rows, GM_ADDR indptr, GM_ADDR tiling_ptr) {
  GlobalTensor<uint32_t> tilingGm;
  tilingGm.SetGlobalBuffer((__gm__ uint32_t *)tiling_ptr, 4);
  uint32_t nnz = tilingGm.GetValue(0);
  uint32_t num_rows = tilingGm.GetValue(1);
  uint32_t blockIdx = tilingGm.GetValue(2);
  uint32_t blockNum = tilingGm.GetValue(3);

  KernelCooToCsrIndptr<int32_t> op;
  op.Init(rows, indptr, nnz, num_rows);
  op.Process(blockIdx, blockNum);
}

template <typename T>
class KernelGather {
 public:
  __aicore__ inline void Init(
      GM_ADDR src, GM_ADDR idx, GM_ADDR dst, uint32_t n) {
    src_gm.SetGlobalBuffer((__gm__ T *)src, n);
    idx_gm.SetGlobalBuffer((__gm__ int64_t *)idx, n);
    dst_gm.SetGlobalBuffer((__gm__ T *)dst, n);
    n_ = n;
  }

  __aicore__ inline void Process() {
    if (n_ == 0) return;
    for (uint32_t i = 0; i < n_; i++) {
      int64_t p = idx_gm.GetValue(i);
      dst_gm.SetValue(i, src_gm.GetValue(static_cast<uint32_t>(p)));
    }
  }

 private:
  GlobalTensor<T> src_gm;
  GlobalTensor<int64_t> idx_gm;
  GlobalTensor<T> dst_gm;
  uint32_t n_;
};

extern "C" __global__ __aicore__ void coo_gather_int32(
    GM_ADDR src, GM_ADDR idx, GM_ADDR dst, GM_ADDR tiling_ptr) {
  GlobalTensor<uint32_t> tilingGm;
  tilingGm.SetGlobalBuffer((__gm__ uint32_t *)tiling_ptr, 1);
  uint32_t n = tilingGm.GetValue(0);
  KernelGather<int32_t> op;
  op.Init(src, idx, dst, n);
  op.Process();
}

extern "C" __global__ __aicore__ void coo_gather_int64(
    GM_ADDR src, GM_ADDR idx, GM_ADDR dst, GM_ADDR tiling_ptr) {
  GlobalTensor<uint32_t> tilingGm;
  tilingGm.SetGlobalBuffer((__gm__ uint32_t *)tiling_ptr, 1);
  uint32_t n = tilingGm.GetValue(0);
  KernelGather<int64_t> op;
  op.Init(src, idx, dst, n);
  op.Process();
}

extern "C" __global__ __aicore__ void coo_gather_float(
    GM_ADDR src, GM_ADDR idx, GM_ADDR dst, GM_ADDR tiling_ptr) {
  GlobalTensor<uint32_t> tilingGm;
  tilingGm.SetGlobalBuffer((__gm__ uint32_t *)tiling_ptr, 1);
  uint32_t n = tilingGm.GetValue(0);
  KernelGather<float> op;
  op.Init(src, idx, dst, n);
  op.Process();
}

extern "C" __global__ __aicore__ void coo_to_csr_indptr_int64(
    GM_ADDR rows, GM_ADDR indptr, GM_ADDR tiling_ptr) {
  GlobalTensor<uint32_t> tilingGm;
  tilingGm.SetGlobalBuffer((__gm__ uint32_t *)tiling_ptr, 4);
  uint32_t nnz = tilingGm.GetValue(0);
  uint32_t num_rows = tilingGm.GetValue(1);
  uint32_t blockIdx = tilingGm.GetValue(2);
  uint32_t blockNum = tilingGm.GetValue(3);

  KernelCooToCsrIndptr<int64_t> op;
  op.Init(rows, indptr, nnz, num_rows);
  op.Process(blockIdx, blockNum);
}
