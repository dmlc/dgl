#include "kernel_operator.h"

using namespace AscendC;

// --- CSRSliceRows (scalar start/end) ---
// Input: indptr (numRows+1), start, num_rows (end-start)
// Output: ret_indptr (num_rows+1) where ret_indptr[i] = indptr[start+i] - indptr[start]

template <typename IdxT>
class KernelCsrSliceRowsScalar {
 public:
  __aicore__ inline KernelCsrSliceRowsScalar() {}

  __aicore__ inline void Init(
      GM_ADDR indptr, GM_ADDR ret_indptr,
      uint32_t start, uint32_t num_rows) {
    indptr_gm.SetGlobalBuffer((__gm__ IdxT *)indptr, start + num_rows + 1);
    ret_indptr_gm.SetGlobalBuffer((__gm__ IdxT *)ret_indptr, num_rows + 1);
    start_ = start;
    num_rows_ = num_rows;
  }

  __aicore__ inline void Process() {
    IdxT base = indptr_gm.GetValue(start_);
    for (uint32_t i = 0; i <= num_rows_; i++) {
      IdxT val = indptr_gm.GetValue(start_ + i);
      ret_indptr_gm.SetValue(i, val - base);
    }
  }

 private:
  GlobalTensor<IdxT> indptr_gm;
  GlobalTensor<IdxT> ret_indptr_gm;
  uint32_t start_;
  uint32_t num_rows_;
};

struct CsrSliceRowsScalarTiling {
  uint32_t start;
  uint32_t num_rows;
};

extern "C" __global__ __aicore__ void csr_slice_rows_scalar_int32(
    GM_ADDR indptr, GM_ADDR ret_indptr, GM_ADDR tiling_ptr) {
  GlobalTensor<uint32_t> tilingGm;
  tilingGm.SetGlobalBuffer((__gm__ uint32_t *)tiling_ptr, 2);
  uint32_t start = tilingGm.GetValue(0);
  uint32_t num_rows = tilingGm.GetValue(1);
  KernelCsrSliceRowsScalar<int32_t> op;
  op.Init(indptr, ret_indptr, start, num_rows);
  op.Process();
}

extern "C" __global__ __aicore__ void csr_slice_rows_scalar_int64(
    GM_ADDR indptr, GM_ADDR ret_indptr, GM_ADDR tiling_ptr) {
  GlobalTensor<uint32_t> tilingGm;
  tilingGm.SetGlobalBuffer((__gm__ uint32_t *)tiling_ptr, 2);
  uint32_t start = tilingGm.GetValue(0);
  uint32_t num_rows = tilingGm.GetValue(1);
  KernelCsrSliceRowsScalar<int64_t> op;
  op.Init(indptr, ret_indptr, start, num_rows);
  op.Process();
}

// --- CSRSliceRows (NDArray rows) ---
// Phase 1: prefix sum -> ret_indptr
// Phase 2: copy indices/data -> ret_indices/ret_data
// Both phases in a single kernel call.
// Output arrays must be pre-allocated with worst-case nnz (csr.indices->shape[0]).

template <typename IdxT>
class KernelCsrSliceRows {
 public:
  __aicore__ inline KernelCsrSliceRows() {}

  __aicore__ inline void Init(
      GM_ADDR indptr, GM_ADDR indices, GM_ADDR data_or_null,
      GM_ADDR rows, GM_ADDR ret_indptr,
      GM_ADDR ret_indices, GM_ADDR ret_data,
      uint32_t n, uint32_t num_cols, int32_t has_data) {
    indptr_gm.SetGlobalBuffer((__gm__ IdxT *)indptr, n + 2);
    indices_gm.SetGlobalBuffer((__gm__ IdxT *)indices, n + 1);
    if (has_data)
      data_gm.SetGlobalBuffer((__gm__ IdxT *)data_or_null, n + 1);
    rows_gm.SetGlobalBuffer((__gm__ IdxT *)rows, n);
    ret_indptr_gm.SetGlobalBuffer((__gm__ IdxT *)ret_indptr, n + 1);
    ret_indices_gm.SetGlobalBuffer((__gm__ IdxT *)ret_indices, n + 1);
    ret_data_gm.SetGlobalBuffer((__gm__ IdxT *)ret_data, n + 1);
    n_ = n;
    num_cols_ = num_cols;
    has_data_ = has_data;
  }

  __aicore__ inline void Process() {
    // Phase 1: prefix sum -> ret_indptr
    IdxT total = 0;
    ret_indptr_gm.SetValue(0, 0);
    for (uint32_t i = 0; i < n_; i++) {
      IdxT row = rows_gm.GetValue(i);
      IdxT start = indptr_gm.GetValue(static_cast<uint32_t>(row));
      IdxT end = indptr_gm.GetValue(static_cast<uint32_t>(row) + 1);
      total += (end - start);
      ret_indptr_gm.SetValue(i + 1, total);
    }

    // Phase 2: copy indices and data
    for (uint32_t i = 0; i < n_; i++) {
      IdxT row = rows_gm.GetValue(i);
      IdxT src_off = indptr_gm.GetValue(static_cast<uint32_t>(row));
      IdxT dst_off = ret_indptr_gm.GetValue(i);
      IdxT next_off = ret_indptr_gm.GetValue(i + 1);
      uint32_t len = static_cast<uint32_t>(next_off - dst_off);
      for (uint32_t j = 0; j < len; j++) {
        IdxT val = indices_gm.GetValue(static_cast<uint32_t>(src_off) + j);
        ret_indices_gm.SetValue(static_cast<uint32_t>(dst_off) + j, val);
        if (has_data_) {
          ret_data_gm.SetValue(
              static_cast<uint32_t>(dst_off) + j,
              data_gm.GetValue(static_cast<uint32_t>(src_off) + j));
        } else {
          ret_data_gm.SetValue(
              static_cast<uint32_t>(dst_off) + j,
              src_off + static_cast<IdxT>(j));
        }
      }
    }
  }

 private:
  GlobalTensor<IdxT> indptr_gm;
  GlobalTensor<IdxT> indices_gm;
  GlobalTensor<IdxT> data_gm;
  GlobalTensor<IdxT> rows_gm;
  GlobalTensor<IdxT> ret_indptr_gm;
  GlobalTensor<IdxT> ret_indices_gm;
  GlobalTensor<IdxT> ret_data_gm;
  uint32_t n_;
  uint32_t num_cols_;
  int32_t has_data_;
};

struct CsrSliceRowsTiling {
  uint32_t n;
  uint32_t num_cols;
  int32_t has_data;
};

extern "C" __global__ __aicore__ void csr_slice_rows_int32(
    GM_ADDR indptr, GM_ADDR indices, GM_ADDR data_or_null,
    GM_ADDR rows, GM_ADDR ret_indptr,
    GM_ADDR ret_indices, GM_ADDR ret_data,
    GM_ADDR tiling_ptr) {
  GlobalTensor<uint32_t> tilingGm;
  tilingGm.SetGlobalBuffer((__gm__ uint32_t *)tiling_ptr, 3);
  uint32_t n = tilingGm.GetValue(0);
  uint32_t num_cols = tilingGm.GetValue(1);
  uint32_t has_data = tilingGm.GetValue(2);
  KernelCsrSliceRows<int32_t> op;
  op.Init(indptr, indices, data_or_null, rows,
          ret_indptr, ret_indices, ret_data,
          n, num_cols, static_cast<int32_t>(has_data));
  op.Process();
}

extern "C" __global__ __aicore__ void csr_slice_rows_int64(
    GM_ADDR indptr, GM_ADDR indices, GM_ADDR data_or_null,
    GM_ADDR rows, GM_ADDR ret_indptr,
    GM_ADDR ret_indices, GM_ADDR ret_data,
    GM_ADDR tiling_ptr) {
  GlobalTensor<uint32_t> tilingGm;
  tilingGm.SetGlobalBuffer((__gm__ uint32_t *)tiling_ptr, 3);
  uint32_t n = tilingGm.GetValue(0);
  uint32_t num_cols = tilingGm.GetValue(1);
  uint32_t has_data = tilingGm.GetValue(2);
  KernelCsrSliceRows<int64_t> op;
  op.Init(indptr, indices, data_or_null, rows,
          ret_indptr, ret_indices, ret_data,
          n, num_cols, static_cast<int32_t>(has_data));
  op.Process();
}

