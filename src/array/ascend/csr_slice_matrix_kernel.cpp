#include "kernel_operator.h"

using namespace AscendC;

// --- CSRSliceMatrix pass 1a: mark valid_col[cols[k]] = k ---
// valid_col is pre-initialized to -1 via aclrtMemsetAsync (DMA engine) before this kernel runs.
template <typename IdxT>
class KernelCsrSliceMatrixMark {
 public:
  __aicore__ inline KernelCsrSliceMatrixMark() {}

  __aicore__ inline void Init(
      GM_ADDR cols, GM_ADDR valid_col,
      uint32_t ncols, uint32_t num_cols) {
    cols_gm.SetGlobalBuffer((__gm__ IdxT *)cols, ncols);
    valid_col_gm.SetGlobalBuffer((__gm__ int64_t *)valid_col, num_cols);
    ncols_ = ncols;
  }

  __aicore__ inline void Process() {
    uint32_t block_id = GetBlockIdx();
    uint32_t block_num = GetBlockNum();
    uint32_t chunk = (ncols_ + block_num - 1) / block_num;
    uint32_t start = block_id * chunk;
    uint32_t end = (start + chunk > ncols_) ? ncols_ : start + chunk;
    for (uint32_t k = start; k < end; k++) {
      IdxT col = cols_gm.GetValue(k);
      valid_col_gm.SetValue(static_cast<uint32_t>(col), static_cast<int64_t>(k));
    }
  }

 private:
  GlobalTensor<IdxT> cols_gm;
  GlobalTensor<int64_t> valid_col_gm;
  uint32_t ncols_;
};

extern "C" __global__ __aicore__ void csr_slice_matrix_mark_valid_col_int32(
    GM_ADDR cols, GM_ADDR valid_col, GM_ADDR tiling_ptr) {
  GlobalTensor<uint32_t> tilingGm;
  tilingGm.SetGlobalBuffer((__gm__ uint32_t *)tiling_ptr, 2);
  uint32_t ncols = tilingGm.GetValue(0);
  uint32_t num_cols = tilingGm.GetValue(1);
  KernelCsrSliceMatrixMark<int32_t> op;
  op.Init(cols, valid_col, ncols, num_cols);
  op.Process();
}

extern "C" __global__ __aicore__ void csr_slice_matrix_mark_valid_col_int64(
    GM_ADDR cols, GM_ADDR valid_col, GM_ADDR tiling_ptr) {
  GlobalTensor<uint32_t> tilingGm;
  tilingGm.SetGlobalBuffer((__gm__ uint32_t *)tiling_ptr, 2);
  uint32_t ncols = tilingGm.GetValue(0);
  uint32_t num_cols = tilingGm.GetValue(1);
  KernelCsrSliceMatrixMark<int64_t> op;
  op.Init(cols, valid_col, ncols, num_cols);
  op.Process();
}

// --- CSRSliceMatrix pass 1c: prefix sum (count valid edges per row) ---
// Single-block: reads valid_col (already fully init+marked), writes ret_indptr
template <typename IdxT>
class KernelCsrSliceMatrixPrefix {
 public:
  __aicore__ inline KernelCsrSliceMatrixPrefix() {}

  __aicore__ inline void Init(
      GM_ADDR indptr, GM_ADDR indices,
      GM_ADDR rows, GM_ADDR valid_col,
      GM_ADDR ret_indptr,
      uint32_t nrows, uint32_t num_cols,
      uint32_t orig_nrows, uint32_t orig_nnz) {
    indptr_gm.SetGlobalBuffer((__gm__ IdxT *)indptr, orig_nrows + 1);
    indices_gm.SetGlobalBuffer((__gm__ IdxT *)indices, orig_nnz);
    rows_gm.SetGlobalBuffer((__gm__ IdxT *)rows, nrows);
    valid_col_gm.SetGlobalBuffer((__gm__ int64_t *)valid_col, num_cols);
    ret_indptr_gm.SetGlobalBuffer((__gm__ IdxT *)ret_indptr, nrows + 1);
    nrows_ = nrows;
  }

  __aicore__ inline void Process() {
    if (GetBlockIdx() != 0) return;
    IdxT total = 0;
    ret_indptr_gm.SetValue(0, 0);
    for (uint32_t i = 0; i < nrows_; i++) {
      IdxT row = rows_gm.GetValue(i);
      IdxT start = indptr_gm.GetValue(static_cast<uint32_t>(row));
      IdxT end = indptr_gm.GetValue(static_cast<uint32_t>(row) + 1);
      IdxT count = 0;
      for (IdxT p = start; p < end; p++) {
        IdxT col = indices_gm.GetValue(static_cast<uint32_t>(p));
        if (valid_col_gm.GetValue(static_cast<uint32_t>(col)) != -1) {
          count++;
        }
      }
      total += count;
      ret_indptr_gm.SetValue(i + 1, total);
    }
  }

 private:
  GlobalTensor<IdxT> indptr_gm;
  GlobalTensor<IdxT> indices_gm;
  GlobalTensor<IdxT> rows_gm;
  GlobalTensor<int64_t> valid_col_gm;
  GlobalTensor<IdxT> ret_indptr_gm;
  uint32_t nrows_;
};

extern "C" __global__ __aicore__ void csr_slice_matrix_prefix_int32(
    GM_ADDR indptr, GM_ADDR indices,
    GM_ADDR rows, GM_ADDR valid_col,
    GM_ADDR ret_indptr,
    GM_ADDR tiling_ptr) {
  GlobalTensor<uint32_t> tilingGm;
  tilingGm.SetGlobalBuffer((__gm__ uint32_t *)tiling_ptr, 4);
  uint32_t nrows = tilingGm.GetValue(0);
  uint32_t num_cols = tilingGm.GetValue(1);
  uint32_t orig_nrows = tilingGm.GetValue(2);
  uint32_t orig_nnz = tilingGm.GetValue(3);
  KernelCsrSliceMatrixPrefix<int32_t> op;
  op.Init(indptr, indices, rows, valid_col, ret_indptr, nrows, num_cols, orig_nrows, orig_nnz);
  op.Process();
}

extern "C" __global__ __aicore__ void csr_slice_matrix_prefix_int64(
    GM_ADDR indptr, GM_ADDR indices,
    GM_ADDR rows, GM_ADDR valid_col,
    GM_ADDR ret_indptr,
    GM_ADDR tiling_ptr) {
  GlobalTensor<uint32_t> tilingGm;
  tilingGm.SetGlobalBuffer((__gm__ uint32_t *)tiling_ptr, 4);
  uint32_t nrows = tilingGm.GetValue(0);
  uint32_t num_cols = tilingGm.GetValue(1);
  uint32_t orig_nrows = tilingGm.GetValue(2);
  uint32_t orig_nnz = tilingGm.GetValue(3);
  KernelCsrSliceMatrixPrefix<int64_t> op;
  op.Init(indptr, indices, rows, valid_col, ret_indptr, nrows, num_cols, orig_nrows, orig_nnz);
  op.Process();
}

// --- CSRSliceMatrix pass 2: copy valid edges with column remap ---
template <typename IdxT>
class KernelCsrSliceMatrixCopy {
 public:
  __aicore__ inline KernelCsrSliceMatrixCopy() {}

  __aicore__ inline void Init(
      GM_ADDR indptr, GM_ADDR indices, GM_ADDR data_or_null,
      GM_ADDR rows, GM_ADDR valid_col,
      GM_ADDR ret_indptr, GM_ADDR ret_indices, GM_ADDR ret_data,
      uint32_t nrows, int32_t has_data, uint32_t num_cols,
      uint32_t orig_nrows, uint32_t orig_nnz) {
    indptr_gm.SetGlobalBuffer((__gm__ IdxT *)indptr, orig_nrows + 1);
    indices_gm.SetGlobalBuffer((__gm__ IdxT *)indices, orig_nnz);
    if (has_data)
      data_gm.SetGlobalBuffer((__gm__ IdxT *)data_or_null, orig_nnz);
    rows_gm.SetGlobalBuffer((__gm__ IdxT *)rows, nrows);
    valid_col_gm.SetGlobalBuffer((__gm__ int64_t *)valid_col, num_cols);
    ret_indptr_gm.SetGlobalBuffer((__gm__ IdxT *)ret_indptr, nrows + 1);
    ret_indices_gm.SetGlobalBuffer((__gm__ IdxT *)ret_indices, nrows + 2);
    ret_data_gm.SetGlobalBuffer((__gm__ IdxT *)ret_data, nrows + 2);
    nrows_ = nrows;
    has_data_ = has_data;
  }

  __aicore__ inline void Process() {
    uint32_t block_id = GetBlockIdx();
    uint32_t block_num = GetBlockNum();
    uint32_t chunk = (nrows_ + block_num - 1) / block_num;
    uint32_t start = block_id * chunk;
    uint32_t end = (start + chunk > nrows_) ? nrows_ : start + chunk;
    for (uint32_t i = start; i < end; i++) {
      IdxT row = rows_gm.GetValue(i);
      IdxT src_off = indptr_gm.GetValue(static_cast<uint32_t>(row));
      IdxT src_end = indptr_gm.GetValue(static_cast<uint32_t>(row) + 1);
      IdxT dst_off = ret_indptr_gm.GetValue(i);
      for (IdxT p = src_off; p < src_end; p++) {
        IdxT col = indices_gm.GetValue(static_cast<uint32_t>(p));
        int64_t new_col = valid_col_gm.GetValue(static_cast<uint32_t>(col));
        if (new_col != -1) {
          uint32_t dp = static_cast<uint32_t>(dst_off);
          ret_indices_gm.SetValue(dp, static_cast<IdxT>(new_col));
          if (has_data_)
            ret_data_gm.SetValue(dp, data_gm.GetValue(static_cast<uint32_t>(p)));
          else
            ret_data_gm.SetValue(dp, p);
          dst_off++;
        }
      }
    }
  }

 private:
  GlobalTensor<IdxT> indptr_gm;
  GlobalTensor<IdxT> indices_gm;
  GlobalTensor<IdxT> data_gm;
  GlobalTensor<IdxT> rows_gm;
  GlobalTensor<int64_t> valid_col_gm;
  GlobalTensor<IdxT> ret_indptr_gm;
  GlobalTensor<IdxT> ret_indices_gm;
  GlobalTensor<IdxT> ret_data_gm;
  uint32_t nrows_;
  int32_t has_data_;
};

struct CsrSliceMatrixCopyTiling {
  uint32_t nrows;
  int32_t has_data;
  uint32_t num_cols;
  uint32_t orig_nrows;
  uint32_t orig_nnz;
};

extern "C" __global__ __aicore__ void csr_slice_matrix_copy_int32(
    GM_ADDR indptr, GM_ADDR indices, GM_ADDR data_or_null,
    GM_ADDR rows, GM_ADDR valid_col,
    GM_ADDR ret_indptr, GM_ADDR ret_indices, GM_ADDR ret_data,
    GM_ADDR tiling_ptr) {
  GlobalTensor<uint32_t> tilingGm;
  tilingGm.SetGlobalBuffer((__gm__ uint32_t *)tiling_ptr, 5);
  uint32_t nrows = tilingGm.GetValue(0);
  uint32_t has_data = tilingGm.GetValue(1);
  uint32_t num_cols = tilingGm.GetValue(2);
  uint32_t orig_nrows = tilingGm.GetValue(3);
  uint32_t orig_nnz = tilingGm.GetValue(4);
  KernelCsrSliceMatrixCopy<int32_t> op;
  op.Init(indptr, indices, data_or_null, rows, valid_col,
          ret_indptr, ret_indices, ret_data, nrows,
          static_cast<int32_t>(has_data), num_cols,
          orig_nrows, orig_nnz);
  op.Process();
}

extern "C" __global__ __aicore__ void csr_slice_matrix_copy_int64(
    GM_ADDR indptr, GM_ADDR indices, GM_ADDR data_or_null,
    GM_ADDR rows, GM_ADDR valid_col,
    GM_ADDR ret_indptr, GM_ADDR ret_indices, GM_ADDR ret_data,
    GM_ADDR tiling_ptr) {
  GlobalTensor<uint32_t> tilingGm;
  tilingGm.SetGlobalBuffer((__gm__ uint32_t *)tiling_ptr, 5);
  uint32_t nrows = tilingGm.GetValue(0);
  uint32_t has_data = tilingGm.GetValue(1);
  uint32_t num_cols = tilingGm.GetValue(2);
  uint32_t orig_nrows = tilingGm.GetValue(3);
  uint32_t orig_nnz = tilingGm.GetValue(4);
  KernelCsrSliceMatrixCopy<int64_t> op;
  op.Init(indptr, indices, data_or_null, rows, valid_col,
          ret_indptr, ret_indices, ret_data, nrows,
          static_cast<int32_t>(has_data), num_cols,
          orig_nrows, orig_nnz);
  op.Process();
}

