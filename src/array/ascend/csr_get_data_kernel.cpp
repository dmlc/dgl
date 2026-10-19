#include "kernel_operator.h"

using namespace AscendC;

template <typename IdxT>
class KernelCsrGetData {
 public:
  __aicore__ inline KernelCsrGetData() {}

  __aicore__ inline void Init(
      GM_ADDR indptr, GM_ADDR indices, GM_ADDR data,
      GM_ADDR rows, GM_ADDR cols, GM_ADDR out,
      uint32_t numRows, uint32_t numCols, uint32_t nnz,
      uint32_t n, int64_t filler) {
    indptr_gm.SetGlobalBuffer((__gm__ IdxT *)indptr, numRows + 1);
    indices_gm.SetGlobalBuffer((__gm__ IdxT *)indices, nnz);
    data_gm.SetGlobalBuffer((__gm__ IdxT *)data, nnz);
    rows_gm.SetGlobalBuffer((__gm__ IdxT *)rows, n);
    cols_gm.SetGlobalBuffer((__gm__ IdxT *)cols, n);
    out_gm.SetGlobalBuffer((__gm__ IdxT *)out, n);
    numRows_ = numRows;
    numCols_ = numCols;
    nnz_ = nnz;
    n_ = n;
    filler_ = filler;
  }

  __aicore__ inline void Process() {
    uint32_t block_id = GetBlockIdx();
    uint32_t block_num = GetBlockNum();
    uint32_t chunk = (n_ + block_num - 1) / block_num;
    uint32_t start = block_id * chunk;
    uint32_t end = (start + chunk > n_) ? n_ : start + chunk;
    for (uint32_t i = start; i < end; i++) {
      IdxT row_id = rows_gm.GetValue(i);
      IdxT col_id = cols_gm.GetValue(i);

      IdxT start_pos = indptr_gm.GetValue(static_cast<uint32_t>(row_id));
      IdxT end_pos = indptr_gm.GetValue(static_cast<uint32_t>(row_id) + 1);

      IdxT result = filler_;
      for (IdxT j = start_pos; j < end_pos; j++) {
        if (indices_gm.GetValue(static_cast<uint32_t>(j)) == col_id) {
          result = data_gm.GetValue(static_cast<uint32_t>(j));
          break;
        }
      }
      out_gm.SetValue(i, result);
    }
  }

 private:
  GlobalTensor<IdxT> indptr_gm;
  GlobalTensor<IdxT> indices_gm;
  GlobalTensor<IdxT> data_gm;
  GlobalTensor<IdxT> rows_gm;
  GlobalTensor<IdxT> cols_gm;
  GlobalTensor<IdxT> out_gm;
  uint32_t numRows_;
  uint32_t numCols_;
  uint32_t nnz_;
  uint32_t n_;
  int64_t filler_;
};

struct CsrGetDataTiling {
  uint32_t numRows;
  uint32_t numCols;
  uint32_t nnz;
  uint32_t n;
  int64_t filler;
};

extern "C" __global__ __aicore__ void csr_get_data_int32(
    GM_ADDR indptr, GM_ADDR indices, GM_ADDR data,
    GM_ADDR rows, GM_ADDR cols, GM_ADDR out,
    GM_ADDR tiling_ptr) {
  GlobalTensor<uint32_t> tilingGm;
  tilingGm.SetGlobalBuffer((__gm__ uint32_t *)tiling_ptr, 5);
  uint32_t numRows = tilingGm.GetValue(0);
  uint32_t numCols = tilingGm.GetValue(1);
  uint32_t nnz = tilingGm.GetValue(2);
  uint32_t n = tilingGm.GetValue(3);

  GlobalTensor<int64_t> tilingI64;
  tilingI64.SetGlobalBuffer((__gm__ int64_t *)((__gm__ uint8_t *)tiling_ptr + 16), 1);
  int64_t filler = tilingI64.GetValue(0);

  KernelCsrGetData<int32_t> op;
  op.Init(indptr, indices, data, rows, cols, out,
          numRows, numCols, nnz, n, filler);
  op.Process();
}

extern "C" __global__ __aicore__ void csr_get_data_int64(
    GM_ADDR indptr, GM_ADDR indices, GM_ADDR data,
    GM_ADDR rows, GM_ADDR cols, GM_ADDR out,
    GM_ADDR tiling_ptr) {
  GlobalTensor<uint32_t> tilingGm;
  tilingGm.SetGlobalBuffer((__gm__ uint32_t *)tiling_ptr, 5);
  uint32_t numRows = tilingGm.GetValue(0);
  uint32_t numCols = tilingGm.GetValue(1);
  uint32_t nnz = tilingGm.GetValue(2);
  uint32_t n = tilingGm.GetValue(3);

  GlobalTensor<int64_t> tilingI64;
  tilingI64.SetGlobalBuffer((__gm__ int64_t *)((__gm__ uint8_t *)tiling_ptr + 16), 1);
  int64_t filler = tilingI64.GetValue(0);

  KernelCsrGetData<int64_t> op;
  op.Init(indptr, indices, data, rows, cols, out,
          numRows, numCols, nnz, n, filler);
  op.Process();
}

template <typename IdxT>
class KernelCsrGetDataWithWeights {
 public:
  __aicore__ inline KernelCsrGetDataWithWeights() {}

  __aicore__ inline void Init(
      GM_ADDR indptr, GM_ADDR indices, GM_ADDR data,
      GM_ADDR rows, GM_ADDR cols, GM_ADDR weights,
      GM_ADDR out, uint32_t numRows, uint32_t numCols,
      uint32_t nnz, uint32_t n, float filler) {
    indptr_gm.SetGlobalBuffer((__gm__ IdxT *)indptr, numRows + 1);
    indices_gm.SetGlobalBuffer((__gm__ IdxT *)indices, nnz);
    data_gm.SetGlobalBuffer((__gm__ IdxT *)data, nnz);
    rows_gm.SetGlobalBuffer((__gm__ IdxT *)rows, n);
    cols_gm.SetGlobalBuffer((__gm__ IdxT *)cols, n);
    weights_gm.SetGlobalBuffer((__gm__ float *)weights, nnz);
    out_gm.SetGlobalBuffer((__gm__ float *)out, n);
    numRows_ = numRows;
    numCols_ = numCols;
    nnz_ = nnz;
    n_ = n;
    filler_ = filler;
  }

  __aicore__ inline void Process() {
    uint32_t block_id = GetBlockIdx();
    uint32_t block_num = GetBlockNum();
    uint32_t chunk = (n_ + block_num - 1) / block_num;
    uint32_t start = block_id * chunk;
    uint32_t end = (start + chunk > n_) ? n_ : start + chunk;
    for (uint32_t i = start; i < end; i++) {
      IdxT row_id = rows_gm.GetValue(i);
      IdxT col_id = cols_gm.GetValue(i);

      IdxT start_pos = indptr_gm.GetValue(static_cast<uint32_t>(row_id));
      IdxT end_pos = indptr_gm.GetValue(static_cast<uint32_t>(row_id) + 1);

      float result = filler_;
      for (IdxT j = start_pos; j < end_pos; j++) {
        if (indices_gm.GetValue(static_cast<uint32_t>(j)) == col_id) {
          result = weights_gm.GetValue(static_cast<uint32_t>(j));
          break;
        }
      }
      out_gm.SetValue(i, result);
    }
  }

 private:
  GlobalTensor<IdxT> indptr_gm;
  GlobalTensor<IdxT> indices_gm;
  GlobalTensor<IdxT> data_gm;
  GlobalTensor<IdxT> rows_gm;
  GlobalTensor<IdxT> cols_gm;
  GlobalTensor<float> weights_gm;
  GlobalTensor<float> out_gm;
  uint32_t numRows_;
  uint32_t numCols_;
  uint32_t nnz_;
  uint32_t n_;
  float filler_;
};

struct CsrGetDataWeightedTiling {
  uint32_t numRows;
  uint32_t numCols;
  uint32_t nnz;
  uint32_t n;
  float filler;
};

extern "C" __global__ __aicore__ void csr_get_data_weighted_int32(
    GM_ADDR indptr, GM_ADDR indices, GM_ADDR data,
    GM_ADDR rows, GM_ADDR cols, GM_ADDR weights,
    GM_ADDR out, GM_ADDR tiling_ptr) {
  GlobalTensor<uint32_t> tilingGm;
  tilingGm.SetGlobalBuffer((__gm__ uint32_t *)tiling_ptr, 5);
  uint32_t numRows = tilingGm.GetValue(0);
  uint32_t numCols = tilingGm.GetValue(1);
  uint32_t nnz = tilingGm.GetValue(2);
  uint32_t n = tilingGm.GetValue(3);

  GlobalTensor<float> tilingF;
  tilingF.SetGlobalBuffer((__gm__ float *)((__gm__ uint8_t *)tiling_ptr + 16), 1);
  float filler = tilingF.GetValue(0);

  KernelCsrGetDataWithWeights<int32_t> op;
  op.Init(indptr, indices, data, rows, cols, weights, out,
          numRows, numCols, nnz, n, filler);
  op.Process();
}

extern "C" __global__ __aicore__ void csr_get_data_weighted_int64(
    GM_ADDR indptr, GM_ADDR indices, GM_ADDR data,
    GM_ADDR rows, GM_ADDR cols, GM_ADDR weights,
    GM_ADDR out, GM_ADDR tiling_ptr) {
  GlobalTensor<uint32_t> tilingGm;
  tilingGm.SetGlobalBuffer((__gm__ uint32_t *)tiling_ptr, 5);
  uint32_t numRows = tilingGm.GetValue(0);
  uint32_t numCols = tilingGm.GetValue(1);
  uint32_t nnz = tilingGm.GetValue(2);
  uint32_t n = tilingGm.GetValue(3);

  GlobalTensor<float> tilingF;
  tilingF.SetGlobalBuffer((__gm__ float *)((__gm__ uint8_t *)tiling_ptr + 16), 1);
  float filler = tilingF.GetValue(0);

  KernelCsrGetDataWithWeights<int64_t> op;
  op.Init(indptr, indices, data, rows, cols, weights, out,
          numRows, numCols, nnz, n, filler);
  op.Process();
}

