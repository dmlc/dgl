/**
 *  Copyright (c) 2020 by Contributors
 * @file array/cuda/coo2csr.cc
 * @brief COO2CSR
 */
#include <dgl/array.h>

#include "../../runtime/cuda/cuda_common.h"
#include "./utils.h"

namespace dgl {

using runtime::NDArray;

namespace aten {
namespace impl {

template <DGLDeviceType XPU, typename IdType>
CSRMatrix COOToCSR(COOMatrix coo) {
  LOG(FATAL) << "Unreachable code.";
  return {};
}

template <>
CSRMatrix COOToCSR<kDGLCUDA, int32_t>(COOMatrix coo) {
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  cudaStream_t stream = runtime::getCurrentCUDAStream();
  // allocate cusparse handle if needed
  if (!thr_entry->cusparse_handle) {
    CUSPARSE_CALL(cusparseCreate(&(thr_entry->cusparse_handle)));
  }
  CUSPARSE_CALL(cusparseSetStream(thr_entry->cusparse_handle, stream));

  bool row_sorted = coo.row_sorted;
  bool col_sorted = coo.col_sorted;
  if (!row_sorted) {
    // we only need to sort the rows to perform conversion
    coo = COOSort(coo, false);
    col_sorted = coo.col_sorted;
  }

  const int64_t nnz = coo.row->shape[0];
  CHECK_NO_OVERFLOW(coo.row->dtype, nnz);
  // TODO(minjie): Many of our current implementation assumes that CSR must have
  //   a data array. This is a temporary workaround. Remove this after:
  //   - The old immutable graph implementation is deprecated.
  //   - The old binary reduce kernel is deprecated.
  if (!COOHasData(coo))
    coo.data = aten::Range(0, nnz, coo.row->dtype.bits, coo.row->ctx);

  NDArray indptr =
      aten::NewIdArray(coo.num_rows + 1, coo.row->ctx, coo.row->dtype.bits);
  int32_t* indptr_ptr = static_cast<int32_t*>(indptr->data);
  CUSPARSE_CALL(cusparseXcoo2csr(
      thr_entry->cusparse_handle, coo.row.Ptr<int32_t>(), nnz, coo.num_rows,
      indptr_ptr, CUSPARSE_INDEX_BASE_ZERO));

  return CSRMatrix(
      coo.num_rows, coo.num_cols, indptr, coo.col, coo.data, col_sorted);
}

/**
 * @brief Search for the insertion positions for needle in the hay.
 *
 * The hay is a list of sorted elements and the result is the insertion position
 * of each needle so that the insertion still gives sorted order.
 *
 * It essentially perform binary search to find upper bound for each needle
 * elements.
 *
 * For example:
 * hay = [0, 0, 1, 2, 2]
 * needle = [0, 1, 2, 3]
 * then,
 * out = [2, 3, 5, 5]
 */
template <typename IdType>
__global__ void _SortedSearchKernelUpperBound(
    const IdType* hay, int64_t hay_size, const IdType* needles,
    int64_t num_needles, IdType* pos) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride_x = gridDim.x * blockDim.x;
  while (tx < num_needles) {
    const IdType ele = needles[tx];
    // binary search
    IdType lo = 0, hi = hay_size;
    while (lo < hi) {
      IdType mid = (lo + hi) >> 1;
      if (hay[mid] <= ele) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    pos[tx] = lo;
    tx += stride_x;
  }
}

template <>
CSRMatrix COOToCSR<kDGLCUDA, int64_t>(COOMatrix coo) {
  const auto& ctx = coo.row->ctx;
  const auto nbits = coo.row->dtype.bits;
  cudaStream_t stream = runtime::getCurrentCUDAStream();
  bool row_sorted = coo.row_sorted;
  bool col_sorted = coo.col_sorted;
  if (!row_sorted) {
    coo = COOSort(coo, false);
    col_sorted = coo.col_sorted;
  }

  const int64_t nnz = coo.row->shape[0];
  // TODO(minjie): Many of our current implementation assumes that CSR must have
  //   a data array. This is a temporary workaround. Remove this after:
  //   - The old immutable graph implementation is deprecated.
  //   - The old binary reduce kernel is deprecated.
  if (!COOHasData(coo))
    coo.data = aten::Range(0, nnz, coo.row->dtype.bits, coo.row->ctx);

  IdArray rowids = Range(0, coo.num_rows, nbits, ctx);
  const int nt = cuda::FindNumThreads(coo.num_rows);
  const int nb = (coo.num_rows + nt - 1) / nt;
  IdArray indptr = Full(0, coo.num_rows + 1, nbits, ctx);
  CUDA_KERNEL_CALL(
      _SortedSearchKernelUpperBound, nb, nt, 0, stream, coo.row.Ptr<int64_t>(),
      nnz, rowids.Ptr<int64_t>(), coo.num_rows, indptr.Ptr<int64_t>() + 1);

  return CSRMatrix(
      coo.num_rows, coo.num_cols, indptr, coo.col, coo.data, col_sorted);
}

template CSRMatrix COOToCSR<kDGLCUDA, int32_t>(COOMatrix coo);
template CSRMatrix COOToCSR<kDGLCUDA, int64_t>(COOMatrix coo);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
