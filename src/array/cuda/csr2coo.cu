/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cuda/csr2coo.cc
 * \brief CSR2COO
 */
#include <dgl/array.h>

#include "../../runtime/cuda/cuda_common.h"
#include "./utils.h"

namespace dgl {

using runtime::NDArray;

namespace aten {
namespace impl {

template <DGLDeviceType XPU, typename IdType>
COOMatrix CSRToCOO(CSRMatrix csr) {
  LOG(FATAL) << "Unreachable codes";
  return {};
}

template <>
COOMatrix CSRToCOO<kDGLCUDA, int32_t>(CSRMatrix csr) {
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  cudaStream_t stream = runtime::getCurrentCUDAStream();
  // allocate cusparse handle if needed
  if (!thr_entry->cusparse_handle) {
    CUSPARSE_CALL(cusparseCreate(&(thr_entry->cusparse_handle)));
  }
  CUSPARSE_CALL(cusparseSetStream(thr_entry->cusparse_handle, stream));

  NDArray indptr = csr.indptr, indices = csr.indices, data = csr.data;
  const int32_t* indptr_ptr = static_cast<int32_t*>(indptr->data);
  NDArray row =
      aten::NewIdArray(indices->shape[0], indptr->ctx, indptr->dtype.bits);
  int32_t* row_ptr = static_cast<int32_t*>(row->data);

  CUSPARSE_CALL(cusparseXcsr2coo(
      thr_entry->cusparse_handle, indptr_ptr, indices->shape[0], csr.num_rows,
      row_ptr, CUSPARSE_INDEX_BASE_ZERO));

  return COOMatrix(
      csr.num_rows, csr.num_cols, row, indices, data, true, csr.sorted);
}

/*!
 * \brief Repeat elements
 * \param val Value to repeat
 * \param repeats Number of repeats for each value
 * \param pos The position of the output buffer to write the value.
 * \param out Output buffer.
 * \param length Number of values
 *
 * For example:
 * val = [3, 0, 1]
 * repeats = [1, 0, 2]
 * pos = [0, 1, 1]  # write to output buffer position 0, 1, 1
 * then,
 * out = [3, 1, 1]
 */
template <typename DType, typename IdType>
__global__ void _RepeatKernel(
    const DType* val, const IdType* pos, DType* out, int64_t n_row,
    int64_t length) {
  IdType tx = static_cast<IdType>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int stride_x = gridDim.x * blockDim.x;
  while (tx < length) {
    IdType i = dgl::cuda::_UpperBound(pos, n_row, tx) - 1;
    out[tx] = val[i];
    tx += stride_x;
  }
}

template <>
COOMatrix CSRToCOO<kDGLCUDA, int64_t>(CSRMatrix csr) {
  const auto& ctx = csr.indptr->ctx;
  cudaStream_t stream = runtime::getCurrentCUDAStream();

  const int64_t nnz = csr.indices->shape[0];
  const auto nbits = csr.indptr->dtype.bits;
  IdArray rowids = Range(0, csr.num_rows, nbits, ctx);
  IdArray ret_row = NewIdArray(nnz, ctx, nbits);

  const int nt = 256;
  const int nb = (nnz + nt - 1) / nt;
  CUDA_KERNEL_CALL(
      _RepeatKernel, nb, nt, 0, stream, rowids.Ptr<int64_t>(),
      csr.indptr.Ptr<int64_t>(), ret_row.Ptr<int64_t>(), csr.num_rows, nnz);

  return COOMatrix(
      csr.num_rows, csr.num_cols, ret_row, csr.indices, csr.data, true,
      csr.sorted);
}

template COOMatrix CSRToCOO<kDGLCUDA, int32_t>(CSRMatrix csr);
template COOMatrix CSRToCOO<kDGLCUDA, int64_t>(CSRMatrix csr);

template <DGLDeviceType XPU, typename IdType>
COOMatrix CSRToCOODataAsOrder(CSRMatrix csr) {
  LOG(FATAL) << "Unreachable codes";
  return {};
}

template <>
COOMatrix CSRToCOODataAsOrder<kDGLCUDA, int32_t>(CSRMatrix csr) {
  COOMatrix coo = CSRToCOO<kDGLCUDA, int32_t>(csr);
  if (aten::IsNullArray(coo.data)) return coo;

  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  auto device = runtime::DeviceAPI::Get(coo.row->ctx);
  cudaStream_t stream = runtime::getCurrentCUDAStream();
  // allocate cusparse handle if needed
  if (!thr_entry->cusparse_handle) {
    CUSPARSE_CALL(cusparseCreate(&(thr_entry->cusparse_handle)));
  }
  CUSPARSE_CALL(cusparseSetStream(thr_entry->cusparse_handle, stream));

  NDArray row = coo.row, col = coo.col, data = coo.data;
  int32_t* row_ptr = static_cast<int32_t*>(row->data);
  int32_t* col_ptr = static_cast<int32_t*>(col->data);
  int32_t* data_ptr = static_cast<int32_t*>(data->data);

  size_t workspace_size = 0;
  CUSPARSE_CALL(cusparseXcoosort_bufferSizeExt(
      thr_entry->cusparse_handle, coo.num_rows, coo.num_cols, row->shape[0],
      data_ptr, row_ptr, &workspace_size));
  void* workspace = device->AllocWorkspace(row->ctx, workspace_size);
  CUSPARSE_CALL(cusparseXcoosortByRow(
      thr_entry->cusparse_handle, coo.num_rows, coo.num_cols, row->shape[0],
      data_ptr, row_ptr, col_ptr, workspace));
  device->FreeWorkspace(row->ctx, workspace);

  // The row and column field have already been reordered according
  // to data, thus the data field will be deprecated.
  coo.data = aten::NullArray();
  coo.row_sorted = false;
  coo.col_sorted = false;
  return coo;
}

template <>
COOMatrix CSRToCOODataAsOrder<kDGLCUDA, int64_t>(CSRMatrix csr) {
  COOMatrix coo = CSRToCOO<kDGLCUDA, int64_t>(csr);
  if (aten::IsNullArray(coo.data)) return coo;
  const auto& sorted = Sort(coo.data);

  coo.row = IndexSelect(coo.row, sorted.second);
  coo.col = IndexSelect(coo.col, sorted.second);

  // The row and column field have already been reordered according
  // to data, thus the data field will be deprecated.
  coo.data = aten::NullArray();
  coo.row_sorted = false;
  coo.col_sorted = false;
  return coo;
}

template COOMatrix CSRToCOODataAsOrder<kDGLCUDA, int32_t>(CSRMatrix csr);
template COOMatrix CSRToCOODataAsOrder<kDGLCUDA, int64_t>(CSRMatrix csr);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
