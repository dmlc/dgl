/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cuda/coo_sort.cc
 * \brief Sort COO index
 */
#include <dgl/array.h>
#include "../../runtime/cuda/cuda_common.h"
#include "./utils.h"

namespace dgl {

using runtime::NDArray;

namespace aten {
namespace impl {

///////////////////////////// COOSort_ /////////////////////////////

template <DLDeviceType XPU, typename IdType>
void COOSort_(COOMatrix* coo, bool sort_column) {
  LOG(FATAL) << "Unreachable codes";
}

template <>
void COOSort_<kDLGPU, int32_t>(COOMatrix* coo, bool sort_column) {
  // TODO(minjie): Current implementation is based on cusparse which only supports
  //   int32_t. To support int64_t, we could use the Radix sort algorithm provided
  //   by CUB.
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  auto device = runtime::DeviceAPI::Get(coo->row->ctx);
  // allocate cusparse handle if needed
  if (!thr_entry->cusparse_handle) {
    CUSPARSE_CALL(cusparseCreate(&(thr_entry->cusparse_handle)));
  }
  CUSPARSE_CALL(cusparseSetStream(thr_entry->cusparse_handle, thr_entry->stream));


  NDArray row = coo->row;
  NDArray col = coo->col;
  if (!aten::COOHasData(*coo))
    coo->data = aten::Range(0, row->shape[0], row->dtype.bits, row->ctx);
  NDArray data = coo->data;
  int32_t* row_ptr = static_cast<int32_t*>(row->data);
  int32_t* col_ptr = static_cast<int32_t*>(col->data);
  int32_t* data_ptr = static_cast<int32_t*>(data->data);

  // sort row
  size_t workspace_size = 0;
  CUSPARSE_CALL(cusparseXcoosort_bufferSizeExt(
      thr_entry->cusparse_handle,
      coo->num_rows, coo->num_cols,
      row->shape[0],
      row_ptr,
      col_ptr,
      &workspace_size));
  void* workspace = device->AllocWorkspace(row->ctx, workspace_size);
  CUSPARSE_CALL(cusparseXcoosortByRow(
      thr_entry->cusparse_handle,
      coo->num_rows, coo->num_cols,
      row->shape[0],
      row_ptr,
      col_ptr,
      data_ptr,
      workspace));
  device->FreeWorkspace(row->ctx, workspace);

  if (sort_column) {
    // First create a row indptr array and then call csrsort
    int32_t* indptr = static_cast<int32_t*>(
        device->AllocWorkspace(row->ctx, (coo->num_rows + 1) * sizeof(int32_t)));
    CUSPARSE_CALL(cusparseXcoo2csr(
          thr_entry->cusparse_handle,
          row_ptr,
          row->shape[0],
          coo->num_rows,
          indptr,
          CUSPARSE_INDEX_BASE_ZERO));
    CUSPARSE_CALL(cusparseXcsrsort_bufferSizeExt(
          thr_entry->cusparse_handle,
          coo->num_rows,
          coo->num_cols,
          row->shape[0],
          indptr,
          col_ptr,
          &workspace_size));
    void* workspace = device->AllocWorkspace(row->ctx, workspace_size);
    cusparseMatDescr_t descr;
    CUSPARSE_CALL(cusparseCreateMatDescr(&descr));
    CUSPARSE_CALL(cusparseXcsrsort(
          thr_entry->cusparse_handle,
          coo->num_rows,
          coo->num_cols,
          row->shape[0],
          descr,
          indptr,
          col_ptr,
          data_ptr,
          workspace));
    CUSPARSE_CALL(cusparseDestroyMatDescr(descr));
    device->FreeWorkspace(row->ctx, workspace);
    device->FreeWorkspace(row->ctx, indptr);
  }

  coo->row_sorted = true;
  coo->col_sorted = sort_column;
}

template <>
void COOSort_<kDLGPU, int64_t>(COOMatrix* coo, bool sort_column) {
  // Always sort the COO to be both row and column sorted.
  IdArray pos = coo->row * coo->num_cols + coo->col;
  const auto& sorted = Sort(pos);
  coo->row = sorted.first / coo->num_cols;
  coo->col = sorted.first % coo->num_cols;
  if (aten::COOHasData(*coo))
    coo->data = IndexSelect(coo->data, sorted.second);
  else
    coo->data = AsNumBits(sorted.second, coo->row->dtype.bits);
  coo->row_sorted = coo->col_sorted = true;
}

template void COOSort_<kDLGPU, int32_t>(COOMatrix* coo, bool sort_column);
template void COOSort_<kDLGPU, int64_t>(COOMatrix* coo, bool sort_column);

///////////////////////////// COOIsSorted /////////////////////////////

template <typename IdType>
__global__ void _COOIsSortedKernel(
    const IdType* row, const IdType* col,
    int64_t nnz, int8_t* row_sorted, int8_t* col_sorted) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride_x = gridDim.x * blockDim.x;
  while (tx < nnz) {
    if (tx == 0) {
      row_sorted[0] = 1;
      col_sorted[0] = 1;
    } else {
      row_sorted[tx] = static_cast<int8_t>(row[tx - 1] <= row[tx]);
      col_sorted[tx] = static_cast<int8_t>(
          row[tx - 1] < row[tx] || col[tx - 1] <= col[tx]);
    }
    tx += stride_x;
  }
}

template <DLDeviceType XPU, typename IdType>
std::pair<bool, bool> COOIsSorted(COOMatrix coo) {
  const int64_t nnz = coo.row->shape[0];
  const auto& ctx = coo.row->ctx;
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  auto device = runtime::DeviceAPI::Get(ctx);
  // We allocate a workspace of 2*nnz bytes. It wastes a little bit memory but should
  // be fine.
  int8_t* row_flags = static_cast<int8_t*>(device->AllocWorkspace(ctx, nnz));
  int8_t* col_flags = static_cast<int8_t*>(device->AllocWorkspace(ctx, nnz));
  const int nt = cuda::FindNumThreads(nnz);
  const int nb = (nnz + nt - 1) / nt;
  CUDA_KERNEL_CALL(_COOIsSortedKernel, nb, nt, 0, thr_entry->stream,
      coo.row.Ptr<IdType>(), coo.col.Ptr<IdType>(),
      nnz, row_flags, col_flags);

  const bool row_sorted = cuda::AllTrue(row_flags, nnz, ctx);
  const bool col_sorted = row_sorted? cuda::AllTrue(col_flags, nnz, ctx) : false;

  device->FreeWorkspace(ctx, row_flags);
  device->FreeWorkspace(ctx, col_flags);

  return {row_sorted, col_sorted};
}

template std::pair<bool, bool> COOIsSorted<kDLGPU, int32_t>(COOMatrix coo);
template std::pair<bool, bool> COOIsSorted<kDLGPU, int64_t>(COOMatrix coo);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
