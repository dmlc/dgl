/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cuda/coo_sort.cc
 * \brief Sort COO index
 */
#include <dgl/array.h>
#include "../../runtime/cuda/cuda_common.h"
#include "../../c_api_common.h"
#include "./utils.h"

namespace dgl {

using runtime::NDArray;

namespace aten {
namespace impl {

///////////////////////////// COOSort_ /////////////////////////////

template <typename IdType>
__global__ void _COOMergeEdgesKernel(
    const IdType* const row, const IdType* const col,
    const int64_t nnz, const int col_bits, IdType * const key) {

  int64_t tx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  if (tx < nnz) {
    key[tx] = row[tx] << col_bits | col[tx];
  }
}

template <typename IdType>
__global__ void _COOSeparateEdgesKernel(
    const IdType* const key, const int64_t nnz, const int col_bits,
    IdType * const row, IdType * const col) {

  int64_t tx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  if (tx < nnz) {
    const IdType k = key[tx];
    row[tx] = k >> col_bits;
    col[tx] = k & ((1<<col_bits)-1);
  }
}



template<typename T>
int _NumberOfBits(const T& val)
{
  if (val == 0) {
    return 0;
  }

  int bits = 1;
  while (bits < sizeof(T)*8 && (1 << (bits-1)) < val) {
    ++bits;
  }

  CHECK(((1<<bits)-1 & val) == val);

  return bits;
}
    
template <DLDeviceType XPU, typename IdType>
void COOSort_(COOMatrix* coo, bool sort_column) {
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  const int row_bits = _NumberOfBits(coo->num_rows);

  const int64_t nnz = coo->row->shape[0];
  if (sort_column) {
    const int col_bits = _NumberOfBits(coo->num_cols);
    const int num_bits = row_bits + col_bits;

    const int nt = 256;
    const int nb = (nnz+nt-1)/nt;
    CHECK(static_cast<int64_t>(nb)*nt >= nnz);

    IdArray pos = aten::NewIdArray(nnz, coo->row->ctx, coo->row->dtype.bits);

    CUDA_KERNEL_CALL(_COOMergeEdgesKernel, nb, nt, 0, thr_entry->stream,
        coo->row.Ptr<IdType>(), coo->col.Ptr<IdType>(), 
        nnz, col_bits, pos.Ptr<IdType>());

    const auto& sorted = Sort(pos, num_bits);

    CUDA_KERNEL_CALL(_COOSeparateEdgesKernel, nb, nt, 0, thr_entry->stream,
        sorted.first.Ptr<IdType>(), nnz, col_bits,
        coo->row.Ptr<IdType>(), coo->col.Ptr<IdType>());

    if (aten::COOHasData(*coo))
      coo->data = IndexSelect(coo->data, sorted.second);
    else
      coo->data = AsNumBits(sorted.second, coo->row->dtype.bits);
    coo->row_sorted = coo->col_sorted = true;
  } else {
    const int num_bits = row_bits;

    const auto& sorted = Sort(coo->row, num_bits);

    coo->row = sorted.first;
    coo->col = IndexSelect(coo->col, sorted.second);

    if (aten::COOHasData(*coo))
      coo->data = IndexSelect(coo->data, sorted.second);
    else
      coo->data = AsNumBits(sorted.second, coo->row->dtype.bits);
    coo->row_sorted = true;
  }
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
