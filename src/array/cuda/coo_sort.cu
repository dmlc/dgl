/**
 *  Copyright (c) 2020 by Contributors
 * @file array/cuda/coo_sort.cc
 * @brief Sort COO index
 */
#include <dgl/array.h>

#include "../../c_api_common.h"
#include "../../runtime/cuda/cuda_common.h"
#include "./utils.h"

namespace dgl {

using runtime::NDArray;

namespace aten {
namespace impl {

///////////////////////////// COOSort_ /////////////////////////////

/**
 * @brief Encode row and column IDs into a single scalar per edge.
 *
 * @tparam IdType The type to encode as.
 * @param row The row (src) IDs per edge.
 * @param col The column (dst) IDs per edge.
 * @param nnz The number of edges.
 * @param col_bits The number of bits used to encode the destination. The row
 * information is packed into the remaining bits.
 * @param key The encoded edges (output).
 */
template <typename IdType>
__global__ void _COOEncodeEdgesKernel(
    const IdType* const row, const IdType* const col, const int64_t nnz,
    const int col_bits, IdType* const key) {
  int64_t tx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  if (tx < nnz) {
    key[tx] = row[tx] << col_bits | col[tx];
  }
}

/**
 * @brief Decode row and column IDs from the encoded edges.
 *
 * @tparam IdType The type the edges are encoded as.
 * @param key The encoded edges.
 * @param nnz The number of edges.
 * @param col_bits The number of bits used to store the column/dst ID.
 * @param row The row (src) IDs per edge (output).
 * @param col The col (dst) IDs per edge (output).
 */
template <typename IdType>
__global__ void _COODecodeEdgesKernel(
    const IdType* const key, const int64_t nnz, const int col_bits,
    IdType* const row, IdType* const col) {
  int64_t tx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  if (tx < nnz) {
    const IdType k = key[tx];
    row[tx] = k >> col_bits;
    col[tx] = k & ((1 << col_bits) - 1);
  }
}

template <DGLDeviceType XPU, typename IdType>
void COOSort_(COOMatrix* coo, bool sort_column) {
  cudaStream_t stream = runtime::getCurrentCUDAStream();
  const int row_bits = cuda::_NumberOfBits(coo->num_rows);

  const int64_t nnz = coo->row->shape[0];
  if (sort_column) {
    const int col_bits = cuda::_NumberOfBits(coo->num_cols);
    const int num_bits = row_bits + col_bits;

    const int nt = 256;
    const int nb = (nnz + nt - 1) / nt;
    CHECK(static_cast<int64_t>(nb) * nt >= nnz);

    IdArray pos = aten::NewIdArray(nnz, coo->row->ctx, coo->row->dtype.bits);

    CUDA_KERNEL_CALL(
        _COOEncodeEdgesKernel, nb, nt, 0, stream, coo->row.Ptr<IdType>(),
        coo->col.Ptr<IdType>(), nnz, col_bits, pos.Ptr<IdType>());

    auto sorted = Sort(pos, num_bits);

    CUDA_KERNEL_CALL(
        _COODecodeEdgesKernel, nb, nt, 0, stream, sorted.first.Ptr<IdType>(),
        nnz, col_bits, coo->row.Ptr<IdType>(), coo->col.Ptr<IdType>());

    if (aten::COOHasData(*coo))
      coo->data = IndexSelect(coo->data, sorted.second);
    else
      coo->data = AsNumBits(sorted.second, coo->row->dtype.bits);
    coo->row_sorted = coo->col_sorted = true;
  } else {
    const int num_bits = row_bits;

    auto sorted = Sort(coo->row, num_bits);

    coo->row = sorted.first;
    coo->col = IndexSelect(coo->col, sorted.second);

    if (aten::COOHasData(*coo))
      coo->data = IndexSelect(coo->data, sorted.second);
    else
      coo->data = AsNumBits(sorted.second, coo->row->dtype.bits);
    coo->row_sorted = true;
  }
}

template void COOSort_<kDGLCUDA, int32_t>(COOMatrix* coo, bool sort_column);
template void COOSort_<kDGLCUDA, int64_t>(COOMatrix* coo, bool sort_column);

///////////////////////////// COOIsSorted /////////////////////////////

template <typename IdType>
__global__ void _COOIsSortedKernel(
    const IdType* row, const IdType* col, int64_t nnz, int8_t* row_sorted,
    int8_t* col_sorted) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride_x = gridDim.x * blockDim.x;
  while (tx < nnz) {
    if (tx == 0) {
      row_sorted[0] = 1;
      col_sorted[0] = 1;
    } else {
      row_sorted[tx] = static_cast<int8_t>(row[tx - 1] <= row[tx]);
      col_sorted[tx] =
          static_cast<int8_t>(row[tx - 1] < row[tx] || col[tx - 1] <= col[tx]);
    }
    tx += stride_x;
  }
}

template <DGLDeviceType XPU, typename IdType>
std::pair<bool, bool> COOIsSorted(COOMatrix coo) {
  const int64_t nnz = coo.row->shape[0];
  const auto& ctx = coo.row->ctx;
  cudaStream_t stream = runtime::getCurrentCUDAStream();
  auto device = runtime::DeviceAPI::Get(ctx);
  // We allocate a workspace of 2*nnz bytes. It wastes a little bit memory but
  // should be fine.
  int8_t* row_flags = static_cast<int8_t*>(device->AllocWorkspace(ctx, nnz));
  int8_t* col_flags = static_cast<int8_t*>(device->AllocWorkspace(ctx, nnz));
  const int nt = cuda::FindNumThreads(nnz);
  const int nb = (nnz + nt - 1) / nt;
  CUDA_KERNEL_CALL(
      _COOIsSortedKernel, nb, nt, 0, stream, coo.row.Ptr<IdType>(),
      coo.col.Ptr<IdType>(), nnz, row_flags, col_flags);

  const bool row_sorted = cuda::AllTrue(row_flags, nnz, ctx);
  const bool col_sorted =
      row_sorted ? cuda::AllTrue(col_flags, nnz, ctx) : false;

  device->FreeWorkspace(ctx, row_flags);
  device->FreeWorkspace(ctx, col_flags);

  return {row_sorted, col_sorted};
}

template std::pair<bool, bool> COOIsSorted<kDGLCUDA, int32_t>(COOMatrix coo);
template std::pair<bool, bool> COOIsSorted<kDGLCUDA, int64_t>(COOMatrix coo);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
