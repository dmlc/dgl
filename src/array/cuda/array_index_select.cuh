/**
 *  Copyright (c) 2021-2022 by Contributors
 * @file array/cuda/array_index_select.cuh
 * @brief Array index select GPU kernel implementation
 */

#ifndef DGL_ARRAY_CUDA_ARRAY_INDEX_SELECT_CUH_
#define DGL_ARRAY_CUDA_ARRAY_INDEX_SELECT_CUH_

namespace dgl {
namespace aten {
namespace impl {

template <typename DType, typename IdType>
__global__ void IndexSelectSingleKernel(
    const DType* array, const IdType* index, const int64_t length,
    const int64_t arr_len, DType* out, const int64_t* perm = nullptr) {
  int64_t tx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride_x = gridDim.x * blockDim.x;
  while (tx < length) {
    assert(index[tx] >= 0 && index[tx] < arr_len);
    const auto out_row = perm ? perm[tx] : tx;
    out[out_row] = array[index[tx]];
    tx += stride_x;
  }
}

template <typename DType, typename IdType>
__global__ void IndexSelectMultiKernel(
    const DType* const array, const int64_t num_feat, const IdType* const index,
    const int64_t length, const int64_t arr_len, DType* const out,
    const int64_t* perm = nullptr) {
  int64_t out_row_index = blockIdx.x * blockDim.y + threadIdx.y;

  const int64_t stride = blockDim.y * gridDim.x;

  while (out_row_index < length) {
    int64_t col = threadIdx.x;
    const int64_t in_row = index[out_row_index];
    assert(in_row >= 0 && in_row < arr_len);
    const auto out_row = perm ? perm[out_row_index] : out_row_index;
    while (col < num_feat) {
      out[out_row * num_feat + col] = array[in_row * num_feat + col];
      col += blockDim.x;
    }
    out_row_index += stride;
  }
}

template <typename DType, typename IdType>
__global__ void IndexScatterSingleKernel(
    const DType* array, const IdType* index, const int64_t length,
    const int64_t arr_len, DType* out) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride_x = gridDim.x * blockDim.x;
  while (tx < length) {
    assert(index[tx] >= 0 && index[tx] < arr_len);
    out[index[tx]] = array[tx];
    tx += stride_x;
  }
}

template <typename DType, typename IdType>
__global__ void IndexScatterMultiKernel(
    const DType* const array, const int64_t num_feat, const IdType* const index,
    const int64_t length, const int64_t arr_len, DType* const out) {
  int64_t in_row = blockIdx.x * blockDim.y + threadIdx.y;

  const int64_t stride = blockDim.y * gridDim.x;

  while (in_row < length) {
    int64_t col = threadIdx.x;
    const int64_t out_row = index[in_row];
    assert(out_row >= 0 && out_row < arr_len);
    while (col < num_feat) {
      out[out_row * num_feat + col] = array[in_row * num_feat + col];
      col += blockDim.x;
    }
    in_row += stride;
  }
}

}  // namespace impl
}  // namespace aten
}  // namespace dgl

#endif  // DGL_ARRAY_CUDA_ARRAY_INDEX_SELECT_CUH_
