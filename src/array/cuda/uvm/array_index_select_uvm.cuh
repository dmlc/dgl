/**
 *  Copyright (c) 2021 by Contributors
 * @file array/cpu/array_index_select_uvm.cuh
 * @brief Array index select GPU kernel implementation
 */

#ifndef DGL_ARRAY_CUDA_UVM_ARRAY_INDEX_SELECT_UVM_CUH_
#define DGL_ARRAY_CUDA_UVM_ARRAY_INDEX_SELECT_UVM_CUH_

#define CACHE_LINE_SIZE 128

namespace dgl {
namespace aten {
namespace impl {

/**
 *  This is a cross-device access version of IndexSelectMultiKernel.
 *  Since the memory access over PCIe is more sensitive to the
 *  data access aligment (cacheline), we need a separate version here.
 */
template <typename DType, typename IdType>
__global__ void IndexSelectMultiKernelAligned(
    const DType* const array, const int64_t num_feat, const IdType* const index,
    const int64_t length, const int64_t arr_len, DType* const out,
    const int64_t* perm = nullptr) {
  int64_t out_row_index = blockIdx.x * blockDim.y + threadIdx.y;

  const int64_t stride = blockDim.y * gridDim.x;

  while (out_row_index < length) {
    int64_t col = threadIdx.x;
    const int64_t in_row = index[out_row_index];
    assert(in_row >= 0 && in_row < arr_len);
    const int64_t idx_offset =
        ((uint64_t)(&array[in_row * num_feat]) % CACHE_LINE_SIZE) /
        sizeof(DType);
    col = col - idx_offset;
    const auto out_row = perm ? perm[out_row_index] : out_row_index;
    while (col < num_feat) {
      if (col >= 0)
        out[out_row * num_feat + col] = array[in_row * num_feat + col];
      col += blockDim.x;
    }
    out_row_index += stride;
  }
}

}  // namespace impl
}  // namespace aten
}  // namespace dgl

#endif  // DGL_ARRAY_CUDA_UVM_ARRAY_INDEX_SELECT_UVM_CUH_
