/*!
 *  Copyright (c) 2021 by Contributors
 * \file array/cpu/array_index_select_uvm.cuh
 * \brief Array index select GPU kernel implementation
 */

#ifndef DGL_ARRAY_CUDA_ARRAY_INDEX_SELECT_UVM_CUH_
#define DGL_ARRAY_CUDA_ARRAY_INDEX_SELECT_UVM_CUH_

#define CACHE_LINE_SIZE 128

namespace dgl {
namespace aten {
namespace impl {

template <typename DType, typename IdType>
__global__ void IndexSelectMultiKernelAligned(
        const DType* const array,
        const int64_t num_feat,
        const IdType* const index,
        const int64_t length,
        DType* const out) {
  int64_t out_row = blockIdx.x*blockDim.y+threadIdx.y;

  const int64_t stride = blockDim.y*gridDim.x;

  while (out_row < length) {
    int64_t col = threadIdx.x;
    const int64_t in_row = index[out_row];
    const int64_t idx_offset =
      ((uint64_t)(&array[in_row*num_feat]) % CACHE_LINE_SIZE) / sizeof(DType);
    col = col - idx_offset;
    while (col < num_feat) {
      if (col >= 0)
        out[out_row*num_feat+col] = array[in_row*num_feat+col];
      col += blockDim.x;
    }
    out_row += stride;
  }
}

}  // namespace impl
}  // namespace aten
}  // namespace dgl

#endif
