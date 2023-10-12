/**
 *  Copyright (c) 2023 by Contributors
 * @file cuda/index_select_impl.cu
 * @brief Index select operator implementation on CUDA.
 */
#include <torch/script.h>

#include <numeric>

#include "../index_select.h"
#include "./macro.h"

namespace graphbolt {
namespace ops {

template <typename DType, typename IdType>
__global__ void IndexSelectMultiKernel(
    const int64_t input_len, const DType* const input, const int64_t feat_size,
    const IdType* const index, const int64_t output_len, DType* const output) {
  int64_t out_row_index = blockIdx.x * blockDim.y + threadIdx.y;

  const int64_t stride = blockDim.y * gridDim.x;

  while (out_row_index < output_len) {
    int64_t column = threadIdx.x;
    const int64_t in_row = index[out_row_index];
    assert(in_row >= 0 && in_row < input_len);
    while (column < feat_size) {
      output[out_row_index * feat_size + column] =
          input[in_row * feat_size + column];
      column += blockDim.x;
    }
    out_row_index += stride;
  }
}

template <typename DType, typename IdType>
torch::Tensor UVAIndexSelectImpl_(torch::Tensor input, torch::Tensor index) {
  const int64_t input_len = input.size(0);
  const int64_t ret_len = index.size(0);
  const int64_t feat_size = std::accumulate(
      input.sizes().begin() + 1, input.sizes().end(), 1, std::multiplies<>());
  torch::Tensor ret = torch::empty(
      {ret_len, feat_size}, torch::TensorOptions()
                                .dtype(input.dtype())
                                .device(c10::DeviceType::CUDA));
  DType* input_ptr = input.data_ptr<DType>();
  IdType* index_ptr = index.data_ptr<IdType>();
  DType* ret_ptr = ret.data_ptr<DType>();
  cudaStream_t stream = 0;
  dim3 block(512, 1);
  // Find the smallest block size that can fit the feat_size.
  while (static_cast<int64_t>(block.x) >= 2 * feat_size) {
    block.x /= 2;
    block.y *= 2;
  }
  const dim3 grid((ret_len + block.y - 1) / block.y);
  GRAPHBOLT_CUDA_KERNEL_CALL(
      IndexSelectMultiKernel, grid, block, 0, stream, input_len, input_ptr,
      feat_size, index_ptr, ret_len, ret_ptr);
  auto ret_shape = std::vector({ret_len});
  ret_shape.insert(
      ret_shape.end(), input.sizes().begin() + 1, input.sizes().end());
  ret = ret.reshape(ret_shape);
  return ret;
}

/**
 * @brief Index select operator implementation on CUDA.
 *
 * The supporting input types are: float, double, int, int64_t.
 * The supporting index types are: int, int64_t.
 */
template <c10::DeviceType XPU>
torch::Tensor UVAIndexSelectImpl(torch::Tensor input, torch::Tensor index) {
  return AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Int, at::ScalarType::Long, input.scalar_type(),
      "UVAIndexSelectImpl", [&] {
        return AT_DISPATCH_INDEX_TYPES(
            index.scalar_type(), "UVAIndexSelectImpl", [&] {
              return UVAIndexSelectImpl_<scalar_t, index_t>(input, index);
            });
      });
}

template torch::Tensor UVAIndexSelectImpl<c10::DeviceType::CUDA>(
    torch::Tensor input, torch::Tensor index);

}  //  namespace ops
}  //  namespace graphbolt
