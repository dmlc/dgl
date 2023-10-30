/**
 *  Copyright (c) 2023 by Contributors
 * @file cuda/index_select_impl.cu
 * @brief Index select operator implementation on CUDA.
 */
#include <c10/cuda/CUDAException.h>
#include <torch/script.h>

#include <numeric>

#include "../index_select.h"
#include "./utils.h"

namespace graphbolt {
namespace ops {

/** @brief Index select operator implementation for feature size 1. */
template <typename DType, typename IdType>
__global__ void IndexSelectSingleKernel(
    const DType* input, const int64_t input_len, const IdType* index,
    const int64_t output_len, DType* output,
    const int64_t* permutation = nullptr) {
  int64_t out_row_index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  while (out_row_index < output_len) {
    assert(index[out_row_index] >= 0 && index[out_row_index] < input_len);
    const auto out_row =
        permutation ? permutation[out_row_index] : out_row_index;
    output[out_row] = input[index[out_row_index]];
    out_row_index += stride;
  }
}

/**
 * @brief Index select operator implementation for feature size > 1.
 */
template <typename DType, typename IdType>
__global__ void IndexSelectMultiKernel(
    const DType* const input, const int64_t input_len,
    const int64_t feature_size, const IdType* const index,
    const int64_t output_len, DType* const output,
    const int64_t* permutation = nullptr) {
  int64_t out_row_index = blockIdx.x * blockDim.y + threadIdx.y;

  const int64_t stride = blockDim.y * gridDim.x;

  while (out_row_index < output_len) {
    int64_t column = threadIdx.x;
    const int64_t in_row = index[out_row_index];
    assert(in_row >= 0 && in_row < input_len);
    const auto out_row =
        permutation ? permutation[out_row_index] : out_row_index;
    while (column < feature_size) {
      output[out_row * feature_size + column] =
          input[in_row * feature_size + column];
      column += blockDim.x;
    }
    out_row_index += stride;
  }
}

/**
 * @brief Index select operator implementation for feature size > 1.
 *
 * @note This is a cross-device access version of IndexSelectMultiKernel. Since
 * the memory access over PCIe is more sensitive to the data access aligment
 * (cacheline), we need a separate version here.
 */
template <typename DType, typename IdType>
__global__ void IndexSelectMultiKernelAligned(
    const DType* const input, const int64_t input_len,
    const int64_t feature_size, const IdType* const index,
    const int64_t output_len, DType* const output,
    const int64_t* permutation = nullptr) {
  int64_t out_row_index = blockIdx.x * blockDim.y + threadIdx.y;

  const int64_t stride = blockDim.y * gridDim.x;

  while (out_row_index < output_len) {
    int64_t col = threadIdx.x;
    const int64_t in_row = index[out_row_index];
    assert(in_row >= 0 && in_row < input_len);
    const int64_t idx_offset =
        ((uint64_t)(&input[in_row * feature_size]) % GPU_CACHE_LINE_SIZE) /
        sizeof(DType);
    col = col - idx_offset;
    const auto out_row =
        permutation ? permutation[out_row_index] : out_row_index;
    while (col < feature_size) {
      if (col >= 0)
        output[out_row * feature_size + col] =
            input[in_row * feature_size + col];
      col += blockDim.x;
    }
    out_row_index += stride;
  }
}

template <typename DType, typename IdType>
torch::Tensor UVAIndexSelectImpl_(torch::Tensor input, torch::Tensor index) {
  const int64_t input_len = input.size(0);
  const int64_t return_len = index.size(0);
  const int64_t feature_size = std::accumulate(
      input.sizes().begin() + 1, input.sizes().end(), 1, std::multiplies<>());
  torch::Tensor ret = torch::empty(
      {return_len, feature_size}, torch::TensorOptions()
                                      .dtype(input.dtype())
                                      .device(c10::DeviceType::CUDA));
  DType* input_ptr = input.data_ptr<DType>();
  DType* ret_ptr = ret.data_ptr<DType>();

  // Sort the index to improve the memory access pattern.
  torch::Tensor sorted_index, permutation;
  std::tie(sorted_index, permutation) = torch::sort(index);
  const IdType* index_sorted_ptr = sorted_index.data_ptr<IdType>();
  const int64_t* permutation_ptr = permutation.data_ptr<int64_t>();

  cudaStream_t stream = 0;

  if (feature_size == 1) {
    const int num_threads = cuda::FindNumThreads(return_len);
    const int num_blocks = (return_len + num_threads - 1) / num_threads;
    IndexSelectSingleKernel<<<num_blocks, num_threads, 0, stream>>>(
        input_ptr, input_len, index_sorted_ptr, return_len, ret_ptr,
        permutation_ptr);
  } else {
    dim3 block(512, 1);
    while (static_cast<int64_t>(block.x) >= 2 * feature_size) {
      block.x >>= 1;
      block.y <<= 1;
    }
    const dim3 grid((return_len + block.y - 1) / block.y);
    if (feature_size * sizeof(DType) < 2 * GPU_CACHE_LINE_SIZE) {
      IndexSelectMultiKernel<<<grid, block, 0, stream>>>(
          input_ptr, input_len, feature_size, index_sorted_ptr, return_len,
          ret_ptr, permutation_ptr);
    } else {
      IndexSelectMultiKernelAligned<<<grid, block, 0, stream>>>(
          input_ptr, input_len, feature_size, index_sorted_ptr, return_len,
          ret_ptr, permutation_ptr);
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  auto return_shape = std::vector<int64_t>({return_len});
  return_shape.insert(
      return_shape.end(), input.sizes().begin() + 1, input.sizes().end());
  ret = ret.reshape(return_shape);
  return ret;
}

/**
 * @brief UVA index select operator implementation on CUDA.
 *
 * The supporting input types are: float, double, int, int64_t.
 * The supporting index types are: int, int64_t.
 */
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

}  //  namespace ops
}  //  namespace graphbolt
