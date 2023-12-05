/**
 *  Copyright (c) 2023 by Contributors
 * @file cuda/index_select_impl.cu
 * @brief Index select operator implementation on CUDA.
 */
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAStream.h>

#include <cub/cub.cuh>
#include <numeric>

#include "../index_select.h"
#include "./common.h"
#include "./utils.h"

namespace graphbolt {
namespace ops {

std::pair<torch::Tensor, torch::Tensor> Sort(
    torch::Tensor input, int num_bits) {
  int64_t num_items = input.size(0);
  // We utilize int64_t for the values array. (torch::kLong == int64_t)
  auto original_idx =
      torch::arange(num_items, input.options().dtype(torch::kLong));
  auto sorted_array = torch::empty_like(input);
  auto sorted_idx = torch::empty_like(original_idx);
  cuda::CUDAWorkspaceAllocator allocator;
  AT_DISPATCH_INDEX_TYPES(
      input.scalar_type(), "SortImpl", ([&] {
        using IdType = index_t;
        const auto input_keys = input.data_ptr<index_t>();
        const int64_t* input_values = original_idx.data_ptr<int64_t>();
        IdType* sorted_keys = sorted_array.data_ptr<index_t>();
        int64_t* sorted_values = sorted_idx.data_ptr<int64_t>();
        cudaStream_t stream = torch::cuda::getDefaultCUDAStream();
        if (num_bits == 0) {
          num_bits = sizeof(index_t) * 8;
        }
        size_t workspace_size = 0;
        CUDA_CALL(cub::DeviceRadixSort::SortPairs(
            nullptr, workspace_size, input_keys, sorted_keys, input_values,
            sorted_values, num_items, 0, num_bits, stream));
        auto temp = allocator.AllocateStorage<char>(workspace_size);
        CUDA_CALL(cub::DeviceRadixSort::SortPairs(
            temp.get(), workspace_size, input_keys, sorted_keys, input_values,
            sorted_values, num_items, 0, num_bits, stream));
      }));
  return std::make_pair(sorted_array, sorted_idx);
}

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
  const int64_t original_feature_size = std::accumulate(
      input.sizes().begin() + 1, input.sizes().end(), 1ll, std::multiplies<>());
  const auto aligned_feature_size =
      input.element_size() * original_feature_size / sizeof(DType);
  torch::Tensor ret = torch::empty(
      {return_len, original_feature_size}, torch::TensorOptions()
                                               .dtype(input.dtype())
                                               .device(c10::DeviceType::CUDA));
  DType* input_ptr = reinterpret_cast<DType*>(input.data_ptr());
  DType* ret_ptr = reinterpret_cast<DType*>(ret.data_ptr());

  // Sort the index to improve the memory access pattern.
  torch::Tensor sorted_index, permutation;
  std::tie(sorted_index, permutation) =
      Sort(index, cuda::NumberOfBits(input_len));
  const IdType* index_sorted_ptr = sorted_index.data_ptr<IdType>();
  const int64_t* permutation_ptr = permutation.data_ptr<int64_t>();

  cudaStream_t stream = torch::cuda::getDefaultCUDAStream();

  if (aligned_feature_size == 1) {
    // Use a single thread to process each output row to avoid wasting threads.
    const int num_threads = cuda::FindNumThreads(return_len);
    const int num_blocks = (return_len + num_threads - 1) / num_threads;
    CUDA_KERNEL_CALL(
        IndexSelectSingleKernel, num_blocks, num_threads, 0, stream, input_ptr,
        input_len, index_sorted_ptr, return_len, ret_ptr, permutation_ptr);
  } else {
    dim3 block(512, 1);
    while (static_cast<int64_t>(block.x) >= 2 * aligned_feature_size) {
      block.x >>= 1;
      block.y <<= 1;
    }
    const dim3 grid((return_len + block.y - 1) / block.y);
    if (aligned_feature_size * sizeof(DType) <= GPU_CACHE_LINE_SIZE) {
      // When feature size is smaller than GPU cache line size, use unaligned
      // version for less SM usage, which is more resource efficient.
      CUDA_KERNEL_CALL(
          IndexSelectMultiKernel, grid, block, 0, stream, input_ptr, input_len,
          aligned_feature_size, index_sorted_ptr, return_len, ret_ptr,
          permutation_ptr);
    } else {
      // Use aligned version to improve the memory access pattern.
      CUDA_KERNEL_CALL(
          IndexSelectMultiKernelAligned, grid, block, 0, stream, input_ptr,
          input_len, aligned_feature_size, index_sorted_ptr, return_len,
          ret_ptr, permutation_ptr);
    }
  }

  auto return_shape = std::vector<int64_t>({return_len});
  return_shape.insert(
      return_shape.end(), input.sizes().begin() + 1, input.sizes().end());
  ret = ret.reshape(return_shape);
  return ret;
}

/**
 * @brief UVA index select operator implementation on CUDA.
 *
 * All basic torch types are supported for input.
 * The supporting index types are: int, int64_t.
 */
torch::Tensor UVAIndexSelectImpl(torch::Tensor input, torch::Tensor index) {
  return AT_DISPATCH_INDEX_TYPES(
      index.scalar_type(), "UVAIndexSelectImpl", ([&] {
        const auto ptr = (size_t)input.data_ptr();
        const int64_t feature_size = std::accumulate(
            input.sizes().begin() + 1, input.sizes().end(), 1ll,
            std::multiplies<>());
        // We perform the copy with datatype of size powers of 2, and the
        // maximum data type we use has 16 bytes. We check the alignment of the
        // pointer and the feature dimensionality to determine the largest
        // type to use for the copy to minimize the number of CUDA threads used.
        // Alignment denotes the maximum suitable alignment and datatype size
        // for the copies.
        const int aligned_access_size =
            std::gcd(16, std::gcd(ptr, input.element_size() * feature_size));
        switch (aligned_access_size) {
          case 1:
            return UVAIndexSelectImpl_<uint8_t, index_t>(input, index);
          case 2:
            return UVAIndexSelectImpl_<uint16_t, index_t>(input, index);
          case 4:
            return UVAIndexSelectImpl_<uint32_t, index_t>(input, index);
          case 8:
            return UVAIndexSelectImpl_<uint64_t, index_t>(input, index);
          case 16:
            return UVAIndexSelectImpl_<float4, index_t>(input, index);
          default:
            TORCH_CHECK(false, "UVAIndexSelectImpl: Unreachable code path!");
            return torch::Tensor{};
        }
      }));
}

}  //  namespace ops
}  //  namespace graphbolt
