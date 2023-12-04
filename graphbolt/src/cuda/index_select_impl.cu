/**
 *  Copyright (c) 2023 by Contributors
 * @file cuda/index_select_impl.cu
 * @brief Index select operator implementation on CUDA.
 */
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAStream.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <cub/cub.cuh>
#include <numeric>

#include "../index_select.h"
#include "./common.h"
#include "./utils.h"

namespace graphbolt {
namespace ops {

constexpr int BLOCK_SIZE = 128;

std::pair<torch::Tensor, torch::Tensor> Sort(
    torch::Tensor input, int num_bits) {
  int64_t num_items = input.size(0);
  // We utilize int64_t for the values array. (torch::kLong == int64_t)
  auto original_idx =
      torch::arange(num_items, input.options().dtype(torch::kLong));
  auto sorted_array = torch::empty_like(input);
  auto sorted_idx = torch::empty_like(original_idx);
  cuda::CUDAWorkspaceAllocator allocator;
  auto stream = c10::cuda::getDefaultCUDAStream();
  AT_DISPATCH_INDEX_TYPES(
      input.scalar_type(), "SortImpl", ([&] {
        const auto input_keys = input.data_ptr<index_t>();
        const int64_t* input_values = original_idx.data_ptr<int64_t>();
        index_t* sorted_keys = sorted_array.data_ptr<index_t>();
        int64_t* sorted_values = sorted_idx.data_ptr<int64_t>();
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

template <typename indptr_t, typename indices_t>
struct AlignmentFunc {
  static_assert(GPU_CACHE_LINE_SIZE % sizeof(indices_t) == 0);
  const indptr_t* in_degree;
  const int64_t* perm;
  int64_t num_rows;
  __host__ __device__ auto operator()(int64_t row) {
    constexpr int num_elements = GPU_CACHE_LINE_SIZE / sizeof(indices_t);
    return thrust::make_tuple(
        in_degree[row],
        (indptr_t)(in_degree[perm ? perm[row % num_rows] : row] + num_elements - 1));
  }
};

template <typename indptr_t, typename indices_t>
__global__ void _CSRRowWiseOneHopExtractorAlignedKernel(
    const indptr_t hop_size, const int64_t num_rows,
    const indptr_t* const indptr, const indptr_t* const subindptr,
    const indptr_t* const subindptr_aligned, const indices_t* const indices,
    indices_t* const hop, const int64_t* const perm) {
  indptr_t tx = static_cast<indptr_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int stride_x = gridDim.x * blockDim.x;

  while (tx < hop_size) {
    const auto rpos_ = cuda::UpperBound(subindptr_aligned, num_rows, tx) - 1;
    const auto rpos = perm ? perm[rpos_] : rpos_;
    const auto out_row = subindptr[rpos];
    const auto d = subindptr[rpos + 1] - out_row;
    const int offset =
        ((size_t)(indices + indptr[rpos] - subindptr_aligned[rpos_]) %
         GPU_CACHE_LINE_SIZE) /
        sizeof(indices_t);
    const auto rofs = tx - subindptr_aligned[rpos_] - offset;
    if (rofs >= 0 && rofs < d) {
      const auto in_idx = indptr[rpos] + rofs;
      assert((size_t)(indices + in_idx - tx) % GPU_CACHE_LINE_SIZE == 0);
      const auto u = indices[in_idx];
      hop[out_row + rofs] = u;
    }
    tx += stride_x;
  }
}

template <typename indptr_t, typename index_t>
struct DegreeFunc {
  const index_t* rows;
  const indptr_t* indptr;
  indptr_t* in_deg;
  indptr_t* inrow_indptr;
  __host__ __device__ auto operator()(int64_t tIdx) {
    const auto out_row = rows[tIdx];
    const auto indptr_val = indptr[out_row];
    const auto degree = indptr[out_row + 1] - indptr_val;
    in_deg[tIdx] = degree;
    inrow_indptr[tIdx] = indptr_val;
  }
};

struct PairSum {
  template <typename indptr_t>
  __host__ __device__ auto operator()(
      thrust::tuple<indptr_t, indptr_t> a,
      thrust::tuple<indptr_t, indptr_t> b) {
    return thrust::make_tuple(
        thrust::get<0>(a) + thrust::get<0>(b),
        thrust::get<1>(a) + thrust::get<1>(b));
  };
};

std::tuple<torch::Tensor, torch::Tensor> UVAIndexSelectCSCImpl(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor index) {
  const auto [sorted, perm_tensor] =
      Sort(index, cuda::NumberOfBits(indptr.size(0) - 1));
  const auto perm = perm_tensor.data_ptr<int64_t>();

  cuda::CUDAWorkspaceAllocator allocator;
  auto stream = c10::cuda::getDefaultCUDAStream();
  const auto exec_policy = thrust::cuda::par_nosync(allocator).on(stream);

  const int64_t num_rows = index.size(0);

  auto subindptr =
      torch::empty(num_rows + 1, index.options().dtype(indptr.scalar_type()));
  torch::Tensor subindices;

  AT_DISPATCH_INTEGRAL_TYPES(
      indptr.scalar_type(), "UVAIndexSelectCSCIndptr", ([&] {
        using indptr_t = scalar_t;
        // Read indptr only once in case it is pinned and access is slow.
        auto sliced_indptr = allocator.AllocateStorage<indptr_t>(num_rows);
        // compute in-degrees
        auto in_deg = allocator.AllocateStorage<indptr_t>(num_rows + 1);
        AT_DISPATCH_INDEX_TYPES(
            index.scalar_type(), "UVAIndexSelectCSCIndex", ([&] {
              thrust::counting_iterator<int64_t> iota(0);
              thrust::for_each(
                  exec_policy, iota, iota + num_rows,
                  DegreeFunc<indptr_t, index_t>{
                      index.data_ptr<index_t>(), indptr.data_ptr<indptr_t>(),
                      in_deg.get(), sliced_indptr.get()});
            }));
        auto copy_indices_fun = [&](auto indices_type_variable) {
          using indices_t = decltype(indices_type_variable);

          indptr_t hop_size, hop_size_aligned;
          auto subindptr_aligned =
              allocator.AllocateStorage<indptr_t>(num_rows + 1);
          {
            thrust::counting_iterator<int64_t> iota(0);
            auto modified_in_deg = thrust::make_transform_iterator(
                iota, AlignmentFunc<indptr_t, indices_t>{
                          in_deg.get(), perm, num_rows});
            auto subindptr_pair = thrust::make_zip_iterator(
                subindptr.data_ptr<indptr_t>(), subindptr_aligned.get());
            thrust::tuple<indptr_t, indptr_t> zero_value{};
            size_t workspace_size = 0;
            CUDA_CALL(cub::DeviceScan::ExclusiveScan(
                nullptr, workspace_size, modified_in_deg, subindptr_pair,
                PairSum{}, zero_value, num_rows + 1, stream));
            auto temp = allocator.AllocateStorage<char>(workspace_size);
            CUDA_CALL(cub::DeviceScan::ExclusiveScan(
                temp.get(), workspace_size, modified_in_deg, subindptr_pair,
                PairSum{}, zero_value, num_rows + 1, stream));
          }
          CUDA_CALL(cudaMemcpyAsync(
              &hop_size_aligned, subindptr_aligned.get() + num_rows,
              sizeof(hop_size_aligned), cudaMemcpyDeviceToHost, stream));
          CUDA_CALL(cudaMemcpyAsync(
              &hop_size, subindptr.data_ptr<indptr_t>() + num_rows,
              sizeof(hop_size), cudaMemcpyDeviceToHost, stream));
          // synchronizes here, we can read hop_size and hop_size_aligned
          cudaStreamSynchronize(stream);
          subindices = torch::empty(
              hop_size, index.options().dtype(indices.scalar_type()));
          const dim3 block(BLOCK_SIZE);
          const dim3 grid((hop_size_aligned + BLOCK_SIZE - 1) / BLOCK_SIZE);
          CUDA_KERNEL_CALL(
              _CSRRowWiseOneHopExtractorAlignedKernel, grid, block, 0, stream,
              hop_size_aligned, num_rows, sliced_indptr.get(),
              subindptr.data_ptr<indptr_t>(), subindptr_aligned.get(),
              reinterpret_cast<indices_t*>(indices.data_ptr()),
              reinterpret_cast<indices_t*>(subindices.data_ptr()), perm);
        };
        switch (indices.element_size()) {
          case 1:
            return copy_indices_fun(uint8_t{});
          case 2:
            return copy_indices_fun(uint16_t{});
          case 4:
            return copy_indices_fun(uint32_t{});
          case 8:
            return copy_indices_fun(uint64_t{});
          case 16:
            return copy_indices_fun(float4{});
        }
      }));
  return std::make_tuple(subindptr, subindices);
}

template <typename indptr_t, typename indices_t>
struct IteratorFunc {
  indptr_t* indptr;
  indices_t* indices;
  __host__ __device__ auto operator()(int64_t i) { return indices + indptr[i]; }
};

template <typename indptr_t, typename indices_t>
struct ConvertToBytes {
  indptr_t* in_degree;
  __host__ __device__ indptr_t operator()(int64_t i) {
    return in_degree[i] * sizeof(indices_t);
  }
};

std::tuple<torch::Tensor, torch::Tensor> IndexSelectCSCImpl(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor index) {
  cuda::CUDAWorkspaceAllocator allocator;
  auto stream = c10::cuda::getDefaultCUDAStream();
  const auto exec_policy = thrust::cuda::par_nosync(allocator).on(stream);

  const int64_t num_rows = index.size(0);

  auto subindptr =
      torch::empty(num_rows + 1, index.options().dtype(indptr.scalar_type()));
  torch::Tensor subindices;
  thrust::counting_iterator<int64_t> iota(0);

  AT_DISPATCH_INTEGRAL_TYPES(
      indptr.scalar_type(), "IndexSelectCSCIndptr", ([&] {
        using indptr_t = scalar_t;
        // Read indptr only once in case it is pinned and access is slow.
        auto sliced_indptr = allocator.AllocateStorage<indptr_t>(num_rows);
        // compute in-degrees
        auto in_deg = allocator.AllocateStorage<indptr_t>(num_rows + 1);
        indptr_t hop_size;
        AT_DISPATCH_INDEX_TYPES(
            index.scalar_type(), "IndexSelectCSCIndex", ([&] {
              thrust::counting_iterator<int64_t> iota(0);
              thrust::for_each(
                  exec_policy, iota, iota + num_rows,
                  DegreeFunc<indptr_t, index_t>{
                      index.data_ptr<index_t>(), indptr.data_ptr<indptr_t>(),
                      in_deg.get(), sliced_indptr.get()});
            }));
        size_t workspace_size = 0;
        CUDA_CALL(cub::DeviceScan::ExclusiveSum(
            nullptr, workspace_size, in_deg.get(),
            subindptr.data_ptr<indptr_t>(), num_rows + 1, stream));
        auto temp = allocator.AllocateStorage<char>(workspace_size);
        CUDA_CALL(cub::DeviceScan::ExclusiveSum(
            temp.get(), workspace_size, in_deg.get(),
            subindptr.data_ptr<indptr_t>(), num_rows + 1, stream));
        // blocking read of hop_size
        CUDA_CALL(cudaMemcpyAsync(
            &hop_size, subindptr.data_ptr<indptr_t>() + num_rows,
            sizeof(hop_size), cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);
        subindices = torch::empty(
            hop_size, index.options().dtype(indices.scalar_type()));
        auto copy_indices_fun = [&](auto indices_type_variable) {
          using indices_t = decltype(indices_type_variable);
          auto input_buffer_it = thrust::make_transform_iterator(
              iota, IteratorFunc<indptr_t, indices_t>{
                        sliced_indptr.get(),
                        reinterpret_cast<indices_t*>(indices.data_ptr())});
          auto output_buffer_it = thrust::make_transform_iterator(
              iota, IteratorFunc<indptr_t, indices_t>{
                        subindptr.data_ptr<indptr_t>(),
                        reinterpret_cast<indices_t*>(subindices.data_ptr())});
          auto buffer_sizes = thrust::make_transform_iterator(
              iota, ConvertToBytes<indptr_t, indices_t>{in_deg.get()});
          constexpr int64_t max_copy_at_once =
              std::numeric_limits<int32_t>::max();
          for (int64_t i = 0; i < num_rows; i += max_copy_at_once) {
            size_t workspace_size = 0;
            CUDA_CALL(cub::DeviceMemcpy::Batched(
                nullptr, workspace_size, input_buffer_it + i,
                output_buffer_it + i, buffer_sizes + i,
                std::min(num_rows - i, max_copy_at_once), stream));
            auto temp = allocator.AllocateStorage<char>(workspace_size);
            CUDA_CALL(cub::DeviceMemcpy::Batched(
                temp.get(), workspace_size, input_buffer_it + i,
                output_buffer_it + i, buffer_sizes + i,
                std::min(num_rows - i, max_copy_at_once), stream));
          }
        };
        switch (indices.element_size()) {
          case 1:
            return copy_indices_fun(uint8_t{});
          case 2:
            return copy_indices_fun(uint16_t{});
          case 4:
            return copy_indices_fun(uint32_t{});
          case 8:
            return copy_indices_fun(uint64_t{});
          case 16:
            return copy_indices_fun(float4{});
        }
      }));
  return std::make_tuple(subindptr, subindices);
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

  cudaStream_t stream = c10::cuda::getDefaultCUDAStream();

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
