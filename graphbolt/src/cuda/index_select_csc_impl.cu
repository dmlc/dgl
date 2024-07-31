/**
 *   Copyright (c) 2023, GT-TDAlab (Muhammed Fatih Balin & Umit V. Catalyurek)
 *   All rights reserved.
 *
 *   Licensed under the Apache License, Version 2.0 (the "License");
 *   you may not use this file except in compliance with the License.
 *   You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *   Unless required by applicable law or agreed to in writing, software
 *   distributed under the License is distributed on an "AS IS" BASIS,
 *   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *   See the License for the specific language governing permissions and
 *   limitations under the License.
 *
 * @file cuda/index_select_csc_impl.cu
 * @brief Index select csc operator implementation on CUDA.
 */
#include <c10/core/ScalarType.h>
#include <graphbolt/cuda_ops.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>

#include <cstdint>
#include <cub/cub.cuh>
#include <numeric>

#include "./common.h"
#include "./max_uva_threads.h"
#include "./utils.h"

namespace graphbolt {
namespace ops {

constexpr int BLOCK_SIZE = CUDA_MAX_NUM_THREADS;

// Given the in_degree array and a permutation, returns in_degree of the output
// and the permuted and modified in_degree of the input. The modified in_degree
// is modified so that there is slack to be able to align as needed.
template <typename indptr_t, typename indices_t>
struct AlignmentFunc {
  static_assert(GPU_CACHE_LINE_SIZE % sizeof(indices_t) == 0);
  const indptr_t* in_degree;
  const int64_t* perm;
  int64_t num_nodes;
  __host__ __device__ auto operator()(int64_t row) {
    constexpr int num_elements = GPU_CACHE_LINE_SIZE / sizeof(indices_t);
    return thrust::make_tuple(
        in_degree[row],
        // A single cache line has num_elements items, we add num_elements - 1
        // to ensure there is enough slack to move forward or backward by
        // num_elements - 1 items if the performed access is not aligned.
        static_cast<indptr_t>(
            in_degree[perm ? perm[row % num_nodes] : row] + num_elements - 1));
  }
};

template <typename indptr_t, typename indices_t, typename coo_rows_t>
__global__ void _CopyIndicesAlignedKernel(
    const indptr_t edge_count, const indptr_t* const indptr,
    const indptr_t* const output_indptr,
    const indptr_t* const output_indptr_aligned, const indices_t* const indices,
    const coo_rows_t* const coo_aligned_rows, indices_t* const output_indices,
    const int64_t* const perm) {
  indptr_t idx = static_cast<indptr_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int stride_x = gridDim.x * blockDim.x;

  while (idx < edge_count) {
    const auto permuted_row_pos = coo_aligned_rows[idx];
    const auto row_pos = perm ? perm[permuted_row_pos] : permuted_row_pos;
    const auto out_row = output_indptr[row_pos];
    const auto d = output_indptr[row_pos + 1] - out_row;
    const int offset = (reinterpret_cast<std::uintptr_t>(
                            indices + indptr[row_pos] -
                            output_indptr_aligned[permuted_row_pos]) %
                        GPU_CACHE_LINE_SIZE) /
                       sizeof(indices_t);
    const auto rofs = idx - output_indptr_aligned[permuted_row_pos] - offset;
    if (rofs >= 0 && rofs < d) {
      const auto in_idx = indptr[row_pos] + rofs;
      assert(
          reinterpret_cast<std::uintptr_t>(indices + in_idx - idx) %
              GPU_CACHE_LINE_SIZE ==
          0);
      const auto u = indices[in_idx];
      output_indices[out_row + rofs] = u;
    }
    idx += stride_x;
  }
}

struct PairSum {
  template <typename indptr_t>
  __host__ __device__ auto operator()(
      const thrust::tuple<indptr_t, indptr_t> a,
      const thrust::tuple<indptr_t, indptr_t> b) {
    return thrust::make_tuple(
        thrust::get<0>(a) + thrust::get<0>(b),
        thrust::get<1>(a) + thrust::get<1>(b));
  };
};

template <typename indptr_t, typename indices_t>
std::tuple<torch::Tensor, torch::Tensor> UVAIndexSelectCSCCopyIndices(
    torch::Tensor indices, const int64_t num_nodes,
    const indptr_t* const in_degree, const indptr_t* const sliced_indptr,
    const int64_t* const perm, torch::TensorOptions options,
    torch::ScalarType indptr_scalar_type,
    torch::optional<int64_t> output_size) {
  auto allocator = cuda::GetAllocator();
  thrust::counting_iterator<int64_t> iota(0);

  // Output indptr for the slice indexed by nodes.
  auto output_indptr =
      torch::empty(num_nodes + 1, options.dtype(indptr_scalar_type));

  auto output_indptr_aligned =
      torch::empty(num_nodes + 1, options.dtype(indptr_scalar_type));
  auto output_indptr_aligned_ptr = output_indptr_aligned.data_ptr<indptr_t>();

  {
    // Returns the actual and modified_indegree as a pair, the
    // latter overestimates the actual indegree for alignment
    // purposes.
    auto modified_in_degree = thrust::make_transform_iterator(
        iota, AlignmentFunc<indptr_t, indices_t>{in_degree, perm, num_nodes});
    auto output_indptr_pair = thrust::make_zip_iterator(
        output_indptr.data_ptr<indptr_t>(), output_indptr_aligned_ptr);
    thrust::tuple<indptr_t, indptr_t> zero_value{};
    // Compute the prefix sum over actual and modified indegrees.
    CUB_CALL(
        DeviceScan::ExclusiveScan, modified_in_degree, output_indptr_pair,
        PairSum{}, zero_value, num_nodes + 1);
  }

  // Copy the actual total number of edges.
  if (!output_size.has_value()) {
    auto edge_count =
        cuda::CopyScalar{output_indptr.data_ptr<indptr_t>() + num_nodes};
    output_size = static_cast<indptr_t>(edge_count);
  }
  // Copy the modified number of edges.
  auto edge_count_aligned_ =
      cuda::CopyScalar{output_indptr_aligned_ptr + num_nodes};
  const int64_t edge_count_aligned = static_cast<indptr_t>(edge_count_aligned_);

  // Allocate output array with actual number of edges.
  torch::Tensor output_indices =
      torch::empty(output_size.value(), options.dtype(indices.scalar_type()));
  const dim3 block(BLOCK_SIZE);
  const dim3 grid(
      (std::min(edge_count_aligned, cuda::max_uva_threads.value_or(1 << 20)) +
       BLOCK_SIZE - 1) /
      BLOCK_SIZE);

  // Find the smallest integer type to store the coo_aligned_rows tensor.
  const int num_bits = cuda::NumberOfBits(num_nodes);
  std::array<int, 4> type_bits = {8, 15, 31, 63};
  const auto type_index =
      std::lower_bound(type_bits.begin(), type_bits.end(), num_bits) -
      type_bits.begin();
  std::array<torch::ScalarType, 5> types = {
      torch::kByte, torch::kInt16, torch::kInt32, torch::kLong, torch::kLong};
  auto coo_dtype = types[type_index];

  auto coo_aligned_rows = ExpandIndptrImpl(
      output_indptr_aligned, coo_dtype, torch::nullopt, edge_count_aligned);

  AT_DISPATCH_INTEGRAL_TYPES(
      coo_dtype, "UVAIndexSelectCSCCopyIndicesCOO", ([&] {
        using coo_rows_t = scalar_t;
        // Perform the actual copying, of the indices array into
        // output_indices in an aligned manner.
        CUDA_KERNEL_CALL(
            _CopyIndicesAlignedKernel, grid, block, 0,
            static_cast<indptr_t>(edge_count_aligned_), sliced_indptr,
            output_indptr.data_ptr<indptr_t>(), output_indptr_aligned_ptr,
            reinterpret_cast<indices_t*>(indices.data_ptr()),
            coo_aligned_rows.data_ptr<coo_rows_t>(),
            reinterpret_cast<indices_t*>(output_indices.data_ptr()), perm);
      }));
  return {output_indptr, output_indices};
}

std::tuple<torch::Tensor, torch::Tensor> UVAIndexSelectCSCImpl(
    torch::Tensor in_degree, torch::Tensor sliced_indptr, torch::Tensor indices,
    torch::Tensor nodes, int num_bits, torch::optional<int64_t> output_size) {
  // Sorting nodes so that accesses over PCI-e are more regular.
  const auto sorted_idx = Sort(nodes, num_bits).second;
  const int64_t num_nodes = nodes.size(0);

  return AT_DISPATCH_INTEGRAL_TYPES(
      sliced_indptr.scalar_type(), "UVAIndexSelectCSCIndptr", ([&] {
        using indptr_t = scalar_t;
        return GRAPHBOLT_DISPATCH_ELEMENT_SIZES(
            indices.element_size(), "UVAIndexSelectCSCCopyIndices", ([&] {
              return UVAIndexSelectCSCCopyIndices<indptr_t, element_size_t>(
                  indices, num_nodes, in_degree.data_ptr<indptr_t>(),
                  sliced_indptr.data_ptr<indptr_t>(),
                  sorted_idx.data_ptr<int64_t>(), nodes.options(),
                  sliced_indptr.scalar_type(), output_size);
            }));
      }));
}

template <typename indptr_t, typename indices_t>
struct IteratorFunc {
  indptr_t* indptr;
  indices_t* indices;
  __host__ __device__ auto operator()(int64_t i) { return indices + indptr[i]; }
};

template <typename indptr_t, typename indices_t>
struct ConvertToBytes {
  const indptr_t* in_degree;
  __host__ __device__ indptr_t operator()(int64_t i) {
    return in_degree[i] * sizeof(indices_t);
  }
};

template <typename indptr_t, typename indices_t>
void IndexSelectCSCCopyIndices(
    const int64_t num_nodes, indices_t* const indices,
    indptr_t* const sliced_indptr, const indptr_t* const in_degree,
    indptr_t* const output_indptr, indices_t* const output_indices) {
  thrust::counting_iterator<int64_t> iota(0);

  auto input_buffer_it = thrust::make_transform_iterator(
      iota, IteratorFunc<indptr_t, indices_t>{sliced_indptr, indices});
  auto output_buffer_it = thrust::make_transform_iterator(
      iota, IteratorFunc<indptr_t, indices_t>{output_indptr, output_indices});
  auto buffer_sizes = thrust::make_transform_iterator(
      iota, ConvertToBytes<indptr_t, indices_t>{in_degree});
  constexpr int64_t max_copy_at_once = std::numeric_limits<int32_t>::max();

  // Performs the copy from indices into output_indices.
  for (int64_t i = 0; i < num_nodes; i += max_copy_at_once) {
    CUB_CALL(
        DeviceMemcpy::Batched, input_buffer_it + i, output_buffer_it + i,
        buffer_sizes + i, std::min(num_nodes - i, max_copy_at_once));
  }
}

std::tuple<torch::Tensor, torch::Tensor> DeviceIndexSelectCSCImpl(
    torch::Tensor in_degree, torch::Tensor sliced_indptr, torch::Tensor indices,
    torch::TensorOptions options, torch::optional<int64_t> output_size) {
  const int64_t num_nodes = sliced_indptr.size(0);
  return AT_DISPATCH_INTEGRAL_TYPES(
      sliced_indptr.scalar_type(), "IndexSelectCSCIndptr", ([&] {
        using indptr_t = scalar_t;
        auto in_degree_ptr = in_degree.data_ptr<indptr_t>();
        auto sliced_indptr_ptr = sliced_indptr.data_ptr<indptr_t>();
        // Output indptr for the slice indexed by nodes.
        torch::Tensor output_indptr = torch::empty(
            num_nodes + 1, options.dtype(sliced_indptr.scalar_type()));

        // Compute the output indptr, output_indptr.
        CUB_CALL(
            DeviceScan::ExclusiveSum, in_degree_ptr,
            output_indptr.data_ptr<indptr_t>(), num_nodes + 1);

        // Number of edges being copied.
        if (!output_size.has_value()) {
          auto edge_count =
              cuda::CopyScalar{output_indptr.data_ptr<indptr_t>() + num_nodes};
          output_size = static_cast<indptr_t>(edge_count);
        }
        // Allocate output array of size number of copied edges.
        torch::Tensor output_indices = torch::empty(
            output_size.value(), options.dtype(indices.scalar_type()));
        GRAPHBOLT_DISPATCH_ELEMENT_SIZES(
            indices.element_size(), "IndexSelectCSCCopyIndices", ([&] {
              using indices_t = element_size_t;
              IndexSelectCSCCopyIndices<indptr_t, indices_t>(
                  num_nodes, reinterpret_cast<indices_t*>(indices.data_ptr()),
                  sliced_indptr_ptr, in_degree_ptr,
                  output_indptr.data_ptr<indptr_t>(),
                  reinterpret_cast<indices_t*>(output_indices.data_ptr()));
            }));
        return std::make_tuple(output_indptr, output_indices);
      }));
}

std::tuple<torch::Tensor, torch::Tensor> IndexSelectCSCImpl(
    torch::Tensor in_degree, torch::Tensor sliced_indptr, torch::Tensor indices,
    torch::Tensor nodes, int64_t nodes_max,
    torch::optional<int64_t> output_size) {
  if (indices.is_pinned()) {
    int num_bits = cuda::NumberOfBits(nodes_max + 1);
    return UVAIndexSelectCSCImpl(
        in_degree, sliced_indptr, indices, nodes, num_bits, output_size);
  } else {
    return DeviceIndexSelectCSCImpl(
        in_degree, sliced_indptr, indices, nodes.options(), output_size);
  }
}

std::tuple<torch::Tensor, torch::Tensor> IndexSelectCSCImpl(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor nodes,
    torch::optional<int64_t> output_size) {
  auto [in_degree, sliced_indptr] = SliceCSCIndptr(indptr, nodes);
  return IndexSelectCSCImpl(
      in_degree, sliced_indptr, indices, nodes, indptr.size(0) - 2,
      output_size);
}

std::tuple<torch::Tensor, std::vector<torch::Tensor>> IndexSelectCSCBatchedImpl(
    torch::Tensor indptr, std::vector<torch::Tensor> indices_list,
    torch::Tensor nodes, bool with_edge_ids,
    torch::optional<int64_t> output_size) {
  auto [in_degree, sliced_indptr] = SliceCSCIndptr(indptr, nodes);
  std::vector<torch::Tensor> results;
  results.reserve(indices_list.size());
  torch::Tensor output_indptr;
  for (auto& indices : indices_list) {
    torch::Tensor output_indices;
    std::tie(output_indptr, output_indices) = IndexSelectCSCImpl(
        in_degree, sliced_indptr, indices, nodes, indptr.size(0) - 2,
        output_size);
    if (!output_size.has_value()) output_size = output_indices.size(0);
    TORCH_CHECK(*output_size == output_indices.size(0));
    results.push_back(output_indices);
  }
  if (with_edge_ids) {
    results.push_back(IndptrEdgeIdsImpl(
        output_indptr, sliced_indptr.scalar_type(), sliced_indptr,
        output_size));
  }
  return {output_indptr, results};
}

}  //  namespace ops
}  //  namespace graphbolt
