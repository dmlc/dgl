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
 * @file cuda/index_select_impl.cu
 * @brief Index select operator implementation on CUDA.
 */
#include <c10/core/ScalarType.h>
#include <curand_kernel.h>
#include <graphbolt/continuous_seed.h>
#include <graphbolt/cuda_ops.h>
#include <graphbolt/cuda_sampling_ops.h>
#include <thrust/copy.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>

#include <algorithm>
#include <array>
#include <cub/cub.cuh>
#if __CUDA_ARCH__ >= 700
#include <cuda/atomic>
#endif  // __CUDA_ARCH__ >= 700
#include <limits>
#include <numeric>
#include <type_traits>

#include "../macro.h"
#include "../random.h"
#include "../utils.h"
#include "./common.h"
#include "./utils.h"

namespace graphbolt {
namespace ops {

constexpr int BLOCK_SIZE = 128;

inline __device__ int64_t AtomicMax(int64_t* const address, const int64_t val) {
  // To match the type of "::atomicCAS", ignore lint warning.
  using Type = unsigned long long int;  // NOLINT

  static_assert(sizeof(Type) == sizeof(*address), "Type width must match");

  return atomicMax(reinterpret_cast<Type*>(address), static_cast<Type>(val));
}

inline __device__ int32_t AtomicMax(int32_t* const address, const int32_t val) {
  // To match the type of "::atomicCAS", ignore lint warning.
  using Type = int;  // NOLINT

  static_assert(sizeof(Type) == sizeof(*address), "Type width must match");

  return atomicMax(reinterpret_cast<Type*>(address), static_cast<Type>(val));
}

/**
 * @brief Performs neighbor sampling and fills the edge_ids array with
 * original edge ids if sliced_indptr is valid. If not, then it fills the edge
 * ids array with numbers upto the node degree.
 */
template <typename indptr_t, typename indices_t>
__global__ void _ComputeRandomsNS(
    const int64_t num_edges, const indptr_t* const sliced_indptr,
    const indptr_t* const sub_indptr, const indptr_t* const output_indptr,
    const indices_t* const csr_rows, const uint64_t random_seed,
    indptr_t* edge_ids) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;

  curandStatePhilox4_32_10_t rng;
  curand_init(random_seed, i, 0, &rng);

  while (i < num_edges) {
    const auto row_position = csr_rows[i];
    const auto row_offset = i - sub_indptr[row_position];
    const auto output_offset = output_indptr[row_position];
    const auto fanout = output_indptr[row_position + 1] - output_offset;
    const auto rnd =
        row_offset < fanout ? row_offset : curand(&rng) % (row_offset + 1);
    if (rnd < fanout) {
      const indptr_t edge_id =
          row_offset + (sliced_indptr ? sliced_indptr[row_position] : 0);
#if __CUDA_ARCH__ >= 700
      ::cuda::atomic_ref<indptr_t, ::cuda::thread_scope_device> a(
          edge_ids[output_offset + rnd]);
      a.fetch_max(edge_id, ::cuda::std::memory_order_relaxed);
#else
      AtomicMax(edge_ids + output_offset + rnd, edge_id);
#endif  // __CUDA_ARCH__
    }

    i += stride;
  }
}

/**
 * @brief Fills the random_arr with random numbers and the edge_ids array with
 * original edge ids. When random_arr is sorted along with edge_ids, the first
 * fanout elements of each row gives us the sampled edges.
 */
template <
    typename float_t, typename indptr_t, typename indices_t, typename weights_t,
    typename edge_id_t>
__global__ void _ComputeRandoms(
    const int64_t num_edges, const indptr_t* const sliced_indptr,
    const indptr_t* const sub_indptr, const indices_t* const csr_rows,
    const weights_t* const sliced_weights, const indices_t* const indices,
    const continuous_seed random_seed, float_t* random_arr,
    edge_id_t* edge_ids) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  const auto labor = indices != nullptr;
  const float_t inf =
      static_cast<float_t>(std::numeric_limits<float>::infinity());

  while (i < num_edges) {
    const auto row_position = csr_rows[i];
    const auto row_offset = i - sub_indptr[row_position];
    const auto in_idx = sliced_indptr[row_position] + row_offset;
    const auto rnd = random_seed.uniform(labor ? indices[in_idx] : i);
    const auto prob =
        sliced_weights ? sliced_weights[i] : static_cast<weights_t>(1);
    const auto exp_rnd = -__logf(rnd);
    const float_t adjusted_rnd =
        prob > 0 ? static_cast<float_t>(exp_rnd / prob) : inf;
    random_arr[i] = adjusted_rnd;
    edge_ids[i] = row_offset;

    i += stride;
  }
}

struct IsPositive {
  template <typename probs_t>
  __host__ __device__ auto operator()(probs_t x) {
    return x > 0;
  }
};

template <typename indptr_t>
struct MinInDegreeFanout {
  const indptr_t* in_degree;
  const int64_t* fanouts;
  size_t num_fanouts;
  __host__ __device__ auto operator()(int64_t i) {
    return static_cast<indptr_t>(
        min(static_cast<int64_t>(in_degree[i]), fanouts[i % num_fanouts]));
  }
};

template <typename indptr_t, typename indices_t>
struct IteratorFunc {
  indptr_t* indptr;
  indices_t* indices;
  __host__ __device__ auto operator()(int64_t i) { return indices + indptr[i]; }
};

template <typename indptr_t>
struct AddOffset {
  indptr_t offset;
  template <typename edge_id_t>
  __host__ __device__ indptr_t operator()(edge_id_t x) {
    return x + offset;
  }
};

template <typename indptr_t, typename indices_t>
struct IteratorFuncAddOffset {
  indptr_t* indptr;
  indptr_t* sliced_indptr;
  indices_t* indices;
  __host__ __device__ auto operator()(int64_t i) {
    return thrust::transform_output_iterator{
        indices + indptr[i], AddOffset<indptr_t>{sliced_indptr[i]}};
  }
};

template <typename indptr_t, typename in_degree_iterator_t>
struct SegmentEndFunc {
  indptr_t* indptr;
  in_degree_iterator_t in_degree;
  __host__ __device__ auto operator()(int64_t i) {
    return indptr[i] + in_degree[i];
  }
};

c10::intrusive_ptr<sampling::FusedSampledSubgraph> SampleNeighbors(
    torch::Tensor indptr, torch::Tensor indices,
    torch::optional<torch::Tensor> seeds,
    torch::optional<std::vector<int64_t>> seed_offsets,
    const std::vector<int64_t>& fanouts, bool replace, bool layer,
    bool returning_indices_is_optional,
    torch::optional<torch::Tensor> type_per_edge,
    torch::optional<torch::Tensor> probs_or_mask,
    torch::optional<torch::Tensor> node_type_offset,
    torch::optional<torch::Dict<std::string, int64_t>> node_type_to_id,
    torch::optional<torch::Dict<std::string, int64_t>> edge_type_to_id,
    torch::optional<torch::Tensor> random_seed_tensor, float seed2_contribution,
    // Optional temporal sampling arguments begin.
    torch::optional<torch::Tensor> seeds_timestamp,
    torch::optional<torch::Tensor> seeds_pre_time_window,
    torch::optional<torch::Tensor> node_timestamp,
    torch::optional<torch::Tensor> edge_timestamp
    // Optional temporal sampling arguments end.
) {
  // When seed_offsets.has_value() in the hetero case, we compute the output of
  // sample_neighbors _convert_to_sampled_subgraph in a fused manner so that
  // _convert_to_sampled_subgraph only has to perform slices over the returned
  // indptr and indices tensors to form CSC outputs for each edge type.
  TORCH_CHECK(!replace, "Sampling with replacement is not supported yet!");
  // Assume that indptr, indices, seeds, type_per_edge and probs_or_mask
  // are all resident on the GPU. If not, it is better to first extract them
  // before calling this function.
  auto allocator = cuda::GetAllocator();
  auto num_rows =
      seeds.has_value() ? seeds.value().size(0) : indptr.size(0) - 1;
  auto fanouts_pinned = torch::empty(
      fanouts.size(),
      c10::TensorOptions().dtype(torch::kLong).pinned_memory(true));
  auto fanouts_pinned_ptr = fanouts_pinned.data_ptr<int64_t>();
  for (size_t i = 0; i < fanouts.size(); i++) {
    fanouts_pinned_ptr[i] =
        fanouts[i] >= 0 ? fanouts[i] : std::numeric_limits<int64_t>::max();
  }
  // Finally, copy the adjusted fanout values to the device memory.
  auto fanouts_device = allocator.AllocateStorage<int64_t>(fanouts.size());
  CUDA_CALL(cudaMemcpyAsync(
      fanouts_device.get(), fanouts_pinned_ptr,
      sizeof(int64_t) * fanouts.size(), cudaMemcpyHostToDevice,
      cuda::GetCurrentStream()));
  auto in_degree_and_sliced_indptr = SliceCSCIndptr(indptr, seeds);
  auto in_degree = std::get<0>(in_degree_and_sliced_indptr);
  auto sliced_indptr = std::get<1>(in_degree_and_sliced_indptr);
  const auto homo_in_degree = in_degree;
  const auto homo_sliced_indptr = sliced_indptr;
  auto max_in_degree = torch::empty(
      1,
      c10::TensorOptions().dtype(in_degree.scalar_type()).pinned_memory(true));
  AT_DISPATCH_INDEX_TYPES(
      indptr.scalar_type(), "SampleNeighborsMaxInDegree", ([&] {
        CUB_CALL(
            DeviceReduce::Max, in_degree.data_ptr<index_t>(),
            max_in_degree.data_ptr<index_t>(), num_rows);
      }));
  // Protect access to max_in_degree with a CUDAEvent
  at::cuda::CUDAEvent max_in_degree_event;
  max_in_degree_event.record();
  torch::optional<int64_t> num_edges;
  torch::Tensor sub_indptr;
  if (!seeds.has_value()) {
    num_edges = indices.size(0);
    sub_indptr = indptr;
  }
  torch::optional<torch::Tensor> sliced_probs_or_mask;
  if (probs_or_mask.has_value()) {
    if (seeds.has_value()) {
      torch::Tensor sliced_probs_or_mask_tensor;
      std::tie(sub_indptr, sliced_probs_or_mask_tensor) = IndexSelectCSCImpl(
          in_degree, sliced_indptr, probs_or_mask.value(), seeds.value(),
          indptr.size(0) - 2, num_edges);
      sliced_probs_or_mask = sliced_probs_or_mask_tensor;
      num_edges = sliced_probs_or_mask_tensor.size(0);
    } else {
      sliced_probs_or_mask = probs_or_mask;
    }
  }
  if (fanouts.size() > 1) {
    torch::Tensor sliced_type_per_edge;
    if (seeds.has_value()) {
      std::tie(sub_indptr, sliced_type_per_edge) = IndexSelectCSCImpl(
          in_degree, sliced_indptr, type_per_edge.value(), seeds.value(),
          indptr.size(0) - 2, num_edges);
    } else {
      sliced_type_per_edge = type_per_edge.value();
    }
    std::tie(sub_indptr, in_degree, sliced_indptr) = SliceCSCIndptrHetero(
        sub_indptr, sliced_type_per_edge, sliced_indptr, fanouts.size());
    num_rows = sliced_indptr.size(0);
    num_edges = sliced_type_per_edge.size(0);
  }
  // If sub_indptr was not computed in the two code blocks above:
  if (seeds.has_value() && !probs_or_mask.has_value() && fanouts.size() <= 1) {
    sub_indptr = ExclusiveCumSum(in_degree);
  }
  torch::optional<torch::Tensor> homo_coo_rows;
  if (seeds_timestamp.has_value()) {
    // Temporal sampling is enabled.
    const auto homo_sub_indptr =
        fanouts.size() > 1 ? ExclusiveCumSum(homo_in_degree) : sub_indptr;
    homo_coo_rows = ExpandIndptrImpl(
        homo_sub_indptr, indices.scalar_type(), torch::nullopt, num_edges);
    num_edges = homo_coo_rows->size(0);
    const auto is_probs_initialized = sliced_probs_or_mask.has_value();
    if (!is_probs_initialized) {
      sliced_probs_or_mask =
          torch::empty(*num_edges, sub_indptr.options().dtype(torch::kBool));
    }
    GRAPHBOLT_DISPATCH_ALL_TYPES(
        sliced_probs_or_mask->scalar_type(),
        "SampleNeighborsTemporalProbsOrMask", ([&] {
          const scalar_t* input_probs_ptr =
              is_probs_initialized ? sliced_probs_or_mask->data_ptr<scalar_t>()
                                   : nullptr;
          auto output_probs_ptr = sliced_probs_or_mask->data_ptr<scalar_t>();
          using timestamp_t = int64_t;
          const auto seeds_timestamp_ptr =
              seeds_timestamp->data_ptr<timestamp_t>();
          const timestamp_t* seeds_pre_time_window_ptr =
              seeds_pre_time_window.has_value()
                  ? seeds_pre_time_window->data_ptr<timestamp_t>()
                  : nullptr;
          const timestamp_t* node_timestamp_ptr =
              node_timestamp.has_value()
                  ? node_timestamp->data_ptr<timestamp_t>()
                  : nullptr;
          const timestamp_t* edge_timestamp_ptr =
              edge_timestamp.has_value()
                  ? edge_timestamp->data_ptr<timestamp_t>()
                  : nullptr;
          AT_DISPATCH_INDEX_TYPES(
              homo_coo_rows->scalar_type(),
              "SampleNeighborsTemporalMaskIndices", ([&] {
                const auto coo_rows_ptr = homo_coo_rows->data_ptr<index_t>();
                const auto indices_ptr = indices.data_ptr<index_t>();
                AT_DISPATCH_INDEX_TYPES(
                    homo_sliced_indptr.scalar_type(),
                    "SampleNeighborsTemporalMaskIndptr", ([&] {
                      const auto sliced_indptr_data =
                          homo_sliced_indptr.data_ptr<index_t>();
                      const auto sub_indptr_data =
                          homo_sub_indptr.data_ptr<index_t>();
                      CUB_CALL(
                          DeviceFor::Bulk, *num_edges,
                          [=] __device__(int64_t i) {
                            const auto row = coo_rows_ptr[i];
                            const auto seed_timestamp =
                                seeds_timestamp_ptr[row];
                            const auto row_offset = i - sub_indptr_data[row];
                            const auto in_idx =
                                sliced_indptr_data[row] + row_offset;
                            bool mask = true;
                            if (node_timestamp_ptr) {
                              const auto index = indices_ptr[in_idx];
                              const auto neighbor_timestamp =
                                  node_timestamp_ptr[index];
                              mask &= neighbor_timestamp < seed_timestamp;
                              if (seeds_pre_time_window_ptr) {
                                mask &= neighbor_timestamp >
                                        seed_timestamp -
                                            seeds_pre_time_window_ptr[row];
                              }
                            }
                            if (edge_timestamp_ptr) {
                              const auto edge_timestamp =
                                  edge_timestamp_ptr[in_idx];
                              mask &= edge_timestamp < seed_timestamp;
                              if (seeds_pre_time_window_ptr) {
                                mask &= edge_timestamp >
                                        seed_timestamp -
                                            seeds_pre_time_window_ptr[row];
                              }
                            }
                            const scalar_t prob = input_probs_ptr
                                                      ? input_probs_ptr[i]
                                                      : scalar_t{1};
                            output_probs_ptr[i] =
                                prob * static_cast<scalar_t>(mask);
                          });
                    }));
              }));
        }));
  }
  const continuous_seed random_seed = [&] {
    if (random_seed_tensor.has_value()) {
      return continuous_seed(random_seed_tensor.value(), seed2_contribution);
    } else {
      return continuous_seed{RandomEngine::ThreadLocal()->RandInt(
          static_cast<int64_t>(0), std::numeric_limits<int64_t>::max())};
    }
  }();
  auto output_indptr = torch::empty_like(sub_indptr);
  torch::Tensor picked_eids;
  torch::optional<torch::Tensor> output_indices;

  AT_DISPATCH_INDEX_TYPES(
      indptr.scalar_type(), "SampleNeighborsIndptr", ([&] {
        using indptr_t = index_t;
        if (sliced_probs_or_mask.has_value()) {
          // Count nonzero probs into in_degree.
          GRAPHBOLT_DISPATCH_ALL_TYPES(
              sliced_probs_or_mask->scalar_type(),
              "SampleNeighborsPositiveProbs", ([&] {
                using probs_t = scalar_t;
                auto is_nonzero = thrust::make_transform_iterator(
                    sliced_probs_or_mask->data_ptr<probs_t>(), IsPositive{});
                CUB_CALL(
                    DeviceSegmentedReduce::Sum, is_nonzero,
                    in_degree.data_ptr<indptr_t>(), num_rows,
                    sub_indptr.data_ptr<indptr_t>(),
                    sub_indptr.data_ptr<indptr_t>() + 1);
              }));
        }
        thrust::counting_iterator<int64_t> iota(0);
        auto sampled_degree = thrust::make_transform_iterator(
            iota, MinInDegreeFanout<indptr_t>{
                      in_degree.data_ptr<indptr_t>(), fanouts_device.get(),
                      fanouts.size()});

        // Compute output_indptr.
        CUB_CALL(
            DeviceScan::ExclusiveSum, sampled_degree,
            output_indptr.data_ptr<indptr_t>(), num_rows + 1);

        auto num_sampled_edges =
            cuda::CopyScalar{output_indptr.data_ptr<indptr_t>() + num_rows};

        // This operation is placed after num_sampled_edges copy is started to
        // hide the latency of copy synchronization later.
        torch::Tensor coo_rows;
        if (!homo_coo_rows.has_value() || fanouts.size() > 1) {
          coo_rows = ExpandIndptrImpl(
              sub_indptr, indices.scalar_type(), torch::nullopt, num_edges);
          num_edges = coo_rows.size(0);
        } else {
          coo_rows = *homo_coo_rows;
        }

        // Find the smallest integer type to store the edge id offsets. We synch
        // the CUDAEvent so that the access is safe.
        auto compute_num_bits = [&] {
          max_in_degree_event.synchronize();
          return cuda::NumberOfBits(max_in_degree.data_ptr<indptr_t>()[0]);
        };
        if (layer || sliced_probs_or_mask.has_value()) {
          const int num_bits = compute_num_bits();
          std::array<int, 4> type_bits = {8, 16, 32, 64};
          const auto type_index =
              std::lower_bound(type_bits.begin(), type_bits.end(), num_bits) -
              type_bits.begin();
          std::array<torch::ScalarType, 5> types = {
              torch::kByte, torch::kInt16, torch::kInt32, torch::kLong,
              torch::kLong};
          auto edge_id_dtype = types[type_index];
          AT_DISPATCH_INTEGRAL_TYPES(
              edge_id_dtype, "SampleNeighborsEdgeIDs", ([&] {
                using edge_id_t = std::make_unsigned_t<scalar_t>;
                TORCH_CHECK(
                    num_bits <= sizeof(edge_id_t) * 8,
                    "Selected edge_id_t must be capable of storing edge_ids.");
                // Using bfloat16 for random numbers works just as reliably as
                // float32 and provides around 30% speedup.
                using rnd_t = nv_bfloat16;
                auto randoms =
                    allocator.AllocateStorage<rnd_t>(num_edges.value());
                auto randoms_sorted =
                    allocator.AllocateStorage<rnd_t>(num_edges.value());
                auto edge_id_segments =
                    allocator.AllocateStorage<edge_id_t>(num_edges.value());
                auto sorted_edge_id_segments =
                    allocator.AllocateStorage<edge_id_t>(num_edges.value());
                AT_DISPATCH_INDEX_TYPES(
                    indices.scalar_type(), "SampleNeighborsIndices", ([&] {
                      using indices_t = index_t;
                      auto probs_or_mask_scalar_type = torch::kFloat32;
                      if (sliced_probs_or_mask.has_value()) {
                        probs_or_mask_scalar_type =
                            sliced_probs_or_mask->scalar_type();
                      }
                      GRAPHBOLT_DISPATCH_ALL_TYPES(
                          probs_or_mask_scalar_type, "SampleNeighborsProbs",
                          ([&] {
                            using probs_t = scalar_t;
                            probs_t* sliced_probs_ptr = nullptr;
                            if (sliced_probs_or_mask.has_value()) {
                              sliced_probs_ptr =
                                  sliced_probs_or_mask->data_ptr<probs_t>();
                            }
                            const indices_t* indices_ptr =
                                layer ? indices.data_ptr<indices_t>() : nullptr;
                            const dim3 block(BLOCK_SIZE);
                            const dim3 grid(
                                (num_edges.value() + BLOCK_SIZE - 1) /
                                BLOCK_SIZE);
                            // Compute row and random number pairs.
                            CUDA_KERNEL_CALL(
                                _ComputeRandoms, grid, block, 0,
                                num_edges.value(),
                                sliced_indptr.data_ptr<indptr_t>(),
                                sub_indptr.data_ptr<indptr_t>(),
                                coo_rows.data_ptr<indices_t>(),
                                sliced_probs_ptr, indices_ptr, random_seed,
                                randoms.get(), edge_id_segments.get());
                          }));
                    }));

                // Sort the random numbers along with edge ids, after
                // sorting the first fanout elements of each row will
                // give us the sampled edges.
                CUB_CALL(
                    DeviceSegmentedSort::SortPairs, randoms.get(),
                    randoms_sorted.get(), edge_id_segments.get(),
                    sorted_edge_id_segments.get(), num_edges.value(), num_rows,
                    sub_indptr.data_ptr<indptr_t>(),
                    sub_indptr.data_ptr<indptr_t>() + 1);

                picked_eids = torch::empty(
                    static_cast<indptr_t>(num_sampled_edges),
                    sub_indptr.options());

                // Need to sort the sampled edges only when fanouts.size() == 1
                // since multiple fanout sampling case is automatically going to
                // be sorted.
                if (type_per_edge && fanouts.size() == 1) {
                  // Ensuring sort result still ends up in
                  // sorted_edge_id_segments
                  std::swap(edge_id_segments, sorted_edge_id_segments);
                  auto sampled_segment_end_it = thrust::make_transform_iterator(
                      iota,
                      SegmentEndFunc<indptr_t, decltype(sampled_degree)>{
                          sub_indptr.data_ptr<indptr_t>(), sampled_degree});
                  CUB_CALL(
                      DeviceSegmentedSort::SortKeys, edge_id_segments.get(),
                      sorted_edge_id_segments.get(), picked_eids.size(0),
                      num_rows, sub_indptr.data_ptr<indptr_t>(),
                      sampled_segment_end_it);
                }

                auto input_buffer_it = thrust::make_transform_iterator(
                    iota, IteratorFunc<indptr_t, edge_id_t>{
                              sub_indptr.data_ptr<indptr_t>(),
                              sorted_edge_id_segments.get()});
                auto output_buffer_it = thrust::make_transform_iterator(
                    iota, IteratorFuncAddOffset<indptr_t, indptr_t>{
                              output_indptr.data_ptr<indptr_t>(),
                              sliced_indptr.data_ptr<indptr_t>(),
                              picked_eids.data_ptr<indptr_t>()});
                constexpr int64_t max_copy_at_once =
                    std::numeric_limits<int32_t>::max();

                // Copy the sampled edge ids into picked_eids tensor.
                for (int64_t i = 0; i < num_rows; i += max_copy_at_once) {
                  CUB_CALL(
                      DeviceCopy::Batched, input_buffer_it + i,
                      output_buffer_it + i, sampled_degree + i,
                      std::min(num_rows - i, max_copy_at_once));
                }
              }));
        } else {  // Non-weighted neighbor sampling.
          picked_eids = torch::zeros(num_edges.value(), sub_indptr.options());
          const auto sort_needed = type_per_edge && fanouts.size() == 1;
          const auto sliced_indptr_ptr =
              sort_needed ? nullptr : sliced_indptr.data_ptr<indptr_t>();

          const dim3 block(BLOCK_SIZE);
          const dim3 grid(
              (std::min(num_edges.value(), static_cast<int64_t>(1 << 20)) +
               BLOCK_SIZE - 1) /
              BLOCK_SIZE);
          AT_DISPATCH_INDEX_TYPES(
              indices.scalar_type(), "SampleNeighborsIndices", ([&] {
                using indices_t = index_t;
                // Compute row and random number pairs.
                CUDA_KERNEL_CALL(
                    _ComputeRandomsNS, grid, block, 0, num_edges.value(),
                    sliced_indptr_ptr, sub_indptr.data_ptr<indptr_t>(),
                    output_indptr.data_ptr<indptr_t>(),
                    coo_rows.data_ptr<indices_t>(), random_seed.get_seed(0),
                    picked_eids.data_ptr<indptr_t>());
              }));

          picked_eids =
              picked_eids.slice(0, 0, static_cast<indptr_t>(num_sampled_edges));

          // Need to sort the sampled edges only when fanouts.size() == 1
          // since multiple fanout sampling case is automatically going to
          // be sorted.
          if (sort_needed) {
            const int num_bits = compute_num_bits();
            std::array<int, 4> type_bits = {8, 15, 31, 63};
            const auto type_index =
                std::lower_bound(type_bits.begin(), type_bits.end(), num_bits) -
                type_bits.begin();
            std::array<torch::ScalarType, 5> types = {
                torch::kByte, torch::kInt16, torch::kInt32, torch::kLong,
                torch::kLong};
            auto edge_id_dtype = types[type_index];
            AT_DISPATCH_INTEGRAL_TYPES(
                edge_id_dtype, "SampleNeighborsEdgeIDs", ([&] {
                  using edge_id_t = scalar_t;
                  TORCH_CHECK(
                      num_bits <= sizeof(edge_id_t) * 8,
                      "Selected edge_id_t must be capable of storing "
                      "edge_ids.");
                  auto picked_offsets = picked_eids.to(edge_id_dtype);
                  auto sorted_offsets = torch::empty_like(picked_offsets);
                  CUB_CALL(
                      DeviceSegmentedSort::SortKeys,
                      picked_offsets.data_ptr<edge_id_t>(),
                      sorted_offsets.data_ptr<edge_id_t>(), picked_eids.size(0),
                      num_rows, output_indptr.data_ptr<indptr_t>(),
                      output_indptr.data_ptr<indptr_t>() + 1);
                  auto edge_id_offsets = ExpandIndptrImpl(
                      output_indptr, picked_eids.scalar_type(), sliced_indptr,
                      picked_eids.size(0));
                  picked_eids = sorted_offsets.to(picked_eids.scalar_type()) +
                                edge_id_offsets;
                }));
          }
        }

        if (!returning_indices_is_optional || utils::is_on_gpu(indices)) {
          output_indices = Gather(indices, picked_eids);
        }
      }));

  torch::optional<torch::Tensor> output_type_per_edge;
  torch::optional<torch::Tensor> edge_offsets;
  if (type_per_edge && seed_offsets) {
    const int64_t num_etypes =
        edge_type_to_id.has_value() ? edge_type_to_id->size() : 1;
    // If we performed homogenous sampling on hetero graph, we have to look at
    // type_per_edge of sampled edges and determine the offsets of different
    // sampled etypes and convert to fused hetero indptr representation.
    if (fanouts.size() == 1) {
      output_type_per_edge = Gather(*type_per_edge, picked_eids);
      torch::Tensor output_in_degree, sliced_output_indptr;
      sliced_output_indptr =
          output_indptr.slice(0, 0, output_indptr.size(0) - 1);
      std::tie(output_indptr, output_in_degree, sliced_output_indptr) =
          SliceCSCIndptrHetero(
              output_indptr, output_type_per_edge.value(), sliced_output_indptr,
              num_etypes);
      // We use num_rows to hold num_seeds * num_etypes. So, it needs to be
      // updated when sampling with a single fanout value when the graph is
      // heterogenous.
      num_rows = sliced_output_indptr.size(0);
    }
    // Here, we check what are the dst node types for the given seeds so that
    // we can compute the output indptr space later.
    std::vector<int64_t> etype_id_to_dst_ntype_id(num_etypes);
    // Here, we check what are the src node types for the given seeds so that
    // we can subtract source node offset from indices later.
    auto etype_id_to_src_ntype_id = torch::empty(
        2 * num_etypes,
        c10::TensorOptions().dtype(torch::kLong).pinned_memory(true));
    auto etype_id_to_src_ntype_id_ptr =
        etype_id_to_src_ntype_id.data_ptr<int64_t>();
    for (auto& etype_and_id : edge_type_to_id.value()) {
      auto etype = etype_and_id.key();
      auto id = etype_and_id.value();
      auto [src_type, dst_type] = utils::parse_src_dst_ntype_from_etype(etype);
      etype_id_to_dst_ntype_id[id] = node_type_to_id->at(dst_type);
      etype_id_to_src_ntype_id_ptr[2 * id] =
          etype_id_to_src_ntype_id_ptr[2 * id + 1] =
              node_type_to_id->at(src_type);
    }
    auto indices_offsets_device = torch::empty(
        etype_id_to_src_ntype_id.size(0),
        picked_eids.options().dtype(torch::kLong));
    AT_DISPATCH_INDEX_TYPES(
        node_type_offset->scalar_type(), "SampleNeighborsNodeTypeOffset", ([&] {
          THRUST_CALL(
              gather, etype_id_to_src_ntype_id_ptr,
              etype_id_to_src_ntype_id_ptr + etype_id_to_src_ntype_id.size(0),
              node_type_offset->data_ptr<index_t>(),
              indices_offsets_device.data_ptr<int64_t>());
        }));
    // For each edge type, we compute the start and end offsets to index into
    // indptr to form the final output_indptr.
    auto indptr_offsets = torch::empty(
        num_etypes * 2,
        c10::TensorOptions().dtype(torch::kLong).pinned_memory(true));
    auto indptr_offsets_ptr = indptr_offsets.data_ptr<int64_t>();
    // We compute the indptr offsets here, right now, output_indptr is of size
    // # seeds * num_etypes + 1. We can simply take slices to get correct output
    // indptr. The final output_indptr is same as current indptr except that
    // some intermediate values are removed to change the node ids space from
    // all of the seed vertices to the node id space of the dst node type of
    // each edge type.
    for (int i = 0; i < num_etypes; i++) {
      indptr_offsets_ptr[2 * i] = num_rows / num_etypes * i +
                                  seed_offsets->at(etype_id_to_dst_ntype_id[i]);
      indptr_offsets_ptr[2 * i + 1] =
          num_rows / num_etypes * i +
          seed_offsets->at(etype_id_to_dst_ntype_id[i] + 1);
    }
    auto permutation = torch::arange(
        0, num_rows * num_etypes, num_etypes, output_indptr.options());
    permutation =
        permutation.remainder(num_rows) + permutation.div(num_rows, "floor");
    // This permutation, when applied sorts the sampled edges with respect to
    // edge types.
    auto [output_in_degree, sliced_output_indptr] =
        SliceCSCIndptr(output_indptr, permutation);
    std::tie(output_indptr, picked_eids) = IndexSelectCSCImpl(
        output_in_degree, sliced_output_indptr, picked_eids, permutation,
        num_rows - 1, picked_eids.size(0));
    edge_offsets = torch::empty(
        num_etypes * 2, c10::TensorOptions()
                            .dtype(output_indptr.scalar_type())
                            .pinned_memory(true));
    auto edge_offsets_device =
        torch::empty(num_etypes * 2, output_indptr.options());
    at::cuda::CUDAEvent edge_offsets_event;
    AT_DISPATCH_INDEX_TYPES(
        indptr.scalar_type(), "SampleNeighborsEdgeOffsets", ([&] {
          auto edge_offsets_pinned_device_pair =
              thrust::make_transform_output_iterator(
                  thrust::make_zip_iterator(
                      edge_offsets->data_ptr<index_t>(),
                      edge_offsets_device.data_ptr<index_t>()),
                  [=] __device__(index_t x) {
                    return thrust::make_tuple(x, x);
                  });
          THRUST_CALL(
              gather, indptr_offsets_ptr,
              indptr_offsets_ptr + indptr_offsets.size(0),
              output_indptr.data_ptr<index_t>(),
              edge_offsets_pinned_device_pair);
        }));
    edge_offsets_event.record();
    if (output_indices.has_value()) {
      auto indices_offset_subtract = ExpandIndptrImpl(
          edge_offsets_device, indices.scalar_type(), indices_offsets_device,
          output_indices->size(0));
      // The output_indices is permuted here.
      std::tie(output_indptr, output_indices) = IndexSelectCSCImpl(
          output_in_degree, sliced_output_indptr, *output_indices, permutation,
          num_rows - 1, output_indices->size(0));
      *output_indices -= indices_offset_subtract;
    }
    auto output_indptr_offsets = torch::empty(
        num_etypes * 2,
        c10::TensorOptions().dtype(torch::kLong).pinned_memory(true));
    auto output_indptr_offsets_ptr = output_indptr_offsets.data_ptr<int64_t>();
    std::vector<torch::Tensor> indptr_list;
    for (int i = 0; i < num_etypes; i++) {
      indptr_list.push_back(output_indptr.slice(
          0, indptr_offsets_ptr[2 * i], indptr_offsets_ptr[2 * i + 1] + 1));
      output_indptr_offsets_ptr[2 * i] =
          i == 0 ? 0 : output_indptr_offsets_ptr[2 * i - 1];
      output_indptr_offsets_ptr[2 * i + 1] =
          output_indptr_offsets_ptr[2 * i] + indptr_list.back().size(0);
    }
    auto output_indptr_offsets_device = torch::empty(
        output_indptr_offsets.size(0),
        output_indptr.options().dtype(torch::kLong));
    THRUST_CALL(
        copy_n, output_indptr_offsets_ptr, output_indptr_offsets.size(0),
        output_indptr_offsets_device.data_ptr<int64_t>());
    // We form the final output indptr by concatenating pieces for different
    // edge types.
    output_indptr = torch::cat(indptr_list);
    auto indptr_offset_subtract = ExpandIndptrImpl(
        output_indptr_offsets_device, indptr.scalar_type(), edge_offsets_device,
        output_indptr.size(0));
    output_indptr -= indptr_offset_subtract;
    edge_offsets_event.synchronize();
    // We read the edge_offsets here, they are in pairs but we don't need it to
    // be in pairs. So we remove the duplicate information from it and turn it
    // into a real offsets array.
    AT_DISPATCH_INDEX_TYPES(
        indptr.scalar_type(), "SampleNeighborsEdgeOffsetsCheck", ([&] {
          auto edge_offsets_ptr = edge_offsets->data_ptr<index_t>();
          TORCH_CHECK(edge_offsets_ptr[0] == 0, "edge_offsets is incorrect.");
          for (int i = 1; i < num_etypes; i++) {
            TORCH_CHECK(
                edge_offsets_ptr[2 * i - 1] == edge_offsets_ptr[2 * i],
                "edge_offsets is incorrect.");
          }
          TORCH_CHECK(
              edge_offsets_ptr[2 * num_etypes - 1] == picked_eids.size(0),
              "edge_offsets is incorrect.");
          for (int i = 0; i < num_etypes; i++) {
            edge_offsets_ptr[i + 1] = edge_offsets_ptr[2 * i + 1];
          }
        }));
    edge_offsets = edge_offsets->slice(0, 0, num_etypes + 1);
  } else {
    // Convert output_indptr back to homo by discarding intermediate offsets.
    output_indptr =
        output_indptr.slice(0, 0, output_indptr.size(0), fanouts.size());
    if (type_per_edge)
      output_type_per_edge = Gather(*type_per_edge, picked_eids);
  }

  return c10::make_intrusive<sampling::FusedSampledSubgraph>(
      output_indptr, output_indices, picked_eids, seeds, torch::nullopt,
      output_type_per_edge, edge_offsets);
}

}  //  namespace ops
}  //  namespace graphbolt
