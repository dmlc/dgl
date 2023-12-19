/**
 *  Copyright (c) 2023 by Contributors
 *  Copyright (c) 2023, GT-TDAlab (Muhammed Fatih Balin & Umit V. Catalyurek)
 * @file cuda/index_select_impl.cu
 * @brief Index select operator implementation on CUDA.
 */
#include <ATen/cuda/CUDAEvent.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAStream.h>
#include <curand_kernel.h>
#include <graphbolt/cuda_ops.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <cub/cub.cuh>
#include <cuda/std/tuple>
#include <limits>
#include <numeric>

#include "../random.h"
#include "./common.h"
#include "./utils.h"

namespace graphbolt {
namespace ops {

constexpr int BLOCK_SIZE = 128;

template <
    typename float_t, typename indptr_t, typename indices_t, typename weights_t>
__global__ void _ComputeRowRandomPairs(
    const int64_t num_edges, const indptr_t* const sliced_indptr,
    const indptr_t* const sub_indptr, const indices_t* const csr_rows,
    const weights_t* const weights, const indices_t* const indices,
    const uint64_t random_seed, ::cuda::std::tuple<indices_t, float_t>* output,
    indptr_t* edge_ids) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  curandStatePhilox4_32_10_t rng;
  const auto labor = indices != nullptr;

  if (!labor) {
    curand_init(random_seed, i, 0, &rng);
  }

  while (i < num_edges) {
    const auto row_position = csr_rows[i];
    const auto row_offset = i - sub_indptr[row_position];
    const auto in_idx = sliced_indptr[row_position] + row_offset;

    if (labor) {
      constexpr uint64_t kCurandSeed = 999961;
      curand_init(kCurandSeed, random_seed, indices[in_idx], &rng);
    }

    const auto rnd = curand_uniform(&rng);
    const auto prob = weights ? weights[in_idx] : static_cast<weights_t>(1);
    const float_t adjusted_prob = -__logf(rnd) / prob;
    output[i] = {row_position, adjusted_prob};
    edge_ids[i] = in_idx;

    i += stride;
  }
}

template <typename indices_t, typename float_t>
struct decomposer_t {
  __host__ __device__ ::cuda::std::tuple<indices_t&, float_t&> operator()(
      ::cuda::std::tuple<indices_t, float_t>& key) const {
    auto& [t, prob] = key;
    return {t, prob};
  }
};

template <typename indptr_t>
struct MinInDegreeFanout {
  const indptr_t* in_degree;
  int64_t fanout;
  __host__ __device__ auto operator()(int64_t i) {
    return static_cast<indptr_t>(
        min(static_cast<int64_t>(in_degree[i]), fanout));
  }
};

template <typename indptr_t, typename indices_t>
struct IteratorFunc {
  indptr_t* indptr;
  indices_t* indices;
  __host__ __device__ auto operator()(int64_t i) { return indices + indptr[i]; }
};

template <typename indices_t>
struct ConvertToBytes {
  template <typename indptr_t>
  __host__ __device__ auto operator()(indptr_t num_elements) {
    return num_elements * sizeof(indices_t);
  }
};

c10::intrusive_ptr<sampling::FusedSampledSubgraph> SampleNeighbors(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor nodes,
    const std::vector<int64_t>& fanouts, bool replace, bool layer,
    bool return_eids, torch::optional<torch::Tensor> type_per_edge,
    torch::optional<torch::Tensor> probs_or_mask,
    torch::optional<int64_t> random_seed) {
  TORCH_CHECK(
      fanouts.size() == 1, "Heterogenous sampling is not supported yet!");
  TORCH_CHECK(!replace, "Sampling with replacement is not supported yet!");
  // We assume that indptr, indices, nodes, type_per_edge and probs_or_mask
  // are all resident on the GPU. If not, it is better to first extract them.
  const auto num_rows = nodes.size(0);
  const auto fanout =
      fanouts[0] >= 0 ? fanouts[0] : std::numeric_limits<int64_t>::max();
  auto in_degree_and_sliced_indptr = SliceCSCIndptr(indptr, nodes);
  auto in_degree = std::get<0>(in_degree_and_sliced_indptr);
  auto sliced_indptr = std::get<1>(in_degree_and_sliced_indptr);
  auto sub_indptr = ExclusiveCumSum(in_degree);
  auto output_indptr = torch::empty_like(sub_indptr);
  auto coo_rows = CSRToCOO(sub_indptr, indices.scalar_type());
  const auto num_edges = coo_rows.size(0);
  auto allocator = cuda::GetAllocator();
  const auto stream = cuda::GetCurrentStream();
  if (!random_seed.has_value()) {
    random_seed = RandomEngine::ThreadLocal()->RandInt(
        static_cast<int64_t>(0), std::numeric_limits<int64_t>::max());
  }
  torch::Tensor picked_eids;

  AT_DISPATCH_INTEGRAL_TYPES(
      indptr.scalar_type(), "SampleNeighborsWithoutReplacementIndptr", ([&] {
        using indptr_t = scalar_t;
        thrust::counting_iterator<int64_t> iota(0);
        auto sampled_degree = thrust::make_transform_iterator(
            iota, MinInDegreeFanout<indptr_t>{
                      in_degree.data_ptr<indptr_t>(), fanout});
        {
          size_t tmp_storage_size = 0;
          cub::DeviceScan::ExclusiveSum(
              nullptr, tmp_storage_size, sampled_degree,
              output_indptr.data_ptr<indptr_t>(), num_rows + 1, stream);
          auto tmp_storage = allocator.AllocateStorage<char>(tmp_storage_size);
          cub::DeviceScan::ExclusiveSum(
              tmp_storage.get(), tmp_storage_size, sampled_degree,
              output_indptr.data_ptr<indptr_t>(), num_rows + 1, stream);
        }
        auto num_sampled_edges = torch::empty(
            1, c10::TensorOptions()
                   .dtype(indptr.scalar_type())
                   .pinned_memory(true));
        CUDA_CALL(cudaMemcpyAsync(
            num_sampled_edges.data_ptr<indptr_t>(),
            output_indptr.data_ptr<indptr_t>() + num_rows, sizeof(indptr_t),
            cudaMemcpyDeviceToHost, stream));
        at::cuda::CUDAEvent copy_event;
        copy_event.record(stream);
        auto sorted_edge_id_segments =
            allocator.AllocateStorage<indptr_t>(num_edges);
        AT_DISPATCH_INTEGRAL_TYPES(
            indices.scalar_type(), "SampleNeighborsWithoutReplacementIndices",
            ([&] {
              using indices_t = scalar_t;
              const indices_t* indices_ptr =
                  layer ? indices.data_ptr<indices_t>() : nullptr;
              auto row_and_prob =
                  allocator
                      .AllocateStorage<::cuda::std::tuple<indices_t, float>>(
                          num_edges);
              auto row_and_prob_sorted =
                  allocator
                      .AllocateStorage<::cuda::std::tuple<indices_t, float>>(
                          num_edges);
              auto probs_or_mask_scalar_type = torch::kFloat32;
              if (probs_or_mask.has_value()) {
                probs_or_mask_scalar_type = probs_or_mask.value().scalar_type();
              }
              GRAPHBOLT_DISPATCH_ALL_TYPES(
                  probs_or_mask_scalar_type,
                  "SampleNeighborsWithoutReplacementProbs", ([&] {
                    using probs_t = scalar_t;
                    probs_t* probs_ptr = nullptr;
                    if (probs_or_mask.has_value()) {
                      probs_ptr = probs_or_mask.value().data_ptr<probs_t>();
                    }
                    auto input_edge_id_segments =
                        allocator.AllocateStorage<indptr_t>(num_edges);
                    const dim3 block(BLOCK_SIZE);
                    const dim3 grid((num_edges + BLOCK_SIZE - 1) / BLOCK_SIZE);
                    CUDA_KERNEL_CALL(
                        _ComputeRowRandomPairs, grid, block, 0, stream,
                        num_edges, sliced_indptr.data_ptr<indptr_t>(),
                        sub_indptr.data_ptr<indptr_t>(),
                        coo_rows.data_ptr<indices_t>(), probs_ptr, indices_ptr,
                        random_seed.value(), row_and_prob.get(),
                        input_edge_id_segments.get());

                    size_t tmp_storage_size = 0;
                    CUDA_CALL(cub::DeviceRadixSort::SortPairs(
                        nullptr, tmp_storage_size, row_and_prob.get(),
                        row_and_prob_sorted.get(), input_edge_id_segments.get(),
                        sorted_edge_id_segments.get(), num_edges,
                        decomposer_t<indices_t, float>{}, 0,
                        sizeof(row_and_prob.get()[0]), stream));
                    auto tmp_storage =
                        allocator.AllocateStorage<char>(tmp_storage_size);
                    CUDA_CALL(cub::DeviceRadixSort::SortPairs(
                        tmp_storage.get(), tmp_storage_size, row_and_prob.get(),
                        row_and_prob_sorted.get(), input_edge_id_segments.get(),
                        sorted_edge_id_segments.get(), num_edges,
                        decomposer_t<indices_t, float>{}, 0,
                        sizeof(row_and_prob.get()[0]), stream));
                  }));
            }));

        // Now we are free to access num_sampled_edges
        copy_event.synchronize();
        picked_eids = torch::empty(
            *num_sampled_edges.data_ptr<indptr_t>(),
            nodes.options().dtype(indices.scalar_type()));

        auto input_buffer_it = thrust::make_transform_iterator(
            iota, IteratorFunc<indptr_t, indptr_t>{
                      sub_indptr.data_ptr<indptr_t>(),
                      sorted_edge_id_segments.get()});
        auto output_buffer_it = thrust::make_transform_iterator(
            iota, IteratorFunc<indptr_t, indptr_t>{
                      output_indptr.data_ptr<indptr_t>(),
                      picked_eids.data_ptr<indptr_t>()});
        auto buffer_sizes = thrust::make_transform_iterator(
            sampled_degree, ConvertToBytes<indptr_t>{});
        constexpr int64_t max_copy_at_once =
            std::numeric_limits<int32_t>::max();

        for (int64_t i = 0; i < num_rows; i += max_copy_at_once) {
          size_t tmp_storage_size = 0;
          CUDA_CALL(cub::DeviceMemcpy::Batched(
              nullptr, tmp_storage_size, input_buffer_it + i,
              output_buffer_it + i, buffer_sizes + i,
              std::min(num_rows - i, max_copy_at_once), stream));
          auto tmp_storage = allocator.AllocateStorage<char>(tmp_storage_size);
          CUDA_CALL(cub::DeviceMemcpy::Batched(
              tmp_storage.get(), tmp_storage_size, input_buffer_it + i,
              output_buffer_it + i, buffer_sizes + i,
              std::min(num_rows - i, max_copy_at_once), stream));
        }
      }));

  auto output_indices = indices.gather(0, picked_eids);
  torch::optional<torch::Tensor> subgraph_reverse_edge_ids = torch::nullopt;
  if (return_eids) subgraph_reverse_edge_ids = std::move(picked_eids);

  return c10::make_intrusive<sampling::FusedSampledSubgraph>(
      output_indptr, output_indices, nodes, torch::nullopt,
      subgraph_reverse_edge_ids, torch::nullopt);
}
}  //  namespace ops
}  //  namespace graphbolt
