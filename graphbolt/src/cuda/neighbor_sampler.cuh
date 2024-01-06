/**
 *  Copyright (c) 2023 by Contributors
 *  Copyright (c) 2023, GT-TDAlab (Muhammed Fatih Balin & Umit V. Catalyurek)
 * @file cuda/neighbor_sampler.cuh
 * @brief SampleNeighbor operator implementation templated on indptr_t for CUDA.
 */
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAStream.h>
#include <curand_kernel.h>
#include <graphbolt/cuda_ops.h>
#include <graphbolt/cuda_sampling_ops.h>
#include <thrust/gather.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>

#include <algorithm>
#include <array>
#include <cub/cub.cuh>
#include <cuda/std/tuple>
#include <limits>
#include <numeric>
#include <type_traits>

#include "../random.h"
#include "./common.h"
#include "./utils.h"

namespace graphbolt {
namespace ops {

constexpr int BLOCK_SIZE = 128;

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
    const weights_t* const weights, const indices_t* const indices,
    const uint64_t random_seed, float_t* random_arr, edge_id_t* edge_ids) {
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
    const auto exp_rnd = -__logf(rnd);
    const float_t adjusted_rnd = prob > 0
                                     ? static_cast<float_t>(exp_rnd / prob)
                                     : std::numeric_limits<float_t>::infinity();
    random_arr[i] = adjusted_rnd;
    edge_ids[i] = row_offset;

    i += stride;
  }
}

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

template <typename indptr_t>
c10::intrusive_ptr<sampling::FusedSampledSubgraph> SampleNeighborsIndptr(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor nodes,
    const std::vector<int64_t>& fanouts, bool replace, bool layer,
    bool return_eids, torch::optional<torch::Tensor> type_per_edge,
    torch::optional<torch::Tensor> probs_or_mask) {
  TORCH_CHECK(!replace, "Sampling with replacement is not supported yet!");
  // Assume that indptr, indices, nodes, type_per_edge and probs_or_mask
  // are all resident on the GPU. If not, it is better to first extract them
  // before calling this function.
  auto allocator = cuda::GetAllocator();
  const auto stream = cuda::GetCurrentStream();
  auto num_rows = nodes.size(0);
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
      sizeof(int64_t) * fanouts.size(), cudaMemcpyHostToDevice, stream));
  auto in_degree_and_sliced_indptr = SliceCSCIndptr(indptr, nodes);
  auto in_degree = std::get<0>(in_degree_and_sliced_indptr);
  auto sliced_indptr = std::get<1>(in_degree_and_sliced_indptr);
  auto sub_indptr = ExclusiveCumSum(in_degree);
  if (fanouts.size() > 1) {
    torch::Tensor sliced_type_per_edge;
    std::tie(sub_indptr, sliced_type_per_edge) =
        IndexSelectCSCImpl(indptr, type_per_edge.value(), nodes);
    std::tie(sub_indptr, in_degree, sliced_indptr) = SliceCSCIndptrHetero(
        sub_indptr, sliced_type_per_edge, sliced_indptr, fanouts.size());
    num_rows = sliced_indptr.size(0);
  }
  auto max_in_degree = torch::empty(
      1,
      c10::TensorOptions().dtype(in_degree.scalar_type()).pinned_memory(true));
  {  // indptr_t
    size_t tmp_storage_size = 0;
    cub::DeviceReduce::Max(
        nullptr, tmp_storage_size, in_degree.data_ptr<indptr_t>(),
        max_in_degree.data_ptr<indptr_t>(), num_rows, stream);
    auto tmp_storage = allocator.AllocateStorage<char>(tmp_storage_size);
    cub::DeviceReduce::Max(
        tmp_storage.get(), tmp_storage_size, in_degree.data_ptr<indptr_t>(),
        max_in_degree.data_ptr<indptr_t>(), num_rows, stream);
  }
  auto coo_rows = CSRToCOO(sub_indptr, indices.scalar_type());
  const auto num_edges = coo_rows.size(0);
  const auto random_seed = RandomEngine::ThreadLocal()->RandInt(
      static_cast<int64_t>(0), std::numeric_limits<int64_t>::max());
  auto output_indptr = torch::empty_like(sub_indptr);
  torch::Tensor picked_eids;
  torch::Tensor output_indices;
  torch::optional<torch::Tensor> output_type_per_edge;

  {  // indptr_t
    thrust::counting_iterator<int64_t> iota(0);
    auto sampled_degree = thrust::make_transform_iterator(
        iota, MinInDegreeFanout<indptr_t>{
                  in_degree.data_ptr<indptr_t>(), fanouts_device.get(),
                  fanouts.size()});

    {  // Compute output_indptr.
      size_t tmp_storage_size = 0;
      cub::DeviceScan::ExclusiveSum(
          nullptr, tmp_storage_size, sampled_degree,
          output_indptr.data_ptr<indptr_t>(), num_rows + 1, stream);
      auto tmp_storage = allocator.AllocateStorage<char>(tmp_storage_size);
      cub::DeviceScan::ExclusiveSum(
          tmp_storage.get(), tmp_storage_size, sampled_degree,
          output_indptr.data_ptr<indptr_t>(), num_rows + 1, stream);
    }

    auto num_sampled_edges =
        cuda::CopyScalar{output_indptr.data_ptr<indptr_t>() + num_rows};

    // Find the smallest integer type to store the edge id offsets.
    // CSRToCOO had synch inside, so it is safe to read max_in_degree now.
    const int num_bits =
        cuda::NumberOfBits(max_in_degree.data_ptr<indptr_t>()[0]);
    std::array<int, 4> type_bits = {8, 16, 32, 64};
    const auto type_index =
        std::lower_bound(type_bits.begin(), type_bits.end(), num_bits) -
        type_bits.begin();
    std::array<torch::ScalarType, 5> types = {
        torch::kByte, torch::kInt16, torch::kInt32, torch::kLong, torch::kLong};
    auto edge_id_dtype = types[type_index];
    AT_DISPATCH_INTEGRAL_TYPES(
        edge_id_dtype, "SampleNeighborsEdgeIDs", ([&] {
          using edge_id_t = std::make_unsigned_t<scalar_t>;
          TORCH_CHECK(
              num_bits <= sizeof(edge_id_t) * 8,
              "Selected edge_id_t must be capable of storing edge_ids.");
          // Using bfloat16 for random numbers works just as reliably as
          // float32 and provides around %30 percent speedup.
          using rnd_t = nv_bfloat16;
          auto randoms = allocator.AllocateStorage<rnd_t>(num_edges);
          auto randoms_sorted = allocator.AllocateStorage<rnd_t>(num_edges);
          auto edge_id_segments =
              allocator.AllocateStorage<edge_id_t>(num_edges);
          auto sorted_edge_id_segments =
              allocator.AllocateStorage<edge_id_t>(num_edges);
          AT_DISPATCH_INTEGRAL_TYPES(
              indices.scalar_type(), "SampleNeighborsIndices", ([&] {
                using indices_t = scalar_t;
                auto probs_or_mask_scalar_type = torch::kFloat32;
                if (probs_or_mask.has_value()) {
                  probs_or_mask_scalar_type =
                      probs_or_mask.value().scalar_type();
                }
                GRAPHBOLT_DISPATCH_ALL_TYPES(
                    probs_or_mask_scalar_type, "SampleNeighborsProbs", ([&] {
                      using probs_t = scalar_t;
                      probs_t* probs_ptr = nullptr;
                      if (probs_or_mask.has_value()) {
                        probs_ptr = probs_or_mask.value().data_ptr<probs_t>();
                      }
                      const indices_t* indices_ptr =
                          layer ? indices.data_ptr<indices_t>() : nullptr;
                      const dim3 block(BLOCK_SIZE);
                      const dim3 grid(
                          (num_edges + BLOCK_SIZE - 1) / BLOCK_SIZE);
                      // Compute row and random number pairs.
                      CUDA_KERNEL_CALL(
                          _ComputeRandoms, grid, block, 0, stream, num_edges,
                          sliced_indptr.data_ptr<indptr_t>(),
                          sub_indptr.data_ptr<indptr_t>(),
                          coo_rows.data_ptr<indices_t>(), probs_ptr,
                          indices_ptr, random_seed, randoms.get(),
                          edge_id_segments.get());
                    }));
              }));

          // Sort the random numbers along with edge ids, after
          // sorting the first fanout elements of each row will
          // give us the sampled edges.
          size_t tmp_storage_size = 0;
          CUDA_CALL(cub::DeviceSegmentedSort::SortPairs(
              nullptr, tmp_storage_size, randoms.get(), randoms_sorted.get(),
              edge_id_segments.get(), sorted_edge_id_segments.get(), num_edges,
              num_rows, sub_indptr.data_ptr<indptr_t>(),
              sub_indptr.data_ptr<indptr_t>() + 1, stream));
          auto tmp_storage = allocator.AllocateStorage<char>(tmp_storage_size);
          CUDA_CALL(cub::DeviceSegmentedSort::SortPairs(
              tmp_storage.get(), tmp_storage_size, randoms.get(),
              randoms_sorted.get(), edge_id_segments.get(),
              sorted_edge_id_segments.get(), num_edges, num_rows,
              sub_indptr.data_ptr<indptr_t>(),
              sub_indptr.data_ptr<indptr_t>() + 1, stream));

          picked_eids = torch::empty(
              static_cast<indptr_t>(num_sampled_edges),
              nodes.options().dtype(indptr.scalar_type()));

          // Need to sort the sampled edges only when fanouts.size() == 1
          // since multiple fanout sampling case is automatically going to
          // be sorted.
          if (type_per_edge && fanouts.size() == 1) {
            // Ensuring sort result still ends up in sorted_edge_id_segments
            std::swap(edge_id_segments, sorted_edge_id_segments);
            auto sampled_segment_end_it = thrust::make_transform_iterator(
                iota, SegmentEndFunc<indptr_t, decltype(sampled_degree)>{
                          sub_indptr.data_ptr<indptr_t>(), sampled_degree});
            size_t tmp_storage_size = 0;
            CUDA_CALL(cub::DeviceSegmentedSort::SortKeys(
                nullptr, tmp_storage_size, edge_id_segments.get(),
                sorted_edge_id_segments.get(), picked_eids.size(0), num_rows,
                sub_indptr.data_ptr<indptr_t>(), sampled_segment_end_it,
                stream));
            auto tmp_storage =
                allocator.AllocateStorage<char>(tmp_storage_size);
            CUDA_CALL(cub::DeviceSegmentedSort::SortKeys(
                tmp_storage.get(), tmp_storage_size, edge_id_segments.get(),
                sorted_edge_id_segments.get(), picked_eids.size(0), num_rows,
                sub_indptr.data_ptr<indptr_t>(), sampled_segment_end_it,
                stream));
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
            size_t tmp_storage_size = 0;
            CUDA_CALL(cub::DeviceCopy::Batched(
                nullptr, tmp_storage_size, input_buffer_it + i,
                output_buffer_it + i, sampled_degree + i,
                std::min(num_rows - i, max_copy_at_once), stream));
            auto tmp_storage =
                allocator.AllocateStorage<char>(tmp_storage_size);
            CUDA_CALL(cub::DeviceCopy::Batched(
                tmp_storage.get(), tmp_storage_size, input_buffer_it + i,
                output_buffer_it + i, sampled_degree + i,
                std::min(num_rows - i, max_copy_at_once), stream));
          }
        }));

    output_indices = torch::empty(
        picked_eids.size(0),
        picked_eids.options().dtype(indices.scalar_type()));

    // Compute: output_indices = indices.gather(0, picked_eids);
    AT_DISPATCH_INTEGRAL_TYPES(
        indices.scalar_type(), "SampleNeighborsOutputIndices", ([&] {
          using indices_t = scalar_t;
          const auto exec_policy =
              thrust::cuda::par_nosync(allocator).on(stream);
          thrust::gather(
              exec_policy, picked_eids.data_ptr<indptr_t>(),
              picked_eids.data_ptr<indptr_t>() + picked_eids.size(0),
              indices.data_ptr<indices_t>(),
              output_indices.data_ptr<indices_t>());
        }));

    if (type_per_edge) {
      // output_type_per_edge = type_per_edge.gather(0, picked_eids);
      // The commented out torch equivalent above does not work when
      // type_per_edge is on pinned memory. That is why, we have to
      // reimplement it, similar to the indices gather operation above.
      auto types = type_per_edge.value();
      output_type_per_edge = torch::empty(
          picked_eids.size(0),
          picked_eids.options().dtype(types.scalar_type()));
      AT_DISPATCH_INTEGRAL_TYPES(
          types.scalar_type(), "SampleNeighborsOutputTypePerEdge", ([&] {
            const auto exec_policy =
                thrust::cuda::par_nosync(allocator).on(stream);
            thrust::gather(
                exec_policy, picked_eids.data_ptr<indptr_t>(),
                picked_eids.data_ptr<indptr_t>() + picked_eids.size(0),
                types.data_ptr<scalar_t>(),
                output_type_per_edge.value().data_ptr<scalar_t>());
          }));
    }
  }

  // Convert output_indptr back to homo by discarding intermediate offsets.
  output_indptr =
      output_indptr.slice(0, 0, output_indptr.size(0), fanouts.size());
  torch::optional<torch::Tensor> subgraph_reverse_edge_ids = torch::nullopt;
  if (return_eids) subgraph_reverse_edge_ids = std::move(picked_eids);

  return c10::make_intrusive<sampling::FusedSampledSubgraph>(
      output_indptr, output_indices, nodes, torch::nullopt,
      subgraph_reverse_edge_ids, output_type_per_edge);
}

#define INSTANTIATE_NEIGHBOR_SAMPLER(type)                              \
  template c10::intrusive_ptr<sampling::FusedSampledSubgraph>           \
  SampleNeighborsIndptr<type>(                                          \
      torch::Tensor indptr, torch::Tensor indices, torch::Tensor nodes, \
      const std::vector<int64_t>& fanouts, bool replace, bool layer,    \
      bool return_eids, torch::optional<torch::Tensor> type_per_edge,   \
      torch::optional<torch::Tensor> probs_or_mask)

}  //  namespace ops
}  //  namespace graphbolt
