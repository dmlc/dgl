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
 * @file cuda/gpu_graph_cache.cu
 * @brief GPU graph cache implementation on CUDA.
 */
#include <graphbolt/cuda_ops.h>
#include <thrust/gather.h>
#include <thrust/transform.h>

#include <cstddef>
#include <cub/cub.cuh>
#include <cuco/static_map.cuh>
#include <cuda/std/atomic>
#include <cuda/stream_ref>
#include <limits>
#include <numeric>
#include <type_traits>

#include "../common.h"
#include "../utils.h"
#include "./gpu_graph_cache.h"

namespace graphbolt {
namespace cuda {

namespace {

constexpr int cg_size = 1;
template <typename index_t>
using probing_t =
    cuco::linear_probing<cg_size, cuco::default_hash_function<index_t>>;
template <typename index_t>
using allocator_t = cuda::CUDAWorkspaceAllocator<cuco::pair<index_t, index_t>>;
template <typename index_t>
using map_t = cuco::static_map<
    index_t, index_t, cuco::extent<int64_t>, ::cuda::thread_scope_device,
    thrust::equal_to<index_t>, probing_t<index_t>, allocator_t<index_t>>;

template <typename index_t, typename map_t>
__global__ void _Insert(
    const int64_t num_nodes, const index_t num_existing, const index_t* seeds,
    const index_t* missing_indices, const index_t* indices, map_t map) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;

  while (i < num_nodes) {
    const auto key = seeds[missing_indices[indices[i]]];

    auto slot = map.find(key);
    slot->second = num_existing + i;

    i += stride;
  }
}

/**
 * @brief For node ids not in the cache, it keeps their access count inside
 * a hash table as (v, -c) where v is the node id and c is the access count.
 * When c == -threshold, it means that v will be inserted into the cache
 * during the call to the replace method. Once v is inserted into the cache,
 * c is assigned to a nonnegative value and indicates the local id of vertex
 * v in the cache.
 *
 * @param num_nodes The number of node ids.
 * @param seeds The node ids the cache is being queried with.
 * @param positions Holds the values found in the hash table.
 * @param map The hash table holding (v, -c) or (v, local_id).
 *
 */
template <typename index_t, typename map_t>
__global__ void _QueryAndIncrement(
    const int64_t num_nodes, const index_t* seeds, index_t* positions,
    map_t map) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;

  while (i < num_nodes) {
    const auto key = seeds[i];

    constexpr index_t minusONE = -1;
    auto [slot, is_new_key] = map.insert_and_find(cuco::pair{key, minusONE});

    int64_t position = -1;

    if (!is_new_key) {
      auto ref = ::cuda::atomic_ref<index_t, ::cuda::thread_scope_device>{
          slot->second};
      position = ref.load(::cuda::memory_order_relaxed);
      if (position < 0) {
        position = ref.fetch_add(-1, ::cuda::memory_order_relaxed) - 1;
      }
    }

    positions[i] = position;

    i += stride;
  }
}

constexpr int kIntBlockSize = 512;
}  // namespace

c10::intrusive_ptr<GpuGraphCache> GpuGraphCache::Create(
    const int64_t num_edges, const int64_t threshold,
    torch::ScalarType indptr_dtype, std::vector<torch::ScalarType> dtypes,
    bool has_original_edge_ids) {
  return c10::make_intrusive<GpuGraphCache>(
      num_edges, threshold, indptr_dtype, dtypes, has_original_edge_ids);
}

GpuGraphCache::GpuGraphCache(
    const int64_t num_edges, const int64_t threshold,
    torch::ScalarType indptr_dtype, std::vector<torch::ScalarType> dtypes,
    bool has_original_edge_ids) {
  const int64_t initial_node_capacity = 1024;
  AT_DISPATCH_INDEX_TYPES(
      dtypes.at(0), "GpuGraphCache::GpuGraphCache", ([&] {
        auto map_temp = map_t<index_t>{
            initial_node_capacity,
            kDoubleLoadFactor,
            cuco::empty_key{static_cast<index_t>(-1)},
            cuco::empty_value{std::numeric_limits<index_t>::lowest()},
            {},
            probing_t<index_t>{},
            {},
            {},
            allocator_t<index_t>{},
            ::cuda::stream_ref{cuda::GetCurrentStream()}};
        map_ = new map_t<index_t>{std::move(map_temp)};
      }));
  C10_CUDA_KERNEL_LAUNCH_CHECK();  // Check the map constructor's success.
  const auto options = torch::TensorOptions().device(c10::DeviceType::CUDA);
  TORCH_CHECK(threshold > 0, "Threshold should be a position integer.");
  threshold_ = threshold;
  device_id_ = cuda::GetCurrentStream().device_index();
  map_size_ = 0;
  num_nodes_ = 0;
  num_edges_ = 0;
  indptr_ =
      torch::zeros(initial_node_capacity + 1, options.dtype(indptr_dtype));
  if (!has_original_edge_ids) {
    offset_ = torch::empty(indptr_.size(0) - 1, indptr_.options());
  }
  for (auto dtype : dtypes) {
    cached_edge_tensors_.push_back(
        torch::empty(num_edges, options.dtype(dtype)));
  }
}

GpuGraphCache::~GpuGraphCache() {
  AT_DISPATCH_INDEX_TYPES(
      cached_edge_tensors_.at(0).scalar_type(), "GpuGraphCache::GpuGraphCache",
      ([&] { delete reinterpret_cast<map_t<index_t>*>(map_); }));
}

std::tuple<torch::Tensor, torch::Tensor, int64_t, int64_t> GpuGraphCache::Query(
    torch::Tensor seeds) {
  TORCH_CHECK(seeds.device().is_cuda(), "Seeds should be on a CUDA device.");
  TORCH_CHECK(
      seeds.device().index() == device_id_,
      "Seeds should be on the correct CUDA device.");
  TORCH_CHECK(seeds.sizes().size() == 1, "Keys should be a 1D tensor.");
  std::lock_guard lock(mtx_);
  auto allocator = cuda::GetAllocator();
  auto index_dtype = cached_edge_tensors_.at(0).scalar_type();
  const dim3 block(kIntBlockSize);
  const dim3 grid((seeds.size(0) + kIntBlockSize - 1) / kIntBlockSize);
  return AT_DISPATCH_INDEX_TYPES(
      index_dtype, "GpuGraphCache::Query", ([&] {
        auto map = reinterpret_cast<map_t<index_t>*>(map_);
        while ((
            map_size_ + seeds.size(0) >= map->capacity() * kDoubleLoadFactor)) {
          map->rehash_async(
              map->capacity() * kIntGrowthFactor,
              ::cuda::stream_ref{cuda::GetCurrentStream()});
        }
        auto positions = torch::empty_like(seeds);
        CUDA_KERNEL_CALL(
            _QueryAndIncrement, grid, block, 0,
            static_cast<int64_t>(seeds.size(0)), seeds.data_ptr<index_t>(),
            positions.data_ptr<index_t>(), map->ref(cuco::insert_and_find));
        auto num_threshold_new_hit =
            allocator.AllocateStorage<thrust::tuple<int64_t, int64_t, int64_t>>(
                1);
        // Since threshold_ is a class member, we want the lambda functions
        // below to only capture this particular variable by reassigning it to a
        // local variable.
        const auto threshold = -threshold_;
        auto is_threshold_new_hit = thrust::make_transform_iterator(
            positions.data_ptr<index_t>(), [=] __host__ __device__(index_t x) {
              int64_t is_threshold = x == threshold;
              int64_t is_new = x == -1;
              int64_t is_hit = x >= 0;
              return thrust::make_tuple(is_threshold, is_new, is_hit);
            });
        CUB_CALL(
            DeviceReduce::Reduce, is_threshold_new_hit,
            num_threshold_new_hit.get(), positions.size(0),
            [] __host__ __device__(
                const thrust::tuple<int64_t, int64_t, int64_t>& a,
                const thrust::tuple<int64_t, int64_t, int64_t>& b) {
              return thrust::make_tuple(
                  thrust::get<0>(a) + thrust::get<0>(b),
                  thrust::get<1>(a) + thrust::get<1>(b),
                  thrust::get<2>(a) + thrust::get<2>(b));
            },
            thrust::tuple<int64_t, int64_t, int64_t>{});
        CopyScalar num_threshold_new_hit_cpu{num_threshold_new_hit.get()};
        thrust::counting_iterator<index_t> iota{0};
        auto position_and_index =
            thrust::make_zip_iterator(positions.data_ptr<index_t>(), iota);
        auto output_positions = torch::empty_like(seeds);
        auto output_indices = torch::empty_like(seeds);
        auto output_position_and_index = thrust::make_zip_iterator(
            output_positions.data_ptr<index_t>(),
            output_indices.data_ptr<index_t>());
        CUB_CALL(
            DevicePartition::If, position_and_index, output_position_and_index,
            cub::DiscardOutputIterator{}, seeds.size(0),
            [] __device__(thrust::tuple<index_t, index_t> & x) {
              return thrust::get<0>(x) >= 0;
            });
        const auto [num_threshold, num_new, num_hit] =
            static_cast<thrust::tuple<int64_t, int64_t, int64_t>>(
                num_threshold_new_hit_cpu);
        map_size_ += num_new;

        return std::make_tuple(
            output_indices, output_positions, num_hit, num_threshold);
      }));
}

c10::intrusive_ptr<
    Future<std::tuple<torch::Tensor, torch::Tensor, int64_t, int64_t>>>
GpuGraphCache::QueryAsync(torch::Tensor seeds) {
  return async([=] { return Query(seeds); }, true);
}

std::tuple<torch::Tensor, std::vector<torch::Tensor>> GpuGraphCache::Replace(
    torch::Tensor seeds, torch::Tensor indices, torch::Tensor positions,
    int64_t num_hit, int64_t num_threshold, torch::Tensor indptr,
    std::vector<torch::Tensor> edge_tensors) {
  const auto with_edge_ids = offset_.has_value();
  // The last element of edge_tensors has the edge ids.
  const auto num_tensors = edge_tensors.size() - with_edge_ids;
  TORCH_CHECK(
      num_tensors == cached_edge_tensors_.size(),
      "Same number of tensors need to be passed!");
  const auto num_nodes = seeds.size(0);
  TORCH_CHECK(
      indptr.size(0) == num_nodes - num_hit + 1,
      "(indptr.size(0) == seeds.size(0) - num_hit + 1) failed.");
  std::lock_guard lock(mtx_);
  const int64_t num_buffers = num_nodes * num_tensors;
  auto allocator = cuda::GetAllocator();
  auto index_dtype = cached_edge_tensors_.at(0).scalar_type();
  return AT_DISPATCH_INDEX_TYPES(
      index_dtype, "GpuGraphCache::Replace", ([&] {
        using indices_t = index_t;
        return AT_DISPATCH_INDEX_TYPES(
            indptr_.scalar_type(), "GpuGraphCache::Replace::copy_prep", ([&] {
              using indptr_t = index_t;
              static_assert(
                  sizeof(int64_t) == sizeof(void*),
                  "Pointers have to be 64-bit.");
              static_assert(
                  sizeof(std::byte) == 1, "Byte needs to have a size of 1.");
              auto cache_missing_dtype = torch::empty(
                  // Below, we use this storage to store a tuple of 4 elements,
                  // since each element is 64-bit, we need 4x int64 storage.
                  4 * num_tensors, c10::TensorOptions()
                                       .dtype(torch::kInt64)
                                       .pinned_memory(true));
              auto cache_missing_dtype_ptr =
                  reinterpret_cast<::cuda::std::tuple<
                      std::byte*, std::byte*, int64_t, int64_t>*>(
                      cache_missing_dtype.data_ptr());
              int64_t total_size = 0;
              for (size_t i = 0; i < num_tensors; i++) {
                TORCH_CHECK(
                    cached_edge_tensors_[i].scalar_type() ==
                        edge_tensors[i].scalar_type(),
                    "The dtypes of edge tensors must match.");
                if (i > 0) {
                  TORCH_CHECK(
                      edge_tensors[i - 1].size(0) == edge_tensors[i].size(0),
                      "The missing edge tensors should have identical size.");
                }
                const int64_t element_size = edge_tensors[i].element_size();
                cache_missing_dtype_ptr[i] = {
                    reinterpret_cast<std::byte*>(
                        cached_edge_tensors_[i].data_ptr()),
                    reinterpret_cast<std::byte*>(edge_tensors[i].data_ptr()),
                    element_size, total_size};
                total_size += element_size;
              }
              auto cache_missing_dtype_dev = allocator.AllocateStorage<
                  ::cuda::std::tuple<std::byte*, std::byte*, int64_t, int64_t>>(
                  num_tensors);
              THRUST_CALL(
                  copy_n, cache_missing_dtype_ptr, num_tensors,
                  cache_missing_dtype_dev.get());

              auto input = allocator.AllocateStorage<std::byte*>(num_buffers);
              auto input_size =
                  allocator.AllocateStorage<size_t>(num_buffers + 1);
              torch::optional<torch::Tensor> edge_id_offsets;
              if (with_edge_ids) {
                edge_id_offsets = torch::empty(
                    num_nodes,
                    seeds.options().dtype(offset_.value().scalar_type()));
              }
              const auto cache_missing_dtype_dev_ptr =
                  cache_missing_dtype_dev.get();
              const auto indices_ptr = indices.data_ptr<indices_t>();
              const auto positions_ptr = positions.data_ptr<indices_t>();
              const auto input_ptr = input.get();
              const auto input_size_ptr = input_size.get();
              const auto edge_id_offsets_ptr =
                  edge_id_offsets ? edge_id_offsets->data_ptr<indptr_t>()
                                  : nullptr;
              const auto cache_indptr = indptr_.data_ptr<indptr_t>();
              const auto missing_indptr = indptr.data_ptr<indptr_t>();
              const auto cache_offset =
                  offset_ ? offset_->data_ptr<indptr_t>() : nullptr;
              const auto missing_edge_ids =
                  edge_id_offsets ? edge_tensors.back().data_ptr<indptr_t>()
                                  : nullptr;
              CUB_CALL(DeviceFor::Bulk, num_buffers, [=] __device__(int64_t i) {
                const auto tensor_idx = i / num_nodes;
                const auto idx = i % num_nodes;
                const auto pos = positions_ptr[idx];
                const auto original_idx = indices_ptr[idx];
                const auto [cache_ptr, missing_ptr, size, cum_size] =
                    cache_missing_dtype_dev_ptr[tensor_idx];
                const auto is_cached = pos >= 0;
                const auto offset = is_cached ? cache_indptr[pos]
                                              : missing_indptr[idx - num_hit];
                const auto offset_end = is_cached
                                            ? cache_indptr[pos + 1]
                                            : missing_indptr[idx - num_hit + 1];
                const auto out_idx = tensor_idx * num_nodes + original_idx;

                input_ptr[out_idx] =
                    (is_cached ? cache_ptr : missing_ptr) + offset * size;
                input_size_ptr[out_idx] = size * (offset_end - offset);
                if (edge_id_offsets_ptr && i < num_nodes) {
                  const auto edge_id =
                      is_cached ? cache_offset[pos] : missing_edge_ids[offset];
                  edge_id_offsets_ptr[out_idx] = edge_id;
                }
              });
              auto output_indptr = torch::empty(
                  num_nodes + 1, seeds.options().dtype(indptr_.scalar_type()));
              auto output_indptr_ptr = output_indptr.data_ptr<indptr_t>();
              const auto element_size =
                  ::cuda::std::get<2>(cache_missing_dtype_ptr[0]);
              auto input_indegree = thrust::make_transform_iterator(
                  input_size_ptr, [=] __host__ __device__(size_t x) {
                    return x / element_size;
                  });
              CUB_CALL(
                  DeviceScan::ExclusiveSum, input_indegree, output_indptr_ptr,
                  num_nodes + 1);
              CopyScalar output_size{output_indptr_ptr + num_nodes};

              if (num_threshold > 0) {
                // Insert the vertices whose access count equal threshold.
                auto missing_positions = positions.slice(0, num_hit);
                auto missing_indices = indices.slice(0, num_hit);

                thrust::counting_iterator<indices_t> iota{0};
                auto threshold = -threshold_;
                auto is_threshold = thrust::make_transform_iterator(
                    missing_positions.data_ptr<indices_t>(),
                    [=] __host__ __device__(indices_t x) {
                      return x == threshold;
                    });
                auto output_indices =
                    torch::empty(num_threshold, seeds.options());
                CUB_CALL(
                    DeviceSelect::Flagged, iota, is_threshold,
                    output_indices.data_ptr<indices_t>(),
                    cub::DiscardOutputIterator{}, missing_positions.size(0));
                auto [in_degree, sliced_indptr] =
                    ops::SliceCSCIndptr(indptr, output_indices);
                while (num_nodes_ + num_threshold >= indptr_.size(0)) {
                  auto new_indptr = torch::empty(
                      indptr_.size(0) * kIntGrowthFactor, indptr_.options());
                  new_indptr.slice(0, 0, indptr_.size(0)) = indptr_;
                  indptr_ = new_indptr;
                  if (offset_) {
                    auto new_offset =
                        torch::empty(indptr_.size(0) - 1, offset_->options());
                    new_offset.slice(0, 0, offset_->size(0)) = *offset_;
                    offset_ = new_offset;
                  }
                }
                torch::Tensor sindptr;
                bool enough_space;
                torch::optional<int64_t> cached_output_size;
                for (size_t i = 0; i < num_tensors; i++) {
                  torch::Tensor sindices;
                  std::tie(sindptr, sindices) = ops::IndexSelectCSCImpl(
                      in_degree, sliced_indptr, edge_tensors[i], output_indices,
                      indptr.size(0) - 2, cached_output_size);
                  cached_output_size = sindices.size(0);
                  enough_space = num_edges_ + *cached_output_size <=
                                 cached_edge_tensors_[i].size(0);
                  if (enough_space) {
                    cached_edge_tensors_[i].slice(
                        0, num_edges_, num_edges_ + *cached_output_size) =
                        sindices;
                  } else
                    break;
                }
                if (enough_space) {
                  auto num_edges = num_edges_;
                  if (offset_) {
                    auto transform_input_it = thrust::make_zip_iterator(
                        sindptr.data_ptr<indptr_t>() + 1,
                        sliced_indptr.data_ptr<indptr_t>());
                    auto transform_output_it = thrust::make_zip_iterator(
                        indptr_.data_ptr<indptr_t>() + num_nodes_ + 1,
                        offset_->data_ptr<indptr_t>() + num_nodes_);
                    THRUST_CALL(
                        transform, transform_input_it,
                        transform_input_it + sindptr.size(0) - 1,
                        transform_output_it,
                        [=] __host__ __device__(
                            const thrust::tuple<indptr_t, indptr_t>& x) {
                          return thrust::make_tuple(
                              thrust::get<0>(x) + num_edges,
                              missing_edge_ids[thrust::get<1>(x)]);
                        });
                  } else {
                    THRUST_CALL(
                        transform, sindptr.data_ptr<indptr_t>() + 1,
                        sindptr.data_ptr<indptr_t>() + sindptr.size(0),
                        indptr_.data_ptr<indptr_t>() + num_nodes_ + 1,
                        [=] __host__ __device__(const indptr_t& x) {
                          return x + num_edges;
                        });
                  }
                  auto map = reinterpret_cast<map_t<indices_t>*>(map_);
                  const dim3 block(kIntBlockSize);
                  const dim3 grid(
                      (num_threshold + kIntBlockSize - 1) / kIntBlockSize);
                  CUDA_KERNEL_CALL(
                      _Insert, grid, block, 0, output_indices.size(0),
                      static_cast<indices_t>(num_nodes_),
                      seeds.data_ptr<indices_t>(),
                      missing_indices.data_ptr<indices_t>(),
                      output_indices.data_ptr<indices_t>(),
                      map->ref(cuco::find));
                  num_edges_ += *cached_output_size;
                  num_nodes_ += num_threshold;
                }
              }

              constexpr int alignment = 128;
              const auto output_allocation_count =
                  (static_cast<indptr_t>(output_size) + alignment - 1) /
                  alignment * alignment;
              auto output_allocation = torch::empty(
                  output_allocation_count * total_size,
                  seeds.options().dtype(torch::kInt8));
              const auto output_allocation_ptr =
                  output_allocation.data_ptr<int8_t>();

              std::vector<torch::Tensor> output_edge_tensors;
              for (size_t i = 0; i < num_tensors; i++) {
                const auto cum_size =
                    ::cuda::std::get<3>(cache_missing_dtype_ptr[i]);
                output_edge_tensors.push_back(
                    output_allocation
                        .slice(0, cum_size * output_allocation_count)
                        .view(edge_tensors[i].scalar_type())
                        .slice(0, 0, static_cast<indptr_t>(output_size)));
              }
              if (edge_id_offsets) {
                // Append the edge ids as the last element of the output.
                output_edge_tensors.push_back(ops::IndptrEdgeIdsImpl(
                    output_indptr, output_indptr.scalar_type(),
                    *edge_id_offsets,
                    static_cast<int64_t>(static_cast<indptr_t>(output_size))));
              }

              {
                thrust::counting_iterator<int64_t> iota{0};
                auto output_buffer_it = thrust::make_transform_iterator(
                    iota, [=] __host__ __device__(int64_t i) {
                      const auto tensor_idx = i / num_nodes;
                      const auto idx = i % num_nodes;
                      const auto offset = output_indptr_ptr[idx];
                      const auto [_0, _1, size, cum_size] =
                          cache_missing_dtype_dev_ptr[tensor_idx];
                      return output_allocation_ptr +
                             cum_size * output_allocation_count + offset * size;
                    });
                constexpr int64_t max_copy_at_once =
                    std::numeric_limits<int32_t>::max();
                for (int64_t i = 0; i < num_buffers; i += max_copy_at_once) {
                  CUB_CALL(
                      DeviceMemcpy::Batched, input.get() + i,
                      output_buffer_it + i, input_size_ptr + i,
                      std::min(num_buffers - i, max_copy_at_once));
                }
              }

              return std::make_tuple(output_indptr, output_edge_tensors);
            }));
      }));
}

c10::intrusive_ptr<
    Future<std::tuple<torch::Tensor, std::vector<torch::Tensor>>>>
GpuGraphCache::ReplaceAsync(
    torch::Tensor seeds, torch::Tensor indices, torch::Tensor positions,
    int64_t num_hit, int64_t num_threshold, torch::Tensor indptr,
    std::vector<torch::Tensor> edge_tensors) {
  return async(
      [=] {
        return Replace(
            seeds, indices, positions, num_hit, num_threshold, indptr,
            edge_tensors);
      },
      true);
}

}  // namespace cuda
}  // namespace graphbolt
