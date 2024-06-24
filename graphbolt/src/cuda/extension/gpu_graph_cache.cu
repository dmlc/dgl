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

#include <cub/cub.cuh>
#include <cuco/static_map.cuh>
#include <cuda/std/atomic>
#include <numeric>

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

template <typename index_t, typename map_t>
__global__ void _QueryAndIncrement(
    const int64_t num_nodes, const index_t threshold, const index_t* seeds,
    index_t* positions, map_t map) {
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
    torch::ScalarType indptr_dtype, std::vector<torch::ScalarType> dtypes) {
  return c10::make_intrusive<GpuGraphCache>(
      num_edges, threshold, indptr_dtype, dtypes);
}

GpuGraphCache::GpuGraphCache(
    const int64_t num_edges, const int64_t threshold,
    torch::ScalarType indptr_dtype, std::vector<torch::ScalarType> dtypes) {
  const int64_t num_nodes = 1024;
  AT_DISPATCH_INDEX_TYPES(
      dtypes.at(0), "GpuGraphCache::GpuGraphCache", ([&] {
        auto map_temp = map_t<index_t>{
            num_nodes,
            kDoubleLoadFactor,
            cuco::empty_key{static_cast<index_t>(-1)},
            cuco::empty_value{std::numeric_limits<index_t>::lowest()},
            {},
            probing_t<index_t>{},
            {},
            {},
            allocator_t<index_t>{},
            cuco::cuda_stream_ref{cuda::GetCurrentStream()}};
        map_ = new map_t<index_t>{std::move(map_temp)};
      }));
  C10_CUDA_KERNEL_LAUNCH_CHECK();  // Check the map constructor's success.
  const auto options = torch::TensorOptions().device(c10::DeviceType::CUDA);
  threshold_ = threshold;
  device_id_ = cuda::GetCurrentStream().device_index();
  map_size_ = 0;
  num_nodes_ = 0;
  num_edges_ = 0;
  indptr_ = torch::zeros(num_nodes + 1, options.dtype(indptr_dtype));
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

struct AddTuple {
  template <typename T>
  __host__ __device__ auto operator()(const T& a, const T& b) const {
    return thrust::make_tuple(
        thrust::get<0>(a) + thrust::get<0>(b),
        thrust::get<1>(a) + thrust::get<1>(b));
  }
};

std::tuple<
    torch::Tensor, std::vector<torch::Tensor>, torch::Tensor, torch::Tensor,
    torch::Tensor, int64_t>
GpuGraphCache::Query(torch::Tensor seeds) {
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
              cuco::cuda_stream_ref{cuda::GetCurrentStream()});
        }
        auto positions =
            torch::empty(seeds.size(0), seeds.options().dtype(index_dtype));
        CUDA_KERNEL_CALL(
            _QueryAndIncrement, grid, block, 0,
            static_cast<int64_t>(seeds.size(0)),
            static_cast<index_t>(threshold_), seeds.data_ptr<index_t>(),
            positions.data_ptr<index_t>(), map->ref(cuco::insert_and_find));
        auto num_cache_enter =
            allocator.AllocateStorage<thrust::tuple<int64_t, int64_t>>(1);
        // Since threshold_ is a class member, we want the lambda functions
        // below to only capture this particular variable by reassigning it to a
        // local variable.
        const auto threshold = -threshold_;
        auto is_threshold = thrust::make_transform_iterator(
            positions.data_ptr<index_t>(), [=] __host__ __device__(index_t x) {
              int64_t is_threshold = x == threshold;
              int64_t is_new = x == -1;
              return thrust::make_tuple(is_threshold, is_new);
            });
        CUB_CALL(
            DeviceReduce::Reduce, is_threshold, num_cache_enter.get(),
            positions.size(0), AddTuple{}, thrust::tuple<int64_t, int64_t>{});
        auto num_cache_enter_cpu = torch::empty(
            2, c10::TensorOptions().dtype(torch::kInt64).pinned_memory(true));
        auto num_cache_enter_cpu_ptr = num_cache_enter_cpu.data_ptr<int64_t>();
        THRUST_CALL(
            copy, num_cache_enter.get(), num_cache_enter.get() + 1,
            reinterpret_cast<thrust::tuple<int64_t, int64_t>*>(
                num_cache_enter_cpu_ptr));
        thrust::counting_iterator<index_t> iota{0};
        auto position_and_index =
            thrust::make_zip_iterator(positions.data_ptr<index_t>(), iota);
        auto output_positions =
            torch::empty(seeds.size(0), seeds.options().dtype(index_dtype));
        auto output_indices =
            torch::empty(seeds.size(0), seeds.options().dtype(index_dtype));
        auto output_position_and_index = thrust::make_zip_iterator(
            output_positions.data_ptr<index_t>(),
            output_indices.data_ptr<index_t>());
        auto num_cache_hit = allocator.AllocateStorage<index_t>(1);
        CUB_CALL(
            DevicePartition::If, position_and_index, output_position_and_index,
            num_cache_hit.get(), seeds.size(0),
            [] __device__(thrust::tuple<index_t, index_t> & x) {
              return thrust::get<0>(x) >= 0;
            });
        CopyScalar num_cache_hit_cpu{num_cache_hit.get()};
        auto selected_positions = output_positions.slice(
            0, 0, static_cast<index_t>(num_cache_hit_cpu));
        auto selected_indices =
            output_indices.slice(0, 0, static_cast<index_t>(num_cache_hit_cpu));
        auto missing_indices =
            output_indices.slice(0, static_cast<index_t>(num_cache_hit_cpu));
        auto missing_positions =
            output_positions.slice(0, static_cast<index_t>(num_cache_hit_cpu));
        torch::Tensor indptr;
        std::vector<torch::Tensor> edge_tensors;
        torch::optional<int64_t> output_size;
        auto [in_degree, sliced_indptr] =
            ops::SliceCSCIndptr(indptr_, selected_positions);
        for (auto& edge_tensor : cached_edge_tensors_) {
          auto [tindptr, indices] = ops::IndexSelectCSCImpl(
              in_degree, sliced_indptr, edge_tensor, selected_positions,
              indptr_.size(0) - 2, output_size);
          output_size = indices.size(0);
          edge_tensors.push_back(indices);
          indptr = tindptr;
        }
        map_size_ += num_cache_enter_cpu_ptr[1];
        return std::make_tuple(
            indptr, edge_tensors, selected_indices, missing_indices,
            missing_positions, num_cache_enter_cpu_ptr[0]);
      }));
}

int64_t GpuGraphCache::Replace(
    torch::Tensor seeds, torch::Tensor missing_indices,
    torch::Tensor missing_positions, int64_t num_entering, torch::Tensor indptr,
    std::vector<torch::Tensor> edge_tensors) {
  TORCH_CHECK(
      edge_tensors.size() == cached_edge_tensors_.size(),
      "Same number of tensors need to be passed!");
  auto allocator = cuda::GetAllocator();
  auto index_dtype = cached_edge_tensors_.at(0).scalar_type();
  return AT_DISPATCH_INDEX_TYPES(
      index_dtype, "GpuGraphCache::Replace", ([&] {
        thrust::counting_iterator<index_t> iota{0};
        auto threshold = -threshold_;
        auto is_threshold = thrust::make_transform_iterator(
            missing_positions.data_ptr<index_t>(),
            [=] __host__ __device__(index_t x) { return x == threshold; });
        auto output_indices =
            torch::empty(num_entering, seeds.options().dtype(index_dtype));
        auto num_cache_entering = allocator.AllocateStorage<index_t>(1);
        CUB_CALL(
            DeviceSelect::Flagged, iota, is_threshold,
            output_indices.data_ptr<index_t>(), num_cache_entering.get(),
            seeds.size(0));
        auto [in_degree, sliced_indptr] =
            ops::SliceCSCIndptr(indptr, output_indices);
        torch::optional<int64_t> output_size;
        while (num_nodes_ + num_entering >= indptr_.size(0)) {
          auto new_indptr = torch::empty(
              indptr_.size(0) * kIntGrowthFactor, indptr_.options());
          new_indptr.slice(0, 0, indptr_.size(0)) = indptr_;
          indptr_ = new_indptr;
        }
        torch::Tensor sindptr;
        bool enough_space;
        for (size_t i = 0; i < edge_tensors.size(); i++) {
          torch::Tensor sindices;
          std::tie(sindptr, sindices) = ops::IndexSelectCSCImpl(
              in_degree, sliced_indptr, edge_tensors[i], output_indices,
              indptr.size(0) - 2, output_size);
          output_size = sindices.size(0);
          enough_space =
              num_edges_ + *output_size <= cached_edge_tensors_.at(0).size(0);
          if (enough_space) {
            cached_edge_tensors_.at(i).slice(
                0, num_edges_, num_edges_ + *output_size) = sindices;
          } else
            break;
        }
        if (enough_space) {
          AT_DISPATCH_INDEX_TYPES(
              sindptr.scalar_type(), "GpuGraphCache::Replace", ([&] {
                auto num_edges = num_edges_;
                THRUST_CALL(
                    transform, sindptr.data_ptr<index_t>() + 1,
                    sindptr.data_ptr<index_t>() + sindptr.size(0),
                    indptr_.data_ptr<index_t>() + num_nodes_ + 1,
                    [=] __host__ __device__(index_t x) {
                      return x + num_edges;
                    });
              }));
          auto map = reinterpret_cast<map_t<index_t>*>(map_);
          const dim3 block(kIntBlockSize);
          const dim3 grid((num_entering + kIntBlockSize - 1) / kIntBlockSize);
          CUDA_KERNEL_CALL(
              _Insert, grid, block, 0, output_indices.size(0),
              static_cast<index_t>(num_nodes_), seeds.data_ptr<index_t>(),
              missing_indices.data_ptr<index_t>(),
              output_indices.data_ptr<index_t>(), map->ref(cuco::find));
          num_edges_ += *output_size;
          num_nodes_ += num_entering;
        }
        return enough_space ? *output_size : 0;
      }));
}

}  // namespace cuda
}  // namespace graphbolt
