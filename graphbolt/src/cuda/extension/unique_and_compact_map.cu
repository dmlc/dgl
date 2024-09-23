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
 * @file cuda/unique_and_compact_map.cu
 * @brief Unique and compact operator implementation on CUDA using hash table.
 */
#include <graphbolt/cuda_ops.h>
#include <thrust/iterator/reverse_iterator.h>
#include <thrust/iterator/tabulate_output_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>

#include <cub/cub.cuh>
#include <cuco/static_map.cuh>
#include <cuda/functional>
#include <cuda/std/atomic>
#include <cuda/std/utility>
#include <cuda/stream_ref>
#include <limits>
#include <numeric>

#include "../common.h"
#include "../cooperative_minibatching_utils.cuh"
#include "../cooperative_minibatching_utils.h"
#include "../utils.h"
#include "./unique_and_compact.h"

namespace graphbolt {
namespace ops {

// Support graphs with up to 2^kNodeIdBits nodes.
constexpr int kNodeIdBits = 40;

template <typename index_t, typename map_t>
__global__ void _InsertAndSetMinBatched(
    const int64_t num_edges, const int32_t* const indexes, index_t** pointers,
    const int64_t* const offsets, map_t map) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;

  while (i < num_edges) {
    const auto tensor_index = indexes[i];
    const auto tensor_offset = i - offsets[tensor_index];
    const int64_t node_id = pointers[tensor_index][tensor_offset];
    const int64_t batch_index = tensor_index / 2;
    const int64_t key = node_id | (batch_index << kNodeIdBits);

    auto [slot, is_new_key] = map.insert_and_find(cuco::pair{key, i});

    if (!is_new_key) {
      auto ref = ::cuda::atomic_ref<int64_t, ::cuda::thread_scope_device>{
          slot->second};
      ref.fetch_min(i, ::cuda::memory_order_relaxed);
    }

    i += stride;
  }
}

template <typename index_t, typename map_t>
__global__ void _MapIdsBatched(
    const int num_batches, const int64_t num_edges,
    const int32_t* const indexes, index_t** pointers,
    const int64_t* const offsets, const int64_t* const unique_ids_offsets,
    const index_t* const index, map_t map, index_t* mapped_ids) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;

  while (i < num_edges) {
    const auto tensor_index = indexes[i];
    int64_t batch_index;

    if (tensor_index >= 2 * num_batches) {
      batch_index = tensor_index - 2 * num_batches;
    } else if (tensor_index & 1) {
      batch_index = tensor_index / 2;
    } else {
      batch_index = -1;
    }

    // Only map src or dst ids.
    if (batch_index >= 0) {
      const auto tensor_offset = i - offsets[tensor_index];
      const int64_t node_id = pointers[tensor_index][tensor_offset];
      const int64_t key = node_id | (batch_index << kNodeIdBits);

      auto slot = map.find(key);
      auto new_id = slot->second;
      if (index) {
        new_id = index[new_id];
      } else {
        new_id -= unique_ids_offsets[batch_index];
      }
      mapped_ids[i] = new_id;
    }

    i += stride;
  }
}

std::vector<
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>>
UniqueAndCompactBatchedHashMapBased(
    const std::vector<torch::Tensor>& src_ids,
    const std::vector<torch::Tensor>& dst_ids,
    const std::vector<torch::Tensor>& unique_dst_ids, const int64_t rank,
    const int64_t world_size) {
  TORCH_CHECK(
      rank < world_size, "rank needs to be smaller than the world_size.");
  TORCH_CHECK(world_size <= std::numeric_limits<uint32_t>::max());
  auto allocator = cuda::GetAllocator();
  auto stream = cuda::GetCurrentStream();
  auto scalar_type = src_ids.at(0).scalar_type();
  constexpr int BLOCK_SIZE = 512;
  const auto num_batches = src_ids.size();
  static_assert(
      sizeof(std::ptrdiff_t) == sizeof(int64_t),
      "Need to be compiled on a 64-bit system.");
  constexpr int batch_id_bits = sizeof(int64_t) * 8 - 1 - kNodeIdBits;
  TORCH_CHECK(
      num_batches <= (1 << batch_id_bits),
      "UniqueAndCompactBatched supports a batch size of up to ",
      1 << batch_id_bits);
  return AT_DISPATCH_INDEX_TYPES(
      scalar_type, "unique_and_compact", ([&] {
        // For 2 batches of inputs, stores the input tensor pointers in the
        // unique_dst, src, unique_dst, src, dst, dst order. Since there are
        // 3 * num_batches input tensors, we need the first 3 * num_batches to
        // store the input tensor pointers. Then, we store offsets in the rest
        // of the 3 * num_batches + 1 space as if they were stored contiguously.
        auto pointers_and_offsets = torch::empty(
            6 * num_batches + 1,
            c10::TensorOptions().dtype(torch::kInt64).pinned_memory(true));
        // Points to the input tensor pointers.
        auto pointers_ptr =
            reinterpret_cast<index_t**>(pointers_and_offsets.data_ptr());
        // Points to the input tensor storage logical offsets.
        auto offsets_ptr =
            pointers_and_offsets.data_ptr<int64_t>() + 3 * num_batches;
        for (std::size_t i = 0; i < num_batches; i++) {
          pointers_ptr[2 * i] = unique_dst_ids.at(i).data_ptr<index_t>();
          offsets_ptr[2 * i] = unique_dst_ids[i].size(0);
          pointers_ptr[2 * i + 1] = src_ids.at(i).data_ptr<index_t>();
          offsets_ptr[2 * i + 1] = src_ids[i].size(0);
          pointers_ptr[2 * num_batches + i] = dst_ids.at(i).data_ptr<index_t>();
          offsets_ptr[2 * num_batches + i] = dst_ids[i].size(0);
        }
        // Finish computing the offsets by taking a cumulative sum.
        std::exclusive_scan(
            offsets_ptr, offsets_ptr + 3 * num_batches + 1, offsets_ptr, 0ll);
        // Device version of the tensors defined above. We store the information
        // initially on the CPU, which are later copied to the device.
        auto pointers_and_offsets_dev = torch::empty(
            pointers_and_offsets.size(0),
            src_ids[0].options().dtype(pointers_and_offsets.scalar_type()));
        auto offsets_dev = pointers_and_offsets_dev.slice(0, 3 * num_batches);
        auto pointers_dev_ptr =
            reinterpret_cast<index_t**>(pointers_and_offsets_dev.data_ptr());
        auto offsets_dev_ptr = offsets_dev.data_ptr<int64_t>();
        CUDA_CALL(cudaMemcpyAsync(
            pointers_dev_ptr, pointers_ptr,
            sizeof(int64_t) * pointers_and_offsets.size(0),
            cudaMemcpyHostToDevice, stream));
        auto indexes = ExpandIndptrImpl(
            offsets_dev, torch::kInt32, torch::nullopt,
            offsets_ptr[3 * num_batches]);
        cuco::static_map map{
            offsets_ptr[2 * num_batches],
            0.5,  // load_factor
            cuco::empty_key{static_cast<int64_t>(-1)},
            cuco::empty_value{static_cast<int64_t>(-1)},
            {},
            cuco::linear_probing<1, cuco::default_hash_function<int64_t>>{},
            {},
            {},
            cuda::CUDAWorkspaceAllocator<cuco::pair<int64_t, int64_t>>{},
            ::cuda::stream_ref{stream},
        };
        C10_CUDA_KERNEL_LAUNCH_CHECK();  // Check the map constructor's success.
        const dim3 block(BLOCK_SIZE);
        const dim3 grid(
            (offsets_ptr[2 * num_batches] + BLOCK_SIZE - 1) / BLOCK_SIZE);
        CUDA_KERNEL_CALL(
            _InsertAndSetMinBatched, grid, block, 0,
            offsets_ptr[2 * num_batches], indexes.data_ptr<int32_t>(),
            pointers_dev_ptr, offsets_dev_ptr, map.ref(cuco::insert_and_find));
        cub::ArgIndexInputIterator index_it(indexes.data_ptr<int32_t>());
        auto input_it = thrust::make_transform_iterator(
            index_it,
            ::cuda::proclaim_return_type<
                ::cuda::std::tuple<int64_t*, index_t, int32_t, bool>>(
                [=, map = map.ref(cuco::find)] __device__(auto it)
                    -> ::cuda::std::tuple<int64_t*, index_t, int32_t, bool> {
                  const auto i = it.key;
                  const auto tensor_index = it.value;
                  const auto tensor_offset = i - offsets_dev_ptr[tensor_index];
                  const int64_t node_id =
                      pointers_dev_ptr[tensor_index][tensor_offset];
                  const auto batch_index = tensor_index / 2;
                  const int64_t key =
                      node_id |
                      (static_cast<int64_t>(batch_index) << kNodeIdBits);
                  const auto batch_offset = offsets_dev_ptr[batch_index * 2];

                  auto slot = map.find(key);
                  const auto valid = slot->second == i;

                  return {&slot->second, node_id, batch_index, valid};
                }));
        torch::optional<torch::Tensor> part_ids;
        if (world_size > 1) {
          part_ids = torch::empty(
              offsets_ptr[2 * num_batches],
              src_ids[0].options().dtype(cuda::kPartDType));
        }
        auto unique_ids =
            torch::empty(offsets_ptr[2 * num_batches], src_ids[0].options());
        auto unique_ids_offsets_dev = torch::full(
            num_batches + 1, std::numeric_limits<int64_t>::max(),
            src_ids[0].options().dtype(torch::kInt64));
        auto unique_ids_offsets_dev_ptr =
            unique_ids_offsets_dev.data_ptr<int64_t>();
        auto output_it = thrust::make_tabulate_output_iterator(
            ::cuda::proclaim_return_type<void>(
                [=, unique_ids_ptr = unique_ids.data_ptr<index_t>(),
                 part_ids_ptr =
                     part_ids ? part_ids->data_ptr<cuda::part_t>() : nullptr,
                 rank = static_cast<uint32_t>(rank),
                 world_size = static_cast<uint32_t>(
                     world_size)] __device__(const int64_t i, const auto& t) {
                  *::cuda::std::get<0>(t) = i;
                  const auto node_id = ::cuda::std::get<1>(t);
                  unique_ids_ptr[i] = node_id;
                  if (part_ids_ptr) {
                    part_ids_ptr[i] =
                        cuda::rank_assignment(node_id, rank, world_size);
                  }
                  const auto batch_index = ::cuda::std::get<2>(t);
                  auto ref =
                      ::cuda::atomic_ref<int64_t, ::cuda::thread_scope_device>{
                          unique_ids_offsets_dev_ptr[batch_index]};
                  ref.fetch_min(i, ::cuda::memory_order_relaxed);
                }));
        CUB_CALL(
            DeviceSelect::If, input_it, output_it,
            unique_ids_offsets_dev_ptr + num_batches,
            offsets_ptr[2 * num_batches],
            ::cuda::proclaim_return_type<bool>([] __device__(const auto& t) {
              return ::cuda::std::get<3>(t);
            }));
        auto unique_ids_offsets = torch::empty(
            num_batches + 1,
            c10::TensorOptions().dtype(torch::kInt64).pinned_memory(true));
        {
          auto unique_ids_offsets_dev2 =
              torch::empty_like(unique_ids_offsets_dev);
          CUB_CALL(
              DeviceScan::InclusiveScan,
              thrust::make_reverse_iterator(
                  num_batches + 1 + unique_ids_offsets_dev_ptr),
              thrust::make_reverse_iterator(
                  num_batches + 1 +
                  thrust::make_transform_output_iterator(
                      thrust::make_zip_iterator(
                          unique_ids_offsets_dev2.data_ptr<int64_t>(),
                          unique_ids_offsets.data_ptr<int64_t>()),
                      ::cuda::proclaim_return_type<
                          thrust::tuple<int64_t, int64_t>>(
                          [=] __device__(const auto x) {
                            return thrust::make_tuple(x, x);
                          }))),
              cub::Min{}, num_batches + 1);
          unique_ids_offsets_dev = unique_ids_offsets_dev2;
          unique_ids_offsets_dev_ptr =
              unique_ids_offsets_dev.data_ptr<int64_t>();
        }
        at::cuda::CUDAEvent unique_ids_offsets_event;
        unique_ids_offsets_event.record();
        torch::optional<torch::Tensor> index;
        if (part_ids) {
          unique_ids_offsets_event.synchronize();
          const auto num_unique =
              unique_ids_offsets.data_ptr<int64_t>()[num_batches];
          unique_ids = unique_ids.slice(0, 0, num_unique);
          part_ids = part_ids->slice(0, 0, num_unique);
          std::tie(
              unique_ids, index, unique_ids_offsets, unique_ids_offsets_event) =
              cuda::RankSortImpl(
                  unique_ids, *part_ids, unique_ids_offsets_dev, world_size);
        }
        auto mapped_ids =
            torch::empty(offsets_ptr[3 * num_batches], unique_ids.options());
        CUDA_KERNEL_CALL(
            _MapIdsBatched, grid, block, 0, num_batches,
            offsets_ptr[3 * num_batches], indexes.data_ptr<int32_t>(),
            pointers_dev_ptr, offsets_dev_ptr, unique_ids_offsets_dev_ptr,
            index ? index->data_ptr<index_t>() : nullptr, map.ref(cuco::find),
            mapped_ids.data_ptr<index_t>());
        std::vector<std::tuple<
            torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>>
            results;
        unique_ids_offsets_event.synchronize();
        auto unique_ids_offsets_ptr = unique_ids_offsets.data_ptr<int64_t>();
        for (int64_t i = 0; i < num_batches; i++) {
          results.emplace_back(
              unique_ids.slice(
                  0, unique_ids_offsets_ptr[i * world_size],
                  unique_ids_offsets_ptr[(i + 1) * world_size]),
              mapped_ids.slice(
                  0, offsets_ptr[2 * i + 1], offsets_ptr[2 * i + 2]),
              mapped_ids.slice(
                  0, offsets_ptr[2 * num_batches + i],
                  offsets_ptr[2 * num_batches + i + 1]),
              unique_ids_offsets.slice(
                  0, i * world_size, (i + 1) * world_size + 1));
        }
        return results;
      }));
}

}  // namespace ops
}  // namespace graphbolt
