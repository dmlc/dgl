/**
 *   Copyright (c) 2024, mfbalin (Muhammed Fatih Balin)
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
 * @file cuda/cooperative_minibatching_utils.cu
 * @brief Cooperative Minibatching (arXiv:2310.12403) utility function
 * implementations in CUDA.
 */
#include <graphbolt/cuda_ops.h>
#include <thrust/scatter.h>
#include <thrust/transform.h>

#include <cub/cub.cuh>
#include <cuda/functional>

#include "../utils.h"
#include "./common.h"
#include "./cooperative_minibatching_utils.cuh"
#include "./cooperative_minibatching_utils.h"
#include "./utils.h"

namespace graphbolt {
namespace cuda {

torch::Tensor RankAssignment(
    torch::Tensor nodes, const int64_t rank, const int64_t world_size) {
  auto part_ids = torch::empty_like(nodes, nodes.options().dtype(kPartDType));
  auto part_ids_ptr = part_ids.data_ptr<part_t>();
  AT_DISPATCH_INDEX_TYPES(
      nodes.scalar_type(), "RankAssignment", ([&] {
        auto nodes_ptr = nodes.data_ptr<index_t>();
        THRUST_CALL(
            transform, nodes_ptr, nodes_ptr + nodes.numel(), part_ids_ptr,
            ::cuda::proclaim_return_type<part_t>(
                [rank = static_cast<uint32_t>(rank),
                 world_size = static_cast<uint32_t>(
                     world_size)] __device__(index_t id) -> part_t {
                  return rank_assignment(id, rank, world_size);
                }));
      }));
  return part_ids;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, at::cuda::CUDAEvent>
RankSortImpl(
    torch::Tensor nodes, torch::Tensor part_ids, torch::Tensor offsets_dev,
    const int64_t world_size) {
  const int num_bits = cuda::NumberOfBits(world_size);
  const auto num_batches = offsets_dev.numel() - 1;
  auto offsets_dev_ptr = offsets_dev.data_ptr<int64_t>();
  auto part_ids_sorted = torch::empty_like(part_ids);
  auto part_ids2 = part_ids.clone();
  auto part_ids2_sorted = torch::empty_like(part_ids2);
  auto nodes_sorted = torch::empty_like(nodes);
  auto index = torch::arange(nodes.numel(), nodes.options());
  auto index_sorted = torch::empty_like(index);
  return AT_DISPATCH_INDEX_TYPES(
      nodes.scalar_type(), "RankSortImpl", ([&] {
        CUB_CALL(
            DeviceSegmentedRadixSort::SortPairs,
            part_ids.data_ptr<cuda::part_t>(),
            part_ids_sorted.data_ptr<cuda::part_t>(), nodes.data_ptr<index_t>(),
            nodes_sorted.data_ptr<index_t>(), nodes.numel(), num_batches,
            offsets_dev_ptr, offsets_dev_ptr + 1, 0, num_bits);
        auto offsets = torch::empty(
            num_batches * world_size + 1, c10::TensorOptions()
                                              .dtype(offsets_dev.scalar_type())
                                              .pinned_memory(true));
        CUB_CALL(
            DeviceFor::Bulk, num_batches * world_size + 1,
            [=, part_ids = part_ids_sorted.data_ptr<cuda::part_t>(),
             offsets = offsets.data_ptr<int64_t>()] __device__(int64_t i) {
              const auto batch_id = i / world_size;
              const auto rank = i % world_size;
              const auto offset_begin = offsets_dev_ptr[batch_id];
              const auto offset_end =
                  offsets_dev_ptr[::cuda::std::min(batch_id + 1, num_batches)];
              offsets[i] = cub::LowerBound(
                               part_ids + offset_begin,
                               offset_end - offset_begin, rank) +
                           offset_begin;
            });
        at::cuda::CUDAEvent offsets_event;
        offsets_event.record();
        CUB_CALL(
            DeviceSegmentedRadixSort::SortPairs,
            part_ids2.data_ptr<cuda::part_t>(),
            part_ids2_sorted.data_ptr<cuda::part_t>(),
            index.data_ptr<index_t>(), index_sorted.data_ptr<index_t>(),
            nodes.numel(), num_batches, offsets_dev_ptr, offsets_dev_ptr + 1, 0,
            num_bits);
        auto values = ops::IndptrEdgeIdsImpl(
            offsets_dev, nodes.scalar_type(), torch::nullopt, nodes.numel());
        THRUST_CALL(
            scatter, values.data_ptr<index_t>(),
            values.data_ptr<index_t>() + values.numel(),
            index_sorted.data_ptr<index_t>(), index.data_ptr<index_t>());
        return std::make_tuple(
            nodes_sorted, index, offsets, std::move(offsets_event));
      }));
}

std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>> RankSort(
    const std::vector<torch::Tensor>& nodes_list, const int64_t rank,
    const int64_t world_size) {
  const auto num_batches = nodes_list.size();
  auto nodes = torch::cat(nodes_list, 0);
  auto offsets = torch::empty(
      num_batches + 1,
      c10::TensorOptions().dtype(torch::kInt64).pinned_memory(true));
  auto offsets_ptr = offsets.data_ptr<int64_t>();
  offsets_ptr[0] = 0;
  for (int64_t i = 0; i < num_batches; i++) {
    offsets_ptr[i + 1] = offsets_ptr[i] + nodes_list[i].numel();
  }
  auto part_ids = RankAssignment(nodes, rank, world_size);
  auto offsets_dev =
      torch::empty_like(offsets, nodes.options().dtype(offsets.scalar_type()));
  CUDA_CALL(cudaMemcpyAsync(
      offsets_dev.data_ptr<int64_t>(), offsets_ptr,
      sizeof(int64_t) * offsets.numel(), cudaMemcpyHostToDevice,
      cuda::GetCurrentStream()));
  auto [nodes_sorted, index_sorted, rank_offsets, rank_offsets_event] =
      RankSortImpl(nodes, part_ids, offsets_dev, world_size);
  std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>> results;
  rank_offsets_event.synchronize();
  for (int64_t i = 0; i < num_batches; i++) {
    results.emplace_back(
        nodes_sorted.slice(0, offsets_ptr[i], offsets_ptr[i + 1]),
        index_sorted.slice(0, offsets_ptr[i], offsets_ptr[i + 1]),
        rank_offsets.slice(0, i * world_size, (i + 1) * world_size + 1));
  }
  return results;
}

c10::intrusive_ptr<Future<
    std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>>>>
RankSortAsync(
    const std::vector<torch::Tensor>& nodes_list, const int64_t rank,
    const int64_t world_size) {
  return async(
      [=] { return RankSort(nodes_list, rank, world_size); },
      utils::is_on_gpu(nodes_list.at(0)));
}

}  // namespace cuda
}  // namespace graphbolt
