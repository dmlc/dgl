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
 * @file cuda/cooperative_minibatching_utils.h
 * @brief Cooperative Minibatching (arXiv:2310.12403) utility function headers
 * in CUDA.
 */
#ifndef GRAPHBOLT_CUDA_COOPERATIVE_MINIBATCHING_UTILS_H_
#define GRAPHBOLT_CUDA_COOPERATIVE_MINIBATCHING_UTILS_H_

#include <ATen/cuda/CUDAEvent.h>
#include <graphbolt/async.h>
#include <torch/script.h>

namespace graphbolt {
namespace cuda {

/**
 * @brief Given node ids, the rank of current GPU and the world size, returns
 * the ranks that the given ids belong in a deterministic manner.
 *
 * @param nodes      Node id tensor to be mapped to a rank in [0, world_size).
 * @param rank       Rank of the current GPU.
 * @param world_size World size, the total number of cooperating GPUs.
 *
 * @return The rank tensor of the GPU the given id tensor is mapped to.
 */
torch::Tensor RankAssignment(
    torch::Tensor nodes, int64_t rank, int64_t world_size);

/**
 * @brief Given node ids, the ranks they belong, the offsets to separate
 * different node types and world size, returns node ids sorted w.r.t. the ranks
 * that the given ids belong along with their new positions.
 *
 * @param nodes        Node id tensor to be mapped to a rank in [0, world_size).
 * @param part_ids     Rank tensor the nodes belong to.
 * @param offsets_dev  Offsets to separate different node types.
 * @param world_size   World size, the total number of cooperating GPUs.
 *
 * @return (sorted_nodes, new_positions, rank_offsets, rank_offsets_event),
 * where the first one includes sorted nodes, the second contains new positions
 * of the given nodes, so that sorted_nodes[new_positions] == nodes, and the
 * third contains the offsets of the sorted_nodes indicating
 * sorted_nodes[rank_offsets[i]: rank_offsets[i + 1]] contains nodes that
 * belongs to the `i`th rank. Before accessing rank_offsets on the CPU,
 * `rank_offsets_event.synchronize()` is required.
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, at::cuda::CUDAEvent>
RankSortImpl(
    torch::Tensor nodes, torch::Tensor part_ids, torch::Tensor offsets_dev,
    int64_t world_size);

/**
 * @brief Given a vector of node ids, the rank of current GPU and the world
 * size, returns node ids sorted w.r.t. the ranks that the given ids belong
 * along with the original positions.
 *
 * @param nodes_list   Node id tensor to be mapped to a rank in [0, world_size).
 * @param rank         Rank of the current GPU.
 * @param world_size   World size, the total number of cooperating GPUs.
 *
 * @return vector of (sorted_nodes, new_positions, rank_offsets), where the
 * first one includes sorted nodes, the second contains new positions of the
 * given nodes, so that sorted_nodes[new_positions] == nodes, and the third
 * contains the offsets of the sorted_nodes indicating
 * sorted_nodes[rank_offsets[i]: rank_offsets[i + 1]] contains nodes that
 * belongs to the `i`th rank.
 */
std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>> RankSort(
    const std::vector<torch::Tensor>& nodes_list, int64_t rank,
    int64_t world_size);

c10::intrusive_ptr<Future<
    std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>>>>
RankSortAsync(
    const std::vector<torch::Tensor>& nodes_list, const int64_t rank,
    const int64_t world_size);

}  // namespace cuda
}  // namespace graphbolt

#endif  // GRAPHBOLT_CUDA_COOPERATIVE_MINIBATCHING_UTILS_H_
