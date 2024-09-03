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
#include <curand_kernel.h>
#include <thrust/transform.h>
#include <torch/script.h>

#include <cuda/functional>

#include "./common.h"
#include "./cooperative_minibatching_utils.h"

namespace graphbolt {
namespace cuda {

torch::Tensor RankAssignment(
    torch::Tensor nodes, const int64_t rank, const int64_t world_size) {
  auto part_ids = torch::empty_like(nodes, nodes.options().dtype(kPartDType));
  auto part_ids_ptr = part_ids.data_ptr<part_t>();
  AT_DISPATCH_INDEX_TYPES(
      nodes.scalar_type(), "unique_and_compact", ([&] {
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

}  // namespace cuda
}  // namespace graphbolt
