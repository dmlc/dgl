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
 * @brief Cooperative Minibatching (arXiv:2310.12403) utility function in CUDA.
 */
#ifndef GRAPHBOLT_CUDA_COOPERATIVE_MINIBATCHING_UTILS_H_
#define GRAPHBOLT_CUDA_COOPERATIVE_MINIBATCHING_UTILS_H_

#include <curand_kernel.h>

namespace graphbolt {
namespace cuda {

/**
 * @brief Given a vertex id, the rank of current GPU and the world size, returns
 * the rank that this id belongs in a deterministic manner.
 *
 * @param id         The node id that will mapped to a rank in [0, world_size).
 * @param rank       The rank of the current GPU.
 * @param world_size The world size, the total number of cooperating GPUs.
 *
 * @return The rank of the GPU the given id is mapped to.
 */
template <typename index_t>
__device__ inline auto PartitionAssignment(
    index_t id, uint32_t rank, uint32_t world_size) {
  // Consider using a faster implementation in the future.
  constexpr uint64_t kCurandSeed = 999961;  // Any random number.
  curandStatePhilox4_32_10_t rng;
  curand_init(kCurandSeed, 0, id, &rng);
  return (curand(&rng) - rank) % world_size;
}

}  // namespace cuda
}  // namespace graphbolt

#endif  // GRAPHBOLT_CUDA_COOPERATIVE_MINIBATCHING_UTILS_H_
