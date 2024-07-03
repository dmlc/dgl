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
 * @file partitioned_cache_policy.h
 * @brief Partitioned cache policy implementation on the CPU.
 */
#ifndef GRAPHBOLT_PARTITIONED_CACHE_H_
#define GRAPHBOLT_PARTITIONED_CACHE_H_

#include <torch/custom_class.h>
#include <torch/torch.h>

#include <pcg_random.hpp>
#include <random>
#include <vector>

#include "./cache_policy.h"

namespace graphbolt {
namespace storage {

template <typename BaseCachePolicy>
struct PartitionedCachePolicy : public torch::CustomClassHolder {
  PartitionedCachePolicy(int64_t capacity, int64_t num_partitions);

  PartitionedCachePolicy() = default;

  /**
   * @brief The policy query function.
   * @param keys The keys to query the cache.
   *
   * @return (positions, indices, missing_keys), where positions has the
   * locations of the keys which were found in the cache, missing_keys has the
   * keys that were not found and indices is defined such that
   * keys[indices[:positions.size(0)]] gives us the found keys and
   * keys[indices[positions.size(0):]] is identical to missing_keys.
   */
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> Query(
      torch::Tensor keys);

  /**
   * @brief The policy replace function.
   * @param keys The keys to query the cache.
   *
   * @return positions tensor is returned holding the locations of the replaced
   * entries in the cache.
   */
  torch::Tensor Replace(torch::Tensor keys);

  static c10::intrusive_ptr<PartitionedCachePolicy> Create(
      int64_t capacity, int64_t num_partitions);

 private:
  static constexpr uint64_t seed = 1e9 + 7;

  int32_t PartAssignment(int64_t key) {
    pcg32 rng(seed, key);
    std::uniform_int_distribution<int32_t> dist(0, policies_.size() - 1);
    return dist(rng);
  }

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> Partition(
      torch::Tensor keys);

  int64_t capacity_;
  std::vector<BaseCachePolicy> policies_;
};

using PartitionedS3FifoCachePolicy = PartitionedCachePolicy<S3FifoCachePolicy>;

}  // namespace storage
}  // namespace graphbolt

#endif  // GRAPHBOLT_PARTITIONED_CACHE_H_
