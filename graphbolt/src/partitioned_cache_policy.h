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

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> Query(
      torch::Tensor keys);

  torch::Tensor Replace(torch::Tensor keys);

  static c10::intrusive_ptr<PartitionedCachePolicy> Create(
      int64_t capacity, int64_t num_partitions);

 private:
  static constexpr uint64_t seed = 1e9 + 7;

  int32_t part_assignment(int64_t key) {
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
