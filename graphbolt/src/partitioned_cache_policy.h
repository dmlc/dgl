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

#include <graphbolt/async.h>
#include <torch/custom_class.h>
#include <torch/torch.h>

#include <mutex>
#include <pcg_random.hpp>
#include <random>
#include <type_traits>
#include <vector>

#include "./cache_policy.h"

namespace graphbolt {
namespace storage {

/**
 * @brief PartitionedCachePolicy works by partitioning the key space to a set
 * number of partitions that is provided as the second argument of its
 * constructor. Since the partitioning is random but deterministic, the caching
 * policy performance is not affected as the key distribution stays the same in
 * each partition.
 **/
class PartitionedCachePolicy : public torch::CustomClassHolder {
 public:
  /**
   * @brief The policy query function.
   * @param capacity The capacity of the cache.
   * @param num_partitions The number of caching policies instantiated in a
   * one-to-one mapping to each partition.
   */
  template <typename CachePolicy>
  PartitionedCachePolicy(CachePolicy, int64_t capacity, int64_t num_partitions);

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
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> Query(
      torch::Tensor keys);

  c10::intrusive_ptr<Future<std::vector<torch::Tensor>>> QueryAsync(
      torch::Tensor keys);

  /**
   * @brief The policy replace function.
   * @param keys The keys to query the cache.
   *
   * @return result, where result[0] is the positions tensor holding the
   * locations of the replaced entries in the cache. The rest of the tensors are
   * (offsets, indices, permuted_keys) output from Partition(keys).
   */
  std::vector<torch::Tensor> Replace(torch::Tensor keys);

  c10::intrusive_ptr<Future<std::vector<torch::Tensor>>> ReplaceAsync(
      torch::Tensor keys);

  template <bool write>
  void ReadingWritingCompletedImpl(
      torch::Tensor keys, std::vector<torch::Tensor>& partition_result);

  /**
   * @brief A reader has finished reading these keys, so they can be
   * evicted.
   * @param keys The keys to unmark.
   * @param partition_result The output of Partition(keys) if available.
   * Otherwise when partition_result.empty() is true, it will be computed.
   */
  void ReadingCompleted(
      torch::Tensor keys, std::vector<torch::Tensor> partition_result);

  /**
   * @brief A writer has finished writing these keys, so they can be evicted.
   * @param keys The keys to unmark.
   * @param partition_result The output of Partition(keys) if available.
   * Otherwise when partition_result.empty() is true, it will be computed.
   */
  void WritingCompleted(
      torch::Tensor keys, std::vector<torch::Tensor> partition_result);

  c10::intrusive_ptr<Future<void>> ReadingCompletedAsync(
      torch::Tensor keys, std::vector<torch::Tensor> partition_result);

  c10::intrusive_ptr<Future<void>> WritingCompletedAsync(
      torch::Tensor keys, std::vector<torch::Tensor> partition_result);

  template <typename CachePolicy>
  static c10::intrusive_ptr<PartitionedCachePolicy> Create(
      int64_t capacity, int64_t num_partitions);

 private:
  static constexpr uint64_t seed = 1e9 + 7;

  /**
   * @brief Deterministic assignment of keys to different parts.
   */
  int32_t PartAssignment(int64_t key) {
    pcg32 rng(seed, key);
    std::uniform_int_distribution<int32_t> dist(0, policies_.size() - 1);
    return dist(rng);
  }

  /**
   * @brief The partition function for a given keys tensor.
   * @param keys The keys to query the cache.
   *
   * @return (offsets, indices, permuted_keys), the returned tensors have the
   * following properties:
   * permuted_keys[offsets[i]: offsets[i + 1]] belong to part i and
   * keys[indices] == permuted_keys
   */
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> Partition(
      torch::Tensor keys);

  int64_t capacity_;
  std::vector<std::unique_ptr<BaseCachePolicy>> policies_;
};

}  // namespace storage
}  // namespace graphbolt

#endif  // GRAPHBOLT_PARTITIONED_CACHE_H_
