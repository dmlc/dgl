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
   * @param offset The offset to be added to the keys.
   *
   * @return (positions, indices, missing_keys, found_ptrs, found_offsets,
   * missing_offsets), where positions has the locations of the keys which were
   * found in the cache, missing_keys has the keys that were not found and
   * indices is defined such that keys[indices[:positions.size(0)]] gives us the
   * keys for the found pointers and keys[indices[positions.size(0):]] is
   * identical to missing_keys. The found_offsets tensor holds the partition
   * offsets for the found pointers. The missing_offsets holds the partition
   * offsets for the missing_keys.
   */
  std::tuple<
      torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
      torch::Tensor>
  Query(torch::Tensor keys, int64_t offset);

  c10::intrusive_ptr<Future<std::vector<torch::Tensor>>> QueryAsync(
      torch::Tensor keys, int64_t offset);

  /**
   * @brief The policy query and then replace function.
   * @param keys The keys to query the cache.
   * @param offset The offset to be added to the keys.
   *
   * @return (positions, indices, pointers, missing_keys, found_offsets,
   * missing_offsets), where positions has the locations of the keys which were
   * emplaced into the cache, pointers point to the emplaced CacheKey pointers
   * in the cache, missing_keys has the keys that were not found and just
   * inserted and indices is defined such that keys[indices[:keys.size(0) -
   * missing_keys.size(0)]] gives us the keys for the found keys and
   * keys[indices[keys.size(0) - missing_keys.size(0):]] is identical to
   * missing_keys. The found_offsets tensor holds the partition offsets for the
   * found pointers. The missing_offsets holds the partition offsets for the
   * missing_keys and missing pointers.
   */
  std::tuple<
      torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
      torch::Tensor>
  QueryAndReplace(torch::Tensor keys, int64_t offset);

  c10::intrusive_ptr<Future<std::vector<torch::Tensor>>> QueryAndReplaceAsync(
      torch::Tensor keys, int64_t offset);

  /**
   * @brief The policy replace function.
   * @param keys The keys to query the cache.
   * @param offsets The partition offsets for the keys.
   * @param offset The offset to be added to the keys.
   *
   * @return (positions, pointers, offsets), where positions holds the locations
   * of the replaced entries in the cache, pointers holds the CacheKey pointers
   * for the inserted keys and offsets holds the partition offsets for pointers.
   */
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> Replace(
      torch::Tensor keys, torch::optional<torch::Tensor> offsets,
      int64_t offset);

  c10::intrusive_ptr<Future<std::vector<torch::Tensor>>> ReplaceAsync(
      torch::Tensor keys, torch::optional<torch::Tensor> offsets,
      int64_t offset);

  template <bool write>
  void ReadingWritingCompletedImpl(
      torch::Tensor pointers, torch::Tensor offsets);

  /**
   * @brief A reader has finished reading these keys, so they can be
   * evicted.
   * @param pointers The CacheKey pointers in the cache to unmark.
   * @param offsets The partition offsets for the pointers.
   */
  void ReadingCompleted(torch::Tensor pointers, torch::Tensor offsets);

  /**
   * @brief A writer has finished writing these keys, so they can be evicted.
   * @param pointers The CacheKey pointers in the cache to unmark.
   * @param offsets The partition offsets for the pointers.
   */
  void WritingCompleted(torch::Tensor pointers, torch::Tensor offsets);

  c10::intrusive_ptr<Future<void>> ReadingCompletedAsync(
      torch::Tensor pointers, torch::Tensor offsets);

  c10::intrusive_ptr<Future<void>> WritingCompletedAsync(
      torch::Tensor pointers, torch::Tensor offsets);

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
  std::mutex mtx_;
};

}  // namespace storage
}  // namespace graphbolt

#endif  // GRAPHBOLT_PARTITIONED_CACHE_H_
