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
 * @file cache_policy.cc
 * @brief Cache policy implementation on the CPU.
 */
#include "./cache_policy.h"

namespace graphbolt {
namespace storage {

template <typename CachePolicy>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
BaseCachePolicy::QueryImpl(CachePolicy& policy, torch::Tensor keys) {
  auto positions = torch::empty_like(
      keys,
      keys.options().dtype(torch::kInt64).pinned_memory(keys.is_pinned()));
  auto indices = torch::empty_like(
      keys,
      keys.options().dtype(torch::kInt64).pinned_memory(keys.is_pinned()));
  auto filtered_keys =
      torch::empty_like(keys, keys.options().pinned_memory(keys.is_pinned()));
  int64_t found_cnt = 0;
  int64_t missing_cnt = keys.size(0);
  AT_DISPATCH_INDEX_TYPES(
      keys.scalar_type(), "BaseCachePolicy::Query::DispatchForKeys", ([&] {
        auto keys_ptr = keys.data_ptr<index_t>();
        auto positions_ptr = positions.data_ptr<int64_t>();
        auto indices_ptr = indices.data_ptr<int64_t>();
        auto filtered_keys_ptr = filtered_keys.data_ptr<index_t>();
        for (int64_t i = 0; i < keys.size(0); i++) {
          const auto key = keys_ptr[i];
          auto pos = policy.Read(key);
          if (pos.has_value()) {
            positions_ptr[found_cnt] = *pos;
            filtered_keys_ptr[found_cnt] = key;
            indices_ptr[found_cnt++] = i;
          } else {
            indices_ptr[--missing_cnt] = i;
            filtered_keys_ptr[missing_cnt] = key;
          }
        }
      }));
  return {
      positions.slice(0, 0, found_cnt), indices,
      filtered_keys.slice(0, found_cnt), filtered_keys.slice(0, 0, found_cnt)};
}

template <typename CachePolicy>
torch::Tensor BaseCachePolicy::ReplaceImpl(
    CachePolicy& policy, torch::Tensor keys) {
  auto positions = torch::empty_like(
      keys,
      keys.options().dtype(torch::kInt64).pinned_memory(keys.is_pinned()));
  AT_DISPATCH_INDEX_TYPES(
      keys.scalar_type(), "BaseCachePolicy::Replace", ([&] {
        auto keys_ptr = keys.data_ptr<index_t>();
        auto positions_ptr = positions.data_ptr<int64_t>();
        phmap::flat_hash_set<int64_t> position_set;
        position_set.reserve(keys.size(0));
        for (int64_t i = 0; i < keys.size(0); i++) {
          const auto key = keys_ptr[i];
          const auto pos_optional = policy.Read(key);
          const auto pos = pos_optional ? *pos_optional : policy.Insert(key);
          positions_ptr[i] = pos;
          TORCH_CHECK(
              std::get<1>(position_set.insert(pos)),
              "Can't insert all, larger cache capacity is needed.");
        }
      }));
  return positions;
}

template <typename CachePolicy>
void BaseCachePolicy::ReadingCompletedImpl(
    CachePolicy& policy, torch::Tensor keys) {
  AT_DISPATCH_INDEX_TYPES(
      keys.scalar_type(), "BaseCachePolicy::ReadingCompleted", ([&] {
        auto keys_ptr = keys.data_ptr<index_t>();
        for (int64_t i = 0; i < keys.size(0); i++) {
          policy.Unmark(keys_ptr[i]);
        }
      }));
}

S3FifoCachePolicy::S3FifoCachePolicy(int64_t capacity)
    : small_queue_(capacity),
      main_queue_(capacity),
      ghost_queue_(capacity - capacity / 10),
      capacity_(capacity),
      cache_usage_(0),
      small_queue_size_target_(capacity / 10) {
  TORCH_CHECK(small_queue_size_target_ > 0, "Capacity is not large enough.");
  ghost_set_.reserve(ghost_queue_.Capacity());
  key_to_cache_key_.reserve(capacity);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
S3FifoCachePolicy::Query(torch::Tensor keys) {
  return QueryImpl(*this, keys);
}

torch::Tensor S3FifoCachePolicy::Replace(torch::Tensor keys) {
  return ReplaceImpl(*this, keys);
}

void S3FifoCachePolicy::ReadingCompleted(torch::Tensor keys) {
  ReadingCompletedImpl(*this, keys);
}

SieveCachePolicy::SieveCachePolicy(int64_t capacity)
    : hand_(queue_.end()), capacity_(capacity), cache_usage_(0) {
  TORCH_CHECK(capacity > 0, "Capacity needs to be positive.");
  key_to_cache_key_.reserve(capacity);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
SieveCachePolicy::Query(torch::Tensor keys) {
  return QueryImpl(*this, keys);
}

torch::Tensor SieveCachePolicy::Replace(torch::Tensor keys) {
  return ReplaceImpl(*this, keys);
}

void SieveCachePolicy::ReadingCompleted(torch::Tensor keys) {
  ReadingCompletedImpl(*this, keys);
}

LruCachePolicy::LruCachePolicy(int64_t capacity)
    : capacity_(capacity), cache_usage_(0) {
  TORCH_CHECK(capacity > 0, "Capacity needs to be positive.");
  key_to_cache_key_.reserve(capacity);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
LruCachePolicy::Query(torch::Tensor keys) {
  return QueryImpl(*this, keys);
}

torch::Tensor LruCachePolicy::Replace(torch::Tensor keys) {
  return ReplaceImpl(*this, keys);
}

void LruCachePolicy::ReadingCompleted(torch::Tensor keys) {
  ReadingCompletedImpl(*this, keys);
}

ClockCachePolicy::ClockCachePolicy(int64_t capacity)
    : queue_(capacity), capacity_(capacity), cache_usage_(0) {
  TORCH_CHECK(capacity > 0, "Capacity needs to be positive.");
  key_to_cache_key_.reserve(capacity);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
ClockCachePolicy::Query(torch::Tensor keys) {
  return QueryImpl(*this, keys);
}

torch::Tensor ClockCachePolicy::Replace(torch::Tensor keys) {
  return ReplaceImpl(*this, keys);
}

void ClockCachePolicy::ReadingCompleted(torch::Tensor keys) {
  ReadingCompletedImpl(*this, keys);
}

}  // namespace storage
}  // namespace graphbolt
