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

#include "./utils.h"

namespace graphbolt {
namespace storage {

template <typename CachePolicy>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
BaseCachePolicy::QueryImpl(CachePolicy& policy, torch::Tensor keys) {
  auto positions = torch::empty_like(
      keys, keys.options()
                .dtype(torch::kInt64)
                .pinned_memory(utils::is_pinned(keys)));
  auto indices = torch::empty_like(
      keys, keys.options()
                .dtype(torch::kInt64)
                .pinned_memory(utils::is_pinned(keys)));
  auto found_ptr_tensor = torch::empty_like(
      keys, keys.options()
                .dtype(torch::kInt64)
                .pinned_memory(utils::is_pinned(keys)));
  auto missing_keys = torch::empty_like(
      keys, keys.options().pinned_memory(utils::is_pinned(keys)));
  int64_t found_cnt = 0;
  int64_t missing_cnt = keys.size(0);
  AT_DISPATCH_INDEX_TYPES(
      keys.scalar_type(), "BaseCachePolicy::Query::DispatchForKeys", ([&] {
        auto keys_ptr = keys.data_ptr<index_t>();
        auto positions_ptr = positions.data_ptr<int64_t>();
        auto indices_ptr = indices.data_ptr<int64_t>();
        static_assert(
            sizeof(CacheKey*) == sizeof(int64_t), "You need 64 bit pointers.");
        auto found_ptr =
            reinterpret_cast<CacheKey**>(found_ptr_tensor.data_ptr<int64_t>());
        auto missing_keys_ptr = missing_keys.data_ptr<index_t>();
        for (int64_t i = 0; i < keys.size(0); i++) {
          const auto key = keys_ptr[i];
          auto cache_key_ptr = policy.Read(key);
          if (cache_key_ptr) {
            positions_ptr[found_cnt] = cache_key_ptr->getPos();
            found_ptr[found_cnt] = cache_key_ptr;
            indices_ptr[found_cnt++] = i;
          } else {
            indices_ptr[--missing_cnt] = i;
            missing_keys_ptr[missing_cnt] = key;
          }
        }
      }));
  return {
      positions.slice(0, 0, found_cnt), indices,
      missing_keys.slice(0, found_cnt),
      found_ptr_tensor.slice(0, 0, found_cnt)};
}

template <typename CachePolicy>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
BaseCachePolicy::QueryAndReplaceImpl(CachePolicy& policy, torch::Tensor keys) {
  auto positions = torch::empty_like(
      keys, keys.options()
                .dtype(torch::kInt64)
                .pinned_memory(utils::is_pinned(keys)));
  auto indices = torch::empty_like(
      keys, keys.options()
                .dtype(torch::kInt64)
                .pinned_memory(utils::is_pinned(keys)));
  auto pointers = torch::empty_like(keys, keys.options().dtype(torch::kInt64));
  auto missing_keys = torch::empty_like(
      keys, keys.options().pinned_memory(utils::is_pinned(keys)));
  int64_t found_cnt = 0;
  int64_t missing_cnt = keys.size(0);
  AT_DISPATCH_INDEX_TYPES(
      keys.scalar_type(), "BaseCachePolicy::Replace", ([&] {
        auto keys_ptr = keys.data_ptr<index_t>();
        auto positions_ptr = positions.data_ptr<int64_t>();
        auto indices_ptr = indices.data_ptr<int64_t>();
        static_assert(
            sizeof(CacheKey*) == sizeof(int64_t), "You need 64 bit pointers.");
        auto pointers_ptr =
            reinterpret_cast<CacheKey**>(pointers.data_ptr<int64_t>());
        auto missing_keys_ptr = missing_keys.data_ptr<index_t>();
        set_t<int64_t> position_set;
        position_set.reserve(keys.size(0));
        // Query and Replace combined.
        for (int64_t i = 0; i < keys.size(0); i++) {
          const auto key = keys_ptr[i];
          const auto [it, can_read] = policy.Emplace(key);
          if (can_read) {
            auto& cache_key = *it->second;
            positions_ptr[found_cnt] = cache_key.getPos();
            pointers_ptr[found_cnt] = &cache_key;
            indices_ptr[found_cnt++] = i;
          } else {
            indices_ptr[--missing_cnt] = i;
            missing_keys_ptr[missing_cnt] = key;
            // Ensure that even if an offset is added, it stays negative.
            auto position = std::numeric_limits<int64_t>::min();
            CacheKey* cache_key_ptr = nullptr;
            if (it->second == policy.getMapSentinelValue()) {
              cache_key_ptr = policy.Insert(it);
              position = cache_key_ptr->getPos();
              TORCH_CHECK(
                  // We check for the uniqueness of the positions.
                  std::get<1>(position_set.insert(position)),
                  "Can't insert all, larger cache capacity is needed.");
            }
            positions_ptr[missing_cnt] = position;
            pointers_ptr[missing_cnt] = cache_key_ptr;
          }
        }
      }));
  return {positions, indices, pointers, missing_keys.slice(0, found_cnt)};
}

template <typename CachePolicy>
std::tuple<torch::Tensor, torch::Tensor> BaseCachePolicy::ReplaceImpl(
    CachePolicy& policy, torch::Tensor keys) {
  auto positions = torch::empty_like(
      keys, keys.options()
                .dtype(torch::kInt64)
                .pinned_memory(utils::is_pinned(keys)));
  auto pointers = torch::empty_like(
      keys, keys.options()
                .dtype(torch::kInt64)
                .pinned_memory(utils::is_pinned(keys)));
  AT_DISPATCH_INDEX_TYPES(
      keys.scalar_type(), "BaseCachePolicy::Replace", ([&] {
        auto keys_ptr = keys.data_ptr<index_t>();
        auto positions_ptr = positions.data_ptr<int64_t>();
        static_assert(
            sizeof(CacheKey*) == sizeof(int64_t), "You need 64 bit pointers.");
        auto pointers_ptr =
            reinterpret_cast<CacheKey**>(pointers.data_ptr<int64_t>());
        set_t<int64_t> position_set;
        position_set.reserve(keys.size(0));
        for (int64_t i = 0; i < keys.size(0); i++) {
          const auto key = keys_ptr[i];
          // Ensure that even if an offset is added, it stays negative.
          auto position = std::numeric_limits<int64_t>::min();
          CacheKey* cache_key_ptr = nullptr;
          const auto [it, _] = policy.Emplace(key);
          if (it->second == policy.getMapSentinelValue()) {
            cache_key_ptr = policy.Insert(it);
            position = cache_key_ptr->getPos();
            TORCH_CHECK(
                // We check for the uniqueness of the positions.
                std::get<1>(position_set.insert(position)),
                "Can't insert all, larger cache capacity is needed.");
          }
          positions_ptr[i] = position;
          pointers_ptr[i] = cache_key_ptr;
        }
      }));
  return {positions, pointers};
}

template <bool write>
void BaseCachePolicy::ReadingWritingCompletedImpl(torch::Tensor pointers) {
  static_assert(
      sizeof(CacheKey*) == sizeof(int64_t), "You need 64 bit pointers.");
  auto pointers_ptr =
      reinterpret_cast<CacheKey**>(pointers.data_ptr<int64_t>());
  for (int64_t i = 0; i < pointers.size(0); i++) {
    const auto pointer = pointers_ptr[i];
    if (!write || pointer) {
      pointer->EndUse<write>();
    }
  }
}

void BaseCachePolicy::ReadingCompleted(torch::Tensor pointers) {
  ReadingWritingCompletedImpl<false>(pointers);
}

void BaseCachePolicy::WritingCompleted(torch::Tensor pointers) {
  ReadingWritingCompletedImpl<true>(pointers);
}

S3FifoCachePolicy::S3FifoCachePolicy(int64_t capacity)
    : BaseCachePolicy(capacity),
      ghost_queue_(capacity - capacity / 10),
      small_queue_size_target_(capacity / 10),
      small_queue_size_(0) {
  TORCH_CHECK(small_queue_size_target_ > 0, "Capacity is not large enough.");
  ghost_set_.reserve(ghost_queue_.Capacity());
  key_to_cache_key_.reserve(kCapacityFactor * (capacity + 1));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
S3FifoCachePolicy::Query(torch::Tensor keys) {
  return QueryImpl(*this, keys);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
S3FifoCachePolicy::QueryAndReplace(torch::Tensor keys) {
  return QueryAndReplaceImpl(*this, keys);
}

std::tuple<torch::Tensor, torch::Tensor> S3FifoCachePolicy::Replace(
    torch::Tensor keys) {
  return ReplaceImpl(*this, keys);
}

SieveCachePolicy::SieveCachePolicy(int64_t capacity)
    // Ensure that queue_ is constructed first before accessing its `.end()`.
    : BaseCachePolicy(capacity), queue_(), hand_(queue_.end()) {
  TORCH_CHECK(capacity > 0, "Capacity needs to be positive.");
  key_to_cache_key_.reserve(kCapacityFactor * (capacity + 1));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
SieveCachePolicy::Query(torch::Tensor keys) {
  return QueryImpl(*this, keys);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
SieveCachePolicy::QueryAndReplace(torch::Tensor keys) {
  return QueryAndReplaceImpl(*this, keys);
}

std::tuple<torch::Tensor, torch::Tensor> SieveCachePolicy::Replace(
    torch::Tensor keys) {
  return ReplaceImpl(*this, keys);
}

LruCachePolicy::LruCachePolicy(int64_t capacity) : BaseCachePolicy(capacity) {
  TORCH_CHECK(capacity > 0, "Capacity needs to be positive.");
  key_to_cache_key_.reserve(kCapacityFactor * (capacity + 1));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
LruCachePolicy::Query(torch::Tensor keys) {
  return QueryImpl(*this, keys);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
LruCachePolicy::QueryAndReplace(torch::Tensor keys) {
  return QueryAndReplaceImpl(*this, keys);
}

std::tuple<torch::Tensor, torch::Tensor> LruCachePolicy::Replace(
    torch::Tensor keys) {
  return ReplaceImpl(*this, keys);
}

ClockCachePolicy::ClockCachePolicy(int64_t capacity)
    : BaseCachePolicy(capacity) {
  TORCH_CHECK(capacity > 0, "Capacity needs to be positive.");
  key_to_cache_key_.reserve(kCapacityFactor * (capacity + 1));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
ClockCachePolicy::Query(torch::Tensor keys) {
  return QueryImpl(*this, keys);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
ClockCachePolicy::QueryAndReplace(torch::Tensor keys) {
  return QueryAndReplaceImpl(*this, keys);
}

std::tuple<torch::Tensor, torch::Tensor> ClockCachePolicy::Replace(
    torch::Tensor keys) {
  return ReplaceImpl(*this, keys);
}

}  // namespace storage
}  // namespace graphbolt
