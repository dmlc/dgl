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

S3FifoCachePolicy::S3FifoCachePolicy(int64_t capacity)
    : small_queue_(capacity / 10),
      main_queue_(capacity - capacity / 10),
      ghost_q_time_(0),
      capacity_(capacity),
      cache_usage_(0) {}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
S3FifoCachePolicy::Query(torch::Tensor keys) {
  auto positions = torch::empty_like(keys, keys.options().dtype(torch::kInt64));
  auto indices = torch::empty_like(keys, keys.options().dtype(torch::kInt64));
  auto missing_keys = torch::empty_like(keys);
  int64_t found_cnt = 0;
  int64_t missing_cnt = keys.size(0);
  AT_DISPATCH_INDEX_TYPES(
      keys.scalar_type(), "S3FifoCachePolicy::Query::DispatchForKeys", ([&] {
        auto keys_ptr = keys.data_ptr<index_t>();
        auto positions_ptr = positions.data_ptr<int64_t>();
        auto indices_ptr = indices.data_ptr<int64_t>();
        auto missing_keys_ptr = missing_keys.data_ptr<index_t>();
        for (int64_t i = 0; i < keys.size(0); i++) {
          const auto key = keys_ptr[i];
          auto it = key_to_cache_key_.find(key);
          if (it != key_to_cache_key_.end()) {
            auto& cache_key = *it->second;
            cache_key.Increment();
            positions_ptr[found_cnt] = cache_key.getPos();
            indices_ptr[found_cnt++] = i;
          } else {
            indices_ptr[--missing_cnt] = i;
            missing_keys_ptr[missing_cnt] = key;
          }
        }
      }));
  return {
      positions.slice(0, 0, found_cnt), indices,
      missing_keys.slice(0, found_cnt)};
}

torch::Tensor S3FifoCachePolicy::Replace(torch::Tensor keys) {
  auto positions = torch::empty_like(keys, keys.options().dtype(torch::kInt64));
  AT_DISPATCH_INDEX_TYPES(
      keys.scalar_type(), "S3FifoCachePolicy::Replace", ([&] {
        auto keys_ptr = keys.data_ptr<index_t>();
        auto positions_ptr = positions.data_ptr<int64_t>();
        for (int64_t i = 0; i < keys.size(0); i++) {
          const auto key = keys_ptr[i];
          auto it = key_to_cache_key_.find(key);
          if (it !=
              key_to_cache_key_.end()) {  // Already in the cache, inc freq.
            auto& cache_key = *it->second;
            cache_key.Increment();
            positions_ptr[i] = cache_key.getPos();
          } else {
            const auto in_ghost_queue = InGhostQueue(key);
            auto& queue = in_ghost_queue ? main_queue_ : small_queue_;
            int64_t pos;
            if (queue.IsFull()) {
              // When the queue is full, we need to make a space by evicting.
              // Inside ghost queue means insertion into M, otherwise S.
              pos = (in_ghost_queue ? EvictM() : EvictS());
            } else {  // If the cache is not full yet, get an unused empty slot.
              pos = cache_usage_++;
            }
            TORCH_CHECK(0 <= pos && pos < capacity_, "Position out of bounds!");
            key_to_cache_key_[key] = queue.Push(CacheKey(key, pos));
            positions_ptr[i] = pos;
          }
        }
      }));
  if (static_cast<int64_t>(ghost_map_.size()) >= 2 * main_queue_.Capacity()) {
    // Here, we ensure that the ghost_map_ does not grow too much.
    decltype(ghost_map_) filtered_map;
    filtered_map.reserve(ghost_map_.size());
    for (const auto [key, timestamp] : ghost_map_) {
      if (ghost_q_time_ - timestamp <= main_queue_.Capacity()) {
        filtered_map[key] = timestamp;
      }
    }
    ghost_map_ = filtered_map;
  }
  return positions;
}

c10::intrusive_ptr<S3FifoCachePolicy> S3FifoCachePolicy::Create(
    int64_t capacity) {
  return c10::make_intrusive<S3FifoCachePolicy>(capacity);
}

}  // namespace storage
}  // namespace graphbolt
