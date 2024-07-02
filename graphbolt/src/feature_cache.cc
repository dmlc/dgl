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
 * @file feature_cache.cc
 * @brief Feature cache implementation on the CPU.
 */
#include "./feature_cache.h"

namespace graphbolt {
namespace storage {

constexpr int kIntGrainSize = 64;

S3FifoCachePolicy::S3FifoCachePolicy(int64_t capacity)
    : S_{capacity / 10},
      M_{capacity - capacity / 10},
      position_q_{0},
      ghost_q_time_{0},
      capacity_{capacity} {}

FeatureCache::FeatureCache(
    const std::vector<int64_t>& shape, torch::ScalarType dtype)
    : tensor_{torch::empty(shape, c10::TensorOptions().dtype(dtype))} {}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
S3FifoCachePolicy::Query(torch::Tensor keys) {
  auto positions = torch::empty_like(keys, keys.options().dtype(torch::kInt64));
  auto indices = torch::empty_like(keys, keys.options().dtype(torch::kInt64));
  auto missing_keys = torch::empty_like(keys);
  int64_t found_cnt = 0;
  int64_t missing_cnt = keys.size(0);
  AT_DISPATCH_INDEX_TYPES(keys.scalar_type(), "S3FifoCachePolicy::Query", ([&] {
                            auto keys_ptr = keys.data_ptr<index_t>();
                            auto positions_ptr = positions.data_ptr<int64_t>();
                            auto indices_ptr = indices.data_ptr<int64_t>();
                            auto missing_keys_ptr =
                                missing_keys.data_ptr<index_t>();
                            for (int64_t i = 0; i < keys.size(0); i++) {
                              const auto key = keys_ptr[i];
                              auto it = position_map_.find(key);
                              if (it != position_map_.end()) {
                                auto& key_ref = *it->second;
                                key_ref.Increment();
                                positions_ptr[found_cnt] = key_ref.position_;
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

torch::Tensor FeatureCache::Query(
    torch::Tensor positions, torch::Tensor indices, int64_t size,
    bool pin_memory) {
  std::vector<int64_t> output_shape{
      tensor_.sizes().begin(), tensor_.sizes().end()};
  output_shape[0] = size;
  auto values =
      torch::empty(output_shape, tensor_.options().pinned_memory(pin_memory));
  const auto row_bytes = values.slice(0, 0, 1).numel() * values.element_size();
  auto values_ptr = reinterpret_cast<std::byte*>(values.data_ptr());
  const auto tensor_ptr = reinterpret_cast<std::byte*>(tensor_.data_ptr());
  const auto positions_ptr = positions.data_ptr<int64_t>();
  const auto indices_ptr = indices.data_ptr<int64_t>();
  torch::parallel_for(
      0, positions.size(0), kIntGrainSize, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; i++) {
          std::memcpy(
              values_ptr + indices_ptr[i] * row_bytes,
              tensor_ptr + positions_ptr[i] * row_bytes, row_bytes);
        }
      });
  return values;
}

torch::Tensor S3FifoCachePolicy::Replace(torch::Tensor keys) {
  auto positions = torch::empty_like(keys, keys.options().dtype(torch::kInt64));
  AT_DISPATCH_INDEX_TYPES(
      keys.scalar_type(), "S3FifoCachePolicy::Replace", ([&] {
        auto keys_ptr = keys.data_ptr<index_t>();
        auto positions_ptr = positions.data_ptr<int64_t>();
        for (int64_t i = 0; i < keys.size(0); i++) {
          const auto key = keys_ptr[i];
          auto it = position_map_.find(key);
          if (it != position_map_.end()) {  // Already in the cache, inc freq.
            auto& key_ref = *it->second;
            key_ref.Increment();
            positions_ptr[i] = key_ref.position_;
          } else {
            const auto inside_G = InG(key);
            auto& Queue = inside_G ? M_ : S_;
            const auto pos = Queue.IsFull() ? (inside_G ? EvictM() : EvictS())
                                            : position_q_++;
            TORCH_CHECK(0 <= pos && pos < capacity_, "Position out of bounds!");
            position_map_[key] = Queue.Push(CacheKey{key, pos});
            positions_ptr[i] = pos;
          }
        }
      }));
  return positions;
}

void FeatureCache::Replace(
    torch::Tensor keys, torch::Tensor values, torch::Tensor positions) {
  if (keys.size(0) > tensor_.size(0)) {
    keys = keys.slice(0, 0, tensor_.size(0));
    values = values.slice(0, 0, tensor_.size(0));
  }
  const auto row_bytes = values.slice(0, 0, 1).numel() * values.element_size();
  auto values_ptr = reinterpret_cast<std::byte*>(values.data_ptr());
  const auto tensor_ptr = reinterpret_cast<std::byte*>(tensor_.data_ptr());
  const auto positions_ptr = positions.data_ptr<int64_t>();
  torch::parallel_for(
      0, positions.size(0), kIntGrainSize, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; i++) {
          std::memcpy(
              tensor_ptr + positions_ptr[i] * row_bytes,
              values_ptr + i * row_bytes, row_bytes);
        }
      });
}

c10::intrusive_ptr<FeatureCache> FeatureCache::Create(
    const std::vector<int64_t>& shape, torch::ScalarType dtype) {
  return c10::make_intrusive<FeatureCache>(shape, dtype);
}

c10::intrusive_ptr<S3FifoCachePolicy> S3FifoCachePolicy::Create(
    int64_t capacity) {
  return c10::make_intrusive<S3FifoCachePolicy>(capacity);
}

}  // namespace storage
}  // namespace graphbolt
