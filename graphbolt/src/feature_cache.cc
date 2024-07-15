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

#include "./index_select.h"

namespace graphbolt {
namespace storage {

constexpr int kIntGrainSize = 64;

FeatureCache::FeatureCache(
    const std::vector<int64_t>& shape, torch::ScalarType dtype, bool pin_memory)
    : tensor_(torch::empty(
          shape, c10::TensorOptions().dtype(dtype).pinned_memory(pin_memory))) {
}

torch::Tensor FeatureCache::Query(
    torch::Tensor positions, torch::Tensor indices, int64_t size) {
  const bool pin_memory = positions.is_pinned() || indices.is_pinned();
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

torch::Tensor FeatureCache::IndexSelect(torch::Tensor positions) {
  return ops::IndexSelect(tensor_, positions);
}

void FeatureCache::Replace(torch::Tensor positions, torch::Tensor values) {
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
    const std::vector<int64_t>& shape, torch::ScalarType dtype,
    bool pin_memory) {
  return c10::make_intrusive<FeatureCache>(shape, dtype, pin_memory);
}

}  // namespace storage
}  // namespace graphbolt
