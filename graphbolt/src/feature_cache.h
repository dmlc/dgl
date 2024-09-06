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
 * @file feature_cache.h
 * @brief Feature cache implementation on the CPU.
 */
#ifndef GRAPHBOLT_FEATURE_CACHE_H_
#define GRAPHBOLT_FEATURE_CACHE_H_

#include <graphbolt/async.h>
#include <torch/custom_class.h>
#include <torch/torch.h>

#include <vector>

namespace graphbolt {
namespace storage {

struct FeatureCache : public torch::CustomClassHolder {
  /**
   * @brief Constructor for the FeatureCache struct.
   *
   * @param shape The shape of the cache.
   * @param dtype The dtype of elements stored in the cache.
   * @param pin_memory Whether to pin the memory of the cache storage tensor.
   */
  FeatureCache(
      const std::vector<int64_t>& shape, torch::ScalarType dtype,
      bool pin_memory);

  bool IsPinned() const { return tensor_.is_pinned(); }

  int64_t NumBytes() const { return tensor_.numel() * tensor_.element_size(); }

  /**
   * @brief The cache query function. Allocates an empty tensor `values` with
   * size as the first dimension and runs
   * values[indices[:positions.size(0)]] = cache_tensor[positions] before
   * returning it.
   *
   * @param positions The positions of the queried items.
   * @param indices The indices of the queried items among the original keys.
   * Only the first portion corresponding to the provided positions tensor is
   * used, e.g. indices[:positions.size(0)].
   * @param size The size of the original keys, hence the first dimension of
   * the output shape.
   *
   * @return The values tensor is returned. Its memory is pinned if pin_memory
   * is true.
   */
  torch::Tensor Query(
      torch::Tensor positions, torch::Tensor indices, int64_t size);

  c10::intrusive_ptr<Future<torch::Tensor>> QueryAsync(
      torch::Tensor positions, torch::Tensor indices, int64_t size);

  /**
   * @brief The cache tensor index_select returns cache_tensor[positions].
   *
   * @param positions The positions of the queried items.
   *
   * @return The values tensor is returned on the same device as positions.
   */
  torch::Tensor IndexSelect(torch::Tensor positions);

  /**
   * @brief The cache replace function.
   *
   * @param positions The positions to replace in the cache.
   * @param values The values to be inserted into the cache.
   */
  void Replace(torch::Tensor positions, torch::Tensor values);

  c10::intrusive_ptr<Future<void>> ReplaceAsync(
      torch::Tensor positions, torch::Tensor values);

  static c10::intrusive_ptr<FeatureCache> Create(
      const std::vector<int64_t>& shape, torch::ScalarType dtype,
      bool pin_memory);

 private:
  torch::Tensor tensor_;
};

}  // namespace storage
}  // namespace graphbolt

#endif  // GRAPHBOLT_FEATURE_CACHE_H_
