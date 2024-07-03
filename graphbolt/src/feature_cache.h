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
   */
  FeatureCache(const std::vector<int64_t>& shape, torch::ScalarType dtype);

  FeatureCache() = default;

  torch::Tensor Query(
      torch::Tensor positions, torch::Tensor indices, int64_t size,
      bool pin_memory);

  void Replace(
      torch::Tensor keys, torch::Tensor values, torch::Tensor positions);

  static c10::intrusive_ptr<FeatureCache> Create(
      const std::vector<int64_t>& shape, torch::ScalarType dtype);

 private:
  torch::Tensor tensor_;
};

}  // namespace storage
}  // namespace graphbolt

#endif  // GRAPHBOLT_FEATURE_CACHE_H_
