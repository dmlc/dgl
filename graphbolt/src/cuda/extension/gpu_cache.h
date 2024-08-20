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
 * @file cuda/gpu_cache.h
 * @brief Header file of HugeCTR gpu_cache wrapper.
 */

#ifndef GRAPHBOLT_GPU_CACHE_H_
#define GRAPHBOLT_GPU_CACHE_H_

#include <graphbolt/async.h>
#include <torch/custom_class.h>
#include <torch/torch.h>

#include <limits>
#include <nv_gpu_cache.hpp>

namespace graphbolt {
namespace cuda {

class GpuCache : public torch::CustomClassHolder {
  using key_t = long long;
  constexpr static int set_associativity = 2;
  constexpr static int WARP_SIZE = 32;
  constexpr static int bucket_size = WARP_SIZE * set_associativity;
  using gpu_cache_t = ::gpu_cache::gpu_cache<
      key_t, uint64_t, std::numeric_limits<key_t>::max(), set_associativity,
      WARP_SIZE>;

 public:
  /**
   * @brief Constructor for the GpuCache struct.
   *
   * @param shape The shape of the GPU cache.
   * @param dtype The datatype of items to be stored.
   */
  GpuCache(const std::vector<int64_t>& shape, torch::ScalarType dtype);

  GpuCache() = default;

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> Query(
      torch::Tensor keys);

  c10::intrusive_ptr<Future<std::vector<torch::Tensor>>> QueryAsync(
      torch::Tensor keys);

  void Replace(torch::Tensor keys, torch::Tensor values);

  static c10::intrusive_ptr<GpuCache> Create(
      const std::vector<int64_t>& shape, torch::ScalarType dtype);

 private:
  std::vector<int64_t> shape_;
  torch::ScalarType dtype_;
  std::unique_ptr<gpu_cache_t> cache_;
  int64_t num_bytes_;
  int64_t num_float_feats_;
  torch::DeviceIndex device_id_;
};

// The cu file in HugeCTR gpu cache uses unsigned int and long long.
// Changing to int64_t results in a mismatch of template arguments.
static_assert(
    sizeof(long long) == sizeof(int64_t),
    "long long and int64_t needs to have the same size.");  // NOLINT

}  // namespace cuda
}  // namespace graphbolt

#endif  // GRAPHBOLT_GPU_CACHE_H_
