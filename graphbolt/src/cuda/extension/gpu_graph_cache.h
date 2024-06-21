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
 * @file cuda/gpu_graph_cache.h
 * @brief Header file of GPU graph cache.
 */

#ifndef GRAPHBOLT_GPU_GRAPH_CACHE_H_
#define GRAPHBOLT_GPU_GRAPH_CACHE_H_

#include <torch/custom_class.h>
#include <torch/torch.h>

#include <limits>
#include <type_traits>

namespace graphbolt {
namespace cuda {

class GpuGraphCache : public torch::CustomClassHolder {
  static constexpr double load_factor = 0.8;
  static constexpr int growth_factor = 2;

 public:
  /**
   * @brief Constructor for the GpuGraphCache struct.
   *
   * @param num_nodes The node capacity of GPU cache.
   * @param num_edges The edge capacity of GPU cache.
   * @param dtype The node id datatype.
   */
  GpuGraphCache(
      const int64_t num_edges, const int64_t threshold,
      torch::ScalarType indptr_dtype, std::vector<torch::ScalarType> dtypes);

  GpuGraphCache() = default;

  ~GpuGraphCache();

  std::tuple<
      torch::Tensor, std::vector<torch::Tensor>, torch::Tensor, torch::Tensor,
      torch::Tensor, int64_t>
  Query(torch::Tensor seeds);

  int64_t Replace(
      torch::Tensor seeds, torch::Tensor missing_indices,
      torch::Tensor missing_positions, int64_t num_entering,
      torch::Tensor indptr, std::vector<torch::Tensor> edge_tensors);

  static c10::intrusive_ptr<GpuGraphCache> Create(
      const int64_t num_edges, const int64_t threshold,
      torch::ScalarType indptr_dtype, std::vector<torch::ScalarType> dtypes);

 private:
  void* map_;
  int64_t threshold_;
  torch::DeviceIndex device_id_;
  int64_t map_size_;
  int64_t map_capacity_;
  int64_t num_nodes_;
  int64_t num_edges_;
  torch::Tensor indptr_;
  std::vector<torch::Tensor> cached_edge_tensors_;
};

}  // namespace cuda
}  // namespace graphbolt

#endif  // GRAPHBOLT_GPU_CACHE_H_
