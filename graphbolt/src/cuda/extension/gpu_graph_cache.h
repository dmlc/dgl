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

#include <graphbolt/async.h>
#include <torch/custom_class.h>
#include <torch/torch.h>

#include <mutex>

namespace graphbolt {
namespace cuda {

class GpuGraphCache : public torch::CustomClassHolder {
  // The load factor of the constructed hash table.
  static constexpr double kDoubleLoadFactor = 0.8;
  // The growth factor of the hash table and the dynamically sized indptr
  // tensor.
  static constexpr int kIntGrowthFactor = 2;

 public:
  /**
   * @brief Constructor for the GpuGraphCache struct.
   *
   * @param num_edges The edge capacity of GPU cache.
   * @param threshold The access threshold before a vertex neighborhood is
   * cached.
   * @param indptr_dtype The node id datatype.
   * @param dtypes The dtypes of the edge tensors to be cached. dtypes[0] is
   * reserved for the indices edge tensor holding node ids.
   * @param has_original_edge_ids Whether the graph to be cached has original
   * edge ids.
   */
  GpuGraphCache(
      const int64_t num_edges, const int64_t threshold,
      torch::ScalarType indptr_dtype, std::vector<torch::ScalarType> dtypes,
      bool has_original_edge_ids);

  GpuGraphCache() = default;

  ~GpuGraphCache();

  /**
   * @brief Queries the cache. Returns tensors indicating which elements are
   * missing.
   *
   * @param seeds The node ids to query the cache with.
   *
   * @return
   * (torch::Tensor, torch::Tensor, int64_t, int64_t) index, position,
   * number of cache hits and number of ids that will enter the cache.
   */
  std::tuple<torch::Tensor, torch::Tensor, int64_t, int64_t> Query(
      torch::Tensor seeds);

  c10::intrusive_ptr<
      Future<std::tuple<torch::Tensor, torch::Tensor, int64_t, int64_t>>>
  QueryAsync(torch::Tensor seeds);

  /**
   * @brief After the graph structure for the missing node ids are fetched, it
   * inserts the node ids which passes the threshold and returns the final
   * output graph structure, combining the information in the cache with the
   * graph structure for the missing node ids.
   *
   * @param seeds The node ids that the cache was queried with.
   * @param indices seeds[indices[:num_hit]] gives us the node ids that were
   * found in the cache
   * @param positions positions[:num_hit] gives where the node ids can be found
   * in the cache.
   * @param num_hit The number of seeds that are already in the cache.
   * @param num_threshold The number of seeds among the missing node ids that
   * will be inserted into the cache.
   * @param indptr The indptr for the missing seeds fetched from remote.
   * @param edge_tensors The edge tensors for the missing seeds. The last
   * element of edge_tensors is treated as the edge ids tensor with
   * indptr_dtype.
   *
   * @return (torch::Tensor, std::vector<torch::Tensor>) The final indptr and
   * edge_tensors, directly corresponding to the seeds tensor.
   */
  std::tuple<torch::Tensor, std::vector<torch::Tensor>> Replace(
      torch::Tensor seeds, torch::Tensor indices, torch::Tensor positions,
      int64_t num_hit, int64_t num_threshold, torch::Tensor indptr,
      std::vector<torch::Tensor> edge_tensors);

  c10::intrusive_ptr<
      Future<std::tuple<torch::Tensor, std::vector<torch::Tensor>>>>
  ReplaceAsync(
      torch::Tensor seeds, torch::Tensor indices, torch::Tensor positions,
      int64_t num_hit, int64_t num_threshold, torch::Tensor indptr,
      std::vector<torch::Tensor> edge_tensors);

  static c10::intrusive_ptr<GpuGraphCache> Create(
      const int64_t num_edges, const int64_t threshold,
      torch::ScalarType indptr_dtype, std::vector<torch::ScalarType> dtypes,
      bool has_original_edge_ids);

 private:
  void* map_;                     // pointer to the hash table.
  int64_t threshold_;             // A positive threshold value.
  torch::DeviceIndex device_id_;  // Which GPU the cache resides in.
  int64_t map_size_;              // The number of nodes inside the hash table.
  int64_t num_nodes_;             // The number of cached nodes in the cache.
  int64_t num_edges_;             // The number of cached edges in the cache.
  torch::Tensor indptr_;          // The cached graph structure indptr tensor.
  torch::optional<torch::Tensor>
      offset_;  // The original graph's sliced_indptr tensor.
  std::vector<torch::Tensor> cached_edge_tensors_;  // The cached graph
                                                    // structure edge tensors.
  std::mutex mtx_;  // Protects the data structure and makes it threadsafe.
};

}  // namespace cuda
}  // namespace graphbolt

#endif  // GRAPHBOLT_GPU_CACHE_H_
