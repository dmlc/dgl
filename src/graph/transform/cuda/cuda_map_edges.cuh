/**
 *  Copyright 2020-2022 Contributors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 * @file graph/transform/cuda/cuda_map_edges.cuh
 * @brief Device level functions for mapping edges.
 */

#ifndef DGL_GRAPH_TRANSFORM_CUDA_CUDA_MAP_EDGES_CUH_
#define DGL_GRAPH_TRANSFORM_CUDA_CUDA_MAP_EDGES_CUH_

#include <dgl/runtime/c_runtime_api.h>
#include <dgl/base_heterograph.h>
#include <cuda_runtime.h>
#include <dgl/runtime/c_runtime_api.h>

#include <algorithm>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include "../../../runtime/cuda/cuda_common.h"
#include "../../../runtime/cuda/cuda_hashtable.cuh"

using namespace dgl::aten;
using namespace dgl::runtime::cuda;

namespace dgl {
namespace transform {

namespace cuda {

template <typename IdType, int BLOCK_SIZE, IdType TILE_SIZE>
__device__ void map_vertex_ids(
    const IdType* const global, IdType* const new_global,
    const IdType num_vertices, const DeviceOrderedHashTable<IdType>& table) {
  assert(BLOCK_SIZE == blockDim.x);

  using Mapping = typename OrderedHashTable<IdType>::Mapping;

  const IdType tile_start = TILE_SIZE * blockIdx.x;
  const IdType tile_end = min(TILE_SIZE * (blockIdx.x + 1), num_vertices);

  for (IdType idx = threadIdx.x + tile_start; idx < tile_end;
       idx += BLOCK_SIZE) {
    const Mapping& mapping = *table.Search(global[idx]);
    new_global[idx] = mapping.local;
  }
}

/**
 * @brief Generate mapped edge endpoint ids.
 *
 * @tparam IdType The type of id.
 * @tparam BLOCK_SIZE The size of each thread block.
 * @tparam TILE_SIZE The number of edges to process per thread block.
 * @param global_srcs_device The source ids to map.
 * @param new_global_srcs_device The mapped source ids (output).
 * @param global_dsts_device The destination ids to map.
 * @param new_global_dsts_device The mapped destination ids (output).
 * @param num_edges The number of edges to map.
 * @param src_mapping The mapping of sources ids.
 * @param src_hash_size The the size of source id hash table/mapping.
 * @param dst_mapping The mapping of destination ids.
 * @param dst_hash_size The the size of destination id hash table/mapping.
 */
template <typename IdType, int BLOCK_SIZE, IdType TILE_SIZE>
__global__ void map_edge_ids(
    const IdType* const global_srcs_device,
    IdType* const new_global_srcs_device,
    const IdType* const global_dsts_device,
    IdType* const new_global_dsts_device, const IdType num_edges,
    DeviceOrderedHashTable<IdType> src_mapping,
    DeviceOrderedHashTable<IdType> dst_mapping) {
  assert(BLOCK_SIZE == blockDim.x);
  assert(2 == gridDim.y);

  if (blockIdx.y == 0) {
    map_vertex_ids<IdType, BLOCK_SIZE, TILE_SIZE>(
        global_srcs_device, new_global_srcs_device, num_edges, src_mapping);
  } else {
    map_vertex_ids<IdType, BLOCK_SIZE, TILE_SIZE>(
        global_dsts_device, new_global_dsts_device, num_edges, dst_mapping);
  }
}

/**
 * @brief Device level node maps for each node type.
 *
 * @param num_nodes Number of nodes per type.
 * @param offset When offset is set to 0, LhsHashTable is identical to
 *        RhsHashTable. Or set to num_nodes.size()/2 to use seperated
 *        LhsHashTable and RhsHashTable.
 * @param ctx The DGL context.
 * @param stream The stream to operate on.
 */
template <typename IdType>
class DeviceNodeMap {
 public:
  using Mapping = typename OrderedHashTable<IdType>::Mapping;

  DeviceNodeMap(
      const std::vector<int64_t>& num_nodes, const int64_t offset,
      DGLContext ctx, cudaStream_t stream)
      : num_types_(num_nodes.size()),
        rhs_offset_(offset),
        hash_tables_(),
        ctx_(ctx) {
    auto device = runtime::DeviceAPI::Get(ctx);

    hash_tables_.reserve(num_types_);
    for (int64_t i = 0; i < num_types_; ++i) {
      hash_tables_.emplace_back(
          new OrderedHashTable<IdType>(num_nodes[i], ctx_, stream));
    }
  }

  OrderedHashTable<IdType>& LhsHashTable(const size_t index) {
    return HashData(index);
  }

  OrderedHashTable<IdType>& RhsHashTable(const size_t index) {
    return HashData(index + rhs_offset_);
  }

  const OrderedHashTable<IdType>& LhsHashTable(const size_t index) const {
    return HashData(index);
  }

  const OrderedHashTable<IdType>& RhsHashTable(const size_t index) const {
    return HashData(index + rhs_offset_);
  }

  IdType LhsHashSize(const size_t index) const { return HashSize(index); }

  IdType RhsHashSize(const size_t index) const {
    return HashSize(rhs_offset_ + index);
  }

  size_t Size() const { return hash_tables_.size(); }

 private:
  int64_t num_types_;
  size_t rhs_offset_;
  std::vector<std::unique_ptr<OrderedHashTable<IdType>>> hash_tables_;
  DGLContext ctx_;

  inline OrderedHashTable<IdType>& HashData(const size_t index) {
    CHECK_LT(index, hash_tables_.size());
    return *hash_tables_[index];
  }

  inline const OrderedHashTable<IdType>& HashData(const size_t index) const {
    CHECK_LT(index, hash_tables_.size());
    return *hash_tables_[index];
  }

  inline IdType HashSize(const size_t index) const {
    return HashData(index).size();
  }
};

template <typename IdType>
inline size_t RoundUpDiv(const IdType num, const size_t divisor) {
  return static_cast<IdType>(num / divisor) + (num % divisor == 0 ? 0 : 1);
}

template <typename IdType>
inline IdType RoundUp(const IdType num, const size_t unit) {
  return RoundUpDiv(num, unit) * unit;
}

template <typename IdType>
std::tuple<std::vector<IdArray>, std::vector<IdArray>> MapEdges(
    HeteroGraphPtr graph, const std::vector<EdgeArray>& edge_sets,
    const DeviceNodeMap<IdType>& node_map, cudaStream_t stream) {
  constexpr const int BLOCK_SIZE = 128;
  constexpr const size_t TILE_SIZE = 1024;

  const auto& ctx = graph->Context();

  std::vector<IdArray> new_lhs;
  new_lhs.reserve(edge_sets.size());
  std::vector<IdArray> new_rhs;
  new_rhs.reserve(edge_sets.size());

  // The next peformance optimization here, is to perform mapping of all edge
  // types in a single kernel launch.
  const int64_t num_edge_sets = static_cast<int64_t>(edge_sets.size());
  for (int64_t etype = 0; etype < num_edge_sets; ++etype) {
    const EdgeArray& edges = edge_sets[etype];
    if (edges.id.defined() && edges.src->shape[0] > 0) {
      const int64_t num_edges = edges.src->shape[0];

      new_lhs.emplace_back(NewIdArray(num_edges, ctx, sizeof(IdType) * 8));
      new_rhs.emplace_back(NewIdArray(num_edges, ctx, sizeof(IdType) * 8));

      const auto src_dst_types = graph->GetEndpointTypes(etype);
      const int src_type = src_dst_types.first;
      const int dst_type = src_dst_types.second;

      const dim3 grid(RoundUpDiv(num_edges, TILE_SIZE), 2);
      const dim3 block(BLOCK_SIZE);

      // map the srcs
      CUDA_KERNEL_CALL(
          (map_edge_ids<IdType, BLOCK_SIZE, TILE_SIZE>), grid, block, 0, stream,
          edges.src.Ptr<IdType>(), new_lhs.back().Ptr<IdType>(),
          edges.dst.Ptr<IdType>(), new_rhs.back().Ptr<IdType>(), num_edges,
          node_map.LhsHashTable(src_type).DeviceHandle(),
          node_map.RhsHashTable(dst_type).DeviceHandle());
    } else {
      new_lhs.emplace_back(
          aten::NullArray(DGLDataType{kDGLInt, sizeof(IdType) * 8, 1}, ctx));
      new_rhs.emplace_back(
          aten::NullArray(DGLDataType{kDGLInt, sizeof(IdType) * 8, 1}, ctx));
    }
  }

  return std::tuple<std::vector<IdArray>, std::vector<IdArray>>(
      std::move(new_lhs), std::move(new_rhs));
}

}  // namespace cuda
}  // namespace transform
}  // namespace dgl

#endif  // DGL_GRAPH_TRANSFORM_CUDA_CUDA_MAP_EDGES_CUH_
