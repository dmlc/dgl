/*!
 *  Copyright 2021 Contributors
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
 * \file graph/transform/cuda_compact_graph.cu
 * \brief Functions to find and eliminate the common isolated nodes across
 * all given graphs with the same set of nodes.
 */


#include <dgl/runtime/device_api.h>
#include <dgl/immutable_graph.h>
#include <cuda_runtime.h>
#include <utility>
#include <algorithm>
#include <memory>

#include "../../../runtime/cuda/cuda_common.h"
#include "../../../runtime/cuda/cuda_hashtable.cuh"
#include "../../heterograph.h"
#include "../compact.h"

using namespace dgl::aten;
using namespace dgl::runtime::cuda;

namespace dgl {
namespace transform {

namespace {

template<typename IdType, int BLOCK_SIZE, IdType TILE_SIZE>
__device__ void map_vertex_ids(
    const IdType * const global,
    IdType * const new_global,
    const IdType num_vertices,
    const DeviceOrderedHashTable<IdType>& table) {
  assert(BLOCK_SIZE == blockDim.x);

  using Mapping = typename OrderedHashTable<IdType>::Mapping;

  const IdType tile_start = TILE_SIZE*blockIdx.x;
  const IdType tile_end = min(TILE_SIZE*(blockIdx.x+1), num_vertices);

  for (IdType idx = threadIdx.x+tile_start; idx < tile_end; idx+=BLOCK_SIZE) {
    const Mapping& mapping = *table.Search(global[idx]);
    new_global[idx] = mapping.local;
  }
}

/**
* \brief Generate mapped edge endpoint ids.
*
* \tparam IdType The type of id.
* \tparam BLOCK_SIZE The size of each thread block.
* \tparam TILE_SIZE The number of edges to process per thread block.
* \param global_srcs_device The source ids to map.
* \param new_global_srcs_device The mapped source ids (output).
* \param global_dsts_device The destination ids to map.
* \param new_global_dsts_device The mapped destination ids (output).
* \param num_edges The number of edges to map.
* \param src_mapping The mapping of sources ids.
* \param src_hash_size The the size of source id hash table/mapping.
* \param dst_mapping The mapping of destination ids.
* \param dst_hash_size The the size of destination id hash table/mapping.
*/
template<typename IdType, int BLOCK_SIZE, IdType TILE_SIZE>
__global__ void map_edge_ids(
    const IdType * const global_srcs_device,
    IdType * const new_global_srcs_device,
    const IdType * const global_dsts_device,
    IdType * const new_global_dsts_device,
    const IdType num_edges,
    DeviceOrderedHashTable<IdType> src_mapping,
    DeviceOrderedHashTable<IdType> dst_mapping) {
  assert(BLOCK_SIZE == blockDim.x);
  assert(2 == gridDim.y);

  if (blockIdx.y == 0) {
    map_vertex_ids<IdType, BLOCK_SIZE, TILE_SIZE>(
        global_srcs_device,
        new_global_srcs_device,
        num_edges,
        src_mapping);
  } else {
    map_vertex_ids<IdType, BLOCK_SIZE, TILE_SIZE>(
        global_dsts_device,
        new_global_dsts_device,
        num_edges,
        dst_mapping);
  }
}

template<typename IdType>
inline size_t RoundUpDiv(
    const IdType num,
    const size_t divisor) {
  return static_cast<IdType>(num/divisor) + (num % divisor == 0 ? 0 : 1);
}


template<typename IdType>
inline IdType RoundUp(
    const IdType num,
    const size_t unit) {
  return RoundUpDiv(num, unit)*unit;
}


template<typename IdType>
class DeviceNodeMap {
 public:
  using Mapping = typename OrderedHashTable<IdType>::Mapping;

  DeviceNodeMap(
      const std::vector<int64_t>& num_nodes,
      DGLContext ctx,
      cudaStream_t stream) :
    num_types_(num_nodes.size()),
    workspaces_(),
    hash_tables_(),
    ctx_(ctx) {
    auto device = runtime::DeviceAPI::Get(ctx);

    hash_tables_.reserve(num_types_);
    workspaces_.reserve(num_types_);
    for (int64_t i = 0; i < num_types_; ++i) {
      hash_tables_.emplace_back(
          new OrderedHashTable<IdType>(
            num_nodes[i],
            ctx_,
            stream));
    }
  }

  OrderedHashTable<IdType>& HashTable(
      const size_t index) {
    return HashData(index);
  }

  const OrderedHashTable<IdType>& HashTable(
      const size_t index) const {
    return HashData(index);
  }

  IdType HashTableSize(
      const size_t index) const {
    return HashSize(index);
  }

  size_t Size() const {
    return hash_tables_.size();
  }

 private:
  int64_t num_types_;
  std::vector<void*> workspaces_;
  std::vector<std::unique_ptr<OrderedHashTable<IdType>>> hash_tables_;
  DGLContext ctx_;

  inline OrderedHashTable<IdType>& HashData(
      const size_t index) {
    CHECK_LT(index, hash_tables_.size());
    return *hash_tables_[index];
  }

  inline const OrderedHashTable<IdType>& HashData(
      const size_t index) const {
    CHECK_LT(index, hash_tables_.size());
    return *hash_tables_[index];
  }

  inline IdType HashSize(
      const size_t index) const {
    return HashData(index).size();
  }
};

/**
* \brief This function builds node maps for each node type, preserving the
* order of the input nodes. Here it is assumed the nodes are not unique,
* and thus a unique list is generated.
*
* \param input_nodes The set of input nodes.
* \param node_maps The node maps to be constructed.
* \param count_unique_device The number of unique nodes (on the GPU).
* \param unique_nodes_device The unique nodes (on the GPU).
* \param stream The stream to operate on.
*/
void BuildNodeMaps(
    const std::vector<IdArray>& input_nodes,
    DeviceNodeMap<IdType> * const node_maps,
    int64_t * const count_unique_device,
    std::vector<IdArray>* const unique_nodes_device,
    cudaStream_t stream) {
  const int64_t num_ntypes = static_cast<int64_t>(input_nodes.size());

  CUDA_CALL(cudaMemsetAsync(
    count_unique_device,
    0,
    num_ntypes*sizeof(*count_unique_device),
    stream));

  // possibly duplicated nodes
  for (int64_t ntype = 0; ntype < num_ntypes; ++ntype) {
    const IdArray& nodes = input_nodes[ntype];
    if (nodes->shape[0] > 0) {
      CHECK_EQ(nodes->ctx.device_type, kDLGPU);
      node_maps->HashTable(ntype).FillWithDuplicates(
          nodes.Ptr<IdType>(),
          nodes->shape[0],
          (*unique_nodes_device)[ntype].Ptr<IdType>(),
          count_unique_device+ntype,
          stream);
    }
  }
}


template<typename IdType>
std::tuple<std::vector<IdArray>, std::vector<IdArray>>
MapEdges(
    HeteroGraphPtr graph,
    const std::vector<EdgeArray>& edge_sets,
    const DeviceNodeMap<IdType>& node_map,
    cudaStream_t stream) {
  constexpr const int BLOCK_SIZE = 128;
  constexpr const size_t TILE_SIZE = 1024;

  const auto& ctx = graph->Context();

  // should map the edges of each graph in parallel or
  // call this function multiple times

  // std::vector<IdArray> new_lhs;
  // new_lhs.reserve(edge_sets.size());
  // std::vector<IdArray> new_rhs;
  // new_rhs.reserve(edge_sets.size());

  // The next peformance optimization here, is to perform mapping of all edge
  // types in a single kernel launch.
  const int64_t num_edge_sets = static_cast<int64_t>(edge_sets.size());
  for (int64_t etype = 0; etype < num_edge_sets; ++etype) {
    const EdgeArray& edges = edge_sets[etype];
    if (edges.id.defined() && edges.src->shape[0] > 0) {
      const int64_t num_edges = edges.src->shape[0];

      // new_lhs.emplace_back(NewIdArray(num_edges, ctx, sizeof(IdType)*8));
      // new_rhs.emplace_back(NewIdArray(num_edges, ctx, sizeof(IdType)*8));

      const auto src_dst_types = graph->GetEndpointTypes(etype);
      const int src_type = src_dst_types.first;
      const int dst_type = src_dst_types.second;

      const dim3 grid(RoundUpDiv(num_edges, TILE_SIZE), 2);
      const dim3 block(BLOCK_SIZE);

      // map the srcs
      map_edge_ids<IdType, BLOCK_SIZE, TILE_SIZE><<<
        grid,
        block,
        0,
        stream>>>(
        edges.src.Ptr<IdType>(),
        new_lhs.back().Ptr<IdType>(),
        edges.dst.Ptr<IdType>(),
        new_rhs.back().Ptr<IdType>(),
        num_edges,
        node_map.LhsHashTable(src_type).DeviceHandle(),
        node_map.RhsHashTable(dst_type).DeviceHandle());
      CUDA_CALL(cudaGetLastError());
    } else {
      new_lhs.emplace_back(
          aten::NullArray(DLDataType{kDLInt, sizeof(IdType)*8, 1}, ctx));
      new_rhs.emplace_back(
          aten::NullArray(DLDataType{kDLInt, sizeof(IdType)*8, 1}, ctx));
    }
  }

  return std::tuple<std::vector<IdArray>, std::vector<IdArray>>(
      std::move(new_lhs), std::move(new_rhs));
}


template<typename IdType>
std::pair<std::vector<HeteroGraphPtr>, std::vector<IdArray>>
CompactGraphsGPU(
    const std::vector<HeteroGraphPtr> &graphs,
    const std::vector<IdArray> &always_preserve) {

  cudaStream_t stream = 0;
  const auto& ctx = graphs[0]->Context();
  auto device = runtime::DeviceAPI::Get(ctx);

  CHECK_EQ(ctx.device_type, kDLGPU);

  // Step 1: Collect the nodes that has connections for each type.

  // Step 2: Relabel the nodes for each type to a smaller ID space
  //         using BuildNodeMap

  // Step 3: Remap the edges of each graph using MapEdges



};

}  // namespace

template<>
std::pair<std::vector<HeteroGraphPtr>, std::vector<IdArray>>
CompactGraphs<kDLGPU, int32_t>(
    const std::vector<HeteroGraphPtr> &graphs,
    const std::vector<IdArray> &always_preserve) {
  return CompactGraphsGPU<int32_t>(graphs, always_preserve);
}

template<>
std::pair<std::vector<HeteroGraphPtr>, std::vector<IdArray>>
CompactGraphs<kDLGPU, int64_t>(
    const std::vector<HeteroGraphPtr> &graphs,
    const std::vector<IdArray> &always_preserve) {
  return CompactGraphsGPU<int64_t>(graphs, always_preserve);
}

}  // namespace transform
}  // namespace dgl
