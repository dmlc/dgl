/**
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
 * @file graph/transform/cuda/cuda_compact_graph.cu
 * @brief Functions to find and eliminate the common isolated nodes across
 * all given graphs with the same set of nodes.
 */

#include <cuda_runtime.h>
#include <dgl/immutable_graph.h>
#include <dgl/runtime/device_api.h>

#include <algorithm>
#include <memory>
#include <utility>

#include "../../../runtime/cuda/cuda_common.h"
#include "../../heterograph.h"
#include "../compact.h"
#include "cuda_map_edges.cuh"

using namespace dgl::aten;
using namespace dgl::runtime::cuda;
using namespace dgl::transform::cuda;

namespace dgl {
namespace transform {

namespace {

/**
 * @brief This function builds node maps for each node type, preserving the
 * order of the input nodes. Here it is assumed the nodes are not unique,
 * and thus a unique list is generated.
 *
 * @param input_nodes The set of input nodes.
 * @param node_maps The node maps to be constructed.
 * @param count_unique_device The number of unique nodes (on the GPU).
 * @param unique_nodes_device The unique nodes (on the GPU).
 * @param stream The stream to operate on.
 */
template <typename IdType>
void BuildNodeMaps(
    const std::vector<IdArray> &input_nodes,
    DeviceNodeMap<IdType> *const node_maps, int64_t *const count_unique_device,
    std::vector<IdArray> *const unique_nodes_device, cudaStream_t stream) {
  const int64_t num_ntypes = static_cast<int64_t>(input_nodes.size());

  CUDA_CALL(cudaMemsetAsync(
      count_unique_device, 0, num_ntypes * sizeof(*count_unique_device),
      stream));

  // possibly duplicated nodes
  for (int64_t ntype = 0; ntype < num_ntypes; ++ntype) {
    const IdArray &nodes = input_nodes[ntype];
    if (nodes->shape[0] > 0) {
      CHECK_EQ(nodes->ctx.device_type, kDGLCUDA);
      node_maps->LhsHashTable(ntype).FillWithDuplicates(
          nodes.Ptr<IdType>(), nodes->shape[0],
          (*unique_nodes_device)[ntype].Ptr<IdType>(),
          count_unique_device + ntype, stream);
    }
  }
}

template <typename IdType>
std::pair<std::vector<HeteroGraphPtr>, std::vector<IdArray>> CompactGraphsGPU(
    const std::vector<HeteroGraphPtr> &graphs,
    const std::vector<IdArray> &always_preserve) {
  const auto &ctx = graphs[0]->Context();
  auto device = runtime::DeviceAPI::Get(ctx);
  cudaStream_t stream = runtime::getCurrentCUDAStream();

  CHECK_EQ(ctx.device_type, kDGLCUDA);

  // Step 1: Collect the nodes that has connections for each type.
  const uint64_t num_ntypes = graphs[0]->NumVertexTypes();
  std::vector<std::vector<EdgeArray>> all_edges(
      graphs.size());  // all_edges[i][etype]

  // count the number of nodes per type
  std::vector<int64_t> max_vertex_cnt(num_ntypes, 0);
  for (size_t i = 0; i < graphs.size(); ++i) {
    const HeteroGraphPtr curr_graph = graphs[i];
    const int64_t num_etypes = curr_graph->NumEdgeTypes();

    for (IdType etype = 0; etype < num_etypes; ++etype) {
      IdType srctype, dsttype;
      std::tie(srctype, dsttype) = curr_graph->GetEndpointTypes(etype);

      const int64_t n_edges = curr_graph->NumEdges(etype);
      max_vertex_cnt[srctype] += n_edges;
      max_vertex_cnt[dsttype] += n_edges;
    }
  }

  for (size_t i = 0; i < always_preserve.size(); ++i) {
    max_vertex_cnt[i] += always_preserve[i]->shape[0];
  }

  // gather all nodes
  std::vector<IdArray> all_nodes(num_ntypes);
  std::vector<int64_t> node_offsets(num_ntypes, 0);

  for (uint64_t ntype = 0; ntype < num_ntypes; ++ntype) {
    all_nodes[ntype] =
        NewIdArray(max_vertex_cnt[ntype], ctx, sizeof(IdType) * 8);
    // copy the nodes in always_preserve
    if (ntype < always_preserve.size() &&
        always_preserve[ntype]->shape[0] > 0) {
      device->CopyDataFromTo(
          always_preserve[ntype].Ptr<IdType>(), 0,
          all_nodes[ntype].Ptr<IdType>(), node_offsets[ntype],
          sizeof(IdType) * always_preserve[ntype]->shape[0],
          always_preserve[ntype]->ctx, all_nodes[ntype]->ctx,
          always_preserve[ntype]->dtype);
      node_offsets[ntype] += sizeof(IdType) * always_preserve[ntype]->shape[0];
    }
  }

  for (size_t i = 0; i < graphs.size(); ++i) {
    const HeteroGraphPtr curr_graph = graphs[i];
    const int64_t num_etypes = curr_graph->NumEdgeTypes();

    all_edges[i].reserve(num_etypes);
    for (int64_t etype = 0; etype < num_etypes; ++etype) {
      dgl_type_t srctype, dsttype;
      std::tie(srctype, dsttype) = curr_graph->GetEndpointTypes(etype);

      const EdgeArray edges = curr_graph->Edges(etype, "eid");

      if (edges.src.defined()) {
        device->CopyDataFromTo(
            edges.src.Ptr<IdType>(), 0, all_nodes[srctype].Ptr<IdType>(),
            node_offsets[srctype], sizeof(IdType) * edges.src->shape[0],
            edges.src->ctx, all_nodes[srctype]->ctx, edges.src->dtype);
        node_offsets[srctype] += sizeof(IdType) * edges.src->shape[0];
      }
      if (edges.dst.defined()) {
        device->CopyDataFromTo(
            edges.dst.Ptr<IdType>(), 0, all_nodes[dsttype].Ptr<IdType>(),
            node_offsets[dsttype], sizeof(IdType) * edges.dst->shape[0],
            edges.dst->ctx, all_nodes[dsttype]->ctx, edges.dst->dtype);
        node_offsets[dsttype] += sizeof(IdType) * edges.dst->shape[0];
      }
      all_edges[i].push_back(edges);
    }
  }

  // Step 2: Relabel the nodes for each type to a smaller ID space
  //         using BuildNodeMaps

  // allocate space for map creation
  // the hashmap on GPU
  DeviceNodeMap<IdType> node_maps(max_vertex_cnt, 0, ctx, stream);
  // number of unique nodes per type on CPU
  std::vector<int64_t> num_induced_nodes(num_ntypes);
  // number of unique nodes per type on GPU
  int64_t *count_unique_device = static_cast<int64_t *>(
      device->AllocWorkspace(ctx, sizeof(int64_t) * num_ntypes));
  // the set of unique nodes per type
  std::vector<IdArray> induced_nodes(num_ntypes);
  for (uint64_t ntype = 0; ntype < num_ntypes; ++ntype) {
    induced_nodes[ntype] =
        NewIdArray(max_vertex_cnt[ntype], ctx, sizeof(IdType) * 8);
  }

  BuildNodeMaps(
      all_nodes, &node_maps, count_unique_device, &induced_nodes, stream);

  device->CopyDataFromTo(
      count_unique_device, 0, num_induced_nodes.data(), 0,
      sizeof(*num_induced_nodes.data()) * num_ntypes, ctx,
      DGLContext{kDGLCPU, 0}, DGLDataType{kDGLInt, 64, 1});
  device->StreamSync(ctx, stream);

  // wait for the node counts to finish transferring
  device->FreeWorkspace(ctx, count_unique_device);

  // resize induced nodes
  for (uint64_t ntype = 0; ntype < num_ntypes; ++ntype) {
    induced_nodes[ntype]->shape[0] = num_induced_nodes[ntype];
  }

  // Step 3: Remap the edges of each graph using MapEdges
  std::vector<HeteroGraphPtr> new_graphs;
  for (size_t i = 0; i < graphs.size(); ++i) {
    const HeteroGraphPtr curr_graph = graphs[i];
    const auto meta_graph = curr_graph->meta_graph();
    const int64_t num_etypes = curr_graph->NumEdgeTypes();

    std::vector<HeteroGraphPtr> rel_graphs;
    rel_graphs.reserve(num_etypes);

    std::vector<IdArray> new_src;
    std::vector<IdArray> new_dst;
    std::tie(new_src, new_dst) =
        MapEdges(curr_graph, all_edges[i], node_maps, stream);

    for (IdType etype = 0; etype < num_etypes; ++etype) {
      IdType srctype, dsttype;
      std::tie(srctype, dsttype) = curr_graph->GetEndpointTypes(etype);

      rel_graphs.push_back(UnitGraph::CreateFromCOO(
          srctype == dsttype ? 1 : 2, induced_nodes[srctype]->shape[0],
          induced_nodes[dsttype]->shape[0], new_src[etype], new_dst[etype]));
    }

    new_graphs.push_back(
        CreateHeteroGraph(meta_graph, rel_graphs, num_induced_nodes));
  }

  return std::make_pair(new_graphs, induced_nodes);
}

}  // namespace

template <>
std::pair<std::vector<HeteroGraphPtr>, std::vector<IdArray>>
CompactGraphs<kDGLCUDA, int32_t>(
    const std::vector<HeteroGraphPtr> &graphs,
    const std::vector<IdArray> &always_preserve) {
  return CompactGraphsGPU<int32_t>(graphs, always_preserve);
}

template <>
std::pair<std::vector<HeteroGraphPtr>, std::vector<IdArray>>
CompactGraphs<kDGLCUDA, int64_t>(
    const std::vector<HeteroGraphPtr> &graphs,
    const std::vector<IdArray> &always_preserve) {
  return CompactGraphsGPU<int64_t>(graphs, always_preserve);
}

}  // namespace transform
}  // namespace dgl
