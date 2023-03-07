/**
 *  Copyright 2019-2021 Contributors
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
 * @file graph/transform/to_block.cc
 * @brief Convert a graph to a bipartite-structured graph.
 *
 * Tested via python wrapper: python/dgl/path/to/to_block.py
 */

#include "to_block.h"

#include <dgl/array.h>
#include <dgl/base_heterograph.h>
#include <dgl/immutable_graph.h>
#include <dgl/packed_func_ext.h>
#include <dgl/runtime/container.h>
#include <dgl/runtime/device_api.h>
#include <dgl/runtime/registry.h>
#include <dgl/transform.h>

#include <tuple>
#include <utility>
#include <vector>

#include "../../array/cpu/concurrent_id_hash_map.h"

namespace dgl {

using namespace dgl::runtime;
using namespace dgl::aten;

namespace transform {

namespace {

template <typename IdType>
struct CPUIdsMapper {
  std::tuple<std::vector<IdArray>, std::vector<IdArray>> operator()(
      const HeteroGraphPtr &graph, bool include_rhs_in_lhs, int64_t num_ntypes,
      const DGLContext &ctx, const std::vector<int64_t> &max_nodes_per_type,
      const std::vector<EdgeArray> &edge_arrays,
      const std::vector<IdArray> &src_nodes,
      const std::vector<IdArray> &rhs_nodes,
      std::vector<IdArray> *const lhs_nodes_ptr,
      std::vector<int64_t> *const num_nodes_per_type_ptr) {
    std::vector<IdArray> &lhs_nodes = *lhs_nodes_ptr;
    std::vector<int64_t> &num_nodes_per_type = *num_nodes_per_type_ptr;

    const bool generate_lhs_nodes = lhs_nodes.empty();
    if (generate_lhs_nodes) {
      lhs_nodes.reserve(num_ntypes);
    }

    std::vector<ConcurrentIdHashMap<IdType>> lhs_nodes_map(num_ntypes);
    std::vector<ConcurrentIdHashMap<IdType>> rhs_nodes_map(num_ntypes);
    for (int64_t ntype = 0; ntype < num_ntypes; ++ntype) {
      IdArray unique_ids =
          aten::NullArray(DGLDataTypeTraits<IdType>::dtype, ctx);
      if (!aten::IsNullArray(src_nodes[ntype])) {
        auto num_seeds = include_rhs_in_lhs ? rhs_nodes[ntype]->shape[0] : 0;
        unique_ids = lhs_nodes_map[ntype].Init(src_nodes[ntype], num_seeds);
      }
      if (generate_lhs_nodes) {
        num_nodes_per_type[ntype] = unique_ids->shape[0];
        lhs_nodes.emplace_back(unique_ids);
      }
    }

    // Skip rhs mapping construction to save efforts when rhs is already
    // contained in lhs.
    if (!include_rhs_in_lhs) {
      for (int64_t ntype = 0; ntype < num_ntypes; ++ntype) {
        if (!aten::IsNullArray(rhs_nodes[ntype])) {
          rhs_nodes_map[ntype].Init(
              rhs_nodes[ntype], rhs_nodes[ntype]->shape[0]);
        }
      }
    }

    // Map node numberings from global to local, and build pointer for CSR.
    std::vector<IdArray> new_lhs;
    std::vector<IdArray> new_rhs;
    new_lhs.reserve(edge_arrays.size());
    new_rhs.reserve(edge_arrays.size());
    const int64_t num_etypes = static_cast<int64_t>(edge_arrays.size());
    for (int64_t etype = 0; etype < num_etypes; ++etype) {
      const EdgeArray &edges = edge_arrays[etype];
      if (edges.id.defined() && !aten::IsNullArray(edges.src)) {
        const auto src_dst_types = graph->GetEndpointTypes(etype);
        const int src_type = src_dst_types.first;
        const int dst_type = src_dst_types.second;
        new_lhs.emplace_back(lhs_nodes_map[src_type].MapIds(edges.src));
        if (include_rhs_in_lhs) {
          new_rhs.emplace_back(lhs_nodes_map[dst_type].MapIds(edges.dst));
        } else {
          new_rhs.emplace_back(rhs_nodes_map[dst_type].MapIds(edges.dst));
        }
      } else {
        new_lhs.emplace_back(
            aten::NullArray(DGLDataTypeTraits<IdType>::dtype, ctx));
        new_rhs.emplace_back(
            aten::NullArray(DGLDataTypeTraits<IdType>::dtype, ctx));
      }
    }
    return std::tuple<std::vector<IdArray>, std::vector<IdArray>>(
        std::move(new_lhs), std::move(new_rhs));
  }
};

// Since partial specialization is not allowed for functions, use this as an
// intermediate for ToBlock where XPU = kDGLCPU.
template <typename IdType>
std::tuple<HeteroGraphPtr, std::vector<IdArray>> ToBlockCPU(
    HeteroGraphPtr graph, const std::vector<IdArray> &rhs_nodes,
    bool include_rhs_in_lhs, std::vector<IdArray> *const lhs_nodes_ptr) {
  return dgl::transform::ProcessToBlock<IdType>(
      graph, rhs_nodes, include_rhs_in_lhs, lhs_nodes_ptr,
      CPUIdsMapper<IdType>());
}

}  // namespace

template <typename IdType>
std::tuple<HeteroGraphPtr, std::vector<IdArray>> ProcessToBlock(
    HeteroGraphPtr graph, const std::vector<IdArray> &rhs_nodes,
    bool include_rhs_in_lhs, std::vector<IdArray> *const lhs_nodes_ptr,
    IdsMapper &&ids_mapper) {
  std::vector<IdArray> &lhs_nodes = *lhs_nodes_ptr;
  const bool generate_lhs_nodes = lhs_nodes.empty();

  const auto &ctx = graph->Context();
  auto device = runtime::DeviceAPI::Get(ctx);

  // Since DST nodes are included in SRC nodes, a common requirement is to fetch
  // the DST node features from the SRC nodes features. To avoid expensive
  // sparse lookup, the function assures that the DST nodes in both SRC and DST
  // sets have the same ids. As a result, given the node feature tensor ``X`` of
  // type ``utype``, the following code finds the corresponding DST node
  // features of type ``vtype``:

  const int64_t num_etypes = graph->NumEdgeTypes();
  const int64_t num_ntypes = graph->NumVertexTypes();

  CHECK(rhs_nodes.size() == static_cast<size_t>(num_ntypes))
      << "rhs_nodes not given for every node type";

  std::vector<EdgeArray> edge_arrays(num_etypes);
  for (int64_t etype = 0; etype < num_etypes; ++etype) {
    const auto src_dst_types = graph->GetEndpointTypes(etype);
    const dgl_type_t dsttype = src_dst_types.second;
    if (!aten::IsNullArray(rhs_nodes[dsttype])) {
      edge_arrays[etype] = graph->Edges(etype);
    }
  }

  // Count lhs and rhs nodes.
  std::vector<int64_t> maxNodesPerType(num_ntypes * 2, 0);
  for (int64_t ntype = 0; ntype < num_ntypes; ++ntype) {
    maxNodesPerType[ntype + num_ntypes] += rhs_nodes[ntype]->shape[0];

    if (generate_lhs_nodes) {
      if (include_rhs_in_lhs) {
        maxNodesPerType[ntype] += rhs_nodes[ntype]->shape[0];
      }
    } else {
      maxNodesPerType[ntype] += lhs_nodes[ntype]->shape[0];
    }
  }
  if (generate_lhs_nodes) {
    // We don't have lhs_nodes, see we need to count inbound edges to get an
    // upper bound.
    for (int64_t etype = 0; etype < num_etypes; ++etype) {
      const auto src_dst_types = graph->GetEndpointTypes(etype);
      const dgl_type_t srctype = src_dst_types.first;
      if (edge_arrays[etype].src.defined()) {
        maxNodesPerType[srctype] += edge_arrays[etype].src->shape[0];
      }
    }
  }

  // Gather lhs_nodes.
  std::vector<IdArray> src_nodes(num_ntypes);
  if (generate_lhs_nodes) {
    std::vector<int64_t> src_node_offsets(num_ntypes, 0);
    for (int64_t ntype = 0; ntype < num_ntypes; ++ntype) {
      src_nodes[ntype] =
          NewIdArray(maxNodesPerType[ntype], ctx, sizeof(IdType) * 8);
      if (include_rhs_in_lhs) {
        // Place rhs nodes first.
        device->CopyDataFromTo(
            rhs_nodes[ntype].Ptr<IdType>(), 0, src_nodes[ntype].Ptr<IdType>(),
            src_node_offsets[ntype],
            sizeof(IdType) * rhs_nodes[ntype]->shape[0], rhs_nodes[ntype]->ctx,
            src_nodes[ntype]->ctx, rhs_nodes[ntype]->dtype);
        src_node_offsets[ntype] += sizeof(IdType) * rhs_nodes[ntype]->shape[0];
      }
    }
    for (int64_t etype = 0; etype < num_etypes; ++etype) {
      const auto src_dst_types = graph->GetEndpointTypes(etype);
      const dgl_type_t srctype = src_dst_types.first;
      if (edge_arrays[etype].src.defined()) {
        device->CopyDataFromTo(
            edge_arrays[etype].src.Ptr<IdType>(), 0,
            src_nodes[srctype].Ptr<IdType>(), src_node_offsets[srctype],
            sizeof(IdType) * edge_arrays[etype].src->shape[0],
            rhs_nodes[srctype]->ctx, src_nodes[srctype]->ctx,
            rhs_nodes[srctype]->dtype);

        src_node_offsets[srctype] +=
            sizeof(IdType) * edge_arrays[etype].src->shape[0];
      }
    }
  } else {
    for (int64_t ntype = 0; ntype < num_ntypes; ++ntype) {
      src_nodes[ntype] = lhs_nodes[ntype];
    }
  }

  std::vector<int64_t> num_nodes_per_type(num_ntypes * 2);
  // Populate RHS nodes from what we already know.
  for (int64_t ntype = 0; ntype < num_ntypes; ++ntype) {
    num_nodes_per_type[num_ntypes + ntype] = rhs_nodes[ntype]->shape[0];
  }

  std::vector<IdArray> new_lhs;
  std::vector<IdArray> new_rhs;
  std::tie(new_lhs, new_rhs) = ids_mapper(
      graph, include_rhs_in_lhs, num_ntypes, ctx, maxNodesPerType, edge_arrays,
      src_nodes, rhs_nodes, lhs_nodes_ptr, &num_nodes_per_type);

  std::vector<IdArray> induced_edges;
  induced_edges.reserve(num_etypes);
  for (int64_t etype = 0; etype < num_etypes; ++etype) {
    if (edge_arrays[etype].id.defined()) {
      induced_edges.push_back(edge_arrays[etype].id);
    } else {
      induced_edges.push_back(
          aten::NullArray(DGLDataType{kDGLInt, sizeof(IdType) * 8, 1}, ctx));
    }
  }

  // Build metagraph.
  const auto meta_graph = graph->meta_graph();
  const EdgeArray etypes = meta_graph->Edges("eid");
  const IdArray new_dst = Add(etypes.dst, num_ntypes);
  const auto new_meta_graph =
      ImmutableGraph::CreateFromCOO(num_ntypes * 2, etypes.src, new_dst);

  // Allocate vector for graph relations while GPU is busy.
  std::vector<HeteroGraphPtr> rel_graphs;
  rel_graphs.reserve(num_etypes);

  // Build the heterograph.
  for (int64_t etype = 0; etype < num_etypes; ++etype) {
    const auto src_dst_types = graph->GetEndpointTypes(etype);
    const dgl_type_t srctype = src_dst_types.first;
    const dgl_type_t dsttype = src_dst_types.second;

    if (rhs_nodes[dsttype]->shape[0] == 0) {
      // No rhs nodes are given for this edge type. Create an empty graph.
      rel_graphs.push_back(CreateFromCOO(
          2, lhs_nodes[srctype]->shape[0], rhs_nodes[dsttype]->shape[0],
          aten::NullArray(DGLDataType{kDGLInt, sizeof(IdType) * 8, 1}, ctx),
          aten::NullArray(DGLDataType{kDGLInt, sizeof(IdType) * 8, 1}, ctx)));
    } else {
      rel_graphs.push_back(CreateFromCOO(
          2, lhs_nodes[srctype]->shape[0], rhs_nodes[dsttype]->shape[0],
          new_lhs[etype], new_rhs[etype]));
    }
  }

  HeteroGraphPtr new_graph =
      CreateHeteroGraph(new_meta_graph, rel_graphs, num_nodes_per_type);

  // Return the new graph, the new src nodes, and new edges.
  return std::make_tuple(new_graph, induced_edges);
}

template std::tuple<HeteroGraphPtr, std::vector<IdArray>>
ProcessToBlock<int32_t>(
    HeteroGraphPtr graph, const std::vector<IdArray> &rhs_nodes,
    bool include_rhs_in_lhs, std::vector<IdArray> *const lhs_nodes_ptr,
    IdsMapper &&get_maping_ids);

template std::tuple<HeteroGraphPtr, std::vector<IdArray>>
ProcessToBlock<int64_t>(
    HeteroGraphPtr graph, const std::vector<IdArray> &rhs_nodes,
    bool include_rhs_in_lhs, std::vector<IdArray> *const lhs_nodes_ptr,
    IdsMapper &&get_maping_ids);

template <>
std::tuple<HeteroGraphPtr, std::vector<IdArray>> ToBlock<kDGLCPU, int32_t>(
    HeteroGraphPtr graph, const std::vector<IdArray> &rhs_nodes,
    bool include_rhs_in_lhs, std::vector<IdArray> *const lhs_nodes) {
  return ToBlockCPU<int32_t>(graph, rhs_nodes, include_rhs_in_lhs, lhs_nodes);
}

template <>
std::tuple<HeteroGraphPtr, std::vector<IdArray>> ToBlock<kDGLCPU, int64_t>(
    HeteroGraphPtr graph, const std::vector<IdArray> &rhs_nodes,
    bool include_rhs_in_lhs, std::vector<IdArray> *const lhs_nodes) {
  return ToBlockCPU<int64_t>(graph, rhs_nodes, include_rhs_in_lhs, lhs_nodes);
}

#ifdef DGL_USE_CUDA

// Forward declaration of GPU ToBlock implementations - actual implementation is
// in
// ./cuda/cuda_to_block.cu
// This is to get around the broken name mangling in VS2019 CL 16.5.5 +
// CUDA 11.3 which complains that the two template specializations have the same
// signature.
std::tuple<HeteroGraphPtr, std::vector<IdArray>> ToBlockGPU32(
    HeteroGraphPtr, const std::vector<IdArray> &, bool,
    std::vector<IdArray> *const);
std::tuple<HeteroGraphPtr, std::vector<IdArray>> ToBlockGPU64(
    HeteroGraphPtr, const std::vector<IdArray> &, bool,
    std::vector<IdArray> *const);

template <>
std::tuple<HeteroGraphPtr, std::vector<IdArray>> ToBlock<kDGLCUDA, int32_t>(
    HeteroGraphPtr graph, const std::vector<IdArray> &rhs_nodes,
    bool include_rhs_in_lhs, std::vector<IdArray> *const lhs_nodes) {
  return ToBlockGPU32(graph, rhs_nodes, include_rhs_in_lhs, lhs_nodes);
}

template <>
std::tuple<HeteroGraphPtr, std::vector<IdArray>> ToBlock<kDGLCUDA, int64_t>(
    HeteroGraphPtr graph, const std::vector<IdArray> &rhs_nodes,
    bool include_rhs_in_lhs, std::vector<IdArray> *const lhs_nodes) {
  return ToBlockGPU64(graph, rhs_nodes, include_rhs_in_lhs, lhs_nodes);
}

#endif  // DGL_USE_CUDA

DGL_REGISTER_GLOBAL("capi._CAPI_DGLToBlock")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      const HeteroGraphRef graph_ref = args[0];
      const std::vector<IdArray> &rhs_nodes =
          ListValueToVector<IdArray>(args[1]);
      const bool include_rhs_in_lhs = args[2];
      std::vector<IdArray> lhs_nodes = ListValueToVector<IdArray>(args[3]);

      HeteroGraphPtr new_graph;
      std::vector<IdArray> induced_edges;

      ATEN_XPU_SWITCH_CUDA(graph_ref->Context().device_type, XPU, "ToBlock", {
        ATEN_ID_TYPE_SWITCH(graph_ref->DataType(), IdType, {
          std::tie(new_graph, induced_edges) = ToBlock<XPU, IdType>(
              graph_ref.sptr(), rhs_nodes, include_rhs_in_lhs, &lhs_nodes);
        });
      });

      List<Value> lhs_nodes_ref;
      for (IdArray &array : lhs_nodes)
        lhs_nodes_ref.push_back(Value(MakeValue(array)));
      List<Value> induced_edges_ref;
      for (IdArray &array : induced_edges)
        induced_edges_ref.push_back(Value(MakeValue(array)));

      List<ObjectRef> ret;
      ret.push_back(HeteroGraphRef(new_graph));
      ret.push_back(lhs_nodes_ref);
      ret.push_back(induced_edges_ref);

      *rv = ret;
    });

};  // namespace transform

};  // namespace dgl
