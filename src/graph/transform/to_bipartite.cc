/*!
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
 * \file graph/transform/to_bipartite.cc
 * \brief Convert a graph to a bipartite-structured graph.
 */

#include "to_bipartite.h"

#include <dgl/array.h>
#include <dgl/base_heterograph.h>
#include <dgl/immutable_graph.h>
#include <dgl/packed_func_ext.h>
#include <dgl/runtime/container.h>
#include <dgl/runtime/registry.h>
#include <dgl/transform.h>

#include <tuple>
#include <utility>
#include <vector>

#include "../../array/cpu/array_utils.h"

namespace dgl {

using namespace dgl::runtime;
using namespace dgl::aten;

namespace transform {

namespace {

// Since partial specialization is not allowed for functions, use this as an
// intermediate for ToBlock where XPU = kDGLCPU.
template <typename IdType>
std::tuple<HeteroGraphPtr, std::vector<IdArray>> ToBlockCPU(
    HeteroGraphPtr graph, const std::vector<IdArray> &rhs_nodes,
    bool include_rhs_in_lhs, std::vector<IdArray> *const lhs_nodes_ptr) {
  std::vector<IdArray> &lhs_nodes = *lhs_nodes_ptr;
  const bool generate_lhs_nodes = lhs_nodes.empty();

  const int64_t num_etypes = graph->NumEdgeTypes();
  const int64_t num_ntypes = graph->NumVertexTypes();
  std::vector<EdgeArray> edge_arrays(num_etypes);

  CHECK(rhs_nodes.size() == static_cast<size_t>(num_ntypes))
      << "rhs_nodes not given for every node type";

  const std::vector<IdHashMap<IdType>> rhs_node_mappings(
      rhs_nodes.begin(), rhs_nodes.end());
  std::vector<IdHashMap<IdType>> lhs_node_mappings;

  if (generate_lhs_nodes) {
    // build lhs_node_mappings -- if we don't have them already
    if (include_rhs_in_lhs)
      lhs_node_mappings = rhs_node_mappings;  // copy
    else
      lhs_node_mappings.resize(num_ntypes);
  } else {
    lhs_node_mappings =
        std::vector<IdHashMap<IdType>>(lhs_nodes.begin(), lhs_nodes.end());
  }

  for (int64_t etype = 0; etype < num_etypes; ++etype) {
    const auto src_dst_types = graph->GetEndpointTypes(etype);
    const dgl_type_t srctype = src_dst_types.first;
    const dgl_type_t dsttype = src_dst_types.second;
    if (!aten::IsNullArray(rhs_nodes[dsttype])) {
      const EdgeArray &edges = graph->Edges(etype);
      if (generate_lhs_nodes) {
        lhs_node_mappings[srctype].Update(edges.src);
      }
      edge_arrays[etype] = edges;
    }
  }

  std::vector<int64_t> num_nodes_per_type;
  num_nodes_per_type.reserve(2 * num_ntypes);

  const auto meta_graph = graph->meta_graph();
  const EdgeArray etypes = meta_graph->Edges("eid");
  const IdArray new_dst = Add(etypes.dst, num_ntypes);
  const auto new_meta_graph =
      ImmutableGraph::CreateFromCOO(num_ntypes * 2, etypes.src, new_dst);

  for (int64_t ntype = 0; ntype < num_ntypes; ++ntype)
    num_nodes_per_type.push_back(lhs_node_mappings[ntype].Size());
  for (int64_t ntype = 0; ntype < num_ntypes; ++ntype)
    num_nodes_per_type.push_back(rhs_node_mappings[ntype].Size());

  std::vector<HeteroGraphPtr> rel_graphs;
  std::vector<IdArray> induced_edges;
  for (int64_t etype = 0; etype < num_etypes; ++etype) {
    const auto src_dst_types = graph->GetEndpointTypes(etype);
    const dgl_type_t srctype = src_dst_types.first;
    const dgl_type_t dsttype = src_dst_types.second;
    const IdHashMap<IdType> &lhs_map = lhs_node_mappings[srctype];
    const IdHashMap<IdType> &rhs_map = rhs_node_mappings[dsttype];
    if (rhs_map.Size() == 0) {
      // No rhs nodes are given for this edge type. Create an empty graph.
      rel_graphs.push_back(CreateFromCOO(
          2, lhs_map.Size(), rhs_map.Size(), aten::NullArray(),
          aten::NullArray()));
      induced_edges.push_back(aten::NullArray());
    } else {
      IdArray new_src = lhs_map.Map(edge_arrays[etype].src, -1);
      IdArray new_dst = rhs_map.Map(edge_arrays[etype].dst, -1);
      // Check whether there are unmapped IDs and raise error.
      for (int64_t i = 0; i < new_dst->shape[0]; ++i)
        CHECK_NE(new_dst.Ptr<IdType>()[i], -1)
            << "Node " << edge_arrays[etype].dst.Ptr<IdType>()[i]
            << " does not exist"
            << " in `rhs_nodes`. Argument `rhs_nodes` must contain all the edge"
            << " destination nodes.";
      rel_graphs.push_back(
          CreateFromCOO(2, lhs_map.Size(), rhs_map.Size(), new_src, new_dst));
      induced_edges.push_back(edge_arrays[etype].id);
    }
  }

  const HeteroGraphPtr new_graph =
      CreateHeteroGraph(new_meta_graph, rel_graphs, num_nodes_per_type);

  if (generate_lhs_nodes) {
    CHECK_EQ(lhs_nodes.size(), 0) << "InteralError: lhs_nodes should be empty "
                                     "when generating it.";
    for (const IdHashMap<IdType> &lhs_map : lhs_node_mappings)
      lhs_nodes.push_back(lhs_map.Values());
  }
  return std::make_tuple(new_graph, induced_edges);
}

}  // namespace

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

DGL_REGISTER_GLOBAL("transform._CAPI_DGLToBlock")
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
