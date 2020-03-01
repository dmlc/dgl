/*!
 *  Copyright (c) 2019 by Contributors
 * \file graph/transform/to_bipartite.cc
 * \brief Convert a graph to a bipartite-structured graph.
 */

#include <dgl/base_heterograph.h>
#include <dgl/transform.h>
#include <dgl/array.h>
#include <dgl/packed_func_ext.h>
#include <dgl/immutable_graph.h>
#include <dgl/runtime/registry.h>
#include <dgl/runtime/container.h>
#include <vector>
#include <tuple>
// TODO(BarclayII): currently ToBipartite depend on IdHashMap<IdType> implementation which
// only works on CPU.  Should fix later to make it device agnostic.
#include "../../array/cpu/array_utils.h"

namespace dgl {

using namespace dgl::runtime;
using namespace dgl::aten;

namespace transform {

namespace {

template<typename IdType>
std::tuple<HeteroGraphPtr, std::vector<IdArray>, std::vector<IdArray>>
ToBipartite(
    HeteroGraphPtr graph,
    const std::vector<IdArray> &rhs_nodes,
    bool include_rhs_in_lhs) {
  std::vector<IdHashMap<IdType>> rhs_node_mappings, lhs_node_mappings;
  const int64_t num_etypes = graph->NumEdgeTypes();
  const int64_t num_ntypes = graph->NumVertexTypes();
  std::vector<EdgeArray> edge_arrays;

  if (rhs_nodes.size() > 0) {
    for (const IdArray &nodes : rhs_nodes)
      rhs_node_mappings.emplace_back(nodes);

    if (include_rhs_in_lhs)
      lhs_node_mappings = rhs_node_mappings;    // copy

    for (int64_t etype = 0; etype < num_etypes; ++etype) {
      const auto src_dst_types = graph->GetEndpointTypes(etype);
      const dgl_type_t srctype = src_dst_types.first;
      const dgl_type_t dsttype = src_dst_types.second;
      const EdgeArray edges = graph->InEdges(etype, rhs_nodes[dsttype]);
      lhs_node_mappings[srctype].Update(edges.src);
      edge_arrays.push_back(edges);
    }
  } else {
    rhs_node_mappings.resize(num_ntypes);
    lhs_node_mappings.resize(num_ntypes);

    for (int64_t etype = 0; etype < num_etypes; ++etype) {
      const auto srctype = graph->GetEndpointTypes(etype).first;
      const EdgeArray edges = graph->Edges(etype, "eid");
      lhs_node_mappings[srctype].Update(edges.src);
      edge_arrays.push_back(edges);
    }
  }

  const auto meta_graph = graph->meta_graph();
  const EdgeArray etypes = meta_graph->Edges("eid");
  const IdArray new_dst = Add(etypes.dst, num_ntypes);
  const auto new_meta_graph = ImmutableGraph::CreateFromCOO(
      num_ntypes * 2, etypes.src, etypes.dst);

  std::vector<HeteroGraphPtr> rel_graphs;
  std::vector<IdArray> induced_edges;
  for (int64_t etype = 0; etype < num_etypes; ++etype) {
    const auto src_dst_types = graph->GetEndpointTypes(etype);
    const dgl_type_t srctype = src_dst_types.first;
    const dgl_type_t dsttype = src_dst_types.second;
    const IdHashMap<IdType> &lhs_map = lhs_node_mappings[srctype];
    const IdHashMap<IdType> &rhs_map = rhs_node_mappings[dsttype];
    rel_graphs.push_back(CreateFromCOO(
        2, lhs_map.Size(), rhs_map.Size(),
        lhs_map.Map(edge_arrays[etype].src, -1),
	rhs_map.Map(edge_arrays[etype].dst, -1)));
    induced_edges.push_back(edge_arrays[etype].id);
  }

  const HeteroGraphPtr new_graph = CreateHeteroGraph(new_meta_graph, rel_graphs);
  std::vector<IdArray> lhs_nodes;
  for (const IdHashMap<IdType> &lhs_map : lhs_node_mappings)
    lhs_nodes.push_back(lhs_map.Values());
  return std::make_tuple(new_graph, lhs_nodes, induced_edges);
}

};  // namespace

std::tuple<HeteroGraphPtr, std::vector<IdArray>, std::vector<IdArray>>
ToBipartite(
    HeteroGraphPtr graph,
    const std::vector<IdArray> &rhs_nodes,
    bool include_rhs_in_lhs) {
  std::tuple<HeteroGraphPtr, std::vector<IdArray>, std::vector<IdArray>> ret;
  ATEN_ID_TYPE_SWITCH(graph->DataType(), IdType, {
    ret = ToBipartite<IdType>(graph, rhs_nodes, include_rhs_in_lhs);
  });
  return ret;
}

DGL_REGISTER_GLOBAL("transform._CAPI_DGLToBipartite")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
    const HeteroGraphRef graph_ref = args[0];
    const std::vector<IdArray> &rhs_nodes = ListValueToVector<IdArray>(args[1]);
    const bool include_rhs_in_lhs = args[2];

    HeteroGraphPtr new_graph;
    std::vector<IdArray> lhs_nodes;
    std::vector<IdArray> induced_edges;
    std::tie(new_graph, lhs_nodes, induced_edges) = ToBipartite(
        graph_ref.sptr(), rhs_nodes, include_rhs_in_lhs);

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
