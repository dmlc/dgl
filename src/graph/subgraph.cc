/*!
 *  Copyright (c) 2020 by Contributors
 * \file graph/subgraph.cc
 * \brief Functions for extracting subgraphs.
 */
#include "./heterograph.h"
using namespace dgl::runtime;

namespace dgl {

HeteroSubgraph InEdgeGraph(
    const HeteroGraphPtr graph, const std::vector<IdArray>& vids, bool relabel_nodes) {
  CHECK_EQ(vids.size(), graph->NumVertexTypes())
    << "Invalid input: the input list size must be the same as the number of vertex types.";
  std::vector<IdArray> induced_edges(graph->NumEdgeTypes());
  for (dgl_type_t etype = 0; etype < graph->NumEdgeTypes(); ++etype) {
    auto pair = graph->meta_graph()->FindEdge(etype);
    const dgl_type_t dst_vtype = pair.second;
    if (aten::IsNullArray(vids[dst_vtype])) {
      induced_edges[etype] = IdArray::Empty({0}, graph->DataType(), graph->Context());
    } else {
      const auto& earr = graph->InEdges(etype, {vids[dst_vtype]});
      induced_edges[etype] = earr.id;
    }
  }
  return graph->EdgeSubgraph(induced_edges, !relabel_nodes);
}

HeteroSubgraph OutEdgeGraph(
    const HeteroGraphPtr graph, const std::vector<IdArray>& vids, bool relabel_nodes) {
  CHECK_EQ(vids.size(), graph->NumVertexTypes())
    << "Invalid input: the input list size must be the same as the number of vertex types.";
  std::vector<IdArray> induced_edges(graph->NumEdgeTypes());
  for (dgl_type_t etype = 0; etype < graph->NumEdgeTypes(); ++etype) {
    auto pair = graph->meta_graph()->FindEdge(etype);
    const dgl_type_t src_vtype = pair.first;
    if (aten::IsNullArray(vids[src_vtype])) {
      induced_edges[etype] = IdArray::Empty({0}, graph->DataType(), graph->Context());
    } else {
      induced_edges[etype] = earr.id;
    }
  }
  return graph->EdgeSubgraph(induced_edges, !relabel_nodes);
}

}  // namespace dgl
