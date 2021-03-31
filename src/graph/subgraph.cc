/*!
 *  Copyright (c) 2020 by Contributors
 * \file graph/subgraph.cc
 * \brief Functions for extracting subgraphs.
 */
#include "./heterograph.h"
using namespace dgl::runtime;

namespace dgl {

HeteroSubgraph InEdgeGraph(const HeteroGraphPtr graph, const std::vector<IdArray>& vids) {
  CHECK_EQ(vids.size(), graph->NumVertexTypes())
    << "Invalid input: the input list size must be the same as the number of vertex types.";
  std::vector<HeteroGraphPtr> subrels(graph->NumEdgeTypes());
  std::vector<IdArray> induced_edges(graph->NumEdgeTypes());
  for (dgl_type_t etype = 0; etype < graph->NumEdgeTypes(); ++etype) {
    auto pair = graph->meta_graph()->FindEdge(etype);
    const dgl_type_t src_vtype = pair.first;
    const dgl_type_t dst_vtype = pair.second;
    auto relgraph = graph->GetRelationGraph(etype);
    if (aten::IsNullArray(vids[dst_vtype])) {
      // create a placeholder graph
      subrels[etype] = UnitGraph::Empty(
        relgraph->NumVertexTypes(),
        graph->NumVertices(src_vtype),
        graph->NumVertices(dst_vtype),
        graph->DataType(), graph->Context());
      induced_edges[etype] = IdArray::Empty({0}, graph->DataType(), graph->Context());
    } else {
      const auto& earr = graph->InEdges(etype, {vids[dst_vtype]});
      subrels[etype] = UnitGraph::CreateFromCOO(
        relgraph->NumVertexTypes(),
        graph->NumVertices(src_vtype),
        graph->NumVertices(dst_vtype),
        earr.src,
        earr.dst);
      induced_edges[etype] = earr.id;
    }
  }
  HeteroSubgraph ret;
  ret.graph = CreateHeteroGraph(graph->meta_graph(), subrels, graph->NumVerticesPerType());
  ret.induced_edges = std::move(induced_edges);
  return ret;
}

HeteroSubgraph OutEdgeGraph(const HeteroGraphPtr graph, const std::vector<IdArray>& vids) {
  CHECK_EQ(vids.size(), graph->NumVertexTypes())
    << "Invalid input: the input list size must be the same as the number of vertex types.";
  std::vector<HeteroGraphPtr> subrels(graph->NumEdgeTypes());
  std::vector<IdArray> induced_edges(graph->NumEdgeTypes());
  for (dgl_type_t etype = 0; etype < graph->NumEdgeTypes(); ++etype) {
    auto pair = graph->meta_graph()->FindEdge(etype);
    const dgl_type_t src_vtype = pair.first;
    const dgl_type_t dst_vtype = pair.second;
    auto relgraph = graph->GetRelationGraph(etype);
    if (aten::IsNullArray(vids[src_vtype])) {
      // create a placeholder graph
      subrels[etype] = UnitGraph::Empty(
        relgraph->NumVertexTypes(),
        graph->NumVertices(src_vtype),
        graph->NumVertices(dst_vtype),
        graph->DataType(), graph->Context());
      induced_edges[etype] = IdArray::Empty({0}, graph->DataType(), graph->Context());
    } else {
      const auto& earr = graph->OutEdges(etype, {vids[src_vtype]});
      subrels[etype] = UnitGraph::CreateFromCOO(
          relgraph->NumVertexTypes(),
          graph->NumVertices(src_vtype),
          graph->NumVertices(dst_vtype),
          earr.src,
          earr.dst);
      induced_edges[etype] = earr.id;
    }
  }
  HeteroSubgraph ret;
  ret.graph = CreateHeteroGraph(graph->meta_graph(), subrels, graph->NumVerticesPerType());
  ret.induced_edges = std::move(induced_edges);
  return ret;
}

}  // namespace dgl
