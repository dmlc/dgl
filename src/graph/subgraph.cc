/**
 *  Copyright (c) 2020 by Contributors
 * @file graph/subgraph.cc
 * @brief Functions for extracting subgraphs.
 */
#include "./heterograph.h"
using namespace dgl::runtime;

namespace dgl {

HeteroSubgraph InEdgeGraphRelabelNodes(
    const HeteroGraphPtr graph, const std::vector<IdArray>& vids) {
  CHECK_EQ(vids.size(), graph->NumVertexTypes())
      << "Invalid input: the input list size must be the same as the number of "
         "vertex types.";
  std::vector<IdArray> eids(graph->NumEdgeTypes());
  DGLContext ctx = aten::GetContextOf(vids);
  for (dgl_type_t etype = 0; etype < graph->NumEdgeTypes(); ++etype) {
    auto pair = graph->meta_graph()->FindEdge(etype);
    const dgl_type_t dst_vtype = pair.second;
    if (aten::IsNullArray(vids[dst_vtype])) {
      eids[etype] = IdArray::Empty({0}, graph->DataType(), ctx);
    } else {
      const auto& earr = graph->InEdges(etype, {vids[dst_vtype]});
      eids[etype] = earr.id;
    }
  }
  return graph->EdgeSubgraph(eids, false);
}

HeteroSubgraph InEdgeGraphNoRelabelNodes(
    const HeteroGraphPtr graph, const std::vector<IdArray>& vids) {
  // TODO(mufei): This should also use EdgeSubgraph once it is supported for CSR
  // graphs
  CHECK_EQ(vids.size(), graph->NumVertexTypes())
      << "Invalid input: the input list size must be the same as the number of "
         "vertex types.";
  std::vector<HeteroGraphPtr> subrels(graph->NumEdgeTypes());
  std::vector<IdArray> induced_edges(graph->NumEdgeTypes());
  DGLContext ctx = aten::GetContextOf(vids);
  for (dgl_type_t etype = 0; etype < graph->NumEdgeTypes(); ++etype) {
    auto pair = graph->meta_graph()->FindEdge(etype);
    const dgl_type_t src_vtype = pair.first;
    const dgl_type_t dst_vtype = pair.second;
    auto relgraph = graph->GetRelationGraph(etype);
    if (aten::IsNullArray(vids[dst_vtype])) {
      // create a placeholder graph
      subrels[etype] = UnitGraph::Empty(
          relgraph->NumVertexTypes(), graph->NumVertices(src_vtype),
          graph->NumVertices(dst_vtype), graph->DataType(), ctx);
      induced_edges[etype] =
          IdArray::Empty({0}, graph->DataType(), graph->Context());
    } else {
      const auto& earr = graph->InEdges(etype, {vids[dst_vtype]});
      subrels[etype] = UnitGraph::CreateFromCOO(
          relgraph->NumVertexTypes(), graph->NumVertices(src_vtype),
          graph->NumVertices(dst_vtype), earr.src, earr.dst);
      induced_edges[etype] = earr.id;
    }
  }
  HeteroSubgraph ret;
  ret.graph = CreateHeteroGraph(
      graph->meta_graph(), subrels, graph->NumVerticesPerType());
  ret.induced_edges = std::move(induced_edges);
  return ret;
}

HeteroSubgraph InEdgeGraph(
    const HeteroGraphPtr graph, const std::vector<IdArray>& vids,
    bool relabel_nodes) {
  if (relabel_nodes) {
    return InEdgeGraphRelabelNodes(graph, vids);
  } else {
    return InEdgeGraphNoRelabelNodes(graph, vids);
  }
}

HeteroSubgraph OutEdgeGraphRelabelNodes(
    const HeteroGraphPtr graph, const std::vector<IdArray>& vids) {
  CHECK_EQ(vids.size(), graph->NumVertexTypes())
      << "Invalid input: the input list size must be the same as the number of "
         "vertex types.";
  std::vector<IdArray> eids(graph->NumEdgeTypes());
  DGLContext ctx = aten::GetContextOf(vids);
  for (dgl_type_t etype = 0; etype < graph->NumEdgeTypes(); ++etype) {
    auto pair = graph->meta_graph()->FindEdge(etype);
    const dgl_type_t src_vtype = pair.first;
    if (aten::IsNullArray(vids[src_vtype])) {
      eids[etype] = IdArray::Empty({0}, graph->DataType(), ctx);
    } else {
      const auto& earr = graph->OutEdges(etype, {vids[src_vtype]});
      eids[etype] = earr.id;
    }
  }
  return graph->EdgeSubgraph(eids, false);
}

HeteroSubgraph OutEdgeGraphNoRelabelNodes(
    const HeteroGraphPtr graph, const std::vector<IdArray>& vids) {
  // TODO(mufei): This should also use EdgeSubgraph once it is supported for CSR
  // graphs
  CHECK_EQ(vids.size(), graph->NumVertexTypes())
      << "Invalid input: the input list size must be the same as the number of "
         "vertex types.";
  std::vector<HeteroGraphPtr> subrels(graph->NumEdgeTypes());
  std::vector<IdArray> induced_edges(graph->NumEdgeTypes());
  DGLContext ctx = aten::GetContextOf(vids);
  for (dgl_type_t etype = 0; etype < graph->NumEdgeTypes(); ++etype) {
    auto pair = graph->meta_graph()->FindEdge(etype);
    const dgl_type_t src_vtype = pair.first;
    const dgl_type_t dst_vtype = pair.second;
    auto relgraph = graph->GetRelationGraph(etype);
    if (aten::IsNullArray(vids[src_vtype])) {
      // create a placeholder graph
      subrels[etype] = UnitGraph::Empty(
          relgraph->NumVertexTypes(), graph->NumVertices(src_vtype),
          graph->NumVertices(dst_vtype), graph->DataType(), ctx);
      induced_edges[etype] =
          IdArray::Empty({0}, graph->DataType(), graph->Context());
    } else {
      const auto& earr = graph->OutEdges(etype, {vids[src_vtype]});
      subrels[etype] = UnitGraph::CreateFromCOO(
          relgraph->NumVertexTypes(), graph->NumVertices(src_vtype),
          graph->NumVertices(dst_vtype), earr.src, earr.dst);
      induced_edges[etype] = earr.id;
    }
  }
  HeteroSubgraph ret;
  ret.graph = CreateHeteroGraph(
      graph->meta_graph(), subrels, graph->NumVerticesPerType());
  ret.induced_edges = std::move(induced_edges);
  return ret;
}

HeteroSubgraph OutEdgeGraph(
    const HeteroGraphPtr graph, const std::vector<IdArray>& vids,
    bool relabel_nodes) {
  if (relabel_nodes) {
    return OutEdgeGraphRelabelNodes(graph, vids);
  } else {
    return OutEdgeGraphNoRelabelNodes(graph, vids);
  }
}

}  // namespace dgl
