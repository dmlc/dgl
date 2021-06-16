/*!
 *  Copyright (c) 2020 by Contributors
 * \file graph/subgraph.cc
 * \brief Functions for extracting subgraphs.
 */
#include "./heterograph.h"
using namespace dgl::runtime;

namespace dgl {

HeteroSubgraph InEdgeGraphRelabelNodes(
    const HeteroGraphPtr graph, const std::vector<IdArray>& vids) {
  CHECK_EQ(vids.size(), graph->NumVertexTypes())
    << "Invalid input: the input list size must be the same as the number of vertex types.";
  HeteroSubgraph ret;
  ret.induced_vertices.resize(graph->NumVertexTypes());
  ret.induced_edges.resize(graph->NumEdgeTypes());
  // NOTE(mufei): InEdgeGraph when relabel_nodes is true is quite complicated in
  // heterograph. This is because we need to make sure bipartite graphs that incident
  // on the same vertex type must have the same ID space. For example, suppose we have
  // following heterograph:
  //
  // Meta graph: A -> B, C -> B
  // UnitGraph graphs:
  // * A -> B: (0, 0), (0, 1)
  // * C -> B: (1, 0), (1, 1)
  //
  // Suppose for A->B, we only keep edge (0, 0), while for C->B we only keep (1, 1). We need
  // to make sure that in the result subgraph, node type B still has two nodes. This means
  // we cannot simply compute InEdgeGraph for C->B which will relabel node#1 of type B to be
  // node #0.
  //
  // One implementation is as follows:
  // (1) For each bipartite graph, slice out the in edges using the given vids.
  // (2) Make a dictionary map<vtype, vector<IdArray>>, where the key is the vertex type
  //     and the value is the incident nodes from the bipartite graphs that has the vertex
  //     type as either srctype or dsttype.
  // (3) Then for each vertex type, use aten::Relabel_ on its vector<IdArray>.
  //     aten::Relabel_ computes the union of the vertex sets and relabel
  //     the unique elements from zero. The returned mapping array is the final induced
  //     vertex set for that vertex type.
  // (4) Use the relabeled edges to construct the bipartite graph.
  // step (1) & (2)
  std::vector<EdgeArray> subedges(graph->NumEdgeTypes());
  std::vector<std::vector<IdArray>> vtype2incnodes(graph->NumVertexTypes());
  for (dgl_type_t etype = 0; etype < graph->NumEdgeTypes(); ++etype) {
    auto pair = graph->meta_graph()->FindEdge(etype);
    const dgl_type_t src_vtype = pair.first;
    const dgl_type_t dst_vtype = pair.second;
    const auto& earr = graph->InEdges(etype, {vids[dst_vtype]});
    vtype2incnodes[src_vtype].push_back(earr.src);
    vtype2incnodes[dst_vtype].push_back(earr.dst);
    subedges[etype] = earr;
  }
  // step (3)
  std::vector<int64_t> num_vertices_per_type(graph->NumVertexTypes());
  for (dgl_type_t vtype = 0; vtype < graph->NumVertexTypes(); ++vtype) {
    ret.induced_vertices[vtype] = aten::Relabel_(vtype2incnodes[vtype]);
    num_vertices_per_type[vtype] = ret.induced_vertices[vtype]->shape[0];
  }
  // step (4)
  std::vector<HeteroGraphPtr> subrels(graph->NumEdgeTypes());
  for (dgl_type_t etype = 0; etype < graph->NumEdgeTypes(); ++etype) {
    auto pair = graph->meta_graph()->FindEdge(etype);
    const dgl_type_t src_vtype = pair.first;
    const dgl_type_t dst_vtype = pair.second;
    if (aten::IsNullArray(vids[dst_vtype])) {
      // create a placeholder graph
      subrels[etype] = UnitGraph::Empty(
        (src_vtype == dst_vtype)? 1 : 2,
        ret.induced_vertices[src_vtype]->shape[0],
        ret.induced_vertices[dst_vtype]->shape[0],
        graph->DataType(), graph->Context());
      ret.induced_edges[etype] = IdArray::Empty({0}, graph->DataType(), graph->Context());
    } else {
      subrels[etype] = UnitGraph::CreateFromCOO(
        (src_vtype == dst_vtype)? 1 : 2,
        ret.induced_vertices[src_vtype]->shape[0],
        ret.induced_vertices[dst_vtype]->shape[0],
        subedges[etype].src,
        subedges[etype].dst);
      ret.induced_edges[etype] = subedges[etype].id;
    }
  }
  ret.graph = HeteroGraphPtr(new HeteroGraph(
    graph->meta_graph(), subrels, std::move(num_vertices_per_type)));
  return ret;
}

HeteroSubgraph InEdgeGraphNoRelabelNodes(
    const HeteroGraphPtr graph, const std::vector<IdArray>& vids) {
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

HeteroSubgraph InEdgeGraph(
    const HeteroGraphPtr graph, const std::vector<IdArray>& vids, bool relabel_nodes) {
  if (relabel_nodes) {
    return InEdgeGraphRelabelNodes(graph, vids);
  } else {
    return InEdgeGraphNoRelabelNodes(graph, vids);
  }
}

HeteroSubgraph OutEdgeGraphRelabelNodes(
    const HeteroGraphPtr graph, const std::vector<IdArray>& vids) {
  CHECK_EQ(vids.size(), graph->NumVertexTypes())
    << "Invalid input: the input list size must be the same as the number of vertex types.";
  HeteroSubgraph ret;
  ret.induced_vertices.resize(graph->NumVertexTypes());
  ret.induced_edges.resize(graph->NumEdgeTypes());
  // NOTE(mufei): OutEdgeGraph when relabel_nodes is true is quite complicated in
  // heterograph. This is because we need to make sure bipartite graphs that incident
  // on the same vertex type must have the same ID space. For example, suppose we have
  // following heterograph:
  //
  // Meta graph: B -> A, B -> C
  // UnitGraph graphs:
  // * B -> A: (0, 0), (0, 1)
  // * B -> C: (1, 0), (1, 1)
  //
  // Suppose for B->A, we only keep edge (0, 1), while for B->C we only keep (1, 1). We need
  // to make sure that in the result subgraph, node type B still has two nodes. This means
  // we cannot simply compute OutEdgeGraph for B->C which will relabel node#1 of type B to be
  // node#0.
  //
  // One implementation is as follows:
  // (1) For each bipartite graph, slice out the out edges using the given vids.
  // (2) Make a dictionary map<vtype, vector<IdArray>>, where the key is the vertex type
  //     and the value is the incident nodes from the bipartite graphs that has the vertex
  //     type as either srctype or dsttype.
  // (3) Then for each vertex type, use aten::Relabel_ on its vector<IdArray>.
  //     aten::Relabel_ computes the union of the vertex sets and relabel
  //     the unique elements from zero. The returned mapping array is the final induced
  //     vertex set for that vertex type.
  // (4) Use the relabeled edges to construct the bipartite graph.
  // step (1) & (2)
  std::vector<EdgeArray> subedges(graph->NumEdgeTypes());
  std::vector<std::vector<IdArray>> vtype2incnodes(graph->NumVertexTypes());
  for (dgl_type_t etype = 0; etype < graph->NumEdgeTypes(); ++etype) {
    auto pair = graph->meta_graph()->FindEdge(etype);
    const dgl_type_t src_vtype = pair.first;
    const dgl_type_t dst_vtype = pair.second;
    const auto& earr = graph->OutEdges(etype, {vids[src_vtype]});
    vtype2incnodes[src_vtype].push_back(earr.src);
    vtype2incnodes[dst_vtype].push_back(earr.dst);
    subedges[etype] = earr;
  }
  // step (3)
  std::vector<int64_t> num_vertices_per_type(graph->NumVertexTypes());
  for (dgl_type_t vtype = 0; vtype < graph->NumVertexTypes(); ++vtype) {
    ret.induced_vertices[vtype] = aten::Relabel_(vtype2incnodes[vtype]);
    num_vertices_per_type[vtype] = ret.induced_vertices[vtype]->shape[0];
  }
  // step (4)
  std::vector<HeteroGraphPtr> subrels(graph->NumEdgeTypes());
  for (dgl_type_t etype = 0; etype < graph->NumEdgeTypes(); ++etype) {
    auto pair = graph->meta_graph()->FindEdge(etype);
    const dgl_type_t src_vtype = pair.first;
    const dgl_type_t dst_vtype = pair.second;
    if (aten::IsNullArray(vids[src_vtype])) {
      // create a placeholder graph
      subrels[etype] = UnitGraph::Empty(
        (src_vtype == dst_vtype)? 1 : 2,
        ret.induced_vertices[src_vtype]->shape[0],
        ret.induced_vertices[dst_vtype]->shape[0],
        graph->DataType(), graph->Context());
      ret.induced_edges[etype] = IdArray::Empty({0}, graph->DataType(), graph->Context());
    } else {
      subrels[etype] = UnitGraph::CreateFromCOO(
        (src_vtype == dst_vtype)? 1 : 2,
        ret.induced_vertices[src_vtype]->shape[0],
        ret.induced_vertices[dst_vtype]->shape[0],
        subedges[etype].src,
        subedges[etype].dst);
      ret.induced_edges[etype] = subedges[etype].id;
    }
  }
  ret.graph = HeteroGraphPtr(new HeteroGraph(
    graph->meta_graph(), subrels, std::move(num_vertices_per_type)));
  return ret;
}

HeteroSubgraph OutEdgeGraphNoRelabelNodes(
    const HeteroGraphPtr graph, const std::vector<IdArray>& vids) {
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

HeteroSubgraph OutEdgeGraph(
    const HeteroGraphPtr graph, const std::vector<IdArray>& vids, bool relabel_nodes) {
  if (relabel_nodes) {
    return OutEdgeGraphRelabelNodes(graph, vids);
  } else {
    return OutEdgeGraphNoRelabelNodes(graph, vids);
  }
}

}  // namespace dgl
