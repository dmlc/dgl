#include "./heterograph.h"
#include "./bipartite.h"

namespace dgl {

HeteroGraph::HeteroGraph(GraphPtr meta_graph, const std::vector<HeteroGraphPtr>& rel_graphs)
  : BaseHeteroGraph(meta_graph), relation_graphs_(rel_graphs) {
  // Sanity check
  CHECK_EQ(meta_graph->NumEdges(), rel_graphs.size());
  // all relation graph must be bipartite graphs
  for (const auto rg : rel_graphs) {
    CHECK_EQ(rg->NumVertexTypes(), 2) << "Each relation graph must be a bipartite graph.";
    CHECK_EQ(rg->NumEdgeTypes(), 1) << "Each relation graph must be a bipartite graph.";
  }
  // create num verts per type
  num_verts_per_type_.resize(meta_graph_->NumVertices(), -1);
  for (dgl_type_t vtype = 0; vtype < meta_graph_->NumVertices(); ++vtype) {
    for (dgl_type_t etype : meta_graph->OutEdgeVec(vtype)) {
      const auto nv = rel_graphs[etype]->NumVertices(Bipartite::kSrcVType);
      if (num_verts_per_type_[vtype] < 0) {
        num_verts_per_type_[vtype] = nv;
      } else {
        CHECK_EQ(num_verts_per_type_[vtype], nv)
          << "Mismatch number of vertices for vertex type " << vtype;
      }
    }
  }
}

bool HeteroGraph::IsMultigraph() const {
  return const_cast<HeteroGraph*>(this)->is_multigraph_.Get([this] () {
      for (const auto hg : relation_graphs_) {
        if (hg->IsMultigraph()) {
          return true;
        }
      }
      return false;
    });
}

HeteroSubgraph HeteroGraph::VertexSubgraph(const std::vector<IdArray>& vids) const {
  CHECK_EQ(vids.size(), NumVertexTypes())
    << "Invalid input: the input list size must be the same as the number of vertex types.";
  HeteroSubgraph ret;
  ret.induced_vertices = vids;
  ret.induced_edges.resize(NumEdgeTypes());
  std::vector<HeteroGraphPtr> subrels(NumEdgeTypes());
  for (dgl_type_t etype = 0; etype < NumEdgeTypes(); ++etype) {
    auto pair = meta_graph_->FindEdge(etype);
    const dgl_type_t src_vtype = pair.first;
    const dgl_type_t dst_vtype = pair.second;
    const auto& rel_vsg = GetRelationGraph(etype)->VertexSubgraph(
        {vids[src_vtype], vids[dst_vtype]});
    subrels[etype] = rel_vsg.graph;
    ret.induced_edges[etype] = rel_vsg.induced_edges[0];
  }
  ret.graph = HeteroGraphPtr(new HeteroGraph(meta_graph_, subrels));
  return ret;
}

HeteroSubgraph HeteroGraph::EdgeSubgraph(
    const std::vector<IdArray>& eids, bool preserve_nodes) const {
  CHECK_EQ(eids.size(), NumEdgeTypes())
    << "Invalid input: the input list size must be the same as the number of edge type.";
  HeteroSubgraph ret;
  ret.induced_vertices.resize(NumVertexTypes());
  ret.induced_edges = eids;
  std::vector<HeteroGraphPtr> subrels(NumEdgeTypes());
  for (dgl_type_t etype = 0; etype < NumEdgeTypes(); ++etype) {
    auto pair = meta_graph_->FindEdge(etype);
    const dgl_type_t src_vtype = pair.first;
    const dgl_type_t dst_vtype = pair.second;
    const auto& rel_vsg = GetRelationGraph(etype)->EdgeSubgraph(
        {eids[etype]}, preserve_nodes);
    subrels[etype] = rel_vsg.graph;
    ret.induced_vertices[src_vtype] = rel_vsg.induced_vertices[0];
    ret.induced_vertices[dst_vtype] = rel_vsg.induced_vertices[1];
  }
  ret.graph = HeteroGraphPtr(new HeteroGraph(meta_graph_, subrels));
  return ret;
}

}  // namespace dgl
