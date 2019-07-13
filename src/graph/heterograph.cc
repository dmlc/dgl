#include "./heterograph.h"
#include <dgl/packed_func_ext.h>
#include "../c_api_common.h"
#include "./bipartite.h"

using namespace dgl::runtime;

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

// creator implementation
HeteroGraphPtr CreateBipartiteFromCOO(
    int64_t num_src, int64_t num_dst, IdArray row, IdArray col) {
  return Bipartite::CreateFromCOO(num_src, num_dst, row, col);
}

HeteroGraphPtr CreateBipartiteFromCSR(
    int64_t num_src, int64_t num_dst,
    IdArray indptr, IdArray indices, IdArray edge_ids) {
  return Bipartite::CreateFromCSR(num_src, num_dst, indptr, indices, edge_ids);
}

HeteroGraphPtr CreateHeteroGraph(
    GraphPtr meta_graph, const std::vector<HeteroGraphPtr>& rel_graphs) {
  return HeteroGraphPtr(new HeteroGraph(meta_graph, rel_graphs));
}

///////////////////////// C APIs /////////////////////////

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLHeteroCreateBipartiteFromCOO")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    int64_t num_src = args[0];
    int64_t num_dst = args[1];
    IdArray row = args[2];
    IdArray col = args[3];
    auto hgptr = CreateBipartiteFromCOO(num_src, num_dst, row, col);
    *rv = HeteroGraphRef(hgptr);
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLHeteroCreateBipartiteFromCSR")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    int64_t num_src = args[0];
    int64_t num_dst = args[1];
    IdArray indptr = args[2];
    IdArray indices = args[3];
    IdArray edge_ids = args[4];
    auto hgptr = CreateBipartiteFromCSR(num_src, num_dst, indptr, indices, edge_ids);
    *rv = HeteroGraphRef(hgptr);
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLHeteroCreateHeteroGraph")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    GraphRef meta_graph = args[0];
    List<HeteroGraphRef> rel_graphs = args[1];
    std::vector<HeteroGraphPtr> rel_ptrs;
    rel_ptrs.reserve(rel_graphs.size());
    for (const auto& ref : rel_graphs) {
      rel_ptrs.push_back(ref.sptr());
    }
    auto hgptr = CreateHeteroGraph(meta_graph.sptr(), rel_ptrs);
    *rv = HeteroGraphRef(hgptr);
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLHeteroGetMetaGraph")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    *rv = GraphRef(hg->meta_graph());
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLHeteroGetRelationGraph")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    dgl_type_t etype = args[1];
    *rv = HeteroGraphRef(hg->GetRelationGraph(etype));
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLHeteroAddVertices")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];

  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLHeteroAddEdge")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];

  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLHeteroAddEdges")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];

  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLHeteroClear")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];

  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLHeteroContext")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];

  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLHeteroNumBits")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];

  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLHeteroIsMultigraph")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];

  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLHeteroIsReadonly")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];

  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLHeteroNumVertices")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];

  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLHeteroNumEdges")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];

  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLHeteroHasVertex")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];

  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLHeteroHasVertices")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];

  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLHeteroHasEdgeBetween")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];

  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLHeteroHasEdgesBetween")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];

  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLHeteroSuccessors")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];

  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLHeteroEdgeId")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];

  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLHeteroEdgeIds")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];

  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLHeteroFindEdge")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];

  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLHeteroFindEdges")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];

  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLHeteroInEdges_1")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];

  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLHeteroInEdges_2")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];

  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLHeteroOutEdges_1")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];

  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLHeteroOutEdges_2")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];

  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLHeteroEdges")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];

  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLHeteroInDegree")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];

  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLHeteroInDegrees")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];

  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLHeteroOutDegree")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];

  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLHeteroOutDegrees")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];

  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLHeteroGetAdj")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];

  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLHeteroVertexSubgraph")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];

  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLHeteroEdgeSubgraph")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];

  });

}  // namespace dgl
