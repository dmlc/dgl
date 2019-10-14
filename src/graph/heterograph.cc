/*!
 *  Copyright (c) 2019 by Contributors
 * \file graph/heterograph.cc
 * \brief Heterograph implementation
 */
#include "./heterograph.h"
#include <dgl/array.h>
#include <dgl/packed_func_ext.h>
#include <dgl/runtime/container.h>
#include "../c_api_common.h"
#include "./unit_graph.h"

using namespace dgl::runtime;

namespace dgl {
namespace {

HeteroSubgraph EdgeSubgraphPreserveNodes(
    const HeteroGraph* hg, const std::vector<IdArray>& eids) {
  CHECK_EQ(eids.size(), hg->NumEdgeTypes())
    << "Invalid input: the input list size must be the same as the number of edge type.";
  HeteroSubgraph ret;
  ret.induced_vertices.resize(hg->NumVertexTypes());
  ret.induced_edges = eids;
  // When preserve_nodes is true, simply compute EdgeSubgraph for each bipartite
  std::vector<HeteroGraphPtr> subrels(hg->NumEdgeTypes());
  for (dgl_type_t etype = 0; etype < hg->NumEdgeTypes(); ++etype) {
    auto pair = hg->meta_graph()->FindEdge(etype);
    const dgl_type_t src_vtype = pair.first;
    const dgl_type_t dst_vtype = pair.second;
    const auto& rel_vsg = hg->GetRelationGraph(etype)->EdgeSubgraph(
        {eids[etype]}, true);
    subrels[etype] = rel_vsg.graph;
    ret.induced_vertices[src_vtype] = rel_vsg.induced_vertices[0];
    ret.induced_vertices[dst_vtype] = rel_vsg.induced_vertices[1];
  }
  ret.graph = HeteroGraphPtr(new HeteroGraph(hg->meta_graph(), subrels));
  return ret;
}

HeteroSubgraph EdgeSubgraphNoPreserveNodes(
    const HeteroGraph* hg, const std::vector<IdArray>& eids) {
  CHECK_EQ(eids.size(), hg->NumEdgeTypes())
    << "Invalid input: the input list size must be the same as the number of edge type.";
  HeteroSubgraph ret;
  ret.induced_vertices.resize(hg->NumVertexTypes());
  ret.induced_edges = eids;
  // NOTE(minjie): EdgeSubgraph when preserve_nodes is false is quite complicated in
  // heterograph. This is because we need to make sure bipartite graphs that incident
  // on the same vertex type must have the same ID space. For example, suppose we have
  // following heterograph:
  //
  // Meta graph: A -> B -> C
  // UnitGraph graphs:
  // * A -> B: (0, 0), (0, 1)
  // * B -> C: (1, 0), (1, 1)
  //
  // Suppose for A->B, we only keep edge (0, 0), while for B->C we only keep (1, 0). We need
  // to make sure that in the result subgraph, node type B still has two nodes. This means
  // we cannot simply compute EdgeSubgraph for B->C which will relabel node#1 of type B to be
  // node #0.
  //
  // One implementation is as follows:
  // (1) For each bipartite graph, slice out the edges using the given eids.
  // (2) Make a dictionary map<vtype, vector<IdArray>>, where the key is the vertex type
  //     and the value is the incident nodes from the bipartite graphs that has the vertex
  //     type as either srctype or dsttype.
  // (3) Then for each vertex type, use aten::Relabel_ on its vector<IdArray>.
  //     aten::Relabel_ computes the union of the vertex sets and relabel
  //     the unique elements from zero. The returned mapping array is the final induced
  //     vertex set for that vertex type.
  // (4) Use the relabeled edges to construct the bipartite graph.
  // step (1) & (2)
  std::vector<EdgeArray> subedges(hg->NumEdgeTypes());
  std::vector<std::vector<IdArray>> vtype2incnodes(hg->NumVertexTypes());
  for (dgl_type_t etype = 0; etype < hg->NumEdgeTypes(); ++etype) {
    auto pair = hg->meta_graph()->FindEdge(etype);
    const dgl_type_t src_vtype = pair.first;
    const dgl_type_t dst_vtype = pair.second;
    auto earray = hg->GetRelationGraph(etype)->FindEdges(0, eids[etype]);
    vtype2incnodes[src_vtype].push_back(earray.src);
    vtype2incnodes[dst_vtype].push_back(earray.dst);
    subedges[etype] = earray;
  }
  // step (3)
  for (dgl_type_t vtype = 0; vtype < hg->NumVertexTypes(); ++vtype) {
    ret.induced_vertices[vtype] = aten::Relabel_(vtype2incnodes[vtype]);
  }
  // step (4)
  std::vector<HeteroGraphPtr> subrels(hg->NumEdgeTypes());
  for (dgl_type_t etype = 0; etype < hg->NumEdgeTypes(); ++etype) {
    auto pair = hg->meta_graph()->FindEdge(etype);
    const dgl_type_t src_vtype = pair.first;
    const dgl_type_t dst_vtype = pair.second;
    subrels[etype] = UnitGraph::CreateFromCOO(
      (src_vtype == dst_vtype)? 1 : 2,
      ret.induced_vertices[src_vtype]->shape[0],
      ret.induced_vertices[dst_vtype]->shape[0],
      subedges[etype].src,
      subedges[etype].dst);
  }
  ret.graph = HeteroGraphPtr(new HeteroGraph(hg->meta_graph(), subrels));
  return ret;
}

}  // namespace

HeteroGraph::HeteroGraph(GraphPtr meta_graph, const std::vector<HeteroGraphPtr>& rel_graphs)
  : BaseHeteroGraph(meta_graph), relation_graphs_(rel_graphs) {
  // Sanity check
  CHECK_EQ(meta_graph->NumEdges(), rel_graphs.size());
  CHECK(!rel_graphs.empty()) << "Empty heterograph is not allowed.";
  // all relation graphs must have only one edge type
  for (const auto rg : rel_graphs) {
    CHECK_EQ(rg->NumEdgeTypes(), 1) << "Each relation graph must have only one edge type.";
  }
  // create num verts per type
  num_verts_per_type_.resize(meta_graph->NumVertices(), -1);

  EdgeArray etype_array = meta_graph->Edges();
  dgl_type_t *srctypes = static_cast<dgl_type_t *>(etype_array.src->data);
  dgl_type_t *dsttypes = static_cast<dgl_type_t *>(etype_array.dst->data);
  dgl_type_t *etypes = static_cast<dgl_type_t *>(etype_array.id->data);

  for (size_t i = 0; i < meta_graph->NumEdges(); ++i) {
    dgl_type_t srctype = srctypes[i];
    dgl_type_t dsttype = dsttypes[i];
    dgl_type_t etype = etypes[i];
    const auto& rg = rel_graphs[etype];
    const auto sty = 0;
    const auto dty = rg->NumVertexTypes() == 1? 0 : 1;
    size_t nv;

    // # nodes of source type
    nv = rg->NumVertices(sty);
    if (num_verts_per_type_[srctype] < 0)
      num_verts_per_type_[srctype] = nv;
    else
      CHECK_EQ(num_verts_per_type_[srctype], nv)
        << "Mismatch number of vertices for vertex type " << srctype;
    // # nodes of destination type
    nv = rg->NumVertices(dty);
    if (num_verts_per_type_[dsttype] < 0)
      num_verts_per_type_[dsttype] = nv;
    else
      CHECK_EQ(num_verts_per_type_[dsttype], nv)
        << "Mismatch number of vertices for vertex type " << dsttype;
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

BoolArray HeteroGraph::HasVertices(dgl_type_t vtype, IdArray vids) const {
  CHECK(aten::IsValidIdArray(vids)) << "Invalid id array input";
  return aten::LT(vids, NumVertices(vtype));
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
    const std::vector<IdArray> rel_vids = (src_vtype == dst_vtype) ?
      std::vector<IdArray>({vids[src_vtype]}) :
      std::vector<IdArray>({vids[src_vtype], vids[dst_vtype]});
    const auto& rel_vsg = GetRelationGraph(etype)->VertexSubgraph(rel_vids);
    subrels[etype] = rel_vsg.graph;
    ret.induced_edges[etype] = rel_vsg.induced_edges[0];
  }
  ret.graph = HeteroGraphPtr(new HeteroGraph(meta_graph_, subrels));
  return ret;
}

HeteroSubgraph HeteroGraph::EdgeSubgraph(
    const std::vector<IdArray>& eids, bool preserve_nodes) const {
  if (preserve_nodes) {
    return EdgeSubgraphPreserveNodes(this, eids);
  } else {
    return EdgeSubgraphNoPreserveNodes(this, eids);
  }
}

FlattenedHeteroGraphPtr HeteroGraph::Flatten(const std::vector<dgl_type_t>& etypes) const {
  std::unordered_map<dgl_type_t, size_t> srctype_offsets, dsttype_offsets;
  size_t src_nodes = 0, dst_nodes = 0;
  std::vector<dgl_id_t> result_src, result_dst;
  std::vector<dgl_type_t> induced_srctype, induced_etype, induced_dsttype;
  std::vector<dgl_id_t> induced_srcid, induced_eid, induced_dstid;
  std::vector<dgl_type_t> srctype_set, dsttype_set;

  // XXXtype_offsets contain the mapping from node type and number of nodes after this
  // loop.
  for (dgl_type_t etype : etypes) {
    auto src_dsttype = meta_graph_->FindEdge(etype);
    dgl_type_t srctype = src_dsttype.first;
    dgl_type_t dsttype = src_dsttype.second;
    size_t num_srctype_nodes = NumVertices(srctype);
    size_t num_dsttype_nodes = NumVertices(dsttype);

    if (srctype_offsets.count(srctype) == 0) {
      srctype_offsets[srctype] = num_srctype_nodes;
      srctype_set.push_back(srctype);
    }
    if (dsttype_offsets.count(dsttype) == 0) {
      dsttype_offsets[dsttype] = num_dsttype_nodes;
      dsttype_set.push_back(dsttype);
    }
  }

  // Sort the node types so that we can compare the sets and decide whether a homograph
  // should be returned.
  std::sort(srctype_set.begin(), srctype_set.end());
  std::sort(dsttype_set.begin(), dsttype_set.end());
  bool homograph = (srctype_set.size() == dsttype_set.size()) &&
    std::equal(srctype_set.begin(), srctype_set.end(), dsttype_set.begin());

  // XXXtype_offsets contain the mapping from node type to node ID offsets after these
  // two loops.
  for (size_t i = 0; i < srctype_set.size(); ++i) {
    dgl_type_t ntype = srctype_set[i];
    size_t num_nodes = srctype_offsets[ntype];
    srctype_offsets[ntype] = src_nodes;
    src_nodes += num_nodes;
    for (size_t j = 0; j < num_nodes; ++j) {
      induced_srctype.push_back(ntype);
      induced_srcid.push_back(j);
    }
  }
  for (size_t i = 0; i < dsttype_set.size(); ++i) {
    dgl_type_t ntype = dsttype_set[i];
    size_t num_nodes = dsttype_offsets[ntype];
    dsttype_offsets[ntype] = dst_nodes;
    dst_nodes += num_nodes;
    for (size_t j = 0; j < num_nodes; ++j) {
      induced_dsttype.push_back(ntype);
      induced_dstid.push_back(j);
    }
  }

  for (dgl_type_t etype : etypes) {
    auto src_dsttype = meta_graph_->FindEdge(etype);
    dgl_type_t srctype = src_dsttype.first;
    dgl_type_t dsttype = src_dsttype.second;
    size_t srctype_offset = srctype_offsets[srctype];
    size_t dsttype_offset = dsttype_offsets[dsttype];

    EdgeArray edges = Edges(etype);
    size_t num_edges = NumEdges(etype);
    const dgl_id_t* edges_src_data = static_cast<const dgl_id_t*>(edges.src->data);
    const dgl_id_t* edges_dst_data = static_cast<const dgl_id_t*>(edges.dst->data);
    const dgl_id_t* edges_eid_data = static_cast<const dgl_id_t*>(edges.id->data);
    // TODO(gq) Use concat?
    for (size_t i = 0; i < num_edges; ++i) {
      result_src.push_back(edges_src_data[i] + srctype_offset);
      result_dst.push_back(edges_dst_data[i] + dsttype_offset);
      induced_etype.push_back(etype);
      induced_eid.push_back(edges_eid_data[i]);
    }
  }

  HeteroGraphPtr gptr = UnitGraph::CreateFromCOO(
      homograph ? 1 : 2,
      src_nodes,
      dst_nodes,
      aten::VecToIdArray(result_src),
      aten::VecToIdArray(result_dst));

  FlattenedHeteroGraph* result = new FlattenedHeteroGraph;
  result->graph = HeteroGraphRef(gptr);
  result->induced_srctype = aten::VecToIdArray(induced_srctype);
  result->induced_srctype_set = aten::VecToIdArray(srctype_set);
  result->induced_srcid = aten::VecToIdArray(induced_srcid);
  result->induced_etype = aten::VecToIdArray(induced_etype);
  result->induced_etype_set = aten::VecToIdArray(etypes);
  result->induced_eid = aten::VecToIdArray(induced_eid);
  result->induced_dsttype = aten::VecToIdArray(induced_dsttype);
  result->induced_dsttype_set = aten::VecToIdArray(dsttype_set);
  result->induced_dstid = aten::VecToIdArray(induced_dstid);
  return FlattenedHeteroGraphPtr(result);
}

// creator implementation
HeteroGraphPtr CreateHeteroGraph(
    GraphPtr meta_graph, const std::vector<HeteroGraphPtr>& rel_graphs) {
  return HeteroGraphPtr(new HeteroGraph(meta_graph, rel_graphs));
}

///////////////////////// C APIs /////////////////////////

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroCreateUnitGraphFromCOO")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    int64_t nvtypes = args[0];
    int64_t num_src = args[1];
    int64_t num_dst = args[2];
    IdArray row = args[3];
    IdArray col = args[4];
    auto hgptr = UnitGraph::CreateFromCOO(nvtypes, num_src, num_dst, row, col);
    *rv = HeteroGraphRef(hgptr);
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroCreateUnitGraphFromCSR")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    int64_t nvtypes = args[0];
    int64_t num_src = args[1];
    int64_t num_dst = args[2];
    IdArray indptr = args[3];
    IdArray indices = args[4];
    IdArray edge_ids = args[5];
    auto hgptr = UnitGraph::CreateFromCSR(
        nvtypes, num_src, num_dst, indptr, indices, edge_ids);
    *rv = HeteroGraphRef(hgptr);
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroCreateHeteroGraph")
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

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroGetMetaGraph")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    *rv = GraphRef(hg->meta_graph());
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroGetRelationGraph")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    dgl_type_t etype = args[1];
    if (hg->NumEdgeTypes() == 1) {
      CHECK_EQ(etype, 0);
      *rv = hg;
    } else {
      *rv = HeteroGraphRef(hg->GetRelationGraph(etype));
    }
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroGetFlattenedGraph")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    List<Value> etypes = args[1];
    std::vector<dgl_id_t> etypes_vec;
    for (Value val : etypes) {
      // (gq) have to decompose it into two statements because of a weird MSVC internal error
      dgl_id_t id = val->data;
      etypes_vec.push_back(id);
    }

    *rv = FlattenedHeteroGraphRef(hg->Flatten(etypes_vec));
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroAddVertices")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    dgl_type_t vtype = args[1];
    int64_t num = args[2];
    hg->AddVertices(vtype, num);
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroAddEdge")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    dgl_type_t etype = args[1];
    dgl_id_t src = args[2];
    dgl_id_t dst = args[3];
    hg->AddEdge(etype, src, dst);
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroAddEdges")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    dgl_type_t etype = args[1];
    IdArray src = args[2];
    IdArray dst = args[3];
    hg->AddEdges(etype, src, dst);
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroClear")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    hg->Clear();
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroContext")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    *rv = hg->Context();
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroNumBits")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    *rv = hg->NumBits();
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroIsMultigraph")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    *rv = hg->IsMultigraph();
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroIsReadonly")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    *rv = hg->IsReadonly();
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroNumVertices")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    dgl_type_t vtype = args[1];
    *rv = static_cast<int64_t>(hg->NumVertices(vtype));
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroNumEdges")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    dgl_type_t etype = args[1];
    *rv = static_cast<int64_t>(hg->NumEdges(etype));
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroHasVertex")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    dgl_type_t vtype = args[1];
    dgl_id_t vid = args[2];
    *rv = hg->HasVertex(vtype, vid);
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroHasVertices")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    dgl_type_t vtype = args[1];
    IdArray vids = args[2];
    *rv = hg->HasVertices(vtype, vids);
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroHasEdgeBetween")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    dgl_type_t etype = args[1];
    dgl_id_t src = args[2];
    dgl_id_t dst = args[3];
    *rv = hg->HasEdgeBetween(etype, src, dst);
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroHasEdgesBetween")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    dgl_type_t etype = args[1];
    IdArray src = args[2];
    IdArray dst = args[3];
    *rv = hg->HasEdgesBetween(etype, src, dst);
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroPredecessors")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    dgl_type_t etype = args[1];
    dgl_id_t dst = args[2];
    *rv = hg->Predecessors(etype, dst);
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroSuccessors")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    dgl_type_t etype = args[1];
    dgl_id_t src = args[2];
    *rv = hg->Successors(etype, src);
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroEdgeId")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    dgl_type_t etype = args[1];
    dgl_id_t src = args[2];
    dgl_id_t dst = args[3];
    *rv = hg->EdgeId(etype, src, dst);
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroEdgeIds")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    dgl_type_t etype = args[1];
    IdArray src = args[2];
    IdArray dst = args[3];
    const auto& ret = hg->EdgeIds(etype, src, dst);
    *rv = ConvertEdgeArrayToPackedFunc(ret);
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroFindEdges")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    dgl_type_t etype = args[1];
    IdArray eids = args[2];
    const auto& ret = hg->FindEdges(etype, eids);
    *rv = ConvertEdgeArrayToPackedFunc(ret);
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroInEdges_1")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    dgl_type_t etype = args[1];
    dgl_id_t vid = args[2];
    const auto& ret = hg->InEdges(etype, vid);
    *rv = ConvertEdgeArrayToPackedFunc(ret);
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroInEdges_2")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    dgl_type_t etype = args[1];
    IdArray vids = args[2];
    const auto& ret = hg->InEdges(etype, vids);
    *rv = ConvertEdgeArrayToPackedFunc(ret);
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroOutEdges_1")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    dgl_type_t etype = args[1];
    dgl_id_t vid = args[2];
    const auto& ret = hg->OutEdges(etype, vid);
    *rv = ConvertEdgeArrayToPackedFunc(ret);
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroOutEdges_2")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    dgl_type_t etype = args[1];
    IdArray vids = args[2];
    const auto& ret = hg->OutEdges(etype, vids);
    *rv = ConvertEdgeArrayToPackedFunc(ret);
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroEdges")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    dgl_type_t etype = args[1];
    std::string order = args[2];
    const auto& ret = hg->Edges(etype, order);
    *rv = ConvertEdgeArrayToPackedFunc(ret);
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroInDegree")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    dgl_type_t etype = args[1];
    dgl_id_t vid = args[2];
    *rv = static_cast<int64_t>(hg->InDegree(etype, vid));
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroInDegrees")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    dgl_type_t etype = args[1];
    IdArray vids = args[2];
    *rv = hg->InDegrees(etype, vids);
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroOutDegree")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    dgl_type_t etype = args[1];
    dgl_id_t vid = args[2];
    *rv = static_cast<int64_t>(hg->OutDegree(etype, vid));
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroOutDegrees")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    dgl_type_t etype = args[1];
    IdArray vids = args[2];
    *rv = hg->OutDegrees(etype, vids);
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroGetAdj")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    dgl_type_t etype = args[1];
    bool transpose = args[2];
    std::string fmt = args[3];
    *rv = ConvertNDArrayVectorToPackedFunc(
        hg->GetAdj(etype, transpose, fmt));
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroVertexSubgraph")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    List<Value> vids = args[1];
    std::vector<IdArray> vid_vec;
    vid_vec.reserve(vids.size());
    for (Value val : vids) {
      vid_vec.push_back(val->data);
    }
    std::shared_ptr<HeteroSubgraph> subg(
        new HeteroSubgraph(hg->VertexSubgraph(vid_vec)));
    *rv = HeteroSubgraphRef(subg);
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroEdgeSubgraph")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    List<Value> eids = args[1];
    bool preserve_nodes = args[2];
    std::vector<IdArray> eid_vec;
    eid_vec.reserve(eids.size());
    for (Value val : eids) {
      eid_vec.push_back(val->data);
    }
    std::shared_ptr<HeteroSubgraph> subg(
        new HeteroSubgraph(hg->EdgeSubgraph(eid_vec, preserve_nodes)));
    *rv = HeteroSubgraphRef(subg);
  });

// HeteroSubgraph C APIs

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroSubgraphGetGraph")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroSubgraphRef subg = args[0];
    *rv = HeteroGraphRef(subg->graph);
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroSubgraphGetInducedVertices")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroSubgraphRef subg = args[0];
    List<Value> induced_verts;
    for (IdArray arr : subg->induced_vertices) {
      induced_verts.push_back(Value(MakeValue(arr)));
    }
    *rv = induced_verts;
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroSubgraphGetInducedEdges")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroSubgraphRef subg = args[0];
    List<Value> induced_edges;
    for (IdArray arr : subg->induced_edges) {
      induced_edges.push_back(Value(MakeValue(arr)));
    }
    *rv = induced_edges;
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroAsNumBits")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    int bits = args[1];
    HeteroGraphPtr hg_new = UnitGraph::AsNumBits(hg.sptr(), bits);
    *rv = HeteroGraphRef(hg_new);
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroCopyTo")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    int device_type = args[1];
    int device_id = args[2];
    DLContext ctx;
    ctx.device_type = static_cast<DLDeviceType>(device_type);
    ctx.device_id = device_id;
    HeteroGraphPtr hg_new = UnitGraph::CopyTo(hg.sptr(), ctx);
    *rv = HeteroGraphRef(hg_new);
  });

}  // namespace dgl
