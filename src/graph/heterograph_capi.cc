/*!
 *  Copyright (c) 2020 by Contributors
 * \file graph/heterograph_capi.cc
 * \brief Heterograph CAPI bindings.
 */
#include "./heterograph.h"
#include <dgl/packed_func_ext.h>
#include <dgl/runtime/container.h>
#include "../c_api_common.h"

using namespace dgl::runtime;

namespace dgl {

///////////////////////// Unitgraph functions /////////////////////////

// XXX(minjie): Ideally, Unitgraph should be invisible to python side

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroCreateUnitGraphFromCOO")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    int64_t nvtypes = args[0];
    int64_t num_src = args[1];
    int64_t num_dst = args[2];
    IdArray row = args[3];
    IdArray col = args[4];
    SparseFormat restrict_format = ParseSparseFormat(args[5]);
    auto hgptr = CreateFromCOO(nvtypes, num_src, num_dst, row, col, restrict_format);
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
    SparseFormat restrict_format = ParseSparseFormat(args[6]);
    auto hgptr = CreateFromCSR(nvtypes, num_src, num_dst, indptr, indices, edge_ids,
                               restrict_format);
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

///////////////////////// HeteroGraph member functions /////////////////////////

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroGetMetaGraph")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    *rv = GraphRef(hg->meta_graph());
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroGetRelationGraph")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    dgl_type_t etype = args[1];
    CHECK_LE(etype, hg->NumEdgeTypes()) << "invalid edge type " << etype;
    // Test if the heterograph is a unit graph.  If so, return itself.
    auto bg = std::dynamic_pointer_cast<UnitGraph>(hg.sptr());
    if (bg != nullptr)
      *rv = bg;
    else
      *rv = HeteroGraphRef(hg->GetRelationGraph(etype));
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

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroDataType")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    *rv = hg->DataType();
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

///////////////////////// HeteroSubgraph members /////////////////////////

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

///////////////////////// Global functions and algorithms /////////////////////////

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

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroDisjointUnion")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    GraphRef meta_graph = args[0];
    List<HeteroGraphRef> component_graphs = args[1];
    std::vector<HeteroGraphPtr> component_ptrs;
    component_ptrs.reserve(component_graphs.size());
    for (const auto& component : component_graphs) {
      component_ptrs.push_back(component.sptr());
    }
    auto hgptr = DisjointUnionHeteroGraph(meta_graph.sptr(), component_ptrs);
    *rv = HeteroGraphRef(hgptr);
});

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroDisjointPartitionBySizes")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    const IdArray vertex_sizes = args[1];
    const IdArray edge_sizes = args[2];
    const auto& ret = DisjointPartitionHeteroBySizes(
      hg->meta_graph(), hg.sptr(), vertex_sizes, edge_sizes);
    List<HeteroGraphRef> ret_list;
    for (HeteroGraphPtr hgptr : ret) {
      ret_list.push_back(HeteroGraphRef(hgptr));
    }
    *rv = ret_list;
});

DGL_REGISTER_GLOBAL("transform._CAPI_DGLInSubgraph")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
    HeteroGraphRef hg = args[0];
    const auto& nodes = ListValueToVector<IdArray>(args[1]);
    std::shared_ptr<HeteroSubgraph> ret(new HeteroSubgraph);
    *ret = InEdgeGraph(hg.sptr(), nodes);
    *rv = HeteroGraphRef(ret);
  });

DGL_REGISTER_GLOBAL("transform._CAPI_DGLOutSubgraph")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
    HeteroGraphRef hg = args[0];
    const auto& nodes = ListValueToVector<IdArray>(args[1]);
    std::shared_ptr<HeteroSubgraph> ret(new HeteroSubgraph);
    *ret = OutEdgeGraph(hg.sptr(), nodes);
    *rv = HeteroGraphRef(ret);
  });

}  // namespace dgl
