/*!
 *  Copyright (c) 2018 by Contributors
 * \file graph/graph.cc
 * \brief DGL graph index APIs
 */
#include <dgl/graph.h>
#include <dgl/immutable_graph.h>
#include <dgl/graph_op.h>
#include <dgl/sampler.h>
#include <dgl/nodeflow.h>
#include "../c_api_common.h"

using dgl::runtime::DGLArgs;
using dgl::runtime::DGLArgValue;
using dgl::runtime::DGLRetValue;
using dgl::runtime::PackedFunc;
using dgl::runtime::NDArray;

namespace dgl {

namespace {
// Convert EdgeArray structure to PackedFunc.
template<class EdgeArray>
PackedFunc ConvertEdgeArrayToPackedFunc(const EdgeArray& ea) {
  auto body = [ea] (DGLArgs args, DGLRetValue* rv) {
      const int which = args[0];
      if (which == 0) {
        *rv = std::move(ea.src);
      } else if (which == 1) {
        *rv = std::move(ea.dst);
      } else if (which == 2) {
        *rv = std::move(ea.id);
      } else {
        LOG(FATAL) << "invalid choice";
      }
    };
  return PackedFunc(body);
}

// Convert CSRArray structure to PackedFunc.
PackedFunc ConvertAdjToPackedFunc(const std::vector<IdArray>& ea) {
  auto body = [ea] (DGLArgs args, DGLRetValue* rv) {
      const int which = args[0];
      if ((size_t) which < ea.size()) {
        *rv = std::move(ea[which]);
      } else {
        LOG(FATAL) << "invalid choice";
      }
    };
  return PackedFunc(body);
}

// Convert Subgraph structure to PackedFunc.
PackedFunc ConvertSubgraphToPackedFunc(const Subgraph& sg) {
  auto body = [sg] (DGLArgs args, DGLRetValue* rv) {
      const int which = args[0];
      if (which == 0) {
        *rv = sg.graph;
      } else if (which == 1) {
        *rv = std::move(sg.induced_vertices);
      } else if (which == 2) {
        *rv = std::move(sg.induced_edges);
      } else {
        LOG(FATAL) << "invalid choice";
      }
    };
  return PackedFunc(body);
}

}  // namespace

namespace {
// This namespace contains template functions for batching
// and unbatching over graph and immutable graph
template<typename T>
void DGLDisjointPartitionByNum(const T *gptr, DGLArgs args, DGLRetValue *rv) {
  int64_t num = args[1];
  std::vector<T> &&rst = GraphOp::DisjointPartitionByNum(gptr, num);
  // return the pointer array as an integer array
  const int64_t len = rst.size();
  NDArray ptr_array = NDArray::Empty({len}, DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
  int64_t *ptr_array_data = static_cast<int64_t *>(ptr_array->data);
  for (size_t i = 0; i < rst.size(); ++i) {
    GraphInterface *ptr = rst[i].Reset();
    ptr_array_data[i] = reinterpret_cast<std::intptr_t>(ptr);
  }
  *rv = ptr_array;
}

template<typename T>
void DGLDisjointUnion(GraphHandle *inhandles, int list_size, DGLRetValue *rv) {
  std::vector<const T *> graphs;
  for (int i = 0; i < list_size; ++i) {
    const GraphInterface *ptr = static_cast<const GraphInterface *>(inhandles[i]);
    const T *gr = dynamic_cast<const T *>(ptr);
    CHECK(gr) << "Error: Attempted to batch MutableGraph with ImmutableGraph";
    graphs.push_back(gr);
  }

  GraphHandle ghandle = GraphOp::DisjointUnion(std::move(graphs)).Reset();
  *rv = ghandle;
}

template<typename T>
void DGLDisjointPartitionBySizes(const T *gptr, const IdArray sizes, DGLRetValue *rv) {
  std::vector<T> &&rst = GraphOp::DisjointPartitionBySizes(gptr, sizes);
  // return the pointer array as an integer array
  const int64_t len = rst.size();
  NDArray ptr_array = NDArray::Empty({len}, DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
  int64_t *ptr_array_data = static_cast<int64_t *>(ptr_array->data);
  for (size_t i = 0; i < rst.size(); ++i) {
    GraphInterface *ptr = rst[i].Reset();
    ptr_array_data[i] = reinterpret_cast<std::intptr_t>(ptr);
  }
  *rv = ptr_array;
}
}  // namespace

///////////////////////////// Graph API ///////////////////////////////////

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphCreateMutable")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    bool multigraph = args[0];
    *rv = GraphRef::Create(multigraph);
  });


DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphCreate")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    const IdArray src_ids = args[0];
    const IdArray dst_ids = args[1];
    const int multigraph = args[2];
    const int64_t num_nodes = args[3];
    const bool readonly = args[4];
    if (readonly) {
      if (multigraph == kBoolUnknown) {
        *rv = ImmutableGraphRef::CreateFromCOO(num_nodes, src_ids, dst_ids);
      } else {
        *rv = ImmutableGraphRef::CreateFromCOO(num_nodes, src_ids, dst_ids, multigraph);
      }
    } else {
      CHECK_NE(multigraph, kBoolUnknown);
      *rv = GraphRef::CreateFromCOO(num_nodes, src_ids, dst_ids, multigraph);
    }
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphCSRCreate")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    const IdArray indptr = args[0];
    const IdArray indices = args[1];
    const std::string shared_mem_name = args[2];
    const int multigraph = args[3];
    const std::string edge_dir = args[4];

    IdArray edge_ids = IdArray::Empty({indices->shape[0]},
                                      DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
    int64_t *edge_data = static_cast<int64_t *>(edge_ids->data);
    for (size_t i = 0; i < edge_ids->shape[0]; i++)
      edge_data[i] = i;
    if (shared_mem_name.empty()) {
      if (multigraph == kBoolUnknown) {
        *rv = ImmutableGraphRef::CreateFromCSR(indptr, indices, edge_ids, edge_dir);
      } else {
        *rv = ImmutableGraphRef::CreateFromCSR(indptr, indices, edge_ids, multigraph, edge_dir);
      }
    } else {
      if (multigraph == kBoolUnknown) {
        *rv = ImmutableGraphRef::CreateFromCSR(
            indptr, indices, edge_ids, edge_dir, shared_mem_name);
      } else {
        *rv = ImmutableGraphRef::CreateFromCSR(indptr, indices, edge_ids,
            multigraph, edge_dir, shared_mem_name);
      }
    }
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphCSRCreateMMap")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    const std::string shared_mem_name = args[0];
    const int64_t num_vertices = args[1];
    const int64_t num_edges = args[2];
    const bool multigraph = args[3];
    const std::string edge_dir = args[4];
    // TODO(minjie): how to know multigraph
    *rv = ImmutableGraphRef::CreateFromCSR(
      shared_mem_name, num_vertices, num_edges, multigraph, edge_dir);
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphAddVertices")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    BaseGraphRef g = args[0];
    uint64_t num_vertices = args[1];
    g->AddVertices(num_vertices);
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphAddEdge")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    BaseGraphRef g = args[0];
    const dgl_id_t src = args[1];
    const dgl_id_t dst = args[2];
    g->AddEdge(src, dst);
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphAddEdges")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    BaseGraphRef g = args[0];
    const IdArray src = args[1];
    const IdArray dst = args[2];
    g->AddEdges(src, dst);
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphClear")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    BaseGraphRef g = args[0];
    g->Clear();
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphIsMultigraph")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
    BaseGraphRef g = args[0];
    *rv = g->IsMultigraph();
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphIsReadonly")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
    BaseGraphRef g = args[0];
    *rv = g->IsReadonly();
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphNumVertices")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    BaseGraphRef g = args[0];
    *rv = static_cast<int64_t>(g->NumVertices());
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphNumEdges")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    BaseGraphRef g = args[0];
    *rv = static_cast<int64_t>(g->NumEdges());
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphHasVertex")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    BaseGraphRef g = args[0];
    const dgl_id_t vid = args[1];
    *rv = g->HasVertex(vid);
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphHasVertices")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    BaseGraphRef g = args[0];
    const IdArray vids = args[1];
    *rv = g->HasVertices(vids);
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLMapSubgraphNID")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    const IdArray parent_vids = args[0];
    const IdArray query = args[1];
    *rv = GraphOp::MapParentIdToSubgraphId(parent_vids, query);
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphHasEdgeBetween")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    BaseGraphRef g = args[0];
    const dgl_id_t src = args[1];
    const dgl_id_t dst = args[2];
    *rv = g->HasEdgeBetween(src, dst);
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphHasEdgesBetween")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    BaseGraphRef g = args[0];
    const IdArray src = args[1];
    const IdArray dst = args[2];
    *rv = g->HasEdgesBetween(src, dst);
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphPredecessors")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    BaseGraphRef g = args[0];
    const dgl_id_t vid = args[1];
    const uint64_t radius = args[2];
    *rv = g->Predecessors(vid, radius);
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphSuccessors")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    BaseGraphRef g = args[0];
    const dgl_id_t vid = args[1];
    const uint64_t radius = args[2];
    *rv = g->Successors(vid, radius);
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphEdgeId")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    BaseGraphRef g = args[0];
    const dgl_id_t src = args[1];
    const dgl_id_t dst = args[2];
    *rv = g->EdgeId(src, dst);
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphEdgeIds")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    BaseGraphRef g = args[0];
    const IdArray src = args[1];
    const IdArray dst = args[2];
    *rv = ConvertEdgeArrayToPackedFunc(g->EdgeIds(src, dst));
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphFindEdges")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    BaseGraphRef g = args[0];
    const IdArray eids = args[1];
    *rv = ConvertEdgeArrayToPackedFunc(g->FindEdges(eids));
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphInEdges_1")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    BaseGraphRef g = args[0];
    const dgl_id_t vid = args[1];
    *rv = ConvertEdgeArrayToPackedFunc(g->InEdges(vid));
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphInEdges_2")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    BaseGraphRef g = args[0];
    const IdArray vids = args[1];
    *rv = ConvertEdgeArrayToPackedFunc(g->InEdges(vids));
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphOutEdges_1")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    BaseGraphRef g = args[0];
    const dgl_id_t vid = args[1];
    *rv = ConvertEdgeArrayToPackedFunc(g->OutEdges(vid));
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphOutEdges_2")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    BaseGraphRef g = args[0];
    const IdArray vids = args[1];
    *rv = ConvertEdgeArrayToPackedFunc(g->OutEdges(vids));
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphEdges")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    BaseGraphRef g = args[0];
    std::string order = args[1];
    *rv = ConvertEdgeArrayToPackedFunc(g->Edges(order));
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphInDegree")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    BaseGraphRef g = args[0];
    const dgl_id_t vid = args[1];
    *rv = static_cast<int64_t>(g->InDegree(vid));
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphInDegrees")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    BaseGraphRef g = args[0];
    const IdArray vids = args[1];
    *rv = g->InDegrees(vids);
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphOutDegree")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    BaseGraphRef g = args[0];
    const dgl_id_t vid = args[1];
    *rv = static_cast<int64_t>(g->OutDegree(vid));
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphOutDegrees")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    BaseGraphRef g = args[0];
    const IdArray vids = args[1];
    *rv = g->OutDegrees(vids);
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphVertexSubgraph")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    BaseGraphRef g = args[0];
    const IdArray vids = args[1];
    *rv = ConvertSubgraphToPackedFunc(g->VertexSubgraph(vids));
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphEdgeSubgraph")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    BaseGraphRef g = args[0];
    const IdArray eids = args[1];
    bool preserve_nodes = args[2];
    *rv = ConvertSubgraphToPackedFunc(g->EdgeSubgraph(eids, preserve_nodes));
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphGetAdj")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    GraphRef g = args[0];
    bool transpose = args[1];
    std::string format = args[2];
    auto res = g->GetAdj(transpose, format);
    *rv = ConvertAdjToPackedFunc(res);
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphContext")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    GraphRef g = args[0];
    *rv = g->Context();
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphNumBits")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    GraphRef g = args[0];
    *rv = g->NumBits();
  });

}  // namespace dgl
