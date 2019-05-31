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
        GraphInterface* gptr = sg.graph->Reset();
        GraphHandle ghandle = gptr;
        *rv = ghandle;
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
    bool multigraph = static_cast<bool>(args[0]);
    GraphHandle ghandle = new Graph(multigraph);
    *rv = ghandle;
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphCreate")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    const IdArray src_ids = args[0];
    const IdArray dst_ids = args[1];
    const bool multigraph = args[2];
    const int64_t num_nodes = args[3];
    const bool readonly = args[4];
    GraphHandle ghandle;
    if (readonly) {
      // TODO(minjie): The array copy here is unnecessary and adds extra overhead.
      //   However, with MXNet backend, the memory would be corrupted if we directly
      //   save the passed-in ndarrays into DGL's graph object. We hope MXNet team
      //   could help look into this.
      COOPtr coo(new COO(num_nodes, Clone(src_ids), Clone(dst_ids), multigraph));
      ghandle = new ImmutableGraph(coo);
    } else {
      ghandle = new Graph(src_ids, dst_ids, num_nodes, multigraph);
    }
    *rv = ghandle;
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphCSRCreate")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    const IdArray indptr = args[0];
    const IdArray indices = args[1];
    const std::string shared_mem_name = args[2];
    const bool multigraph = static_cast<bool>(args[3]);
    const std::string edge_dir = args[4];
    CSRPtr csr;

    IdArray edge_ids = IdArray::Empty({indices->shape[0]},
                                      DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
    int64_t *edge_data = static_cast<int64_t *>(edge_ids->data);
    for (size_t i = 0; i < edge_ids->shape[0]; i++)
      edge_data[i] = i;
    if (shared_mem_name.empty())
      // TODO(minjie): The array copy here is unnecessary and adds extra overhead.
      //   However, with MXNet backend, the memory would be corrupted if we directly
      //   save the passed-in ndarrays into DGL's graph object. We hope MXNet team
      //   could help look into this.
      csr.reset(new CSR(Clone(indptr), Clone(indices), Clone(edge_ids), multigraph));
    else
      csr.reset(new CSR(indptr, indices, edge_ids, multigraph, shared_mem_name));

    GraphHandle ghandle;
    if (edge_dir == "in")
      ghandle = new ImmutableGraph(csr, nullptr);
    else
      ghandle = new ImmutableGraph(nullptr, csr);
    *rv = ghandle;
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphCSRCreateMMap")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    const std::string shared_mem_name = args[0];
    const int64_t num_vertices = args[1];
    const int64_t num_edges = args[2];
    const bool multigraph = args[3];
    const std::string edge_dir = args[4];
    // TODO(minjie): how to know multigraph
    CSRPtr csr(new CSR(shared_mem_name, num_vertices, num_edges, multigraph));
    GraphHandle ghandle;
    if (edge_dir == "in")
      ghandle = new ImmutableGraph(csr, nullptr);
    else
      ghandle = new ImmutableGraph(nullptr, csr);
    *rv = ghandle;
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphFree")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    GraphHandle ghandle = args[0];
    GraphInterface* gptr = static_cast<GraphInterface*>(ghandle);
    delete gptr;
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphAddVertices")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    GraphHandle ghandle = args[0];
    GraphInterface* gptr = static_cast<GraphInterface*>(ghandle);
    uint64_t num_vertices = args[1];
    gptr->AddVertices(num_vertices);
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphAddEdge")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    GraphHandle ghandle = args[0];
    GraphInterface* gptr = static_cast<GraphInterface*>(ghandle);
    const dgl_id_t src = args[1];
    const dgl_id_t dst = args[2];
    gptr->AddEdge(src, dst);
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphAddEdges")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    GraphHandle ghandle = args[0];
    GraphInterface* gptr = static_cast<GraphInterface*>(ghandle);
    const IdArray src = args[1];
    const IdArray dst = args[2];
    gptr->AddEdges(src, dst);
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphClear")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    GraphHandle ghandle = args[0];
    GraphInterface* gptr = static_cast<GraphInterface*>(ghandle);
    gptr->Clear();
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphIsMultigraph")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
    GraphHandle ghandle = args[0];
    // NOTE: not const since we have caches
    const GraphInterface* gptr = static_cast<GraphInterface*>(ghandle);
    *rv = gptr->IsMultigraph();
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphIsReadonly")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
    GraphHandle ghandle = args[0];
    // NOTE: not const since we have caches
    const GraphInterface* gptr = static_cast<GraphInterface*>(ghandle);
    *rv = gptr->IsReadonly();
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphNumVertices")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    GraphHandle ghandle = args[0];
    const GraphInterface* gptr = static_cast<GraphInterface*>(ghandle);
    *rv = static_cast<int64_t>(gptr->NumVertices());
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphNumEdges")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    GraphHandle ghandle = args[0];
    const GraphInterface* gptr = static_cast<GraphInterface*>(ghandle);
    *rv = static_cast<int64_t>(gptr->NumEdges());
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphHasVertex")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    GraphHandle ghandle = args[0];
    const GraphInterface* gptr = static_cast<GraphInterface*>(ghandle);
    const dgl_id_t vid = args[1];
    *rv = gptr->HasVertex(vid);
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphHasVertices")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    GraphHandle ghandle = args[0];
    const GraphInterface* gptr = static_cast<GraphInterface*>(ghandle);
    const IdArray vids = args[1];
    *rv = gptr->HasVertices(vids);
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLMapSubgraphNID")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    const IdArray parent_vids = args[0];
    const IdArray query = args[1];
    *rv = GraphOp::MapParentIdToSubgraphId(parent_vids, query);
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphHasEdgeBetween")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    GraphHandle ghandle = args[0];
    const GraphInterface* gptr = static_cast<GraphInterface*>(ghandle);
    const dgl_id_t src = args[1];
    const dgl_id_t dst = args[2];
    *rv = gptr->HasEdgeBetween(src, dst);
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphHasEdgesBetween")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    GraphHandle ghandle = args[0];
    const GraphInterface* gptr = static_cast<GraphInterface*>(ghandle);
    const IdArray src = args[1];
    const IdArray dst = args[2];
    *rv = gptr->HasEdgesBetween(src, dst);
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphPredecessors")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    GraphHandle ghandle = args[0];
    const GraphInterface* gptr = static_cast<GraphInterface*>(ghandle);
    const dgl_id_t vid = args[1];
    const uint64_t radius = args[2];
    *rv = gptr->Predecessors(vid, radius);
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphSuccessors")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    GraphHandle ghandle = args[0];
    const GraphInterface* gptr = static_cast<GraphInterface*>(ghandle);
    const dgl_id_t vid = args[1];
    const uint64_t radius = args[2];
    *rv = gptr->Successors(vid, radius);
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphEdgeId")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    GraphHandle ghandle = args[0];
    const GraphInterface* gptr = static_cast<GraphInterface*>(ghandle);
    const dgl_id_t src = args[1];
    const dgl_id_t dst = args[2];
    *rv = gptr->EdgeId(src, dst);
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphEdgeIds")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    GraphHandle ghandle = args[0];
    const GraphInterface* gptr = static_cast<GraphInterface*>(ghandle);
    const IdArray src = args[1];
    const IdArray dst = args[2];
    *rv = ConvertEdgeArrayToPackedFunc(gptr->EdgeIds(src, dst));
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphFindEdges")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    GraphHandle ghandle = args[0];
    const GraphInterface* gptr = static_cast<GraphInterface*>(ghandle);
    const IdArray eids = args[1];
    *rv = ConvertEdgeArrayToPackedFunc(gptr->FindEdges(eids));
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphInEdges_1")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    GraphHandle ghandle = args[0];
    const GraphInterface* gptr = static_cast<GraphInterface*>(ghandle);
    const dgl_id_t vid = args[1];
    *rv = ConvertEdgeArrayToPackedFunc(gptr->InEdges(vid));
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphInEdges_2")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    GraphHandle ghandle = args[0];
    const GraphInterface* gptr = static_cast<GraphInterface*>(ghandle);
    const IdArray vids = args[1];
    *rv = ConvertEdgeArrayToPackedFunc(gptr->InEdges(vids));
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphOutEdges_1")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    GraphHandle ghandle = args[0];
    const GraphInterface* gptr = static_cast<GraphInterface*>(ghandle);
    const dgl_id_t vid = args[1];
    *rv = ConvertEdgeArrayToPackedFunc(gptr->OutEdges(vid));
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphOutEdges_2")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    GraphHandle ghandle = args[0];
    const GraphInterface* gptr = static_cast<GraphInterface*>(ghandle);
    const IdArray vids = args[1];
    *rv = ConvertEdgeArrayToPackedFunc(gptr->OutEdges(vids));
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphEdges")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    GraphHandle ghandle = args[0];
    const GraphInterface* gptr = static_cast<GraphInterface*>(ghandle);
    std::string order = args[1];
    *rv = ConvertEdgeArrayToPackedFunc(gptr->Edges(order));
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphInDegree")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    GraphHandle ghandle = args[0];
    const GraphInterface* gptr = static_cast<GraphInterface*>(ghandle);
    const dgl_id_t vid = args[1];
    *rv = static_cast<int64_t>(gptr->InDegree(vid));
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphInDegrees")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    GraphHandle ghandle = args[0];
    const GraphInterface* gptr = static_cast<GraphInterface*>(ghandle);
    const IdArray vids = args[1];
    *rv = gptr->InDegrees(vids);
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphOutDegree")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    GraphHandle ghandle = args[0];
    const GraphInterface* gptr = static_cast<GraphInterface*>(ghandle);
    const dgl_id_t vid = args[1];
    *rv = static_cast<int64_t>(gptr->OutDegree(vid));
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphOutDegrees")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    GraphHandle ghandle = args[0];
    const GraphInterface* gptr = static_cast<GraphInterface*>(ghandle);
    const IdArray vids = args[1];
    *rv = gptr->OutDegrees(vids);
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphVertexSubgraph")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    GraphHandle ghandle = args[0];
    const GraphInterface* gptr = static_cast<GraphInterface*>(ghandle);
    const IdArray vids = args[1];
    *rv = ConvertSubgraphToPackedFunc(gptr->VertexSubgraph(vids));
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphEdgeSubgraph")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    GraphHandle ghandle = args[0];
    const GraphInterface *gptr = static_cast<GraphInterface*>(ghandle);
    const IdArray eids = args[1];
    *rv = ConvertSubgraphToPackedFunc(gptr->EdgeSubgraph(eids));
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLDisjointUnion")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    void* list = args[0];
    GraphHandle* inhandles = static_cast<GraphHandle*>(list);
    int list_size = args[1];
    const GraphInterface *ptr = static_cast<const GraphInterface *>(inhandles[0]);
    const ImmutableGraph *im_gr = dynamic_cast<const ImmutableGraph *>(ptr);
    const Graph *gr = dynamic_cast<const Graph *>(ptr);
    if (gr) {
      DGLDisjointUnion<Graph>(inhandles, list_size, rv);
    } else {
      CHECK(im_gr) << "Args[0] is not a list of valid DGLGraph";
      DGLDisjointUnion<ImmutableGraph>(inhandles, list_size, rv);
    }
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLDisjointPartitionByNum")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    GraphHandle ghandle = args[0];
    const GraphInterface *ptr = static_cast<const GraphInterface *>(ghandle);
    const Graph* gptr = dynamic_cast<const Graph*>(ptr);
    const ImmutableGraph* im_gptr = dynamic_cast<const ImmutableGraph*>(ptr);
    if (gptr) {
      DGLDisjointPartitionByNum(gptr, args, rv);
    } else {
      CHECK(im_gptr) << "Args[0] is not a valid DGLGraph";
      DGLDisjointPartitionByNum(im_gptr, args, rv);
    }
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLDisjointPartitionBySizes")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    GraphHandle ghandle = args[0];
    const IdArray sizes = args[1];
    const GraphInterface *ptr = static_cast<const GraphInterface *>(ghandle);
    const Graph* gptr = dynamic_cast<const Graph*>(ptr);
    const ImmutableGraph* im_gptr = dynamic_cast<const ImmutableGraph*>(ptr);
    if (gptr) {
      DGLDisjointPartitionBySizes(gptr, sizes, rv);
    } else {
      CHECK(im_gptr) << "Args[0] is not a valid DGLGraph";
      DGLDisjointPartitionBySizes(im_gptr, sizes, rv);
    }
});

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphLineGraph")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    GraphHandle ghandle = args[0];
    bool backtracking = args[1];
    const GraphInterface *ptr = static_cast<const GraphInterface *>(ghandle);
    const Graph* gptr = dynamic_cast<const Graph*>(ptr);
    CHECK(gptr) << "_CAPI_DGLGraphLineGraph isn't implemented in immutable graph";
    Graph* lgptr = new Graph();
    *lgptr = GraphOp::LineGraph(gptr, backtracking);
    GraphHandle lghandle = lgptr;
    *rv = lghandle;
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphGetAdj")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    GraphHandle ghandle = args[0];
    bool transpose = args[1];
    std::string format = args[2];
    const GraphInterface *ptr = static_cast<const GraphInterface *>(ghandle);
    auto res = ptr->GetAdj(transpose, format);
    *rv = ConvertAdjToPackedFunc(res);
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLToImmutable")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    GraphHandle ghandle = args[0];
    const GraphInterface *ptr = static_cast<GraphInterface *>(ghandle);
    GraphHandle newhandle = new ImmutableGraph(ImmutableGraph::ToImmutable(ptr));
    *rv = newhandle;
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphContext")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    GraphHandle ghandle = args[0];
    const GraphInterface *ptr = static_cast<GraphInterface *>(ghandle);
    *rv = ptr->Context();
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLImmutableGraphCopyTo")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    GraphHandle ghandle = args[0];
    const int device_type = args[1];
    const int device_id = args[2];
    DLContext ctx;
    ctx.device_type = static_cast<DLDeviceType>(device_type);
    ctx.device_id = device_id;
    const GraphInterface *ptr = static_cast<GraphInterface *>(ghandle);
    const ImmutableGraph *ig = dynamic_cast<const ImmutableGraph*>(ptr);
    CHECK(ig) << "Invalid argument: must be an immutable graph object.";
    GraphHandle newhandle = new ImmutableGraph(ig->CopyTo(ctx));
    *rv = newhandle;
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphNumBits")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    GraphHandle ghandle = args[0];
    const GraphInterface *ptr = static_cast<GraphInterface *>(ghandle);
    *rv = ptr->NumBits();
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLImmutableGraphAsNumBits")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    GraphHandle ghandle = args[0];
    int bits = args[1];
    const GraphInterface *ptr = static_cast<GraphInterface *>(ghandle);
    const ImmutableGraph *ig = dynamic_cast<const ImmutableGraph*>(ptr);
    CHECK(ig) << "Invalid argument: must be an immutable graph object.";
    GraphHandle newhandle = new ImmutableGraph(ig->AsNumBits(bits));
    *rv = newhandle;
  });

}  // namespace dgl
