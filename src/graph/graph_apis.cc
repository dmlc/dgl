/*!
 *  Copyright (c) 2018 by Contributors
 * \file graph/graph.cc
 * \brief DGL graph index APIs
 */
#include <dgl/graph.h>
#include <dgl/immutable_graph.h>
#include <dgl/graph_op.h>
#include <dgl/sampler.h>
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

// Convert Sampled Subgraph structures to PackedFunc.
PackedFunc ConvertSubgraphToPackedFunc(const std::vector<SampledSubgraph>& sg) {
  auto body = [sg] (DGLArgs args, DGLRetValue* rv) {
      const int which = args[0];
      if (which < static_cast<int>(sg.size())) {
        GraphInterface* gptr = sg[which].graph->Reset();
        GraphHandle ghandle = gptr;
        *rv = ghandle;
      } else if (which >= static_cast<int>(sg.size()) && which < static_cast<int>(sg.size()) * 2) {
        *rv = std::move(sg[which - sg.size()].induced_vertices);
      } else if (which >= static_cast<int>(sg.size()) * 2
                 && which < static_cast<int>(sg.size()) * 3) {
        *rv = std::move(sg[which - sg.size() * 2].induced_edges);
      } else if (which >= static_cast<int>(sg.size()) * 3
                 && which < static_cast<int>(sg.size()) * 4) {
        *rv = std::move(sg[which - sg.size() * 3].layer_offsets);
      } else if (which >= static_cast<int>(sg.size()) * 4
                 && which < static_cast<int>(sg.size()) * 5) {
        *rv = std::move(sg[which - sg.size() * 4].sample_prob);
      } else {
        LOG(FATAL) << "invalid choice";
      }
    };
  // TODO(minjie): figure out a better way of returning a complex results.
  return PackedFunc(body);
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
    const IdArray src_ids = IdArray::FromDLPack(CreateTmpDLManagedTensor(args[0]));
    const IdArray dst_ids = IdArray::FromDLPack(CreateTmpDLManagedTensor(args[1]));
    const IdArray edge_ids = IdArray::FromDLPack(CreateTmpDLManagedTensor(args[2]));
    const bool multigraph = static_cast<bool>(args[3]);
    const int64_t num_nodes = static_cast<int64_t>(args[4]);
    const bool readonly = static_cast<bool>(args[5]);
    GraphHandle ghandle;
    if (readonly)
      ghandle = new ImmutableGraph(src_ids, dst_ids, edge_ids, num_nodes, multigraph);
    else
      ghandle = new Graph(src_ids, dst_ids, edge_ids, num_nodes, multigraph);
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
    const IdArray src = IdArray::FromDLPack(CreateTmpDLManagedTensor(args[1]));
    const IdArray dst = IdArray::FromDLPack(CreateTmpDLManagedTensor(args[2]));
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
    const IdArray vids = IdArray::FromDLPack(CreateTmpDLManagedTensor(args[1]));
    *rv = gptr->HasVertices(vids);
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLMapSubgraphNID")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    const IdArray parent_vids = IdArray::FromDLPack(CreateTmpDLManagedTensor(args[0]));
    const IdArray query = IdArray::FromDLPack(CreateTmpDLManagedTensor(args[1]));
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
    const IdArray src = IdArray::FromDLPack(CreateTmpDLManagedTensor(args[1]));
    const IdArray dst = IdArray::FromDLPack(CreateTmpDLManagedTensor(args[2]));
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
    const IdArray src = IdArray::FromDLPack(CreateTmpDLManagedTensor(args[1]));
    const IdArray dst = IdArray::FromDLPack(CreateTmpDLManagedTensor(args[2]));
    *rv = ConvertEdgeArrayToPackedFunc(gptr->EdgeIds(src, dst));
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphFindEdges")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    GraphHandle ghandle = args[0];
    const GraphInterface* gptr = static_cast<GraphInterface*>(ghandle);
    const IdArray eids = IdArray::FromDLPack(CreateTmpDLManagedTensor(args[1]));
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
    const IdArray vids = IdArray::FromDLPack(CreateTmpDLManagedTensor(args[1]));
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
    const IdArray vids = IdArray::FromDLPack(CreateTmpDLManagedTensor(args[1]));
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
    const IdArray vids = IdArray::FromDLPack(CreateTmpDLManagedTensor(args[1]));
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
    const IdArray vids = IdArray::FromDLPack(CreateTmpDLManagedTensor(args[1]));
    *rv = gptr->OutDegrees(vids);
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphVertexSubgraph")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    GraphHandle ghandle = args[0];
    const GraphInterface* gptr = static_cast<GraphInterface*>(ghandle);
    const IdArray vids = IdArray::FromDLPack(CreateTmpDLManagedTensor(args[1]));
    *rv = ConvertSubgraphToPackedFunc(gptr->VertexSubgraph(vids));
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphEdgeSubgraph")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    GraphHandle ghandle = args[0];
    const GraphInterface *gptr = static_cast<GraphInterface*>(ghandle);
    const IdArray eids = IdArray::FromDLPack(CreateTmpDLManagedTensor(args[1]));
    *rv = ConvertSubgraphToPackedFunc(gptr->EdgeSubgraph(eids));
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLDisjointUnion")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    void* list = args[0];
    GraphHandle* inhandles = static_cast<GraphHandle*>(list);
    int list_size = args[1];
    std::vector<const Graph*> graphs;
    for (int i = 0; i < list_size; ++i) {
      const GraphInterface *ptr = static_cast<const GraphInterface *>(inhandles[i]);
      const Graph* gr = dynamic_cast<const Graph*>(ptr);
      CHECK(gr) << "_CAPI_DGLDisjointUnion isn't implemented in immutable graph";
      graphs.push_back(gr);
    }
    Graph* gptr = new Graph();
    *gptr = GraphOp::DisjointUnion(std::move(graphs));
    GraphHandle ghandle = gptr;
    *rv = ghandle;
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLDisjointPartitionByNum")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    GraphHandle ghandle = args[0];
    const GraphInterface *ptr = static_cast<const GraphInterface *>(ghandle);
    const Graph* gptr = dynamic_cast<const Graph*>(ptr);
    CHECK(gptr) << "_CAPI_DGLDisjointPartitionByNum isn't implemented in immutable graph";
    int64_t num = args[1];
    std::vector<Graph>&& rst = GraphOp::DisjointPartitionByNum(gptr, num);
    // return the pointer array as an integer array
    const int64_t len = rst.size();
    NDArray ptr_array = NDArray::Empty({len}, DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
    int64_t* ptr_array_data = static_cast<int64_t*>(ptr_array->data);
    for (size_t i = 0; i < rst.size(); ++i) {
      Graph* ptr = new Graph();
      *ptr = std::move(rst[i]);
      ptr_array_data[i] = reinterpret_cast<std::intptr_t>(ptr);
    }
    *rv = ptr_array;
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLDisjointPartitionBySizes")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    GraphHandle ghandle = args[0];
    const GraphInterface *ptr = static_cast<const GraphInterface *>(ghandle);
    const Graph* gptr = dynamic_cast<const Graph*>(ptr);
    CHECK(gptr) << "_CAPI_DGLDisjointPartitionBySizes isn't implemented in immutable graph";
    const IdArray sizes = IdArray::FromDLPack(CreateTmpDLManagedTensor(args[1]));
    std::vector<Graph>&& rst = GraphOp::DisjointPartitionBySizes(gptr, sizes);
    // return the pointer array as an integer array
    const int64_t len = rst.size();
    NDArray ptr_array = NDArray::Empty({len}, DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
    int64_t* ptr_array_data = static_cast<int64_t*>(ptr_array->data);
    for (size_t i = 0; i < rst.size(); ++i) {
      Graph* ptr = new Graph();
      *ptr = std::move(rst[i]);
      ptr_array_data[i] = reinterpret_cast<std::intptr_t>(ptr);
    }
    *rv = ptr_array;
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

template<int num_seeds>
void CAPI_NeighborUniformSample(DGLArgs args, DGLRetValue* rv) {
  GraphHandle ghandle = args[0];
  std::vector<IdArray> seeds(num_seeds);
  for (size_t i = 0; i < seeds.size(); i++)
    seeds[i] = IdArray::FromDLPack(CreateTmpDLManagedTensor(args[i + 1]));
  std::string neigh_type = args[num_seeds + 1];
  const int num_hops = args[num_seeds + 2];
  const int num_neighbors = args[num_seeds + 3];
  const int num_valid_seeds = args[num_seeds + 4];
  const GraphInterface *ptr = static_cast<const GraphInterface *>(ghandle);
  const ImmutableGraph *gptr = dynamic_cast<const ImmutableGraph*>(ptr);
  CHECK(gptr) << "sampling isn't implemented in mutable graph";
  CHECK(num_valid_seeds <= num_seeds);
  std::vector<SampledSubgraph> subgs(seeds.size());
#pragma omp parallel for
  for (int i = 0; i < num_valid_seeds; i++) {
    subgs[i] = gptr->NeighborUniformSample(seeds[i], neigh_type, num_hops, num_neighbors);
  }
  *rv = ConvertSubgraphToPackedFunc(subgs);
}

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphUniformSampling")
.set_body(CAPI_NeighborUniformSample<1>);
DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphUniformSampling2")
.set_body(CAPI_NeighborUniformSample<2>);
DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphUniformSampling4")
.set_body(CAPI_NeighborUniformSample<4>);
DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphUniformSampling8")
.set_body(CAPI_NeighborUniformSample<8>);
DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphUniformSampling16")
.set_body(CAPI_NeighborUniformSample<16>);
DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphUniformSampling32")
.set_body(CAPI_NeighborUniformSample<32>);
DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphUniformSampling64")
.set_body(CAPI_NeighborUniformSample<64>);
DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphUniformSampling128")
.set_body(CAPI_NeighborUniformSample<128>);

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphGetAdj")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    GraphHandle ghandle = args[0];
    bool transpose = args[1];
    std::string format = args[2];
    const GraphInterface *ptr = static_cast<const GraphInterface *>(ghandle);
    auto res = ptr->GetAdj(transpose, format);
    *rv = ConvertAdjToPackedFunc(res);
  });

}  // namespace dgl
