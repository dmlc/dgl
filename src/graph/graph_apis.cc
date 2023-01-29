/**
 *  Copyright (c) 2018 by Contributors
 * @file graph/graph.cc
 * @brief DGL graph index APIs
 */
#include <dgl/graph.h>
#include <dgl/graph_op.h>
#include <dgl/immutable_graph.h>
#include <dgl/nodeflow.h>
#include <dgl/packed_func_ext.h>
#include <dgl/sampler.h>

#include "../c_api_common.h"

using dgl::runtime::DGLArgs;
using dgl::runtime::DGLArgValue;
using dgl::runtime::DGLRetValue;
using dgl::runtime::NDArray;
using dgl::runtime::PackedFunc;

namespace dgl {

///////////////////////////// Graph API ///////////////////////////////////

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphCreateMutable")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      *rv = GraphRef(Graph::Create());
    });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphCreate")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      const IdArray src_ids = args[0];
      const IdArray dst_ids = args[1];
      const int64_t num_nodes = args[2];
      const bool readonly = args[3];
      if (readonly) {
        *rv = GraphRef(
            ImmutableGraph::CreateFromCOO(num_nodes, src_ids, dst_ids));
      } else {
        *rv = GraphRef(Graph::CreateFromCOO(num_nodes, src_ids, dst_ids));
      }
    });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphCSRCreate")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      const IdArray indptr = args[0];
      const IdArray indices = args[1];
      const std::string edge_dir = args[2];

      IdArray edge_ids = IdArray::Empty(
          {indices->shape[0]}, DGLDataType{kDGLInt, 64, 1},
          DGLContext{kDGLCPU, 0});
      int64_t* edge_data = static_cast<int64_t*>(edge_ids->data);
      for (int64_t i = 0; i < edge_ids->shape[0]; i++) edge_data[i] = i;
      *rv = GraphRef(
          ImmutableGraph::CreateFromCSR(indptr, indices, edge_ids, edge_dir));
    });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphCSRCreateMMap")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      const std::string shared_mem_name = args[0];
      *rv = GraphRef(ImmutableGraph::CreateFromCSR(shared_mem_name));
    });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphAddVertices")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      GraphRef g = args[0];
      uint64_t num_vertices = args[1];
      g->AddVertices(num_vertices);
    });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphAddEdge")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      GraphRef g = args[0];
      const dgl_id_t src = args[1];
      const dgl_id_t dst = args[2];
      g->AddEdge(src, dst);
    });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphAddEdges")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      GraphRef g = args[0];
      const IdArray src = args[1];
      const IdArray dst = args[2];
      g->AddEdges(src, dst);
    });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphClear")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      GraphRef g = args[0];
      g->Clear();
    });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphIsMultigraph")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      GraphRef g = args[0];
      *rv = g->IsMultigraph();
    });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphIsReadonly")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      GraphRef g = args[0];
      *rv = g->IsReadonly();
    });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphNumVertices")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      GraphRef g = args[0];
      *rv = static_cast<int64_t>(g->NumVertices());
    });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphNumEdges")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      GraphRef g = args[0];
      *rv = static_cast<int64_t>(g->NumEdges());
    });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphHasVertex")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      GraphRef g = args[0];
      const dgl_id_t vid = args[1];
      *rv = g->HasVertex(vid);
    });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphHasVertices")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      GraphRef g = args[0];
      const IdArray vids = args[1];
      *rv = g->HasVertices(vids);
    });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphHasEdgeBetween")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      GraphRef g = args[0];
      const dgl_id_t src = args[1];
      const dgl_id_t dst = args[2];
      *rv = g->HasEdgeBetween(src, dst);
    });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphHasEdgesBetween")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      GraphRef g = args[0];
      const IdArray src = args[1];
      const IdArray dst = args[2];
      *rv = g->HasEdgesBetween(src, dst);
    });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphPredecessors")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      GraphRef g = args[0];
      const dgl_id_t vid = args[1];
      const uint64_t radius = args[2];
      *rv = g->Predecessors(vid, radius);
    });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphSuccessors")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      GraphRef g = args[0];
      const dgl_id_t vid = args[1];
      const uint64_t radius = args[2];
      *rv = g->Successors(vid, radius);
    });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphEdgeId")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      GraphRef g = args[0];
      const dgl_id_t src = args[1];
      const dgl_id_t dst = args[2];
      *rv = g->EdgeId(src, dst);
    });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphEdgeIds")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      GraphRef g = args[0];
      const IdArray src = args[1];
      const IdArray dst = args[2];
      *rv = ConvertEdgeArrayToPackedFunc(g->EdgeIds(src, dst));
    });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphFindEdge")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      GraphRef g = args[0];
      const dgl_id_t eid = args[1];
      const auto& pair = g->FindEdge(eid);
      *rv = PackedFunc([pair](DGLArgs args, DGLRetValue* rv) {
        const int choice = args[0];
        const int64_t ret = (choice == 0 ? pair.first : pair.second);
        *rv = ret;
      });
    });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphFindEdges")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      GraphRef g = args[0];
      const IdArray eids = args[1];
      *rv = ConvertEdgeArrayToPackedFunc(g->FindEdges(eids));
    });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphInEdges_1")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      GraphRef g = args[0];
      const dgl_id_t vid = args[1];
      *rv = ConvertEdgeArrayToPackedFunc(g->InEdges(vid));
    });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphInEdges_2")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      GraphRef g = args[0];
      const IdArray vids = args[1];
      *rv = ConvertEdgeArrayToPackedFunc(g->InEdges(vids));
    });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphOutEdges_1")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      GraphRef g = args[0];
      const dgl_id_t vid = args[1];
      *rv = ConvertEdgeArrayToPackedFunc(g->OutEdges(vid));
    });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphOutEdges_2")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      GraphRef g = args[0];
      const IdArray vids = args[1];
      *rv = ConvertEdgeArrayToPackedFunc(g->OutEdges(vids));
    });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphEdges")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      GraphRef g = args[0];
      std::string order = args[1];
      *rv = ConvertEdgeArrayToPackedFunc(g->Edges(order));
    });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphInDegree")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      GraphRef g = args[0];
      const dgl_id_t vid = args[1];
      *rv = static_cast<int64_t>(g->InDegree(vid));
    });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphInDegrees")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      GraphRef g = args[0];
      const IdArray vids = args[1];
      *rv = g->InDegrees(vids);
    });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphOutDegree")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      GraphRef g = args[0];
      const dgl_id_t vid = args[1];
      *rv = static_cast<int64_t>(g->OutDegree(vid));
    });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphOutDegrees")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      GraphRef g = args[0];
      const IdArray vids = args[1];
      *rv = g->OutDegrees(vids);
    });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphVertexSubgraph")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      GraphRef g = args[0];
      const IdArray vids = args[1];
      std::shared_ptr<Subgraph> subg(new Subgraph(g->VertexSubgraph(vids)));
      *rv = SubgraphRef(subg);
    });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphEdgeSubgraph")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      GraphRef g = args[0];
      const IdArray eids = args[1];
      bool preserve_nodes = args[2];
      std::shared_ptr<Subgraph> subg(
          new Subgraph(g->EdgeSubgraph(eids, preserve_nodes)));
      *rv = SubgraphRef(subg);
    });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphGetAdj")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      GraphRef g = args[0];
      bool transpose = args[1];
      std::string format = args[2];
      auto res = g->GetAdj(transpose, format);
      *rv = ConvertNDArrayVectorToPackedFunc(res);
    });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphContext")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      GraphRef g = args[0];
      *rv = g->Context();
    });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphNumBits")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      GraphRef g = args[0];
      *rv = g->NumBits();
    });

// Subgraph C APIs

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLSubgraphGetGraph")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      SubgraphRef subg = args[0];
      *rv = GraphRef(subg->graph);
    });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLSubgraphGetInducedVertices")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      SubgraphRef subg = args[0];
      *rv = subg->induced_vertices;
    });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLSubgraphGetInducedEdges")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      SubgraphRef subg = args[0];
      *rv = subg->induced_edges;
    });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLSortAdj")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      GraphRef g = args[0];
      g->SortCSR();
    });

}  // namespace dgl
