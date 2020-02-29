/*!
 *  Copyright (c) 2020 by Contributors
 * \file graph/pickle.cc
 * \brief Functions for pickle and unpickle a graph
 */
#include "./heterograph.h"
#include <dgl/packed_func_ext.h>
#include <dgl/runtime/container.h>
#include "../c_api_common.h"

using namespace dgl::runtime;

namespace dgl {

HeteroPickleStates HeteroPickle(HeteroGraphPtr graph) {
  // format array is always in int32
  IdArray fmts = aten::NewIdArray(graph->NumEdgeTypes(), DLContext{kDLCPU, 0}, 32);
  int32_t* fmts_data = static_cast<int32_t*>(fmts->data);
  std::vector<IdArray> adjs;
  for (dgl_type_t etype = 0; etype < graph->NumEdgeTypes(); ++etype) {
    SparseFormat fmt = graph->SelectFormat(etype, SparseFormat::ANY);
    fmts_data[etype] = static_cast<int32_t>(fmt);
    switch (fmt) {
      case SparseFormat::COO:
        {
          const auto& adj = graph->GetCOOMatrix(etype);
          adjs.push_back(adj.row);
          adjs.push_back(adj.col);
          adjs.push_back(adj.data);
          break;
        }
      case SparseFormat::CSR:
        {
          const auto& adj = graph->GetCSRMatrix(etype);
          adjs.push_back(adj.indptr);
          adjs.push_back(adj.indices);
          adjs.push_back(adj.data);
          break;
        }
      case SparseFormat::CSC:
        {
          const auto& adj = graph->GetCSCMatrix(etype);
          adjs.push_back(adj.indptr);
          adjs.push_back(adj.indices);
          adjs.push_back(adj.data);
          break;
        }
      default:
        LOG(FATAL) << "Unsupported sparse format.";
    }
  }
  IdArray num_nodes = aten::NewIdArray(graph->NumVertexTypes(), graph->Context(), 64);
  int64_t* num_nodes_data = static_cast<int64_t*>(num_nodes->data);
  for (int64_t ntype = 0; ntype < graph->NumVertexTypes(); ++ntype)
    num_nodes_data[ntype] = graph->NumVertices(ntype);
  HeteroPickleStates states;
  states.metagraph = graph->meta_graph();
  states.num_nodes = num_nodes;
  states.formats = fmts;
  states.adj_arrays = std::move(adjs);
  return states;
}

HeteroGraphPtr HeteroUnpickle(const HeteroPickleStates& states) {
  const auto metagraph = states.metagraph;
  CHECK_EQ(states.num_nodes->shape[0], metagraph->NumVertices());
  CHECK_EQ(states.formats->shape[0], metagraph->NumEdges());
  const int64_t* num_nodes = static_cast<int64_t*>(states.num_nodes->data);
  const int32_t* fmts = static_cast<int32_t*>(states.formats->data);
  std::vector<HeteroGraphPtr> relgraphs(metagraph->NumEdges());
  for (dgl_type_t etype = 0; etype < metagraph->NumEdges(); ++etype) {
    const SparseFormat fmt = static_cast<SparseFormat>(fmts[etype]);
    const size_t offset = 3 * etype;
    switch (fmt) {
      case SparseFormat::COO:
        {
          const aten::COOMatrix coo();
        relgraphs[etype] = UnitGraph::CreateFromCOO
        }
      case SparseFormat::CSR:
      case SparseFormat::CSC:
      default:
        LOG(FATL) << "Unsupported sparse format.";
    }
  }
}

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroPickleStatesGetMetagraph")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroPickleStatesRef st = args[0];
    *rv = GraphRef(st->metagraph);
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroPickleStatesGetNumNodes")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroPickleStatesRef st = args[0];
    *rv = st->num_nodes;
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroPickleStatesGetFormats")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroPickleStatesRef st = args[0];
    *rv = st->formats;
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroPickleStatesGetAdjArrays")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroPickleStatesRef st = args[0];
    List<Value> adj_arrays;
    for (IdArray arr : st->adj_arrays) {
      adj_arrays.push_back(Value(MakeValue(arr)));
    }
    *rv = adj_arrays;
  });

}  // namespace dgl
