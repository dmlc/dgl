/*!
 *  Copyright (c) 2020 by Contributors
 * \file graph/pickle.cc
 * \brief Functions for pickle and unpickle a graph
 */
#include <dgl/packed_func_ext.h>
#include <dgl/runtime/container.h>
#include "./heterograph.h"
#include "../c_api_common.h"

using namespace dgl::runtime;

namespace dgl {

HeteroPickleStates HeteroPickle(HeteroGraphPtr graph) {
  HeteroPickleStates states;
  states.metagraph = graph->meta_graph();
  states.adjs.resize(graph->NumEdgeTypes());
  for (dgl_type_t etype = 0; etype < graph->NumEdgeTypes(); ++etype) {
    SparseFormat fmt = graph->SelectFormat(etype, SparseFormat::ANY);
    states.adjs[etype] = std::make_shared<SparseMatrix>();
    switch (fmt) {
      case SparseFormat::COO:
        *states.adjs[etype] = graph->GetCOOMatrix(etype).ToSparseMatrix();
        break;
      case SparseFormat::CSR:
      case SparseFormat::CSC:
        *states.adjs[etype] = graph->GetCSRMatrix(etype).ToSparseMatrix();
        break;
      default:
        LOG(FATAL) << "Unsupported sparse format.";
    }
  }
  return states;
}

HeteroGraphPtr HeteroUnpickle(const HeteroPickleStates& states) {
  const auto metagraph = states.metagraph;
  CHECK_EQ(states.adjs.size(), metagraph->NumEdges());
  std::vector<HeteroGraphPtr> relgraphs(metagraph->NumEdges());
  for (dgl_type_t etype = 0; etype < metagraph->NumEdges(); ++etype) {
    const auto& pair = metagraph->FindEdge(etype);
    const dgl_type_t srctype = pair.first;
    const dgl_type_t dsttype = pair.second;
    const int64_t num_vtypes = (srctype == dsttype)? 1 : 2;
    const SparseFormat fmt = static_cast<SparseFormat>(states.adjs[etype]->format);
    switch (fmt) {
      case SparseFormat::COO:
        relgraphs[etype] = UnitGraph::CreateFromCOO(
            num_vtypes, aten::COOMatrix(*states.adjs[etype]));
        break;
      case SparseFormat::CSR:
        relgraphs[etype] = UnitGraph::CreateFromCSR(
            num_vtypes, aten::CSRMatrix(*states.adjs[etype]));
        break;
      case SparseFormat::CSC:
      default:
        LOG(FATAL) << "Unsupported sparse format.";
    }
  }
  return CreateHeteroGraph(metagraph, relgraphs);
}

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroPickleStatesGetMetagraph")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroPickleStatesRef st = args[0];
    *rv = GraphRef(st->metagraph);
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroPickleStatesGetAdjs")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroPickleStatesRef st = args[0];
    std::vector<SparseMatrixRef> refs(st->adjs.begin(), st->adjs.end());
    *rv = List<SparseMatrixRef>(refs);
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLCreateHeteroPickleStates")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    GraphRef metagraph = args[0];
    List<SparseMatrixRef> adjs = args[1];
    std::shared_ptr<HeteroPickleStates> st( new HeteroPickleStates );
    st->metagraph = metagraph.sptr();
    st->adjs.reserve(adjs.size());
    for (const auto& ref : adjs)
      st->adjs.push_back(ref.sptr());
    *rv = HeteroPickleStatesRef(st);
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroPickle")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef ref = args[0];
    std::shared_ptr<HeteroPickleStates> st( new HeteroPickleStates );
    *st = HeteroPickle(ref.sptr());
    *rv = HeteroPickleStatesRef(st);
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroUnpickle")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroPickleStatesRef ref = args[0];
    HeteroGraphPtr graph = HeteroUnpickle(*ref.sptr());
    *rv = HeteroGraphRef(graph);
  });

}  // namespace dgl
