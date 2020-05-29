/*!
 *  Copyright (c) 2020 by Contributors
 * \file graph/pickle.cc
 * \brief Functions for pickle and unpickle a graph
 */
#include <dgl/packed_func_ext.h>
#include <dgl/runtime/container.h>
#include <dgl/immutable_graph.h>
#include <dgl/graph_serializer.h>
#include <dmlc/memory_io.h>
#include "./heterograph.h"
#include "../c_api_common.h"

using namespace dgl::runtime;

namespace dgl {

HeteroPickleStates HeteroPickle(HeteroGraphPtr graph) {
  HeteroPickleStates states;
  dmlc::MemoryStringStream ofs(&states.meta);
  dmlc::Stream *strm = &ofs;
  strm->Write(ImmutableGraph::ToImmutable(graph->meta_graph()));
  strm->Write(graph->NumVerticesPerType());
  std::vector<SparseFormat> fmts;
  std::vector<char> flags;
  for (dgl_type_t etype = 0; etype < graph->NumEdgeTypes(); ++etype) {
    SparseFormat fmt = graph->SelectFormat(etype, SparseFormat::kAny);
    switch (fmt) {
      case SparseFormat::kCOO: {
        fmts.push_back(SparseFormat::kCOO);
        const auto &coo = graph->GetCOOMatrix(etype);
        flags.push_back(coo.row_sorted);
        flags.push_back(coo.col_sorted);
        states.arrays.push_back(coo.row);
        states.arrays.push_back(coo.col);
        break;
      }
      case SparseFormat::kCSR:
      case SparseFormat::kCSC: {
        fmts.push_back(SparseFormat::kCSR);
        const auto &csr = graph->GetCSRMatrix(etype);
        flags.push_back(csr.sorted);
        states.arrays.push_back(csr.indptr);
        states.arrays.push_back(csr.indices);
        states.arrays.push_back(csr.data);
        break;
      }
      default:
        LOG(FATAL) << "Unsupported sparse format.";
    }
  }
  strm->Write(fmts);
  strm->Write(flags);
  return states;
}

HeteroGraphPtr HeteroUnpickle(const HeteroPickleStates& states) {
  char *buf = const_cast<char *>(states.meta.c_str());  // a readonly stream?
  dmlc::MemoryFixedSizeStream ifs(buf, states.meta.size());
  dmlc::Stream *strm = &ifs;
  auto meta_imgraph = Serializer::make_shared<ImmutableGraph>();
  CHECK(strm->Read(&meta_imgraph)) << "Invalid meta graph";
  GraphPtr metagraph = meta_imgraph;
  std::vector<HeteroGraphPtr> relgraphs(metagraph->NumEdges());
  std::vector<int64_t> num_nodes_per_type;
  CHECK(strm->Read(&num_nodes_per_type)) << "Invalid num_nodes_per_type";

  std::vector<SparseFormat> fmts;
  std::vector<char> flags;
  CHECK(strm->Read(&fmts)) << "Invalid sparse matrix format";
  CHECK(strm->Read(&flags)) << "Invalid flags";
  auto fmt_itr = fmts.begin();
  auto flags_itr = flags.begin();
  auto array_itr = states.arrays.begin();
  for (dgl_type_t etype = 0; etype < metagraph->NumEdges(); ++etype) {
    const auto& pair = metagraph->FindEdge(etype);
    const dgl_type_t srctype = pair.first;
    const dgl_type_t dsttype = pair.second;
    const int64_t num_vtypes = (srctype == dsttype)? 1 : 2;
    int64_t num_src = num_nodes_per_type[srctype];
    int64_t num_dst = num_nodes_per_type[dsttype];

    CHECK(fmt_itr != fmts.end());
    const auto fmt = *(fmt_itr++);
    HeteroGraphPtr relgraph;
    switch (fmt) {
      case SparseFormat::kCOO: {
        CHECK_GE(states.arrays.end() - array_itr, 2);
        const auto &row = *(array_itr++);
        const auto &col = *(array_itr++);
        CHECK_GE(flags.end() - flags_itr, 2);
        bool rsorted = *(flags_itr++);
        bool csorted = *(flags_itr++);

        auto coo = aten::COOMatrix(num_src, num_dst, row, col, aten::NullArray(), rsorted, csorted);
        relgraph = CreateFromCOO(num_vtypes, coo);
        break;
      }
      case SparseFormat::kCSR: {
        CHECK_GE(states.arrays.end() - array_itr, 3);
        const auto &indptr = *(array_itr++);
        const auto &indices = *(array_itr++);
        const auto &edge_id = *(array_itr++);
        CHECK(flags_itr != flags.end());
        bool sorted = *(flags_itr++);
        auto csr = aten::CSRMatrix(num_src, num_dst, indptr, indices, edge_id, sorted);
        relgraph = CreateFromCSR(num_vtypes, csr);
        break;
      }
      case SparseFormat::kCSC:
      default:
        LOG(FATAL) << "Unsupported sparse format.";
    }
    relgraphs[etype] = relgraph;
  }
  return CreateHeteroGraph(metagraph, relgraphs, num_nodes_per_type);
}

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroPickleStatesGetMeta")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroPickleStatesRef st = args[0];
    *rv = st->meta;
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroPickleStatesGetArrays")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroPickleStatesRef st = args[0];
    *rv = ConvertNDArrayVectorToPackedFunc(st->arrays);
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroPickleStatesGetArraysNum")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroPickleStatesRef st = args[0];
    *rv = static_cast<int64_t>(st->arrays.size());
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLCreateHeteroPickleStates")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    std::string meta = args[0];
    const List<Value> arrays = args[1];
    std::shared_ptr<HeteroPickleStates> st( new HeteroPickleStates );
    st->meta = meta;
    st->arrays.reserve(arrays.size());
    for (const auto& ref : arrays) {
      st->arrays.push_back(ref->data);
    }
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
