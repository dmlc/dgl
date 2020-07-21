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
#include "unit_graph.h"

using namespace dgl::runtime;

namespace dgl {

HeteroPickleStates HeteroPickle(HeteroGraphPtr graph) {
  HeteroPickleStates states;
  dmlc::MemoryStringStream ofs(&states.meta);
  dmlc::Stream *strm = &ofs;
  strm->Write(ImmutableGraph::ToImmutable(graph->meta_graph()));
  strm->Write(graph->NumVerticesPerType());
  for (dgl_type_t etype = 0; etype < graph->NumEdgeTypes(); ++etype) {
    SparseFormat fmt = graph->SelectFormat(etype, all_code);
    switch (fmt) {
      case SparseFormat::kCOO: {
        strm->Write(SparseFormat::kCOO);
        const auto &coo = graph->GetCOOMatrix(etype);
        strm->Write(coo.row_sorted);
        strm->Write(coo.col_sorted);
        states.arrays.push_back(coo.row);
        states.arrays.push_back(coo.col);
        break;
      }
      case SparseFormat::kCSR:
      case SparseFormat::kCSC: {
        strm->Write(SparseFormat::kCSR);
        const auto &csr = graph->GetCSRMatrix(etype);
        strm->Write(csr.sorted);
        states.arrays.push_back(csr.indptr);
        states.arrays.push_back(csr.indices);
        states.arrays.push_back(csr.data);
        break;
      }
      default:
        LOG(FATAL) << "Unsupported sparse format.";
    }
  }
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

  auto array_itr = states.arrays.begin();
  for (dgl_type_t etype = 0; etype < metagraph->NumEdges(); ++etype) {
    const auto& pair = metagraph->FindEdge(etype);
    const dgl_type_t srctype = pair.first;
    const dgl_type_t dsttype = pair.second;
    const int64_t num_vtypes = (srctype == dsttype)? 1 : 2;
    int64_t num_src = num_nodes_per_type[srctype];
    int64_t num_dst = num_nodes_per_type[dsttype];
    SparseFormat fmt;
    CHECK(strm->Read(&fmt)) << "Invalid SparseFormat";
    HeteroGraphPtr relgraph;
    switch (fmt) {
      case SparseFormat::kCOO: {
        CHECK_GE(states.arrays.end() - array_itr, 2);
        const auto &row = *(array_itr++);
        const auto &col = *(array_itr++);
        bool rsorted;
        bool csorted;
        CHECK(strm->Read(&rsorted)) << "Invalid flag 'rsorted'";
        CHECK(strm->Read(&csorted)) << "Invalid flag 'csorted'";
        auto coo = aten::COOMatrix(num_src, num_dst, row, col, aten::NullArray(), rsorted, csorted);
        // TODO(zihao) fix
        relgraph = CreateFromCOO(num_vtypes, coo, all_code);
        break;
      }
      case SparseFormat::kCSR: {
        CHECK_GE(states.arrays.end() - array_itr, 3);
        const auto &indptr = *(array_itr++);
        const auto &indices = *(array_itr++);
        const auto &edge_id = *(array_itr++);
        bool sorted;
        CHECK(strm->Read(&sorted)) << "Invalid flag 'sorted'";
        auto csr = aten::CSRMatrix(num_src, num_dst, indptr, indices, edge_id, sorted);
        // TODO(zihao) fix
        relgraph = CreateFromCSR(num_vtypes, csr, all_code);
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

// For backward compatibility
HeteroGraphPtr HeteroUnpickleOld(const HeteroPickleStates& states) {
  const auto metagraph = states.metagraph;
  const auto &num_nodes_per_type = states.num_nodes_per_type;
  CHECK_EQ(states.adjs.size(), metagraph->NumEdges());
  std::vector<HeteroGraphPtr> relgraphs(metagraph->NumEdges());
  for (dgl_type_t etype = 0; etype < metagraph->NumEdges(); ++etype) {
    const auto& pair = metagraph->FindEdge(etype);
    const dgl_type_t srctype = pair.first;
    const dgl_type_t dsttype = pair.second;
    const int64_t num_vtypes = (srctype == dsttype)? 1 : 2;
    const SparseFormat fmt = static_cast<SparseFormat>(states.adjs[etype]->format);
    switch (fmt) {
      case SparseFormat::kCOO:
        relgraphs[etype] = UnitGraph::CreateFromCOO(
            num_vtypes, aten::COOMatrix(*states.adjs[etype]));
        break;
      case SparseFormat::kCSR:
        relgraphs[etype] = UnitGraph::CreateFromCSR(
            num_vtypes, aten::CSRMatrix(*states.adjs[etype]));
        break;
      case SparseFormat::kCSC:
      default:
        LOG(FATAL) << "Unsupported sparse format.";
    }
  }
  return CreateHeteroGraph(metagraph, relgraphs, num_nodes_per_type);
}

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroPickleStatesGetVersion")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroPickleStatesRef st = args[0];
    *rv = st->version;
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroPickleStatesGetMeta")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroPickleStatesRef st = args[0];
    DGLByteArray buf;
    buf.data = st->meta.c_str();
    buf.size = st->meta.size();
    *rv = buf;
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
    st->version = 1;
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
    HeteroGraphPtr graph;
    switch (ref->version) {
      case 0:
        graph = HeteroUnpickleOld(*ref.sptr());
        break;
      case 1:
        graph = HeteroUnpickle(*ref.sptr());
        break;
      default:
        LOG(FATAL) << "Version can only be 0 or 1.";
    }
    *rv = HeteroGraphRef(graph);
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLCreateHeteroPickleStatesOld")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    GraphRef metagraph = args[0];
    IdArray num_nodes_per_type = args[1];
    List<SparseMatrixRef> adjs = args[2];
    std::shared_ptr<HeteroPickleStates> st( new HeteroPickleStates );
    st->version = 0;
    st->metagraph = metagraph.sptr();
    st->num_nodes_per_type = num_nodes_per_type.ToVector<int64_t>();
    st->adjs.reserve(adjs.size());
    for (const auto& ref : adjs)
      st->adjs.push_back(ref.sptr());
    *rv = HeteroPickleStatesRef(st);
  });
}  // namespace dgl
