/*!
 *  Copyright (c) 2019 by Contributors
 * \file graph/transform/remove_edges.cc
 * \brief Remove edges.
 */

#include <dgl/base_heterograph.h>
#include <dgl/transform.h>
#include <dgl/array.h>
#include <dgl/packed_func_ext.h>
#include <vector>
#include <utility>
#include <tuple>

namespace dgl {

using namespace dgl::runtime;
using namespace dgl::aten;

namespace transform {

namespace {

std::pair<HeteroGraphPtr, std::vector<IdArray>>
RemoveEdges(const HeteroGraphPtr graph, const std::vector<IdArray> &eids) {
  std::vector<IdArray> induced_eids;
  std::vector<HeteroGraphPtr> rel_graphs;
  const int64_t num_etypes = graph->NumEdgeTypes();

  for (dgl_type_t etype = 0; etype < num_etypes; ++etype) {
    const SparseFormat fmt = graph->SelectFormat(etype, SparseFormat::kAny);
    const auto src_dst_types = graph->GetEndpointTypes(etype);
    const dgl_type_t srctype = src_dst_types.first;
    const dgl_type_t dsttype = src_dst_types.second;
    const int num_ntypes_rel = (srctype == dsttype) ? 1 : 2;
    const int64_t num_nodes_srctype = graph->NumVertices(srctype);
    const int64_t num_nodes_dsttype = graph->NumVertices(dsttype);
    HeteroGraphPtr new_rel_graph;
    IdArray induced_eids_rel;

    if (fmt == SparseFormat::kCOO) {
      const COOMatrix &coo = graph->GetCOOMatrix(etype);
      auto result = COORemove(coo, eids[etype]);
      new_rel_graph = CreateFromCOO(
          num_ntypes_rel, num_nodes_srctype, num_nodes_dsttype,
          result.first.row, result.first.col);
      induced_eids_rel = result.second;
    } else if (fmt == SparseFormat::kCSR) {
      const CSRMatrix &csr = graph->GetCSRMatrix(etype);
      auto result = CSRRemove(csr, eids[etype]);
      CSRMatrix new_csr = std::move(result.first);
      const IdArray new_eids = Range(
          0, new_csr.indices->shape[0], new_csr.indices->dtype.bits, new_csr.indices->ctx);
      new_rel_graph = CreateFromCSR(
          num_ntypes_rel, num_nodes_srctype, num_nodes_dsttype,
          new_csr.indptr, new_csr.indices, new_eids);
      induced_eids_rel = result.second;
    } else if (fmt == SparseFormat::kCSC) {
      const CSRMatrix &csc = graph->GetCSCMatrix(etype);
      auto result = CSRRemove(csc, eids[etype]);
      CSRMatrix new_csr = CSRTranspose(result.first);
      const IdArray new_eids = Range(
          0, new_csr.indices->shape[0], new_csr.indices->dtype.bits, new_csr.indices->ctx);
      new_rel_graph = CreateFromCSR(
          num_ntypes_rel, num_nodes_srctype, num_nodes_dsttype,
          new_csr.indptr, new_csr.indices, new_eids);
      induced_eids_rel = result.second;
    }

    rel_graphs.push_back(new_rel_graph);
    induced_eids.push_back(induced_eids_rel);
  }

  const HeteroGraphPtr new_graph = CreateHeteroGraph(graph->meta_graph(), rel_graphs);
  return std::make_pair(new_graph, induced_eids);
}

};  // namespace

DGL_REGISTER_GLOBAL("transform._CAPI_DGLRemoveEdges")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
    const HeteroGraphRef graph_ref = args[0];
    const std::vector<IdArray> &eids = ListValueToVector(args[1]);

    HeteroGraphPtr new_graph;
    std::vector<IdArray> induced_eids;
    std::tie(new_graph, induced_eids) = RemoveEdges(graph_ref.sptr(), eids);

    List<Value> induced_eids_ref;
    for (IdArray &array : induced_eids)
      induced_eids_ref.push_back(Value(MakeValue(array)));

    List<ObjectRef> ret;
    ret.push_back(HeteroGraphRef(new_graph));
    ret.push_back(induced_eids_ref);

    *rv = ret;
  });

};  // namespace transform

};  // namespace dgl
