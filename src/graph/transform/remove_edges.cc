/**
 *  Copyright (c) 2019 by Contributors
 * @file graph/transform/remove_edges.cc
 * @brief Remove edges.
 */

#include <dgl/array.h>
#include <dgl/base_heterograph.h>
#include <dgl/packed_func_ext.h>
#include <dgl/runtime/container.h>
#include <dgl/runtime/registry.h>
#include <dgl/transform.h>

#include <tuple>
#include <utility>
#include <vector>

namespace dgl {

using namespace dgl::runtime;
using namespace dgl::aten;

namespace transform {

std::pair<HeteroGraphPtr, std::vector<IdArray>> RemoveEdges(
    const HeteroGraphPtr graph, const std::vector<IdArray> &eids) {
  std::vector<IdArray> induced_eids;
  std::vector<HeteroGraphPtr> rel_graphs;
  const int64_t num_etypes = graph->NumEdgeTypes();

  for (int64_t etype = 0; etype < num_etypes; ++etype) {
    const SparseFormat fmt = graph->SelectFormat(etype, COO_CODE);
    const auto src_dst_types = graph->GetEndpointTypes(etype);
    const dgl_type_t srctype = src_dst_types.first;
    const dgl_type_t dsttype = src_dst_types.second;
    const int num_ntypes_rel = (srctype == dsttype) ? 1 : 2;
    HeteroGraphPtr new_rel_graph;
    IdArray induced_eids_rel;

    if (fmt == SparseFormat::kCOO) {
      const COOMatrix &coo = graph->GetCOOMatrix(etype);
      const COOMatrix &result = COORemove(coo, eids[etype]);
      new_rel_graph = CreateFromCOO(
          num_ntypes_rel, result.num_rows, result.num_cols, result.row,
          result.col);
      induced_eids_rel = result.data;
    } else if (fmt == SparseFormat::kCSR) {
      const CSRMatrix &csr = graph->GetCSRMatrix(etype);
      const CSRMatrix &result = CSRRemove(csr, eids[etype]);
      new_rel_graph = CreateFromCSR(
          num_ntypes_rel, result.num_rows, result.num_cols, result.indptr,
          result.indices,
          // TODO(BarclayII): make CSR support null eid array
          Range(
              0, result.indices->shape[0], result.indices->dtype.bits,
              result.indices->ctx));
      induced_eids_rel = result.data;
    } else if (fmt == SparseFormat::kCSC) {
      const CSRMatrix &csc = graph->GetCSCMatrix(etype);
      const CSRMatrix &result = CSRRemove(csc, eids[etype]);
      new_rel_graph = CreateFromCSC(
          num_ntypes_rel, result.num_rows, result.num_cols, result.indptr,
          result.indices,
          // TODO(BarclayII): make CSR support null eid array
          Range(
              0, result.indices->shape[0], result.indices->dtype.bits,
              result.indices->ctx));
      induced_eids_rel = result.data;
    }

    rel_graphs.push_back(new_rel_graph);
    induced_eids.push_back(induced_eids_rel);
  }

  const HeteroGraphPtr new_graph = CreateHeteroGraph(
      graph->meta_graph(), rel_graphs, graph->NumVerticesPerType());
  return std::make_pair(new_graph, induced_eids);
}

DGL_REGISTER_GLOBAL("transform._CAPI_DGLRemoveEdges")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      const HeteroGraphRef graph_ref = args[0];
      const std::vector<IdArray> &eids = ListValueToVector<IdArray>(args[1]);

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
