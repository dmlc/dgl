/*!
 *  Copyright (c) 2019 by Contributors
 * \file graph/transform/select_topk.cc
 * \brief Create a subgraph by selecting inbound edges with top-K weights.
 */

#include <dgl/base_heterograph.h>
#include <dgl/transform.h>
#include <dgl/array.h>
#include <dgl/packed_func_ext.h>
#include <vector>
#include <tuple>
#include "../unit_graph.h"
#include "../../c_api_common.h"

namespace dgl {

using namespace dgl::runtime;
using namespace dgl::aten;

namespace transform {

std::tuple<HeteroGraphPtr, std::vector<IdArray>, std::vector<NDArray>> SelectTopK(
    const HeteroGraphPtr graph,
    const std::vector<NDArray> &weights,
    int K,
    bool inbound,
    bool smallest) {
  const int64_t num_etypes = graph->NumEdgeTypes();
  const auto metagraph = graph->meta_graph();

  std::vector<IdArray> induced_edges(num_etypes);
  std::vector<NDArray> induced_values(num_etypes);
  std::vector<HeteroGraphPtr> rel_graphs(num_etypes);

  for (dgl_type_t etype = 0; etype < num_etypes; ++etype) {
    const HeteroGraphPtr rel_graph = graph->GetRelationGraph(etype);
    const COOMatrix adj = rel_graph->GetCOOMatrix(0);
    const COOMatrix sorted_adj = COOSort(inbound ? COOTranspose(adj) : adj);
    const auto selected = COORowwiseTopKNonZero(sorted_adj, K, weights[etype], smallest);

    induced_edges[etype] = selected.first.data;
    induced_values[etype] = selected.second;
    if (inbound)
      rel_graphs[etype] = UnitGraph::CreateFromCOO(
          rel_graph->NumVertexTypes(),
          selected.first.num_cols,
          selected.first.num_rows,
          selected.first.col,
          selected.first.row);
    else
      rel_graphs[etype] = UnitGraph::CreateFromCOO(
          rel_graph->NumVertexTypes(),
          selected.first.num_rows,
          selected.first.num_cols,
          selected.first.row,
          selected.first.col);
  }

  const HeteroGraphPtr new_graph = CreateHeteroGraph(metagraph, rel_graphs);
  return std::make_tuple(new_graph, induced_edges, induced_values);
}

DGL_REGISTER_GLOBAL("transform._CAPI_DGLSelectTopK")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    const HeteroGraphRef graph_ref = args[0];
    List<Value> weights_ref = args[1];
    int K = args[2];
    bool inbound = args[3];
    bool smallest = args[4];
    std::vector<NDArray> weights;
    for (const auto &ref : weights_ref)
      weights.push_back(ref->data);

    const auto result = SelectTopK(graph_ref.sptr(), weights, K, inbound, smallest);

    List<Value> induced_edges, induced_values;
    for (const IdArray &ids : std::get<1>(result))
      induced_edges.push_back(Value(MakeValue(ids)));
    for (const IdArray &values : std::get<2>(result))
      induced_values.push_back(Value(MakeValue(values)));

    List<ObjectRef> result_ref;
    result_ref.push_back(HeteroGraphRef(std::get<0>(result)));
    result_ref.push_back(induced_edges);
    result_ref.push_back(induced_values);

    *rv = result_ref;
  });

};  // namespace transform

};  // namespace dgl

