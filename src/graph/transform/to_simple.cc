/*!
 *  Copyright (c) 2019 by Contributors
 * \file graph/transform/to_simple.cc
 * \brief Convert multigraphs to simple graphs
 */

#include <dgl/base_heterograph.h>
#include <dgl/transform.h>
#include <dgl/array.h>
#include <dgl/packed_func_ext.h>
#include <vector>
#include <utility>
#include "../unit_graph.h"
#include "../../c_api_common.h"

namespace dgl {

using namespace dgl::runtime;
using namespace dgl::aten;

namespace transform {

std::tuple<HeteroGraphPtr, std::vector<IdArray>, std::vector<IdArray>>
ToSimpleGraph(const HeteroGraphPtr graph) {
  const int64_t num_etypes = graph->NumEdgeTypes();
  const auto metagraph = graph->meta_graph();

  std::vector<IdArray> counts(num_etypes), edge_maps(num_etypes);
  std::vector<HeteroGraphPtr> rel_graphs(num_etypes);

  for (int64_t etype = 0; etype < num_etypes; ++etype) {
    const auto vtypes = graph->GetEndpointTypes(etype);
    const COOMatrix adj = graph->GetCOOMatrix(etype);
    const COOMatrix sorted_adj = COOSort(adj, true);
    const IdArray eids_shuffled = sorted_adj.data;
    const auto &coalesced_result = COOCoalesce(sorted_adj);
    const COOMatrix &coalesced_adj = coalesced_result.first;
    const IdArray &count = coalesced_result.second;

    /*
     * eids_shuffled actually already contains the mapping from old edge space to the
     * new one:
     *
     * * eids_shuffled[0:count[0]] indicates the original edge IDs that coalesced into new
     *   edge #0.
     * * eids_shuffled[count[0]:count[0] + count[1]] indicates those that coalesced into
     *   new edge #1.
     * * eids_shuffled[count[0] + count[1]:count[0] + count[1] + count[2]] indicates those
     *   that coalesced into new edge #2.
     * * etc.
     *
     * Here, we need to translate eids_shuffled to an array "eids_remapped" such that
     * eids_remapped[i] indicates the new edge ID the old edge #i is mapped to.  The
     * translation can simply be achieved by (in numpy code):
     *
     *     new_eid_for_eids_shuffled = np.range(len(count)).repeat(count)
     *     eids_remapped = np.zeros_like(new_eid_for_eids_shuffled)
     *     eids_remapped[eids_shuffled] = new_eid_for_eids_shuffled
     */
    const IdArray new_eids = Range(
        0, coalesced_adj.row->shape[0], coalesced_adj.row->dtype.bits, coalesced_adj.row->ctx);
    const IdArray eids_remapped = Scatter(Repeat(new_eids, count), eids_shuffled);

    edge_maps[etype] = eids_remapped;
    counts[etype] = count;
    rel_graphs[etype] = UnitGraph::CreateFromCOO(
        vtypes.first == vtypes.second ? 1 : 2,
        coalesced_adj.num_rows,
        coalesced_adj.num_cols,
        coalesced_adj.row,
        coalesced_adj.col);
  }

  const HeteroGraphPtr result = CreateHeteroGraph(
      metagraph, rel_graphs, graph->NumVerticesPerType());

  return std::make_tuple(result, counts, edge_maps);
}

DGL_REGISTER_GLOBAL("transform._CAPI_DGLToSimpleHetero")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
    const HeteroGraphRef graph_ref = args[0];

    const auto result = ToSimpleGraph(graph_ref.sptr());

    List<Value> counts, edge_maps;
    for (const IdArray &count : std::get<1>(result))
      counts.push_back(Value(MakeValue(count)));
    for (const IdArray &edge_map : std::get<2>(result))
      edge_maps.push_back(Value(MakeValue(edge_map)));

    List<ObjectRef> ret;
    ret.push_back(HeteroGraphRef(std::get<0>(result)));
    ret.push_back(counts);
    ret.push_back(edge_maps);

    *rv = ret;
  });

};  // namespace transform

};  // namespace dgl
