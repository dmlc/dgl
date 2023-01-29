/**
 *  Copyright (c) 2019 by Contributors
 * @file graph/transform/to_simple.cc
 * @brief Convert multigraphs to simple graphs
 */

#include <dgl/array.h>
#include <dgl/base_heterograph.h>
#include <dgl/packed_func_ext.h>
#include <dgl/transform.h>

#include <utility>
#include <vector>

#include "../../c_api_common.h"
#include "../heterograph.h"
#include "../unit_graph.h"

namespace dgl {

using namespace dgl::runtime;
using namespace dgl::aten;

namespace transform {

std::tuple<HeteroGraphPtr, std::vector<IdArray>, std::vector<IdArray>>
ToSimpleGraph(const HeteroGraphPtr graph) {
  const int64_t num_etypes = graph->NumEdgeTypes();
  const auto metagraph = graph->meta_graph();
  const auto &ugs =
      std::dynamic_pointer_cast<HeteroGraph>(graph)->relation_graphs();

  std::vector<IdArray> counts(num_etypes), edge_maps(num_etypes);
  std::vector<HeteroGraphPtr> rel_graphs(num_etypes);

  for (int64_t etype = 0; etype < num_etypes; ++etype) {
    const auto result = ugs[etype]->ToSimple();
    std::tie(rel_graphs[etype], counts[etype], edge_maps[etype]) = result;
  }

  const HeteroGraphPtr result =
      CreateHeteroGraph(metagraph, rel_graphs, graph->NumVerticesPerType());

  return std::make_tuple(result, counts, edge_maps);
}

DGL_REGISTER_GLOBAL("transform._CAPI_DGLToSimpleHetero")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
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
