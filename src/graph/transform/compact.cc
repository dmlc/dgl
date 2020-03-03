/*!
 *  Copyright (c) 2019 by Contributors
 * \file graph/transform/compact.cc
 * \brief Compact graph implementation
 */

#include <dgl/base_heterograph.h>
#include <dgl/transform.h>
#include <dgl/array.h>
#include <dgl/packed_func_ext.h>
#include <vector>
#include <utility>
#include "../../c_api_common.h"
#include "../unit_graph.h"
// TODO(BarclayII): currently CompactGraphs depend on IdHashMap implementation which
// only works on CPU.  Should fix later to make it device agnostic.
#include "../../array/cpu/array_utils.h"

namespace dgl {

using namespace dgl::runtime;
using namespace dgl::aten;

namespace transform {

namespace {

template<typename IdType>
std::pair<std::vector<HeteroGraphPtr>, std::vector<IdArray>>
CompactGraphs(
    const std::vector<HeteroGraphPtr> &graphs,
    const std::vector<IdArray> &always_preserve) {
  // TODO(BarclayII): check whether the node space and metagraph of each graph is the same.
  // Step 1: Collect the nodes that has connections for each type.
  std::vector<aten::IdHashMap<IdType>> hashmaps(graphs[0]->NumVertexTypes());
  std::vector<std::vector<EdgeArray>> all_edges(graphs.size());   // all_edges[i][etype]

  for (size_t i = 0; i < always_preserve.size(); ++i)
    hashmaps[i].Update(always_preserve[i]);

  for (size_t i = 0; i < graphs.size(); ++i) {
    const HeteroGraphPtr curr_graph = graphs[i];
    const int64_t num_etypes = curr_graph->NumEdgeTypes();

    for (IdType etype = 0; etype < num_etypes; ++etype) {
      IdType srctype, dsttype;
      std::tie(srctype, dsttype) = curr_graph->GetEndpointTypes(etype);

      const EdgeArray edges = curr_graph->Edges(etype, "eid");

      hashmaps[srctype].Update(edges.src);
      hashmaps[dsttype].Update(edges.dst);

      all_edges[i].push_back(edges);
    }
  }

  // Step 2: Relabel the nodes for each type to a smaller ID space and save the mapping.
  std::vector<IdArray> induced_nodes;
  for (auto &hashmap : hashmaps)
    induced_nodes.push_back(hashmap.Values());

  // Step 3: Remap the edges of each graph.
  std::vector<HeteroGraphPtr> new_graphs;
  for (size_t i = 0; i < graphs.size(); ++i) {
    std::vector<HeteroGraphPtr> rel_graphs;
    const HeteroGraphPtr curr_graph = graphs[i];
    const auto meta_graph = curr_graph->meta_graph();
    const int64_t num_etypes = curr_graph->NumEdgeTypes();

    for (IdType etype = 0; etype < num_etypes; ++etype) {
      IdType srctype, dsttype;
      std::tie(srctype, dsttype) = curr_graph->GetEndpointTypes(etype);
      const EdgeArray &edges = all_edges[i][etype];

      const IdArray mapped_rows = hashmaps[srctype].Map(edges.src, -1);
      const IdArray mapped_cols = hashmaps[dsttype].Map(edges.dst, -1);

      rel_graphs.push_back(UnitGraph::CreateFromCOO(
          srctype == dsttype ? 1 : 2,
          induced_nodes[srctype]->shape[0],
          induced_nodes[dsttype]->shape[0],
          mapped_rows,
          mapped_cols));
    }

    new_graphs.push_back(CreateHeteroGraph(meta_graph, rel_graphs));
  }

  return std::make_pair(new_graphs, induced_nodes);
}

};  // namespace

std::pair<std::vector<HeteroGraphPtr>, std::vector<IdArray>>
CompactGraphs(
    const std::vector<HeteroGraphPtr> &graphs,
    const std::vector<IdArray> &always_preserve) {
  std::pair<std::vector<HeteroGraphPtr>, std::vector<IdArray>> result;
  // TODO(BarclayII): check for all IdArrays
  CHECK(graphs[0]->DataType() == always_preserve[0]->dtype) << "data type mismatch.";
  ATEN_ID_TYPE_SWITCH(graphs[0]->DataType(), IdType, {
    result = CompactGraphs<IdType>(graphs, always_preserve);
  });
  return result;
}

DGL_REGISTER_GLOBAL("transform._CAPI_DGLCompactGraphs")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    List<HeteroGraphRef> graph_refs = args[0];
    List<Value> always_preserve_refs = args[1];

    std::vector<HeteroGraphPtr> graphs;
    std::vector<IdArray> always_preserve;
    for (HeteroGraphRef gref : graph_refs)
      graphs.push_back(gref.sptr());
    for (Value array : always_preserve_refs)
      always_preserve.push_back(array->data);

    const auto &result_pair = CompactGraphs(graphs, always_preserve);

    List<HeteroGraphRef> compacted_graph_refs;
    List<Value> induced_nodes;

    for (const HeteroGraphPtr g : result_pair.first)
      compacted_graph_refs.push_back(HeteroGraphRef(g));
    for (const IdArray &ids : result_pair.second)
      induced_nodes.push_back(Value(MakeValue(ids)));

    List<ObjectRef> result;
    result.push_back(compacted_graph_refs);
    result.push_back(induced_nodes);

    *rv = result;
  });

};  // namespace transform

};  // namespace dgl
