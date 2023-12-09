/**
 *  Copyright 2019-2021 Contributors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 * @file graph/transform/compact.cc
 * @brief Compact graph implementation
 */

#include "compact.h"

#include <dgl/array.h>
#include <dgl/base_heterograph.h>
#include <dgl/packed_func_ext.h>
#include <dgl/runtime/container.h>
#include <dgl/runtime/registry.h>
#include <dgl/transform.h>

#include <utility>
#include <vector>

#include "../../c_api_common.h"
#include "../unit_graph.h"
// TODO(BarclayII): currently CompactGraphs depend on IdHashMap implementation
// which only works on CPU.  Should fix later to make it device agnostic.
#include "../../array/cpu/array_utils.h"

namespace dgl {

using namespace dgl::runtime;
using namespace dgl::aten;

namespace transform {

namespace {

template <typename IdType>
std::pair<std::vector<HeteroGraphPtr>, std::vector<IdArray>> CompactGraphsCPU(
    const std::vector<HeteroGraphPtr> &graphs,
    const std::vector<IdArray> &always_preserve) {
  // TODO(BarclayII): check whether the node space and metagraph of each graph
  // is the same. Step 1: Collect the nodes that has connections for each type.
  const int64_t num_ntypes = graphs[0]->NumVertexTypes();
  std::vector<aten::IdHashMap<IdType>> hashmaps(num_ntypes);
  std::vector<std::vector<EdgeArray>> all_edges(
      graphs.size());  // all_edges[i][etype]

  std::vector<int64_t> max_vertex_cnt(num_ntypes, 0);
  for (size_t i = 0; i < graphs.size(); ++i) {
    const HeteroGraphPtr curr_graph = graphs[i];
    const int64_t num_etypes = curr_graph->NumEdgeTypes();

    for (IdType etype = 0; etype < num_etypes; ++etype) {
      IdType srctype, dsttype;
      std::tie(srctype, dsttype) = curr_graph->GetEndpointTypes(etype);

      const int64_t n_edges = curr_graph->NumEdges(etype);
      max_vertex_cnt[srctype] += n_edges;
      max_vertex_cnt[dsttype] += n_edges;
    }
  }

  // Reserve the space for hash maps before ahead to aoivd rehashing
  for (size_t i = 0; i < static_cast<size_t>(num_ntypes); ++i) {
    if (i < always_preserve.size())
      hashmaps[i].Reserve(always_preserve[i]->shape[0] + max_vertex_cnt[i]);
    else
      hashmaps[i].Reserve(max_vertex_cnt[i]);
  }

  for (size_t i = 0; i < always_preserve.size(); ++i) {
    hashmaps[i].Update(always_preserve[i]);
  }

  for (size_t i = 0; i < graphs.size(); ++i) {
    const HeteroGraphPtr curr_graph = graphs[i];
    const int64_t num_etypes = curr_graph->NumEdgeTypes();

    all_edges[i].reserve(num_etypes);
    for (IdType etype = 0; etype < num_etypes; ++etype) {
      IdType srctype, dsttype;
      std::tie(srctype, dsttype) = curr_graph->GetEndpointTypes(etype);

      const EdgeArray edges = curr_graph->Edges(etype, "eid");

      hashmaps[srctype].Update(edges.src);
      hashmaps[dsttype].Update(edges.dst);

      all_edges[i].push_back(edges);
    }
  }

  // Step 2: Relabel the nodes for each type to a smaller ID space and save the
  // mapping.
  std::vector<IdArray> induced_nodes(num_ntypes);
  std::vector<int64_t> num_induced_nodes(num_ntypes);
  for (int64_t i = 0; i < num_ntypes; ++i) {
    induced_nodes[i] = hashmaps[i].Values();
    num_induced_nodes[i] = hashmaps[i].Size();
  }

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
          srctype == dsttype ? 1 : 2, induced_nodes[srctype]->shape[0],
          induced_nodes[dsttype]->shape[0], mapped_rows, mapped_cols));
    }

    new_graphs.push_back(
        CreateHeteroGraph(meta_graph, rel_graphs, num_induced_nodes));
  }

  return std::make_pair(new_graphs, induced_nodes);
}

};  // namespace

template <>
std::pair<std::vector<HeteroGraphPtr>, std::vector<IdArray>>
CompactGraphs<kDGLCPU, int32_t>(
    const std::vector<HeteroGraphPtr> &graphs,
    const std::vector<IdArray> &always_preserve) {
  return CompactGraphsCPU<int32_t>(graphs, always_preserve);
}

template <>
std::pair<std::vector<HeteroGraphPtr>, std::vector<IdArray>>
CompactGraphs<kDGLCPU, int64_t>(
    const std::vector<HeteroGraphPtr> &graphs,
    const std::vector<IdArray> &always_preserve) {
  return CompactGraphsCPU<int64_t>(graphs, always_preserve);
}

DGL_REGISTER_GLOBAL("transform._CAPI_DGLCompactGraphs")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      List<HeteroGraphRef> graph_refs = args[0];
      List<Value> always_preserve_refs = args[1];

      std::vector<HeteroGraphPtr> graphs;
      std::vector<IdArray> always_preserve;
      for (HeteroGraphRef gref : graph_refs) graphs.push_back(gref.sptr());
      for (Value array : always_preserve_refs)
        always_preserve.push_back(array->data);

      // TODO(BarclayII): check for all IdArrays
      CHECK(graphs[0]->DataType() == always_preserve[0]->dtype)
          << "data type mismatch.";

      std::pair<std::vector<HeteroGraphPtr>, std::vector<IdArray>> result_pair;

      ATEN_XPU_SWITCH_CUDA(
          graphs[0]->Context().device_type, XPU, "CompactGraphs", {
            ATEN_ID_TYPE_SWITCH(graphs[0]->DataType(), IdType, {
              result_pair = CompactGraphs<XPU, IdType>(graphs, always_preserve);
            });
          });

      List<HeteroGraphRef> compacted_graph_refs;
      List<Value> induced_nodes;

      for (const HeteroGraphPtr &g : result_pair.first)
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
