/*!
 *  Copyright (c) 2019 by Contributors
 * \file graph/transform/remove_edges.cc
 * \brief Remove edges.
 */

#include <dgl/array.h>
#include <dgl/base_heterograph.h>
#include <dgl/packed_func_ext.h>
#include <dgl/runtime/container.h>
#include <dgl/runtime/registry.h>
#include <dgl/transform.h>
#include <parallel_hashmap/phmap.h>

#include <tuple>
#include <utility>
#include <vector>

#include "../heterograph.h"
#include "dgl/aten/array_ops.h"
#include "dgl/aten/types.h"

namespace dgl {

using namespace dgl::runtime;
using namespace dgl::aten;

namespace transform {

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_HeteroGraphGetSubgraphWithHalo")
  .set_body([](DGLArgs args, DGLRetValue *rv) {
    HeteroGraphRef graph = args[0];
    IdArray nodes = args[1];
    int num_hops = args[2];
    auto hg = std::dynamic_pointer_cast<HeteroGraph>(graph.sptr());
    CHECK(hg) << "Invalid Heterograph pointer";
    std::vector<IdArray> vid_vec = {nodes};
    std::shared_ptr<HeteroSubgraph> subg(
      new HeteroSubgraph(hg->VertexSubgraph(vid_vec)));
    IdArray halo_in_nodes = nodes;
    std::vector<IdArray> halo_edges_vec;
    std::vector<IdArray> halo_nodes_vec;
    for (int i = 0; i < num_hops; i++) {
      EdgeArray halo_in_edges = hg->InEdges(0, halo_in_nodes);
      halo_edges_vec.emplace_back(std::move(halo_in_edges.id));
      halo_in_nodes = halo_in_edges.src;
      halo_nodes_vec.emplace_back(halo_in_edges.src);
    }
    auto all_edges_with_extra = aten::Concat(halo_edges_vec);
    auto all_nodes_with_extra = aten::Concat(halo_nodes_vec);
    auto all_halo_edge_ids = aten::Diff1d(all_edges_with_extra, subg->induced_edges[0]);
    auto all_halo_node_ids = aten::Diff1d(all_nodes_with_extra, nodes);
    auto all_sungraph_edges = aten::Concat({subg->induced_edges[0], all_halo_edge_ids});
    // auto out = hg->EdgeSubgraph({all_edges}, true);
    List<ObjectRef> ret;

    std::shared_ptr<HeteroSubgraph> out(
      new HeteroSubgraph(hg->EdgeSubgraph({all_sungraph_edges}, true)));
    ret.push_back(HeteroSubgraphRef(out));
    // ret.push_back(HeteroSubgraphRef(out));
    ret.push_back(Value(MakeValue(all_halo_node_ids)));
    ret.push_back(Value(MakeValue(all_halo_edge_ids)));

    *rv = ret;
  });
};  // namespace transform

};  // namespace dgl
