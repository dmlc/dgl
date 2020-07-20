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

#include <algorithm>
#include <iostream>
#include <iterator>
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

/*! \brief Return the sorted unique values in arr1 that are not in arr2.
 * Same as np.setdiff1d */
IdArray SetDiff1d(IdArray arr1, IdArray arr2);

/*! \brief Return the sorted unique values in arr. Same as np.unique */
IdArray Unique(IdArray arr);

namespace impl {

template <DLDeviceType XPU, typename DType>
IdArray SetDiff1d(IdArray arr1, IdArray arr2);

template <DLDeviceType XPU, typename DType>
IdArray Unique(IdArray arr);

///////////////////////////// SetDiff1d /////////////////////////////

template <DLDeviceType XPU, typename IdType>
IdArray SetDiff1d(IdArray arr1, IdArray arr2) {
  CHECK(arr1->ndim == 1) << "SetDiff1d only supports 1D array";
  CHECK(arr2->ndim == 1) << "SetDiff1d only supports 1D array";
  IdArray unique_arr1 = Unique<XPU, IdType>(arr1);
  IdArray unique_arr2 = Unique<XPU, IdType>(arr2);
  const IdType* unique_arr1_data = static_cast<IdType*>(unique_arr1->data);
  const IdType* unique_arr2_data = static_cast<IdType*>(unique_arr2->data);
  std::vector<IdType> diff;
  std::set_difference(
    unique_arr1_data, unique_arr1_data + unique_arr1->shape[0],
    unique_arr2_data, unique_arr2_data + unique_arr2->shape[0],
    std::inserter(diff, diff.begin()));

  return VecToIdArray(diff, sizeof(IdType) * 8);
}

template IdArray SetDiff1d<kDLCPU, int32_t>(IdArray arr1, IdArray arr2);
template IdArray SetDiff1d<kDLCPU, int64_t>(IdArray arr1, IdArray arr2);

///////////////////////////// Unique /////////////////////////////

template <DLDeviceType XPU, typename IdType>
IdArray Unique(IdArray arr) {
  CHECK(arr->ndim == 1) << "Unique only supports 1D array";
  const IdType* arr_data = static_cast<IdType*>(arr->data);
  std::vector<IdType> arr_vec(arr_data, arr_data + arr->shape[0]);
  std::sort(arr_vec.begin(), arr_vec.end());
  arr_vec.erase(std::unique(arr_vec.begin(), arr_vec.end()), arr_vec.end());
  IdArray ret_ndarray = VecToIdArray(arr_vec, sizeof(IdType) * 8);
  return ret_ndarray;
}

template IdArray Unique<kDLCPU, int32_t>(IdArray arr);
template IdArray Unique<kDLCPU, int64_t>(IdArray arr);

};  // namespace impl

// clang-format off

IdArray SetDiff1d(IdArray lhs, IdArray rhs) {
  IdArray ret;
  CHECK_SAME_CONTEXT(lhs, rhs);
  CHECK_SAME_DTYPE(lhs, rhs);
  ATEN_XPU_SWITCH(lhs->ctx.device_type, XPU, "SetDiff1d", {
    ATEN_ID_TYPE_SWITCH(lhs->dtype, IdType, {
      ret = impl::SetDiff1d<XPU, IdType>(lhs, rhs);
    });
  });
  return ret;
}


IdArray Unique(IdArray arr) {
  IdArray ret;
  ATEN_XPU_SWITCH(arr->ctx.device_type, XPU, "Unique", {
    ATEN_ID_TYPE_SWITCH(arr->dtype, IdType, {
      ret = impl::Unique<XPU, IdType>(arr);
    });
  });
  return ret;
}

// clang-format on

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_HeteroGraphGetSubgraphWithHalo")
  .set_body([](DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef graph = args[0];
    IdArray nodes = args[1];
    int num_hops = args[2];
    auto hg = std::dynamic_pointer_cast<HeteroGraph>(graph.sptr());
    CHECK(hg) << "Invalid Heterograph pointer";
    std::vector<IdArray> vid_vec = {nodes};
    std::shared_ptr<HeteroSubgraph> subg(
      new HeteroSubgraph(hg->VertexSubgraph(vid_vec)));

    IdArray inner_nodes = nodes;

    std::vector<IdArray> outer_edge_ids_x_vec;
    std::vector<IdArray> outer_node_ids_x_vec;

    for (int i = 0; i < num_hops; i++) {
      EdgeArray halo_in_edges = hg->InEdges(0, inner_nodes);
      outer_edge_ids_x_vec.emplace_back(halo_in_edges.id);
      inner_nodes = halo_in_edges.src;
      outer_node_ids_x_vec.emplace_back(halo_in_edges.src);
    }
    auto all_edges_with_extra = aten::Concat(outer_edge_ids_x_vec);
    auto all_nodes_with_extra = aten::Concat(outer_node_ids_x_vec);
    auto all_halo_edge_ids =
      SetDiff1d(all_edges_with_extra, subg->induced_edges[0]);
    auto all_subgraph_edges =
      Concat({subg->induced_edges[0], all_halo_edge_ids});

    List<ObjectRef> ret;

    std::shared_ptr<HeteroSubgraph> out(
      new HeteroSubgraph(hg->EdgeSubgraph({all_subgraph_edges}, true)));
    int64_t total_num_edges = all_subgraph_edges->shape[0];
    int64_t inner_num_edges = subg->induced_edges[0]->shape[0];

    auto outer_edge_ids = aten::IndexSelect(out->graph->Edges(0, "eid").id,
                                            inner_num_edges, total_num_edges);
    auto outer_node_ids =
      SetDiff1d(out->induced_vertices[0], subg->induced_vertices[0]);
    ret.push_back(HeteroSubgraphRef(out));
    ret.push_back(Value(MakeValue(outer_node_ids)));
    ret.push_back(Value(MakeValue(outer_edge_ids)));
    *rv = ret;
  });
};  // namespace transform

};  // namespace dgl
