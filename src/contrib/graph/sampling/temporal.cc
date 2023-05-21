/**
 *  Copyright (c) 2020-2023 by Contributors
 * @file contrib/graph/sampling/temporal.cc
 * @brief Definition of temporal neighbor sampling APIs.
 */

#include <dgl/array.h>
#include <dgl/aten/macro.h>
#include <dgl/packed_func_ext.h>
#include <dgl/runtime/container.h>

#include "../../../c_api_common.h"
#include "../../../graph/unit_graph.h"

using namespace dgl::runtime;
using namespace dgl::aten;

namespace dgl {
namespace sampling {

DGLContext _GetContext(
    const std::unordered_map<std::string, NDArray>& ndarray_map) {
  CHECK(ndarray_map.size() > 0);
  for (const auto& item : ndarray_map)
    return item.second->ctx;
  return DGLContext();
}

void _CheckContext(
    const std::unordered_map<std::string, NDArray>& ndarray_map,
    const DGLContext& ctx) {
  for (const auto& item : ndarray_map) {
    CHECK_EQ(item.second->ctx, ctx);
  }
}

void _CheckTimestamp(
    const std::unordered_map<std::string, NDArray>& timestamp_map) {
  const auto int64_dtype = DGLDataType{kDGLInt, 64, 1};
  for (const auto& item : timestamp_map) {
    CHECK_EQ(item.second->dtype, int64_dtype);
    CHECK_EQ(item.second->ndim, 1);
  }
}

HeteroSubgraph TemporalSampleNeighbors(
    const HeteroGraphPtr hg,
    const std::vector<std::string>& vtype_names,
    const std::unordered_map<std::string, NDArray>& nodes,
    const std::vector<int64_t>& fanouts,
    const std::unordered_map<std::string, NDArray>& timestamp,
    bool replace) {
  // Sanity check.
  CHECK_EQ(fanouts.size(), hg->NumEdgeTypes());
  CHECK_EQ(timestamp.size(), hg->NumVertexTypes());

  DGLContext ctx = _GetContext(nodes);
  CHECK_EQ(ctx.device_type, kDGLCPU)
    << "Temporal neighbor sampling does not support GPU.";
  _CheckContext(nodes, ctx);
  _CheckContext(timestamp, ctx);

  _CheckTimestamp(timestamp);

  std::vector<HeteroGraphPtr> subrels(hg->NumEdgeTypes());
  std::vector<IdArray> induced_edges(hg->NumEdgeTypes());
  for (dgl_type_t etype = 0; etype < hg->NumEdgeTypes(); ++etype) {
    auto pair = hg->meta_graph()->FindEdge(etype);
    const dgl_type_t src_vtype = pair.first;
    const dgl_type_t dst_vtype = pair.second;
    const auto& dst_type_name = vtype_names[dst_vtype];
    const int64_t num_nodes = nodes.count(dst_type_name) == 1? nodes.at(dst_type_name)->shape[0] : 0;

    if (num_nodes == 0 || fanouts[etype] == 0) {
      // Nothing to sample for this etype, create a placeholder relation graph
      subrels[etype] = UnitGraph::Empty(
          hg->GetRelationGraph(etype)->NumVertexTypes(),
          hg->NumVertices(src_vtype), hg->NumVertices(dst_vtype),
          hg->DataType(), ctx);
      induced_edges[etype] = aten::NullArray(hg->DataType(), ctx);
    } else {
      COOMatrix sampled_coo;
      /*auto sampled_coo = aten::CSRRowWiseTemporalSampling(
          hg->GetCSCMatrix(etype),
          nodes.at(dst_vtype),
          fanouts[etype],
          timestamp.at(dst_vtype),
          timestamp.at(src_vtype),
          replace
      );*/
      CHECK(false);
      subrels[etype] = UnitGraph::CreateFromCOO(
          hg->GetRelationGraph(etype)->NumVertexTypes(), sampled_coo.num_rows,
          sampled_coo.num_cols, sampled_coo.row, sampled_coo.col);
      induced_edges[etype] = sampled_coo.data;
    }
  }
  HeteroSubgraph ret;
  ret.graph =
      CreateHeteroGraph(hg->meta_graph(), subrels, hg->NumVerticesPerType());
  ret.induced_vertices.resize(hg->NumVertexTypes());
  ret.induced_edges = std::move(induced_edges);
  return ret;
}

DGL_REGISTER_GLOBAL("contrib.sampling.temporal._CAPI_DGLTemporalSampleNeighbors")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      HeteroGraphRef hg = args[0];
      const auto& vtype_names = ListValueToVector<std::string>(args[1]);
      const auto& nodes = MapValueToUnorderedMap<NDArray>(args[2]);
      const auto& fanout = ListValueToVector<int64_t>(args[3]);
      const auto& timestamp = MapValueToUnorderedMap<NDArray>(args[4]);
      const bool replace = args[5];

      std::shared_ptr<HeteroSubgraph> subg(new HeteroSubgraph);
      *subg = sampling::TemporalSampleNeighbors(
          hg.sptr(), vtype_names, nodes, fanout, timestamp, replace);

      *rv = HeteroSubgraphRef(subg);
    });

}  // namespace sampling
}  // namespace dgl
