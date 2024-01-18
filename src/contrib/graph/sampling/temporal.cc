/**
 *  Copyright (c) 2020-2023 by Contributors
 * @file contrib/graph/sampling/temporal.cc
 * @brief Definition of temporal neighbor sampling APIs.
 */

#include <dgl/array.h>
#include <dgl/aten/macro.h>
#include <dgl/packed_func_ext.h>
#include <dgl/random.h>
#include <dgl/runtime/container.h>

#include "../../../array/cpu/rowwise_pick.h"
#include "../../../c_api_common.h"
#include "../../../graph/unit_graph.h"

using namespace dgl::runtime;
using namespace dgl::aten;

namespace dgl {
namespace sampling {
namespace {

void _CheckIdType(
    DGLDataType idtype,
    const std::unordered_map<std::string, NDArray>& ndarray_map) {
  for (const auto& item : ndarray_map) {
    CHECK(item.second.defined());
    CHECK_EQ(item.second->dtype, idtype);
  }
}

DGLContext _GetContext(
    const std::unordered_map<std::string, NDArray>& ndarray_map) {
  CHECK(!ndarray_map.empty());
  for (const auto& item : ndarray_map) {
    CHECK(item.second.defined());
    return item.second->ctx;
  }
  return DGLContext();
}

void _CheckContext(
    const std::unordered_map<std::string, NDArray>& ndarray_map,
    const DGLContext& ctx) {
  for (const auto& item : ndarray_map) {
    CHECK(item.second.defined());
    CHECK_EQ(item.second->ctx, ctx);
  }
}

void _CheckTimestamp(
    const std::unordered_map<std::string, NDArray>& timestamp_map) {
  const auto int64_dtype = DGLDataType{kDGLInt, 64, 1};
  for (const auto& item : timestamp_map) {
    CHECK(item.second.defined());
    CHECK_EQ(item.second->dtype, int64_dtype);
    CHECK_EQ(item.second->ndim, 1);
  }
}

template <typename IdxType>
inline aten::impl::NumPicksFn<IdxType> _GetNumPicksFn(
    int64_t num_samples, NDArray row_timestamp, NDArray col_timestamp,
    bool replace) {
  aten::impl::NumPicksFn<IdxType> num_picks_fn =
      [num_samples, row_timestamp, col_timestamp, replace](
          IdxType rowid, IdxType off, IdxType len, const IdxType* col,
          const IdxType* data) {
        const int64_t max_num_picks = (num_samples == -1) ? len : num_samples;
        const auto row_ts = row_timestamp.Ptr<int64_t>()[rowid];
        IdxType num = 0;
        for (IdxType i = off; i < off + len; ++i) {
          const auto col_ts = col_timestamp.Ptr<int64_t>()[col[i]];
          if (col_ts <= row_ts) ++num;
        }

        if (replace) {
          return static_cast<IdxType>(num == 0 ? 0 : num_samples);
        } else {
          return std::min(static_cast<IdxType>(max_num_picks), num);
        }
      };
  return num_picks_fn;
}

template <typename IdxType>
inline aten::impl::PickFn<IdxType> _GetPickFn(
    int64_t num_samples, NDArray row_timestamp, NDArray col_timestamp,
    bool replace) {
  aten::impl::PickFn<IdxType> pick_fn =
      [num_samples, row_timestamp, col_timestamp, replace](
          IdxType rowid, IdxType off, IdxType len, IdxType num_picks,
          const IdxType* col, const IdxType* data, IdxType* out_idx) {
        const auto row_ts = row_timestamp.Ptr<int64_t>()[rowid];
        std::vector<IdxType> candidate_col;
        candidate_col.reserve(len);
        for (IdxType i = off; i < off + len; ++i) {
          const auto ts = col_timestamp.Ptr<int64_t>()[col[i]];
          if (ts <= row_ts) candidate_col.push_back(i);
        }
        RandomEngine::ThreadLocal()->UniformChoice<IdxType>(
            num_picks, candidate_col.size(), out_idx, replace);
        for (int64_t j = 0; j < num_picks; ++j) {
          out_idx[j] = candidate_col[out_idx[j]];
        }
      };
  return pick_fn;
}

COOMatrix CSRRowWiseTemporalSampling(
    CSRMatrix mat, IdArray rows, int64_t num_samples, NDArray row_timestamp,
    NDArray col_timestamp, bool replace) {
  // If num_samples is -1, select all neighbors without replacement.
  replace = (replace && num_samples != -1);
  COOMatrix ret;
  ATEN_CSR_SWITCH(mat, XPU, IdType, "CSRRowWiseTemporalSampling", {
    CHECK(XPU == kDGLCPU);
    for (int64_t i = 0; i < rows->shape[0]; ++i) {
      const IdType rowid = rows.Ptr<IdType>()[i];
      CHECK_LT(rowid, mat.num_rows) << "Row ID (" << rowid << ") out of range.";
    }
    auto num_picks_fn = _GetNumPicksFn<IdType>(
        num_samples, row_timestamp, col_timestamp, replace);
    auto pick_fn =
        _GetPickFn<IdType>(num_samples, row_timestamp, col_timestamp, replace);
    ret = aten::impl::CSRRowWisePick(
        mat, rows, num_samples, replace, pick_fn, num_picks_fn);
  });
  return ret;
}

}  // namespace

HeteroSubgraph TemporalSampleNeighbors(
    const HeteroGraphPtr hg, const std::vector<std::string>& vtype_names,
    const std::unordered_map<std::string, NDArray>& nodes,
    const std::vector<int64_t>& fanouts,
    const std::unordered_map<std::string, NDArray>& timestamp, bool replace) {
  // Sanity check.
  CHECK_EQ(fanouts.size(), hg->NumEdgeTypes());
  CHECK_EQ(timestamp.size(), hg->NumVertexTypes());

  _CheckIdType(hg->DataType(), nodes);

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
    const auto& src_type_name = vtype_names[src_vtype];
    const int64_t num_nodes =
        nodes.count(dst_type_name) == 1 ? nodes.at(dst_type_name)->shape[0] : 0;

    if (num_nodes == 0 || fanouts[etype] == 0) {
      // Nothing to sample for this etype, create a placeholder relation graph
      subrels[etype] = UnitGraph::Empty(
          hg->GetRelationGraph(etype)->NumVertexTypes(),
          hg->NumVertices(src_vtype), hg->NumVertices(dst_vtype),
          hg->DataType(), ctx);
      induced_edges[etype] = aten::NullArray(hg->DataType(), ctx);
    } else {
      auto sampled_coo = CSRRowWiseTemporalSampling(
          hg->GetCSCMatrix(etype), nodes.at(dst_type_name), fanouts[etype],
          timestamp.at(dst_type_name), timestamp.at(src_type_name), replace);
      sampled_coo = aten::COOTranspose(sampled_coo);
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

DGL_REGISTER_GLOBAL(
    "contrib.sampling.temporal._CAPI_DGLTemporalSampleNeighbors")
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
