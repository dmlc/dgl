/*!
 *  Copyright (c) 2020 by Contributors
 * \file graph/sampling/neighbor_cpu.h
 * \brief CPU implementation of neighborhood-based sampling algorithms.
 */

#ifndef DGL_GRAPH_SAMPLING_NEIGHBOR_NEIGHBOR_CPU_H_
#define DGL_GRAPH_SAMPLING_NEIGHBOR_NEIGHBOR_CPU_H_

#include <dgl/base_heterograph.h>
#include <dgl/array.h>
#include <dgl/random.h>
#include "../../unit_graph.h"
#include <vector>

namespace dgl {
namespace sampling {
namespace impl {
namespace {
// Equivalent to numpy expression: array[idx[offset:offset+len]]
template <typename IdxType>
inline FloatArray LightSlice(FloatArray array, IdArray idx,
                             int64_t offset, int64_t len) {
  FloatArray ret = FloatArray::Empty({len}, array->dtype, array->ctx);
  const IdxType* idx_data = static_cast<IdxType*>(idx->data);
  ATEN_FLOAT_TYPE_SWITCH(array->dtype, DType, "array", {
    const DType* array_data = static_cast<DType*>(array->data);
    DType* ret_data = static_cast<DType*>(ret->data);
    for (int64_t j = offset; j < offset + len; ++j)
      ret_data[j] = array_data[idx_data[j]];
  });
  return ret;
}
}  // namespace

template<typename IdxType>
HeteroGraphPtr CPUSampleNeighbors(
    const HeteroGraphPtr hg,
    const std::vector<IdArray>& nodes,
    const std::vector<int64_t>& fanouts,
    EdgeDir dir,
    const std::vector<FloatArray>& prob,
    bool replace) {
  std::vector<HeteroGraphPtr> subrels(hg->NumEdgeTypes());
  for (dgl_type_t etype = 0; etype < hg->NumEdgeTypes(); ++etype) {
    auto pair = hg->meta_graph()->FindEdge(etype);
    const dgl_type_t src_vtype = pair.first;
    const dgl_type_t dst_vtype = pair.second;
    const int64_t num_nodes = nodes[src_vtype]->shape[0];
    if (num_nodes == 0) {
      // No node provided in the type, create a placeholder relation graph
      IdArray row = IdArray::Empty({}, hg->DataType(), hg->Context());
      IdArray col = IdArray::Empty({}, hg->DataType(), hg->Context());
      subrels.push_back(UnitGraph::CreateFromCOO(
          hg->GetRelationGraph(etype)->NumVertexTypes(),
          hg->NumVertices(src_vtype),
          hg->NumVertices(dst_vtype),
          row, col));
      continue;
    }
    // sample from one relation graph
    const IdxType* nodes_data = static_cast<IdxType*>(nodes[src_vtype]->data);
    const std::vector<IdArray>& adj = hg->GetAdj(etype, dir == EdgeDir::kOut, "csr");
    const IdxType* indptr = static_cast<IdxType*>(adj[0]->data);
    const IdxType* indices = static_cast<IdxType*>(adj[1]->data);
    const int64_t fanout = fanouts[etype];
    // To leverage OMP parallelization, we first create two vectors to store
    // sampled src and dst indices. Each vector is of length num_nodes * fanout.
    // For nodes whose neighborhood size < fanout, the indices are padded with -1.
    // We then remove -1 elements in the two vectors to produce the final result.
    std::vector<IdxType> row(num_nodes * fanout, -1), col(num_nodes * fanout, -1);
#pragma omp parallel for
    for (int64_t i = 0; i < num_nodes; ++i) {
      const IdxType nid = nodes_data[i];
      const IdxType off = indptr[nid];
      const IdxType len = indptr[nid + 1] - off;
      if (len <= fanout && !replace) {
        // neighborhood size <= fanout and w/o replacement, take all neighbors
        for (int64_t j = 0; j < len; ++j) {
          row[i * fanout + j] = nid;
          col[i * fanout + j] = indices[off + j];
        }
      } else {
        IdArray chosen;
        if (prob[etype]->shape[0] == 0) {
          // empty prob array; assume uniform
          //chosen = RandomEngine::ThreadLocal()->UniformChoice<IdxType>(
              //fanout, len, replace);
        } else {
          FloatArray prob_selected = LightSlice(prob[etype], adj[2], off, len);
          chosen = RandomEngine::ThreadLocal()->Choice<IdxType>(
              fanout, prob_selected, replace);
        }
        const IdxType* chosen_data = static_cast<IdxType*>(chosen->data);
        for (int64_t j = 0; j < fanout; ++j) {
          row[i * fanout + j] = nid;
          col[i * fanout + j] = indices[off + chosen_data[j]];
        }
      }
    }
  }
}

}  // namespace impl
}  // namespace sampling
}  // namespace dgl

#endif  // DGL_GRAPH_SAMPLING_NEIGHBOR_NEIGHBOR_CPU_H_
