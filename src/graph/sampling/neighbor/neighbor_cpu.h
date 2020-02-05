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
    if (nodes[src_vtype]->shape[0] == 0) {
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
    std::vector<IdArray> adj = hg->GetAdj(etype, dir == EdgeDir::kOut, "csr");
    FloatArray prob_etype = prob[etype];
    if (prob_etype->shape[0] == 0) {
      // empty probability array; assume uniform
    }
  }
}

}  // namespace impl
}  // namespace sampling
}  // namespace dgl

#endif  // DGL_GRAPH_SAMPLING_NEIGHBOR_NEIGHBOR_CPU_H_
