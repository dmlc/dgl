/*!
 *  Copyright (c) 2020 by Contributors
 * \file graph/sampling/neighbor_cpu.h
 * \brief CPU implementation of neighborhood-based sampling algorithms.
 */

#ifndef DGL_GRAPH_SAMPLING_NEIGHBOR_NEIGHBOR_CPU_H_
#define DGL_GRAPH_SAMPLING_NEIGHBOR_NEIGHBOR_CPU_H_

#include <dgl/base_heterograph.h>
#include <dgl/array.h>
#include "../../unit_graph.h"
#include <vector>
#include <algorithm>

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
    for (int64_t j = 0; j < len; ++j) {
      ret_data[j] = array_data[idx_data[offset + j]];
    }
  });
  return ret;
}
}  // namespace

// Customizable functor for choosing from population (0~N-1). The edge weight
// array could be empty. Otherwise, its length is population.
template <typename IdxType>
using ChooseFunc = std::function<
  // choosed index
  IdArray(int64_t,  // how many to choose
          int64_t,  // population
          const FloatArray)>;   // edge weight

/*
 */
template <typename IdxType>
HeteroGraphPtr CPUSampleNeighbors(
    const HeteroGraphPtr hg,
    const std::vector<IdArray>& nodes,
    const std::vector<int64_t>& fanouts,
    EdgeDir dir,
    const std::vector<FloatArray>& weight,
    bool replace,
    ChooseFunc<IdxType> choose_fn) {
  std::vector<HeteroGraphPtr> subrels(hg->NumEdgeTypes());
  for (dgl_type_t etype = 0; etype < hg->NumEdgeTypes(); ++etype) {
    auto pair = hg->meta_graph()->FindEdge(etype);
    const dgl_type_t src_vtype = pair.first;
    const dgl_type_t dst_vtype = pair.second;
    const IdArray nodes_ntype = nodes[(dir == EdgeDir::kOut)? src_vtype : dst_vtype];
    const int64_t num_nodes = nodes_ntype->shape[0];
    if (num_nodes == 0) {
      // No node provided in the type, create a placeholder relation graph
      IdArray row = IdArray::Empty({0}, hg->DataType(), hg->Context());
      IdArray col = IdArray::Empty({0}, hg->DataType(), hg->Context());
      subrels[etype] = UnitGraph::CreateFromCOO(
        hg->GetRelationGraph(etype)->NumVertexTypes(),
        hg->NumVertices(src_vtype),
        hg->NumVertices(dst_vtype),
        row, col);
      continue;
    }
    // sample from one relation graph
    const IdxType* nodes_data = static_cast<IdxType*>(nodes_ntype->data);
    const std::vector<IdArray>& adj = hg->GetAdj(etype, dir == EdgeDir::kOut, "csr");
    const IdxType* indptr = static_cast<IdxType*>(adj[0]->data);
    const IdxType* indices = static_cast<IdxType*>(adj[1]->data);
    const int64_t fanout = fanouts[etype];

    // To leverage OMP parallelization, we create two arrays to store
    // sampled src and dst indices. Each array is of length num_nodes * fanout.
    // For nodes whose neighborhood size < fanout, the indices are padded with -1.
    //
    // We check whether all the given nodes
    // have at least fanout number of neighbors when replace is false.
    //
    // If the check holds, remove -1 elements by remove_if operation, which simply
    // moves valid elements to the head of arrays and create a view of the original
    // array. The implementation consumes a little extra memory than the actual requirement.
    //
    // Otherwise, directly use the row and col arrays to construct the result graph.
    
    bool all_has_fanout = true;
    if (replace) {
      all_has_fanout = true;
    } else {
#pragma omp parallel for reduction(&&:all_has_fanout)
      for (int64_t i = 0; i < num_nodes; ++i) {
        const IdxType nid = nodes_data[i];
        const IdxType len = indptr[nid + 1] - indptr[nid];
        all_has_fanout = all_has_fanout && (len >= fanout);
      }
    }

    //LOG(INFO) << "all_has_fanout: " << all_has_fanout;

    IdArray row = aten::Full(-1, num_nodes * fanout, sizeof(IdxType) * 8, hg->Context());
    IdArray col = aten::Full(-1, num_nodes * fanout, sizeof(IdxType) * 8, hg->Context());

    IdxType* row_data = static_cast<IdxType*>(row->data);
    IdxType* col_data = static_cast<IdxType*>(col->data);

//#pragma omp parallel for
    for (int64_t i = 0; i < num_nodes; ++i) {
      const IdxType nid = nodes_data[i];
      const IdxType off = indptr[nid];
      const IdxType len = indptr[nid + 1] - off;
      //LOG(INFO) << "nid=" << nid << " off=" << off << " len=" << len;
      if (len <= fanout && !replace) {
        // neighborhood size <= fanout and w/o replacement, take all neighbors
        for (int64_t j = 0; j < len; ++j) {
          row_data[i * fanout + j] = nid;
          col_data[i * fanout + j] = indices[off + j];
        }
      } else {
        FloatArray weight_selected = weight[etype];
        if (weight[etype]->shape[0] != 0) {
          weight_selected = LightSlice<IdxType>(weight[etype], adj[2], off, len);
        }
        IdArray chosen = choose_fn(fanout, len, weight_selected);
        const IdxType* chosen_data = static_cast<IdxType*>(chosen->data);
        for (int64_t j = 0; j < fanout; ++j) {
          row_data[i * fanout + j] = nid;
          col_data[i * fanout + j] = indices[off + chosen_data[j]];
        }
      }
    }

    if (!all_has_fanout) {
      // correct the array by remove_if
      IdxType* new_row_end = std::remove_if(row_data, row_data + num_nodes * fanout,
                                            [] (IdxType i) { return i == -1; });
      IdxType* new_col_end = std::remove_if(col_data, col_data + num_nodes * fanout,
                                            [] (IdxType i) { return i == -1; });
      const int64_t new_len = (new_row_end - row_data);
      CHECK_EQ(new_col_end - col_data, new_len);
      CHECK_LT(new_len, num_nodes * fanout);
      row = row.CreateView({new_len}, row->dtype);
      col = col.CreateView({new_len}, col->dtype);
    }

    if (dir == EdgeDir::kIn)
      std::swap(row, col);

    subrels[etype] = UnitGraph::CreateFromCOO(
      hg->GetRelationGraph(etype)->NumVertexTypes(),
      hg->NumVertices(src_vtype),
      hg->NumVertices(dst_vtype),
      row, col);
  }

  return CreateHeteroGraph(hg->meta_graph(), subrels);
}

}  // namespace impl
}  // namespace sampling
}  // namespace dgl

#endif  // DGL_GRAPH_SAMPLING_NEIGHBOR_NEIGHBOR_CPU_H_
