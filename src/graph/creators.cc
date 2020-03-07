/*!
 *  Copyright (c) 2020 by Contributors
 * \file graph/creators.cc
 * \brief Functions for constructing graphs.
 */
#include "./heterograph.h"
using namespace dgl::runtime;

namespace dgl {

// creator implementation
HeteroGraphPtr CreateHeteroGraph(
    GraphPtr meta_graph, const std::vector<HeteroGraphPtr>& rel_graphs) {
  return HeteroGraphPtr(new HeteroGraph(meta_graph, rel_graphs));
}

HeteroGraphPtr CreateHeteroGraph(
    GraphPtr meta_graph,
    const std::vector<HeteroGraphPtr>& rel_graphs,
    const std::vector<int64_t>& num_nodes_per_type) {
  return HeteroGraphPtr(new HeteroGraph(meta_graph, rel_graphs, num_nodes_per_type));
}

HeteroGraphPtr CreateHeteroGraph(
    GraphPtr meta_graph,
    const std::vector<HeteroGraphPtr>& rel_graphs,
    std::vector<int64_t>&& num_nodes_per_type) {
  return HeteroGraphPtr(new HeteroGraph(meta_graph, rel_graphs, std::move(num_nodes_per_type)));
}

HeteroGraphPtr CreateFromCOO(
    int64_t num_vtypes, int64_t num_src, int64_t num_dst,
    IdArray row, IdArray col, SparseFormat restrict_format) {
  auto unit_g = UnitGraph::CreateFromCOO(
      num_vtypes, num_src, num_dst, row, col, restrict_format);
  return HeteroGraphPtr(new HeteroGraph(unit_g->meta_graph(), {unit_g}));
}

HeteroGraphPtr CreateFromCSR(
    int64_t num_vtypes, int64_t num_src, int64_t num_dst,
    IdArray indptr, IdArray indices, IdArray edge_ids,
    SparseFormat restrict_format) {
  auto unit_g = UnitGraph::CreateFromCSR(
      num_vtypes, num_src, num_dst, indptr, indices, edge_ids, restrict_format);
  return HeteroGraphPtr(new HeteroGraph(unit_g->meta_graph(), {unit_g}));
}

}  // namespace dgl
