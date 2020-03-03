/*!
 *  Copyright (c) 2020 by Contributors
 * \file graph/transform/union_partition.cc
 * \brief Functions for partition, union multiple graphs.
 */
#include "../heterograph.h"
using namespace dgl::runtime;

namespace dgl {

HeteroGraphPtr DisjointUnionHeteroGraph(
    GraphPtr meta_graph, const std::vector<HeteroGraphPtr>& component_graphs) {
  CHECK_GT(component_graphs.size(), 0) << "Input graph list is empty";
  std::vector<HeteroGraphPtr> rel_graphs(meta_graph->NumEdges());

  // Loop over all canonical etypes
  for (dgl_type_t etype = 0; etype < meta_graph->NumEdges(); ++etype) {
    auto pair = meta_graph->FindEdge(etype);
    const dgl_type_t src_vtype = pair.first;
    const dgl_type_t dst_vtype = pair.second;
    dgl_id_t src_offset = 0, dst_offset = 0;
    std::vector<dgl_id_t> result_src, result_dst;

    // Loop over all graphs
    for (size_t i = 0; i < component_graphs.size(); ++i) {
      const auto& cg = component_graphs[i];
      EdgeArray edges = cg->Edges(etype);
      size_t num_edges = cg->NumEdges(etype);
      const dgl_id_t* edges_src_data = static_cast<const dgl_id_t*>(edges.src->data);
      const dgl_id_t* edges_dst_data = static_cast<const dgl_id_t*>(edges.dst->data);

      // Loop over all edges
      for (size_t j = 0; j < num_edges; ++j) {
        // TODO(mufei): Should use array operations to implement this.
        result_src.push_back(edges_src_data[j] + src_offset);
        result_dst.push_back(edges_dst_data[j] + dst_offset);
      }
      // Update offsets
      src_offset += cg->NumVertices(src_vtype);
      dst_offset += cg->NumVertices(dst_vtype);
    }
    HeteroGraphPtr rgptr = UnitGraph::CreateFromCOO(
      (src_vtype == dst_vtype)? 1 : 2,
      src_offset,
      dst_offset,
      aten::VecToIdArray(result_src),
      aten::VecToIdArray(result_dst));
    rel_graphs[etype] = rgptr;
  }
  return HeteroGraphPtr(new HeteroGraph(meta_graph, rel_graphs));
}

std::vector<HeteroGraphPtr> DisjointPartitionHeteroBySizes(
    GraphPtr meta_graph, HeteroGraphPtr batched_graph, IdArray vertex_sizes, IdArray edge_sizes) {
  // Sanity check for vertex sizes
  const uint64_t len_vertex_sizes = vertex_sizes->shape[0];
  const uint64_t* vertex_sizes_data = static_cast<uint64_t*>(vertex_sizes->data);
  const uint64_t num_vertex_types = meta_graph->NumVertices();
  const uint64_t batch_size = len_vertex_sizes / num_vertex_types;
  // Map vertex type to the corresponding node cum sum
  std::vector<std::vector<uint64_t>> vertex_cumsum;
  vertex_cumsum.resize(num_vertex_types);
  // Loop over all vertex types
  for (uint64_t vtype = 0; vtype < num_vertex_types; ++vtype) {
    vertex_cumsum[vtype].push_back(0);
    for (uint64_t g = 0; g < batch_size; ++g) {
      // We've flattened the number of vertices in the batch for all types
      vertex_cumsum[vtype].push_back(
        vertex_cumsum[vtype][g] + vertex_sizes_data[vtype * batch_size + g]);
    }
    CHECK_EQ(vertex_cumsum[vtype][batch_size], batched_graph->NumVertices(vtype))
      << "Sum of the given sizes must equal to the number of nodes for type " << vtype;
  }

  // Sanity check for edge sizes
  const uint64_t* edge_sizes_data = static_cast<uint64_t*>(edge_sizes->data);
  const uint64_t num_edge_types = meta_graph->NumEdges();
  // Map edge type to the corresponding edge cum sum
  std::vector<std::vector<uint64_t>> edge_cumsum;
  edge_cumsum.resize(num_edge_types);
  // Loop over all edge types
  for (uint64_t etype = 0; etype < num_edge_types; ++etype) {
    edge_cumsum[etype].push_back(0);
    for (uint64_t g = 0; g < batch_size; ++g) {
      // We've flattened the number of edges in the batch for all types
      edge_cumsum[etype].push_back(
        edge_cumsum[etype][g] + edge_sizes_data[etype * batch_size + g]);
    }
    CHECK_EQ(edge_cumsum[etype][batch_size], batched_graph->NumEdges(etype))
      << "Sum of the given sizes must equal to the number of edges for type " << etype;
  }

  // Construct relation graphs for unbatched graphs
  std::vector<std::vector<HeteroGraphPtr>> rel_graphs;
  rel_graphs.resize(batch_size);
  // Loop over all edge types
  for (uint64_t etype = 0; etype < num_edge_types; ++etype) {
    auto pair = meta_graph->FindEdge(etype);
    const dgl_type_t src_vtype = pair.first;
    const dgl_type_t dst_vtype = pair.second;
    EdgeArray edges = batched_graph->Edges(etype);
    const dgl_id_t* edges_src_data = static_cast<const dgl_id_t*>(edges.src->data);
    const dgl_id_t* edges_dst_data = static_cast<const dgl_id_t*>(edges.dst->data);
    // Loop over all graphs to be unbatched
    for (uint64_t g = 0; g < batch_size; ++g) {
      std::vector<dgl_id_t> result_src, result_dst;
      // Loop over the chunk of edges for the specified graph and edge type
      for (uint64_t e = edge_cumsum[etype][g]; e < edge_cumsum[etype][g + 1]; ++e) {
        // TODO(mufei): Should use array operations to implement this.
        result_src.push_back(edges_src_data[e] - vertex_cumsum[src_vtype][g]);
        result_dst.push_back(edges_dst_data[e] - vertex_cumsum[dst_vtype][g]);
      }
      HeteroGraphPtr rgptr = UnitGraph::CreateFromCOO(
        (src_vtype == dst_vtype)? 1 : 2,
        vertex_sizes_data[src_vtype * batch_size + g],
        vertex_sizes_data[dst_vtype * batch_size + g],
        aten::VecToIdArray(result_src),
        aten::VecToIdArray(result_dst));
      rel_graphs[g].push_back(rgptr);
    }
  }

  std::vector<HeteroGraphPtr> rst;
  for (uint64_t g = 0; g < batch_size; ++g) {
    rst.push_back(HeteroGraphPtr(new HeteroGraph(meta_graph, rel_graphs[g])));
  }
  return rst;
}

}  // namespace dgl
