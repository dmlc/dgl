/*!
 *  Copyright (c) 2018 by Contributors
 * \file graph/graph.cc
 * \brief Graph operation implementation
 */
#include <dgl/graph_op.h>
#include <algorithm>

namespace dgl {
namespace {
inline bool IsValidIdArray(const IdArray& arr) {
  return arr->ctx.device_type == kDLCPU && arr->ndim == 1
    && arr->dtype.code == kDLInt && arr->dtype.bits == 64;
}
}  // namespace

Graph GraphOp::LineGraph(const Graph* g, bool backtracking) {
  Graph lg;
  lg.AddVertices(g->NumEdges());
  for (size_t i = 0; i < g->all_edges_src_.size(); ++i) {
    const auto u = g->all_edges_src_[i];
    const auto v = g->all_edges_dst_[i];
    for (size_t j = 0; j < g->adjlist_[v].succ.size(); ++j) {
      if (backtracking || (!backtracking && g->adjlist_[v].succ[j] != u)) {
        lg.AddEdge(i, g->adjlist_[v].edge_id[j]);
      }
    }
  }

  return lg;
}

Graph GraphOp::DisjointUnion(std::vector<const Graph*> graphs) {
  Graph rst;
  uint64_t cumsum = 0;
  for (const Graph* gr : graphs) {
    rst.AddVertices(gr->NumVertices());
    for (uint64_t i = 0; i < gr->NumEdges(); ++i) {
      rst.AddEdge(gr->all_edges_src_[i] + cumsum, gr->all_edges_dst_[i] + cumsum);
    }
    cumsum += gr->NumVertices();
  }
  return rst;
}

std::vector<Graph> GraphOp::DisjointPartitionByNum(const Graph* graph, int64_t num) {
  CHECK(num != 0 && graph->NumVertices() % num == 0)
    << "Number of partitions must evenly divide the number of nodes.";
  IdArray sizes = IdArray::Empty({num}, DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
  int64_t* sizes_data = static_cast<int64_t*>(sizes->data);
  std::fill(sizes_data, sizes_data + num, graph->NumVertices() / num);
  return DisjointPartitionBySizes(graph, sizes);
}

std::vector<Graph> GraphOp::DisjointPartitionBySizes(const Graph* graph, IdArray sizes) {
  const int64_t len = sizes->shape[0];
  const int64_t* sizes_data = static_cast<int64_t*>(sizes->data);
  std::vector<int64_t> cumsum;
  cumsum.push_back(0);
  for (int64_t i = 0; i < len; ++i) {
    cumsum.push_back(cumsum[i] + sizes_data[i]);
  }
  CHECK_EQ(cumsum[len], graph->NumVertices())
    << "Sum of the given sizes must equal to the number of nodes.";
  dgl_id_t node_offset = 0, edge_offset = 0;
  std::vector<Graph> rst(len);
  for (int64_t i = 0; i < len; ++i) {
    // copy adj
    rst[i].adjlist_.insert(rst[i].adjlist_.end(),
        graph->adjlist_.begin() + node_offset,
        graph->adjlist_.begin() + node_offset + sizes_data[i]);
    rst[i].reverse_adjlist_.insert(rst[i].reverse_adjlist_.end(),
        graph->reverse_adjlist_.begin() + node_offset,
        graph->reverse_adjlist_.begin() + node_offset + sizes_data[i]);
    // relabel adjs
    size_t num_edges = 0;
    for (auto& elist : rst[i].adjlist_) {
      for (size_t j = 0; j < elist.succ.size(); ++j) {
        elist.succ[j] -= node_offset;
        elist.edge_id[j] -= edge_offset;
      }
      num_edges += elist.succ.size();
    }
    for (auto& elist : rst[i].reverse_adjlist_) {
      for (size_t j = 0; j < elist.succ.size(); ++j) {
        elist.succ[j] -= node_offset;
        elist.edge_id[j] -= edge_offset;
      }
    }
    // copy edges
    rst[i].all_edges_src_.reserve(num_edges);
    rst[i].all_edges_dst_.reserve(num_edges);
    rst[i].num_edges_ = num_edges;
    for (size_t j = edge_offset; j < edge_offset + num_edges; ++j) {
      rst[i].all_edges_src_.push_back(graph->all_edges_src_[j] - node_offset);
      rst[i].all_edges_dst_.push_back(graph->all_edges_dst_[j] - node_offset);
    }
    // update offset
    CHECK_EQ(rst[i].NumVertices(), sizes_data[i]);
    CHECK_EQ(rst[i].NumEdges(), num_edges);
    node_offset += sizes_data[i];
    edge_offset += num_edges;
  }
  return rst;
}

IdArray GraphOp::MapParentIdToSubgraphId(IdArray parent_vids, IdArray query) {
  CHECK(IsValidIdArray(parent_vids)) << "Invalid parent id array.";
  CHECK(IsValidIdArray(query)) << "Invalid query id array.";
  const auto parent_len = parent_vids->shape[0];
  const auto query_len = query->shape[0];
  const dgl_id_t* parent_data = static_cast<dgl_id_t*>(parent_vids->data);
  const dgl_id_t* query_data = static_cast<dgl_id_t*>(query->data);
  IdArray rst = IdArray::Empty({query_len}, DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
  dgl_id_t* rst_data = static_cast<dgl_id_t*>(rst->data);

  const bool is_sorted = std::is_sorted(parent_data, parent_data + parent_len);
  if (is_sorted) {
#pragma omp parallel for
    for (int64_t i = 0; i < query_len; i++) {
      const dgl_id_t id = query_data[i];
      const auto it = std::find(parent_data, parent_data + parent_len, id);
      // If the vertex Id doesn't exist, the vid in the subgraph is -1.
      if (it != parent_data + parent_len) {
        rst_data[i] = it - parent_data;
      } else {
        rst_data[i] = -1;
      }
    }
  } else {
    std::unordered_map<dgl_id_t, dgl_id_t> parent_map;
    for (int64_t i = 0; i < parent_len; i++) {
      const dgl_id_t id = parent_data[i];
      parent_map[id] = i;
    }
#pragma omp parallel for
    for (int64_t i = 0; i < query_len; i++) {
      const dgl_id_t id = query_data[i];
      auto it = parent_map.find(id);
      // If the vertex Id doesn't exist, the vid in the subgraph is -1.
      if (it != parent_map.end()) {
        rst_data[i] = it->second;
      } else {
        rst_data[i] = -1;
      }
    }
  }
  return rst;
}

IdArray GraphOp::ExpandIds(IdArray ids, IdArray offset) {
  const auto id_len = ids->shape[0];
  const auto off_len = offset->shape[0];
  CHECK_EQ(id_len + 1, off_len);
  const dgl_id_t *id_data = static_cast<dgl_id_t*>(ids->data);
  const dgl_id_t *off_data = static_cast<dgl_id_t*>(offset->data);
  const int64_t len = off_data[off_len - 1];
  IdArray rst = IdArray::Empty({len}, DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
  dgl_id_t *rst_data = static_cast<dgl_id_t*>(rst->data);
  for (int64_t i = 0; i < id_len; i++) {
    const int64_t local_len = off_data[i + 1] - off_data[i];
    for (int64_t j = 0; j < local_len; j++) {
      rst_data[off_data[i] + j] = id_data[i];
    }
  }
  return rst;
}

}  // namespace dgl
