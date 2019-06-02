/*!
 *  Copyright (c) 2018 by Contributors
 * \file graph/graph.cc
 * \brief Graph operation implementation
 */
#include <dgl/graph_op.h>
#include <dgl/immutable_graph.h>
#include <algorithm>
#include "../c_api_common.h"

namespace dgl {
namespace {
// generate consecutive dgl ids
class RangeIter : public std::iterator<std::input_iterator_tag, dgl_id_t> {
 public:
  explicit RangeIter(dgl_id_t from): cur_(from) {}

  RangeIter& operator++() {
    ++cur_;
    return *this;
  }

  RangeIter operator++(int) {
    RangeIter retval = *this;
    ++cur_;
    return retval;
  }
  bool operator==(RangeIter other) const {
    return cur_ == other.cur_;
  }
  bool operator!=(RangeIter other) const {
    return cur_ != other.cur_;
  }
  dgl_id_t operator*() const {
    return cur_;
  }

 private:
  dgl_id_t cur_;
};
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


ImmutableGraph GraphOp::DisjointUnion(std::vector<const ImmutableGraph *> graphs) {
  int64_t num_nodes = 0;
  int64_t num_edges = 0;
  for (const ImmutableGraph *gr : graphs) {
    num_nodes += gr->NumVertices();
    num_edges += gr->NumEdges();
  }
  IdArray indptr_arr = NewIdArray(num_nodes + 1);
  IdArray indices_arr = NewIdArray(num_edges);
  IdArray edge_ids_arr = NewIdArray(num_edges);
  dgl_id_t* indptr = static_cast<dgl_id_t*>(indptr_arr->data);
  dgl_id_t* indices = static_cast<dgl_id_t*>(indices_arr->data);
  dgl_id_t* edge_ids = static_cast<dgl_id_t*>(edge_ids_arr->data);

  indptr[0] = 0;
  dgl_id_t cum_num_nodes = 0;
  dgl_id_t cum_num_edges = 0;
  for (const ImmutableGraph *gr : graphs) {
    const CSRPtr g_csrptr = gr->GetInCSR();
    const int64_t g_num_nodes = g_csrptr->NumVertices();
    const int64_t g_num_edges = g_csrptr->NumEdges();
    dgl_id_t* g_indptr = static_cast<dgl_id_t*>(g_csrptr->indptr()->data);
    dgl_id_t* g_indices = static_cast<dgl_id_t*>(g_csrptr->indices()->data);
    dgl_id_t* g_edge_ids = static_cast<dgl_id_t*>(g_csrptr->edge_ids()->data);
    for (dgl_id_t i = 1; i < g_num_nodes + 1; ++i) {
      indptr[cum_num_nodes + i] = g_indptr[i] + cum_num_edges;
    }
    for (dgl_id_t i = 0; i < g_num_edges; ++i) {
      indices[cum_num_edges + i] = g_indices[i] + cum_num_nodes;
    }

    for (dgl_id_t i = 0; i < g_num_edges; ++i) {
      edge_ids[cum_num_edges + i] = g_edge_ids[i] + cum_num_edges;
    }
    cum_num_nodes += g_num_nodes;
    cum_num_edges += g_num_edges;
  }

  CSRPtr batched_csr_ptr = CSRPtr(new CSR(indptr_arr, indices_arr, edge_ids_arr));
  return ImmutableGraph(batched_csr_ptr, nullptr);
}

std::vector<ImmutableGraph> GraphOp::DisjointPartitionByNum(const ImmutableGraph *graph,
        int64_t num) {
  CHECK(num != 0 && graph->NumVertices() % num == 0)
    << "Number of partitions must evenly divide the number of nodes.";
  IdArray sizes = IdArray::Empty({num}, DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
  int64_t *sizes_data = static_cast<int64_t *>(sizes->data);
  std::fill(sizes_data, sizes_data + num, graph->NumVertices() / num);
  return DisjointPartitionBySizes(graph, sizes);
}

std::vector<ImmutableGraph> GraphOp::DisjointPartitionBySizes(const ImmutableGraph *batched_graph,
        IdArray sizes) {
  // TODO(minjie): use array views to speedup this operation
  const int64_t len = sizes->shape[0];
  const int64_t *sizes_data = static_cast<int64_t *>(sizes->data);
  std::vector<int64_t> cumsum;
  cumsum.reserve(len + 1);
  cumsum.push_back(0);
  for (int64_t i = 0; i < len; ++i) {
    cumsum.push_back(cumsum[i] + sizes_data[i]);
  }
  CHECK_EQ(cumsum[len], batched_graph->NumVertices())
    << "Sum of the given sizes must equal to the number of nodes.";
  std::vector<ImmutableGraph> rst;
  CSRPtr in_csr_ptr = batched_graph->GetInCSR();
  const dgl_id_t* indptr = static_cast<dgl_id_t*>(in_csr_ptr->indptr()->data);
  const dgl_id_t* indices = static_cast<dgl_id_t*>(in_csr_ptr->indices()->data);
  const dgl_id_t* edge_ids = static_cast<dgl_id_t*>(in_csr_ptr->edge_ids()->data);
  dgl_id_t cum_sum_edges = 0;
  for (int64_t i = 0; i < len; ++i) {
    const int64_t start_pos = cumsum[i];
    const int64_t end_pos = cumsum[i + 1];
    const int64_t g_num_nodes = sizes_data[i];
    const int64_t g_num_edges = indptr[end_pos] - indptr[start_pos];
    IdArray indptr_arr = NewIdArray(g_num_nodes + 1);
    IdArray indices_arr = NewIdArray(g_num_edges);
    IdArray edge_ids_arr = NewIdArray(g_num_edges);
    dgl_id_t* g_indptr = static_cast<dgl_id_t*>(indptr_arr->data);
    dgl_id_t* g_indices = static_cast<dgl_id_t*>(indices_arr->data);
    dgl_id_t* g_edge_ids = static_cast<dgl_id_t*>(edge_ids_arr->data);

    const dgl_id_t idoff = indptr[start_pos];
    g_indptr[0] = 0;
    for (int l = start_pos + 1; l < end_pos + 1; ++l) {
      g_indptr[l - start_pos] = indptr[l] - indptr[start_pos];
    }

    for (int j = indptr[start_pos]; j < indptr[end_pos]; ++j) {
      g_indices[j - idoff] = indices[j] - cumsum[i];
    }

    for (int k = indptr[start_pos]; k < indptr[end_pos]; ++k) {
      g_edge_ids[k - idoff] = edge_ids[k] - cum_sum_edges;
    }

    cum_sum_edges += g_num_edges;
    CSRPtr g_in_csr_ptr = CSRPtr(new CSR(indptr_arr, indices_arr, edge_ids_arr));
    rst.emplace_back(g_in_csr_ptr, nullptr);
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

ImmutableGraph GraphOp::ToSimpleGraph(const GraphInterface* graph) {
  std::vector<dgl_id_t> indptr(graph->NumVertices() + 1), indices;
  indptr[0] = 0;
  for (dgl_id_t src = 0; src < graph->NumVertices(); ++src) {
    std::unordered_set<dgl_id_t> hashmap;
    for (const dgl_id_t dst : graph->SuccVec(src)) {
      if (!hashmap.count(dst)) {
        indices.push_back(dst);
        hashmap.insert(dst);
      }
    }
    indptr[src+1] = indices.size();
  }
  CSRPtr csr(new CSR(graph->NumVertices(), indices.size(),
        indptr.begin(), indices.begin(), RangeIter(0), false));
  return ImmutableGraph(csr);
}

Graph GraphOp::BidirectedGraph(const Graph* g) {
  Graph bg;
  bg.AddVertices(g->NumVertices());
  std::unordered_map<int, std::unordered_map<int, int>> n_e;
  for (size_t i = 0; i < g->all_edges_src_.size(); ++i) {
    const auto u = g->all_edges_src_[i];
    const auto v = g->all_edges_dst_[i];
    n_e[u][v]++;
  }
  for (dgl_id_t u = 0; u < g->NumVertices(); ++u) {
    for (dgl_id_t v = u; v < g->NumVertices(); ++v) {
      const auto new_n_e = std::max(n_e[u][v], n_e[v][u]);
      if (new_n_e > 0) {
        IdArray us = IdArray::Empty({new_n_e}, DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
        int64_t* us_data = static_cast<int64_t*>(us->data);
        std::fill(us_data, us_data + new_n_e, u);
        if (u == v) {
          bg.AddEdges(us, us);
        } else {
          IdArray vs = IdArray::Empty({new_n_e}, DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
          int64_t* vs_data = static_cast<int64_t*>(vs->data);
          std::fill(vs_data, vs_data + new_n_e, v);
          bg.AddEdges(us, vs);
          bg.AddEdges(vs, us);
        }
      }
    }
  }
  return bg;
}

}  // namespace dgl
