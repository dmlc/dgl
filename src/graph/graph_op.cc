/**
 *  Copyright (c) 2018 by Contributors
 * @file graph/graph.cc
 * @brief Graph operation implementation
 */
#include <dgl/array.h>
#include <dgl/graph_op.h>
#include <dgl/immutable_graph.h>
#include <dgl/packed_func_ext.h>
#include <dgl/runtime/container.h>
#include <dgl/runtime/parallel_for.h>

#include <algorithm>

#include "../c_api_common.h"

using namespace dgl::runtime;

namespace dgl {
namespace {
// generate consecutive dgl ids
class RangeIter : public std::iterator<std::input_iterator_tag, dgl_id_t> {
 public:
  explicit RangeIter(dgl_id_t from) : cur_(from) {}

  RangeIter& operator++() {
    ++cur_;
    return *this;
  }

  RangeIter operator++(int) {
    RangeIter retval = *this;
    ++cur_;
    return retval;
  }
  bool operator==(RangeIter other) const { return cur_ == other.cur_; }
  bool operator!=(RangeIter other) const { return cur_ != other.cur_; }
  dgl_id_t operator*() const { return cur_; }

 private:
  dgl_id_t cur_;
};

bool IsMutable(GraphPtr g) {
  MutableGraphPtr mg = std::dynamic_pointer_cast<Graph>(g);
  return mg != nullptr;
}

}  // namespace

GraphPtr GraphOp::Reverse(GraphPtr g) {
  ImmutableGraphPtr ig = std::dynamic_pointer_cast<ImmutableGraph>(g);
  CHECK(ig) << "Reverse is only supported on immutable graph";
  return ig->Reverse();
}

GraphPtr GraphOp::LineGraph(GraphPtr g, bool backtracking) {
  MutableGraphPtr mg = std::dynamic_pointer_cast<Graph>(g);
  CHECK(mg) << "Line graph transformation is only supported on mutable graph";
  MutableGraphPtr lg = Graph::Create();
  lg->AddVertices(g->NumEdges());
  for (size_t i = 0; i < mg->all_edges_src_.size(); ++i) {
    const auto u = mg->all_edges_src_[i];
    const auto v = mg->all_edges_dst_[i];
    for (size_t j = 0; j < mg->adjlist_[v].succ.size(); ++j) {
      if (backtracking || (!backtracking && mg->adjlist_[v].succ[j] != u)) {
        lg->AddEdge(i, mg->adjlist_[v].edge_id[j]);
      }
    }
  }
  return lg;
}

GraphPtr GraphOp::DisjointUnion(std::vector<GraphPtr> graphs) {
  CHECK_GT(graphs.size(), 0) << "Input graph list is empty";
  if (IsMutable(graphs[0])) {
    // Disjointly union of a list of mutable graph inputs. The result is
    // also a mutable graph.
    MutableGraphPtr rst = Graph::Create();
    uint64_t cumsum = 0;
    for (GraphPtr gr : graphs) {
      MutableGraphPtr mg = std::dynamic_pointer_cast<Graph>(gr);
      CHECK(mg) << "All the input graphs should be mutable graphs.";
      rst->AddVertices(gr->NumVertices());
      for (uint64_t i = 0; i < gr->NumEdges(); ++i) {
        // TODO(minjie): quite ugly to expose internal members
        rst->AddEdge(
            mg->all_edges_src_[i] + cumsum, mg->all_edges_dst_[i] + cumsum);
      }
      cumsum += gr->NumVertices();
    }
    return rst;
  } else {
    // Disjointly union of a list of immutable graph inputs. The result is
    // also an immutable graph.
    int64_t num_nodes = 0;
    int64_t num_edges = 0;
    for (auto gr : graphs) {
      num_nodes += gr->NumVertices();
      num_edges += gr->NumEdges();
    }
    IdArray indptr_arr = aten::NewIdArray(num_nodes + 1);
    IdArray indices_arr = aten::NewIdArray(num_edges);
    IdArray edge_ids_arr = aten::NewIdArray(num_edges);
    dgl_id_t* indptr = static_cast<dgl_id_t*>(indptr_arr->data);
    dgl_id_t* indices = static_cast<dgl_id_t*>(indices_arr->data);
    dgl_id_t* edge_ids = static_cast<dgl_id_t*>(edge_ids_arr->data);

    indptr[0] = 0;
    dgl_id_t cum_num_nodes = 0;
    dgl_id_t cum_num_edges = 0;
    for (auto g : graphs) {
      ImmutableGraphPtr gr = std::dynamic_pointer_cast<ImmutableGraph>(g);
      CHECK(gr) << "All the input graphs should be immutable graphs.";
      // TODO(minjie): why in csr?
      const CSRPtr g_csrptr = gr->GetInCSR();
      const uint64_t g_num_nodes = g_csrptr->NumVertices();
      const uint64_t g_num_edges = g_csrptr->NumEdges();
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

    return ImmutableGraph::CreateFromCSR(
        indptr_arr, indices_arr, edge_ids_arr, "in");
  }
}

std::vector<GraphPtr> GraphOp::DisjointPartitionByNum(
    GraphPtr graph, int64_t num) {
  CHECK(num != 0 && graph->NumVertices() % num == 0)
      << "Number of partitions must evenly divide the number of nodes.";
  IdArray sizes = IdArray::Empty(
      {num}, DGLDataType{kDGLInt, 64, 1}, DGLContext{kDGLCPU, 0});
  int64_t* sizes_data = static_cast<int64_t*>(sizes->data);
  std::fill(sizes_data, sizes_data + num, graph->NumVertices() / num);
  return DisjointPartitionBySizes(graph, sizes);
}

std::vector<GraphPtr> GraphOp::DisjointPartitionBySizes(
    GraphPtr batched_graph, IdArray sizes) {
  const int64_t len = sizes->shape[0];
  const int64_t* sizes_data = static_cast<int64_t*>(sizes->data);
  std::vector<int64_t> cumsum;
  cumsum.push_back(0);
  for (int64_t i = 0; i < len; ++i) {
    cumsum.push_back(cumsum[i] + sizes_data[i]);
  }
  CHECK_EQ(cumsum[len], batched_graph->NumVertices())
      << "Sum of the given sizes must equal to the number of nodes.";

  std::vector<GraphPtr> rst;
  if (IsMutable(batched_graph)) {
    // Input is a mutable graph. Partition it into several mutable graphs.
    MutableGraphPtr graph = std::dynamic_pointer_cast<Graph>(batched_graph);
    dgl_id_t node_offset = 0, edge_offset = 0;
    for (int64_t i = 0; i < len; ++i) {
      MutableGraphPtr mg = Graph::Create();
      // TODO(minjie): quite ugly to expose internal members
      // copy adj
      mg->adjlist_.insert(
          mg->adjlist_.end(), graph->adjlist_.begin() + node_offset,
          graph->adjlist_.begin() + node_offset + sizes_data[i]);
      mg->reverse_adjlist_.insert(
          mg->reverse_adjlist_.end(),
          graph->reverse_adjlist_.begin() + node_offset,
          graph->reverse_adjlist_.begin() + node_offset + sizes_data[i]);
      // relabel adjs
      size_t num_edges = 0;
      for (auto& elist : mg->adjlist_) {
        for (size_t j = 0; j < elist.succ.size(); ++j) {
          elist.succ[j] -= node_offset;
          elist.edge_id[j] -= edge_offset;
        }
        num_edges += elist.succ.size();
      }
      for (auto& elist : mg->reverse_adjlist_) {
        for (size_t j = 0; j < elist.succ.size(); ++j) {
          elist.succ[j] -= node_offset;
          elist.edge_id[j] -= edge_offset;
        }
      }
      // copy edges
      mg->all_edges_src_.reserve(num_edges);
      mg->all_edges_dst_.reserve(num_edges);
      mg->num_edges_ = num_edges;
      for (size_t j = edge_offset; j < edge_offset + num_edges; ++j) {
        mg->all_edges_src_.push_back(graph->all_edges_src_[j] - node_offset);
        mg->all_edges_dst_.push_back(graph->all_edges_dst_[j] - node_offset);
      }
      // push to rst
      rst.push_back(mg);
      // update offset
      CHECK_EQ(rst[i]->NumVertices(), sizes_data[i]);
      CHECK_EQ(rst[i]->NumEdges(), num_edges);
      node_offset += sizes_data[i];
      edge_offset += num_edges;
    }
  } else {
    // Input is an immutable graph. Partition it into several multiple graphs.
    ImmutableGraphPtr graph =
        std::dynamic_pointer_cast<ImmutableGraph>(batched_graph);
    // TODO(minjie): why in csr?
    CSRPtr in_csr_ptr = graph->GetInCSR();
    const dgl_id_t* indptr = static_cast<dgl_id_t*>(in_csr_ptr->indptr()->data);
    const dgl_id_t* indices =
        static_cast<dgl_id_t*>(in_csr_ptr->indices()->data);
    const dgl_id_t* edge_ids =
        static_cast<dgl_id_t*>(in_csr_ptr->edge_ids()->data);
    dgl_id_t cum_sum_edges = 0;
    for (int64_t i = 0; i < len; ++i) {
      const int64_t start_pos = cumsum[i];
      const int64_t end_pos = cumsum[i + 1];
      const int64_t g_num_nodes = sizes_data[i];
      const int64_t g_num_edges = indptr[end_pos] - indptr[start_pos];
      IdArray indptr_arr = aten::NewIdArray(g_num_nodes + 1);
      IdArray indices_arr = aten::NewIdArray(g_num_edges);
      IdArray edge_ids_arr = aten::NewIdArray(g_num_edges);
      dgl_id_t* g_indptr = static_cast<dgl_id_t*>(indptr_arr->data);
      dgl_id_t* g_indices = static_cast<dgl_id_t*>(indices_arr->data);
      dgl_id_t* g_edge_ids = static_cast<dgl_id_t*>(edge_ids_arr->data);

      const dgl_id_t idoff = indptr[start_pos];
      g_indptr[0] = 0;
      for (int l = start_pos + 1; l < end_pos + 1; ++l) {
        g_indptr[l - start_pos] = indptr[l] - indptr[start_pos];
      }

      for (dgl_id_t j = indptr[start_pos]; j < indptr[end_pos]; ++j) {
        g_indices[j - idoff] = indices[j] - cumsum[i];
      }

      for (dgl_id_t k = indptr[start_pos]; k < indptr[end_pos]; ++k) {
        g_edge_ids[k - idoff] = edge_ids[k] - cum_sum_edges;
      }

      cum_sum_edges += g_num_edges;
      rst.push_back(ImmutableGraph::CreateFromCSR(
          indptr_arr, indices_arr, edge_ids_arr, "in"));
    }
  }
  return rst;
}

IdArray GraphOp::MapParentIdToSubgraphId(IdArray parent_vids, IdArray query) {
  CHECK(aten::IsValidIdArray(parent_vids)) << "Invalid parent id array.";
  CHECK(aten::IsValidIdArray(query)) << "Invalid query id array.";
  const auto parent_len = parent_vids->shape[0];
  const auto query_len = query->shape[0];
  const dgl_id_t* parent_data = static_cast<dgl_id_t*>(parent_vids->data);
  const dgl_id_t* query_data = static_cast<dgl_id_t*>(query->data);
  IdArray rst = IdArray::Empty(
      {query_len}, DGLDataType{kDGLInt, 64, 1}, DGLContext{kDGLCPU, 0});
  dgl_id_t* rst_data = static_cast<dgl_id_t*>(rst->data);

  const bool is_sorted = std::is_sorted(parent_data, parent_data + parent_len);
  if (is_sorted) {
    runtime::parallel_for(0, query_len, [&](size_t b, size_t e) {
      for (auto i = b; i < e; ++i) {
        const dgl_id_t id = query_data[i];
        const auto it = std::find(parent_data, parent_data + parent_len, id);
        // If the vertex Id doesn't exist, the vid in the subgraph is -1.
        if (it != parent_data + parent_len) {
          rst_data[i] = it - parent_data;
        } else {
          rst_data[i] = -1;
        }
      }
    });
  } else {
    std::unordered_map<dgl_id_t, dgl_id_t> parent_map;
    for (int64_t i = 0; i < parent_len; i++) {
      const dgl_id_t id = parent_data[i];
      parent_map[id] = i;
    }
    runtime::parallel_for(0, query_len, [&](size_t b, size_t e) {
      for (auto i = b; i < e; ++i) {
        const dgl_id_t id = query_data[i];
        auto it = parent_map.find(id);
        // If the vertex Id doesn't exist, the vid in the subgraph is -1.
        if (it != parent_map.end()) {
          rst_data[i] = it->second;
        } else {
          rst_data[i] = -1;
        }
      }
    });
  }
  return rst;
}

IdArray GraphOp::ExpandIds(IdArray ids, IdArray offset) {
  const auto id_len = ids->shape[0];
  const auto off_len = offset->shape[0];
  CHECK_EQ(id_len + 1, off_len);
  const dgl_id_t* id_data = static_cast<dgl_id_t*>(ids->data);
  const dgl_id_t* off_data = static_cast<dgl_id_t*>(offset->data);
  const int64_t len = off_data[off_len - 1];
  IdArray rst = IdArray::Empty(
      {len}, DGLDataType{kDGLInt, 64, 1}, DGLContext{kDGLCPU, 0});
  dgl_id_t* rst_data = static_cast<dgl_id_t*>(rst->data);
  for (int64_t i = 0; i < id_len; i++) {
    const int64_t local_len = off_data[i + 1] - off_data[i];
    for (int64_t j = 0; j < local_len; j++) {
      rst_data[off_data[i] + j] = id_data[i];
    }
  }
  return rst;
}

GraphPtr GraphOp::ToSimpleGraph(GraphPtr graph) {
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
    indptr[src + 1] = indices.size();
  }
  CSRPtr csr(new CSR(
      graph->NumVertices(), indices.size(), indptr.begin(), indices.begin(),
      RangeIter(0)));
  return std::make_shared<ImmutableGraph>(csr);
}

GraphPtr GraphOp::ToBidirectedMutableGraph(GraphPtr g) {
  std::unordered_map<int, std::unordered_map<int, int>> n_e;
  for (dgl_id_t u = 0; u < g->NumVertices(); ++u) {
    for (const dgl_id_t v : g->SuccVec(u)) {
      n_e[u][v]++;
    }
  }

  GraphPtr bg = Graph::Create();
  bg->AddVertices(g->NumVertices());
  for (dgl_id_t u = 0; u < g->NumVertices(); ++u) {
    for (dgl_id_t v = u; v < g->NumVertices(); ++v) {
      const auto new_n_e = std::max(n_e[u][v], n_e[v][u]);
      if (new_n_e > 0) {
        IdArray us = aten::NewIdArray(new_n_e);
        dgl_id_t* us_data = static_cast<dgl_id_t*>(us->data);
        std::fill(us_data, us_data + new_n_e, u);
        if (u == v) {
          bg->AddEdges(us, us);
        } else {
          IdArray vs = aten::NewIdArray(new_n_e);
          dgl_id_t* vs_data = static_cast<dgl_id_t*>(vs->data);
          std::fill(vs_data, vs_data + new_n_e, v);
          bg->AddEdges(us, vs);
          bg->AddEdges(vs, us);
        }
      }
    }
  }
  return bg;
}

GraphPtr GraphOp::ToBidirectedImmutableGraph(GraphPtr g) {
  std::unordered_map<int, std::unordered_map<int, int>> n_e;
  for (dgl_id_t u = 0; u < g->NumVertices(); ++u) {
    for (const dgl_id_t v : g->SuccVec(u)) {
      n_e[u][v]++;
    }
  }

  std::vector<dgl_id_t> srcs, dsts;
  for (dgl_id_t u = 0; u < g->NumVertices(); ++u) {
    std::unordered_set<dgl_id_t> hashmap;
    std::vector<dgl_id_t> nbrs;
    for (const dgl_id_t v : g->PredVec(u)) {
      if (!hashmap.count(v)) {
        nbrs.push_back(v);
        hashmap.insert(v);
      }
    }
    for (const dgl_id_t v : g->SuccVec(u)) {
      if (!hashmap.count(v)) {
        nbrs.push_back(v);
        hashmap.insert(v);
      }
    }
    for (const dgl_id_t v : nbrs) {
      const auto new_n_e = std::max(n_e[u][v], n_e[v][u]);
      for (int i = 0; i < new_n_e; ++i) {
        srcs.push_back(v);
        dsts.push_back(u);
      }
    }
  }

  IdArray srcs_array = aten::VecToIdArray(srcs);
  IdArray dsts_array = aten::VecToIdArray(dsts);
  return ImmutableGraph::CreateFromCOO(
      g->NumVertices(), srcs_array, dsts_array);
}

HaloSubgraph GraphOp::GetSubgraphWithHalo(
    GraphPtr g, IdArray nodes, int num_hops) {
  const dgl_id_t* nid = static_cast<dgl_id_t*>(nodes->data);
  const auto id_len = nodes->shape[0];
  // A map contains all nodes in the subgraph.
  // The key is the old node Ids, the value indicates whether a node is a inner
  // node.
  std::unordered_map<dgl_id_t, bool> all_nodes;
  // The old Ids of all nodes. We want to preserve the order of the nodes in the
  // vector. The first few nodes are the inner nodes in the subgraph.
  std::vector<dgl_id_t> old_node_ids(nid, nid + id_len);
  std::vector<std::vector<dgl_id_t>> outer_nodes(num_hops);
  for (int64_t i = 0; i < id_len; i++) all_nodes[nid[i]] = true;
  auto orig_nodes = all_nodes;

  std::vector<dgl_id_t> edge_src, edge_dst, edge_eid;

  // When we deal with in-edges, we need to do two things:
  // * find the edges inside the partition and the edges between partitions.
  // * find the nodes outside the partition that connect the partition.
  EdgeArray in_edges = g->InEdges(nodes);
  auto src = in_edges.src;
  auto dst = in_edges.dst;
  auto eid = in_edges.id;
  auto num_edges = eid->shape[0];
  const dgl_id_t* src_data = static_cast<dgl_id_t*>(src->data);
  const dgl_id_t* dst_data = static_cast<dgl_id_t*>(dst->data);
  const dgl_id_t* eid_data = static_cast<dgl_id_t*>(eid->data);
  for (int64_t i = 0; i < num_edges; i++) {
    // We check if the source node is in the original node.
    auto it1 = orig_nodes.find(src_data[i]);
    if (it1 != orig_nodes.end() || num_hops > 0) {
      edge_src.push_back(src_data[i]);
      edge_dst.push_back(dst_data[i]);
      edge_eid.push_back(eid_data[i]);
    }
    // We need to expand only if the node hasn't been seen before.
    auto it = all_nodes.find(src_data[i]);
    if (it == all_nodes.end() && num_hops > 0) {
      all_nodes[src_data[i]] = false;
      old_node_ids.push_back(src_data[i]);
      outer_nodes[0].push_back(src_data[i]);
    }
  }

  // Now we need to traverse the graph with the in-edges to access nodes
  // and edges more hops away.
  for (int k = 1; k < num_hops; k++) {
    const std::vector<dgl_id_t>& nodes = outer_nodes[k - 1];
    EdgeArray in_edges = g->InEdges(aten::VecToIdArray(nodes));
    auto src = in_edges.src;
    auto dst = in_edges.dst;
    auto eid = in_edges.id;
    auto num_edges = eid->shape[0];
    const dgl_id_t* src_data = static_cast<dgl_id_t*>(src->data);
    const dgl_id_t* dst_data = static_cast<dgl_id_t*>(dst->data);
    const dgl_id_t* eid_data = static_cast<dgl_id_t*>(eid->data);
    for (int64_t i = 0; i < num_edges; i++) {
      edge_src.push_back(src_data[i]);
      edge_dst.push_back(dst_data[i]);
      edge_eid.push_back(eid_data[i]);
      // If we haven't seen this node.
      auto it = all_nodes.find(src_data[i]);
      if (it == all_nodes.end()) {
        all_nodes[src_data[i]] = false;
        old_node_ids.push_back(src_data[i]);
        outer_nodes[k].push_back(src_data[i]);
      }
    }
  }

  // We assign new Ids to the nodes in the subgraph. We ensure that the HALO
  // nodes are behind the input nodes.
  std::unordered_map<dgl_id_t, dgl_id_t> old2new;
  for (size_t i = 0; i < old_node_ids.size(); i++) {
    old2new[old_node_ids[i]] = i;
  }

  num_edges = edge_src.size();
  IdArray new_src = IdArray::Empty(
      {num_edges}, DGLDataType{kDGLInt, 64, 1}, DGLContext{kDGLCPU, 0});
  IdArray new_dst = IdArray::Empty(
      {num_edges}, DGLDataType{kDGLInt, 64, 1}, DGLContext{kDGLCPU, 0});
  dgl_id_t* new_src_data = static_cast<dgl_id_t*>(new_src->data);
  dgl_id_t* new_dst_data = static_cast<dgl_id_t*>(new_dst->data);
  for (size_t i = 0; i < edge_src.size(); i++) {
    new_src_data[i] = old2new[edge_src[i]];
    new_dst_data[i] = old2new[edge_dst[i]];
  }

  std::vector<int> inner_nodes(old_node_ids.size());
  for (size_t i = 0; i < old_node_ids.size(); i++) {
    dgl_id_t old_nid = old_node_ids[i];
    inner_nodes[i] = all_nodes[old_nid];
  }

  GraphPtr subg =
      ImmutableGraph::CreateFromCOO(old_node_ids.size(), new_src, new_dst);
  HaloSubgraph halo_subg;
  halo_subg.graph = subg;
  halo_subg.induced_vertices = aten::VecToIdArray(old_node_ids);
  halo_subg.induced_edges = aten::VecToIdArray(edge_eid);
  // TODO(zhengda) we need to switch to 8 bytes afterwards.
  halo_subg.inner_nodes = aten::VecToIdArray<int>(inner_nodes, 32);
  return halo_subg;
}

GraphPtr GraphOp::ReorderImmutableGraph(
    ImmutableGraphPtr ig, IdArray new_order) {
  CSRPtr in_csr, out_csr;
  COOPtr coo;
  // We only need to reorder one of the graph structure.
  if (ig->HasInCSR()) {
    in_csr = ig->GetInCSR();
    auto csrmat = in_csr->ToCSRMatrix();
    auto new_csrmat = aten::CSRReorder(csrmat, new_order, new_order);
    in_csr =
        CSRPtr(new CSR(new_csrmat.indptr, new_csrmat.indices, new_csrmat.data));
  } else if (ig->HasOutCSR()) {
    out_csr = ig->GetOutCSR();
    auto csrmat = out_csr->ToCSRMatrix();
    auto new_csrmat = aten::CSRReorder(csrmat, new_order, new_order);
    out_csr =
        CSRPtr(new CSR(new_csrmat.indptr, new_csrmat.indices, new_csrmat.data));
  } else {
    coo = ig->GetCOO();
    auto coomat = coo->ToCOOMatrix();
    auto new_coomat = aten::COOReorder(coomat, new_order, new_order);
    coo = COOPtr(new COO(ig->NumVertices(), new_coomat.row, new_coomat.col));
  }
  if (in_csr || out_csr)
    return GraphPtr(new ImmutableGraph(in_csr, out_csr));
  else
    return GraphPtr(new ImmutableGraph(coo));
}

DGL_REGISTER_GLOBAL("transform._CAPI_DGLPartitionWithHalo")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      GraphRef graph = args[0];
      IdArray node_parts = args[1];
      int num_hops = args[2];

      const dgl_id_t* part_data = static_cast<dgl_id_t*>(node_parts->data);
      int64_t num_nodes = node_parts->shape[0];
      std::unordered_map<int, std::vector<dgl_id_t>> part_map;
      for (int64_t i = 0; i < num_nodes; i++) {
        dgl_id_t part_id = part_data[i];
        auto it = part_map.find(part_id);
        if (it == part_map.end()) {
          std::vector<dgl_id_t> vec;
          vec.push_back(i);
          part_map[part_id] = vec;
        } else {
          it->second.push_back(i);
        }
      }
      std::vector<int> part_ids;
      std::vector<std::vector<dgl_id_t>> part_nodes;
      int max_part_id = 0;
      for (auto it = part_map.begin(); it != part_map.end(); it++) {
        max_part_id = std::max(it->first, max_part_id);
        part_ids.push_back(it->first);
        part_nodes.push_back(it->second);
      }
      auto graph_ptr = std::dynamic_pointer_cast<ImmutableGraph>(graph.sptr());
      CHECK(graph_ptr) << "The input graph has to be an immutable graph";
      // When we construct subgraphs, we only access in-edges.
      // We need to make sure the in-CSR exists. Otherwise, we'll
      // try to construct in-CSR in openmp for loop, which will lead
      // to some unexpected results.
      graph_ptr->GetInCSR();
      std::vector<std::shared_ptr<HaloSubgraph>> subgs(max_part_id + 1);
      int num_partitions = part_nodes.size();
      runtime::parallel_for(0, num_partitions, [&](size_t b, size_t e) {
        for (auto i = b; i < e; ++i) {
          auto nodes = aten::VecToIdArray(part_nodes[i]);
          HaloSubgraph subg =
              GraphOp::GetSubgraphWithHalo(graph_ptr, nodes, num_hops);
          std::shared_ptr<HaloSubgraph> subg_ptr(new HaloSubgraph(subg));
          int part_id = part_ids[i];
          subgs[part_id] = subg_ptr;
        }
      });
      List<SubgraphRef> ret_list;
      for (size_t i = 0; i < subgs.size(); i++) {
        ret_list.push_back(SubgraphRef(subgs[i]));
      }
      *rv = ret_list;
    });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGetSubgraphWithHalo")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      GraphRef graph = args[0];
      IdArray nodes = args[1];
      int num_hops = args[2];
      HaloSubgraph subg =
          GraphOp::GetSubgraphWithHalo(graph.sptr(), nodes, num_hops);
      std::shared_ptr<HaloSubgraph> subg_ptr(new HaloSubgraph(subg));
      *rv = SubgraphRef(subg_ptr);
    });

DGL_REGISTER_GLOBAL("graph_index._CAPI_GetHaloSubgraphInnerNodes")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      SubgraphRef g = args[0];
      auto gptr = std::dynamic_pointer_cast<HaloSubgraph>(g.sptr());
      CHECK(gptr) << "The input graph has to be immutable graph";
      *rv = gptr->inner_nodes;
    });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLDisjointUnion")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      List<GraphRef> graphs = args[0];
      std::vector<GraphPtr> ptrs(graphs.size());
      for (size_t i = 0; i < graphs.size(); ++i) {
        ptrs[i] = graphs[i].sptr();
      }
      *rv = GraphOp::DisjointUnion(ptrs);
    });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLDisjointPartitionByNum")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      GraphRef g = args[0];
      int64_t num = args[1];
      const auto& ret = GraphOp::DisjointPartitionByNum(g.sptr(), num);
      List<GraphRef> ret_list;
      for (GraphPtr gp : ret) {
        ret_list.push_back(GraphRef(gp));
      }
      *rv = ret_list;
    });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLDisjointPartitionBySizes")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      GraphRef g = args[0];
      const IdArray sizes = args[1];
      const auto& ret = GraphOp::DisjointPartitionBySizes(g.sptr(), sizes);
      List<GraphRef> ret_list;
      for (GraphPtr gp : ret) {
        ret_list.push_back(GraphRef(gp));
      }
      *rv = ret_list;
    });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphLineGraph")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      GraphRef g = args[0];
      bool backtracking = args[1];
      *rv = GraphOp::LineGraph(g.sptr(), backtracking);
    });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLToImmutable")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      GraphRef g = args[0];
      *rv = ImmutableGraph::ToImmutable(g.sptr());
    });

DGL_REGISTER_GLOBAL("transform._CAPI_DGLToSimpleGraph")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      GraphRef g = args[0];
      *rv = GraphOp::ToSimpleGraph(g.sptr());
    });

DGL_REGISTER_GLOBAL("transform._CAPI_DGLToBidirectedMutableGraph")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      GraphRef g = args[0];
      *rv = GraphOp::ToBidirectedMutableGraph(g.sptr());
    });

DGL_REGISTER_GLOBAL("transform._CAPI_DGLReorderGraph")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      GraphRef g = args[0];
      const IdArray new_order = args[1];
      auto gptr = std::dynamic_pointer_cast<ImmutableGraph>(g.sptr());
      CHECK(gptr) << "The input graph has to be immutable graph";
      *rv = GraphOp::ReorderImmutableGraph(gptr, new_order);
    });

DGL_REGISTER_GLOBAL("transform._CAPI_DGLReassignEdges")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      GraphRef graph = args[0];
      bool is_incsr = args[1];
      auto gptr = std::dynamic_pointer_cast<ImmutableGraph>(graph.sptr());
      CHECK(gptr) << "We can only reassign edge Ids on immutable graphs";
      CSRPtr csr = is_incsr ? gptr->GetInCSR() : gptr->GetOutCSR();
      auto csrmat = csr->ToCSRMatrix();
      int64_t num_edges = csrmat.data->shape[0];
      IdArray new_data =
          IdArray::Empty({num_edges}, csrmat.data->dtype, csrmat.data->ctx);
      // Return the original edge Ids.
      *rv = new_data;
      // TODO(zhengda) I need to invalidate out-CSR and COO.

      // Generate new edge Ids.
      // TODO(zhengda) after assignment, we actually don't need to store them
      // physically.
      ATEN_ID_TYPE_SWITCH(new_data->dtype, IdType, {
        IdType* typed_new_data = static_cast<IdType*>(new_data->data);
        IdType* typed_data = static_cast<IdType*>(csrmat.data->data);
        for (int64_t i = 0; i < num_edges; i++) {
          typed_new_data[i] = typed_data[i];
          typed_data[i] = i;
        }
      });
    });

DGL_REGISTER_GLOBAL("transform._CAPI_DGLToBidirectedImmutableGraph")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      GraphRef g = args[0];
      auto gptr = g.sptr();
      auto immutable_g = std::dynamic_pointer_cast<ImmutableGraph>(gptr);
      GraphPtr ret;
      // For immutable graphs, we can try a faster version.
      if (immutable_g) {
        ret = GraphOp::ToBidirectedSimpleImmutableGraph(immutable_g);
      }
      // If the above option doesn't work, we call a general implementation.
      if (!ret) {
        ret = GraphOp::ToBidirectedImmutableGraph(gptr);
      }
      *rv = ret;
    });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLMapSubgraphNID")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      const IdArray parent_vids = args[0];
      const IdArray query = args[1];
      *rv = GraphOp::MapParentIdToSubgraphId(parent_vids, query);
    });

template <class IdType>
IdArray MapIds(
    IdArray ids, IdArray range_starts, IdArray range_ends, IdArray typed_map,
    int num_parts, int num_types) {
  int64_t num_ids = ids->shape[0];
  int64_t num_ranges = range_starts->shape[0];
  IdArray ret = IdArray::Empty({num_ids * 2}, ids->dtype, ids->ctx);

  const IdType* range_start_data = static_cast<IdType*>(range_starts->data);
  const IdType* range_end_data = static_cast<IdType*>(range_ends->data);
  const IdType* ids_data = static_cast<IdType*>(ids->data);
  const IdType* typed_map_data = static_cast<IdType*>(typed_map->data);
  IdType* types_data = static_cast<IdType*>(ret->data);
  IdType* per_type_ids_data = static_cast<IdType*>(ret->data) + num_ids;
  runtime::parallel_for(0, ids->shape[0], [&](size_t b, size_t e) {
    for (auto i = b; i < e; ++i) {
      IdType id = ids_data[i];
      auto it =
          std::lower_bound(range_end_data, range_end_data + num_ranges, id);
      // The range must exist.
      BUG_IF_FAIL(it != range_end_data + num_ranges);
      size_t range_id = it - range_end_data;
      int type_id = range_id % num_types;
      types_data[i] = type_id;
      int part_id = range_id / num_types;
      BUG_IF_FAIL(part_id < num_parts);
      if (part_id == 0) {
        per_type_ids_data[i] = id - range_start_data[range_id];
      } else {
        per_type_ids_data[i] =
            id - range_start_data[range_id] +
            typed_map_data[num_parts * type_id + part_id - 1];
      }
    }
  });
  return ret;
}

DGL_REGISTER_GLOBAL("distributed.id_map._CAPI_DGLHeteroMapIds")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      const IdArray ids = args[0];
      const IdArray range_starts = args[1];
      const IdArray range_ends = args[2];
      const IdArray typed_map = args[3];
      int num_parts = args[4];
      int num_types = args[5];
      int num_ranges = range_starts->shape[0];

      CHECK_EQ(range_starts->dtype.bits, ids->dtype.bits);
      CHECK_EQ(range_ends->dtype.bits, ids->dtype.bits);
      CHECK_EQ(typed_map->dtype.bits, ids->dtype.bits);
      CHECK_EQ(num_ranges, num_parts * num_types);
      CHECK_EQ(num_ranges, range_ends->shape[0]);

      IdArray ret;
      ATEN_ID_TYPE_SWITCH(ids->dtype, IdType, {
        ret = MapIds<IdType>(
            ids, range_starts, range_ends, typed_map, num_parts, num_types);
      });
      *rv = ret;
    });

}  // namespace dgl
