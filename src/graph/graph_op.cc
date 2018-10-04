// Graph operation implementation
#include <dgl/graph_op.h>
#include <algorithm>

namespace dgl {

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
  /*for (int64_t i = 0; i < len; ++i) {
    rst[i].AddVertices(sizes_data[i]);
  }
  for (dgl_id_t eid = 0; eid < graph->num_edges_; ++eid) {
    const dgl_id_t src = graph->all_edges_src_[eid];
    const dgl_id_t dst = graph->all_edges_dst_[eid];
    size_t src_select = 0, dst_select = 0;
    for (size_t i = 1; i < cumsum.size(); ++i) { // TODO: replace with binary search
      if (cumsum[i] > src) {
        src_select = i;
        break;
      }
    }
    for (size_t i = 1; i < cumsum.size(); ++i) { // TODO: replace with binary search
      if (cumsum[i] > dst) {
        dst_select = i;
        break;
      }
    }
    if (src_select != dst_select) {
      // the edge is ignored if across two partitions
      continue;
    }
    const int64_t offset = cumsum[src_select - 1];
    rst[src_select - 1].AddEdge(src - offset, dst - offset);
  }*/
  return rst;
}

}  // namespace dgl
