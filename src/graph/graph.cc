// Graph class implementation
#include <algorithm>
#include <dgl/graph.h>

namespace dgl {
namespace {
inline bool IsValidIdArray(const IdArray& arr) {
  return arr->ctx.device_type == kDLCPU && arr->ndim == 1
    && arr->dtype.code == kDLInt && arr->dtype.bits == 64;
}
}  // namespace

void Graph::AddVertices(uint64_t num_vertices) {
  CHECK(!read_only_) << "Graph is read-only. Mutations are not allowed.";
  adjlist_.resize(adjlist_.size() + num_vertices);
  reverse_adjlist_.resize(reverse_adjlist_.size() + num_vertices);
}

void Graph::AddEdge(dgl_id_t src, dgl_id_t dst) {
  CHECK(!read_only_) << "Graph is read-only. Mutations are not allowed.";
  CHECK(HasVertex(src) && HasVertex(dst))
    << "In valid vertices: " << src << " " << dst;
  dgl_id_t eid = num_edges_++;
  adjlist_[src].succ.push_back(dst);
  adjlist_[src].edge_id.push_back(eid);
  reverse_adjlist_[dst].succ.push_back(src);
  reverse_adjlist_[dst].edge_id.push_back(eid);
  all_edges_src_.push_back(src);
  all_edges_dst_.push_back(dst);
}

void Graph::AddEdges(IdArray src_ids, IdArray dst_ids) {
  CHECK(!read_only_) << "Graph is read-only. Mutations are not allowed.";
  CHECK(IsValidIdArray(src_ids)) << "Invalid src id array.";
  CHECK(IsValidIdArray(dst_ids)) << "Invalid dst id array.";
  const auto srclen = src_ids->shape[0];
  const auto dstlen = dst_ids->shape[0];
  const int64_t* src_data = static_cast<int64_t*>(src_ids->data);
  const int64_t* dst_data = static_cast<int64_t*>(dst_ids->data);
  if (srclen == 1) {
    // one-many
    for (int64_t i = 0; i < dstlen; ++i) {
      AddEdge(src_data[0], dst_data[i]);
    }
  } else if (dstlen == 1) {
    // many-one
    for (int64_t i = 0; i < srclen; ++i) {
      AddEdge(src_data[i], dst_data[0]);
    }
  } else {
    // many-many
    CHECK(srclen == dstlen) << "Invalid src and dst id array.";
    for (int64_t i = 0; i < srclen; ++i) {
      AddEdge(src_data[i], dst_data[i]);
    }
  }
}

BoolArray Graph::HasVertices(IdArray vids) const {
  CHECK(IsValidIdArray(vids)) << "Invalid vertex id array.";
  const auto len = vids->shape[0];
  BoolArray rst = BoolArray::Empty({len}, vids->dtype, vids->ctx);
  const int64_t* vid_data = static_cast<int64_t*>(vids->data);
  int64_t* rst_data = static_cast<int64_t*>(rst->data);
  const int64_t nverts = NumVertices();
  for (int64_t i = 0; i < len; ++i) {
    rst_data[i] = (vid_data[i] < nverts)? 1 : 0;
  }
  return rst;
}

// O(E)
bool Graph::HasEdge(dgl_id_t src, dgl_id_t dst) const {
  if (!HasVertex(src) || !HasVertex(dst)) return false;
  const auto& succ = adjlist_[src].succ;
  return std::find(succ.begin(), succ.end(), dst) != succ.end();
}

// O(E*K) pretty slow
BoolArray Graph::HasEdges(IdArray src_ids, IdArray dst_ids) const {
  CHECK(IsValidIdArray(src_ids)) << "Invalid src id array.";
  CHECK(IsValidIdArray(dst_ids)) << "Invalid dst id array.";
  const auto srclen = src_ids->shape[0];
  const auto dstlen = dst_ids->shape[0];
  const auto rstlen = std::max(srclen, dstlen);
  BoolArray rst = BoolArray::Empty({rstlen}, src_ids->dtype, src_ids->ctx);
  int64_t* rst_data = static_cast<int64_t*>(rst->data);
  const int64_t* src_data = static_cast<int64_t*>(src_ids->data);
  const int64_t* dst_data = static_cast<int64_t*>(dst_ids->data);
  if (srclen == 1) {
    // one-many
    for (int64_t i = 0; i < dstlen; ++i) {
      rst_data[i] = HasEdge(src_data[0], dst_data[i])? 1 : 0;
    }
  } else if (dstlen == 1) {
    // many-one
    for (int64_t i = 0; i < srclen; ++i) {
      rst_data[i] = HasEdge(src_data[i], dst_data[0])? 1 : 0;
    }
  } else {
    // many-many
    CHECK(srclen == dstlen) << "Invalid src and dst id array.";
    for (int64_t i = 0; i < srclen; ++i) {
      rst_data[i] = HasEdge(src_data[i], dst_data[i])? 1 : 0;
    }
  }
  return rst;
}

// The data is copy-out; support zero-copy?
IdArray Graph::Predecessors(dgl_id_t vid, uint64_t radius) const {
  CHECK(HasVertex(vid)) << "invalid vertex: " << vid;
  CHECK(radius >= 1) << "invalid radius: " << radius;
  const auto& pred = reverse_adjlist_[vid].succ;
  const int64_t len = pred.size();
  IdArray rst = IdArray::Empty({len}, DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
  int64_t* rst_data = static_cast<int64_t*>(rst->data);
  for (int64_t i = 0; i < len; ++i) {
    rst_data[i] = pred[i];
  }
  return rst;
}

// The data is copy-out; support zero-copy?
IdArray Graph::Successors(dgl_id_t vid, uint64_t radius) const {
  CHECK(HasVertex(vid)) << "invalid vertex: " << vid;
  CHECK(radius >= 1) << "invalid radius: " << radius;
  const auto& succ = adjlist_[vid].succ;
  const int64_t len = succ.size();
  IdArray rst = IdArray::Empty({len}, DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
  int64_t* rst_data = static_cast<int64_t*>(rst->data);
  for (int64_t i = 0; i < len; ++i) {
    rst_data[i] = succ[i];
  }
  return rst;
}

// O(E)
dgl_id_t Graph::EdgeId(dgl_id_t src, dgl_id_t dst) const {
  CHECK(HasVertex(src)) << "invalid edge: " << src << " -> " << dst;
  const auto& succ = adjlist_[src].succ;
  for (size_t i = 0; i < succ.size(); ++i) {
    if (succ[i] == dst) {
      return adjlist_[src].edge_id[i];
    }
  }
  LOG(FATAL) << "invalid edge: " << src << " -> " << dst;
  return 0;
}

// O(E*k) pretty slow
IdArray Graph::EdgeIds(IdArray src_ids, IdArray dst_ids) const {
  CHECK(IsValidIdArray(src_ids)) << "Invalid src id array.";
  CHECK(IsValidIdArray(dst_ids)) << "Invalid dst id array.";
  const auto srclen = src_ids->shape[0];
  const auto dstlen = dst_ids->shape[0];
  const auto rstlen = std::max(srclen, dstlen);
  IdArray rst = IdArray::Empty({rstlen}, src_ids->dtype, src_ids->ctx);
  int64_t* rst_data = static_cast<int64_t*>(rst->data);
  const int64_t* src_data = static_cast<int64_t*>(src_ids->data);
  const int64_t* dst_data = static_cast<int64_t*>(dst_ids->data);
  if (srclen == 1) {
    // one-many
    for (int64_t i = 0; i < dstlen; ++i) {
      rst_data[i] = EdgeId(src_data[0], dst_data[i]);
    }
  } else if (dstlen == 1) {
    // many-one
    for (int64_t i = 0; i < srclen; ++i) {
      rst_data[i] = EdgeId(src_data[i], dst_data[0]);
    }
  } else {
    // many-many
    CHECK(srclen == dstlen) << "Invalid src and dst id array.";
    for (int64_t i = 0; i < srclen; ++i) {
      rst_data[i] = EdgeId(src_data[i], dst_data[i]);
    }
  }
  return rst;
}

// O(E)
Graph::EdgeArray Graph::InEdges(dgl_id_t vid) const {
  CHECK(HasVertex(vid)) << "invalid vertex: " << vid;
  const int64_t len = reverse_adjlist_[vid].succ.size();
  IdArray src = IdArray::Empty({len}, DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
  IdArray dst = IdArray::Empty({len}, DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
  IdArray eid = IdArray::Empty({len}, DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
  int64_t* src_data = static_cast<int64_t*>(src->data);
  int64_t* dst_data = static_cast<int64_t*>(dst->data);
  int64_t* eid_data = static_cast<int64_t*>(eid->data);
  for (int64_t i = 0; i < len; ++i) {
    src_data[i] = reverse_adjlist_[vid].succ[i];
    eid_data[i] = reverse_adjlist_[vid].edge_id[i];
  }
  std::fill(dst_data, dst_data + len, vid);
  return EdgeArray{src, dst, eid};
}

// O(E)
Graph::EdgeArray Graph::InEdges(IdArray vids) const {
  CHECK(IsValidIdArray(vids)) << "Invalid vertex id array.";
  const auto len = vids->shape[0];
  const int64_t* vid_data = static_cast<int64_t*>(vids->data);
  int64_t rstlen = 0;
  for (int64_t i = 0; i < len; ++i) {
    CHECK(HasVertex(vid_data[i])) << "Invalid vertex: " << vid_data[i];
    rstlen += reverse_adjlist_[vid_data[i]].succ.size();
  }
  IdArray src = IdArray::Empty({rstlen}, vids->dtype, vids->ctx);
  IdArray dst = IdArray::Empty({rstlen}, vids->dtype, vids->ctx);
  IdArray eid = IdArray::Empty({rstlen}, vids->dtype, vids->ctx);
  int64_t* src_ptr = static_cast<int64_t*>(src->data);
  int64_t* dst_ptr = static_cast<int64_t*>(dst->data);
  int64_t* eid_ptr = static_cast<int64_t*>(eid->data);
  for (int64_t i = 0; i < len; ++i) {
    const auto& pred = reverse_adjlist_[vid_data[i]].succ;
    const auto& eids = reverse_adjlist_[vid_data[i]].edge_id;
    for (size_t j = 0; j < pred.size(); ++j) {
      *(src_ptr++) = pred[j];
      *(dst_ptr++) = vid_data[i];
      *(eid_ptr++) = eids[j];
    }
  }
  return EdgeArray{src, dst, eid};
}

// O(E)
Graph::EdgeArray Graph::OutEdges(dgl_id_t vid) const {
  CHECK(HasVertex(vid)) << "invalid vertex: " << vid;
  const int64_t len = adjlist_[vid].succ.size();
  IdArray src = IdArray::Empty({len}, DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
  IdArray dst = IdArray::Empty({len}, DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
  IdArray eid = IdArray::Empty({len}, DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
  int64_t* src_data = static_cast<int64_t*>(src->data);
  int64_t* dst_data = static_cast<int64_t*>(dst->data);
  int64_t* eid_data = static_cast<int64_t*>(eid->data);
  for (int64_t i = 0; i < len; ++i) {
    dst_data[i] = adjlist_[vid].succ[i];
    eid_data[i] = adjlist_[vid].edge_id[i];
  }
  std::fill(src_data, src_data + len, vid);
  return EdgeArray{src, dst, eid};
}

// O(E)
Graph::EdgeArray Graph::OutEdges(IdArray vids) const {
  CHECK(IsValidIdArray(vids)) << "Invalid vertex id array.";
  const auto len = vids->shape[0];
  const int64_t* vid_data = static_cast<int64_t*>(vids->data);
  int64_t rstlen = 0;
  for (int64_t i = 0; i < len; ++i) {
    CHECK(HasVertex(vid_data[i])) << "Invalid vertex: " << vid_data[i];
    rstlen += adjlist_[vid_data[i]].succ.size();
  }
  IdArray src = IdArray::Empty({rstlen}, vids->dtype, vids->ctx);
  IdArray dst = IdArray::Empty({rstlen}, vids->dtype, vids->ctx);
  IdArray eid = IdArray::Empty({rstlen}, vids->dtype, vids->ctx);
  int64_t* src_ptr = static_cast<int64_t*>(src->data);
  int64_t* dst_ptr = static_cast<int64_t*>(dst->data);
  int64_t* eid_ptr = static_cast<int64_t*>(eid->data);
  for (int64_t i = 0; i < len; ++i) {
    const auto& succ = adjlist_[vid_data[i]].succ;
    const auto& eids = adjlist_[vid_data[i]].edge_id;
    for (size_t j = 0; j < succ.size(); ++j) {
      *(src_ptr++) = vid_data[i];
      *(dst_ptr++) = succ[j];
      *(eid_ptr++) = eids[j];
    }
  }
  return EdgeArray{src, dst, eid};
}

// O(E*log(E)) if sort is required; otherwise, O(E)
Graph::EdgeArray Graph::Edges(bool sorted) const {
  const int64_t len = num_edges_;
  IdArray src = IdArray::Empty({len}, DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
  IdArray dst = IdArray::Empty({len}, DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
  IdArray eid = IdArray::Empty({len}, DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});

  if (sorted) {
    typedef std::tuple<int64_t, int64_t, int64_t> Tuple;
    std::vector<Tuple> tuples;
    tuples.reserve(len);
    for (uint64_t eid = 0; eid < num_edges_; ++eid) {
      tuples.emplace_back(all_edges_src_[eid], all_edges_dst_[eid], eid);
    }
    // sort according to src and dst ids
    std::sort(tuples.begin(), tuples.end(),
        [] (const Tuple& t1, const Tuple& t2) {
          return std::get<0>(t1) < std::get<0>(t2)
            || (std::get<0>(t1) == std::get<0>(t2) && std::get<1>(t1) < std::get<1>(t2));
        });
    
    // make return arrays
    int64_t* src_ptr = static_cast<int64_t*>(src->data);
    int64_t* dst_ptr = static_cast<int64_t*>(dst->data);
    int64_t* eid_ptr = static_cast<int64_t*>(eid->data);
    for (size_t i = 0; i < tuples.size(); ++i) {
      src_ptr[i] = std::get<0>(tuples[i]);
      dst_ptr[i] = std::get<1>(tuples[i]);
      eid_ptr[i] = std::get<2>(tuples[i]);
    }
  } else {
    int64_t* src_ptr = static_cast<int64_t*>(src->data);
    int64_t* dst_ptr = static_cast<int64_t*>(dst->data);
    int64_t* eid_ptr = static_cast<int64_t*>(eid->data);
    std::copy(all_edges_src_.begin(), all_edges_src_.end(), src_ptr);
    std::copy(all_edges_dst_.begin(), all_edges_dst_.end(), dst_ptr);
    for (uint64_t eid = 0; eid < num_edges_; ++eid) {
      eid_ptr[eid] = eid;
    }
  }

  return EdgeArray{src, dst, eid};
}

// O(V)
DegreeArray Graph::InDegrees(IdArray vids) const {
  CHECK(IsValidIdArray(vids)) << "Invalid vertex id array.";
  const auto len = vids->shape[0];
  const int64_t* vid_data = static_cast<int64_t*>(vids->data);
  DegreeArray rst = DegreeArray::Empty({len}, vids->dtype, vids->ctx);
  int64_t* rst_data = static_cast<int64_t*>(rst->data);
  for (int64_t i = 0; i < len; ++i) {
    const auto vid = vid_data[i];
    CHECK(HasVertex(vid)) << "Invalid vertex: " << vid;
    rst_data[i] = reverse_adjlist_[vid].succ.size();
  }
  return rst;
}

// O(V)
DegreeArray Graph::OutDegrees(IdArray vids) const {
  CHECK(IsValidIdArray(vids)) << "Invalid vertex id array.";
  const auto len = vids->shape[0];
  const int64_t* vid_data = static_cast<int64_t*>(vids->data);
  DegreeArray rst = DegreeArray::Empty({len}, vids->dtype, vids->ctx);
  int64_t* rst_data = static_cast<int64_t*>(rst->data);
  for (int64_t i = 0; i < len; ++i) {
    const auto vid = vid_data[i];
    CHECK(HasVertex(vid)) << "Invalid vertex: " << vid;
    rst_data[i] = adjlist_[vid].succ.size();
  }
  return rst;
}

Graph Graph::Subgraph(IdArray vids) const {
  LOG(FATAL) << "not implemented";
  return *this;
}

Graph Graph::EdgeSubgraph(IdArray src, IdArray dst) const {
  LOG(FATAL) << "not implemented";
  return *this;
}

Graph Graph::Reverse() const {
  LOG(FATAL) << "not implemented";
  return *this;
}

Graph Graph::Merge(std::vector<const Graph*> graphs) {
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

}  // namespace dgl
