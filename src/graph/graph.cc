/*!
 *  Copyright (c) 2018 by Contributors
 * \file graph/graph.cc
 * \brief DGL graph index implementation
 */
#include <dgl/graph.h>
#include <algorithm>
#include <unordered_map>
#include <set>
#include <stack>
#include <functional>
#include <tuple>

namespace dgl {
namespace {
inline bool IsValidIdArray(const IdArray& arr) {
  return arr->ctx.device_type == kDLCPU && arr->ndim == 1
    && arr->dtype.code == kDLInt && arr->dtype.bits == 64;
}

inline IdArray IdVectorToIdArray(const IdVector& v) {
  const int64_t len = v.size();
  auto a = IdArray::Empty({len}, DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
  std::copy(v.begin(), v.end(), static_cast<dgl_id_t*>(a->data));
  return a;
}

inline std::pair<IdArray, IdArray> ZipIdVectorsInIdArray(std::vector<IdVector> &id_vectors) {
  auto size = [](const IdVector &x) { return x.size(); };
  std::vector<size_t> s(id_vectors.size());
  std::transform(id_vectors.begin(), id_vectors.end(), s.begin(), size);
  auto max_s = *std::max_element(s.begin(), s.end());
  std::set<dgl_id_t> included;
  for (size_t i = 0; i < id_vectors.size(); ++i) {
    included.insert(i);
  }
  std::vector<size_t> t(max_s);
  for (size_t i = 0; i < max_s; ++i) {
    for (auto j : included) {
      if (i == id_vectors[j].size()) {
        included.erase(j);
      }
    }
    t[i] = included.size();
  }

  for (size_t i = 0; i < id_vectors.size(); ++i) {
    included.insert(i);
  }
  const int64_t sum_t = std::accumulate(t.begin(), t.end(), 0);
  auto id_array = IdArray::Empty({sum_t}, DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
  auto id_data = static_cast<int64_t*>(id_array->data);
  for (size_t i = 0; i < max_s; ++i) {
    for (auto j : included) {
      if (i < id_vectors[j].size()) {
        id_data[i * id_vectors.size() + j] = id_vectors[j][i];
      } else {
        included.erase(j);
      }
    }
  }
  auto s_array = IdVectorToIdArray(t);
  return std::make_pair(id_array, s_array);
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
    << "Invalid vertices: src=" << src << " dst=" << dst;

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
bool Graph::HasEdgeBetween(dgl_id_t src, dgl_id_t dst) const {
  if (!HasVertex(src) || !HasVertex(dst)) return false;
  const auto& succ = adjlist_[src].succ;
  return std::find(succ.begin(), succ.end(), dst) != succ.end();
}

// O(E*k) pretty slow
BoolArray Graph::HasEdgesBetween(IdArray src_ids, IdArray dst_ids) const {
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
      rst_data[i] = HasEdgeBetween(src_data[0], dst_data[i])? 1 : 0;
    }
  } else if (dstlen == 1) {
    // many-one
    for (int64_t i = 0; i < srclen; ++i) {
      rst_data[i] = HasEdgeBetween(src_data[i], dst_data[0])? 1 : 0;
    }
  } else {
    // many-many
    CHECK(srclen == dstlen) << "Invalid src and dst id array.";
    for (int64_t i = 0; i < srclen; ++i) {
      rst_data[i] = HasEdgeBetween(src_data[i], dst_data[i])? 1 : 0;
    }
  }
  return rst;
}

// The data is copy-out; support zero-copy?
IdArray Graph::Predecessors(dgl_id_t vid, uint64_t radius) const {
  CHECK(HasVertex(vid)) << "invalid vertex: " << vid;
  CHECK(radius >= 1) << "invalid radius: " << radius;
  std::set<dgl_id_t> vset;

  for (auto& it : reverse_adjlist_[vid].succ)
    vset.insert(it);

  const int64_t len = vset.size();
  IdArray rst = IdArray::Empty({len}, DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
  int64_t* rst_data = static_cast<int64_t*>(rst->data);

  std::copy(vset.begin(), vset.end(), rst_data);
  return rst;
}

// The data is copy-out; support zero-copy?
IdArray Graph::Successors(dgl_id_t vid, uint64_t radius) const {
  CHECK(HasVertex(vid)) << "invalid vertex: " << vid;
  CHECK(radius >= 1) << "invalid radius: " << radius;
  std::set<dgl_id_t> vset;

  for (auto& it : adjlist_[vid].succ)
    vset.insert(it);

  const int64_t len = vset.size();
  IdArray rst = IdArray::Empty({len}, DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
  int64_t* rst_data = static_cast<int64_t*>(rst->data);

  std::copy(vset.begin(), vset.end(), rst_data);
  return rst;
}

// O(E)
IdArray Graph::EdgeId(dgl_id_t src, dgl_id_t dst) const {
  CHECK(HasVertex(src) && HasVertex(dst)) << "invalid edge: " << src << " -> " << dst;

  const auto& succ = adjlist_[src].succ;
  std::vector<dgl_id_t> edgelist;

  for (size_t i = 0; i < succ.size(); ++i) {
    if (succ[i] == dst)
      edgelist.push_back(adjlist_[src].edge_id[i]);
  }

  // FIXME: signed?  Also it seems that we are using int64_t everywhere...
  const int64_t len = edgelist.size();
  IdArray rst = IdArray::Empty({len}, DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
  // FIXME: signed?
  int64_t* rst_data = static_cast<int64_t*>(rst->data);

  std::copy(edgelist.begin(), edgelist.end(), rst_data);

  return rst;
}

// O(E*k) pretty slow
Graph::EdgeArray Graph::EdgeIds(IdArray src_ids, IdArray dst_ids) const {
  CHECK(IsValidIdArray(src_ids)) << "Invalid src id array.";
  CHECK(IsValidIdArray(dst_ids)) << "Invalid dst id array.";
  const auto srclen = src_ids->shape[0];
  const auto dstlen = dst_ids->shape[0];
  int64_t i, j;

  CHECK((srclen == dstlen) || (srclen == 1) || (dstlen == 1))
    << "Invalid src and dst id array.";

  const int64_t src_stride = (srclen == 1 && dstlen != 1) ? 0 : 1;
  const int64_t dst_stride = (dstlen == 1 && srclen != 1) ? 0 : 1;
  const int64_t* src_data = static_cast<int64_t*>(src_ids->data);
  const int64_t* dst_data = static_cast<int64_t*>(dst_ids->data);

  std::vector<dgl_id_t> src, dst, eid;

  for (i = 0, j = 0; i < srclen && j < dstlen; i += src_stride, j += dst_stride) {
    const dgl_id_t src_id = src_data[i], dst_id = dst_data[j];
    const auto& succ = adjlist_[src_id].succ;
    for (size_t k = 0; k < succ.size(); ++k) {
      if (succ[k] == dst_id) {
        src.push_back(src_id);
        dst.push_back(dst_id);
        eid.push_back(adjlist_[src_id].edge_id[k]);
      }
    }
  }

  int64_t rstlen = src.size();
  IdArray rst_src = IdArray::Empty({rstlen}, src_ids->dtype, src_ids->ctx);
  IdArray rst_dst = IdArray::Empty({rstlen}, src_ids->dtype, src_ids->ctx);
  IdArray rst_eid = IdArray::Empty({rstlen}, src_ids->dtype, src_ids->ctx);
  int64_t* rst_src_data = static_cast<int64_t*>(rst_src->data);
  int64_t* rst_dst_data = static_cast<int64_t*>(rst_dst->data);
  int64_t* rst_eid_data = static_cast<int64_t*>(rst_eid->data);

  std::copy(src.begin(), src.end(), rst_src_data);
  std::copy(dst.begin(), dst.end(), rst_dst_data);
  std::copy(eid.begin(), eid.end(), rst_eid_data);

  return EdgeArray{rst_src, rst_dst, rst_eid};
}

Graph::EdgeArray Graph::FindEdges(IdArray eids) const {
  int64_t len = eids->shape[0];

  IdArray rst_src = IdArray::Empty({len}, eids->dtype, eids->ctx);
  IdArray rst_dst = IdArray::Empty({len}, eids->dtype, eids->ctx);
  IdArray rst_eid = IdArray::Empty({len}, eids->dtype, eids->ctx);
  int64_t* eid_data = static_cast<int64_t*>(eids->data);
  int64_t* rst_src_data = static_cast<int64_t*>(rst_src->data);
  int64_t* rst_dst_data = static_cast<int64_t*>(rst_dst->data);
  int64_t* rst_eid_data = static_cast<int64_t*>(rst_eid->data);

  for (uint64_t i = 0; i < (uint64_t)len; ++i) {
    dgl_id_t eid = eid_data[i];
    if (eid >= num_edges_)
      LOG(FATAL) << "invalid edge id:" << eid;

    rst_src_data[i] = all_edges_src_[eid];
    rst_dst_data[i] = all_edges_dst_[eid];
    rst_eid_data[i] = eid;
  }

  return EdgeArray{rst_src, rst_dst, rst_eid};
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

std::pair<IdArray, IdArray> Graph::BFS(IdArray src, bool out) const {
  std::vector<dgl_id_t> vv;
  auto src_data = static_cast<int64_t*>(src->data);
  std::copy(src_data, src_data + src->shape[0], std::back_inserter(vv));
  size_t i = 0;
  size_t j = vv.size();
  auto adj = &(out ? adjlist_ : reverse_adjlist_);
  std::vector<bool> visited(adjlist_.size());
  for (auto v : vv) {
    visited[v] = true;
  }
  std::vector<size_t> ss;
  while (i < j) {
    for (size_t k = i; k != j; k++) {
      auto succ = (*adj)[vv[k]].succ;
      for (auto v : succ) {
        if (!visited[v]) {
            visited[v] = true;
            vv.push_back(v);
        }
      }
    }
    ss.push_back(j - i);
    i = j;
    j = vv.size();
  }
  return std::make_pair(IdVectorToIdArray(vv), IdVectorToIdArray(ss));
}

std::tuple<IdVector, IdVector, IdVector> Graph::DFSLabeledEdges_(dgl_id_t source, bool out, bool reverse_edge, bool nontree_edge) const {
  enum EdgeType { forward, reverse, nontree };
  IdVector src;
  IdVector dst;
  IdVector type;
  auto adjlist = out ? adjlist_ : reverse_adjlist_;
  typedef std::pair<dgl_id_t, std::vector<dgl_id_t>::iterator> crux_t;
  std::stack<crux_t> stack;
  stack.push(std::make_pair(source, adjlist[source].succ.begin()));
  std::vector<bool> visited(adjlist_.size());
  visited[source] = true;
  while (!stack.empty()) {
    auto parent = stack.top().first;
    auto children = stack.top().second;
    if (children != adjlist[parent].succ.end()) {
      auto child = *children;
      ++stack.top().second;
      if (visited[child] && nontree_edge) {
        src.push_back(parent);
        dst.push_back(child);
        type.push_back(nontree);
      } else if (!visited[child]) {
        src.push_back(parent);
        dst.push_back(child);
        type.push_back(forward);
        visited[child] = true;
        stack.push(std::make_pair(child, adjlist[child].succ.begin()));
      }
    } else {
      stack.pop();
      if (!stack.empty() && reverse_edge) {
        src.push_back(stack.top().first);
        dst.push_back(parent);
        type.push_back(reverse);
      }
    }
  }
  return std::make_tuple(src, dst, type);
}

std::tuple<IdArray, IdArray, IdArray, IdArray> Graph::DFSLabeledEdges(IdArray source, bool out, bool reverse_edge, bool nontree_edge) const {
  // TODO(gaiyu): const
  int64_t *source_data = static_cast<int64_t*>(source->data);
  std::vector<IdVector> src(source->shape[0]);
  std::vector<IdVector> dst(source->shape[0]);
  std::vector<IdVector> type(source->shape[0]);
  for (int64_t i = 0; i < source->shape[0]; i++) {
    auto ret = DFSLabeledEdges_(source_data[i], out, reverse_edge, nontree_edge);
    src[i] = std::get<0>(ret);
    dst[i] = std::get<1>(ret);
    type[i] = std::get<2>(ret);
  }
  auto src_pair = ZipIdVectorsInIdArray(src);
  auto dst_pair = ZipIdVectorsInIdArray(dst);
  auto type_pair = ZipIdVectorsInIdArray(type);
  return std::make_tuple(src_pair.first, dst_pair.first, type_pair.first, src_pair.second);
}

std::pair<IdArray, IdArray> Graph::TopologicalTraversal(bool out) const {
  IdVector uu(adjlist_.size());
  std::iota(uu.begin(), uu.end(), 0);
  auto u_array = IdVectorToIdArray(uu);
  auto deg = out ? InDegrees(u_array) : OutDegrees(u_array);
  auto deg_data = static_cast<int64_t*>(deg->data);
  IdVector vv;
  for (int64_t k = 0; k < deg->shape[0]; k++) {
    if (deg_data[k] == 0) {
      vv.push_back(k);
    }
  }
  size_t i = 0;
  size_t j = vv.size();
  auto adj = &(out ? adjlist_ : reverse_adjlist_);
  IdVector ss;
  while (i < j) {
    for (size_t k = i; k != j; k++) {
      auto succ = (*adj)[vv[k]].succ;
      for (auto v : succ) {
        if (--(deg_data[v]) == 0) {
          vv.push_back(v);
        }
      }
    }
    ss.push_back(j - i);
    i = j;
    j = vv.size();
  }
  return std::make_pair(IdVectorToIdArray(vv), IdVectorToIdArray(ss));
}

Subgraph Graph::VertexSubgraph(IdArray vids) const {
  CHECK(IsValidIdArray(vids)) << "Invalid vertex id array.";
  const auto len = vids->shape[0];
  std::unordered_map<dgl_id_t, dgl_id_t> oldv2newv;
  std::vector<dgl_id_t> edges;
  const int64_t* vid_data = static_cast<int64_t*>(vids->data);
  for (int64_t i = 0; i < len; ++i) {
    oldv2newv[vid_data[i]] = i;
  }
  Subgraph rst;
  rst.induced_vertices = vids;
  rst.graph.AddVertices(len);
  for (int64_t i = 0; i < len; ++i) {
    const dgl_id_t oldvid = vid_data[i];
    const dgl_id_t newvid = i;
    for (size_t j = 0; j < adjlist_[oldvid].succ.size(); ++j) {
      const dgl_id_t oldsucc = adjlist_[oldvid].succ[j];
      if (oldv2newv.count(oldsucc)) {
        const dgl_id_t newsucc = oldv2newv[oldsucc];
        edges.push_back(adjlist_[oldvid].edge_id[j]);
        rst.graph.AddEdge(newvid, newsucc);
      }
    }
  }
  rst.induced_edges = IdArray::Empty({static_cast<int64_t>(edges.size())}, vids->dtype, vids->ctx);
  std::copy(edges.begin(), edges.end(), static_cast<int64_t*>(rst.induced_edges->data));
  return rst;
}

Subgraph Graph::EdgeSubgraph(IdArray eids) const {
  CHECK(IsValidIdArray(eids)) << "Invalid vertex id array.";

  const auto len = eids->shape[0];
  std::unordered_map<dgl_id_t, dgl_id_t> oldv2newv;
  std::vector<dgl_id_t> nodes;
  const int64_t* eid_data = static_cast<int64_t*>(eids->data);

  for (int64_t i = 0; i < len; ++i) {
    dgl_id_t src_id = all_edges_src_[eid_data[i]];
    dgl_id_t dst_id = all_edges_dst_[eid_data[i]];
    if (oldv2newv.insert(std::make_pair(src_id, oldv2newv.size())).second)
      nodes.push_back(src_id);
    if (oldv2newv.insert(std::make_pair(dst_id, oldv2newv.size())).second)
      nodes.push_back(dst_id);
  }

  Subgraph rst;
  rst.induced_edges = eids;
  rst.graph.AddVertices(nodes.size());

  for (int64_t i = 0; i < len; ++i) {
    dgl_id_t src_id = all_edges_src_[eid_data[i]];
    dgl_id_t dst_id = all_edges_dst_[eid_data[i]];
    rst.graph.AddEdge(oldv2newv[src_id], oldv2newv[dst_id]);
  }

  rst.induced_vertices = IdArray::Empty(
      {static_cast<int64_t>(nodes.size())}, eids->dtype, eids->ctx);
  std::copy(nodes.begin(), nodes.end(), static_cast<int64_t*>(rst.induced_vertices->data));

  return rst;
}

Graph Graph::Reverse() const {
  LOG(FATAL) << "not implemented";
  return *this;
}

}  // namespace dgl
