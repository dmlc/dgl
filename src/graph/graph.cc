/*!
 *  Copyright (c) 2018 by Contributors
 * \file graph/graph.cc
 * \brief DGL graph index implementation
 */
#include <dgl/graph.h>
#include <algorithm>
#include <array>
#include <set>
#include <functional>
#include <tuple>
#include "../c_api_common.h"

namespace dgl {
void Graph::AddVertices(uint64_t num_vertices) {
  CHECK(!read_only_) << "Graph is read-only. Mutations are not allowed.";
  grow(adjlist_, num_vertices);
  grow(reverse_adjlist_, num_vertices);
  grow(src_count_buffer_, num_vertices);
  grow(dst_count_buffer_, num_vertices);
  grow(offset_buffer_, num_vertices);
}

void Graph::InsertEdgeId_(EdgeList& el, dgl_id_t dst, dgl_id_t eid) {
  auto pos = std::upper_bound(el.succ.begin(), el.succ.end(), dst);
  auto offset = pos - el.succ.begin();

  CHECK(!(!is_multigraph_ && (pos != el.succ.begin()) && (*(pos - 1) == dst)))
      << "duplicate edge added to simple graph.";
  el.succ.insert(pos, dst);
  el.edge_id.insert(el.edge_id.begin() + offset, eid);
}

void Graph::AddEdge(dgl_id_t src, dgl_id_t dst) {
  CHECK(!read_only_) << "Graph is read-only. Mutations are not allowed.";
  CHECK(HasVertex(src) && HasVertex(dst))
    << "Invalid vertices: src=" << src << " dst=" << dst;

  dgl_id_t eid = num_edges_++;
  InsertEdgeId_(adjlist_[src], dst, eid);
  InsertEdgeId_(reverse_adjlist_[dst], src, eid);

  all_edges_src_.push_back(src);
  all_edges_dst_.push_back(dst);
}

bool Graph::CheckDuplicates_(const EdgeList& el, const dgl_id_t* dst, bool broadcast,
    const std::vector<dgl_id_t>& eid, size_t begin, size_t end) const {
  size_t i, j, size = el.edge_id.size();

  for (i = 0, j = begin; i < size && j < end; ) {
    const auto idx = broadcast ? 0 : eid[j];
    if (el.succ[i] < dst[idx])
      ++i;
    else if (el.succ[i] > dst[idx])
      ++j;
    else
      return true;
  }

  return false;
}

void Graph::MergeSorted_(EdgeList& el, const dgl_id_t* dst, bool broadcast,
    const std::vector<dgl_id_t>& eid, int64_t begin, int64_t end) {
  int64_t i, j, p;
  int64_t n_edges = end - begin, oldsize = el.succ.size(), newsize = oldsize + n_edges;

  el.succ.resize(newsize);
  el.edge_id.resize(newsize);

  for (i = oldsize - 1, j = end - 1, p = newsize - 1;
       j >= begin && i >= 0; --p) {
    const auto idx = broadcast ? 0 : eid[j];
    if (el.succ[i] > dst[idx]) {
      el.succ[p] = el.succ[i];
      el.edge_id[p] = el.edge_id[i];
      --i;
    } else {
      el.succ[p] = dst[idx];
      el.edge_id[p] = eid[j] + num_edges_;
      --j;
    }
  }

  for ( ; j >= begin; --j, --p) {
    el.succ[p] = dst[broadcast ? 0 : eid[j]];
    el.edge_id[p] = eid[j] + num_edges_;
  }
}

bool Graph::InsertEdges_(AdjacencyList& adjlist,
                         dgl_id_t* lo,
                         dgl_id_t* hi,
                         bool lo_broadcast,
                         bool hi_broadcast,
                         size_t len,
                         const std::vector<dgl_id_t>& lo_count_buffer,
                         const std::vector<dgl_id_t>& hi_count_buffer,
                         bool check_duplicates) {
  size_t num_vertices = adjlist_.size();
  size_t i, off;

  for (i = 0, off = 0; i < num_vertices; off += lo_count_buffer[i++])
    offset_buffer_[i] = off;
  for (i = 0; i < len; ++i)
    eid_buffer_[offset_buffer_[lo[lo_broadcast ? 0 : i]]++] = i;
  for (i = 0, off = 0; i < num_vertices; off += hi_count_buffer[i++])
    offset_buffer_[i] = off;
  for (i = 0; i < len; ++i)
    eid_buffer2_[offset_buffer_[hi[hi_broadcast ? 0 : eid_buffer_[i]]]++] = eid_buffer_[i];
  // eid_buffer2_ now contains the argsort of hi-lo pairs

  if (!is_multigraph_ && check_duplicates) {
    for (i = 1; i < len; ++i) {
      if (hi[hi_broadcast ? 0 : eid_buffer2_[i]] ==
          hi[hi_broadcast ? 0 : eid_buffer2_[i-1]] &&
          lo[lo_broadcast ? 0 : eid_buffer2_[i]] ==
          lo[lo_broadcast ? 0 : eid_buffer2_[i-1]])
        return false;
    }

    for (i = 0, off = 0; i < num_vertices; off += hi_count_buffer[i++]) {
      if (hi_count_buffer[i] > 0 &&
          CheckDuplicates_(adjlist[i], lo, lo_broadcast, eid_buffer2_, off,
              off + hi_count_buffer[i]))
        return false;
    }
  }

  for (i = 0, off = 0; i < num_vertices; off += hi_count_buffer[i++]) {
    if (hi_count_buffer[i] > 0)
      MergeSorted_(adjlist[i], lo, lo_broadcast, eid_buffer2_, off,
          off + hi_count_buffer[i]);
  }

  return true;
}

// O(V + E)
void Graph::AddEdges(IdArray src_ids, IdArray dst_ids) {
  CHECK(!read_only_) << "Graph is read-only. Mutations are not allowed.";
  CHECK(IsValidIdArray(src_ids)) << "Invalid src id array.";
  CHECK(IsValidIdArray(dst_ids)) << "Invalid dst id array.";
  const auto srclen = src_ids->shape[0];
  const auto dstlen = dst_ids->shape[0];
  const size_t len = (size_t)std::max(srclen, dstlen);
  dgl_id_t* src_data = static_cast<dgl_id_t*>(src_ids->data);
  dgl_id_t* dst_data = static_cast<dgl_id_t*>(dst_ids->data);
  bool duplicate = false;
  bool src_broadcast = false, dst_broadcast = false;

  std::fill(src_count_buffer_.begin(), src_count_buffer_.end(), 0);
  std::fill(dst_count_buffer_.begin(), dst_count_buffer_.end(), 0);
  eid_buffer_.resize(len);
  eid_buffer2_.resize(len);

  // replace src_data or dst_data with new array for broadcasting
  // (TODO better broadcasting handling)
  if (srclen == 1) {
    src_broadcast = true;
  } else if (dstlen == 1) {
    dst_broadcast = true;
  } else {
    CHECK(srclen == dstlen) << "Invalid dst and src array.";
  }

  size_t src_nonzeros = 0, dst_nonzeros = 0;
  for (size_t i = 0; i < len; ++i) {
    if (src_count_buffer_[src_data[src_broadcast ? 0 : i]]++ == 0)
      ++src_nonzeros;
    if (dst_count_buffer_[dst_data[dst_broadcast ? 0 : i]]++ == 0)
      ++dst_nonzeros;
  }

  if (src_nonzeros < dst_nonzeros) {
    if (!InsertEdges_(adjlist_, dst_data, src_data, dst_broadcast, src_broadcast, len,
          dst_count_buffer_, src_count_buffer_, true)) {
      duplicate = true;
      goto complete;
    }
    InsertEdges_(reverse_adjlist_, src_data, dst_data, src_broadcast, dst_broadcast, len,
        src_count_buffer_, dst_count_buffer_, false);
  } else {
    if (!InsertEdges_(reverse_adjlist_, src_data, dst_data, src_broadcast, dst_broadcast,
          len, src_count_buffer_, dst_count_buffer_, true)) {
      duplicate = true;
      goto complete;
    }
    InsertEdges_(adjlist_, dst_data, src_data, dst_broadcast, src_broadcast, len,
        dst_count_buffer_, src_count_buffer_, false);
  }

  for (size_t i = 0; i < len; ++i) {
    all_edges_src_.push_back(src_data[src_broadcast ? 0 : i]);
    all_edges_dst_.push_back(dst_data[dst_broadcast ? 0 : i]);
  }

  num_edges_ += len;

complete:
  CHECK(!duplicate) << "duplicate edges to be added in simple graph";
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

// O(log E)
bool Graph::HasEdgeBetween(dgl_id_t src, dgl_id_t dst) const {
  if (!HasVertex(src) || !HasVertex(dst)) return false;
  const auto& succ = adjlist_[src].succ;
  return std::binary_search(succ.begin(), succ.end(), dst);
}

// O(log E*k)
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

void Graph::EdgeIdRange_(dgl_id_t src, dgl_id_t dst, size_t* begin, size_t* end,
    bool* reverse) const {
  const auto& succ = adjlist_[src].succ;
  const auto& reverse_succ = reverse_adjlist_[dst].succ;
  if (succ.size() > reverse_succ.size()) {
    const auto& bounds = std::equal_range(
        reverse_succ.begin(), reverse_succ.end(), src);
    *begin = bounds.first - reverse_succ.begin();
    *end = bounds.second - reverse_succ.begin();
    *reverse = true;
  } else {
    const auto& bounds = std::equal_range(succ.begin(), succ.end(), dst);
    *begin = bounds.first - succ.begin();
    *end = bounds.second - succ.begin();
    *reverse = false;
  }
}

// O(logE)
IdArray Graph::EdgeId(dgl_id_t src, dgl_id_t dst) const {
  CHECK(HasVertex(src) && HasVertex(dst)) << "invalid edge: " << src << " -> " << dst;
  size_t begin, end;
  bool reverse;

  EdgeIdRange_(src, dst, &begin, &end, &reverse);
  const int64_t len = end - begin;
  IdArray rst = IdArray::Empty({len}, DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
  int64_t* rst_data = static_cast<int64_t*>(rst->data);
  const auto& edge_id = !reverse ? adjlist_[src].edge_id : reverse_adjlist_[dst].edge_id;
  std::copy(edge_id.begin() + begin, edge_id.begin() + end, rst_data);

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
    CHECK(HasVertex(src_id) && HasVertex(dst_id)) <<
        "invalid edge: " << src_id << " -> " << dst_id;
    size_t begin, end;
    bool reverse;

    EdgeIdRange_(src_id, dst_id, &begin, &end, &reverse);
    const int64_t len = end - begin;
    const auto& edge_id = !reverse ? adjlist_[src_id].edge_id :
      reverse_adjlist_[dst_id].edge_id;

    for (int64_t k = 0; k < len; ++k) {
      src.push_back(src_id);
      dst.push_back(dst_id);
      eid.push_back(edge_id[begin + k]);
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

// O(E)
Graph::EdgeArray Graph::Edges(bool sorted) const {
  const int64_t len = num_edges_;
  const size_t num_vertices = NumVertices();
  IdArray src = IdArray::Empty({len}, DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
  IdArray dst = IdArray::Empty({len}, DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
  IdArray eid = IdArray::Empty({len}, DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});

  if (sorted) {
    // make return arrays
    int64_t* src_ptr = static_cast<int64_t*>(src->data);
    int64_t* dst_ptr = static_cast<int64_t*>(dst->data);
    int64_t* eid_ptr = static_cast<int64_t*>(eid->data);
    size_t p = 0;
    for (dgl_id_t v = 0; v < num_vertices; ++v) {
      const auto& el = adjlist_[v];
      for (size_t i = 0; i < el.succ.size(); ++i, ++p) {
        src_ptr[p] = v;
        dst_ptr[p] = el.succ[i];
        eid_ptr[p] = el.edge_id[i];
      }
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
  rst.graph.is_multigraph_ = is_multigraph_;
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
  rst.graph.is_multigraph_ = is_multigraph_;
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
