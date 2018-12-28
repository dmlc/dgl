/*!
 *  Copyright (c) 2018 by Contributors
 * \file graph/immutable_graph.cc
 * \brief DGL immutable graph index implementation
 */

#include <dgl/immutable_graph.h>

#include "../c_api_common.h"

namespace dgl {

ImmutableGraph::EdgeArray ImmutableGraph::csr::GetEdges(dgl_id_t vid) const {
  CHECK(HasVertex(vid)) << "invalid vertex: " << vid;
  int64_t off = this->indptr[vid];
  const int64_t len = this->GetDegree(vid);
  IdArray src = IdArray::Empty({len}, DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
  IdArray dst = IdArray::Empty({len}, DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
  IdArray eid = IdArray::Empty({len}, DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
  dgl_id_t* src_data = static_cast<dgl_id_t*>(src->data);
  dgl_id_t* dst_data = static_cast<dgl_id_t*>(dst->data);
  dgl_id_t* eid_data = static_cast<dgl_id_t*>(eid->data);
  for (int64_t i = 0; i < len; ++i) {
    src_data[i] = this->indices[off + i];
    eid_data[i] = this->edge_ids[off + i];
  }
  std::fill(dst_data, dst_data + len, vid);
  return ImmutableGraph::EdgeArray{src, dst, eid};
}

ImmutableGraph::EdgeArray ImmutableGraph::csr::GetEdges(IdArray vids) const {
  CHECK(IsValidIdArray(vids)) << "Invalid vertex id array.";
  const auto len = vids->shape[0];
  const dgl_id_t* vid_data = static_cast<dgl_id_t*>(vids->data);
  int64_t rstlen = 0;
  for (int64_t i = 0; i < len; ++i) {
    dgl_id_t vid = vid_data[i];
    CHECK(HasVertex(vid)) << "Invalid vertex: " << vid;
    rstlen += this->GetDegree(vid);
  }
  IdArray src = IdArray::Empty({rstlen}, vids->dtype, vids->ctx);
  IdArray dst = IdArray::Empty({rstlen}, vids->dtype, vids->ctx);
  IdArray eid = IdArray::Empty({rstlen}, vids->dtype, vids->ctx);
  dgl_id_t* src_ptr = static_cast<dgl_id_t*>(src->data);
  dgl_id_t* dst_ptr = static_cast<dgl_id_t*>(dst->data);
  dgl_id_t* eid_ptr = static_cast<dgl_id_t*>(eid->data);
  for (int64_t i = 0; i < len; ++i) {
    dgl_id_t vid = vid_data[i];
    int64_t off = this->indptr[vid];
    const int64_t len = this->GetDegree(vid);
    const auto *pred = &this->indices[off];
    const auto *eids = &this->edge_ids[off];
    for (int64_t j = 0; j < len; ++j) {
      *(src_ptr++) = pred[j];
      *(dst_ptr++) = vid;
      *(eid_ptr++) = eids[j];
    }
  }
  return ImmutableGraph::EdgeArray{src, dst, eid};
}

DegreeArray ImmutableGraph::csr::GetDegrees(IdArray vids) const {
  CHECK(IsValidIdArray(vids)) << "Invalid vertex id array.";
  const auto len = vids->shape[0];
  const dgl_id_t* vid_data = static_cast<dgl_id_t*>(vids->data);
  DegreeArray rst = DegreeArray::Empty({len}, vids->dtype, vids->ctx);
  dgl_id_t* rst_data = static_cast<dgl_id_t*>(rst->data);
  for (int64_t i = 0; i < len; ++i) {
    const auto vid = vid_data[i];
    CHECK(HasVertex(vid)) << "Invalid vertex: " << vid;
    rst_data[i] = this->GetDegree(vid);
  }
  return rst;
}

class Bitmap {
  const size_t size = 1024 * 1024 * 4;
  const size_t mask = size - 1;
  std::vector<bool> map;

  size_t hash(dgl_id_t id) const {
    return id & mask;
  }
 public:
  Bitmap(const dgl_id_t *vid_data, int64_t len): map(size) {
    for (int64_t i = 0; i < len; ++i) {
      map[hash(vid_data[i])] = 1;
    }
  }

  bool test(dgl_id_t id) const {
    return map[hash(id)];
  }
};

/*
 * This uses a hashtable to check if a node is in the given node list.
 */
class HashTableChecker {
  std::unordered_map<dgl_id_t, dgl_id_t> oldv2newv;
  Bitmap map;

 public:
  HashTableChecker(const dgl_id_t *vid_data, int64_t len): map(vid_data, len) {
    oldv2newv.reserve(len);
    for (int64_t i = 0; i < len; ++i) {
      oldv2newv[vid_data[i]] = i;
    }
  }

  void CollectOnRow(const dgl_id_t col_idx[], const dgl_id_t eids[], size_t row_len,
                    std::vector<dgl_id_t> *new_col_idx,
                    std::vector<dgl_id_t> *orig_eids) {
    // TODO(zhengda) I need to make sure the column index in each row is sorted.
    for (size_t j = 0; j < row_len; ++j) {
      const dgl_id_t oldsucc = col_idx[j];
      const dgl_id_t eid = eids[j];
      Collect(oldsucc, eid, new_col_idx, orig_eids);
    }
  }

  void Collect(const dgl_id_t old_id, const dgl_id_t old_eid,
               std::vector<dgl_id_t> *col_idx,
               std::vector<dgl_id_t> *orig_eids) {
    if (!map.test(old_id))
      return;

    auto it = oldv2newv.find(old_id);
    if (it != oldv2newv.end()) {
      const dgl_id_t new_id = it->second;
      col_idx->push_back(new_id);
      if (orig_eids)
        orig_eids->push_back(old_eid);
    }
  }
};

std::pair<std::shared_ptr<ImmutableGraph::csr>, IdArray> ImmutableGraph::csr::VertexSubgraph(IdArray vids) const {
  const dgl_id_t* vid_data = static_cast<dgl_id_t*>(vids->data);
  const int64_t len = vids->shape[0];

  HashTableChecker def_check(vid_data, len);
  // check if varr is sorted.
  CHECK(std::is_sorted(vid_data, vid_data + len)) << "The input vertex list has to be sorted";

  // Collect the non-zero entries in from the original graph.
  std::vector<dgl_id_t> orig_edge_ids;
  orig_edge_ids.reserve(len * 50);
  auto sub_csr = std::make_shared<csr>(len, len * 50);
  sub_csr->indptr[0] = 0;
  for (int64_t i = 0; i < len; ++i) {
    const dgl_id_t oldvid = vid_data[i];
    CHECK_LT(oldvid, NumVertices()) << "Vertex Id " << oldvid << " isn't in a graph of "
        << NumVertices() << " vertices";
    size_t row_start = indptr[oldvid];
    size_t row_len = indptr[oldvid + 1] - indptr[oldvid];
    def_check.CollectOnRow(&indices[row_start], &edge_ids[row_start], row_len,
                           &sub_csr->indices, &orig_edge_ids);
    sub_csr->indptr[i + 1] = sub_csr->indices.size();
  }

  // Store the non-zeros in a subgraph with edge attributes of new edge ids.
  sub_csr->edge_ids.resize(sub_csr->indices.size());
  for (int64_t i = 0; i < sub_csr->edge_ids.size(); i++)
    sub_csr->edge_ids[i] = i;

  IdArray rst_eids = IdArray::Empty({static_cast<int64_t>(orig_edge_ids.size())},
                                    DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
  dgl_id_t* eid_data = static_cast<dgl_id_t*>(rst_eids->data);
  std::copy(orig_edge_ids.begin(), orig_edge_ids.end(), eid_data);

  return std::pair<std::shared_ptr<ImmutableGraph::csr>, IdArray>(sub_csr, rst_eids);
}

std::shared_ptr<ImmutableGraph::csr> ImmutableGraph::csr::from_edges(std::vector<edge> &edges,
                                                                     int sort_on) {
  assert(sort_on == 0 || sort_on == 1);
  int other_end = sort_on == 1 ? 0 : 1;
  // TODO(zhengda) we should sort in parallel.
  struct compare {
    int sort_on;
    int other_end;
    compare(int sort_on, int other_end) {
      this->sort_on = sort_on;
      this->other_end = other_end;
    }
    bool operator()(const edge &e1, const edge &e2) {
      if (e1.end_points[sort_on] == e2.end_points[sort_on])
        return e1.end_points[other_end] < e2.end_points[other_end];
      else
        return e1.end_points[sort_on] < e2.end_points[sort_on];
    }
  };
  std::sort(edges.begin(), edges.end(), compare(sort_on, other_end));
  std::shared_ptr<csr> t = std::make_shared<csr>(0, 0);
  t->indices.resize(edges.size());
  t->edge_ids.resize(edges.size());
  t->indptr.push_back(0);
  for (size_t i = 0; i < edges.size(); i++) {
    t->indices[i] = edges[i].end_points[other_end];
    t->edge_ids[i] = edges[i].edge_id;
    dgl_id_t vid = edges[i].end_points[sort_on];
    int64_t off;
    if (t->indptr.empty())
      off = 0;
    else
      off = t->indptr.back();
    while (vid > 0 && t->indptr.size() < static_cast<size_t>(vid - 1))
      t->indptr.push_back(off);
    if (t->indptr.size() < vid)
      t->indptr.push_back(i);
    assert(t->indptr.size() == vid + 1);
  }
  t->indptr.push_back(edges.size());
  return t;
}

std::shared_ptr<ImmutableGraph::csr> ImmutableGraph::csr::Transpose() const {
  std::vector<edge> edges(NumEdges());
  for (size_t i = 0; i < NumVertices(); i++) {
    const dgl_id_t *indices_begin = &indices[indptr[i]];
    const dgl_id_t *eid_begin = &edge_ids[indptr[i]];
    for (size_t j = 0; j < GetDegree(i); j++) {
      edge e;
      e.end_points[0] = i;
      e.end_points[1] = indices_begin[j];
      e.edge_id = eid_begin[j];
      edges[indptr[i] + j] = e;
    }
  }
  return from_edges(edges, 1);
}

ImmutableGraph::ImmutableGraph(IdArray src_ids, IdArray dst_ids, IdArray edge_ids,
                               bool multigraph) : is_multigraph_(multigraph) {
  int64_t len = src_ids->shape[0];
  assert(len == dst_ids->shape[0]);
  assert(len == edge_ids->shape[0]);
  const dgl_id_t *src_data = static_cast<dgl_id_t*>(src_ids->data);
  const dgl_id_t *dst_data = static_cast<dgl_id_t*>(dst_ids->data);
  const dgl_id_t *edge_data = static_cast<dgl_id_t*>(edge_ids->data);
  std::vector<edge> edges(len);
  for (size_t i = 0; i < edges.size(); i++) {
    edge e;
    e.end_points[0] = src_data[i];
    e.end_points[1] = dst_data[i];
    e.edge_id = edge_data[i];
    edges[i] = e;
  }
  in_csr_ = csr::from_edges(edges, 1);
  out_csr_ = csr::from_edges(edges, 0);
}

BoolArray ImmutableGraph::HasVertices(IdArray vids) const {
  CHECK(IsValidIdArray(vids)) << "Invalid vertex id array.";
  const auto len = vids->shape[0];
  BoolArray rst = BoolArray::Empty({len}, vids->dtype, vids->ctx);
  const dgl_id_t* vid_data = static_cast<dgl_id_t*>(vids->data);
  dgl_id_t* rst_data = static_cast<dgl_id_t*>(rst->data);
  const int64_t nverts = NumVertices();
  for (int64_t i = 0; i < len; ++i) {
    rst_data[i] = (vid_data[i] < nverts)? 1 : 0;
  }
  return rst;
}

BoolArray ImmutableGraph::HasEdgesBetween(IdArray src_ids, IdArray dst_ids) const {
  CHECK(IsValidIdArray(src_ids)) << "Invalid src id array.";
  CHECK(IsValidIdArray(dst_ids)) << "Invalid dst id array.";
  const auto srclen = src_ids->shape[0];
  const auto dstlen = dst_ids->shape[0];
  const auto rstlen = std::max(srclen, dstlen);
  BoolArray rst = BoolArray::Empty({rstlen}, src_ids->dtype, src_ids->ctx);
  dgl_id_t* rst_data = static_cast<dgl_id_t*>(rst->data);
  const dgl_id_t* src_data = static_cast<dgl_id_t*>(src_ids->data);
  const dgl_id_t* dst_data = static_cast<dgl_id_t*>(dst_ids->data);
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

IdArray ImmutableGraph::Predecessors(dgl_id_t vid, uint64_t radius) const {
  CHECK(HasVertex(vid)) << "invalid vertex: " << vid;
  CHECK(radius >= 1) << "invalid radius: " << radius;

  auto pred = this->GetInCSR()->GetIndexRef(vid);
  const int64_t len = pred.second - pred.first;
  IdArray rst = IdArray::Empty({len}, DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
  dgl_id_t* rst_data = static_cast<dgl_id_t*>(rst->data);

  std::copy(pred.first, pred.second, rst_data);
  return rst;
}

IdArray ImmutableGraph::Successors(dgl_id_t vid, uint64_t radius) const {
  CHECK(HasVertex(vid)) << "invalid vertex: " << vid;
  CHECK(radius >= 1) << "invalid radius: " << radius;

  auto succ = this->GetOutCSR()->GetIndexRef(vid);
  const int64_t len = succ.second - succ.first;
  IdArray rst = IdArray::Empty({len}, DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
  dgl_id_t* rst_data = static_cast<dgl_id_t*>(rst->data);

  std::copy(succ.first, succ.second, rst_data);
  return rst;
}

std::pair<const dgl_id_t *, const dgl_id_t *> ImmutableGraph::GetInEdgeIdRef(dgl_id_t src,
                                                                         dgl_id_t dst) const {
  assert(this->in_csr_);
  auto pred = this->in_csr_->GetIndexRef(dst);
  auto it = std::lower_bound(pred.first, pred.second, src);
  // If there doesn't exist edges between the two nodes.
  if (it == pred.second || *it != src)
    return std::pair<const dgl_id_t *, const dgl_id_t *>(nullptr, nullptr);

  size_t off = it - in_csr_->indices.data();
  assert(off < in_csr_->indices.size());
  const dgl_id_t *start = &in_csr_->edge_ids[off];
  int64_t len = 0;
  // There are edges between the source and the destination.
  for (auto it1 = it; it1 != pred.second && *it1 == src; it1++, len++);
  return std::pair<const dgl_id_t *, const dgl_id_t *>(start, start + len);
}

std::pair<const dgl_id_t *, const dgl_id_t *> ImmutableGraph::GetOutEdgeIdRef(dgl_id_t src,
                                                                              dgl_id_t dst) const {
  assert(this->out_csr_);
  auto succ = this->out_csr_->GetIndexRef(src);
  auto it = std::lower_bound(succ.first, succ.second, dst);
  // If there doesn't exist edges between the two nodes.
  if (it == succ.second || *it != dst)
    return std::pair<const dgl_id_t *, const dgl_id_t *>(nullptr, nullptr);

  size_t off = it - out_csr_->indices.data();
  assert(off < out_csr_->indices.size());
  const dgl_id_t *start = &out_csr_->edge_ids[off];
  int64_t len = 0;
  // There are edges between the source and the destination.
  for (auto it1 = it; it1 != succ.second && *it1 == dst; it1++, len++);
  return std::pair<const dgl_id_t *, const dgl_id_t *>(start, start + len);
}

IdArray ImmutableGraph::EdgeId(dgl_id_t src, dgl_id_t dst) const {
  CHECK(HasVertex(src) && HasVertex(dst)) << "invalid edge: " << src << " -> " << dst;

  std::pair<const dgl_id_t *, const dgl_id_t *> edge_ids;
  if (in_csr_)
    edge_ids = GetInEdgeIdRef(src, dst);
  else
    edge_ids = GetOutEdgeIdRef(src, dst);
  int64_t len = edge_ids.second - edge_ids.first;
  IdArray rst = IdArray::Empty({len}, DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
  dgl_id_t* rst_data = static_cast<dgl_id_t*>(rst->data);
  if (len > 0)
    std::copy(edge_ids.first, edge_ids.second, rst_data);

  return rst;
}

ImmutableGraph::EdgeArray ImmutableGraph::EdgeIds(IdArray src_ids, IdArray dst_ids) const {
  CHECK(IsValidIdArray(src_ids)) << "Invalid src id array.";
  CHECK(IsValidIdArray(dst_ids)) << "Invalid dst id array.";
  const auto srclen = src_ids->shape[0];
  const auto dstlen = dst_ids->shape[0];
  int64_t i, j;

  CHECK((srclen == dstlen) || (srclen == 1) || (dstlen == 1))
    << "Invalid src and dst id array.";

  const int src_stride = (srclen == 1 && dstlen != 1) ? 0 : 1;
  const int dst_stride = (dstlen == 1 && srclen != 1) ? 0 : 1;
  const dgl_id_t* src_data = static_cast<dgl_id_t*>(src_ids->data);
  const dgl_id_t* dst_data = static_cast<dgl_id_t*>(dst_ids->data);

  std::vector<dgl_id_t> src, dst, eid;

  for (i = 0, j = 0; i < srclen && j < dstlen; i += src_stride, j += dst_stride) {
    const dgl_id_t src_id = src_data[i], dst_id = dst_data[j];
    CHECK(HasVertex(src_id) && HasVertex(dst_id)) <<
        "invalid edge: " << src_id << " -> " << dst_id;

    std::pair<const dgl_id_t *, const dgl_id_t *> edges;
    if (this->in_csr_)
      edges = this->GetInEdgeIdRef(src_id, dst_id);
    else 
      edges = this->GetOutEdgeIdRef(src_id, dst_id);

    size_t len = edges.second - edges.first;
    for (size_t i = 0; i < len; i++) {
        src.push_back(src_id);
        dst.push_back(dst_id);
        eid.push_back(edges.first[i]);
    }
  }

  int64_t rstlen = src.size();
  IdArray rst_src = IdArray::Empty({rstlen}, src_ids->dtype, src_ids->ctx);
  IdArray rst_dst = IdArray::Empty({rstlen}, src_ids->dtype, src_ids->ctx);
  IdArray rst_eid = IdArray::Empty({rstlen}, src_ids->dtype, src_ids->ctx);
  dgl_id_t* rst_src_data = static_cast<dgl_id_t*>(rst_src->data);
  dgl_id_t* rst_dst_data = static_cast<dgl_id_t*>(rst_dst->data);
  dgl_id_t* rst_eid_data = static_cast<dgl_id_t*>(rst_eid->data);

  std::copy(src.begin(), src.end(), rst_src_data);
  std::copy(dst.begin(), dst.end(), rst_dst_data);
  std::copy(eid.begin(), eid.end(), rst_eid_data);

  return ImmutableGraph::EdgeArray{rst_src, rst_dst, rst_eid};
}

ImmutableGraph::EdgeArray ImmutableGraph::Edges(bool sorted) const {
  int64_t rstlen = NumEdges();
  IdArray rst_src = IdArray::Empty({rstlen}, DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
  IdArray rst_dst = IdArray::Empty({rstlen}, DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
  IdArray rst_eid = IdArray::Empty({rstlen}, DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
  dgl_id_t* rst_src_data = static_cast<dgl_id_t*>(rst_src->data);
  dgl_id_t* rst_dst_data = static_cast<dgl_id_t*>(rst_dst->data);
  dgl_id_t* rst_eid_data = static_cast<dgl_id_t*>(rst_eid->data);

  auto out_csr = GetOutCSR();
  // If sorted, the returned edges are sorted by the source Id and dest Id.
  for (size_t i = 0; i < out_csr->indptr.size() - 1; i++) {
    std::fill(rst_src_data + out_csr->indptr[i], rst_src_data + out_csr->indptr[i + 1],
              static_cast<dgl_id_t>(i));
  }
  std::copy(out_csr->indices.begin(), out_csr->indices.end(), rst_dst_data);
  std::copy(out_csr->edge_ids.begin(), out_csr->edge_ids.end(), rst_eid_data);

  // TODO(zhengda) do I need to sort the edges if sorted = false?

  return ImmutableGraph::EdgeArray{rst_src, rst_dst, rst_eid};
}

ImmutableSubgraph ImmutableGraph::VertexSubgraph(IdArray vids) const {
  ImmutableSubgraph subg;
  std::pair<std::shared_ptr<csr>, IdArray> ret;
  if (in_csr_) {
    ret = in_csr_->VertexSubgraph(vids);
    subg.graph = ImmutableGraph(ret.first, nullptr, IsMultigraph());
  } else {
    assert(out_csr_);
    ret = out_csr_->VertexSubgraph(vids);
    subg.graph = ImmutableGraph(nullptr, ret.first, IsMultigraph());
  }
  subg.induced_vertices = vids;
  subg.induced_edges = ret.second;
  return subg;
}

ImmutableSubgraph ImmutableGraph::EdgeSubgraph(IdArray eids) const {
  return ImmutableSubgraph();
}

}  // namespace dgl
