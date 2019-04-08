/*!
 *  Copyright (c) 2018 by Contributors
 * \file graph/immutable_graph.cc
 * \brief DGL immutable graph index implementation
 */

#include <string.h>
#include <sys/types.h>
#include <dgl/immutable_graph.h>

#ifdef _MSC_VER
#define _CRT_RAND_S
#endif

#include "../c_api_common.h"

namespace dgl {

template<class ForwardIt, class T>
bool binary_search(ForwardIt first, ForwardIt last, const T& value) {
  first = std::lower_bound(first, last, value);
  return (!(first == last) && !(value < *first));
}

ImmutableGraph::CSR::CSR(int64_t num_vertices, int64_t expected_num_edges):
    indptr(num_vertices + 1), indices(expected_num_edges), edge_ids(expected_num_edges) {
  indptr.resize(num_vertices + 1);
}

ImmutableGraph::CSR::CSR(IdArray indptr_arr, IdArray index_arr, IdArray edge_id_arr):
    indptr(indptr_arr->shape[0]), indices(index_arr->shape[0]), edge_ids(index_arr->shape[0]) {
  size_t num_vertices = indptr_arr->shape[0] - 1;
  size_t num_edges = index_arr->shape[0];
  const int64_t *indptr_data = static_cast<int64_t*>(indptr_arr->data);
  const dgl_id_t *indices_data = static_cast<dgl_id_t*>(index_arr->data);
  const dgl_id_t *edge_id_data = static_cast<dgl_id_t*>(edge_id_arr->data);
  CHECK_EQ(indptr_data[0], 0);
  CHECK_EQ(indptr_data[num_vertices], num_edges);
  indptr.insert_back(indptr_data, num_vertices + 1);
  indices.insert_back(indices_data, num_edges);
  edge_ids.insert_back(edge_id_data, num_edges);
}

ImmutableGraph::CSR::CSR(IdArray indptr_arr, IdArray index_arr, IdArray edge_id_arr,
                         const std::string &shared_mem_name) {
#ifndef _WIN32
  size_t num_vertices = indptr_arr->shape[0] - 1;
  size_t num_edges = index_arr->shape[0];
  CHECK_EQ(num_edges, edge_id_arr->shape[0]);
  size_t file_size = (num_vertices + 1) * sizeof(int64_t) + num_edges * sizeof(dgl_id_t) * 2;

  auto mem = std::make_shared<runtime::SharedMemory>(shared_mem_name);
  auto ptr = mem->create_new(file_size);

  int64_t *addr1 = static_cast<int64_t *>(ptr);
  indptr.init(addr1, num_vertices + 1);
  void *addr = addr1 + num_vertices + 1;
  dgl_id_t *addr2 = static_cast<dgl_id_t *>(addr);
  indices.init(addr2, num_edges);
  addr = addr2 + num_edges;
  dgl_id_t *addr3 = static_cast<dgl_id_t *>(addr);
  edge_ids.init(addr3, num_edges);

  const int64_t *indptr_data = static_cast<int64_t*>(indptr_arr->data);
  const dgl_id_t *indices_data = static_cast<dgl_id_t*>(index_arr->data);
  const dgl_id_t *edge_id_data = static_cast<dgl_id_t*>(edge_id_arr->data);
  CHECK_EQ(indptr_data[0], 0);
  CHECK_EQ(indptr_data[num_vertices], num_edges);
  indptr.insert_back(indptr_data, num_vertices + 1);
  indices.insert_back(indices_data, num_edges);
  edge_ids.insert_back(edge_id_data, num_edges);
  this->mem = mem;
#else
  LOG(FATAL) << "ImmutableGraph doesn't support shared memory in Windows yet";
#endif  // _WIN32
}

ImmutableGraph::CSR::CSR(const std::string &shared_mem_name,
                         size_t num_vertices, size_t num_edges) {
#ifndef _WIN32
  size_t file_size = (num_vertices + 1) * sizeof(int64_t) + num_edges * sizeof(dgl_id_t) * 2;
  auto mem = std::make_shared<runtime::SharedMemory>(shared_mem_name);
  auto ptr = mem->open(file_size);

  int64_t *addr1 = static_cast<int64_t *>(ptr);
  indptr.init(addr1, num_vertices + 1, num_vertices + 1);
  void *addr = addr1 + num_vertices + 1;
  dgl_id_t *addr2 = static_cast<dgl_id_t *>(addr);
  indices.init(addr2, num_edges, num_edges);
  addr = addr2 + num_edges;
  dgl_id_t *addr3 = static_cast<dgl_id_t *>(addr);
  edge_ids.init(addr3, num_edges, num_edges);
  this->mem = mem;
#else
  LOG(FATAL) << "ImmutableGraph doesn't support shared memory in Windows yet";
#endif  // _WIN32
}

ImmutableGraph::EdgeArray ImmutableGraph::CSR::GetEdges(dgl_id_t vid) const {
  CHECK(HasVertex(vid)) << "invalid vertex: " << vid;
  const int64_t off = this->indptr[vid];
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

ImmutableGraph::EdgeArray ImmutableGraph::CSR::GetEdges(IdArray vids) const {
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
    const int64_t deg = this->GetDegree(vid);
    if (deg == 0)
      continue;
    const auto *pred = &this->indices[off];
    const auto *eids = &this->edge_ids[off];
    for (int64_t j = 0; j < deg; ++j) {
      *(src_ptr++) = pred[j];
      *(dst_ptr++) = vid;
      *(eid_ptr++) = eids[j];
    }
  }
  return ImmutableGraph::EdgeArray{src, dst, eid};
}

DegreeArray ImmutableGraph::CSR::GetDegrees(IdArray vids) const {
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
  // This bitmap is used as a bloom filter to remove some lookups.
  // Hashtable is very slow. Using bloom filter can significantly speed up lookups.
  Bitmap map;

  /*
   * This is to test if a vertex is in the induced subgraph.
   * If it is, the edge on this vertex and the source vertex will be collected.
   * `old_id` is the vertex we test, `old_eid` is the edge Id between the `old_id`
   * and the source vertex. `col_idx` and `orig_eids` store the collected edges.
   */
  void Collect(const dgl_id_t old_id, const dgl_id_t old_eid,
               ImmutableGraph::CSR::vector<dgl_id_t> *col_idx,
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

 public:
  HashTableChecker(const dgl_id_t *vid_data, int64_t len): map(vid_data, len) {
    oldv2newv.reserve(len);
    for (int64_t i = 0; i < len; ++i) {
      oldv2newv[vid_data[i]] = i;
    }
  }

  /*
   * This is to collect edges from the neighborhood of a vertex.
   * `neigh_idx`, `eids` and `row_len` indicates the neighbor list of the vertex.
   * The collected edges are stored in `new_neigh_idx` and `orig_eids`.
   */
  void CollectOnRow(const dgl_id_t neigh_idx[], const dgl_id_t eids[], size_t row_len,
                    ImmutableGraph::CSR::vector<dgl_id_t> *new_neigh_idx,
                    std::vector<dgl_id_t> *orig_eids) {
    // TODO(zhengda) I need to make sure the column index in each row is sorted.
    for (size_t j = 0; j < row_len; ++j) {
      const dgl_id_t oldsucc = neigh_idx[j];
      const dgl_id_t eid = eids[j];
      Collect(oldsucc, eid, new_neigh_idx, orig_eids);
    }
  }
};

ImmutableGraph::EdgeList::Ptr ImmutableGraph::EdgeList::FromCSR(
    const CSR::vector<int64_t>& indptr,
    const CSR::vector<dgl_id_t>& indices,
    const CSR::vector<dgl_id_t>& edge_ids,
    bool in_csr) {
  const auto n = indptr.size() - 1;
  const auto len = edge_ids.size();
  auto t = std::make_shared<EdgeList>(len, n);
  for (size_t i = 0; i < indptr.size() - 1; i++) {
    for (int64_t j = indptr[i]; j < indptr[i + 1]; j++) {
      dgl_id_t row = i, col = indices[j];
      if (in_csr)
        std::swap(row, col);
      t->register_edge(edge_ids[j], row, col);
    }
  }
  return t;
}

std::pair<ImmutableGraph::CSR::Ptr, IdArray> ImmutableGraph::CSR::VertexSubgraph(
    IdArray vids) const {
  CHECK(IsValidIdArray(vids)) << "Invalid vertex id array.";
  const dgl_id_t* vid_data = static_cast<dgl_id_t*>(vids->data);
  const int64_t len = vids->shape[0];

  HashTableChecker def_check(vid_data, len);
  // check if vid_data is sorted.
  CHECK(std::is_sorted(vid_data, vid_data + len)) << "The input vertex list has to be sorted";

  // Collect the non-zero entries in from the original graph.
  std::vector<dgl_id_t> orig_edge_ids;
  orig_edge_ids.reserve(len);
  auto sub_csr = std::make_shared<CSR>(len, len);
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
  for (size_t i = 0; i < sub_csr->edge_ids.size(); i++)
    sub_csr->edge_ids[i] = i;

  IdArray rst_eids = IdArray::Empty({static_cast<int64_t>(orig_edge_ids.size())},
                                    DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
  dgl_id_t* eid_data = static_cast<dgl_id_t*>(rst_eids->data);
  std::copy(orig_edge_ids.begin(), orig_edge_ids.end(), eid_data);

  return std::pair<ImmutableGraph::CSR::Ptr, IdArray>(sub_csr, rst_eids);
}

std::pair<ImmutableGraph::CSR::Ptr, IdArray> ImmutableGraph::CSR::EdgeSubgraph(
    IdArray eids, EdgeList::Ptr edge_list) const {
  // Return sub_csr and vids array.
  CHECK(IsValidIdArray(eids)) << "Invalid edge id array.";
  const dgl_id_t* eid_data = static_cast<dgl_id_t*>(eids->data);
  const int64_t len = eids->shape[0];
  std::vector<dgl_id_t> nodes;
  std::unordered_map<dgl_id_t, dgl_id_t> oldv2newv;
  std::vector<Edge> edges;

  for (int64_t i = 0; i < len; i++) {
    dgl_id_t src_id = edge_list->src_points[eid_data[i]];
    dgl_id_t dst_id = edge_list->dst_points[eid_data[i]];

    // pair<iterator, bool>, the second indicates whether the insertion is successful or not.
    auto src_pair = oldv2newv.insert(std::make_pair(src_id, oldv2newv.size()));
    auto dst_pair = oldv2newv.insert(std::make_pair(dst_id, oldv2newv.size()));
    if (src_pair.second)
      nodes.push_back(src_id);
    if (dst_pair.second)
      nodes.push_back(dst_id);
    edges.push_back(Edge{src_pair.first->second, dst_pair.first->second, static_cast<dgl_id_t>(i)});
  }

  const size_t n = oldv2newv.size();
  auto sub_csr = CSR::FromEdges(&edges, 0, n);

  IdArray rst_vids = IdArray::Empty({static_cast<int64_t>(nodes.size())},
      DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
  dgl_id_t* vid_data = static_cast<dgl_id_t*>(rst_vids->data);
  std::copy(nodes.begin(), nodes.end(), vid_data);

  return std::make_pair(sub_csr, rst_vids);
}


ImmutableGraph::CSR::Ptr ImmutableGraph::CSR::FromEdges(std::vector<Edge> *edges,
                                                        int sort_on, uint64_t num_nodes) {
  CHECK(sort_on == 0 || sort_on == 1) << "we must sort on the first or the second vector";
  int other_end = sort_on == 1 ? 0 : 1;
  // TODO(zhengda) we should sort in parallel.
  std::sort(edges->begin(), edges->end(), [sort_on, other_end](const Edge &e1, const Edge &e2) {
    if (e1.end_points[sort_on] == e2.end_points[sort_on]) {
      return e1.end_points[other_end] < e2.end_points[other_end];
    } else {
      return e1.end_points[sort_on] < e2.end_points[sort_on];
    }
  });
  auto t = std::make_shared<CSR>(0, 0);
  t->indices.resize(edges->size());
  t->edge_ids.resize(edges->size());
  for (size_t i = 0; i < edges->size(); i++) {
    t->indices[i] = edges->at(i).end_points[other_end];
    CHECK(t->indices[i] < num_nodes);
    t->edge_ids[i] = edges->at(i).edge_id;
    dgl_id_t vid = edges->at(i).end_points[sort_on];
    CHECK(vid < num_nodes);
    while (vid > 0 && t->indptr.size() <= static_cast<size_t>(vid)) {
      t->indptr.push_back(i);
    }
    CHECK(t->indptr.size() == vid + 1);
  }
  while (t->indptr.size() < num_nodes + 1) {
    t->indptr.push_back(edges->size());
  }
  CHECK(t->indptr.size() == num_nodes + 1);
  return t;
}

void ImmutableGraph::CSR::ReadAllEdges(std::vector<Edge> *edges) const {
  edges->resize(NumEdges());
  for (size_t i = 0; i < NumVertices(); i++) {
    // If all the remaining nodes don't have edges.
    if (indptr[i] == indptr[NumVertices()])
      break;
    const dgl_id_t *indices_begin = &indices[indptr[i]];
    const dgl_id_t *eid_begin = &edge_ids[indptr[i]];
    for (size_t j = 0; j < GetDegree(i); j++) {
      Edge e;
      e.end_points[0] = i;
      e.end_points[1] = indices_begin[j];
      e.edge_id = eid_begin[j];
      (*edges)[indptr[i] + j] = e;
    }
  }
}

ImmutableGraph::CSR::Ptr ImmutableGraph::CSR::Transpose() const {
  std::vector<Edge> edges;
  ReadAllEdges(&edges);
  return FromEdges(&edges, 1, NumVertices());
}

ImmutableGraph::ImmutableGraph(IdArray src_ids, IdArray dst_ids, IdArray edge_ids, size_t num_nodes,
                               bool multigraph) : is_multigraph_(multigraph) {
  CHECK(IsValidIdArray(src_ids)) << "Invalid vertex id array.";
  CHECK(IsValidIdArray(dst_ids)) << "Invalid vertex id array.";
  CHECK(IsValidIdArray(edge_ids)) << "Invalid vertex id array.";
  const int64_t len = src_ids->shape[0];
  CHECK(len == dst_ids->shape[0]);
  CHECK(len == edge_ids->shape[0]);
  const dgl_id_t *src_data = static_cast<dgl_id_t*>(src_ids->data);
  const dgl_id_t *dst_data = static_cast<dgl_id_t*>(dst_ids->data);
  const dgl_id_t *edge_data = static_cast<dgl_id_t*>(edge_ids->data);
  std::vector<Edge> edges(len);
  for (size_t i = 0; i < edges.size(); i++) {
    Edge e;
    e.end_points[0] = src_data[i];
    e.end_points[1] = dst_data[i];
    e.edge_id = edge_data[i];
    edges[i] = e;
  }
  in_csr_ = CSR::FromEdges(&edges, 1, num_nodes);
  out_csr_ = CSR::FromEdges(&edges, 0, num_nodes);
}

BoolArray ImmutableGraph::HasVertices(IdArray vids) const {
  CHECK(IsValidIdArray(vids)) << "Invalid vertex id array.";
  const auto len = vids->shape[0];
  BoolArray rst = BoolArray::Empty({len}, vids->dtype, vids->ctx);
  const dgl_id_t* vid_data = static_cast<dgl_id_t*>(vids->data);
  dgl_id_t* rst_data = static_cast<dgl_id_t*>(rst->data);
  const uint64_t nverts = NumVertices();
  for (int64_t i = 0; i < len; ++i) {
    rst_data[i] = (vid_data[i] < nverts)? 1 : 0;
  }
  return rst;
}

bool ImmutableGraph::HasEdgeBetween(dgl_id_t src, dgl_id_t dst) const {
  if (!HasVertex(src) || !HasVertex(dst)) return false;
  if (this->in_csr_) {
    auto pred = this->in_csr_->GetIndexRef(dst);
    return dgl::binary_search(pred.begin(), pred.end(), src);
  } else {
    CHECK(this->out_csr_) << "one of the CSRs must exist";
    auto succ = this->out_csr_->GetIndexRef(src);
    return dgl::binary_search(succ.begin(), succ.end(), dst);
  }
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
  const int64_t len = pred.size();
  IdArray rst = IdArray::Empty({len}, DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
  dgl_id_t* rst_data = static_cast<dgl_id_t*>(rst->data);

  std::copy(pred.begin(), pred.end(), rst_data);
  return rst;
}

IdArray ImmutableGraph::Successors(dgl_id_t vid, uint64_t radius) const {
  CHECK(HasVertex(vid)) << "invalid vertex: " << vid;
  CHECK(radius >= 1) << "invalid radius: " << radius;

  auto succ = this->GetOutCSR()->GetIndexRef(vid);
  const int64_t len = succ.size();
  IdArray rst = IdArray::Empty({len}, DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
  dgl_id_t* rst_data = static_cast<dgl_id_t*>(rst->data);

  std::copy(succ.begin(), succ.end(), rst_data);
  return rst;
}

DGLIdIters ImmutableGraph::GetInEdgeIdRef(dgl_id_t src, dgl_id_t dst) const {
  CHECK(this->in_csr_);
  auto pred = this->in_csr_->GetIndexRef(dst);
  auto it = std::lower_bound(pred.begin(), pred.end(), src);
  // If there doesn't exist edges between the two nodes.
  if (it == pred.end() || *it != src) {
    return DGLIdIters(it, it);
  }

  size_t off = it - in_csr_->indices.begin();
  CHECK(off < in_csr_->indices.size());
  auto start = in_csr_->edge_ids.begin() + off;
  int64_t len = 0;
  // There are edges between the source and the destination.
  for (auto it1 = it; it1 != pred.end() && *it1 == src; it1++, len++) {}
  return DGLIdIters(start, start + len);
}

DGLIdIters ImmutableGraph::GetOutEdgeIdRef(dgl_id_t src, dgl_id_t dst) const {
  CHECK(this->out_csr_);
  auto succ = this->out_csr_->GetIndexRef(src);
  auto it = std::lower_bound(succ.begin(), succ.end(), dst);
  // If there doesn't exist edges between the two nodes.
  if (it == succ.end() || *it != dst) {
    return DGLIdIters(it, it);
  }

  size_t off = it - out_csr_->indices.begin();
  CHECK(off < out_csr_->indices.size());
  auto start = out_csr_->edge_ids.begin() + off;
  int64_t len = 0;
  // There are edges between the source and the destination.
  for (auto it1 = it; it1 != succ.end() && *it1 == dst; it1++, len++) {}
  return DGLIdIters(start, start + len);
}

IdArray ImmutableGraph::EdgeId(dgl_id_t src, dgl_id_t dst) const {
  CHECK(HasVertex(src) && HasVertex(dst)) << "invalid edge: " << src << " -> " << dst;

  auto edge_ids = in_csr_ ? GetInEdgeIdRef(src, dst) : GetOutEdgeIdRef(src, dst);
  int64_t len = edge_ids.size();
  IdArray rst = IdArray::Empty({len}, DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
  dgl_id_t* rst_data = static_cast<dgl_id_t*>(rst->data);
  if (len > 0) {
    std::copy(edge_ids.begin(), edge_ids.end(), rst_data);
  }

  return rst;
}

ImmutableGraph::EdgeArray ImmutableGraph::EdgeIds(IdArray src_ids, IdArray dst_ids) const {
  CHECK(IsValidIdArray(src_ids)) << "Invalid src id array.";
  CHECK(IsValidIdArray(dst_ids)) << "Invalid dst id array.";
  const auto srclen = src_ids->shape[0];
  const auto dstlen = dst_ids->shape[0];

  CHECK((srclen == dstlen) || (srclen == 1) || (dstlen == 1))
    << "Invalid src and dst id array.";

  const int src_stride = (srclen == 1 && dstlen != 1) ? 0 : 1;
  const int dst_stride = (dstlen == 1 && srclen != 1) ? 0 : 1;
  const dgl_id_t* src_data = static_cast<dgl_id_t*>(src_ids->data);
  const dgl_id_t* dst_data = static_cast<dgl_id_t*>(dst_ids->data);

  std::vector<dgl_id_t> src, dst, eid;

  for (int64_t i = 0, j = 0; i < srclen && j < dstlen; i += src_stride, j += dst_stride) {
    const dgl_id_t src_id = src_data[i], dst_id = dst_data[j];
    CHECK(HasVertex(src_id) && HasVertex(dst_id)) <<
        "invalid edge: " << src_id << " -> " << dst_id;

    auto edges = this->in_csr_ ? GetInEdgeIdRef(src_id, dst_id) : GetOutEdgeIdRef(src_id, dst_id);
    for (size_t k = 0; k < edges.size(); k++) {
        src.push_back(src_id);
        dst.push_back(dst_id);
        eid.push_back(edges[k]);
    }
  }

  const int64_t rstlen = src.size();
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

std::pair<dgl_id_t, dgl_id_t> ImmutableGraph::FindEdge(dgl_id_t eid) const {
  dgl_id_t row = 0, col = 0;
  auto edge_list = GetEdgeList();
  CHECK(eid < NumEdges()) << "Invalid edge id " << eid;
  row = edge_list->src_points[eid];
  col = edge_list->dst_points[eid];
  CHECK(row < NumVertices() && col < NumVertices()) << "Invalid edge id " << eid;
  return std::pair<dgl_id_t, dgl_id_t>(row, col);
}

ImmutableGraph::EdgeArray ImmutableGraph::FindEdges(IdArray eids) const {
  CHECK(IsValidIdArray(eids)) << "Invalid edge id array";
  dgl_id_t* eid_data = static_cast<dgl_id_t*>(eids->data);
  int64_t len = eids->shape[0];
  IdArray rst_src = IdArray::Empty({len}, eids->dtype, eids->ctx);
  IdArray rst_dst = IdArray::Empty({len}, eids->dtype, eids->ctx);
  dgl_id_t* rst_src_data = static_cast<dgl_id_t*>(rst_src->data);
  dgl_id_t* rst_dst_data = static_cast<dgl_id_t*>(rst_dst->data);

  for (int64_t i = 0; i < len; i++) {
    auto edge = ImmutableGraph::FindEdge(eid_data[i]);
    rst_src_data[i] = edge.first;
    rst_dst_data[i] = edge.second;
  }

  return ImmutableGraph::EdgeArray{rst_src, rst_dst, eids};
}

ImmutableGraph::EdgeArray ImmutableGraph::Edges(const std::string &order) const {
  int64_t rstlen = NumEdges();
  IdArray rst_src = IdArray::Empty({rstlen}, DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
  IdArray rst_dst = IdArray::Empty({rstlen}, DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
  IdArray rst_eid = IdArray::Empty({rstlen}, DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
  dgl_id_t* rst_src_data = static_cast<dgl_id_t*>(rst_src->data);
  dgl_id_t* rst_dst_data = static_cast<dgl_id_t*>(rst_dst->data);
  dgl_id_t* rst_eid_data = static_cast<dgl_id_t*>(rst_eid->data);

  if (order == "srcdst") {
    auto out_csr = GetOutCSR();
    // If sorted, the returned edges are sorted by the source Id and dest Id.
    for (size_t i = 0; i < out_csr->indptr.size() - 1; i++) {
      std::fill(rst_src_data + out_csr->indptr[i], rst_src_data + out_csr->indptr[i + 1],
          static_cast<dgl_id_t>(i));
    }
    std::copy(out_csr->indices.begin(), out_csr->indices.end(), rst_dst_data);
    std::copy(out_csr->edge_ids.begin(), out_csr->edge_ids.end(), rst_eid_data);
  } else if (order.empty() || order == "eid") {
    std::vector<Edge> edges;
    auto out_csr = GetOutCSR();
    out_csr->ReadAllEdges(&edges);
    std::sort(edges.begin(), edges.end(), [](const Edge &e1, const Edge &e2) {
      return e1.edge_id < e2.edge_id;
    });
    for (size_t i = 0; i < edges.size(); i++) {
      rst_src_data[i] = edges[i].end_points[0];
      rst_dst_data[i] = edges[i].end_points[1];
      rst_eid_data[i] = edges[i].edge_id;
    }
  } else {
    LOG(FATAL) << "unsupported order " << order;
  }

  return ImmutableGraph::EdgeArray{rst_src, rst_dst, rst_eid};
}

Subgraph ImmutableGraph::VertexSubgraph(IdArray vids) const {
  Subgraph subg;
  std::pair<CSR::Ptr, IdArray> ret;
  // We prefer to generate a subgraph for out-csr first.
  if (out_csr_) {
    ret = out_csr_->VertexSubgraph(vids);
    subg.graph = GraphPtr(new ImmutableGraph(nullptr, ret.first, IsMultigraph()));
  } else {
    CHECK(in_csr_);
    ret = in_csr_->VertexSubgraph(vids);
    // When we generate a subgraph, it may be used by only accessing in-edges or out-edges.
    // We don't need to generate both.
    subg.graph = GraphPtr(new ImmutableGraph(ret.first, nullptr, IsMultigraph()));
  }
  subg.induced_vertices = vids;
  subg.induced_edges = ret.second;
  return subg;
}

Subgraph ImmutableGraph::EdgeSubgraph(IdArray eids) const {
  Subgraph subg;
  std::pair<CSR::Ptr, IdArray> ret;
  auto edge_list = GetEdgeList();
  if (out_csr_) {
    ret = out_csr_->EdgeSubgraph(eids, edge_list);
    subg.graph = GraphPtr(new ImmutableGraph(nullptr, ret.first, IsMultigraph()));
  } else {
    ret = in_csr_->EdgeSubgraph(eids, edge_list);
    subg.graph = GraphPtr(new ImmutableGraph(ret.first, nullptr, IsMultigraph()));
  }
  subg.induced_edges = eids;
  subg.induced_vertices = ret.second;
  return subg;
}

ImmutableGraph::CSRArray GetCSRArray(ImmutableGraph::CSR::Ptr csr, size_t start, size_t end) {
  size_t num_rows = end - start;
  size_t nnz = csr->indptr[end] - csr->indptr[start];
  IdArray indptr = IdArray::Empty({static_cast<int64_t>(num_rows + 1)},
                                  DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
  IdArray indices = IdArray::Empty({static_cast<int64_t>(nnz)},
                                   DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
  IdArray eids = IdArray::Empty({static_cast<int64_t>(nnz)},
                                DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
  int64_t *indptr_data = static_cast<int64_t*>(indptr->data);
  dgl_id_t* indices_data = static_cast<dgl_id_t*>(indices->data);
  dgl_id_t* eid_data = static_cast<dgl_id_t*>(eids->data);
  for (size_t i = start; i < end + 1; i++)
    indptr_data[i - start] = csr->indptr[i] - csr->indptr[start];
  std::copy(csr->indices.begin() + csr->indptr[start],
            csr->indices.begin() + csr->indptr[end], indices_data);
  std::copy(csr->edge_ids.begin() + csr->indptr[start],
            csr->edge_ids.begin() + csr->indptr[end], eid_data);
  return ImmutableGraph::CSRArray{indptr, indices, eids};
}

ImmutableGraph::CSRArray ImmutableGraph::GetInCSRArray(size_t start, size_t end) const {
  return GetCSRArray(GetInCSR(), start, end);
}

ImmutableGraph::CSRArray ImmutableGraph::GetOutCSRArray(size_t start, size_t end) const {
  return GetCSRArray(GetOutCSR(), start, end);
}

std::vector<IdArray> ImmutableGraph::GetAdj(bool transpose, const std::string &fmt) const {
  if (fmt == "csr") {
    CSRArray arrs = transpose ? this->GetOutCSRArray(0, NumVertices())
        : this->GetInCSRArray(0, NumVertices());
    return std::vector<IdArray>{arrs.indptr, arrs.indices, arrs.id};
  } else if (fmt == "coo") {
    int64_t num_edges = this->NumEdges();
    IdArray idx = IdArray::Empty({2 * num_edges}, DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
    IdArray eid = IdArray::Empty({num_edges}, DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
    CSR::Ptr csr = transpose ? GetOutCSR() : GetInCSR();
    int64_t *idx_data = static_cast<int64_t*>(idx->data);
    dgl_id_t *eid_data = static_cast<dgl_id_t*>(eid->data);
    for (size_t i = 0; i < csr->indptr.size() - 1; i++) {
      for (int64_t j = csr->indptr[i]; j < csr->indptr[i + 1]; j++)
        idx_data[j] = i;
    }
    std::copy(csr->indices.begin(), csr->indices.end(), idx_data + num_edges);
    std::copy(csr->edge_ids.begin(), csr->edge_ids.end(), eid_data);
    return std::vector<IdArray>{idx, eid};
  } else {
    LOG(FATAL) << "unsupported adjacency matrix format";
    return std::vector<IdArray>();
  }
}

}  // namespace dgl
