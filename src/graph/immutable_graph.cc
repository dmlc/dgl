/*!
 *  Copyright (c) 2018 by Contributors
 * \file graph/immutable_graph.cc
 * \brief DGL immutable graph index implementation
 */

#include <string.h>
#include <sys/types.h>
#include <dgl/immutable_graph.h>

#include "../c_api_common.h"
#include "./common.h"

#ifdef _MSC_VER
#define _CRT_RAND_S
#endif

namespace dgl {
namespace {
template<class ForwardIt, class T>
bool BinarySearch(ForwardIt first, ForwardIt last, const T& value) {
  first = std::lower_bound(first, last, value);
  return (!(first == last) && !(value < *first));
}
}  // namespace

//////////////////////////////////////////////////////////
//
// CSR graph implementation
//
//////////////////////////////////////////////////////////

CSR::CSR(int64_t num_vertices, int64_t num_edges) {
  indptr_ = NewIdArray(num_vertices + 1);
  indices_ = NewIdArray(num_edges);
  edge_ids_ = NewIdArray(num_edges);
}

CSR::CSR(IdArray indptr, IdArray indices, IdArray edge_ids)
  : indptr_(indptr), indices_(indices), edge_ids_(edge_ids) {
  CHECK(IsValidIdArray(indptr));
  CHECK(IsValidIdArray(indices));
  CHECK(IsValidIdArray(edge_ids));
  CHECK_EQ(indices_->shape[0], edge_ids_->shape[0]);
}

CSR::CSR(IdArray indptr, IdArray indices, IdArray edge_ids,
         const std::string &shared_mem_name) {
#ifndef _WIN32
  CHECK(IsValidIdArray(indptr));
  CHECK(IsValidIdArray(indices));
  CHECK(IsValidIdArray(edge_ids));
  CHECK_EQ(indices_->shape[0], edge_ids_->shape[0]);
  const int64_t num_verts = indptr->shape[0] - 1;
  const int64_t num_edges = indices->shape[0];
  const int64_t file_size = (num_verts + 1 + num_edges * 2) * sizeof(dgl_id_t);

  IdArray sm_array = IdArray::EmptyShared(
      shared_mem_name, {file_size}, DLDataType{kDLInt, 8, 1}, DLContext{kDLCPU, 0}, true);
  // Create views from the shared memory array. Note that we don't need to save
  //   the sm_array because the refcount is maintained by the view arrays.
  indptr_ = sm_array.CreateView({num_verts + 1}, DLDataType{kDLInt, 64, 1});
  indices_ = sm_array.CreateView({num_edges}, DLDataType{kDLInt, 64, 1},
      (num_verts + 1) * sizeof(dgl_id_t));
  edge_ids_ = sm_array.CreateView({num_edges}, DLDataType{kDLInt, 64, 1},
      (num_verts + 1 + num_edges) * sizeof(dgl_id_t));
  // copy the given data into the shared memory arrays
  indptr_.CopyFrom(indptr);
  indices_.CopyFrom(indices);
  edge_ids_.CopyFrom(edge_ids);
#else
  LOG(FATAL) << "ImmutableGraph doesn't support shared memory in Windows yet";
#endif  // _WIN32
}

CSR::CSR(const std::string &shared_mem_name,
         int64_t num_verts, int64_t num_edges) {
#ifndef _WIN32
  const int64_t file_size = (num_verts + 1 + num_edges * 2) * sizeof(dgl_id_t);
  IdArray sm_array = IdArray::EmptyShared(
      shared_mem_name, {file_size}, DLDataType{kDLInt, 8, 1}, DLContext{kDLCPU, 0}, true);
  // Create views from the shared memory array. Note that we don't need to save
  //   the sm_array because the refcount is maintained by the view arrays.
  indptr_ = sm_array.CreateView({num_verts + 1}, DLDataType{kDLInt, 64, 1});
  indices_ = sm_array.CreateView({num_edges}, DLDataType{kDLInt, 64, 1},
      (num_verts + 1) * sizeof(dgl_id_t));
  edge_ids_ = sm_array.CreateView({num_edges}, DLDataType{kDLInt, 64, 1},
      (num_verts + 1 + num_edges) * sizeof(dgl_id_t));
#else
  LOG(FATAL) << "ImmutableGraph doesn't support shared memory in Windows yet";
#endif  // _WIN32
}

ImmutableGraph::EdgeArray CSR::OutEdges(dgl_id_t vid) const {
  CHECK(HasVertex(vid)) << "invalid vertex: " << vid;
  const dgl_id_t* indptr_data = static_cast<dgl_id_t*>(indptr_->data);
  const dgl_id_t* indices_data = static_cast<dgl_id_t*>(indices_->data);
  const dgl_id_t* edge_ids_data = static_cast<dgl_id_t*>(edge_ids_->data);
  const dgl_id_t off = indptr_data[vid];
  const int64_t len = OutDegree(vid);
  IdArray src = NewIdArray(len);
  IdArray dst = NewIdArray(len);
  IdArray eid = NewIdArray(len);
  dgl_id_t* src_data = static_cast<dgl_id_t*>(src->data);
  dgl_id_t* dst_data = static_cast<dgl_id_t*>(dst->data);
  dgl_id_t* eid_data = static_cast<dgl_id_t*>(eid->data);
  std::fill(src_data, src_data + len, vid);
  std::copy(indices_data + off, indices_data + off + len, dst_data);
  std::copy(edge_ids_data + off, edge_ids_data + off + len, eid_data);
  return ImmutableGraph::EdgeArray{src, dst, eid};
}

ImmutableGraph::EdgeArray CSR::OutEdges(IdArray vids) const {
  CHECK(IsValidIdArray(vids)) << "Invalid vertex id array.";
  const dgl_id_t* indptr_data = static_cast<dgl_id_t*>(indptr_->data);
  const dgl_id_t* indices_data = static_cast<dgl_id_t*>(indices_->data);
  const dgl_id_t* edge_ids_data = static_cast<dgl_id_t*>(edge_ids_->data);
  const auto len = vids->shape[0];
  const dgl_id_t* vid_data = static_cast<dgl_id_t*>(vids->data);
  int64_t rstlen = 0;
  for (int64_t i = 0; i < len; ++i) {
    dgl_id_t vid = vid_data[i];
    CHECK(HasVertex(vid)) << "Invalid vertex: " << vid;
    rstlen += OutDegree(vid);
  }
  IdArray src = NewIdArray(rstlen);
  IdArray dst = NewIdArray(rstlen);
  IdArray eid = NewIdArray(rstlen);
  dgl_id_t* src_ptr = static_cast<dgl_id_t*>(src->data);
  dgl_id_t* dst_ptr = static_cast<dgl_id_t*>(dst->data);
  dgl_id_t* eid_ptr = static_cast<dgl_id_t*>(eid->data);
  for (int64_t i = 0; i < len; ++i) {
    const dgl_id_t vid = vid_data[i];
    const dgl_id_t off = indptr_data[vid];
    const int64_t deg = OutDegree(vid);
    if (deg == 0)
      continue;
    const auto *succ = indices_data + off;
    const auto *eids = edge_ids_data + off;
    for (int64_t j = 0; j < deg; ++j) {
      *(src_ptr++) = vid;
      *(dst_ptr++) = succ[j];
      *(eid_ptr++) = eids[j];
    }
  }
  return ImmutableGraph::EdgeArray{src, dst, eid};
}

DegreeArray CSR::OutDegrees(IdArray vids) const {
  CHECK(IsValidIdArray(vids)) << "Invalid vertex id array.";
  const auto len = vids->shape[0];
  const dgl_id_t* vid_data = static_cast<dgl_id_t*>(vids->data);
  DegreeArray rst = DegreeArray::Empty({len}, vids->dtype, vids->ctx);
  dgl_id_t* rst_data = static_cast<dgl_id_t*>(rst->data);
  for (int64_t i = 0; i < len; ++i) {
    const auto vid = vid_data[i];
    CHECK(HasVertex(vid)) << "Invalid vertex: " << vid;
    rst_data[i] = OutDegree(vid);
  }
  return rst;
}

bool CSR::HasEdgeBetween(dgl_id_t src, dgl_id_t dst) const {
  CHECK(HasVertex(src)) << "Invalid vertex id: " << src;
  CHECK(HasVertex(dst)) << "Invalid vertex id: " << dst;
  const auto& range = GetEdgeRange(src, dst);
  return range.first != range.second;
}

IdArray CSR::Successors(dgl_id_t vid, uint64_t radius) const {
  CHECK(HasVertex(vid)) << "invalid vertex: " << vid;
  CHECK(radius == 1) << "invalid radius: " << radius;
  const dgl_id_t* indptr_data = static_cast<dgl_id_t*>(indptr_->data);
  const dgl_id_t* indices_data = static_cast<dgl_id_t*>(indices_->data);
  const int64_t len = indptr_data[vid + 1] - indptr_data[vid];
  IdArray rst = NewIdArray(len);
  dgl_id_t* rst_data = static_cast<dgl_id_t*>(rst->data);
  std::copy(indices_data + indptr_data[vid],
            indices_data + indptr_data[vid + 1],
            rst_data);
  return rst;
}

IdArray CSR::EdgeId(dgl_id_t src, dgl_id_t dst) const {
  CHECK(HasVertex(src)) << "invalid vertex: " << src;
  CHECK(HasVertex(dst)) << "invalid vertex: " << dst;
  const auto& range = GetEdgeRange(src, dst);
  if (range.first == range.second) {
    LOG(FATAL) << "Invalid edge: " << src << " -> " << dst;
  }
  const dgl_id_t* eid_data = static_cast<dgl_id_t*>(edge_ids_->data);
  IdArray ret = NewIdArray(range.second - range.first);
  dgl_id_t* ret_data = static_cast<dgl_id_t*>(ret->data);
  std::copy(eid_data + range.first, eid_data + range.second, ret_data);
  return ret;
}

CSR::EdgeArray CSR::EdgeIds(IdArray src_ids, IdArray dst_ids) const {
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
  const dgl_id_t* eid_data = static_cast<dgl_id_t*>(edge_ids_->data);

  std::vector<dgl_id_t> src, dst, eid;

  // TODO(minjie): Below algorithm can be improved. Currently, the complexity is
  //   O(K*log(E)), where K is the length of given src/dst pairs and E is the number
  //   of edges. This can be improved to O(K + E) since the CSR structure is sorted.
  for (int64_t i = 0, j = 0; i < srclen && j < dstlen; i += src_stride, j += dst_stride) {
    const dgl_id_t src_id = src_data[i], dst_id = dst_data[j];
    CHECK(HasVertex(src_id) && HasVertex(dst_id)) <<
        "invalid edge: " << src_id << " -> " << dst_id;

    const auto& range = GetEdgeRange(src_id, dst_id);
    if (range.first == range.second) {
      LOG(FATAL) << "Invalid edge: " << src_id << " -> " << dst_id;
    }
    for (int64_t k = range.first; k < range.second; k++) {
        src.push_back(src_id);
        dst.push_back(dst_id);
        eid.push_back(eid_data[k]);
    }
  }

  const int64_t rstlen = src.size();
  IdArray rst_src = NewIdArray(rstlen);
  IdArray rst_dst = NewIdArray(rstlen);
  IdArray rst_eid = NewIdArray(rstlen);
  dgl_id_t* rst_src_data = static_cast<dgl_id_t*>(rst_src->data);
  dgl_id_t* rst_dst_data = static_cast<dgl_id_t*>(rst_dst->data);
  dgl_id_t* rst_eid_data = static_cast<dgl_id_t*>(rst_eid->data);

  std::copy(src.begin(), src.end(), rst_src_data);
  std::copy(dst.begin(), dst.end(), rst_dst_data);
  std::copy(eid.begin(), eid.end(), rst_eid_data);

  return CSR::EdgeArray{rst_src, rst_dst, rst_eid};
}

CSR::EdgeArray CSR::Edges(const std::string &order) const {
  CHECK(order.empty() || order == std::string("srcdst"))
    << "COO only support Edges of order \"srcdst\","
    << " but got \"" << order << "\".";
  const int64_t rstlen = NumEdges();
  const dgl_id_t* indptr_data = static_cast<dgl_id_t*>(indptr_->data);
  IdArray rst_src = NewIdArray(rstlen);
  dgl_id_t* rst_src_data = static_cast<dgl_id_t*>(rst_src->data);

  // If sorted, the returned edges are sorted by the source Id and dest Id.
  for (dgl_id_t src = 0; src < NumVertices(); ++src) {
    std::fill(rst_src_data + indptr_data[src],
              rst_src_data + indptr_data[src + 1],
              src);
  }

  return CSR::EdgeArray{rst_src, indices_, edge_ids_};
}

Subgraph CSR::VertexSubgraph(IdArray vids) const {
  // TODO
}

Subgraph CSR::EdgeSubgraph(IdArray eids) const {
  // TODO
}

CSRPtr CSR::Transpose() const {
  // TODO
}

COOPtr CSR::ToCOO() const {
  // TODO
}

std::pair<int64_t, int64_t> CSR::GetEdgeRange(dgl_id_t src, dgl_id_t dst) const {
  const dgl_id_t* indptr_data = static_cast<dgl_id_t*>(indptr_->data);
  const dgl_id_t* indices_data = static_cast<dgl_id_t*>(indices_->data);
  auto start = indices_data + indptr_data[src];
  auto last = indices_data + indptr_data[src + 1];
  auto first = std::lower_bound(start, last, dst);
  int64_t len = 0;
  for (auto it = first; it != last && *it == dst; ++it, ++len);
  return std::make_pair(first - start, len);
}

//////////////////////////////////////////////////////////
//
// COO graph implementation
//
//////////////////////////////////////////////////////////

COO::EdgeArray COO::FindEdges(IdArray eids) const {
  CHECK(IsValidIdArray(eids)) << "Invalid edge id array";
  dgl_id_t* eid_data = static_cast<dgl_id_t*>(eids->data);
  int64_t len = eids->shape[0];
  IdArray rst_src = IdArray::Empty({len}, eids->dtype, eids->ctx);
  IdArray rst_dst = IdArray::Empty({len}, eids->dtype, eids->ctx);
  dgl_id_t* rst_src_data = static_cast<dgl_id_t*>(rst_src->data);
  dgl_id_t* rst_dst_data = static_cast<dgl_id_t*>(rst_dst->data);

  for (int64_t i = 0; i < len; i++) {
    auto edge = COO::FindEdge(eid_data[i]);
    rst_src_data[i] = edge.first;
    rst_dst_data[i] = edge.second;
  }

  return COO::EdgeArray{rst_src, rst_dst, eids};
}

COO::EdgeArray COO::Edges(const std::string &order) const {
  const int64_t rstlen = NumEdges();
  CHECK(order.empty() || order == std::string("eid"))
    << "COO only support Edges of order \"eid\", but got \""
    << order << "\".";
  IdArray rst_eid = NewIdArray(rstlen);
  dgl_id_t* rst_eid_data = static_cast<dgl_id_t*>(rst_eid->data);
  for (int64_t i = 0; i < rstlen; ++i) {
    rst_eid_data[i] = i;
  }
  return EdgeArray{src_, dst_, rst_eid};
}

Subgraph COO::EdgeSubgraph(IdArray eids) const {
  // TODO
}

CSRPtr COO::ToCSR() const {
  // TODO
}

//////////////////////////////////////////////////////////
//
// immutable graph implementation
//
//////////////////////////////////////////////////////////

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

ImmutableGraph::EdgeArray ImmutableGraph::Edges(const std::string &order) const {
  if (order.empty()) {
    // arbitrary order
    return AnyGraph()->Edges(order);
  } else if (order == std::string("srcdst")) {
    return GetOutCSR()->Edges(order);
  } else if (order == std::string("eid")) {
    return GetCOO()->Edges(order);
  } else {
    LOG(FATAL) << "Unsupported order request: " << order;
  }
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
