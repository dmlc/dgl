/*!
 *  Copyright (c) 2018 by Contributors
 * \file graph/immutable_graph.cc
 * \brief DGL immutable graph index implementation
 */

#include <string.h>
#include <bitset>
#include <numeric>
#include <dgl/immutable_graph.h>

#include "../c_api_common.h"
#include "./common.h"

namespace dgl {
namespace {
/*!
 * \brief A hashmap that maps each ids in the given array to new ids starting from zero.
 */
class IdHashMap {
 public:
  // Construct the hashmap using the given id arrays.
  // The id array could contain duplicates.
  IdHashMap(IdArray ids) {
    const dgl_id_t* ids_data = static_cast<dgl_id_t*>(ids->data);
    const int64_t len = ids->shape[0];
    dgl_id_t newid = 0;
    for (int64_t i = 0; i < len; ++i) {
      const dgl_id_t id = ids_data[i];
      if (!Contains(id)) {
        oldv2newv_[id] = newid++;
        filter_.flip(id & kFilterMask);
      }
    }
  }

  // Return true if the given id is contained in this hashmap.
  bool Contains(dgl_id_t id) const {
    return filter_.test(id & kFilterMask) && oldv2newv_.count(id);
  }

  // Return the new id of the given id.
  dgl_id_t Map(dgl_id_t id) const {
    return oldv2newv_.at(id);
  }

 private:
  static constexpr int32_t kFilterMask = 0xFFFFFF;
  static constexpr int32_t kFilterSize = kFilterMask + 1;
  // This bitmap is used as a bloom filter to remove some lookups.
  // Hashtable is very slow. Using bloom filter can significantly speed up lookups.
  std::bitset<kFilterSize> filter_;
  // The hashmap from old vid to new vid
  std::unordered_map<dgl_id_t, dgl_id_t> oldv2newv_;
};
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
  const dgl_id_t* indptr_data = static_cast<dgl_id_t*>(indptr_->data);
  const dgl_id_t* indices_data = static_cast<dgl_id_t*>(indices_->data);
  for (dgl_id_t i = indptr_data[src]; i < indptr_data[src+1]; ++i) {
    if (indices_data[i] == dst) {
      return true;
    }
  }
  return false;
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
  std::vector<dgl_id_t> ids;
  const dgl_id_t* indptr_data = static_cast<dgl_id_t*>(indptr_->data);
  const dgl_id_t* indices_data = static_cast<dgl_id_t*>(indices_->data);
  const dgl_id_t* eid_data = static_cast<dgl_id_t*>(edge_ids_->data);
  for (dgl_id_t i = indptr_data[src]; i < indptr_data[src+1]; ++i) {
    if (indices_data[i] == dst) {
      ids.push_back(eid_data[i]);
    }
  }
  return VecToIdArray(ids);
}

CSR::EdgeArray CSR::EdgeIds(IdArray src_ids, IdArray dst_ids) const {
  // TODO(minjie): more efficient implementation for simple graph
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

  const dgl_id_t* indptr_data = static_cast<dgl_id_t*>(indptr_->data);
  const dgl_id_t* indices_data = static_cast<dgl_id_t*>(indices_->data);
  const dgl_id_t* eid_data = static_cast<dgl_id_t*>(edge_ids_->data);

  std::vector<dgl_id_t> src, dst, eid;

  for (int64_t i = 0, j = 0; i < srclen && j < dstlen; i += src_stride, j += dst_stride) {
    const dgl_id_t src_id = src_data[i], dst_id = dst_data[j];
    CHECK(HasVertex(src_id) && HasVertex(dst_id)) <<
        "invalid edge: " << src_id << " -> " << dst_id;
    for (dgl_id_t i = indptr_data[src_id]; i < indptr_data[src_id+1]; ++i) {
      if (indices_data[i] == dst_id) {
          src.push_back(src_id);
          dst.push_back(dst_id);
          eid.push_back(eid_data[i]);
      }
    }
  }
  return CSR::EdgeArray{VecToIdArray(src), VecToIdArray(dst), VecToIdArray(eid)};
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
  CHECK(IsValidIdArray(vids)) << "Invalid vertex id array.";
  IdHashMap hashmap(vids);
  const dgl_id_t* vid_data = static_cast<dgl_id_t*>(vids->data);
  const int64_t len = vids->shape[0];
  // check if vid_data is sorted.
  //CHECK(std::is_sorted(vid_data, vid_data + len)) << "The input vertex list has to be sorted";

  const dgl_id_t* indptr_data = static_cast<dgl_id_t*>(indptr_->data);
  const dgl_id_t* indices_data = static_cast<dgl_id_t*>(indices_->data);
  const dgl_id_t* eid_data = static_cast<dgl_id_t*>(edge_ids_->data);

  std::vector<dgl_id_t> sub_indptr, sub_indices, sub_eids, induced_edges;
  sub_indptr.resize(len + 1, 0);
  for (int64_t i = 0; i < len; ++i) {
    // NOTE: newv == i
    const dgl_id_t oldv = vid_data[i];
    CHECK(HasVertex(oldv)) << "Invalid vertex: " << oldv;
    for (dgl_id_t olde = indptr_data[oldv]; olde < indptr_data[oldv+1]; ++olde) {
      const dgl_id_t oldu = indices_data[olde];
      if (hashmap.Contains(oldu)) {
        const dgl_id_t newu = hashmap.Map(oldu);
        ++sub_indptr[i];
        sub_indices.push_back(newu);
        induced_edges.push_back(eid_data[olde]);
      }
    }
  }
  sub_eids.resize(sub_indices.size());
  std::iota(sub_eids.begin(), sub_eids.end(), 0);

  // cumsum sub_indptr
  for (int64_t i = 0, cumsum = 0; i < len; ++i) {
    const dgl_id_t temp = sub_indptr[i];
    sub_indptr[i] = cumsum;
    cumsum += temp;
  }
  sub_indptr[len] = sub_indices.size();

  CSRPtr subcsr(new CSR(
        VecToIdArray(sub_indptr), VecToIdArray(sub_indices), VecToIdArray(sub_eids)));
  return Subgraph{subcsr, vids, VecToIdArray(induced_edges)};
}

// complexity: time O(E + V), space O(1)
CSRPtr CSR::Transpose() const {
  const int64_t N = NumVertices();
  const int64_t M = NumEdges();
  const dgl_id_t* Ap = static_cast<dgl_id_t*>(indptr_->data);
  const dgl_id_t* Aj = static_cast<dgl_id_t*>(indices_->data);
  const dgl_id_t* Ax = static_cast<dgl_id_t*>(edge_ids_->data);
  IdArray ret_indptr = NewIdArray(N + 1);
  IdArray ret_indices = NewIdArray(M);
  IdArray ret_edge_ids = NewIdArray(M);
  dgl_id_t* Bp = static_cast<dgl_id_t*>(ret_indptr->data);
  dgl_id_t* Bi = static_cast<dgl_id_t*>(ret_indices->data);
  dgl_id_t* Bx = static_cast<dgl_id_t*>(ret_edge_ids->data);

  std::fill(Bp, Bp + N, 0);

  for (int64_t j = 0; j < M; ++j) {
    Bp[Aj[j]]++;
  }

  // cumsum
  for (int64_t i = 0, cumsum = 0; i < N; ++i) {
    const dgl_id_t temp = Bp[i];
    Bp[i] = cumsum;
    cumsum += temp;
  }
  Bp[N] = M;

  for (int64_t i = 0; i < N; ++i) {
    for (int64_t j = Ap[i]; j < Ap[i+1]; ++j) {
      const dgl_id_t dst = Aj[j];
      Bi[Bp[dst]] = i;
      Bx[Bp[dst]] = Ax[j];
      Bp[dst]++;
    }
  }

  // correct the indptr
  for (int64_t i = 0, last = 0; i <= N; ++i) {
    dgl_id_t temp = Bp[i];
    Bp[i] = last;
    last = temp;
  }

  return CSRPtr(new CSR(ret_indptr, ret_indices, ret_edge_ids));
}

// complexity: time O(E + V), space O(1)
COOPtr CSR::ToCOO() const {
  const dgl_id_t* indptr_data = static_cast<dgl_id_t*>(indptr_->data);
  const dgl_id_t* indices_data = static_cast<dgl_id_t*>(indices_->data);
  const dgl_id_t* eid_data = static_cast<dgl_id_t*>(edge_ids_->data);
  IdArray ret_src = NewIdArray(NumEdges());
  IdArray ret_dst = NewIdArray(NumEdges());
  dgl_id_t* ret_src_data = static_cast<dgl_id_t*>(ret_src->data);
  dgl_id_t* ret_dst_data = static_cast<dgl_id_t*>(ret_dst->data);
  // scatter by edge id
  for (dgl_id_t src = 0; src < NumVertices(); ++src) {
    for (dgl_id_t eid = indptr_data[src]; eid < indptr_data[src + 1]; ++eid) {
      const dgl_id_t dst = indices_data[eid];
      ret_src_data[eid_data[eid]] = src;
      ret_dst_data[eid_data[eid]] = dst;
    }
  }
  return COOPtr(new COO(NumVertices(), ret_src, ret_dst));
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
  IdArray rst_src = NewIdArray(len);
  IdArray rst_dst = NewIdArray(len);
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
  CHECK(IsValidIdArray(eids));
  const dgl_id_t* src_data = static_cast<dgl_id_t*>(src_->data);
  const dgl_id_t* dst_data = static_cast<dgl_id_t*>(dst_->data);
  const dgl_id_t* eids_data = static_cast<dgl_id_t*>(eids->data);
  IdArray new_src = NewIdArray(eids->shape[0]);
  IdArray new_dst = NewIdArray(eids->shape[0]);
  dgl_id_t* new_src_data = static_cast<dgl_id_t*>(new_src->data);
  dgl_id_t* new_dst_data = static_cast<dgl_id_t*>(new_dst->data);
  dgl_id_t newid = 0;
  std::unordered_map<dgl_id_t, dgl_id_t> oldv2newv;

  for (dgl_id_t i = 0; i < eids->shape[0]; ++i) {
    const dgl_id_t eid = eids_data[i];
    const dgl_id_t src = src_data[eid];
    const dgl_id_t dst = dst_data[eid];
    if (!oldv2newv.count(src)) {
      oldv2newv[src] = newid++;
    }
    if (!oldv2newv.count(dst)) {
      oldv2newv[dst] = newid++;
    }
    *(new_src_data++) = oldv2newv[src];
    *(new_dst_data++) = oldv2newv[dst];
  }

  // induced nodes
  IdArray induced_nodes = NewIdArray(newid);
  dgl_id_t* induced_nodes_data = static_cast<dgl_id_t*>(induced_nodes->data);
  for (const auto& kv : oldv2newv) {
    induced_nodes_data[kv.second] = kv.first;
  }

  COOPtr subcoo(new COO(newid, new_src, new_dst));
  return Subgraph{subcoo, induced_nodes, eids};
}

// complexity: time O(E + V), space O(1)
CSRPtr COO::ToCSR() const {
  const int64_t N = num_vertices_;
  const int64_t M = src_->shape[0];
  const dgl_id_t* src_data = static_cast<dgl_id_t*>(src_->data);
  const dgl_id_t* dst_data = static_cast<dgl_id_t*>(dst_->data);
  IdArray indptr = NewIdArray(N + 1);
  IdArray indices = NewIdArray(M);
  IdArray edge_ids = NewIdArray(M);

  dgl_id_t* Bp = static_cast<dgl_id_t*>(indptr->data);
  dgl_id_t* Bi = static_cast<dgl_id_t*>(indices->data);
  dgl_id_t* Bx = static_cast<dgl_id_t*>(edge_ids->data);

  std::fill(Bp, Bp + N, 0);

  for (int64_t i = 0; i < M; ++i) {
    Bp[src_data[i]]++;
  }

  // cumsum
  for (int64_t i = 0, cumsum = 0; i < N; ++i) {
    const dgl_id_t temp = Bp[i];
    Bp[i] = cumsum;
    cumsum += temp;
  }
  Bp[N] = M;

  for (int64_t i = 0; i < M; ++i) {
    const dgl_id_t src = src_data[i];
    const dgl_id_t dst = dst_data[i];
    Bi[Bp[src]] = dst;
    Bx[Bp[src]] = i;
    Bp[src]++;
  }

  // correct the indptr
  for (int64_t i = 0, last = 0; i <= N; ++i) {
    dgl_id_t temp = Bp[i];
    Bp[i] = last;
    last = temp;
  }

  return CSRPtr(new CSR(indptr, indices, edge_ids));
}

//////////////////////////////////////////////////////////
//
// immutable graph implementation
//
//////////////////////////////////////////////////////////

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
  return {};
}

Subgraph ImmutableGraph::VertexSubgraph(IdArray vids) const {
  // We prefer to generate a subgraph from out-csr.
  auto sg = GetOutCSR()->VertexSubgraph(vids);
  CSRPtr subcsr = std::dynamic_pointer_cast<CSR>(sg.graph);
  return Subgraph{GraphPtr(new ImmutableGraph(subcsr)),
                  sg.induced_vertices, sg.induced_edges};
}

Subgraph ImmutableGraph::EdgeSubgraph(IdArray eids) const {
  // We prefer to generate a subgraph from out-csr.
  auto sg = GetCOO()->EdgeSubgraph(eids);
  COOPtr subcoo = std::dynamic_pointer_cast<COO>(sg.graph);
  return Subgraph{GraphPtr(new ImmutableGraph(subcoo)),
                  sg.induced_vertices, sg.induced_edges};
}

std::vector<IdArray> ImmutableGraph::GetAdj(bool transpose, const std::string &fmt) const {
  if (fmt == std::string("csr")) {
    return transpose? GetInCSR()->GetAdj(false, "csr") : GetOutCSR()->GetAdj(false, "csr");
  } else if (fmt == std::string("coo")) {
    return GetCOO()->GetAdj(transpose, fmt);
  } else {
    LOG(FATAL) << "unsupported adjacency matrix format: " << fmt;
    return {};
  }
}

}  // namespace dgl
