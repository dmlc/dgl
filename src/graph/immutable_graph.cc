/*!
 *  Copyright (c) 2018 by Contributors
 * \file graph/immutable_graph.cc
 * \brief DGL immutable graph index implementation
 */

#include <dgl/immutable_graph.h>
#include <string.h>
#include <bitset>
#include <numeric>
#include <tuple>

#include "../c_api_common.h"

namespace dgl {
namespace {
std::tuple<IdArray, IdArray, IdArray> MapFromSharedMemory(
  const std::string &shared_mem_name, int64_t num_verts, int64_t num_edges, bool is_create) {
#ifndef _WIN32
  const int64_t file_size = (num_verts + 1 + num_edges * 2) * sizeof(dgl_id_t);

  IdArray sm_array = IdArray::EmptyShared(
      shared_mem_name, {file_size}, DLDataType{kDLInt, 8, 1}, DLContext{kDLCPU, 0}, is_create);
  // Create views from the shared memory array. Note that we don't need to save
  //   the sm_array because the refcount is maintained by the view arrays.
  IdArray indptr = sm_array.CreateView({num_verts + 1}, DLDataType{kDLInt, 64, 1});
  IdArray indices = sm_array.CreateView({num_edges}, DLDataType{kDLInt, 64, 1},
      (num_verts + 1) * sizeof(dgl_id_t));
  IdArray edge_ids = sm_array.CreateView({num_edges}, DLDataType{kDLInt, 64, 1},
      (num_verts + 1 + num_edges) * sizeof(dgl_id_t));
  return std::make_tuple(indptr, indices, edge_ids);
#else
  LOG(FATAL) << "CSR graph doesn't support shared memory in Windows yet";
  return {};
#endif  // _WIN32
}
}  // namespace

//////////////////////////////////////////////////////////
//
// CSR graph implementation
//
//////////////////////////////////////////////////////////

CSR::CSR(int64_t num_vertices, int64_t num_edges, bool is_multigraph)
  : is_multigraph_(is_multigraph) {
  adj_.indptr = aten::NewIdArray(num_vertices + 1);
  adj_.indices = aten::NewIdArray(num_edges);
  adj_.data = aten::NewIdArray(num_edges);
}

CSR::CSR(IdArray indptr, IdArray indices, IdArray edge_ids) {
  CHECK(IsValidIdArray(indptr));
  CHECK(IsValidIdArray(indices));
  CHECK(IsValidIdArray(edge_ids));
  CHECK_EQ(indices->shape[0], edge_ids->shape[0]);
  const int64_t N = indptr->shape[0];
  adj_ = aten::CSRMatrix{N, N, indptr, indices, edge_ids};
}

CSR::CSR(IdArray indptr, IdArray indices, IdArray edge_ids, bool is_multigraph)
  : is_multigraph_(is_multigraph) {
  CHECK(IsValidIdArray(indptr));
  CHECK(IsValidIdArray(indices));
  CHECK(IsValidIdArray(edge_ids));
  CHECK_EQ(indices->shape[0], edge_ids->shape[0]);
  const int64_t N = indptr->shape[0];
  adj_ = aten::CSRMatrix{N, N, indptr, indices, edge_ids};
}

CSR::CSR(IdArray indptr, IdArray indices, IdArray edge_ids,
         const std::string &shared_mem_name): shared_mem_name_(shared_mem_name) {
  CHECK(IsValidIdArray(indptr));
  CHECK(IsValidIdArray(indices));
  CHECK(IsValidIdArray(edge_ids));
  CHECK_EQ(indices->shape[0], edge_ids->shape[0]);
  const int64_t num_verts = indptr->shape[0] - 1;
  const int64_t num_edges = indices->shape[0];
  adj_.num_rows = num_verts;
  adj_.num_cols = num_verts;
  std::tie(adj_.indptr, adj_.indices, adj_.data) = MapFromSharedMemory(
      shared_mem_name, num_verts, num_edges, true);
  // copy the given data into the shared memory arrays
  adj_.indptr.CopyFrom(indptr);
  adj_.indices.CopyFrom(indices);
  adj_.data.CopyFrom(edge_ids);
}

CSR::CSR(IdArray indptr, IdArray indices, IdArray edge_ids, bool is_multigraph,
         const std::string &shared_mem_name): is_multigraph_(is_multigraph),
         shared_mem_name_(shared_mem_name) {
  CHECK(IsValidIdArray(indptr));
  CHECK(IsValidIdArray(indices));
  CHECK(IsValidIdArray(edge_ids));
  CHECK_EQ(indices->shape[0], edge_ids->shape[0]);
  const int64_t num_verts = indptr->shape[0] - 1;
  const int64_t num_edges = indices->shape[0];
  adj_.num_rows = num_verts;
  adj_.num_cols = num_verts;
  std::tie(adj_.indptr, adj_.indices, adj_.data) = MapFromSharedMemory(
      shared_mem_name, num_verts, num_edges, true);
  // copy the given data into the shared memory arrays
  adj_.indptr.CopyFrom(indptr);
  adj_.indices.CopyFrom(indices);
  adj_.data.CopyFrom(edge_ids);
}

CSR::CSR(const std::string &shared_mem_name,
         int64_t num_verts, int64_t num_edges, bool is_multigraph)
  : is_multigraph_(is_multigraph), shared_mem_name_(shared_mem_name) {
  adj_.num_rows = num_verts;
  adj_.num_cols = num_verts;
  std::tie(adj_.indptr, adj_.indices, adj_.data) = MapFromSharedMemory(
      shared_mem_name, num_verts, num_edges, false);
}

bool CSR::IsMultigraph() const {
  // The lambda will be called the first time to initialize the is_multigraph flag.
  return const_cast<CSR*>(this)->is_multigraph_.Get([this] () {
      return aten::CSRHasDuplicate(adj_);
      /*
      const dgl_id_t* indptr_data = static_cast<dgl_id_t*>(indptr_->data);
      const dgl_id_t* indices_data = static_cast<dgl_id_t*>(indices_->data);
      for (dgl_id_t src = 0; src < NumVertices(); ++src) {
        std::unordered_set<dgl_id_t> hashmap;
        for (dgl_id_t eid = indptr_data[src]; eid < indptr_data[src+1]; ++eid) {
          const dgl_id_t dst = indices_data[eid];
          if (hashmap.count(dst)) {
            return true;
          } else {
            hashmap.insert(dst);
          }
        }
      }
      return false;
      */
    });
}

CSR::EdgeArray CSR::OutEdges(dgl_id_t vid) const {
  CHECK(HasVertex(vid)) << "invalid vertex: " << vid;
  IdArray ret_dst = aten::CSRGetRowColumnIndices(adj_, vid);
  IdArray ret_eid = aten::CSRGetRowData(adj_, vid);
  IdArray ret_src = aten::Full(vid, ret_dst->shape[0], NumBits(), ret_dst->ctx);
  return CSR::EdgeArray{ret_src, ret_dst, ret_eid};
}

CSR::EdgeArray CSR::OutEdges(IdArray vids) const {
  CHECK(IsValidIdArray(vids)) << "Invalid vertex id array.";
  auto csrsubmat = aten::CSRSliceRows(adj_, vids);
  auto coosubmat = aten::CSRToCOO(csrsubmat, false);
  return CSR::EdgeArray{coosubmat.row, coosubmat.col, coosubmat.data};

  /*
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
  return CSR::EdgeArray{src, dst, eid};
  */
}

DegreeArray CSR::OutDegrees(IdArray vids) const {
  CHECK(IsValidIdArray(vids)) << "Invalid vertex id array.";
  return aten::CSRGetRowNNZ(adj_, vids);
  /*
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
  */
}

bool CSR::HasEdgeBetween(dgl_id_t src, dgl_id_t dst) const {
  CHECK(HasVertex(src)) << "Invalid vertex id: " << src;
  CHECK(HasVertex(dst)) << "Invalid vertex id: " << dst;
  return aten::CSRIsNonZero(adj_, src, dst);
}

IdArray CSR::Successors(dgl_id_t vid, uint64_t radius) const {
  CHECK(HasVertex(vid)) << "invalid vertex: " << vid;
  CHECK(radius == 1) << "invalid radius: " << radius;
  return aten::CSRGetRowColumnIndices(adj_, vid);
}

IdArray CSR::EdgeId(dgl_id_t src, dgl_id_t dst) const {
  CHECK(HasVertex(src)) << "invalid vertex: " << src;
  CHECK(HasVertex(dst)) << "invalid vertex: " << dst;
  return aten::CSRGetData(adj_, src, dst);
}

CSR::EdgeArray CSR::EdgeIds(IdArray src_ids, IdArray dst_ids) const {
  const auto& arrs = aten::CSRGetDataAndIndices(adj_, src_ids, dst_ids);
  return CSR::EdgeArray{arrs[0], arrs[1], arrs[2]};
}

CSR::EdgeArray CSR::Edges(const std::string &order) const {
  CHECK(order.empty() || order == std::string("srcdst"))
    << "CSR only support Edges of order \"srcdst\","
    << " but got \"" << order << "\".";
  const auto& coo = aten::CSRToCOO(adj_, false);
  return CSR::EdgeArray{coo.row, coo.col, coo.data};
}

Subgraph CSR::VertexSubgraph(IdArray vids) const {
  CHECK(IsValidIdArray(vids)) << "Invalid vertex id array.";
  const auto& submat = aten::CSRSliceMatrix(adj_, vids, vids);
  IdArray sub_eids = aten::Range(0, submat.data->shape[0], NumBits(), Context());
  CSRPtr subcsr(new CSR(submat.indptr, submat.indices, sub_eids));
  return Subgraph{subcsr, vids, submat.data};
}

CSRPtr CSR::Transpose() const {
  const auto& trans = aten::CSRTranspose(adj_);
  return CSRPtr(new CSR(trans.indptr, trans.indices, trans.data));
}

COOPtr CSR::ToCOO() const {
  const auto& coo = aten::CSRToCOO(adj_, true);
  return COOPtr(new COO(NumVertices(), coo.row, coo.col));
}

CSR CSR::CopyTo(const DLContext& ctx) const {
  if (Context() == ctx) {
    return *this;
  } else {
    CSR ret(adj_.indptr.CopyTo(ctx),
            adj_.indices.CopyTo(ctx),
            adj_.data.CopyTo(ctx));
    ret.is_multigraph_ = is_multigraph_;
    return ret;
  }
}

CSR CSR::CopyToSharedMem(const std::string &name) const {
  if (IsSharedMem()) {
    CHECK(name == shared_mem_name_);
    return *this;
  } else {
    return CSR(adj_.indptr, adj_.indices, adj_.data, name);
  }
}

CSR CSR::AsNumBits(uint8_t bits) const {
  if (NumBits() == bits) {
    return *this;
  } else {
    CSR ret(aten::AsNumBits(adj_.indptr, bits),
            aten::AsNumBits(adj_.indices, bits),
            aten::AsNumBits(adj_.data, bits));
    ret.is_multigraph_ = is_multigraph_;
    return ret;
  }
}

DGLIdIters CSR::SuccVec(dgl_id_t vid) const {
  // TODO(minjie): This still assumes the data type and device context
  //   of this graph. Should fix later.
  const dgl_id_t* indptr_data = static_cast<dgl_id_t*>(adj_.indptr->data);
  const dgl_id_t* indices_data = static_cast<dgl_id_t*>(adj_.indices->data);
  const dgl_id_t start = indptr_data[vid];
  const dgl_id_t end = indptr_data[vid + 1];
  return DGLIdIters(indices_data + start, indices_data + end);
}

DGLIdIters CSR::OutEdgeVec(dgl_id_t vid) const {
  // TODO(minjie): This still assumes the data type and device context
  //   of this graph. Should fix later.
  const dgl_id_t* indptr_data = static_cast<dgl_id_t*>(adj_.indptr->data);
  const dgl_id_t* eid_data = static_cast<dgl_id_t*>(adj_.data->data);
  const dgl_id_t start = indptr_data[vid];
  const dgl_id_t end = indptr_data[vid + 1];
  return DGLIdIters(eid_data + start, eid_data + end);
}

//////////////////////////////////////////////////////////
//
// COO graph implementation
//
//////////////////////////////////////////////////////////
COO::COO(int64_t num_vertices, IdArray src, IdArray dst)
  : num_vertices_(num_vertices), src_(src), dst_(dst) {
  CHECK(IsValidIdArray(src));
  CHECK(IsValidIdArray(dst));
  CHECK_EQ(src->shape[0], dst->shape[0]);
}

COO::COO(int64_t num_vertices, IdArray src, IdArray dst, bool is_multigraph)
  : num_vertices_(num_vertices), src_(src), dst_(dst), is_multigraph_(is_multigraph) {
  CHECK(IsValidIdArray(src));
  CHECK(IsValidIdArray(dst));
  CHECK_EQ(src->shape[0], dst->shape[0]);
}

bool COO::IsMultigraph() const {
  // The lambda will be called the first time to initialize the is_multigraph flag.
  return const_cast<COO*>(this)->is_multigraph_.Get([this] () {
      std::unordered_set<std::pair<dgl_id_t, dgl_id_t>, PairHash> hashmap;
      const dgl_id_t* src_data = static_cast<dgl_id_t*>(src_->data);
      const dgl_id_t* dst_data = static_cast<dgl_id_t*>(dst_->data);
      for (dgl_id_t eid = 0; eid < NumEdges(); ++eid) {
        const auto& p = std::make_pair(src_data[eid], dst_data[eid]);
        if (hashmap.count(p)) {
          return true;
        } else {
          hashmap.insert(p);
        }
      }
      return false;
    });
}

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
  std::iota(rst_eid_data, rst_eid_data + rstlen, 0);
  return EdgeArray{src_, dst_, rst_eid};
}

Subgraph COO::EdgeSubgraph(IdArray eids, bool preserve_nodes) const {
  CHECK(IsValidIdArray(eids)) << "Invalid edge id array.";
  const dgl_id_t* src_data = static_cast<dgl_id_t*>(src_->data);
  const dgl_id_t* dst_data = static_cast<dgl_id_t*>(dst_->data);
  const dgl_id_t* eids_data = static_cast<dgl_id_t*>(eids->data);
  IdArray new_src = NewIdArray(eids->shape[0]);
  IdArray new_dst = NewIdArray(eids->shape[0]);
  dgl_id_t* new_src_data = static_cast<dgl_id_t*>(new_src->data);
  dgl_id_t* new_dst_data = static_cast<dgl_id_t*>(new_dst->data);
  if (!preserve_nodes) {
    dgl_id_t newid = 0;
    std::unordered_map<dgl_id_t, dgl_id_t> oldv2newv;

    for (int64_t i = 0; i < eids->shape[0]; ++i) {
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
  } else {
    for (int64_t i = 0; i < eids->shape[0]; ++i) {
      const dgl_id_t eid = eids_data[i];
      const dgl_id_t src = src_data[eid];
      const dgl_id_t dst = dst_data[eid];
      *(new_src_data++) = src;
      *(new_dst_data++) = dst;
    }

    IdArray induced_nodes = NewIdArray(NumVertices());
    dgl_id_t* induced_nodes_data = static_cast<dgl_id_t*>(induced_nodes->data);
    for (int64_t i = 0; i < NumVertices(); ++i)
      *(induced_nodes_data++) = i;

    COOPtr subcoo(new COO(NumVertices(), new_src, new_dst));
    return Subgraph{subcoo, induced_nodes, eids};
  }
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

COO COO::CopyTo(const DLContext& ctx) const {
  if (Context() == ctx) {
    return *this;
  } else {
    // TODO(minjie): change to use constructor later
    COO ret;
    ret.num_vertices_ = num_vertices_;
    ret.src_ = src_.CopyTo(ctx);
    ret.dst_ = dst_.CopyTo(ctx);
    ret.is_multigraph_ = is_multigraph_;
    return ret;
  }
}

COO COO::CopyToSharedMem(const std::string &name) const {
  LOG(FATAL) << "COO doesn't supprt shared memory yet";
}

COO COO::AsNumBits(uint8_t bits) const {
  if (NumBits() == bits) {
    return *this;
  } else {
    // TODO(minjie): change to use constructor later
    COO ret;
    ret.num_vertices_ = num_vertices_;
    ret.src_ = dgl::AsNumBits(src_, bits);
    ret.dst_ = dgl::AsNumBits(dst_, bits);
    ret.is_multigraph_ = is_multigraph_;
    return ret;
  }
}

//////////////////////////////////////////////////////////
//
// immutable graph implementation
//
//////////////////////////////////////////////////////////

ImmutableGraph::EdgeArray ImmutableGraph::Edges(const std::string &order) const {
  if (order.empty()) {
    // arbitrary order
    if (in_csr_) {
      // transpose
      const auto& edges = in_csr_->Edges(order);
      return EdgeArray{edges.dst, edges.src, edges.id};
    } else {
      return AnyGraph()->Edges(order);
    }
  } else if (order == std::string("srcdst")) {
    // TODO(minjie): CSR only guarantees "src" to be sorted.
    //   Maybe we should relax this requirement?
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

Subgraph ImmutableGraph::EdgeSubgraph(IdArray eids, bool preserve_nodes) const {
  // We prefer to generate a subgraph from out-csr.
  auto sg = GetCOO()->EdgeSubgraph(eids, preserve_nodes);
  COOPtr subcoo = std::dynamic_pointer_cast<COO>(sg.graph);
  return Subgraph{GraphPtr(new ImmutableGraph(subcoo)),
                  sg.induced_vertices, sg.induced_edges};
}

std::vector<IdArray> ImmutableGraph::GetAdj(bool transpose, const std::string &fmt) const {
  // TODO(minjie): Our current semantics of adjacency matrix is row for dst nodes and col for
  //   src nodes. Therefore, we need to flip the transpose flag. For example, transpose=False
  //   is equal to in edge CSR.
  //   We have this behavior because previously we use framework's SPMM and we don't cache
  //   reverse adj. This is not intuitive and also not consistent with networkx's
  //   to_scipy_sparse_matrix. With the upcoming custom kernel change, we should change the
  //   behavior and make row for src and col for dst.
  if (fmt == std::string("csr")) {
    return transpose? GetOutCSR()->GetAdj(false, "csr") : GetInCSR()->GetAdj(false, "csr");
  } else if (fmt == std::string("coo")) {
    return GetCOO()->GetAdj(!transpose, fmt);
  } else {
    LOG(FATAL) << "unsupported adjacency matrix format: " << fmt;
    return {};
  }
}

ImmutableGraph ImmutableGraph::ToImmutable(const GraphInterface* graph) {
  const ImmutableGraph* ig = dynamic_cast<const ImmutableGraph*>(graph);
  if (ig) {
    return *ig;
  } else {
    const auto& adj = graph->GetAdj(true, "csr");
    CSRPtr csr(new CSR(adj[0], adj[1], adj[2]));
    return ImmutableGraph(nullptr, csr);
  }
}

ImmutableGraph ImmutableGraph::CopyTo(const DLContext& ctx) const {
  if (ctx == Context()) {
    return *this;
  }
  // TODO(minjie): since we don't have GPU implementation of COO<->CSR,
  //   we make sure that this graph (on CPU) has materialized CSR,
  //   and then copy them to other context (usually GPU). This should
  //   be fixed later.
  CSRPtr new_incsr = CSRPtr(new CSR(GetInCSR()->CopyTo(ctx)));
  CSRPtr new_outcsr = CSRPtr(new CSR(GetOutCSR()->CopyTo(ctx)));
  return ImmutableGraph(new_incsr, new_outcsr);
}

ImmutableGraph ImmutableGraph::CopyToSharedMem(const std::string &edge_dir,
                                               const std::string &name) const {
  CSRPtr new_incsr, new_outcsr;
  std::string shared_mem_name = GetSharedMemName(name, edge_dir);
  if (edge_dir == std::string("in"))
    new_incsr = CSRPtr(new CSR(GetInCSR()->CopyToSharedMem(shared_mem_name)));
  else if (edge_dir == std::string("out"))
    new_outcsr = CSRPtr(new CSR(GetOutCSR()->CopyToSharedMem(shared_mem_name)));
  return ImmutableGraph(new_incsr, new_outcsr, name);
}

ImmutableGraph ImmutableGraph::AsNumBits(uint8_t bits) const {
  if (NumBits() == bits) {
    return *this;
  } else {
    // TODO(minjie): since we don't have int32 operations,
    //   we make sure that this graph (on CPU) has materialized CSR,
    //   and then copy them to other context (usually GPU). This should
    //   be fixed later.
    CSRPtr new_incsr = CSRPtr(new CSR(GetInCSR()->AsNumBits(bits)));
    CSRPtr new_outcsr = CSRPtr(new CSR(GetOutCSR()->AsNumBits(bits)));
    return ImmutableGraph(new_incsr, new_outcsr);
  }
}

}  // namespace dgl
