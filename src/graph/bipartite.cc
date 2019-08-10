/*!
 *  Copyright (c) 2019 by Contributors
 * \file graph/bipartite.cc
 * \brief Bipartite graph implementation
 */
#include <dgl/array.h>
#include <dgl/lazy.h>
#include <dgl/immutable_graph.h>
#include <dgl/base_heterograph.h>

#include "./bipartite.h"
#include "../c_api_common.h"

namespace dgl {

namespace {

inline GraphPtr CreateBipartiteMetaGraph() {
  std::vector<int64_t> row_vec(1, Bipartite::kSrcVType);
  std::vector<int64_t> col_vec(1, Bipartite::kDstVType);
  IdArray row = aten::VecToIdArray(row_vec);
  IdArray col = aten::VecToIdArray(col_vec);
  GraphPtr g = ImmutableGraph::CreateFromCOO(2, row, col);
  return g;
}
const GraphPtr kBipartiteMetaGraph = CreateBipartiteMetaGraph();

};  // namespace

//////////////////////////////////////////////////////////
//
// COO graph implementation
//
//////////////////////////////////////////////////////////

Bipartite::COO::COO(
    int64_t num_src,
    int64_t num_dst,
    IdArray src,
    IdArray dst)
  : BaseHeteroGraph(kBipartiteMetaGraph) {
  adj_ = aten::COOMatrix{num_src, num_dst, src, dst};
}

Bipartite::COO::COO(
    int64_t num_src,
    int64_t num_dst,
    IdArray src,
    IdArray dst,
    bool is_multigraph)
  : BaseHeteroGraph(kBipartiteMetaGraph),
    is_multigraph_(is_multigraph) {
  adj_ = aten::COOMatrix{num_src, num_dst, src, dst};
}

Bipartite::COO::COO(const aten::COOMatrix& coo)
  : BaseHeteroGraph(kBipartiteMetaGraph), adj_(coo) {
}

HeteroSubgraph Bipartite::COO::EdgeSubgraph(
    const std::vector<IdArray>& eids, bool preserve_nodes) const {
  CHECK_EQ(eids.size(), 1) << "Edge type number mismatch.";
  HeteroSubgraph subg;
  if (!preserve_nodes) {
    IdArray new_src = aten::IndexSelect(adj_.row, eids[0]);
    IdArray new_dst = aten::IndexSelect(adj_.col, eids[0]);
    subg.induced_vertices.emplace_back(aten::Relabel_({new_src}));
    subg.induced_vertices.emplace_back(aten::Relabel_({new_dst}));
    const auto new_nsrc = subg.induced_vertices[0]->shape[0];
    const auto new_ndst = subg.induced_vertices[1]->shape[0];
    subg.graph = std::make_shared<COO>(
        new_nsrc, new_ndst, new_src, new_dst);
    subg.induced_edges = eids;
  } else {
    IdArray new_src = aten::IndexSelect(adj_.row, eids[0]);
    IdArray new_dst = aten::IndexSelect(adj_.col, eids[0]);
    subg.induced_vertices.emplace_back(aten::Range(0, NumVertices(0), NumBits(), Context()));
    subg.induced_vertices.emplace_back(aten::Range(0, NumVertices(1), NumBits(), Context()));
    subg.graph = std::make_shared<COO>(
        NumVertices(0), NumVertices(1), new_src, new_dst);
    subg.induced_edges = eids;
  }
  return subg;
}

//////////////////////////////////////////////////////////
//
// CSR graph implementation
//
//////////////////////////////////////////////////////////

Bipartite::CSR::CSR(
    int64_t num_src,
    int64_t num_dst,
    IdArray indptr,
    IdArray indices,
    IdArray edge_ids)
  : BaseHeteroGraph(kBipartiteMetaGraph) {
  adj_ = aten::CSRMatrix{num_src, num_dst, indptr, indices, edge_ids};
}

Bipartite::CSR::CSR(
    int64_t num_src,
    int64_t num_dst,
    IdArray indptr,
    IdArray indices,
    IdArray edge_ids,
    bool is_multigraph)
  : BaseHeteroGraph(kBipartiteMetaGraph),
    is_multigraph_(is_multigraph) {
  adj_ = aten::CSRMatrix{num_src, num_dst, indptr, indices, edge_ids};
}

Bipartite::CSR::CSR(const aten::CSRMatrix& csr)
  : BaseHeteroGraph(kBipartiteMetaGraph), adj_(csr) {
}

//////////////////////////////////////////////////////////
//
// bipartite graph implementation
//
//////////////////////////////////////////////////////////

DLContext Bipartite::Context() const {
  return GetAny()->Context();
}

uint8_t Bipartite::NumBits() const {
  return GetAny()->NumBits();
}

bool Bipartite::IsMultigraph() const {
  return GetAny()->IsMultigraph();
}

uint64_t Bipartite::NumVertices(dgl_type_t vtype) const {
  return GetAny()->NumVertices(vtype);
}

uint64_t Bipartite::NumEdges(dgl_type_t etype) const {
  return GetAny()->NumEdges(etype);
}

bool Bipartite::HasVertex(dgl_type_t vtype, dgl_id_t vid) const {
  return GetAny()->HasVertex(vtype, vid);
}

BoolArray Bipartite::HasVertices(dgl_type_t vtype, IdArray vids) const {
  CHECK(aten::IsValidIdArray(vids)) << "Invalid id array input";
  return aten::LT(vids, NumVertices(vtype));
}

bool Bipartite::HasEdgeBetween(dgl_type_t etype, dgl_id_t src, dgl_id_t dst) const {
  if (in_csr_) {
    return in_csr_->HasEdgeBetween(etype, dst, src);
  } else {
    return GetOutCSR()->HasEdgeBetween(etype, src, dst);
  }
}

BoolArray Bipartite::HasEdgesBetween(
    dgl_type_t etype, IdArray src, IdArray dst) const {
  if (in_csr_) {
    return in_csr_->HasEdgesBetween(etype, dst, src);
  } else {
    return GetOutCSR()->HasEdgesBetween(etype, src, dst);
  }
}

IdArray Bipartite::Predecessors(dgl_type_t etype, dgl_id_t dst) const {
  return GetInCSR()->Successors(etype, dst);
}

IdArray Bipartite::Successors(dgl_type_t etype, dgl_id_t src) const {
  return GetOutCSR()->Successors(etype, src);
}

IdArray Bipartite::EdgeId(dgl_type_t etype, dgl_id_t src, dgl_id_t dst) const {
  if (in_csr_) {
    return in_csr_->EdgeId(etype, dst, src);
  } else {
    return GetOutCSR()->EdgeId(etype, src, dst);
  }
}

EdgeArray Bipartite::EdgeIds(dgl_type_t etype, IdArray src, IdArray dst) const {
  if (in_csr_) {
    EdgeArray edges = in_csr_->EdgeIds(etype, dst, src);
    return EdgeArray{edges.dst, edges.src, edges.id};
  } else {
    return GetOutCSR()->EdgeIds(etype, src, dst);
  }
}

std::pair<dgl_id_t, dgl_id_t> Bipartite::FindEdge(dgl_type_t etype, dgl_id_t eid) const {
  return GetCOO()->FindEdge(etype, eid);
}

EdgeArray Bipartite::FindEdges(dgl_type_t etype, IdArray eids) const {
  return GetCOO()->FindEdges(etype, eids);
}

EdgeArray Bipartite::InEdges(dgl_type_t etype, dgl_id_t vid) const {
  const EdgeArray& ret = GetInCSR()->OutEdges(etype, vid);
  return {ret.dst, ret.src, ret.id};
}

EdgeArray Bipartite::InEdges(dgl_type_t etype, IdArray vids) const {
  const EdgeArray& ret = GetInCSR()->OutEdges(etype, vids);
  return {ret.dst, ret.src, ret.id};
}

EdgeArray Bipartite::OutEdges(dgl_type_t etype, dgl_id_t vid) const {
  return GetOutCSR()->OutEdges(etype, vid);
}

EdgeArray Bipartite::OutEdges(dgl_type_t etype, IdArray vids) const {
  return GetOutCSR()->OutEdges(etype, vids);
}

EdgeArray Bipartite::Edges(dgl_type_t etype, const std::string &order) const {
  if (order.empty()) {
    // arbitrary order
    if (in_csr_) {
      // transpose
      const auto& edges = in_csr_->Edges(etype, order);
      return EdgeArray{edges.dst, edges.src, edges.id};
    } else {
      return GetAny()->Edges(etype, order);
    }
  } else if (order == std::string("srcdst")) {
    // TODO(minjie): CSR only guarantees "src" to be sorted.
    //   Maybe we should relax this requirement?
    return GetOutCSR()->Edges(etype, order);
  } else if (order == std::string("eid")) {
    return GetCOO()->Edges(etype, order);
  } else {
    LOG(FATAL) << "Unsupported order request: " << order;
  }
  return {};
}

uint64_t Bipartite::InDegree(dgl_type_t etype, dgl_id_t vid) const {
  return GetInCSR()->OutDegree(etype, vid);
}

DegreeArray Bipartite::InDegrees(dgl_type_t etype, IdArray vids) const {
  return GetInCSR()->OutDegrees(etype, vids);
}

uint64_t Bipartite::OutDegree(dgl_type_t etype, dgl_id_t vid) const {
  return GetOutCSR()->OutDegree(etype, vid);
}

DegreeArray Bipartite::OutDegrees(dgl_type_t etype, IdArray vids) const {
  return GetOutCSR()->OutDegrees(etype, vids);
}

DGLIdIters Bipartite::SuccVec(dgl_type_t etype, dgl_id_t vid) const {
  return GetOutCSR()->SuccVec(etype, vid);
}

DGLIdIters Bipartite::OutEdgeVec(dgl_type_t etype, dgl_id_t vid) const {
  return GetOutCSR()->OutEdgeVec(etype, vid);
}

DGLIdIters Bipartite::PredVec(dgl_type_t etype, dgl_id_t vid) const {
  return GetInCSR()->SuccVec(etype, vid);
}

DGLIdIters Bipartite::InEdgeVec(dgl_type_t etype, dgl_id_t vid) const {
  return GetInCSR()->OutEdgeVec(etype, vid);
}

std::vector<IdArray> Bipartite::GetAdj(
    dgl_type_t etype, bool transpose, const std::string &fmt) const {
  // TODO(minjie): Our current semantics of adjacency matrix is row for dst nodes and col for
  //   src nodes. Therefore, we need to flip the transpose flag. For example, transpose=False
  //   is equal to in edge CSR.
  //   We have this behavior because previously we use framework's SPMM and we don't cache
  //   reverse adj. This is not intuitive and also not consistent with networkx's
  //   to_scipy_sparse_matrix. With the upcoming custom kernel change, we should change the
  //   behavior and make row for src and col for dst.
  if (fmt == std::string("csr")) {
    return transpose? GetOutCSR()->GetAdj(etype, false, "csr")
      : GetInCSR()->GetAdj(etype, false, "csr");
  } else if (fmt == std::string("coo")) {
    return GetCOO()->GetAdj(etype, !transpose, fmt);
  } else {
    LOG(FATAL) << "unsupported adjacency matrix format: " << fmt;
    return {};
  }
}

HeteroSubgraph Bipartite::VertexSubgraph(const std::vector<IdArray>& vids) const {
  // We prefer to generate a subgraph from out-csr.
  auto sg = GetOutCSR()->VertexSubgraph(vids);
  CSRPtr subcsr = std::dynamic_pointer_cast<CSR>(sg.graph);
  HeteroSubgraph ret;
  ret.graph = HeteroGraphPtr(new Bipartite(nullptr, subcsr, nullptr));
  ret.induced_vertices = std::move(sg.induced_vertices);
  ret.induced_edges = std::move(sg.induced_edges);
  return ret;
}

HeteroSubgraph Bipartite::EdgeSubgraph(
    const std::vector<IdArray>& eids, bool preserve_nodes) const {
  auto sg = GetCOO()->EdgeSubgraph(eids, preserve_nodes);
  COOPtr subcoo = std::dynamic_pointer_cast<COO>(sg.graph);
  HeteroSubgraph ret;
  ret.graph = HeteroGraphPtr(new Bipartite(nullptr, nullptr, subcoo));
  ret.induced_vertices = std::move(sg.induced_vertices);
  ret.induced_edges = std::move(sg.induced_edges);
  return ret;
}

HeteroGraphPtr Bipartite::CreateFromCOO(
    int64_t num_src, int64_t num_dst,
    IdArray row, IdArray col) {
  COOPtr coo(new COO(num_src, num_dst, row, col));
  return HeteroGraphPtr(new Bipartite(nullptr, nullptr, coo));
}

HeteroGraphPtr Bipartite::CreateFromCSR(
    int64_t num_src, int64_t num_dst,
    IdArray indptr, IdArray indices, IdArray edge_ids) {
  CSRPtr csr(new CSR(num_src, num_dst, indptr, indices, edge_ids));
  return HeteroGraphPtr(new Bipartite(nullptr, csr, nullptr));
}

HeteroGraphPtr Bipartite::AsNumBits(HeteroGraphPtr g, uint8_t bits) {
  if (g->NumBits() == bits) {
    return g;
  } else {
    // TODO(minjie): since we don't have int32 operations,
    //   we make sure that this graph (on CPU) has materialized CSR,
    //   and then copy them to other context (usually GPU). This should
    //   be fixed later.
    auto bg = std::dynamic_pointer_cast<Bipartite>(g);
    CHECK_NOTNULL(bg);
    CSRPtr new_incsr = CSRPtr(new CSR(bg->GetInCSR()->AsNumBits(bits)));
    CSRPtr new_outcsr = CSRPtr(new CSR(bg->GetOutCSR()->AsNumBits(bits)));
    return HeteroGraphPtr(new Bipartite(new_incsr, new_outcsr, nullptr));
  }
}

HeteroGraphPtr Bipartite::CopyTo(HeteroGraphPtr g, const DLContext& ctx) {
  if (ctx == g->Context()) {
    return g;
  }
  // TODO(minjie): since we don't have GPU implementation of COO<->CSR,
  //   we make sure that this graph (on CPU) has materialized CSR,
  //   and then copy them to other context (usually GPU). This should
  //   be fixed later.
  auto bg = std::dynamic_pointer_cast<Bipartite>(g);
  CHECK_NOTNULL(bg);
  CSRPtr new_incsr = CSRPtr(new CSR(bg->GetInCSR()->CopyTo(ctx)));
  CSRPtr new_outcsr = CSRPtr(new CSR(bg->GetOutCSR()->CopyTo(ctx)));
  return HeteroGraphPtr(new Bipartite(new_incsr, new_outcsr, nullptr));
}

Bipartite::Bipartite(CSRPtr in_csr, CSRPtr out_csr, COOPtr coo)
  : BaseHeteroGraph(kBipartiteMetaGraph), in_csr_(in_csr), out_csr_(out_csr), coo_(coo) {
  CHECK(GetAny()) << "At least one graph structure should exist.";
}

Bipartite::CSRPtr Bipartite::GetInCSR() const {
  if (!in_csr_) {
    if (out_csr_) {
      const auto& newadj = aten::CSRTranspose(out_csr_->adj());
      const_cast<Bipartite*>(this)->in_csr_ = std::make_shared<CSR>(newadj);
    } else {
      CHECK(coo_) << "None of CSR, COO exist";
      const auto& adj = coo_->adj();
      const auto& newadj = aten::COOToCSR(
          aten::COOMatrix{adj.num_cols, adj.num_rows, adj.col, adj.row});
      const_cast<Bipartite*>(this)->in_csr_ = std::make_shared<CSR>(newadj);
    }
  }
  return in_csr_;
}

/* !\brief Return out csr. If not exist, transpose the other one.*/
Bipartite::CSRPtr Bipartite::GetOutCSR() const {
  if (!out_csr_) {
    if (in_csr_) {
      const auto& newadj = aten::CSRTranspose(in_csr_->adj());
      const_cast<Bipartite*>(this)->out_csr_ = std::make_shared<CSR>(newadj);
    } else {
      CHECK(coo_) << "None of CSR, COO exist";
      const auto& newadj = aten::COOToCSR(coo_->adj());
      const_cast<Bipartite*>(this)->out_csr_ = std::make_shared<CSR>(newadj);
    }
  }
  return out_csr_;
}

/* !\brief Return coo. If not exist, create from csr.*/
Bipartite::COOPtr Bipartite::GetCOO() const {
  if (!coo_) {
    if (in_csr_) {
      const auto& newadj = aten::CSRToCOO(in_csr_->adj(), true);
      const_cast<Bipartite*>(this)->coo_ = std::make_shared<COO>(
          aten::COOMatrix{newadj.num_cols, newadj.num_rows, newadj.col, newadj.row});
    } else {
      CHECK(out_csr_) << "Both CSR are missing.";
      const auto& newadj = aten::CSRToCOO(out_csr_->adj(), true);
      const_cast<Bipartite*>(this)->coo_ = std::make_shared<COO>(newadj);
    }
  }
  return coo_;
}

HeteroGraphPtr Bipartite::GetAny() const {
  if (in_csr_) {
    return in_csr_;
  } else if (out_csr_) {
    return out_csr_;
  } else {
    return coo_;
  }
}

}  // namespace dgl
