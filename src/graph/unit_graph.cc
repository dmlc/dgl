/*!
 *  Copyright (c) 2019 by Contributors
 * \file graph/unit_graph.cc
 * \brief UnitGraph graph implementation
 */
#include <dgl/array.h>
#include <dgl/lazy.h>
#include <dgl/immutable_graph.h>
#include <dgl/base_heterograph.h>

#include "./unit_graph.h"
#include "../c_api_common.h"

namespace dgl {

namespace {
// create metagraph of one node type
inline GraphPtr CreateUnitGraphMetaGraph1() {
  // a self-loop edge 0->0
  std::vector<int64_t> row_vec(1, 0);
  std::vector<int64_t> col_vec(1, 0);
  IdArray row = aten::VecToIdArray(row_vec);
  IdArray col = aten::VecToIdArray(col_vec);
  GraphPtr g = ImmutableGraph::CreateFromCOO(1, row, col);
  return g;
}

// create metagraph of two node types
inline GraphPtr CreateUnitGraphMetaGraph2() {
  // an edge 0->1
  std::vector<int64_t> row_vec(1, 0);
  std::vector<int64_t> col_vec(1, 1);
  IdArray row = aten::VecToIdArray(row_vec);
  IdArray col = aten::VecToIdArray(col_vec);
  GraphPtr g = ImmutableGraph::CreateFromCOO(2, row, col);
  return g;
}

inline GraphPtr CreateUnitGraphMetaGraph(int num_vtypes) {
  static GraphPtr mg1 = CreateUnitGraphMetaGraph1();
  static GraphPtr mg2 = CreateUnitGraphMetaGraph2();
  if (num_vtypes == 1)
    return mg1;
  else if (num_vtypes == 2)
    return mg2;
  else
    LOG(FATAL) << "Invalid number of vertex types. Must be 1 or 2.";
  return {};
}

};  // namespace

//////////////////////////////////////////////////////////
//
// COO graph implementation
//
//////////////////////////////////////////////////////////

class UnitGraph::COO : public BaseHeteroGraph {
 public:
  COO(GraphPtr metagraph, int64_t num_src, int64_t num_dst, IdArray src, IdArray dst)
    : BaseHeteroGraph(metagraph) {
    CHECK(aten::IsValidIdArray(src));
    CHECK(aten::IsValidIdArray(dst));
    CHECK_EQ(src->shape[0], dst->shape[0]) << "Input arrays should have the same length.";
    adj_ = aten::COOMatrix{num_src, num_dst, src, dst};
  }

  COO(GraphPtr metagraph, int64_t num_src, int64_t num_dst,
      IdArray src, IdArray dst, bool is_multigraph)
    : BaseHeteroGraph(metagraph),
      is_multigraph_(is_multigraph) {
    CHECK(aten::IsValidIdArray(src));
    CHECK(aten::IsValidIdArray(dst));
    CHECK_EQ(src->shape[0], dst->shape[0]) << "Input arrays should have the same length.";
    adj_ = aten::COOMatrix{num_src, num_dst, src, dst};
  }

  explicit COO(GraphPtr metagraph, const aten::COOMatrix& coo)
    : BaseHeteroGraph(metagraph), adj_(coo) {}

  inline dgl_type_t SrcType() const {
    return 0;
  }

  inline dgl_type_t DstType() const {
    return NumVertexTypes() == 1? 0 : 1;
  }

  inline dgl_type_t EdgeType() const {
    return 0;
  }

  HeteroGraphPtr GetRelationGraph(dgl_type_t etype) const override {
    LOG(FATAL) << "The method shouldn't be called for UnitGraph graph. "
      << "The relation graph is simply this graph itself.";
    return {};
  }

  void AddVertices(dgl_type_t vtype, uint64_t num_vertices) override {
    LOG(FATAL) << "UnitGraph graph is not mutable.";
  }

  void AddEdge(dgl_type_t etype, dgl_id_t src, dgl_id_t dst) override {
    LOG(FATAL) << "UnitGraph graph is not mutable.";
  }

  void AddEdges(dgl_type_t etype, IdArray src_ids, IdArray dst_ids) override {
    LOG(FATAL) << "UnitGraph graph is not mutable.";
  }

  void Clear() override {
    LOG(FATAL) << "UnitGraph graph is not mutable.";
  }

  DLContext Context() const override {
    return adj_.row->ctx;
  }

  uint8_t NumBits() const override {
    return adj_.row->dtype.bits;
  }

  bool IsMultigraph() const override {
    return const_cast<COO*>(this)->is_multigraph_.Get([this] () {
        return aten::COOHasDuplicate(adj_);
      });
  }

  bool IsReadonly() const override {
    return true;
  }

  uint64_t NumVertices(dgl_type_t vtype) const override {
    if (vtype == SrcType()) {
      return adj_.num_rows;
    } else if (vtype == DstType()) {
      return adj_.num_cols;
    } else {
      LOG(FATAL) << "Invalid vertex type: " << vtype;
      return 0;
    }
  }

  uint64_t NumEdges(dgl_type_t etype) const override {
    return adj_.row->shape[0];
  }

  bool HasVertex(dgl_type_t vtype, dgl_id_t vid) const override {
    return vid < NumVertices(vtype);
  }

  BoolArray HasVertices(dgl_type_t vtype, IdArray vids) const override {
    LOG(FATAL) << "Not enabled for COO graph";
    return {};
  }

  bool HasEdgeBetween(dgl_type_t etype, dgl_id_t src, dgl_id_t dst) const override {
    LOG(INFO) << "Not enabled for COO graph.";
    return {};
  }

  BoolArray HasEdgesBetween(dgl_type_t etype, IdArray src_ids, IdArray dst_ids) const override {
    LOG(INFO) << "Not enabled for COO graph.";
    return {};
  }

  IdArray Predecessors(dgl_type_t etype, dgl_id_t dst) const override {
    LOG(INFO) << "Not enabled for COO graph.";
    return {};
  }

  IdArray Successors(dgl_type_t etype, dgl_id_t src) const override {
    LOG(INFO) << "Not enabled for COO graph.";
    return {};
  }

  IdArray EdgeId(dgl_type_t etype, dgl_id_t src, dgl_id_t dst) const override {
    LOG(INFO) << "Not enabled for COO graph.";
    return {};
  }

  EdgeArray EdgeIds(dgl_type_t etype, IdArray src, IdArray dst) const override {
    LOG(INFO) << "Not enabled for COO graph.";
    return {};
  }

  std::pair<dgl_id_t, dgl_id_t> FindEdge(dgl_type_t etype, dgl_id_t eid) const override {
    CHECK(eid < NumEdges(etype)) << "Invalid edge id: " << eid;
    const auto src = aten::IndexSelect(adj_.row, eid);
    const auto dst = aten::IndexSelect(adj_.col, eid);
    return std::pair<dgl_id_t, dgl_id_t>(src, dst);
  }

  EdgeArray FindEdges(dgl_type_t etype, IdArray eids) const override {
    CHECK(aten::IsValidIdArray(eids)) << "Invalid edge id array";
    return EdgeArray{aten::IndexSelect(adj_.row, eids),
                     aten::IndexSelect(adj_.col, eids),
                     eids};
  }

  EdgeArray InEdges(dgl_type_t etype, dgl_id_t vid) const override {
    LOG(INFO) << "Not enabled for COO graph.";
    return {};
  }

  EdgeArray InEdges(dgl_type_t etype, IdArray vids) const override {
    LOG(INFO) << "Not enabled for COO graph.";
    return {};
  }

  EdgeArray OutEdges(dgl_type_t etype, dgl_id_t vid) const override {
    LOG(INFO) << "Not enabled for COO graph.";
    return {};
  }

  EdgeArray OutEdges(dgl_type_t etype, IdArray vids) const override {
    LOG(INFO) << "Not enabled for COO graph.";
    return {};
  }

  EdgeArray Edges(dgl_type_t etype, const std::string &order = "") const override {
    CHECK(order.empty() || order == std::string("eid"))
      << "COO only support Edges of order \"eid\", but got \""
      << order << "\".";
    IdArray rst_eid = aten::Range(0, NumEdges(etype), NumBits(), Context());
    return EdgeArray{adj_.row, adj_.col, rst_eid};
  }

  uint64_t InDegree(dgl_type_t etype, dgl_id_t vid) const override {
    LOG(INFO) << "Not enabled for COO graph.";
    return {};
  }

  DegreeArray InDegrees(dgl_type_t etype, IdArray vids) const override {
    LOG(INFO) << "Not enabled for COO graph.";
    return {};
  }

  uint64_t OutDegree(dgl_type_t etype, dgl_id_t vid) const override {
    LOG(INFO) << "Not enabled for COO graph.";
    return {};
  }

  DegreeArray OutDegrees(dgl_type_t etype, IdArray vids) const override {
    LOG(INFO) << "Not enabled for COO graph.";
    return {};
  }

  DGLIdIters SuccVec(dgl_type_t etype, dgl_id_t vid) const override {
    LOG(INFO) << "Not enabled for COO graph.";
    return {};
  }

  DGLIdIters OutEdgeVec(dgl_type_t etype, dgl_id_t vid) const override {
    LOG(INFO) << "Not enabled for COO graph.";
    return {};
  }

  DGLIdIters PredVec(dgl_type_t etype, dgl_id_t vid) const override {
    LOG(INFO) << "Not enabled for COO graph.";
    return {};
  }

  DGLIdIters InEdgeVec(dgl_type_t etype, dgl_id_t vid) const override {
    LOG(INFO) << "Not enabled for COO graph.";
    return {};
  }

  std::vector<IdArray> GetAdj(
      dgl_type_t etype, bool transpose, const std::string &fmt) const override {
    CHECK(fmt == "coo") << "Not valid adj format request.";
    if (transpose) {
      return {aten::HStack(adj_.col, adj_.row)};
    } else {
      return {aten::HStack(adj_.row, adj_.col)};
    }
  }

  HeteroSubgraph VertexSubgraph(const std::vector<IdArray>& vids) const override {
    LOG(INFO) << "Not enabled for COO graph.";
    return {};
  }

  HeteroSubgraph EdgeSubgraph(
      const std::vector<IdArray>& eids, bool preserve_nodes = false) const override {
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
          meta_graph(), new_nsrc, new_ndst, new_src, new_dst);
      subg.induced_edges = eids;
    } else {
      IdArray new_src = aten::IndexSelect(adj_.row, eids[0]);
      IdArray new_dst = aten::IndexSelect(adj_.col, eids[0]);
      subg.induced_vertices.emplace_back(aten::Range(0, NumVertices(0), NumBits(), Context()));
      subg.induced_vertices.emplace_back(aten::Range(0, NumVertices(1), NumBits(), Context()));
      subg.graph = std::make_shared<COO>(
          meta_graph(), NumVertices(0), NumVertices(1), new_src, new_dst);
      subg.induced_edges = eids;
    }
    return subg;
  }

  aten::COOMatrix adj() const {
    return adj_;
  }

 private:
  /*! \brief internal adjacency matrix. Data array is empty */
  aten::COOMatrix adj_;

  /*! \brief multi-graph flag */
  Lazy<bool> is_multigraph_;
};

//////////////////////////////////////////////////////////
//
// CSR graph implementation
//
//////////////////////////////////////////////////////////

/*! \brief CSR graph */
class UnitGraph::CSR : public BaseHeteroGraph {
 public:
  CSR(GraphPtr metagraph, int64_t num_src, int64_t num_dst,
      IdArray indptr, IdArray indices, IdArray edge_ids)
    : BaseHeteroGraph(metagraph) {
    CHECK(aten::IsValidIdArray(indptr));
    CHECK(aten::IsValidIdArray(indices));
    CHECK(aten::IsValidIdArray(edge_ids));
    CHECK_EQ(indices->shape[0], edge_ids->shape[0])
      << "indices and edge id arrays should have the same length";
    adj_ = aten::CSRMatrix{num_src, num_dst, indptr, indices, edge_ids};
    sorted_ = false;
  }

  CSR(GraphPtr metagraph, int64_t num_src, int64_t num_dst,
      IdArray indptr, IdArray indices, IdArray edge_ids, bool is_multigraph)
    : BaseHeteroGraph(metagraph), is_multigraph_(is_multigraph) {
    CHECK(aten::IsValidIdArray(indptr));
    CHECK(aten::IsValidIdArray(indices));
    CHECK(aten::IsValidIdArray(edge_ids));
    CHECK_EQ(indices->shape[0], edge_ids->shape[0])
      << "indices and edge id arrays should have the same length";
    adj_ = aten::CSRMatrix{num_src, num_dst, indptr, indices, edge_ids};
    sorted_ = false;
  }

  explicit CSR(GraphPtr metagraph, const aten::CSRMatrix& csr)
    : BaseHeteroGraph(metagraph), adj_(csr) {
    sorted_ = false;
  }

  inline dgl_type_t SrcType() const {
    return 0;
  }

  inline dgl_type_t DstType() const {
    return NumVertexTypes() == 1? 0 : 1;
  }

  inline dgl_type_t EdgeType() const {
    return 0;
  }

  HeteroGraphPtr GetRelationGraph(dgl_type_t etype) const override {
    LOG(FATAL) << "The method shouldn't be called for UnitGraph graph. "
      << "The relation graph is simply this graph itself.";
    return {};
  }

  void AddVertices(dgl_type_t vtype, uint64_t num_vertices) override {
    LOG(FATAL) << "UnitGraph graph is not mutable.";
  }

  void AddEdge(dgl_type_t etype, dgl_id_t src, dgl_id_t dst) override {
    LOG(FATAL) << "UnitGraph graph is not mutable.";
  }

  void AddEdges(dgl_type_t etype, IdArray src_ids, IdArray dst_ids) override {
    LOG(FATAL) << "UnitGraph graph is not mutable.";
  }

  void Clear() override {
    LOG(FATAL) << "UnitGraph graph is not mutable.";
  }

  DLContext Context() const override {
    return adj_.indices->ctx;
  }

  uint8_t NumBits() const override {
    return adj_.indices->dtype.bits;
  }

  CSR AsNumBits(uint8_t bits) const {
    if (NumBits() == bits) {
      return *this;
    } else {
      CSR ret(
          meta_graph_,
          adj_.num_rows, adj_.num_cols,
          aten::AsNumBits(adj_.indptr, bits),
          aten::AsNumBits(adj_.indices, bits),
          aten::AsNumBits(adj_.data, bits));
      ret.is_multigraph_ = is_multigraph_;
      return ret;
    }
  }

  CSR CopyTo(const DLContext& ctx) const {
    if (Context() == ctx) {
      return *this;
    } else {
      CSR ret(
          meta_graph_,
          adj_.num_rows, adj_.num_cols,
          adj_.indptr.CopyTo(ctx),
          adj_.indices.CopyTo(ctx),
          adj_.data.CopyTo(ctx));
      ret.is_multigraph_ = is_multigraph_;
      return ret;
    }
  }

  bool IsMultigraph() const override {
    return const_cast<CSR*>(this)->is_multigraph_.Get([this] () {
        return aten::CSRHasDuplicate(adj_);
      });
  }

  bool IsReadonly() const override {
    return true;
  }

  uint64_t NumVertices(dgl_type_t vtype) const override {
    if (vtype == SrcType()) {
      return adj_.num_rows;
    } else if (vtype == DstType()) {
      return adj_.num_cols;
    } else {
      LOG(FATAL) << "Invalid vertex type: " << vtype;
      return 0;
    }
  }

  uint64_t NumEdges(dgl_type_t etype) const override {
    return adj_.indices->shape[0];
  }

  bool HasVertex(dgl_type_t vtype, dgl_id_t vid) const override {
    return vid < NumVertices(vtype);
  }

  BoolArray HasVertices(dgl_type_t vtype, IdArray vids) const override {
    LOG(FATAL) << "Not enabled for COO graph";
    return {};
  }

  bool HasEdgeBetween(dgl_type_t etype, dgl_id_t src, dgl_id_t dst) const override {
    CHECK(HasVertex(SrcType(), src)) << "Invalid src vertex id: " << src;
    CHECK(HasVertex(DstType(), dst)) << "Invalid dst vertex id: " << dst;
    return aten::CSRIsNonZero(adj_, src, dst);
  }

  BoolArray HasEdgesBetween(dgl_type_t etype, IdArray src_ids, IdArray dst_ids) const override {
    CHECK(aten::IsValidIdArray(src_ids)) << "Invalid vertex id array.";
    CHECK(aten::IsValidIdArray(dst_ids)) << "Invalid vertex id array.";
    return aten::CSRIsNonZero(adj_, src_ids, dst_ids);
  }

  IdArray Predecessors(dgl_type_t etype, dgl_id_t dst) const override {
    LOG(INFO) << "Not enabled for CSR graph.";
    return {};
  }

  IdArray Successors(dgl_type_t etype, dgl_id_t src) const override {
    CHECK(HasVertex(SrcType(), src)) << "Invalid src vertex id: " << src;
    return aten::CSRGetRowColumnIndices(adj_, src);
  }

  IdArray EdgeId(dgl_type_t etype, dgl_id_t src, dgl_id_t dst) const override {
    CHECK(HasVertex(SrcType(), src)) << "Invalid src vertex id: " << src;
    CHECK(HasVertex(DstType(), dst)) << "Invalid dst vertex id: " << dst;
    return aten::CSRGetData(adj_, src, dst);
  }

  EdgeArray EdgeIds(dgl_type_t etype, IdArray src, IdArray dst) const override {
    CHECK(aten::IsValidIdArray(src)) << "Invalid vertex id array.";
    CHECK(aten::IsValidIdArray(dst)) << "Invalid vertex id array.";
    const auto& arrs = aten::CSRGetDataAndIndices(adj_, src, dst);
    return EdgeArray{arrs[0], arrs[1], arrs[2]};
  }

  std::pair<dgl_id_t, dgl_id_t> FindEdge(dgl_type_t etype, dgl_id_t eid) const override {
    LOG(INFO) << "Not enabled for CSR graph.";
    return {};
  }

  EdgeArray FindEdges(dgl_type_t etype, IdArray eids) const override {
    LOG(INFO) << "Not enabled for CSR graph.";
    return {};
  }

  EdgeArray InEdges(dgl_type_t etype, dgl_id_t vid) const override {
    LOG(INFO) << "Not enabled for CSR graph.";
    return {};
  }

  EdgeArray InEdges(dgl_type_t etype, IdArray vids) const override {
    LOG(INFO) << "Not enabled for CSR graph.";
    return {};
  }

  EdgeArray OutEdges(dgl_type_t etype, dgl_id_t vid) const override {
    CHECK(HasVertex(SrcType(), vid)) << "Invalid src vertex id: " << vid;
    IdArray ret_dst = aten::CSRGetRowColumnIndices(adj_, vid);
    IdArray ret_eid = aten::CSRGetRowData(adj_, vid);
    IdArray ret_src = aten::Full(vid, ret_dst->shape[0], NumBits(), ret_dst->ctx);
    return EdgeArray{ret_src, ret_dst, ret_eid};
  }

  EdgeArray OutEdges(dgl_type_t etype, IdArray vids) const override {
    CHECK(aten::IsValidIdArray(vids)) << "Invalid vertex id array.";
    auto csrsubmat = aten::CSRSliceRows(adj_, vids);
    auto coosubmat = aten::CSRToCOO(csrsubmat, false);
    // Note that the row id in the csr submat is relabled, so
    // we need to recover it using an index select.
    auto row = aten::IndexSelect(vids, coosubmat.row);
    return EdgeArray{row, coosubmat.col, coosubmat.data};
  }

  EdgeArray Edges(dgl_type_t etype, const std::string &order = "") const override {
    CHECK(order.empty() || order == std::string("srcdst"))
      << "CSR only support Edges of order \"srcdst\","
      << " but got \"" << order << "\".";
    const auto& coo = aten::CSRToCOO(adj_, false);
    return EdgeArray{coo.row, coo.col, coo.data};
  }

  uint64_t InDegree(dgl_type_t etype, dgl_id_t vid) const override {
    LOG(INFO) << "Not enabled for CSR graph.";
    return {};
  }

  DegreeArray InDegrees(dgl_type_t etype, IdArray vids) const override {
    LOG(INFO) << "Not enabled for CSR graph.";
    return {};
  }

  uint64_t OutDegree(dgl_type_t etype, dgl_id_t vid) const override {
    CHECK(HasVertex(SrcType(), vid)) << "Invalid src vertex id: " << vid;
    return aten::CSRGetRowNNZ(adj_, vid);
  }

  DegreeArray OutDegrees(dgl_type_t etype, IdArray vids) const override {
    CHECK(aten::IsValidIdArray(vids)) << "Invalid vertex id array.";
    return aten::CSRGetRowNNZ(adj_, vids);
  }

  DGLIdIters SuccVec(dgl_type_t etype, dgl_id_t vid) const override {
    // TODO(minjie): This still assumes the data type and device context
    //   of this graph. Should fix later.
    const dgl_id_t* indptr_data = static_cast<dgl_id_t*>(adj_.indptr->data);
    const dgl_id_t* indices_data = static_cast<dgl_id_t*>(adj_.indices->data);
    const dgl_id_t start = indptr_data[vid];
    const dgl_id_t end = indptr_data[vid + 1];
    return DGLIdIters(indices_data + start, indices_data + end);
  }

  DGLIdIters OutEdgeVec(dgl_type_t etype, dgl_id_t vid) const override {
    // TODO(minjie): This still assumes the data type and device context
    //   of this graph. Should fix later.
    const dgl_id_t* indptr_data = static_cast<dgl_id_t*>(adj_.indptr->data);
    const dgl_id_t* eid_data = static_cast<dgl_id_t*>(adj_.data->data);
    const dgl_id_t start = indptr_data[vid];
    const dgl_id_t end = indptr_data[vid + 1];
    return DGLIdIters(eid_data + start, eid_data + end);
  }

  DGLIdIters PredVec(dgl_type_t etype, dgl_id_t vid) const override {
    LOG(INFO) << "Not enabled for CSR graph.";
    return {};
  }

  DGLIdIters InEdgeVec(dgl_type_t etype, dgl_id_t vid) const override {
    LOG(INFO) << "Not enabled for CSR graph.";
    return {};
  }

  std::vector<IdArray> GetAdj(
      dgl_type_t etype, bool transpose, const std::string &fmt) const override {
    CHECK(!transpose && fmt == "csr") << "Not valid adj format request.";
    return {adj_.indptr, adj_.indices, adj_.data};
  }

  HeteroSubgraph VertexSubgraph(const std::vector<IdArray>& vids) const override {
    CHECK_EQ(vids.size(), NumVertexTypes()) << "Number of vertex types mismatch";
    auto srcvids = vids[SrcType()], dstvids = vids[DstType()];
    CHECK(aten::IsValidIdArray(srcvids)) << "Invalid vertex id array.";
    CHECK(aten::IsValidIdArray(dstvids)) << "Invalid vertex id array.";
    HeteroSubgraph subg;
    const auto& submat = aten::CSRSliceMatrix(adj_, srcvids, dstvids);
    IdArray sub_eids = aten::Range(0, submat.data->shape[0], NumBits(), Context());
    subg.graph = std::make_shared<CSR>(meta_graph(), submat.num_rows, submat.num_cols,
        submat.indptr, submat.indices, sub_eids);
    subg.induced_vertices = vids;
    subg.induced_edges.emplace_back(submat.data);
    return subg;
  }

  HeteroSubgraph EdgeSubgraph(
      const std::vector<IdArray>& eids, bool preserve_nodes = false) const override {
    LOG(INFO) << "Not enabled for CSR graph.";
    return {};
  }

  aten::CSRMatrix adj() const {
    return adj_;
  }

 private:
  /*! \brief internal adjacency matrix. Data array stores edge ids */
  aten::CSRMatrix adj_;

  /*! \brief multi-graph flag */
  Lazy<bool> is_multigraph_;

  /*! \brief indicate that the edges are stored in the sorted order. */
  bool sorted_;
};

//////////////////////////////////////////////////////////
//
// unit graph implementation
//
//////////////////////////////////////////////////////////

DLContext UnitGraph::Context() const {
  return GetAny()->Context();
}

uint8_t UnitGraph::NumBits() const {
  return GetAny()->NumBits();
}

bool UnitGraph::IsMultigraph() const {
  return GetAny()->IsMultigraph();
}

uint64_t UnitGraph::NumVertices(dgl_type_t vtype) const {
  if (in_csr_) {
    vtype = (vtype == SrcType()) ? DstType() : SrcType();
    return in_csr_->NumVertices(vtype);
  } else if (out_csr_) {
    return out_csr_->NumVertices(vtype);
  } else {
    return GetCOO()->NumVertices(vtype);
  }
}

uint64_t UnitGraph::NumEdges(dgl_type_t etype) const {
  return GetAny()->NumEdges(etype);
}

bool UnitGraph::HasVertex(dgl_type_t vtype, dgl_id_t vid) const {
  if (in_csr_) {
    vtype = (vtype == SrcType()) ? DstType() : SrcType();
    return in_csr_->HasVertex(vtype, vid);
  } else if (out_csr_) {
    return out_csr_->HasVertex(vtype, vid);
  } else {
    return GetCOO()->HasVertex(vtype, vid);
  }
}

BoolArray UnitGraph::HasVertices(dgl_type_t vtype, IdArray vids) const {
  CHECK(aten::IsValidIdArray(vids)) << "Invalid id array input";
  return aten::LT(vids, NumVertices(vtype));
}

bool UnitGraph::HasEdgeBetween(dgl_type_t etype, dgl_id_t src, dgl_id_t dst) const {
  if (in_csr_) {
    return in_csr_->HasEdgeBetween(etype, dst, src);
  } else {
    return GetOutCSR()->HasEdgeBetween(etype, src, dst);
  }
}

BoolArray UnitGraph::HasEdgesBetween(
    dgl_type_t etype, IdArray src, IdArray dst) const {
  if (in_csr_) {
    return in_csr_->HasEdgesBetween(etype, dst, src);
  } else {
    return GetOutCSR()->HasEdgesBetween(etype, src, dst);
  }
}

IdArray UnitGraph::Predecessors(dgl_type_t etype, dgl_id_t dst) const {
  return GetInCSR()->Successors(etype, dst);
}

IdArray UnitGraph::Successors(dgl_type_t etype, dgl_id_t src) const {
  return GetOutCSR()->Successors(etype, src);
}

IdArray UnitGraph::EdgeId(dgl_type_t etype, dgl_id_t src, dgl_id_t dst) const {
  if (in_csr_) {
    return in_csr_->EdgeId(etype, dst, src);
  } else {
    return GetOutCSR()->EdgeId(etype, src, dst);
  }
}

EdgeArray UnitGraph::EdgeIds(dgl_type_t etype, IdArray src, IdArray dst) const {
  if (in_csr_) {
    EdgeArray edges = in_csr_->EdgeIds(etype, dst, src);
    return EdgeArray{edges.dst, edges.src, edges.id};
  } else {
    return GetOutCSR()->EdgeIds(etype, src, dst);
  }
}

std::pair<dgl_id_t, dgl_id_t> UnitGraph::FindEdge(dgl_type_t etype, dgl_id_t eid) const {
  return GetCOO()->FindEdge(etype, eid);
}

EdgeArray UnitGraph::FindEdges(dgl_type_t etype, IdArray eids) const {
  return GetCOO()->FindEdges(etype, eids);
}

EdgeArray UnitGraph::InEdges(dgl_type_t etype, dgl_id_t vid) const {
  const EdgeArray& ret = GetInCSR()->OutEdges(etype, vid);
  return {ret.dst, ret.src, ret.id};
}

EdgeArray UnitGraph::InEdges(dgl_type_t etype, IdArray vids) const {
  const EdgeArray& ret = GetInCSR()->OutEdges(etype, vids);
  return {ret.dst, ret.src, ret.id};
}

EdgeArray UnitGraph::OutEdges(dgl_type_t etype, dgl_id_t vid) const {
  return GetOutCSR()->OutEdges(etype, vid);
}

EdgeArray UnitGraph::OutEdges(dgl_type_t etype, IdArray vids) const {
  return GetOutCSR()->OutEdges(etype, vids);
}

EdgeArray UnitGraph::Edges(dgl_type_t etype, const std::string &order) const {
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

uint64_t UnitGraph::InDegree(dgl_type_t etype, dgl_id_t vid) const {
  return GetInCSR()->OutDegree(etype, vid);
}

DegreeArray UnitGraph::InDegrees(dgl_type_t etype, IdArray vids) const {
  return GetInCSR()->OutDegrees(etype, vids);
}

uint64_t UnitGraph::OutDegree(dgl_type_t etype, dgl_id_t vid) const {
  return GetOutCSR()->OutDegree(etype, vid);
}

DegreeArray UnitGraph::OutDegrees(dgl_type_t etype, IdArray vids) const {
  return GetOutCSR()->OutDegrees(etype, vids);
}

DGLIdIters UnitGraph::SuccVec(dgl_type_t etype, dgl_id_t vid) const {
  return GetOutCSR()->SuccVec(etype, vid);
}

DGLIdIters UnitGraph::OutEdgeVec(dgl_type_t etype, dgl_id_t vid) const {
  return GetOutCSR()->OutEdgeVec(etype, vid);
}

DGLIdIters UnitGraph::PredVec(dgl_type_t etype, dgl_id_t vid) const {
  return GetInCSR()->SuccVec(etype, vid);
}

DGLIdIters UnitGraph::InEdgeVec(dgl_type_t etype, dgl_id_t vid) const {
  return GetInCSR()->OutEdgeVec(etype, vid);
}

std::vector<IdArray> UnitGraph::GetAdj(
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

HeteroSubgraph UnitGraph::VertexSubgraph(const std::vector<IdArray>& vids) const {
  // We prefer to generate a subgraph from out-csr.
  auto sg = GetOutCSR()->VertexSubgraph(vids);
  CSRPtr subcsr = std::dynamic_pointer_cast<CSR>(sg.graph);
  HeteroSubgraph ret;
  ret.graph = HeteroGraphPtr(new UnitGraph(meta_graph(), nullptr, subcsr, nullptr));
  ret.induced_vertices = std::move(sg.induced_vertices);
  ret.induced_edges = std::move(sg.induced_edges);
  return ret;
}

HeteroSubgraph UnitGraph::EdgeSubgraph(
    const std::vector<IdArray>& eids, bool preserve_nodes) const {
  auto sg = GetCOO()->EdgeSubgraph(eids, preserve_nodes);
  COOPtr subcoo = std::dynamic_pointer_cast<COO>(sg.graph);
  HeteroSubgraph ret;
  ret.graph = HeteroGraphPtr(new UnitGraph(meta_graph(), nullptr, nullptr, subcoo));
  ret.induced_vertices = std::move(sg.induced_vertices);
  ret.induced_edges = std::move(sg.induced_edges);
  return ret;
}

HeteroGraphPtr UnitGraph::CreateFromCOO(
    int64_t num_vtypes, int64_t num_src, int64_t num_dst, IdArray row, IdArray col) {
  CHECK(num_vtypes == 1 || num_vtypes == 2);
  if (num_vtypes == 1)
    CHECK_EQ(num_src, num_dst);
  auto mg = CreateUnitGraphMetaGraph(num_vtypes);
  COOPtr coo(new COO(mg, num_src, num_dst, row, col));
  return HeteroGraphPtr(new UnitGraph(mg, nullptr, nullptr, coo));
}

HeteroGraphPtr UnitGraph::CreateFromCSR(
    int64_t num_vtypes, int64_t num_src, int64_t num_dst,
    IdArray indptr, IdArray indices, IdArray edge_ids) {
  CHECK(num_vtypes == 1 || num_vtypes == 2);
  if (num_vtypes == 1)
    CHECK_EQ(num_src, num_dst);
  auto mg = CreateUnitGraphMetaGraph(num_vtypes);
  CSRPtr csr(new CSR(mg, num_src, num_dst, indptr, indices, edge_ids));
  return HeteroGraphPtr(new UnitGraph(mg, nullptr, csr, nullptr));
}

HeteroGraphPtr UnitGraph::AsNumBits(HeteroGraphPtr g, uint8_t bits) {
  if (g->NumBits() == bits) {
    return g;
  } else {
    // TODO(minjie): since we don't have int32 operations,
    //   we make sure that this graph (on CPU) has materialized CSR,
    //   and then copy them to other context (usually GPU). This should
    //   be fixed later.
    auto bg = std::dynamic_pointer_cast<UnitGraph>(g);
    CHECK_NOTNULL(bg);
    CSRPtr new_incsr = CSRPtr(new CSR(bg->GetInCSR()->AsNumBits(bits)));
    CSRPtr new_outcsr = CSRPtr(new CSR(bg->GetOutCSR()->AsNumBits(bits)));
    return HeteroGraphPtr(new UnitGraph(g->meta_graph(), new_incsr, new_outcsr, nullptr));
  }
}

HeteroGraphPtr UnitGraph::CopyTo(HeteroGraphPtr g, const DLContext& ctx) {
  if (ctx == g->Context()) {
    return g;
  }
  // TODO(minjie): since we don't have GPU implementation of COO<->CSR,
  //   we make sure that this graph (on CPU) has materialized CSR,
  //   and then copy them to other context (usually GPU). This should
  //   be fixed later.
  auto bg = std::dynamic_pointer_cast<UnitGraph>(g);
  CHECK_NOTNULL(bg);
  CSRPtr new_incsr = CSRPtr(new CSR(bg->GetInCSR()->CopyTo(ctx)));
  CSRPtr new_outcsr = CSRPtr(new CSR(bg->GetOutCSR()->CopyTo(ctx)));
  return HeteroGraphPtr(new UnitGraph(g->meta_graph(), new_incsr, new_outcsr, nullptr));
}

UnitGraph::UnitGraph(GraphPtr metagraph, CSRPtr in_csr, CSRPtr out_csr, COOPtr coo)
  : BaseHeteroGraph(metagraph), in_csr_(in_csr), out_csr_(out_csr), coo_(coo) {
  CHECK(GetAny()) << "At least one graph structure should exist.";
}

UnitGraph::CSRPtr UnitGraph::GetInCSR() const {
  if (!in_csr_) {
    if (out_csr_) {
      const auto& newadj = aten::CSRTranspose(out_csr_->adj());
      const_cast<UnitGraph*>(this)->in_csr_ = std::make_shared<CSR>(meta_graph(), newadj);
    } else {
      CHECK(coo_) << "None of CSR, COO exist";
      const auto& adj = coo_->adj();
      const auto& newadj = aten::COOToCSR(
          aten::COOMatrix{adj.num_cols, adj.num_rows, adj.col, adj.row});
      const_cast<UnitGraph*>(this)->in_csr_ = std::make_shared<CSR>(meta_graph(), newadj);
    }
  }
  return in_csr_;
}

/* !\brief Return out csr. If not exist, transpose the other one.*/
UnitGraph::CSRPtr UnitGraph::GetOutCSR() const {
  if (!out_csr_) {
    if (in_csr_) {
      const auto& newadj = aten::CSRTranspose(in_csr_->adj());
      const_cast<UnitGraph*>(this)->out_csr_ = std::make_shared<CSR>(meta_graph(), newadj);
    } else {
      CHECK(coo_) << "None of CSR, COO exist";
      const auto& newadj = aten::COOToCSR(coo_->adj());
      const_cast<UnitGraph*>(this)->out_csr_ = std::make_shared<CSR>(meta_graph(), newadj);
    }
  }
  return out_csr_;
}

/* !\brief Return coo. If not exist, create from csr.*/
UnitGraph::COOPtr UnitGraph::GetCOO() const {
  if (!coo_) {
    if (in_csr_) {
      const auto& newadj = aten::CSRToCOO(in_csr_->adj(), true);
      const_cast<UnitGraph*>(this)->coo_ = std::make_shared<COO>(
          meta_graph(),
          aten::COOMatrix{newadj.num_cols, newadj.num_rows, newadj.col, newadj.row});
    } else {
      CHECK(out_csr_) << "Both CSR are missing.";
      const auto& newadj = aten::CSRToCOO(out_csr_->adj(), true);
      const_cast<UnitGraph*>(this)->coo_ = std::make_shared<COO>(meta_graph(), newadj);
    }
  }
  return coo_;
}

aten::CSRMatrix UnitGraph::GetInCSRMatrix() const {
  return GetInCSR()->adj();
}

aten::CSRMatrix UnitGraph::GetOutCSRMatrix() const {
  return GetOutCSR()->adj();
}

aten::COOMatrix UnitGraph::GetCOOMatrix() const {
  return GetCOO()->adj();
}

HeteroGraphPtr UnitGraph::GetAny() const {
  if (in_csr_) {
    return in_csr_;
  } else if (out_csr_) {
    return out_csr_;
  } else {
    return coo_;
  }
}

}  // namespace dgl
