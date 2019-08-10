/*!
 *  Copyright (c) 2019 by Contributors
 * \file graph/bipartite.h
 * \brief Bipartite graph
 */

#ifndef DGL_GRAPH_BIPARTITE_H_
#define DGL_GRAPH_BIPARTITE_H_

#include <dgl/base_heterograph.h>
#include <dgl/lazy.h>
#include <utility>
#include <string>
#include <vector>

#include "../c_api_common.h"

namespace dgl {

/*!
 * \brief Bipartite graph
 *
 * Bipartite graph is a special type of heterograph which has two types
 * of nodes: "Src" and "Dst". All the edges are from "Src" type nodes to
 * "Dst" type nodes, so there is no edge among nodes of the same type.
 */
class Bipartite : public BaseHeteroGraph {
 public:
  /*! \brief source node group type */
  static constexpr dgl_type_t kSrcVType = 0;
  /*! \brief destination node group type */
  static constexpr dgl_type_t kDstVType = 1;
  /*! \brief edge group type */
  static constexpr dgl_type_t kEType = 0;

  // internal data structure
  class COO;
  class CSR;
  typedef std::shared_ptr<COO> COOPtr;
  typedef std::shared_ptr<CSR> CSRPtr;

  uint64_t NumVertexTypes() const override {
    return 2;
  }

  uint64_t NumEdgeTypes() const override {
    return 1;
  }

  HeteroGraphPtr GetRelationGraph(dgl_type_t etype) const override {
    LOG(FATAL) << "The method shouldn't be called for Bipartite graph. "
      << "The relation graph is simply this graph itself.";
    return {};
  }

  void AddVertices(dgl_type_t vtype, uint64_t num_vertices) override {
    LOG(FATAL) << "Bipartite graph is not mutable.";
  }

  void AddEdge(dgl_type_t etype, dgl_id_t src, dgl_id_t dst) override {
    LOG(FATAL) << "Bipartite graph is not mutable.";
  }

  void AddEdges(dgl_type_t etype, IdArray src_ids, IdArray dst_ids) override {
    LOG(FATAL) << "Bipartite graph is not mutable.";
  }

  void Clear() override {
    LOG(FATAL) << "Bipartite graph is not mutable.";
  }

  DLContext Context() const override;

  uint8_t NumBits() const override;

  bool IsMultigraph() const override;

  bool IsReadonly() const override {
    return true;
  }

  uint64_t NumVertices(dgl_type_t vtype) const override;

  uint64_t NumEdges(dgl_type_t etype) const override;

  bool HasVertex(dgl_type_t vtype, dgl_id_t vid) const override;

  BoolArray HasVertices(dgl_type_t vtype, IdArray vids) const override;

  bool HasEdgeBetween(dgl_type_t etype, dgl_id_t src, dgl_id_t dst) const override;

  BoolArray HasEdgesBetween(dgl_type_t etype, IdArray src_ids, IdArray dst_ids) const override;

  IdArray Predecessors(dgl_type_t etype, dgl_id_t dst) const override;

  IdArray Successors(dgl_type_t etype, dgl_id_t src) const override;

  IdArray EdgeId(dgl_type_t etype, dgl_id_t src, dgl_id_t dst) const override;

  EdgeArray EdgeIds(dgl_type_t etype, IdArray src, IdArray dst) const override;

  std::pair<dgl_id_t, dgl_id_t> FindEdge(dgl_type_t etype, dgl_id_t eid) const override;

  EdgeArray FindEdges(dgl_type_t etype, IdArray eids) const override;

  EdgeArray InEdges(dgl_type_t etype, dgl_id_t vid) const override;

  EdgeArray InEdges(dgl_type_t etype, IdArray vids) const override;

  EdgeArray OutEdges(dgl_type_t etype, dgl_id_t vid) const override;

  EdgeArray OutEdges(dgl_type_t etype, IdArray vids) const override;

  EdgeArray Edges(dgl_type_t etype, const std::string &order = "") const override;

  uint64_t InDegree(dgl_type_t etype, dgl_id_t vid) const override;

  DegreeArray InDegrees(dgl_type_t etype, IdArray vids) const override;

  uint64_t OutDegree(dgl_type_t etype, dgl_id_t vid) const override;

  DegreeArray OutDegrees(dgl_type_t etype, IdArray vids) const override;

  DGLIdIters SuccVec(dgl_type_t etype, dgl_id_t vid) const override;

  DGLIdIters OutEdgeVec(dgl_type_t etype, dgl_id_t vid) const override;

  DGLIdIters PredVec(dgl_type_t etype, dgl_id_t vid) const override;

  DGLIdIters InEdgeVec(dgl_type_t etype, dgl_id_t vid) const override;

  std::vector<IdArray> GetAdj(
      dgl_type_t etype, bool transpose, const std::string &fmt) const override;

  HeteroSubgraph VertexSubgraph(const std::vector<IdArray>& vids) const override;

  HeteroSubgraph EdgeSubgraph(
      const std::vector<IdArray>& eids, bool preserve_nodes = false) const override;

  // creators
  /*! \brief Create a bipartite graph from COO arrays */
  static HeteroGraphPtr CreateFromCOO(int64_t num_src, int64_t num_dst,
      IdArray row, IdArray col);

  /*! \brief Create a bipartite graph from (out) CSR arrays */
  static HeteroGraphPtr CreateFromCSR(
      int64_t num_src, int64_t num_dst,
      IdArray indptr, IdArray indices, IdArray edge_ids);

  /*! \brief Convert the graph to use the given number of bits for storage */
  static HeteroGraphPtr AsNumBits(HeteroGraphPtr g, uint8_t bits);

  /*! \brief Copy the data to another context */
  static HeteroGraphPtr CopyTo(HeteroGraphPtr g, const DLContext& ctx);

  /*! \return Return the in-edge CSR format. Create from other format if not exist. */
  CSRPtr GetInCSR() const;

  /*! \return Return the out-edge CSR format. Create from other format if not exist. */
  CSRPtr GetOutCSR() const;

  /*! \return Return the COO format. Create from other format if not exist. */
  COOPtr GetCOO() const;

 private:
  Bipartite(CSRPtr in_csr, CSRPtr out_csr, COOPtr coo);

  /*! \return Return any existing format. */
  HeteroGraphPtr GetAny() const;

  // Graph stored in different format. We use an on-demand strategy: the format is
  // only materialized if the operation that suitable for it is invoked.
  /*! \brief CSR graph that stores reverse edges */
  CSRPtr in_csr_;
  /*! \brief CSR representation */
  CSRPtr out_csr_;
  /*! \brief COO representation */
  COOPtr coo_;
};

//////////////////////////////////////////////////////////
//
// COO graph implementation
//
//////////////////////////////////////////////////////////

/*! \brief COO graph */
class Bipartite::COO : public BaseHeteroGraph {
 public:
  COO(int64_t num_src, int64_t num_dst, IdArray src, IdArray dst);
  COO(int64_t num_src, int64_t num_dst, IdArray src, IdArray dst, bool is_multigraph);
  explicit COO(const aten::COOMatrix& coo);

  uint64_t NumVertexTypes() const override {
    return 2;
  }
  uint64_t NumEdgeTypes() const override {
    return 1;
  }

  HeteroGraphPtr GetRelationGraph(dgl_type_t etype) const override {
    LOG(FATAL) << "The method shouldn't be called for Bipartite graph. "
      << "The relation graph is simply this graph itself.";
    return {};
  }

  void AddVertices(dgl_type_t vtype, uint64_t num_vertices) override {
    LOG(FATAL) << "Bipartite graph is not mutable.";
  }

  void AddEdge(dgl_type_t etype, dgl_id_t src, dgl_id_t dst) override {
    LOG(FATAL) << "Bipartite graph is not mutable.";
  }

  void AddEdges(dgl_type_t etype, IdArray src_ids, IdArray dst_ids) override {
    LOG(FATAL) << "Bipartite graph is not mutable.";
  }

  void Clear() override {
    LOG(FATAL) << "Bipartite graph is not mutable.";
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
    if (vtype == Bipartite::kSrcVType) {
      return adj_.num_rows;
    } else if (vtype == Bipartite::kDstVType) {
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
      const std::vector<IdArray>& eids, bool preserve_nodes = false) const override;

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
class Bipartite::CSR : public BaseHeteroGraph {
 public:
  CSR(int64_t num_src, int64_t num_dst,
      IdArray indptr, IdArray indices, IdArray edge_ids);

  CSR(int64_t num_src, int64_t num_dst,
      IdArray indptr, IdArray indices, IdArray edge_ids, bool is_multigraph);

  explicit CSR(const aten::CSRMatrix& csr);

  uint64_t NumVertexTypes() const override {
    return 2;
  }
  uint64_t NumEdgeTypes() const override {
    return 1;
  }

  HeteroGraphPtr GetRelationGraph(dgl_type_t etype) const override {
    LOG(FATAL) << "The method shouldn't be called for Bipartite graph. "
      << "The relation graph is simply this graph itself.";
    return {};
  }

  void AddVertices(dgl_type_t vtype, uint64_t num_vertices) override {
    LOG(FATAL) << "Bipartite graph is not mutable.";
  }

  void AddEdge(dgl_type_t etype, dgl_id_t src, dgl_id_t dst) override {
    LOG(FATAL) << "Bipartite graph is not mutable.";
  }

  void AddEdges(dgl_type_t etype, IdArray src_ids, IdArray dst_ids) override {
    LOG(FATAL) << "Bipartite graph is not mutable.";
  }

  void Clear() override {
    LOG(FATAL) << "Bipartite graph is not mutable.";
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
    if (vtype == Bipartite::kSrcVType) {
      return adj_.num_rows;
    } else if (vtype == Bipartite::kDstVType) {
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
    CHECK(HasVertex(0, src)) << "Invalid src vertex id: " << src;
    CHECK(HasVertex(1, dst)) << "Invalid dst vertex id: " << dst;
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
    CHECK(HasVertex(0, src)) << "Invalid src vertex id: " << src;
    return aten::CSRGetRowColumnIndices(adj_, src);
  }

  IdArray EdgeId(dgl_type_t etype, dgl_id_t src, dgl_id_t dst) const override {
    CHECK(HasVertex(0, src)) << "Invalid src vertex id: " << src;
    CHECK(HasVertex(1, dst)) << "Invalid dst vertex id: " << dst;
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
    CHECK(HasVertex(0, vid)) << "Invalid src vertex id: " << vid;
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
    CHECK(HasVertex(0, vid)) << "Invalid src vertex id: " << vid;
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
    CHECK_EQ(vids.size(), 2) << "Number of vertex types mismatch";
    CHECK(aten::IsValidIdArray(vids[0])) << "Invalid vertex id array.";
    CHECK(aten::IsValidIdArray(vids[1])) << "Invalid vertex id array.";
    HeteroSubgraph subg;
    const auto& submat = aten::CSRSliceMatrix(adj_, vids[0], vids[1]);
    IdArray sub_eids = aten::Range(0, submat.data->shape[0], NumBits(), Context());
    subg.graph = std::make_shared<CSR>(submat.num_rows, submat.num_cols,
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
};

};  // namespace dgl

#endif  // DGL_GRAPH_BIPARTITE_H_
