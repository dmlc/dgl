/*!
 *  Copyright (c) 2019 by Contributors
 * \file graph/unit_graph.h
 * \brief UnitGraph graph
 */

#ifndef DGL_GRAPH_UNIT_GRAPH_H_
#define DGL_GRAPH_UNIT_GRAPH_H_

#include <dgl/base_heterograph.h>
#include <dgl/lazy.h>
#include <dgl/array.h>
#include <utility>
#include <string>
#include <vector>

#include "../c_api_common.h"

namespace dgl {

/*!
 * \brief UnitGraph graph
 *
 * UnitGraph graph is a special type of heterograph which
 * (1) Have two types of nodes: "Src" and "Dst". All the edges are
 *     from "Src" type nodes to "Dst" type nodes, so there is no edge among
 *     nodes of the same type. Thus, its metagraph has two nodes and one edge
 *     between them.
 * (2) Have only one type of nodes and edges. Thus, its metagraph has one node
 *     and one self-loop edge.
 */
class UnitGraph : public BaseHeteroGraph {
 public:
  // internal data structure
  class COO;
  class CSR;
  typedef std::shared_ptr<COO> COOPtr;
  typedef std::shared_ptr<CSR> CSRPtr;

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
  /*! \brief Create a graph from COO arrays */
  static HeteroGraphPtr CreateFromCOO(
      int64_t num_vtypes, int64_t num_src, int64_t num_dst,
      IdArray row, IdArray col);

  /*! \brief Create a graph from (out) CSR arrays */
  static HeteroGraphPtr CreateFromCSR(
      int64_t num_vtypes, int64_t num_src, int64_t num_dst,
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

  /*! \return Return the in-edge CSR in the matrix form */
  aten::CSRMatrix GetInCSRMatrix() const;

  /*! \return Return the out-edge CSR in the matrix form */
  aten::CSRMatrix GetOutCSRMatrix() const;

  /*! \return Return the COO matrix form */
  aten::COOMatrix GetCOOMatrix() const;

 private:
  /*!
   * \brief constructor
   * \param metagraph metagraph
   * \param in_csr in edge csr
   * \param out_csr out edge csr
   * \param coo coo
   */
  UnitGraph(GraphPtr metagraph, CSRPtr in_csr, CSRPtr out_csr, COOPtr coo);

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

};  // namespace dgl

#endif  // DGL_GRAPH_UNIT_GRAPH_H_
