/*!
 *  Copyright (c) 2019 by Contributors
 * \file dgl/heterograph_interface.h
 * \brief DGL heterogeneous graph index class.
 */

#ifndef DGL_BASE_HETEROGRAPH_H_
#define DGL_BASE_HETEROGRAPH_H_

#include <string>
#include <vector>
#include <utility>
#include <algorithm>
#include <memory>

#include "./runtime/object.h"
#include "graph_interface.h"
#include "array.h"

namespace dgl {

// Forward declaration
class BaseHeteroGraph;
typedef std::shared_ptr<BaseHeteroGraph> HeteroGraphPtr;
struct HeteroSubgraph;

/*!
 * \brief Base heterogenous graph.
 *
 * In heterograph, nodes represent entities and edges represent relations.
 * Nodes and edges are associated with types. The same pair of entity types
 * can have multiple relation types between them, but relation type **uniquely**
 * identifies the source and destination entity types.
 *
 * In a high-level, a heterograph is a data structure composed of:
 *  - A meta-graph that stores the entity-entity relation graph.
 *  - A dictionary of relation type to the bipartite graph representing the
 *    actual connections among entity nodes.
 */
class BaseHeteroGraph : public runtime::Object {
 public:
  explicit BaseHeteroGraph(GraphPtr meta_graph): meta_graph_(meta_graph) {}

  virtual ~BaseHeteroGraph() = default;

  ////////////////////////// query/operations on meta graph ////////////////////////

  /*! \return the number of vertex types */
  virtual uint64_t NumVertexTypes() const = 0;

  /*! \return the number of edge types */
  virtual uint64_t NumEdgeTypes() const = 0;

  /*! \return the meta graph */
  virtual GraphPtr meta_graph() const {
    return meta_graph_;
  }

  /*!
   * \brief Return the bipartite graph of the given edge type.
   * \param etype The edge type.
   * \return The bipartite graph.
   */
  virtual HeteroGraphPtr GetRelationGraph(dgl_type_t etype) const = 0;

  ////////////////////////// query/operations on realized graph ////////////////////////

  /*! \brief Add vertices to the given vertex type */
  virtual void AddVertices(dgl_type_t vtype, uint64_t num_vertices) = 0;

  /*! \brief Add one edge to the given edge type */
  virtual void AddEdge(dgl_type_t etype, dgl_id_t src, dgl_id_t dst) = 0;

  /*! \brief Add edges to the given edge type */
  virtual void AddEdges(dgl_type_t etype, IdArray src_ids, IdArray dst_ids) = 0;

  /*!
   * \brief Clear the graph. Remove all vertices/edges.
   */
  virtual void Clear() = 0;

  /*!
   * \brief Get the device context of this graph.
   */
  virtual DLContext Context() const = 0;

  /*!
   * \brief Get the number of integer bits used to store node/edge ids (32 or 64).
   */
  virtual uint8_t NumBits() const = 0;

  /*!
   * \return whether the graph is a multigraph
   */
  virtual bool IsMultigraph() const = 0;

  /*! \return whether the graph is read-only */
  virtual bool IsReadonly() const = 0;

  /*! \return the number of vertices in the graph.*/
  virtual uint64_t NumVertices(dgl_type_t vtype) const = 0;

  /*! \return the number of edges in the graph.*/
  virtual uint64_t NumEdges(dgl_type_t etype) const = 0;

  /*! \return true if the given vertex is in the graph.*/
  virtual bool HasVertex(dgl_type_t vtype, dgl_id_t vid) const = 0;

  /*! \return a 0-1 array indicating whether the given vertices are in the graph.*/
  virtual BoolArray HasVertices(dgl_type_t vtype, IdArray vids) const = 0;

  /*! \return true if the given edge is in the graph.*/
  virtual bool HasEdgeBetween(dgl_type_t etype, dgl_id_t src, dgl_id_t dst) const = 0;

  /*! \return a 0-1 array indicating whether the given edges are in the graph.*/
  virtual BoolArray HasEdgesBetween(dgl_type_t etype, IdArray src_ids, IdArray dst_ids) const = 0;

  /*!
   * \brief Find the predecessors of a vertex.
   * \note The given vertex should belong to the source vertex type
   *       of the given edge type.
   * \param etype The edge type
   * \param vid The vertex id.
   * \return the predecessor id array.
   */
  virtual IdArray Predecessors(dgl_type_t etype, dgl_id_t dst) const = 0;

  /*!
   * \brief Find the successors of a vertex.
   * \note The given vertex should belong to the dest vertex type
   *       of the given edge type.
   * \param etype The edge type
   * \param vid The vertex id.
   * \return the successor id array.
   */
  virtual IdArray Successors(dgl_type_t etype, dgl_id_t src) const = 0;

  /*!
   * \brief Get all edge ids between the two given endpoints
   * \note The given src and dst vertices should belong to the source vertex type
   *       and the dest vertex type of the given edge type, respectively.
   * \param etype The edge type
   * \param src The source vertex.
   * \param dst The destination vertex.
   * \return the edge id array.
   */
  virtual IdArray EdgeId(dgl_type_t etype, dgl_id_t src, dgl_id_t dst) const = 0;

  /*!
   * \brief Get all edge ids between the given endpoint pairs.
   * \note Edges are associated with an integer id start from zero.
   *       The id is assigned when the edge is being added to the graph.
   *       If duplicate pairs exist, the returned edge IDs will also duplicate.
   *       The order of returned edge IDs will follow the order of src-dst pairs
   *       first, and ties are broken by the order of edge ID.
   * \param etype The edge type
   * \param src The src vertex ids.
   * \param dst The dst vertex ids.
   * \return EdgeArray containing all edges between all pairs.
   */
  virtual EdgeArray EdgeIds(dgl_type_t etype, IdArray src, IdArray dst) const = 0;

  /*!
   * \brief Find the edge ID and return the pair of endpoints
   * \param etype The edge type
   * \param eid The edge ID
   * \return a pair whose first element is the source and the second the destination.
   */
  virtual std::pair<dgl_id_t, dgl_id_t> FindEdge(dgl_type_t etype, dgl_id_t eid) const = 0;

  /*!
   * \brief Find the edge IDs and return their source and target node IDs.
   * \param etype The edge type
   * \param eids The edge ID array.
   * \return EdgeArray containing all edges with id in eid.  The order is preserved.
   */
  virtual EdgeArray FindEdges(dgl_type_t etype, IdArray eids) const = 0;

  /*!
   * \brief Get the in edges of the vertex.
   * \note The given vertex should belong to the dest vertex type
   *       of the given edge type.
   * \param etype The edge type
   * \param vid The vertex id.
   * \return the edges
   */
  virtual EdgeArray InEdges(dgl_type_t etype, dgl_id_t vid) const = 0;

  /*!
   * \brief Get the in edges of the vertices.
   * \note The given vertex should belong to the dest vertex type
   *       of the given edge type.
   * \param etype The edge type
   * \param vids The vertex id array.
   * \return the id arrays of the two endpoints of the edges.
   */
  virtual EdgeArray InEdges(dgl_type_t etype, IdArray vids) const = 0;

  /*!
   * \brief Get the out edges of the vertex.
   * \note The given vertex should belong to the source vertex type
   *       of the given edge type.
   * \param etype The edge type
   * \param vid The vertex id.
   * \return the id arrays of the two endpoints of the edges.
   */
  virtual EdgeArray OutEdges(dgl_type_t etype, dgl_id_t vid) const = 0;

  /*!
   * \brief Get the out edges of the vertices.
   * \note The given vertex should belong to the source vertex type
   *       of the given edge type.
   * \param etype The edge type
   * \param vids The vertex id array.
   * \return the id arrays of the two endpoints of the edges.
   */
  virtual EdgeArray OutEdges(dgl_type_t etype, IdArray vids) const = 0;

  /*!
   * \brief Get all the edges in the graph.
   * \note If order is "srcdst", the returned edges list is sorted by their src and
   *       dst ids. If order is "eid", they are in their edge id order.
   *       Otherwise, in the arbitrary order.
   * \param etype The edge type
   * \param order The order of the returned edge list.
   * \return the id arrays of the two endpoints of the edges.
   */
  virtual EdgeArray Edges(dgl_type_t etype, const std::string &order = "") const = 0;

  /*!
   * \brief Get the in degree of the given vertex.
   * \note The given vertex should belong to the dest vertex type
   *       of the given edge type.
   * \param etype The edge type
   * \param vid The vertex id.
   * \return the in degree
   */
  virtual uint64_t InDegree(dgl_type_t etype, dgl_id_t vid) const = 0;

  /*!
   * \brief Get the in degrees of the given vertices.
   * \note The given vertex should belong to the dest vertex type
   *       of the given edge type.
   * \param etype The edge type
   * \param vid The vertex id array.
   * \return the in degree array
   */
  virtual DegreeArray InDegrees(dgl_type_t etype, IdArray vids) const = 0;

  /*!
   * \brief Get the out degree of the given vertex.
   * \note The given vertex should belong to the source vertex type
   *       of the given edge type.
   * \param etype The edge type
   * \param vid The vertex id.
   * \return the out degree
   */
  virtual uint64_t OutDegree(dgl_type_t etype, dgl_id_t vid) const = 0;

  /*!
   * \brief Get the out degrees of the given vertices.
   * \note The given vertex should belong to the source vertex type
   *       of the given edge type.
   * \param etype The edge type
   * \param vid The vertex id array.
   * \return the out degree array
   */
  virtual DegreeArray OutDegrees(dgl_type_t etype, IdArray vids) const = 0;

  /*!
   * \brief Return the successor vector
   * \note The given vertex should belong to the source vertex type
   *       of the given edge type.
   * \param vid The vertex id.
   * \return the successor vector iterator pair.
   */
  virtual DGLIdIters SuccVec(dgl_type_t etype, dgl_id_t vid) const = 0;

  /*!
   * \brief Return the out edge id vector
   * \note The given vertex should belong to the source vertex type
   *       of the given edge type.
   * \param vid The vertex id.
   * \return the out edge id vector iterator pair.
   */
  virtual DGLIdIters OutEdgeVec(dgl_type_t etype, dgl_id_t vid) const = 0;

  /*!
   * \brief Return the predecessor vector
   * \note The given vertex should belong to the dest vertex type
   *       of the given edge type.
   * \param vid The vertex id.
   * \return the predecessor vector iterator pair.
   */
  virtual DGLIdIters PredVec(dgl_type_t etype, dgl_id_t vid) const = 0;

  /*!
   * \brief Return the in edge id vector
   * \note The given vertex should belong to the dest vertex type
   *       of the given edge type.
   * \param vid The vertex id.
   * \return the in edge id vector iterator pair.
   */
  virtual DGLIdIters InEdgeVec(dgl_type_t etype, dgl_id_t vid) const = 0;

  /*!
   * \brief Get the adjacency matrix of the graph.
   *
   * By default, a row of returned adjacency matrix represents the destination
   * of an edge and the column represents the source.
   *
   * If the fmt is 'csr', the function should return three arrays, representing
   *  indptr, indices and edge ids
   *
   * If the fmt is 'coo', the function should return one array of shape (2, nnz),
   * representing a horitonzal stack of row and col indices.
   *
   * \param transpose A flag to transpose the returned adjacency matrix.
   * \param fmt the format of the returned adjacency matrix.
   * \return a vector of IdArrays.
   */
  virtual std::vector<IdArray> GetAdj(
      dgl_type_t etype, bool transpose, const std::string &fmt) const = 0;

  /*!
   * \brief Extract the induced subgraph by the given vertices.
   * 
   * The length of the given vector should be equal to the number of vertex types.
   * Empty arrays can be provided if no vertex is needed for the type. The result
   * subgraph has the same meta graph with the parent, but some types can have no
   * node/edge.
   *
   * \param vids the induced vertices per type.
   * \return the subgraph.
   */
  virtual HeteroSubgraph VertexSubgraph(const std::vector<IdArray>& vids) const = 0;

  /*!
   * \brief Extract the induced subgraph by the given edges.
   * 
   * The length of the given vector should be equal to the number of edge types.
   * Empty arrays can be provided if no edge is needed for the type. The result
   * subgraph has the same meta graph with the parent, but some types can have no
   * node/edge.
   *
   * \param eids The edges in the subgraph.
   * \param preserve_nodes If true, the vertices will not be relabeled, so some vertices
   *                       may have no incident edges.
   * \return the subgraph.
   */
  virtual HeteroSubgraph EdgeSubgraph(
      const std::vector<IdArray>& eids, bool preserve_nodes = false) const = 0;

  static constexpr const char* _type_key = "graph.HeteroGraph";
  DGL_DECLARE_OBJECT_TYPE_INFO(BaseHeteroGraph, runtime::Object);

 protected:
  /*! \brief meta graph */
  GraphPtr meta_graph_;
};

// Define HeteroGraphRef
DGL_DEFINE_OBJECT_REF(HeteroGraphRef, BaseHeteroGraph);

/*! \brief Heter-subgraph data structure */
struct HeteroSubgraph : public runtime::Object {
  /*! \brief The heterograph. */
  HeteroGraphPtr graph;
  /*!
   * \brief The induced vertex ids of each entity type.
   * The vector length is equal to the number of vertex types in the parent graph.
   */
  std::vector<IdArray> induced_vertices;
  /*!
   * \brief The induced vertex ids of each entity type.
   * The vector length is equal to the number of vertex types in the parent graph.
   */
  std::vector<IdArray> induced_edges;

  static constexpr const char* _type_key = "graph.HeteroSubgraph";
  DGL_DECLARE_OBJECT_TYPE_INFO(HeteroSubgraph, runtime::Object);
};

// Define HeteroSubgraphRef
DGL_DEFINE_OBJECT_REF(HeteroSubgraphRef, HeteroSubgraph);

// creators

/*! \brief Create a bipartite graph from COO arrays */
HeteroGraphPtr CreateBipartiteFromCOO(
    int64_t num_src, int64_t num_dst, IdArray row, IdArray col);

/*! \brief Create a bipartite graph from (out) CSR arrays */
HeteroGraphPtr CreateBipartiteFromCSR(
    int64_t num_src, int64_t num_dst,
    IdArray indptr, IdArray indices, IdArray edge_ids);

/*! \brief Create a heterograph from meta graph and a list of bipartite graph */
HeteroGraphPtr CreateHeteroGraph(
    GraphPtr meta_graph, const std::vector<HeteroGraphPtr>& rel_graphs);

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

 private:
  // internal data structure
  class COO;
  class CSR;
  typedef std::shared_ptr<COO> COOPtr;
  typedef std::shared_ptr<CSR> CSRPtr;

  Bipartite(CSRPtr in_csr, CSRPtr out_csr, COOPtr coo);

  /*! \return Return the in-edge CSR format. Create from other format if not exist. */
  CSRPtr GetInCSR() const;

  /*! \return Return the out-edge CSR format. Create from other format if not exist. */
  CSRPtr GetOutCSR() const;

  /*! \return Return the COO format. Create from other format if not exist. */
  COOPtr GetCOO() const;

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
  COO(int64_t num_src, int64_t num_dst,
               IdArray src, IdArray dst)
    : BaseHeteroGraph(kBipartiteMetaGraph) {
    adj_ = aten::COOMatrix{num_src, num_dst, src, dst};
  }
  COO(int64_t num_src, int64_t num_dst,
               IdArray src, IdArray dst, bool is_multigraph)
    : BaseHeteroGraph(kBipartiteMetaGraph),
      is_multigraph_(is_multigraph) {
    adj_ = aten::COOMatrix{num_src, num_dst, src, dst};
  }
  explicit COO(const aten::COOMatrix& coo)
    : BaseHeteroGraph(kBipartiteMetaGraph), adj_(coo) {}

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
    CHECK(IsValidIdArray(eids)) << "Invalid edge id array";
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
      IdArray indptr, IdArray indices, IdArray edge_ids)
    : BaseHeteroGraph(kBipartiteMetaGraph) {
    adj_ = aten::CSRMatrix{num_src, num_dst, indptr, indices, edge_ids};
  }

  CSR(int64_t num_src, int64_t num_dst,
      IdArray indptr, IdArray indices, IdArray edge_ids, bool is_multigraph)
    : BaseHeteroGraph(kBipartiteMetaGraph),
      is_multigraph_(is_multigraph) {
    adj_ = aten::CSRMatrix{num_src, num_dst, indptr, indices, edge_ids};
  }

  explicit CSR(const aten::CSRMatrix& csr)
    : BaseHeteroGraph(kBipartiteMetaGraph), adj_(csr) {}

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
    CHECK(IsValidIdArray(src_ids)) << "Invalid vertex id array.";
    CHECK(IsValidIdArray(dst_ids)) << "Invalid vertex id array.";
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
    CHECK(IsValidIdArray(src)) << "Invalid vertex id array.";
    CHECK(IsValidIdArray(dst)) << "Invalid vertex id array.";
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
    CHECK(IsValidIdArray(vids)) << "Invalid vertex id array.";
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
    CHECK(IsValidIdArray(vids)) << "Invalid vertex id array.";
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
    CHECK(IsValidIdArray(vids[0])) << "Invalid vertex id array.";
    CHECK(IsValidIdArray(vids[1])) << "Invalid vertex id array.";
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

#endif  // DGL_BASE_HETEROGRAPH_H_
