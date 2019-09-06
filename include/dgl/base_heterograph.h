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
class FlattenedHeteroGraph;
typedef std::shared_ptr<BaseHeteroGraph> HeteroGraphPtr;
typedef std::shared_ptr<FlattenedHeteroGraph> FlattenedHeteroGraphPtr;
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
  virtual uint64_t NumVertexTypes() const {
    return meta_graph_->NumVertices();
  }

  /*! \return the number of edge types */
  virtual uint64_t NumEdgeTypes() const {
    return meta_graph_->NumEdges();
  }

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

  /*!
   * \brief Convert the list of requested unitgraph graphs into a single unitgraph graph.
   *
   * \param etypes The list of edge type IDs.
   * \return The flattened graph, with induced source/edge/destination types/IDs.
   */
  virtual FlattenedHeteroGraphPtr Flatten(const std::vector<dgl_type_t>& etypes) const {
    LOG(FATAL) << "Flatten operation unsupported";
    return nullptr;
  }

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

/*! \brief The flattened heterograph */
struct FlattenedHeteroGraph : public runtime::Object {
  /*! \brief The graph */
  HeteroGraphRef graph;
  /*!
   * \brief Mapping from source node ID to node type in parent graph
   * \note The induced type array guarantees that the same type always appear contiguously.
   */
  IdArray induced_srctype;
  /*!
   * \brief The set of node types in parent graph appearing in source nodes.
   */
  IdArray induced_srctype_set;
  /*! \brief Mapping from source node ID to local node ID in parent graph */
  IdArray induced_srcid;
  /*!
   * \brief Mapping from edge ID to edge type in parent graph
   * \note The induced type array guarantees that the same type always appear contiguously.
   */
  IdArray induced_etype;
  /*!
   * \brief The set of edge types in parent graph appearing in edges.
   */
  IdArray induced_etype_set;
  /*! \brief Mapping from edge ID to local edge ID in parent graph */
  IdArray induced_eid;
  /*!
   * \brief Mapping from destination node ID to node type in parent graph
   * \note The induced type array guarantees that the same type always appear contiguously.
   */
  IdArray induced_dsttype;
  /*!
   * \brief The set of node types in parent graph appearing in destination nodes.
   */
  IdArray induced_dsttype_set;
  /*! \brief Mapping from destination node ID to local node ID in parent graph */
  IdArray induced_dstid;

  void VisitAttrs(runtime::AttrVisitor *v) final {
    v->Visit("graph", &graph);
    v->Visit("induced_srctype", &induced_srctype);
    v->Visit("induced_srctype_set", &induced_srctype_set);
    v->Visit("induced_srcid", &induced_srcid);
    v->Visit("induced_etype", &induced_etype);
    v->Visit("induced_etype_set", &induced_etype_set);
    v->Visit("induced_eid", &induced_eid);
    v->Visit("induced_dsttype", &induced_dsttype);
    v->Visit("induced_dsttype_set", &induced_dsttype_set);
    v->Visit("induced_dstid", &induced_dstid);
  }

  static constexpr const char* _type_key = "graph.FlattenedHeteroGraph";
  DGL_DECLARE_OBJECT_TYPE_INFO(FlattenedHeteroGraph, runtime::Object);
};
DGL_DEFINE_OBJECT_REF(FlattenedHeteroGraphRef, FlattenedHeteroGraph);

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

};  // namespace dgl

#endif  // DGL_BASE_HETEROGRAPH_H_
