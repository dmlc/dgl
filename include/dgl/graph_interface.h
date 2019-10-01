/*!
 *  Copyright (c) 2018 by Contributors
 * \file dgl/graph_interface.h
 * \brief DGL graph index class.
 */
#ifndef DGL_GRAPH_INTERFACE_H_
#define DGL_GRAPH_INTERFACE_H_

#include <string>
#include <vector>
#include <utility>
#include <algorithm>

#include "./runtime/object.h"
#include "array.h"

namespace dgl {

const dgl_id_t DGL_INVALID_ID = static_cast<dgl_id_t>(-1);

/*!
 * \brief This class references data in std::vector.
 *
 * This isn't a STL-style iterator. It provides a STL data container interface.
 * but it doesn't own data itself. instead, it only references data in std::vector.
 */
class DGLIdIters {
 public:
  /* !\brief default constructor to create an empty range */
  DGLIdIters() {}
  /* !\brief constructor with given begin and end */
  DGLIdIters(const dgl_id_t *begin, const dgl_id_t *end) {
    this->begin_ = begin;
    this->end_ = end;
  }
  const dgl_id_t *begin() const {
    return this->begin_;
  }
  const dgl_id_t *end() const {
    return this->end_;
  }
  dgl_id_t operator[](int64_t i) const {
    return *(this->begin_ + i);
  }
  size_t size() const {
    return this->end_ - this->begin_;
  }
 private:
  const dgl_id_t *begin_{nullptr}, *end_{nullptr};
};

/* \brief structure used to represent a list of edges */
typedef struct {
  /* \brief the two endpoints and the id of the edge */
  IdArray src, dst, id;
} EdgeArray;

// forward declaration
struct Subgraph;
class GraphRef;
class GraphInterface;
typedef std::shared_ptr<GraphInterface> GraphPtr;

/*!
 * \brief dgl graph index interface.
 *
 * DGL's graph is directed. Vertices are integers enumerated from zero.
 *
 * When calling functions supporing multiple edges (e.g. AddEdges, HasEdges),
 * the input edges are represented by two id arrays for source and destination
 * vertex ids. In the general case, the two arrays should have the same length.
 * If the length of src id array is one, it represents one-many connections.
 * If the length of dst id array is one, it represents many-one connections.
 */
class GraphInterface : public runtime::Object {
 public:
  virtual ~GraphInterface() = default;

  /*!
   * \brief Add vertices to the graph.
   * \note Since vertices are integers enumerated from zero, only the number of
   *       vertices to be added needs to be specified.
   * \param num_vertices The number of vertices to be added.
   */
  virtual void AddVertices(uint64_t num_vertices) = 0;

  /*!
   * \brief Add one edge to the graph.
   * \param src The source vertex.
   * \param dst The destination vertex.
   */
  virtual void AddEdge(dgl_id_t src, dgl_id_t dst) = 0;

  /*!
   * \brief Add edges to the graph.
   * \param src_ids The source vertex id array.
   * \param dst_ids The destination vertex id array.
   */
  virtual void AddEdges(IdArray src_ids, IdArray dst_ids) = 0;

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

  /*!
   * \return whether the graph is read-only
   */
  virtual bool IsReadonly() const = 0;

  /*! \return the number of vertices in the graph.*/
  virtual uint64_t NumVertices() const = 0;

  /*! \return the number of edges in the graph.*/
  virtual uint64_t NumEdges() const = 0;

  /*! \return true if the given vertex is in the graph.*/
  virtual bool HasVertex(dgl_id_t vid) const {
    return vid < NumVertices();
  }

  /*! \return a 0-1 array indicating whether the given vertices are in the graph.*/
  virtual BoolArray HasVertices(IdArray vids) const = 0;

  /*! \return true if the given edge is in the graph.*/
  virtual bool HasEdgeBetween(dgl_id_t src, dgl_id_t dst) const = 0;

  /*! \return a 0-1 array indicating whether the given edges are in the graph.*/
  virtual BoolArray HasEdgesBetween(IdArray src_ids, IdArray dst_ids) const = 0;

  /*!
   * \brief Find the predecessors of a vertex.
   * \param vid The vertex id.
   * \param radius The radius of the neighborhood. Default is immediate neighbor (radius=1).
   * \return the predecessor id array.
   */
  virtual IdArray Predecessors(dgl_id_t vid, uint64_t radius = 1) const = 0;

  /*!
   * \brief Find the successors of a vertex.
   * \param vid The vertex id.
   * \param radius The radius of the neighborhood. Default is immediate neighbor (radius=1).
   * \return the successor id array.
   */
  virtual IdArray Successors(dgl_id_t vid, uint64_t radius = 1) const = 0;

  /*!
   * \brief Get all edge ids between the two given endpoints
   * \note Edges are associated with an integer id start from zero.
   *       The id is assigned when the edge is being added to the graph.
   * \param src The source vertex.
   * \param dst The destination vertex.
   * \return the edge id array.
   */
  virtual IdArray EdgeId(dgl_id_t src, dgl_id_t dst) const = 0;

  /*!
   * \brief Get all edge ids between the given endpoint pairs.
   * \note Edges are associated with an integer id start from zero.
   *       The id is assigned when the edge is being added to the graph.
   *       If duplicate pairs exist, the returned edge IDs will also duplicate.
   *       The order of returned edge IDs will follow the order of src-dst pairs
   *       first, and ties are broken by the order of edge ID.
   * \return EdgeArray containing all edges between all pairs.
   */
  virtual EdgeArray EdgeIds(IdArray src, IdArray dst) const = 0;

  /*!
   * \brief Find the edge ID and return the pair of endpoints
   * \param eid The edge ID
   * \return a pair whose first element is the source and the second the destination.
   */
  virtual std::pair<dgl_id_t, dgl_id_t> FindEdge(dgl_id_t eid) const = 0;

  /*!
   * \brief Find the edge IDs and return their source and target node IDs.
   * \param eids The edge ID array.
   * \return EdgeArray containing all edges with id in eid.  The order is preserved.
   */
  virtual EdgeArray FindEdges(IdArray eids) const = 0;

  /*!
   * \brief Get the in edges of the vertex.
   * \note The returned dst id array is filled with vid.
   * \param vid The vertex id.
   * \return the edges
   */
  virtual EdgeArray InEdges(dgl_id_t vid) const = 0;

  /*!
   * \brief Get the in edges of the vertices.
   * \param vids The vertex id array.
   * \return the id arrays of the two endpoints of the edges.
   */
  virtual EdgeArray InEdges(IdArray vids) const = 0;

  /*!
   * \brief Get the out edges of the vertex.
   * \note The returned src id array is filled with vid.
   * \param vid The vertex id.
   * \return the id arrays of the two endpoints of the edges.
   */
  virtual EdgeArray OutEdges(dgl_id_t vid) const = 0;

  /*!
   * \brief Get the out edges of the vertices.
   * \param vids The vertex id array.
   * \return the id arrays of the two endpoints of the edges.
   */
  virtual EdgeArray OutEdges(IdArray vids) const = 0;

  /*!
   * \brief Get all the edges in the graph.
   * \note If order is "srcdst", the returned edges list is sorted by their src and
   *       dst ids. If order is "eid", they are in their edge id order.
   *       Otherwise, in the arbitrary order.
   * \param order The order of the returned edge list.
   * \return the id arrays of the two endpoints of the edges.
   */
  virtual EdgeArray Edges(const std::string &order = "") const = 0;

  /*!
   * \brief Get the in degree of the given vertex.
   * \param vid The vertex id.
   * \return the in degree
   */
  virtual uint64_t InDegree(dgl_id_t vid) const = 0;

  /*!
   * \brief Get the in degrees of the given vertices.
   * \param vid The vertex id array.
   * \return the in degree array
   */
  virtual DegreeArray InDegrees(IdArray vids) const = 0;

  /*!
   * \brief Get the out degree of the given vertex.
   * \param vid The vertex id.
   * \return the out degree
   */
  virtual uint64_t OutDegree(dgl_id_t vid) const = 0;

  /*!
   * \brief Get the out degrees of the given vertices.
   * \param vid The vertex id array.
   * \return the out degree array
   */
  virtual DegreeArray OutDegrees(IdArray vids) const = 0;

  /*!
   * \brief Construct the induced subgraph of the given vertices.
   *
   * The induced subgraph is a subgraph formed by specifying a set of vertices V' and then
   * selecting all of the edges from the original graph that connect two vertices in V'.
   *
   * Vertices and edges in the original graph will be "reindexed" to local index. The local
   * index of the vertices preserve the order of the given id array, while the local index
   * of the edges preserve the index order in the original graph. Vertices not in the
   * original graph are ignored.
   *
   * The result subgraph is read-only.
   *
   * \param vids The vertices in the subgraph.
   * \return the induced subgraph
   */
  virtual Subgraph VertexSubgraph(IdArray vids) const = 0;

  /*!
   * \brief Construct the induced edge subgraph of the given edges.
   *
   * The induced edges subgraph is a subgraph formed by specifying a set of edges E' and then
   * selecting all of the nodes from the original graph that are endpoints in E'.
   *
   * Vertices and edges in the original graph will be "reindexed" to local index. The local
   * index of the edges preserve the order of the given id array, while the local index
   * of the vertices preserve the index order in the original graph. Edges not in the
   * original graph are ignored.
   *
   * The result subgraph is read-only.
   *
   * \param eids The edges in the subgraph.
   * \param preserve_nodes If true, the vertices will not be relabeled, so some vertices
   *                       may have no incident edges.
   * \return the induced edge subgraph
   */
  virtual Subgraph EdgeSubgraph(IdArray eids, bool preserve_nodes = false) const = 0;

  /*!
   * \brief Return the successor vector
   * \param vid The vertex id.
   * \return the successor vector iterator pair.
   */
  virtual DGLIdIters SuccVec(dgl_id_t vid) const = 0;

  /*!
   * \brief Return the out edge id vector
   * \param vid The vertex id.
   * \return the out edge id vector iterator pair.
   */
  virtual DGLIdIters OutEdgeVec(dgl_id_t vid) const = 0;

  /*!
   * \brief Return the predecessor vector
   * \param vid The vertex id.
   * \return the predecessor vector iterator pair.
   */
  virtual DGLIdIters PredVec(dgl_id_t vid) const = 0;

  /*!
   * \brief Return the in edge id vector
   * \param vid The vertex id.
   * \return the in edge id vector iterator pair.
   */
  virtual DGLIdIters InEdgeVec(dgl_id_t vid) const = 0;

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
  virtual std::vector<IdArray> GetAdj(bool transpose, const std::string &fmt) const = 0;

  /*!
   * \brief Sort the columns in CSR.
   *
   * This sorts the columns in each row based on the column Ids.
   * The edge ids should be sorted accordingly.
   */
  virtual void SortCSR() {
  }

  static constexpr const char* _type_key = "graph.Graph";
  DGL_DECLARE_OBJECT_TYPE_INFO(GraphInterface, runtime::Object);
};

// Define GraphRef
DGL_DEFINE_OBJECT_REF(GraphRef, GraphInterface);

/*! \brief Subgraph data structure */
struct Subgraph : public runtime::Object {
  /*! \brief The graph. */
  GraphPtr graph;
  /*!
   * \brief The induced vertex ids.
   * \note This is also a map from the new vertex id to the vertex id in the parent graph.
   */
  IdArray induced_vertices;
  /*!
   * \brief The induced edge ids.
   * \note This is also a map from the new edge id to the edge id in the parent graph.
   */
  IdArray induced_edges;

  static constexpr const char* _type_key = "graph.Subgraph";
  DGL_DECLARE_OBJECT_TYPE_INFO(Subgraph, runtime::Object);
};

/*! \brief Subgraph data structure for negative subgraph */
struct NegSubgraph: public Subgraph {
  /*! \brief The existence of the negative edges in the parent graph. */
  IdArray exist;

  /*! \brief The Ids of head nodes */
  IdArray head_nid;

  /*! \brief The Ids of tail nodes */
  IdArray tail_nid;
};

// Define SubgraphRef
DGL_DEFINE_OBJECT_REF(SubgraphRef, Subgraph);

}  // namespace dgl

#endif  // DGL_GRAPH_INTERFACE_H_
