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
#include "runtime/ndarray.h"

namespace dgl {

typedef uint64_t dgl_id_t;
typedef dgl::runtime::NDArray IdArray;
typedef dgl::runtime::NDArray DegreeArray;
typedef dgl::runtime::NDArray BoolArray;
typedef dgl::runtime::NDArray IntArray;
typedef dgl::runtime::NDArray FloatArray;

struct Subgraph;
struct NodeFlow;

const dgl_id_t DGL_INVALID_ID = static_cast<dgl_id_t>(-1);

/*!
 * \brief This class references data in std::vector.
 *
 * This isn't a STL-style iterator. It provides a STL data container interface.
 * but it doesn't own data itself. instead, it only references data in std::vector.
 */
class DGLIdIters {
  const dgl_id_t *begin_, *end_;
 public:
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
};

class GraphInterface;
typedef std::shared_ptr<GraphInterface> GraphPtr;

/*!
 * \brief dgl graph index interface.
 *
 * DGL's graph is directed. Vertices are integers enumerated from zero.
 */
class GraphInterface {
 public:
  /* \brief structure used to represent a list of edges */
  typedef struct {
    /* \brief the two endpoints and the id of the edge */
    IdArray src, dst, id;
  } EdgeArray;

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
   * \note not const since we have caches
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
  virtual bool HasVertex(dgl_id_t vid) const = 0;

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
   * \return the induced edge subgraph
   */
  virtual Subgraph EdgeSubgraph(IdArray eids) const = 0;

  /*!
   * \brief Return a new graph with all the edges reversed.
   *
   * The returned graph preserves the vertex and edge index in the original graph.
   *
   * \return the reversed graph
   */
  virtual GraphPtr Reverse() const = 0;

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
   * \brief Reset the data in the graph and move its data to the returned graph object.
   * \return a raw pointer to the graph object.
   */
  virtual GraphInterface *Reset() = 0;

  /*!
   * \brief Get the adjacency matrix of the graph.
   *
   * By default, a row of returned adjacency matrix represents the destination
   * of an edge and the column represents the source.
   * \param transpose A flag to transpose the returned adjacency matrix.
   * \param fmt the format of the returned adjacency matrix.
   * \return a vector of IdArrays.
   */
  virtual std::vector<IdArray> GetAdj(bool transpose, const std::string &fmt) const = 0;
};

/*! \brief Subgraph data structure */
struct Subgraph {
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
};

}  // namespace dgl

#endif  // DGL_GRAPH_INTERFACE_H_
