// DGL Graph interface
#ifndef DGL_DGLGRAPH_H_
#define DGL_DGLGRAPH_H_

#include <stdint.h>
#include "runtime/ndarray.h"
#include "./vector_view.h"

namespace dgl {

typedef uint64_t dgl_id_t;
typedef tvm::runtime::NDArray IdArray;
typedef tvm::runtime::NDArray DegreeArray;
typedef tvm::runtime::NDArray BoolArray;

class DGLGraph;
class DGLSubGraph;

/*!
 * \brief Base dgl graph class.
 *
 * DGLGraph is a directed graph. Vertices are integers enumerated from zero. Edges
 * are uniquely identified by the two endpoints. Multi-edge is currently not
 * supported.
 *
 * Removal of vertices/edges is not allowed. Instead, the graph can only be "cleared"
 * by removing all the vertices and edges.
 */
class DGLGraph {
 public:
   /*! \brief default constructor */
  DGLGraph() {}
  /*!
   * \brief Add vertices to the graph.
   * \note Since vertices are integers enumerated from zero, only the number of
   *       vertices to be added needs to be specified.
   * \param num_vertices The number of vertices to be added.
   */
  void AddVertices(uint64_t num_vertices);
  /*!
   * \brief Add one edge to the graph.
   * \param src The source vertex.
   * \param dst The destination vertex.
   */
  void AddEdge(dgl_id_t src, dgl_id_t dst);
  /*!
   * \brief Add edges to the graph.
   * \param src_ids The source vertex id array.
   * \param dst_ids The destination vertex id array.
   */
  void AddEdges(IdArray src_ids, IdArray dst_ids);
  /*!
   * \brief Clear the graph. Remove all vertices/edges.
   */
  void Clear();
  /*! \return the number of vertices in the graph.*/
  uint64_t NumVertices() const;
  /*! \return the number of edges in the graph.*/
  uint64_t NumEdges() const;
  /*! \return true if the given vertex is in the graph.*/
  bool HasVertex(dgl_id_t vid) const;
  /*! \return a 0-1 array indicating whether the given vertices are in the graph.*/
  BoolArray HasVertices(IdArray vids) const;
  /*! \return true if the given edge is in the graph.*/
  bool HasEdge(dgl_id_t src, dgl_id_t dst) const;
  /*! \return a 0-1 array indicating whether the given edges are in the graph.*/
  BoolArray HasEdges(IdArray src_ids, IdArray dst_ids) const;
  /*!
   * \brief Find the predecessors of a vertex.
   * \param vid The vertex id.
   * \return the predecessor id array.
   */
  IdArray Predecessors(dgl_id_t vid) const;
  /*!
   * \brief Find the successors of a vertex.
   * \param vid The vertex id.
   * \return the successor id array.
   */
  IdArray Successors(dgl_id_t vid) const;
  /*!
   * \brief Get the id of the given edge.
   * \note Edges are associated with an integer id start from zero.
   *       The id is assigned when the edge is being added to the graph.
   * \param src The source vertex.
   * \param dst The destination vertex.
   * \return the edge id.
   */
  dgl_id_t EdgeId(dgl_id_t src, dgl_id_t dst) const;
  /*!
   * \brief Get the id of the given edges.
   * \note Edges are associated with an integer id start from zero.
   *       The id is assigned when the edge is being added to the graph.
   * \return the edge id array.
   */
  IdArray EdgeIds(IdArray src, IdArray dst) const;
  /*!
   * \brief Get the in edges of the vertex.
   * \param vid The vertex id.
   * \return the id arrays of the two endpoints of the edges.
   */
  std::pair<IdArray, IdArray> InEdges(dgl_id_t vid) const;
  /*!
   * \brief Get the in edges of the vertices.
   * \param vids The vertex id array.
   * \return the id arrays of the two endpoints of the edges.
   */
  std::pair<IdArray, IdArray> InEdges(IdArray vids) const;
  /*!
   * \brief Get the out edges of the vertex.
   * \param vid The vertex id.
   * \return the id arrays of the two endpoints of the edges.
   */
  std::pair<IdArray, IdArray> OutEdges(dgl_id_t vid) const;
  /*!
   * \brief Get the out edges of the vertices.
   * \param vids The vertex id array.
   * \return the id arrays of the two endpoints of the edges.
   */
  std::pair<IdArray, IdArray> OutEdges(IdArray vids) const;
  /*!
   * \brief Get all the edges in the graph.
   * \return the id arrays of the two endpoints of the edges.
   */
  std::pair<IdArray, IdArray> Edges() const;
  /*!
   * \brief Get the in degree of the given vertex.
   * \param vid The vertex id.
   * \return the in degree
   */
  uint64_t InDegree(dgl_id_t vid) const;
  /*!
   * \brief Get the in degrees of the given vertices.
   * \param vid The vertex id array.
   * \return the in degree array
   */
  DegreeArray InDegrees(IdArray vids) const;
  /*!
   * \brief Get the out degree of the given vertex.
   * \param vid The vertex id.
   * \return the out degree
   */
  uint64_t OutDegree(dgl_id_t vid) const;
  /*!
   * \brief Get the out degrees of the given vertices.
   * \param vid The vertex id array.
   * \return the out degree array
   */
  DegreeArray OutDegrees(IdArray vids) const;
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
  DGLGraph Subgraph(IdArray vids) const;
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
   * \param vids The edges in the subgraph.
   * \return the induced edge subgraph
   */
  DGLGraph EdgeSubgraph(IdArray src, IdArray dst) const;
  /*!
   * \brief Return a new graph with all the edges reversed.
   *
   * The returned graph preserves the vertex and edge index in the original graph.
   *
   * \return the reversed graph
   */
  DGLGraph Reverse() const;

 private:
  /*! \brief Internal edge list type */
  struct EdgeList {
    /*! \brief successor vertex list */
    vector_view<dgl_id_t> succ;
    /*! \brief predecessor vertex list */
    vector_view<dgl_id_t> pred;
    /*! \brief (local) edge id property */
    std::vector<dgl_id_t> edge_id;
  };
  /*! \brief Adjacency list using vector storage */
  // TODO(minjie): adjacent list is good for graph mutation and finding pred/succ.
  // It is not good for getting all the edges of the graph. If the graph is known
  // to be static, how to design a data structure to speed this up? This idea can
  // be further extended. For example, CSC/CSR graph storage is known to be more
  // compact than adjlist, but is difficult to be mutated. Shall we switch to a CSR/CSC
  // graph structure at some point? When shall such conversion happen? Which one
  // will more likely to be a bottleneck? memory or computation?
  vector_view<EdgeList> adjlist_;
  /*! \brief read only flag */
  bool read_only_{false};
};

}  // namespace dgl

#endif  // DGL_DGLGRAPH_H_
