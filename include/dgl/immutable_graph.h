/*!
 *  Copyright (c) 2018 by Contributors
 * \file dgl/graph.h
 * \brief DGL immutable graph index class.
 */
#ifndef DGL_IMMUTABLE_GRAPH_H_
#define DGL_IMMUTABLE_GRAPH_H_

#include <vector>
#include <cstdint>
#include <utility>
#include <tuple>
#include "runtime/ndarray.h"
#include "graph.h"

namespace dgl {

/*!
 * \brief Base dgl immutable graph index class.
 *
 */
class ImmutableGraph {
 public:
  typedef struct {
    std::vector<int64_t> indptr;
    std::vector<int64_t> indices;
    std::vector<int64_t> edge_ids;
  } csr;

  ImmutableGraph(std::shared_ptr<csr> in_csr, std::shared_ptr<csr> out_csr,
      bool multigraph = false) : is_multigraph_(multigraph) {
    this->in_csr_ = in_csr;
    this->out_csr_ = out_csr;
  }

  /*! \brief default constructor */
  explicit ImmutableGraph(bool multigraph = false) : is_multigraph_(multigraph) {}

  /*! \brief default copy constructor */
  ImmutableGraph(const ImmutableGraph& other) = default;

#ifndef _MSC_VER
  /*! \brief default move constructor */
  ImmutableGraph(ImmutableGraph&& other) = default;
#else
  ImmutableGraph(ImmutableGraph&& other) {
    // TODO
  }
#endif  // _MSC_VER

  /*! \brief default assign constructor */
  ImmutableGraph& operator=(const ImmutableGraph& other) = default;

  /*! \brief default destructor */
  ~ImmutableGraph() = default;

  /*!
   * \note not const since we have caches
   * \return whether the graph is a multigraph
   */
  bool IsMultigraph() const {
    return is_multigraph_;
  }

  /*! \return the number of vertices in the graph.*/
  uint64_t NumVertices() const {
    return in_csr_->indptr.size() - 1;
  }

  /*! \return the number of edges in the graph.*/
  uint64_t NumEdges() const {
    return in_csr_->indices.size();
  }

  /*! \return true if the given vertex is in the graph.*/
  bool HasVertex(dgl_id_t vid) const {
    return vid < NumVertices();
  }

  /*! \return a 0-1 array indicating whether the given vertices are in the graph.*/
  BoolArray HasVertices(IdArray vids) const;

  /*! \return true if the given edge is in the graph.*/
  bool HasEdgeBetween(dgl_id_t src, dgl_id_t dst) const;

  /*! \return a 0-1 array indicating whether the given edges are in the graph.*/
  BoolArray HasEdgesBetween(IdArray src_ids, IdArray dst_ids) const;

  /*!
   * \brief Find the predecessors of a vertex.
   * \param vid The vertex id.
   * \param radius The radius of the neighborhood. Default is immediate neighbor (radius=1).
   * \return the predecessor id array.
   */
  IdArray Predecessors(dgl_id_t vid, uint64_t radius = 1) const;

  /*!
   * \brief Find the successors of a vertex.
   * \param vid The vertex id.
   * \param radius The radius of the neighborhood. Default is immediate neighbor (radius=1).
   * \return the successor id array.
   */
  IdArray Successors(dgl_id_t vid, uint64_t radius = 1) const;

  /*!
   * \brief Get all edge ids between the two given endpoints
   * \note Edges are associated with an integer id start from zero.
   *       The id is assigned when the edge is being added to the graph.
   * \param src The source vertex.
   * \param dst The destination vertex.
   * \return the edge id array.
   */
  IdArray EdgeId(dgl_id_t src, dgl_id_t dst) const;

  /*!
   * \brief Get all edge ids between the given endpoint pairs.
   * \note Edges are associated with an integer id start from zero.
   *       The id is assigned when the edge is being added to the graph.
   *       If duplicate pairs exist, the returned edge IDs will also duplicate.
   *       The order of returned edge IDs will follow the order of src-dst pairs
   *       first, and ties are broken by the order of edge ID.
   * \return EdgeArray containing all edges between all pairs.
   */
  EdgeArray EdgeIds(IdArray src, IdArray dst) const;

  /*!
   * \brief Find the edge ID and return the pair of endpoints
   * \param eid The edge ID
   * \return a pair whose first element is the source and the second the destination.
   */
  std::pair<dgl_id_t, dgl_id_t> FindEdge(dgl_id_t eid) const {
    dgl_id_t src_id = in_csr_->indices[eid];
    auto it = std::lower_bound(in_csr-->indptr.begin(), in_csr_->indptr.end(), eid);
    assert(it != in_csr_->indptr.end());
    dgl_id_t dst_id;
    if (*it == eid)
      dst_id = it - in_csr_->indptr.begin();
    else
      dst_id = it - in_csr_->indptr.begin() - 1;
    assert(in_csr_->edge_ids[in_csr_->indptr[dst_id]] <= eid);
    assert(dst_id >= 0);
    return std::make_pair(src_id, dst_id);
  }

  /*!
   * \brief Find the edge IDs and return their source and target node IDs.
   * \param eids The edge ID array.
   * \return EdgeArray containing all edges with id in eid.  The order is preserved.
   */
  EdgeArray FindEdges(IdArray eids) const;

  /*!
   * \brief Get the in edges of the vertex.
   * \note The returned dst id array is filled with vid.
   * \param vid The vertex id.
   * \return the edges
   */
  EdgeArray InEdges(dgl_id_t vid) const;

  /*!
   * \brief Get the in edges of the vertices.
   * \param vids The vertex id array.
   * \return the id arrays of the two endpoints of the edges.
   */
  EdgeArray InEdges(IdArray vids) const;

  /*!
   * \brief Get the out edges of the vertex.
   * \note The returned src id array is filled with vid.
   * \param vid The vertex id.
   * \return the id arrays of the two endpoints of the edges.
   */
  EdgeArray OutEdges(dgl_id_t vid) const;

  /*!
   * \brief Get the out edges of the vertices.
   * \param vids The vertex id array.
   * \return the id arrays of the two endpoints of the edges.
   */
  EdgeArray OutEdges(IdArray vids) const;

  /*!
   * \brief Get all the edges in the graph.
   * \note If sorted is true, the returned edges list is sorted by their src and
   *       dst ids. Otherwise, they are in their edge id order.
   * \param sorted Whether the returned edge list is sorted by their src and dst ids
   * \return the id arrays of the two endpoints of the edges.
   */
  EdgeArray Edges(bool sorted = false) const;

  /*!
   * \brief Get the in degree of the given vertex.
   * \param vid The vertex id.
   * \return the in degree
   */
  uint64_t InDegree(dgl_id_t vid) const {
    CHECK(HasVertex(vid)) << "invalid vertex: " << vid;
    return in_csr_->indptr[vid + 1] - in_csr_->indptr[vid];
  }

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
  uint64_t OutDegree(dgl_id_t vid) const {
    CHECK(HasVertex(vid)) << "invalid vertex: " << vid;
    return out_csr_->indptr[vid + 1] - out_csr_->indptr[vid];
  }

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
  Subgraph VertexSubgraph(IdArray vids) const;

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
  Subgraph EdgeSubgraph(IdArray eids) const;

  /*!
   * \brief Return a new graph with all the edges reversed.
   *
   * The returned graph preserves the vertex and edge index in the original graph.
   *
   * \return the reversed graph
   */
  ImmutableGraph Reverse() const {
    return ImmutableGraph(out_csr, in_csr, is_multigraph_);
  }

  /*!
   * \brief Return the successor vector
   * \param vid The vertex id.
   * \return the successor vector
   */
  const std::vector<dgl_id_t>& SuccVec(dgl_id_t vid) const {
    return adjlist_[vid].succ;
  }

  /*!
   * \brief Return the out edge id vector
   * \param vid The vertex id.
   * \return the out edge id vector
   */
  const std::vector<dgl_id_t>& OutEdgeVec(dgl_id_t vid) const {
    return adjlist_[vid].edge_id;
  }

  /*!
   * \brief Return the predecessor vector
   * \param vid The vertex id.
   * \return the predecessor vector
   */
  const std::vector<dgl_id_t>& PredVec(dgl_id_t vid) const {
    return reverse_adjlist_[vid].succ;
  }

  /*!
   * \brief Return the in edge id vector
   * \param vid The vertex id.
   * \return the in edge id vector
   */
  const std::vector<dgl_id_t>& InEdgeVec(dgl_id_t vid) const {
    return reverse_adjlist_[vid].edge_id;
  }

 protected:
  friend class GraphOp;
  // Store the in-edges.
  std::shared_ptr<csr> in_csr_;
  // Store the out-edges.
  std::shared_ptr<csr> out_csr_;
  /*!
   * \brief Whether if this is a multigraph.
   *
   * When a multiedge is added, this flag switches to true.
   */
  bool is_multigraph_ = false;
};

/*! \brief Subgraph data structure */
struct Subgraph {
  /*! \brief The graph. */
  Graph graph;
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

#endif  // DGL_IMMUTABLE_GRAPH_H_

