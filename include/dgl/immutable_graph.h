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

struct ImmutableSubgraph;

/*!
 * \brief Base dgl immutable graph index class.
 *
 */
class ImmutableGraph {
 public:
  typedef struct {
    /* \brief the two endpoints and the id of the edge */
    IdArray src, dst, id;
  } EdgeArray;

  struct csr {
    std::vector<int64_t> indptr;
    std::vector<dgl_id_t> indices;
    std::vector<dgl_id_t> edge_ids;

    csr(int64_t num_vertices, int64_t expected_num_edges) {
      indptr.resize(num_vertices + 1);
      indices.reserve(expected_num_edges);
      edge_ids.reserve(expected_num_edges);
    }

    bool HasVertex(dgl_id_t vid) const {
      return vid < NumVertices();
    }

    uint64_t NumVertices() const {
      return indptr.size() - 1;
    }

    uint64_t NumEdges() const {
      return indices.size();
    }

    int64_t GetDegree(dgl_id_t vid) const {
      return indptr[vid + 1] - indptr[vid];
    }
    DegreeArray GetDegrees(IdArray vids) const;
    EdgeArray GetEdges(dgl_id_t vid) const;
    EdgeArray GetEdges(IdArray vids) const;
    std::pair<const dgl_id_t *, const dgl_id_t *> GetIndexRef(dgl_id_t v) const {
      int64_t start = indptr[v];
      int64_t end = indptr[v + 1];
      return std::pair<const dgl_id_t *, const dgl_id_t *>(&indices[start], &indices[end]);
    }
    std::shared_ptr<csr> Transpose() const;
    std::pair<std::shared_ptr<csr>, IdArray> VertexSubgraph(IdArray vids) const;
  };

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
    if (in_csr_)
      return in_csr_->NumVertices();
    else
      return out_csr_->NumVertices();
  }

  /*! \return the number of edges in the graph.*/
  uint64_t NumEdges() const {
    if (in_csr_)
      return in_csr_->NumEdges();
    else
      return out_csr_->NumEdges();
  }

  /*! \return true if the given vertex is in the graph.*/
  bool HasVertex(dgl_id_t vid) const {
    return vid < NumVertices();
  }

  /*! \return a 0-1 array indicating whether the given vertices are in the graph.*/
  BoolArray HasVertices(IdArray vids) const;

  /*! \return true if the given edge is in the graph.*/
  bool HasEdgeBetween(dgl_id_t src, dgl_id_t dst) const {
    if (!HasVertex(src) || !HasVertex(dst)) return false;
    if (this->in_csr_) {
      auto pred = this->in_csr_->GetIndexRef(dst);
      return std::binary_search(pred.first, pred.second, src);
    } else {
      assert(this->out_csr_);
      auto succ = this->out_csr_->GetIndexRef(src);
      return std::binary_search(succ.first, succ.second, dst);
    }
  }

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
   * \brief Get the in edges of the vertex.
   * \note The returned dst id array is filled with vid.
   * \param vid The vertex id.
   * \return the edges
   */
  EdgeArray InEdges(dgl_id_t vid) const {
    return this->GetInCSR()->GetEdges(vid);
  }

  /*!
   * \brief Get the in edges of the vertices.
   * \param vids The vertex id array.
   * \return the id arrays of the two endpoints of the edges.
   */
  EdgeArray InEdges(IdArray vids) const {
    return this->GetInCSR()->GetEdges(vids);
  }

  /*!
   * \brief Get the out edges of the vertex.
   * \note The returned src id array is filled with vid.
   * \param vid The vertex id.
   * \return the id arrays of the two endpoints of the edges.
   */
  EdgeArray OutEdges(dgl_id_t vid) const {
    return this->GetOutCSR()->GetEdges(vid);
  }

  /*!
   * \brief Get the out edges of the vertices.
   * \param vids The vertex id array.
   * \return the id arrays of the two endpoints of the edges.
   */
  EdgeArray OutEdges(IdArray vids) const {
    return this->GetOutCSR()->GetEdges(vids);
  }

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
    return this->GetInCSR()->GetDegree(vid);
  }

  /*!
   * \brief Get the in degrees of the given vertices.
   * \param vid The vertex id array.
   * \return the in degree array
   */
  DegreeArray InDegrees(IdArray vids) const {
    return this->GetInCSR()->GetDegrees(vids);
  }

  /*!
   * \brief Get the out degree of the given vertex.
   * \param vid The vertex id.
   * \return the out degree
   */
  uint64_t OutDegree(dgl_id_t vid) const {
    CHECK(HasVertex(vid)) << "invalid vertex: " << vid;
    return this->GetOutCSR()->GetDegree(vid);
  }

  /*!
   * \brief Get the out degrees of the given vertices.
   * \param vid The vertex id array.
   * \return the out degree array
   */
  DegreeArray OutDegrees(IdArray vids) const {
    return this->GetOutCSR()->GetDegrees(vids);
  }

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
  ImmutableSubgraph VertexSubgraph(IdArray vids) const;

  std::vector<ImmutableSubgraph> VertexSubgraphs(const std::vector<IdArray> &vids) const;

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
  ImmutableSubgraph EdgeSubgraph(IdArray eids) const;

  std::vector<ImmutableSubgraph> EdgeSubgraphs(std::vector<IdArray> eids) const;

  /*!
   * \brief Return a new graph with all the edges reversed.
   *
   * The returned graph preserves the vertex and edge index in the original graph.
   *
   * \return the reversed graph
   */
  ImmutableGraph Reverse() const {
    return ImmutableGraph(out_csr_, in_csr_, is_multigraph_);
  }

  /*!
   * \brief Return the successor vector
   * \param vid The vertex id.
   * \return the successor vector
   */
  std::vector<dgl_id_t> SuccVec(dgl_id_t vid) const {
    return std::vector<dgl_id_t>(out_csr_->indices.begin() + out_csr_->indptr[vid],
                                 out_csr_->indices.begin() + out_csr_->indptr[vid + 1]);
  }

  /*!
   * \brief Return the out edge id vector
   * \param vid The vertex id.
   * \return the out edge id vector
   */
  std::vector<dgl_id_t> OutEdgeVec(dgl_id_t vid) const {
    return std::vector<dgl_id_t>(out_csr_->edge_ids.begin() + out_csr_->indptr[vid],
                                 out_csr_->edge_ids.begin() + out_csr_->indptr[vid + 1]);
  }

  /*!
   * \brief Return the predecessor vector
   * \param vid The vertex id.
   * \return the predecessor vector
   */
  std::vector<dgl_id_t> PredVec(dgl_id_t vid) const {
    return std::vector<dgl_id_t>(in_csr_->indices.begin() + in_csr_->indptr[vid],
                                 in_csr_->indices.begin() + in_csr_->indptr[vid + 1]);
  }

  /*!
   * \brief Return the in edge id vector
   * \param vid The vertex id.
   * \return the in edge id vector
   */
  std::vector<dgl_id_t> InEdgeVec(dgl_id_t vid) const {
    return std::vector<dgl_id_t>(in_csr_->edge_ids.begin() + in_csr_->indptr[vid],
                                 in_csr_->edge_ids.begin() + in_csr_->indptr[vid + 1]);
  }

 protected:
  std::pair<const dgl_id_t *, const dgl_id_t *> GetInEdgeIdRef(dgl_id_t src, dgl_id_t dst) const;
  std::pair<const dgl_id_t *, const dgl_id_t *> GetOutEdgeIdRef(dgl_id_t src, dgl_id_t dst) const;

  /*
   * When we get in csr or out csr, we try to get the one cached in the structure.
   * If not, we transpose the other one to get the one we need.
   */
  std::shared_ptr<csr> GetInCSR() const {
    if (in_csr_) {
      return in_csr_;
    } else {
      assert(out_csr_ != nullptr);
      const_cast<ImmutableGraph *>(this)->in_csr_ = out_csr_->Transpose();
      return in_csr_;
    }
  }
  std::shared_ptr<csr> GetOutCSR() const {
    if (out_csr_) {
      return out_csr_;
    } else {
      assert(in_csr_ != nullptr);
      const_cast<ImmutableGraph *>(this)->out_csr_ = in_csr_->Transpose();
      return out_csr_;
    }
  }

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
struct ImmutableSubgraph {
  /*! \brief The graph. */
  ImmutableGraph graph;
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

