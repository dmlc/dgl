/*!
 *  Copyright (c) 2018 by Contributors
 * \file dgl/immutable_graph.h
 * \brief DGL immutable graph index class.
 */
#ifndef DGL_IMMUTABLE_GRAPH_H_
#define DGL_IMMUTABLE_GRAPH_H_

#include <vector>
#include <string>
#include <cstdint>
#include <utility>
#include <tuple>
#include <algorithm>
#include "runtime/ndarray.h"
#include "graph_interface.h"
#include "lazy.h"

namespace dgl {

class CSR;
class COO;
typedef std::shared_ptr<CSR> CSRPtr;
typedef std::shared_ptr<COO> COOPtr;

/*!
 * \brief Graph class stored using CSR structure.
 */
class CSR : public GraphInterface {
 public:
  // Create a csr graph that has the given number of verts and edges.
  CSR(int64_t num_vertices, int64_t num_edges, bool is_multigraph);
  // Create a csr graph whose memory is stored in the shared memory
  //   that has the given number of verts and edges.
  CSR(const std::string &shared_mem_name,
      int64_t num_vertices, int64_t num_edges, bool is_multigraph);

  // Create a csr graph that shares the given indptr and indices.
  CSR(IdArray indptr, IdArray indices, IdArray edge_ids);
  CSR(IdArray indptr, IdArray indices, IdArray edge_ids, bool is_multigraph);

  // Create a csr graph by data iterator
  template <typename IndptrIter, typename IndicesIter, typename EdgeIdIter>
  CSR(int64_t num_vertices, int64_t num_edges,
      IndptrIter indptr_begin, IndicesIter indices_begin, EdgeIdIter edge_ids_begin,
      bool is_multigraph);

  // Create a csr graph whose memory is stored in the shared memory
  //   and the structure is given by the indptr and indcies.
  CSR(IdArray indptr, IdArray indices, IdArray edge_ids,
      const std::string &shared_mem_name);
  CSR(IdArray indptr, IdArray indices, IdArray edge_ids, bool is_multigraph,
      const std::string &shared_mem_name);

  void AddVertices(uint64_t num_vertices) override {
    LOG(FATAL) << "CSR graph does not allow mutation.";
  }

  void AddEdge(dgl_id_t src, dgl_id_t dst) override {
    LOG(FATAL) << "CSR graph does not allow mutation.";
  }

  void AddEdges(IdArray src_ids, IdArray dst_ids) override {
    LOG(FATAL) << "CSR graph does not allow mutation.";
  }

  void Clear() override {
    LOG(FATAL) << "CSR graph does not allow mutation.";
  }

  bool IsMultigraph() const override;

  bool IsReadonly() const override {
    return true;
  }

  uint64_t NumVertices() const override {
    return indptr_->shape[0] - 1;
  }

  uint64_t NumEdges() const override {
    return indices_->shape[0];
  }

  bool HasVertex(dgl_id_t vid) const override {
    return vid < NumVertices() && vid >= 0;
  }

  bool HasEdgeBetween(dgl_id_t src, dgl_id_t dst) const override;

  IdArray Predecessors(dgl_id_t vid, uint64_t radius = 1) const override {
    LOG(FATAL) << "CSR graph does not support efficient predecessor query."
      << " Please use successors on the reverse CSR graph.";
    return {};
  }

  IdArray Successors(dgl_id_t vid, uint64_t radius = 1) const override;

  IdArray EdgeId(dgl_id_t src, dgl_id_t dst) const override;

  EdgeArray EdgeIds(IdArray src, IdArray dst) const override;

  std::pair<dgl_id_t, dgl_id_t> FindEdge(dgl_id_t eid) const override {
    LOG(FATAL) << "CSR graph does not support efficient FindEdge."
      << " Please use COO graph.";
    return {};
  }

  EdgeArray FindEdges(IdArray eids) const override {
    LOG(FATAL) << "CSR graph does not support efficient FindEdges."
      << " Please use COO graph.";
    return {};
  }

  EdgeArray InEdges(dgl_id_t vid) const override {
    LOG(FATAL) << "CSR graph does not support efficient inedges query."
      << " Please use outedges on the reverse CSR graph.";
    return {};
  }

  EdgeArray InEdges(IdArray vids) const override {
    LOG(FATAL) << "CSR graph does not support efficient inedges query."
      << " Please use outedges on the reverse CSR graph.";
    return {};
  }

  EdgeArray OutEdges(dgl_id_t vid) const override;

  EdgeArray OutEdges(IdArray vids) const override;

  EdgeArray Edges(const std::string &order = "") const override;

  uint64_t InDegree(dgl_id_t vid) const override {
    LOG(FATAL) << "CSR graph does not support efficient indegree query."
      << " Please use outdegree on the reverse CSR graph.";
    return 0;
  }

  DegreeArray InDegrees(IdArray vids) const override {
    LOG(FATAL) << "CSR graph does not support efficient indegree query."
      << " Please use outdegree on the reverse CSR graph.";
    return {};
  }

  uint64_t OutDegree(dgl_id_t vid) const override {
    const int64_t* indptr_data = static_cast<int64_t*>(indptr_->data);
    return indptr_data[vid + 1] - indptr_data[vid];
  }

  DegreeArray OutDegrees(IdArray vids) const override;

  Subgraph VertexSubgraph(IdArray vids) const override;

  Subgraph EdgeSubgraph(IdArray eids) const override {
    LOG(FATAL) << "CSR graph does not support efficient EdgeSubgraph."
      << " Please use COO graph instead.";
    return {};
  }

  GraphPtr Reverse() const override {
    return Transpose();
  }

  DGLIdIters SuccVec(dgl_id_t vid) const override {
    const dgl_id_t* indptr_data = static_cast<dgl_id_t*>(indptr_->data);
    const dgl_id_t* indices_data = static_cast<dgl_id_t*>(indices_->data);
    const dgl_id_t start = indptr_data[vid];
    const dgl_id_t end = indptr_data[vid + 1];
    return DGLIdIters(indices_data + start, indices_data + end);
  }

  DGLIdIters OutEdgeVec(dgl_id_t vid) const override {
    const dgl_id_t* indptr_data = static_cast<dgl_id_t*>(indptr_->data);
    const dgl_id_t* eid_data = static_cast<dgl_id_t*>(edge_ids_->data);
    const dgl_id_t start = indptr_data[vid];
    const dgl_id_t end = indptr_data[vid + 1];
    return DGLIdIters(eid_data + start, eid_data + end);
  }

  DGLIdIters PredVec(dgl_id_t vid) const override {
    LOG(FATAL) << "CSR graph does not support efficient PredVec."
      << " Please use SuccVec on the reverse CSR graph.";
    return DGLIdIters(nullptr, nullptr);
  }

  DGLIdIters InEdgeVec(dgl_id_t vid) const override {
    LOG(FATAL) << "CSR graph does not support efficient InEdgeVec."
      << " Please use OutEdgeVec on the reverse CSR graph.";
    return DGLIdIters(nullptr, nullptr);
  }

  GraphInterface *Reset() override {
    CSR* gptr = new CSR();
    *gptr = std::move(*this);
    return gptr;
  }

  std::vector<IdArray> GetAdj(bool transpose, const std::string &fmt) const override {
    CHECK(!transpose && fmt == "csr") << "Not valid adj format request.";
    return {indptr_, indices_, edge_ids_};
  }

  /*! \brief Return the reverse of this CSR graph (i.e, a CSC graph) */
  CSRPtr Transpose() const;

  /*! \brief Convert this CSR to COO */
  COOPtr ToCOO() const;

  /*!
   * \return the csr matrix that represents this graph.
   * \note The csr matrix shares the storage with this graph.
   *       The data field of the CSR matrix stores the edge ids.
   */
  CSRMatrix ToCSRMatrix() const {
    return CSRMatrix{indptr_, indices_, edge_ids_};
  }

  // member getters

  IdArray indptr() const { return indptr_; }

  IdArray indices() const { return indices_; }

  IdArray edge_ids() const { return edge_ids_; }

 private:
  /*! \brief prive default constructor */
  CSR() {}

  // The CSR arrays.
  //  - The index is 0-based.
  //  - The out edges of vertex v is stored from `indices_[indptr_[v]]` to
  //    `indices_[indptr_[v+1]]` (exclusive).
  //  - The indices are *not* necessarily sorted.
  // TODO(minjie): in the future, we should separate CSR and graph. A general CSR
  //   is not necessarily a graph, but graph operations could be implemented by
  //   CSR matrix operations. CSR matrix operations would be backed by different
  //   devices (CPU, CUDA, ...), while graph interface will not be aware of that.
  IdArray indptr_, indices_, edge_ids_;

  // whether the graph is a multi-graph
  LazyObject<bool> is_multigraph_;
};

class COO : public GraphInterface {
 public:
  // Create a coo graph that shares the given src and dst
  COO(int64_t num_vertices, IdArray src, IdArray dst);
  COO(int64_t num_vertices, IdArray src, IdArray dst, bool is_multigraph);

  // TODO(da): add constructor for creating COO from shared memory

  void AddVertices(uint64_t num_vertices) override {
    LOG(FATAL) << "CSR graph does not allow mutation.";
  }

  void AddEdge(dgl_id_t src, dgl_id_t dst) override {
    LOG(FATAL) << "CSR graph does not allow mutation.";
  }

  void AddEdges(IdArray src_ids, IdArray dst_ids) override {
    LOG(FATAL) << "CSR graph does not allow mutation.";
  }

  void Clear() override {
    LOG(FATAL) << "CSR graph does not allow mutation.";
  }

  bool IsMultigraph() const override;

  bool IsReadonly() const override {
    return true;
  }

  uint64_t NumVertices() const override {
    return num_vertices_;
  }

  uint64_t NumEdges() const override {
    return src_->shape[0];
  }

  bool HasVertex(dgl_id_t vid) const override {
    return vid < NumVertices() && vid >= 0;
  }

  bool HasEdgeBetween(dgl_id_t src, dgl_id_t dst) const override {
    LOG(FATAL) << "COO graph does not support efficient HasEdgeBetween."
      << " Please use CSR graph or AdjList graph instead.";
    return false;
  }

  IdArray Predecessors(dgl_id_t vid, uint64_t radius = 1) const override {
    LOG(FATAL) << "COO graph does not support efficient Predecessors."
      << " Please use CSR graph or AdjList graph instead.";
    return {};
  }

  IdArray Successors(dgl_id_t vid, uint64_t radius = 1) const override {
    LOG(FATAL) << "COO graph does not support efficient Successors."
      << " Please use CSR graph or AdjList graph instead.";
    return {};
  }

  IdArray EdgeId(dgl_id_t src, dgl_id_t dst) const override {
    LOG(FATAL) << "COO graph does not support efficient EdgeId."
      << " Please use CSR graph or AdjList graph instead.";
    return {};
  }

  EdgeArray EdgeIds(IdArray src, IdArray dst) const override {
    LOG(FATAL) << "COO graph does not support efficient EdgeId."
      << " Please use CSR graph or AdjList graph instead.";
    return {};
  }

  std::pair<dgl_id_t, dgl_id_t> FindEdge(dgl_id_t eid) const override {
    CHECK(eid < NumEdges() && eid >= 0) << "Invalid edge id: " << eid;
    const dgl_id_t* src_data = static_cast<dgl_id_t*>(src_->data);
    const dgl_id_t* dst_data = static_cast<dgl_id_t*>(dst_->data);
    return std::make_pair(src_data[eid], dst_data[eid]);
  }

  EdgeArray FindEdges(IdArray eids) const override;

  EdgeArray InEdges(dgl_id_t vid) const override {
    LOG(FATAL) << "COO graph does not support efficient InEdges."
      << " Please use CSR graph or AdjList graph instead.";
    return {};
  }

  EdgeArray InEdges(IdArray vids) const override {
    LOG(FATAL) << "COO graph does not support efficient InEdges."
      << " Please use CSR graph or AdjList graph instead.";
    return {};
  }

  EdgeArray OutEdges(dgl_id_t vid) const override {
    LOG(FATAL) << "COO graph does not support efficient OutEdges."
      << " Please use CSR graph or AdjList graph instead.";
    return {};
  }

  EdgeArray OutEdges(IdArray vids) const override {
    LOG(FATAL) << "COO graph does not support efficient OutEdges."
      << " Please use CSR graph or AdjList graph instead.";
    return {};
  }

  EdgeArray Edges(const std::string &order = "") const override;

  uint64_t InDegree(dgl_id_t vid) const override {
    LOG(FATAL) << "COO graph does not support efficient InDegree."
      << " Please use CSR graph or AdjList graph instead.";
    return 0;
  }

  DegreeArray InDegrees(IdArray vids) const override {
    LOG(FATAL) << "COO graph does not support efficient InDegrees."
      << " Please use CSR graph or AdjList graph instead.";
    return {};
  }

  uint64_t OutDegree(dgl_id_t vid) const override {
    LOG(FATAL) << "COO graph does not support efficient OutDegree."
      << " Please use CSR graph or AdjList graph instead.";
    return 0;
  }

  DegreeArray OutDegrees(IdArray vids) const override {
    LOG(FATAL) << "COO graph does not support efficient OutDegrees."
      << " Please use CSR graph or AdjList graph instead.";
    return {};
  }

  Subgraph VertexSubgraph(IdArray vids) const override {
    LOG(FATAL) << "COO graph does not support efficient VertexSubgraph."
      << " Please use CSR graph or AdjList graph instead.";
    return {};
  }

  Subgraph EdgeSubgraph(IdArray eids) const override;

  GraphPtr Reverse() const override {
    return Transpose();
  }

  DGLIdIters SuccVec(dgl_id_t vid) const override {
    LOG(FATAL) << "COO graph does not support efficient SuccVec."
      << " Please use CSR graph or AdjList graph instead.";
    return DGLIdIters(nullptr, nullptr);
  }

  DGLIdIters OutEdgeVec(dgl_id_t vid) const override {
    LOG(FATAL) << "COO graph does not support efficient OutEdgeVec."
      << " Please use CSR graph or AdjList graph instead.";
    return DGLIdIters(nullptr, nullptr);
  }

  DGLIdIters PredVec(dgl_id_t vid) const override {
    LOG(FATAL) << "COO graph does not support efficient PredVec."
      << " Please use CSR graph or AdjList graph instead.";
    return DGLIdIters(nullptr, nullptr);
  }

  DGLIdIters InEdgeVec(dgl_id_t vid) const override {
    LOG(FATAL) << "COO graph does not support efficient InEdgeVec."
      << " Please use CSR graph or AdjList graph instead.";
    return DGLIdIters(nullptr, nullptr);
  }

  GraphInterface *Reset() override {
    COO* gptr = new COO();
    *gptr = std::move(*this);
    return gptr;
  }

  std::vector<IdArray> GetAdj(bool transpose, const std::string &fmt) const override {
    CHECK(fmt == "coo") << "Not valid adj format request.";
    if (transpose) {
      return {HStack(dst_, src_)};
    } else {
      return {HStack(src_, dst_)};
    }
  }

  /*! \brief Return the transpose of this COO */
  COOPtr Transpose() const {
    return COOPtr(new COO(num_vertices_, dst_, src_));
  }

  /*! \brief Convert this COO to CSR */
  CSRPtr ToCSR() const;

  /*!
   * \brief Get the coo matrix that represents this graph.
   * \note The coo matrix shares the storage with this graph.
   *       The data field of the coo matrix is none.
   */
  COOMatrix ToCOOMatrix() const {
    return COOMatrix{src_, dst_, {}};
  }

  // member getters

  IdArray src() const { return src_; }

  IdArray dst() const { return dst_; }

 private:
  /* !\brief private default constructor */
  COO() {}

  /*! \brief number of vertices */
  int64_t num_vertices_;
  /*! \brief coordinate arrays */
  IdArray src_, dst_;
  /*! \brief whether the graph is a multi-graph */
  LazyObject<bool> is_multigraph_;
};

/*!
 * \brief DGL immutable graph index class.
 *
 * DGL's graph is directed. Vertices are integers enumerated from zero.
 */
class ImmutableGraph: public GraphInterface {
 public:
  /*! \brief Construct an immutable graph from the COO format. */
  explicit ImmutableGraph(COOPtr coo): coo_(coo) { }
  /*!
   * \brief Construct an immutable graph from the CSR format.
   *
   * For a single graph, we need two CSRs, one stores the in-edges of vertices and
   * the other stores the out-edges of vertices. These two CSRs stores the same edges.
   * The reason we need both is that some operators are faster on in-edge CSR and
   * the other operators are faster on out-edge CSR.
   *
   * However, not both CSRs are required. Technically, one CSR contains all information.
   * Thus, when we construct a temporary graphs (e.g., the sampled subgraphs), we only
   * construct one of the CSRs that runs fast for some operations we expect and construct
   * the other CSR on demand.
   */
  ImmutableGraph(CSRPtr in_csr, CSRPtr out_csr)
    : in_csr_(in_csr), out_csr_(out_csr) {
    CHECK(in_csr_ || out_csr_) << "Both CSR are missing.";
  }

  /*! \brief Construct an immutable graph from one CSR. */
  explicit ImmutableGraph(CSRPtr csr): out_csr_(csr) { }

  /*! \brief default copy constructor */
  ImmutableGraph(const ImmutableGraph& other) = default;

#ifndef _MSC_VER
  /*! \brief default move constructor */
  ImmutableGraph(ImmutableGraph&& other) = default;
#else
  ImmutableGraph(ImmutableGraph&& other) {
    this->in_csr_ = other.in_csr_;
    this->out_csr_ = other.out_csr_;
    this->coo_ = other.coo_;
    other.in_csr_ = nullptr;
    other.out_csr_ = nullptr;
    other.coo_ = nullptr;
  }
#endif  // _MSC_VER

  /*! \brief default assign constructor */
  ImmutableGraph& operator=(const ImmutableGraph& other) = default;

  /*! \brief default destructor */
  ~ImmutableGraph() = default;

  void AddVertices(uint64_t num_vertices) override {
    LOG(FATAL) << "AddVertices isn't supported in ImmutableGraph";
  }

  void AddEdge(dgl_id_t src, dgl_id_t dst) override {
    LOG(FATAL) << "AddEdge isn't supported in ImmutableGraph";
  }

  void AddEdges(IdArray src_ids, IdArray dst_ids) override {
    LOG(FATAL) << "AddEdges isn't supported in ImmutableGraph";
  }

  void Clear() override {
    LOG(FATAL) << "Clear isn't supported in ImmutableGraph";
  }

  /*!
   * \note not const since we have caches
   * \return whether the graph is a multigraph
   */
  bool IsMultigraph() const override {
    return AnyGraph()->IsMultigraph();
  }

  /*!
   * \return whether the graph is read-only
   */
  bool IsReadonly() const override {
    return true;
  }

  /*! \return the number of vertices in the graph.*/
  uint64_t NumVertices() const override {
    return AnyGraph()->NumVertices();
  }

  /*! \return the number of edges in the graph.*/
  uint64_t NumEdges() const override {
    return AnyGraph()->NumEdges();
  }

  /*! \return true if the given vertex is in the graph.*/
  bool HasVertex(dgl_id_t vid) const override {
    return vid < NumVertices();
  }

  /*! \return true if the given edge is in the graph.*/
  bool HasEdgeBetween(dgl_id_t src, dgl_id_t dst) const override {
    if (in_csr_) {
      return in_csr_->HasEdgeBetween(dst, src);
    } else {
      return GetOutCSR()->HasEdgeBetween(src, dst);
    }
  }

  /*!
   * \brief Find the predecessors of a vertex.
   * \param vid The vertex id.
   * \param radius The radius of the neighborhood. Default is immediate neighbor (radius=1).
   * \return the predecessor id array.
   */
  IdArray Predecessors(dgl_id_t vid, uint64_t radius = 1) const override {
    return GetInCSR()->Successors(vid, radius);
  }

  /*!
   * \brief Find the successors of a vertex.
   * \param vid The vertex id.
   * \param radius The radius of the neighborhood. Default is immediate neighbor (radius=1).
   * \return the successor id array.
   */
  IdArray Successors(dgl_id_t vid, uint64_t radius = 1) const override {
    return GetOutCSR()->Successors(vid, radius);
  }

  /*!
   * \brief Get all edge ids between the two given endpoints
   * \note Edges are associated with an integer id start from zero.
   *       The id is assigned when the edge is being added to the graph.
   * \param src The source vertex.
   * \param dst The destination vertex.
   * \return the edge id array.
   */
  IdArray EdgeId(dgl_id_t src, dgl_id_t dst) const override {
    if (in_csr_) {
      return in_csr_->EdgeId(dst, src);
    } else {
      return GetOutCSR()->EdgeId(src, dst);
    }
  }

  /*!
   * \brief Get all edge ids between the given endpoint pairs.
   * \note Edges are associated with an integer id start from zero.
   *       The id is assigned when the edge is being added to the graph.
   *       If duplicate pairs exist, the returned edge IDs will also duplicate.
   *       The order of returned edge IDs will follow the order of src-dst pairs
   *       first, and ties are broken by the order of edge ID.
   * \return EdgeArray containing all edges between all pairs.
   */
  EdgeArray EdgeIds(IdArray src, IdArray dst) const override {
    if (in_csr_) {
      EdgeArray edges = in_csr_->EdgeIds(dst, src);
      return EdgeArray{edges.dst, edges.src, edges.id};
    } else {
      return GetOutCSR()->EdgeIds(src, dst);
    }
  }

  /*!
   * \brief Find the edge ID and return the pair of endpoints
   * \param eid The edge ID
   * \return a pair whose first element is the source and the second the destination.
   */
  std::pair<dgl_id_t, dgl_id_t> FindEdge(dgl_id_t eid) const override {
    return GetCOO()->FindEdge(eid);
  }

  /*!
   * \brief Find the edge IDs and return their source and target node IDs.
   * \param eids The edge ID array.
   * \return EdgeArray containing all edges with id in eid.  The order is preserved.
   */
  EdgeArray FindEdges(IdArray eids) const override {
    return GetCOO()->FindEdges(eids);
  }

  /*!
   * \brief Get the in edges of the vertex.
   * \note The returned dst id array is filled with vid.
   * \param vid The vertex id.
   * \return the edges
   */
  EdgeArray InEdges(dgl_id_t vid) const override {
    const EdgeArray& ret = GetInCSR()->OutEdges(vid);
    return {ret.dst, ret.src, ret.id};
  }

  /*!
   * \brief Get the in edges of the vertices.
   * \param vids The vertex id array.
   * \return the id arrays of the two endpoints of the edges.
   */
  EdgeArray InEdges(IdArray vids) const override {
    const EdgeArray& ret = GetInCSR()->OutEdges(vids);
    return {ret.dst, ret.src, ret.id};
  }

  /*!
   * \brief Get the out edges of the vertex.
   * \note The returned src id array is filled with vid.
   * \param vid The vertex id.
   * \return the id arrays of the two endpoints of the edges.
   */
  EdgeArray OutEdges(dgl_id_t vid) const override {
    return GetOutCSR()->OutEdges(vid);
  }

  /*!
   * \brief Get the out edges of the vertices.
   * \param vids The vertex id array.
   * \return the id arrays of the two endpoints of the edges.
   */
  EdgeArray OutEdges(IdArray vids) const override {
    return GetOutCSR()->OutEdges(vids);
  }

  /*!
   * \brief Get all the edges in the graph.
   * \note If sorted is true, the returned edges list is sorted by their src and
   *       dst ids. Otherwise, they are in their edge id order.
   * \param sorted Whether the returned edge list is sorted by their src and dst ids
   * \return the id arrays of the two endpoints of the edges.
   */
  EdgeArray Edges(const std::string &order = "") const override;

  /*!
   * \brief Get the in degree of the given vertex.
   * \param vid The vertex id.
   * \return the in degree
   */
  uint64_t InDegree(dgl_id_t vid) const override {
    return GetInCSR()->OutDegree(vid);
  }

  /*!
   * \brief Get the in degrees of the given vertices.
   * \param vid The vertex id array.
   * \return the in degree array
   */
  DegreeArray InDegrees(IdArray vids) const override {
    return GetInCSR()->OutDegrees(vids);
  }

  /*!
   * \brief Get the out degree of the given vertex.
   * \param vid The vertex id.
   * \return the out degree
   */
  uint64_t OutDegree(dgl_id_t vid) const override {
    return GetOutCSR()->OutDegree(vid);
  }

  /*!
   * \brief Get the out degrees of the given vertices.
   * \param vid The vertex id array.
   * \return the out degree array
   */
  DegreeArray OutDegrees(IdArray vids) const override {
    return GetOutCSR()->OutDegrees(vids);
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
  Subgraph VertexSubgraph(IdArray vids) const override;

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
  Subgraph EdgeSubgraph(IdArray eids) const override;

  /*!
   * \brief Return a new graph with all the edges reversed.
   *
   * The returned graph preserves the vertex and edge index in the original graph.
   *
   * \return the reversed graph
   */
  GraphPtr Reverse() const override {
    if (coo_) {
      return GraphPtr(new ImmutableGraph(out_csr_, in_csr_, coo_->Transpose()));
    } else {
      return GraphPtr(new ImmutableGraph(out_csr_, in_csr_));
    }
  }

  /*!
   * \brief Return the successor vector
   * \param vid The vertex id.
   * \return the successor vector
   */
  DGLIdIters SuccVec(dgl_id_t vid) const override {
    return GetOutCSR()->SuccVec(vid);
  }

  /*!
   * \brief Return the out edge id vector
   * \param vid The vertex id.
   * \return the out edge id vector
   */
  DGLIdIters OutEdgeVec(dgl_id_t vid) const override {
    return GetOutCSR()->OutEdgeVec(vid);
  }

  /*!
   * \brief Return the predecessor vector
   * \param vid The vertex id.
   * \return the predecessor vector
   */
  DGLIdIters PredVec(dgl_id_t vid) const override {
    return GetInCSR()->SuccVec(vid);
  }

  /*!
   * \brief Return the in edge id vector
   * \param vid The vertex id.
   * \return the in edge id vector
   */
  DGLIdIters InEdgeVec(dgl_id_t vid) const override {
    return GetInCSR()->OutEdgeVec(vid);
  }

  /*!
   * \brief Reset the data in the graph and move its data to the returned graph object.
   * \return a raw pointer to the graph object.
   */
  GraphInterface *Reset() override {
    ImmutableGraph* gptr = new ImmutableGraph();
    *gptr = std::move(*this);
    return gptr;
  }

  /*!
   * \brief Get the adjacency matrix of the graph.
   *
   * By default, a row of returned adjacency matrix represents the destination
   * of an edge and the column represents the source.
   * \param transpose A flag to transpose the returned adjacency matrix.
   * \param fmt the format of the returned adjacency matrix.
   * \return a vector of three IdArray.
   */
  std::vector<IdArray> GetAdj(bool transpose, const std::string &fmt) const override;

  /* !\brief Return in csr. If not exist, transpose the other one.*/
  CSRPtr GetInCSR() const {
    if (!in_csr_) {
      if (out_csr_) {
        const_cast<ImmutableGraph*>(this)->in_csr_ = out_csr_->Transpose();
      } else {
        CHECK(coo_) << "None of CSR, COO exist";
        const_cast<ImmutableGraph*>(this)->in_csr_ = coo_->Transpose()->ToCSR();
      }
    }
    return in_csr_;
  }

  /* !\brief Return out csr. If not exist, transpose the other one.*/
  CSRPtr GetOutCSR() const {
    if (!out_csr_) {
      if (in_csr_) {
        const_cast<ImmutableGraph*>(this)->out_csr_ = in_csr_->Transpose();
      } else {
        CHECK(coo_) << "None of CSR, COO exist";
        const_cast<ImmutableGraph*>(this)->out_csr_ = coo_->ToCSR();
      }
    }
    return out_csr_;
  }

  /* !\brief Return coo. If not exist, create from csr.*/
  COOPtr GetCOO() const {
    if (!coo_) {
      if (in_csr_) {
        const_cast<ImmutableGraph*>(this)->coo_ = in_csr_->ToCOO()->Transpose();
      } else {
        CHECK(out_csr_) << "Both CSR are missing.";
        const_cast<ImmutableGraph*>(this)->coo_ = out_csr_->ToCOO();
      }
    }
    return coo_;
  }

 protected:
  /* !\brief internal default constructor */
  ImmutableGraph() {}

  /* !\brief internal constructor for all the members */
  ImmutableGraph(CSRPtr in_csr, CSRPtr out_csr, COOPtr coo)
    : in_csr_(in_csr), out_csr_(out_csr), coo_(coo) {
    CHECK(AnyGraph()) << "At least one graph structure should exist.";
  }

  /* !\brief return pointer to any available graph structure */
  GraphPtr AnyGraph() const {
    if (in_csr_) {
      return in_csr_;
    } else if (out_csr_) {
      return out_csr_;
    } else {
      return coo_;
    }
  }

  // Store the in csr (i.e, the reverse csr)
  CSRPtr in_csr_;
  // Store the out csr (i.e, the normal csr)
  CSRPtr out_csr_;
  // Store the edge list indexed by edge id (COO)
  COOPtr coo_;
};

// inline implementations

template <typename IndptrIter, typename IndicesIter, typename EdgeIdIter>
CSR::CSR(int64_t num_vertices, int64_t num_edges,
    IndptrIter indptr_begin, IndicesIter indices_begin, EdgeIdIter edge_ids_begin,
    bool is_multigraph): is_multigraph_(is_multigraph) {
  indptr_ = NewIdArray(num_vertices + 1);
  indices_ = NewIdArray(num_edges);
  edge_ids_ = NewIdArray(num_edges);
  dgl_id_t* indptr_data = static_cast<dgl_id_t*>(indptr_->data);
  dgl_id_t* indices_data = static_cast<dgl_id_t*>(indices_->data);
  dgl_id_t* edge_ids_data = static_cast<dgl_id_t*>(edge_ids_->data);
  for (int64_t i = 0; i < num_vertices + 1; ++i)
    *(indptr_data++) = *(indptr_begin++);
  for (int64_t i = 0; i < num_edges; ++i) {
    *(indices_data++) = *(indices_begin++);
    *(edge_ids_data++) = *(edge_ids_begin++);
  }
}

class ImmutableBiGraph: public ImmutableGraph {
 public:
  /*! \brief Construct an immutable graph from the COO format. */
  explicit ImmutableBiGraph(COOPtr coo, size_t num_src, size_t num_dst): ImmutableGraph(coo) {
    this->num_src = num_src;
    this->num_dst = num_dst;
  }
  /*! \brief Construct an immutable graph from the CSR format. */
  ImmutableBiGraph(CSRPtr in_csr, CSRPtr out_csr, size_t num_src, size_t num_dst)
    : ImmutableGraph(in_csr, out_csr) {
    CHECK(in_csr_ || out_csr_) << "Both CSR are missing.";
    this->num_src = num_src;
    this->num_dst = num_dst;
  }

  /*! \brief Construct an immutable graph from one CSR. */
  ImmutableBiGraph(CSRPtr csr, size_t num_src, size_t num_dst): ImmutableGraph(csr) {
    this->num_src = num_src;
    this->num_dst = num_dst;
  }

  std::vector<IdArray> GetAdj(bool transpose, const std::string &fmt) const override;

 private:
  size_t num_src;
  size_t num_dst;
};

}  // namespace dgl

#endif  // DGL_IMMUTABLE_GRAPH_H_
