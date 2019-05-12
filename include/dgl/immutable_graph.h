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
#include <dgl/runtime/ndarray.h>
#include <dgl/graph_interface.h>
#include <dgl/lazy_pointer.h>

namespace dgl {

class CSR;
class COO;

class CSR : public GraphInterface {
 public:
  typedef std::shared_ptr<CSR> Ptr;
  // Create a csr graph that has the given number of verts and edges.
  CSR(int64_t num_vertices, int64_t num_edges);
  // Create a csr graph that shares the given indptr and indices.
  CSR(IdArray indptr, IdArray indices, IdArray edge_ids);
  // Create a csr graph whose memory is stored in the shared memory
  //   and the structure is given by the indptr and indcies.
  CSR(IdArray indptr, IdArray indices, IdArray edge_ids,
      const std::string &shared_mem_name);
  // Create a csr graph whose memory is stored in the shared memory
  //   that has the given number of verts and edges.
  CSR(const std::string &shared_mem_name, size_t num_vertices, size_t num_edges);

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

  bool IsMultigraph() const override {
    return false;
  }

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

  BoolArray HasEdgesBetween(IdArray src_ids, IdArray dst_ids) const override;

  IdArray Predecessors(dgl_id_t vid, uint64_t radius = 1) const override {
    LOG(FATAL) << "CSR graph does not support efficient predecessor query."
      << " Please use successors on the reverse CSR graph.";
    return {};
  }

  IdArray Successors(dgl_id_t vid, uint64_t radius = 1) const override;

  IdArray EdgeId(dgl_id_t src, dgl_id_t dst) const override;

  std::pair<dgl_id_t, dgl_id_t> FindEdge(dgl_id_t eid) const override;

  EdgeArray FindEdges(IdArray eids) const override;

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

  EdgeArray Edges(const std::string &order = "") const override {
    LOG(FATAL) << "CSR graph does not support efficient edges query.";
    return {};
  }

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

  Subgraph EdgeSubgraph(IdArray eids) const override;

  GraphPtr Reverse() const override;

  DGLIdIters SuccVec(dgl_id_t vid) const override {
    const int64_t* indptr_data = static_cast<int64_t*>(indptr_->data);
    const dgl_id_t* indices_data = static_cast<dgl_id_t*>(indices_->data);
    const int64_t start = indptr_data[vid];
    const int64_t end = indptr_data[vid + 1];
    return DGLIdIters(indices_data + start, indices_data + end);
  }

  DGLIdIters OutEdgeVec(dgl_id_t vid) const override {
    const int64_t* indptr_data = static_cast<int64_t*>(indptr_->data);
    const dgl_id_t* eid_data = static_cast<dgl_id_t*>(edge_ids_->data);
    const int64_t start = indptr_data[vid];
    const int64_t end = indptr_data[vid + 1];
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

  GraphInterface *Reset() override;

  std::vector<IdArray> GetAdj(bool transpose, const std::string &fmt) const override;

  CSR::Ptr Transpose() const;

  COO::Ptr ToCOO() const;

 private:
  IdArray indptr_, indices_, edge_ids_;

 private:
#ifndef _WIN32
  std::shared_ptr<runtime::SharedMemory> mem;
#endif  // _WIN32
};

/*!
 * \brief DGL immutable graph index class.
 *
 * DGL's graph is directed. Vertices are integers enumerated from zero.
 */
class ImmutableGraph: public GraphInterface {
 public:
  /*
  typedef struct {
    IdArray indptr, indices, id;
  } CSRArray;

  struct Edge {
    dgl_id_t end_points[2];
    dgl_id_t edge_id;
  };

  // Edge list indexed by edge id;
  struct EdgeList {
    typedef std::shared_ptr<EdgeList> Ptr;
    std::vector<dgl_id_t> src_points;
    std::vector<dgl_id_t> dst_points;

    EdgeList(int64_t len, dgl_id_t val) {
      src_points.resize(len, val);
      dst_points.resize(len, val);
    }

    void register_edge(dgl_id_t eid, dgl_id_t src, dgl_id_t dst) {
      CHECK_LT(eid, src_points.size()) << "Invalid edge id " << eid;
      src_points[eid] = src;
      dst_points[eid] = dst;
    }

    static EdgeList::Ptr FromCSR(
        const CSR::vector<int64_t>& indptr,
        const CSR::vector<dgl_id_t>& indices,
        const CSR::vector<dgl_id_t>& edge_ids,
        bool in_csr);
  };
  */

  /*! \brief Construct an immutable graph from the COO format. */
  ImmutableGraph(IdArray src_ids, IdArray dst_ids, IdArray edge_ids, size_t num_nodes,
                 bool multigraph = false);

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
  ImmutableGraph(CSR::Ptr in_csr, CSR::Ptr out_csr,
                 bool multigraph = false) : is_multigraph_(multigraph) {
    this->in_csr_ = in_csr;
    this->out_csr_ = out_csr;
    CHECK(this->in_csr_ != nullptr || this->out_csr_ != nullptr)
                   << "there must exist one of the CSRs";
  }

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
    this->is_multigraph_ = other.is_multigraph_;
    other.in_csr_ = nullptr;
    other.out_csr_ = nullptr;
    other.coo_ = nullptr;
    other->is_multigraph_ = false;
  }
#endif  // _MSC_VER

  /*! \brief default assign constructor */
  ImmutableGraph& operator=(const ImmutableGraph& other) = default;

  /*! \brief default destructor */
  ~ImmutableGraph() = default;

  /*!
   * \brief Add vertices to the graph.
   * \note Since vertices are integers enumerated from zero, only the number of
   *       vertices to be added needs to be specified.
   * \param num_vertices The number of vertices to be added.
   */
  void AddVertices(uint64_t num_vertices) override {
    LOG(FATAL) << "AddVertices isn't supported in ImmutableGraph";
  }

  /*!
   * \brief Add one edge to the graph.
   * \param src The source vertex.
   * \param dst The destination vertex.
   */
  void AddEdge(dgl_id_t src, dgl_id_t dst) override {
    LOG(FATAL) << "AddEdge isn't supported in ImmutableGraph";
  }

  /*!
   * \brief Add edges to the graph.
   * \param src_ids The source vertex id array.
   * \param dst_ids The destination vertex id array.
   */
  void AddEdges(IdArray src_ids, IdArray dst_ids) override {
    LOG(FATAL) << "AddEdges isn't supported in ImmutableGraph";
  }

  /*!
   * \brief Clear the graph. Remove all vertices/edges.
   */
  void Clear() override {
    LOG(FATAL) << "Clear isn't supported in ImmutableGraph";
  }

  /*!
   * \note not const since we have caches
   * \return whether the graph is a multigraph
   */
  bool IsMultigraph() const override {
    return is_multigraph_;
  }

  /*!
   * \return whether the graph is read-only
   */
  virtual bool IsReadonly() const override {
    return true;
  }

  /*! \return the number of vertices in the graph.*/
  uint64_t NumVertices() const override;
  /*{
    if (in_csr_)
      return in_csr_->NumVertices();
    else
      return out_csr_->NumVertices();
  }*/

  /*! \return the number of edges in the graph.*/
  uint64_t NumEdges() const override;
  /*{
    if (in_csr_)
      return in_csr_->NumEdges();
    else
      return out_csr_->NumEdges();
  }*/

  /*! \return true if the given vertex is in the graph.*/
  bool HasVertex(dgl_id_t vid) const override {
    return vid < NumVertices();
  }

  /*! \return a 0-1 array indicating whether the given vertices are in the graph.*/
  BoolArray HasVertices(IdArray vids) const override;

  /*! \return true if the given edge is in the graph.*/
  bool HasEdgeBetween(dgl_id_t src, dgl_id_t dst) const override;

  /*! \return a 0-1 array indicating whether the given edges are in the graph.*/
  BoolArray HasEdgesBetween(IdArray src_ids, IdArray dst_ids) const override;

  /*!
   * \brief Find the predecessors of a vertex.
   * \param vid The vertex id.
   * \param radius The radius of the neighborhood. Default is immediate neighbor (radius=1).
   * \return the predecessor id array.
   */
  IdArray Predecessors(dgl_id_t vid, uint64_t radius = 1) const override;

  /*!
   * \brief Find the successors of a vertex.
   * \param vid The vertex id.
   * \param radius The radius of the neighborhood. Default is immediate neighbor (radius=1).
   * \return the successor id array.
   */
  IdArray Successors(dgl_id_t vid, uint64_t radius = 1) const override;

  /*!
   * \brief Get all edge ids between the two given endpoints
   * \note Edges are associated with an integer id start from zero.
   *       The id is assigned when the edge is being added to the graph.
   * \param src The source vertex.
   * \param dst The destination vertex.
   * \return the edge id array.
   */
  IdArray EdgeId(dgl_id_t src, dgl_id_t dst) const override;

  /*!
   * \brief Get all edge ids between the given endpoint pairs.
   * \note Edges are associated with an integer id start from zero.
   *       The id is assigned when the edge is being added to the graph.
   *       If duplicate pairs exist, the returned edge IDs will also duplicate.
   *       The order of returned edge IDs will follow the order of src-dst pairs
   *       first, and ties are broken by the order of edge ID.
   * \return EdgeArray containing all edges between all pairs.
   */
  EdgeArray EdgeIds(IdArray src, IdArray dst) const override;

  /*!
   * \brief Find the edge ID and return the pair of endpoints
   * \param eid The edge ID
   * \return a pair whose first element is the source and the second the destination.
   */
  std::pair<dgl_id_t, dgl_id_t> FindEdge(dgl_id_t eid) const override;

  /*!
   * \brief Find the edge IDs and return their source and target node IDs.
   * \param eids The edge ID array.
   * \return EdgeArray containing all edges with id in eid.  The order is preserved.
   */
  EdgeArray FindEdges(IdArray eids) const override;

  /*!
   * \brief Get the in edges of the vertex.
   * \note The returned dst id array is filled with vid.
   * \param vid The vertex id.
   * \return the edges
   */
  EdgeArray InEdges(dgl_id_t vid) const override;
  /*{
    return this->GetInCSR()->GetEdges(vid);
  }*/

  /*!
   * \brief Get the in edges of the vertices.
   * \param vids The vertex id array.
   * \return the id arrays of the two endpoints of the edges.
   */
  EdgeArray InEdges(IdArray vids) const override;
  /*{
    return this->GetInCSR()->GetEdges(vids);
  }*/

  /*!
   * \brief Get the out edges of the vertex.
   * \note The returned src id array is filled with vid.
   * \param vid The vertex id.
   * \return the id arrays of the two endpoints of the edges.
   */
  EdgeArray OutEdges(dgl_id_t vid) const override;
  /*{
    auto ret = this->GetOutCSR()->GetEdges(vid);
    // We should reverse the source and destination in the edge array.
    return ImmutableGraph::EdgeArray{ret.dst, ret.src, ret.id};
  }*/

  /*!
   * \brief Get the out edges of the vertices.
   * \param vids The vertex id array.
   * \return the id arrays of the two endpoints of the edges.
   */
  EdgeArray OutEdges(IdArray vids) const override;
  /*{
    auto ret = this->GetOutCSR()->GetEdges(vids);
    return ImmutableGraph::EdgeArray{ret.dst, ret.src, ret.id};
  }*/

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
  uint64_t InDegree(dgl_id_t vid) const override;
  /*{
    CHECK(HasVertex(vid)) << "invalid vertex: " << vid;
    return this->GetInCSR()->GetDegree(vid);
  }*/

  /*!
   * \brief Get the in degrees of the given vertices.
   * \param vid The vertex id array.
   * \return the in degree array
   */
  DegreeArray InDegrees(IdArray vids) const override;
  /*{
    return this->GetInCSR()->GetDegrees(vids);
  }*/

  /*!
   * \brief Get the out degree of the given vertex.
   * \param vid The vertex id.
   * \return the out degree
   */
  uint64_t OutDegree(dgl_id_t vid) const override;
  /*{
    CHECK(HasVertex(vid)) << "invalid vertex: " << vid;
    return this->GetOutCSR()->GetDegree(vid);
  }*/

  /*!
   * \brief Get the out degrees of the given vertices.
   * \param vid The vertex id array.
   * \return the out degree array
   */
  DegreeArray OutDegrees(IdArray vids) const override;
  /*{
    return this->GetOutCSR()->GetDegrees(vids);
  }*/

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
  GraphPtr Reverse() const override;
  /*{
    return GraphPtr(new ImmutableGraph(out_csr_, in_csr_, is_multigraph_));
  }*/

  /*!
   * \brief Return the successor vector
   * \param vid The vertex id.
   * \return the successor vector
   */
  DGLIdIters SuccVec(dgl_id_t vid) const override;
  /*{
    return DGLIdIters(out_csr_->indices.begin() + out_csr_->indptr[vid],
                      out_csr_->indices.begin() + out_csr_->indptr[vid + 1]);
  }*/

  /*!
   * \brief Return the out edge id vector
   * \param vid The vertex id.
   * \return the out edge id vector
   */
  DGLIdIters OutEdgeVec(dgl_id_t vid) const override;
  /*{
    return DGLIdIters(out_csr_->edge_ids.begin() + out_csr_->indptr[vid],
                      out_csr_->edge_ids.begin() + out_csr_->indptr[vid + 1]);
  }*/

  /*!
   * \brief Return the predecessor vector
   * \param vid The vertex id.
   * \return the predecessor vector
   */
  DGLIdIters PredVec(dgl_id_t vid) const override;
  /*{
    return DGLIdIters(in_csr_->indices.begin() + in_csr_->indptr[vid],
                      in_csr_->indices.begin() + in_csr_->indptr[vid + 1]);
  }*/

  /*!
   * \brief Return the in edge id vector
   * \param vid The vertex id.
   * \return the in edge id vector
   */
  DGLIdIters InEdgeVec(dgl_id_t vid) const override;
  /*{
    return DGLIdIters(in_csr_->edge_ids.begin() + in_csr_->indptr[vid],
                      in_csr_->edge_ids.begin() + in_csr_->indptr[vid + 1]);
  }*/

  /*!
   * \brief Reset the data in the graph and move its data to the returned graph object.
   * \return a raw pointer to the graph object.
   */
  virtual GraphInterface *Reset() {
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
  virtual std::vector<IdArray> GetAdj(bool transpose, const std::string &fmt) const;

  /* !\brief Return in csr. If not exist, transpose the other one.*/
  CSR::Ptr GetInCSR() const {
    return in_csr_.Get([&] () {
        CHECK(out_csr_) << "Both CSR are missing.";
        return out_csr_->Transpose(); });
  }

  /* !\brief Return out csr. If not exist, transpose the other one.*/
  CSR::Ptr GetOutCSR() const {
    return out_csr_.Get([&] () {
        CHECK(in_csr_) << "Both CSR are missing.";
        return in_csr_->Transpose(); });
  }

  /*
   * The edge list is required for FindEdge/FindEdges/EdgeSubgraph, if no such function is called, we would not create edge list.
   * if such function is called the first time, we create a edge list from one of the graph's csr representations,
   * if we have called such function before, we get the one cached in the structure.
   */
  EdgeList::Ptr GetEdgeList() const {
    if (edge_list_)
      return edge_list_;
    if (in_csr_) {
      const_cast<ImmutableGraph *>(this)->edge_list_ =\
        EdgeList::FromCSR(in_csr_->indptr, in_csr_->indices, in_csr_->edge_ids, true);
    } else {
      CHECK(out_csr_ != nullptr) << "one of the CSRs must exist";
      const_cast<ImmutableGraph *>(this)->edge_list_ =\
        EdgeList::FromCSR(out_csr_->indptr, out_csr_->indices, out_csr_->edge_ids, false);
    }
    return edge_list_;
  }

  /*!
   * \brief Get the CSR array that represents the in-edges.
   * This method copies data from std::vector to IdArray.
   * \param start the first row to copy.
   * \param end the last row to copy (exclusive).
   * \return the CSR array.
   */
  CSRArray GetInCSRArray(size_t start, size_t end) const;

  /*!
   * \brief Get the CSR array that represents the out-edges.
   * This method copies data from std::vector to IdArray.
   * \param start the first row to copy.
   * \param end the last row to copy (exclusive).
   * \return the CSR array.
   */
  CSRArray GetOutCSRArray(size_t start, size_t end) const;

 protected:
  DGLIdIters GetInEdgeIdRef(dgl_id_t src, dgl_id_t dst) const;
  DGLIdIters GetOutEdgeIdRef(dgl_id_t src, dgl_id_t dst) const;

  /*!
   * \brief Compact a subgraph.
   * In a sampled subgraph, the vertex Id is still in the ones in the original graph.
   * We want to convert them to the subgraph Ids.
   */
  void CompactSubgraph(IdArray induced_vertices);

  // Store the in csr
  mutable LazyPointer<CSR> in_csr_;
  // Store the out csr
  mutable LazyPointer<CSR> out_csr_;
  // Store the edge list indexed by edge id (COO)
  mutable LazyPointer<COO> coo_;
  /*!
   * \brief Whether if this is a multigraph.
   *
   * When a multiedge is added, this flag switches to true.
   */
  bool is_multigraph_ = false;
};

}  // namespace dgl

#endif  // DGL_IMMUTABLE_GRAPH_H_
