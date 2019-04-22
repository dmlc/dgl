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

namespace dgl {

/*!
 * \brief DGL immutable graph index class.
 *
 * DGL's graph is directed. Vertices are integers enumerated from zero.
 */
class ImmutableGraph: public GraphInterface {
 public:
  typedef struct {
    IdArray indptr, indices, id;
  } CSRArray;

  struct Edge {
    dgl_id_t end_points[2];
    dgl_id_t edge_id;
  };

  struct EdgeList;

  struct CSR {
    typedef std::shared_ptr<CSR> Ptr;

    /*
     * This vector provides interfaces similar to std::vector.
     * The main difference is that the memory used by the vector can be allocated
     * outside the vector. The main use case is that the vector can use the shared
     * memory that is created by another process. In this way, we can access the
     * graph structure loaded in another process.
     */
    template<class T>
    class vector {
     public:
      vector() {
        this->arr = nullptr;
        this->capacity = 0;
        this->curr = 0;
        this->own = false;
      }

      /*
       * Create a vector whose memory is allocated outside.
       * Here there are no elements in the vector.
       */
      vector(T *arr, size_t size) {
        this->arr = arr;
        this->capacity = size;
        this->curr = 0;
        this->own = false;
      }

      /*
       * Create a vector whose memory is allocated by the vector.
       * Here there are no elements in the vector.
       */
      explicit vector(size_t size) {
        this->arr = static_cast<T *>(malloc(size * sizeof(T)));
        this->capacity = size;
        this->curr = 0;
        this->own = true;
      }

      ~vector() {
        // If the memory is allocated by the vector, it should be free'd.
        if (this->own) {
          free(this->arr);
        }
      }

      vector(const vector &other) = delete;

      /*
       * Initialize the vector whose memory is allocated outside.
       * There are no elements in the vector.
       */
      void init(T *arr, size_t size) {
        CHECK(this->arr == nullptr);
        this->arr = arr;
        this->capacity = size;
        this->curr = 0;
        this->own = false;
      }

      /*
       * Initialize the vector whose memory is allocated outside.
       * There are elements in the vector.
       */
      void init(T *arr, size_t capacity, size_t size) {
        CHECK(this->arr == nullptr);
        CHECK_LE(size, capacity);
        this->arr = arr;
        this->capacity = capacity;
        this->curr = size;
        this->own = false;
      }

      /* Similar to std::vector::push_back. */
      void push_back(T val) {
        // If the vector doesn't own the memory, it can't adjust its memory size.
        if (!this->own) {
          CHECK_LT(curr, capacity);
        } else if (curr == capacity) {
          this->capacity = this->capacity * 2;
          this->arr = static_cast<T *>(realloc(this->arr, this->capacity * sizeof(T)));
          CHECK(this->arr) << "can't allocate memory for a larger vector.";
        }
        this->arr[curr++] = val;
      }

      /*
       * This inserts multiple elements to the back of the vector.
       */
      void insert_back(const T* val, size_t len) {
        if (!this->own) {
          CHECK_LE(curr + len, capacity);
        } else if (curr + len > capacity) {
          this->capacity = curr + len;
          this->arr = static_cast<T *>(realloc(this->arr, this->capacity * sizeof(T)));
          CHECK(this->arr) << "can't allocate memory for a larger vector.";
        }
        std::copy(val, val + len, this->arr + curr);
        curr += len;
      }

      /*
       * Similar to std::vector::[].
       * It checks the boundary of the vector.
       */
      T &operator[](size_t idx) {
        CHECK_LT(idx, curr);
        return this->arr[idx];
      }

      /*
       * Similar to std::vector::[].
       * It checks the boundary of the vector.
       */
      const T &operator[](size_t idx) const {
        CHECK_LT(idx, curr);
        return this->arr[idx];
      }

      /* Similar to std::vector::size. */
      size_t size() const {
        return this->curr;
      }

      /* Similar to std::vector::resize. */
      void resize(size_t new_size) {
        if (!this->own) {
          CHECK_LE(new_size, capacity);
        } else if (new_size > capacity) {
          this->capacity = new_size;
          this->arr = static_cast<T *>(realloc(this->arr, this->capacity * sizeof(T)));
          CHECK(this->arr) << "can't allocate memory for a larger vector.";
        }
        for (size_t i = this->curr; i < new_size; i++)
          this->arr[i] = 0;
        this->curr = new_size;
      }

      /* Similar to std::vector::clear. */
      void clear() {
        this->curr = 0;
      }

      /* Similar to std::vector::data. */
      const T *data() const {
        return this->arr;
      }

      /* Similar to std::vector::data. */
      T *data() {
        return this->arr;
      }

      /*
       * This is to simulate begin() of std::vector.
       * However, it returns the raw pointer instead of iterator.
       */
      const T *begin() const {
        return this->arr;
      }

      /*
       * This is to simulate begin() of std::vector.
       * However, it returns the raw pointer instead of iterator.
       */
      T *begin() {
        return this->arr;
      }

      /*
       * This is to simulate end() of std::vector.
       * However, it returns the raw pointer instead of iterator.
       */
      const T *end() const {
        return this->arr + this->curr;
      }

      /*
       * This is to simulate end() of std::vector.
       * However, it returns the raw pointer instead of iterator.
       */
      T *end() {
        return this->arr + this->curr;
      }

     private:
      /*
       * \brief the raw array that contains elements of type T.
       *
       * The vector may or may not own the memory of the raw array.
       */
      T *arr;
      /* \brief the memory size of the raw array. */
      size_t capacity;
      /* \brief the number of elements in the array. */
      size_t curr;
      /* \brief whether the vector owns the memory. */
      bool own;
    };

    vector<int64_t> indptr;
    vector<dgl_id_t> indices;
    vector<dgl_id_t> edge_ids;

    CSR(int64_t num_vertices, int64_t expected_num_edges);
    CSR(IdArray indptr, IdArray indices, IdArray edge_ids);
    CSR(IdArray indptr, IdArray indices, IdArray edge_ids,
        const std::string &shared_mem_name);
    CSR(const std::string &shared_mem_name, size_t num_vertices, size_t num_edges);

    bool HasVertex(dgl_id_t vid) const {
      return vid < NumVertices();
    }

    uint64_t NumVertices() const {
      return indptr.size() - 1;
    }

    uint64_t NumEdges() const {
      return indices.size();
    }

    /* This gets the sum of vertex degrees in the range. */
    uint64_t GetDegree(dgl_id_t start, dgl_id_t end) const {
      return indptr[end] - indptr[start];
    }

    uint64_t GetDegree(dgl_id_t vid) const {
      return indptr[vid + 1] - indptr[vid];
    }
    DegreeArray GetDegrees(IdArray vids) const;
    EdgeArray GetEdges(dgl_id_t vid) const;
    EdgeArray GetEdges(IdArray vids) const;
    /* \brief this returns the start and end position of the column indices corresponding v. */
    DGLIdIters GetIndexRef(dgl_id_t v) const {
      const int64_t start = indptr[v];
      const int64_t end = indptr[v + 1];
      return DGLIdIters(indices.begin() + start, indices.begin() + end);
    }
    /*
     * Read all edges and store them in the vector.
     */
    void ReadAllEdges(std::vector<Edge> *edges) const;
    CSR::Ptr Transpose() const;
    std::pair<CSR::Ptr, IdArray> VertexSubgraph(IdArray vids) const;
    std::pair<CSR::Ptr, IdArray> EdgeSubgraph(IdArray eids,
                                              std::shared_ptr<EdgeList> edge_list) const;
    /*
     * Construct a CSR from a list of edges.
     *
     * When constructing a CSR, we need to sort the edge list. To reduce the overhead,
     * we simply sort on the input edge list. We allow sorting on both end points of an edge,
     * which is specified by `sort_on`.
     */
    static CSR::Ptr FromEdges(std::vector<Edge> *edges, int sort_on, uint64_t num_nodes);

   private:
#ifndef _WIN32
    std::shared_ptr<runtime::SharedMemory> mem;
#endif  // _WIN32
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

  /*! \brief default constructor */
  explicit ImmutableGraph(bool multigraph = false) : is_multigraph_(multigraph) {}

  /*! \brief default copy constructor */
  ImmutableGraph(const ImmutableGraph& other) = default;

#ifndef _MSC_VER
  /*! \brief default move constructor */
  ImmutableGraph(ImmutableGraph&& other) = default;
#else
  ImmutableGraph(ImmutableGraph&& other) {
    this->in_csr_ = other.in_csr_;
    this->out_csr_ = other.out_csr_;
    this->is_multigraph_ = other.is_multigraph_;
    other.in_csr_ = nullptr;
    other.out_csr_ = nullptr;
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
  void AddVertices(uint64_t num_vertices) {
    LOG(FATAL) << "AddVertices isn't supported in ImmutableGraph";
  }

  /*!
   * \brief Add one edge to the graph.
   * \param src The source vertex.
   * \param dst The destination vertex.
   */
  void AddEdge(dgl_id_t src, dgl_id_t dst) {
    LOG(FATAL) << "AddEdge isn't supported in ImmutableGraph";
  }

  /*!
   * \brief Add edges to the graph.
   * \param src_ids The source vertex id array.
   * \param dst_ids The destination vertex id array.
   */
  void AddEdges(IdArray src_ids, IdArray dst_ids) {
    LOG(FATAL) << "AddEdges isn't supported in ImmutableGraph";
  }

  /*!
   * \brief Clear the graph. Remove all vertices/edges.
   */
  void Clear() {
    LOG(FATAL) << "Clear isn't supported in ImmutableGraph";
  }

  /*!
   * \note not const since we have caches
   * \return whether the graph is a multigraph
   */
  bool IsMultigraph() const {
    return is_multigraph_;
  }

  /*!
   * \return whether the graph is read-only
   */
  virtual bool IsReadonly() const {
    return true;
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
  std::pair<dgl_id_t, dgl_id_t> FindEdge(dgl_id_t eid) const;

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
    auto ret = this->GetOutCSR()->GetEdges(vid);
    // We should reverse the source and destination in the edge array.
    return ImmutableGraph::EdgeArray{ret.dst, ret.src, ret.id};
  }

  /*!
   * \brief Get the out edges of the vertices.
   * \param vids The vertex id array.
   * \return the id arrays of the two endpoints of the edges.
   */
  EdgeArray OutEdges(IdArray vids) const {
    auto ret = this->GetOutCSR()->GetEdges(vids);
    return ImmutableGraph::EdgeArray{ret.dst, ret.src, ret.id};
  }

  /*!
   * \brief Get all the edges in the graph.
   * \note If sorted is true, the returned edges list is sorted by their src and
   *       dst ids. Otherwise, they are in their edge id order.
   * \param sorted Whether the returned edge list is sorted by their src and dst ids
   * \return the id arrays of the two endpoints of the edges.
   */
  EdgeArray Edges(const std::string &order = "") const;

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
  GraphPtr Reverse() const {
    return GraphPtr(new ImmutableGraph(out_csr_, in_csr_, is_multigraph_));
  }

  /*!
   * \brief Return the successor vector
   * \param vid The vertex id.
   * \return the successor vector
   */
  DGLIdIters SuccVec(dgl_id_t vid) const {
    return DGLIdIters(out_csr_->indices.begin() + out_csr_->indptr[vid],
                      out_csr_->indices.begin() + out_csr_->indptr[vid + 1]);
  }

  /*!
   * \brief Return the out edge id vector
   * \param vid The vertex id.
   * \return the out edge id vector
   */
  DGLIdIters OutEdgeVec(dgl_id_t vid) const {
    return DGLIdIters(out_csr_->edge_ids.begin() + out_csr_->indptr[vid],
                      out_csr_->edge_ids.begin() + out_csr_->indptr[vid + 1]);
  }

  /*!
   * \brief Return the predecessor vector
   * \param vid The vertex id.
   * \return the predecessor vector
   */
  DGLIdIters PredVec(dgl_id_t vid) const {
    return DGLIdIters(in_csr_->indices.begin() + in_csr_->indptr[vid],
                      in_csr_->indices.begin() + in_csr_->indptr[vid + 1]);
  }

  /*!
   * \brief Return the in edge id vector
   * \param vid The vertex id.
   * \return the in edge id vector
   */
  DGLIdIters InEdgeVec(dgl_id_t vid) const {
    return DGLIdIters(in_csr_->edge_ids.begin() + in_csr_->indptr[vid],
                      in_csr_->edge_ids.begin() + in_csr_->indptr[vid + 1]);
  }

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

  /*
   * The immutable graph may only contain one of the CSRs (e.g., the sampled subgraphs).
   * When we get in csr or out csr, we try to get the one cached in the structure.
   * If not, we transpose the other one to get the one we need.
   */
  CSR::Ptr GetInCSR() const {
    if (in_csr_) {
      return in_csr_;
    } else {
      CHECK(out_csr_ != nullptr) << "one of the CSRs must exist";
      const_cast<ImmutableGraph *>(this)->in_csr_ = out_csr_->Transpose();
      return in_csr_;
    }
  }
  CSR::Ptr GetOutCSR() const {
    if (out_csr_) {
      return out_csr_;
    } else {
      CHECK(in_csr_ != nullptr) << "one of the CSRs must exist";
      const_cast<ImmutableGraph *>(this)->out_csr_ = in_csr_->Transpose();
      return out_csr_;
    }
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

  // Store the in-edges.
  CSR::Ptr in_csr_;
  // Store the out-edges.
  CSR::Ptr out_csr_;
  // Store the edge list indexed by edge id
  EdgeList::Ptr edge_list_;
  /*!
   * \brief Whether if this is a multigraph.
   *
   * When a multiedge is added, this flag switches to true.
   */
  bool is_multigraph_ = false;
};

}  // namespace dgl

#endif  // DGL_IMMUTABLE_GRAPH_H_

