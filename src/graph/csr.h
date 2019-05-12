#pragma once

#include <dgl/graph_interface.h>
#include <memory>

namespace dgl {

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

class CSR : public GraphInterface {
 public:
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

 private:
  //typedef std::shared_ptr<CSR> Ptr;

  //vector<int64_t> indptr;
  //vector<dgl_id_t> indices;
  //vector<dgl_id_t> edge_ids;
  IdArray indptr_, indices_, edge_ids_;

//
//  /* This gets the sum of vertex degrees in the range. */
//  uint64_t GetDegree(dgl_id_t start, dgl_id_t end) const {
//    return indptr[end] - indptr[start];
//  }
//
//  DegreeArray GetDegrees(IdArray vids) const;
//  EdgeArray GetEdges(dgl_id_t vid) const;
//  EdgeArray GetEdges(IdArray vids) const;
//  /* \brief this returns the start and end position of the column indices corresponding v. */
//  DGLIdIters GetIndexRef(dgl_id_t v) const {
//    const int64_t start = indptr[v];
//    const int64_t end = indptr[v + 1];
//    return DGLIdIters(indices.begin() + start, indices.begin() + end);
//  }
//  /*
//   * Read all edges and store them in the vector.
//   */
//  void ReadAllEdges(std::vector<Edge> *edges) const;
//  CSR::Ptr Transpose() const;
//  std::pair<CSR::Ptr, IdArray> VertexSubgraph(IdArray vids) const;
//  std::pair<CSR::Ptr, IdArray> EdgeSubgraph(IdArray eids,
//                                            std::shared_ptr<EdgeList> edge_list) const;
//  /*
//   * Construct a CSR from a list of edges.
//   *
//   * When constructing a CSR, we need to sort the edge list. To reduce the overhead,
//   * we simply sort on the input edge list. We allow sorting on both end points of an edge,
//   * which is specified by `sort_on`.
//   */
//  static CSR::Ptr FromEdges(std::vector<Edge> *edges, int sort_on, uint64_t num_nodes);

 private:
#ifndef _WIN32
  std::shared_ptr<runtime::SharedMemory> mem;
#endif  // _WIN32
};

}  // namespace dgl
