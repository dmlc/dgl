/*!
 *  Copyright (c) 2018 by Contributors
 * \file dgl/immutable_graph.h
 * \brief DGL immutable graph index class.
 */
#ifndef DGL_VECTOR_H_
#define DGL_VECTOR_H_

#include <algorithm>

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
  T *arr;
  size_t capacity;
  size_t curr;
  bool own;
};

}  // namespace dgl

#endif  // DGL_VECTOR_H_

