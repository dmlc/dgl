/**
 *  Copyright (c) 2020 by Contributors
 * @file dgl/array_iterator.h
 * @brief Various iterators.
 */
#ifndef DGL_ARRAY_ITERATOR_H_
#define DGL_ARRAY_ITERATOR_H_

#ifdef __CUDA_ARCH__
#define CUB_INLINE __host__ __device__ __forceinline__
#else
#define CUB_INLINE inline
#endif  // __CUDA_ARCH__

#include <algorithm>
#include <iterator>
#include <utility>

namespace dgl {
namespace aten {

using std::swap;

// Make std::pair work on both host and device
template <typename DType>
struct Pair {
  Pair() = default;
  Pair(const Pair& other) = default;
  Pair(Pair&& other) = default;
  CUB_INLINE Pair(DType a, DType b) : first(a), second(b) {}
  CUB_INLINE Pair& operator=(const Pair& other) {
    first = other.first;
    second = other.second;
    return *this;
  }
  CUB_INLINE operator std::pair<DType, DType>() const {
    return std::make_pair(first, second);
  }
  CUB_INLINE bool operator==(const Pair& other) const {
    return (first == other.first) && (second == other.second);
  }
  CUB_INLINE void swap(const Pair& other) const {
    std::swap(first, other.first);
    std::swap(second, other.second);
  }
  DType first, second;
};

template <typename DType>
CUB_INLINE void swap(const Pair<DType>& r1, const Pair<DType>& r2) {
  r1.swap(r2);
}

// PairRef and PairIterator that serves as an iterator over a pair of arrays in
// a zipped fashion like zip(a, b).
template <typename DType>
struct PairRef {
  PairRef() = delete;
  PairRef(const PairRef& other) = default;
  PairRef(PairRef&& other) = default;
  CUB_INLINE PairRef(DType* const r, DType* const c) : a(r), b(c) {}
  CUB_INLINE PairRef& operator=(const PairRef& other) {
    *a = *other.a;
    *b = *other.b;
    return *this;
  }
  CUB_INLINE PairRef& operator=(const Pair<DType>& val) {
    *a = val.first;
    *b = val.second;
    return *this;
  }
  CUB_INLINE operator Pair<DType>() const { return Pair<DType>(*a, *b); }
  CUB_INLINE operator std::pair<DType, DType>() const {
    return std::make_pair(*a, *b);
  }
  CUB_INLINE bool operator==(const PairRef& other) const {
    return (*a == *(other.a)) && (*b == *(other.b));
  }
  CUB_INLINE void swap(const PairRef& other) const {
    std::swap(*a, *other.a);
    std::swap(*b, *other.b);
  }
  DType *a, *b;
};

template <typename DType>
CUB_INLINE void swap(const PairRef<DType>& r1, const PairRef<DType>& r2) {
  r1.swap(r2);
}

template <typename DType>
struct PairIterator : public std::iterator<
                          std::random_access_iterator_tag, Pair<DType>,
                          std::ptrdiff_t, Pair<DType*>, PairRef<DType>> {
  PairIterator() = default;
  PairIterator(const PairIterator& other) = default;
  PairIterator(PairIterator&& other) = default;
  CUB_INLINE PairIterator(DType* x, DType* y) : a(x), b(y) {}
  PairIterator& operator=(const PairIterator& other) = default;
  PairIterator& operator=(PairIterator&& other) = default;
  ~PairIterator() = default;
  CUB_INLINE bool operator==(const PairIterator& other) const {
    return a == other.a;
  }
  CUB_INLINE bool operator!=(const PairIterator& other) const {
    return a != other.a;
  }
  CUB_INLINE bool operator<(const PairIterator& other) const {
    return a < other.a;
  }
  CUB_INLINE bool operator>(const PairIterator& other) const {
    return a > other.a;
  }
  CUB_INLINE bool operator<=(const PairIterator& other) const {
    return a <= other.a;
  }
  CUB_INLINE bool operator>=(const PairIterator& other) const {
    return a >= other.a;
  }
  CUB_INLINE PairIterator& operator+=(const std::ptrdiff_t& movement) {
    a += movement;
    b += movement;
    return *this;
  }
  CUB_INLINE PairIterator& operator-=(const std::ptrdiff_t& movement) {
    a -= movement;
    b -= movement;
    return *this;
  }
  CUB_INLINE PairIterator& operator++() {
    ++a;
    ++b;
    return *this;
  }
  CUB_INLINE PairIterator& operator--() {
    --a;
    --b;
    return *this;
  }
  CUB_INLINE PairIterator operator++(int) {
    PairIterator ret(*this);
    operator++();
    return ret;
  }
  CUB_INLINE PairIterator operator--(int) {
    PairIterator ret(*this);
    operator--();
    return ret;
  }
  CUB_INLINE PairIterator operator+(const std::ptrdiff_t& movement) const {
    return PairIterator(a + movement, b + movement);
  }
  CUB_INLINE PairIterator operator-(const std::ptrdiff_t& movement) const {
    return PairIterator(a - movement, b - movement);
  }
  CUB_INLINE std::ptrdiff_t operator-(const PairIterator& other) const {
    return a - other.a;
  }
  CUB_INLINE PairRef<DType> operator*() const { return PairRef<DType>(a, b); }
  CUB_INLINE PairRef<DType> operator*() { return PairRef<DType>(a, b); }
  CUB_INLINE PairRef<DType> operator[](size_t offset) const {
    return PairRef<DType>(a + offset, b + offset);
  }
  CUB_INLINE PairRef<DType> operator[](size_t offset) {
    return PairRef<DType>(a + offset, b + offset);
  }
  DType *a, *b;
};

};  // namespace aten
};  // namespace dgl

#endif  // DGL_ARRAY_ITERATOR_H_
