/**
 *  Copyright (c) 2020 by Contributors
 * @file array/cpu/array_sort.cc
 * @brief Array sort CPU implementation
 */
#include <dgl/array.h>
#ifdef PARALLEL_ALGORITHMS
#include <parallel/algorithm>
#endif
#include <algorithm>
#include <iterator>

namespace {

template <typename V1, typename V2>
struct PairRef {
  PairRef() = delete;
  PairRef(const PairRef& other) = default;
  PairRef(PairRef&& other) = default;
  PairRef(V1* const r, V2* const c) : row(r), col(c) {}

  PairRef& operator=(const PairRef& other) {
    *row = *other.row;
    *col = *other.col;
    return *this;
  }
  PairRef& operator=(const std::pair<V1, V2>& val) {
    *row = std::get<0>(val);
    *col = std::get<1>(val);
    return *this;
  }

  operator std::pair<V1, V2>() const { return std::make_pair(*row, *col); }

  void Swap(const PairRef& other) const {
    std::swap(*row, *other.row);
    std::swap(*col, *other.col);
  }

  V1* row;
  V2* col;
};

using std::swap;
template <typename V1, typename V2>
void swap(const PairRef<V1, V2>& r1, const PairRef<V1, V2>& r2) {
  r1.Swap(r2);
}

template <typename V1, typename V2>
struct PairIterator
    : public std::iterator<
          std::random_access_iterator_tag, std::pair<V1, V2>, std::ptrdiff_t,
          std::pair<V1*, V2*>, PairRef<V1, V2>> {
  PairIterator() = default;
  PairIterator(const PairIterator& other) = default;
  PairIterator(PairIterator&& other) = default;
  PairIterator(V1* r, V2* c) : row(r), col(c) {}

  PairIterator& operator=(const PairIterator& other) = default;
  PairIterator& operator=(PairIterator&& other) = default;
  ~PairIterator() = default;

  bool operator==(const PairIterator& other) const { return row == other.row; }

  bool operator!=(const PairIterator& other) const { return row != other.row; }

  bool operator<(const PairIterator& other) const { return row < other.row; }

  bool operator>(const PairIterator& other) const { return row > other.row; }

  bool operator<=(const PairIterator& other) const { return row <= other.row; }

  bool operator>=(const PairIterator& other) const { return row >= other.row; }

  PairIterator& operator+=(const std::ptrdiff_t& movement) {
    row += movement;
    col += movement;
    return *this;
  }

  PairIterator& operator-=(const std::ptrdiff_t& movement) {
    row -= movement;
    col -= movement;
    return *this;
  }

  PairIterator& operator++() { return operator+=(1); }

  PairIterator& operator--() { return operator-=(1); }

  PairIterator operator++(int) {
    PairIterator ret(*this);
    operator++();
    return ret;
  }

  PairIterator operator--(int) {
    PairIterator ret(*this);
    operator--();
    return ret;
  }

  PairIterator operator+(const std::ptrdiff_t& movement) const {
    PairIterator ret(*this);
    ret += movement;
    return ret;
  }

  PairIterator operator-(const std::ptrdiff_t& movement) const {
    PairIterator ret(*this);
    ret -= movement;
    return ret;
  }

  std::ptrdiff_t operator-(const PairIterator& other) const {
    return row - other.row;
  }

  PairRef<V1, V2> operator*() const { return PairRef<V1, V2>(row, col); }
  PairRef<V1, V2> operator*() { return PairRef<V1, V2>(row, col); }

  // required for random access iterators in VS2019
  PairRef<V1, V2> operator[](size_t offset) const {
    return PairRef<V1, V2>(row + offset, col + offset);
  }

  V1* row;
  V2* col;
};

}  // namespace

namespace dgl {
using runtime::NDArray;
namespace aten {
namespace impl {

template <DGLDeviceType XPU, typename IdType>
std::pair<IdArray, IdArray> Sort(IdArray array, int /* num_bits */) {
  const int64_t nitem = array->shape[0];
  IdArray val = array.Clone();
  IdArray idx = aten::Range(0, nitem, 64, array->ctx);
  IdType* val_data = val.Ptr<IdType>();
  int64_t* idx_data = idx.Ptr<int64_t>();
  typedef std::pair<IdType, int64_t> Pair;
#ifdef PARALLEL_ALGORITHMS
  __gnu_parallel::sort(
#else
  std::sort(
#endif
      PairIterator<IdType, int64_t>(val_data, idx_data),
      PairIterator<IdType, int64_t>(val_data, idx_data) + nitem,
      [](const Pair& a, const Pair& b) {
        return std::get<0>(a) < std::get<0>(b);
      });
  return std::make_pair(val, idx);
}

template std::pair<IdArray, IdArray> Sort<kDGLCPU, int32_t>(
    IdArray, int num_bits);
template std::pair<IdArray, IdArray> Sort<kDGLCPU, int64_t>(
    IdArray, int num_bits);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
