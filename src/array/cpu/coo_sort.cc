/**
 *  Copyright (c) 2020 by Contributors
 * @file array/cpu/coo_sort.cc
 * @brief COO sorting
 */
#include <dgl/array.h>
#ifdef PARALLEL_ALGORITHMS
#include <parallel/algorithm>
#endif
#include <algorithm>
#include <iterator>
#include <numeric>
#include <tuple>
#include <vector>

namespace {

template <typename IdType>
struct TupleRef {
  TupleRef() = delete;
  TupleRef(const TupleRef& other) = default;
  TupleRef(TupleRef&& other) = default;
  TupleRef(IdType* const r, IdType* const c, IdType* const d)
      : row(r), col(c), data(d) {}

  TupleRef& operator=(const TupleRef& other) {
    *row = *other.row;
    *col = *other.col;
    *data = *other.data;
    return *this;
  }
  TupleRef& operator=(const std::tuple<IdType, IdType, IdType>& val) {
    *row = std::get<0>(val);
    *col = std::get<1>(val);
    *data = std::get<2>(val);
    return *this;
  }

  operator std::tuple<IdType, IdType, IdType>() const {
    return std::make_tuple(*row, *col, *data);
  }

  void Swap(const TupleRef& other) const {
    std::swap(*row, *other.row);
    std::swap(*col, *other.col);
    std::swap(*data, *other.data);
  }

  IdType *row, *col, *data;
};

using std::swap;
template <typename IdType>
void swap(const TupleRef<IdType>& r1, const TupleRef<IdType>& r2) {
  r1.Swap(r2);
}

template <typename IdType>
struct CooIterator
    : public std::iterator<
          std::random_access_iterator_tag, std::tuple<IdType, IdType, IdType>,
          std::ptrdiff_t, std::tuple<IdType*, IdType*, IdType*>,
          TupleRef<IdType>> {
  CooIterator() = default;
  CooIterator(const CooIterator& other) = default;
  CooIterator(CooIterator&& other) = default;
  CooIterator(IdType* r, IdType* c, IdType* d) : row(r), col(c), data(d) {}

  CooIterator& operator=(const CooIterator& other) = default;
  CooIterator& operator=(CooIterator&& other) = default;
  ~CooIterator() = default;

  bool operator==(const CooIterator& other) const { return row == other.row; }

  bool operator!=(const CooIterator& other) const { return row != other.row; }

  bool operator<(const CooIterator& other) const { return row < other.row; }

  bool operator>(const CooIterator& other) const { return row > other.row; }

  bool operator<=(const CooIterator& other) const { return row <= other.row; }

  bool operator>=(const CooIterator& other) const { return row >= other.row; }

  CooIterator& operator+=(const std::ptrdiff_t& movement) {
    row += movement;
    col += movement;
    data += movement;
    return *this;
  }

  CooIterator& operator-=(const std::ptrdiff_t& movement) {
    row -= movement;
    col -= movement;
    data -= movement;
    return *this;
  }

  CooIterator& operator++() { return operator+=(1); }

  CooIterator& operator--() { return operator-=(1); }

  CooIterator operator++(int) {
    CooIterator ret(*this);
    operator++();
    return ret;
  }

  CooIterator operator--(int) {
    CooIterator ret(*this);
    operator--();
    return ret;
  }

  CooIterator operator+(const std::ptrdiff_t& movement) const {
    CooIterator ret(*this);
    ret += movement;
    return ret;
  }

  CooIterator operator-(const std::ptrdiff_t& movement) const {
    CooIterator ret(*this);
    ret -= movement;
    return ret;
  }

  std::ptrdiff_t operator-(const CooIterator& other) const {
    return row - other.row;
  }

  TupleRef<IdType> operator*() const {
    return TupleRef<IdType>(row, col, data);
  }
  TupleRef<IdType> operator*() { return TupleRef<IdType>(row, col, data); }

  // required for random access iterators in VS2019
  TupleRef<IdType> operator[](size_t offset) const {
    return TupleRef<IdType>(row + offset, col + offset, data + offset);
  }

  IdType *row, *col, *data;
};

}  // namespace

namespace dgl {
namespace aten {
namespace impl {

///////////////////////////// COOSort_ /////////////////////////////

template <DGLDeviceType XPU, typename IdType>
void COOSort_(COOMatrix* coo, bool sort_column) {
  const int64_t nnz = coo->row->shape[0];
  IdType* coo_row = coo->row.Ptr<IdType>();
  IdType* coo_col = coo->col.Ptr<IdType>();
  if (!COOHasData(*coo))
    coo->data = aten::Range(0, nnz, coo->row->dtype.bits, coo->row->ctx);
  IdType* coo_data = coo->data.Ptr<IdType>();

  typedef std::tuple<IdType, IdType, IdType> Tuple;

  // Arg sort
  if (sort_column) {
#ifdef PARALLEL_ALGORITHMS
    __gnu_parallel::sort(
#else
    std::sort(
#endif
        CooIterator<IdType>(coo_row, coo_col, coo_data),
        CooIterator<IdType>(coo_row, coo_col, coo_data) + nnz,
        [](const Tuple& a, const Tuple& b) {
          return (std::get<0>(a) != std::get<0>(b))
                     ? (std::get<0>(a) < std::get<0>(b))
                     : (std::get<1>(a) < std::get<1>(b));
        });
  } else {
#ifdef PARALLEL_ALGORITHMS
    __gnu_parallel::sort(
#else
    std::sort(
#endif
        CooIterator<IdType>(coo_row, coo_col, coo_data),
        CooIterator<IdType>(coo_row, coo_col, coo_data) + nnz,
        [](const Tuple& a, const Tuple& b) {
          return std::get<0>(a) < std::get<0>(b);
        });
  }

  coo->row_sorted = true;
  coo->col_sorted = sort_column;
}

template void COOSort_<kDGLCPU, int32_t>(COOMatrix*, bool);
template void COOSort_<kDGLCPU, int64_t>(COOMatrix*, bool);

///////////////////////////// COOIsSorted /////////////////////////////

template <DGLDeviceType XPU, typename IdType>
std::pair<bool, bool> COOIsSorted(COOMatrix coo) {
  const int64_t nnz = coo.row->shape[0];
  IdType* row = coo.row.Ptr<IdType>();
  IdType* col = coo.col.Ptr<IdType>();
  bool row_sorted = true;
  bool col_sorted = true;
  for (int64_t i = 1; row_sorted && i < nnz; ++i) {
    row_sorted = (row[i - 1] <= row[i]);
    col_sorted = col_sorted && (row[i - 1] < row[i] || col[i - 1] <= col[i]);
  }
  if (!row_sorted) col_sorted = false;
  return {row_sorted, col_sorted};
}

template std::pair<bool, bool> COOIsSorted<kDGLCPU, int32_t>(COOMatrix coo);
template std::pair<bool, bool> COOIsSorted<kDGLCPU, int64_t>(COOMatrix coo);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
