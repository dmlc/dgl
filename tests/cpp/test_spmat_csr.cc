#include <dgl/array.h>
#include <gtest/gtest.h>

#include "./common.h"

using namespace dgl;
using namespace dgl::runtime;

namespace {

template <typename IDX>
aten::CSRMatrix CSR1(DGLContext ctx = CTX) {
  // [[0, 1, 1, 0, 0],
  //  [1, 0, 0, 0, 0],
  //  [0, 0, 1, 1, 0],
  //  [0, 0, 0, 0, 0]]
  // data: [0, 2, 3, 1, 4]
  return aten::CSRMatrix(
      4, 5,
      aten::VecToIdArray(
          std::vector<IDX>({0, 2, 3, 5, 5}), sizeof(IDX) * 8, ctx),
      aten::VecToIdArray(
          std::vector<IDX>({1, 2, 0, 3, 2}), sizeof(IDX) * 8, ctx),
      aten::VecToIdArray(
          std::vector<IDX>({0, 2, 3, 4, 1}), sizeof(IDX) * 8, ctx),
      false);
}

template <typename IDX>
aten::CSRMatrix CSR2(DGLContext ctx = CTX) {
  // has duplicate entries
  // [[0, 1, 2, 0, 0],
  //  [1, 0, 0, 0, 0],
  //  [0, 0, 1, 1, 0],
  //  [0, 0, 0, 0, 0]]
  // data: [0, 2, 5, 3, 1, 4]
  return aten::CSRMatrix(
      4, 5,
      aten::VecToIdArray(
          std::vector<IDX>({0, 3, 4, 6, 6}), sizeof(IDX) * 8, ctx),
      aten::VecToIdArray(
          std::vector<IDX>({1, 2, 2, 0, 2, 3}), sizeof(IDX) * 8, ctx),
      aten::VecToIdArray(
          std::vector<IDX>({0, 2, 5, 3, 1, 4}), sizeof(IDX) * 8, ctx),
      false);
}

template <typename IDX>
aten::CSRMatrix CSR3(DGLContext ctx = CTX) {
  // has duplicate entries and the columns are not sorted
  // [[0, 1, 1, 1, 0, 0],
  //  [1, 0, 0, 0, 0, 0],
  //  [0, 0, 1, 1, 0, 0],
  //  [0, 0, 0, 0, 0, 0],
  //  [1, 1, 1, 0, 0, 0],
  //  [0, 0, 0, 1, 0, 0],
  //  [0, 0, 0, 0, 0, 0],
  //  [1, 2, 1, 1, 0, 0],
  //  [0, 1, 0, 0, 0, 1]],
  // data: [5, 2, 0, 3, 1, 4, 8, 7, 6, 9, 12, 13, 11, 10, 14, 15, 16]
  return aten::CSRMatrix(
      9, 6,
      aten::VecToIdArray(
          std::vector<IDX>({0, 3, 4, 6, 6, 9, 10, 10, 15, 17}), sizeof(IDX) * 8,
          ctx),
      aten::VecToIdArray(
          std::vector<IDX>({3, 2, 1, 0, 2, 3, 1, 2, 0, 3, 1, 2, 1, 3, 0, 5, 1}),
          sizeof(IDX) * 8, ctx),
      aten::VecToIdArray(
          std::vector<IDX>(
              {0, 2, 5, 3, 1, 4, 6, 8, 7, 9, 13, 10, 11, 14, 12, 16, 15}),
          sizeof(IDX) * 8, ctx),
      false);
}

template <typename IDX>
aten::COOMatrix COO1(DGLContext ctx = CTX) {
  // [[0, 1, 1, 0, 0],
  //  [1, 0, 0, 0, 0],
  //  [0, 0, 1, 1, 0],
  //  [0, 0, 0, 0, 0]]
  // data: [0, 2, 3, 1, 4]
  // row : [0, 2, 0, 1, 2]
  // col : [1, 2, 2, 0, 3]
  return aten::COOMatrix(
      4, 5,
      aten::VecToIdArray(
          std::vector<IDX>({0, 2, 0, 1, 2}), sizeof(IDX) * 8, ctx),
      aten::VecToIdArray(
          std::vector<IDX>({1, 2, 2, 0, 3}), sizeof(IDX) * 8, ctx),
      aten::VecToIdArray(
          std::vector<IDX>({0, 3, 1, 2, 4}), sizeof(IDX) * 8, ctx));
}

template <typename IDX>
aten::COOMatrix COO2(DGLContext ctx = CTX) {
  // has duplicate entries
  // [[0, 1, 2, 0, 0],
  //  [1, 0, 0, 0, 0],
  //  [0, 0, 1, 1, 0],
  //  [0, 0, 0, 0, 0]]
  // data: [0, 2, 5, 3, 1, 4]
  // row : [0, 2, 0, 1, 2, 0]
  // col : [1, 2, 2, 0, 3, 2]
  return aten::COOMatrix(
      4, 5,
      aten::VecToIdArray(
          std::vector<IDX>({0, 2, 0, 1, 2, 0}), sizeof(IDX) * 8, ctx),
      aten::VecToIdArray(
          std::vector<IDX>({1, 2, 2, 0, 3, 2}), sizeof(IDX) * 8, ctx),
      aten::VecToIdArray(
          std::vector<IDX>({0, 1, 2, 3, 4, 5}), sizeof(IDX) * 8, ctx));
}

template <typename IDX>
aten::CSRMatrix SR_CSR3(DGLContext ctx) {
  // [[0, 1, 2, 0, 0],
  //  [1, 0, 0, 0, 0],
  //  [0, 0, 1, 1, 0],
  //  [0, 0, 0, 0, 0]]
  return aten::CSRMatrix(
      4, 5,
      aten::VecToIdArray(
          std::vector<IDX>({0, 3, 4, 6, 6}), sizeof(IDX) * 8, ctx),
      aten::VecToIdArray(
          std::vector<IDX>({2, 1, 2, 0, 2, 3}), sizeof(IDX) * 8, ctx),
      aten::VecToIdArray(
          std::vector<IDX>({0, 2, 5, 3, 1, 4}), sizeof(IDX) * 8, ctx),
      false);
}

template <typename IDX>
aten::CSRMatrix SRC_CSR3(DGLContext ctx) {
  // [[0, 1, 2, 0, 0],
  //  [1, 0, 0, 0, 0],
  //  [0, 0, 1, 1, 0],
  //  [0, 0, 0, 0, 0]]
  return aten::CSRMatrix(
      4, 5,
      aten::VecToIdArray(
          std::vector<IDX>({0, 3, 4, 6, 6}), sizeof(IDX) * 8, ctx),
      aten::VecToIdArray(
          std::vector<IDX>({1, 2, 2, 0, 2, 3}), sizeof(IDX) * 8, ctx),
      aten::VecToIdArray(
          std::vector<IDX>({2, 0, 5, 3, 1, 4}), sizeof(IDX) * 8, ctx),
      false);
}

template <typename IDX>
aten::COOMatrix COO3(DGLContext ctx) {
  // has duplicate entries
  // [[0, 1, 2, 0, 0],
  //  [1, 0, 0, 0, 0],
  //  [0, 0, 1, 1, 0],
  //  [0, 0, 0, 0, 0]]
  // row : [0, 2, 0, 1, 2, 0]
  // col : [2, 2, 1, 0, 3, 2]
  return aten::COOMatrix(
      4, 5,
      aten::VecToIdArray(
          std::vector<IDX>({0, 2, 0, 1, 2, 0}), sizeof(IDX) * 8, ctx),
      aten::VecToIdArray(
          std::vector<IDX>({2, 2, 1, 0, 3, 2}), sizeof(IDX) * 8, ctx));
}

}  // namespace

template <typename IDX>
void _TestCSRIsNonZero1(DGLContext ctx) {
  auto csr = CSR1<IDX>(ctx);
  ASSERT_TRUE(aten::CSRIsNonZero(csr, 0, 1));
  ASSERT_FALSE(aten::CSRIsNonZero(csr, 0, 0));
  IdArray r =
      aten::VecToIdArray(std::vector<IDX>({2, 2, 0, 0}), sizeof(IDX) * 8, ctx);
  IdArray c =
      aten::VecToIdArray(std::vector<IDX>({1, 1, 1, 3}), sizeof(IDX) * 8, ctx);
  IdArray x = aten::CSRIsNonZero(csr, r, c);
  IdArray tx =
      aten::VecToIdArray(std::vector<IDX>({0, 0, 1, 0}), sizeof(IDX) * 8, ctx);
  ASSERT_TRUE(ArrayEQ<IDX>(x, tx));
}

template <typename IDX>
void _TestCSRIsNonZero2(DGLContext ctx) {
  auto csr = CSR3<IDX>(ctx);
  ASSERT_TRUE(aten::CSRIsNonZero(csr, 0, 1));
  ASSERT_FALSE(aten::CSRIsNonZero(csr, 0, 0));
  IdArray r = aten::VecToIdArray(
      std::vector<IDX>({
          0,
          0,
          0,
          0,
          0,
      }),
      sizeof(IDX) * 8, ctx);
  IdArray c = aten::VecToIdArray(
      std::vector<IDX>({
          0,
          1,
          2,
          3,
          4,
      }),
      sizeof(IDX) * 8, ctx);
  IdArray x = aten::CSRIsNonZero(csr, r, c);
  IdArray tx = aten::VecToIdArray(
      std::vector<IDX>({0, 1, 1, 1, 0}), sizeof(IDX) * 8, ctx);
  ASSERT_TRUE(ArrayEQ<IDX>(x, tx)) << " x = " << x << ", tx = " << tx;
}

TEST(SpmatTest, TestCSRIsNonZero) {
  _TestCSRIsNonZero1<int32_t>(CPU);
  _TestCSRIsNonZero1<int64_t>(CPU);
  _TestCSRIsNonZero2<int32_t>(CPU);
  _TestCSRIsNonZero2<int64_t>(CPU);
#ifdef DGL_USE_CUDA
  _TestCSRIsNonZero1<int32_t>(GPU);
  _TestCSRIsNonZero1<int64_t>(GPU);
  _TestCSRIsNonZero2<int32_t>(GPU);
  _TestCSRIsNonZero2<int64_t>(GPU);
#endif
}

template <typename IDX>
void _TestCSRGetRowNNZ(DGLContext ctx) {
  auto csr = CSR2<IDX>(ctx);
  ASSERT_EQ(aten::CSRGetRowNNZ(csr, 0), 3);
  ASSERT_EQ(aten::CSRGetRowNNZ(csr, 3), 0);
  IdArray r =
      aten::VecToIdArray(std::vector<IDX>({0, 3}), sizeof(IDX) * 8, ctx);
  IdArray x = aten::CSRGetRowNNZ(csr, r);
  IdArray tx =
      aten::VecToIdArray(std::vector<IDX>({3, 0}), sizeof(IDX) * 8, ctx);
  ASSERT_TRUE(ArrayEQ<IDX>(x, tx));
}

TEST(SpmatTest, TestCSRGetRowNNZ) {
  _TestCSRGetRowNNZ<int32_t>(CPU);
  _TestCSRGetRowNNZ<int64_t>(CPU);
#ifdef DGL_USE_CUDA
  _TestCSRGetRowNNZ<int32_t>(GPU);
  _TestCSRGetRowNNZ<int64_t>(GPU);
#endif
}

template <typename IDX>
void _TestCSRGetRowColumnIndices(DGLContext ctx) {
  auto csr = CSR2<IDX>(ctx);
  auto x = aten::CSRGetRowColumnIndices(csr, 0);
  auto tx =
      aten::VecToIdArray(std::vector<IDX>({1, 2, 2}), sizeof(IDX) * 8, ctx);
  ASSERT_TRUE(ArrayEQ<IDX>(x, tx));
  x = aten::CSRGetRowColumnIndices(csr, 1);
  tx = aten::VecToIdArray(std::vector<IDX>({0}), sizeof(IDX) * 8, ctx);
  ASSERT_TRUE(ArrayEQ<IDX>(x, tx));
  x = aten::CSRGetRowColumnIndices(csr, 3);
  tx = aten::VecToIdArray(std::vector<IDX>({}), sizeof(IDX) * 8, ctx);
  ASSERT_TRUE(ArrayEQ<IDX>(x, tx));
}

TEST(SpmatTest, TestCSRGetRowColumnIndices) {
  _TestCSRGetRowColumnIndices<int32_t>(CPU);
  _TestCSRGetRowColumnIndices<int64_t>(CPU);
#ifdef DGL_USE_CUDA
  _TestCSRGetRowColumnIndices<int32_t>(GPU);
  _TestCSRGetRowColumnIndices<int64_t>(GPU);
#endif
}

template <typename IDX>
void _TestCSRGetRowData(DGLContext ctx) {
  auto csr = CSR2<IDX>(ctx);
  auto x = aten::CSRGetRowData(csr, 0);
  auto tx =
      aten::VecToIdArray(std::vector<IDX>({0, 2, 5}), sizeof(IDX) * 8, ctx);
  ASSERT_TRUE(ArrayEQ<IDX>(x, tx));
  x = aten::CSRGetRowData(csr, 1);
  tx = aten::VecToIdArray(std::vector<IDX>({3}), sizeof(IDX) * 8, ctx);
  ASSERT_TRUE(ArrayEQ<IDX>(x, tx));
  x = aten::CSRGetRowData(csr, 3);
  tx = aten::VecToIdArray(std::vector<IDX>({}), sizeof(IDX) * 8, ctx);
  ASSERT_TRUE(ArrayEQ<IDX>(x, tx));
}

TEST(SpmatTest, TestCSRGetRowData) {
  _TestCSRGetRowData<int32_t>(CPU);
  _TestCSRGetRowData<int64_t>(CPU);
#ifdef DGL_USE_CUDA
  _TestCSRGetRowData<int32_t>(GPU);
  _TestCSRGetRowData<int64_t>(GPU);
#endif
}

template <typename IDX>
void _TestCSRGetData(DGLContext ctx) {
  auto csr = CSR2<IDX>(ctx);
  // test get all data
  auto x = aten::CSRGetAllData(csr, 0, 0);
  auto tx = aten::VecToIdArray(std::vector<IDX>({}), sizeof(IDX) * 8, ctx);
  ASSERT_TRUE(ArrayEQ<IDX>(x, tx));
  x = aten::CSRGetAllData(csr, 0, 2);
  tx = aten::VecToIdArray(std::vector<IDX>({2, 5}), sizeof(IDX) * 8, ctx);
  ASSERT_TRUE(ArrayEQ<IDX>(x, tx));

  // test get data
  auto r =
      aten::VecToIdArray(std::vector<IDX>({0, 0, 0}), sizeof(IDX) * 8, ctx);
  auto c =
      aten::VecToIdArray(std::vector<IDX>({0, 1, 2}), sizeof(IDX) * 8, ctx);
  x = aten::CSRGetData(csr, r, c);
  tx = aten::VecToIdArray(std::vector<IDX>({-1, 0, 2}), sizeof(IDX) * 8, ctx);
  ASSERT_TRUE(ArrayEQ<IDX>(x, tx));

  // test get data on sorted
  csr = aten::CSRSort(csr);
  r = aten::VecToIdArray(std::vector<IDX>({0, 0, 0}), sizeof(IDX) * 8, ctx);
  c = aten::VecToIdArray(std::vector<IDX>({0, 1, 2}), sizeof(IDX) * 8, ctx);
  x = aten::CSRGetData(csr, r, c);
  tx = aten::VecToIdArray(std::vector<IDX>({-1, 0, 2}), sizeof(IDX) * 8, ctx);
  ASSERT_TRUE(ArrayEQ<IDX>(x, tx));

  // test get data w/ broadcasting
  r = aten::VecToIdArray(std::vector<IDX>({0}), sizeof(IDX) * 8, ctx);
  c = aten::VecToIdArray(std::vector<IDX>({0, 1, 2}), sizeof(IDX) * 8, ctx);
  x = aten::CSRGetData(csr, r, c);
  tx = aten::VecToIdArray(std::vector<IDX>({-1, 0, 2}), sizeof(IDX) * 8, ctx);
  ASSERT_TRUE(ArrayEQ<IDX>(x, tx));
}

TEST(SpmatTest, CSRGetData) {
  _TestCSRGetData<int32_t>(CPU);
  _TestCSRGetData<int64_t>(CPU);
#ifdef DGL_USE_CUDA
  _TestCSRGetData<int32_t>(GPU);
  _TestCSRGetData<int64_t>(GPU);
#endif
}

template <typename IDX>
void _TestCSRGetDataAndIndices(DGLContext ctx) {
  auto csr = CSR2<IDX>(ctx);
  auto r =
      aten::VecToIdArray(std::vector<IDX>({0, 0, 0}), sizeof(IDX) * 8, ctx);
  auto c =
      aten::VecToIdArray(std::vector<IDX>({0, 1, 2}), sizeof(IDX) * 8, ctx);
  auto x = aten::CSRGetDataAndIndices(csr, r, c);
  auto tr =
      aten::VecToIdArray(std::vector<IDX>({0, 0, 0}), sizeof(IDX) * 8, ctx);
  auto tc =
      aten::VecToIdArray(std::vector<IDX>({1, 2, 2}), sizeof(IDX) * 8, ctx);
  auto td =
      aten::VecToIdArray(std::vector<IDX>({0, 2, 5}), sizeof(IDX) * 8, ctx);
  ASSERT_TRUE(ArrayEQ<IDX>(x[0], tr));
  ASSERT_TRUE(ArrayEQ<IDX>(x[1], tc));
  ASSERT_TRUE(ArrayEQ<IDX>(x[2], td));
}

TEST(SpmatTest, CSRGetDataAndIndices) {
  _TestCSRGetDataAndIndices<int32_t>(CPU);
  _TestCSRGetDataAndIndices<int64_t>(CPU);
#ifdef DGL_USE_CUDA
  _TestCSRGetDataAndIndices<int32_t>(GPU);
  _TestCSRGetDataAndIndices<int64_t>(GPU);
#endif
}

template <typename IDX>
void _TestCSRTranspose(DGLContext ctx) {
  auto csr = CSR2<IDX>(ctx);
  auto csr_t = aten::CSRTranspose(csr);
  // [[0, 1, 0, 0],
  //  [1, 0, 0, 0],
  //  [2, 0, 1, 0],
  //  [0, 0, 1, 0],
  //  [0, 0, 0, 0]]
  // data: [3, 0, 2, 5, 1, 4]
  ASSERT_EQ(csr_t.num_rows, 5);
  ASSERT_EQ(csr_t.num_cols, 4);
  auto tp = aten::VecToIdArray(
      std::vector<IDX>({0, 1, 2, 5, 6, 6}), sizeof(IDX) * 8, ctx);
  auto ti = aten::VecToIdArray(
      std::vector<IDX>({1, 0, 0, 0, 2, 2}), sizeof(IDX) * 8, ctx);
  auto td = aten::VecToIdArray(
      std::vector<IDX>({3, 0, 2, 5, 1, 4}), sizeof(IDX) * 8, ctx);
  ASSERT_TRUE(ArrayEQ<IDX>(csr_t.indptr, tp));
  ASSERT_TRUE(ArrayEQ<IDX>(csr_t.indices, ti));
  ASSERT_TRUE(ArrayEQ<IDX>(csr_t.data, td));
}

TEST(SpmatTest, CSRTranspose) {
  _TestCSRTranspose<int32_t>(CPU);
  _TestCSRTranspose<int64_t>(CPU);
#ifdef DGL_USE_CUDA
  _TestCSRTranspose<int32_t>(GPU);
  _TestCSRTranspose<int64_t>(GPU);
#endif
}

template <typename IDX>
void _TestCSRToCOO(DGLContext ctx) {
  auto csr = CSR2<IDX>(ctx);
  {
    auto coo = CSRToCOO(csr, false);
    ASSERT_EQ(coo.num_rows, 4);
    ASSERT_EQ(coo.num_cols, 5);
    ASSERT_TRUE(coo.row_sorted);
    auto tr = aten::VecToIdArray(
        std::vector<IDX>({0, 0, 0, 1, 2, 2}), sizeof(IDX) * 8, ctx);
    ASSERT_TRUE(ArrayEQ<IDX>(coo.row, tr));
    ASSERT_TRUE(ArrayEQ<IDX>(coo.col, csr.indices));
    ASSERT_TRUE(ArrayEQ<IDX>(coo.data, csr.data));

    // convert from sorted csr
    auto s_csr = CSRSort(csr);
    coo = CSRToCOO(s_csr, false);
    ASSERT_EQ(coo.num_rows, 4);
    ASSERT_EQ(coo.num_cols, 5);
    ASSERT_TRUE(coo.row_sorted);
    ASSERT_TRUE(coo.col_sorted);
    tr = aten::VecToIdArray(
        std::vector<IDX>({0, 0, 0, 1, 2, 2}), sizeof(IDX) * 8, ctx);
    ASSERT_TRUE(ArrayEQ<IDX>(coo.row, tr));
    ASSERT_TRUE(ArrayEQ<IDX>(coo.col, s_csr.indices));
    ASSERT_TRUE(ArrayEQ<IDX>(coo.data, s_csr.data));
  }
  {
    auto coo = CSRToCOO(csr, true);
    ASSERT_EQ(coo.num_rows, 4);
    ASSERT_EQ(coo.num_cols, 5);
    auto tcoo = COO2<IDX>(ctx);
    ASSERT_TRUE(ArrayEQ<IDX>(coo.row, tcoo.row));
    ASSERT_TRUE(ArrayEQ<IDX>(coo.col, tcoo.col));
  }
}

TEST(SpmatTest, CSRToCOO) {
  _TestCSRToCOO<int32_t>(CPU);
  _TestCSRToCOO<int64_t>(CPU);
#if DGL_USE_CUDA
  _TestCSRToCOO<int32_t>(GPU);
  _TestCSRToCOO<int64_t>(GPU);
#endif
}

template <typename IDX>
void _TestCSRSliceRows(DGLContext ctx) {
  auto csr = CSR2<IDX>(ctx);
  auto x = aten::CSRSliceRows(csr, 1, 4);
  //  [1, 0, 0, 0, 0],
  //  [0, 0, 1, 1, 0],
  //  [0, 0, 0, 0, 0]]
  // data: [3, 1, 4]
  ASSERT_EQ(x.num_rows, 3);
  ASSERT_EQ(x.num_cols, 5);
  auto tp =
      aten::VecToIdArray(std::vector<IDX>({0, 1, 3, 3}), sizeof(IDX) * 8, ctx);
  auto ti =
      aten::VecToIdArray(std::vector<IDX>({0, 2, 3}), sizeof(IDX) * 8, ctx);
  auto td =
      aten::VecToIdArray(std::vector<IDX>({3, 1, 4}), sizeof(IDX) * 8, ctx);
  ASSERT_TRUE(ArrayEQ<IDX>(x.indptr, tp));
  ASSERT_TRUE(ArrayEQ<IDX>(x.indices, ti));
  ASSERT_TRUE(ArrayEQ<IDX>(x.data, td));

  auto r =
      aten::VecToIdArray(std::vector<IDX>({0, 1, 3}), sizeof(IDX) * 8, ctx);
  x = aten::CSRSliceRows(csr, r);
  // [[0, 1, 2, 0, 0],
  //  [1, 0, 0, 0, 0],
  //  [0, 0, 0, 0, 0]]
  // data: [0, 2, 5, 3]
  tp = aten::VecToIdArray(std::vector<IDX>({0, 3, 4, 4}), sizeof(IDX) * 8, ctx);
  ti = aten::VecToIdArray(std::vector<IDX>({1, 2, 2, 0}), sizeof(IDX) * 8, ctx);
  td = aten::VecToIdArray(std::vector<IDX>({0, 2, 5, 3}), sizeof(IDX) * 8, ctx);
  ASSERT_TRUE(ArrayEQ<IDX>(x.indptr, tp));
  ASSERT_TRUE(ArrayEQ<IDX>(x.indices, ti));
  ASSERT_TRUE(ArrayEQ<IDX>(x.data, td));

  // Testing non-increasing row id based slicing
  r = aten::VecToIdArray(std::vector<IDX>({3, 2, 1}), sizeof(IDX) * 8, ctx);
  x = aten::CSRSliceRows(csr, r);
  // [[0, 0, 0, 0, 0],
  //  [0, 0, 1, 1, 0],
  //  [1, 0, 0, 0, 0]]
  // data: [1, 4, 3]
  tp = aten::VecToIdArray(std::vector<IDX>({0, 0, 2, 3}), sizeof(IDX) * 8, ctx);
  ti = aten::VecToIdArray(std::vector<IDX>({2, 3, 0}), sizeof(IDX) * 8, ctx);
  td = aten::VecToIdArray(std::vector<IDX>({1, 4, 3}), sizeof(IDX) * 8, ctx);
  ASSERT_TRUE(ArrayEQ<IDX>(x.indptr, tp));
  ASSERT_TRUE(ArrayEQ<IDX>(x.indices, ti));
  ASSERT_TRUE(ArrayEQ<IDX>(x.data, td));

  // Testing zero-degree row slicing with different rows
  r = aten::VecToIdArray(
      std::vector<IDX>({1, 3, 0, 3, 2}), sizeof(IDX) * 8, ctx);
  x = aten::CSRSliceRows(csr, r);
  // [[1, 0, 0, 0, 0],
  //  [0, 0, 0, 0, 0],
  //  [0, 1, 2, 0, 0],
  //  [0, 0, 0, 0, 0],
  //  [0, 0, 1, 1, 0]]
  // data: [3, 0, 2, 5, 1, 4]
  tp = aten::VecToIdArray(
      std::vector<IDX>({0, 1, 1, 4, 4, 6}), sizeof(IDX) * 8, ctx);
  ti = aten::VecToIdArray(
      std::vector<IDX>({0, 1, 2, 2, 2, 3}), sizeof(IDX) * 8, ctx);
  td = aten::VecToIdArray(
      std::vector<IDX>({3, 0, 2, 5, 1, 4}), sizeof(IDX) * 8, ctx);
  ASSERT_TRUE(ArrayEQ<IDX>(x.indptr, tp));
  ASSERT_TRUE(ArrayEQ<IDX>(x.indices, ti));
  ASSERT_TRUE(ArrayEQ<IDX>(x.data, td));

  // Testing empty output (i.e. sliced rows will be zero-degree)
  r = aten::VecToIdArray(std::vector<IDX>({3, 3, 3}), sizeof(IDX) * 8, ctx);
  x = aten::CSRSliceRows(csr, r);
  // [[0, 0, 0, 0, 0],
  //  [0, 0, 0, 0, 0],
  //  [0, 0, 0, 0, 0]]
  // data: []
  tp = aten::VecToIdArray(std::vector<IDX>({0, 0, 0, 0}), sizeof(IDX) * 8, ctx);
  ti = aten::VecToIdArray(std::vector<IDX>({}), sizeof(IDX) * 8, ctx);
  td = aten::VecToIdArray(std::vector<IDX>({}), sizeof(IDX) * 8, ctx);
  ASSERT_TRUE(ArrayEQ<IDX>(x.indptr, tp));
  ASSERT_TRUE(ArrayEQ<IDX>(x.indices, ti));
  ASSERT_TRUE(ArrayEQ<IDX>(x.data, td));

  // Testing constant output: we pick last row with at least one nnz
  r = aten::VecToIdArray(std::vector<IDX>({2, 2, 2}), sizeof(IDX) * 8, ctx);
  x = aten::CSRSliceRows(csr, r);
  // [[0, 0, 1, 1, 0],
  //  [0, 0, 1, 1, 0],
  //  [0, 0, 1, 1, 0]]
  // data: [1, 4, 1, 4, 1, 4]
  tp = aten::VecToIdArray(std::vector<IDX>({0, 2, 4, 6}), sizeof(IDX) * 8, ctx);
  ti = aten::VecToIdArray(
      std::vector<IDX>({2, 3, 2, 3, 2, 3}), sizeof(IDX) * 8, ctx);
  td = aten::VecToIdArray(
      std::vector<IDX>({1, 4, 1, 4, 1, 4}), sizeof(IDX) * 8, ctx);
  ASSERT_TRUE(ArrayEQ<IDX>(x.indptr, tp));
  ASSERT_TRUE(ArrayEQ<IDX>(x.indices, ti));
  ASSERT_TRUE(ArrayEQ<IDX>(x.data, td));
}

TEST(SpmatTest, TestCSRSliceRows) {
  _TestCSRSliceRows<int32_t>(CPU);
  _TestCSRSliceRows<int64_t>(CPU);
#ifdef DGL_USE_CUDA
  _TestCSRSliceRows<int32_t>(GPU);
  _TestCSRSliceRows<int64_t>(GPU);
#endif
}

template <typename IDX>
void _TestCSRSliceMatrix1(DGLContext ctx) {
  auto csr = CSR2<IDX>(ctx);
  {
    // square
    auto r =
        aten::VecToIdArray(std::vector<IDX>({0, 1, 3}), sizeof(IDX) * 8, ctx);
    auto c =
        aten::VecToIdArray(std::vector<IDX>({1, 2, 3}), sizeof(IDX) * 8, ctx);
    auto x = aten::CSRSliceMatrix(csr, r, c);
    // [[1, 2, 0],
    //  [0, 0, 0],
    //  [0, 0, 0]]
    // data: [0, 2, 5]
    ASSERT_EQ(x.num_rows, 3);
    ASSERT_EQ(x.num_cols, 3);
    auto tp = aten::VecToIdArray(
        std::vector<IDX>({0, 3, 3, 3}), sizeof(IDX) * 8, ctx);
    auto ti =
        aten::VecToIdArray(std::vector<IDX>({0, 1, 1}), sizeof(IDX) * 8, ctx);
    auto td =
        aten::VecToIdArray(std::vector<IDX>({0, 2, 5}), sizeof(IDX) * 8, ctx);
    ASSERT_TRUE(ArrayEQ<IDX>(x.indptr, tp));
    ASSERT_TRUE(ArrayEQ<IDX>(x.indices, ti));
    ASSERT_TRUE(ArrayEQ<IDX>(x.data, td));
  }
  {
    // non-square
    auto r =
        aten::VecToIdArray(std::vector<IDX>({0, 1, 2}), sizeof(IDX) * 8, ctx);
    auto c = aten::VecToIdArray(std::vector<IDX>({0, 1}), sizeof(IDX) * 8, ctx);
    auto x = aten::CSRSliceMatrix(csr, r, c);
    // [[0, 1],
    //  [1, 0],
    //  [0, 0]]
    // data: [0, 3]
    ASSERT_EQ(x.num_rows, 3);
    ASSERT_EQ(x.num_cols, 2);
    auto tp = aten::VecToIdArray(
        std::vector<IDX>({0, 1, 2, 2}), sizeof(IDX) * 8, ctx);
    auto ti =
        aten::VecToIdArray(std::vector<IDX>({1, 0}), sizeof(IDX) * 8, ctx);
    auto td =
        aten::VecToIdArray(std::vector<IDX>({0, 3}), sizeof(IDX) * 8, ctx);
    ASSERT_TRUE(ArrayEQ<IDX>(x.indptr, tp));
    ASSERT_TRUE(ArrayEQ<IDX>(x.indices, ti));
    ASSERT_TRUE(ArrayEQ<IDX>(x.data, td));
  }
  {
    // empty slice
    auto r = aten::VecToIdArray(std::vector<IDX>({2, 3}), sizeof(IDX) * 8, ctx);
    auto c = aten::VecToIdArray(std::vector<IDX>({0, 1}), sizeof(IDX) * 8, ctx);
    auto x = aten::CSRSliceMatrix(csr, r, c);
    // [[0, 0],
    //  [0, 0]]
    // data: []
    ASSERT_EQ(x.num_rows, 2);
    ASSERT_EQ(x.num_cols, 2);
    auto tp =
        aten::VecToIdArray(std::vector<IDX>({0, 0, 0}), sizeof(IDX) * 8, ctx);
    auto ti = aten::VecToIdArray(std::vector<IDX>({}), sizeof(IDX) * 8, ctx);
    auto td = aten::VecToIdArray(std::vector<IDX>({}), sizeof(IDX) * 8, ctx);
    ASSERT_TRUE(ArrayEQ<IDX>(x.indptr, tp));
    ASSERT_TRUE(ArrayEQ<IDX>(x.indices, ti));
    ASSERT_TRUE(ArrayEQ<IDX>(x.data, td));
  }
}

template <typename IDX>
void _TestCSRSliceMatrix2(DGLContext ctx) {
  auto csr = CSR3<IDX>(ctx);
  {
    // square
    auto r =
        aten::VecToIdArray(std::vector<IDX>({0, 1, 3}), sizeof(IDX) * 8, ctx);
    auto c =
        aten::VecToIdArray(std::vector<IDX>({1, 2, 3}), sizeof(IDX) * 8, ctx);
    auto x = aten::CSRSliceMatrix(csr, r, c);
    // [[1, 1, 1],
    //  [0, 0, 0],
    //  [0, 0, 0]]
    // data: [5, 2, 0]
    ASSERT_EQ(x.num_rows, 3);
    ASSERT_EQ(x.num_cols, 3);
    auto tp = aten::VecToIdArray(
        std::vector<IDX>({0, 3, 3, 3}), sizeof(IDX) * 8, ctx);
    // indexes are in reverse order in CSR3
    auto ti =
        aten::VecToIdArray(std::vector<IDX>({2, 1, 0}), sizeof(IDX) * 8, ctx);
    auto td =
        aten::VecToIdArray(std::vector<IDX>({0, 2, 5}), sizeof(IDX) * 8, ctx);
    ASSERT_TRUE(ArrayEQ<IDX>(x.indptr, tp));
    ASSERT_TRUE(ArrayEQ<IDX>(x.indices, ti));
    ASSERT_TRUE(ArrayEQ<IDX>(x.data, td));
  }
  {
    // non-square
    auto r =
        aten::VecToIdArray(std::vector<IDX>({0, 1, 2}), sizeof(IDX) * 8, ctx);
    auto c = aten::VecToIdArray(std::vector<IDX>({0, 1}), sizeof(IDX) * 8, ctx);
    auto x = aten::CSRSliceMatrix(csr, r, c);
    // [[0, 1],
    //  [1, 0],
    //  [0, 0]]
    // data: [0, 3]
    ASSERT_EQ(x.num_rows, 3);
    ASSERT_EQ(x.num_cols, 2);
    auto tp = aten::VecToIdArray(
        std::vector<IDX>({0, 1, 2, 2}), sizeof(IDX) * 8, ctx);
    auto ti =
        aten::VecToIdArray(std::vector<IDX>({1, 0}), sizeof(IDX) * 8, ctx);
    auto td =
        aten::VecToIdArray(std::vector<IDX>({5, 3}), sizeof(IDX) * 8, ctx);
    ASSERT_TRUE(ArrayEQ<IDX>(x.indptr, tp));
    ASSERT_TRUE(ArrayEQ<IDX>(x.indices, ti));
    ASSERT_TRUE(ArrayEQ<IDX>(x.data, td));
  }
  {
    // empty slice
    auto r = aten::VecToIdArray(std::vector<IDX>({2, 3}), sizeof(IDX) * 8, ctx);
    auto c = aten::VecToIdArray(std::vector<IDX>({0, 1}), sizeof(IDX) * 8, ctx);
    auto x = aten::CSRSliceMatrix(csr, r, c);
    // [[0, 0],
    //  [0, 0]]
    // data: []
    ASSERT_EQ(x.num_rows, 2);
    ASSERT_EQ(x.num_cols, 2);
    auto tp =
        aten::VecToIdArray(std::vector<IDX>({0, 0, 0}), sizeof(IDX) * 8, ctx);
    auto ti = aten::VecToIdArray(std::vector<IDX>({}), sizeof(IDX) * 8, ctx);
    auto td = aten::VecToIdArray(std::vector<IDX>({}), sizeof(IDX) * 8, ctx);
    ASSERT_TRUE(ArrayEQ<IDX>(x.indptr, tp));
    ASSERT_TRUE(ArrayEQ<IDX>(x.indices, ti));
    ASSERT_TRUE(ArrayEQ<IDX>(x.data, td));
  }
}

TEST(SpmatTest, CSRSliceMatrix) {
  _TestCSRSliceMatrix1<int32_t>(CPU);
  _TestCSRSliceMatrix1<int64_t>(CPU);
  _TestCSRSliceMatrix2<int32_t>(CPU);
  _TestCSRSliceMatrix2<int64_t>(CPU);
#ifdef DGL_USE_CUDA
  _TestCSRSliceMatrix1<int32_t>(GPU);
  _TestCSRSliceMatrix1<int64_t>(GPU);
  _TestCSRSliceMatrix2<int32_t>(GPU);
  _TestCSRSliceMatrix2<int64_t>(GPU);
#endif
}

template <typename IDX>
void _TestCSRHasDuplicate(DGLContext ctx) {
  auto csr = CSR1<IDX>(ctx);
  ASSERT_FALSE(aten::CSRHasDuplicate(csr));
  csr = CSR2<IDX>(ctx);
  ASSERT_TRUE(aten::CSRHasDuplicate(csr));
}

TEST(SpmatTest, CSRHasDuplicate) {
  _TestCSRHasDuplicate<int32_t>(CPU);
  _TestCSRHasDuplicate<int64_t>(CPU);
#ifdef DGL_USE_CUDA
  _TestCSRHasDuplicate<int32_t>(GPU);
  _TestCSRHasDuplicate<int64_t>(GPU);
#endif
}

template <typename IDX>
void _TestCSRSort(DGLContext ctx) {
  auto csr = CSR1<IDX>(ctx);
  ASSERT_FALSE(aten::CSRIsSorted(csr));
  auto csr1 = aten::CSRSort(csr);
  ASSERT_FALSE(aten::CSRIsSorted(csr));
  ASSERT_TRUE(aten::CSRIsSorted(csr1));
  ASSERT_TRUE(csr1.sorted);
  aten::CSRSort_(&csr);
  ASSERT_TRUE(aten::CSRIsSorted(csr));
  ASSERT_TRUE(csr.sorted);
  csr = CSR2<IDX>(ctx);
  ASSERT_TRUE(aten::CSRIsSorted(csr));
}

TEST(SpmatTest, CSRSort) {
  _TestCSRSort<int32_t>(CPU);
  _TestCSRSort<int64_t>(CPU);
#ifdef DGL_USE_CUDA
  _TestCSRSort<int32_t>(GPU);
  _TestCSRSort<int64_t>(GPU);
#endif
}

template <typename IDX>
void _TestCSRReorder() {
  auto csr = CSR2<IDX>();
  auto new_row =
      aten::VecToIdArray(std::vector<IDX>({2, 0, 3, 1}), sizeof(IDX) * 8, CTX);
  auto new_col = aten::VecToIdArray(
      std::vector<IDX>({2, 0, 4, 3, 1}), sizeof(IDX) * 8, CTX);
  auto new_csr = CSRReorder(csr, new_row, new_col);
  ASSERT_EQ(new_csr.num_rows, csr.num_rows);
  ASSERT_EQ(new_csr.num_cols, csr.num_cols);
}

TEST(SpmatTest, TestCSRReorder) {
  _TestCSRReorder<int32_t>();
  _TestCSRReorder<int64_t>();
}
