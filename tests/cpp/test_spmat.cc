#include <gtest/gtest.h>
#include <dgl/array.h>
#include "./common.h"

using namespace dgl;
using namespace dgl::runtime;

namespace {

template <typename IDX>
aten::CSRMatrix CSR1() {
  // [[0, 1, 1, 0, 0],
  //  [1, 0, 0, 0, 0],
  //  [0, 0, 1, 1, 0],
  //  [0, 0, 0, 0, 0]]
  // data: [0, 2, 3, 1, 4]
  return aten::CSRMatrix(
      4, 5,
      aten::VecToIdArray(std::vector<IDX>({0, 2, 3, 5, 5}), sizeof(IDX)*8, CTX),
      aten::VecToIdArray(std::vector<IDX>({1, 2, 0, 2, 3}), sizeof(IDX)*8, CTX),
      aten::VecToIdArray(std::vector<IDX>({0, 2, 3, 1, 4}), sizeof(IDX)*8, CTX),
      false);
}

template <typename IDX>
aten::CSRMatrix CSR2() {
  // has duplicate entries
  // [[0, 1, 2, 0, 0],
  //  [1, 0, 0, 0, 0],
  //  [0, 0, 1, 1, 0],
  //  [0, 0, 0, 0, 0]]
  // data: [0, 2, 5, 3, 1, 4]
  return aten::CSRMatrix(
      4, 5,
      aten::VecToIdArray(std::vector<IDX>({0, 3, 4, 6, 6}), sizeof(IDX)*8, CTX),
      aten::VecToIdArray(std::vector<IDX>({1, 2, 2, 0, 2, 3}), sizeof(IDX)*8, CTX),
      aten::VecToIdArray(std::vector<IDX>({0, 2, 5, 3, 1, 4}), sizeof(IDX)*8, CTX),
      false);
}

template <typename IDX>
aten::COOMatrix COO1() {
  // [[0, 1, 1, 0, 0],
  //  [1, 0, 0, 0, 0],
  //  [0, 0, 1, 1, 0],
  //  [0, 0, 0, 0, 0]]
  // data: [0, 2, 3, 1, 4]
  // row : [0, 2, 0, 1, 2]
  // col : [1, 2, 2, 0, 3]
  return aten::COOMatrix(
      4, 5,
      aten::VecToIdArray(std::vector<IDX>({0, 2, 0, 1, 2}), sizeof(IDX)*8, CTX),
      aten::VecToIdArray(std::vector<IDX>({1, 2, 2, 0, 3}), sizeof(IDX)*8, CTX));
}

template <typename IDX>
aten::COOMatrix COO2() {
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
      aten::VecToIdArray(std::vector<IDX>({0, 2, 0, 1, 2, 0}), sizeof(IDX)*8, CTX),
      aten::VecToIdArray(std::vector<IDX>({1, 2, 2, 0, 3, 2}), sizeof(IDX)*8, CTX));
}

template <typename IDX>
aten::CSRMatrix SR_CSR3() {
  // [[0, 1, 2, 0, 0],
  //  [1, 0, 0, 0, 0],
  //  [0, 0, 1, 1, 0],
  //  [0, 0, 0, 0, 0]]
  return aten::CSRMatrix(
      4, 5,
      aten::VecToIdArray(std::vector<IDX>({0, 3, 4, 6, 6}), sizeof(IDX)*8, CTX),
      aten::VecToIdArray(std::vector<IDX>({2, 1, 2, 0, 2, 3}), sizeof(IDX)*8, CTX),
      aten::VecToIdArray(std::vector<IDX>({0, 2, 5, 3, 1, 4}), sizeof(IDX)*8, CTX),
      false);
}

template <typename IDX>
aten::CSRMatrix SRC_CSR3() {
  // [[0, 1, 2, 0, 0],
  //  [1, 0, 0, 0, 0],
  //  [0, 0, 1, 1, 0],
  //  [0, 0, 0, 0, 0]]
  return aten::CSRMatrix(
      4, 5,
      aten::VecToIdArray(std::vector<IDX>({0, 3, 4, 6, 6}), sizeof(IDX)*8, CTX),
      aten::VecToIdArray(std::vector<IDX>({1, 2, 2, 0, 2, 3}), sizeof(IDX)*8, CTX),
      aten::VecToIdArray(std::vector<IDX>({2, 0, 5, 3, 1, 4}), sizeof(IDX)*8, CTX),
      false);
}

template <typename IDX>
aten::COOMatrix COO3() {
  // has duplicate entries
  // [[0, 1, 2, 0, 0],
  //  [1, 0, 0, 0, 0],
  //  [0, 0, 1, 1, 0],
  //  [0, 0, 0, 0, 0]]
  // row : [0, 2, 0, 1, 2, 0]
  // col : [2, 2, 1, 0, 3, 2]
  return aten::COOMatrix(
      4, 5,
      aten::VecToIdArray(std::vector<IDX>({0, 2, 0, 1, 2, 0}), sizeof(IDX)*8, CTX),
      aten::VecToIdArray(std::vector<IDX>({2, 2, 1, 0, 3, 2}), sizeof(IDX)*8, CTX));
}

}

template <typename IDX>
void _TestCSRIsNonZero() {
  auto csr = CSR1<IDX>();
  ASSERT_TRUE(aten::CSRIsNonZero(csr, 0, 1));
  ASSERT_FALSE(aten::CSRIsNonZero(csr, 0, 0));
  IdArray r = aten::VecToIdArray(std::vector<IDX>({2, 2, 0, 0}), sizeof(IDX)*8, CTX);
  IdArray c = aten::VecToIdArray(std::vector<IDX>({1, 1, 1, 3}), sizeof(IDX)*8, CTX);
  IdArray x = aten::CSRIsNonZero(csr, r, c);
  IdArray tx = aten::VecToIdArray(std::vector<IDX>({0, 0, 1, 0}), sizeof(IDX)*8, CTX);
  ASSERT_TRUE(ArrayEQ<IDX>(x, tx));
}

TEST(SpmatTest, TestCSRIsNonZero) {
  _TestCSRIsNonZero<int32_t>();
  _TestCSRIsNonZero<int64_t>();
}

template <typename IDX>
void _TestCSRGetRowNNZ() {
  auto csr = CSR2<IDX>();
  ASSERT_EQ(aten::CSRGetRowNNZ(csr, 0), 3);
  ASSERT_EQ(aten::CSRGetRowNNZ(csr, 3), 0);
  IdArray r = aten::VecToIdArray(std::vector<IDX>({0, 3}), sizeof(IDX)*8, CTX);
  IdArray x = aten::CSRGetRowNNZ(csr, r);
  IdArray tx = aten::VecToIdArray(std::vector<IDX>({3, 0}), sizeof(IDX)*8, CTX);
  ASSERT_TRUE(ArrayEQ<IDX>(x, tx));
}

TEST(SpmatTest, TestCSRGetRowNNZ) {
  _TestCSRGetRowNNZ<int32_t>();
  _TestCSRGetRowNNZ<int64_t>();
}

template <typename IDX>
void _TestCSRGetRowColumnIndices() {
  auto csr = CSR2<IDX>();
  auto x = aten::CSRGetRowColumnIndices(csr, 0);
  auto tx = aten::VecToIdArray(std::vector<IDX>({1, 2, 2}), sizeof(IDX)*8, CTX);
  ASSERT_TRUE(ArrayEQ<IDX>(x, tx));
  x = aten::CSRGetRowColumnIndices(csr, 1);
  tx = aten::VecToIdArray(std::vector<IDX>({0}), sizeof(IDX)*8, CTX);
  ASSERT_TRUE(ArrayEQ<IDX>(x, tx));
  x = aten::CSRGetRowColumnIndices(csr, 3);
  tx = aten::VecToIdArray(std::vector<IDX>({}), sizeof(IDX)*8, CTX);
  ASSERT_TRUE(ArrayEQ<IDX>(x, tx));
}

TEST(SpmatTest, TestCSRGetRowColumnIndices) {
  _TestCSRGetRowColumnIndices<int32_t>();
  _TestCSRGetRowColumnIndices<int64_t>();
}

template <typename IDX>
void _TestCSRGetRowData() {
  auto csr = CSR2<IDX>();
  auto x = aten::CSRGetRowData(csr, 0);
  auto tx = aten::VecToIdArray(std::vector<IDX>({0, 2, 5}), sizeof(IDX)*8, CTX);
  ASSERT_TRUE(ArrayEQ<IDX>(x, tx));
  x = aten::CSRGetRowData(csr, 1);
  tx = aten::VecToIdArray(std::vector<IDX>({3}), sizeof(IDX)*8, CTX);
  ASSERT_TRUE(ArrayEQ<IDX>(x, tx));
  x = aten::CSRGetRowData(csr, 3);
  tx = aten::VecToIdArray(std::vector<IDX>({}), sizeof(IDX)*8, CTX);
  ASSERT_TRUE(ArrayEQ<IDX>(x, tx));
}

TEST(SpmatTest, TestCSRGetRowData) {
  _TestCSRGetRowData<int32_t>();
  _TestCSRGetRowData<int64_t>();
}

template <typename IDX>
void _TestCSRGetData() {
  auto csr = CSR2<IDX>();
  auto x = aten::CSRGetData(csr, 0, 0);
  auto tx = aten::VecToIdArray(std::vector<IDX>({}), sizeof(IDX)*8, CTX);
  ASSERT_TRUE(ArrayEQ<IDX>(x, tx));
  x = aten::CSRGetData(csr, 0, 2);
  tx = aten::VecToIdArray(std::vector<IDX>({2, 5}), sizeof(IDX)*8, CTX);
  ASSERT_TRUE(ArrayEQ<IDX>(x, tx));

  auto r = aten::VecToIdArray(std::vector<IDX>({0, 0, 0}), sizeof(IDX)*8, CTX);
  auto c = aten::VecToIdArray(std::vector<IDX>({0, 1, 2}), sizeof(IDX)*8, CTX);
  x = aten::CSRGetData(csr, r, c);
  tx = aten::VecToIdArray(std::vector<IDX>({0, 2, 5}), sizeof(IDX)*8, CTX);
  ASSERT_TRUE(ArrayEQ<IDX>(x, tx));
}

TEST(SpmatTest, TestCSRGetData) {
  _TestCSRGetData<int32_t>();
  _TestCSRGetData<int64_t>();
}

template <typename IDX>
void _TestCSRGetDataAndIndices() {
  auto csr = CSR2<IDX>();
  auto r = aten::VecToIdArray(std::vector<IDX>({0, 0, 0}), sizeof(IDX)*8, CTX);
  auto c = aten::VecToIdArray(std::vector<IDX>({0, 1, 2}), sizeof(IDX)*8, CTX);
  auto x = aten::CSRGetDataAndIndices(csr, r, c);
  auto tr = aten::VecToIdArray(std::vector<IDX>({0, 0, 0}), sizeof(IDX)*8, CTX);
  auto tc = aten::VecToIdArray(std::vector<IDX>({1, 2, 2}), sizeof(IDX)*8, CTX);
  auto td = aten::VecToIdArray(std::vector<IDX>({0, 2, 5}), sizeof(IDX)*8, CTX);
  ASSERT_TRUE(ArrayEQ<IDX>(x[0], tr));
  ASSERT_TRUE(ArrayEQ<IDX>(x[1], tc));
  ASSERT_TRUE(ArrayEQ<IDX>(x[2], td));
}

TEST(SpmatTest, TestCSRGetDataAndIndices) {
  _TestCSRGetDataAndIndices<int32_t>();
  _TestCSRGetDataAndIndices<int64_t>();
}

template <typename IDX>
void _TestCSRTranspose() {
  auto csr = CSR2<IDX>();
  auto csr_t = aten::CSRTranspose(csr);
  // [[0, 1, 0, 0],
  //  [1, 0, 0, 0],
  //  [2, 0, 1, 0],
  //  [0, 0, 1, 0],
  //  [0, 0, 0, 0]]
  // data: [3, 0, 2, 5, 1, 4]
  ASSERT_EQ(csr_t.num_rows, 5);
  ASSERT_EQ(csr_t.num_cols, 4);
  auto tp = aten::VecToIdArray(std::vector<IDX>({0, 1, 2, 5, 6, 6}), sizeof(IDX)*8, CTX);
  auto ti = aten::VecToIdArray(std::vector<IDX>({1, 0, 0, 0, 2, 2}), sizeof(IDX)*8, CTX);
  auto td = aten::VecToIdArray(std::vector<IDX>({3, 0, 2, 5, 1, 4}), sizeof(IDX)*8, CTX);
  ASSERT_TRUE(ArrayEQ<IDX>(csr_t.indptr, tp));
  ASSERT_TRUE(ArrayEQ<IDX>(csr_t.indices, ti));
  ASSERT_TRUE(ArrayEQ<IDX>(csr_t.data, td));
}

TEST(SpmatTest, TestCSRTranspose) {
  _TestCSRTranspose<int32_t>();
  _TestCSRTranspose<int64_t>();
}

template <typename IDX>
void _TestCSRToCOO() {
  auto csr = CSR2<IDX>();
  {
  auto coo = CSRToCOO(csr, false);
  ASSERT_EQ(coo.num_rows, 4);
  ASSERT_EQ(coo.num_cols, 5);
  auto tr = aten::VecToIdArray(std::vector<IDX>({0, 0, 0, 1, 2, 2}), sizeof(IDX)*8, CTX);
  auto tc = aten::VecToIdArray(std::vector<IDX>({1, 2, 2, 0, 2, 3}), sizeof(IDX)*8, CTX);
  auto td = aten::VecToIdArray(std::vector<IDX>({0, 2, 5, 3, 1, 4}), sizeof(IDX)*8, CTX);
  ASSERT_TRUE(ArrayEQ<IDX>(coo.row, tr));
  ASSERT_TRUE(ArrayEQ<IDX>(coo.col, tc));
  ASSERT_TRUE(ArrayEQ<IDX>(coo.data, td));
  }
  {
  auto coo = CSRToCOO(csr, true);
  ASSERT_EQ(coo.num_rows, 4);
  ASSERT_EQ(coo.num_cols, 5);
  auto tcoo = COO2<IDX>();
  ASSERT_TRUE(ArrayEQ<IDX>(coo.row, tcoo.row));
  ASSERT_TRUE(ArrayEQ<IDX>(coo.col, tcoo.col));
  }
}

TEST(SpmatTest, TestCSRToCOO) {
  _TestCSRToCOO<int32_t>();
  _TestCSRToCOO<int64_t>();
}

template <typename IDX>
void _TestCSRSliceRows() {
  auto csr = CSR2<IDX>();
  auto x = aten::CSRSliceRows(csr, 1, 4);
  //  [1, 0, 0, 0, 0],
  //  [0, 0, 1, 1, 0],
  //  [0, 0, 0, 0, 0]]
  // data: [3, 1, 4]
  ASSERT_EQ(x.num_rows, 3);
  ASSERT_EQ(x.num_cols, 5);
  auto tp = aten::VecToIdArray(std::vector<IDX>({0, 1, 3, 3}), sizeof(IDX)*8, CTX);
  auto ti = aten::VecToIdArray(std::vector<IDX>({0, 2, 3}), sizeof(IDX)*8, CTX);
  auto td = aten::VecToIdArray(std::vector<IDX>({3, 1, 4}), sizeof(IDX)*8, CTX);
  ASSERT_TRUE(ArrayEQ<IDX>(x.indptr, tp));
  ASSERT_TRUE(ArrayEQ<IDX>(x.indices, ti));
  ASSERT_TRUE(ArrayEQ<IDX>(x.data, td));

  auto r = aten::VecToIdArray(std::vector<IDX>({0, 1, 3}), sizeof(IDX)*8, CTX);
  x = aten::CSRSliceRows(csr, r);
  // [[0, 1, 2, 0, 0],
  //  [1, 0, 0, 0, 0],
  //  [0, 0, 0, 0, 0]]
  // data: [0, 2, 5, 3]
  tp = aten::VecToIdArray(std::vector<IDX>({0, 3, 4, 4}), sizeof(IDX)*8, CTX);
  ti = aten::VecToIdArray(std::vector<IDX>({1, 2, 2, 0}), sizeof(IDX)*8, CTX);
  td = aten::VecToIdArray(std::vector<IDX>({0, 2, 5, 3}), sizeof(IDX)*8, CTX);
  ASSERT_TRUE(ArrayEQ<IDX>(x.indptr, tp));
  ASSERT_TRUE(ArrayEQ<IDX>(x.indices, ti));
  ASSERT_TRUE(ArrayEQ<IDX>(x.data, td));
}

TEST(SpmatTest, TestCSRSliceRows) {
  _TestCSRSliceRows<int32_t>();
  _TestCSRSliceRows<int64_t>();
}

template <typename IDX>
void _TestCSRSliceMatrix() {
  auto csr = CSR2<IDX>();
  auto r = aten::VecToIdArray(std::vector<IDX>({0, 1, 3}), sizeof(IDX)*8, CTX);
  auto c = aten::VecToIdArray(std::vector<IDX>({1, 2, 3}), sizeof(IDX)*8, CTX);
  auto x = aten::CSRSliceMatrix(csr, r, c);
  // [[1, 2, 0],
  //  [0, 0, 0],
  //  [0, 0, 0]]
  // data: [0, 2, 5]
  ASSERT_EQ(x.num_rows, 3);
  ASSERT_EQ(x.num_cols, 3);
  auto tp = aten::VecToIdArray(std::vector<IDX>({0, 3, 3, 3}), sizeof(IDX)*8, CTX);
  auto ti = aten::VecToIdArray(std::vector<IDX>({0, 1, 1}), sizeof(IDX)*8, CTX);
  auto td = aten::VecToIdArray(std::vector<IDX>({0, 2, 5}), sizeof(IDX)*8, CTX);
  ASSERT_TRUE(ArrayEQ<IDX>(x.indptr, tp));
  ASSERT_TRUE(ArrayEQ<IDX>(x.indices, ti));
  ASSERT_TRUE(ArrayEQ<IDX>(x.data, td));
}

TEST(SpmatTest, TestCSRSliceMatrix) {
  _TestCSRSliceMatrix<int32_t>();
  _TestCSRSliceMatrix<int64_t>();
}

template <typename IDX>
void _TestCSRHasDuplicate() {
  auto csr = CSR1<IDX>();
  ASSERT_FALSE(aten::CSRHasDuplicate(csr));
  csr = CSR2<IDX>();
  ASSERT_TRUE(aten::CSRHasDuplicate(csr));
}

TEST(SpmatTest, TestCSRHasDuplicate) {
  _TestCSRHasDuplicate<int32_t>();
  _TestCSRHasDuplicate<int64_t>();
}

template <typename IDX>
void _TestCOOToCSR() {
  auto coo = COO1<IDX>();
  auto csr = CSR1<IDX>();
  auto tcsr = aten::COOToCSR(coo);
  ASSERT_EQ(coo.num_rows, csr.num_rows);
  ASSERT_EQ(coo.num_cols, csr.num_cols);
  ASSERT_TRUE(ArrayEQ<IDX>(csr.indptr, tcsr.indptr));
  ASSERT_TRUE(ArrayEQ<IDX>(csr.indices, tcsr.indices));
  ASSERT_TRUE(ArrayEQ<IDX>(csr.data, tcsr.data));

  coo = COO2<IDX>();
  csr = CSR2<IDX>();
  tcsr = aten::COOToCSR(coo);
  ASSERT_EQ(coo.num_rows, csr.num_rows);
  ASSERT_EQ(coo.num_cols, csr.num_cols);
  ASSERT_TRUE(ArrayEQ<IDX>(csr.indptr, tcsr.indptr));
  ASSERT_TRUE(ArrayEQ<IDX>(csr.indices, tcsr.indices));
  ASSERT_TRUE(ArrayEQ<IDX>(csr.data, tcsr.data));

  coo = COO1<IDX>();
  auto rs_coo = aten::COOSort(coo, false);
  auto rs_csr = CSR1<IDX>();
  auto rs_tcsr = aten::COOToCSR(rs_coo);
  ASSERT_EQ(coo.num_rows, rs_tcsr.num_rows);
  ASSERT_EQ(coo.num_cols, rs_tcsr.num_cols);
  ASSERT_TRUE(ArrayEQ<IDX>(rs_csr.indptr, rs_tcsr.indptr));
  ASSERT_TRUE(ArrayEQ<IDX>(rs_csr.indices, rs_tcsr.indices));
  ASSERT_TRUE(ArrayEQ<IDX>(rs_csr.data, rs_tcsr.data));

  coo = COO3<IDX>();
  rs_coo = aten::COOSort(coo, false);
  rs_csr = SR_CSR3<IDX>();
  rs_tcsr = aten::COOToCSR(rs_coo);
  ASSERT_EQ(coo.num_rows, rs_tcsr.num_rows);
  ASSERT_EQ(coo.num_cols, rs_tcsr.num_cols);
  ASSERT_TRUE(ArrayEQ<IDX>(rs_csr.indptr, rs_tcsr.indptr));
  ASSERT_TRUE(ArrayEQ<IDX>(rs_csr.indices, rs_tcsr.indices));
  ASSERT_TRUE(ArrayEQ<IDX>(rs_csr.data, rs_tcsr.data));

  coo = COO1<IDX>();
  auto src_coo = aten::COOSort(coo, true);
  auto src_csr = CSR1<IDX>();
  auto src_tcsr = aten::COOToCSR(src_coo);
  ASSERT_EQ(coo.num_rows, src_tcsr.num_rows);
  ASSERT_EQ(coo.num_cols, src_tcsr.num_cols);
  ASSERT_TRUE(ArrayEQ<IDX>(src_csr.indptr, src_tcsr.indptr));
  ASSERT_TRUE(ArrayEQ<IDX>(src_csr.indices, src_tcsr.indices));
  ASSERT_TRUE(ArrayEQ<IDX>(src_csr.data, src_tcsr.data));

  coo = COO3<IDX>();
  src_coo = aten::COOSort(coo, true);
  src_csr = SRC_CSR3<IDX>();
  src_tcsr = aten::COOToCSR(src_coo);
  ASSERT_EQ(coo.num_rows, src_tcsr.num_rows);
  ASSERT_EQ(coo.num_cols, src_tcsr.num_cols);
  ASSERT_TRUE(ArrayEQ<IDX>(src_csr.indptr, src_tcsr.indptr));
  ASSERT_TRUE(ArrayEQ<IDX>(src_csr.indices, src_tcsr.indices));
  ASSERT_TRUE(ArrayEQ<IDX>(src_csr.data, src_tcsr.data));
}

TEST(SpmatTest, TestCOOToCSR) {
  _TestCOOToCSR<int32_t>();
  _TestCOOToCSR<int64_t>();
}

template <typename IDX>
void _TestCOOHasDuplicate() {
  auto csr = COO1<IDX>();
  ASSERT_FALSE(aten::COOHasDuplicate(csr));
  csr = COO2<IDX>();
  ASSERT_TRUE(aten::COOHasDuplicate(csr));
}

TEST(SpmatTest, TestCOOHasDuplicate) {
  _TestCOOHasDuplicate<int32_t>();
  _TestCOOHasDuplicate<int64_t>();
}

template <typename IDX>
void _TestCOOSort() {
  auto coo = COO3<IDX>();
  auto sr_coo = COOSort(coo, false);
  ASSERT_EQ(coo.num_rows, sr_coo.num_rows);
  ASSERT_EQ(coo.num_cols, sr_coo.num_cols);
  auto src_coo = COOSort(coo, true);
  ASSERT_EQ(coo.num_rows, src_coo.num_rows);
  ASSERT_EQ(coo.num_cols, src_coo.num_cols);

  // COO3
  // [[0, 1, 2, 0, 0],
  //  [1, 0, 0, 0, 0],
  //  [0, 0, 1, 1, 0],
  //  [0, 0, 0, 0, 0]]
  // data: [0, 1, 2, 3, 4, 5]
  // row : [0, 2, 0, 1, 2, 0]
  // col : [2, 2, 1, 0, 3, 2]
  // Row Sorted
  // data: [0, 2, 5, 3, 1, 4]
  // row : [0, 0, 0, 1, 2, 2]
  // col : [2, 1, 2, 0, 2, 3]
  // Row Col Sorted
  // data: [2, 0, 5, 3, 1, 4]
  // row : [0, 0, 0, 1, 2, 2]
  // col : [1, 2, 2, 0, 2, 3]
  auto sort_row = aten::VecToIdArray(
    std::vector<IDX>({0, 0, 0, 1, 2, 2}), sizeof(IDX)*8, CTX);
  auto unsort_col = aten::VecToIdArray(
    std::vector<IDX>({2, 1, 2, 0, 2, 3}), sizeof(IDX)*8, CTX);
  auto unsort_col_data = aten::VecToIdArray(
    std::vector<IDX>({0, 2, 5, 3, 1, 4}), sizeof(IDX)*8, CTX);
  auto sort_col = aten::VecToIdArray(
    std::vector<IDX>({1, 2, 2, 0, 2, 3}), sizeof(IDX)*8, CTX);
  auto sort_col_data = aten::VecToIdArray(
    std::vector<IDX>({2, 0, 5, 3, 1, 4}), sizeof(IDX)*8, CTX);

  ASSERT_TRUE(ArrayEQ<IDX>(sr_coo.row, sort_row));
  ASSERT_TRUE(ArrayEQ<IDX>(sr_coo.col, unsort_col));
  ASSERT_TRUE(ArrayEQ<IDX>(sr_coo.data, unsort_col_data));
  ASSERT_TRUE(ArrayEQ<IDX>(src_coo.row, sort_row));
  ASSERT_TRUE(ArrayEQ<IDX>(src_coo.col, sort_col));
  ASSERT_TRUE(ArrayEQ<IDX>(src_coo.data, sort_col_data));
}

TEST(SpmatTest, TestCOOSort) {
  _TestCOOSort<int32_t>();
  _TestCOOSort<int64_t>();
}
