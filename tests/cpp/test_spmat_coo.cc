#include <dgl/array.h>
#include <dmlc/omp.h>
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
          std::vector<IDX>({1, 2, 0, 2, 3}), sizeof(IDX) * 8, ctx),
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
          std::vector<IDX>({1, 2, 2, 0, 3}), sizeof(IDX) * 8, ctx));
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
          std::vector<IDX>({1, 2, 2, 0, 3, 2}), sizeof(IDX) * 8, ctx));
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

struct SparseCOOCSR {
  static constexpr uint64_t NUM_ROWS = 100;
  static constexpr uint64_t NUM_COLS = 150;
  static constexpr uint64_t NUM_NZ = 5;
  template <typename IDX>
  static aten::COOMatrix COOSparse(const DGLContext &ctx = CTX) {
    return aten::COOMatrix(
        NUM_ROWS, NUM_COLS,
        aten::VecToIdArray(
            std::vector<IDX>({0, 1, 2, 3, 4}), sizeof(IDX) * 8, ctx),
        aten::VecToIdArray(
            std::vector<IDX>({1, 2, 3, 4, 5}), sizeof(IDX) * 8, ctx));
  }

  template <typename IDX>
  static aten::CSRMatrix CSRSparse(const DGLContext &ctx = CTX) {
    auto &&indptr = std::vector<IDX>(NUM_ROWS + 1, NUM_NZ);
    for (size_t i = 0; i < NUM_NZ; ++i) {
      indptr[i + 1] = static_cast<IDX>(i + 1);
    }
    indptr[0] = 0;
    return aten::CSRMatrix(
        NUM_ROWS, NUM_COLS, aten::VecToIdArray(indptr, sizeof(IDX) * 8, ctx),
        aten::VecToIdArray(
            std::vector<IDX>({1, 2, 3, 4, 5}), sizeof(IDX) * 8, ctx),
        aten::VecToIdArray(
            std::vector<IDX>({1, 1, 1, 1, 1}), sizeof(IDX) * 8, ctx),
        false);
  }
};

bool isSparseCOO(
    const int64_t &num_threads, const int64_t &num_nodes,
    const int64_t &num_edges) {
  // refer to COOToCSR<>() in ~dgl/src/array/cpu/spmat_op_impl_coo for details.
  return num_threads * num_nodes > 4 * num_edges;
}

template <typename IDX>
aten::COOMatrix RowSorted_NullData_COO(DGLContext ctx = CTX) {
  // [[0, 1, 1, 0, 0],
  //  [1, 0, 0, 0, 0],
  //  [0, 0, 1, 1, 0],
  //  [0, 0, 0, 0, 0]]
  // row : [0, 0, 1, 2, 2]
  // col : [1, 2, 0, 2, 3]
  return aten::COOMatrix(
      4, 5,
      aten::VecToIdArray(
          std::vector<IDX>({0, 0, 1, 2, 2}), sizeof(IDX) * 8, ctx),
      aten::VecToIdArray(
          std::vector<IDX>({1, 2, 0, 2, 3}), sizeof(IDX) * 8, ctx),
      aten::NullArray(), true, false);
}

template <typename IDX>
aten::CSRMatrix RowSorted_NullData_CSR(DGLContext ctx = CTX) {
  // [[0, 1, 1, 0, 0],
  //  [1, 0, 0, 0, 0],
  //  [0, 0, 1, 1, 0],
  //  [0, 0, 0, 0, 0]]
  // data: [0, 1, 2, 3, 4]
  return aten::CSRMatrix(
      4, 5,
      aten::VecToIdArray(
          std::vector<IDX>({0, 2, 3, 5, 5}), sizeof(IDX) * 8, ctx),
      aten::VecToIdArray(
          std::vector<IDX>({1, 2, 0, 2, 3}), sizeof(IDX) * 8, ctx),
      aten::VecToIdArray(
          std::vector<IDX>({0, 1, 2, 3, 4}), sizeof(IDX) * 8, ctx),
      false);
}
}  // namespace

template <typename IDX>
void _TestCOOToCSR(DGLContext ctx) {
  auto coo = COO1<IDX>(ctx);
  auto csr = CSR1<IDX>(ctx);
  auto tcsr = aten::COOToCSR(coo);
  ASSERT_FALSE(coo.row_sorted);
  ASSERT_FALSE(
      isSparseCOO(omp_get_num_threads(), coo.num_rows, coo.row->shape[0]));
  ASSERT_EQ(csr.num_rows, tcsr.num_rows);
  ASSERT_EQ(csr.num_cols, tcsr.num_cols);
  ASSERT_TRUE(ArrayEQ<IDX>(csr.indptr, tcsr.indptr));
  ASSERT_TRUE(ArrayEQ<IDX>(csr.indices, tcsr.indices));

  coo = COO2<IDX>(ctx);
  csr = CSR2<IDX>(ctx);
  tcsr = aten::COOToCSR(coo);
  ASSERT_EQ(coo.num_rows, csr.num_rows);
  ASSERT_EQ(coo.num_cols, csr.num_cols);
  ASSERT_TRUE(ArrayEQ<IDX>(csr.indptr, tcsr.indptr));

  // Convert from row sorted coo
  coo = COO1<IDX>(ctx);
  auto rs_coo = aten::COOSort(coo, false);
  auto rs_csr = CSR1<IDX>(ctx);
  auto rs_tcsr = aten::COOToCSR(rs_coo);
  ASSERT_TRUE(rs_coo.row_sorted);
  ASSERT_EQ(coo.num_rows, rs_tcsr.num_rows);
  ASSERT_EQ(coo.num_cols, rs_tcsr.num_cols);
  ASSERT_TRUE(ArrayEQ<IDX>(rs_csr.indptr, rs_tcsr.indptr));
  ASSERT_TRUE(ArrayEQ<IDX>(rs_tcsr.indices, rs_coo.col));
  ASSERT_TRUE(ArrayEQ<IDX>(rs_tcsr.data, rs_coo.data));

  coo = COO3<IDX>(ctx);
  rs_coo = aten::COOSort(coo, false);
  rs_csr = SR_CSR3<IDX>(ctx);
  rs_tcsr = aten::COOToCSR(rs_coo);
  ASSERT_EQ(coo.num_rows, rs_tcsr.num_rows);
  ASSERT_EQ(coo.num_cols, rs_tcsr.num_cols);
  ASSERT_TRUE(ArrayEQ<IDX>(rs_csr.indptr, rs_tcsr.indptr));
  ASSERT_TRUE(ArrayEQ<IDX>(rs_tcsr.indices, rs_coo.col));
  ASSERT_TRUE(ArrayEQ<IDX>(rs_tcsr.data, rs_coo.data));

  rs_coo = RowSorted_NullData_COO<IDX>(ctx);
  ASSERT_TRUE(rs_coo.row_sorted);
  rs_csr = RowSorted_NullData_CSR<IDX>(ctx);
  rs_tcsr = aten::COOToCSR(rs_coo);
  ASSERT_EQ(coo.num_rows, rs_tcsr.num_rows);
  ASSERT_EQ(rs_csr.num_rows, rs_tcsr.num_rows);
  ASSERT_EQ(coo.num_cols, rs_tcsr.num_cols);
  ASSERT_EQ(rs_csr.num_cols, rs_tcsr.num_cols);
  ASSERT_TRUE(ArrayEQ<IDX>(rs_csr.indptr, rs_tcsr.indptr));
  ASSERT_TRUE(ArrayEQ<IDX>(rs_csr.indices, rs_tcsr.indices));
  ASSERT_TRUE(ArrayEQ<IDX>(rs_csr.data, rs_tcsr.data));
  ASSERT_TRUE(ArrayEQ<IDX>(rs_coo.col, rs_tcsr.indices));
  ASSERT_FALSE(ArrayEQ<IDX>(rs_coo.data, rs_tcsr.data));

  // Convert from col sorted coo
  coo = COO1<IDX>(ctx);
  auto src_coo = aten::COOSort(coo, true);
  auto src_csr = CSR1<IDX>(ctx);
  auto src_tcsr = aten::COOToCSR(src_coo);
  ASSERT_EQ(coo.num_rows, src_tcsr.num_rows);
  ASSERT_EQ(coo.num_cols, src_tcsr.num_cols);
  ASSERT_TRUE(src_tcsr.sorted);
  ASSERT_TRUE(ArrayEQ<IDX>(src_tcsr.indptr, src_csr.indptr));
  ASSERT_TRUE(ArrayEQ<IDX>(src_tcsr.indices, src_coo.col));
  ASSERT_TRUE(ArrayEQ<IDX>(src_tcsr.data, src_coo.data));

  coo = COO3<IDX>(ctx);
  src_coo = aten::COOSort(coo, true);
  src_csr = SRC_CSR3<IDX>(ctx);
  src_tcsr = aten::COOToCSR(src_coo);
  ASSERT_EQ(coo.num_rows, src_tcsr.num_rows);
  ASSERT_EQ(coo.num_cols, src_tcsr.num_cols);
  ASSERT_TRUE(src_tcsr.sorted);
  ASSERT_TRUE(ArrayEQ<IDX>(src_tcsr.indptr, src_csr.indptr));
  ASSERT_TRUE(ArrayEQ<IDX>(src_tcsr.indices, src_coo.col));
  ASSERT_TRUE(ArrayEQ<IDX>(src_tcsr.data, src_coo.data));

  coo = SparseCOOCSR::COOSparse<IDX>(ctx);
  csr = SparseCOOCSR::CSRSparse<IDX>(ctx);
  tcsr = aten::COOToCSR(coo);
  ASSERT_FALSE(coo.row_sorted);
  ASSERT_TRUE(
      isSparseCOO(omp_get_num_threads(), coo.num_rows, coo.row->shape[0]));
  ASSERT_EQ(csr.num_rows, tcsr.num_rows);
  ASSERT_EQ(csr.num_cols, tcsr.num_cols);
  ASSERT_TRUE(ArrayEQ<IDX>(csr.indptr, tcsr.indptr));
  ASSERT_TRUE(ArrayEQ<IDX>(csr.indices, tcsr.indices));
}

TEST(SpmatTest, COOToCSR) {
  _TestCOOToCSR<int32_t>(CPU);
  _TestCOOToCSR<int64_t>(CPU);
#ifdef DGL_USE_CUDA
  _TestCOOToCSR<int32_t>(GPU);
  _TestCOOToCSR<int64_t>(GPU);
#endif
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
void _TestCOOSort(DGLContext ctx) {
  auto coo = COO3<IDX>(ctx);

  auto sr_coo = COOSort(coo, false);
  ASSERT_EQ(coo.num_rows, sr_coo.num_rows);
  ASSERT_EQ(coo.num_cols, sr_coo.num_cols);
  ASSERT_TRUE(sr_coo.row_sorted);
  auto flags = COOIsSorted(sr_coo);
  ASSERT_TRUE(flags.first);
  flags = COOIsSorted(coo);  // original coo should stay the same
  ASSERT_FALSE(flags.first);
  ASSERT_FALSE(flags.second);

  auto src_coo = COOSort(coo, true);
  ASSERT_EQ(coo.num_rows, src_coo.num_rows);
  ASSERT_EQ(coo.num_cols, src_coo.num_cols);
  ASSERT_TRUE(src_coo.row_sorted);
  ASSERT_TRUE(src_coo.col_sorted);
  flags = COOIsSorted(src_coo);
  ASSERT_TRUE(flags.first);
  ASSERT_TRUE(flags.second);

  // sort inplace
  COOSort_(&coo);
  ASSERT_TRUE(coo.row_sorted);
  flags = COOIsSorted(coo);
  ASSERT_TRUE(flags.first);
  COOSort_(&coo, true);
  ASSERT_TRUE(coo.row_sorted);
  ASSERT_TRUE(coo.col_sorted);
  flags = COOIsSorted(coo);
  ASSERT_TRUE(flags.first);
  ASSERT_TRUE(flags.second);

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
      std::vector<IDX>({0, 0, 0, 1, 2, 2}), sizeof(IDX) * 8, ctx);
  auto sort_col = aten::VecToIdArray(
      std::vector<IDX>({1, 2, 2, 0, 2, 3}), sizeof(IDX) * 8, ctx);
  auto sort_col_data = aten::VecToIdArray(
      std::vector<IDX>({2, 0, 5, 3, 1, 4}), sizeof(IDX) * 8, ctx);

  ASSERT_TRUE(ArrayEQ<IDX>(sr_coo.row, sort_row));
  ASSERT_TRUE(ArrayEQ<IDX>(src_coo.row, sort_row));
  ASSERT_TRUE(ArrayEQ<IDX>(src_coo.col, sort_col));
  ASSERT_TRUE(ArrayEQ<IDX>(src_coo.data, sort_col_data));
}

TEST(SpmatTest, COOSort) {
  _TestCOOSort<int32_t>(CPU);
  _TestCOOSort<int64_t>(CPU);
#ifdef DGL_USE_CUDA
  _TestCOOSort<int32_t>(GPU);
  _TestCOOSort<int64_t>(GPU);
#endif
}

template <typename IDX>
void _TestCOOReorder() {
  auto coo = COO2<IDX>();
  auto new_row =
      aten::VecToIdArray(std::vector<IDX>({2, 0, 3, 1}), sizeof(IDX) * 8, CTX);
  auto new_col = aten::VecToIdArray(
      std::vector<IDX>({2, 0, 4, 3, 1}), sizeof(IDX) * 8, CTX);
  auto new_coo = COOReorder(coo, new_row, new_col);
  ASSERT_EQ(new_coo.num_rows, coo.num_rows);
  ASSERT_EQ(new_coo.num_cols, coo.num_cols);
}

TEST(SpmatTest, TestCOOReorder) {
  _TestCOOReorder<int32_t>();
  _TestCOOReorder<int64_t>();
}

template <typename IDX>
void _TestCOOGetData(DGLContext ctx) {
  auto coo = COO2<IDX>(ctx);
  // test get all data
  auto x = aten::COOGetAllData(coo, 0, 0);
  auto tx = aten::VecToIdArray(std::vector<IDX>({}), sizeof(IDX) * 8, ctx);
  ASSERT_TRUE(ArrayEQ<IDX>(x, tx));
  x = aten::COOGetAllData(coo, 0, 2);
  tx = aten::VecToIdArray(std::vector<IDX>({2, 5}), sizeof(IDX) * 8, ctx);
  ASSERT_TRUE(ArrayEQ<IDX>(x, tx));

  // test get data
  auto r =
      aten::VecToIdArray(std::vector<IDX>({0, 0, 0}), sizeof(IDX) * 8, ctx);
  auto c =
      aten::VecToIdArray(std::vector<IDX>({0, 1, 2}), sizeof(IDX) * 8, ctx);
  x = aten::COOGetData(coo, r, c);
  tx = aten::VecToIdArray(std::vector<IDX>({-1, 0, 2}), sizeof(IDX) * 8, ctx);
  ASSERT_TRUE(ArrayEQ<IDX>(x, tx));

  // test get data on sorted
  coo = aten::COOSort(coo);
  r = aten::VecToIdArray(std::vector<IDX>({0, 0, 0}), sizeof(IDX) * 8, ctx);
  c = aten::VecToIdArray(std::vector<IDX>({0, 1, 2}), sizeof(IDX) * 8, ctx);
  x = aten::COOGetData(coo, r, c);
  tx = aten::VecToIdArray(std::vector<IDX>({-1, 0, 2}), sizeof(IDX) * 8, ctx);
  ASSERT_TRUE(ArrayEQ<IDX>(x, tx));

  // test get data w/ broadcasting
  r = aten::VecToIdArray(std::vector<IDX>({0}), sizeof(IDX) * 8, ctx);
  c = aten::VecToIdArray(std::vector<IDX>({0, 1, 2}), sizeof(IDX) * 8, ctx);
  x = aten::COOGetData(coo, r, c);
  tx = aten::VecToIdArray(std::vector<IDX>({-1, 0, 2}), sizeof(IDX) * 8, ctx);
  ASSERT_TRUE(ArrayEQ<IDX>(x, tx));
}

TEST(SpmatTest, COOGetData) {
  _TestCOOGetData<int32_t>(CPU);
  _TestCOOGetData<int64_t>(CPU);
  //#ifdef DGL_USE_CUDA
  //_TestCOOGetData<int32_t>(GPU);
  //_TestCOOGetData<int64_t>(GPU);
  //#endif
}

template <typename IDX>
void _TestCOOGetDataAndIndices() {
  auto csr = COO2<IDX>();
  auto r =
      aten::VecToIdArray(std::vector<IDX>({0, 0, 0}), sizeof(IDX) * 8, CTX);
  auto c =
      aten::VecToIdArray(std::vector<IDX>({0, 1, 2}), sizeof(IDX) * 8, CTX);
  auto x = aten::COOGetDataAndIndices(csr, r, c);
  auto tr =
      aten::VecToIdArray(std::vector<IDX>({0, 0, 0}), sizeof(IDX) * 8, CTX);
  auto tc =
      aten::VecToIdArray(std::vector<IDX>({1, 2, 2}), sizeof(IDX) * 8, CTX);
  auto td =
      aten::VecToIdArray(std::vector<IDX>({0, 2, 5}), sizeof(IDX) * 8, CTX);
  ASSERT_TRUE(ArrayEQ<IDX>(x[0], tr));
  ASSERT_TRUE(ArrayEQ<IDX>(x[1], tc));
  ASSERT_TRUE(ArrayEQ<IDX>(x[2], td));
}

TEST(SpmatTest, COOGetDataAndIndices) {
  _TestCOOGetDataAndIndices<int32_t>();
  _TestCOOGetDataAndIndices<int64_t>();
}
