#include <dgl/array.h>
#include <gtest/gtest.h>

#include "./common.h"

using namespace dgl;
using namespace dgl::runtime;

TEST(ArrayTest, TestCreate) {
  IdArray a = aten::NewIdArray(100, CTX, 32);
  ASSERT_EQ(a->dtype.bits, 32);
  ASSERT_EQ(a->shape[0], 100);

  a = aten::NewIdArray(0);
  ASSERT_EQ(a->shape[0], 0);

  std::vector<int64_t> vec = {2, 94, 232, 30};
  a = aten::VecToIdArray(vec, 32);
  ASSERT_EQ(Len(a), vec.size());
  ASSERT_EQ(a->dtype.bits, 32);
  for (int i = 0; i < Len(a); ++i) {
    ASSERT_EQ(Ptr<int32_t>(a)[i], vec[i]);
  }

  a = aten::VecToIdArray(std::vector<int32_t>());
  ASSERT_EQ(Len(a), 0);
};

void _TestRange(DGLContext ctx) {
  IdArray a = aten::Range(10, 10, 64, ctx);
  ASSERT_EQ(Len(a), 0);
  a = aten::Range(10, 20, 32, ctx);
  ASSERT_EQ(Len(a), 10);
  ASSERT_EQ(a->dtype.bits, 32);
  a = a.CopyTo(CPU);
  for (int i = 0; i < 10; ++i) ASSERT_EQ(Ptr<int32_t>(a)[i], i + 10);
}

TEST(ArrayTest, TestRange) {
  _TestRange(CPU);
#ifdef DGL_USE_CUDA
  _TestRange(GPU);
#endif
};

TEST(ArrayTest, TestFull) {
  IdArray a = aten::Full(-100, 0, 32, CTX);
  ASSERT_EQ(Len(a), 0);
  a = aten::Full(-100, 13, 64, CTX);
  ASSERT_EQ(Len(a), 13);
  ASSERT_EQ(a->dtype.bits, 64);
  for (int i = 0; i < 13; ++i) ASSERT_EQ(Ptr<int64_t>(a)[i], -100);
};

TEST(ArrayTest, TestClone) {
  IdArray a = aten::NewIdArray(0);
  IdArray b = aten::Clone(a);
  ASSERT_EQ(Len(b), 0);

  a = aten::Range(0, 10, 32, CTX);
  b = aten::Clone(a);
  for (int i = 0; i < 10; ++i) {
    ASSERT_EQ(PI32(b)[i], i);
  }
  PI32(b)[0] = -1;
  for (int i = 0; i < 10; ++i) {
    ASSERT_EQ(PI32(a)[i], i);
  }
};

void _TestNumBits(DGLContext ctx) {
  IdArray a = aten::Range(0, 10, 32, ctx);
  a = aten::AsNumBits(a, 64);
  ASSERT_EQ(a->dtype.bits, 64);
  a = a.CopyTo(CPU);
  for (int i = 0; i < 10; ++i) ASSERT_EQ(PI64(a)[i], i);
}

TEST(ArrayTest, TestAsNumBits) {
  _TestNumBits(CPU);
#ifdef DGL_USE_CUDA
  _TestNumBits(GPU);
#endif
};

template <typename IDX>
void _TestArith(DGLContext ctx) {
  const int N = 100;
  IdArray a = aten::Full(-10, N, sizeof(IDX) * 8, ctx);
  IdArray b = aten::Full(7, N, sizeof(IDX) * 8, ctx);

  IdArray c = a + b;
  c = c.CopyTo(CPU);
  for (int i = 0; i < N; ++i) ASSERT_EQ(Ptr<IDX>(c)[i], -3);
  c = a - b;
  c = c.CopyTo(CPU);
  for (int i = 0; i < N; ++i) ASSERT_EQ(Ptr<IDX>(c)[i], -17);
  c = a * b;
  c = c.CopyTo(CPU);
  for (int i = 0; i < N; ++i) ASSERT_EQ(Ptr<IDX>(c)[i], -70);
  c = a / b;
  c = c.CopyTo(CPU);
  for (int i = 0; i < N; ++i) ASSERT_EQ(Ptr<IDX>(c)[i], -1);
  c = -a;
  c = c.CopyTo(CPU);
  for (int i = 0; i < N; ++i) ASSERT_EQ(Ptr<IDX>(c)[i], 10);
  c = (-a) % b;
  c = c.CopyTo(CPU);
  for (int i = 0; i < N; ++i) ASSERT_EQ(Ptr<IDX>(c)[i], 3);

  const int val = -3;
  c = aten::Add(a, val);
  c = c.CopyTo(CPU);
  for (int i = 0; i < N; ++i) ASSERT_EQ(Ptr<IDX>(c)[i], -13);
  c = aten::Sub(a, val);
  c = c.CopyTo(CPU);
  for (int i = 0; i < N; ++i) ASSERT_EQ(Ptr<IDX>(c)[i], -7);
  c = aten::Mul(a, val);
  c = c.CopyTo(CPU);
  for (int i = 0; i < N; ++i) ASSERT_EQ(Ptr<IDX>(c)[i], 30);
  c = aten::Div(a, val);
  c = c.CopyTo(CPU);
  for (int i = 0; i < N; ++i) ASSERT_EQ(Ptr<IDX>(c)[i], 3);
  c = b % 3;
  c = c.CopyTo(CPU);
  for (int i = 0; i < N; ++i) ASSERT_EQ(Ptr<IDX>(c)[i], 1);

  c = aten::Add(val, b);
  c = c.CopyTo(CPU);
  for (int i = 0; i < N; ++i) ASSERT_EQ(Ptr<IDX>(c)[i], 4);
  c = aten::Sub(val, b);
  c = c.CopyTo(CPU);
  for (int i = 0; i < N; ++i) ASSERT_EQ(Ptr<IDX>(c)[i], -10);
  c = aten::Mul(val, b);
  c = c.CopyTo(CPU);
  for (int i = 0; i < N; ++i) ASSERT_EQ(Ptr<IDX>(c)[i], -21);
  c = aten::Div(val, b);
  c = c.CopyTo(CPU);
  for (int i = 0; i < N; ++i) ASSERT_EQ(Ptr<IDX>(c)[i], 0);
  c = 3 % b;
  c = c.CopyTo(CPU);
  for (int i = 0; i < N; ++i) ASSERT_EQ(Ptr<IDX>(c)[i], 3);

  a = aten::Range(0, N, sizeof(IDX) * 8, ctx);
  c = a < 50;
  c = c.CopyTo(CPU);
  for (int i = 0; i < N; ++i) ASSERT_EQ(Ptr<IDX>(c)[i], (int)(i < 50));

  c = a > 50;
  c = c.CopyTo(CPU);
  for (int i = 0; i < N; ++i) ASSERT_EQ(Ptr<IDX>(c)[i], (int)(i > 50));

  c = a >= 50;
  c = c.CopyTo(CPU);
  for (int i = 0; i < N; ++i) ASSERT_EQ(Ptr<IDX>(c)[i], (int)(i >= 50));

  c = a <= 50;
  c = c.CopyTo(CPU);
  for (int i = 0; i < N; ++i) ASSERT_EQ(Ptr<IDX>(c)[i], (int)(i <= 50));

  c = a == 50;
  c = c.CopyTo(CPU);
  for (int i = 0; i < N; ++i) ASSERT_EQ(Ptr<IDX>(c)[i], (int)(i == 50));

  c = a != 50;
  c = c.CopyTo(CPU);
  for (int i = 0; i < N; ++i) ASSERT_EQ(Ptr<IDX>(c)[i], (int)(i != 50));
}

TEST(ArrayTest, Arith) {
  _TestArith<int32_t>(CPU);
  _TestArith<int64_t>(CPU);
#ifdef DGL_USE_CUDA
  _TestArith<int32_t>(GPU);
  _TestArith<int64_t>(GPU);
#endif
};

template <typename IDX>
void _TestHStack(DGLContext ctx) {
  IdArray a = aten::Range(0, 100, sizeof(IDX) * 8, ctx);
  IdArray b = aten::Range(100, 200, sizeof(IDX) * 8, ctx);
  IdArray c = aten::HStack(a, b).CopyTo(aten::CPU);
  ASSERT_EQ(c->ndim, 1);
  ASSERT_EQ(c->shape[0], 200);
  for (int i = 0; i < 200; ++i) ASSERT_EQ(Ptr<IDX>(c)[i], i);
}

TEST(ArrayTest, HStack) {
  _TestHStack<int32_t>(CPU);
  _TestHStack<int64_t>(CPU);
#ifdef DGL_USE_CUDA
  _TestHStack<int32_t>(GPU);
  _TestHStack<int64_t>(GPU);
#endif
}

template <typename IDX>
void _TestIndexSelect(DGLContext ctx) {
  IdArray a = aten::Range(0, 100, sizeof(IDX) * 8, ctx);
  ASSERT_EQ(aten::IndexSelect<int>(a, 50), 50);
  ASSERT_TRUE(ArrayEQ<IDX>(
      aten::IndexSelect(a, 10, 20), aten::Range(10, 20, sizeof(IDX) * 8, ctx)));
  IdArray b =
      aten::VecToIdArray(std::vector<IDX>({0, 20, 10}), sizeof(IDX) * 8, ctx);
  IdArray c = aten::IndexSelect(a, b);
  ASSERT_TRUE(ArrayEQ<IDX>(b, c));
}

TEST(ArrayTest, TestIndexSelect) {
  _TestIndexSelect<int32_t>(CPU);
  _TestIndexSelect<int64_t>(CPU);
#ifdef DGL_USE_CUDA
  _TestIndexSelect<int32_t>(GPU);
  _TestIndexSelect<int64_t>(GPU);
#endif
}

template <typename IDX>
void _TestRelabel_(DGLContext ctx) {
  IdArray a =
      aten::VecToIdArray(std::vector<IDX>({0, 20, 10}), sizeof(IDX) * 8, ctx);
  IdArray b =
      aten::VecToIdArray(std::vector<IDX>({20, 5, 6}), sizeof(IDX) * 8, ctx);
  IdArray c = aten::Relabel_({a, b});

  IdArray ta =
      aten::VecToIdArray(std::vector<IDX>({0, 1, 2}), sizeof(IDX) * 8, ctx);
  IdArray tb =
      aten::VecToIdArray(std::vector<IDX>({1, 3, 4}), sizeof(IDX) * 8, ctx);
  IdArray tc = aten::VecToIdArray(
      std::vector<IDX>({0, 20, 10, 5, 6}), sizeof(IDX) * 8, ctx);

  ASSERT_TRUE(ArrayEQ<IDX>(a, ta));
  ASSERT_TRUE(ArrayEQ<IDX>(b, tb));
  ASSERT_TRUE(ArrayEQ<IDX>(c, tc));
}

TEST(ArrayTest, TestRelabel_) {
  _TestRelabel_<int32_t>(CPU);
  _TestRelabel_<int64_t>(CPU);
#ifdef DGL_USE_CUDA
  _TestRelabel_<int32_t>(GPU);
  _TestRelabel_<int64_t>(GPU);
#endif
}

template <typename IDX>
void _TestConcat(DGLContext ctx) {
  IdArray a =
      aten::VecToIdArray(std::vector<IDX>({1, 2, 3}), sizeof(IDX) * 8, CTX);
  IdArray b =
      aten::VecToIdArray(std::vector<IDX>({4, 5, 6}), sizeof(IDX) * 8, CTX);
  IdArray tc = aten::VecToIdArray(
      std::vector<IDX>({1, 2, 3, 4, 5, 6}), sizeof(IDX) * 8, CTX);
  IdArray c = aten::Concat(std::vector<IdArray>{a, b});
  ASSERT_TRUE(ArrayEQ<IDX>(c, tc));
  IdArray d = aten::Concat(std::vector<IdArray>{a, b, c});
  IdArray td = aten::VecToIdArray(
      std::vector<IDX>({1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6}), sizeof(IDX) * 8,
      CTX);
  ASSERT_TRUE(ArrayEQ<IDX>(d, td));
}

template <typename IdType>
void _TestToSimpleCsr(DGLContext ctx) {
  /**
   * A = [[0, 0, 0, 0],
   *      [1, 0, 0, 1],
   *      [1, 1, 1, 1],
   *      [3, 2, 2, 3],
   *      [2, 0, 0, 2]]
   *
   * B = CSRToSimple(A)
   * B = [[0, 0, 0, 0],
   *      [1, 0, 0, 1],
   *      [1, 1, 1, 1],
   *      [1, 1, 1, 1],
   *      [1, 0, 0, 1]]
   */
  IdArray a_indptr = aten::VecToIdArray(
      std::vector<IdType>({0, 0, 2, 6, 16, 20}), sizeof(IdType) * 8, CTX);
  IdArray a_indices = aten::VecToIdArray(
      std::vector<IdType>(
          {0, 3, 0, 1, 2, 3, 0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 0, 0, 3, 3}),
      sizeof(IdType) * 8, CTX);
  IdArray b_indptr = aten::VecToIdArray(
      std::vector<IdType>({0, 0, 2, 6, 10, 12}), sizeof(IdType) * 8, CTX);
  IdArray b_indices = aten::VecToIdArray(
      std::vector<IdType>({0, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 3}),
      sizeof(IdType) * 8, CTX);
  IdArray cnt = aten::VecToIdArray(
      std::vector<IdType>({1, 1, 1, 1, 1, 1, 3, 2, 2, 3, 2, 2}),
      sizeof(IdType) * 8, CTX);
  IdArray map = aten::VecToIdArray(
      std::vector<IdType>(
          {0, 1, 2, 3, 4, 5, 6, 6, 6, 7, 7, 8, 8, 9, 9, 9, 10, 10, 11, 11}),
      sizeof(IdType) * 8, CTX);
  const aten::CSRMatrix &csr_a =
      aten::CSRMatrix(5, 4, a_indptr, a_indices, aten::NullArray(), true);
  auto ret = CSRToSimple(csr_a);
  aten::CSRMatrix csr_b = std::get<0>(ret);
  IdArray ecnt = std::get<1>(ret);
  IdArray emap = std::get<2>(ret);
  ASSERT_EQ(csr_b.num_rows, 5);
  ASSERT_EQ(csr_b.num_cols, 4);
  ASSERT_TRUE(ArrayEQ<IdType>(csr_b.indptr, b_indptr));
  ASSERT_TRUE(ArrayEQ<IdType>(csr_b.indices, b_indices));
  ASSERT_TRUE(ArrayEQ<IdType>(ecnt, cnt));
  ASSERT_TRUE(ArrayEQ<IdType>(emap, map));
  ASSERT_TRUE(csr_b.sorted);

  // a not sorted
  a_indices = aten::VecToIdArray(
      std::vector<IdType>(
          {0, 3, 0, 1, 2, 3, 3, 0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 3, 0, 3}),
      sizeof(IdType) * 8, CTX);
  map = aten::VecToIdArray(
      std::vector<IdType>(
          {0, 1, 2, 3, 4, 5, 9, 6, 6, 7, 7, 8, 8, 9, 9, 6, 10, 11, 10, 11}),
      sizeof(IdType) * 8, CTX);
  const aten::CSRMatrix &csr_a2 =
      aten::CSRMatrix(5, 4, a_indptr, a_indices, aten::NullArray(), false);
  ret = CSRToSimple(csr_a2);
  csr_b = std::get<0>(ret);
  ecnt = std::get<1>(ret);
  emap = std::get<2>(ret);
  ASSERT_EQ(csr_b.num_rows, 5);
  ASSERT_EQ(csr_b.num_cols, 4);
  ASSERT_TRUE(ArrayEQ<IdType>(csr_b.indptr, b_indptr));
  ASSERT_TRUE(ArrayEQ<IdType>(csr_b.indices, b_indices));
  ASSERT_TRUE(ArrayEQ<IdType>(ecnt, cnt));
  ASSERT_TRUE(ArrayEQ<IdType>(emap, map));
  ASSERT_TRUE(csr_b.sorted);
}

TEST(MatrixTest, TestToSimpleCsr) {
  _TestToSimpleCsr<int32_t>(CPU);
  _TestToSimpleCsr<int64_t>(CPU);
}

template <typename IdType>
void _TestToSimpleCoo(DGLContext ctx) {
  /**
   * A = [[0, 0, 0, 0],
   *      [1, 0, 0, 1],
   *      [1, 1, 1, 1],
   *      [3, 2, 2, 3],
   *      [2, 0, 0, 2]]
   *
   * B = CSRToSimple(A)
   * B = [[0, 0, 0, 0],
   *      [1, 0, 0, 1],
   *      [1, 1, 1, 1],
   *      [1, 1, 1, 1],
   *      [1, 0, 0, 1]]
   */
  IdArray a_row = aten::VecToIdArray(
      std::vector<IdType>(
          {1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4}),
      sizeof(IdType) * 8, CTX);
  IdArray a_col = aten::VecToIdArray(
      std::vector<IdType>(
          {0, 3, 0, 1, 2, 3, 0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 0, 0, 3, 3}),
      sizeof(IdType) * 8, CTX);
  IdArray b_row = aten::VecToIdArray(
      std::vector<IdType>({1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4}),
      sizeof(IdType) * 8, CTX);
  IdArray b_col = aten::VecToIdArray(
      std::vector<IdType>({0, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 3}),
      sizeof(IdType) * 8, CTX);
  IdArray cnt = aten::VecToIdArray(
      std::vector<IdType>({1, 1, 1, 1, 1, 1, 3, 2, 2, 3, 2, 2}),
      sizeof(IdType) * 8, CTX);
  IdArray map = aten::VecToIdArray(
      std::vector<IdType>(
          {0, 1, 2, 3, 4, 5, 6, 6, 6, 7, 7, 8, 8, 9, 9, 9, 10, 10, 11, 11}),
      sizeof(IdType) * 8, CTX);
  const aten::COOMatrix &coo_a =
      aten::COOMatrix(5, 4, a_row, a_col, aten::NullArray(), true, true);
  auto ret = COOToSimple(coo_a);
  aten::COOMatrix coo_b = std::get<0>(ret);
  IdArray ecnt = std::get<1>(ret);
  IdArray emap = std::get<2>(ret);
  ASSERT_EQ(coo_b.num_rows, 5);
  ASSERT_EQ(coo_b.num_cols, 4);
  ASSERT_TRUE(ArrayEQ<IdType>(coo_b.row, b_row));
  ASSERT_TRUE(ArrayEQ<IdType>(coo_b.col, b_col));
  ASSERT_TRUE(ArrayEQ<IdType>(ecnt, cnt));
  ASSERT_TRUE(ArrayEQ<IdType>(emap, map));
  ASSERT_FALSE(COOHasData(coo_b));
  ASSERT_TRUE(coo_b.row_sorted);
  ASSERT_TRUE(coo_b.col_sorted);

  // a not sorted
  a_row = aten::VecToIdArray(
      std::vector<IdType>(
          {1, 2, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4}),
      sizeof(IdType) * 8, CTX);
  a_col = aten::VecToIdArray(
      std::vector<IdType>(
          {0, 0, 3, 1, 2, 3, 0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 0, 3, 0, 3}),
      sizeof(IdType) * 8, CTX);
  map = aten::VecToIdArray(
      std::vector<IdType>(
          {0, 2, 1, 3, 4, 5, 6, 6, 6, 7, 7, 8, 8, 9, 9, 9, 10, 11, 10, 11}),
      sizeof(IdType) * 8, CTX);
  const aten::COOMatrix &coo_a2 =
      aten::COOMatrix(5, 4, a_row, a_col, aten::NullArray(), false, false);
  ret = COOToSimple(coo_a2);
  coo_b = std::get<0>(ret);
  ecnt = std::get<1>(ret);
  emap = std::get<2>(ret);
  ASSERT_EQ(coo_b.num_rows, 5);
  ASSERT_EQ(coo_b.num_cols, 4);
  ASSERT_TRUE(ArrayEQ<IdType>(coo_b.row, b_row));
  ASSERT_TRUE(ArrayEQ<IdType>(coo_b.col, b_col));
  ASSERT_TRUE(ArrayEQ<IdType>(ecnt, cnt));
  ASSERT_TRUE(ArrayEQ<IdType>(emap, map));
  ASSERT_FALSE(COOHasData(coo_b));
  ASSERT_TRUE(coo_b.row_sorted);
  ASSERT_TRUE(coo_b.col_sorted);
}

TEST(MatrixTest, TestToSimpleCoo) {
  _TestToSimpleCoo<int32_t>(CPU);
  _TestToSimpleCoo<int64_t>(CPU);
}

template <typename IdType>
void _TestDisjointUnionPartitionCoo(DGLContext ctx) {
  /**
   * A = [[0, 0, 1],
   *      [1, 0, 1],
   *      [0, 1, 0]]
   *
   * B = [[1, 1, 0],
   *      [0, 1, 0]]
   *
   * C = [[1]]
   *
   * AB = [[0, 0, 1, 0, 0, 0],
   *       [1, 0, 1, 0, 0, 0],
   *       [0, 1, 0, 0, 0, 0],
   *       [0, 0, 0, 1, 1, 0],
   *       [0, 0, 0, 0, 1, 0]]
   *
   * ABC = [[0, 0, 1, 0, 0, 0, 0],
   *        [1, 0, 1, 0, 0, 0, 0],
   *        [0, 1, 0, 0, 0, 0, 0],
   *        [0, 0, 0, 1, 1, 0, 0],
   *        [0, 0, 0, 0, 1, 0, 0],
   *        [0, 0, 0, 0, 0, 0, 1]]
   */
  IdArray a_row = aten::VecToIdArray(
      std::vector<IdType>({0, 1, 1, 2}), sizeof(IdType) * 8, CTX);
  IdArray a_col = aten::VecToIdArray(
      std::vector<IdType>({2, 0, 2, 1}), sizeof(IdType) * 8, CTX);
  IdArray b_row = aten::VecToIdArray(
      std::vector<IdType>({0, 0, 1}), sizeof(IdType) * 8, CTX);
  IdArray b_col = aten::VecToIdArray(
      std::vector<IdType>({0, 1, 1}), sizeof(IdType) * 8, CTX);
  IdArray b_data = aten::VecToIdArray(
      std::vector<IdType>({2, 0, 1}), sizeof(IdType) * 8, CTX);
  IdArray c_row =
      aten::VecToIdArray(std::vector<IdType>({0}), sizeof(IdType) * 8, CTX);
  IdArray c_col =
      aten::VecToIdArray(std::vector<IdType>({0}), sizeof(IdType) * 8, CTX);
  IdArray ab_row = aten::VecToIdArray(
      std::vector<IdType>({0, 1, 1, 2, 3, 3, 4}), sizeof(IdType) * 8, CTX);
  IdArray ab_col = aten::VecToIdArray(
      std::vector<IdType>({2, 0, 2, 1, 3, 4, 4}), sizeof(IdType) * 8, CTX);
  IdArray ab_data = aten::VecToIdArray(
      std::vector<IdType>({0, 1, 2, 3, 6, 4, 5}), sizeof(IdType) * 8, CTX);
  IdArray abc_row = aten::VecToIdArray(
      std::vector<IdType>({0, 1, 1, 2, 3, 3, 4, 5}), sizeof(IdType) * 8, CTX);
  IdArray abc_col = aten::VecToIdArray(
      std::vector<IdType>({2, 0, 2, 1, 3, 4, 4, 6}), sizeof(IdType) * 8, CTX);
  IdArray abc_data = aten::VecToIdArray(
      std::vector<IdType>({0, 1, 2, 3, 6, 4, 5, 7}), sizeof(IdType) * 8, CTX);
  const aten::COOMatrix &coo_a =
      aten::COOMatrix(3, 3, a_row, a_col, aten::NullArray(), true, false);
  const aten::COOMatrix &coo_b =
      aten::COOMatrix(2, 3, b_row, b_col, b_data, true, true);
  const aten::COOMatrix &coo_c =
      aten::COOMatrix(1, 1, c_row, c_col, aten::NullArray(), true, true);

  const std::vector<aten::COOMatrix> coos_ab({coo_a, coo_b});
  const aten::COOMatrix &coo_ab = aten::DisjointUnionCoo(coos_ab);
  ASSERT_EQ(coo_ab.num_rows, 5);
  ASSERT_EQ(coo_ab.num_cols, 6);
  ASSERT_TRUE(ArrayEQ<IdType>(coo_ab.row, ab_row));
  ASSERT_TRUE(ArrayEQ<IdType>(coo_ab.col, ab_col));
  ASSERT_TRUE(ArrayEQ<IdType>(coo_ab.data, ab_data));
  ASSERT_TRUE(coo_ab.row_sorted);
  ASSERT_FALSE(coo_ab.col_sorted);

  const std::vector<uint64_t> edge_cumsum({0, 4, 7});
  const std::vector<uint64_t> src_vertex_cumsum({0, 3, 5});
  const std::vector<uint64_t> dst_vertex_cumsum({0, 3, 6});
  const std::vector<aten::COOMatrix> &p_coos =
      aten::DisjointPartitionCooBySizes(
          coo_ab, 2, edge_cumsum, src_vertex_cumsum, dst_vertex_cumsum);
  ASSERT_EQ(p_coos[0].num_rows, coo_a.num_rows);
  ASSERT_EQ(p_coos[0].num_cols, coo_a.num_cols);
  ASSERT_EQ(p_coos[1].num_rows, coo_b.num_rows);
  ASSERT_EQ(p_coos[1].num_cols, coo_b.num_cols);
  ASSERT_TRUE(ArrayEQ<IdType>(p_coos[0].row, coo_a.row));
  ASSERT_TRUE(ArrayEQ<IdType>(p_coos[0].col, coo_a.col));
  ASSERT_TRUE(ArrayEQ<IdType>(p_coos[1].row, coo_b.row));
  ASSERT_TRUE(ArrayEQ<IdType>(p_coos[1].col, coo_b.col));
  ASSERT_TRUE(ArrayEQ<IdType>(p_coos[1].data, coo_b.data));
  ASSERT_TRUE(p_coos[0].row_sorted);
  ASSERT_FALSE(p_coos[0].col_sorted);
  ASSERT_TRUE(p_coos[1].row_sorted);
  ASSERT_FALSE(p_coos[1].col_sorted);

  const std::vector<aten::COOMatrix> coos_abc({coo_a, coo_b, coo_c});
  const aten::COOMatrix &coo_abc = aten::DisjointUnionCoo(coos_abc);
  ASSERT_EQ(coo_abc.num_rows, 6);
  ASSERT_EQ(coo_abc.num_cols, 7);
  ASSERT_TRUE(ArrayEQ<IdType>(coo_abc.row, abc_row));
  ASSERT_TRUE(ArrayEQ<IdType>(coo_abc.col, abc_col));
  ASSERT_TRUE(ArrayEQ<IdType>(coo_abc.data, abc_data));
  ASSERT_TRUE(coo_abc.row_sorted);
  ASSERT_FALSE(coo_abc.col_sorted);

  const std::vector<uint64_t> edge_cumsum_abc({0, 4, 7, 8});
  const std::vector<uint64_t> src_vertex_cumsum_abc({0, 3, 5, 6});
  const std::vector<uint64_t> dst_vertex_cumsum_abc({0, 3, 6, 7});
  const std::vector<aten::COOMatrix> &p_coos_abc =
      aten::DisjointPartitionCooBySizes(
          coo_abc, 3, edge_cumsum_abc, src_vertex_cumsum_abc,
          dst_vertex_cumsum_abc);
  ASSERT_EQ(p_coos_abc[0].num_rows, coo_a.num_rows);
  ASSERT_EQ(p_coos_abc[0].num_cols, coo_a.num_cols);
  ASSERT_EQ(p_coos_abc[1].num_rows, coo_b.num_rows);
  ASSERT_EQ(p_coos_abc[1].num_cols, coo_b.num_cols);
  ASSERT_EQ(p_coos_abc[2].num_rows, coo_c.num_rows);
  ASSERT_EQ(p_coos_abc[2].num_cols, coo_c.num_cols);
  ASSERT_TRUE(ArrayEQ<IdType>(p_coos_abc[0].row, coo_a.row));
  ASSERT_TRUE(ArrayEQ<IdType>(p_coos_abc[0].col, coo_a.col));
  ASSERT_TRUE(ArrayEQ<IdType>(p_coos_abc[1].row, coo_b.row));
  ASSERT_TRUE(ArrayEQ<IdType>(p_coos_abc[1].col, coo_b.col));
  ASSERT_TRUE(ArrayEQ<IdType>(p_coos_abc[1].data, coo_b.data));
  ASSERT_TRUE(ArrayEQ<IdType>(p_coos_abc[2].row, coo_c.row));
  ASSERT_TRUE(ArrayEQ<IdType>(p_coos_abc[2].col, coo_c.col));
  ASSERT_TRUE(p_coos_abc[0].row_sorted);
  ASSERT_FALSE(p_coos_abc[0].col_sorted);
  ASSERT_TRUE(p_coos_abc[1].row_sorted);
  ASSERT_FALSE(p_coos_abc[1].col_sorted);
  ASSERT_TRUE(p_coos_abc[2].row_sorted);
  ASSERT_FALSE(p_coos_abc[2].col_sorted);
}

TEST(DisjointUnionTest, TestDisjointUnionPartitionCoo) {
  _TestDisjointUnionPartitionCoo<int32_t>(CPU);
  _TestDisjointUnionPartitionCoo<int64_t>(CPU);
#ifdef DGL_USE_CUDA
  _TestDisjointUnionPartitionCoo<int32_t>(GPU);
  _TestDisjointUnionPartitionCoo<int64_t>(GPU);
#endif
}

template <typename IdType>
void _TestDisjointUnionPartitionCsr(DGLContext ctx) {
  /**
   * A = [[0, 0, 1],
   *      [1, 0, 1],
   *      [0, 1, 0]]
   *
   * B = [[1, 1, 0],
   *      [0, 1, 0]]
   *
   * C = [[1]]
   *
   * BC = [[1, 1, 0, 0],
   *       [0, 1, 0, 0],
   *       [0, 0, 0, 1]],
   *
   * ABC = [[0, 0, 1, 0, 0, 0, 0],
   *        [1, 0, 1, 0, 0, 0, 0],
   *        [0, 1, 0, 0, 0, 0, 0],
   *        [0, 0, 0, 1, 1, 0, 0],
   *        [0, 0, 0, 0, 1, 0, 0],
   *        [0, 0, 0, 0, 0, 0, 1]]
   */
  IdArray a_indptr = aten::VecToIdArray(
      std::vector<IdType>({0, 1, 3, 4}), sizeof(IdType) * 8, CTX);
  IdArray a_indices = aten::VecToIdArray(
      std::vector<IdType>({2, 0, 2, 1}), sizeof(IdType) * 8, CTX);
  IdArray b_indptr = aten::VecToIdArray(
      std::vector<IdType>({0, 2, 3}), sizeof(IdType) * 8, CTX);
  IdArray b_indices = aten::VecToIdArray(
      std::vector<IdType>({0, 1, 1}), sizeof(IdType) * 8, CTX);
  IdArray b_data = aten::VecToIdArray(
      std::vector<IdType>({2, 0, 1}), sizeof(IdType) * 8, CTX);
  IdArray c_indptr =
      aten::VecToIdArray(std::vector<IdType>({0, 1}), sizeof(IdType) * 8, CTX);
  IdArray c_indices =
      aten::VecToIdArray(std::vector<IdType>({0}), sizeof(IdType) * 8, CTX);
  IdArray bc_indptr = aten::VecToIdArray(
      std::vector<IdType>({0, 2, 3, 4}), sizeof(IdType) * 8, CTX);
  IdArray bc_indices = aten::VecToIdArray(
      std::vector<IdType>({0, 1, 1, 3}), sizeof(IdType) * 8, CTX);
  IdArray bc_data = aten::VecToIdArray(
      std::vector<IdType>({2, 0, 1, 3}), sizeof(IdType) * 8, CTX);
  IdArray abc_indptr = aten::VecToIdArray(
      std::vector<IdType>({0, 1, 3, 4, 6, 7, 8}), sizeof(IdType) * 8, CTX);
  IdArray abc_indices = aten::VecToIdArray(
      std::vector<IdType>({2, 0, 2, 1, 3, 4, 4, 6}), sizeof(IdType) * 8, CTX);
  IdArray abc_data = aten::VecToIdArray(
      std::vector<IdType>({0, 1, 2, 3, 6, 4, 5, 7}), sizeof(IdType) * 8, CTX);
  const aten::CSRMatrix &csr_a =
      aten::CSRMatrix(3, 3, a_indptr, a_indices, aten::NullArray(), false);
  const aten::CSRMatrix &csr_b =
      aten::CSRMatrix(2, 3, b_indptr, b_indices, b_data, true);
  const aten::CSRMatrix &csr_c =
      aten::CSRMatrix(1, 1, c_indptr, c_indices, aten::NullArray(), true);

  const std::vector<aten::CSRMatrix> csrs_bc({csr_b, csr_c});
  const aten::CSRMatrix &csr_bc = aten::DisjointUnionCsr(csrs_bc);
  ASSERT_EQ(csr_bc.num_rows, 3);
  ASSERT_EQ(csr_bc.num_cols, 4);
  ASSERT_TRUE(ArrayEQ<IdType>(csr_bc.indptr, bc_indptr));
  ASSERT_TRUE(ArrayEQ<IdType>(csr_bc.indices, bc_indices));
  ASSERT_TRUE(ArrayEQ<IdType>(csr_bc.data, bc_data));
  ASSERT_TRUE(csr_bc.sorted);

  const std::vector<uint64_t> edge_cumsum({0, 3, 4});
  const std::vector<uint64_t> src_vertex_cumsum({0, 2, 3});
  const std::vector<uint64_t> dst_vertex_cumsum({0, 3, 4});
  const std::vector<aten::CSRMatrix> &p_csrs =
      aten::DisjointPartitionCsrBySizes(
          csr_bc, 2, edge_cumsum, src_vertex_cumsum, dst_vertex_cumsum);
  ASSERT_EQ(p_csrs[0].num_rows, csr_b.num_rows);
  ASSERT_EQ(p_csrs[0].num_cols, csr_b.num_cols);
  ASSERT_EQ(p_csrs[1].num_rows, csr_c.num_rows);
  ASSERT_EQ(p_csrs[1].num_cols, csr_c.num_cols);
  ASSERT_TRUE(ArrayEQ<IdType>(p_csrs[0].indptr, csr_b.indptr));
  ASSERT_TRUE(ArrayEQ<IdType>(p_csrs[0].indices, csr_b.indices));
  ASSERT_TRUE(ArrayEQ<IdType>(p_csrs[0].data, csr_b.data));
  ASSERT_TRUE(ArrayEQ<IdType>(p_csrs[1].indptr, csr_c.indptr));
  ASSERT_TRUE(ArrayEQ<IdType>(p_csrs[1].indices, csr_c.indices));
  ASSERT_TRUE(p_csrs[0].sorted);
  ASSERT_TRUE(p_csrs[1].sorted);

  const std::vector<aten::CSRMatrix> csrs_abc({csr_a, csr_b, csr_c});
  const aten::CSRMatrix &csr_abc = aten::DisjointUnionCsr(csrs_abc);
  ASSERT_EQ(csr_abc.num_rows, 6);
  ASSERT_EQ(csr_abc.num_cols, 7);
  ASSERT_TRUE(ArrayEQ<IdType>(csr_abc.indptr, abc_indptr));
  ASSERT_TRUE(ArrayEQ<IdType>(csr_abc.indices, abc_indices));
  ASSERT_TRUE(ArrayEQ<IdType>(csr_abc.data, abc_data));
  ASSERT_FALSE(csr_abc.sorted);

  const std::vector<uint64_t> edge_cumsum_abc({0, 4, 7, 8});
  const std::vector<uint64_t> src_vertex_cumsum_abc({0, 3, 5, 6});
  const std::vector<uint64_t> dst_vertex_cumsum_abc({0, 3, 6, 7});
  const std::vector<aten::CSRMatrix> &p_csrs_abc =
      aten::DisjointPartitionCsrBySizes(
          csr_abc, 3, edge_cumsum_abc, src_vertex_cumsum_abc,
          dst_vertex_cumsum_abc);
  ASSERT_EQ(p_csrs_abc[0].num_rows, csr_a.num_rows);
  ASSERT_EQ(p_csrs_abc[0].num_cols, csr_a.num_cols);
  ASSERT_EQ(p_csrs_abc[1].num_rows, csr_b.num_rows);
  ASSERT_EQ(p_csrs_abc[1].num_cols, csr_b.num_cols);
  ASSERT_EQ(p_csrs_abc[2].num_rows, csr_c.num_rows);
  ASSERT_EQ(p_csrs_abc[2].num_cols, csr_c.num_cols);
  ASSERT_TRUE(ArrayEQ<IdType>(p_csrs_abc[0].indptr, csr_a.indptr));
  ASSERT_TRUE(ArrayEQ<IdType>(p_csrs_abc[0].indices, csr_a.indices));
  ASSERT_TRUE(ArrayEQ<IdType>(p_csrs_abc[1].indptr, csr_b.indptr));
  ASSERT_TRUE(ArrayEQ<IdType>(p_csrs_abc[1].indices, csr_b.indices));
  ASSERT_TRUE(ArrayEQ<IdType>(p_csrs_abc[1].data, csr_b.data));
  ASSERT_TRUE(ArrayEQ<IdType>(p_csrs_abc[2].indptr, csr_c.indptr));
  ASSERT_TRUE(ArrayEQ<IdType>(p_csrs_abc[2].indices, csr_c.indices));
  ASSERT_FALSE(p_csrs_abc[0].sorted);
  ASSERT_FALSE(p_csrs_abc[1].sorted);
  ASSERT_FALSE(p_csrs_abc[2].sorted);
}

TEST(DisjointUnionTest, TestDisjointUnionPartitionCsr) {
  _TestDisjointUnionPartitionCsr<int32_t>(CPU);
  _TestDisjointUnionPartitionCsr<int64_t>(CPU);
#ifdef DGL_USE_CUDA
  _TestDisjointUnionPartitionCsr<int32_t>(GPU);
  _TestDisjointUnionPartitionCsr<int64_t>(GPU);
#endif
}

template <typename IdType>
void _TestSliceContiguousChunkCoo(DGLContext ctx) {
  /**
   * A = [[1, 0, 0, 0],
   *      [0, 0, 1, 0],
   *      [0, 0, 0, 0]]
   *
   * B = [[1, 0, 0],
   *      [0, 0, 1]]
   *
   * C = [[0]]
   *
   */
  IdArray a_row =
      aten::VecToIdArray(std::vector<IdType>({0, 1}), sizeof(IdType) * 8, CTX);
  IdArray a_col =
      aten::VecToIdArray(std::vector<IdType>({0, 2}), sizeof(IdType) * 8, CTX);
  const aten::COOMatrix &coo_a =
      aten::COOMatrix(3, 4, a_row, a_col, aten::NullArray(), true, false);

  IdArray b_row =
      aten::VecToIdArray(std::vector<IdType>({0, 1}), sizeof(IdType) * 8, CTX);
  IdArray b_col =
      aten::VecToIdArray(std::vector<IdType>({0, 2}), sizeof(IdType) * 8, CTX);
  const aten::COOMatrix &coo_b_raw =
      aten::COOMatrix(2, 3, b_row, b_col, aten::NullArray(), true, false);

  const std::vector<uint64_t> edge_range_b({0, 2});
  const std::vector<uint64_t> src_vertex_range_b({0, 2});
  const std::vector<uint64_t> dst_vertex_range_b({0, 3});
  const aten::COOMatrix &coo_b = aten::COOSliceContiguousChunk(
      coo_a, edge_range_b, src_vertex_range_b, dst_vertex_range_b);
  ASSERT_EQ(coo_b_raw.num_rows, coo_b.num_rows);
  ASSERT_EQ(coo_b_raw.num_cols, coo_b.num_cols);
  ASSERT_TRUE(ArrayEQ<IdType>(coo_b_raw.row, coo_b.row));
  ASSERT_TRUE(ArrayEQ<IdType>(coo_b_raw.col, coo_b.col));
  ASSERT_TRUE(coo_b.row_sorted);
  ASSERT_FALSE(coo_b.col_sorted);

  IdArray c_row =
      aten::VecToIdArray(std::vector<IdType>({}), sizeof(IdType) * 8, CTX);
  IdArray c_col =
      aten::VecToIdArray(std::vector<IdType>({}), sizeof(IdType) * 8, CTX);
  const aten::COOMatrix &coo_c_raw =
      aten::COOMatrix(1, 1, c_row, c_col, aten::NullArray(), true, false);

  const std::vector<uint64_t> edge_range_c({2, 2});
  const std::vector<uint64_t> src_vertex_range_c({2, 3});
  const std::vector<uint64_t> dst_vertex_range_c({3, 4});
  const aten::COOMatrix &coo_c = aten::COOSliceContiguousChunk(
      coo_a, edge_range_c, src_vertex_range_c, dst_vertex_range_c);
  ASSERT_EQ(coo_c_raw.num_rows, coo_c.num_rows);
  ASSERT_EQ(coo_c_raw.num_cols, coo_c.num_cols);
  ASSERT_TRUE(ArrayEQ<IdType>(coo_c.row, c_row));
  ASSERT_TRUE(ArrayEQ<IdType>(coo_c.col, c_col));
  ASSERT_TRUE(coo_c.row_sorted);
  ASSERT_FALSE(coo_c.col_sorted);
}

TEST(SliceContiguousChunk, TestSliceContiguousChunkCoo) {
  _TestSliceContiguousChunkCoo<int32_t>(CPU);
  _TestSliceContiguousChunkCoo<int64_t>(CPU);
#ifdef DGL_USE_CUDA
  _TestSliceContiguousChunkCoo<int32_t>(GPU);
  _TestSliceContiguousChunkCoo<int64_t>(GPU);
#endif
}

template <typename IdType>
void _TestSliceContiguousChunkCsr(DGLContext ctx) {
  /**
   * A = [[1, 0, 0, 0],
   *      [0, 0, 1, 0],
   *      [0, 0, 0, 0]]
   *
   * B = [[1, 0, 0],
   *      [0, 0, 1]]
   *
   * C = [[0]]
   *
   */
  IdArray a_indptr = aten::VecToIdArray(
      std::vector<IdType>({0, 1, 2, 2}), sizeof(IdType) * 8, CTX);
  IdArray a_indices =
      aten::VecToIdArray(std::vector<IdType>({0, 2}), sizeof(IdType) * 8, CTX);
  const aten::CSRMatrix &csr_a =
      aten::CSRMatrix(3, 4, a_indptr, a_indices, aten::NullArray(), false);

  IdArray b_indptr = aten::VecToIdArray(
      std::vector<IdType>({0, 1, 2}), sizeof(IdType) * 8, CTX);
  IdArray b_indices =
      aten::VecToIdArray(std::vector<IdType>({0, 2}), sizeof(IdType) * 8, CTX);
  const aten::CSRMatrix &csr_b_raw =
      aten::CSRMatrix(2, 3, b_indptr, b_indices, aten::NullArray(), false);

  const std::vector<uint64_t> edge_range_b({0, 2});
  const std::vector<uint64_t> src_vertex_range_b({0, 2});
  const std::vector<uint64_t> dst_vertex_range_b({0, 3});
  const aten::CSRMatrix &csr_b = aten::CSRSliceContiguousChunk(
      csr_a, edge_range_b, src_vertex_range_b, dst_vertex_range_b);
  ASSERT_EQ(csr_b.num_rows, csr_b_raw.num_rows);
  ASSERT_EQ(csr_b.num_cols, csr_b_raw.num_cols);
  ASSERT_TRUE(ArrayEQ<IdType>(csr_b.indptr, csr_b_raw.indptr));
  ASSERT_TRUE(ArrayEQ<IdType>(csr_b.indices, csr_b_raw.indices));
  ASSERT_FALSE(csr_b.sorted);

  const std::vector<uint64_t> edge_range_c({2, 2});
  const std::vector<uint64_t> src_vertex_range_c({2, 3});
  const std::vector<uint64_t> dst_vertex_range_c({3, 4});
  const aten::CSRMatrix &csr_c = aten::CSRSliceContiguousChunk(
      csr_a, edge_range_c, src_vertex_range_c, dst_vertex_range_c);

  int64_t indptr_len = src_vertex_range_c[1] - src_vertex_range_c[0] + 1;
  IdArray c_indptr = aten::Full(0, indptr_len, sizeof(IdType) * 8, CTX);
  IdArray c_indices =
      aten::VecToIdArray(std::vector<IdType>({}), sizeof(IdType) * 8, CTX);
  const aten::CSRMatrix &csr_c_raw =
      aten::CSRMatrix(1, 1, c_indptr, c_indices, aten::NullArray(), false);

  ASSERT_EQ(csr_c.num_rows, csr_c_raw.num_rows);
  ASSERT_EQ(csr_c.num_cols, csr_c_raw.num_cols);
  ASSERT_TRUE(ArrayEQ<IdType>(csr_c.indptr, c_indptr));
  ASSERT_TRUE(ArrayEQ<IdType>(csr_c.indices, c_indices));
  ASSERT_FALSE(csr_c.sorted);
}

TEST(SliceContiguousChunk, TestSliceContiguousChunkCsr) {
  _TestSliceContiguousChunkCsr<int32_t>(CPU);
  _TestSliceContiguousChunkCsr<int64_t>(CPU);
#ifdef DGL_USE_CUDA
  _TestSliceContiguousChunkCsr<int32_t>(GPU);
  _TestSliceContiguousChunkCsr<int64_t>(GPU);
#endif
}

template <typename IdType>
void _TestMatrixUnionCsr(DGLContext ctx) {
  /**
   * A = [[0, 0, 0, 0],
   *      [0, 0, 0, 0],
   *      [0, 1, 0, 0],
   *      [1, 1, 1, 1],
   *      [0, 1, 1, 0],
   *      [1, 0, 0, 1]]
   *
   * B = [[0, 0, 0, 0],
   *      [1, 0, 0, 1],
   *      [0, 0, 1, 0],
   *      [1, 0, 0, 1],
   *      [1, 0, 0, 1]]
   *      [1, 0, 0, 1]]
   *
   * C = UnionCsr({A, B})
   *
   * C = [[0, 0, 0, 0],
   *      [1, 0, 0, 1],
   *      [0, 1, 1, 0],
   *      [2, 1, 1, 2],
   *      [1, 1, 1, 1]]
   *      [2, 0, 0, 2]]
   *
   * D = [[1, 0, 0, 0],
   *      [0, 0, 0, 0],
   *      [0, 0, 0, 0],
   *      [0, 0, 0, 0],
   *      [0, 0, 0, 0],
   *      [1, 0, 0, 1]]
   *
   * C = UnionCsr({A, B, D})
   *
   * C = [[1, 0, 0, 0],
   *      [1, 0, 0, 1],
   *      [0, 1, 1, 0],
   *      [2, 1, 1, 2],
   *      [1, 1, 1, 1]]
   *      [3, 0, 0, 3]]
   */
  IdArray a_indptr = aten::VecToIdArray(
      std::vector<IdType>({0, 0, 0, 1, 5, 7, 9}), sizeof(IdType) * 8, CTX);
  IdArray a_indices = aten::VecToIdArray(
      std::vector<IdType>({1, 0, 1, 2, 3, 1, 2, 0, 3}), sizeof(IdType) * 8,
      CTX);
  IdArray b_indptr = aten::VecToIdArray(
      std::vector<IdType>({0, 0, 2, 3, 5, 7, 9}), sizeof(IdType) * 8, CTX);
  IdArray b_indices = aten::VecToIdArray(
      std::vector<IdType>({0, 3, 2, 0, 3, 0, 3, 0, 3}), sizeof(IdType) * 8,
      CTX);
  IdArray c_indptr = aten::VecToIdArray(
      std::vector<IdType>({0, 0, 2, 4, 10, 14, 18}), sizeof(IdType) * 8, CTX);
  IdArray c_indices = aten::VecToIdArray(
      std::vector<IdType>(
          {0, 3, 1, 2, 0, 0, 1, 2, 3, 3, 0, 1, 2, 3, 0, 0, 3, 3}),
      sizeof(IdType) * 8, CTX);
  IdArray c_data = aten::VecToIdArray(
      std::vector<IdType>(
          {9, 10, 0, 11, 1, 12, 2, 3, 4, 13, 14, 5, 6, 15, 7, 16, 8, 17}),
      sizeof(IdType) * 8, CTX);

  const aten::CSRMatrix &csr_a =
      aten::CSRMatrix(6, 4, a_indptr, a_indices, aten::NullArray(), true);
  const aten::CSRMatrix &csr_b =
      aten::CSRMatrix(6, 4, b_indptr, b_indices, aten::NullArray(), true);

  const aten::CSRMatrix &csr_aUb = aten::UnionCsr({csr_a, csr_b});
  ASSERT_EQ(csr_aUb.num_rows, 6);
  ASSERT_EQ(csr_aUb.num_cols, 4);
  ASSERT_TRUE(ArrayEQ<IdType>(csr_aUb.indptr, c_indptr));
  ASSERT_TRUE(ArrayEQ<IdType>(csr_aUb.indices, c_indices));
  ASSERT_TRUE(ArrayEQ<IdType>(csr_aUb.data, c_data));
  ASSERT_TRUE(csr_aUb.sorted);

  IdArray a_data = aten::VecToIdArray(
      std::vector<IdType>({8, 7, 6, 5, 4, 3, 2, 1, 0}), sizeof(IdType) * 8,
      CTX);

  c_data = aten::VecToIdArray(
      std::vector<IdType>(
          {9, 10, 8, 11, 7, 12, 6, 5, 4, 13, 14, 3, 2, 15, 1, 16, 0, 17}),
      sizeof(IdType) * 8, CTX);
  const aten::CSRMatrix &csr_ad =
      aten::CSRMatrix(6, 4, a_indptr, a_indices, a_data, true);
  const aten::CSRMatrix &csr_adUb = aten::UnionCsr({csr_ad, csr_b});
  ASSERT_EQ(csr_adUb.num_rows, 6);
  ASSERT_EQ(csr_adUb.num_cols, 4);
  ASSERT_TRUE(ArrayEQ<IdType>(csr_adUb.indptr, c_indptr));
  ASSERT_TRUE(ArrayEQ<IdType>(csr_adUb.indices, c_indices));
  ASSERT_TRUE(ArrayEQ<IdType>(csr_adUb.data, c_data));
  ASSERT_TRUE(csr_adUb.sorted);

  IdArray b_indices2 = aten::VecToIdArray(
      std::vector<IdType>({0, 3, 2, 0, 3, 3, 0, 0, 3}), sizeof(IdType) * 8,
      CTX);
  c_indices = aten::VecToIdArray(
      std::vector<IdType>(
          {0, 3, 1, 2, 0, 1, 2, 3, 0, 3, 1, 2, 3, 0, 0, 3, 0, 3}),
      sizeof(IdType) * 8, CTX);
  c_data = aten::VecToIdArray(
      std::vector<IdType>(
          {9, 10, 0, 11, 1, 2, 3, 4, 12, 13, 5, 6, 14, 15, 7, 8, 16, 17}),
      sizeof(IdType) * 8, CTX);
  const aten::CSRMatrix &csr_b2 =
      aten::CSRMatrix(6, 4, b_indptr, b_indices2, aten::NullArray(), false);
  const aten::CSRMatrix &csr_aUb2 = aten::UnionCsr({csr_a, csr_b2});
  ASSERT_EQ(csr_aUb2.num_rows, 6);
  ASSERT_EQ(csr_aUb2.num_cols, 4);
  ASSERT_TRUE(ArrayEQ<IdType>(csr_aUb2.indptr, c_indptr));
  ASSERT_TRUE(ArrayEQ<IdType>(csr_aUb2.indices, c_indices));
  ASSERT_TRUE(ArrayEQ<IdType>(csr_aUb2.data, c_data));
  ASSERT_FALSE(csr_aUb2.sorted);

  IdArray a_indices2 = aten::VecToIdArray(
      std::vector<IdType>({1, 3, 2, 1, 0, 1, 2, 0, 3}), sizeof(IdType) * 8,
      CTX);
  c_indices = aten::VecToIdArray(
      std::vector<IdType>(
          {0, 3, 1, 2, 3, 2, 1, 0, 0, 3, 1, 2, 0, 3, 0, 3, 0, 3}),
      sizeof(IdType) * 8, CTX);
  const aten::CSRMatrix &csr_a2 =
      aten::CSRMatrix(6, 4, a_indptr, a_indices2, aten::NullArray(), false);
  const aten::CSRMatrix &csr_aUb3 = aten::UnionCsr({csr_a2, csr_b});
  ASSERT_EQ(csr_aUb3.num_rows, 6);
  ASSERT_EQ(csr_aUb3.num_cols, 4);
  ASSERT_TRUE(ArrayEQ<IdType>(csr_aUb3.indptr, c_indptr));
  ASSERT_TRUE(ArrayEQ<IdType>(csr_aUb3.indices, c_indices));
  ASSERT_TRUE(ArrayEQ<IdType>(csr_aUb3.data, c_data));
  ASSERT_FALSE(csr_aUb3.sorted);

  c_indices = aten::VecToIdArray(
      std::vector<IdType>(
          {0, 3, 1, 2, 3, 2, 1, 0, 0, 3, 1, 2, 3, 0, 0, 3, 0, 3}),
      sizeof(IdType) * 8, CTX);
  const aten::CSRMatrix &csr_aUb4 = aten::UnionCsr({csr_a2, csr_b2});
  ASSERT_EQ(csr_aUb4.num_rows, 6);
  ASSERT_EQ(csr_aUb4.num_cols, 4);
  ASSERT_TRUE(ArrayEQ<IdType>(csr_aUb4.indptr, c_indptr));
  ASSERT_TRUE(ArrayEQ<IdType>(csr_aUb4.indices, c_indices));
  ASSERT_TRUE(ArrayEQ<IdType>(csr_aUb4.data, c_data));
  ASSERT_FALSE(csr_aUb4.sorted);

  IdArray d_indptr = aten::VecToIdArray(
      std::vector<IdType>({0, 1, 1, 1, 1, 1, 3}), sizeof(IdType) * 8, CTX);
  IdArray d_indices = aten::VecToIdArray(
      std::vector<IdType>({0, 0, 3}), sizeof(IdType) * 8, CTX);
  c_indptr = aten::VecToIdArray(
      std::vector<IdType>({0, 1, 3, 5, 11, 15, 21}), sizeof(IdType) * 8, CTX);
  c_indices = aten::VecToIdArray(
      std::vector<IdType>(
          {0, 0, 3, 1, 2, 0, 0, 1, 2, 3, 3, 0, 1, 2, 3, 0, 0, 0, 3, 3, 3}),
      sizeof(IdType) * 8, CTX);
  c_data = aten::VecToIdArray(
      std::vector<IdType>({18, 9, 10, 8,  11, 7,  12, 6, 5,  4, 13,
                           14, 3, 2,  15, 1,  16, 19, 0, 17, 20}),
      sizeof(IdType) * 8, CTX);
  const aten::CSRMatrix &csr_d =
      aten::CSRMatrix(6, 4, d_indptr, d_indices, aten::NullArray(), true);
  const aten::CSRMatrix &csr_aUbUd = aten::UnionCsr({csr_ad, csr_b, csr_d});
  ASSERT_EQ(csr_aUbUd.num_rows, 6);
  ASSERT_EQ(csr_aUbUd.num_cols, 4);
  ASSERT_TRUE(ArrayEQ<IdType>(csr_aUbUd.indptr, c_indptr));
  ASSERT_TRUE(ArrayEQ<IdType>(csr_aUbUd.indices, c_indices));
  ASSERT_TRUE(ArrayEQ<IdType>(csr_aUbUd.data, c_data));
  ASSERT_TRUE(csr_aUbUd.sorted);

  c_indices = aten::VecToIdArray(
      std::vector<IdType>(
          {0, 0, 3, 1, 2, 3, 2, 1, 0, 0, 3, 1, 2, 3, 0, 0, 3, 0, 3, 0, 3}),
      sizeof(IdType) * 8, CTX);
  c_data = aten::VecToIdArray(
      std::vector<IdType>({18, 9, 10, 0,  11, 1, 2,  3,  4,  12, 13,
                           5,  6, 14, 15, 7,  8, 16, 17, 19, 20}),
      sizeof(IdType) * 8, CTX);

  const aten::CSRMatrix &csr_aUbUd2 = aten::UnionCsr({csr_a2, csr_b2, csr_d});
  ASSERT_EQ(csr_aUbUd2.num_rows, 6);
  ASSERT_EQ(csr_aUbUd2.num_cols, 4);
  ASSERT_TRUE(ArrayEQ<IdType>(csr_aUbUd2.indptr, c_indptr));
  ASSERT_TRUE(ArrayEQ<IdType>(csr_aUbUd2.indices, c_indices));
  ASSERT_TRUE(ArrayEQ<IdType>(csr_aUbUd2.data, c_data));
  ASSERT_FALSE(csr_aUbUd2.sorted);
}

TEST(MatrixUnionTest, TestMatrixUnionCsr) {
  _TestMatrixUnionCsr<int32_t>(CPU);
  _TestMatrixUnionCsr<int64_t>(CPU);
}

template <typename IdType>
void _TestMatrixUnionCoo(DGLContext ctx) {
  /**
   * A = [[0, 0, 0, 0],
   *      [0, 0, 0, 0],
   *      [0, 1, 0, 0],
   *      [1, 1, 1, 1],
   *      [0, 1, 1, 0],
   *      [1, 0, 0, 1]]
   *
   * B = [[0, 0, 0, 0],
   *      [1, 0, 0, 1],
   *      [0, 0, 1, 0],
   *      [1, 0, 0, 1],
   *      [1, 0, 0, 1]]
   *      [1, 0, 0, 1]]
   *
   * C = UnionCsr({A, B})
   *
   * C = [[0, 0, 0, 0],
   *      [1, 0, 0, 1],
   *      [0, 1, 1, 0],
   *      [2, 1, 1, 2],
   *      [1, 1, 1, 1]]
   *      [2, 0, 0, 2]]
   *
   * D = [[1, 0, 0, 0],
   *      [0, 0, 0, 0],
   *      [0, 0, 0, 0],
   *      [0, 0, 0, 0],
   *      [0, 0, 0, 0],
   *      [1, 0, 0, 1]]
   *
   * C = UnionCsr({A, B, D})
   *
   * C = [[1, 0, 0, 0],
   *      [1, 0, 0, 1],
   *      [0, 1, 1, 0],
   *      [2, 1, 1, 2],
   *      [1, 1, 1, 1]]
   *      [3, 0, 0, 3]]
   */
  IdArray a_row = aten::VecToIdArray(
      std::vector<IdType>({2, 3, 3, 3, 3, 4, 4, 5, 5}), sizeof(IdType) * 8,
      CTX);
  IdArray a_col = aten::VecToIdArray(
      std::vector<IdType>({1, 0, 1, 2, 3, 1, 2, 0, 3}), sizeof(IdType) * 8,
      CTX);
  IdArray b_row = aten::VecToIdArray(
      std::vector<IdType>({1, 1, 2, 3, 3, 4, 4, 5, 5}), sizeof(IdType) * 8,
      CTX);
  IdArray b_col = aten::VecToIdArray(
      std::vector<IdType>({0, 3, 2, 0, 3, 0, 3, 0, 3}), sizeof(IdType) * 8,
      CTX);
  IdArray c_row = aten::VecToIdArray(
      std::vector<IdType>(
          {2, 3, 3, 3, 3, 4, 4, 5, 5, 1, 1, 2, 3, 3, 4, 4, 5, 5}),
      sizeof(IdType) * 8, CTX);
  IdArray c_col = aten::VecToIdArray(
      std::vector<IdType>(
          {1, 0, 1, 2, 3, 1, 2, 0, 3, 0, 3, 2, 0, 3, 0, 3, 0, 3}),
      sizeof(IdType) * 8, CTX);
  const aten::COOMatrix &coo_a =
      aten::COOMatrix(6, 4, a_row, a_col, aten::NullArray(), true, true);
  const aten::COOMatrix &coo_b =
      aten::COOMatrix(6, 4, b_row, b_col, aten::NullArray(), true, true);
  const std::vector<aten::COOMatrix> coos_ab({coo_a, coo_b});
  const aten::COOMatrix &coo_ab = aten::UnionCoo(coos_ab);
  ASSERT_EQ(coo_ab.num_rows, 6);
  ASSERT_EQ(coo_ab.num_cols, 4);
  ASSERT_TRUE(ArrayEQ<IdType>(coo_ab.row, c_row));
  ASSERT_TRUE(ArrayEQ<IdType>(coo_ab.col, c_col));
  ASSERT_FALSE(COOHasData(coo_ab));
  ASSERT_FALSE(coo_ab.row_sorted);
  ASSERT_FALSE(coo_ab.col_sorted);

  IdArray a_data = aten::VecToIdArray(
      std::vector<IdType>({2, 1, 0, 3, 4, 5, 6, 7, 8}), sizeof(IdType) * 8,
      CTX);

  IdArray c_data = aten::VecToIdArray(
      std::vector<IdType>(
          {2, 1, 0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17}),
      sizeof(IdType) * 8, CTX);
  const aten::COOMatrix &coo_a2 =
      aten::COOMatrix(6, 4, a_row, a_col, a_data, true, true);
  const std::vector<aten::COOMatrix> coos_ab2({coo_a2, coo_b});
  const aten::COOMatrix &coo_ab2 = aten::UnionCoo(coos_ab2);
  ASSERT_EQ(coo_ab2.num_rows, 6);
  ASSERT_EQ(coo_ab2.num_cols, 4);
  ASSERT_TRUE(ArrayEQ<IdType>(coo_ab2.row, c_row));
  ASSERT_TRUE(ArrayEQ<IdType>(coo_ab2.col, c_col));
  ASSERT_TRUE(COOHasData(coo_ab2));
  ASSERT_TRUE(ArrayEQ<IdType>(coo_ab2.data, c_data));
  ASSERT_FALSE(coo_ab2.row_sorted);
  ASSERT_FALSE(coo_ab2.col_sorted);

  IdArray b_data = aten::VecToIdArray(
      std::vector<IdType>({0, 1, 2, 3, 4, 5, 6, 8, 7}), sizeof(IdType) * 8,
      CTX);
  c_data = aten::VecToIdArray(
      std::vector<IdType>(
          {2, 1, 0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 16}),
      sizeof(IdType) * 8, CTX);
  const aten::COOMatrix &coo_b2 =
      aten::COOMatrix(6, 4, b_row, b_col, b_data, true, true);
  const std::vector<aten::COOMatrix> coos_ab3({coo_a2, coo_b2});
  const aten::COOMatrix &coo_ab3 = aten::UnionCoo(coos_ab3);
  ASSERT_EQ(coo_ab3.num_rows, 6);
  ASSERT_EQ(coo_ab3.num_cols, 4);
  ASSERT_TRUE(ArrayEQ<IdType>(coo_ab3.row, c_row));
  ASSERT_TRUE(ArrayEQ<IdType>(coo_ab3.col, c_col));
  ASSERT_TRUE(COOHasData(coo_ab3));
  ASSERT_TRUE(ArrayEQ<IdType>(coo_ab3.data, c_data));
  ASSERT_FALSE(coo_ab3.row_sorted);
  ASSERT_FALSE(coo_ab3.col_sorted);

  c_data = aten::VecToIdArray(
      std::vector<IdType>(
          {2, 1, 0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 16}),
      sizeof(IdType) * 8, CTX);

  const std::vector<aten::COOMatrix> coos_ab4({coo_a2, coo_b2});
  const aten::COOMatrix &coo_ab4 = aten::UnionCoo(coos_ab4);
  ASSERT_EQ(coo_ab4.num_rows, 6);
  ASSERT_EQ(coo_ab4.num_cols, 4);
  ASSERT_TRUE(ArrayEQ<IdType>(coo_ab4.row, c_row));
  ASSERT_TRUE(ArrayEQ<IdType>(coo_ab4.col, c_col));
  ASSERT_TRUE(COOHasData(coo_ab4));
  ASSERT_TRUE(ArrayEQ<IdType>(coo_ab4.data, c_data));
  ASSERT_FALSE(coo_ab4.row_sorted);
  ASSERT_FALSE(coo_ab4.col_sorted);

  IdArray d_row = aten::VecToIdArray(
      std::vector<IdType>({0, 5, 5}), sizeof(IdType) * 8, CTX);
  IdArray d_col = aten::VecToIdArray(
      std::vector<IdType>({0, 0, 3}), sizeof(IdType) * 8, CTX);
  c_row = aten::VecToIdArray(
      std::vector<IdType>(
          {2, 3, 3, 3, 3, 4, 4, 5, 5, 1, 1, 2, 3, 3, 4, 4, 5, 5, 0, 5, 5}),
      sizeof(IdType) * 8, CTX);
  c_col = aten::VecToIdArray(
      std::vector<IdType>(
          {1, 0, 1, 2, 3, 1, 2, 0, 3, 0, 3, 2, 0, 3, 0, 3, 0, 3, 0, 0, 3}),
      sizeof(IdType) * 8, CTX);

  const aten::COOMatrix &coo_d =
      aten::COOMatrix(6, 4, d_row, d_col, aten::NullArray(), true, true);
  const aten::COOMatrix &csr_aUbUd = aten::UnionCoo({coo_a, coo_b, coo_d});
  ASSERT_EQ(csr_aUbUd.num_rows, 6);
  ASSERT_EQ(csr_aUbUd.num_cols, 4);
  ASSERT_TRUE(ArrayEQ<IdType>(csr_aUbUd.row, c_row));
  ASSERT_TRUE(ArrayEQ<IdType>(csr_aUbUd.col, c_col));
  ASSERT_FALSE(COOHasData(csr_aUbUd));
  ASSERT_FALSE(csr_aUbUd.row_sorted);
  ASSERT_FALSE(csr_aUbUd.col_sorted);

  c_data = aten::VecToIdArray(
      std::vector<IdType>({2,  1,  0,  3,  4,  5,  6,  7,  8,  9, 10,
                           11, 12, 13, 14, 15, 17, 16, 18, 19, 20}),
      sizeof(IdType) * 8, CTX);

  const aten::COOMatrix &csr_aUbUd2 = aten::UnionCoo({coo_a2, coo_b2, coo_d});
  ASSERT_EQ(csr_aUbUd2.num_rows, 6);
  ASSERT_EQ(csr_aUbUd2.num_cols, 4);
  ASSERT_TRUE(ArrayEQ<IdType>(csr_aUbUd2.row, c_row));
  ASSERT_TRUE(ArrayEQ<IdType>(csr_aUbUd2.col, c_col));
  ASSERT_TRUE(COOHasData(csr_aUbUd2));
  ASSERT_TRUE(ArrayEQ<IdType>(csr_aUbUd2.data, c_data));
  ASSERT_FALSE(csr_aUbUd2.row_sorted);
  ASSERT_FALSE(csr_aUbUd2.col_sorted);
}

TEST(MatrixUnionTest, TestMatrixUnionCoo) {
  _TestMatrixUnionCoo<int32_t>(CPU);
  _TestMatrixUnionCoo<int64_t>(CPU);
}

template <typename IDX>
void _TestCumSum(DGLContext ctx) {
  IdArray a = aten::VecToIdArray(
      std::vector<IDX>({8, 6, 7, 5, 3, 0, 9}), sizeof(IDX) * 8, ctx);
  {
    IdArray tb = aten::VecToIdArray(
        std::vector<IDX>({8, 14, 21, 26, 29, 29, 38}), sizeof(IDX) * 8, ctx);
    IdArray b = aten::CumSum(a);
    ASSERT_TRUE(ArrayEQ<IDX>(b, tb));
  }
  {
    IdArray tb = aten::VecToIdArray(
        std::vector<IDX>({0, 8, 14, 21, 26, 29, 29, 38}), sizeof(IDX) * 8, ctx);
    IdArray b = aten::CumSum(a, true);
    ASSERT_TRUE(ArrayEQ<IDX>(b, tb));
  }
  a = aten::VecToIdArray(std::vector<IDX>({}), sizeof(IDX) * 8, ctx);
  {
    IdArray tb = aten::VecToIdArray(std::vector<IDX>({}), sizeof(IDX) * 8, ctx);
    IdArray b = aten::CumSum(a);
    ASSERT_TRUE(ArrayEQ<IDX>(b, tb));
  }
  {
    IdArray tb = aten::VecToIdArray(std::vector<IDX>({}), sizeof(IDX) * 8, ctx);
    IdArray b = aten::CumSum(a);
    ASSERT_TRUE(ArrayEQ<IDX>(b, tb));
  }
}

TEST(ArrayTest, CumSum) {
  _TestCumSum<int32_t>(CPU);
  _TestCumSum<int64_t>(CPU);
#ifdef DGL_USE_CUDA
  _TestCumSum<int32_t>(GPU);
  _TestCumSum<int64_t>(GPU);
#endif
}

template <typename IDX, typename D>
void _TestScatter_(DGLContext ctx) {
  IdArray out = aten::Full(1, 10, 8 * sizeof(IDX), ctx);
  IdArray idx =
      aten::VecToIdArray(std::vector<IDX>({2, 3, 9}), sizeof(IDX) * 8, ctx);
  IdArray val =
      aten::VecToIdArray(std::vector<IDX>({-20, 30, 90}), sizeof(IDX) * 8, ctx);
  aten::Scatter_(idx, val, out);
  IdArray tout = aten::VecToIdArray(
      std::vector<IDX>({1, 1, -20, 30, 1, 1, 1, 1, 1, 90}), sizeof(IDX) * 8,
      ctx);
  ASSERT_TRUE(ArrayEQ<IDX>(out, tout));
}

TEST(ArrayTest, Scatter_) {
  _TestScatter_<int32_t, int32_t>(CPU);
  _TestScatter_<int64_t, int32_t>(CPU);
  _TestScatter_<int32_t, int64_t>(CPU);
  _TestScatter_<int64_t, int64_t>(CPU);
#ifdef DGL_USE_CUDA
  _TestScatter_<int32_t, int32_t>(GPU);
  _TestScatter_<int64_t, int32_t>(GPU);
  _TestScatter_<int32_t, int64_t>(GPU);
  _TestScatter_<int64_t, int64_t>(GPU);
#endif
}

template <typename IDX>
void _TestNonZero(DGLContext ctx) {
  auto val = aten::VecToIdArray(
      std::vector<IDX>({0, 1, 2, 0, -10, 0, 0, 23}), sizeof(IDX) * 8, ctx);
  auto idx = aten::NonZero(val);
  auto tidx = aten::VecToIdArray(std::vector<int64_t>({1, 2, 4, 7}), 64, ctx);
  ASSERT_TRUE(ArrayEQ<IDX>(idx, tidx));

  val = aten::VecToIdArray(std::vector<IDX>({}), sizeof(IDX) * 8, ctx);
  idx = aten::NonZero(val);
  tidx = aten::VecToIdArray(std::vector<int64_t>({}), 64, ctx);
  ASSERT_TRUE(ArrayEQ<IDX>(idx, tidx));

  val =
      aten::VecToIdArray(std::vector<IDX>({0, 0, 0, 0}), sizeof(IDX) * 8, ctx);
  idx = aten::NonZero(val);
  tidx = aten::VecToIdArray(std::vector<int64_t>({}), 64, ctx);
  ASSERT_TRUE(ArrayEQ<IDX>(idx, tidx));

  val = aten::Full(1, 3, sizeof(IDX) * 8, ctx);
  idx = aten::NonZero(val);
  tidx = aten::VecToIdArray(std::vector<int64_t>({0, 1, 2}), 64, ctx);
  ASSERT_TRUE(ArrayEQ<IDX>(idx, tidx));
}

TEST(ArrayTest, NonZero) {
  _TestNonZero<int32_t>(CPU);
  _TestNonZero<int64_t>(CPU);
#ifdef DGL_USE_CUDA
  _TestNonZero<int32_t>(GPU);
  _TestNonZero<int64_t>(GPU);
#endif
}

template <typename IdType>
void _TestLineGraphCOO(DGLContext ctx) {
  /**
   * A = [[0, 0, 1, 0],
   *      [1, 0, 1, 0],
   *      [1, 1, 0, 0],
   *      [0, 0, 0, 1]]
   * row: 0 1 1 2 2 3
   * col: 2 0 2 0 1 3
   * ID:  0 1 2 3 4 5
   *
   * B = COOLineGraph(A, backtracking=False)
   *
   * B = [[0, 0, 0, 0, 1, 0],
   *      [1, 0, 0, 0, 0, 0],
   *      [0, 0, 0, 1, 0, 0],
   *      [0, 0, 0, 0, 0, 0],
   *      [0, 1, 0, 0, 0, 0],
   *      [0, 0, 0, 0, 0, 0]]
   *
   * C = COOLineGraph(A, backtracking=True)
   *
   * C = [[0, 0, 0, 1, 1, 0],
   *      [1, 0, 0, 0, 0, 0],
   *      [0, 0, 0, 1, 1, 0],
   *      [1, 0, 0, 0, 0, 0],
   *      [0, 1, 1, 0, 0, 0],
   *      [0, 0, 0, 0, 0, 0]]
   */
  IdArray a_row = aten::VecToIdArray(
      std::vector<IdType>({0, 1, 1, 2, 2, 3}), sizeof(IdType) * 8, ctx);
  IdArray a_col = aten::VecToIdArray(
      std::vector<IdType>({2, 0, 2, 0, 1, 3}), sizeof(IdType) * 8, ctx);
  IdArray b_row = aten::VecToIdArray(
      std::vector<IdType>({0, 1, 2, 4}), sizeof(IdType) * 8, ctx);
  IdArray b_col = aten::VecToIdArray(
      std::vector<IdType>({4, 0, 3, 1}), sizeof(IdType) * 8, ctx);
  IdArray c_row = aten::VecToIdArray(
      std::vector<IdType>({0, 0, 1, 2, 2, 3, 4, 4}), sizeof(IdType) * 8, ctx);
  IdArray c_col = aten::VecToIdArray(
      std::vector<IdType>({3, 4, 0, 3, 4, 0, 1, 2}), sizeof(IdType) * 8, ctx);

  const aten::COOMatrix &coo_a =
      aten::COOMatrix(4, 4, a_row, a_col, aten::NullArray(), true, false);

  const aten::COOMatrix &l_coo = COOLineGraph(coo_a, false);
  ASSERT_EQ(l_coo.num_rows, 6);
  ASSERT_EQ(l_coo.num_cols, 6);
  ASSERT_TRUE(ArrayEQ<IdType>(l_coo.row, b_row));
  ASSERT_TRUE(ArrayEQ<IdType>(l_coo.col, b_col));
  ASSERT_FALSE(l_coo.row_sorted);
  ASSERT_FALSE(l_coo.col_sorted);

  const aten::COOMatrix &l_coo2 = COOLineGraph(coo_a, true);
  ASSERT_EQ(l_coo2.num_rows, 6);
  ASSERT_EQ(l_coo2.num_cols, 6);
  ASSERT_TRUE(ArrayEQ<IdType>(l_coo2.row, c_row));
  ASSERT_TRUE(ArrayEQ<IdType>(l_coo2.col, c_col));
  ASSERT_FALSE(l_coo2.row_sorted);
  ASSERT_FALSE(l_coo2.col_sorted);

  IdArray a_data = aten::VecToIdArray(
      std::vector<IdType>({4, 5, 0, 1, 2, 3}), sizeof(IdType) * 8, ctx);
  b_row = aten::VecToIdArray(
      std::vector<IdType>({4, 5, 0, 2}), sizeof(IdType) * 8, ctx);
  b_col = aten::VecToIdArray(
      std::vector<IdType>({2, 4, 1, 5}), sizeof(IdType) * 8, ctx);
  c_row = aten::VecToIdArray(
      std::vector<IdType>({4, 4, 5, 0, 0, 1, 2, 2}), sizeof(IdType) * 8, ctx);
  c_col = aten::VecToIdArray(
      std::vector<IdType>({1, 2, 4, 1, 2, 4, 5, 0}), sizeof(IdType) * 8, ctx);
  const aten::COOMatrix &coo_ad =
      aten::COOMatrix(4, 4, a_row, a_col, a_data, true, false);
  const aten::COOMatrix &ld_coo = COOLineGraph(coo_ad, false);
  ASSERT_EQ(ld_coo.num_rows, 6);
  ASSERT_EQ(ld_coo.num_cols, 6);
  ASSERT_TRUE(ArrayEQ<IdType>(ld_coo.row, b_row));
  ASSERT_TRUE(ArrayEQ<IdType>(ld_coo.col, b_col));
  ASSERT_FALSE(ld_coo.row_sorted);
  ASSERT_FALSE(ld_coo.col_sorted);

  const aten::COOMatrix &ld_coo2 = COOLineGraph(coo_ad, true);
  ASSERT_EQ(ld_coo2.num_rows, 6);
  ASSERT_EQ(ld_coo2.num_cols, 6);
  ASSERT_TRUE(ArrayEQ<IdType>(ld_coo2.row, c_row));
  ASSERT_TRUE(ArrayEQ<IdType>(ld_coo2.col, c_col));
  ASSERT_FALSE(ld_coo2.row_sorted);
  ASSERT_FALSE(ld_coo2.col_sorted);
}

TEST(LineGraphTest, LineGraphCOO) {
  _TestLineGraphCOO<int32_t>(CPU);
  _TestLineGraphCOO<int64_t>(CPU);
}

template <typename IDX>
void _TestSort(DGLContext ctx) {
  // case 1
  IdArray a = aten::VecToIdArray(
      std::vector<IDX>({8, 6, 7, 5, 3, 0, 9}), sizeof(IDX) * 8, ctx);
  IdArray sorted_a = aten::VecToIdArray(
      std::vector<IDX>({0, 3, 5, 6, 7, 8, 9}), sizeof(IDX) * 8, ctx);
  IdArray sorted_idx =
      aten::VecToIdArray(std::vector<IDX>({5, 4, 3, 1, 2, 0, 6}), 64, ctx);

  IdArray sorted, idx;
  std::tie(sorted, idx) = aten::Sort(a);
  ASSERT_TRUE(ArrayEQ<IDX>(sorted, sorted_a));
  ASSERT_TRUE(ArrayEQ<IDX>(idx, sorted_idx));

  // case 2: empty array
  a = aten::VecToIdArray(std::vector<IDX>({}), sizeof(IDX) * 8, ctx);
  sorted_a = aten::VecToIdArray(std::vector<IDX>({}), sizeof(IDX) * 8, ctx);
  sorted_idx = aten::VecToIdArray(std::vector<IDX>({}), 64, ctx);
  std::tie(sorted, idx) = aten::Sort(a);
  ASSERT_TRUE(ArrayEQ<IDX>(sorted, sorted_a));
  ASSERT_TRUE(ArrayEQ<IDX>(idx, sorted_idx));

  // case 3: array with one element
  a = aten::VecToIdArray(std::vector<IDX>({2}), sizeof(IDX) * 8, ctx);
  sorted_a = aten::VecToIdArray(std::vector<IDX>({2}), sizeof(IDX) * 8, ctx);
  sorted_idx = aten::VecToIdArray(std::vector<IDX>({0}), 64, ctx);
  std::tie(sorted, idx) = aten::Sort(a);
  ASSERT_TRUE(ArrayEQ<IDX>(sorted, sorted_a));
  ASSERT_TRUE(ArrayEQ<IDX>(idx, sorted_idx));
}

TEST(ArrayTest, Sort) {
  _TestSort<int32_t>(CPU);
  _TestSort<int64_t>(CPU);
#ifdef DGL_USE_CUDA
  _TestSort<int32_t>(GPU);
  _TestSort<int64_t>(GPU);
#endif
}

TEST(ArrayTest, BFloatCast) {
  for (int i = -100; i < 100; ++i) {
    float a = i;
    BFloat16 b = a;
    float a_casted = b;
    ASSERT_FLOAT_EQ(a, a_casted);
  }
}
