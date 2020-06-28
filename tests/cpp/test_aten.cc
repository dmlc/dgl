#include <gtest/gtest.h>
#include <dgl/array.h>
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

void _TestRange(DLContext ctx) {
  IdArray a = aten::Range(10, 10, 64, ctx);
  ASSERT_EQ(Len(a), 0);
  a = aten::Range(10, 20, 32, ctx);
  ASSERT_EQ(Len(a), 10);
  ASSERT_EQ(a->dtype.bits, 32);
  a = a.CopyTo(CPU);
  for (int i = 0; i < 10; ++i)
    ASSERT_EQ(Ptr<int32_t>(a)[i], i + 10);
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
  for (int i = 0; i < 13; ++i)
    ASSERT_EQ(Ptr<int64_t>(a)[i], -100);
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

void _TestNumBits(DLContext ctx) {
  IdArray a = aten::Range(0, 10, 32, ctx);
  a = aten::AsNumBits(a, 64);
  ASSERT_EQ(a->dtype.bits, 64);
  a = a.CopyTo(CPU);
  for (int i = 0; i < 10; ++i)
    ASSERT_EQ(PI64(a)[i], i);
}

TEST(ArrayTest, TestAsNumBits) {
  _TestNumBits(CPU);
#ifdef DGL_USE_CUDA
  _TestNumBits(GPU);
#endif
};

template <typename IDX>
void _TestArith(DLContext ctx) {
  const int N = 100;
  IdArray a = aten::Full(-10, N, sizeof(IDX)*8, ctx);
  IdArray b = aten::Full(7, N, sizeof(IDX)*8, ctx);

  IdArray c = a + b;
  c = c.CopyTo(CPU);
  for (int i = 0; i < N; ++i)
    ASSERT_EQ(Ptr<IDX>(c)[i], -3);
  c = a - b;
  c = c.CopyTo(CPU);
  for (int i = 0; i < N; ++i)
    ASSERT_EQ(Ptr<IDX>(c)[i], -17);
  c = a * b;
  c = c.CopyTo(CPU);
  for (int i = 0; i < N; ++i)
    ASSERT_EQ(Ptr<IDX>(c)[i], -70);
  c = a / b;
  c = c.CopyTo(CPU);
  for (int i = 0; i < N; ++i)
    ASSERT_EQ(Ptr<IDX>(c)[i], -1);
  c = -a;
  c = c.CopyTo(CPU);
  for (int i = 0; i < N; ++i)
    ASSERT_EQ(Ptr<IDX>(c)[i], 10);

  const int val = -3;
  c = aten::Add(a, val);
  c = c.CopyTo(CPU);
  for (int i = 0; i < N; ++i)
    ASSERT_EQ(Ptr<IDX>(c)[i], -13);
  c = aten::Sub(a, val);
  c = c.CopyTo(CPU);
  for (int i = 0; i < N; ++i)
    ASSERT_EQ(Ptr<IDX>(c)[i], -7);
  c = aten::Mul(a, val);
  c = c.CopyTo(CPU);
  for (int i = 0; i < N; ++i)
    ASSERT_EQ(Ptr<IDX>(c)[i], 30);
  c = aten::Div(a, val);
  c = c.CopyTo(CPU);
  for (int i = 0; i < N; ++i)
    ASSERT_EQ(Ptr<IDX>(c)[i], 3);
  c = aten::Add(val, b);
  c = c.CopyTo(CPU);
  for (int i = 0; i < N; ++i)
    ASSERT_EQ(Ptr<IDX>(c)[i], 4);
  c = aten::Sub(val, b);
  c = c.CopyTo(CPU);
  for (int i = 0; i < N; ++i)
    ASSERT_EQ(Ptr<IDX>(c)[i], -10);
  c = aten::Mul(val, b);
  c = c.CopyTo(CPU);
  for (int i = 0; i < N; ++i)
    ASSERT_EQ(Ptr<IDX>(c)[i], -21);
  c = aten::Div(val, b);
  c = c.CopyTo(CPU);
  for (int i = 0; i < N; ++i)
    ASSERT_EQ(Ptr<IDX>(c)[i], 0);

  a = aten::Range(0, N, sizeof(IDX)*8, ctx);
  c = a < 50;
  c = c.CopyTo(CPU);
  for (int i = 0; i < N; ++i)
    ASSERT_EQ(Ptr<IDX>(c)[i], (int)(i < 50));

  c = a > 50;
  c = c.CopyTo(CPU);
  for (int i = 0; i < N; ++i)
    ASSERT_EQ(Ptr<IDX>(c)[i], (int)(i > 50));

  c = a >= 50;
  c = c.CopyTo(CPU);
  for (int i = 0; i < N; ++i)
    ASSERT_EQ(Ptr<IDX>(c)[i], (int)(i >= 50));

  c = a <= 50;
  c = c.CopyTo(CPU);
  for (int i = 0; i < N; ++i)
    ASSERT_EQ(Ptr<IDX>(c)[i], (int)(i <= 50));

  c = a == 50;
  c = c.CopyTo(CPU);
  for (int i = 0; i < N; ++i)
    ASSERT_EQ(Ptr<IDX>(c)[i], (int)(i == 50));

  c = a != 50;
  c = c.CopyTo(CPU);
  for (int i = 0; i < N; ++i)
    ASSERT_EQ(Ptr<IDX>(c)[i], (int)(i != 50));

}

TEST(ArrayTest, TestArith) {
  _TestArith<int32_t>(CPU);
  _TestArith<int64_t>(CPU);
#ifdef DGL_USE_CUDA
  _TestArith<int32_t>(GPU);
  _TestArith<int64_t>(GPU);
#endif
};

template <typename IDX>
void _TestHStack() {
  IdArray a = aten::Range(0, 100, sizeof(IDX)*8, CTX);
  IdArray b = aten::Range(100, 200, sizeof(IDX)*8, CTX);
  IdArray c = aten::HStack(a, b);
  ASSERT_EQ(c->ndim, 1);
  ASSERT_EQ(c->shape[0], 200);
  for (int i = 0; i < 200; ++i)
    ASSERT_EQ(Ptr<IDX>(c)[i], i);
}

TEST(ArrayTest, TestHStack) {
  _TestHStack<int32_t>();
  _TestHStack<int64_t>();
}

template <typename IDX>
void _TestIndexSelect(DLContext ctx) {
  IdArray a = aten::Range(0, 100, sizeof(IDX)*8, ctx);
  ASSERT_EQ(aten::IndexSelect<int>(a, 50), 50);
  ASSERT_TRUE(ArrayEQ<IDX>(aten::IndexSelect(a, 10, 20),
        aten::Range(10, 20, sizeof(IDX)*8, ctx)));
  IdArray b = aten::VecToIdArray(std::vector<IDX>({0, 20, 10}), sizeof(IDX)*8, ctx);
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
void _TestRelabel_() {
  IdArray a = aten::VecToIdArray(std::vector<IDX>({0, 20, 10}), sizeof(IDX)*8, CTX);
  IdArray b = aten::VecToIdArray(std::vector<IDX>({20, 5, 6}), sizeof(IDX)*8, CTX);
  IdArray c = aten::Relabel_({a, b});

  IdArray ta = aten::VecToIdArray(std::vector<IDX>({0, 1, 2}), sizeof(IDX)*8, CTX);
  IdArray tb = aten::VecToIdArray(std::vector<IDX>({1, 3, 4}), sizeof(IDX)*8, CTX);
  IdArray tc = aten::VecToIdArray(std::vector<IDX>({0, 20, 10, 5, 6}), sizeof(IDX)*8, CTX);

  ASSERT_TRUE(ArrayEQ<IDX>(a, ta));
  ASSERT_TRUE(ArrayEQ<IDX>(b, tb));
  ASSERT_TRUE(ArrayEQ<IDX>(c, tc));
}

TEST(ArrayTest, TestRelabel_) {
  _TestRelabel_<int32_t>();
  _TestRelabel_<int64_t>();
}

template <typename IDX>
void _TestConcat(DLContext ctx) {
  IdArray a = aten::VecToIdArray(std::vector<IDX>({1, 2, 3}), sizeof(IDX)*8, CTX);
  IdArray b = aten::VecToIdArray(std::vector<IDX>({4, 5, 6}), sizeof(IDX)*8, CTX);
  IdArray tc = aten::VecToIdArray(std::vector<IDX>({1, 2, 3, 4, 5, 6}), sizeof(IDX)*8, CTX);
  IdArray c = aten::Concat(std::vector<IdArray>{a, b});
  ASSERT_TRUE(ArrayEQ<IDX>(c, tc));
  IdArray d = aten::Concat(std::vector<IdArray>{a, b, c});
  IdArray td = aten::VecToIdArray(std::vector<IDX>({1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6}),
                                  sizeof(IDX)*8, CTX);
  ASSERT_TRUE(ArrayEQ<IDX>(d, td));
}

TEST(ArrayTest, TestConcat) {
  _TestConcat<int32_t>(CPU);
  _TestConcat<int64_t>(CPU);
  _TestConcat<float>(CPU);
  _TestConcat<double>(CPU);
#ifdef DGL_USE_CUDA
  _TestConcat<int32_t>(GPU);
  _TestConcat<int64_t>(GPU);
  _TestConcat<float>(GPU);
  _TestConcat<double>(GPU);
#endif
}


template <typename IdType>
void _TestDisjointUnionPartitionCoo(DLContext ctx) {
  /*
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
  IdArray a_row =
    aten::VecToIdArray(std::vector<IdType>({0, 1, 1, 2}), sizeof(IdType)*8, CTX);
  IdArray a_col =
    aten::VecToIdArray(std::vector<IdType>({2, 0, 2, 1}), sizeof(IdType)*8, CTX);
  IdArray b_row =
    aten::VecToIdArray(std::vector<IdType>({0, 0, 1}), sizeof(IdType)*8, CTX);
  IdArray b_col =
    aten::VecToIdArray(std::vector<IdType>({0, 1, 1}), sizeof(IdType)*8, CTX);
  IdArray b_data =
    aten::VecToIdArray(std::vector<IdType>({2, 0, 1}), sizeof(IdType)*8, CTX);
  IdArray c_row =
    aten::VecToIdArray(std::vector<IdType>({0}), sizeof(IdType)*8, CTX);
  IdArray c_col =
    aten::VecToIdArray(std::vector<IdType>({0}), sizeof(IdType)*8, CTX);
  IdArray ab_row =
    aten::VecToIdArray(std::vector<IdType>({0, 1, 1, 2, 3, 3, 4}), sizeof(IdType)*8, CTX);
  IdArray ab_col =
    aten::VecToIdArray(std::vector<IdType>({2, 0, 2, 1, 3, 4, 4}), sizeof(IdType)*8, CTX);
  IdArray ab_data =
    aten::VecToIdArray(std::vector<IdType>({0, 1, 2, 3, 6, 4, 5}), sizeof(IdType)*8, CTX);
  IdArray abc_row =
    aten::VecToIdArray(std::vector<IdType>({0, 1, 1, 2, 3, 3, 4, 5}), sizeof(IdType)*8, CTX);
  IdArray abc_col =
    aten::VecToIdArray(std::vector<IdType>({2, 0, 2, 1, 3, 4, 4, 6}), sizeof(IdType)*8, CTX);
  IdArray abc_data =
    aten::VecToIdArray(std::vector<IdType>({0, 1, 2, 3, 6, 4, 5, 7}), sizeof(IdType)*8, CTX);
  const aten::COOMatrix &coo_a = aten::COOMatrix(
    3,
    3,
    a_row,
    a_col,
    aten::NullArray(),
    true,
    false);
  const aten::COOMatrix &coo_b = aten::COOMatrix(
    2,
    3,
    b_row,
    b_col,
    b_data,
    true,
    true);
  const aten::COOMatrix &coo_c = aten::COOMatrix(
    1,
    1,
    c_row,
    c_col,
    aten::NullArray(),
    true,
    true);

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
  const std::vector<aten::COOMatrix> &p_coos = aten::DisjointPartitionCooBySizes(
    coo_ab,
    2,
    edge_cumsum,
    src_vertex_cumsum,
    dst_vertex_cumsum);
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
  const std::vector<aten::COOMatrix> &p_coos_abc = aten::DisjointPartitionCooBySizes(
    coo_abc,
    3,
    edge_cumsum_abc,
    src_vertex_cumsum_abc,
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
void _TestDisjointUnionPartitionCsr(DLContext ctx) {
  /*
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
  IdArray a_indptr =
    aten::VecToIdArray(std::vector<IdType>({0, 1, 3, 4}), sizeof(IdType)*8, CTX);
  IdArray a_indices =
    aten::VecToIdArray(std::vector<IdType>({2, 0, 2, 1}), sizeof(IdType)*8, CTX);
  IdArray b_indptr =
    aten::VecToIdArray(std::vector<IdType>({0, 2, 3}), sizeof(IdType)*8, CTX);
  IdArray b_indices =
    aten::VecToIdArray(std::vector<IdType>({0, 1, 1}), sizeof(IdType)*8, CTX);
  IdArray b_data =
    aten::VecToIdArray(std::vector<IdType>({2, 0, 1}), sizeof(IdType)*8, CTX);
  IdArray c_indptr =
    aten::VecToIdArray(std::vector<IdType>({0, 1}), sizeof(IdType)*8, CTX);
  IdArray c_indices =
    aten::VecToIdArray(std::vector<IdType>({0}), sizeof(IdType)*8, CTX);
  IdArray bc_indptr =
    aten::VecToIdArray(std::vector<IdType>({0, 2, 3, 4}), sizeof(IdType)*8, CTX);
  IdArray bc_indices =
    aten::VecToIdArray(std::vector<IdType>({0, 1, 1, 3}), sizeof(IdType)*8, CTX);
  IdArray bc_data =
    aten::VecToIdArray(std::vector<IdType>({2, 0, 1, 3}), sizeof(IdType)*8, CTX);
  IdArray abc_indptr =
    aten::VecToIdArray(std::vector<IdType>({0, 1, 3, 4, 6, 7, 8}), sizeof(IdType)*8, CTX);
  IdArray abc_indices =
    aten::VecToIdArray(std::vector<IdType>({2, 0, 2, 1, 3, 4, 4, 6}), sizeof(IdType)*8, CTX);
  IdArray abc_data =
    aten::VecToIdArray(std::vector<IdType>({0, 1, 2, 3, 6, 4, 5, 7}), sizeof(IdType)*8, CTX);
  const aten::CSRMatrix &csr_a = aten::CSRMatrix(
    3,
    3,
    a_indptr,
    a_indices,
    aten::NullArray(),
    false);
  const aten::CSRMatrix &csr_b = aten::CSRMatrix(
    2,
    3,
    b_indptr,
    b_indices,
    b_data,
    true);
  const aten::CSRMatrix &csr_c = aten::CSRMatrix(
    1,
    1,
    c_indptr,
    c_indices,
    aten::NullArray(),
    true);

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
  const std::vector<aten::CSRMatrix> &p_csrs = aten::DisjointPartitionCsrBySizes(
    csr_bc,
    2,
    edge_cumsum,
    src_vertex_cumsum,
    dst_vertex_cumsum);
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
  const std::vector<aten::CSRMatrix> &p_csrs_abc = aten::DisjointPartitionCsrBySizes(
    csr_abc,
    3,
    edge_cumsum_abc,
    src_vertex_cumsum_abc,
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

template <typename IDX>
void _TestCumSum(DLContext ctx) {
  IdArray a = aten::VecToIdArray(std::vector<IDX>({8, 6, 7, 5, 3, 0, 9}),
      sizeof(IDX)*8, ctx);
  {
    IdArray tb = aten::VecToIdArray(std::vector<IDX>({8, 14, 21, 26, 29, 29, 38}),
        sizeof(IDX)*8, ctx);
    IdArray b = aten::CumSum(a);
    ASSERT_TRUE(ArrayEQ<IDX>(b, tb));
  }
  {
    IdArray tb = aten::VecToIdArray(std::vector<IDX>({0, 8, 14, 21, 26, 29, 29, 38}),
        sizeof(IDX)*8, ctx);
    IdArray b = aten::CumSum(a, true);
    ASSERT_TRUE(ArrayEQ<IDX>(b, tb));
  }
  a = aten::VecToIdArray(std::vector<IDX>({}), sizeof(IDX)*8, ctx);
  {
    IdArray tb = aten::VecToIdArray(std::vector<IDX>({}), sizeof(IDX)*8, ctx);
    IdArray b = aten::CumSum(a);
    ASSERT_TRUE(ArrayEQ<IDX>(b, tb));
  }
  {
    IdArray tb = aten::VecToIdArray(std::vector<IDX>({}), sizeof(IDX)*8, ctx);
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
