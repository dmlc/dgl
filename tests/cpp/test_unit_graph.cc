/**
 *  Copyright (c) 2019 by Contributors
 * @file test_unit_graph.cc
 * @brief Test UnitGraph
 */
#include <dgl/array.h>
#include <dgl/immutable_graph.h>
#include <dgl/runtime/device_api.h>
#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "../../src/graph/unit_graph.h"
#include "./../src/graph/heterograph.h"
#include "./common.h"

using namespace dgl;
using namespace dgl::runtime;

template <typename IdType>
aten::CSRMatrix CSR1(DGLContext ctx) {
  /**
   * G = [[0, 0, 1],
   *      [1, 0, 1],
   *      [0, 1, 0],
   *      [1, 0, 1]]
   */
  IdArray g_indptr = aten::VecToIdArray(
      std::vector<IdType>({0, 1, 3, 4, 6}), sizeof(IdType) * 8, CTX);
  IdArray g_indices = aten::VecToIdArray(
      std::vector<IdType>({2, 0, 2, 1, 0, 2}), sizeof(IdType) * 8, CTX);

  const aten::CSRMatrix &csr_a =
      aten::CSRMatrix(4, 3, g_indptr, g_indices, aten::NullArray(), false);
  return csr_a;
}

template aten::CSRMatrix CSR1<int32_t>(DGLContext ctx);
template aten::CSRMatrix CSR1<int64_t>(DGLContext ctx);

template <typename IdType>
aten::COOMatrix COO1(DGLContext ctx) {
  /**
   * G = [[1, 1, 0],
   *      [0, 1, 0]]
   */
  IdArray g_row = aten::VecToIdArray(
      std::vector<IdType>({0, 0, 1}), sizeof(IdType) * 8, CTX);
  IdArray g_col = aten::VecToIdArray(
      std::vector<IdType>({0, 1, 1}), sizeof(IdType) * 8, CTX);
  const aten::COOMatrix &coo =
      aten::COOMatrix(2, 3, g_row, g_col, aten::NullArray(), true, true);

  return coo;
}

template aten::COOMatrix COO1<int32_t>(DGLContext ctx);
template aten::COOMatrix COO1<int64_t>(DGLContext ctx);

template <typename IdType>
void _TestUnitGraph_InOutDegrees(DGLContext ctx) {
  /**
  InDegree(s) is available only if COO or CSC formats permitted.
  OutDegree(s) is available only if COO or CSR formats permitted.
  */

  // COO
  {
    const aten::COOMatrix &coo = COO1<IdType>(ctx);
    auto &&g = CreateFromCOO(2, coo, COO_CODE);
    ASSERT_EQ(g->InDegree(0, 0), 1);
    auto &&nids = aten::Range(0, g->NumVertices(0), g->NumBits(), g->Context());
    ASSERT_TRUE(ArrayEQ<IdType>(
        g->InDegrees(0, nids),
        aten::VecToIdArray<IdType>({1, 2}, g->NumBits(), g->Context())));
    ASSERT_EQ(g->OutDegree(0, 0), 2);
    ASSERT_TRUE(ArrayEQ<IdType>(
        g->OutDegrees(0, nids),
        aten::VecToIdArray<IdType>({2, 1}, g->NumBits(), g->Context())));
  }
  // CSC
  {
    const aten::CSRMatrix &csr = CSR1<IdType>(ctx);
    auto &&g = CreateFromCSC(2, csr, CSC_CODE);
    ASSERT_EQ(g->InDegree(0, 0), 1);
    auto &&nids = aten::Range(0, g->NumVertices(0), g->NumBits(), g->Context());
    ASSERT_TRUE(ArrayEQ<IdType>(
        g->InDegrees(0, nids),
        aten::VecToIdArray<IdType>({1, 2, 1}, g->NumBits(), g->Context())));
    EXPECT_ANY_THROW(g->OutDegree(0, 0));
    EXPECT_ANY_THROW(g->OutDegrees(0, nids));
  }
  // CSR
  {
    const aten::CSRMatrix &csr = CSR1<IdType>(ctx);
    auto &&g = CreateFromCSR(2, csr, CSR_CODE);
    ASSERT_EQ(g->OutDegree(0, 0), 1);
    auto &&nids = aten::Range(0, g->NumVertices(0), g->NumBits(), g->Context());
    ASSERT_TRUE(ArrayEQ<IdType>(
        g->OutDegrees(0, nids),
        aten::VecToIdArray<IdType>({1, 2, 1, 2}, g->NumBits(), g->Context())));
    EXPECT_ANY_THROW(g->InDegree(0, 0));
    EXPECT_ANY_THROW(g->InDegrees(0, nids));
  }
}

template <typename IdType>
void _TestUnitGraph(DGLContext ctx) {
  const aten::CSRMatrix &csr = CSR1<IdType>(ctx);
  const aten::COOMatrix &coo = COO1<IdType>(ctx);

  auto g = CreateFromCSC(2, csr);
  ASSERT_EQ(g->GetCreatedFormats(), 4);

  g = CreateFromCSR(2, csr);
  ASSERT_EQ(g->GetCreatedFormats(), 2);

  g = CreateFromCOO(2, coo);
  ASSERT_EQ(g->GetCreatedFormats(), 1);

  auto src = aten::VecToIdArray<int64_t>({1, 2, 5, 3});
  auto dst = aten::VecToIdArray<int64_t>({1, 6, 2, 6});
  auto mg = dgl::UnitGraph::CreateFromCOO(2, 9, 8, src, dst, COO_CODE);
  ASSERT_EQ(mg->GetCreatedFormats(), 1);
  auto hmg = dgl::UnitGraph::CreateFromCOO(1, 8, 8, src, dst, COO_CODE);
  auto img = std::dynamic_pointer_cast<ImmutableGraph>(hmg->AsImmutableGraph());
  ASSERT_TRUE(img != nullptr);
  mg = dgl::UnitGraph::CreateFromCOO(2, 9, 8, src, dst, CSR_CODE | COO_CODE);
  ASSERT_EQ(mg->GetCreatedFormats(), 1);
  hmg = dgl::UnitGraph::CreateFromCOO(1, 8, 8, src, dst, CSR_CODE | COO_CODE);
  img = std::dynamic_pointer_cast<ImmutableGraph>(hmg->AsImmutableGraph());
  ASSERT_TRUE(img != nullptr);
  mg = dgl::UnitGraph::CreateFromCOO(2, 9, 8, src, dst, CSC_CODE | COO_CODE);
  ASSERT_EQ(mg->GetCreatedFormats(), 1);
  hmg = dgl::UnitGraph::CreateFromCOO(1, 8, 8, src, dst, CSC_CODE | COO_CODE);
  img = std::dynamic_pointer_cast<ImmutableGraph>(hmg->AsImmutableGraph());
  ASSERT_TRUE(img != nullptr);

  g = CreateFromCSC(2, csr);
  ASSERT_EQ(g->GetCreatedFormats(), 4);

  g = CreateFromCSR(2, csr);
  ASSERT_EQ(g->GetCreatedFormats(), 2);

  g = CreateFromCOO(2, coo);
  ASSERT_EQ(g->GetCreatedFormats(), 1);
}

template <typename IdType>
void _TestUnitGraph_GetInCSR(DGLContext ctx) {
  const aten::CSRMatrix &csr = CSR1<IdType>(ctx);
  const aten::COOMatrix &coo = COO1<IdType>(ctx);

  auto g = CreateFromCSC(2, csr);
  auto in_csr_matrix = g->GetCSCMatrix(0);
  ASSERT_EQ(in_csr_matrix.num_rows, csr.num_rows);
  ASSERT_EQ(in_csr_matrix.num_cols, csr.num_cols);
  ASSERT_EQ(g->GetCreatedFormats(), 4);

  // test out csr
  g = CreateFromCSR(2, csr);
  auto g_ptr = g->GetGraphInFormat(CSC_CODE);
  in_csr_matrix = g_ptr->GetCSCMatrix(0);
  ASSERT_EQ(in_csr_matrix.num_cols, csr.num_rows);
  ASSERT_EQ(in_csr_matrix.num_rows, csr.num_cols);
  ASSERT_EQ(g->GetCreatedFormats(), 2);
  in_csr_matrix = g->GetCSCMatrix(0);
  ASSERT_EQ(in_csr_matrix.num_cols, csr.num_rows);
  ASSERT_EQ(in_csr_matrix.num_rows, csr.num_cols);
  ASSERT_EQ(g->GetCreatedFormats(), 6);

  // test out coo
  g = CreateFromCOO(2, coo);
  g_ptr = g->GetGraphInFormat(CSC_CODE);
  in_csr_matrix = g_ptr->GetCSCMatrix(0);
  ASSERT_EQ(in_csr_matrix.num_cols, coo.num_rows);
  ASSERT_EQ(in_csr_matrix.num_rows, coo.num_cols);
  ASSERT_EQ(g->GetCreatedFormats(), 1);

  in_csr_matrix = g->GetCSCMatrix(0);
  ASSERT_EQ(in_csr_matrix.num_cols, coo.num_rows);
  ASSERT_EQ(in_csr_matrix.num_rows, coo.num_cols);
  ASSERT_EQ(g->GetCreatedFormats(), 5);
}

template <typename IdType>
void _TestUnitGraph_GetOutCSR(DGLContext ctx) {
  const aten::CSRMatrix &csr = CSR1<IdType>(ctx);
  const aten::COOMatrix &coo = COO1<IdType>(ctx);

  auto g = CreateFromCSC(2, csr);
  auto g_ptr = g->GetGraphInFormat(CSR_CODE);
  auto out_csr_matrix = g_ptr->GetCSRMatrix(0);
  ASSERT_EQ(out_csr_matrix.num_cols, csr.num_rows);
  ASSERT_EQ(out_csr_matrix.num_rows, csr.num_cols);
  ASSERT_EQ(g->GetCreatedFormats(), 4);
  out_csr_matrix = g->GetCSRMatrix(0);
  ASSERT_EQ(out_csr_matrix.num_cols, csr.num_rows);
  ASSERT_EQ(out_csr_matrix.num_rows, csr.num_cols);
  ASSERT_EQ(g->GetCreatedFormats(), 6);

  // test out csr
  g = CreateFromCSR(2, csr);
  out_csr_matrix = g->GetCSRMatrix(0);
  ASSERT_EQ(out_csr_matrix.num_rows, csr.num_rows);
  ASSERT_EQ(out_csr_matrix.num_cols, csr.num_cols);
  ASSERT_EQ(g->GetCreatedFormats(), 2);

  // test out coo
  g = CreateFromCOO(2, coo);
  g_ptr = g->GetGraphInFormat(CSR_CODE);
  out_csr_matrix = g_ptr->GetCSRMatrix(0);
  ASSERT_EQ(out_csr_matrix.num_rows, coo.num_rows);
  ASSERT_EQ(out_csr_matrix.num_cols, coo.num_cols);
  ASSERT_EQ(g->GetCreatedFormats(), 1);

  out_csr_matrix = g->GetCSRMatrix(0);
  ASSERT_EQ(out_csr_matrix.num_rows, coo.num_rows);
  ASSERT_EQ(out_csr_matrix.num_cols, coo.num_cols);
  ASSERT_EQ(g->GetCreatedFormats(), 3);
}

template <typename IdType>
void _TestUnitGraph_GetCOO(DGLContext ctx) {
  const aten::CSRMatrix &csr = CSR1<IdType>(ctx);
  const aten::COOMatrix &coo = COO1<IdType>(ctx);

  auto g = CreateFromCSC(2, csr);
  auto g_ptr = g->GetGraphInFormat(COO_CODE);
  auto out_coo_matrix = g_ptr->GetCOOMatrix(0);
  ASSERT_EQ(out_coo_matrix.num_cols, csr.num_rows);
  ASSERT_EQ(out_coo_matrix.num_rows, csr.num_cols);
  ASSERT_EQ(g->GetCreatedFormats(), 4);
  out_coo_matrix = g->GetCOOMatrix(0);
  ASSERT_EQ(out_coo_matrix.num_cols, csr.num_rows);
  ASSERT_EQ(out_coo_matrix.num_rows, csr.num_cols);
  ASSERT_EQ(g->GetCreatedFormats(), 5);

  // test out csr
  g = CreateFromCSR(2, csr);
  g_ptr = g->GetGraphInFormat(COO_CODE);
  out_coo_matrix = g_ptr->GetCOOMatrix(0);
  ASSERT_EQ(out_coo_matrix.num_rows, csr.num_rows);
  ASSERT_EQ(out_coo_matrix.num_cols, csr.num_cols);
  ASSERT_EQ(g->GetCreatedFormats(), 2);
  out_coo_matrix = g->GetCOOMatrix(0);
  ASSERT_EQ(out_coo_matrix.num_rows, csr.num_rows);
  ASSERT_EQ(out_coo_matrix.num_cols, csr.num_cols);
  ASSERT_EQ(g->GetCreatedFormats(), 3);

  // test out coo
  g = CreateFromCOO(2, coo);
  out_coo_matrix = g->GetCOOMatrix(0);
  ASSERT_EQ(out_coo_matrix.num_rows, coo.num_rows);
  ASSERT_EQ(out_coo_matrix.num_cols, coo.num_cols);
  ASSERT_EQ(g->GetCreatedFormats(), 1);
}

template <typename IdType>
void _TestUnitGraph_Reserve(DGLContext ctx) {
  const aten::CSRMatrix &csr = CSR1<IdType>(ctx);
  const aten::COOMatrix &coo = COO1<IdType>(ctx);

  auto g = CreateFromCSC(2, csr);
  ASSERT_EQ(g->GetCreatedFormats(), 4);
  auto r_g =
      std::dynamic_pointer_cast<UnitGraph>(g->GetRelationGraph(0))->Reverse();
  ASSERT_EQ(r_g->GetCreatedFormats(), 2);
  aten::CSRMatrix g_in_csr = g->GetCSCMatrix(0);
  aten::CSRMatrix r_g_out_csr = r_g->GetCSRMatrix(0);
  ASSERT_TRUE(g_in_csr.indptr->data == r_g_out_csr.indptr->data);
  ASSERT_TRUE(g_in_csr.indices->data == r_g_out_csr.indices->data);
  aten::CSRMatrix g_out_csr = g->GetCSRMatrix(0);
  ASSERT_EQ(g->GetCreatedFormats(), 6);
  ASSERT_EQ(r_g->GetCreatedFormats(), 6);
  aten::CSRMatrix r_g_in_csr = r_g->GetCSCMatrix(0);
  ASSERT_TRUE(g_out_csr.indptr->data == r_g_in_csr.indptr->data);
  ASSERT_TRUE(g_out_csr.indices->data == r_g_in_csr.indices->data);
  aten::COOMatrix g_coo = g->GetCOOMatrix(0);
  ASSERT_EQ(g->GetCreatedFormats(), 7);
  ASSERT_EQ(r_g->GetCreatedFormats(), 6);
  aten::COOMatrix r_g_coo = r_g->GetCOOMatrix(0);
  ASSERT_EQ(r_g->GetCreatedFormats(), 7);
  ASSERT_EQ(g_coo.num_rows, r_g_coo.num_cols);
  ASSERT_EQ(g_coo.num_cols, r_g_coo.num_rows);
  ASSERT_TRUE(ArrayEQ<IdType>(g_coo.row, r_g_coo.col));
  ASSERT_TRUE(ArrayEQ<IdType>(g_coo.col, r_g_coo.row));

  // test out csr
  g = CreateFromCSR(2, csr);
  ASSERT_EQ(g->GetCreatedFormats(), 2);
  r_g = std::dynamic_pointer_cast<UnitGraph>(g->GetRelationGraph(0))->Reverse();
  ASSERT_EQ(r_g->GetCreatedFormats(), 4);
  g_out_csr = g->GetCSRMatrix(0);
  r_g_in_csr = r_g->GetCSCMatrix(0);
  ASSERT_TRUE(g_out_csr.indptr->data == r_g_in_csr.indptr->data);
  ASSERT_TRUE(g_out_csr.indices->data == r_g_in_csr.indices->data);
  g_in_csr = g->GetCSCMatrix(0);
  ASSERT_EQ(g->GetCreatedFormats(), 6);
  ASSERT_EQ(r_g->GetCreatedFormats(), 6);
  r_g_out_csr = r_g->GetCSRMatrix(0);
  ASSERT_TRUE(g_in_csr.indptr->data == r_g_out_csr.indptr->data);
  ASSERT_TRUE(g_in_csr.indices->data == r_g_out_csr.indices->data);
  g_coo = g->GetCOOMatrix(0);
  ASSERT_EQ(g->GetCreatedFormats(), 7);
  ASSERT_EQ(r_g->GetCreatedFormats(), 6);
  r_g_coo = r_g->GetCOOMatrix(0);
  ASSERT_EQ(r_g->GetCreatedFormats(), 7);
  ASSERT_EQ(g_coo.num_rows, r_g_coo.num_cols);
  ASSERT_EQ(g_coo.num_cols, r_g_coo.num_rows);
  ASSERT_TRUE(ArrayEQ<IdType>(g_coo.row, r_g_coo.col));
  ASSERT_TRUE(ArrayEQ<IdType>(g_coo.col, r_g_coo.row));

  // test out coo
  g = CreateFromCOO(2, coo);
  ASSERT_EQ(g->GetCreatedFormats(), 1);
  r_g = std::dynamic_pointer_cast<UnitGraph>(g->GetRelationGraph(0))->Reverse();
  ASSERT_EQ(r_g->GetCreatedFormats(), 1);
  g_coo = g->GetCOOMatrix(0);
  r_g_coo = r_g->GetCOOMatrix(0);
  ASSERT_EQ(g_coo.num_rows, r_g_coo.num_cols);
  ASSERT_EQ(g_coo.num_cols, r_g_coo.num_rows);
  ASSERT_TRUE(g_coo.row->data == r_g_coo.col->data);
  ASSERT_TRUE(g_coo.col->data == r_g_coo.row->data);
  g_in_csr = g->GetCSCMatrix(0);
  ASSERT_EQ(g->GetCreatedFormats(), 5);
  ASSERT_EQ(r_g->GetCreatedFormats(), 3);
  r_g_out_csr = r_g->GetCSRMatrix(0);
  ASSERT_TRUE(g_in_csr.indptr->data == r_g_out_csr.indptr->data);
  ASSERT_TRUE(g_in_csr.indices->data == r_g_out_csr.indices->data);
  g_out_csr = g->GetCSRMatrix(0);
  ASSERT_EQ(g->GetCreatedFormats(), 7);
  ASSERT_EQ(r_g->GetCreatedFormats(), 7);
  r_g_in_csr = r_g->GetCSCMatrix(0);
  ASSERT_TRUE(g_out_csr.indptr->data == r_g_in_csr.indptr->data);
  ASSERT_TRUE(g_out_csr.indices->data == r_g_in_csr.indices->data);
}

template <typename IdType>
void _TestUnitGraph_CopyTo(
    const DGLContext &src_ctx, const DGLContext &dst_ctx) {
  const aten::CSRMatrix &csr = CSR1<IdType>(src_ctx);
  const aten::COOMatrix &coo = COO1<IdType>(src_ctx);

  auto device = dgl::runtime::DeviceAPI::Get(dst_ctx);
  // We don't allow SetStream in DGL for now.
  auto stream = nullptr;

  auto g = dgl::UnitGraph::CreateFromCSC(2, csr);
  ASSERT_EQ(g->GetCreatedFormats(), 4);
  auto cg = dgl::UnitGraph::CopyTo(g, dst_ctx);
  device->StreamSync(dst_ctx, stream);
  ASSERT_EQ(cg->GetCreatedFormats(), 4);

  g = dgl::UnitGraph::CreateFromCSR(2, csr);
  ASSERT_EQ(g->GetCreatedFormats(), 2);
  cg = dgl::UnitGraph::CopyTo(g, dst_ctx);
  device->StreamSync(dst_ctx, stream);
  ASSERT_EQ(cg->GetCreatedFormats(), 2);

  g = dgl::UnitGraph::CreateFromCOO(2, coo);
  ASSERT_EQ(g->GetCreatedFormats(), 1);
  cg = dgl::UnitGraph::CopyTo(g, dst_ctx);
  device->StreamSync(dst_ctx, stream);
  ASSERT_EQ(cg->GetCreatedFormats(), 1);
}

TEST(UniGraphTest, TestUnitGraph_CopyTo) {
  _TestUnitGraph_CopyTo<int32_t>(CPU, CPU);
  _TestUnitGraph_CopyTo<int64_t>(CPU, CPU);
#ifdef DGL_USE_CUDA
  _TestUnitGraph_CopyTo<int32_t>(CPU, GPU);
  _TestUnitGraph_CopyTo<int32_t>(GPU, GPU);
  _TestUnitGraph_CopyTo<int32_t>(GPU, CPU);
  _TestUnitGraph_CopyTo<int64_t>(CPU, GPU);
  _TestUnitGraph_CopyTo<int64_t>(GPU, GPU);
  _TestUnitGraph_CopyTo<int64_t>(GPU, CPU);
#endif
}

TEST(UniGraphTest, TestUnitGraph_InOutDegrees) {
  _TestUnitGraph_InOutDegrees<int32_t>(CPU);
  _TestUnitGraph_InOutDegrees<int64_t>(CPU);
#ifdef DGL_USE_CUDA
  _TestUnitGraph_InOutDegrees<int32_t>(GPU);
  _TestUnitGraph_InOutDegrees<int64_t>(GPU);
#endif
}

TEST(UniGraphTest, TestUnitGraph_Create) {
  _TestUnitGraph<int32_t>(CPU);
  _TestUnitGraph<int64_t>(CPU);
#ifdef DGL_USE_CUDA
  _TestUnitGraph<int32_t>(GPU);
  _TestUnitGraph<int64_t>(GPU);
#endif
}

TEST(UniGraphTest, TestUnitGraph_GetInCSR) {
  _TestUnitGraph_GetInCSR<int32_t>(CPU);
  _TestUnitGraph_GetInCSR<int64_t>(CPU);
#ifdef DGL_USE_CUDA
  _TestUnitGraph_GetInCSR<int32_t>(GPU);
  _TestUnitGraph_GetInCSR<int64_t>(GPU);
#endif
}

TEST(UniGraphTest, TestUnitGraph_GetOutCSR) {
  _TestUnitGraph_GetOutCSR<int32_t>(CPU);
  _TestUnitGraph_GetOutCSR<int64_t>(CPU);
#ifdef DGL_USE_CUDA
  _TestUnitGraph_GetOutCSR<int32_t>(GPU);
  _TestUnitGraph_GetOutCSR<int64_t>(GPU);
#endif
}

TEST(UniGraphTest, TestUnitGraph_GetCOO) {
  _TestUnitGraph_GetCOO<int32_t>(CPU);
  _TestUnitGraph_GetCOO<int64_t>(CPU);
#ifdef DGL_USE_CUDA
  _TestUnitGraph_GetCOO<int32_t>(GPU);
  _TestUnitGraph_GetCOO<int64_t>(GPU);
#endif
}

TEST(UniGraphTest, TestUnitGraph_Reserve) {
  _TestUnitGraph_Reserve<int32_t>(CPU);
  _TestUnitGraph_Reserve<int64_t>(CPU);
#ifdef DGL_USE_CUDA
  _TestUnitGraph_Reserve<int32_t>(GPU);
  _TestUnitGraph_Reserve<int64_t>(GPU);
#endif
}
