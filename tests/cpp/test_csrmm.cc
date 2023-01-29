#include <dgl/array.h>
#include <dgl/kernel.h>
#include <gtest/gtest.h>

#include "../../src/array/cpu/array_utils.h"  // PairHash
#include "./common.h"

using namespace dgl;
using namespace dgl::runtime;

namespace {

// Unit tests:
// CSRMM(A, B) == A_mm_B
// CSRSum({A, C}) == A_plus_C
// CSRMask(A, C) = A_mask_C

template <typename IdType, typename DType>
std::unordered_map<std::pair<IdType, IdType>, DType, aten::PairHash> COOToMap(
    aten::COOMatrix coo, NDArray weights) {
  std::unordered_map<std::pair<IdType, IdType>, DType, aten::PairHash> map;

  for (int64_t i = 0; i < coo.row->shape[0]; ++i) {
    IdType irow = aten::IndexSelect<IdType>(coo.row, i);
    IdType icol = aten::IndexSelect<IdType>(coo.col, i);
    IdType ieid =
        aten::COOHasData(coo) ? aten::IndexSelect<IdType>(coo.data, i) : i;
    DType idata = aten::IndexSelect<DType>(weights, ieid);
    map.insert({{irow, icol}, idata});
  }
  return map;
}

template <typename IdType, typename DType>
bool CSRIsClose(
    aten::CSRMatrix A, aten::CSRMatrix B, NDArray A_weights, NDArray B_weights,
    DType rtol, DType atol) {
  auto Amap = COOToMap<IdType, DType>(CSRToCOO(A, false), A_weights);
  auto Bmap = COOToMap<IdType, DType>(CSRToCOO(B, false), B_weights);

  if (Amap.size() != Bmap.size()) return false;

  for (auto itA : Amap) {
    auto itB = Bmap.find(itA.first);
    if (itB == Bmap.end()) return false;
    if (fabs(itA.second - itB->second) >= rtol * fabs(itA.second) + atol)
      return false;
  }

  return true;
}

template <typename IdType, typename DType>
std::pair<aten::CSRMatrix, NDArray> CSR_A(DGLContext ctx = CTX) {
  // matrix([[0. , 0. , 1. , 0.7, 0. ],
  //         [0. , 0. , 0.5, 0.+, 0. ],
  //         [0.4, 0.7, 0. , 0.2, 0. ],
  //         [0. , 0. , 0. , 0. , 0.2]])
  // (0.+ indicates that the entry exists but the value is 0.)
  auto csr = aten::CSRMatrix(
      4, 5, NDArray::FromVector(std::vector<IdType>({0, 2, 4, 7, 8}), ctx),
      NDArray::FromVector(std::vector<IdType>({2, 3, 2, 3, 0, 1, 3, 4}), ctx),
      NDArray::FromVector(std::vector<IdType>({1, 0, 2, 3, 4, 5, 6, 7}), ctx));
  auto weights = NDArray::FromVector(
      std::vector<DType>({0.7, 1.0, 0.5, 0.0, 0.4, 0.7, 0.2, 0.2}), ctx);
  return {csr, weights};
}

template <typename IdType, typename DType>
std::pair<aten::CSRMatrix, NDArray> CSR_B(DGLContext ctx = CTX) {
  // matrix([[0. , 0.9, 0. , 0.6, 0. , 0.3],
  //         [0. , 0. , 0. , 0. , 0. , 0.4],
  //         [0.+, 0. , 0. , 0. , 0. , 0.9],
  //         [0.8, 0.2, 0.3, 0.2, 0. , 0. ],
  //         [0.2, 0.4, 0. , 0. , 0. , 0. ]])
  // (0.+ indicates that the entry exists but the value is 0.)
  auto csr = aten::CSRMatrix(
      5, 6, NDArray::FromVector(std::vector<IdType>({0, 3, 4, 6, 10, 12}), ctx),
      NDArray::FromVector(
          std::vector<IdType>({1, 3, 5, 5, 0, 5, 0, 1, 2, 3, 0, 1}), ctx));
  auto weights = NDArray::FromVector(
      std::vector<DType>(
          {0.9, 0.6, 0.3, 0.4, 0.0, 0.9, 0.8, 0.2, 0.3, 0.2, 0.2, 0.4}),
      ctx);
  return {csr, weights};
}

template <typename IdType, typename DType>
std::pair<aten::CSRMatrix, NDArray> CSR_C(DGLContext ctx = CTX) {
  // matrix([[0. , 0. , 0. , 0.2, 0. ],
  //         [0. , 0. , 0. , 0.5, 0.4],
  //         [0. , 0.2, 0. , 0.9, 0.2],
  //         [0. , 1. , 0. , 0.7, 0. ]])
  auto csr = aten::CSRMatrix(
      4, 5, NDArray::FromVector(std::vector<IdType>({0, 1, 3, 6, 8}), ctx),
      NDArray::FromVector(std::vector<IdType>({3, 3, 4, 1, 3, 4, 1, 3}), ctx));
  auto weights = NDArray::FromVector(
      std::vector<DType>({0.2, 0.5, 0.4, 0.2, 0.9, 0.2, 1., 0.7}), ctx);
  return {csr, weights};
}

template <typename IdType, typename DType>
std::pair<aten::CSRMatrix, NDArray> CSR_A_mm_B(DGLContext ctx = CTX) {
  // matrix([[0.56, 0.14, 0.21, 0.14, 0.  , 0.9 ],
  //         [0.+ , 0.+ , 0.+ , 0.+ , 0.  , 0.45],
  //         [0.16, 0.4 , 0.06, 0.28, 0.  , 0.4 ],
  //         [0.04, 0.08, 0.  , 0.  , 0.  , 0.  ]])
  // (0.+ indicates that the entry exists but the value is 0.)
  auto csr = aten::CSRMatrix(
      4, 6, NDArray::FromVector(std::vector<IdType>({0, 5, 10, 15, 17}), ctx),
      NDArray::FromVector(
          std::vector<IdType>(
              {0, 1, 2, 3, 5, 0, 1, 2, 3, 5, 0, 1, 2, 3, 5, 0, 1}),
          ctx));
  auto weights = NDArray::FromVector(
      std::vector<DType>(
          {0.56, 0.14, 0.21, 0.14, 0.9, 0., 0., 0., 0., 0.45, 0.16, 0.4, 0.06,
           0.28, 0.4, 0.04, 0.08}),
      ctx);
  return {csr, weights};
}

template <typename IdType, typename DType>
std::pair<aten::CSRMatrix, NDArray> CSR_A_plus_C(DGLContext ctx = CTX) {
  auto csr = aten::CSRMatrix(
      4, 5, NDArray::FromVector(std::vector<IdType>({0, 2, 5, 9, 12}), ctx),
      NDArray::FromVector(
          std::vector<IdType>({2, 3, 2, 3, 4, 0, 1, 3, 4, 1, 3, 4}), ctx));
  auto weights = NDArray::FromVector(
      std::vector<DType>(
          {1., 0.9, 0.5, 0.5, 0.4, 0.4, 0.9, 1.1, 0.2, 1., 0.7, 0.2}),
      ctx);
  return {csr, weights};
}

template <typename DType>
NDArray CSR_A_mask_C(DGLContext ctx = CTX) {
  return NDArray::FromVector(
      std::vector<DType>({0.7, 0.0, 0.0, 0.7, 0.2, 0.0, 0.0, 0.0}), ctx);
}

template <typename IdType, typename DType>
void _TestCsrmm(DGLContext ctx = CTX) {
  auto A = CSR_A<IdType, DType>(ctx);
  auto B = CSR_B<IdType, DType>(ctx);
  auto A_mm_B = aten::CSRMM(A.first, A.second, B.first, B.second);
  auto A_mm_B2 = CSR_A_mm_B<IdType, DType>(ctx);
  bool result = CSRIsClose<IdType, DType>(
      A_mm_B.first, A_mm_B2.first, A_mm_B.second, A_mm_B2.second, 1e-4, 1e-4);
  ASSERT_TRUE(result);
}

template <typename IdType, typename DType>
void _TestCsrsum(DGLContext ctx = CTX) {
  auto A = CSR_A<IdType, DType>(ctx);
  auto C = CSR_C<IdType, DType>(ctx);
  auto A_plus_C = aten::CSRSum({A.first, C.first}, {A.second, C.second});
  auto A_plus_C2 = CSR_A_plus_C<IdType, DType>(ctx);
  bool result = CSRIsClose<IdType, DType>(
      A_plus_C.first, A_plus_C2.first, A_plus_C.second, A_plus_C2.second, 1e-4,
      1e-4);
  ASSERT_TRUE(result);
}

template <typename IdType, typename DType>
void _TestCsrmask(DGLContext ctx = CTX) {
  auto A = CSR_A<IdType, DType>(ctx);
  auto C = CSR_C<IdType, DType>(ctx);
  auto C_coo = CSRToCOO(C.first, false);
  auto A_mask_C =
      aten::CSRGetData<DType>(A.first, C_coo.row, C_coo.col, A.second, 0);
  auto A_mask_C2 = CSR_A_mask_C<DType>(ctx);
  ASSERT_TRUE(ArrayEQ<DType>(A_mask_C, A_mask_C2));
}

TEST(CsrmmTest, TestCsrmm) {
  _TestCsrmm<int32_t, float>(CPU);
  _TestCsrmm<int32_t, double>(CPU);
  _TestCsrmm<int64_t, float>(CPU);
  _TestCsrmm<int64_t, double>(CPU);
#ifdef DGL_USE_CUDA
  _TestCsrmm<int32_t, float>(GPU);
  _TestCsrmm<int32_t, double>(GPU);
  _TestCsrmm<int64_t, float>(GPU);
  _TestCsrmm<int64_t, double>(GPU);
#endif
}

TEST(CsrmmTest, TestCsrsum) {
  _TestCsrsum<int32_t, float>(CPU);
  _TestCsrsum<int32_t, double>(CPU);
  _TestCsrsum<int64_t, float>(CPU);
  _TestCsrsum<int64_t, double>(CPU);
#ifdef DGL_USE_CUDA
  _TestCsrsum<int32_t, float>(GPU);
  _TestCsrsum<int32_t, double>(GPU);
  _TestCsrsum<int64_t, float>(GPU);
  _TestCsrsum<int64_t, double>(GPU);
#endif
}

TEST(CsrmmTest, TestCsrmask) {
  _TestCsrmask<int32_t, float>(CPU);
  _TestCsrmask<int32_t, double>(CPU);
  _TestCsrmask<int64_t, float>(CPU);
  _TestCsrmask<int64_t, double>(CPU);
#ifdef DGL_USE_CUDA
  _TestCsrmask<int32_t, float>(GPU);
  _TestCsrmask<int32_t, double>(GPU);
  _TestCsrmask<int64_t, float>(GPU);
  _TestCsrmask<int64_t, double>(GPU);
#endif
}

};  // namespace
