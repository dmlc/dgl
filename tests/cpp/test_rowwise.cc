#include <gtest/gtest.h>
#include <dgl/array.h>
#include <tuple>
#include <set>
#include "./common.h"

using namespace dgl;
using namespace dgl::runtime;
using namespace dgl::aten;

template <typename Idx>
std::set<std::tuple<Idx, Idx, Idx>> AllEdgeSet(bool has_data) {
  if (has_data) {
    std::set<std::tuple<Idx, Idx, Idx>> eset;
    eset.insert({0, 0, 2});
    eset.insert({0, 1, 3});
    eset.insert({1, 1, 0});
    eset.insert({3, 2, 1});
    eset.insert({3, 3, 4});
    return eset;
  } else {
    std::set<std::tuple<Idx, Idx, Idx>> eset;
    eset.insert({0, 0, 0});
    eset.insert({0, 1, 1});
    eset.insert({1, 1, 2});
    eset.insert({3, 2, 3});
    eset.insert({3, 3, 4});
    return eset;
  }
}

template <typename Idx>
std::set<std::tuple<Idx, Idx, Idx>> ToEdgeSet(COOMatrix mat) {
  std::set<std::tuple<Idx, Idx, Idx>> eset;
  Idx* row = static_cast<Idx*>(mat.row->data);
  Idx* col = static_cast<Idx*>(mat.col->data);
  Idx* data = static_cast<Idx*>(mat.data->data);
  for (int64_t i = 0; i < mat.row->shape[0]; ++i) {
    std::cout << row[i] << " " << col[i] <<  " " << data[i] << std::endl;
    eset.emplace(row[i], col[i], data[i]);
  }
  return eset;
}

template <typename Idx>
void CheckSampledResult(COOMatrix mat, IdArray rows, bool has_data) {
  ASSERT_EQ(mat.num_rows, 4);
  ASSERT_EQ(mat.num_cols, 4);
  Idx* row = static_cast<Idx*>(mat.row->data);
  Idx* col = static_cast<Idx*>(mat.col->data);
  Idx* data = static_cast<Idx*>(mat.data->data);
  const auto& gt = AllEdgeSet<Idx>(has_data);
  for (int64_t i = 0; i < mat.row->shape[0]; ++i) {
    ASSERT_TRUE(gt.count(std::make_tuple(row[i], col[i], data[i])));
    ASSERT_TRUE(IsInArray(rows, row[i]));
  }
}

template <typename Idx>
CSRMatrix CSR(bool has_data) {
  IdArray indptr = NDArray::FromVector(std::vector<Idx>({0, 2, 3, 3, 5}));
  IdArray indices = NDArray::FromVector(std::vector<Idx>({0, 1, 1, 2, 3}));
  IdArray data = NDArray::FromVector(std::vector<Idx>({2, 3, 0, 1, 4}));
  if (has_data)
    return CSRMatrix(4, 4, indptr, indices, data);
  else
    return CSRMatrix(4, 4, indptr, indices);
}

template <typename Idx>
COOMatrix COO(bool has_data) {
  IdArray row = NDArray::FromVector(std::vector<Idx>({0, 0, 1, 3, 3}));
  IdArray col = NDArray::FromVector(std::vector<Idx>({0, 1, 1, 2, 3}));
  IdArray data = NDArray::FromVector(std::vector<Idx>({2, 3, 0, 1, 4}));
  if (has_data)
    return COOMatrix(4, 4, row, col, data);
  else
    return COOMatrix(4, 4, row, col);
}

template <typename Idx, typename FloatType>
void _TestCSRSampling(bool has_data) {
  auto mat = CSR<Idx>(has_data);
  FloatArray prob = NDArray::FromVector(
      std::vector<FloatType>({.5, .5, .5, .5, .5}));
  IdArray rows = NDArray::FromVector(std::vector<Idx>({0, 3}));
  for (int k = 0; k < 10; ++k) {
    auto rst = CSRRowWiseSampling(mat, rows, 2, prob, true);
    CheckSampledResult<Idx>(rst, rows, has_data);
    std::cout << "..............." << std::endl;
    auto eset = ToEdgeSet<Idx>(rst);
  }
  ASSERT_TRUE(false);
  for (int k = 0; k < 10; ++k) {
    auto rst = CSRRowWiseSampling(mat, rows, 2, prob, false);
    CheckSampledResult<Idx>(rst, rows, has_data);
    auto eset = ToEdgeSet<Idx>(rst);
    ASSERT_EQ(eset.size(), 4);
    if (has_data) {
      ASSERT_TRUE(eset.count(std::make_tuple(0, 0, 2)));
      ASSERT_TRUE(eset.count(std::make_tuple(0, 1, 3)));
      ASSERT_TRUE(eset.count(std::make_tuple(3, 2, 1)));
      ASSERT_TRUE(eset.count(std::make_tuple(3, 3, 4)));
    } else {
      ASSERT_TRUE(eset.count(std::make_tuple(0, 0, 0)));
      ASSERT_TRUE(eset.count(std::make_tuple(0, 1, 1)));
      ASSERT_TRUE(eset.count(std::make_tuple(3, 2, 3)));
      ASSERT_TRUE(eset.count(std::make_tuple(3, 3, 4)));
    }
  }
  prob = NDArray::FromVector(
      std::vector<FloatType>({.5, .0, .5, .0, .5}));
  for (int k = 0; k < 100; ++k) {
    std::cout << "..................." << k << std::endl;
    auto rst = CSRRowWiseSampling(mat, rows, 2, prob, true);
    CheckSampledResult<Idx>(rst, rows, has_data);
    auto eset = ToEdgeSet<Idx>(rst);
    if (has_data) {
      ASSERT_FALSE(eset.count(std::make_tuple(0, 1, 3)));
      ASSERT_FALSE(eset.count(std::make_tuple(3, 2, 1)));
    } else {
      ASSERT_FALSE(eset.count(std::make_tuple(0, 1, 1)));
      ASSERT_FALSE(eset.count(std::make_tuple(3, 2, 3)));
    }
  }
}

TEST(RowwiseTest, TestCSRSampling) {
  _TestCSRSampling<int32_t, float>(true);
  //_TestCSRSampling<int64_t, float>(true);
  //_TestCSRSampling<int32_t, double>(true);
  //_TestCSRSampling<int64_t, double>(true);
  //_TestCSRSampling<int32_t, float>(false);
  //_TestCSRSampling<int64_t, float>(false);
  //_TestCSRSampling<int32_t, double>(false);
  //_TestCSRSampling<int64_t, double>(false);
}

template <typename Idx, typename FloatType>
void _TestCSRSamplingUniform(bool has_data) {
  auto mat = CSR<Idx>(has_data);
  FloatArray prob;
  IdArray rows = NDArray::FromVector(std::vector<Idx>({0, 3}));
  for (int k = 0; k < 10; ++k) {
    auto rst = CSRRowWiseSampling(mat, rows, 2, prob, true);
    CheckSampledResult<Idx>(rst, rows, has_data);
  }
  for (int k = 0; k < 10; ++k) {
    auto rst = CSRRowWiseSampling(mat, rows, 2, prob, false);
    CheckSampledResult<Idx>(rst, rows, has_data);
    auto eset = ToEdgeSet<Idx>(rst);
    if (has_data) {
      ASSERT_TRUE(eset.count(std::make_tuple(0, 0, 2)));
      ASSERT_TRUE(eset.count(std::make_tuple(0, 1, 3)));
      ASSERT_TRUE(eset.count(std::make_tuple(3, 2, 1)));
      ASSERT_TRUE(eset.count(std::make_tuple(3, 3, 4)));
    } else {
      ASSERT_TRUE(eset.count(std::make_tuple(0, 0, 0)));
      ASSERT_TRUE(eset.count(std::make_tuple(0, 1, 1)));
      ASSERT_TRUE(eset.count(std::make_tuple(3, 2, 3)));
      ASSERT_TRUE(eset.count(std::make_tuple(3, 3, 4)));
    }
  }
}

TEST(RowwiseTest, TestCSRSamplingUniform) {
  _TestCSRSamplingUniform<int32_t, float>(true);
  _TestCSRSamplingUniform<int64_t, float>(true);
  _TestCSRSamplingUniform<int32_t, double>(true);
  _TestCSRSamplingUniform<int64_t, double>(true);
  _TestCSRSamplingUniform<int32_t, float>(false);
  _TestCSRSamplingUniform<int64_t, float>(false);
  _TestCSRSamplingUniform<int32_t, double>(false);
  _TestCSRSamplingUniform<int64_t, double>(false);
}
