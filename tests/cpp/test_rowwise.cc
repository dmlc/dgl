#include <gtest/gtest.h>
#include <dgl/array.h>
#include <tuple>
#include <set>
#include "./common.h"

using namespace dgl;
using namespace dgl::runtime;
using namespace dgl::aten;

template <typename Idx>
using ETuple = std::tuple<Idx, Idx, Idx>;

template <typename Idx>
std::set<ETuple<Idx>> AllEdgeSet(bool has_data) {
  if (has_data) {
    std::set<ETuple<Idx>> eset;
    eset.insert(ETuple<Idx>{0, 0, 2});
    eset.insert(ETuple<Idx>{0, 1, 3});
    eset.insert(ETuple<Idx>{1, 1, 0});
    eset.insert(ETuple<Idx>{3, 2, 1});
    eset.insert(ETuple<Idx>{3, 3, 4});
    return eset;
  } else {
    std::set<ETuple<Idx>> eset;
    eset.insert(ETuple<Idx>{0, 0, 0});
    eset.insert(ETuple<Idx>{0, 1, 1});
    eset.insert(ETuple<Idx>{1, 1, 2});
    eset.insert(ETuple<Idx>{3, 2, 3});
    eset.insert(ETuple<Idx>{3, 3, 4});
    return eset;
  }
}

template <typename Idx>
std::set<ETuple<Idx>> ToEdgeSet(COOMatrix mat) {
  std::set<ETuple<Idx>> eset;
  Idx* row = static_cast<Idx*>(mat.row->data);
  Idx* col = static_cast<Idx*>(mat.col->data);
  Idx* data = static_cast<Idx*>(mat.data->data);
  for (int64_t i = 0; i < mat.row->shape[0]; ++i) {
    //std::cout << row[i] << " " << col[i] <<  " " << data[i] << std::endl;
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
  }
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
      std::vector<FloatType>({.0, .5, .5, .0, .5}));
  for (int k = 0; k < 100; ++k) {
    auto rst = CSRRowWiseSampling(mat, rows, 2, prob, true);
    CheckSampledResult<Idx>(rst, rows, has_data);
    auto eset = ToEdgeSet<Idx>(rst);
    if (has_data) {
      ASSERT_FALSE(eset.count(std::make_tuple(0, 1, 3)));
    } else {
      ASSERT_FALSE(eset.count(std::make_tuple(0, 0, 0)));
      ASSERT_FALSE(eset.count(std::make_tuple(3, 2, 3)));
    }
  }
}

TEST(RowwiseTest, TestCSRSampling) {
  _TestCSRSampling<int32_t, float>(true);
  _TestCSRSampling<int64_t, float>(true);
  _TestCSRSampling<int32_t, double>(true);
  _TestCSRSampling<int64_t, double>(true);
  _TestCSRSampling<int32_t, float>(false);
  _TestCSRSampling<int64_t, float>(false);
  _TestCSRSampling<int32_t, double>(false);
  _TestCSRSampling<int64_t, double>(false);
}

template <typename Idx, typename FloatType>
void _TestCSRSamplingUniform(bool has_data) {
  auto mat = CSR<Idx>(has_data);
  FloatArray prob = aten::NullArray();
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


template <typename Idx, typename FloatType>
void _TestCOOSampling(bool has_data) {
  auto mat = COO<Idx>(has_data);
  FloatArray prob = NDArray::FromVector(
      std::vector<FloatType>({.5, .5, .5, .5, .5}));
  IdArray rows = NDArray::FromVector(std::vector<Idx>({0, 3}));
  for (int k = 0; k < 10; ++k) {
    auto rst = COORowWiseSampling(mat, rows, 2, prob, true);
    CheckSampledResult<Idx>(rst, rows, has_data);
  }
  for (int k = 0; k < 10; ++k) {
    auto rst = COORowWiseSampling(mat, rows, 2, prob, false);
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
      std::vector<FloatType>({.0, .5, .5, .0, .5}));
  for (int k = 0; k < 100; ++k) {
    auto rst = COORowWiseSampling(mat, rows, 2, prob, true);
    CheckSampledResult<Idx>(rst, rows, has_data);
    auto eset = ToEdgeSet<Idx>(rst);
    if (has_data) {
      ASSERT_FALSE(eset.count(std::make_tuple(0, 1, 3)));
    } else {
      ASSERT_FALSE(eset.count(std::make_tuple(0, 0, 0)));
      ASSERT_FALSE(eset.count(std::make_tuple(3, 2, 3)));
    }
  }
}

TEST(RowwiseTest, TestCOOSampling) {
  _TestCOOSampling<int32_t, float>(true);
  _TestCOOSampling<int64_t, float>(true);
  _TestCOOSampling<int32_t, double>(true);
  _TestCOOSampling<int64_t, double>(true);
  _TestCOOSampling<int32_t, float>(false);
  _TestCOOSampling<int64_t, float>(false);
  _TestCOOSampling<int32_t, double>(false);
  _TestCOOSampling<int64_t, double>(false);
}

template <typename Idx, typename FloatType>
void _TestCOOSamplingUniform(bool has_data) {
  auto mat = COO<Idx>(has_data);
  FloatArray prob = aten::NullArray();
  IdArray rows = NDArray::FromVector(std::vector<Idx>({0, 3}));
  for (int k = 0; k < 10; ++k) {
    auto rst = COORowWiseSampling(mat, rows, 2, prob, true);
    CheckSampledResult<Idx>(rst, rows, has_data);
  }
  for (int k = 0; k < 10; ++k) {
    auto rst = COORowWiseSampling(mat, rows, 2, prob, false);
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

TEST(RowwiseTest, TestCOOSamplingUniform) {
  _TestCOOSamplingUniform<int32_t, float>(true);
  _TestCOOSamplingUniform<int64_t, float>(true);
  _TestCOOSamplingUniform<int32_t, double>(true);
  _TestCOOSamplingUniform<int64_t, double>(true);
  _TestCOOSamplingUniform<int32_t, float>(false);
  _TestCOOSamplingUniform<int64_t, float>(false);
  _TestCOOSamplingUniform<int32_t, double>(false);
  _TestCOOSamplingUniform<int64_t, double>(false);
}

template <typename Idx, typename FloatType>
void _TestCSRTopk(bool has_data) {
  auto mat = CSR<Idx>(has_data);
  FloatArray weight = NDArray::FromVector(
      std::vector<FloatType>({.1, .0, -.1, .2, .5}));
  // -.1, .2, .1, .0, .5
  IdArray rows = NDArray::FromVector(std::vector<Idx>({0, 3}));

  {
  auto rst = CSRRowWiseTopk(mat, rows, 1, weight, true);
  auto eset = ToEdgeSet<Idx>(rst);
  ASSERT_EQ(eset.size(), 2);
  if (has_data) {
    ASSERT_TRUE(eset.count(std::make_tuple(0, 0, 2)));
    ASSERT_TRUE(eset.count(std::make_tuple(3, 2, 1)));
  } else {
    ASSERT_TRUE(eset.count(std::make_tuple(0, 1, 1)));
    ASSERT_TRUE(eset.count(std::make_tuple(3, 2, 3)));
  }
  }

  {
  auto rst = CSRRowWiseTopk(mat, rows, 1, weight, false);
  auto eset = ToEdgeSet<Idx>(rst);
  ASSERT_EQ(eset.size(), 2);
  if (has_data) {
    ASSERT_TRUE(eset.count(std::make_tuple(0, 1, 3)));
    ASSERT_TRUE(eset.count(std::make_tuple(3, 3, 4)));
  } else {
    ASSERT_TRUE(eset.count(std::make_tuple(0, 0, 0)));
    ASSERT_TRUE(eset.count(std::make_tuple(3, 3, 4)));
  }
  }
}

TEST(RowwiseTest, TestCSRTopk) {
  _TestCSRTopk<int32_t, float>(true);
  _TestCSRTopk<int64_t, float>(true);
  _TestCSRTopk<int32_t, double>(true);
  _TestCSRTopk<int64_t, double>(true);
  _TestCSRTopk<int32_t, float>(false);
  _TestCSRTopk<int64_t, float>(false);
  _TestCSRTopk<int32_t, double>(false);
  _TestCSRTopk<int64_t, double>(false);
}


template <typename Idx, typename FloatType>
void _TestCOOTopk(bool has_data) {
  auto mat = COO<Idx>(has_data);
  FloatArray weight = NDArray::FromVector(
      std::vector<FloatType>({.1, .0, -.1, .2, .5}));
  // -.1, .2, .1, .0, .5
  IdArray rows = NDArray::FromVector(std::vector<Idx>({0, 3}));

  {
  auto rst = COORowWiseTopk(mat, rows, 1, weight, true);
  auto eset = ToEdgeSet<Idx>(rst);
  ASSERT_EQ(eset.size(), 2);
  if (has_data) {
    ASSERT_TRUE(eset.count(std::make_tuple(0, 0, 2)));
    ASSERT_TRUE(eset.count(std::make_tuple(3, 2, 1)));
  } else {
    ASSERT_TRUE(eset.count(std::make_tuple(0, 1, 1)));
    ASSERT_TRUE(eset.count(std::make_tuple(3, 2, 3)));
  }
  }

  {
  auto rst = COORowWiseTopk(mat, rows, 1, weight, false);
  auto eset = ToEdgeSet<Idx>(rst);
  ASSERT_EQ(eset.size(), 2);
  if (has_data) {
    ASSERT_TRUE(eset.count(std::make_tuple(0, 1, 3)));
    ASSERT_TRUE(eset.count(std::make_tuple(3, 3, 4)));
  } else {
    ASSERT_TRUE(eset.count(std::make_tuple(0, 0, 0)));
    ASSERT_TRUE(eset.count(std::make_tuple(3, 3, 4)));
  }
  }
}

TEST(RowwiseTest, TestCOOTopk) {
  _TestCOOTopk<int32_t, float>(true);
  _TestCOOTopk<int64_t, float>(true);
  _TestCOOTopk<int32_t, double>(true);
  _TestCOOTopk<int64_t, double>(true);
  _TestCOOTopk<int32_t, float>(false);
  _TestCOOTopk<int64_t, float>(false);
  _TestCOOTopk<int32_t, double>(false);
  _TestCOOTopk<int64_t, double>(false);
}
