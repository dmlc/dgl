#include <dgl/array.h>
#include <gtest/gtest.h>

#include <set>
#include <tuple>

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
std::set<ETuple<Idx>> AllEdgePerEtypeSet(bool has_data) {
  if (has_data) {
    std::set<ETuple<Idx>> eset;
    eset.insert(ETuple<Idx>{0, 0, 0});
    eset.insert(ETuple<Idx>{0, 1, 1});
    eset.insert(ETuple<Idx>{0, 2, 4});
    eset.insert(ETuple<Idx>{0, 3, 6});
    eset.insert(ETuple<Idx>{3, 2, 5});
    eset.insert(ETuple<Idx>{3, 3, 3});
    return eset;
  } else {
    std::set<ETuple<Idx>> eset;
    eset.insert(ETuple<Idx>{0, 0, 0});
    eset.insert(ETuple<Idx>{0, 1, 1});
    eset.insert(ETuple<Idx>{0, 2, 2});
    eset.insert(ETuple<Idx>{0, 3, 3});
    eset.insert(ETuple<Idx>{3, 3, 5});
    eset.insert(ETuple<Idx>{3, 2, 6});
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
    // std::cout << row[i] << " " << col[i] <<  " " << data[i] << std::endl;
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
void CheckSampledPerEtypeResult(COOMatrix mat, IdArray rows, bool has_data) {
  ASSERT_EQ(mat.num_rows, 4);
  ASSERT_EQ(mat.num_cols, 4);
  Idx* row = static_cast<Idx*>(mat.row->data);
  Idx* col = static_cast<Idx*>(mat.col->data);
  Idx* data = static_cast<Idx*>(mat.data->data);
  const auto& gt = AllEdgePerEtypeSet<Idx>(has_data);
  for (int64_t i = 0; i < mat.row->shape[0]; ++i) {
    int64_t count = gt.count(std::make_tuple(row[i], col[i], data[i]));
    ASSERT_TRUE(count);
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

template <typename Idx>
std::pair<CSRMatrix, std::vector<int64_t>> CSREtypes(bool has_data) {
  IdArray indptr = NDArray::FromVector(std::vector<Idx>({0, 4, 5, 5, 7}));
  IdArray indices =
      NDArray::FromVector(std::vector<Idx>({0, 1, 2, 3, 1, 3, 2}));
  IdArray data = NDArray::FromVector(std::vector<Idx>({0, 1, 4, 6, 2, 3, 5}));
  auto eid2etype_offsets = std::vector<int64_t>({0, 4, 5, 6, 7});
  if (has_data)
    return {CSRMatrix(4, 4, indptr, indices, data), eid2etype_offsets};
  else
    return {CSRMatrix(4, 4, indptr, indices), eid2etype_offsets};
}

template <typename Idx>
std::pair<COOMatrix, std::vector<int64_t>> COOEtypes(bool has_data) {
  IdArray row = NDArray::FromVector(std::vector<Idx>({0, 0, 0, 0, 1, 3, 3}));
  IdArray col = NDArray::FromVector(std::vector<Idx>({0, 1, 2, 3, 1, 3, 2}));
  IdArray data = NDArray::FromVector(std::vector<Idx>({0, 1, 4, 6, 2, 3, 5}));
  auto eid2etype_offsets = std::vector<int64_t>({0, 4, 5, 6, 7});
  if (has_data)
    return {COOMatrix(4, 4, row, col, data), eid2etype_offsets};
  else
    return {COOMatrix(4, 4, row, col), eid2etype_offsets};
}

template <typename Idx, typename FloatType>
void _TestCSRSampling(bool has_data) {
  auto mat = CSR<Idx>(has_data);
  FloatArray prob =
      NDArray::FromVector(std::vector<FloatType>({.5, .5, .5, .5, .5}));
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
  prob = NDArray::FromVector(std::vector<FloatType>({.0, .5, .5, .0, .5}));
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
void _TestCSRPerEtypeSampling(bool has_data) {
  auto pair = CSREtypes<Idx>(has_data);
  auto mat = pair.first;
  auto eid2etype_offset = pair.second;
  std::vector<FloatArray> prob = {
      NDArray::FromVector(std::vector<FloatType>({.5, .5, .5, .5})),
      NDArray::FromVector(std::vector<FloatType>({.5})),
      NDArray::FromVector(std::vector<FloatType>({.5})),
      NDArray::FromVector(std::vector<FloatType>({.5}))};
  IdArray rows = NDArray::FromVector(std::vector<Idx>({0, 3}));
  for (int k = 0; k < 10; ++k) {
    auto rst = CSRRowWisePerEtypeSampling(
        mat, rows, eid2etype_offset, {2, 2, 2, 2}, prob, true);
    CheckSampledPerEtypeResult<Idx>(rst, rows, has_data);
  }
  for (int k = 0; k < 10; ++k) {
    auto rst = CSRRowWisePerEtypeSampling(
        mat, rows, eid2etype_offset, {2, 2, 2, 2}, prob, false);
    CheckSampledPerEtypeResult<Idx>(rst, rows, has_data);
    auto eset = ToEdgeSet<Idx>(rst);
    if (has_data) {
      int counts = 0;
      counts += eset.count(std::make_tuple(0, 0, 0));
      counts += eset.count(std::make_tuple(0, 1, 1));
      ASSERT_EQ(counts, 2);
      counts = 0;
      counts += eset.count(std::make_tuple(0, 2, 4));
      ASSERT_EQ(counts, 1);
      counts = 0;
      counts += eset.count(std::make_tuple(0, 3, 6));
      ASSERT_EQ(counts, 1);
      counts = 0;
      counts += eset.count(std::make_tuple(1, 1, 2));
      ASSERT_EQ(counts, 0);
      counts = 0;
      counts += eset.count(std::make_tuple(3, 2, 5));
      ASSERT_EQ(counts, 1);
      counts = 0;
      counts += eset.count(std::make_tuple(3, 3, 3));
      ASSERT_EQ(counts, 1);
    } else {
      int counts = 0;
      counts += eset.count(std::make_tuple(0, 0, 0));
      counts += eset.count(std::make_tuple(0, 1, 1));
      counts += eset.count(std::make_tuple(0, 2, 2));
      counts += eset.count(std::make_tuple(0, 3, 3));
      ASSERT_EQ(counts, 2);
      counts = 0;
      counts += eset.count(std::make_tuple(1, 1, 4));
      ASSERT_EQ(counts, 0);
      counts = 0;
      counts += eset.count(std::make_tuple(3, 3, 5));
      ASSERT_EQ(counts, 1);
      counts = 0;
      counts += eset.count(std::make_tuple(3, 2, 6));
      ASSERT_EQ(counts, 1);
    }
  }

  prob = {
      NDArray::FromVector(std::vector<FloatType>({.0, .5, .0, .0})),
      NDArray::FromVector(std::vector<FloatType>({.5})),
      NDArray::FromVector(std::vector<FloatType>({.5})),
      NDArray::FromVector(std::vector<FloatType>({.5}))};
  for (int k = 0; k < 10; ++k) {
    auto rst = CSRRowWisePerEtypeSampling(
        mat, rows, eid2etype_offset, {2, 2, 2, 2}, prob, true);
    CheckSampledPerEtypeResult<Idx>(rst, rows, has_data);
    auto eset = ToEdgeSet<Idx>(rst);
    if (has_data) {
      ASSERT_FALSE(eset.count(std::make_tuple(0, 0, 0)));
    } else {
      ASSERT_FALSE(eset.count(std::make_tuple(0, 0, 0)));
      ASSERT_FALSE(eset.count(std::make_tuple(0, 2, 2)));
      ASSERT_FALSE(eset.count(std::make_tuple(0, 3, 3)));
    }
  }
}

template <typename Idx, typename FloatType>
void _TestCSRPerEtypeSamplingSorted() {
  auto pair = CSREtypes<Idx>(true);
  auto mat = pair.first;
  auto eid2etype_offset = pair.second;
  std::vector<FloatArray> prob = {
      NDArray::FromVector(std::vector<FloatType>({.5, .5, .5, .5})),
      NDArray::FromVector(std::vector<FloatType>({.5})),
      NDArray::FromVector(std::vector<FloatType>({.5})),
      NDArray::FromVector(std::vector<FloatType>({.5}))};
  IdArray rows = NDArray::FromVector(std::vector<Idx>({0, 3}));
  for (int k = 0; k < 10; ++k) {
    auto rst = CSRRowWisePerEtypeSampling(
        mat, rows, eid2etype_offset, {2, 2, 2, 2}, prob, true, true);
    CheckSampledPerEtypeResult<Idx>(rst, rows, true);
  }
  for (int k = 0; k < 10; ++k) {
    auto rst = CSRRowWisePerEtypeSampling(
        mat, rows, eid2etype_offset, {2, 2, 2, 2}, prob, false, true);
    CheckSampledPerEtypeResult<Idx>(rst, rows, true);
    auto eset = ToEdgeSet<Idx>(rst);
    int counts = 0;
    counts += eset.count(std::make_tuple(0, 0, 0));
    counts += eset.count(std::make_tuple(0, 1, 1));
    ASSERT_EQ(counts, 2);
    counts = 0;
    counts += eset.count(std::make_tuple(0, 2, 4));
    ASSERT_EQ(counts, 1);
    counts = 0;
    counts += eset.count(std::make_tuple(0, 3, 6));
    ASSERT_EQ(counts, 1);
    counts = 0;
    counts += eset.count(std::make_tuple(1, 1, 2));
    ASSERT_EQ(counts, 0);
    counts = 0;
    counts += eset.count(std::make_tuple(3, 2, 5));
    ASSERT_EQ(counts, 1);
    counts = 0;
    counts += eset.count(std::make_tuple(3, 3, 3));
    ASSERT_EQ(counts, 1);
  }

  prob = {
      NDArray::FromVector(std::vector<FloatType>({.0, .5, .0, .0})),
      NDArray::FromVector(std::vector<FloatType>({.5})),
      NDArray::FromVector(std::vector<FloatType>({.5})),
      NDArray::FromVector(std::vector<FloatType>({.5}))};
  for (int k = 0; k < 10; ++k) {
    auto rst = CSRRowWisePerEtypeSampling(
        mat, rows, eid2etype_offset, {2, 2, 2, 2}, prob, true, true);
    CheckSampledPerEtypeResult<Idx>(rst, rows, true);
    auto eset = ToEdgeSet<Idx>(rst);
    ASSERT_FALSE(eset.count(std::make_tuple(0, 0, 0)));
  }
}

TEST(RowwiseTest, TestCSRPerEtypeSampling) {
  _TestCSRPerEtypeSampling<int32_t, float>(true);
  _TestCSRPerEtypeSampling<int64_t, float>(true);
  _TestCSRPerEtypeSampling<int32_t, double>(true);
  _TestCSRPerEtypeSampling<int64_t, double>(true);
  _TestCSRPerEtypeSampling<int32_t, float>(false);
  _TestCSRPerEtypeSampling<int64_t, float>(false);
  _TestCSRPerEtypeSampling<int32_t, double>(false);
  _TestCSRPerEtypeSampling<int64_t, double>(false);
  _TestCSRPerEtypeSamplingSorted<int32_t, float>();
  _TestCSRPerEtypeSamplingSorted<int64_t, float>();
  _TestCSRPerEtypeSamplingSorted<int32_t, double>();
  _TestCSRPerEtypeSamplingSorted<int64_t, double>();
}

template <typename Idx, typename FloatType>
void _TestCSRPerEtypeSamplingUniform(bool has_data) {
  auto pair = CSREtypes<Idx>(has_data);
  auto mat = pair.first;
  auto eid2etype_offset = pair.second;
  std::vector<FloatArray> prob = {
      aten::NullArray(), aten::NullArray(), aten::NullArray(),
      aten::NullArray()};
  IdArray rows = NDArray::FromVector(std::vector<Idx>({0, 3}));
  for (int k = 0; k < 10; ++k) {
    auto rst = CSRRowWisePerEtypeSampling(
        mat, rows, eid2etype_offset, {2, 2, 2, 2}, prob, true);
    CheckSampledPerEtypeResult<Idx>(rst, rows, has_data);
  }
  for (int k = 0; k < 10; ++k) {
    auto rst = CSRRowWisePerEtypeSampling(
        mat, rows, eid2etype_offset, {2, 2, 2, 2}, prob, false);
    CheckSampledPerEtypeResult<Idx>(rst, rows, has_data);
    auto eset = ToEdgeSet<Idx>(rst);
    if (has_data) {
      int counts = 0;
      counts += eset.count(std::make_tuple(0, 0, 0));
      counts += eset.count(std::make_tuple(0, 1, 1));
      ASSERT_EQ(counts, 2);
      counts = 0;
      counts += eset.count(std::make_tuple(0, 2, 4));
      ASSERT_EQ(counts, 1);
      counts = 0;
      counts += eset.count(std::make_tuple(0, 3, 6));
      ASSERT_EQ(counts, 1);
      counts = 0;
      counts += eset.count(std::make_tuple(1, 1, 2));
      ASSERT_EQ(counts, 0);
      counts = 0;
      counts += eset.count(std::make_tuple(3, 2, 5));
      ASSERT_EQ(counts, 1);
      counts = 0;
      counts += eset.count(std::make_tuple(3, 3, 3));
      ASSERT_EQ(counts, 1);
    } else {
      int counts = 0;
      counts += eset.count(std::make_tuple(0, 0, 0));
      counts += eset.count(std::make_tuple(0, 1, 1));
      counts += eset.count(std::make_tuple(0, 2, 2));
      counts += eset.count(std::make_tuple(0, 3, 3));
      ASSERT_EQ(counts, 2);
      counts = 0;
      counts += eset.count(std::make_tuple(1, 1, 4));
      ASSERT_EQ(counts, 0);
      counts = 0;
      counts += eset.count(std::make_tuple(3, 3, 5));
      ASSERT_EQ(counts, 1);
      counts = 0;
      counts += eset.count(std::make_tuple(3, 2, 6));
      ASSERT_EQ(counts, 1);
    }
  }
}

template <typename Idx, typename FloatType>
void _TestCSRPerEtypeSamplingUniformSorted() {
  auto pair = CSREtypes<Idx>(true);
  auto mat = pair.first;
  auto eid2etype_offset = pair.second;
  std::vector<FloatArray> prob = {
      aten::NullArray(), aten::NullArray(), aten::NullArray(),
      aten::NullArray()};
  IdArray rows = NDArray::FromVector(std::vector<Idx>({0, 3}));
  for (int k = 0; k < 10; ++k) {
    auto rst = CSRRowWisePerEtypeSampling(
        mat, rows, eid2etype_offset, {2, 2, 2, 2}, prob, true, true);
    CheckSampledPerEtypeResult<Idx>(rst, rows, true);
  }
  for (int k = 0; k < 10; ++k) {
    auto rst = CSRRowWisePerEtypeSampling(
        mat, rows, eid2etype_offset, {2, 2, 2, 2}, prob, false, true);
    CheckSampledPerEtypeResult<Idx>(rst, rows, true);
    auto eset = ToEdgeSet<Idx>(rst);
    int counts = 0;
    counts += eset.count(std::make_tuple(0, 0, 0));
    counts += eset.count(std::make_tuple(0, 1, 1));
    ASSERT_EQ(counts, 2);
    counts = 0;
    counts += eset.count(std::make_tuple(0, 2, 4));
    ASSERT_EQ(counts, 1);
    counts = 0;
    counts += eset.count(std::make_tuple(0, 3, 6));
    ASSERT_EQ(counts, 1);
    counts = 0;
    counts += eset.count(std::make_tuple(1, 1, 2));
    ASSERT_EQ(counts, 0);
    counts = 0;
    counts += eset.count(std::make_tuple(3, 2, 5));
    ASSERT_EQ(counts, 1);
    counts = 0;
    counts += eset.count(std::make_tuple(3, 3, 3));
    ASSERT_EQ(counts, 1);
  }
}

TEST(RowwiseTest, TestCSRPerEtypeSamplingUniform) {
  _TestCSRPerEtypeSamplingUniform<int32_t, float>(true);
  _TestCSRPerEtypeSamplingUniform<int64_t, float>(true);
  _TestCSRPerEtypeSamplingUniform<int32_t, double>(true);
  _TestCSRPerEtypeSamplingUniform<int64_t, double>(true);
  _TestCSRPerEtypeSamplingUniform<int32_t, float>(false);
  _TestCSRPerEtypeSamplingUniform<int64_t, float>(false);
  _TestCSRPerEtypeSamplingUniform<int32_t, double>(false);
  _TestCSRPerEtypeSamplingUniform<int64_t, double>(false);
  _TestCSRPerEtypeSamplingUniformSorted<int32_t, float>();
  _TestCSRPerEtypeSamplingUniformSorted<int64_t, float>();
  _TestCSRPerEtypeSamplingUniformSorted<int32_t, double>();
  _TestCSRPerEtypeSamplingUniformSorted<int64_t, double>();
}

template <typename Idx, typename FloatType>
void _TestCOOSampling(bool has_data) {
  auto mat = COO<Idx>(has_data);
  FloatArray prob =
      NDArray::FromVector(std::vector<FloatType>({.5, .5, .5, .5, .5}));
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
  prob = NDArray::FromVector(std::vector<FloatType>({.0, .5, .5, .0, .5}));
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

// COOPerEtypeSampling with rowwise_etype_sorted == true is not meaningful as
// it's never used in practice.

template <typename Idx, typename FloatType>
void _TestCOOPerEtypeSampling(bool has_data) {
  auto pair = COOEtypes<Idx>(has_data);
  auto mat = pair.first;
  auto eid2etype_offset = pair.second;
  std::vector<FloatArray> prob = {
      NDArray::FromVector(std::vector<FloatType>({.5, .5, .5, .5})),
      NDArray::FromVector(std::vector<FloatType>({.5})),
      NDArray::FromVector(std::vector<FloatType>({.5})),
      NDArray::FromVector(std::vector<FloatType>({.5}))};
  IdArray rows = NDArray::FromVector(std::vector<Idx>({0, 3}));
  for (int k = 0; k < 10; ++k) {
    auto rst = COORowWisePerEtypeSampling(
        mat, rows, eid2etype_offset, {2, 2, 2, 2}, prob, true);
    CheckSampledPerEtypeResult<Idx>(rst, rows, has_data);
  }
  for (int k = 0; k < 10; ++k) {
    auto rst = COORowWisePerEtypeSampling(
        mat, rows, eid2etype_offset, {2, 2, 2, 2}, prob, false);
    CheckSampledPerEtypeResult<Idx>(rst, rows, has_data);
    auto eset = ToEdgeSet<Idx>(rst);
    if (has_data) {
      int counts = 0;
      counts += eset.count(std::make_tuple(0, 0, 0));
      counts += eset.count(std::make_tuple(0, 1, 1));
      ASSERT_EQ(counts, 2);
      counts = 0;
      counts += eset.count(std::make_tuple(0, 2, 4));
      ASSERT_EQ(counts, 1);
      counts = 0;
      counts += eset.count(std::make_tuple(0, 3, 6));
      ASSERT_EQ(counts, 1);
      counts = 0;
      counts += eset.count(std::make_tuple(1, 1, 2));
      ASSERT_EQ(counts, 0);
      counts = 0;
      counts += eset.count(std::make_tuple(3, 2, 5));
      ASSERT_EQ(counts, 1);
      counts = 0;
      counts += eset.count(std::make_tuple(3, 3, 3));
      ASSERT_EQ(counts, 1);
    } else {
      int counts = 0;
      counts += eset.count(std::make_tuple(0, 0, 0));
      counts += eset.count(std::make_tuple(0, 1, 1));
      counts += eset.count(std::make_tuple(0, 2, 2));
      counts += eset.count(std::make_tuple(0, 3, 3));
      ASSERT_EQ(counts, 2);
      counts = 0;
      counts += eset.count(std::make_tuple(1, 1, 4));
      ASSERT_EQ(counts, 0);
      counts = 0;
      counts += eset.count(std::make_tuple(3, 3, 5));
      ASSERT_EQ(counts, 1);
      counts = 0;
      counts += eset.count(std::make_tuple(3, 2, 6));
      ASSERT_EQ(counts, 1);
    }
  }

  prob = {
      NDArray::FromVector(std::vector<FloatType>({.0, .5, .0, .0})),
      NDArray::FromVector(std::vector<FloatType>({.5})),
      NDArray::FromVector(std::vector<FloatType>({.5})),
      NDArray::FromVector(std::vector<FloatType>({.5}))};
  for (int k = 0; k < 10; ++k) {
    auto rst = COORowWisePerEtypeSampling(
        mat, rows, eid2etype_offset, {2, 2, 2, 2}, prob, true);
    CheckSampledPerEtypeResult<Idx>(rst, rows, has_data);
    auto eset = ToEdgeSet<Idx>(rst);
    if (has_data) {
      ASSERT_FALSE(eset.count(std::make_tuple(0, 0, 0)));
    } else {
      ASSERT_FALSE(eset.count(std::make_tuple(0, 0, 0)));
      ASSERT_FALSE(eset.count(std::make_tuple(0, 2, 2)));
      ASSERT_FALSE(eset.count(std::make_tuple(0, 3, 3)));
    }
  }
}

TEST(RowwiseTest, TestCOOPerEtypeSampling) {
  _TestCOOPerEtypeSampling<int32_t, float>(true);
  _TestCOOPerEtypeSampling<int64_t, float>(true);
  _TestCOOPerEtypeSampling<int32_t, double>(true);
  _TestCOOPerEtypeSampling<int64_t, double>(true);
  _TestCOOPerEtypeSampling<int32_t, float>(false);
  _TestCOOPerEtypeSampling<int64_t, float>(false);
  _TestCOOPerEtypeSampling<int32_t, double>(false);
  _TestCOOPerEtypeSampling<int64_t, double>(false);
}

template <typename Idx, typename FloatType>
void _TestCOOPerEtypeSamplingUniform(bool has_data) {
  auto pair = COOEtypes<Idx>(has_data);
  auto mat = pair.first;
  auto eid2etype_offset = pair.second;
  std::vector<FloatArray> prob = {
      aten::NullArray(), aten::NullArray(), aten::NullArray(),
      aten::NullArray()};
  IdArray rows = NDArray::FromVector(std::vector<Idx>({0, 3}));
  for (int k = 0; k < 10; ++k) {
    auto rst = COORowWisePerEtypeSampling(
        mat, rows, eid2etype_offset, {2, 2, 2, 2}, prob, true);
    CheckSampledPerEtypeResult<Idx>(rst, rows, has_data);
  }
  for (int k = 0; k < 10; ++k) {
    auto rst = COORowWisePerEtypeSampling(
        mat, rows, eid2etype_offset, {2, 2, 2, 2}, prob, false);
    CheckSampledPerEtypeResult<Idx>(rst, rows, has_data);
    auto eset = ToEdgeSet<Idx>(rst);
    if (has_data) {
      int counts = 0;
      counts += eset.count(std::make_tuple(0, 0, 0));
      counts += eset.count(std::make_tuple(0, 1, 1));
      ASSERT_EQ(counts, 2);
      counts = 0;
      counts += eset.count(std::make_tuple(0, 2, 4));
      ASSERT_EQ(counts, 1);
      counts = 0;
      counts += eset.count(std::make_tuple(0, 3, 6));
      ASSERT_EQ(counts, 1);
      counts = 0;
      counts += eset.count(std::make_tuple(1, 1, 2));
      ASSERT_EQ(counts, 0);
      counts = 0;
      counts += eset.count(std::make_tuple(3, 2, 5));
      ASSERT_EQ(counts, 1);
      counts = 0;
      counts += eset.count(std::make_tuple(3, 3, 3));
      ASSERT_EQ(counts, 1);
    } else {
      int counts = 0;
      counts += eset.count(std::make_tuple(0, 0, 0));
      counts += eset.count(std::make_tuple(0, 1, 1));
      counts += eset.count(std::make_tuple(0, 2, 2));
      counts += eset.count(std::make_tuple(0, 3, 3));
      ASSERT_EQ(counts, 2);
      counts = 0;
      counts += eset.count(std::make_tuple(1, 1, 4));
      ASSERT_EQ(counts, 0);
      counts = 0;
      counts += eset.count(std::make_tuple(3, 3, 5));
      ASSERT_EQ(counts, 1);
      counts = 0;
      counts += eset.count(std::make_tuple(3, 2, 6));
      ASSERT_EQ(counts, 1);
    }
  }
}

TEST(RowwiseTest, TestCOOPerEtypeSamplingUniform) {
  _TestCOOPerEtypeSamplingUniform<int32_t, float>(true);
  _TestCOOPerEtypeSamplingUniform<int64_t, float>(true);
  _TestCOOPerEtypeSamplingUniform<int32_t, double>(true);
  _TestCOOPerEtypeSamplingUniform<int64_t, double>(true);
  _TestCOOPerEtypeSamplingUniform<int32_t, float>(false);
  _TestCOOPerEtypeSamplingUniform<int64_t, float>(false);
  _TestCOOPerEtypeSamplingUniform<int32_t, double>(false);
  _TestCOOPerEtypeSamplingUniform<int64_t, double>(false);
}

template <typename Idx, typename FloatType>
void _TestCSRTopk(bool has_data) {
  auto mat = CSR<Idx>(has_data);
  FloatArray weight =
      NDArray::FromVector(std::vector<FloatType>({.1f, .0f, -.1f, .2f, .5f}));
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
  FloatArray weight =
      NDArray::FromVector(std::vector<FloatType>({.1f, .0f, -.1f, .2f, .5f}));
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

template <typename Idx, typename FloatType>
void _TestCSRSamplingBiased(bool has_data) {
  auto mat = CSR<Idx>(has_data);
  // 0 - 0,1
  // 1 - 1
  // 3 - 2,3
  NDArray tag_offset = NDArray::FromVector(
      std::vector<Idx>({0, 1, 2, 0, 0, 1, 0, 0, 0, 0, 1, 2}));
  tag_offset = tag_offset.CreateView({4, 3}, tag_offset->dtype);
  IdArray rows = NDArray::FromVector(std::vector<Idx>({0, 1, 3}));
  FloatArray bias = NDArray::FromVector(std::vector<FloatType>({0, 0.5}));
  for (int k = 0; k < 10; ++k) {
    auto rst = CSRRowWiseSamplingBiased(mat, rows, 1, tag_offset, bias, false);
    CheckSampledResult<Idx>(rst, rows, has_data);
    auto eset = ToEdgeSet<Idx>(rst);
    if (has_data) {
      ASSERT_TRUE(eset.count(std::make_tuple(0, 1, 3)));
      ASSERT_TRUE(eset.count(std::make_tuple(1, 1, 0)));
      ASSERT_TRUE(eset.count(std::make_tuple(3, 3, 4)));
    } else {
      ASSERT_TRUE(eset.count(std::make_tuple(0, 1, 1)));
      ASSERT_TRUE(eset.count(std::make_tuple(1, 1, 2)));
      ASSERT_TRUE(eset.count(std::make_tuple(3, 3, 4)));
    }
  }
  for (int k = 0; k < 10; ++k) {
    auto rst = CSRRowWiseSamplingBiased(mat, rows, 3, tag_offset, bias, true);
    CheckSampledResult<Idx>(rst, rows, has_data);
    auto eset = ToEdgeSet<Idx>(rst);
    if (has_data) {
      ASSERT_TRUE(eset.count(std::make_tuple(0, 1, 3)));
      ASSERT_TRUE(eset.count(std::make_tuple(1, 1, 0)));
      ASSERT_TRUE(eset.count(std::make_tuple(3, 3, 4)));
      ASSERT_FALSE(eset.count(std::make_tuple(0, 0, 2)));
      ASSERT_FALSE(eset.count(std::make_tuple(3, 2, 1)));
    } else {
      ASSERT_TRUE(eset.count(std::make_tuple(0, 1, 1)));
      ASSERT_TRUE(eset.count(std::make_tuple(1, 1, 2)));
      ASSERT_TRUE(eset.count(std::make_tuple(3, 3, 4)));
      ASSERT_FALSE(eset.count(std::make_tuple(0, 0, 0)));
      ASSERT_FALSE(eset.count(std::make_tuple(3, 2, 3)));
    }
  }
}

TEST(RowwiseTest, TestCSRSamplingBiased) {
  _TestCSRSamplingBiased<int32_t, float>(true);
  _TestCSRSamplingBiased<int32_t, float>(false);
  _TestCSRSamplingBiased<int64_t, float>(true);
  _TestCSRSamplingBiased<int64_t, float>(false);
  _TestCSRSamplingBiased<int32_t, double>(true);
  _TestCSRSamplingBiased<int32_t, double>(false);
  _TestCSRSamplingBiased<int64_t, double>(true);
  _TestCSRSamplingBiased<int64_t, double>(false);
}
