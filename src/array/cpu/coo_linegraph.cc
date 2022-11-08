/**
 *  Copyright (c) 2020 by Contributors
 * @file array/cpu/coo_line_graph.cc
 * @brief COO LineGraph
 */

#include <dgl/array.h>

#include <algorithm>
#include <iterator>
#include <numeric>
#include <vector>

namespace dgl {
namespace aten {
namespace impl {

template <DGLDeviceType XPU, typename IdType>
COOMatrix COOLineGraph(const COOMatrix& coo, bool backtracking) {
  const int64_t nnz = coo.row->shape[0];
  IdType* coo_row = coo.row.Ptr<IdType>();
  IdType* coo_col = coo.col.Ptr<IdType>();
  IdArray data = COOHasData(coo)
                     ? coo.data
                     : Range(0, nnz, coo.row->dtype.bits, coo.row->ctx);
  IdType* data_data = data.Ptr<IdType>();
  std::vector<IdType> new_row;
  std::vector<IdType> new_col;

  for (int64_t i = 0; i < nnz; ++i) {
    IdType u = coo_row[i];
    IdType v = coo_col[i];
    for (int64_t j = 0; j < nnz; ++j) {
      // no self-loop
      if (i == j) continue;

      // succ_u == v
      // if not backtracking succ_u != u
      if (v == coo_row[j] && (backtracking || u != coo_col[j])) {
        new_row.push_back(data_data[i]);
        new_col.push_back(data_data[j]);
      }
    }
  }

  COOMatrix res = COOMatrix(
      nnz, nnz, NDArray::FromVector(new_row), NDArray::FromVector(new_col),
      NullArray(), false, false);
  return res;
}

template COOMatrix COOLineGraph<kDGLCPU, int32_t>(
    const COOMatrix& coo, bool backtracking);
template COOMatrix COOLineGraph<kDGLCPU, int64_t>(
    const COOMatrix& coo, bool backtracking);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
