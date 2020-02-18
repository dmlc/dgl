/*!
 *  Copyright (c) 2019 by Contributors
 * \file array/cpu/coo_coalesce.cc
 * \brief COO coalescing
 */

#include <dgl/array.h>
#include <vector>

namespace dgl {

namespace aten {

namespace impl {

template <DLDeviceType XPU, typename IdType>
COOMatrix COOCoalesce(COOMatrix coo) {
  CHECK(coo.row_sorted && coo.col_sorted) << "[BUG] Coalescing is only supported on sorted matrix";

  const int64_t nnz = coo.row->shape[0];
  const IdType* coo_row_data = static_cast<IdType*>(coo.row->data);
  const IdType* coo_col_data = static_cast<IdType*>(coo.col->data);

  std::vector<IdType> new_row, new_col, new_data;
  IdType prev_row = -1, prev_col = -1;
  for (int64_t i = 0; i < nnz; ++i) {
    const IdType curr_row = coo_row_data[i];
    const IdType curr_col = coo_col_data[i];
    if (curr_row == prev_row && curr_col == prev_col) {
      ++new_data[new_data.size() - 1];
    } else {
      new_row.push_back(curr_row);
      new_col.push_back(curr_col);
      new_data.push_back(1);
      prev_row = curr_row;
      prev_col = curr_col;
    }
  }

  return COOMatrix{
      coo.num_rows, coo.num_cols, NDArray::FromVector(new_row), NDArray::FromVector(new_col),
      NDArray::FromVector(new_data), true};
}

template COOMatrix COOCoalesce<kDLCPU, int32_t>(COOMatrix);
template COOMatrix COOCoalesce<kDLCPU, int64_t>(COOMatrix);

};  // namespace impl
};  // namespace aten
};  // namespace dgl
