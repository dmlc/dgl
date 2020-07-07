/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cpu/coo_to_simple.cc
 * \brief COO sorting
 */
#include <dgl/array.h>
#include <numeric>
#include <algorithm>
#include <vector>

namespace dgl {
namespace aten {
namespace impl {

template <DLDeviceType XPU, typename IdType>
COOMatrix COOToSimple(const COOMatrix& coo) {
  CHECK_EQ(coo.row_sorted, true) << "Input csr of COOToSimple should be sorted";
  CHECK_EQ(coo.col_sorted, true) << "Input csr of COOToSimple should be column sorted";

  const IdType *row_data = static_cast<IdType*>(coo.row->data);
  const IdType *col_data = static_cast<IdType*>(coo.col->data);
  // TODO(xiangsx): FIXME coo.data is dropped.
  std::vector<IdType> row;
  std::vector<IdType> col;
  IdType last_row = -1;
  IdType last_col = -1;

  for (int64_t i = 0; i < coo.row->shape[0]; ++i) {
    if (last_row != row_data[i] || last_col != col_data[i]) {
      last_row = row_data[i];
      last_col = col_data[i];
      row.push_back(last_row);
      col.push_back(last_col);
    }
  }

  return COOMatrix(
    coo.num_rows,
    coo.num_cols,
    IdArray::FromVector(row),
    IdArray::FromVector(col),
    NullArray(),
    true,
    true);
}

template COOMatrix COOToSimple<kDLCPU, int32_t>(const COOMatrix&);
template COOMatrix COOToSimple<kDLCPU, int64_t>(const COOMatrix&);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
