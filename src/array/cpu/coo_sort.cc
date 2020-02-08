/*!
 *  Copyright (c) 2019 by Contributors
 * \file array/cpu/coo_sort.cc
 * \brief COO sorting
 */

#include <dgl/array.h>
#include <numeric>
#include <algorithm>
#include <vector>

namespace dgl {

namespace aten {

namespace impl {

template <DLDeviceType XPU, typename IdType, typename DType>
COOMatrix COOSort(COOMatrix coo) {
  const int64_t nnz = coo.row->shape[0];
  const IdType* coo_row_data = static_cast<IdType*>(coo.row->data);
  const IdType* coo_col_data = static_cast<IdType*>(coo.col->data);
  DType* coo_data;

  // If COO Data array is empty, create and populate it with 0..NNZ-1 (i.e. default)
  if (!COOHasData(coo)) {
    // Assumes IDType and DType are equivalent
    coo.data = NewIdArray(nnz, coo.row->ctx, coo.row->dtype.bits);
    coo_data = static_cast<DType*>(coo.data->data);
    std::iota(coo_data, coo_data + nnz, 0);
  } else {
    coo_data = static_cast<DType*>(coo.data->data);
  }

  // Argsort
  std::vector<IdType> shuffle(nnz);
  std::iota(shuffle.begin(), shuffle.end(), 0);
  std::sort(
      shuffle.begin(),
      shuffle.end(),
      [coo_row_data, coo_col_data, coo_data](IdType a, IdType b) {
        return (
            (coo_row_data[a] < coo_row_data[b]) &&
            (coo_col_data[a] < coo_col_data[b]));
      });

  IdArray new_row = IdArray::Empty({nnz}, coo.row->dtype, coo.row->ctx);
  IdArray new_col = IdArray::Empty({nnz}, coo.col->dtype, coo.col->ctx);
  NDArray new_data = NDArray::Empty({nnz}, coo.data->dtype, coo.data->ctx);
  IdType* new_row_data = static_cast<IdType*>(new_row->data);
  IdType* new_col_data = static_cast<IdType*>(new_col->data);
  DType *new_data_data = static_cast<DType*>(new_data->data);

  // Reorder according to shuffle
  for (IdType i = 0; i < nnz; ++i) {
    new_row_data[i] = coo_row_data[shuffle[i]];
    new_col_data[i] = coo_col_data[shuffle[i]];
    new_data_data[i] = coo_data[shuffle[i]];
  }

  return COOMatrix{
      coo.num_rows, coo.num_cols, std::move(new_row), std::move(new_col),
      std::move(new_data), true};
}

template COOMatrix COOSort<kDLCPU, int32_t, int32_t>(COOMatrix);
template COOMatrix COOSort<kDLCPU, int64_t, int64_t>(COOMatrix);

};  // namespace impl
};  // namespace aten
};  // namespace dgl
