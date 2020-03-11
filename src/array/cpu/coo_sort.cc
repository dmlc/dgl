/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cpu/coo_sort.cc
 * \brief COO sorting
 */
#include <dgl/array.h>
#ifdef PARALLEL_ALGORITHMS
#include <parallel/algorithm>
#endif
#include <numeric>
#include <algorithm>
#include <vector>

namespace dgl {
namespace aten {
namespace impl {

template <DLDeviceType XPU, typename IdType>
COOMatrix COOSort(COOMatrix coo, bool sort_column) {
  const int64_t nnz = coo.row->shape[0];
  const IdType* coo_row_data = static_cast<IdType*>(coo.row->data);
  const IdType* coo_col_data = static_cast<IdType*>(coo.col->data);

  // Argsort
  IdArray new_row = IdArray::Empty({nnz}, coo.row->dtype, coo.row->ctx);
  IdArray new_col = IdArray::Empty({nnz}, coo.col->dtype, coo.col->ctx);
  IdArray new_idx = IdArray::Empty({nnz}, coo.row->dtype, coo.row->ctx);
  IdType* new_row_data = static_cast<IdType*>(new_row->data);
  IdType* new_col_data = static_cast<IdType*>(new_col->data);
  IdType* new_idx_data = static_cast<IdType*>(new_idx->data);
  std::iota(new_idx_data, new_idx_data + nnz, 0);
  if (sort_column) {
#ifdef PARALLEL_ALGORITHMS
    __gnu_parallel::sort(
#else
    std::sort(
#endif
        new_idx_data,
        new_idx_data + nnz,
        [coo_row_data, coo_col_data](const IdType a, const IdType b) {
          return (coo_row_data[a] != coo_row_data[b]) ?
            (coo_row_data[a] < coo_row_data[b]) :
            (coo_col_data[a] < coo_col_data[b]);
        });
  } else {
#ifdef PARALLEL_ALGORITHMS
    __gnu_parallel::sort(
#else
    std::sort(
#endif
        new_idx_data,
        new_idx_data + nnz,
        [coo_row_data](const IdType a, const IdType b) {
          return coo_row_data[a] < coo_row_data[b];
        });
  }

  // Reorder according to shuffle
#pragma omp parallel for
  for (IdType i = 0; i < nnz; ++i) {
    new_row_data[i] = coo_row_data[new_idx_data[i]];
    new_col_data[i] = coo_col_data[new_idx_data[i]];
  }

  if (COOHasData(coo)) {
    const IdType* coo_data_data = static_cast<IdType*>(coo.data->data);
    IdArray new_data = IdArray::Empty({nnz}, coo.row->dtype, coo.row->ctx);
    IdType* new_data_data = static_cast<IdType*>(new_data->data);
#pragma omp parallel for
    for (IdType i = 0; i < nnz; ++i) {
      new_data_data[i] = coo_data_data[new_idx_data[i]];
    }

    new_idx = new_data;
  }

  return COOMatrix{
      coo.num_rows, coo.num_cols, std::move(new_row), std::move(new_col),
      std::move(new_idx), true, sort_column};
}

template COOMatrix COOSort<kDLCPU, int32_t>(COOMatrix, bool);
template COOMatrix COOSort<kDLCPU, int64_t>(COOMatrix, bool);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
