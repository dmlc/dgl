/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cpu/rowwise_pick.h
 * \brief Template implementation for rowwise pick operators.
 */
#ifndef DGL_ARRAY_CPU_ROWWISE_PICK_H_
#define DGL_ARRAY_CPU_ROWWISE_PICK_H_

#include <dgl/array.h>
#include <functional>

namespace dgl {
namespace aten {
namespace impl {

// Lambda function for picking elements from one row.
// \param rowid The row to pick from.
// \param out_row Row index of the picked element.
// \param out_col Col index of the picked element.
// \param out_idx Data index of the picked element.
template <typename IdxType>
using PickFn = std::function<void(
    IdxType rowid, IdxType* out_row, IdxType* out_col, IdxType* out_idx)>;

// Template for picking non-zero values row-wise. The implementation utilizes
// OpenMP parallelization on rows because each row performs computation independently.
template <typename IdxType>
COOMatrix CSRRowWisePick(CSRMatrix mat, IdArray rows,
                         int64_t num_picks, bool replace, PickFn<IdxType> pick_fn) {
  const IdxType* indptr = static_cast<IdxType*>(mat.indptr->data);
  const IdxType* indices = static_cast<IdxType*>(mat.indices->data);
  const IdxType* rows_data = static_cast<IdxType*>(rows->data);
  const int64_t num_rows = rows->shape[0];
  const auto& ctx = mat.indptr->ctx;

  // To leverage OMP parallelization, we create two arrays to store
  // picked src and dst indices. Each array is of length num_rows * num_picks.
  // For rows whose nnz < num_picks, the indices are padded with -1.
  //
  // We check whether all the given rows
  // have at least num_picks number of nnz when replace is false.
  //
  // If the check holds, remove -1 elements by remove_if operation, which simply
  // moves valid elements to the head of arrays and create a view of the original
  // array. The implementation consumes a little extra memory than the actual requirement.
  //
  // Otherwise, directly use the row and col arrays to construct the result COO matrix.
  IdArray picked_row = aten::Full(-1, num_rows * num_picks, sizeof(IdxType) * 8, ctx);
  IdArray picked_col = aten::Full(-1, num_rows * num_picks, sizeof(IdxType) * 8, ctx);
  IdArray picked_idx = aten::Full(-1, num_rows * num_picks, sizeof(IdxType) * 8, ctx);
  IdxType* picked_rdata = static_cast<IdxType*>(picked_row->data);
  IdxType* picked_cdata = static_cast<IdxType*>(picked_col->data);
  IdxType* picked_idata = static_cast<IdxType*>(picked_idx->data);

  bool all_has_fanout = true;
  if (replace) {
    all_has_fanout = true;
  } else {
#pragma omp parallel for reduction(&&:all_has_fanout)
    for (int64_t i = 0; i < num_rows; ++i) {
      const IdxType rid = rows_data[i];
      const IdxType len = indptr[rid + 1] - indptr[rid];
      all_has_fanout = all_has_fanout && (len >= num_picks);
    }
  }

#pragma omp parallel for
  for (int64_t i = 0; i < num_rows; ++i) {
    const IdxType rid = rows_data[i];
    const IdxType off = indptr[rid];
    const IdxType len = indptr[rid + 1] - off;
    if (len <= num_picks && !replace) {
      // nnz <= num_picks and w/o replacement, take all nnz
      for (int64_t j = 0; j < len; ++j) {
        picked_rdata[i * num_picks + j] = rid;
        picked_cdata[i * num_picks + j] = indices[off + j];
      }
    } else {
      pick_fn(rid,
              picked_rdata + i * num_picks,
              picked_cdata + i * num_picks,
              picked_idata + i * num_picks);
    }
  }

  if (!all_has_fanout) {
    // correct the array by remove_if
    IdxType* new_row_end = std::remove_if(picked_rdata, picked_rdata + num_rows * num_picks,
                                          [] (IdxType i) { return i == -1; });
    IdxType* new_col_end = std::remove_if(picked_cdata, picked_cdata + num_rows * num_picks,
                                          [] (IdxType i) { return i == -1; });
    const int64_t new_len = (new_row_end - picked_rdata);
    CHECK_EQ(new_col_end - picked_cdata, new_len);
    CHECK_LT(new_len, num_rows * num_picks);
    picked_row = picked_row.CreateView({new_len}, picked_row->dtype);
    picked_col = picked_col.CreateView({new_len}, picked_col->dtype);
  }

  return COOMatrix(mat.num_rows, mat.num_cols,
                   picked_row, picked_col, picked_idx, true);
}


template <typename IdxType>
COOMatrix COORowWisePick(COOMatrix mat, IdArray rows,
                         int64_t num_picks, bool replace, PickFn<IdxType> pick_fn) {
  using namespace aten;
  const CSRMatrix csr = COOToCSR(COOSliceRows(mat, rows));
  const COOMatrix picked = CSRRowWisePick<IdxType>(csr, rows, num_picks, replace, pick_fn);
  // map the row index to the correct one
  const IdArray corrected_row = IndexSelect(rows, picked.row);
  // map the data index to the correct one

  return COOMatrix(mat.num_rows, mat.num_cols,
                   picked_row, picked_col, picked_idx, true);
}

}  // namespace impl
}  // namespace aten
}  // namespace dgl

#endif  // DGL_ARRAY_CPU_ROWWISE_PICK_H_
