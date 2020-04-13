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

// User-defined function for picking elements from one row.
//
// The column indices of the given row are stored in
//   [col + off, col + off + len)
//
// Similarly, the data indices are stored in
//   [data + off, data + off + len)
// Data index pointer could be NULL, which means data[i] == i
//
// *ATTENTION*: This function will be invoked concurrently. Please make sure
// it is thread-safe.
//
// \param rowid The row to pick from.
// \param off Starting offset of this row.
// \param len NNZ of the row.
// \param col Pointer of the column indices.
// \param data Pointer of the data indices.
// \param out_idx Picked indices in [off, off + len).
template <typename IdxType>
using PickFn = std::function<void(
    IdxType rowid, IdxType off, IdxType len,
    const IdxType* col, const IdxType* data,
    IdxType* out_idx)>;

// Template for picking non-zero values row-wise. The implementation utilizes
// OpenMP parallelization on rows because each row performs computation independently.
template <typename IdxType>
COOMatrix CSRRowWisePick(CSRMatrix mat, IdArray rows,
                         int64_t num_picks, bool replace, PickFn<IdxType> pick_fn) {
  using namespace aten;
  const IdxType* indptr = static_cast<IdxType*>(mat.indptr->data);
  const IdxType* indices = static_cast<IdxType*>(mat.indices->data);
  const IdxType* data = CSRHasData(mat)? static_cast<IdxType*>(mat.data->data) : nullptr;
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
  //
  // [02/29/2020 update]: OMP is disabled for now since batch-wise parallelism is more
  //   significant. (minjie)
  IdArray picked_row = Full(-1, num_rows * num_picks, sizeof(IdxType) * 8, ctx);
  IdArray picked_col = Full(-1, num_rows * num_picks, sizeof(IdxType) * 8, ctx);
  IdArray picked_idx = Full(-1, num_rows * num_picks, sizeof(IdxType) * 8, ctx);
  IdxType* picked_rdata = static_cast<IdxType*>(picked_row->data);
  IdxType* picked_cdata = static_cast<IdxType*>(picked_col->data);
  IdxType* picked_idata = static_cast<IdxType*>(picked_idx->data);

  bool all_has_fanout = true;
#pragma omp parallel for reduction(&&:all_has_fanout)
  for (int64_t i = 0; i < num_rows; ++i) {
    const IdxType rid = rows_data[i];
    const IdxType len = indptr[rid + 1] - indptr[rid];
    // If a node has no neighbor then all_has_fanout must be false even if replace is
    // true.
    all_has_fanout = all_has_fanout && (len >= (replace ? 1 : num_picks));
  }

#pragma omp parallel for
  for (int64_t i = 0; i < num_rows; ++i) {
    const IdxType rid = rows_data[i];
    CHECK_LT(rid, mat.num_rows);
    const IdxType off = indptr[rid];
    const IdxType len = indptr[rid + 1] - off;
    if (len == 0)
      continue;

    if (len <= num_picks && !replace) {
      // nnz <= num_picks and w/o replacement, take all nnz
      for (int64_t j = 0; j < len; ++j) {
        picked_rdata[i * num_picks + j] = rid;
        picked_cdata[i * num_picks + j] = indices[off + j];
        picked_idata[i * num_picks + j] = data? data[off + j] : off + j;
      }
    } else {
      pick_fn(rid, off, len,
              indices, data,
              picked_idata + i * num_picks);
      for (int64_t j = 0; j < num_picks; ++j) {
        const IdxType picked = picked_idata[i * num_picks + j];
        picked_rdata[i * num_picks + j] = rid;
        picked_cdata[i * num_picks + j] = indices[picked];
        picked_idata[i * num_picks + j] = data? data[picked] : picked;
      }
    }
  }

  if (!all_has_fanout) {
    // correct the array by remove_if
    IdxType* new_row_end = std::remove_if(picked_rdata, picked_rdata + num_rows * num_picks,
                                          [] (IdxType i) { return i == -1; });
    IdxType* new_col_end = std::remove_if(picked_cdata, picked_cdata + num_rows * num_picks,
                                          [] (IdxType i) { return i == -1; });
    IdxType* new_idx_end = std::remove_if(picked_idata, picked_idata + num_rows * num_picks,
                                          [] (IdxType i) { return i == -1; });
    const int64_t new_len = (new_row_end - picked_rdata);
    CHECK_EQ(new_col_end - picked_cdata, new_len);
    CHECK_EQ(new_idx_end - picked_idata, new_len);
    picked_row = picked_row.CreateView({new_len}, picked_row->dtype);
    picked_col = picked_col.CreateView({new_len}, picked_col->dtype);
    picked_idx = picked_idx.CreateView({new_len}, picked_idx->dtype);
  }

  return COOMatrix(mat.num_rows, mat.num_cols,
                   picked_row, picked_col, picked_idx);
}

// Template for picking non-zero values row-wise. The implementation first slices
// out the corresponding rows and then converts it to CSR format. It then performs
// row-wise pick on the CSR matrix and rectifies the returned results.
template <typename IdxType>
COOMatrix COORowWisePick(COOMatrix mat, IdArray rows,
                         int64_t num_picks, bool replace, PickFn<IdxType> pick_fn) {
  using namespace aten;
  const auto& csr = COOToCSR(COOSliceRows(mat, rows));
  const IdArray new_rows = Range(0, rows->shape[0], rows->dtype.bits, rows->ctx);
  const auto& picked = CSRRowWisePick<IdxType>(csr, new_rows, num_picks, replace, pick_fn);
  return COOMatrix(mat.num_rows, mat.num_cols,
                   IndexSelect(rows, picked.row),  // map the row index to the correct one
                   picked.col,
                   picked.data);
}

}  // namespace impl
}  // namespace aten
}  // namespace dgl

#endif  // DGL_ARRAY_CPU_ROWWISE_PICK_H_
