/*!
 *  Copyright (c) 2020-2022 by Contributors
 * \file array/cpu/rowwise_pick.h
 * \brief Template implementation for rowwise pick operators.
 */
#ifndef DGL_ARRAY_CPU_ROWWISE_PICK_H_
#define DGL_ARRAY_CPU_ROWWISE_PICK_H_

#include <dgl/array.h>
#include <dmlc/omp.h>
#include <dgl/runtime/parallel_for.h>
#include <functional>
#include <algorithm>
#include <string>
#include <vector>
#include <memory>

#include "rowwise_pick_utils.h"

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
// \param num_picks Number of picks that should be done on the row.
// \param col Pointer of the column indices.
// \param data Pointer of the data indices.
// \param out_idx Picked indices in [off, off + len).
template <typename IdxType>
using PickFn = std::function<void(
    IdxType rowid, IdxType off, IdxType len, IdxType num_picks,
    const IdxType* col, const IdxType* data,
    IdxType* out_idx)>;

// User-defined function for determining the number of picks for one row.
//
// The result will be passed as the argument \a num_picks in the picking
// function with type \a PickFn<IdxType>.  Note that the result has to be
// non-negative.
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
// \return The number of entries to pick.  Must be non-negative.
template <typename IdxType>
using NumPicksFn = std::function<IdxType(
    IdxType rowid, IdxType off, IdxType len,
    const IdxType* col, const IdxType* data)>;


// Template for picking non-zero values row-wise. The implementation utilizes
// OpenMP parallelization on rows because each row performs computation independently.
template <typename IdxType>
CSRMatrix CSRRowWisePickPartial(
    CSRMatrix mat, IdArray rows, int64_t max_num_picks,
    PickFn<IdxType> pick_fn, NumPicksFn<IdxType> num_picks_fn) {
  using namespace aten;
  const IdxType* indptr = mat.indptr.Ptr<IdxType>();
  const IdxType* indices = mat.indices.Ptr<IdxType>();
  const IdxType* data = CSRHasData(mat)? mat.data.Ptr<IdxType>() : nullptr;
  const IdxType* rows_data = rows.Ptr<IdxType>();
  const int64_t num_rows = rows->shape[0];
  const auto& ctx = mat.indptr->ctx;

  IdArray picked_row_indptr = NDArray::Empty({num_rows + 1},
                                              DGLDataTypeTraits<IdxType>::dtype,
                                              ctx);
  IdArray picked_col = NDArray::Empty({num_rows * max_num_picks},
                                      DGLDataTypeTraits<IdxType>::dtype,
                                      ctx);
  IdArray picked_idx = NDArray::Empty({num_rows * max_num_picks},
                                      DGLDataTypeTraits<IdxType>::dtype,
                                      ctx);
  IdxType* picked_row_indptr_data = static_cast<IdxType*>(picked_row_indptr->data);
  picked_row_indptr_data[0] = 0;
  IdxType* picked_cdata = static_cast<IdxType*>(picked_col->data);
  IdxType* picked_idata = static_cast<IdxType*>(picked_idx->data);

  const int num_threads = omp_get_max_threads();
  std::vector<int64_t> global_prefix(num_threads+1, 0);

  // NOTE: Not using multiple runtime::parallel_for to save the overhead of launching
  // OpenMP thread groups.
  // We should benchmark the overhead of launching separate OpenMP thread groups and compare
  // with the implementation here.
#pragma omp parallel num_threads(num_threads)
  {
    const int thread_id = omp_get_thread_num();

    const int64_t start_i = thread_id * (num_rows/num_threads) +
        std::min(static_cast<int64_t>(thread_id), num_rows % num_threads);
    const int64_t end_i = (thread_id + 1) * (num_rows/num_threads) +
        std::min(static_cast<int64_t>(thread_id + 1), num_rows % num_threads);
    BUG_IF_FAIL(thread_id + 1 < num_threads || end_i == num_rows);

    // Part 1: determine the number of picks for each row as well as the indptr to return.
    for (int64_t i = start_i; i < end_i; ++i) {
      // build prefix-sum
      const IdxType rid = rows_data[i];
      const IdxType off = indptr[rid];
      const IdxType len = indptr[rid + 1] - off;

      const IdxType num_picks = num_picks_fn(rid, off, len, indices, data);
      picked_row_indptr_data[i + 1] = num_picks;
      global_prefix[thread_id + 1] += num_picks;
    }

    #pragma omp barrier
    #pragma omp master
    {
      for (int t = 0; t < num_threads; ++t)
        global_prefix[t+1] += global_prefix[t];
    }
    #pragma omp barrier

    // No need to accumulate picked_row_indptr_data[start_i] here as it is handled by
    // the global prefix in the loop below.
    for (int64_t i = start_i + 1; i < end_i; ++i)
      picked_row_indptr_data[i + 1] += picked_row_indptr_data[i];
    for (int64_t i = start_i; i < end_i; ++i)
      picked_row_indptr_data[i + 1] += global_prefix[thread_id];

    // Part 2: pick the neighbors.
    for (int64_t i = start_i; i < end_i; ++i) {
      const IdxType rid = rows_data[i];

      const IdxType off = indptr[rid];
      const IdxType len = indptr[rid + 1] - off;
      const int64_t row_offset = picked_row_indptr_data[i];
      const int64_t num_picks = picked_row_indptr_data[i + 1] - picked_row_indptr_data[i];
      if (num_picks == 0)
        continue;

      pick_fn(
          rid, off, len, static_cast<IdxType>(num_picks),
          indices, data, picked_idata + row_offset);
      for (int64_t j = 0; j < num_picks; ++j) {
        const IdxType picked = picked_idata[row_offset + j];
        picked_cdata[row_offset + j] = indices[picked];
        picked_idata[row_offset + j] = data? data[picked] : picked;
      }
    }
  }

  const int64_t new_len = global_prefix.back();
  picked_col = picked_col.CreateView({new_len}, picked_col->dtype);
  picked_idx = picked_idx.CreateView({new_len}, picked_idx->dtype);

  return CSRMatrix(mat.num_rows, mat.num_cols,
                   picked_row_indptr, picked_col, picked_idx);
}

template <typename IdxType>
COOMatrix CSRRowWisePick(
    CSRMatrix mat, IdArray rows, int64_t max_num_picks,
    PickFn<IdxType> pick_fn, NumPicksFn<IdxType> num_picks_fn) {
  CSRMatrix csr = CSRRowWisePickPartial(
      mat, rows, max_num_picks, pick_fn, num_picks_fn);
  return RowWisePickPartialCSRToCOO<IdxType>(csr, rows);
}

// Template for picking non-zero values row-wise. The implementation first slices
// out the corresponding rows and then converts it to CSR format. It then performs
// row-wise pick on the CSR matrix and rectifies the returned results.
template <typename IdxType>
COOMatrix COORowWisePick(COOMatrix mat, IdArray rows,
                         int64_t num_picks, PickFn<IdxType> pick_fn,
                         NumPicksFn<IdxType> num_picks_fn) {
  using namespace aten;
  const auto& csr = COOToCSR(COOSliceRows(mat, rows));
  const IdArray new_rows = Range(0, rows->shape[0], rows->dtype.bits, rows->ctx);
  const auto& picked = CSRRowWisePick<IdxType>(csr, new_rows, num_picks, pick_fn, num_picks_fn);
  return COOMatrix(mat.num_rows, mat.num_cols,
                   IndexSelect(rows, picked.row),  // map the row index to the correct one
                   picked.col,
                   picked.data);
}

}  // namespace impl
}  // namespace aten
}  // namespace dgl

#endif  // DGL_ARRAY_CPU_ROWWISE_PICK_H_
