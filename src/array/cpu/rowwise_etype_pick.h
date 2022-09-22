/*!
 *  Copyright (c) 2022 by Contributors
 * \file array/cpu/rowwise_etype_pick.h
 * \brief Template implementation for rowwise pick operators.
 */
#ifndef DGL_ARRAY_CPU_ROWWISE_ETYPE_PICK_H_
#define DGL_ARRAY_CPU_ROWWISE_ETYPE_PICK_H_

#include <dgl/array.h>
#include <dmlc/omp.h>
#include <dgl/runtime/parallel_for.h>
#include <functional>
#include <algorithm>
#include <string>
#include <vector>
#include <memory>
#include <numeric>

#include "rowwise_pick_utils.h"

namespace dgl {
namespace aten {
namespace impl {


// User-defined function for picking elements within a row and one edge type.
//
// Data index pointer could be NULL, which means data[i] == off+et_idx[et_offset+i])
//
// *ATTENTION*: This function will be invoked concurrently. Please make sure
// it is thread-safe.
//
// \param rid The row to pick from.
// \param off Starting offset of this row.
// \param len NNZ of the row.
// \param num_picks Number of picks that should be done on the row.
// \param col Pointer of the column indices.
// \param data Pointer of the data indices.
// \param et The current edge type.
// \param et_off Starting offset of this range.
// \param et_len Length of the range.
// \param et_idx A map from local idx to column id.
//               etypes[data[et_idx[et_off:et_off + et_len]]] == et.
//               nullptr represents identity mapping (i.e. et_idx[i] == i).
// \param out_idx Picked indices putting into [et_off, et_off + et_len).
template <typename IdxType, typename EType>
using ETypePickFn = std::function<void(
    IdxType rid, IdxType off, IdxType len, IdxType num_picks,
    const IdxType* col, const IdxType* data,
    EType et, IdxType et_off, IdxType et_len, const IdxType* et_idx,
    IdxType* out_idx)>;


// User-defined function for determining the number of picks for one row and one edge type.
//
// Data index pointer could be NULL, which means data[i] == off+et_idx[et_offset+i])
//
// *ATTENTION*: This function will be invoked concurrently. Please make sure
// it is thread-safe.
//
// \param rid The row to pick from.
// \param off Starting offset of this row.
// \param len NNZ of the row.
// \param col Pointer of the column indices.
// \param data Pointer of the data indices.
// \param et The current edge type.
// \param et_off Starting offset of this range.
// \param et_len Length of the range.
// \param et_idx A map from local idx to column id.
//               etypes[data[et_idx[et_off:et_off + et_len]]] == et.
//               nullptr represents identity mapping (i.e. et_idx[i] == i).
// \return The number of entries to pick.  Must be non-negative.
template <typename IdxType, typename EType>
using ETypeNumPicksFn = std::function<IdxType(
    IdxType rid, IdxType off, IdxType len, const IdxType* col, const IdxType* data,
    EType et, IdxType et_off, IdxType et_len, const IdxType* et_idx)>;

// Template for picking non-zero values row-wise. The implementation utilizes
// OpenMP parallelization on rows because each row performs computation independently.
template <typename IdxType, typename EType>
CSRMatrix CSRRowWisePerEtypePickPartial(
    CSRMatrix mat, IdArray rows, IdArray etypes,
    const std::vector<int64_t>& max_num_picks,
    bool etype_sorted,
    ETypePickFn<IdxType, EType> pick_fn,
    ETypeNumPicksFn<IdxType, EType> num_picks_fn) {
  using namespace aten;
  const DGLDataType idtype = DGLDataTypeTraits<IdxType>::dtype;
  const IdxType* indptr = mat.indptr.Ptr<IdxType>();
  const IdxType* indices = mat.indices.Ptr<IdxType>();
  const IdxType* data = CSRHasData(mat)? mat.data.Ptr<IdxType>() : nullptr;
  const IdxType* rows_data = rows.Ptr<IdxType>();
  const EType* etype_data = etypes.Ptr<EType>();
  const int64_t num_rows = rows->shape[0];
  const auto& ctx = mat.indptr->ctx;
  const int64_t num_etypes = max_num_picks.size();
  const int64_t total_max_num_picks = std::accumulate(
      max_num_picks.begin(), max_num_picks.end(), 0L);

  // preallocate the results
  IdArray picked_row_indptr = NDArray::Empty({num_rows + 1}, idtype, ctx);
  IdArray picked_col = NDArray::Empty({num_rows * total_max_num_picks}, idtype, ctx);
  IdArray picked_idx = NDArray::Empty({num_rows * total_max_num_picks}, idtype, ctx);
  IdxType* picked_row_indptr_data = picked_row_indptr.Ptr<IdxType>();
  IdxType* picked_cdata = picked_col.Ptr<IdxType>();
  IdxType* picked_idata = picked_idx.Ptr<IdxType>();

  // the offset of incident edges with a given type at each row
  IdArray off_etypes_per_row = Full(0, num_rows * num_etypes + 1, idtype.bits, ctx);
  IdxType* off_etypes_per_row_data = off_etypes_per_row.Ptr<IdxType>();

  // the offset of picks for each edge type at each row
  IdArray off_picked_per_row = NDArray::Empty(
      {num_rows * num_etypes + 1}, idtype, ctx);
  IdxType* off_picked_per_row_data = off_picked_per_row.Ptr<IdxType>();

  // Determine the size of the sorted edge type index array
  // NOTE: these variables are only used if etype_sorted is False.
  IdArray et_idx_indptr, et_idx;
  IdxType* et_idx_indptr_data = nullptr;
  IdxType* et_idx_data = nullptr;
  if (!etype_sorted) {
    et_idx_indptr = NDArray::Empty({num_rows + 1}, idtype, ctx);
    et_idx_indptr_data = et_idx_indptr.Ptr<IdxType>();
    et_idx_indptr_data[0] = 0;
    for (IdxType i = 0; i < num_rows; ++i) {
      const IdxType rid = rows_data[i];
      const IdxType len = indptr[rid + 1] - indptr[rid];
      et_idx_indptr_data[i + 1] = et_idx_indptr_data[i] + len;
    }
    // Pre-allocate the argsort array of the edge type IDs.
    et_idx = NDArray::Empty({et_idx_indptr_data[num_rows]}, idtype, ctx);
    et_idx_data = et_idx.Ptr<IdxType>();
  }

  runtime::parallel_for(0, num_rows, [=] (size_t start_i, size_t end_i) {
    // Step 1: sort edge type IDs per node if necessary
    if (!etype_sorted) {
      for (size_t i = start_i; i < end_i; ++i) {
        const IdxType rid = rows_data[i];
        std::iota(
            et_idx_data + et_idx_indptr_data[i],
            et_idx_data + et_idx_indptr_data[i + 1],
            indptr[rid]);
        std::sort(
            et_idx_data + et_idx_indptr_data[i],
            et_idx_data + et_idx_indptr_data[i + 1],
            [etype_data, data](IdxType i1, IdxType i2) {
              return etype_data[data ? data[i1] : i1] < etype_data[data ? data[i2] : i2];
            });
      }
    }

    // Step 2: determine the number of incident edges with the same edge type per node
    for (size_t i = start_i; i < end_i; ++i) {
      int64_t start_j = !etype_sorted ? et_idx_indptr_data[i] : indptr[i];
      int64_t end_j = !etype_sorted ? et_idx_indptr_data[i + 1] : indptr[i + 1];
      for (int64_t j = start_j; j < end_j; ++j) {
        const IdxType loc = !etype_sorted ? et_idx_data[j] : j;
        const IdxType eid = data ? data[loc] : loc;
        const EType et = etype_data[eid];

        CHECK_LT(et, num_etypes) << "Length of fanout list is " << num_etypes
          << " but found edge type ID " << et << " that is larger.";
        if (j != end_j - 1) {
          const IdxType nextloc = !etype_sorted ? et_idx_data[j + 1] : j + 1;
          const IdxType next_eid = data ? data[nextloc] : nextloc;
          // Must hold if etype_sorted is False since we sorted it in Step 1.
          CHECK_LE(et, etype_data[next_eid]) << "Edge type IDs not sorted by row.";
        }
        ++off_etypes_per_row_data[i * num_etypes + et + 1];
      }
    }

    #pragma omp master
    {
      off_etypes_per_row_data[0] = 0;
      for (int64_t i = 0; i < num_rows * num_etypes; ++i)
        off_etypes_per_row_data[i + 1] += off_etypes_per_row_data[i];
    }

    // Step 3: determine the number of picks for each row and each edge type as well
    // as the indptr to return.
    for (size_t i = start_i; i < end_i; ++i) {
      const IdxType rid = rows_data[i];
      const IdxType off = indptr[rid];
      const IdxType len = indptr[rid + 1] - off;
      for (int64_t et = 0; et < num_etypes; ++et) {
        const IdxType et_off = off_etypes_per_row_data[i * num_etypes + et];
        const IdxType et_len = off_etypes_per_row_data[i * num_etypes + et + 1] - et_off;
        const IdxType num_picks = num_picks_fn(
            rid, off, len, indices, data, et, et_off, et_len, et_idx_data);
        off_picked_per_row_data[i * num_etypes + et + 1] = num_picks;
      }
    }

    #pragma omp master
    {
      off_picked_per_row_data[0] = 0;
      for (int64_t i = 0; i < num_rows * num_etypes; ++i)
        off_picked_per_row_data[i + 1] += off_picked_per_row_data[i];
      picked_row_indptr_data[num_rows] = off_picked_per_row_data[num_rows * num_etypes];
    }

    for (size_t i = start_i; i < end_i; ++i)
      picked_row_indptr_data[i] = off_picked_per_row_data[i * num_etypes];

    for (size_t i = start_i; i < end_i; ++i) {
      const IdxType rid = rows_data[i];
      const IdxType off = indptr[rid];
      const IdxType len = indptr[rid + 1] - off;
      for (int64_t et = 0; et < num_etypes; ++et) {
        const IdxType et_off = off_etypes_per_row_data[i * num_etypes + et];
        const IdxType et_len = off_etypes_per_row_data[i * num_etypes + et + 1] - et_off;
        const IdxType pick_off = off_picked_per_row_data[i * num_etypes + et];
        const IdxType num_picks = off_picked_per_row_data[i * num_etypes + et + 1] - pick_off;
        pick_fn(
            rid, off, len, num_picks, indices, data, et, et_off, et_len, et_idx_data,
            picked_idata + pick_off);
        for (int64_t j = 0; j < num_picks; ++j) {
          const IdxType picked = picked_idata[pick_off + j];
          picked_cdata[pick_off + j] = indices[picked];
          picked_idata[pick_off + j] = data ? data[picked] : picked;
        }
      }
    }
  });

  const int64_t new_len = off_picked_per_row_data[num_rows * num_etypes];
  picked_col = picked_col.CreateView({new_len}, picked_col->dtype);
  picked_idx = picked_idx.CreateView({new_len}, picked_idx->dtype);

  return CSRMatrix(num_rows, mat.num_cols, picked_row_indptr, picked_col, picked_idx);
}

template <typename IdxType, typename EType>
COOMatrix CSRRowWisePerEtypePick(
    CSRMatrix mat, IdArray rows, IdArray etypes,
    const std::vector<int64_t>& max_num_picks, bool etype_sorted,
    ETypePickFn<IdxType, EType> pick_fn, ETypeNumPicksFn<IdxType, EType> num_picks_fn) {
  CSRMatrix csr = CSRRowWisePerEtypePickPartial<IdxType, EType>(
      mat, rows, etypes, max_num_picks, etype_sorted, pick_fn, num_picks_fn);
  return RowWisePickPartialCSRToCOO<IdxType>(csr, rows, mat.num_rows);
}

// Template for picking non-zero values row-wise. The implementation first slices
// out the corresponding rows and then converts it to CSR format. It then performs
// row-wise pick on the CSR matrix and rectifies the returned results.
template <typename IdxType, typename EType>
COOMatrix COORowWisePerEtypePick(COOMatrix mat, IdArray rows, IdArray etypes,
                                 const std::vector<int64_t>& max_num_picks,
                                 bool etype_sorted, ETypePickFn<IdxType, EType> pick_fn,
                                 ETypeNumPicksFn<IdxType, EType> num_picks_fn) {
  using namespace aten;
  const auto& csr = COOToCSR(COOSliceRows(mat, rows));
  const IdArray new_rows = Range(0, rows->shape[0], rows->dtype.bits, rows->ctx);
  const auto& picked = CSRRowWisePerEtypePick<IdxType, EType>(
    csr, new_rows, etypes, max_num_picks, etype_sorted, pick_fn, num_picks_fn);
  return COOMatrix(mat.num_rows, mat.num_cols,
                   IndexSelect(rows, picked.row),  // map the row index to the correct one
                   picked.col,
                   picked.data);
}

}  // namespace impl
}  // namespace aten
}  // namespace dgl

#endif  // DGL_ARRAY_CPU_ROWWISE_ETYPE_PICK_H_
