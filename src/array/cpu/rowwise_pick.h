/*!
 *  Copyright (c) 2020 by Contributors
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
template <typename IdxType>
using NumPicksFn = std::function<IdxType(
    IdxType rowid, IdxType off, IdxType len,
    const IdxType* col, const IdxType* data);


// User-defined function for picking elements from a range within a row.
//
// The column indices of each element is in
//   off + et_idx[et_offset+i]), where i is in [et_offset, et_offset+et_len)
//
// Similarly, the data indices are stored in
//   data[off+et_idx[et_offset+i])]
// Data index pointer could be NULL, which means data[i] == off+et_idx[et_offset+i])
//
// *ATTENTION*: This function will be invoked concurrently. Please make sure
// it is thread-safe.
//
// \param off Starting offset of this row.
// \param et_offset Starting offset of this range.
// \param cur_et The edge type.
// \param et_len Length of the range.
// \param et_idx A map from local idx to column id.
// \param data Pointer of the data indices.
// \param out_idx Picked indices in [et_offset, et_offset + et_len).
template <typename IdxType>
using RangePickFn = std::function<void(
    IdxType off, IdxType et_offset, IdxType cur_et, IdxType et_len,
    const std::vector<IdxType> &et_idx, const IdxType* data,
    IdxType* out_idx)>;

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
                                              DLDataTypeTraits<IdxType>::dtype,
                                              ctx);
  IdArray picked_col = NDArray::Empty({num_rows * max_num_picks},
                                      DLDataTypeTraits<IdxType>::dtype,
                                      ctx);
  IdArray picked_idx = NDArray::Empty({num_rows * max_num_picks},
                                      DLDataTypeTraits<IdxType>::dtype,
                                      ctx);
  IdxType* picked_row_indptr_data = static_cast<IdxType*>(picked_row_indptr->data);
  IdxType* picked_cdata = static_cast<IdxType*>(picked_col->data);
  IdxType* picked_idata = static_cast<IdxType*>(picked_idx->data);

  const int num_threads = omp_get_max_threads();
  std::vector<int64_t> global_prefix(num_threads+1, 0);

  // NOTE: Not using two runtime::parallel_for to save the overhead of launching two
  // OpenMP thread groups.
  // We should benchmark the overhead of launching two OpenMP thread groups and compare
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
    const int64_t num_local = end_i - start_i;
    for (int64_t i = start_i; i < end_i; ++i) {
      // build prefix-sum
      const int64_t local_i = i-start_i;
      const IdxType rid = rows_data[i];
      const IdxType off = indptr[rid];
      const IdxType len = indptr[rid + 1] - off;

      const IdxType num_picks = num_picks_fn(rid, off, len, indices, data);
      picked_row_indptr_data[i] = num_picks;
      global_prefix[thread_id + 1] += num_picks;
    }

    #pragma omp barrier
    #pragma omp master
    {
      for (int t = 0; t < num_threads; ++t)
        global_prefix[t+1] += global_prefix[t];
      picked_row_indptr_data[num_rows] = global_prefix[num_threads];
    }

    #pragma omp barrier
    for (int i = start_i; i < end_i; ++i)
      picked_row_indptr_data[i] += global_prefix[thread_id];

    // Part 2: pick the neighbors.
    const IdxType thread_offset = global_prefix[thread_id];
    for (int64_t i = start_i; i < end_i; ++i) {
      const IdxType rid = rows_data[i];

      const IdxType off = indptr[rid];
      const IdxType len = indptr[rid + 1] - off;
      const int64_t local_i = i - start_i;
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
  IdArray picked_rows = IdArray::Empty(
      {csr.indices->shape[0]}, csr.indices->dtype, csr.indices->ctx);
  IdxType* picked_rows_data = picked_rows.Ptr<IdxType>();
  const IdxType* indptr_data = csr.indptr.Ptr<IdxType>();
  const IdxType* rows_data = rows.Ptr<IdxType>();
  int64_t num_rows = rows->shape[0];

  runtime::parallel_for(0, num_rows, [&](size_t b, size_t e) {
    for (size_t i = b; i < e; ++i) {
      for (size_t j = indptr_data[i]; j < indptr_data[i + 1]; ++j)
        picked_rows_data[j] = rows_data[i];
    }
  });
  return COOMatrix(csr.num_rows, csr.num_cols, picked_rows, csr.indices, csr.data);
}

// Template for picking non-zero values row-wise. The implementation utilizes
// OpenMP parallelization on rows because each row performs computation independently.
template <typename IdxType>
COOMatrix CSRRowWisePerEtypePick(CSRMatrix mat, IdArray rows, IdArray etypes,
                                 const std::vector<int64_t>& num_picks, bool replace,
                                 bool etype_sorted, RangePickFn<IdxType> pick_fn) {
  using namespace aten;
  const IdxType* indptr = mat.indptr.Ptr<IdxType>();
  const IdxType* indices = mat.indices.Ptr<IdxType>();
  const IdxType* data = CSRHasData(mat)? mat.data.Ptr<IdxType>() : nullptr;
  const IdxType* rows_data = rows.Ptr<IdxType>();
  const int32_t* etype_data = etypes.Ptr<int32_t>();
  const int64_t num_rows = rows->shape[0];
  const auto& ctx = mat.indptr->ctx;
  const int64_t num_etypes = num_picks.size();
  CHECK_EQ(etypes->dtype.bits / 8, sizeof(int32_t)) << "etypes must be int32";
  std::vector<IdArray> picked_rows(rows->shape[0]);
  std::vector<IdArray> picked_cols(rows->shape[0]);
  std::vector<IdArray> picked_idxs(rows->shape[0]);

  // Check if the number of picks have the same value.
  // If so, we can potentially speed up if we have a node with total number of neighbors
  // less than the given number of picks with replace=False.
  bool same_num_pick = true;
  int64_t num_pick_value = num_picks[0];
  for (int64_t num_pick : num_picks) {
    if (num_pick_value != num_pick) {
      same_num_pick = false;
      break;
    }
  }

  runtime::parallel_for(0, num_rows, [&](size_t b, size_t e) {
    for (size_t i = b; i < e; ++i) {
      const IdxType rid = rows_data[i];
      CHECK_LT(rid, mat.num_rows);
      const IdxType off = indptr[rid];
      const IdxType len = indptr[rid + 1] - off;

      // do something here
      if (len == 0) {
        picked_rows[i] = NewIdArray(0, ctx, sizeof(IdxType) * 8);
        picked_cols[i] = NewIdArray(0, ctx, sizeof(IdxType) * 8);
        picked_idxs[i] = NewIdArray(0, ctx, sizeof(IdxType) * 8);
        continue;
      }

      // fast path
      if (same_num_pick && len <= num_pick_value && !replace) {
        IdArray rows = Full(rid, len, sizeof(IdxType) * 8, ctx);
        IdArray cols = Full(-1, len, sizeof(IdxType) * 8, ctx);
        IdArray idx = Full(-1, len, sizeof(IdxType) * 8, ctx);
        IdxType* cdata = cols.Ptr<IdxType>();
        IdxType* idata = idx.Ptr<IdxType>();
        for (int64_t j = 0; j < len; ++j) {
          cdata[j] = indices[off + j];
          idata[j] = data ? data[off + j] : off + j;
        }
        picked_rows[i] = rows;
        picked_cols[i] = cols;
        picked_idxs[i] = idx;
      } else {
        // need to do per edge type sample
        std::vector<IdxType> rows;
        std::vector<IdxType> cols;
        std::vector<IdxType> idx;

        std::vector<IdxType> et(len);
        std::vector<IdxType> et_idx(len);
        std::iota(et_idx.begin(), et_idx.end(), 0);
        for (int64_t j = 0; j < len; ++j) {
          et[j] = data ? etype_data[data[off+j]] : etype_data[off+j];
        }
        if (!etype_sorted)  // the edge type is sorted, not need to sort it
          std::sort(et_idx.begin(), et_idx.end(),
                    [&et](IdxType i1, IdxType i2) {return et[i1] < et[i2];});
        CHECK(et[et_idx[len - 1]] < num_etypes) <<
          "etype values exceed the number of fanout elements";

        IdxType cur_et = et[et_idx[0]];
        int64_t et_offset = 0;
        int64_t et_len = 1;
        for (int64_t j = 0; j < len; ++j) {
          CHECK((j + 1 == len) || (et[et_idx[j]] <= et[et_idx[j + 1]]))
              << "Edge type is not sorted. Please sort in advance or specify "
                 "'etype_sorted' as false.";
          if ((j + 1 == len) || cur_et != et[et_idx[j + 1]]) {
            // 1 end of the current etype
            // 2 end of the row
            // random pick for current etype
            if (et_len <= num_picks[cur_et] && !replace) {
              // fast path, select all
              for (int64_t k = 0; k < et_len; ++k) {
                rows.push_back(rid);
                cols.push_back(indices[off+et_idx[et_offset+k]]);
                if (data)
                  idx.push_back(data[off+et_idx[et_offset+k]]);
                else
                  idx.push_back(off+et_idx[et_offset+k]);
              }
            } else {
              IdArray picked_idx = Full(-1, num_picks[cur_et], sizeof(IdxType) * 8, ctx);
              IdxType* picked_idata = static_cast<IdxType*>(picked_idx->data);

              // need call random pick
              pick_fn(off, et_offset, cur_et,
                      et_len, et_idx,
                      data, picked_idata);
              for (int64_t k = 0; k < num_picks[cur_et]; ++k) {
                const IdxType picked = picked_idata[k];
                rows.push_back(rid);
                cols.push_back(indices[off+et_idx[et_offset+picked]]);
                if (data)
                  idx.push_back(data[off+et_idx[et_offset+picked]]);
                else
                  idx.push_back(off+et_idx[et_offset+picked]);
              }
            }

            if (j+1 == len)
              break;
            // next etype
            cur_et = et[et_idx[j+1]];
            et_offset = j+1;
            et_len = 1;
          } else {
            et_len++;
          }
        }

        picked_rows[i] = VecToIdArray(rows, sizeof(IdxType) * 8, ctx);
        picked_cols[i] = VecToIdArray(cols, sizeof(IdxType) * 8, ctx);
        picked_idxs[i] = VecToIdArray(idx, sizeof(IdxType) * 8, ctx);
      }  // end processing one row

      CHECK_EQ(picked_rows[i]->shape[0], picked_cols[i]->shape[0]);
      CHECK_EQ(picked_rows[i]->shape[0], picked_idxs[i]->shape[0]);
    }  // end processing all rows
  });

  IdArray picked_row = Concat(picked_rows);
  IdArray picked_col = Concat(picked_cols);
  IdArray picked_idx = Concat(picked_idxs);
  return COOMatrix(mat.num_rows, mat.num_cols,
                   picked_row, picked_col, picked_idx);
}

// Template for picking non-zero values row-wise. The implementation first slices
// out the corresponding rows and then converts it to CSR format. It then performs
// row-wise pick on the CSR matrix and rectifies the returned results.
template <typename IdxType>
COOMatrix COORowWisePick(COOMatrix mat, IdArray rows,
                         int64_t num_picks, PickFn<IdxType> pick_fn) {
  using namespace aten;
  const auto& csr = COOToCSR(COOSliceRows(mat, rows));
  const IdArray new_rows = Range(0, rows->shape[0], rows->dtype.bits, rows->ctx);
  const auto& picked = CSRRowWisePick<IdxType>(csr, new_rows, num_picks, pick_fn);
  return COOMatrix(mat.num_rows, mat.num_cols,
                   IndexSelect(rows, picked.row),  // map the row index to the correct one
                   picked.col,
                   picked.data);
}

// Template for picking non-zero values row-wise. The implementation first slices
// out the corresponding rows and then converts it to CSR format. It then performs
// row-wise pick on the CSR matrix and rectifies the returned results.
template <typename IdxType>
COOMatrix COORowWisePerEtypePick(COOMatrix mat, IdArray rows, IdArray etypes,
                                 const std::vector<int64_t>& num_picks, bool replace,
                                 bool etype_sorted, RangePickFn<IdxType> pick_fn) {
  using namespace aten;
  const auto& csr = COOToCSR(COOSliceRows(mat, rows));
  const IdArray new_rows = Range(0, rows->shape[0], rows->dtype.bits, rows->ctx);
  const auto& picked = CSRRowWisePerEtypePick<IdxType>(
    csr, new_rows, etypes, num_picks, replace, etype_sorted, pick_fn);
  return COOMatrix(mat.num_rows, mat.num_cols,
                   IndexSelect(rows, picked.row),  // map the row index to the correct one
                   picked.col,
                   picked.data);
}

// Template for picking non-zero values row-wise. The implementation utilizes
// OpenMP parallelization on rows because each row performs computation independently.
template <typename IdxType, typename EType>
CSRMatrix CSRRowWisePerEtypePickUnsorted(
    CSRMatrix mat, IdArray rows, IdArray etypes,
    const std::vector<int64_t>& max_num_picks, RangePickFn<IdxType> pick_fn) {
  using namespace aten;
  const DLDataType idtype = DLDataTypeTraits<IdxType>::dtype;
  const DLDataType etype_idtype = DLDataTypeTraits<EType>::dtype;
  const IdxType* indptr = mat.indptr.Ptr<IdxType>();
  const IdxType* indices = mat.indices.Ptr<IdxType>();
  const IdxType* data = CSRHasData(mat)? mat.data.Ptr<IdxType>() : nullptr;
  const IdxType* rows_data = rows.Ptr<IdxType>();
  const EType* etype_data = etypes.Ptr<EType>();
  const int64_t num_rows = rows->shape[0];
  const auto& ctx = mat.indptr->ctx;
  const int64_t num_etypes = num_picks.size();
  const int64_t total_max_num_picks = std::accumulate(
      max_num_picks.begin(), max_num_picks.end(), 0L);
  const int num_threads = omp_get_max_threads();

  // preallocate the results
  IdArray picked_row_indptr = NDArray::Empty({num_rows + 1}, idtype, ctx);
  IdArray picked_col = NDArray::Empty({num_rows * total_max_num_picks}, idtype, ctx);
  IdArray picked_idx = NDArray::Empty({num_rows * total_max_num_picks}, idtype, ctx);
  IdxType* picked_row_indptr_data = picked_row_indptr.Ptr<IdxType>();
  IdxType* picked_cdata = picked_col.Ptr<IdxType>();
  IdxType* picked_idata = picked_idx.Ptr<IdxType>();

  // the offset of incident edges with a given type
  IdArray off_etypes_per_row = NDArray::Empty(
      {num_rows * num_etypes + 1}, idtype, ctx);
  IdxType* off_etypes_per_row_data = off_etypes_per_row.Ptr<IdxType>();

  // the number of picks for each edge type at each row
  IdArray off_picks_per_etype_row = NDArray::Empty(
      {num_rows * num_etypes + 1}, idtype, ctx);
  IdxType* off_picks_per_etype_row_data = off_picks_per_etype_row.Ptr<IdxType>();

  // Determine the size of the sorted edge type index array
  IdArray et_idx_indptr = NDArray::Empty({num_rows + 1}, idtype, ctx);
  IdxType* et_idx_indptr_data = et_idx_indptr.Ptr<IdxType>();
  et_idx_indptr[0] = 0;
  for (IdxType i = 0; i < num_rows; ++i) {
    const IdxType rid = rows_data[i];
    const IdxType len = indptr[rid + 1] - indptr[rid];
    et_idx_indptr[i + 1] = et_idx_indptr[i] + len;
  }
  // Pre-allocate the argsort array of the edge type IDs.
  IdArray et_idx = NDArray::Empty({et_idx_indptr[num_rows]}, idtype, ctx);
  IdxType* et_idx_data = et_idx.Ptr<IdxType>();

  // NOTE: Not using two runtime::parallel_for to save the overhead of launching two
  // OpenMP thread groups.
  // We should benchmark the overhead of launching two OpenMP thread groups and compare
  // with the implementation here.
#pragma omp parallel num_threads(num_threads)
  {
    const int thread_id = omp_get_thread_num();
    const int64_t start_i = thread_id * (num_rows/num_threads) +
        std::min(static_cast<int64_t>(thread_id), num_rows % num_threads);
    const int64_t end_i = (thread_id + 1) * (num_rows/num_threads) +
        std::min(static_cast<int64_t>(thread_id + 1), num_rows % num_threads);
    BUG_IF_FAIL(thread_id + 1 < num_threads || end_i == num_rows);

    // Part 1: sort edge type IDs per node if necessary
    for (int64_t i = start_i; i < end_i; ++i) {
      const IdxType rid = rows_data[i];
      std::iota(
          et_idx_data + et_idx_indptr_data[i],
          et_idx_data + et_idx_indptr_data[i + 1],
          indptr[rid]);
      std::sort(
          et_idx_data + et_idx_indptr_data[i],
          et_idx_data + et_idx_indptr_data[i + 1],
          [&etype_data](IdxType i1, IdxType i2) {
            return etype_data[i1] < etype_data[i2];
          });
    }

    // Part 2: determine the number of incident edges with the same edge type per node
    for (int64_t i = start_i; i < end_i; ++i) {
      const IdxType rid = rows_data[i];
      for (int64_t j = et_idx_indptr_data[i]; j < et_idx_indptr_data[i + 1]; ++j) {
        const IdxType et = etype_data[et_idx_data[j]];
        CHECK_LT(et, num_etypes) << "Length of fanout list is " << num_etypes
          << " but found edge type ID " << et << " that is larger.";
        ++off_etypes_per_row_data[i * num_etypes + et + 1];
      }
    }

#pragma omp barrier
#pragma omp master
    {
      off_etypes_per_row_data[0] = 0;
      for (int64_t i = 0; i < num_rows * num_etypes; ++i)
        off_etypes_per_row_data[i + 1] += off_etypes_per_row_data[i];
    }
#pragma omp barrier

    // Part 3: determine the number of picks for each row and each edge type as well
    // as the indptr to return.
    const int64_t num_local = end_i - start_i;
    for (int64_t i = start_i; i < end_i; ++i) {
      const int64_t local_i = i - start_i;
      const IdxType rid = rows_data[i];
      const IdxType off = indptr[rid];
      const IdxType len = indptr[rid + 1] - off;
      int64_t prev_j = off;
      for (int64_t et = 0; et < num_etypes; ++et) {
        const IdxType num_picks = num_picks_fn(
            rid, off, len, et, indices, data,
            et_idx_data + off_etypes_per_row_data[i * num_etypes + et]);
        off_picks_per_etype_row[i * num_etypes + et + 1] = num_picks;
      }
    }

#pragma omp barrier
#pragma omp master
    {
      off_picks_per_etype_row[0] = 0;
      for (int64_t i = 0; i < num_rows * num_etypes; ++i)
        off_picks_per_etype_row[i + 1] += off_picks_per_etype_row[i];
    }
#pragma omp barrier

    // Part 4: pick the neighbors
    // TODO
  }
}

}  // namespace impl
}  // namespace aten
}  // namespace dgl

#endif  // DGL_ARRAY_CPU_ROWWISE_PICK_H_
