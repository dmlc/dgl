/*!
 *  Copyright (c) 2019 by Contributors
 * \file array/cpu/spmat_op_impl.cc
 * \brief CPU implementation of COO sparse matrix operators
 */
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <tuple>
#include "array_utils.h"

namespace dgl {

using runtime::NDArray;

namespace aten {
namespace impl {

/*
 * TODO(BarclayII):
 * For row-major sorted COOs, we have faster implementation with binary search,
 * sorted search, etc.  Later we should benchmark how much we can gain with
 * sorted COOs on hypersparse graphs.
 */

///////////////////////////// COOIsNonZero /////////////////////////////

template <DLDeviceType XPU, typename IdType>
bool COOIsNonZero(COOMatrix coo, int64_t row, int64_t col) {
  CHECK(row >= 0 && row < coo.num_rows) << "Invalid row index: " << row;
  CHECK(col >= 0 && col < coo.num_cols) << "Invalid col index: " << col;
  const IdType* coo_row_data = static_cast<IdType*>(coo.row->data);
  const IdType* coo_col_data = static_cast<IdType*>(coo.col->data);
  for (int64_t i = 0; i < coo.row->shape[0]; ++i) {
    if (coo_row_data[i] == row && coo_col_data[i] == col)
      return true;
  }
  return false;
}

template bool COOIsNonZero<kDLCPU, int32_t>(COOMatrix, int64_t, int64_t);
template bool COOIsNonZero<kDLCPU, int64_t>(COOMatrix, int64_t, int64_t);

template <DLDeviceType XPU, typename IdType>
NDArray COOIsNonZero(COOMatrix coo, NDArray row, NDArray col) {
  const auto rowlen = row->shape[0];
  const auto collen = col->shape[0];
  const auto rstlen = std::max(rowlen, collen);
  NDArray rst = NDArray::Empty({rstlen}, row->dtype, row->ctx);
  IdType* rst_data = static_cast<IdType*>(rst->data);
  const IdType* row_data = static_cast<IdType*>(row->data);
  const IdType* col_data = static_cast<IdType*>(col->data);
  const int64_t row_stride = (rowlen == 1 && collen != 1) ? 0 : 1;
  const int64_t col_stride = (collen == 1 && rowlen != 1) ? 0 : 1;
  const int64_t kmax = std::max(rowlen, collen);
#pragma omp parallel for
  for (int64_t k = 0; k < kmax; ++k) {
    int64_t i = row_stride * k;
    int64_t j = col_stride * k;
    rst_data[k] = COOIsNonZero<XPU, IdType>(coo, row_data[i], col_data[j])? 1 : 0;
  }
  return rst;
}

template NDArray COOIsNonZero<kDLCPU, int32_t>(COOMatrix, NDArray, NDArray);
template NDArray COOIsNonZero<kDLCPU, int64_t>(COOMatrix, NDArray, NDArray);

///////////////////////////// COOHasDuplicate /////////////////////////////

template <DLDeviceType XPU, typename IdType>
bool COOHasDuplicate(COOMatrix coo) {
  std::unordered_set<std::pair<IdType, IdType>, PairHash> hashmap;
  const IdType* src_data = static_cast<IdType*>(coo.row->data);
  const IdType* dst_data = static_cast<IdType*>(coo.col->data);
  const auto nnz = coo.row->shape[0];
  for (IdType eid = 0; eid < nnz; ++eid) {
    const auto& p = std::make_pair(src_data[eid], dst_data[eid]);
    if (hashmap.count(p)) {
      return true;
    } else {
      hashmap.insert(p);
    }
  }
  return false;
}

template bool COOHasDuplicate<kDLCPU, int32_t>(COOMatrix coo);
template bool COOHasDuplicate<kDLCPU, int64_t>(COOMatrix coo);

///////////////////////////// COOGetRowNNZ /////////////////////////////

template <DLDeviceType XPU, typename IdType>
int64_t COOGetRowNNZ(COOMatrix coo, int64_t row) {
  CHECK(row >= 0 && row < coo.num_rows) << "Invalid row index: " << row;
  const IdType* coo_row_data = static_cast<IdType*>(coo.row->data);
  int64_t result = 0;
  for (int64_t i = 0; i < coo.row->shape[0]; ++i) {
    if (coo_row_data[i] == row)
      ++result;
  }
  return result;
}

template int64_t COOGetRowNNZ<kDLCPU, int32_t>(COOMatrix, int64_t);
template int64_t COOGetRowNNZ<kDLCPU, int64_t>(COOMatrix, int64_t);

template <DLDeviceType XPU, typename IdType>
NDArray COOGetRowNNZ(COOMatrix coo, NDArray rows) {
  CHECK_SAME_DTYPE(coo.col, rows);
  const auto len = rows->shape[0];
  const IdType* vid_data = static_cast<IdType*>(rows->data);
  NDArray rst = NDArray::Empty({len}, rows->dtype, rows->ctx);
  IdType* rst_data = static_cast<IdType*>(rst->data);
#pragma omp parallel for
  for (int64_t i = 0; i < len; ++i)
    rst_data[i] = COOGetRowNNZ<XPU, IdType>(coo, vid_data[i]);
  return rst;
}

template NDArray COOGetRowNNZ<kDLCPU, int32_t>(COOMatrix, NDArray);
template NDArray COOGetRowNNZ<kDLCPU, int64_t>(COOMatrix, NDArray);

///////////////////////////// COOGetRowDataAndIndices /////////////////////////////

template <DLDeviceType XPU, typename IdType>
std::pair<NDArray, NDArray> COOGetRowDataAndIndices(
    COOMatrix coo, int64_t row) {
  CHECK(row >= 0 && row < coo.num_rows) << "Invalid row index: " << row;

  const IdType* coo_row_data = static_cast<IdType*>(coo.row->data);
  const IdType* coo_col_data = static_cast<IdType*>(coo.col->data);
  const IdType* coo_data = COOHasData(coo) ? static_cast<IdType*>(coo.data->data) : nullptr;

  std::vector<IdType> indices;
  std::vector<IdType> data;

  for (int64_t i = 0; i < coo.row->shape[0]; ++i) {
    if (coo_row_data[i] == row) {
      indices.push_back(coo_col_data[i]);
      data.push_back(coo_data ? coo_data[i] : i);
    }
  }

  return std::make_pair(NDArray::FromVector(data), NDArray::FromVector(indices));
}

template std::pair<NDArray, NDArray>
COOGetRowDataAndIndices<kDLCPU, int32_t>(COOMatrix, int64_t);
template std::pair<NDArray, NDArray>
COOGetRowDataAndIndices<kDLCPU, int64_t>(COOMatrix, int64_t);

///////////////////////////// COOGetData /////////////////////////////

template <DLDeviceType XPU, typename IdType>
IdArray COOGetData(COOMatrix coo, IdArray rows, IdArray cols) {
  const int64_t rowlen = rows->shape[0];
  const int64_t collen = cols->shape[0];
  CHECK((rowlen == collen) || (rowlen == 1) || (collen == 1))
    << "Invalid row and col Id array:" << rows << " " << cols;
  const int64_t row_stride = (rowlen == 1 && collen != 1) ? 0 : 1;
  const int64_t col_stride = (collen == 1 && rowlen != 1) ? 0 : 1;
  const IdType* row_data = rows.Ptr<IdType>();
  const IdType* col_data = cols.Ptr<IdType>();

  const IdType* coo_row = coo.row.Ptr<IdType>();
  const IdType* coo_col = coo.col.Ptr<IdType>();
  const IdType* data = COOHasData(coo) ? coo.data.Ptr<IdType>() : nullptr;
  const int64_t nnz = coo.row->shape[0];

  const int64_t retlen = std::max(rowlen, collen);
  IdArray ret = Full(-1, retlen, rows->dtype.bits, rows->ctx);
  IdType* ret_data = ret.Ptr<IdType>();

  // TODO(minjie): We might need to consider sorting the COO beforehand especially
  //   when the number of (row, col) pairs is large. Need more benchmarks to justify
  //   the choice.

  if (coo.row_sorted) {
#pragma omp parallel for
    for (int64_t p = 0; p < retlen; ++p) {
      const IdType row_id = row_data[p * row_stride], col_id = col_data[p * col_stride];
      auto it = std::lower_bound(coo_row, coo_row + nnz, row_id);
      for (; it < coo_row + nnz && *it == row_id; ++it) {
        const auto idx = it - coo_row;
        if (coo_col[idx] == col_id) {
          ret_data[p] = data? data[idx] : idx;
          break;
        }
      }
    }
  } else {
#pragma omp parallel for
    for (int64_t p = 0; p < retlen; ++p) {
      const IdType row_id = row_data[p * row_stride], col_id = col_data[p * col_stride];
      for (int64_t idx = 0; idx < nnz; ++idx) {
        if (coo_row[idx] == row_id && coo_col[idx] == col_id) {
          ret_data[p] = data? data[idx] : idx;
          break;
        }
      }
    }
  }

  return ret;
}

template IdArray COOGetData<kDLCPU, int32_t>(COOMatrix, IdArray, IdArray);
template IdArray COOGetData<kDLCPU, int64_t>(COOMatrix, IdArray, IdArray);

///////////////////////////// COOGetDataAndIndices /////////////////////////////

template <DLDeviceType XPU, typename IdType>
std::vector<NDArray> COOGetDataAndIndices(COOMatrix coo, NDArray rows,
                                          NDArray cols) {
  CHECK_SAME_DTYPE(coo.col, rows);
  CHECK_SAME_DTYPE(coo.col, cols);
  const int64_t rowlen = rows->shape[0];
  const int64_t collen = cols->shape[0];
  const int64_t len = std::max(rowlen, collen);

  CHECK((rowlen == collen) || (rowlen == 1) || (collen == 1))
    << "Invalid row and col id array.";

  const int64_t row_stride = (rowlen == 1 && collen != 1) ? 0 : 1;
  const int64_t col_stride = (collen == 1 && rowlen != 1) ? 0 : 1;
  const IdType* row_data = static_cast<IdType*>(rows->data);
  const IdType* col_data = static_cast<IdType*>(cols->data);

  const IdType* coo_row_data = static_cast<IdType*>(coo.row->data);
  const IdType* coo_col_data = static_cast<IdType*>(coo.col->data);
  const IdType* data = COOHasData(coo) ? static_cast<IdType*>(coo.data->data) : nullptr;

  std::vector<IdType> ret_rows, ret_cols;
  std::vector<IdType> ret_data;
  ret_rows.reserve(len);
  ret_cols.reserve(len);
  ret_data.reserve(len);

  // NOTE(BarclayII): With a small number of lookups, linear scan is faster.
  // The threshold 200 comes from benchmarking both algorithms on a P3.8x instance.
  // I also tried sorting plus binary search.  The speed gain is only significant for
  // medium-sized graphs and lookups, so I didn't include it.
  if (len >= 200) {
    // TODO(BarclayII) Ideally we would want to cache this object.  However I'm not sure
    // what is the best way to do so since this object is valid for CPU only.
    std::unordered_multimap<std::pair<IdType, IdType>, IdType, PairHash> pair_map;
    pair_map.reserve(coo.row->shape[0]);
    for (int64_t k = 0; k < coo.row->shape[0]; ++k)
      pair_map.emplace(std::make_pair(coo_row_data[k], coo_col_data[k]), data ? data[k]: k);

    for (int64_t i = 0, j = 0; i < rowlen && j < collen; i += row_stride, j += col_stride) {
      const IdType row_id = row_data[i], col_id = col_data[j];
      CHECK(row_id >= 0 && row_id < coo.num_rows) << "Invalid row index: " << row_id;
      CHECK(col_id >= 0 && col_id < coo.num_cols) << "Invalid col index: " << col_id;
      auto range = pair_map.equal_range({row_id, col_id});
      for (auto it = range.first; it != range.second; ++it) {
        ret_rows.push_back(row_id);
        ret_cols.push_back(col_id);
        ret_data.push_back(it->second);
      }
    }
  } else {
    for (int64_t i = 0, j = 0; i < rowlen && j < collen; i += row_stride, j += col_stride) {
      const IdType row_id = row_data[i], col_id = col_data[j];
      CHECK(row_id >= 0 && row_id < coo.num_rows) << "Invalid row index: " << row_id;
      CHECK(col_id >= 0 && col_id < coo.num_cols) << "Invalid col index: " << col_id;
      for (int64_t k = 0; k < coo.row->shape[0]; ++k) {
        if (coo_row_data[k] == row_id && coo_col_data[k] == col_id) {
          ret_rows.push_back(row_id);
          ret_cols.push_back(col_id);
          ret_data.push_back(data ? data[k] : k);
        }
      }
    }
  }

  return {NDArray::FromVector(ret_rows),
          NDArray::FromVector(ret_cols),
          NDArray::FromVector(ret_data)};
}

template std::vector<NDArray> COOGetDataAndIndices<kDLCPU, int32_t>(
    COOMatrix coo, NDArray rows, NDArray cols);
template std::vector<NDArray> COOGetDataAndIndices<kDLCPU, int64_t>(
    COOMatrix coo, NDArray rows, NDArray cols);

///////////////////////////// COOTranspose /////////////////////////////

template <DLDeviceType XPU, typename IdType>
COOMatrix COOTranspose(COOMatrix coo) {
  return COOMatrix{coo.num_cols, coo.num_rows, coo.col, coo.row, coo.data};
}

template COOMatrix COOTranspose<kDLCPU, int32_t>(COOMatrix coo);
template COOMatrix COOTranspose<kDLCPU, int64_t>(COOMatrix coo);

///////////////////////////// COOToCSR /////////////////////////////

// complexity: time O(NNZ), space O(1)
template <DLDeviceType XPU, typename IdType>
CSRMatrix COOToCSR(COOMatrix coo) {
  const int64_t N = coo.num_rows;
  const int64_t NNZ = coo.row->shape[0];
  const IdType* row_data = static_cast<IdType*>(coo.row->data);
  const IdType* col_data = static_cast<IdType*>(coo.col->data);
  const IdType* data = COOHasData(coo)? static_cast<IdType*>(coo.data->data) : nullptr;

  NDArray ret_indptr = NDArray::Empty({N + 1}, coo.row->dtype, coo.row->ctx);
  NDArray ret_indices;
  NDArray ret_data;

  bool row_sorted = coo.row_sorted;
  bool col_sorted = coo.col_sorted;
  if (!row_sorted) {
    // It is possible that the flag is simply not set (default value is false),
    // so we still perform a linear scan to check the flag.
    std::tie(row_sorted, col_sorted) = COOIsSorted(coo);
  }

  if (row_sorted) {
    // compute indptr
    IdType* Bp = static_cast<IdType*>(ret_indptr->data);
    Bp[0] = 0;
    int64_t j = 0;
    for (int64_t i = 0; i < N; ++i) {
      const int64_t k = j;
      for (; j < NNZ && row_data[j] == i; ++j) {}
      Bp[i + 1] = Bp[i] + j - k;
    }

    // TODO(minjie): Many of our current implementation assumes that CSR must have
    //   a data array. This is a temporary workaround. Remove this after:
    //   - The old immutable graph implementation is deprecated.
    //   - The old binary reduce kernel is deprecated.
    if (!COOHasData(coo))
      coo.data = aten::Range(0, NNZ, coo.row->dtype.bits, coo.row->ctx);

    // compute indices and data
    ret_indices = coo.col;
    ret_data = coo.data;
  } else {
    // compute indptr
    IdType* Bp = static_cast<IdType*>(ret_indptr->data);
    std::fill(Bp, Bp + N, 0);
    for (int64_t i = 0; i < NNZ; ++i) {
      Bp[row_data[i]]++;
    }

    // cumsum
    for (int64_t i = 0, cumsum = 0; i < N; ++i) {
      const IdType temp = Bp[i];
      Bp[i] = cumsum;
      cumsum += temp;
    }
    Bp[N] = NNZ;

    // compute indices and data
    ret_indices = NDArray::Empty({NNZ}, coo.row->dtype, coo.row->ctx);
    ret_data = NDArray::Empty({NNZ}, coo.row->dtype, coo.row->ctx);
    IdType* Bi = static_cast<IdType*>(ret_indices->data);
    IdType* Bx = static_cast<IdType*>(ret_data->data);

    for (int64_t i = 0; i < NNZ; ++i) {
      const IdType r = row_data[i];
      Bi[Bp[r]] = col_data[i];
      Bx[Bp[r]] = data? data[i] : i;
      Bp[r]++;
    }

    // correct the indptr
    for (int64_t i = 0, last = 0; i <= N; ++i) {
      IdType temp = Bp[i];
      Bp[i] = last;
      last = temp;
    }
  }

  return CSRMatrix(coo.num_rows, coo.num_cols,
                   ret_indptr, ret_indices, ret_data,
                   col_sorted);
}

template CSRMatrix COOToCSR<kDLCPU, int32_t>(COOMatrix coo);
template CSRMatrix COOToCSR<kDLCPU, int64_t>(COOMatrix coo);

///////////////////////////// COOSliceRows /////////////////////////////

template <DLDeviceType XPU, typename IdType>
COOMatrix COOSliceRows(COOMatrix coo, int64_t start, int64_t end) {
  // TODO(minjie): use binary search when coo.row_sorted is true
  CHECK(start >= 0 && start < coo.num_rows) << "Invalid start row " << start;
  CHECK(end > 0 && end <= coo.num_rows) << "Invalid end row " << end;

  const IdType* coo_row_data = static_cast<IdType*>(coo.row->data);
  const IdType* coo_col_data = static_cast<IdType*>(coo.col->data);
  const IdType* coo_data = COOHasData(coo) ? static_cast<IdType*>(coo.data->data) : nullptr;

  std::vector<IdType> ret_row, ret_col;
  std::vector<IdType> ret_data;

  for (int64_t i = 0; i < coo.row->shape[0]; ++i) {
    const IdType row_id = coo_row_data[i];
    const IdType col_id = coo_col_data[i];
    if (row_id < end && row_id >= start) {
      ret_row.push_back(row_id - start);
      ret_col.push_back(col_id);
      ret_data.push_back(coo_data ? coo_data[i] : i);
    }
  }
  return COOMatrix(
    end - start,
    coo.num_cols,
    NDArray::FromVector(ret_row),
    NDArray::FromVector(ret_col),
    NDArray::FromVector(ret_data),
    coo.row_sorted,
    coo.col_sorted);
}

template COOMatrix COOSliceRows<kDLCPU, int32_t>(COOMatrix, int64_t, int64_t);
template COOMatrix COOSliceRows<kDLCPU, int64_t>(COOMatrix, int64_t, int64_t);

template <DLDeviceType XPU, typename IdType>
COOMatrix COOSliceRows(COOMatrix coo, NDArray rows) {
  const IdType* coo_row_data = static_cast<IdType*>(coo.row->data);
  const IdType* coo_col_data = static_cast<IdType*>(coo.col->data);
  const IdType* coo_data = COOHasData(coo) ? static_cast<IdType*>(coo.data->data) : nullptr;

  std::vector<IdType> ret_row, ret_col;
  std::vector<IdType> ret_data;

  IdHashMap<IdType> hashmap(rows);

  for (int64_t i = 0; i < coo.row->shape[0]; ++i) {
    const IdType row_id = coo_row_data[i];
    const IdType col_id = coo_col_data[i];
    const IdType mapped_row_id = hashmap.Map(row_id, -1);
    if (mapped_row_id != -1) {
      ret_row.push_back(mapped_row_id);
      ret_col.push_back(col_id);
      ret_data.push_back(coo_data ? coo_data[i] : i);
    }
  }

  return COOMatrix{
    rows->shape[0],
    coo.num_cols,
    NDArray::FromVector(ret_row),
    NDArray::FromVector(ret_col),
    NDArray::FromVector(ret_data),
    coo.row_sorted, coo.col_sorted};
}

template COOMatrix COOSliceRows<kDLCPU, int32_t>(COOMatrix , NDArray);
template COOMatrix COOSliceRows<kDLCPU, int64_t>(COOMatrix , NDArray);

///////////////////////////// COOSliceMatrix /////////////////////////////

template <DLDeviceType XPU, typename IdType>
COOMatrix COOSliceMatrix(COOMatrix coo, runtime::NDArray rows, runtime::NDArray cols) {
  const IdType* coo_row_data = static_cast<IdType*>(coo.row->data);
  const IdType* coo_col_data = static_cast<IdType*>(coo.col->data);
  const IdType* coo_data = COOHasData(coo) ? static_cast<IdType*>(coo.data->data) : nullptr;

  IdHashMap<IdType> row_map(rows), col_map(cols);

  std::vector<IdType> ret_row, ret_col;
  std::vector<IdType> ret_data;

  for (int64_t i = 0; i < coo.row->shape[0]; ++i) {
    const IdType row_id = coo_row_data[i];
    const IdType col_id = coo_col_data[i];
    const IdType mapped_row_id = row_map.Map(row_id, -1);
    if (mapped_row_id != -1) {
      const IdType mapped_col_id = col_map.Map(col_id, -1);
      if (mapped_col_id != -1) {
        ret_row.push_back(mapped_row_id);
        ret_col.push_back(mapped_col_id);
        ret_data.push_back(coo_data ? coo_data[i] : i);
      }
    }
  }

  return COOMatrix(rows->shape[0], cols->shape[0],
                   NDArray::FromVector(ret_row),
                   NDArray::FromVector(ret_col),
                   NDArray::FromVector(ret_data),
                   coo.row_sorted, coo.col_sorted);
}

template COOMatrix COOSliceMatrix<kDLCPU, int32_t>(
    COOMatrix coo, runtime::NDArray rows, runtime::NDArray cols);
template COOMatrix COOSliceMatrix<kDLCPU, int64_t>(
    COOMatrix coo, runtime::NDArray rows, runtime::NDArray cols);


///////////////////////////// COOReorder /////////////////////////////

template <DLDeviceType XPU, typename IdType>
COOMatrix COOReorder(COOMatrix coo, runtime::NDArray new_row_id_arr,
                     runtime::NDArray new_col_id_arr) {
  CHECK_SAME_DTYPE(coo.row, new_row_id_arr);
  CHECK_SAME_DTYPE(coo.col, new_col_id_arr);

  // Input COO
  const IdType* in_rows = static_cast<IdType*>(coo.row->data);
  const IdType* in_cols = static_cast<IdType*>(coo.col->data);
  int64_t num_rows = coo.num_rows;
  int64_t num_cols = coo.num_cols;
  int64_t nnz = coo.row->shape[0];
  CHECK_EQ(num_rows, new_row_id_arr->shape[0])
      << "The new row Id array needs to be the same as the number of rows of COO";
  CHECK_EQ(num_cols, new_col_id_arr->shape[0])
      << "The new col Id array needs to be the same as the number of cols of COO";

  // New row/col Ids.
  const IdType* new_row_ids = static_cast<IdType*>(new_row_id_arr->data);
  const IdType* new_col_ids = static_cast<IdType*>(new_col_id_arr->data);

  // Output COO
  NDArray out_row_arr = NDArray::Empty({nnz}, coo.row->dtype, coo.row->ctx);
  NDArray out_col_arr = NDArray::Empty({nnz}, coo.col->dtype, coo.col->ctx);
  NDArray out_data_arr = COOHasData(coo) ? coo.data : NullArray();
  IdType *out_row = static_cast<IdType*>(out_row_arr->data);
  IdType *out_col = static_cast<IdType*>(out_col_arr->data);

#pragma omp parallel for
  for (int64_t i = 0; i < nnz; i++) {
    out_row[i] = new_row_ids[in_rows[i]];
    out_col[i] = new_col_ids[in_cols[i]];
  }
  return COOMatrix(num_rows, num_cols, out_row_arr, out_col_arr, out_data_arr);
}

template COOMatrix COOReorder<kDLCPU, int64_t>(COOMatrix csr, runtime::NDArray new_row_ids,
                                               runtime::NDArray new_col_ids);
template COOMatrix COOReorder<kDLCPU, int32_t>(COOMatrix csr, runtime::NDArray new_row_ids,
                                               runtime::NDArray new_col_ids);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
