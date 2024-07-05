/**
 *  Copyright (c) 2019 by Contributors
 * @file array/cpu/spmat_op_impl.cc
 * @brief CPU implementation of COO sparse matrix operators
 */
#include <dgl/runtime/parallel_for.h>
#include <dmlc/omp.h>

#include <numeric>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "array_utils.h"

namespace dgl {

using runtime::NDArray;
using runtime::parallel_for;

namespace aten {
namespace impl {

/**
 * TODO(BarclayII):
 * For row-major sorted COOs, we have faster implementation with binary search,
 * sorted search, etc.  Later we should benchmark how much we can gain with
 * sorted COOs on hypersparse graphs.
 */

///////////////////////////// COOIsNonZero /////////////////////////////

template <DGLDeviceType XPU, typename IdType>
bool COOIsNonZero(COOMatrix coo, int64_t row, int64_t col) {
  CHECK(row >= 0 && row < coo.num_rows) << "Invalid row index: " << row;
  CHECK(col >= 0 && col < coo.num_cols) << "Invalid col index: " << col;
  const IdType *coo_row_data = static_cast<IdType *>(coo.row->data);
  const IdType *coo_col_data = static_cast<IdType *>(coo.col->data);
  for (int64_t i = 0; i < coo.row->shape[0]; ++i) {
    if (coo_row_data[i] == row && coo_col_data[i] == col) return true;
  }
  return false;
}

template bool COOIsNonZero<kDGLCPU, int32_t>(COOMatrix, int64_t, int64_t);
template bool COOIsNonZero<kDGLCPU, int64_t>(COOMatrix, int64_t, int64_t);

template <DGLDeviceType XPU, typename IdType>
NDArray COOIsNonZero(COOMatrix coo, NDArray row, NDArray col) {
  const auto rowlen = row->shape[0];
  const auto collen = col->shape[0];
  const auto rstlen = std::max(rowlen, collen);
  NDArray rst = NDArray::Empty({rstlen}, row->dtype, row->ctx);
  IdType *rst_data = static_cast<IdType *>(rst->data);
  const IdType *row_data = static_cast<IdType *>(row->data);
  const IdType *col_data = static_cast<IdType *>(col->data);
  const int64_t row_stride = (rowlen == 1 && collen != 1) ? 0 : 1;
  const int64_t col_stride = (collen == 1 && rowlen != 1) ? 0 : 1;
  const int64_t kmax = std::max(rowlen, collen);
  parallel_for(0, kmax, [=](size_t b, size_t e) {
    for (auto k = b; k < e; ++k) {
      int64_t i = row_stride * k;
      int64_t j = col_stride * k;
      rst_data[k] =
          COOIsNonZero<XPU, IdType>(coo, row_data[i], col_data[j]) ? 1 : 0;
    }
  });
  return rst;
}

template NDArray COOIsNonZero<kDGLCPU, int32_t>(COOMatrix, NDArray, NDArray);
template NDArray COOIsNonZero<kDGLCPU, int64_t>(COOMatrix, NDArray, NDArray);

///////////////////////////// COOHasDuplicate /////////////////////////////

template <DGLDeviceType XPU, typename IdType>
bool COOHasDuplicate(COOMatrix coo) {
  std::unordered_set<std::pair<IdType, IdType>, PairHash> hashmap;
  const IdType *src_data = static_cast<IdType *>(coo.row->data);
  const IdType *dst_data = static_cast<IdType *>(coo.col->data);
  const auto nnz = coo.row->shape[0];
  for (IdType eid = 0; eid < nnz; ++eid) {
    const auto &p = std::make_pair(src_data[eid], dst_data[eid]);
    if (hashmap.count(p)) {
      return true;
    } else {
      hashmap.insert(p);
    }
  }
  return false;
}

template bool COOHasDuplicate<kDGLCPU, int32_t>(COOMatrix coo);
template bool COOHasDuplicate<kDGLCPU, int64_t>(COOMatrix coo);

///////////////////////////// COOGetRowNNZ /////////////////////////////

template <DGLDeviceType XPU, typename IdType>
int64_t COOGetRowNNZ(COOMatrix coo, int64_t row) {
  CHECK(row >= 0 && row < coo.num_rows) << "Invalid row index: " << row;
  const IdType *coo_row_data = static_cast<IdType *>(coo.row->data);
  int64_t result = 0;
  for (int64_t i = 0; i < coo.row->shape[0]; ++i) {
    if (coo_row_data[i] == row) ++result;
  }
  return result;
}

template int64_t COOGetRowNNZ<kDGLCPU, int32_t>(COOMatrix, int64_t);
template int64_t COOGetRowNNZ<kDGLCPU, int64_t>(COOMatrix, int64_t);

template <DGLDeviceType XPU, typename IdType>
NDArray COOGetRowNNZ(COOMatrix coo, NDArray rows) {
  CHECK_SAME_DTYPE(coo.col, rows);
  const auto len = rows->shape[0];
  const IdType *vid_data = static_cast<IdType *>(rows->data);
  NDArray rst = NDArray::Empty({len}, rows->dtype, rows->ctx);
  IdType *rst_data = static_cast<IdType *>(rst->data);
#pragma omp parallel for
  for (int64_t i = 0; i < len; ++i) {
    rst_data[i] = COOGetRowNNZ<XPU, IdType>(coo, vid_data[i]);
  }
  return rst;
}

template NDArray COOGetRowNNZ<kDGLCPU, int32_t>(COOMatrix, NDArray);
template NDArray COOGetRowNNZ<kDGLCPU, int64_t>(COOMatrix, NDArray);

////////////////////////// COOGetRowDataAndIndices /////////////////////////////

template <DGLDeviceType XPU, typename IdType>
std::pair<NDArray, NDArray> COOGetRowDataAndIndices(
    COOMatrix coo, int64_t row) {
  CHECK(row >= 0 && row < coo.num_rows) << "Invalid row index: " << row;

  const IdType *coo_row_data = static_cast<IdType *>(coo.row->data);
  const IdType *coo_col_data = static_cast<IdType *>(coo.col->data);
  const IdType *coo_data =
      COOHasData(coo) ? static_cast<IdType *>(coo.data->data) : nullptr;

  std::vector<IdType> indices;
  std::vector<IdType> data;

  for (int64_t i = 0; i < coo.row->shape[0]; ++i) {
    if (coo_row_data[i] == row) {
      indices.push_back(coo_col_data[i]);
      data.push_back(coo_data ? coo_data[i] : i);
    }
  }

  return std::make_pair(
      NDArray::FromVector(data), NDArray::FromVector(indices));
}

template std::pair<NDArray, NDArray> COOGetRowDataAndIndices<kDGLCPU, int32_t>(
    COOMatrix, int64_t);
template std::pair<NDArray, NDArray> COOGetRowDataAndIndices<kDGLCPU, int64_t>(
    COOMatrix, int64_t);

///////////////////////////// COOGetData /////////////////////////////

template <DGLDeviceType XPU, typename IdType>
IdArray COOGetData(COOMatrix coo, IdArray rows, IdArray cols) {
  const int64_t rowlen = rows->shape[0];
  const int64_t collen = cols->shape[0];
  CHECK((rowlen == collen) || (rowlen == 1) || (collen == 1))
      << "Invalid row and col Id array:" << rows << " " << cols;
  const int64_t row_stride = (rowlen == 1 && collen != 1) ? 0 : 1;
  const int64_t col_stride = (collen == 1 && rowlen != 1) ? 0 : 1;
  const IdType *row_data = rows.Ptr<IdType>();
  const IdType *col_data = cols.Ptr<IdType>();

  const IdType *coo_row = coo.row.Ptr<IdType>();
  const IdType *coo_col = coo.col.Ptr<IdType>();
  const IdType *data = COOHasData(coo) ? coo.data.Ptr<IdType>() : nullptr;
  const int64_t nnz = coo.row->shape[0];

  const int64_t retlen = std::max(rowlen, collen);
  IdArray ret = Full(-1, retlen, rows->dtype.bits, rows->ctx);
  IdType *ret_data = ret.Ptr<IdType>();

  // TODO(minjie): We might need to consider sorting the COO beforehand
  // especially when the number of (row, col) pairs is large. Need more
  // benchmarks to justify the choice.

  if (coo.row_sorted) {
    parallel_for(0, retlen, [&](size_t b, size_t e) {
      for (auto p = b; p < e; ++p) {
        const IdType row_id = row_data[p * row_stride],
                     col_id = col_data[p * col_stride];
        auto it = std::lower_bound(coo_row, coo_row + nnz, row_id);
        for (; it < coo_row + nnz && *it == row_id; ++it) {
          const auto idx = it - coo_row;
          if (coo_col[idx] == col_id) {
            ret_data[p] = data ? data[idx] : idx;
            break;
          }
        }
      }
    });
  } else {
#pragma omp parallel for
    for (int64_t p = 0; p < retlen; ++p) {
      const IdType row_id = row_data[p * row_stride],
                   col_id = col_data[p * col_stride];
      for (int64_t idx = 0; idx < nnz; ++idx) {
        if (coo_row[idx] == row_id && coo_col[idx] == col_id) {
          ret_data[p] = data ? data[idx] : idx;
          break;
        }
      }
    }
  }

  return ret;
}

template IdArray COOGetData<kDGLCPU, int32_t>(COOMatrix, IdArray, IdArray);
template IdArray COOGetData<kDGLCPU, int64_t>(COOMatrix, IdArray, IdArray);

///////////////////////////// COOGetDataAndIndices /////////////////////////////

template <DGLDeviceType XPU, typename IdType>
std::vector<NDArray> COOGetDataAndIndices(
    COOMatrix coo, NDArray rows, NDArray cols) {
  CHECK_SAME_DTYPE(coo.col, rows);
  CHECK_SAME_DTYPE(coo.col, cols);
  const int64_t rowlen = rows->shape[0];
  const int64_t collen = cols->shape[0];
  const int64_t len = std::max(rowlen, collen);

  CHECK((rowlen == collen) || (rowlen == 1) || (collen == 1))
      << "Invalid row and col id array.";

  const int64_t row_stride = (rowlen == 1 && collen != 1) ? 0 : 1;
  const int64_t col_stride = (collen == 1 && rowlen != 1) ? 0 : 1;
  const IdType *row_data = static_cast<IdType *>(rows->data);
  const IdType *col_data = static_cast<IdType *>(cols->data);

  const IdType *coo_row_data = static_cast<IdType *>(coo.row->data);
  const IdType *coo_col_data = static_cast<IdType *>(coo.col->data);
  const IdType *data =
      COOHasData(coo) ? static_cast<IdType *>(coo.data->data) : nullptr;

  std::vector<IdType> ret_rows, ret_cols;
  std::vector<IdType> ret_data;
  ret_rows.reserve(len);
  ret_cols.reserve(len);
  ret_data.reserve(len);

  // NOTE(BarclayII): With a small number of lookups, linear scan is faster.
  // The threshold 200 comes from benchmarking both algorithms on a P3.8x
  // instance. I also tried sorting plus binary search.  The speed gain is only
  // significant for medium-sized graphs and lookups, so I didn't include it.
  if (len >= 200) {
    // TODO(BarclayII) Ideally we would want to cache this object.  However I'm
    // not sure what is the best way to do so since this object is valid for CPU
    // only.
    std::unordered_multimap<std::pair<IdType, IdType>, IdType, PairHash>
        pair_map;
    pair_map.reserve(coo.row->shape[0]);
    for (int64_t k = 0; k < coo.row->shape[0]; ++k)
      pair_map.emplace(
          std::make_pair(coo_row_data[k], coo_col_data[k]), data ? data[k] : k);

    for (int64_t i = 0, j = 0; i < rowlen && j < collen;
         i += row_stride, j += col_stride) {
      const IdType row_id = row_data[i], col_id = col_data[j];
      CHECK(row_id >= 0 && row_id < coo.num_rows)
          << "Invalid row index: " << row_id;
      CHECK(col_id >= 0 && col_id < coo.num_cols)
          << "Invalid col index: " << col_id;
      auto range = pair_map.equal_range({row_id, col_id});
      for (auto it = range.first; it != range.second; ++it) {
        ret_rows.push_back(row_id);
        ret_cols.push_back(col_id);
        ret_data.push_back(it->second);
      }
    }
  } else {
    for (int64_t i = 0, j = 0; i < rowlen && j < collen;
         i += row_stride, j += col_stride) {
      const IdType row_id = row_data[i], col_id = col_data[j];
      CHECK(row_id >= 0 && row_id < coo.num_rows)
          << "Invalid row index: " << row_id;
      CHECK(col_id >= 0 && col_id < coo.num_cols)
          << "Invalid col index: " << col_id;
      for (int64_t k = 0; k < coo.row->shape[0]; ++k) {
        if (coo_row_data[k] == row_id && coo_col_data[k] == col_id) {
          ret_rows.push_back(row_id);
          ret_cols.push_back(col_id);
          ret_data.push_back(data ? data[k] : k);
        }
      }
    }
  }

  return {
      NDArray::FromVector(ret_rows), NDArray::FromVector(ret_cols),
      NDArray::FromVector(ret_data)};
}

template std::vector<NDArray> COOGetDataAndIndices<kDGLCPU, int32_t>(
    COOMatrix coo, NDArray rows, NDArray cols);
template std::vector<NDArray> COOGetDataAndIndices<kDGLCPU, int64_t>(
    COOMatrix coo, NDArray rows, NDArray cols);

///////////////////////////// COOTranspose /////////////////////////////

template <DGLDeviceType XPU, typename IdType>
COOMatrix COOTranspose(COOMatrix coo) {
  return COOMatrix{coo.num_cols, coo.num_rows, coo.col, coo.row, coo.data};
}

template COOMatrix COOTranspose<kDGLCPU, int32_t>(COOMatrix coo);
template COOMatrix COOTranspose<kDGLCPU, int64_t>(COOMatrix coo);

///////////////////////////// COOToCSR /////////////////////////////
namespace {

template <class IdType>
CSRMatrix SortedCOOToCSR(const COOMatrix &coo) {
  const int64_t N = coo.num_rows;
  const int64_t NNZ = coo.row->shape[0];
  const IdType *const row_data = static_cast<IdType *>(coo.row->data);
  const IdType *const data =
      COOHasData(coo) ? static_cast<IdType *>(coo.data->data) : nullptr;

  NDArray ret_indptr = NDArray::Empty({N + 1}, coo.row->dtype, coo.row->ctx);
  NDArray ret_indices = coo.col;
  NDArray ret_data = data == nullptr
                         ? NDArray::Empty({NNZ}, coo.row->dtype, coo.row->ctx)
                         : coo.data;

  // compute indptr
  IdType *const Bp = static_cast<IdType *>(ret_indptr->data);
  Bp[0] = 0;

  IdType *const fill_data =
      data ? nullptr : static_cast<IdType *>(ret_data->data);

  if (NNZ > 0) {
    auto num_threads = omp_get_max_threads();
    parallel_for(0, num_threads, [&](int b, int e) {
      for (auto thread_id = b; thread_id < e; ++thread_id) {
        // We partition the set the of non-zeros among the threads
        const int64_t nz_chunk = (NNZ + num_threads - 1) / num_threads;
        const int64_t nz_start = thread_id * nz_chunk;
        const int64_t nz_end = std::min(NNZ, nz_start + nz_chunk);

        // Each thread searchs the row array for a change, and marks it's
        // location in Bp. Threads, other than the first, start at the last
        // index covered by the previous, in order to detect changes in the row
        // array between thread partitions. This means that each thread after
        // the first, searches the range [nz_start-1, nz_end). That is,
        // if we had 10 non-zeros, and 4 threads, the indexes searched by each
        // thread would be:
        // 0: [0, 1, 2]
        // 1: [2, 3, 4, 5]
        // 2: [5, 6, 7, 8]
        // 3: [8, 9]
        //
        // That way, if the row array were [0, 0, 1, 2, 2, 2, 4, 5, 5, 6], each
        // change in row would be captured by one thread:
        //
        // 0: [0, 0, 1] - row 0
        // 1: [1, 2, 2, 2] - row 1
        // 2: [2, 4, 5, 5] - rows 2, 3, and 4
        // 3: [5, 6] - rows 5 and 6
        //
        int64_t row = 0;
        if (nz_start < nz_end) {
          row = nz_start == 0 ? 0 : row_data[nz_start - 1];
          for (int64_t i = nz_start; i < nz_end; ++i) {
            while (row != row_data[i]) {
              ++row;
              Bp[row] = i;
            }
          }

          // We will not detect the row change for the last row, nor any empty
          // rows at the end of the matrix, so the last active thread needs
          // mark all remaining rows in Bp with NNZ.
          if (nz_end == NNZ) {
            while (row < N) {
              ++row;
              Bp[row] = NNZ;
            }
          }

          if (fill_data) {
            // TODO(minjie): Many of our current implementation assumes that CSR
            // must have
            //   a data array. This is a temporary workaround. Remove this
            //   after:
            //   - The old immutable graph implementation is deprecated.
            //   - The old binary reduce kernel is deprecated.
            std::iota(fill_data + nz_start, fill_data + nz_end, nz_start);
          }
        }
      }
    });
  } else {
    std::fill(Bp, Bp + N + 1, 0);
  }

  return CSRMatrix(
      coo.num_rows, coo.num_cols, ret_indptr, ret_indices, ret_data,
      coo.col_sorted);
}

template <class IdType>
CSRMatrix UnSortedSparseCOOToCSR(const COOMatrix &coo) {
  // Unsigned version of the original integer index data type.
  // It avoids overflow in (N + num_threads) and (n_start + n_chunk) below.
  typedef typename std::make_unsigned<IdType>::type UIdType;

  const UIdType N = coo.num_rows;
  const int64_t NNZ = coo.row->shape[0];
  const IdType *const row_data = static_cast<IdType *>(coo.row->data);
  const IdType *const col_data = static_cast<IdType *>(coo.col->data);
  const IdType *const data =
      COOHasData(coo) ? static_cast<IdType *>(coo.data->data) : nullptr;

  NDArray ret_indptr = NDArray::Empty(
      {static_cast<int64_t>(N) + 1}, coo.row->dtype, coo.row->ctx);
  NDArray ret_indices = NDArray::Empty({NNZ}, coo.row->dtype, coo.row->ctx);
  NDArray ret_data = NDArray::Empty({NNZ}, coo.row->dtype, coo.row->ctx);
  IdType *const Bp = static_cast<IdType *>(ret_indptr->data);
  Bp[N] = 0;
  IdType *const Bi = static_cast<IdType *>(ret_indices->data);
  IdType *const Bx = static_cast<IdType *>(ret_data->data);

  // store sorted data and original index.
  NDArray sorted_data = NDArray::Empty({NNZ}, coo.row->dtype, coo.row->ctx);
  NDArray sorted_data_pos = NDArray::Empty({NNZ}, coo.row->dtype, coo.row->ctx);
  IdType *const Sx = static_cast<IdType *>(sorted_data->data);
  IdType *const Si = static_cast<IdType *>(sorted_data_pos->data);

  // Lower number of threads if cost of parallelization is grater than gain
  // from making calculation parallel.
  const int64_t min_chunk_size = 1000;
  const int64_t num_threads_for_batch = 2 + (NNZ + N) / min_chunk_size;
  const int num_threads_required = std::min(
      static_cast<int64_t>(omp_get_max_threads()), num_threads_for_batch);

  // record row_idx in each thread.
  std::vector<std::vector<int64_t>> p_sum(
      num_threads_required, std::vector<int64_t>(num_threads_required));

#pragma omp parallel num_threads(num_threads_required)
  {
    const int num_threads = omp_get_num_threads();
    const int thread_id = omp_get_thread_num();
    CHECK_LT(thread_id, num_threads);

    const int64_t nz_chunk = (NNZ + num_threads - 1) / num_threads;
    const int64_t nz_start = thread_id * nz_chunk;
    const int64_t nz_end = std::min(NNZ, nz_start + nz_chunk);

    const UIdType n_chunk = (N + num_threads - 1) / num_threads;
    const UIdType n_start = thread_id * n_chunk;
    const UIdType n_end = std::min(N, n_start + n_chunk);

    for (auto i = n_start; i < n_end; ++i) {
      Bp[i] = 0;
    }

    // iterate on NNZ data and count row_idx.
    for (auto i = nz_start; i < nz_end; ++i) {
      const IdType row_idx = row_data[i];
      const IdType row_thread_id = row_idx / n_chunk;
      ++p_sum[thread_id][row_thread_id];
    }

#pragma omp barrier
#pragma omp master
    // accumulate row_idx.
    {
      int64_t cum = 0;
      for (int j = 0; j < num_threads; ++j) {
        for (int i = 0; i < num_threads; ++i) {
          auto tmp = p_sum[i][j];
          p_sum[i][j] = cum;
          cum += tmp;
        }
      }
      CHECK_EQ(cum, NNZ);
    }
#pragma omp barrier
    const int64_t i_start = p_sum[0][thread_id];
    const int64_t i_end =
        thread_id + 1 == num_threads ? NNZ : p_sum[0][thread_id + 1];
#pragma omp barrier

    // sort data by row_idx and place into Sx/Si.
    auto &data_pos = p_sum[thread_id];
    for (auto i = nz_start; i < nz_end; ++i) {
      const IdType row_idx = row_data[i];
      const IdType row_thread_id = row_idx / n_chunk;
      const int64_t pos = data_pos[row_thread_id]++;
      Sx[pos] = data == nullptr ? i : data[i];
      Si[pos] = i;
    }

#pragma omp barrier

    // Now we're able to do coo2csr on sorted data in each thread in parallel.
    // compute data number on each row_idx.
    for (auto i = i_start; i < i_end; ++i) {
      const UIdType row_idx = row_data[Si[i]];
      ++Bp[row_idx + 1];
    }

    // accumulate on each row
    IdType cumsum = i_start;
    for (auto i = n_start + 1; i <= n_end; ++i) {
      const auto tmp = Bp[i];
      Bp[i] = cumsum;
      cumsum += tmp;
    }

    // update Bi/Bp/Bx
    for (auto i = i_start; i < i_end; ++i) {
      const UIdType row_idx = row_data[Si[i]];
      const int64_t dest = (Bp[row_idx + 1]++);
      Bi[dest] = col_data[Si[i]];
      Bx[dest] = Sx[i];
    }
  }
  return CSRMatrix(
      coo.num_rows, coo.num_cols, ret_indptr, ret_indices, ret_data,
      coo.col_sorted);
}

template <class IdType>
CSRMatrix UnSortedDenseCOOToCSR(const COOMatrix &coo) {
  // Unsigned version of the original integer index data type.
  // It avoids overflow in (N + num_threads) and (n_start + n_chunk) below.
  typedef typename std::make_unsigned<IdType>::type UIdType;

  const UIdType N = coo.num_rows;
  const int64_t NNZ = coo.row->shape[0];
  const IdType *const row_data = static_cast<IdType *>(coo.row->data);
  const IdType *const col_data = static_cast<IdType *>(coo.col->data);
  const IdType *const data =
      COOHasData(coo) ? static_cast<IdType *>(coo.data->data) : nullptr;

  NDArray ret_indptr = NDArray::Empty(
      {static_cast<int64_t>(N) + 1}, coo.row->dtype, coo.row->ctx);
  NDArray ret_indices = NDArray::Empty({NNZ}, coo.row->dtype, coo.row->ctx);
  NDArray ret_data = NDArray::Empty({NNZ}, coo.row->dtype, coo.row->ctx);
  IdType *const Bp = static_cast<IdType *>(ret_indptr->data);
  Bp[0] = 0;
  IdType *const Bi = static_cast<IdType *>(ret_indices->data);
  IdType *const Bx = static_cast<IdType *>(ret_data->data);

  // the offset within each row, that each thread will write to
  std::vector<std::vector<IdType>> local_ptrs;
  std::vector<int64_t> thread_prefixsum;

#pragma omp parallel
  {
    const int num_threads = omp_get_num_threads();
    const int thread_id = omp_get_thread_num();
    CHECK_LT(thread_id, num_threads);

    const int64_t nz_chunk = (NNZ + num_threads - 1) / num_threads;
    const int64_t nz_start = thread_id * nz_chunk;
    const int64_t nz_end = std::min(NNZ, nz_start + nz_chunk);

    const UIdType n_chunk = (N + num_threads - 1) / num_threads;
    const UIdType n_start = thread_id * n_chunk;
    const UIdType n_end = std::min(N, n_start + n_chunk);

#pragma omp master
    {
      local_ptrs.resize(num_threads);
      thread_prefixsum.resize(num_threads + 1);
    }

#pragma omp barrier
    local_ptrs[thread_id].resize(N, 0);

    for (int64_t i = nz_start; i < nz_end; ++i) {
      ++local_ptrs[thread_id][row_data[i]];
    }

#pragma omp barrier
    // compute prefixsum in parallel
    int64_t sum = 0;
    for (UIdType i = n_start; i < n_end; ++i) {
      IdType tmp = 0;
      for (int j = 0; j < num_threads; ++j) {
        auto previous = local_ptrs[j][i];
        local_ptrs[j][i] = tmp;
        tmp += previous;
      }
      sum += tmp;
      Bp[i + 1] = sum;
    }
    thread_prefixsum[thread_id + 1] = sum;

#pragma omp barrier
#pragma omp master
    {
      for (int i = 0; i < num_threads; ++i) {
        thread_prefixsum[i + 1] += thread_prefixsum[i];
      }
      CHECK_EQ(thread_prefixsum[num_threads], NNZ);
    }
#pragma omp barrier

    sum = thread_prefixsum[thread_id];
    for (UIdType i = n_start; i < n_end; ++i) {
      Bp[i + 1] += sum;
    }

#pragma omp barrier
    for (int64_t i = nz_start; i < nz_end; ++i) {
      const IdType r = row_data[i];
      const int64_t index = Bp[r] + local_ptrs[thread_id][r]++;
      Bi[index] = col_data[i];
      Bx[index] = data ? data[i] : i;
    }
  }
  CHECK_EQ(Bp[N], NNZ);

  return CSRMatrix(
      coo.num_rows, coo.num_cols, ret_indptr, ret_indices, ret_data,
      coo.col_sorted);
}

// complexity: time O(NNZ), space O(1)
template <typename IdType>
CSRMatrix UnSortedSmallCOOToCSR(COOMatrix coo) {
  const int64_t N = coo.num_rows;
  const int64_t NNZ = coo.row->shape[0];
  const IdType *row_data = static_cast<IdType *>(coo.row->data);
  const IdType *col_data = static_cast<IdType *>(coo.col->data);
  const IdType *data =
      COOHasData(coo) ? static_cast<IdType *>(coo.data->data) : nullptr;
  NDArray ret_indptr = NDArray::Empty({N + 1}, coo.row->dtype, coo.row->ctx);
  NDArray ret_indices = NDArray::Empty({NNZ}, coo.row->dtype, coo.row->ctx);
  NDArray ret_data = NDArray::Empty({NNZ}, coo.row->dtype, coo.row->ctx);
  IdType *Bp = static_cast<IdType *>(ret_indptr->data);
  IdType *Bi = static_cast<IdType *>(ret_indices->data);
  IdType *Bx = static_cast<IdType *>(ret_data->data);

  // Count elements in each row
  std::fill(Bp, Bp + N, 0);
  for (int64_t i = 0; i < NNZ; ++i) {
    Bp[row_data[i]]++;
  }

  // Convert to indexes
  for (IdType i = 0, cumsum = 0; i < N; ++i) {
    const IdType temp = Bp[i];
    Bp[i] = cumsum;
    cumsum += temp;
  }

  for (int64_t i = 0; i < NNZ; ++i) {
    const IdType r = row_data[i];
    Bi[Bp[r]] = col_data[i];
    Bx[Bp[r]] = data ? data[i] : i;
    Bp[r]++;
  }

  // Restore the indptr
  for (int64_t i = N; i > 0; --i) {
    Bp[i] = Bp[i - 1];
  }
  Bp[0] = 0;

  return CSRMatrix(
      coo.num_rows, coo.num_cols, ret_indptr, ret_indices, ret_data,
      coo.col_sorted);
}

enum class COOToCSRAlg {
  sorted = 0,
  unsortedSmall,
  unsortedSparse,
  unsortedDense
};

/**
 * Chose COO to CSR format conversion algorithm for given COO matrix according
 * to heuristic based on measured performance.
 *
 * Implementation and complexity details. N: num_nodes, NNZ: num_edges, P:
 * num_threads.
 *   1. If row is sorted in COO, SortedCOOToCSR<> is applied. Time: O(NNZ/P),
 * space: O(1).
 *   2 If row is NOT sorted in COO and graph is small (small number of NNZ),
 * UnSortedSmallCOOToCSR<> is applied. Time: O(NNZ), space O(N).
 *   3 If row is NOT sorted in COO and graph is sparse (low average degree),
 * UnSortedSparseCOOToCSR<> is applied. Time: O(NNZ/P + N/P + P^2),
 * space O(NNZ + P^2).
 *   4. If row is NOT sorted in COO and graph is dense (medium/high average
 * degree), UnSortedDenseCOOToCSR<> is applied. Time: O(NNZ/P + N/P),
 * space O(NNZ + N*P).
 *
 * Note:
 *   If you change this function, change also _TestCOOToCSRAlgs in
 * tests/cpp/test_spmat_coo.cc
 */
template <typename IdType>
inline COOToCSRAlg WhichCOOToCSR(const COOMatrix &coo) {
  if (coo.row_sorted) {
    return COOToCSRAlg::sorted;
  } else {
#ifdef _WIN32
    // On Windows omp_get_max_threads() gives larger value than later OMP can
    // spawn.
    int64_t num_threads;
#pragma omp parallel
#pragma master
    { num_threads = omp_get_num_threads(); }
#else
    const int64_t num_threads = omp_get_max_threads();
#endif
    const int64_t N = coo.num_rows;
    const int64_t NNZ = coo.row->shape[0];
    // Parameters below are heuristically chosen according to measured
    // performance.
    const int64_t type_scale = sizeof(IdType) >> 1;
    const int64_t small = 50 * num_threads * type_scale * type_scale;
    if (NNZ < small || num_threads == 1) {
      // For relatively small number of non zero elements cost of spread
      // algorithm between threads is bigger than improvements from using
      // many cores
      return COOToCSRAlg::unsortedSmall;
    } else if (type_scale * NNZ < num_threads * N) {
      // For relatively small number of non zero elements in matrix, sparse
      // parallel version of algorithm is more efficient than dense.
      return COOToCSRAlg::unsortedSparse;
    }
    return COOToCSRAlg::unsortedDense;
  }
}

}  // namespace

template <DGLDeviceType XPU, typename IdType>
CSRMatrix COOToCSR(COOMatrix coo) {
  CHECK_NO_OVERFLOW(coo.row->dtype, coo.row->shape[0]);
  switch (WhichCOOToCSR<IdType>(coo)) {
    case COOToCSRAlg::sorted:
      return SortedCOOToCSR<IdType>(coo);
    case COOToCSRAlg::unsortedSmall:
    default:
      return UnSortedSmallCOOToCSR<IdType>(coo);
    case COOToCSRAlg::unsortedSparse:
      return UnSortedSparseCOOToCSR<IdType>(coo);
    case COOToCSRAlg::unsortedDense:
      return UnSortedDenseCOOToCSR<IdType>(coo);
  }
}

template CSRMatrix COOToCSR<kDGLCPU, int32_t>(COOMatrix coo);
template CSRMatrix COOToCSR<kDGLCPU, int64_t>(COOMatrix coo);

///////////////////////////// COOSliceRows /////////////////////////////

template <DGLDeviceType XPU, typename IdType>
COOMatrix COOSliceRows(COOMatrix coo, int64_t start, int64_t end) {
  // TODO(minjie): use binary search when coo.row_sorted is true
  CHECK(start >= 0 && start < coo.num_rows) << "Invalid start row " << start;
  CHECK(end > 0 && end <= coo.num_rows) << "Invalid end row " << end;

  const IdType *coo_row_data = static_cast<IdType *>(coo.row->data);
  const IdType *coo_col_data = static_cast<IdType *>(coo.col->data);
  const IdType *coo_data =
      COOHasData(coo) ? static_cast<IdType *>(coo.data->data) : nullptr;

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
      end - start, coo.num_cols, NDArray::FromVector(ret_row),
      NDArray::FromVector(ret_col), NDArray::FromVector(ret_data),
      coo.row_sorted, coo.col_sorted);
}

template COOMatrix COOSliceRows<kDGLCPU, int32_t>(COOMatrix, int64_t, int64_t);
template COOMatrix COOSliceRows<kDGLCPU, int64_t>(COOMatrix, int64_t, int64_t);

template <DGLDeviceType XPU, typename IdType>
COOMatrix COOSliceRows(COOMatrix coo, NDArray rows) {
  const IdType *coo_row_data = static_cast<IdType *>(coo.row->data);
  const IdType *coo_col_data = static_cast<IdType *>(coo.col->data);
  const IdType *coo_data =
      COOHasData(coo) ? static_cast<IdType *>(coo.data->data) : nullptr;

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
      coo.row_sorted,
      coo.col_sorted};
}

template COOMatrix COOSliceRows<kDGLCPU, int32_t>(COOMatrix, NDArray);
template COOMatrix COOSliceRows<kDGLCPU, int64_t>(COOMatrix, NDArray);

///////////////////////////// COOSliceMatrix /////////////////////////////

template <DGLDeviceType XPU, typename IdType>
COOMatrix COOSliceMatrix(
    COOMatrix coo, runtime::NDArray rows, runtime::NDArray cols) {
  const IdType *coo_row_data = static_cast<IdType *>(coo.row->data);
  const IdType *coo_col_data = static_cast<IdType *>(coo.col->data);
  const IdType *coo_data =
      COOHasData(coo) ? static_cast<IdType *>(coo.data->data) : nullptr;

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

  return COOMatrix(
      rows->shape[0], cols->shape[0], NDArray::FromVector(ret_row),
      NDArray::FromVector(ret_col), NDArray::FromVector(ret_data),
      coo.row_sorted, coo.col_sorted);
}

template COOMatrix COOSliceMatrix<kDGLCPU, int32_t>(
    COOMatrix coo, runtime::NDArray rows, runtime::NDArray cols);
template COOMatrix COOSliceMatrix<kDGLCPU, int64_t>(
    COOMatrix coo, runtime::NDArray rows, runtime::NDArray cols);

///////////////////////////// COOReorder /////////////////////////////

template <DGLDeviceType XPU, typename IdType>
COOMatrix COOReorder(
    COOMatrix coo, runtime::NDArray new_row_id_arr,
    runtime::NDArray new_col_id_arr) {
  CHECK_SAME_DTYPE(coo.row, new_row_id_arr);
  CHECK_SAME_DTYPE(coo.col, new_col_id_arr);

  // Input COO
  const IdType *in_rows = static_cast<IdType *>(coo.row->data);
  const IdType *in_cols = static_cast<IdType *>(coo.col->data);
  int64_t num_rows = coo.num_rows;
  int64_t num_cols = coo.num_cols;
  int64_t nnz = coo.row->shape[0];
  CHECK_EQ(num_rows, new_row_id_arr->shape[0])
      << "The new row Id array needs to be the same as the number of rows of "
         "COO";
  CHECK_EQ(num_cols, new_col_id_arr->shape[0])
      << "The new col Id array needs to be the same as the number of cols of "
         "COO";

  // New row/col Ids.
  const IdType *new_row_ids = static_cast<IdType *>(new_row_id_arr->data);
  const IdType *new_col_ids = static_cast<IdType *>(new_col_id_arr->data);

  // Output COO
  NDArray out_row_arr = NDArray::Empty({nnz}, coo.row->dtype, coo.row->ctx);
  NDArray out_col_arr = NDArray::Empty({nnz}, coo.col->dtype, coo.col->ctx);
  NDArray out_data_arr = COOHasData(coo) ? coo.data : NullArray();
  IdType *out_row = static_cast<IdType *>(out_row_arr->data);
  IdType *out_col = static_cast<IdType *>(out_col_arr->data);

  parallel_for(0, nnz, [=](size_t b, size_t e) {
    for (auto i = b; i < e; ++i) {
      out_row[i] = new_row_ids[in_rows[i]];
      out_col[i] = new_col_ids[in_cols[i]];
    }
  });
  return COOMatrix(num_rows, num_cols, out_row_arr, out_col_arr, out_data_arr);
}

template COOMatrix COOReorder<kDGLCPU, int64_t>(
    COOMatrix csr, runtime::NDArray new_row_ids, runtime::NDArray new_col_ids);
template COOMatrix COOReorder<kDGLCPU, int32_t>(
    COOMatrix csr, runtime::NDArray new_row_ids, runtime::NDArray new_col_ids);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
