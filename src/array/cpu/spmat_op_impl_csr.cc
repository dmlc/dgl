/**
 *  Copyright (c) 2019 by Contributors
 * @file array/cpu/spmat_op_impl_csr.cc
 * @brief CSR matrix operator CPU implementation
 */
#include <dgl/array.h>
#include <dgl/runtime/parallel_for.h>

#include <atomic>
#include <numeric>
#include <unordered_set>
#include <vector>

#include "array_utils.h"

namespace dgl {

using runtime::NDArray;
using runtime::parallel_for;

namespace aten {
namespace impl {

///////////////////////////// CSRIsNonZero /////////////////////////////

template <DGLDeviceType XPU, typename IdType>
bool CSRIsNonZero(CSRMatrix csr, int64_t row, int64_t col) {
  const IdType* indptr_data = static_cast<IdType*>(csr.indptr->data);
  const IdType* indices_data = static_cast<IdType*>(csr.indices->data);
  if (csr.sorted) {
    const IdType* start = indices_data + indptr_data[row];
    const IdType* end = indices_data + indptr_data[row + 1];
    return std::binary_search(start, end, col);
  } else {
    for (IdType i = indptr_data[row]; i < indptr_data[row + 1]; ++i) {
      if (indices_data[i] == col) {
        return true;
      }
    }
  }
  return false;
}

template bool CSRIsNonZero<kDGLCPU, int32_t>(CSRMatrix, int64_t, int64_t);
template bool CSRIsNonZero<kDGLCPU, int64_t>(CSRMatrix, int64_t, int64_t);

template <DGLDeviceType XPU, typename IdType>
NDArray CSRIsNonZero(CSRMatrix csr, NDArray row, NDArray col) {
  const auto rowlen = row->shape[0];
  const auto collen = col->shape[0];
  const auto rstlen = std::max(rowlen, collen);
  NDArray rst = NDArray::Empty({rstlen}, row->dtype, row->ctx);
  IdType* rst_data = static_cast<IdType*>(rst->data);
  const IdType* row_data = static_cast<IdType*>(row->data);
  const IdType* col_data = static_cast<IdType*>(col->data);
  const int64_t row_stride = (rowlen == 1 && collen != 1) ? 0 : 1;
  const int64_t col_stride = (collen == 1 && rowlen != 1) ? 0 : 1;
  runtime::parallel_for(
      0, std::max(rowlen, collen), 1, [=](int64_t b, int64_t e) {
        int64_t i = (row_stride == 0) ? 0 : b;
        int64_t j = (col_stride == 0) ? 0 : b;
        for (int64_t k = b; i < e && j < e;
             i += row_stride, j += col_stride, ++k)
          rst_data[k] =
              CSRIsNonZero<XPU, IdType>(csr, row_data[i], col_data[j]) ? 1 : 0;
      });
  return rst;
}

template NDArray CSRIsNonZero<kDGLCPU, int32_t>(CSRMatrix, NDArray, NDArray);
template NDArray CSRIsNonZero<kDGLCPU, int64_t>(CSRMatrix, NDArray, NDArray);

///////////////////////////// CSRHasDuplicate /////////////////////////////

template <DGLDeviceType XPU, typename IdType>
bool CSRHasDuplicate(CSRMatrix csr) {
  const IdType* indptr_data = static_cast<IdType*>(csr.indptr->data);
  const IdType* indices_data = static_cast<IdType*>(csr.indices->data);
  for (IdType src = 0; src < csr.num_rows; ++src) {
    std::unordered_set<IdType> hashmap;
    for (IdType eid = indptr_data[src]; eid < indptr_data[src + 1]; ++eid) {
      const IdType dst = indices_data[eid];
      if (hashmap.count(dst)) {
        return true;
      } else {
        hashmap.insert(dst);
      }
    }
  }
  return false;
}

template bool CSRHasDuplicate<kDGLCPU, int32_t>(CSRMatrix csr);
template bool CSRHasDuplicate<kDGLCPU, int64_t>(CSRMatrix csr);

///////////////////////////// CSRGetRowNNZ /////////////////////////////

template <DGLDeviceType XPU, typename IdType>
int64_t CSRGetRowNNZ(CSRMatrix csr, int64_t row) {
  const IdType* indptr_data = static_cast<IdType*>(csr.indptr->data);
  return indptr_data[row + 1] - indptr_data[row];
}

template int64_t CSRGetRowNNZ<kDGLCPU, int32_t>(CSRMatrix, int64_t);
template int64_t CSRGetRowNNZ<kDGLCPU, int64_t>(CSRMatrix, int64_t);

template <DGLDeviceType XPU, typename IdType>
NDArray CSRGetRowNNZ(CSRMatrix csr, NDArray rows) {
  CHECK_SAME_DTYPE(csr.indices, rows);
  const auto len = rows->shape[0];
  const IdType* vid_data = static_cast<IdType*>(rows->data);
  const IdType* indptr_data = static_cast<IdType*>(csr.indptr->data);
  NDArray rst = NDArray::Empty({len}, rows->dtype, rows->ctx);
  IdType* rst_data = static_cast<IdType*>(rst->data);
  for (int64_t i = 0; i < len; ++i) {
    const auto vid = vid_data[i];
    rst_data[i] = indptr_data[vid + 1] - indptr_data[vid];
  }
  return rst;
}

template NDArray CSRGetRowNNZ<kDGLCPU, int32_t>(CSRMatrix, NDArray);
template NDArray CSRGetRowNNZ<kDGLCPU, int64_t>(CSRMatrix, NDArray);

/////////////////////////// CSRGetRowColumnIndices /////////////////////////////

template <DGLDeviceType XPU, typename IdType>
NDArray CSRGetRowColumnIndices(CSRMatrix csr, int64_t row) {
  const int64_t len = impl::CSRGetRowNNZ<XPU, IdType>(csr, row);
  const IdType* indptr_data = static_cast<IdType*>(csr.indptr->data);
  const int64_t offset = indptr_data[row] * sizeof(IdType);
  return csr.indices.CreateView({len}, csr.indices->dtype, offset);
}

template NDArray CSRGetRowColumnIndices<kDGLCPU, int32_t>(CSRMatrix, int64_t);
template NDArray CSRGetRowColumnIndices<kDGLCPU, int64_t>(CSRMatrix, int64_t);

///////////////////////////// CSRGetRowData /////////////////////////////

template <DGLDeviceType XPU, typename IdType>
NDArray CSRGetRowData(CSRMatrix csr, int64_t row) {
  const int64_t len = impl::CSRGetRowNNZ<XPU, IdType>(csr, row);
  const IdType* indptr_data = static_cast<IdType*>(csr.indptr->data);
  const int64_t offset = indptr_data[row] * sizeof(IdType);
  if (CSRHasData(csr))
    return csr.data.CreateView({len}, csr.data->dtype, offset);
  else
    return aten::Range(
        offset, offset + len, csr.indptr->dtype.bits, csr.indptr->ctx);
}

template NDArray CSRGetRowData<kDGLCPU, int32_t>(CSRMatrix, int64_t);
template NDArray CSRGetRowData<kDGLCPU, int64_t>(CSRMatrix, int64_t);

///////////////////////////// CSRGetData /////////////////////////////
///////////////////////////// CSRGetDataAndIndices /////////////////////////////

template <DGLDeviceType XPU, typename IdType>
void CollectDataIndicesFromSorted(
    const IdType* indices_data, const IdType* data, const IdType start,
    const IdType end, const IdType col, std::vector<IdType>* col_vec,
    std::vector<IdType>* ret_vec) {
  const IdType* start_ptr = indices_data + start;
  const IdType* end_ptr = indices_data + end;
  auto it = std::lower_bound(start_ptr, end_ptr, col);
  // This might be a multi-graph. We need to collect all of the matched
  // columns.
  for (; it != end_ptr; it++) {
    // If the col exist
    if (*it == col) {
      IdType idx = it - indices_data;
      col_vec->push_back(indices_data[idx]);
      ret_vec->push_back(data[idx]);
    } else {
      // If we find a column that is different, we can stop searching now.
      break;
    }
  }
}

template <DGLDeviceType XPU, typename IdType>
std::vector<NDArray> CSRGetDataAndIndices(
    CSRMatrix csr, NDArray rows, NDArray cols) {
  // TODO(minjie): more efficient implementation for matrix without duplicate
  // entries
  const int64_t rowlen = rows->shape[0];
  const int64_t collen = cols->shape[0];

  CHECK((rowlen == collen) || (rowlen == 1) || (collen == 1))
      << "Invalid row and col id array.";

  const int64_t row_stride = (rowlen == 1 && collen != 1) ? 0 : 1;
  const int64_t col_stride = (collen == 1 && rowlen != 1) ? 0 : 1;
  const IdType* row_data = static_cast<IdType*>(rows->data);
  const IdType* col_data = static_cast<IdType*>(cols->data);

  const IdType* indptr_data = static_cast<IdType*>(csr.indptr->data);
  const IdType* indices_data = static_cast<IdType*>(csr.indices->data);
  const IdType* data =
      CSRHasData(csr) ? static_cast<IdType*>(csr.data->data) : nullptr;

  std::vector<IdType> ret_rows, ret_cols;
  std::vector<IdType> ret_data;

  for (int64_t i = 0, j = 0; i < rowlen && j < collen;
       i += row_stride, j += col_stride) {
    const IdType row_id = row_data[i], col_id = col_data[j];
    CHECK(row_id >= 0 && row_id < csr.num_rows)
        << "Invalid row index: " << row_id;
    CHECK(col_id >= 0 && col_id < csr.num_cols)
        << "Invalid col index: " << col_id;
    if (csr.sorted) {
      // Here we collect col indices and data.
      CollectDataIndicesFromSorted<XPU, IdType>(
          indices_data, data, indptr_data[row_id], indptr_data[row_id + 1],
          col_id, &ret_cols, &ret_data);
      // We need to add row Ids.
      while (ret_rows.size() < ret_data.size()) {
        ret_rows.push_back(row_id);
      }
    } else {
      for (IdType i = indptr_data[row_id]; i < indptr_data[row_id + 1]; ++i) {
        if (indices_data[i] == col_id) {
          ret_rows.push_back(row_id);
          ret_cols.push_back(col_id);
          ret_data.push_back(data ? data[i] : i);
        }
      }
    }
  }

  return {
      NDArray::FromVector(ret_rows, csr.indptr->ctx),
      NDArray::FromVector(ret_cols, csr.indptr->ctx),
      NDArray::FromVector(ret_data, csr.data->ctx)};
}

template std::vector<NDArray> CSRGetDataAndIndices<kDGLCPU, int32_t>(
    CSRMatrix csr, NDArray rows, NDArray cols);
template std::vector<NDArray> CSRGetDataAndIndices<kDGLCPU, int64_t>(
    CSRMatrix csr, NDArray rows, NDArray cols);

///////////////////////////// CSRTranspose /////////////////////////////

// for a matrix of shape (N, M) and NNZ
// complexity: time O(NNZ + max(N, M)), space O(1)
template <DGLDeviceType XPU, typename IdType>
CSRMatrix CSRTranspose(CSRMatrix csr) {
  const int64_t N = csr.num_rows;
  const int64_t M = csr.num_cols;
  const int64_t nnz = csr.indices->shape[0];
  const IdType* Ap = static_cast<IdType*>(csr.indptr->data);
  const IdType* Aj = static_cast<IdType*>(csr.indices->data);
  const IdType* Ax =
      CSRHasData(csr) ? static_cast<IdType*>(csr.data->data) : nullptr;
  NDArray ret_indptr =
      NDArray::Empty({M + 1}, csr.indptr->dtype, csr.indptr->ctx);
  NDArray ret_indices =
      NDArray::Empty({nnz}, csr.indices->dtype, csr.indices->ctx);
  NDArray ret_data = NDArray::Empty({nnz}, csr.indptr->dtype, csr.indptr->ctx);
  IdType* Bp = static_cast<IdType*>(ret_indptr->data);
  IdType* Bi = static_cast<IdType*>(ret_indices->data);
  IdType* Bx = static_cast<IdType*>(ret_data->data);

  std::fill(Bp, Bp + M, 0);

  for (int64_t j = 0; j < nnz; ++j) {
    Bp[Aj[j]]++;
  }

  // cumsum
  for (int64_t i = 0, cumsum = 0; i < M; ++i) {
    const IdType temp = Bp[i];
    Bp[i] = cumsum;
    cumsum += temp;
  }
  Bp[M] = nnz;

  for (int64_t i = 0; i < N; ++i) {
    for (IdType j = Ap[i]; j < Ap[i + 1]; ++j) {
      const IdType dst = Aj[j];
      Bi[Bp[dst]] = i;
      Bx[Bp[dst]] = Ax ? Ax[j] : j;
      Bp[dst]++;
    }
  }

  // correct the indptr
  for (int64_t i = 0, last = 0; i <= M; ++i) {
    IdType temp = Bp[i];
    Bp[i] = last;
    last = temp;
  }

  return CSRMatrix{
      csr.num_cols, csr.num_rows, ret_indptr, ret_indices, ret_data};
}

template CSRMatrix CSRTranspose<kDGLCPU, int32_t>(CSRMatrix csr);
template CSRMatrix CSRTranspose<kDGLCPU, int64_t>(CSRMatrix csr);

///////////////////////////// CSRToCOO /////////////////////////////
template <DGLDeviceType XPU, typename IdType>
COOMatrix CSRToCOO(CSRMatrix csr) {
  const int64_t nnz = csr.indices->shape[0];
  const IdType* indptr_data = static_cast<IdType*>(csr.indptr->data);
  NDArray ret_row = NDArray::Empty({nnz}, csr.indices->dtype, csr.indices->ctx);
  IdType* ret_row_data = static_cast<IdType*>(ret_row->data);
  parallel_for(0, csr.indptr->shape[0] - 1, 10000, [=](int64_t b, int64_t e) {
    for (auto i = b; i < e; ++i) {
      std::fill(
          ret_row_data + indptr_data[i], ret_row_data + indptr_data[i + 1], i);
    }
  });
  return COOMatrix(
      csr.num_rows, csr.num_cols, ret_row, csr.indices, csr.data, true,
      csr.sorted);
}

template COOMatrix CSRToCOO<kDGLCPU, int32_t>(CSRMatrix csr);
template COOMatrix CSRToCOO<kDGLCPU, int64_t>(CSRMatrix csr);

// complexity: time O(NNZ), space O(1)
template <DGLDeviceType XPU, typename IdType>
COOMatrix CSRToCOODataAsOrder(CSRMatrix csr) {
  const int64_t N = csr.num_rows;
  const int64_t M = csr.num_cols;
  const int64_t nnz = csr.indices->shape[0];
  const IdType* indptr_data = static_cast<IdType*>(csr.indptr->data);
  const IdType* indices_data = static_cast<IdType*>(csr.indices->data);
  // data array should have the same type as the indices arrays
  const IdType* data =
      CSRHasData(csr) ? static_cast<IdType*>(csr.data->data) : nullptr;
  NDArray ret_row = NDArray::Empty({nnz}, csr.indices->dtype, csr.indices->ctx);
  NDArray ret_col = NDArray::Empty({nnz}, csr.indices->dtype, csr.indices->ctx);
  IdType* ret_row_data = static_cast<IdType*>(ret_row->data);
  IdType* ret_col_data = static_cast<IdType*>(ret_col->data);
  // scatter using the indices in the data array
  parallel_for(0, N, 10000, [=](int64_t b, int64_t e) {
    for (auto row = b; row < e; ++row) {
      for (IdType j = indptr_data[row]; j < indptr_data[row + 1]; ++j) {
        const IdType col = indices_data[j];
        ret_row_data[data ? data[j] : j] = row;
        ret_col_data[data ? data[j] : j] = col;
      }
    }
  });
  return COOMatrix(N, M, ret_row, ret_col);
}

template COOMatrix CSRToCOODataAsOrder<kDGLCPU, int32_t>(CSRMatrix csr);
template COOMatrix CSRToCOODataAsOrder<kDGLCPU, int64_t>(CSRMatrix csr);

///////////////////////////// CSRSliceRows /////////////////////////////

template <DGLDeviceType XPU, typename IdType>
CSRMatrix CSRSliceRows(CSRMatrix csr, int64_t start, int64_t end) {
  const IdType* indptr = static_cast<IdType*>(csr.indptr->data);
  const int64_t num_rows = end - start;
  const int64_t nnz = indptr[end] - indptr[start];
  IdArray ret_indptr =
      IdArray::Empty({num_rows + 1}, csr.indptr->dtype, csr.indices->ctx);
  IdType* r_indptr = static_cast<IdType*>(ret_indptr->data);
  for (int64_t i = start; i < end + 1; ++i) {
    r_indptr[i - start] = indptr[i] - indptr[start];
  }
  // indices and data can be view arrays
  IdArray ret_indices = csr.indices.CreateView(
      {nnz}, csr.indices->dtype, indptr[start] * sizeof(IdType));
  IdArray ret_data;
  if (CSRHasData(csr))
    ret_data = csr.data.CreateView(
        {nnz}, csr.data->dtype, indptr[start] * sizeof(IdType));
  else
    ret_data = aten::Range(
        indptr[start], indptr[end], csr.indptr->dtype.bits, csr.indptr->ctx);
  return CSRMatrix(
      num_rows, csr.num_cols, ret_indptr, ret_indices, ret_data, csr.sorted);
}

template CSRMatrix CSRSliceRows<kDGLCPU, int32_t>(CSRMatrix, int64_t, int64_t);
template CSRMatrix CSRSliceRows<kDGLCPU, int64_t>(CSRMatrix, int64_t, int64_t);

template <DGLDeviceType XPU, typename IdType>
CSRMatrix CSRSliceRows(CSRMatrix csr, NDArray rows) {
  CHECK_SAME_DTYPE(csr.indices, rows);
  const IdType* indptr_data = static_cast<IdType*>(csr.indptr->data);
  const IdType* indices_data = static_cast<IdType*>(csr.indices->data);
  const IdType* data =
      CSRHasData(csr) ? static_cast<IdType*>(csr.data->data) : nullptr;
  const auto len = rows->shape[0];
  const IdType* rows_data = static_cast<IdType*>(rows->data);
  int64_t nnz = 0;

  CSRMatrix ret;
  ret.num_rows = len;
  ret.num_cols = csr.num_cols;
  ret.indptr = NDArray::Empty({len + 1}, csr.indptr->dtype, csr.indices->ctx);

  IdType* ret_indptr_data = static_cast<IdType*>(ret.indptr->data);
  ret_indptr_data[0] = 0;

  std::vector<IdType> sums;

  std::atomic_flag err_flag = ATOMIC_FLAG_INIT;
  bool err = false;
  std::stringstream err_msg_stream;

// Perform two-round parallel prefix sum using OpenMP
#pragma omp parallel
  {
    int64_t tid = omp_get_thread_num();
    int64_t num_threads = omp_get_num_threads();

#pragma omp single
    {
      sums.resize(num_threads + 1);
      sums[0] = 0;
    }

    int64_t sum = 0;

// First round of parallel prefix sum. All threads perform local prefix sums.
#pragma omp for schedule(static) nowait
    for (int64_t i = 0; i < len; ++i) {
      int64_t rid = rows_data[i];
      if (rid >= csr.num_rows) {
        if (!err_flag.test_and_set()) {
          err_msg_stream << "expect row ID " << rid
                         << " to be less than number of rows " << csr.num_rows;
          err = true;
        }
      } else {
        sum += indptr_data[rid + 1] - indptr_data[rid];
        ret_indptr_data[i + 1] = sum;
      }
    }
    sums[tid + 1] = sum;
#pragma omp barrier

#pragma omp single
    {
      for (int64_t i = 1; i < num_threads; ++i) sums[i] += sums[i - 1];
    }

    int64_t offset = sums[tid];

// Second round of parallel prefix sum. Update the local prefix sums.
#pragma omp for schedule(static)
    for (int64_t i = 0; i < len; ++i) ret_indptr_data[i + 1] += offset;
  }
  if (err) {
    LOG(FATAL) << err_msg_stream.str();
    return ret;
  }

  // After the prefix sum, the last element of ret_indptr_data holds the
  // sum of all elements
  nnz = ret_indptr_data[len];

  ret.indices = NDArray::Empty({nnz}, csr.indices->dtype, csr.indices->ctx);
  ret.data = NDArray::Empty({nnz}, csr.indptr->dtype, csr.indptr->ctx);
  ret.sorted = csr.sorted;

  IdType* ret_indices_data = static_cast<IdType*>(ret.indices->data);
  IdType* ret_data = static_cast<IdType*>(ret.data->data);

  parallel_for(0, len, [=](int64_t b, int64_t e) {
    for (auto i = b; i < e; ++i) {
      const IdType rid = rows_data[i];
      // note: zero is allowed
      std::copy(
          indices_data + indptr_data[rid], indices_data + indptr_data[rid + 1],
          ret_indices_data + ret_indptr_data[i]);
      if (data)
        std::copy(
            data + indptr_data[rid], data + indptr_data[rid + 1],
            ret_data + ret_indptr_data[i]);
      else
        std::iota(
            ret_data + ret_indptr_data[i], ret_data + ret_indptr_data[i + 1],
            indptr_data[rid]);
    }
  });
  return ret;
}

template CSRMatrix CSRSliceRows<kDGLCPU, int32_t>(CSRMatrix, NDArray);
template CSRMatrix CSRSliceRows<kDGLCPU, int64_t>(CSRMatrix, NDArray);

///////////////////////////// CSRSliceMatrix /////////////////////////////

template <DGLDeviceType XPU, typename IdType>
CSRMatrix CSRSliceMatrix(
    CSRMatrix csr, runtime::NDArray rows, runtime::NDArray cols) {
  IdHashMap<IdType> hashmap(cols);
  const int64_t new_nrows = rows->shape[0];
  const int64_t new_ncols = cols->shape[0];
  const IdType* rows_data = static_cast<IdType*>(rows->data);
  const bool has_data = CSRHasData(csr);

  const IdType* indptr_data = static_cast<IdType*>(csr.indptr->data);
  const IdType* indices_data = static_cast<IdType*>(csr.indices->data);
  const IdType* data =
      has_data ? static_cast<IdType*>(csr.data->data) : nullptr;

  std::vector<IdType> sub_indptr, sub_indices;
  std::vector<IdType> sub_data;
  sub_indptr.resize(new_nrows + 1, 0);
  const IdType kInvalidId = new_ncols + 1;
  for (int64_t i = 0; i < new_nrows; ++i) {
    // NOTE: newi == i
    const IdType oldi = rows_data[i];
    CHECK(oldi >= 0 && oldi < csr.num_rows) << "Invalid row index: " << oldi;
    for (IdType p = indptr_data[oldi]; p < indptr_data[oldi + 1]; ++p) {
      const IdType oldj = indices_data[p];
      const IdType newj = hashmap.Map(oldj, kInvalidId);
      if (newj != kInvalidId) {
        ++sub_indptr[i];
        sub_indices.push_back(newj);
        sub_data.push_back(has_data ? data[p] : p);
      }
    }
  }

  // cumsum sub_indptr
  for (int64_t i = 0, cumsum = 0; i < new_nrows; ++i) {
    const IdType temp = sub_indptr[i];
    sub_indptr[i] = cumsum;
    cumsum += temp;
  }
  sub_indptr[new_nrows] = sub_indices.size();

  const int64_t nnz = sub_data.size();
  NDArray sub_data_arr =
      NDArray::Empty({nnz}, csr.indptr->dtype, csr.indptr->ctx);
  IdType* ptr = static_cast<IdType*>(sub_data_arr->data);
  std::copy(sub_data.begin(), sub_data.end(), ptr);
  return CSRMatrix{
      new_nrows, new_ncols, NDArray::FromVector(sub_indptr, csr.indptr->ctx),
      NDArray::FromVector(sub_indices, csr.indptr->ctx), sub_data_arr};
}

template CSRMatrix CSRSliceMatrix<kDGLCPU, int32_t>(
    CSRMatrix csr, runtime::NDArray rows, runtime::NDArray cols);
template CSRMatrix CSRSliceMatrix<kDGLCPU, int64_t>(
    CSRMatrix csr, runtime::NDArray rows, runtime::NDArray cols);

///////////////////////////// CSRReorder /////////////////////////////

template <DGLDeviceType XPU, typename IdType>
CSRMatrix CSRReorder(
    CSRMatrix csr, runtime::NDArray new_row_id_arr,
    runtime::NDArray new_col_id_arr) {
  CHECK_SAME_DTYPE(csr.indices, new_row_id_arr);
  CHECK_SAME_DTYPE(csr.indices, new_col_id_arr);

  // Input CSR
  const IdType* in_indptr = static_cast<IdType*>(csr.indptr->data);
  const IdType* in_indices = static_cast<IdType*>(csr.indices->data);
  const IdType* in_data = static_cast<IdType*>(csr.data->data);
  int64_t num_rows = csr.num_rows;
  int64_t num_cols = csr.num_cols;
  int64_t nnz = csr.indices->shape[0];
  CHECK_EQ(nnz, in_indptr[num_rows]);
  CHECK_EQ(num_rows, new_row_id_arr->shape[0])
      << "The new row Id array needs to be the same as the number of rows of "
         "CSR";
  CHECK_EQ(num_cols, new_col_id_arr->shape[0])
      << "The new col Id array needs to be the same as the number of cols of "
         "CSR";

  // New row/col Ids.
  const IdType* new_row_ids = static_cast<IdType*>(new_row_id_arr->data);
  const IdType* new_col_ids = static_cast<IdType*>(new_col_id_arr->data);

  // Output CSR
  NDArray out_indptr_arr =
      NDArray::Empty({num_rows + 1}, csr.indptr->dtype, csr.indptr->ctx);
  NDArray out_indices_arr =
      NDArray::Empty({nnz}, csr.indices->dtype, csr.indices->ctx);
  NDArray out_data_arr = NDArray::Empty({nnz}, csr.data->dtype, csr.data->ctx);
  IdType* out_indptr = static_cast<IdType*>(out_indptr_arr->data);
  IdType* out_indices = static_cast<IdType*>(out_indices_arr->data);
  IdType* out_data = static_cast<IdType*>(out_data_arr->data);

  // Compute the length of rows for the new matrix.
  std::vector<IdType> new_row_lens(num_rows, -1);
  parallel_for(0, num_rows, [=, &new_row_lens](size_t b, size_t e) {
    for (auto i = b; i < e; ++i) {
      int64_t new_row_id = new_row_ids[i];
      new_row_lens[new_row_id] = in_indptr[i + 1] - in_indptr[i];
    }
  });
  // Compute the starting location of each row in the new matrix.
  out_indptr[0] = 0;
  // This is sequential. It should be pretty fast.
  for (int64_t i = 0; i < num_rows; i++) {
    CHECK_GE(new_row_lens[i], 0);
    out_indptr[i + 1] = out_indptr[i] + new_row_lens[i];
  }
  CHECK_EQ(out_indptr[num_rows], nnz);
  // Copy indieces and data with the new order.
  // Here I iterate rows in the order of the old matrix.
  parallel_for(0, num_rows, [=](size_t b, size_t e) {
    for (auto i = b; i < e; ++i) {
      const IdType* in_row = in_indices + in_indptr[i];
      const IdType* in_row_data = in_data + in_indptr[i];

      int64_t new_row_id = new_row_ids[i];
      IdType* out_row = out_indices + out_indptr[new_row_id];
      IdType* out_row_data = out_data + out_indptr[new_row_id];

      int64_t row_len = new_row_lens[new_row_id];
      // Here I iterate col indices in a row in the order of the old matrix.
      for (int64_t j = 0; j < row_len; j++) {
        out_row[j] = new_col_ids[in_row[j]];
        out_row_data[j] = in_row_data[j];
      }
      // TODO(zhengda) maybe we should sort the column indices.
    }
  });
  return CSRMatrix(
      num_rows, num_cols, out_indptr_arr, out_indices_arr, out_data_arr);
}

template CSRMatrix CSRReorder<kDGLCPU, int64_t>(
    CSRMatrix csr, runtime::NDArray new_row_ids, runtime::NDArray new_col_ids);
template CSRMatrix CSRReorder<kDGLCPU, int32_t>(
    CSRMatrix csr, runtime::NDArray new_row_ids, runtime::NDArray new_col_ids);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
