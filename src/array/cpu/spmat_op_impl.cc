/*!
 *  Copyright (c) 2019 by Contributors
 * \file array/cpu/spmat_op_impl.cc
 * \brief Sparse matrix operator CPU implementation
 */
#include <dgl/array.h>
#include <vector>

namespace dgl {

using runtime::NDArray;

namespace aten {
namespace impl {
namespace {
/*!
 * \brief A hashmap that maps each ids in the given array to new ids starting from zero.
 */
template <typename IdType>
class IdHashMap {
 public:
  // Construct the hashmap using the given id arrays.
  // The id array could contain duplicates.
  explicit IdHashMap(IdArray ids): filter_(kFilterSize, false) {
    const IdType* ids_data = static_cast<IdType*>(ids->data);
    const int64_t len = ids->shape[0];
    IdType newid = 0;
    for (int64_t i = 0; i < len; ++i) {
      const IdType id = ids_data[i];
      if (!Contains(id)) {
        oldv2newv_[id] = newid++;
        filter_[id & kFilterMask] = true;
      }
    }
  }

  // Return true if the given id is contained in this hashmap.
  bool Contains(IdType id) const {
    return filter_[id & kFilterMask] && oldv2newv_.count(id);
  }

  // Return the new id of the given id. If the given id is not contained
  // in the hash map, returns the default_val instead.
  IdType Map(IdType id, IdType default_val) const {
    if (filter_[id & kFilterMask]) {
      auto it = oldv2newv_.find(id);
      return (it == oldv2newv_.end()) ? default_val : it->second;
    } else {
      return default_val;
    }
  }

 private:
  static constexpr int32_t kFilterMask = 0xFFFFFF;
  static constexpr int32_t kFilterSize = kFilterMask + 1;
  // This bitmap is used as a bloom filter to remove some lookups.
  // Hashtable is very slow. Using bloom filter can significantly speed up lookups.
  std::vector<bool> filter_;
  // The hashmap from old vid to new vid
  std::unordered_map<IdType, IdType> oldv2newv_;
};

struct PairHash {
  template <class T1, class T2>
  std::size_t operator() (const std::pair<T1, T2>& pair) const {
    return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
  }
};
}  // namespace

///////////////////////////// CSRIsNonZero /////////////////////////////

template <DLDeviceType XPU, typename IdType, typename DType>
bool CSRIsNonZero(CSRMatrix csr, int64_t row, int64_t col) {
  CHECK(row >= 0 && row < csr.num_rows) << "Invalid row index: " << row;
  CHECK(col >= 0 && col < csr.num_cols) << "Invalid col index: " << col;
  const IdType* indptr_data = static_cast<IdType*>(csr.indptr->data);
  const IdType* indices_data = static_cast<IdType*>(csr.indices->data);
  for (IdType i = indptr_data[row]; i < indptr_data[col+1]; ++i) {
    if (indices_data[i] == col) {
      return true;
    }
  }
  return false;
}

template bool CSRIsNonZero<kDLCPU, int32_t, int32_t>(CSRMatrix, int64_t, int64_t);
template bool CSRIsNonZero<kDLCPU, int64_t, int64_t>(CSRMatrix, int64_t, int64_t);

///////////////////////////// CSRGetRowNNZ /////////////////////////////

template <DLDeviceType XPU, typename IdType, typename DType>
int64_t CSRGetRowNNZ(CSRMatrix csr, int64_t row) {
  CHECK(row >= 0 && row < csr.num_rows) << "Invalid row index: " << row;
  const IdType* indptr_data = static_cast<IdType*>(csr.indptr->data);
  return indptr_data[row + 1] - indptr_data[row];
}

template int64_t CSRGetRowNNZ<kDLCPU, int32_t, int32_t>(CSRMatrix, int64_t);
template int64_t CSRGetRowNNZ<kDLCPU, int64_t, int64_t>(CSRMatrix, int64_t);

///////////////////////////// CSRGetRowColumnIndices /////////////////////////////

template <DLDeviceType XPU, typename IdType, typename DType>
NDArray CSRGetRowColumnIndices(CSRMatrix csr, int64_t row) {
  CHECK(row >= 0 && row < csr.num_rows) << "Invalid row index: " << row;
  const int64_t len = CSRGetRowNNZ(csr, row);
  const IdType* indptr_data = static_cast<IdType*>(csr.indptr->data);
  const int64_t offset = indptr_data[row] * sizeof(IdType);
  return csr.indices.CreateView({len}, csr.indices->dtype, offset);
}

template NDArray CSRGetRowColumnIndices<kDLCPU, int32_t, int32_t>(CSRMatrix, int64_t);
template NDArray CSRGetRowColumnIndices<kDLCPU, int64_t, int64_t>(CSRMatrix, int64_t);

///////////////////////////// CSRGetRowData /////////////////////////////

template <DLDeviceType XPU, typename IdType, typename DType>
NDArray CSRGetRowData(CSRMatrix csr, int64_t row) {
  CHECK(row >= 0 && row < csr.num_rows) << "Invalid row index: " << row;
  const int64_t len = CSRGetRowNNZ(csr, row);
  const IdType* indptr_data = static_cast<IdType*>(csr.indptr->data);
  const int64_t offset = indptr_data[row] * sizeof(DType);
  return csr.indices.CreateView({len}, csr.data->dtype, offset);
}

template NDArray CSRGetRowData<kDLCPU, int32_t, int32_t>(CSRMatrix, int64_t);
template NDArray CSRGetRowData<kDLCPU, int64_t, int64_t>(CSRMatrix, int64_t);

///////////////////////////// CSRGetData /////////////////////////////

template <DLDeviceType XPU, typename IdType, typename DType>
NDArray CSRGetData(CSRMatrix csr, int64_t row, int64_t col) {
  // TODO(minjie): use more efficient binary search when the column indices
  CHECK(row >= 0 && row < csr.num_rows) << "Invalid row index: " << row;
  CHECK(col >= 0 && col < csr.num_cols) << "Invalid col index: " << col;
  std::vector<DType> ret_vec;
  const IdType* indptr_data = static_cast<IdType*>(csr.indptr->data);
  const IdType* indices_data = static_cast<IdType*>(csr.indices->data);
  const DType* data = static_cast<DType*>(csr.data->data);
  for (dgl_id_t i = indptr_data[row]; i < indptr_data[row+1]; ++i) {
    if (indices_data[i] == col) {
      ret_vec.push_back(data[i]);
    }
  }
  const int64_t len = ret_vec.size();
  NDArray ret_arr = NDArray::Empty({len}, csr.data->dtype, csr.data->ctx);
  DType* ptr = static_cast<DType*>(ret_arr->data);
  std::copy(ret_vec.begin(), ret_vec.end(), ptr);
  return ret_arr;
}

template NDArray CSRGetData<kDLCPU, int32_t, int32_t>(CSRMatrix, int64_t, int64_t);
template NDArray CSRGetData<kDLCPU, int64_t, int64_t>(CSRMatrix, int64_t, int64_t);

template <DLDeviceType XPU, typename IdType, typename DType>
NDArray CSRGetData(CSRMatrix csr, NDArray rows, NDArray cols) {
  // TODO(minjie): more efficient implementation for matrix without duplicate entries
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
  const DType* data = static_cast<DType*>(csr.data->data);

  std::vector<DType> ret_vec;

  for (int64_t i = 0, j = 0; i < rowlen && j < collen; i += row_stride, j += col_stride) {
    const IdType row_id = row_data[i], col_id = col_data[j];
    CHECK(row_id >= 0 && row_id < csr.num_rows) << "Invalid row index: " << row_id;
    CHECK(col_id >= 0 && col_id < csr.num_cols) << "Invalid col index: " << col_id;
    for (IdType i = indptr_data[row_id]; i < indptr_data[row_id+1]; ++i) {
      if (indices_data[i] == col_id) {
          ret_vec.push_back(data[i]);
      }
    }
  }

  const int64_t len = ret_vec.size();
  NDArray ret_arr = NDArray::Empty({len}, csr.data->dtype, csr.data->ctx);
  DType* ptr = static_cast<DType*>(ret_arr->data);
  std::copy(ret_vec.begin(), ret_vec.end(), ptr);
  return ret_arr;
}

template NDArray CSRGetData<kDLCPU, int32_t, int32_t>(CSRMatrix csr, NDArray rows, NDArray cols);
template NDArray CSRGetData<kDLCPU, int64_t, int64_t>(CSRMatrix csr, NDArray rows, NDArray cols);

///////////////////////////// CSRGetDataAndIndices /////////////////////////////

template <DLDeviceType XPU, typename IdType, typename DType>
std::vector<NDArray> CSRGetDataAndIndices(CSRMatrix csr, NDArray rows, NDArray cols) {
  // TODO(minjie): more efficient implementation for matrix without duplicate entries
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
  const DType* data = static_cast<DType*>(csr.data->data);

  std::vector<DType> ret_rows, ret_cols, ret_data;

  for (int64_t i = 0, j = 0; i < rowlen && j < collen; i += row_stride, j += col_stride) {
    const IdType row_id = row_data[i], col_id = col_data[j];
    CHECK(row_id >= 0 && row_id < csr.num_rows) << "Invalid row index: " << row_id;
    CHECK(col_id >= 0 && col_id < csr.num_cols) << "Invalid col index: " << col_id;
    for (IdType i = indptr_data[row_id]; i < indptr_data[row_id+1]; ++i) {
      if (indices_data[i] == col_id) {
          ret_rows.push_back(row_id);
          ret_cols.push_back(col_id);
          ret_data.push_back(data[i]);
      }
    }
  }

  return {VecToIdArray(ret_rows), VecToIdArray(ret_cols), VecToIdArray(ret_data)};
}

template std::vector<NDArray> CSRGetDataAndIndices<kDLCPU, int32_t, int32_t>(
    CSRMatrix csr, NDArray rows, NDArray cols);
template std::vector<NDArray> CSRGetDataAndIndices<kDLCPU, int64_t, int64_t>(
    CSRMatrix csr, NDArray rows, NDArray cols);

///////////////////////////// CSRTranspose /////////////////////////////

// for a matrix of shape (N, M) and NNZ
// complexity: time O(NNZ + max(N, M)), space O(1)
template <DLDeviceType XPU, typename IdType, typename DType>
CSRMatrix CSRTranspose(CSRMatrix csr) {
  const int64_t N = csr.num_rows;
  const int64_t M = csr.num_cols;
  const int64_t nnz = csr.indices->shape[0];
  const IdType* Ap = static_cast<IdType*>(csr.indptr->data);
  const IdType* Aj = static_cast<IdType*>(csr.indices->data);
  const DType* Ax = static_cast<DType*>(csr.data->data);
  NDArray ret_indptr = NDArray::Empty({M + 1}, csr.indptr->dtype, csr.indptr->ctx);
  NDArray ret_indices = NDArray::Empty({nnz}, csr.indices->dtype, csr.indices->ctx);
  NDArray ret_data = NDArray::Empty({nnz}, csr.data->dtype, csr.data->ctx);
  IdType* Bp = static_cast<IdType*>(ret_indptr->data);
  IdType* Bi = static_cast<IdType*>(ret_indices->data);
  DType* Bx = static_cast<DType*>(ret_data->data);

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
    for (IdType j = Ap[i]; j < Ap[i+1]; ++j) {
      const IdType dst = Aj[j];
      Bi[Bp[dst]] = i;
      Bx[Bp[dst]] = Ax[j];
      Bp[dst]++;
    }
  }

  // correct the indptr
  for (int64_t i = 0, last = 0; i <= M; ++i) {
    IdType temp = Bp[i];
    Bp[i] = last;
    last = temp;
  }

  return CSRMatrix{csr.num_cols, csr.num_rows, ret_indptr, ret_indices, ret_data};
}

template CSRMatrix CSRTranspose<kDLCPU, int32_t, int32_t>(CSRMatrix csr);
template CSRMatrix CSRTranspose<kDLCPU, int64_t, int64_t>(CSRMatrix csr);

///////////////////////////// CSRToCOO /////////////////////////////
template <DLDeviceType XPU, typename IdType>
COOMatrix CSRToCOO(CSRMatrix csr) {
  const int64_t nnz = csr.indices->shape[0];
  const IdType* indptr_data = static_cast<IdType*>(csr.indptr->data);
  NDArray ret_row = NDArray::Empty({nnz}, csr.indices->dtype, csr.indices->ctx);
  IdType* ret_row_data = static_cast<IdType*>(ret_row->data);
  for (IdType i = 0; i < csr.indptr->shape[0] - 1; ++i) {
    std::fill(ret_row_data + indptr_data[i],
              ret_row_data + indptr_data[i + 1],
              i);
  }
  return COOMatrix{csr.num_rows, csr.num_cols, ret_row, csr.indices, csr.data};
}

template COOMatrix CSRToCOO<kDLCPU, int32_t>(CSRMatrix csr);
template COOMatrix CSRToCOO<kDLCPU, int64_t>(CSRMatrix csr);

// complexity: time O(NNZ), space O(1)
template <DLDeviceType XPU, typename IdType>
COOMatrix CSRToCOODataAsOrder(CSRMatrix csr) {
  const int64_t N = csr.num_rows;
  const int64_t M = csr.num_cols;
  const int64_t nnz = csr.indices->shape[0];
  const IdType* indptr_data = static_cast<IdType*>(csr.indptr->data);
  const IdType* indices_data = static_cast<IdType*>(csr.indices->data);
  // data array should have the same type as the indices arrays
  const IdType* data = static_cast<IdType*>(csr.data->data);
  NDArray ret_row = NDArray::Empty({nnz}, csr.indices->dtype, csr.indices->ctx);
  NDArray ret_col = NDArray::Empty({nnz}, csr.indices->dtype, csr.indices->ctx);
  IdType* ret_row_data = static_cast<IdType*>(ret_row->data);
  IdType* ret_col_data = static_cast<IdType*>(ret_col->data);
  // scatter using the indices in the data array
  for (IdType row = 0; row < N; ++row) {
    for (IdType j = indptr_data[row]; j < indptr_data[row + 1]; ++j) {
      const IdType col = indices_data[j];
      ret_row_data[data[j]] = row;
      ret_col_data[data[j]] = col;
    }
  }
  COOMatrix coo;
  coo.num_rows = N;
  coo.num_cols = M;
  coo.row = ret_row;
  coo.col = ret_col;
  // no data array
  return coo;
}

template COOMatrix CSRToCOODataAsOrder<kDLCPU, int32_t>(CSRMatrix csr);
template COOMatrix CSRToCOODataAsOrder<kDLCPU, int64_t>(CSRMatrix csr);

///////////////////////////// CSRSliceRows /////////////////////////////

template <DLDeviceType XPU, typename IdType, typename DType>
CSRMatrix CSRSliceRows(CSRMatrix csr, int64_t start, int64_t end) {
  const IdType* indptr = static_cast<IdType*>(csr.indptr->data);
  const int64_t num_rows = end - start;
  const int64_t nnz = indptr[end] - indptr[start];
  CSRMatrix ret;
  ret.num_rows = num_rows;
  ret.num_cols = csr.num_cols;
  ret.indptr = NDArray::Empty({num_rows + 1}, csr.indptr->dtype, csr.indices->ctx);
  ret.indices = NDArray::Empty({nnz}, csr.indices->dtype, csr.indices->ctx);
  ret.data = NDArray::Empty({nnz}, csr.data->dtype, csr.data->ctx);
  IdType* r_indptr = static_cast<IdType*>(ret.indptr->data);
  for (int64_t i = start; i < end + 1; ++i) {
    r_indptr[i - start] = indptr[i] - indptr[start];
  }
  // indices and data can be view arrays
  ret.indices = csr.indices.CreateView({nnz}, csr.indices->dtype, indptr[start] * sizeof(IdType));
  ret.data = csr.data.CreateView({nnz}, csr.data->dtype, indptr[start] * sizeof(DType));
  return ret;
}

template CSRMatrix CSRSliceRows<kDLCPU, int32_t, int32_t>(CSRMatrix , int64_t , int64_t );
template CSRMatrix CSRSliceRows<kDLCPU, int64_t, int64_t>(CSRMatrix , int64_t , int64_t );

///////////////////////////// CSRSliceMatrix /////////////////////////////
template <DLDeviceType XPU, typename IdType, typename DType>
CSRMatrix CSRSliceMatrix(CSRMatrix csr, runtime::NDArray rows, runtime::NDArray cols) {
  IdHashMap<IdType> hashmap(cols);
  const int64_t new_nrows = rows->shape[0];
  const int64_t new_ncols = cols->shape[0];
  const IdType* rows_data = static_cast<IdType*>(rows->data);

  const IdType* indptr_data = static_cast<IdType*>(csr.indptr->data);
  const IdType* indices_data = static_cast<IdType*>(csr.indices->data);
  const DType* data = static_cast<DType*>(csr.data->data);

  std::vector<IdType> sub_indptr, sub_indices;
  std::vector<DType> sub_data;
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
        sub_data.push_back(data[p]);
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
  NDArray sub_data_arr = NDArray::Empty({nnz}, csr.data->dtype, csr.data->ctx);
  DType* ptr = static_cast<DType*>(sub_data_arr->data);
  std::copy(sub_data.begin(), sub_data.end(), ptr);
  return CSRMatrix{new_nrows, new_ncols,
    VecToIdArray(sub_indptr), VecToIdArray(sub_indices), sub_data_arr};
}

template CSRMatrix CSRSliceMatrix<kDLCPU, int32_t, int32_t>(
    CSRMatrix csr, runtime::NDArray rows, runtime::NDArray cols);
template CSRMatrix CSRSliceMatrix<kDLCPU, int64_t, int64_t>(
    CSRMatrix csr, runtime::NDArray rows, runtime::NDArray cols);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
