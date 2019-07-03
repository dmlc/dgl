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
  // TODO(minjie): only id data is supported
  return VecToIdArray(ret_vec);
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

  std::vector<DType> ret_data;

  for (int64_t i = 0, j = 0; i < rowlen && j < collen; i += row_stride, j += col_stride) {
    const IdType row_id = row_data[i], col_id = col_data[j];
    CHECK(row_id >= 0 && row_id < csr.num_rows) << "Invalid row index: " << row_id;
    CHECK(col_id >= 0 && col_id < csr.num_cols) << "Invalid col index: " << col_id;
    for (IdType i = indptr_data[row_id]; i < indptr_data[row_id+1]; ++i) {
      if (indices_data[i] == col_id) {
          //row.push_back(row_id);
          //col.push_back(col_id);
          ret_data.push_back(data[i]);
      }
    }
  }

  return VecToIdArray(ret_data);
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

///////////////////////////// CSRToCOODataAsOrder /////////////////////////////

// complexity: time O(NNZ), space O(1)
template <DLDeviceType XPU, typename IdType>
COOMatrix CSRToCOODataAsOrder(CSRMatrix csr) {
  const int64_t N = csr.num_rows;
  const int64_t M = csr.num_cols;
  const int64_t nnz = csr.indices->shape[0];
  const IdType* indptr_data = static_cast<IdType*>(csr.indptr->data);
  const IdType* indices_data = static_cast<IdType*>(csr.indices->data);
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

}  // namespace impl
}  // namespace aten
}  // namespace dgl
