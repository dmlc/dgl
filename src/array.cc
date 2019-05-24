/*!
 *  Copyright (c) 2019 by Contributors
 * \file array.cc
 * \brief DGL array utilities implementation
 */
#include <dgl/array.h>

namespace dgl {

// TODO(minjie): currently these operators are only on CPU.

IdArray NewIdArray(int64_t length) {
  return IdArray::Empty({length}, DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
}

IdArray VecToIdArray(const std::vector<dgl_id_t>& vec) {
  IdArray ret = NewIdArray(vec.size());
  std::copy(vec.begin(), vec.end(), static_cast<dgl_id_t*>(ret->data));
  return ret;
}

IdArray Clone(IdArray arr) {
  IdArray ret = NewIdArray(arr->shape[0]);
  ret.CopyFrom(arr);
  return ret;
}

IdArray Add(IdArray lhs, IdArray rhs) {
  IdArray ret = NewIdArray(lhs->shape[0]);
  const dgl_id_t* lhs_data = static_cast<dgl_id_t*>(lhs->data);
  const dgl_id_t* rhs_data = static_cast<dgl_id_t*>(rhs->data);
  dgl_id_t* ret_data = static_cast<dgl_id_t*>(ret->data);
  for (int64_t i = 0; i < lhs->shape[0]; ++i) {
    ret_data[i] = lhs_data[i] + rhs_data[i];
  }
  return ret;
}

IdArray Sub(IdArray lhs, IdArray rhs) {
  IdArray ret = NewIdArray(lhs->shape[0]);
  const dgl_id_t* lhs_data = static_cast<dgl_id_t*>(lhs->data);
  const dgl_id_t* rhs_data = static_cast<dgl_id_t*>(rhs->data);
  dgl_id_t* ret_data = static_cast<dgl_id_t*>(ret->data);
  for (int64_t i = 0; i < lhs->shape[0]; ++i) {
    ret_data[i] = lhs_data[i] - rhs_data[i];
  }
  return ret;
}

IdArray Mul(IdArray lhs, IdArray rhs) {
  IdArray ret = NewIdArray(lhs->shape[0]);
  const dgl_id_t* lhs_data = static_cast<dgl_id_t*>(lhs->data);
  const dgl_id_t* rhs_data = static_cast<dgl_id_t*>(rhs->data);
  dgl_id_t* ret_data = static_cast<dgl_id_t*>(ret->data);
  for (int64_t i = 0; i < lhs->shape[0]; ++i) {
    ret_data[i] = lhs_data[i] * rhs_data[i];
  }
  return ret;
}

IdArray Div(IdArray lhs, IdArray rhs) {
  IdArray ret = NewIdArray(lhs->shape[0]);
  const dgl_id_t* lhs_data = static_cast<dgl_id_t*>(lhs->data);
  const dgl_id_t* rhs_data = static_cast<dgl_id_t*>(rhs->data);
  dgl_id_t* ret_data = static_cast<dgl_id_t*>(ret->data);
  for (int64_t i = 0; i < lhs->shape[0]; ++i) {
    ret_data[i] = lhs_data[i] / rhs_data[i];
  }
  return ret;
}

IdArray Add(IdArray lhs, dgl_id_t rhs) {
  IdArray ret = NewIdArray(lhs->shape[0]);
  const dgl_id_t* lhs_data = static_cast<dgl_id_t*>(lhs->data);
  dgl_id_t* ret_data = static_cast<dgl_id_t*>(ret->data);
  for (int64_t i = 0; i < lhs->shape[0]; ++i) {
    ret_data[i] = lhs_data[i] + rhs;
  }
  return ret;
}

IdArray Sub(IdArray lhs, dgl_id_t rhs) {
  IdArray ret = NewIdArray(lhs->shape[0]);
  const dgl_id_t* lhs_data = static_cast<dgl_id_t*>(lhs->data);
  dgl_id_t* ret_data = static_cast<dgl_id_t*>(ret->data);
  for (int64_t i = 0; i < lhs->shape[0]; ++i) {
    ret_data[i] = lhs_data[i] - rhs;
  }
  return ret;
}

IdArray Mul(IdArray lhs, dgl_id_t rhs) {
  IdArray ret = NewIdArray(lhs->shape[0]);
  const dgl_id_t* lhs_data = static_cast<dgl_id_t*>(lhs->data);
  dgl_id_t* ret_data = static_cast<dgl_id_t*>(ret->data);
  for (int64_t i = 0; i < lhs->shape[0]; ++i) {
    ret_data[i] = lhs_data[i] * rhs;
  }
  return ret;
}

IdArray Div(IdArray lhs, dgl_id_t rhs) {
  IdArray ret = NewIdArray(lhs->shape[0]);
  const dgl_id_t* lhs_data = static_cast<dgl_id_t*>(lhs->data);
  dgl_id_t* ret_data = static_cast<dgl_id_t*>(ret->data);
  for (int64_t i = 0; i < lhs->shape[0]; ++i) {
    ret_data[i] = lhs_data[i] / rhs;
  }
  return ret;
}

IdArray Add(dgl_id_t lhs, IdArray rhs) {
  return Add(rhs, lhs);
}

IdArray Sub(dgl_id_t lhs, IdArray rhs) {
  IdArray ret = NewIdArray(rhs->shape[0]);
  const dgl_id_t* rhs_data = static_cast<dgl_id_t*>(rhs->data);
  dgl_id_t* ret_data = static_cast<dgl_id_t*>(ret->data);
  for (int64_t i = 0; i < rhs->shape[0]; ++i) {
    ret_data[i] = lhs - rhs_data[i];
  }
  return ret;
}

IdArray Mul(dgl_id_t lhs, IdArray rhs) {
  return Mul(rhs, lhs);
}

IdArray Div(dgl_id_t lhs, IdArray rhs) {
  IdArray ret = NewIdArray(rhs->shape[0]);
  const dgl_id_t* rhs_data = static_cast<dgl_id_t*>(rhs->data);
  dgl_id_t* ret_data = static_cast<dgl_id_t*>(ret->data);
  for (int64_t i = 0; i < rhs->shape[0]; ++i) {
    ret_data[i] = lhs / rhs_data[i];
  }
  return ret;
}

IdArray HStack(IdArray arr1, IdArray arr2) {
  CHECK_EQ(arr1->shape[0], arr2->shape[0]);
  const int64_t L = arr1->shape[0];
  IdArray ret = NewIdArray(2 * L);
  const dgl_id_t* arr1_data = static_cast<dgl_id_t*>(arr1->data);
  const dgl_id_t* arr2_data = static_cast<dgl_id_t*>(arr2->data);
  dgl_id_t* ret_data = static_cast<dgl_id_t*>(ret->data);
  for (int64_t i = 0; i < L; ++i) {
    ret_data[i] = arr1_data[i];
    ret_data[i + L] = arr2_data[i];
  }
  return ret;
}

CSRMatrix SliceRows(const CSRMatrix& csr, int64_t start, int64_t end) {
  const dgl_id_t* indptr = static_cast<dgl_id_t*>(csr.indptr->data);
  const dgl_id_t* indices = static_cast<dgl_id_t*>(csr.indices->data);
  const dgl_id_t* data = static_cast<dgl_id_t*>(csr.data->data);
  const int64_t num_rows = end - start;
  const int64_t nnz = indptr[end] - indptr[start];
  CSRMatrix ret;
  ret.indptr = NewIdArray(num_rows + 1);
  ret.indices = NewIdArray(nnz);
  ret.data = NewIdArray(nnz);
  dgl_id_t* r_indptr = static_cast<dgl_id_t*>(ret.indptr->data);
  dgl_id_t* r_indices = static_cast<dgl_id_t*>(ret.indices->data);
  dgl_id_t* r_data = static_cast<dgl_id_t*>(ret.data->data);
  for (int64_t i = start; i < end + 1; ++i) {
    r_indptr[i - start] = indptr[i] - indptr[start];
  }
  std::copy(indices + indptr[start], indices + indptr[end], r_indices);
  std::copy(data + indptr[start], data + indptr[end], r_data);
  return ret;
}

}  // namespace dgl
