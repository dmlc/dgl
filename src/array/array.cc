/*!
 *  Copyright (c) 2019 by Contributors
 * \file array/array.cc
 * \brief DGL array utilities implementation
 */
#include <dgl/array.h>

namespace dgl {

using runtime::NDArray;

namespace aten {

// TODO(minjie): currently these operators are only on CPU.

IdArray NewIdArray(int64_t length, DLContext ctx, uint8_t nbits) {
  return IdArray::Empty({length}, DLDataType{kDLInt, nbits, 1}, ctx);
}

BoolArray NewBoolArray(int64_t length, DLContext ctx) {
  return BoolArray::Empty({length}, DLDataType{kDLInt, 64, 1}, ctx);
}

IdArray VecToIdArray(const std::vector<int32_t>& vec, DLContext ctx) {
  IdArray ret = NewIdArray(vec.size(), DLContext{kDLCPU, 0}, 32);
  std::copy(vec.begin(), vec.end(), static_cast<int32_t*>(ret->data));
  return ret.CopyTo(ctx);
}

IdArray VecToIdArray(const std::vector<int64_t>& vec, DLContext ctx) {
  IdArray ret = NewIdArray(vec.size(), DLContext{kDLCPU, 0}, 64);
  std::copy(vec.begin(), vec.end(), static_cast<int64_t*>(ret->data));
  return ret.CopyTo(ctx);
}

IdArray Clone(IdArray arr) {
  IdArray ret = NewIdArray(arr->shape[0]);
  ret.CopyFrom(arr);
  return ret;
}

IdArray AsNumBits(IdArray arr, uint8_t bits) {
  // TODO
}

IdArray Add(IdArray lhs, IdArray rhs) {
  // TODO
}

IdArray Sub(IdArray lhs, IdArray rhs) {
  // TODO
}

IdArray Mul(IdArray lhs, IdArray rhs) {
  // TODO
}

IdArray Div(IdArray lhs, IdArray rhs) {
  // TODO
}

IdArray Add(IdArray lhs, dgl_id_t rhs) {
  // TODO
}

IdArray Sub(IdArray lhs, dgl_id_t rhs) {
  // TODO
}

IdArray Mul(IdArray lhs, dgl_id_t rhs) {
  // TODO
}

IdArray Div(IdArray lhs, dgl_id_t rhs) {
  // TODO
}

IdArray Add(dgl_id_t lhs, IdArray rhs) {
  return Add(rhs, lhs);
}

IdArray Sub(dgl_id_t lhs, IdArray rhs) {
  // TODO
}

IdArray Mul(dgl_id_t lhs, IdArray rhs) {
  return Mul(rhs, lhs);
}

IdArray Div(dgl_id_t lhs, IdArray rhs) {
  // TODO
}

IdArray HStack(IdArray arr1, IdArray arr2) {
  // TODO
}

CSRMatrix SliceRows(const CSRMatrix& csr, int64_t start, int64_t end) {
  const dgl_id_t* indptr = static_cast<dgl_id_t*>(csr.indptr->data);
  const dgl_id_t* indices = static_cast<dgl_id_t*>(csr.indices->data);
  const dgl_id_t* data = static_cast<dgl_id_t*>(csr.data->data);
  const int64_t num_rows = end - start;
  const int64_t nnz = indptr[end] - indptr[start];
  CSRMatrix ret;
  ret.num_rows = num_rows;
  ret.num_cols = csr.num_cols;
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

}  // namespace aten
}  // namespace dgl
