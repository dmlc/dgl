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

CSRMatrix SliceRows(const CSRMatrix& csr, int64_t start, int64_t end) {
  // TODO
  LOG(FATAL) << "Not implemented.";
  return {};
}

}  // namespace dgl
