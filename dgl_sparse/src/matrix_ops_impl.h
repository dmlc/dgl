/**
 *  Copyright (c) 2023 by Contributors
 * @file matrix_ops_impl.h
 * @brief DGL C++ sparse matrix operator implementations.
 */
#ifndef DGL_SPARSE_MATRIX_OPS_IMPL_H_
#define DGL_SPARSE_MATRIX_OPS_IMPL_H_

#include <sparse/sparse_format.h>
#include <sparse/sparse_matrix.h>

#include <tuple>

#include "./utils.h"

namespace dgl {
namespace sparse {

template <c10::DeviceType XPU, typename IdType, typename ValType>
std::tuple<c10::intrusive_ptr<SparseMatrix>, torch::optional<torch::Tensor>>
CompactImpl(
    const c10::intrusive_ptr<SparseMatrix>& mat, int64_t dim,
    torch::optional<torch::Tensor> leading_indices) {
  auto csr = dim == 0 ? mat->CSRPtr() : mat->CSCPtr();
  auto ptr = csr->indptr;
  auto idx = csr->indices;
  auto val = mat->value();
  if (csr->value_indices.has_value()) {
    val = val.index_select(0, csr->value_indices.value());
  }
  auto ptr_acc = ptr.accessor<IdType, 1>();
  auto idx_acc = idx.accessor<IdType, 1>();
  auto val_acc = val.accessor<ValType, 1>();

  IdType edg_len = 0;
  std::vector<IdType> nnz, ret_ptr, ret_idx;
  std::vector<ValType> ret_val;
  // Bitmap indicates whether it has included in leading_indices
  std::vector<bool> bitmap(ptr.size(-1));
  ret_ptr.push_back(0);
  /// [ Debug and delete soon ]
  // printf("CSR format\n");
  // for (int i = 0; i < ptr.size(-1); i++) printf("%ld ", ptr_acc[i]);
  // printf("\n");
  // for (int i = 0; i < idx.size(-1); i++) printf("%ld ", idx_acc[i]);
  // printf("\n");
  // for (int i = 0; i < val.size(-1); i++) printf("%.3f ", val_acc[i]);
  // printf("\n-------- \n");

  if (leading_indices.has_value()) {
    torch::Tensor lead = leading_indices.value();
    auto lead_acc = lead.accessor<IdType, 1>();
    for (int i = 0; i < leading_indices->size(-1); i++) {
      for (int j = ptr_acc[lead_acc[i]]; j < ptr_acc[lead_acc[i] + 1]; j++) {
        ret_idx.push_back(idx_acc[j]);
        ret_val.push_back(val_acc[j]);
      }
      edg_len += ptr_acc[lead_acc[i] + 1] - ptr_acc[lead_acc[i]];
      ret_ptr.push_back(edg_len);
      nnz.push_back(lead_acc[i]);
      bitmap[lead_acc[i]] = true;
    }
  }
  for (int i = 0; i < ptr.size(-1) - 1; i++) {
    // If the row not in leading_indices and has non-zero elements.
    if (!bitmap[i] && ptr_acc[i] != ptr_acc[i + 1]) {
      for (int j = ptr_acc[i]; j < ptr_acc[i + 1]; j++) {
        ret_idx.push_back(idx_acc[j]);
        ret_val.push_back(val_acc[j]);
      }
      edg_len += ptr_acc[i + 1] - ptr_acc[i];
      ret_ptr.push_back(edg_len);
      nnz.push_back(i);
    }
  }
  long int nnz_size = nnz.size();
  if (dim == 0) {
    auto ret = SparseMatrix::FromCSR(
        VectorToTorchTensor(ret_ptr), VectorToTorchTensor(ret_idx),
        VectorToTorchTensor(ret_val),
        std::vector<int64_t>{nnz_size, csr->num_cols});
    auto ret_idx = torch::optional<torch::Tensor>(VectorToTorchTensor(nnz));
    return {ret, ret_idx};
  } else {
    auto ret = SparseMatrix::FromCSC(
        VectorToTorchTensor(ret_ptr), VectorToTorchTensor(ret_idx),
        VectorToTorchTensor(ret_val),
        std::vector<int64_t>{csr->num_cols, nnz_size});
    auto ret_idx = torch::optional<torch::Tensor>(VectorToTorchTensor(nnz));
    return {ret, ret_idx};
  }
}

}  // namespace sparse
}  // namespace dgl

#endif  // DGL_SPARSE_MATRIX_OPS_IMPL_H_
