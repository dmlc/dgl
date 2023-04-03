/**
 *  Copyright (c) 2022 by Contributors
 * @file utils.h
 * @brief DGL C++ sparse API utilities
 */
#ifndef DGL_SPARSE_UTILS_H_
#define DGL_SPARSE_UTILS_H_

// clang-format off
#include <sparse/dgl_headers.h>
// clang-format on

#include <ATen/DLConvertor.h>
#include <sparse/sparse_matrix.h>
#include <torch/custom_class.h>
#include <torch/script.h>

namespace dgl {
namespace sparse {

/** @brief Find a proper sparse format for two sparse matrices. It chooses
 * COO if anyone of the sparse matrices has COO format. If none of them has
 * COO, it tries CSR and CSC in the same manner. */
inline static SparseFormat FindAnyExistingFormat(
    const c10::intrusive_ptr<SparseMatrix>& A,
    const c10::intrusive_ptr<SparseMatrix>& B) {
  SparseFormat fmt;
  if (A->HasCOO() || B->HasCOO()) {
    fmt = SparseFormat::kCOO;
  } else if (A->HasCSR() || B->HasCSR()) {
    fmt = SparseFormat::kCSR;
  } else {
    fmt = SparseFormat::kCSC;
  }
  return fmt;
}

/** @brief Check whether two matrices has the same dtype and shape for
 * elementwise operators. */
inline static void ElementwiseOpSanityCheck(
    const c10::intrusive_ptr<SparseMatrix>& A,
    const c10::intrusive_ptr<SparseMatrix>& B) {
  TORCH_CHECK(
      A->value().dtype() == B->value().dtype(),
      "Elementwise operators"
      " do not support two sparse matrices with different dtypes.");
  TORCH_CHECK(
      A->shape()[0] == B->shape()[0] && A->shape()[1] == B->shape()[1],
      "Elementwise operators do not support two sparse matrices with different"
      " shapes.");
}

/** @brief Convert a Torch tensor to a DGL array. */
inline static runtime::NDArray TorchTensorToDGLArray(torch::Tensor tensor) {
  return runtime::DLPackConvert::FromDLPack(at::toDLPack(tensor.contiguous()));
}

/** @brief Convert a DGL array to a Torch tensor. */
inline static torch::Tensor DGLArrayToTorchTensor(runtime::NDArray array) {
  return at::fromDLPack(runtime::DLPackConvert::ToDLPack(array));
}

/** @brief Convert an optional Torch tensor to a DGL array. */
inline static runtime::NDArray OptionalTorchTensorToDGLArray(
    torch::optional<torch::Tensor> tensor) {
  if (!tensor.has_value()) {
    return aten::NullArray();
  }
  return TorchTensorToDGLArray(tensor.value());
}

/** @brief Convert a DGL array to an optional Torch tensor. */
inline static torch::optional<torch::Tensor> DGLArrayToOptionalTorchTensor(
    runtime::NDArray array) {
  if (aten::IsNullArray(array)) {
    return torch::optional<torch::Tensor>();
  }
  return torch::make_optional<torch::Tensor>(DGLArrayToTorchTensor(array));
}

}  // namespace sparse
}  // namespace dgl

#endif  // DGL_SPARSE_UTILS_H_
