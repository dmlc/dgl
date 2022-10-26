/*!
 *  Copyright (c) 2022 by Contributors
 * \file sparse/elementwise_op.h
 * \brief DGL C++ sparse elementwise operators
 */
#ifndef DGL_SPARSE_ELEMENTWISE_OP_H_
#define DGL_SPARSE_ELEMENTWISE_OP_H_

#include <torch/custom_class.h>

#include <memory>

#include "./sparse_matrix.h"
#include "./utils.h"

namespace dgl {
namespace sparse {

c10::intrusive_ptr<SparseMatrix> Add(
    c10::intrusive_ptr<SparseMatrix> A, c10::intrusive_ptr<SparseMatrix> B) {
  SPARSE_FORMAT_SELECT_BINARY(A, B, fmt, {
    if (fmt == SparseFormat::kCOO) {
      auto value = A->Value() + B->Value();
      auto ret = CreateFromCOOPtr(A->COOPtr(), value, A->Shape());
      return ret;
    } else {
      // TODO
    }
  });
}

}  // namespace sparse
}  // namespace dgl

#endif  //  DGL_SPARSE_ELEMENTWISE_OP_H_