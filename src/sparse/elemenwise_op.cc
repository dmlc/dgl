/*!
 *  Copyright (c) 2022 by Contributors
 * \file sparse/elementwise_op.h
 * \brief DGL C++ sparse elementwise operator implementation
 */
#include <torch/custom_class.h>
#include <torch/script.h>

#include <memory>

#include "./elementwise_op.h"
#include "./sparse_matrix.h"
#include "./utils.h"

namespace dgl {
namespace sparse {

c10::intrusive_ptr<SparseMatrix> SpMatAddSpMat(
    const c10::intrusive_ptr<SparseMatrix>& A, const c10::intrusive_ptr<SparseMatrix>& B) {
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
