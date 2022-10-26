/*!
 *  Copyright (c) 2022 by Contributors
 * \file sparse/elementwise_op.h
 * \brief DGL C++ sparse elementwise operators
 */
#ifndef DGL_SPARSE_ELEMENTWISE_OP_H_
#define DGL_SPARSE_ELEMENTWISE_OP_H_

#include <torch/custom_class.h>

#include "./sparse_matrix.h"

namespace dgl {
namespace sparse {

c10::intrusive_ptr<SparseMatrix> Add(
    c10::intrusive_ptr<SparseMatrix> A, c10::intrusive_ptr<SparseMatrix> B);

}  // namespace sparse
}  // namespace dgl

#endif  // DGL_SPARSE_ELEMENTWISE_OP_H_