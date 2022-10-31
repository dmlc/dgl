/*!
 *  Copyright (c) 2022 by Contributors
 * \file sparse/pybind.cc
 * \brief DGL sparse library Python binding
 */
#include <torch/custom_class.h>
#include <torch/script.h>

#include "./elementwise_op.h"
#include "./sparse_matrix.h"

namespace dgl {
namespace sparse {

TORCH_LIBRARY(dgl_sparse, m) {
  m.class_<SparseMatrix>("SparseMatrix")
      .def("coo", &SparseMatrix::COOTensors)
      .def("val", &SparseMatrix::Value)
      .def("shape", &SparseMatrix::Shape);
  m.def("create_from_coo", &CreateFromCOO)
      .def("spmat_add_spmat", &SpMatAddSpMat);
}

}  // namespace sparse
}  // namespace dgl
