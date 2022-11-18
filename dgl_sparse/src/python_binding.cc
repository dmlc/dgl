/**
 *  Copyright (c) 2022 by Contributors
 * @file python_binding.cc
 * @brief DGL sparse library Python binding.
 */
// clang-format off
#include <sparse/dgl_headers.h>
// clang-format on

#include <sparse/elementwise_op.h>
#include <sparse/sparse_matrix.h>
#include <torch/custom_class.h>
#include <torch/script.h>

namespace dgl {
namespace sparse {

TORCH_LIBRARY(dgl_sparse, m) {
  m.class_<SparseMatrix>("SparseMatrix")
      .def("val", &SparseMatrix::value)
      .def("nnz", &SparseMatrix::nnz)
      .def("device", &SparseMatrix::device)
      .def("shape", &SparseMatrix::shape)
      .def("coo", &SparseMatrix::COOTensors)
      .def("csr", &SparseMatrix::CSRTensors)
      .def("csc", &SparseMatrix::CSCTensors)
      .def("set_val", &SparseMatrix::SetValue);
  m.def("create_from_coo", &CreateFromCOO)
      .def("create_from_csr", &CreateFromCSR)
      .def("create_from_csc", &CreateFromCSC)
      .def("spsp_add", &SpSpAdd);
}

}  // namespace sparse
}  // namespace dgl
