/**
 *  Copyright (c) 2022 by Contributors
 * @file python_binding.cc
 * @brief DGL sparse library Python binding.
 */
// clang-format off
#include <sparse/dgl_headers.h>
// clang-format on

#include <sparse/elementwise_op.h>
#include <sparse/matrix_ops.h>
#include <sparse/reduction.h>
#include <sparse/sddmm.h>
#include <sparse/softmax.h>
#include <sparse/sparse_matrix.h>
#include <sparse/spmm.h>
#include <sparse/spspmm.h>
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
      .def("indices", &SparseMatrix::Indices)
      .def("csr", &SparseMatrix::CSRTensors)
      .def("csc", &SparseMatrix::CSCTensors)
      .def("transpose", &SparseMatrix::Transpose)
      .def("coalesce", &SparseMatrix::Coalesce)
      .def("has_duplicate", &SparseMatrix::HasDuplicate)
      .def("is_diag", &SparseMatrix::HasDiag)
      .def("index_select", &SparseMatrix::IndexSelect)
      .def("range_select", &SparseMatrix::RangeSelect)
      .def("sample", &SparseMatrix::Sample);
  m.def("from_coo", &SparseMatrix::FromCOO)
      .def("from_csr", &SparseMatrix::FromCSR)
      .def("from_csc", &SparseMatrix::FromCSC)
      .def("from_diag", &SparseMatrix::FromDiag)
      .def("spsp_add", &SpSpAdd)
      .def("spsp_mul", &SpSpMul)
      .def("spsp_div", &SpSpDiv)
      .def("reduce", &Reduce)
      .def("sum", &ReduceSum)
      .def("smean", &ReduceMean)
      .def("smin", &ReduceMin)
      .def("smax", &ReduceMax)
      .def("sprod", &ReduceProd)
      .def("val_like", &SparseMatrix::ValLike)
      .def("spmm", &SpMM)
      .def("sddmm", &SDDMM)
      .def("softmax", &Softmax)
      .def("spspmm", &SpSpMM)
      .def("compact", &Compact);
}

}  // namespace sparse
}  // namespace dgl
