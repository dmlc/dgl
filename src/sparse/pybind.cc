#include <torch/custom_class.h>
#include <torch/script.h>

#include "./sparse_matrix.h"

TORCH_LIBRARY(dgl_sparse, m) {
  m.class_<SparseMatrix>("SparseMatrix")
      .def(torch::init<torch::Tensor, torch::Tensor,
                       torch::optional<torch::Tensor>,
                       const std::vector<int64_t>&>(),
      .def("row", &SparseMatrix::Row))
}