 /**
 *  Copyright (c) 2023 by Contributors
 * @file python_binding.cc
 * @brief Graph bolt library Python binding.
 */

#include "csr_graph.h"

namespace graphbolt {
namespace sampling {

TORCH_LIBRARY(graphbolt, m) {
  m.class_<CSR>("CsrGraph")
      .def("num_rows", &CSR::NumRows)
      .def("num_cols", &CSR::NumCols)
      .def("indptr", &CSR::IndPtr)
      .def("indices", &CSR::Indices)
      .def("set_hetero_info", &CSR::SetHeteroInfo)
      .def("is_hetero", &CSR::IsHeterogeneous)
      .def("node_types", &CSR::GetNodeTypes)
      .def("edge_types", &CSR::GetEdgeTypes)
      .def("per_edge_types", &CSR::GetPerEdgeTypes);
  m.def("from_csr", &CSR::FromCSR);
}

}  // namespace sampling
}  // namespace graphbolt
