 /**
 *  Copyright (c) 2023 by Contributors
 * @file python_binding.cc
 * @brief Graph bolt library Python binding.
 */

#include "csr_sampling_graph.h"

namespace graphbolt {
namespace sampling {

TORCH_LIBRARY(graphbolt, m) {
  m.class_<CSRSamplingGraph>("CsrGraph")
      .def("num_rows", &CSRSamplingGraph::NumRows)
      .def("num_cols", &CSRSamplingGraph::NumCols)
      .def("num_nodes", &CSRSamplingGraph::NumNodes)
      .def("num_edges", &CSRSamplingGraph::NumEdges)
      .def("indptr", &CSRSamplingGraph::IndPtr)
      .def("indices", &CSRSamplingGraph::Indices)
      .def("set_hetero_info", &CSRSamplingGraph::SetHeteroInfo)
      .def("is_heterogeneous", &CSRSamplingGraph::IsHeterogeneous)
      .def("node_types", &CSRSamplingGraph::NodeTypes)
      .def("edge_types", &CSRSamplingGraph::EdgeTypes)
      .def("node_type_offset", &CSRSamplingGraph::NodeTypeOffset)
      .def("per_edge_type", &CSRSamplingGraph::PerEdgeType);
  m.def("from_csr", &CSRSamplingGraph::FromCSR);
}

}  // namespace sampling
}  // namespace graphbolt
