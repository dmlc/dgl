/**
 *  Copyright (c) 2023 by Contributors
 * @file python_binding.cc
 * @brief Graph bolt library Python binding.
 */

#include "csc_sampling_graph.h"

namespace graphbolt {
namespace sampling {

TORCH_LIBRARY(graphbolt, m) {
  m.class_<CSCSamplingGraph>("CSCSamplingGraph")
      .def("num_rows", &CSCSamplingGraph::NumRows)
      .def("num_cols", &CSCSamplingGraph::NumCols)
      .def("num_nodes", &CSCSamplingGraph::NumNodes)
      .def("num_edges", &CSCSamplingGraph::NumEdges)
      .def("csc_indptr", &CSCSamplingGraph::CSCIndptr)
      .def("indices", &CSCSamplingGraph::Indices)
      .def("is_heterogeneous", &CSCSamplingGraph::IsHeterogeneous)
      .def("node_types", &CSCSamplingGraph::NodeTypes)
      .def("edge_types", &CSCSamplingGraph::EdgeTypes)
      .def("node_type_offset", &CSCSamplingGraph::NodeTypeOffset)
      .def("per_edge_type", &CSCSamplingGraph::PerEdgeType);
  m.def("from_csc", &CSCSamplingGraph::FromCSC);
  m.def("from_csc_with_hetero_info", &CSCSamplingGraph::FromCSCWithHeteroInfo);
}

}  // namespace sampling
}  // namespace graphbolt
