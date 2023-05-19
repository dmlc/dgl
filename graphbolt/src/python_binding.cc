/**
 *  Copyright (c) 2023 by Contributors
 * @file python_binding.cc
 * @brief Graph bolt library Python binding.
 */

#include "csc_sampling_graph.h"
#include "serialize.h"

namespace graphbolt {
namespace sampling {

TORCH_LIBRARY(graphbolt, m) {
  m.class_<CSCSamplingGraph>("CSCSamplingGraph")
      .def("num_nodes", &CSCSamplingGraph::NumNodes)
      .def("num_edges", &CSCSamplingGraph::NumEdges)
      .def("csc_indptr", &CSCSamplingGraph::CSCIndptr)
      .def("indices", &CSCSamplingGraph::Indices)
      .def("is_heterogeneous", &CSCSamplingGraph::IsHeterogeneous)
      .def("node_types", &CSCSamplingGraph::NodeTypes)
      .def("edge_types", &CSCSamplingGraph::EdgeTypes)
      .def("node_type_offset", &CSCSamplingGraph::NodeTypeOffset)
      .def("type_per_edge", &CSCSamplingGraph::TypePerEdge);
  m.def("from_csc", &CSCSamplingGraph::FromCSC);
  m.def("from_csc_with_hetero_info", &CSCSamplingGraph::FromCSCWithHeteroInfo);
  m.def("load_csc_sampling_graph", &LoadCSCSamplingGraph);
  m.def("save_csc_sampling_graph", &SaveCSCSamplingGraph);
}

}  // namespace sampling
}  // namespace graphbolt