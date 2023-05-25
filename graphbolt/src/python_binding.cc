/**
 *  Copyright (c) 2023 by Contributors
 * @file python_binding.cc
 * @brief Graph bolt library Python binding.
 */

#include <graphbolt/csc_sampling_graph.h>

namespace graphbolt {
namespace sampling {

TORCH_LIBRARY(graphbolt, m) {
  m.class_<SampledSubgraph>("SampledSubgraph")
      .def(torch::init<>())
      .def_readwrite("indptr", &SampledSubgraph::indptr)
      .def_readwrite("indices", &SampledSubgraph::indices)
      .def_readwrite("reverse_row_node_ids", &SampledSubgraph::reverse_row_node_ids)
      .def_readwrite("reverse_column_node_ids", &SampledSubgraph::reverse_column_node_ids)
      .def_readwrite("reverse_edge_ids", &SampledSubgraph::reverse_edge_ids)
      .def_readwrite("type_per_edge", &SampledSubgraph::type_per_edge);
  m.class_<CSCSamplingGraph>("CSCSamplingGraph")
      .def("num_nodes", &CSCSamplingGraph::NumNodes)
      .def("num_edges", &CSCSamplingGraph::NumEdges)
      .def("csc_indptr", &CSCSamplingGraph::CSCIndptr)
      .def("indices", &CSCSamplingGraph::Indices)
      .def("node_type_offset", &CSCSamplingGraph::NodeTypeOffset)
      .def("type_per_edge", &CSCSamplingGraph::TypePerEdge)
      .def("sample_etype_neighbors", &CSCSamplingGraph::SampleEtypeNeighbors);
  m.def("from_csc", &CSCSamplingGraph::FromCSC);
}

}  // namespace sampling
}  // namespace graphbolt
