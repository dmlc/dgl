/**
 *  Copyright (c) 2023 by Contributors
 * @file python_binding.cc
 * @brief Graph bolt library Python binding.
 */

#include <graphbolt/fused_csc_sampling_graph.h>
#include <graphbolt/isin.h>
#include <graphbolt/serialize.h>
#include <graphbolt/unique_and_compact.h>

#ifdef GRAPHBOLT_USE_CUDA
#include "./cuda/max_uva_threads.h"
#endif
#include "./expand_indptr.h"
#include "./index_select.h"
#include "./random.h"

#ifdef GRAPHBOLT_USE_CUDA
#include "./cuda/gpu_cache.h"
#endif

namespace graphbolt {
namespace sampling {

TORCH_LIBRARY(graphbolt, m) {
  m.impl_abstract_pystub("graphbolt", "//dgl.graphbolt");
  m.class_<FusedSampledSubgraph>("FusedSampledSubgraph")
      .def(torch::init<>())
      .def_readwrite("indptr", &FusedSampledSubgraph::indptr)
      .def_readwrite("indices", &FusedSampledSubgraph::indices)
      .def_readwrite(
          "original_row_node_ids", &FusedSampledSubgraph::original_row_node_ids)
      .def_readwrite(
          "original_column_node_ids",
          &FusedSampledSubgraph::original_column_node_ids)
      .def_readwrite(
          "original_edge_ids", &FusedSampledSubgraph::original_edge_ids)
      .def_readwrite("type_per_edge", &FusedSampledSubgraph::type_per_edge);
  m.class_<FusedCSCSamplingGraph>("FusedCSCSamplingGraph")
      .def("num_nodes", &FusedCSCSamplingGraph::NumNodes)
      .def("num_edges", &FusedCSCSamplingGraph::NumEdges)
      .def("csc_indptr", &FusedCSCSamplingGraph::CSCIndptr)
      .def("indices", &FusedCSCSamplingGraph::Indices)
      .def("node_type_offset", &FusedCSCSamplingGraph::NodeTypeOffset)
      .def("type_per_edge", &FusedCSCSamplingGraph::TypePerEdge)
      .def("node_type_to_id", &FusedCSCSamplingGraph::NodeTypeToID)
      .def("edge_type_to_id", &FusedCSCSamplingGraph::EdgeTypeToID)
      .def("node_attributes", &FusedCSCSamplingGraph::NodeAttributes)
      .def("edge_attributes", &FusedCSCSamplingGraph::EdgeAttributes)
      .def("set_csc_indptr", &FusedCSCSamplingGraph::SetCSCIndptr)
      .def("set_indices", &FusedCSCSamplingGraph::SetIndices)
      .def("set_node_type_offset", &FusedCSCSamplingGraph::SetNodeTypeOffset)
      .def("set_type_per_edge", &FusedCSCSamplingGraph::SetTypePerEdge)
      .def("set_node_type_to_id", &FusedCSCSamplingGraph::SetNodeTypeToID)
      .def("set_edge_type_to_id", &FusedCSCSamplingGraph::SetEdgeTypeToID)
      .def("set_node_attributes", &FusedCSCSamplingGraph::SetNodeAttributes)
      .def("set_edge_attributes", &FusedCSCSamplingGraph::SetEdgeAttributes)
      .def("in_subgraph", &FusedCSCSamplingGraph::InSubgraph)
      .def("sample_neighbors", &FusedCSCSamplingGraph::SampleNeighbors)
      .def(
          "temporal_sample_neighbors",
          &FusedCSCSamplingGraph::TemporalSampleNeighbors)
      .def("copy_to_shared_memory", &FusedCSCSamplingGraph::CopyToSharedMemory)
      .def_pickle(
          // __getstate__
          [](const c10::intrusive_ptr<FusedCSCSamplingGraph>& self)
              -> torch::Dict<
                  std::string, torch::Dict<std::string, torch::Tensor>> {
            return self->GetState();
          },
          // __setstate__
          [](torch::Dict<std::string, torch::Dict<std::string, torch::Tensor>>
                 state) -> c10::intrusive_ptr<FusedCSCSamplingGraph> {
            auto g = c10::make_intrusive<FusedCSCSamplingGraph>();
            g->SetState(state);
            return g;
          });
#ifdef GRAPHBOLT_USE_CUDA
  m.class_<cuda::GpuCache>("GpuCache")
      .def("query", &cuda::GpuCache::Query)
      .def("replace", &cuda::GpuCache::Replace);
  m.def("gpu_cache", &cuda::GpuCache::Create);
#endif
  m.def("fused_csc_sampling_graph", &FusedCSCSamplingGraph::Create);
  m.def(
      "load_from_shared_memory", &FusedCSCSamplingGraph::LoadFromSharedMemory);
  m.def("unique_and_compact", &UniqueAndCompact);
  m.def("isin", &IsIn);
  m.def("index_select", &ops::IndexSelect);
  m.def("index_select_csc", &ops::IndexSelectCSC);
  m.def(
      "expand_indptr(Tensor indptr, ScalarType dtype, Tensor? node_ids, int? "
      "output_size) -> Tensor",
      {at::Tag::pt2_compliant_tag});
  m.def("set_seed", &RandomEngine::SetManualSeed);
#ifdef GRAPHBOLT_USE_CUDA
  m.def("set_max_uva_threads", &cuda::set_max_uva_threads);
#endif
}

}  // namespace sampling
}  // namespace graphbolt
