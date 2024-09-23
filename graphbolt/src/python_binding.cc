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
#include "./cuda/cooperative_minibatching_utils.h"
#include "./cuda/max_uva_threads.h"
#endif
#include "./cnumpy.h"
#include "./feature_cache.h"
#include "./index_select.h"
#include "./io_uring.h"
#include "./partitioned_cache_policy.h"
#include "./random.h"
#include "./utils.h"

#ifdef GRAPHBOLT_USE_CUDA
#include "./cuda/extension/gpu_cache.h"
#include "./cuda/extension/gpu_graph_cache.h"
#endif

namespace graphbolt {
namespace sampling {

TORCH_LIBRARY(graphbolt, m) {
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
      .def_readwrite("type_per_edge", &FusedSampledSubgraph::type_per_edge)
      .def_readwrite("etype_offsets", &FusedSampledSubgraph::etype_offsets);
  m.class_<Future<void>>("VoidFuture").def("wait", &Future<void>::Wait);
  m.class_<Future<torch::Tensor>>("TensorFuture")
      .def("wait", &Future<torch::Tensor>::Wait);
  m.class_<Future<std::vector<torch::Tensor>>>("TensorListFuture")
      .def("wait", &Future<std::vector<torch::Tensor>>::Wait);
  m.class_<Future<c10::intrusive_ptr<FusedSampledSubgraph>>>(
       "FusedSampledSubgraphFuture")
      .def("wait", &Future<c10::intrusive_ptr<FusedSampledSubgraph>>::Wait);
  m.class_<Future<std::vector<
      std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>>>>(
       "UniqueAndCompactBatchedFuture")
      .def(
          "wait",
          &Future<std::vector<std::tuple<
              torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>>>::
              Wait);
  m.class_<Future<
      std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>>>>(
       "RankSortFuture")
      .def(
          "wait",
          &Future<std::vector<
              std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>>>::Wait);
  m.class_<Future<std::tuple<torch::Tensor, torch::Tensor, int64_t, int64_t>>>(
       "GpuGraphCacheQueryFuture")
      .def(
          "wait",
          &Future<std::tuple<torch::Tensor, torch::Tensor, int64_t, int64_t>>::
              Wait);
  m.class_<Future<std::tuple<torch::Tensor, std::vector<torch::Tensor>>>>(
       "GpuGraphCacheReplaceFuture")
      .def(
          "wait",
          &Future<std::tuple<torch::Tensor, std::vector<torch::Tensor>>>::Wait);
  m.class_<storage::OnDiskNpyArray>("OnDiskNpyArray")
      .def("index_select", &storage::OnDiskNpyArray::IndexSelect);
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
      .def("node_attribute", &FusedCSCSamplingGraph::NodeAttribute)
      .def("edge_attribute", &FusedCSCSamplingGraph::EdgeAttribute)
      .def("set_csc_indptr", &FusedCSCSamplingGraph::SetCSCIndptr)
      .def("set_indices", &FusedCSCSamplingGraph::SetIndices)
      .def("set_node_type_offset", &FusedCSCSamplingGraph::SetNodeTypeOffset)
      .def("set_type_per_edge", &FusedCSCSamplingGraph::SetTypePerEdge)
      .def("set_node_type_to_id", &FusedCSCSamplingGraph::SetNodeTypeToID)
      .def("set_edge_type_to_id", &FusedCSCSamplingGraph::SetEdgeTypeToID)
      .def("set_node_attributes", &FusedCSCSamplingGraph::SetNodeAttributes)
      .def("set_edge_attributes", &FusedCSCSamplingGraph::SetEdgeAttributes)
      .def("add_node_attribute", &FusedCSCSamplingGraph::AddNodeAttribute)
      .def("add_edge_attribute", &FusedCSCSamplingGraph::AddEdgeAttribute)
      .def("in_subgraph", &FusedCSCSamplingGraph::InSubgraph)
      .def("sample_neighbors", &FusedCSCSamplingGraph::SampleNeighbors)
      .def(
          "sample_neighbors_async",
          &FusedCSCSamplingGraph::SampleNeighborsAsync)
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
      .def("query_async", &cuda::GpuCache::QueryAsync)
      .def("replace", &cuda::GpuCache::Replace);
  m.def("gpu_cache", &cuda::GpuCache::Create);
  m.class_<cuda::GpuGraphCache>("GpuGraphCache")
      .def("query", &cuda::GpuGraphCache::Query)
      .def("query_async", &cuda::GpuGraphCache::QueryAsync)
      .def("replace", &cuda::GpuGraphCache::Replace)
      .def("replace_async", &cuda::GpuGraphCache::ReplaceAsync);
  m.def("gpu_graph_cache", &cuda::GpuGraphCache::Create);
#endif
  m.def("fused_csc_sampling_graph", &FusedCSCSamplingGraph::Create);
  m.class_<storage::PartitionedCachePolicy>("PartitionedCachePolicy")
      .def("query", &storage::PartitionedCachePolicy::Query)
      .def("query_async", &storage::PartitionedCachePolicy::QueryAsync)
      .def(
          "query_and_replace",
          &storage::PartitionedCachePolicy::QueryAndReplace)
      .def(
          "query_and_replace_async",
          &storage::PartitionedCachePolicy::QueryAndReplaceAsync)
      .def("replace", &storage::PartitionedCachePolicy::Replace)
      .def("replace_async", &storage::PartitionedCachePolicy::ReplaceAsync)
      .def(
          "reading_completed",
          &storage::PartitionedCachePolicy::ReadingCompleted)
      .def(
          "reading_completed_async",
          &storage::PartitionedCachePolicy::ReadingCompletedAsync)
      .def(
          "writing_completed",
          &storage::PartitionedCachePolicy::WritingCompleted)
      .def(
          "writing_completed_async",
          &storage::PartitionedCachePolicy::WritingCompletedAsync);
  m.def(
      "s3_fifo_cache_policy",
      &storage::PartitionedCachePolicy::Create<storage::S3FifoCachePolicy>);
  m.def(
      "sieve_cache_policy",
      &storage::PartitionedCachePolicy::Create<storage::SieveCachePolicy>);
  m.def(
      "lru_cache_policy",
      &storage::PartitionedCachePolicy::Create<storage::LruCachePolicy>);
  m.def(
      "clock_cache_policy",
      &storage::PartitionedCachePolicy::Create<storage::ClockCachePolicy>);
  m.class_<storage::FeatureCache>("FeatureCache")
      .def("is_pinned", &storage::FeatureCache::IsPinned)
      .def_property("nbytes", &storage::FeatureCache::NumBytes)
      .def("index_select", &storage::FeatureCache::IndexSelect)
      .def("query", &storage::FeatureCache::Query)
      .def("query_async", &storage::FeatureCache::QueryAsync)
      .def("replace", &storage::FeatureCache::Replace)
      .def("replace_async", &storage::FeatureCache::ReplaceAsync);
  m.def("feature_cache", &storage::FeatureCache::Create);
  m.def(
      "load_from_shared_memory", &FusedCSCSamplingGraph::LoadFromSharedMemory);
  m.def("unique_and_compact", &UniqueAndCompact);
  m.def("unique_and_compact_batched", &UniqueAndCompactBatched);
  m.def("unique_and_compact_batched_async", &UniqueAndCompactBatchedAsync);
  m.def("isin", &IsIn);
  m.def("is_not_in_index", &IsNotInIndex);
  m.def("is_not_in_index_async", &IsNotInIndexAsync);
  m.def("index_select", &ops::IndexSelect);
  m.def("index_select_async", &ops::IndexSelectAsync);
  m.def("scatter_async", &ops::ScatterAsync);
  m.def("index_select_csc", &ops::IndexSelectCSC);
  m.def("index_select_csc_batched", &ops::IndexSelectCSCBatched);
  m.def("index_select_csc_batched_async", &ops::IndexSelectCSCBatchedAsync);
  m.def("ondisk_npy_array", &storage::OnDiskNpyArray::Create);
  m.def("detect_io_uring", &io_uring::IsAvailable);
  m.def("set_num_io_uring_threads", &io_uring::SetNumThreads);
  m.def("set_worker_id", &utils::SetWorkerId);
  m.def("set_seed", &RandomEngine::SetManualSeed);
#ifdef GRAPHBOLT_USE_CUDA
  m.def("set_max_uva_threads", &cuda::set_max_uva_threads);
  m.def("rank_sort", &cuda::RankSort);
  m.def("rank_sort_async", &cuda::RankSortAsync);
#endif
#ifdef HAS_IMPL_ABSTRACT_PYSTUB
  m.impl_abstract_pystub("dgl.graphbolt.base", "//dgl.graphbolt.base");
#endif
  m.def(
      "expand_indptr(Tensor indptr, ScalarType dtype, Tensor? node_ids, "
      "SymInt? output_size) -> Tensor"
#ifdef HAS_PT2_COMPLIANT_TAG
      ,
      {at::Tag::pt2_compliant_tag}
#endif
  );
  m.def(
      "indptr_edge_ids(Tensor indptr, ScalarType dtype, Tensor? offset, "
      "SymInt? output_size) -> "
      "Tensor"
#ifdef HAS_PT2_COMPLIANT_TAG
      ,
      {at::Tag::pt2_compliant_tag}
#endif
  );
}

}  // namespace sampling
}  // namespace graphbolt
