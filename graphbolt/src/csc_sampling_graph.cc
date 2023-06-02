/**
 *  Copyright (c) 2023 by Contributors
 * @file csc_sampling_graph.cc
 * @brief Source file of sampling graph.
 */

#include <graphbolt/csc_sampling_graph.h>
#include <graphbolt/serialize.h>
#include <torch/torch.h>

#include <tuple>
#include <vector>

#include "./shared_memory_utils.h"

namespace graphbolt {
namespace sampling {

CSCSamplingGraph::CSCSamplingGraph(
    const torch::Tensor& indptr, const torch::Tensor& indices,
    const torch::optional<torch::Tensor>& node_type_offset,
    const torch::optional<torch::Tensor>& type_per_edge)
    : indptr_(indptr),
      indices_(indices),
      node_type_offset_(node_type_offset),
      type_per_edge_(type_per_edge) {
  TORCH_CHECK(indptr.dim() == 1);
  TORCH_CHECK(indices.dim() == 1);
  TORCH_CHECK(indptr.device() == indices.device());
}

c10::intrusive_ptr<CSCSamplingGraph> CSCSamplingGraph::FromCSC(
    const torch::Tensor& indptr, const torch::Tensor& indices,
    const torch::optional<torch::Tensor>& node_type_offset,
    const torch::optional<torch::Tensor>& type_per_edge) {
  if (node_type_offset.has_value()) {
    auto& offset = node_type_offset.value();
    TORCH_CHECK(offset.dim() == 1);
  }
  if (type_per_edge.has_value()) {
    TORCH_CHECK(type_per_edge.value().dim() == 1);
    TORCH_CHECK(type_per_edge.value().size(0) == indices.size(0));
  }

  return c10::make_intrusive<CSCSamplingGraph>(
      indptr, indices, node_type_offset, type_per_edge);
}

void CSCSamplingGraph::Load(torch::serialize::InputArchive& archive) {
  const int64_t magic_num =
      read_from_archive(archive, "CSCSamplingGraph/magic_num").toInt();
  TORCH_CHECK(
      magic_num == kCSCSamplingGraphSerializeMagic,
      "Magic numbers mismatch when loading CSCSamplingGraph.");
  indptr_ = read_from_archive(archive, "CSCSamplingGraph/indptr").toTensor();
  indices_ = read_from_archive(archive, "CSCSamplingGraph/indices").toTensor();
  if (read_from_archive(archive, "CSCSamplingGraph/has_node_type_offset")
          .toBool()) {
    node_type_offset_ =
        read_from_archive(archive, "CSCSamplingGraph/node_type_offset")
            .toTensor();
  }
  if (read_from_archive(archive, "CSCSamplingGraph/has_type_per_edge")
          .toBool()) {
    type_per_edge_ =
        read_from_archive(archive, "CSCSamplingGraph/type_per_edge").toTensor();
  }
}

void CSCSamplingGraph::Save(torch::serialize::OutputArchive& archive) const {
  archive.write("CSCSamplingGraph/magic_num", kCSCSamplingGraphSerializeMagic);
  archive.write("CSCSamplingGraph/indptr", indptr_);
  archive.write("CSCSamplingGraph/indices", indices_);
  archive.write(
      "CSCSamplingGraph/has_node_type_offset", node_type_offset_.has_value());
  if (node_type_offset_) {
    archive.write(
        "CSCSamplingGraph/node_type_offset", node_type_offset_.value());
  }
  archive.write(
      "CSCSamplingGraph/has_type_per_edge", type_per_edge_.has_value());
  if (type_per_edge_) {
    archive.write("CSCSamplingGraph/type_per_edge", type_per_edge_.value());
  }
}

c10::intrusive_ptr<SampledSubgraph> CSCSamplingGraph::InSubgraph(
    const torch::Tensor& nodes) const {
  using namespace torch::indexing;
  const int32_t kDefaultGrainSize = 100;
  torch::Tensor indptr = torch::zeros_like(indptr_);
  const size_t num_seeds = nodes.size(0);
  std::vector<torch::Tensor> indices_arr(num_seeds);
  std::vector<torch::Tensor> edge_ids_arr(num_seeds);
  std::vector<torch::Tensor> type_per_edge_arr(num_seeds);
  torch::parallel_for(
      0, num_seeds, kDefaultGrainSize, [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
          const int64_t node_id = nodes[i].item<int64_t>();
          const int64_t start_idx = indptr_[node_id].item<int64_t>();
          const int64_t end_idx = indptr_[node_id + 1].item<int64_t>();
          indptr[node_id + 1] = end_idx - start_idx;
          indices_arr[i] = indices_.slice(0, start_idx, end_idx);
          edge_ids_arr[i] = torch::arange(start_idx, end_idx);
          if (type_per_edge_) {
            type_per_edge_arr[i] =
                type_per_edge_.value().slice(0, start_idx, end_idx);
          }
        }
      });

  const auto& nonzero_idx = torch::nonzero(indptr).reshape(-1);
  torch::Tensor compact_indptr =
      torch::zeros({nonzero_idx.size(0) + 1}, indptr_.dtype());
  compact_indptr.index_put_({Slice(1, None)}, indptr.index({nonzero_idx}));
  return c10::make_intrusive<SampledSubgraph>(
      compact_indptr.cumsum(0), torch::cat(indices_arr), nonzero_idx - 1,
      torch::arange(0, NumNodes()), torch::cat(edge_ids_arr),
      type_per_edge_
          ? torch::optional<torch::Tensor>{torch::cat(type_per_edge_arr)}
          : torch::nullopt);
}

c10::intrusive_ptr<SampledSubgraph> CSCSamplingGraph::SampleNeighbors(
    const torch::Tensor& nodes) const {
  const int64_t num_nodes = nodes.size(0);

  std::vector<torch::Tensor> picked_neighbors_per_node(num_nodes);
  torch::Tensor num_picked_neighbors_per_node =
      torch::zeros({num_nodes + 1}, indptr_.options());

  torch::parallel_for(0, num_nodes, 32, [&](size_t b, size_t e) {
    for (size_t i = b; i < e; ++i) {
      const auto nid = nodes[i].item<int64_t>();
      TORCH_CHECK(
          nid >= 0 && nid < NumNodes(),
          "The seed nodes' IDs should fall within the range of the graph's "
          "node IDs.");
      const auto offset = indptr_[nid].item<int64_t>();
      const auto num_neighbors = indptr_[nid + 1].item<int64_t>() - offset;

      if (num_neighbors == 0) {
        // Initialization is performed here because all tensors will be
        // concatenated in the master thread, and having an undefined tensor
        // during concatenation can result in a crash.
        picked_neighbors_per_node[i] = torch::tensor({}, indptr_.options());
        continue;
      }

      picked_neighbors_per_node[i] =
          torch::arange(offset, offset + num_neighbors);
      num_picked_neighbors_per_node[i + 1] = num_neighbors;
    }
  });  // End of the thread.

  torch::Tensor subgraph_indptr =
      torch::cumsum(num_picked_neighbors_per_node, 0);

  torch::Tensor picked_eids = torch::cat(picked_neighbors_per_node);
  torch::Tensor subgraph_indices =
      torch::index_select(indices_, 0, picked_eids);

  return c10::make_intrusive<SampledSubgraph>(
      subgraph_indptr, subgraph_indices, nodes, torch::nullopt, torch::nullopt,
      torch::nullopt);
}

c10::intrusive_ptr<CSCSamplingGraph>
CSCSamplingGraph::BuildGraphFromSharedMemoryTensors(
    std::tuple<
        SharedMemoryPtr, SharedMemoryPtr,
        std::vector<torch::optional<torch::Tensor>>>&& shared_memory_tensors) {
  auto& optional_tensors = std::get<2>(shared_memory_tensors);
  auto graph = c10::make_intrusive<CSCSamplingGraph>(
      optional_tensors[0].value(), optional_tensors[1].value(),
      optional_tensors[2], optional_tensors[3]);
  graph->tensor_meta_shm_ = std::move(std::get<0>(shared_memory_tensors));
  graph->tensor_data_shm_ = std::move(std::get<1>(shared_memory_tensors));
  return graph;
}

c10::intrusive_ptr<CSCSamplingGraph> CSCSamplingGraph::CopyToSharedMemory(
    const std::string& shared_memory_name) {
  auto optional_tensors = std::vector<torch::optional<torch::Tensor>>{
      indptr_, indices_, node_type_offset_, type_per_edge_};
  auto shared_memory_tensors = CopyTensorsToSharedMemory(
      shared_memory_name, optional_tensors, SERIALIZED_METAINFO_SIZE_MAX);
  return BuildGraphFromSharedMemoryTensors(std::move(shared_memory_tensors));
}

c10::intrusive_ptr<CSCSamplingGraph> CSCSamplingGraph::LoadFromSharedMemory(
    const std::string& shared_memory_name) {
  auto shared_memory_tensors = LoadTensorsFromSharedMemory(
      shared_memory_name, SERIALIZED_METAINFO_SIZE_MAX);
  return BuildGraphFromSharedMemoryTensors(std::move(shared_memory_tensors));
}

}  // namespace sampling
}  // namespace graphbolt
