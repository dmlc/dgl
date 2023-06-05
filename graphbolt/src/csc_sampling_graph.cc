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
    const torch::Tensor& nodes, const std::vector<int64_t>& fanouts,
    bool replace, bool return_eids,
    torch::optional<torch::Tensor> probs_or_mask) const {
  const int64_t num_nodes = nodes.size(0);
  // Note probs will be passed as input for 'torch.multinomial' in deeper stack,
  // which doesn't support 'torch.half' and 'torch.bool' data types. To avoid
  // crashes, convert 'probs_or_mask' to 'float32' data type.
  if (probs_or_mask.has_value() &&
      (probs_or_mask.value().dtype() == torch::kBool ||
       probs_or_mask.value().dtype() == torch::kFloat16)) {
    probs_or_mask = probs_or_mask.value().to(torch::kFloat32);
  }
  // If true, perform sampling for each edge type of each node, otherwise just
  // sample once for each node with no regard of edge types.
  bool consider_etype = (fanouts.size() > 1);
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

      if (consider_etype) {
        picked_neighbors_per_node[i] = PickByEtype(
            offset, num_neighbors, fanouts, replace, indptr_.options(),
            type_per_edge_.value(), probs_or_mask);
      } else {
        picked_neighbors_per_node[i] = Pick(
            offset, num_neighbors, fanouts[0], replace, indptr_.options(),
            probs_or_mask);
      }
      num_picked_neighbors_per_node[i + 1] =
          picked_neighbors_per_node[i].size(0);
    }
  });  // End of the thread.

  torch::Tensor subgraph_indptr =
      torch::cumsum(num_picked_neighbors_per_node, 0);

  torch::Tensor picked_eids = torch::cat(picked_neighbors_per_node);
  torch::Tensor subgraph_indices =
      torch::index_select(indices_, 0, picked_eids);
  torch::optional<torch::Tensor> subgraph_type_per_edge = torch::nullopt;
  if (type_per_edge_.has_value())
    subgraph_type_per_edge =
        torch::index_select(type_per_edge_.value(), 0, picked_eids);
  torch::optional<torch::Tensor> subgraph_reverse_edge_ids = torch::nullopt;
  if (return_eids) subgraph_reverse_edge_ids = std::move(picked_eids);
  return c10::make_intrusive<SampledSubgraph>(
      subgraph_indptr, subgraph_indices, nodes, torch::nullopt,
      subgraph_reverse_edge_ids, subgraph_type_per_edge);
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

/**
 * @brief Perform uniform sampling of elements and return the sampled indices.
 *
 * @param offset The starting edge ID for the connected neighbors of the sampled
 * node.
 * @param num_neighbors The number of neighbors to pick.
 * @param fanout The number of edges to be sampled for each node. It should be
 * >= 0 or -1.
 *  - When the value is -1, all neighbors will be chosen for sampling. It is
 * equivalent to selecting all neighbors with non-zero probability when the
 * fanout is >= the number of neighbors (and replacement is set to false).
 *  - When the value is a non-negative integer, it serves as a minimum
 * threshold for selecting neighbors.
 * @param replace Boolean indicating whether the sample is preformed with or
 * without replacement. If True, a value can be selected multiple times.
 * Otherwise, each value can be selected only once.
 * @param options Tensor options specifying the desired data type of the result.
 *
 * @return A tensor containing the picked neighbors.
 */
inline torch::Tensor UniformPick(
    int64_t offset, int64_t num_neighbors, int64_t fanout, bool replace,
    const torch::TensorOptions& options) {
  torch::Tensor picked_neighbors;
  if ((fanout == -1) || (num_neighbors <= fanout && !replace)) {
    picked_neighbors = torch::arange(offset, offset + num_neighbors, options);
  } else {
    if (replace) {
      picked_neighbors =
          torch::randint(offset, offset + num_neighbors, {fanout}, options);
    } else {
      picked_neighbors = torch::randperm(num_neighbors, options);
      picked_neighbors = picked_neighbors.slice(0, 0, fanout) + offset;
    }
  }
  return picked_neighbors;
}

/**
 * @brief Perform non-uniform sampling of elements based on probabilities and
 * return the sampled indices.
 *
 * If 'probs_or_mask' is provided, it indicates that the sampling is
 * non-uniform. In such cases:
 * - When the number of neighbors with non-zero probability is less than or
 * equal to fanout, all neighbors with non-zero probability will be selected.
 * - When the number of neighbors with non-zero probability exceeds fanout, the
 * sampling process will select 'fanout' elements based on their respective
 * probabilities. Higher probabilities will increase the chances of being chosen
 * during the sampling process.
 *
 * @param offset The starting edge ID for the connected neighbors of the sampled
 * node.
 * @param num_neighbors The number of neighbors to pick.
 * @param fanout The number of edges to be sampled for each node. It should be
 * >= 0 or -1.
 *  - When the value is -1, all neighbors will be chosen for sampling. It is
 * equivalent to selecting all neighbors with non-zero probability when the
 * fanout is >= the number of neighbors (and replacement is set to false).
 *  - When the value is a non-negative integer, it serves as a minimum
 * threshold for selecting neighbors.
 * @param replace Boolean indicating whether the sample is preformed with or
 * without replacement. If True, a value can be selected multiple times.
 * Otherwise, each value can be selected only once.
 * @param options Tensor options specifying the desired data type of the result.
 * @param probs_or_mask Optional tensor containing the (unnormalized)
 * probabilities associated with each neighboring edge of a node in the original
 * graph. It must be a 1D floating-point tensor with the number of elements
 * equal to the number of edges in the graph.
 *
 * @return A tensor containing the picked neighbors.
 */
inline torch::Tensor NonUniformPick(
    int64_t offset, int64_t num_neighbors, int64_t fanout, bool replace,
    const torch::TensorOptions& options,
    const torch::optional<torch::Tensor>& probs_or_mask) {
  torch::Tensor picked_neighbors;
  auto local_probs =
      probs_or_mask.value().slice(0, offset, offset + num_neighbors);
  auto positive_probs_indices = local_probs.nonzero().squeeze(1);
  auto num_positive_probs = positive_probs_indices.size(0);
  if (num_positive_probs == 0) return torch::tensor({}, options);
  if ((fanout == -1) || (num_positive_probs <= fanout && !replace)) {
    picked_neighbors = torch::arange(offset, offset + num_neighbors, options);
    picked_neighbors =
        torch::index_select(picked_neighbors, 0, positive_probs_indices);
  } else {
    if (!replace) fanout = std::min(fanout, num_positive_probs);
    picked_neighbors =
        torch::multinomial(local_probs, fanout, replace) + offset;
  }
  return picked_neighbors;
}

torch::Tensor Pick(
    int64_t offset, int64_t num_neighbors, int64_t fanout, bool replace,
    const torch::TensorOptions& options,
    const torch::optional<torch::Tensor>& probs_or_mask) {
  if (probs_or_mask.has_value()) {
    return NonUniformPick(
        offset, num_neighbors, fanout, replace, options, probs_or_mask);
  } else {
    return UniformPick(offset, num_neighbors, fanout, replace, options);
  }
}

torch::Tensor PickByEtype(
    int64_t offset, int64_t num_neighbors, const std::vector<int64_t>& fanouts,
    bool replace, const torch::TensorOptions& options,
    const torch::Tensor& type_per_edge,
    const torch::optional<torch::Tensor>& probs_or_mask) {
  std::vector<torch::Tensor> picked_neighbors(
      fanouts.size(), torch::tensor({}, options));
  int64_t etype_begin = offset;
  int64_t etype_end = offset;
  while (etype_end < offset + num_neighbors) {
    int64_t etype = type_per_edge[etype_end].item<int64_t>();
    int64_t fanout = fanouts[etype];
    while (etype_end < offset + num_neighbors &&
           type_per_edge[etype_end].item<int64_t>() == etype) {
      etype_end++;
    }
    // Do sampling for one etype.
    if (fanout != 0) {
      picked_neighbors[etype] = Pick(
          etype_begin, etype_end - etype_begin, fanout, replace, options,
          probs_or_mask);
    }
    etype_begin = etype_end;
  }

  return torch::cat(picked_neighbors, 0);
}

}  // namespace sampling
}  // namespace graphbolt
