/**
 *  Copyright (c) 2023 by Contributors
 * @file fused_csc_sampling_graph.cc
 * @brief Source file of sampling graph.
 */

#include <graphbolt/cuda_sampling_ops.h>
#include <graphbolt/fused_csc_sampling_graph.h>
#include <graphbolt/serialize.h>
#include <torch/torch.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <tuple>
#include <type_traits>
#include <vector>

#include "./expand_indptr.h"
#include "./index_select.h"
#include "./macro.h"
#include "./random.h"
#include "./shared_memory_helper.h"
#include "./utils.h"

namespace {
torch::optional<torch::Dict<std::string, torch::Tensor>> TensorizeDict(
    const torch::optional<torch::Dict<std::string, int64_t>>& dict) {
  if (!dict.has_value()) {
    return torch::nullopt;
  }
  torch::Dict<std::string, torch::Tensor> result;
  for (const auto& pair : dict.value()) {
    result.insert(pair.key(), torch::tensor(pair.value(), torch::kInt64));
  }
  return result;
}

torch::optional<torch::Dict<std::string, int64_t>> DetensorizeDict(
    const torch::optional<torch::Dict<std::string, torch::Tensor>>& dict) {
  if (!dict.has_value()) {
    return torch::nullopt;
  }
  torch::Dict<std::string, int64_t> result;
  for (const auto& pair : dict.value()) {
    result.insert(pair.key(), pair.value().item<int64_t>());
  }
  return result;
}
}  // namespace

namespace graphbolt {
namespace sampling {

static const int kPickleVersion = 6199;

FusedCSCSamplingGraph::FusedCSCSamplingGraph(
    const torch::Tensor& indptr, const torch::Tensor& indices,
    const torch::optional<torch::Tensor>& node_type_offset,
    const torch::optional<torch::Tensor>& type_per_edge,
    const torch::optional<NodeTypeToIDMap>& node_type_to_id,
    const torch::optional<EdgeTypeToIDMap>& edge_type_to_id,
    const torch::optional<NodeAttrMap>& node_attributes,
    const torch::optional<EdgeAttrMap>& edge_attributes)
    : indptr_(indptr),
      indices_(indices),
      node_type_offset_(node_type_offset),
      type_per_edge_(type_per_edge),
      node_type_to_id_(node_type_to_id),
      edge_type_to_id_(edge_type_to_id),
      node_attributes_(node_attributes),
      edge_attributes_(edge_attributes) {
  TORCH_CHECK(indptr.dim() == 1);
  TORCH_CHECK(indices.dim() == 1);
  TORCH_CHECK(indptr.device() == indices.device());
}

c10::intrusive_ptr<FusedCSCSamplingGraph> FusedCSCSamplingGraph::Create(
    const torch::Tensor& indptr, const torch::Tensor& indices,
    const torch::optional<torch::Tensor>& node_type_offset,
    const torch::optional<torch::Tensor>& type_per_edge,
    const torch::optional<NodeTypeToIDMap>& node_type_to_id,
    const torch::optional<EdgeTypeToIDMap>& edge_type_to_id,
    const torch::optional<NodeAttrMap>& node_attributes,
    const torch::optional<EdgeAttrMap>& edge_attributes) {
  if (node_type_offset.has_value()) {
    auto& offset = node_type_offset.value();
    TORCH_CHECK(offset.dim() == 1);
    TORCH_CHECK(node_type_to_id.has_value());
    TORCH_CHECK(
        offset.size(0) ==
        static_cast<int64_t>(node_type_to_id.value().size() + 1));
  }
  if (type_per_edge.has_value()) {
    TORCH_CHECK(type_per_edge.value().dim() == 1);
    TORCH_CHECK(type_per_edge.value().size(0) == indices.size(0));
    TORCH_CHECK(edge_type_to_id.has_value());
  }
  if (node_attributes.has_value()) {
    for (const auto& pair : node_attributes.value()) {
      TORCH_CHECK(
          pair.value().size(0) == indptr.size(0) - 1,
          "Expected node_attribute.size(0) and num_nodes to be equal, "
          "but node_attribute.size(0) was ",
          pair.value().size(0), ", and num_nodes was ", indptr.size(0) - 1,
          ".");
    }
  }
  if (edge_attributes.has_value()) {
    for (const auto& pair : edge_attributes.value()) {
      TORCH_CHECK(
          pair.value().size(0) == indices.size(0),
          "Expected edge_attribute.size(0) and num_edges to be equal, "
          "but edge_attribute.size(0) was ",
          pair.value().size(0), ", and num_edges was ", indices.size(0), ".");
    }
  }
  return c10::make_intrusive<FusedCSCSamplingGraph>(
      indptr, indices, node_type_offset, type_per_edge, node_type_to_id,
      edge_type_to_id, node_attributes, edge_attributes);
}

void FusedCSCSamplingGraph::Load(torch::serialize::InputArchive& archive) {
  const int64_t magic_num =
      read_from_archive<int64_t>(archive, "FusedCSCSamplingGraph/magic_num");
  TORCH_CHECK(
      magic_num == kCSCSamplingGraphSerializeMagic,
      "Magic numbers mismatch when loading FusedCSCSamplingGraph.");
  indptr_ =
      read_from_archive<torch::Tensor>(archive, "FusedCSCSamplingGraph/indptr");
  indices_ = read_from_archive<torch::Tensor>(
      archive, "FusedCSCSamplingGraph/indices");
  if (read_from_archive<bool>(
          archive, "FusedCSCSamplingGraph/has_node_type_offset")) {
    node_type_offset_ = read_from_archive<torch::Tensor>(
        archive, "FusedCSCSamplingGraph/node_type_offset");
  }
  if (read_from_archive<bool>(
          archive, "FusedCSCSamplingGraph/has_type_per_edge")) {
    type_per_edge_ = read_from_archive<torch::Tensor>(
        archive, "FusedCSCSamplingGraph/type_per_edge");
  }

  if (read_from_archive<bool>(
          archive, "FusedCSCSamplingGraph/has_node_type_to_id")) {
    node_type_to_id_ = read_from_archive<NodeTypeToIDMap>(
        archive, "FusedCSCSamplingGraph/node_type_to_id");
  }

  if (read_from_archive<bool>(
          archive, "FusedCSCSamplingGraph/has_edge_type_to_id")) {
    edge_type_to_id_ = read_from_archive<EdgeTypeToIDMap>(
        archive, "FusedCSCSamplingGraph/edge_type_to_id");
  }

  if (read_from_archive<bool>(
          archive, "FusedCSCSamplingGraph/has_node_attributes")) {
    node_attributes_ = read_from_archive<NodeAttrMap>(
        archive, "FusedCSCSamplingGraph/node_attributes");
  }
  if (read_from_archive<bool>(
          archive, "FusedCSCSamplingGraph/has_edge_attributes")) {
    edge_attributes_ = read_from_archive<EdgeAttrMap>(
        archive, "FusedCSCSamplingGraph/edge_attributes");
  }
}

void FusedCSCSamplingGraph::Save(
    torch::serialize::OutputArchive& archive) const {
  archive.write(
      "FusedCSCSamplingGraph/magic_num", kCSCSamplingGraphSerializeMagic);
  archive.write("FusedCSCSamplingGraph/indptr", indptr_);
  archive.write("FusedCSCSamplingGraph/indices", indices_);
  archive.write(
      "FusedCSCSamplingGraph/has_node_type_offset",
      node_type_offset_.has_value());
  if (node_type_offset_) {
    archive.write(
        "FusedCSCSamplingGraph/node_type_offset", node_type_offset_.value());
  }
  archive.write(
      "FusedCSCSamplingGraph/has_type_per_edge", type_per_edge_.has_value());
  if (type_per_edge_) {
    archive.write(
        "FusedCSCSamplingGraph/type_per_edge", type_per_edge_.value());
  }
  archive.write(
      "FusedCSCSamplingGraph/has_node_type_to_id",
      node_type_to_id_.has_value());
  if (node_type_to_id_) {
    archive.write(
        "FusedCSCSamplingGraph/node_type_to_id", node_type_to_id_.value());
  }
  archive.write(
      "FusedCSCSamplingGraph/has_edge_type_to_id",
      edge_type_to_id_.has_value());
  if (edge_type_to_id_) {
    archive.write(
        "FusedCSCSamplingGraph/edge_type_to_id", edge_type_to_id_.value());
  }
  archive.write(
      "FusedCSCSamplingGraph/has_node_attributes",
      node_attributes_.has_value());
  if (node_attributes_) {
    archive.write(
        "FusedCSCSamplingGraph/node_attributes", node_attributes_.value());
  }
  archive.write(
      "FusedCSCSamplingGraph/has_edge_attributes",
      edge_attributes_.has_value());
  if (edge_attributes_) {
    archive.write(
        "FusedCSCSamplingGraph/edge_attributes", edge_attributes_.value());
  }
}

void FusedCSCSamplingGraph::SetState(
    const torch::Dict<std::string, torch::Dict<std::string, torch::Tensor>>&
        state) {
  // State is a dict of dicts. The tensor-type attributes are stored in the dict
  // with key "independent_tensors". The dict-type attributes (edge_attributes)
  // are stored directly with the their name as the key.
  const auto& independent_tensors = state.at("independent_tensors");
  TORCH_CHECK(
      independent_tensors.at("version_number")
          .equal(torch::tensor({kPickleVersion})),
      "Version number mismatches when loading pickled FusedCSCSamplingGraph.")
  indptr_ = independent_tensors.at("indptr");
  indices_ = independent_tensors.at("indices");
  if (independent_tensors.find("node_type_offset") !=
      independent_tensors.end()) {
    node_type_offset_ = independent_tensors.at("node_type_offset");
  }
  if (independent_tensors.find("type_per_edge") != independent_tensors.end()) {
    type_per_edge_ = independent_tensors.at("type_per_edge");
  }
  if (state.find("node_type_to_id") != state.end()) {
    node_type_to_id_ = DetensorizeDict(state.at("node_type_to_id"));
  }
  if (state.find("edge_type_to_id") != state.end()) {
    edge_type_to_id_ = DetensorizeDict(state.at("edge_type_to_id"));
  }
  if (state.find("node_attributes") != state.end()) {
    node_attributes_ = state.at("node_attributes");
  }
  if (state.find("edge_attributes") != state.end()) {
    edge_attributes_ = state.at("edge_attributes");
  }
}

torch::Dict<std::string, torch::Dict<std::string, torch::Tensor>>
FusedCSCSamplingGraph::GetState() const {
  // State is a dict of dicts. The tensor-type attributes are stored in the dict
  // with key "independent_tensors". The dict-type attributes (edge_attributes)
  // are stored directly with the their name as the key.
  torch::Dict<std::string, torch::Dict<std::string, torch::Tensor>> state;
  torch::Dict<std::string, torch::Tensor> independent_tensors;
  // Serialization version number. It indicates the serialization method of the
  // whole state.
  independent_tensors.insert("version_number", torch::tensor({kPickleVersion}));
  independent_tensors.insert("indptr", indptr_);
  independent_tensors.insert("indices", indices_);
  if (node_type_offset_.has_value()) {
    independent_tensors.insert("node_type_offset", node_type_offset_.value());
  }
  if (type_per_edge_.has_value()) {
    independent_tensors.insert("type_per_edge", type_per_edge_.value());
  }
  state.insert("independent_tensors", independent_tensors);
  if (node_type_to_id_.has_value()) {
    state.insert("node_type_to_id", TensorizeDict(node_type_to_id_).value());
  }
  if (edge_type_to_id_.has_value()) {
    state.insert("edge_type_to_id", TensorizeDict(edge_type_to_id_).value());
  }
  if (node_attributes_.has_value()) {
    state.insert("node_attributes", node_attributes_.value());
  }
  if (edge_attributes_.has_value()) {
    state.insert("edge_attributes", edge_attributes_.value());
  }
  return state;
}

c10::intrusive_ptr<FusedSampledSubgraph> FusedCSCSamplingGraph::InSubgraph(
    const torch::Tensor& nodes) const {
  if (utils::is_on_gpu(nodes) && utils::is_accessible_from_gpu(indptr_) &&
      utils::is_accessible_from_gpu(indices_) &&
      (!type_per_edge_.has_value() ||
       utils::is_accessible_from_gpu(type_per_edge_.value()))) {
    GRAPHBOLT_DISPATCH_CUDA_ONLY_DEVICE(c10::DeviceType::CUDA, "InSubgraph", {
      return ops::InSubgraph(indptr_, indices_, nodes, type_per_edge_);
    });
  }
  std::vector<torch::Tensor> tensors{indices_};
  if (type_per_edge_.has_value()) {
    tensors.push_back(*type_per_edge_);
  }

  auto [output_indptr, results] =
      ops::IndexSelectCSCBatched(indptr_, tensors, nodes, true, torch::nullopt);
  torch::optional<torch::Tensor> type_per_edge;
  if (type_per_edge_.has_value()) {
    type_per_edge = results.at(1);
  }

  return c10::make_intrusive<FusedSampledSubgraph>(
      // original_row_node_ids is not computed here and is unused.
      output_indptr, results.at(0), results.back(), nodes, torch::nullopt,
      type_per_edge);
}

/**
 * @brief Get a lambda function which counts the number of the neighbors to be
 * sampled.
 *
 * @param fanouts The number of edges to be sampled for each node with or
 * without considering edge types.
 * @param replace Boolean indicating whether the sample is performed with or
 * without replacement. If True, a value can be selected multiple times.
 * Otherwise, each value can be selected only once.
 * @param type_per_edge A tensor representing the type of each edge, if
 * present.
 * @param probs_or_mask Optional tensor containing the (unnormalized)
 * probabilities associated with each neighboring edge of a node in the original
 * graph. It must be a 1D floating-point tensor with the number of elements
 * equal to the number of edges in the graph.
 *
 * @return A lambda function (int64_t seed_offset, int64_t offset, int64_t
 * num_neighbors) -> torch::Tensor, which takes seed offset (the offset of the
 * seed to sample), offset (the starting edge ID of the given node) and
 * num_neighbors (number of neighbors) as params and returns the pick number of
 * the given node.
 */
auto GetNumPickFn(
    const std::vector<int64_t>& fanouts, bool replace,
    const torch::optional<torch::Tensor>& type_per_edge,
    const torch::optional<torch::Tensor>& probs_or_mask,
    bool with_seed_offsets) {
  // If fanouts.size() > 1, returns the total number of all edge types of the
  // given node.
  return [&fanouts, replace, &probs_or_mask, &type_per_edge, with_seed_offsets](
             int64_t offset, int64_t num_neighbors, auto num_picked_ptr,
             int64_t seed_index,
             const std::vector<int64_t>& etype_id_to_num_picked_offset) {
    if (fanouts.size() > 1) {
      NumPickByEtype(
          with_seed_offsets, fanouts, replace, type_per_edge.value(),
          probs_or_mask, offset, num_neighbors, num_picked_ptr, seed_index,
          etype_id_to_num_picked_offset);
    } else {
      NumPick(
          fanouts[0], replace, probs_or_mask, offset, num_neighbors,
          num_picked_ptr + seed_index);
    }
  };
}

auto GetTemporalNumPickFn(
    torch::Tensor seed_timestamp, torch::Tensor csc_indices,
    const std::vector<int64_t>& fanouts, bool replace,
    const torch::optional<torch::Tensor>& type_per_edge,
    const torch::optional<torch::Tensor>& seed_pre_time_window,
    const torch::optional<torch::Tensor>& probs_or_mask,
    const torch::optional<torch::Tensor>& node_timestamp,
    const torch::optional<torch::Tensor>& edge_timestamp) {
  // If fanouts.size() > 1, returns the total number of all edge types of the
  // given node.
  return [&seed_timestamp, &csc_indices, &fanouts, replace,
          &seed_pre_time_window, &probs_or_mask, &type_per_edge,
          &node_timestamp, &edge_timestamp](
             int64_t seed_offset, int64_t offset, int64_t num_neighbors) {
    if (fanouts.size() > 1) {
      return TemporalNumPickByEtype(
          seed_timestamp, csc_indices, fanouts, replace, type_per_edge.value(),
          seed_pre_time_window, probs_or_mask, node_timestamp, edge_timestamp,
          seed_offset, offset, num_neighbors);
    } else {
      return TemporalNumPick(
          seed_timestamp, csc_indices, fanouts[0], replace,
          seed_pre_time_window, probs_or_mask, node_timestamp, edge_timestamp,
          seed_offset, offset, num_neighbors);
    }
  };
}

/**
 * @brief Get a lambda function which contains the sampling process.
 *
 * @param fanouts The number of edges to be sampled for each node with or
 * without considering edge types.
 * @param replace Boolean indicating whether the sample is performed with or
 * without replacement. If True, a value can be selected multiple times.
 * Otherwise, each value can be selected only once.
 * @param options Tensor options specifying the desired data type of the result.
 * @param type_per_edge A tensor representing the type of each edge, if
 * present.
 * @param probs_or_mask Optional tensor containing the (unnormalized)
 * probabilities associated with each neighboring edge of a node in the original
 * graph. It must be a 1D floating-point tensor with the number of elements
 * equal to the number of edges in the graph.
 * @param args Contains sampling algorithm specific arguments.
 *
 * @return A lambda function: (int64_t seed_offset, int64_t offset, int64_t
 * num_neighbors, PickedType* picked_data_ptr) -> torch::Tensor, which takes
 * seed_offset (the offset of the seed to sample), offset (the starting edge ID
 * of the given node) and num_neighbors (number of neighbors) as params and puts
 * the picked neighbors at the address specified by picked_data_ptr.
 */
template <SamplerType S>
auto GetPickFn(
    const std::vector<int64_t>& fanouts, bool replace,
    const torch::TensorOptions& options,
    const torch::optional<torch::Tensor>& type_per_edge,
    const torch::optional<torch::Tensor>& probs_or_mask, bool with_seed_offsets,
    SamplerArgs<S> args) {
  return [&fanouts, replace, &options, &type_per_edge, &probs_or_mask, args,
          with_seed_offsets](
             int64_t offset, int64_t num_neighbors, auto picked_data_ptr,
             int64_t seed_offset, auto subgraph_indptr_ptr,
             const std::vector<int64_t>& etype_id_to_num_picked_offset) {
    // If fanouts.size() > 1, perform sampling for each edge type of each
    // node; otherwise just sample once for each node with no regard of edge
    // types.
    if (fanouts.size() > 1) {
      return PickByEtype(
          with_seed_offsets, offset, num_neighbors, fanouts, replace, options,
          type_per_edge.value(), probs_or_mask, args, picked_data_ptr,
          seed_offset, subgraph_indptr_ptr, etype_id_to_num_picked_offset);
    } else {
      picked_data_ptr += subgraph_indptr_ptr[seed_offset];
      int64_t num_sampled = Pick(
          offset, num_neighbors, fanouts[0], replace, options, probs_or_mask,
          args, picked_data_ptr);
      if (type_per_edge) {
        std::sort(picked_data_ptr, picked_data_ptr + num_sampled);
      }
      return num_sampled;
    }
  };
}

template <SamplerType S>
auto GetTemporalPickFn(
    torch::Tensor seed_timestamp, torch::Tensor csc_indices,
    const std::vector<int64_t>& fanouts, bool replace,
    const torch::TensorOptions& options,
    const torch::optional<torch::Tensor>& type_per_edge,
    const torch::optional<torch::Tensor>& seed_pre_time_window,
    const torch::optional<torch::Tensor>& probs_or_mask,
    const torch::optional<torch::Tensor>& node_timestamp,
    const torch::optional<torch::Tensor>& edge_timestamp, SamplerArgs<S> args) {
  return [&seed_timestamp, &csc_indices, &fanouts, replace, &options,
          &type_per_edge, &seed_pre_time_window, &probs_or_mask,
          &node_timestamp, &edge_timestamp, args](
             int64_t seed_offset, int64_t offset, int64_t num_neighbors,
             auto picked_data_ptr) {
    // If fanouts.size() > 1, perform sampling for each edge type of each
    // node; otherwise just sample once for each node with no regard of edge
    // types.
    if (fanouts.size() > 1) {
      return TemporalPickByEtype(
          seed_timestamp, csc_indices, seed_offset, offset, num_neighbors,
          fanouts, replace, options, type_per_edge.value(),
          seed_pre_time_window, probs_or_mask, node_timestamp, edge_timestamp,
          args, picked_data_ptr);
    } else {
      int64_t num_sampled = TemporalPick(
          seed_timestamp, csc_indices, seed_offset, offset, num_neighbors,
          fanouts[0], replace, options, seed_pre_time_window, probs_or_mask,
          node_timestamp, edge_timestamp, args, picked_data_ptr);
      if (type_per_edge.has_value()) {
        std::sort(picked_data_ptr, picked_data_ptr + num_sampled);
      }
      return num_sampled;
    }
  };
}

template <TemporalOption Temporal, typename NumPickFn, typename PickFn>
c10::intrusive_ptr<FusedSampledSubgraph>
FusedCSCSamplingGraph::SampleNeighborsImpl(
    const torch::Tensor& seeds,
    const torch::optional<std::vector<int64_t>>& seed_offsets,
    const std::vector<int64_t>& fanouts, NumPickFn num_pick_fn,
    PickFn pick_fn) const {
  const int64_t num_seeds = seeds.size(0);
  const auto indptr_options = indptr_.options();

  // Calculate GrainSize for parallel_for.
  // Set the default grain size to 64.
  const int64_t grain_size = 64;
  torch::Tensor picked_eids;
  torch::Tensor subgraph_indptr;
  torch::Tensor subgraph_indices;
  torch::optional<torch::Tensor> subgraph_type_per_edge = torch::nullopt;
  torch::optional<torch::Tensor> edge_offsets = torch::nullopt;

  bool with_seed_offsets = seed_offsets.has_value();
  bool hetero_with_seed_offsets = with_seed_offsets && fanouts.size() > 1 &&
                                  Temporal == TemporalOption::NOT_TEMPORAL;

  // Get the number of edge types. If it's homo or if the size of fanouts is 1
  // (hetero graph but sampled as a homo graph), set num_etypes as 1.
  // In temporal sampling, this will not be used for now since the logic hasn't
  // been adopted for temporal sampling.
  const int64_t num_etypes =
      (edge_type_to_id_.has_value() && hetero_with_seed_offsets)
          ? edge_type_to_id_->size()
          : 1;
  std::vector<int64_t> etype_id_to_src_ntype_id(num_etypes);
  std::vector<int64_t> etype_id_to_dst_ntype_id(num_etypes);
  torch::optional<torch::Tensor> subgraph_indptr_substract = torch::nullopt;
  // The pick numbers are stored in a single tensor by the order of etype. Each
  // etype corresponds to a group of seeds whose ntype are the same as the
  // dst_type. `etype_id_to_num_picked_offset` indicates the beginning offset
  // where each etype's corresponding seeds' pick numbers are stored in the pick
  // number tensor.
  std::vector<int64_t> etype_id_to_num_picked_offset(num_etypes + 1);
  if (hetero_with_seed_offsets) {
    for (auto& etype_and_id : edge_type_to_id_.value()) {
      auto etype = etype_and_id.key();
      auto id = etype_and_id.value();
      auto [src_type, dst_type] = utils::parse_src_dst_ntype_from_etype(etype);
      auto dst_ntype_id = node_type_to_id_->at(dst_type);
      etype_id_to_src_ntype_id[id] = node_type_to_id_->at(src_type);
      etype_id_to_dst_ntype_id[id] = dst_ntype_id;
      etype_id_to_num_picked_offset[id + 1] =
          seed_offsets->at(dst_ntype_id + 1) - seed_offsets->at(dst_ntype_id) +
          1;
    }
    std::partial_sum(
        etype_id_to_num_picked_offset.begin(),
        etype_id_to_num_picked_offset.end(),
        etype_id_to_num_picked_offset.begin());
  } else {
    etype_id_to_dst_ntype_id[0] = 0;
    etype_id_to_num_picked_offset[1] = num_seeds + 1;
  }
  // `num_rows` indicates the length of `num_picked_neighbors_per_node`, which
  // is used for storing pick numbers. In non-temporal hetero sampling, it
  // equals to sum_{etype} #seeds with ntype=dst_type(etype). In homo sampling,
  // it equals to `num_seeds`.
  const int64_t num_rows = etype_id_to_num_picked_offset[num_etypes];
  torch::Tensor num_picked_neighbors_per_node =
      // Need to use zeros because all nodes don't have all etypes.
      torch::zeros({num_rows}, indptr_options);

  AT_DISPATCH_INDEX_TYPES(
      indptr_.scalar_type(), "SampleNeighborsImplWrappedWithIndptr", ([&] {
        using indptr_t = index_t;
        AT_DISPATCH_INDEX_TYPES(
            seeds.scalar_type(), "SampleNeighborsImplWrappedWithSeeds", ([&] {
              using seeds_t = index_t;
              const auto indptr_data = indptr_.data_ptr<indptr_t>();
              const auto num_picked_neighbors_data_ptr =
                  num_picked_neighbors_per_node.data_ptr<indptr_t>();
              num_picked_neighbors_data_ptr[0] = 0;
              const auto seeds_data_ptr = seeds.data_ptr<seeds_t>();

              // Step 1. Calculate pick number of each node.
              torch::parallel_for(
                  0, num_seeds, grain_size, [&](int64_t begin, int64_t end) {
                    for (int64_t i = begin; i < end; ++i) {
                      const auto nid = seeds_data_ptr[i];
                      TORCH_CHECK(
                          nid >= 0 && nid < NumNodes(),
                          "The seed nodes' IDs should fall within the range of "
                          "the graph's node IDs.");
                      const auto offset = indptr_data[nid];
                      const auto num_neighbors = indptr_data[nid + 1] - offset;

                      if constexpr (Temporal == TemporalOption::TEMPORAL) {
                        num_picked_neighbors_data_ptr[i + 1] =
                            num_neighbors == 0
                                ? 0
                                : num_pick_fn(i, offset, num_neighbors);
                      } else {
                        const auto seed_type_id =
                            (hetero_with_seed_offsets)
                                ? std::upper_bound(
                                      seed_offsets->begin(),
                                      seed_offsets->end(), i) -
                                      seed_offsets->begin() - 1
                                : 0;
                        // `seed_index` indicates the index of the current
                        // seed within the group of seeds which have the same
                        // node type.
                        const auto seed_index =
                            (hetero_with_seed_offsets)
                                ? i - seed_offsets->at(seed_type_id)
                                : i;
                        num_pick_fn(
                            offset, num_neighbors,
                            num_picked_neighbors_data_ptr + 1, seed_index,
                            etype_id_to_num_picked_offset);
                      }
                    }
                  });

              // Step 2. Calculate prefix sum to get total length and offsets of
              // each node. It's also the indptr of the generated subgraph.
              subgraph_indptr = num_picked_neighbors_per_node.cumsum(
                  0, indptr_.scalar_type());
              auto subgraph_indptr_data_ptr =
                  subgraph_indptr.data_ptr<indptr_t>();

              if (hetero_with_seed_offsets) {
                torch::Tensor num_picked_offset_tensor =
                    torch::empty({num_etypes + 1}, indptr_options);
                const auto num_picked_offset_data_ptr =
                    num_picked_offset_tensor.data_ptr<indptr_t>();
                std::copy(
                    etype_id_to_num_picked_offset.begin(),
                    etype_id_to_num_picked_offset.end(),
                    num_picked_offset_data_ptr);
                torch::Tensor substract_offset =
                    torch::empty({num_etypes}, indptr_options);
                const auto substract_offset_data_ptr =
                    substract_offset.data_ptr<indptr_t>();
                for (auto i = 0; i < num_etypes; ++i) {
                  // Collect the total pick number subtract offsets.
                  substract_offset_data_ptr[i] = subgraph_indptr_data_ptr
                      [etype_id_to_num_picked_offset[i]];
                }
                subgraph_indptr_substract = ops::ExpandIndptr(
                    num_picked_offset_tensor, indptr_.scalar_type(),
                    substract_offset);
              }

              // When doing non-temporal hetero sampling, we generate an
              // edge_offsets tensor.
              if (hetero_with_seed_offsets) {
                edge_offsets = torch::empty({num_etypes + 1}, indptr_options);
                auto edge_offsets_data_ptr =
                    edge_offsets.value().data_ptr<indptr_t>();
                edge_offsets_data_ptr[0] = 0;
                for (auto i = 0; i < num_etypes; ++i) {
                  edge_offsets_data_ptr[i + 1] = subgraph_indptr_data_ptr
                      [etype_id_to_num_picked_offset[i + 1] - 1];
                }
              }

              // Step 3. Allocate the tensor for picked neighbors.
              const auto total_length =
                  subgraph_indptr.data_ptr<indptr_t>()[num_rows - 1];
              picked_eids = torch::empty({total_length}, indptr_options);
              subgraph_indices =
                  torch::empty({total_length}, indices_.options());
              if (!hetero_with_seed_offsets && type_per_edge_.has_value()) {
                subgraph_type_per_edge = torch::empty(
                    {total_length}, type_per_edge_.value().options());
              }

              auto picked_eids_data_ptr = picked_eids.data_ptr<indptr_t>();
              torch::parallel_for(
                  0, num_seeds, grain_size, [&](int64_t begin, int64_t end) {
                    for (int64_t i = begin; i < end; ++i) {
                      const auto nid = seeds_data_ptr[i];
                      const auto offset = indptr_data[nid];
                      const auto num_neighbors = indptr_data[nid + 1] - offset;
                      auto picked_number = 0;
                      const auto seed_type_id =
                          (hetero_with_seed_offsets)
                              ? std::upper_bound(
                                    seed_offsets->begin(), seed_offsets->end(),
                                    i) -
                                    seed_offsets->begin() - 1
                              : 0;
                      const auto seed_index =
                          (hetero_with_seed_offsets)
                              ? i - seed_offsets->at(seed_type_id)
                              : i;

                      // Step 4. Pick neighbors for each node.
                      if constexpr (Temporal == TemporalOption::TEMPORAL) {
                        picked_number = num_picked_neighbors_data_ptr[i + 1];
                        auto picked_offset = subgraph_indptr_data_ptr[i];
                        if (picked_number > 0) {
                          auto actual_picked_count = pick_fn(
                              i, offset, num_neighbors,
                              picked_eids_data_ptr + picked_offset);
                          TORCH_CHECK(
                              actual_picked_count == picked_number,
                              "Actual picked count doesn't match the calculated"
                              " pick number.");
                        }
                      } else {
                        picked_number = pick_fn(
                            offset, num_neighbors, picked_eids_data_ptr,
                            seed_index, subgraph_indptr_data_ptr,
                            etype_id_to_num_picked_offset);
                        if (!hetero_with_seed_offsets) {
                          TORCH_CHECK(
                              num_picked_neighbors_data_ptr[i + 1] ==
                                  picked_number,
                              "Actual picked count doesn't match the calculated"
                              " pick number.");
                        }
                      }

                      // Step 5. Calculate other attributes and return the
                      // subgraph.
                      if (picked_number > 0) {
                        // indices dtype and seeds dtype is required to be same.
                        using index_t = seeds_t;
                        auto subgraph_indices_data_ptr =
                            subgraph_indices.data_ptr<index_t>();
                        auto indices_data_ptr = indices_.data_ptr<index_t>();
                        for (auto i = 0; i < num_etypes; ++i) {
                          if (etype_id_to_dst_ntype_id[i] != seed_type_id)
                            continue;
                          const auto indptr_offset =
                              with_seed_offsets
                                  ? etype_id_to_num_picked_offset[i] +
                                        seed_index
                                  : seed_index;
                          const auto picked_begin =
                              subgraph_indptr_data_ptr[indptr_offset];
                          const auto picked_end =
                              subgraph_indptr_data_ptr[indptr_offset + 1];
                          for (auto j = picked_begin; j < picked_end; ++j) {
                            subgraph_indices_data_ptr[j] =
                                indices_data_ptr[picked_eids_data_ptr[j]];
                            if (hetero_with_seed_offsets &&
                                node_type_offset_.has_value()) {
                              // Substract the node type offset from
                              // subgraph indices. Assuming
                              // node_type_offset has the same dtype as
                              // indices.
                              auto node_type_offset_data =
                                  node_type_offset_.value().data_ptr<index_t>();
                              subgraph_indices_data_ptr[j] -=
                                  node_type_offset_data
                                      [etype_id_to_src_ntype_id[i]];
                            }
                          }
                        }

                        if (!hetero_with_seed_offsets &&
                            type_per_edge_.has_value()) {
                          // When hetero graph is sampled as a homo graph, we
                          // still generate type_per_edge tensor for this
                          // situation.
                          AT_DISPATCH_INTEGRAL_TYPES(
                              subgraph_type_per_edge.value().scalar_type(),
                              "IndexSelectTypePerEdge", ([&] {
                                auto subgraph_type_per_edge_data_ptr =
                                    subgraph_type_per_edge.value()
                                        .data_ptr<scalar_t>();
                                auto type_per_edge_data_ptr =
                                    type_per_edge_.value().data_ptr<scalar_t>();
                                const auto picked_offset =
                                    subgraph_indptr_data_ptr[seed_index];
                                for (auto j = picked_offset;
                                     j < picked_offset + picked_number; ++j)
                                  subgraph_type_per_edge_data_ptr[j] =
                                      type_per_edge_data_ptr
                                          [picked_eids_data_ptr[j]];
                              }));
                        }
                      }
                    }
                  });
            }));
      }));

  if (subgraph_indptr_substract.has_value()) {
    subgraph_indptr -= subgraph_indptr_substract.value();
  }

  return c10::make_intrusive<FusedSampledSubgraph>(
      subgraph_indptr, subgraph_indices, picked_eids, seeds, torch::nullopt,
      subgraph_type_per_edge, edge_offsets);
}

c10::intrusive_ptr<FusedSampledSubgraph> FusedCSCSamplingGraph::SampleNeighbors(
    torch::optional<torch::Tensor> seeds,
    torch::optional<std::vector<int64_t>> seed_offsets,
    const std::vector<int64_t>& fanouts, bool replace, bool layer,
    bool returning_indices_is_optional,
    torch::optional<torch::Tensor> probs_or_mask,
    torch::optional<torch::Tensor> random_seed,
    double seed2_contribution) const {
  // If seeds does not have a value, then we expect all arguments to be resident
  // on the GPU. If seeds has a value, then we expect them to be accessible from
  // GPU. This is required for the dispatch to work when CUDA is not available.
  if (((!seeds.has_value() && utils::is_on_gpu(indptr_) &&
        utils::is_on_gpu(indices_) &&
        (!probs_or_mask.has_value() ||
         utils::is_on_gpu(probs_or_mask.value())) &&
        (!type_per_edge_.has_value() ||
         utils::is_on_gpu(type_per_edge_.value()))) ||
       (seeds.has_value() && utils::is_on_gpu(seeds.value()) &&
        utils::is_accessible_from_gpu(indptr_) &&
        utils::is_accessible_from_gpu(indices_) &&
        (!probs_or_mask.has_value() ||
         utils::is_accessible_from_gpu(probs_or_mask.value())) &&
        (!type_per_edge_.has_value() ||
         utils::is_accessible_from_gpu(type_per_edge_.value())))) &&
      !replace) {
    GRAPHBOLT_DISPATCH_CUDA_ONLY_DEVICE(
        c10::DeviceType::CUDA, "SampleNeighbors", {
          return ops::SampleNeighbors(
              indptr_, indices_, seeds, seed_offsets, fanouts, replace, layer,
              returning_indices_is_optional, type_per_edge_, probs_or_mask,
              node_type_offset_, node_type_to_id_, edge_type_to_id_,
              random_seed, seed2_contribution);
        });
  }
  TORCH_CHECK(seeds.has_value(), "Nodes can not be None on the CPU.");

  if (probs_or_mask.has_value()) {
    // Note probs will be passed as input for 'torch.multinomial' in deeper
    // stack, which doesn't support 'torch.half' and 'torch.bool' data types. To
    // avoid crashes, convert 'probs_or_mask' to 'float32' data type.
    if (probs_or_mask.value().dtype() == torch::kBool ||
        probs_or_mask.value().dtype() == torch::kFloat16) {
      probs_or_mask = probs_or_mask.value().to(torch::kFloat32);
    }
  }

  bool with_seed_offsets = seed_offsets.has_value();

  if (layer) {
    if (random_seed.has_value() && random_seed->numel() >= 2) {
      SamplerArgs<SamplerType::LABOR_DEPENDENT> args{
          indices_,
          {random_seed.value(), static_cast<float>(seed2_contribution)},
          NumNodes()};
      return SampleNeighborsImpl<TemporalOption::NOT_TEMPORAL>(
          seeds.value(), seed_offsets, fanouts,
          GetNumPickFn(
              fanouts, replace, type_per_edge_, probs_or_mask,
              with_seed_offsets),
          GetPickFn(
              fanouts, replace, indptr_.options(), type_per_edge_,
              probs_or_mask, with_seed_offsets, args));
    } else {
      auto args = [&] {
        if (random_seed.has_value() && random_seed->numel() == 1) {
          return SamplerArgs<SamplerType::LABOR>{
              indices_, random_seed.value(), NumNodes()};
        } else {
          return SamplerArgs<SamplerType::LABOR>{
              indices_,
              RandomEngine::ThreadLocal()->RandInt(
                  static_cast<int64_t>(0), std::numeric_limits<int64_t>::max()),
              NumNodes()};
        }
      }();
      return SampleNeighborsImpl<TemporalOption::NOT_TEMPORAL>(
          seeds.value(), seed_offsets, fanouts,
          GetNumPickFn(
              fanouts, replace, type_per_edge_, probs_or_mask,
              with_seed_offsets),
          GetPickFn(
              fanouts, replace, indptr_.options(), type_per_edge_,
              probs_or_mask, with_seed_offsets, args));
    }
  } else {
    SamplerArgs<SamplerType::NEIGHBOR> args;
    return SampleNeighborsImpl<TemporalOption::NOT_TEMPORAL>(
        seeds.value(), seed_offsets, fanouts,
        GetNumPickFn(
            fanouts, replace, type_per_edge_, probs_or_mask, with_seed_offsets),
        GetPickFn(
            fanouts, replace, indptr_.options(), type_per_edge_, probs_or_mask,
            with_seed_offsets, args));
  }
}

c10::intrusive_ptr<Future<c10::intrusive_ptr<FusedSampledSubgraph>>>
FusedCSCSamplingGraph::SampleNeighborsAsync(
    torch::optional<torch::Tensor> seeds,
    torch::optional<std::vector<int64_t>> seed_offsets,
    const std::vector<int64_t>& fanouts, bool replace, bool layer,
    bool returning_indices_is_optional,
    torch::optional<torch::Tensor> probs_or_mask,
    torch::optional<torch::Tensor> random_seed,
    double seed2_contribution) const {
  return async(
      [=] {
        return this->SampleNeighbors(
            seeds, seed_offsets, fanouts, replace, layer,
            returning_indices_is_optional, probs_or_mask, random_seed,
            seed2_contribution);
      },
      (seeds.has_value() && utils::is_on_gpu(*seeds)) ||
          utils::is_on_gpu(indptr_));
}

c10::intrusive_ptr<FusedSampledSubgraph>
FusedCSCSamplingGraph::TemporalSampleNeighbors(
    const torch::optional<torch::Tensor>& seeds,
    const torch::optional<std::vector<int64_t>>& seed_offsets,
    const torch::Tensor& seeds_timestamp, const std::vector<int64_t>& fanouts,
    bool replace, bool layer, bool returning_indices_is_optional,
    torch::optional<torch::Tensor> seeds_pre_time_window,
    torch::optional<torch::Tensor> probs_or_mask,
    torch::optional<std::string> node_timestamp_attr_name,
    torch::optional<std::string> edge_timestamp_attr_name,
    torch::optional<torch::Tensor> random_seed,
    double seed2_contribution) const {
  // 1. Get the timestamp attribute for nodes of the graph
  const auto node_timestamp = this->NodeAttribute(node_timestamp_attr_name);
  // 2. Get the timestamp attribute for edges of the graph
  const auto edge_timestamp = this->EdgeAttribute(edge_timestamp_attr_name);
  // If seeds does not have a value, then we expect all arguments to be resident
  // on the GPU. If seeds has a value, then we expect them to be accessible from
  // GPU. This is required for the dispatch to work when CUDA is not available.
  if (((!seeds.has_value() && utils::is_on_gpu(indptr_) &&
        utils::is_on_gpu(indices_) &&
        (!probs_or_mask.has_value() ||
         utils::is_on_gpu(probs_or_mask.value())) &&
        (!type_per_edge_.has_value() ||
         utils::is_on_gpu(type_per_edge_.value()))) ||
       (seeds.has_value() && utils::is_on_gpu(seeds.value()) &&
        utils::is_accessible_from_gpu(indptr_) &&
        utils::is_accessible_from_gpu(indices_) &&
        (!probs_or_mask.has_value() ||
         utils::is_accessible_from_gpu(probs_or_mask.value())) &&
        (!type_per_edge_.has_value() ||
         utils::is_accessible_from_gpu(type_per_edge_.value())))) &&
      utils::is_accessible_from_gpu(seeds_timestamp) &&
      (!seeds_pre_time_window.has_value() ||
       utils::is_accessible_from_gpu(*seeds_pre_time_window)) &&
      (!node_timestamp.has_value() ||
       utils::is_accessible_from_gpu(*node_timestamp)) &&
      (!edge_timestamp.has_value() ||
       utils::is_accessible_from_gpu(*edge_timestamp)) &&
      !replace) {
    GRAPHBOLT_DISPATCH_CUDA_ONLY_DEVICE(
        c10::DeviceType::CUDA, "SampleNeighbors", {
          return ops::SampleNeighbors(
              indptr_, indices_, seeds, seed_offsets, fanouts, replace, layer,
              returning_indices_is_optional, type_per_edge_, probs_or_mask,
              node_type_offset_, node_type_to_id_, edge_type_to_id_,
              random_seed, seed2_contribution, seeds_timestamp,
              seeds_pre_time_window, node_timestamp, edge_timestamp);
        });
  }
  TORCH_CHECK(seeds.has_value(), "Nodes can not be None for CPU.");
  // 3. Get probs_or_mask.
  if (probs_or_mask.has_value()) {
    // Note probs will be passed as input for 'torch.multinomial' in deeper
    // stack, which doesn't support 'torch.half' and 'torch.bool' data types. To
    // avoid crashes, convert 'probs_or_mask' to 'float32' data type.
    if (probs_or_mask.value().dtype() == torch::kBool ||
        probs_or_mask.value().dtype() == torch::kFloat16) {
      probs_or_mask = probs_or_mask.value().to(torch::kFloat32);
    }
  }
  // 4. Call SampleNeighborsImpl
  if (layer) {
    if (random_seed.has_value() && random_seed->numel() >= 2) {
      SamplerArgs<SamplerType::LABOR_DEPENDENT> args{
          indices_,
          {random_seed.value(), static_cast<float>(seed2_contribution)},
          NumNodes()};
      return SampleNeighborsImpl<TemporalOption::TEMPORAL>(
          *seeds, seed_offsets, fanouts,
          GetTemporalNumPickFn(
              seeds_timestamp, indices_, fanouts, replace, type_per_edge_,
              seeds_pre_time_window, probs_or_mask, node_timestamp,
              edge_timestamp),
          GetTemporalPickFn(
              seeds_timestamp, indices_, fanouts, replace, indptr_.options(),
              type_per_edge_, seeds_pre_time_window, probs_or_mask,
              node_timestamp, edge_timestamp, args));
    } else {
      auto args = [&] {
        if (random_seed.has_value() && random_seed->numel() == 1) {
          return SamplerArgs<SamplerType::LABOR>{
              indices_, random_seed.value(), NumNodes()};
        } else {
          return SamplerArgs<SamplerType::LABOR>{
              indices_,
              RandomEngine::ThreadLocal()->RandInt(
                  static_cast<int64_t>(0), std::numeric_limits<int64_t>::max()),
              NumNodes()};
        }
      }();
      return SampleNeighborsImpl<TemporalOption::TEMPORAL>(
          *seeds, seed_offsets, fanouts,
          GetTemporalNumPickFn(
              seeds_timestamp, indices_, fanouts, replace, type_per_edge_,
              seeds_pre_time_window, probs_or_mask, node_timestamp,
              edge_timestamp),
          GetTemporalPickFn(
              seeds_timestamp, indices_, fanouts, replace, indptr_.options(),
              type_per_edge_, seeds_pre_time_window, probs_or_mask,
              node_timestamp, edge_timestamp, args));
    }
  } else {
    SamplerArgs<SamplerType::NEIGHBOR> args;
    return SampleNeighborsImpl<TemporalOption::TEMPORAL>(
        *seeds, seed_offsets, fanouts,
        GetTemporalNumPickFn(
            seeds_timestamp, this->indices_, fanouts, replace, type_per_edge_,
            seeds_pre_time_window, probs_or_mask, node_timestamp,
            edge_timestamp),
        GetTemporalPickFn(
            seeds_timestamp, this->indices_, fanouts, replace,
            indptr_.options(), type_per_edge_, seeds_pre_time_window,
            probs_or_mask, node_timestamp, edge_timestamp, args));
  }
}

static c10::intrusive_ptr<FusedCSCSamplingGraph>
BuildGraphFromSharedMemoryHelper(SharedMemoryHelper&& helper) {
  helper.InitializeRead();
  auto indptr = helper.ReadTorchTensor();
  auto indices = helper.ReadTorchTensor();
  auto node_type_offset = helper.ReadTorchTensor();
  auto type_per_edge = helper.ReadTorchTensor();
  auto node_type_to_id = DetensorizeDict(helper.ReadTorchTensorDict());
  auto edge_type_to_id = DetensorizeDict(helper.ReadTorchTensorDict());
  auto node_attributes = helper.ReadTorchTensorDict();
  auto edge_attributes = helper.ReadTorchTensorDict();
  auto graph = c10::make_intrusive<FusedCSCSamplingGraph>(
      indptr.value(), indices.value(), node_type_offset, type_per_edge,
      node_type_to_id, edge_type_to_id, node_attributes, edge_attributes);
  auto shared_memory = helper.ReleaseSharedMemory();
  graph->HoldSharedMemoryObject(
      std::move(shared_memory.first), std::move(shared_memory.second));
  return graph;
}

c10::intrusive_ptr<FusedCSCSamplingGraph>
FusedCSCSamplingGraph::CopyToSharedMemory(
    const std::string& shared_memory_name) {
  SharedMemoryHelper helper(shared_memory_name);
  helper.WriteTorchTensor(indptr_);
  helper.WriteTorchTensor(indices_);
  helper.WriteTorchTensor(node_type_offset_);
  helper.WriteTorchTensor(type_per_edge_);
  helper.WriteTorchTensorDict(TensorizeDict(node_type_to_id_));
  helper.WriteTorchTensorDict(TensorizeDict(edge_type_to_id_));
  helper.WriteTorchTensorDict(node_attributes_);
  helper.WriteTorchTensorDict(edge_attributes_);
  helper.Flush();
  return BuildGraphFromSharedMemoryHelper(std::move(helper));
}

c10::intrusive_ptr<FusedCSCSamplingGraph>
FusedCSCSamplingGraph::LoadFromSharedMemory(
    const std::string& shared_memory_name) {
  SharedMemoryHelper helper(shared_memory_name);
  return BuildGraphFromSharedMemoryHelper(std::move(helper));
}

void FusedCSCSamplingGraph::HoldSharedMemoryObject(
    SharedMemoryPtr tensor_metadata_shm, SharedMemoryPtr tensor_data_shm) {
  tensor_metadata_shm_ = std::move(tensor_metadata_shm);
  tensor_data_shm_ = std::move(tensor_data_shm);
}

template <typename PickedNumType>
void NumPick(
    int64_t fanout, bool replace,
    const torch::optional<torch::Tensor>& probs_or_mask, int64_t offset,
    int64_t num_neighbors, PickedNumType* picked_num_ptr) {
  int64_t num_valid_neighbors = num_neighbors;
  if (probs_or_mask.has_value() && num_neighbors > 0) {
    // Subtract the count of zeros in probs_or_mask.
    AT_DISPATCH_ALL_TYPES(
        probs_or_mask.value().scalar_type(), "CountZero", ([&] {
          scalar_t* probs_data_ptr = probs_or_mask.value().data_ptr<scalar_t>();
          num_valid_neighbors -= std::count(
              probs_data_ptr + offset, probs_data_ptr + offset + num_neighbors,
              0);
        }));
  }
  if (num_valid_neighbors == 0 || fanout == -1) {
    *picked_num_ptr = num_valid_neighbors;
  } else {
    *picked_num_ptr = replace ? fanout : std::min(fanout, num_valid_neighbors);
  }
}

torch::Tensor TemporalMask(
    int64_t seed_timestamp, torch::Tensor csc_indices,
    const torch::optional<int64_t>& seed_pre_time_window,
    const torch::optional<torch::Tensor>& probs_or_mask,
    const torch::optional<torch::Tensor>& node_timestamp,
    const torch::optional<torch::Tensor>& edge_timestamp,
    std::pair<int64_t, int64_t> edge_range) {
  auto [l, r] = edge_range;
  torch::Tensor mask = torch::ones({r - l}, torch::kBool);
  if (node_timestamp.has_value()) {
    auto neighbor_timestamp =
        node_timestamp.value().index_select(0, csc_indices.slice(0, l, r));
    mask &= neighbor_timestamp < seed_timestamp;
    if (seed_pre_time_window.has_value())
      mask &=
          neighbor_timestamp > seed_timestamp - seed_pre_time_window.value();
  }
  if (edge_timestamp.has_value()) {
    auto edge_ts = edge_timestamp.value().slice(0, l, r);
    mask &= edge_ts < seed_timestamp;
    if (seed_pre_time_window.has_value())
      mask &= edge_ts > seed_timestamp - seed_pre_time_window.value();
  }
  if (probs_or_mask.has_value()) {
    mask &= probs_or_mask.value().slice(0, l, r) != 0;
  }
  return mask;
}

/**
 * @brief Fast path for temporal sampling without probability. It is used when
 * the number of neighbors is large. It randomly samples neighbors and checks
 * the timestamp of the neighbors. It is successful if the number of sampled
 * neighbors in kTriedThreshold trials is equal to the fanout.
 */
std::pair<bool, std::vector<int64_t>> FastTemporalPick(
    torch::Tensor seed_timestamp, torch::Tensor csc_indices, int64_t fanout,
    bool replace, const torch::optional<torch::Tensor>& seed_pre_time_window,
    const torch::optional<torch::Tensor>& node_timestamp,
    const torch::optional<torch::Tensor>& edge_timestamp, int64_t seed_offset,
    int64_t offset, int64_t num_neighbors) {
  constexpr int64_t kTriedThreshold = 1000;
  auto timestamp = utils::GetValueByIndex<int64_t>(seed_timestamp, seed_offset);
  torch::optional<int64_t> time_window = torch::nullopt;
  if (seed_pre_time_window.has_value()) {
    time_window = utils::GetValueByIndex<int64_t>(
        seed_pre_time_window.value(), seed_offset);
  }
  std::vector<int64_t> sampled_edges;
  sampled_edges.reserve(fanout);
  std::set<int64_t> sampled_edge_set;
  int64_t sample_count = 0;
  int64_t tried = 0;
  while (sample_count < fanout && tried < kTriedThreshold) {
    int64_t edge_id =
        RandomEngine::ThreadLocal()->RandInt(offset, offset + num_neighbors);
    ++tried;
    if (!replace && sampled_edge_set.count(edge_id) > 0) {
      continue;
    }
    if (node_timestamp.has_value()) {
      bool flag = true;
      AT_DISPATCH_INDEX_TYPES(
          csc_indices.scalar_type(), "CheckNodeTimeStamp", ([&] {
            int64_t neighbor_id =
                utils::GetValueByIndex<index_t>(csc_indices, edge_id);
            auto neighbor_ts = utils::GetValueByIndex<int64_t>(
                node_timestamp.value(), neighbor_id);
            if (neighbor_ts >= timestamp ||
                (time_window.has_value() &&
                 neighbor_ts <= (timestamp - time_window.value())))
              flag = false;
          }));
      if (!flag) continue;
    }
    if (edge_timestamp.has_value()) {
      auto edge_ts =
          utils::GetValueByIndex<int64_t>(edge_timestamp.value(), edge_id);
      if (edge_ts >= timestamp ||
          (time_window.has_value() &&
           edge_ts <= (timestamp - time_window.value())))
        continue;
      continue;
    }
    if (!replace) {
      sampled_edge_set.insert(edge_id);
    }
    sampled_edges.push_back(edge_id);
    sample_count++;
  }
  if (sample_count < fanout) {
    return {false, {}};
  }
  return {true, sampled_edges};
}

int64_t TemporalNumPick(
    torch::Tensor seed_timestamp, torch::Tensor csc_indics, int64_t fanout,
    bool replace, const torch::optional<torch::Tensor>& seed_pre_time_window,
    const torch::optional<torch::Tensor>& probs_or_mask,
    const torch::optional<torch::Tensor>& node_timestamp,
    const torch::optional<torch::Tensor>& edge_timestamp, int64_t seed_offset,
    int64_t offset, int64_t num_neighbors) {
  constexpr int64_t kFastPathThreshold = 1000;
  if (num_neighbors > kFastPathThreshold && !probs_or_mask.has_value()) {
    // TODO: Currently we use the fast path both in TemporalNumPick and
    // TemporalPick. We may only sample once in TemporalNumPick and use the
    // sampled edges in TemporalPick to avoid sampling twice.
    auto [success, sampled_edges] = FastTemporalPick(
        seed_timestamp, csc_indics, fanout, replace, seed_pre_time_window,
        node_timestamp, edge_timestamp, seed_offset, offset, num_neighbors);
    if (success) return sampled_edges.size();
  }
  torch::optional<int64_t> time_window = torch::nullopt;
  if (seed_pre_time_window.has_value()) {
    time_window = utils::GetValueByIndex<int64_t>(
        seed_pre_time_window.value(), seed_offset);
  }
  auto mask = TemporalMask(
      utils::GetValueByIndex<int64_t>(seed_timestamp, seed_offset), csc_indics,
      time_window, probs_or_mask, node_timestamp, edge_timestamp,
      {offset, offset + num_neighbors});
  int64_t num_valid_neighbors = utils::GetValueByIndex<int64_t>(mask.sum(), 0);
  if (num_valid_neighbors == 0 || fanout == -1) return num_valid_neighbors;
  return replace ? fanout : std::min(fanout, num_valid_neighbors);
}

template <typename PickedNumType>
void NumPickByEtype(
    bool with_seed_offsets, const std::vector<int64_t>& fanouts, bool replace,
    const torch::Tensor& type_per_edge,
    const torch::optional<torch::Tensor>& probs_or_mask, int64_t offset,
    int64_t num_neighbors, PickedNumType* num_picked_ptr, int64_t seed_index,
    const std::vector<int64_t>& etype_id_to_num_picked_offset) {
  int64_t etype_begin = offset;
  const int64_t end = offset + num_neighbors;
  PickedNumType total_count = 0;
  AT_DISPATCH_INTEGRAL_TYPES(
      type_per_edge.scalar_type(), "NumPickFnByEtype", ([&] {
        const scalar_t* type_per_edge_data = type_per_edge.data_ptr<scalar_t>();
        while (etype_begin < end) {
          scalar_t etype = type_per_edge_data[etype_begin];
          TORCH_CHECK(
              etype >= 0 && etype < (int64_t)fanouts.size(),
              "Etype values exceed the number of fanouts.");
          auto etype_end_it = std::upper_bound(
              type_per_edge_data + etype_begin, type_per_edge_data + end,
              etype);
          int64_t etype_end = etype_end_it - type_per_edge_data;
          // Do sampling for one etype.
          if (with_seed_offsets) {
            // The pick numbers aren't stored continuously, but separately for
            // each different etype.
            const auto offset =
                etype_id_to_num_picked_offset[etype] + seed_index;
            NumPick(
                fanouts[etype], replace, probs_or_mask, etype_begin,
                etype_end - etype_begin, num_picked_ptr + offset);
          } else {
            PickedNumType picked_count = 0;
            NumPick(
                fanouts[etype], replace, probs_or_mask, etype_begin,
                etype_end - etype_begin, &picked_count);
            total_count += picked_count;
          }
          etype_begin = etype_end;
        }
      }));
  if (!with_seed_offsets) {
    num_picked_ptr[seed_index] = total_count;
  }
}

int64_t TemporalNumPickByEtype(
    torch::Tensor seed_timestamp, torch::Tensor csc_indices,
    const std::vector<int64_t>& fanouts, bool replace,
    const torch::Tensor& type_per_edge,
    const torch::optional<torch::Tensor>& seed_pre_time_window,
    const torch::optional<torch::Tensor>& probs_or_mask,
    const torch::optional<torch::Tensor>& node_timestamp,
    const torch::optional<torch::Tensor>& edge_timestamp, int64_t seed_offset,
    int64_t offset, int64_t num_neighbors) {
  int64_t etype_begin = offset;
  const int64_t end = offset + num_neighbors;
  int64_t total_count = 0;
  AT_DISPATCH_INTEGRAL_TYPES(
      type_per_edge.scalar_type(), "TemporalNumPickFnByEtype", ([&] {
        const scalar_t* type_per_edge_data = type_per_edge.data_ptr<scalar_t>();
        while (etype_begin < end) {
          scalar_t etype = type_per_edge_data[etype_begin];
          TORCH_CHECK(
              etype >= 0 && etype < (int64_t)fanouts.size(),
              "Etype values exceed the number of fanouts.");
          auto etype_end_it = std::upper_bound(
              type_per_edge_data + etype_begin, type_per_edge_data + end,
              etype);
          int64_t etype_end = etype_end_it - type_per_edge_data;
          // Do sampling for one etype.
          total_count += TemporalNumPick(
              seed_timestamp, csc_indices, fanouts[etype], replace,
              seed_pre_time_window, probs_or_mask, node_timestamp,
              edge_timestamp, seed_offset, etype_begin,
              etype_end - etype_begin);
          etype_begin = etype_end;
        }
      }));
  return total_count;
}

/**
 * @brief Perform uniform sampling of elements and return the sampled indices.
 *
 * @param offset The starting edge ID for the connected neighbors of the sampled
 * node.
 * @param num_neighbors The number of neighbors to pick.
 * @param fanout The number of edges to be sampled for each node. It should be
 * >= 0 or -1.
 *  - When the value is -1, all neighbors will be sampled once regardless of
 * replacement. It is equivalent to selecting all neighbors when the fanout is
 * >= the number of neighbors (and replacement is set to false).
 *  - When the value is a non-negative integer, it serves as a minimum
 * threshold for selecting neighbors.
 * @param replace Boolean indicating whether the sample is performed with or
 * without replacement. If True, a value can be selected multiple times.
 * Otherwise, each value can be selected only once.
 * @param options Tensor options specifying the desired data type of the result.
 * @param picked_data_ptr The destination address where the picked neighbors
 * should be put. Enough memory space should be allocated in advance.
 */
template <typename PickedType>
inline int64_t UniformPick(
    int64_t offset, int64_t num_neighbors, int64_t fanout, bool replace,
    const torch::TensorOptions& options, PickedType* picked_data_ptr) {
  if ((fanout == -1) || (num_neighbors <= fanout && !replace)) {
    std::iota(picked_data_ptr, picked_data_ptr + num_neighbors, offset);
    return num_neighbors;
  } else if (replace) {
    std::memcpy(
        picked_data_ptr,
        torch::randint(offset, offset + num_neighbors, {fanout}, options)
            .data_ptr<PickedType>(),
        fanout * sizeof(PickedType));
    return fanout;
  } else {
    // We use different sampling strategies for different sampling case.
    if (fanout >= num_neighbors / 10) {
      // [Algorithm]
      // This algorithm is conceptually related to the Fisher-Yates
      // shuffle.
      //
      // [Complexity Analysis]
      // This algorithm's memory complexity is O(num_neighbors), but
      // it generates fewer random numbers (O(fanout)).
      //
      // (Compare) Reservoir algorithm is one of the most classical
      // sampling algorithms. Both the reservoir algorithm and our
      // algorithm offer distinct advantages, we need to compare to
      // illustrate our trade-offs.
      // The reservoir algorithm is memory-efficient (O(fanout)) but
      // creates many random numbers (O(num_neighbors)), which is
      // costly.
      //
      // [Practical Consideration]
      // Use this algorithm when `fanout >= num_neighbors / 10` to
      // reduce computation.
      // In this scenarios above, memory complexity is not a concern due
      // to the small size of both `fanout` and `num_neighbors`. And it
      // is efficient to allocate a small amount of memory. So the
      // algorithm performence is great in this case.
      std::vector<PickedType> seq(num_neighbors);
      // Assign the seq with [offset, offset + num_neighbors].
      std::iota(seq.begin(), seq.end(), offset);
      for (int64_t i = 0; i < fanout; ++i) {
        auto j = RandomEngine::ThreadLocal()->RandInt(i, num_neighbors);
        std::swap(seq[i], seq[j]);
      }
      // Save the randomly sampled fanout elements to the output tensor.
      std::copy(seq.begin(), seq.begin() + fanout, picked_data_ptr);
      return fanout;
    } else if (fanout < 64) {
      // [Algorithm]
      // Use linear search to verify uniqueness.
      //
      // [Complexity Analysis]
      // Since the set of numbers is small (up to 64), so it is more
      // cost-effective for the CPU to use this algorithm.
      auto begin = picked_data_ptr;
      auto end = picked_data_ptr + fanout;

      while (begin != end) {
        // Put the new random number in the last position.
        *begin = RandomEngine::ThreadLocal()->RandInt(
            offset, offset + num_neighbors);
        // Check if a new value doesn't exist in current
        // range(picked_data_ptr, begin). Otherwise get a new
        // value until we haven't unique range of elements.
        auto it = std::find(picked_data_ptr, begin, *begin);
        if (it == begin) ++begin;
      }
      return fanout;
    } else {
      // [Algorithm]
      // Use hash-set to verify uniqueness. In the best scenario, the
      // time complexity is O(fanout), assuming no conflicts occur.
      //
      // [Complexity Analysis]
      // Let K = (fanout / num_neighbors), the expected number of extra
      // sampling steps is roughly K^2 / (1-K) * num_neighbors, which
      // means in the worst case scenario, the time complexity is
      // O(num_neighbors^2).
      //
      // [Practical Consideration]
      // In practice, we set the threshold K to 1/10. This trade-off is
      // due to the slower performance of std::unordered_set, which
      // would otherwise increase the sampling cost. By doing so, we
      // achieve a balance between theoretical efficiency and practical
      // performance.
      std::unordered_set<PickedType> picked_set;
      while (static_cast<int64_t>(picked_set.size()) < fanout) {
        picked_set.insert(RandomEngine::ThreadLocal()->RandInt(
            offset, offset + num_neighbors));
      }
      std::copy(picked_set.begin(), picked_set.end(), picked_data_ptr);
      return picked_set.size();
    }
  }
}

/** @brief An operator to perform non-uniform sampling. */
static torch::Tensor NonUniformPickOp(
    torch::Tensor probs, int64_t fanout, bool replace) {
  auto positive_probs_indices = probs.nonzero().squeeze(1);
  auto num_positive_probs = positive_probs_indices.size(0);
  if (num_positive_probs == 0) return torch::empty({0}, torch::kLong);
  if ((fanout == -1) || (num_positive_probs <= fanout && !replace)) {
    return positive_probs_indices;
  }
  if (!replace) fanout = std::min(fanout, num_positive_probs);
  if (fanout == 0) return torch::empty({0}, torch::kLong);
  auto ret_tensor = torch::empty({fanout}, torch::kLong);
  auto ret_ptr = ret_tensor.data_ptr<int64_t>();
  AT_DISPATCH_FLOATING_TYPES(
      probs.scalar_type(), "MultinomialSampling", ([&] {
        auto probs_data_ptr = probs.data_ptr<scalar_t>();
        auto positive_probs_indices_ptr =
            positive_probs_indices.data_ptr<int64_t>();

        if (!replace) {
          // The algorithm is from gumbel softmax.
          // s = argmax( logp - log(-log(eps)) ) where eps ~ U(0, 1).
          // Here we can apply exp to the formula which will not affect result
          // of argmax or topk. Then we have
          // s = argmax( p / (-log(eps)) ) where eps ~ U(0, 1).
          // We can also simplify the formula above by
          // s = argmax( p / q ) where q ~ Exp(1).
          if (fanout == 1) {
            // Return argmax(p / q).
            scalar_t max_prob = 0;
            int64_t max_prob_index = -1;
            // We only care about the neighbors with non-zero probability.
            for (auto i = 0; i < num_positive_probs; ++i) {
              // Calculate (p / q) for the current neighbor.
              scalar_t current_prob =
                  probs_data_ptr[positive_probs_indices_ptr[i]] /
                  RandomEngine::ThreadLocal()->Exponential(1.);
              if (current_prob > max_prob) {
                max_prob = current_prob;
                max_prob_index = positive_probs_indices_ptr[i];
              }
            }
            ret_ptr[0] = max_prob_index;
          } else {
            // Return topk(p / q).
            std::vector<std::pair<scalar_t, int64_t>> q(num_positive_probs);
            for (auto i = 0; i < num_positive_probs; ++i) {
              q[i].first = probs_data_ptr[positive_probs_indices_ptr[i]] /
                           RandomEngine::ThreadLocal()->Exponential(1.);
              q[i].second = positive_probs_indices_ptr[i];
            }
            if (fanout < num_positive_probs / 64) {
              // Use partial_sort.
              std::partial_sort(
                  q.begin(), q.begin() + fanout, q.end(), std::greater{});
              for (auto i = 0; i < fanout; ++i) {
                ret_ptr[i] = q[i].second;
              }
            } else {
              // Use nth_element.
              std::nth_element(
                  q.begin(), q.begin() + fanout - 1, q.end(), std::greater{});
              for (auto i = 0; i < fanout; ++i) {
                ret_ptr[i] = q[i].second;
              }
            }
          }
        } else {
          // Calculate cumulative sum of probabilities.
          std::vector<scalar_t> prefix_sum_probs(num_positive_probs);
          scalar_t sum_probs = 0;
          for (auto i = 0; i < num_positive_probs; ++i) {
            sum_probs += probs_data_ptr[positive_probs_indices_ptr[i]];
            prefix_sum_probs[i] = sum_probs;
          }
          // Normalize.
          if ((sum_probs > 1.00001) || (sum_probs < 0.99999)) {
            for (auto i = 0; i < num_positive_probs; ++i) {
              prefix_sum_probs[i] /= sum_probs;
            }
          }
          for (auto i = 0; i < fanout; ++i) {
            // Sample a probability mass from a uniform distribution.
            double uniform_sample =
                RandomEngine::ThreadLocal()->Uniform(0., 1.);
            // Use a binary search to find the index.
            int sampled_index = std::lower_bound(
                                    prefix_sum_probs.begin(),
                                    prefix_sum_probs.end(), uniform_sample) -
                                prefix_sum_probs.begin();
            ret_ptr[i] = positive_probs_indices_ptr[sampled_index];
          }
        }
      }));
  return ret_tensor;
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
 *  - When the value is -1, all neighbors with non-zero probability will be
 * sampled once regardless of replacement. It is equivalent to selecting all
 * neighbors with non-zero probability when the fanout is >= the number of
 * neighbors (and replacement is set to false).
 *  - When the value is a non-negative integer, it serves as a minimum
 * threshold for selecting neighbors.
 * @param replace Boolean indicating whether the sample is performed with or
 * without replacement. If True, a value can be selected multiple times.
 * Otherwise, each value can be selected only once.
 * @param options Tensor options specifying the desired data type of the result.
 * @param probs_or_mask Optional tensor containing the (unnormalized)
 * probabilities associated with each neighboring edge of a node in the original
 * graph. It must be a 1D floating-point tensor with the number of elements
 * equal to the number of edges in the graph.
 * @param picked_data_ptr The destination address where the picked neighbors
 * should be put. Enough memory space should be allocated in advance.
 */
template <typename PickedType>
inline int64_t NonUniformPick(
    int64_t offset, int64_t num_neighbors, int64_t fanout, bool replace,
    const torch::TensorOptions& options, const torch::Tensor& probs_or_mask,
    PickedType* picked_data_ptr) {
  auto local_probs =
      probs_or_mask.size(0) > num_neighbors
          ? probs_or_mask.slice(0, offset, offset + num_neighbors)
          : probs_or_mask;
  auto picked_indices = NonUniformPickOp(local_probs, fanout, replace);
  auto picked_indices_ptr = picked_indices.data_ptr<int64_t>();
  for (int i = 0; i < picked_indices.numel(); ++i) {
    picked_data_ptr[i] =
        static_cast<PickedType>(picked_indices_ptr[i]) + offset;
  }
  return picked_indices.numel();
}

template <typename PickedType>
int64_t Pick(
    int64_t offset, int64_t num_neighbors, int64_t fanout, bool replace,
    const torch::TensorOptions& options,
    const torch::optional<torch::Tensor>& probs_or_mask,
    SamplerArgs<SamplerType::NEIGHBOR> args, PickedType* picked_data_ptr) {
  if (fanout == 0 || num_neighbors == 0) return 0;
  if (probs_or_mask.has_value()) {
    return NonUniformPick(
        offset, num_neighbors, fanout, replace, options, probs_or_mask.value(),
        picked_data_ptr);
  } else {
    return UniformPick(
        offset, num_neighbors, fanout, replace, options, picked_data_ptr);
  }
}

template <SamplerType S, typename PickedType>
int64_t TemporalPick(
    torch::Tensor seed_timestamp, torch::Tensor csc_indices,
    int64_t seed_offset, int64_t offset, int64_t num_neighbors, int64_t fanout,
    bool replace, const torch::TensorOptions& options,
    const torch::optional<torch::Tensor>& seed_pre_time_window,
    const torch::optional<torch::Tensor>& probs_or_mask,
    const torch::optional<torch::Tensor>& node_timestamp,
    const torch::optional<torch::Tensor>& edge_timestamp, SamplerArgs<S> args,
    PickedType* picked_data_ptr) {
  constexpr int64_t kFastPathThreshold = 1000;
  if (S == SamplerType::NEIGHBOR && num_neighbors > kFastPathThreshold &&
      !probs_or_mask.has_value()) {
    auto [success, sampled_edges] = FastTemporalPick(
        seed_timestamp, csc_indices, fanout, replace, seed_pre_time_window,
        node_timestamp, edge_timestamp, seed_offset, offset, num_neighbors);
    if (success) {
      for (size_t i = 0; i < sampled_edges.size(); ++i) {
        picked_data_ptr[i] = static_cast<PickedType>(sampled_edges[i]);
      }
      return sampled_edges.size();
    }
  }
  torch::optional<int64_t> time_window = torch::nullopt;
  if (seed_pre_time_window.has_value()) {
    time_window = utils::GetValueByIndex<int64_t>(
        seed_pre_time_window.value(), seed_offset);
  }
  auto mask = TemporalMask(
      utils::GetValueByIndex<int64_t>(seed_timestamp, seed_offset), csc_indices,
      time_window, probs_or_mask, node_timestamp, edge_timestamp,
      {offset, offset + num_neighbors});
  torch::Tensor masked_prob;
  if (probs_or_mask.has_value()) {
    masked_prob =
        probs_or_mask.value().slice(0, offset, offset + num_neighbors) * mask;
  } else {
    masked_prob = S == SamplerType::NEIGHBOR ? mask.to(torch::kFloat32) : mask;
  }
  if constexpr (S == SamplerType::NEIGHBOR) {
    auto picked_indices = NonUniformPickOp(masked_prob, fanout, replace);
    auto picked_indices_ptr = picked_indices.data_ptr<int64_t>();
    for (int i = 0; i < picked_indices.numel(); ++i) {
      picked_data_ptr[i] =
          static_cast<PickedType>(picked_indices_ptr[i]) + offset;
    }
    return picked_indices.numel();
  }
  if constexpr (is_labor(S)) {
    return Pick(
        offset, num_neighbors, fanout, replace, options, masked_prob, args,
        picked_data_ptr);
  }
}

template <SamplerType S, typename PickedType>
int64_t PickByEtype(
    bool with_seed_offsets, int64_t offset, int64_t num_neighbors,
    const std::vector<int64_t>& fanouts, bool replace,
    const torch::TensorOptions& options, const torch::Tensor& type_per_edge,
    const torch::optional<torch::Tensor>& probs_or_mask, SamplerArgs<S> args,
    PickedType* picked_data_ptr, int64_t seed_index,
    PickedType* subgraph_indptr_ptr,
    const std::vector<int64_t>& etype_id_to_num_picked_offset) {
  int64_t etype_begin = offset;
  int64_t etype_end = offset;
  int64_t picked_total_count = 0;
  AT_DISPATCH_INTEGRAL_TYPES(
      type_per_edge.scalar_type(), "PickByEtype", ([&] {
        const scalar_t* type_per_edge_data = type_per_edge.data_ptr<scalar_t>();
        const auto end = offset + num_neighbors;
        while (etype_begin < end) {
          scalar_t etype = type_per_edge_data[etype_begin];
          TORCH_CHECK(
              etype >= 0 && etype < (int64_t)fanouts.size(),
              "Etype values exceed the number of fanouts.");
          int64_t fanout = fanouts[etype];
          auto etype_end_it = std::upper_bound(
              type_per_edge_data + etype_begin, type_per_edge_data + end,
              etype);
          etype_end = etype_end_it - type_per_edge_data;
          // Do sampling for one etype. The picked nodes aren't stored
          // continuously, but separately for each different etype.
          if (fanout != 0) {
            auto picked_count = 0;
            if (with_seed_offsets) {
              const auto indptr_offset =
                  etype_id_to_num_picked_offset[etype] + seed_index;
              picked_count = Pick(
                  etype_begin, etype_end - etype_begin, fanout, replace,
                  options, probs_or_mask, args,
                  picked_data_ptr + subgraph_indptr_ptr[indptr_offset]);
              TORCH_CHECK(
                  subgraph_indptr_ptr[indptr_offset + 1] -
                          subgraph_indptr_ptr[indptr_offset] ==
                      picked_count,
                  "Actual picked count doesn't match the calculated "
                  "pick number.");
            } else {
              picked_count = Pick(
                  etype_begin, etype_end - etype_begin, fanout, replace,
                  options, probs_or_mask, args,
                  picked_data_ptr + subgraph_indptr_ptr[seed_index] +
                      picked_total_count);
            }
            picked_total_count += picked_count;
          }
          etype_begin = etype_end;
        }
      }));
  return picked_total_count;
}

template <SamplerType S, typename PickedType>
int64_t TemporalPickByEtype(
    torch::Tensor seed_timestamp, torch::Tensor csc_indices,
    int64_t seed_offset, int64_t offset, int64_t num_neighbors,
    const std::vector<int64_t>& fanouts, bool replace,
    const torch::TensorOptions& options, const torch::Tensor& type_per_edge,
    const torch::optional<torch::Tensor>& seed_pre_time_window,
    const torch::optional<torch::Tensor>& probs_or_mask,
    const torch::optional<torch::Tensor>& node_timestamp,
    const torch::optional<torch::Tensor>& edge_timestamp, SamplerArgs<S> args,
    PickedType* picked_data_ptr) {
  int64_t etype_begin = offset;
  int64_t etype_end = offset;
  int64_t pick_offset = 0;
  AT_DISPATCH_INTEGRAL_TYPES(
      type_per_edge.scalar_type(), "TemporalPickByEtype", ([&] {
        const scalar_t* type_per_edge_data = type_per_edge.data_ptr<scalar_t>();
        const auto end = offset + num_neighbors;
        while (etype_begin < end) {
          scalar_t etype = type_per_edge_data[etype_begin];
          TORCH_CHECK(
              etype >= 0 && etype < (int64_t)fanouts.size(),
              "Etype values exceed the number of fanouts.");
          int64_t fanout = fanouts[etype];
          auto etype_end_it = std::upper_bound(
              type_per_edge_data + etype_begin, type_per_edge_data + end,
              etype);
          etype_end = etype_end_it - type_per_edge_data;
          // Do sampling for one etype.
          if (fanout != 0) {
            int64_t picked_count = TemporalPick(
                seed_timestamp, csc_indices, seed_offset, etype_begin,
                etype_end - etype_begin, fanout, replace, options,
                seed_pre_time_window, probs_or_mask, node_timestamp,
                edge_timestamp, args, picked_data_ptr + pick_offset);
            pick_offset += picked_count;
          }
          etype_begin = etype_end;
        }
      }));
  return pick_offset;
}

template <SamplerType S, typename PickedType>
std::enable_if_t<is_labor(S), int64_t> Pick(
    int64_t offset, int64_t num_neighbors, int64_t fanout, bool replace,
    const torch::TensorOptions& options,
    const torch::optional<torch::Tensor>& probs_or_mask, SamplerArgs<S> args,
    PickedType* picked_data_ptr) {
  if (fanout == 0 || num_neighbors == 0) return 0;
  if (probs_or_mask.has_value()) {
    if (fanout < 0) {
      return NonUniformPick(
          offset, num_neighbors, fanout, replace, options,
          probs_or_mask.value(), picked_data_ptr);
    } else {
      int64_t picked_count;
      GRAPHBOLT_DISPATCH_ALL_TYPES(
          probs_or_mask.value().scalar_type(), "LaborPickFloatType", ([&] {
            if (replace) {
              picked_count = LaborPick<true, true, scalar_t>(
                  offset, num_neighbors, fanout, options, probs_or_mask, args,
                  picked_data_ptr);
            } else {
              picked_count = LaborPick<true, false, scalar_t>(
                  offset, num_neighbors, fanout, options, probs_or_mask, args,
                  picked_data_ptr);
            }
          }));
      return picked_count;
    }
  } else if (fanout < 0) {
    return UniformPick(
        offset, num_neighbors, fanout, replace, options, picked_data_ptr);
  } else if (replace) {
    return LaborPick<false, true, float>(
        offset, num_neighbors, fanout, options,
        /* probs_or_mask= */ torch::nullopt, args, picked_data_ptr);
  } else {  // replace = false
    return LaborPick<false, false, float>(
        offset, num_neighbors, fanout, options,
        /* probs_or_mask= */ torch::nullopt, args, picked_data_ptr);
  }
}

template <typename T, typename U>
inline void safe_divide(T& a, U b) {
  a = b > 0 ? (T)(a / b) : std::numeric_limits<T>::infinity();
}

namespace labor {

template <typename T>
inline T invcdf(T u, int64_t n, T rem) {
  constexpr T one = 1;
  return rem * (one - std::pow(one - u, one / n));
}

template <typename T, typename seed_t>
inline T jth_sorted_uniform_random(
    seed_t seed, int64_t t, int64_t c, int64_t j, T& rem, int64_t n) {
  const T u = seed.uniform(t + j * c);
  // https://mathematica.stackexchange.com/a/256707
  rem -= invcdf(u, n, rem);
  return 1 - rem;
}

};  // namespace labor

/**
 * @brief Perform uniform-nonuniform sampling of elements depending on the
 * template parameter NonUniform and return the sampled indices.
 *
 * @param offset The starting edge ID for the connected neighbors of the sampled
 * node.
 * @param num_neighbors The number of neighbors to pick.
 * @param fanout The number of edges to be sampled for each node. It should be
 * >= 0 or -1.
 *  - When the value is -1, all neighbors (with non-zero probability, if
 * weighted) will be sampled once regardless of replacement. It is equivalent to
 * selecting all neighbors with non-zero probability when the fanout is >= the
 * number of neighbors (and replacement is set to false).
 *  - When the value is a non-negative integer, it serves as a minimum
 * threshold for selecting neighbors.
 * @param options Tensor options specifying the desired data type of the result.
 * @param probs_or_mask Optional tensor containing the (unnormalized)
 * probabilities associated with each neighboring edge of a node in the original
 * graph. It must be a 1D floating-point tensor with the number of elements
 * equal to the number of edges in the graph.
 * @param args Contains labor specific arguments.
 * @param picked_data_ptr The destination address where the picked neighbors
 * should be put. Enough memory space should be allocated in advance.
 */
template <
    bool NonUniform, bool Replace, typename ProbsType, SamplerType S,
    typename PickedType, int StackSize>
inline std::enable_if_t<is_labor(S), int64_t> LaborPick(
    int64_t offset, int64_t num_neighbors, int64_t fanout,
    const torch::TensorOptions& options,
    const torch::optional<torch::Tensor>& probs_or_mask, SamplerArgs<S> args,
    PickedType* picked_data_ptr) {
  fanout = Replace ? fanout : std::min(fanout, num_neighbors);
  if (!NonUniform && !Replace && fanout >= num_neighbors) {
    std::iota(picked_data_ptr, picked_data_ptr + num_neighbors, offset);
    return num_neighbors;
  }
  // Assuming max_degree of a vertex is <= 4 billion.
  std::array<std::pair<float, uint32_t>, StackSize> heap;
  auto heap_data = heap.data();
  torch::Tensor heap_tensor;
  if (fanout > StackSize) {
    constexpr int factor = sizeof(heap_data[0]) / sizeof(int32_t);
    heap_tensor = torch::empty({fanout * factor}, torch::kInt32);
    heap_data = reinterpret_cast<std::pair<float, uint32_t>*>(
        heap_tensor.data_ptr<int32_t>());
  }
  const ProbsType* local_probs_data =
      NonUniform ? probs_or_mask.value().data_ptr<ProbsType>() + offset
                 : nullptr;
  if (NonUniform && probs_or_mask.value().size(0) <= num_neighbors) {
    local_probs_data -= offset;
  }
  AT_DISPATCH_INDEX_TYPES(
      args.indices.scalar_type(), "LaborPickMain", ([&] {
        const auto local_indices_data =
            reinterpret_cast<index_t*>(args.indices.data_ptr()) + offset;
        if constexpr (Replace) {
          // [Algorithm] @mfbalin
          // Use a max-heap to get rid of the big random numbers and filter the
          // smallest fanout of them. Implements arXiv:2210.13339 Section A.3.
          // Unlike sampling without replacement below, the same item can be
          // included fanout times in our sample. Thus, we sort and pick the
          // smallest fanout random numbers out of num_neighbors * fanout of
          // them. Each item has fanout many random numbers in the race and the
          // smallest fanout of them get picked. Instead of generating
          // fanout * num_neighbors random numbers and increase the complexity,
          // I devised an algorithm to generate the fanout numbers for an item
          // in a sorted manner on demand, meaning we continue generating random
          // numbers for an item only if it has been sampled that many times
          // already.
          // https://gist.github.com/mfbalin/096dcad5e3b1f6a59ff7ff2f9f541618
          //
          // [Complexity Analysis]
          // Will modify the heap at most linear in O(num_neighbors + fanout)
          // and each modification takes O(log(fanout)). So the total complexity
          // is O((fanout + num_neighbors) log(fanout)). It is possible to
          // decrease the logarithmic factor down to
          // O(log(min(fanout, num_neighbors))).
          std::array<float, StackSize> remaining;
          auto remaining_data = remaining.data();
          torch::Tensor remaining_tensor;
          if (num_neighbors > StackSize) {
            remaining_tensor = torch::empty({num_neighbors}, torch::kFloat32);
            remaining_data = remaining_tensor.data_ptr<float>();
          }
          std::fill_n(remaining_data, num_neighbors, 1.f);
          auto heap_end = heap_data;
          const auto init_count = (num_neighbors + fanout - 1) / num_neighbors;
          auto sample_neighbor_i_with_index_t_jth_time =
              [&](index_t t, int64_t j, uint32_t i) {
                auto rnd = labor::jth_sorted_uniform_random(
                    args.random_seed, t, args.num_nodes, j, remaining_data[i],
                    fanout - j);  // r_t
                if constexpr (NonUniform) {
                  safe_divide(rnd, local_probs_data[i]);
                }  // r_t / \pi_t
                if (heap_end < heap_data + fanout) {
                  heap_end[0] = std::make_pair(rnd, i);
                  if (++heap_end >= heap_data + fanout) {
                    std::make_heap(heap_data, heap_data + fanout);
                  }
                  return false;
                } else if (rnd < heap_data[0].first) {
                  std::pop_heap(heap_data, heap_data + fanout);
                  heap_data[fanout - 1] = std::make_pair(rnd, i);
                  std::push_heap(heap_data, heap_data + fanout);
                  return false;
                } else {
                  remaining_data[i] = -1;
                  return true;
                }
              };
          for (uint32_t i = 0; i < num_neighbors; ++i) {
            const auto t = local_indices_data[i];
            for (int64_t j = 0; j < init_count; j++) {
              sample_neighbor_i_with_index_t_jth_time(t, j, i);
            }
          }
          for (uint32_t i = 0; i < num_neighbors; ++i) {
            if (remaining_data[i] == -1) continue;
            const auto t = local_indices_data[i];
            for (int64_t j = init_count; j < fanout; ++j) {
              if (sample_neighbor_i_with_index_t_jth_time(t, j, i)) break;
            }
          }
        } else {
          // [Algorithm]
          // Use a max-heap to get rid of the big random numbers and filter the
          // smallest fanout of them. Implements arXiv:2210.13339 Section A.3.
          //
          // [Complexity Analysis]
          // the first for loop and std::make_heap runs in time O(fanouts).
          // The next for loop compares each random number to the current
          // minimum fanout numbers. For any given i, the probability that the
          // current random number will replace any number in the heap is fanout
          // / i. Summing from i=fanout to num_neighbors, we get f * (H_n -
          // H_f), where n is num_neighbors and f is fanout, H_f is \sum_j=1^f
          // 1/j. In the end H_n - H_f = O(log n/f), there are n - f iterations,
          // each heap operation takes time log f, so the total complexity is
          // O(f + (n - f)
          // + f log(n/f) log f) = O(n + f log(f) log(n/f)). If f << n (f is a
          // constant in almost all cases), then the average complexity is
          // O(num_neighbors).
          for (uint32_t i = 0; i < fanout; ++i) {
            const auto t = local_indices_data[i];
            auto rnd = args.random_seed.uniform(t);  // r_t
            if constexpr (NonUniform) {
              safe_divide(rnd, local_probs_data[i]);
            }  // r_t / \pi_t
            heap_data[i] = std::make_pair(rnd, i);
          }
          if (!NonUniform || fanout < num_neighbors) {
            std::make_heap(heap_data, heap_data + fanout);
          }
          for (uint32_t i = fanout; i < num_neighbors; ++i) {
            const auto t = local_indices_data[i];
            auto rnd = args.random_seed.uniform(t);  // r_t
            if constexpr (NonUniform) {
              safe_divide(rnd, local_probs_data[i]);
            }  // r_t / \pi_t
            if (rnd < heap_data[0].first) {
              std::pop_heap(heap_data, heap_data + fanout);
              heap_data[fanout - 1] = std::make_pair(rnd, i);
              std::push_heap(heap_data, heap_data + fanout);
            }
          }
        }
      }));
  int64_t num_sampled = 0;
  for (int64_t i = 0; i < fanout; ++i) {
    const auto [rnd, j] = heap_data[i];
    if (!NonUniform || rnd < std::numeric_limits<float>::infinity()) {
      picked_data_ptr[num_sampled++] = offset + j;
    }
  }
  return num_sampled;
}

}  // namespace sampling
}  // namespace graphbolt
