/**
 *  Copyright (c) 2023 by Contributors
 * @file csc_sampling_graph.cc
 * @brief Source file of sampling graph.
 */

#include <graphbolt/csc_sampling_graph.h>
#include <graphbolt/serialize.h>
#include <torch/torch.h>

#include <cmath>
#include <limits>
#include <numeric>
#include <tuple>
#include <vector>

#include "./random.h"
#include "./shared_memory_utils.h"

namespace graphbolt {
namespace sampling {

CSCSamplingGraph::CSCSamplingGraph(
    const torch::Tensor& indptr, const torch::Tensor& indices,
    const torch::optional<torch::Tensor>& node_type_offset,
    const torch::optional<torch::Tensor>& type_per_edge,
    const torch::optional<EdgeAttrMap>& edge_attributes)
    : indptr_(indptr),
      indices_(indices),
      node_type_offset_(node_type_offset),
      type_per_edge_(type_per_edge),
      edge_attributes_(edge_attributes) {
  TORCH_CHECK(indptr.dim() == 1);
  TORCH_CHECK(indices.dim() == 1);
  TORCH_CHECK(indptr.device() == indices.device());
}

c10::intrusive_ptr<CSCSamplingGraph> CSCSamplingGraph::FromCSC(
    const torch::Tensor& indptr, const torch::Tensor& indices,
    const torch::optional<torch::Tensor>& node_type_offset,
    const torch::optional<torch::Tensor>& type_per_edge,
    const torch::optional<EdgeAttrMap>& edge_attributes) {
  if (node_type_offset.has_value()) {
    auto& offset = node_type_offset.value();
    TORCH_CHECK(offset.dim() == 1);
  }
  if (type_per_edge.has_value()) {
    TORCH_CHECK(type_per_edge.value().dim() == 1);
    TORCH_CHECK(type_per_edge.value().size(0) == indices.size(0));
  }
  if (edge_attributes.has_value()) {
    for (const auto& pair : edge_attributes.value()) {
      TORCH_CHECK(pair.value().size(0) == indices.size(0));
    }
  }
  return c10::make_intrusive<CSCSamplingGraph>(
      indptr, indices, node_type_offset, type_per_edge, edge_attributes);
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
 * @return A lambda function (int64_t offset, int64_t num_neighbors) ->
 * torch::Tensor, which takes offset (the starting edge ID of the given node)
 * and num_neighbors (number of neighbors) as params and returns the pick number
 * of the given node.
 */
auto GetNumPickFn(
    const std::vector<int64_t>& fanouts, bool replace,
    const torch::optional<torch::Tensor>& type_per_edge,
    const torch::optional<torch::Tensor>& probs_or_mask) {
  // If fanouts.size() > 1, returns the total number of all edge types of the
  // given node.
  return [&fanouts, replace, &probs_or_mask, &type_per_edge](
             int64_t offset, int64_t num_neighbors) {
    if (fanouts.size() > 1) {
      return NumPickByEtype(
          fanouts, replace, type_per_edge.value(), probs_or_mask, offset,
          num_neighbors);
    } else {
      return NumPick(fanouts[0], replace, probs_or_mask, offset, num_neighbors);
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
 * @return A lambda function: (int64_t offset, int64_t num_neighbors,
 * PickedType* picked_data_ptr) -> torch::Tensor, which takes offset (the
 * starting edge ID of the given node) and num_neighbors (number of neighbors)
 * as params and puts the picked neighbors at the address specified by
 * picked_data_ptr.
 */
template <SamplerType S>
auto GetPickFn(
    const std::vector<int64_t>& fanouts, bool replace,
    const torch::TensorOptions& options,
    const torch::optional<torch::Tensor>& type_per_edge,
    const torch::optional<torch::Tensor>& probs_or_mask, SamplerArgs<S> args) {
  return [&fanouts, replace, &options, &type_per_edge, &probs_or_mask, args](
             int64_t offset, int64_t num_neighbors, auto picked_data_ptr) {
    // If fanouts.size() > 1, perform sampling for each edge type of each
    // node; otherwise just sample once for each node with no regard of edge
    // types.
    if (fanouts.size() > 1) {
      return PickByEtype(
          offset, num_neighbors, fanouts, replace, options,
          type_per_edge.value(), probs_or_mask, args, picked_data_ptr);
    } else {
      return Pick(
          offset, num_neighbors, fanouts[0], replace, options, probs_or_mask,
          args, picked_data_ptr);
    }
  };
}

template <typename NumPickFn, typename PickFn>
c10::intrusive_ptr<SampledSubgraph> CSCSamplingGraph::SampleNeighborsImpl(
    const torch::Tensor& nodes, bool return_eids, NumPickFn num_pick_fn,
    PickFn pick_fn) const {
  const int64_t num_nodes = nodes.size(0);
  const auto indptr_options = indptr_.options();
  torch::Tensor num_picked_neighbors_per_node =
      torch::empty({num_nodes + 1}, indptr_options);

  // Calculate GrainSize for parallel_for.
  // Set the default grain size to 64.
  const int64_t grain_size = 64;
  torch::Tensor picked_eids;
  torch::Tensor subgraph_indptr;
  torch::Tensor subgraph_indices;
  torch::optional<torch::Tensor> subgraph_type_per_edge = torch::nullopt;

  AT_DISPATCH_INTEGRAL_TYPES(
      indptr_.scalar_type(), "SampleNeighborsImpl", ([&] {
        const scalar_t* indptr_data = indptr_.data_ptr<scalar_t>();
        auto num_picked_neighbors_data_ptr =
            num_picked_neighbors_per_node.data_ptr<scalar_t>();
        num_picked_neighbors_data_ptr[0] = 0;
        const auto nodes_data_ptr = nodes.data_ptr<int64_t>();

        // Step 1. Calculate pick number of each node.
        torch::parallel_for(
            0, num_nodes, grain_size, [&](int64_t begin, int64_t end) {
              for (int64_t i = begin; i < end; ++i) {
                const auto nid = nodes_data_ptr[i];
                TORCH_CHECK(
                    nid >= 0 && nid < NumNodes(),
                    "The seed nodes' IDs should fall within the range of the "
                    "graph's node IDs.");
                const auto offset = indptr_data[nid];
                const auto num_neighbors = indptr_data[nid + 1] - offset;

                num_picked_neighbors_data_ptr[i + 1] =
                    num_neighbors == 0 ? 0 : num_pick_fn(offset, num_neighbors);
              }
            });

        // Step 2. Calculate prefix sum to get total length and offsets of each
        // node. It's also the indptr of the generated subgraph.
        subgraph_indptr = torch::cumsum(num_picked_neighbors_per_node, 0);

        // Step 3. Allocate the tensor for picked neighbors.
        const auto total_length =
            subgraph_indptr.data_ptr<scalar_t>()[num_nodes];
        picked_eids = torch::empty({total_length}, indptr_options);
        subgraph_indices = torch::empty({total_length}, indices_.options());
        if (type_per_edge_.has_value()) {
          subgraph_type_per_edge =
              torch::empty({total_length}, type_per_edge_.value().options());
        }

        // Step 4. Pick neighbors for each node.
        auto picked_eids_data_ptr = picked_eids.data_ptr<scalar_t>();
        auto subgraph_indptr_data_ptr = subgraph_indptr.data_ptr<scalar_t>();
        torch::parallel_for(
            0, num_nodes, grain_size, [&](int64_t begin, int64_t end) {
              for (int64_t i = begin; i < end; ++i) {
                const auto nid = nodes_data_ptr[i];
                const auto offset = indptr_data[nid];
                const auto num_neighbors = indptr_data[nid + 1] - offset;
                const auto picked_number = num_picked_neighbors_data_ptr[i + 1];
                const auto picked_offset = subgraph_indptr_data_ptr[i];
                if (picked_number > 0) {
                  auto actual_picked_count = pick_fn(
                      offset, num_neighbors,
                      picked_eids_data_ptr + picked_offset);
                  TORCH_CHECK(
                      actual_picked_count == picked_number,
                      "Actual picked count doesn't match the calculated pick "
                      "number.");

                  // Step 5. Calculate other attributes and return the subgraph.
                  AT_DISPATCH_INTEGRAL_TYPES(
                      subgraph_indices.scalar_type(),
                      "IndexSelectSubgraphIndices", ([&] {
                        auto subgraph_indices_data_ptr =
                            subgraph_indices.data_ptr<scalar_t>();
                        auto indices_data_ptr = indices_.data_ptr<scalar_t>();
                        for (auto i = picked_offset;
                             i < picked_offset + picked_number; ++i) {
                          subgraph_indices_data_ptr[i] =
                              indices_data_ptr[picked_eids_data_ptr[i]];
                        }
                      }));
                  if (type_per_edge_.has_value()) {
                    AT_DISPATCH_INTEGRAL_TYPES(
                        subgraph_type_per_edge.value().scalar_type(),
                        "IndexSelectTypePerEdge", ([&] {
                          auto subgraph_type_per_edge_data_ptr =
                              subgraph_type_per_edge.value()
                                  .data_ptr<scalar_t>();
                          auto type_per_edge_data_ptr =
                              type_per_edge_.value().data_ptr<scalar_t>();
                          for (auto i = picked_offset;
                               i < picked_offset + picked_number; ++i) {
                            subgraph_type_per_edge_data_ptr[i] =
                                type_per_edge_data_ptr[picked_eids_data_ptr[i]];
                          }
                        }));
                  }
                }
              }
            });
      }));

  torch::optional<torch::Tensor> subgraph_reverse_edge_ids = torch::nullopt;
  if (return_eids) subgraph_reverse_edge_ids = std::move(picked_eids);

  return c10::make_intrusive<SampledSubgraph>(
      subgraph_indptr, subgraph_indices, nodes, torch::nullopt,
      subgraph_reverse_edge_ids, subgraph_type_per_edge);
}

c10::intrusive_ptr<SampledSubgraph> CSCSamplingGraph::SampleNeighbors(
    const torch::Tensor& nodes, const std::vector<int64_t>& fanouts,
    bool replace, bool layer, bool return_eids,
    torch::optional<std::string> probs_name) const {
  torch::optional<torch::Tensor> probs_or_mask = torch::nullopt;
  if (probs_name.has_value() && !probs_name.value().empty()) {
    probs_or_mask = edge_attributes_.value().at(probs_name.value());
    // Note probs will be passed as input for 'torch.multinomial' in deeper
    // stack, which doesn't support 'torch.half' and 'torch.bool' data types. To
    // avoid crashes, convert 'probs_or_mask' to 'float32' data type.
    if (probs_or_mask.value().dtype() == torch::kBool ||
        probs_or_mask.value().dtype() == torch::kFloat16) {
      probs_or_mask = probs_or_mask.value().to(torch::kFloat32);
    }
  }

  if (layer) {
    const int64_t random_seed = RandomEngine::ThreadLocal()->RandInt(
        static_cast<int64_t>(0), std::numeric_limits<int64_t>::max());
    SamplerArgs<SamplerType::LABOR> args{indices_, random_seed, NumNodes()};
    return SampleNeighborsImpl(
        nodes, return_eids,
        GetNumPickFn(fanouts, replace, type_per_edge_, probs_or_mask),
        GetPickFn(
            fanouts, replace, indptr_.options(), type_per_edge_, probs_or_mask,
            args));
  } else {
    SamplerArgs<SamplerType::NEIGHBOR> args;
    return SampleNeighborsImpl(
        nodes, return_eids,
        GetNumPickFn(fanouts, replace, type_per_edge_, probs_or_mask),
        GetPickFn(
            fanouts, replace, indptr_.options(), type_per_edge_, probs_or_mask,
            args));
  }
}

std::tuple<torch::Tensor, torch::Tensor>
CSCSamplingGraph::SampleNegativeEdgesUniform(
    const std::tuple<torch::Tensor, torch::Tensor>& node_pairs,
    int64_t negative_ratio, int64_t max_node_id) const {
  torch::Tensor pos_src;
  std::tie(pos_src, std::ignore) = node_pairs;
  auto neg_len = pos_src.size(0) * negative_ratio;
  auto neg_src = pos_src.repeat(negative_ratio);
  auto neg_dst = torch::randint(0, max_node_id, {neg_len}, pos_src.options());
  return std::make_tuple(neg_src, neg_dst);
}

c10::intrusive_ptr<CSCSamplingGraph>
CSCSamplingGraph::BuildGraphFromSharedMemoryTensors(
    std::tuple<
        SharedMemoryPtr, SharedMemoryPtr,
        std::vector<torch::optional<torch::Tensor>>>&& shared_memory_tensors) {
  auto& optional_tensors = std::get<2>(shared_memory_tensors);
  auto graph = c10::make_intrusive<CSCSamplingGraph>(
      optional_tensors[0].value(), optional_tensors[1].value(),
      optional_tensors[2], optional_tensors[3], torch::nullopt);
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

int64_t NumPick(
    int64_t fanout, bool replace,
    const torch::optional<torch::Tensor>& probs_or_mask, int64_t offset,
    int64_t num_neighbors) {
  int64_t num_valid_neighbors = num_neighbors;
  if (probs_or_mask.has_value()) {
    // Subtract the count of zeros in probs_or_mask.
    AT_DISPATCH_ALL_TYPES(
        probs_or_mask.value().scalar_type(), "CountZero", ([&] {
          scalar_t* probs_data_ptr = probs_or_mask.value().data_ptr<scalar_t>();
          num_valid_neighbors -= std::count(
              probs_data_ptr + offset, probs_data_ptr + offset + num_neighbors,
              0);
        }));
  }
  if (num_valid_neighbors == 0 || fanout == -1) return num_valid_neighbors;
  return replace ? fanout : std::min(fanout, num_valid_neighbors);
}

int64_t NumPickByEtype(
    const std::vector<int64_t>& fanouts, bool replace,
    const torch::Tensor& type_per_edge,
    const torch::optional<torch::Tensor>& probs_or_mask, int64_t offset,
    int64_t num_neighbors) {
  int64_t etype_begin = offset;
  const int64_t end = offset + num_neighbors;
  int64_t total_count = 0;
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
          total_count += NumPick(
              fanouts[etype], replace, probs_or_mask, etype_begin,
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
    const torch::TensorOptions& options,
    const torch::optional<torch::Tensor>& probs_or_mask,
    PickedType* picked_data_ptr) {
  auto local_probs =
      probs_or_mask.value().slice(0, offset, offset + num_neighbors);
  auto positive_probs_indices = local_probs.nonzero().squeeze(1);
  auto num_positive_probs = positive_probs_indices.size(0);
  if (num_positive_probs == 0) return 0;
  if ((fanout == -1) || (num_positive_probs <= fanout && !replace)) {
    std::memcpy(
        picked_data_ptr,
        (positive_probs_indices + offset).data_ptr<PickedType>(),
        num_positive_probs * sizeof(PickedType));
    return num_positive_probs;
  } else {
    if (!replace) fanout = std::min(fanout, num_positive_probs);
    std::memcpy(
        picked_data_ptr,
        (torch::multinomial(local_probs, fanout, replace) + offset)
            .data_ptr<PickedType>(),
        fanout * sizeof(PickedType));
    return fanout;
  }
}

template <typename PickedType>
int64_t Pick(
    int64_t offset, int64_t num_neighbors, int64_t fanout, bool replace,
    const torch::TensorOptions& options,
    const torch::optional<torch::Tensor>& probs_or_mask,
    SamplerArgs<SamplerType::NEIGHBOR> args, PickedType* picked_data_ptr) {
  if (probs_or_mask.has_value()) {
    return NonUniformPick(
        offset, num_neighbors, fanout, replace, options, probs_or_mask,
        picked_data_ptr);
  } else {
    return UniformPick(
        offset, num_neighbors, fanout, replace, options, picked_data_ptr);
  }
}

template <SamplerType S, typename PickedType>
int64_t PickByEtype(
    int64_t offset, int64_t num_neighbors, const std::vector<int64_t>& fanouts,
    bool replace, const torch::TensorOptions& options,
    const torch::Tensor& type_per_edge,
    const torch::optional<torch::Tensor>& probs_or_mask, SamplerArgs<S> args,
    PickedType* picked_data_ptr) {
  int64_t etype_begin = offset;
  int64_t etype_end = offset;
  int64_t pick_offset = 0;
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
          // Do sampling for one etype.
          if (fanout != 0) {
            int64_t picked_count = Pick(
                etype_begin, etype_end - etype_begin, fanout, replace, options,
                probs_or_mask, args, picked_data_ptr + pick_offset);
            pick_offset += picked_count;
          }
          etype_begin = etype_end;
        }
      }));
  return pick_offset;
}

template <typename PickedType>
int64_t Pick(
    int64_t offset, int64_t num_neighbors, int64_t fanout, bool replace,
    const torch::TensorOptions& options,
    const torch::optional<torch::Tensor>& probs_or_mask,
    SamplerArgs<SamplerType::LABOR> args, PickedType* picked_data_ptr) {
  if (fanout == 0) return 0;
  if (probs_or_mask.has_value()) {
    if (fanout < 0) {
      return NonUniformPick(
          offset, num_neighbors, fanout, replace, options, probs_or_mask,
          picked_data_ptr);
    } else {
      int64_t picked_count;
      AT_DISPATCH_FLOATING_TYPES(
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
    return LaborPick<false, true>(
        offset, num_neighbors, fanout, options,
        /* probs_or_mask= */ torch::nullopt, args, picked_data_ptr);
  } else {  // replace = false
    return LaborPick<false, false>(
        offset, num_neighbors, fanout, options,
        /* probs_or_mask= */ torch::nullopt, args, picked_data_ptr);
  }
}

template <typename T, typename U>
inline void safe_divide(T& a, U b) {
  a = b > 0 ? (T)(a / b) : std::numeric_limits<T>::infinity();
}

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
    bool NonUniform, bool Replace, typename ProbsType, typename PickedType>
inline int64_t LaborPick(
    int64_t offset, int64_t num_neighbors, int64_t fanout,
    const torch::TensorOptions& options,
    const torch::optional<torch::Tensor>& probs_or_mask,
    SamplerArgs<SamplerType::LABOR> args, PickedType* picked_data_ptr) {
  fanout = Replace ? fanout : std::min(fanout, num_neighbors);
  if (!NonUniform && !Replace && fanout >= num_neighbors) {
    std::iota(picked_data_ptr, picked_data_ptr + num_neighbors, offset);
    return num_neighbors;
  }
  torch::Tensor heap_tensor = torch::empty({fanout * 2}, torch::kInt32);
  // Assuming max_degree of a vertex is <= 4 billion.
  auto heap_data = reinterpret_cast<std::pair<float, uint32_t>*>(
      heap_tensor.data_ptr<int32_t>());
  const ProbsType* local_probs_data =
      NonUniform ? probs_or_mask.value().data_ptr<ProbsType>() + offset
                 : nullptr;
  AT_DISPATCH_INTEGRAL_TYPES(
      args.indices.scalar_type(), "LaborPickMain", ([&] {
        const scalar_t* local_indices_data =
            args.indices.data_ptr<scalar_t>() + offset;
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
          torch::Tensor remaining =
              torch::ones({num_neighbors}, torch::kFloat32);
          float* rem_data = remaining.data_ptr<float>();
          auto heap_end = heap_data;
          const auto init_count = (num_neighbors + fanout - 1) / num_neighbors;
          auto sample_neighbor_i_with_index_t_jth_time =
              [&](scalar_t t, int64_t j, uint32_t i) {
                auto rnd = labor::jth_sorted_uniform_random(
                    args.random_seed, t, args.num_nodes, j, rem_data[i],
                    fanout - j);  // r_t
                if constexpr (NonUniform) {
                  safe_divide(rnd, local_probs_data[i]);
                }  // r_t / \pi_t
                if (heap_end < heap_data + fanout) {
                  heap_end[0] = std::make_pair(rnd, i);
                  std::push_heap(heap_data, ++heap_end);
                  return false;
                } else if (rnd < heap_data[0].first) {
                  std::pop_heap(heap_data, heap_data + fanout);
                  heap_data[fanout - 1] = std::make_pair(rnd, i);
                  std::push_heap(heap_data, heap_data + fanout);
                  return false;
                } else {
                  rem_data[i] = -1;
                  return true;
                }
              };
          for (uint32_t i = 0; i < num_neighbors; ++i) {
            for (int64_t j = 0; j < init_count; j++) {
              const auto t = local_indices_data[i];
              sample_neighbor_i_with_index_t_jth_time(t, j, i);
            }
          }
          for (uint32_t i = 0; i < num_neighbors; ++i) {
            if (rem_data[i] == -1) continue;
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
            auto rnd =
                labor::uniform_random<float>(args.random_seed, t);  // r_t
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
            auto rnd =
                labor::uniform_random<float>(args.random_seed, t);  // r_t
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
  TORCH_CHECK(
      !Replace || num_sampled == fanout || num_sampled == 0,
      "Sampling with replacement should sample exactly fanout neighbors or 0!");
  return num_sampled;
}

}  // namespace sampling
}  // namespace graphbolt
