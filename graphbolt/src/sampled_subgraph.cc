/**
 *  Copyright (c) 2023 by Contributors
 * @file sampled_subgraph.cc
 * @brief Source file of sampled subgraph.
 */

#include <graphbolt/sampled_subgraph.h>
#include <graphbolt/serialize.h>
#include <torch/torch.h>

#include <vector>

namespace graphbolt {
namespace sampling {

/**
 * @brief Version number to indicate graph version in serialization and
 * deserialization.
 */
static constexpr int64_t kSampledSubgraphSerializeVersionNumber = 1;

std::vector<torch::Tensor> SampledSubgraph::GetState() {
  std::vector<torch::Tensor> state;

  // Version number.
  torch::Tensor version_num_tensor =
      torch::ones(1, torch::TensorOptions().dtype(torch::kInt64)) *
      kSampledSubgraphSerializeVersionNumber;
  state.push_back(version_num_tensor);

  // Tensors.
  state.push_back(indptr);
  state.push_back(indices);
  state.push_back(reverse_column_node_ids);

  // Optional tensors.
  static torch::Tensor true_tensor =
      torch::ones(1, torch::TensorOptions().dtype(torch::kInt32));
  static torch::Tensor false_tensor =
      torch::zeros(1, torch::TensorOptions().dtype(torch::kInt32));
  if (reverse_row_node_ids.has_value()) {
    state.push_back(true_tensor);
    state.push_back(reverse_row_node_ids.value());
  } else {
    state.push_back(false_tensor);
  }
  if (reverse_edge_ids.has_value()) {
    state.push_back(true_tensor);
    state.push_back(reverse_edge_ids.value());
  } else {
    state.push_back(false_tensor);
  }
  if (type_per_edge.has_value()) {
    state.push_back(true_tensor);
    state.push_back(type_per_edge.value());
  } else {
    state.push_back(false_tensor);
  }

  return state;
}

void SampledSubgraph::SetState(std::vector<torch::Tensor>& state) {
  // Iterator.
  uint32_t i = 0;

  // Version number.
  torch::Tensor& version_num_tensor = state[i++];
  torch::Tensor current_version_num_tensor =
      torch::ones(1, torch::TensorOptions().dtype(torch::kInt64)) *
      SampledSubgraph::kSampledSubgraphSerializeVersionNumber;
  TORCH_CHECK(
      version_num_tensor.equal(current_version_num_tensor),
      "Version number mismatch when deserializing SampledSubgraph.");

  // Tensors.
  indptr = state[i++];
  indices = state[i++];
  reverse_column_node_ids = state[i++];

  // Optional tensors.
  static torch::Tensor true_tensor =
      torch::ones(1, torch::TensorOptions().dtype(torch::kInt32));
  reverse_row_node_ids = torch::nullopt;
  reverse_edge_ids = torch::nullopt;
  type_per_edge = torch::nullopt;
  if (state[i++].equal(true_tensor)) {
    reverse_row_node_ids = state[i++];
  }
  if (state[i++].equal(true_tensor)) {
    reverse_edge_ids = state[i++];
  }
  if (state[i++].equal(true_tensor)) {
    type_per_edge = state[i++];
  }
}

}  // namespace sampling
}  // namespace graphbolt
