/**
 *  Copyright (c) 2023 by Contributors
 * @file graphbolt/sampled_subgraph.h
 * @brief Header file of sampled sub graph.
 */

#include <torch/custom_class.h>
#include <torch/torch.h>

namespace graphbolt {
namespace sampling {

struct SampledSubgraph : torch::CustomClassHolder {
 public:
  /**
   * @brief Constructor for the SampledSubgraph struct.
   *
   * @param indptr CSC format index pointer array.
   * @param indices CSC format index array.
   * @param node_type_offset Offset array of node type.
   * @param type_per_edge Type id of each edge.
   * @param reverse_edge_ids Reversed edge ids in the original graph.
   */
  SampledSubgraph(
      torch::Tensor indptr, torch::Tensor indices,
      torch::optional<torch::Tensor> node_type_offset = torch::nullopt,
      torch::optional<torch::Tensor> type_per_edge = torch::nullopt,
      torch::optional<torch::Tensor> reverse_edge_ids = torch::nullopt)
      : indptr(indptr),
        indices(indices),
        node_type_offset(node_type_offset),
        type_per_edge(type_per_edge),
        reverse_edge_ids(reverse_edge_ids) {}

  /** @brief CSC format index pointer array, where the implicit node ids are
   * already compacted. */
  torch::Tensor indptr;

  /** @brief CSC format index array, where the node ids are original node ids
   * without compaction. */
  torch::Tensor indices;

  /**
   * @brief Offset array of node type. The length of it is equal to the number
   * of node types + 1. The tensor is in ascending order as nodes of the same
   * type have continuous IDs, and larger node IDs are paired with larger node
   * type IDs. Its first value is 0 and last value is the number of nodes. And
   * nodes with ID between `node_type_offset_[i] ~ node_type_offset_[i+1]` are
   * of type id `i`.
   */
  torch::optional<torch::Tensor> node_type_offset;

  /**
   * @brief Type id of each edge, where type id is the corresponding index of
   * edge types. The length of it is equal to the number of edges in the
   * subgraph.
   */
  torch::optional<torch::Tensor> type_per_edge;

  /**
   * @brief Reversed edge ids in the original graph, the edge with id
   * `reverse_edge_ids[i]` in the original graph is mapped to i in this
   * subgraph. This is useful when edge features are needed.
   */
  torch::optional<torch::Tensor> reverse_edge_ids;
};

}  // namespace sampling
}  // namespace graphbolt
