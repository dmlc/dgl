/**
 *  Copyright (c) 2023 by Contributors
 * @file graphbolt/sampled_subgraph.h
 * @brief Header file of sampled sub graph.
 */

#ifndef GRAPHBOLT_SAMPLED_SUBGRAPH_H_
#define GRAPHBOLT_SAMPLED_SUBGRAPH_H_

#include <torch/custom_class.h>
#include <torch/torch.h>

namespace graphbolt {
namespace sampling {

/**
 * @brief Struct representing a sampled subgraph.
 *
 * Example usage:
 *
 * Suppose the subgraph has 3 nodes and 9 edges.
 * ```
 * auto indptr = torch::tensor({0, 2, 3, 4}, {torch::kInt64});
 * auto indices = torch::tensor({55, 101, 3, 3}, {torch::kInt64});
 * auto reverse_row_node_ids = torch::tensor({3, 3, 101}, {torch::kInt64});
 *
 * SampledSubgraph sampledSubgraph(indptr, indices, reverse_row_node_ids);
 * ```
 *
 * The `reverse_row_node_ids` indicates that nodes `[3, 3, 101]` in the
 * original graph are mapped to `[0, 1, 2]` in this subgraph, and because
 * `reverse_column_node_ids` is `Null`, `{55, 101, 3, 3}` in `indices` is just
 * the original node ids without compaction.
 *
 * If `reverse_column_node_ids = torch::tensor({55, 101, 3}, {torch::kInt64})`,
 * it would indicate a different mapping for the column nodes. Note this is
 * inconsistent with row, which is legal, as `3` is mapped to `0` and `1` in the
 * row while `2` in the column.
 */
struct SampledSubgraph : torch::CustomClassHolder {
 public:
  /**
   * @brief Constructor for the SampledSubgraph struct.
   *
   * @param indptr CSC format index pointer array.
   * @param indices CSC format index array.
   * @param reverse_row_node_ids Row's reverse edge ids in the original graph.
   * @param node_type_offset Offset array of node type.
   * @param type_per_edge Type id of each edge.
   * @param reverse_edge_ids Reverse edge ids in the original graph.
   * @param reverse_column_node_ids Column's reverse edge ids in the original
   * graph.
   */
  SampledSubgraph(
      torch::Tensor indptr, torch::Tensor indices,
      torch::Tensor reverse_row_node_ids,
      torch::optional<torch::Tensor> node_type_offset = torch::nullopt,
      torch::optional<torch::Tensor> type_per_edge = torch::nullopt,
      torch::optional<torch::Tensor> reverse_column_node_ids = torch::nullopt,
      torch::optional<torch::Tensor> reverse_edge_ids = torch::nullopt)
      : indptr(indptr),
        indices(indices),
        reverse_row_node_ids(reverse_row_node_ids),
        node_type_offset(node_type_offset),
        type_per_edge(type_per_edge),
        reverse_column_node_ids(reverse_column_node_ids),
        reverse_edge_ids(reverse_edge_ids) {}

  /**
   * @brief CSC format index pointer array, where the implicit node ids are
   * already compacted. And the original ids are stored in the
   * `reverse_row_node_ids` field.
   */
  torch::Tensor indptr;

  /**
   * @brief CSC format index array, where the node ids can be compacted ids or
   * original ids. If compacted, the original ids are stored in the
   * `reverse_column_node_ids` field.
   */
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
   * @brief Row's reverse node ids in the original graph. A graph structure can
   * be treated as a coordinated row and column pair, and this is the the mapped
   * ids of the row.
   *
   * @note This is required and the mapping relations can be inconsistent with
   * column's.
   */
  torch::Tensor reverse_row_node_ids;

  /**
   * @brief Column's reverse node ids in the original graph. A graph structure
   * can be treated as a coordinated row and column pair, and this is the the
   * mapped ids of the column.
   *
   * @note This is optional and the mapping relations can be inconsistent with
   * row's.
   */
  torch::optional<torch::Tensor> reverse_column_node_ids;

  /**
   * @brief Reverse edge ids in the original graph, the edge with id
   * `reverse_edge_ids[i]` in the original graph is mapped to `i` in this
   * subgraph. This is useful when edge features are needed.
   */
  torch::optional<torch::Tensor> reverse_edge_ids;
};

}  // namespace sampling
}  // namespace graphbolt

#endif  // GRAPHBOLT_SAMPLED_SUBGRAPH_H_
