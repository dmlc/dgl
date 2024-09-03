/**
 *  Copyright (c) 2023 by Contributors
 * @file graphbolt/fused_sampled_subgraph.h
 * @brief Header file of sampled sub graph.
 */

#ifndef GRAPHBOLT_FUSED_SAMPLED_SUBGRAPH_H_
#define GRAPHBOLT_FUSED_SAMPLED_SUBGRAPH_H_

#include <torch/custom_class.h>
#include <torch/torch.h>

namespace graphbolt {
namespace sampling {

/**
 * @brief Struct representing a sampled subgraph.
 *
 * Example usage:
 *
 * Suppose the subgraph has 3 nodes and 4 edges.
 * ```
 * auto indptr = torch::tensor({0, 2, 3, 4}, {torch::kInt64});
 * auto indices = torch::tensor({55, 101, 3, 3}, {torch::kInt64});
 * auto original_column_node_ids = torch::tensor({3, 3, 101}, {torch::kInt64});
 *
 * FusedSampledSubgraph sampledSubgraph(indptr, indices,
 * original_column_node_ids);
 * ```
 *
 * The `original_column_node_ids` indicates that nodes `[3, 3, 101]` in the
 * original graph are mapped to `[0, 1, 2]` in this subgraph, and because
 * `original_row_node_ids` is `Null`, `{55, 101, 3, 3}` in `indices` is just
 * the original node ids without compaction.
 *
 * If `original_row_node_ids = torch::tensor({55, 101, 3}, {torch::kInt64})`,
 * it would indicate a different mapping for the row nodes. Note this is
 * inconsistent with column, which is legal, as `3` is mapped to `0` and `1` in
 * the column while `2` in the row.
 */
struct FusedSampledSubgraph : torch::CustomClassHolder {
 public:
  /**
   * @brief Constructor for the FusedSampledSubgraph struct.
   *
   * @param indptr CSC format index pointer array.
   * @param indices CSC format index array.
   * @param original_column_node_ids Row's reverse node ids in the original
   * graph.
   * @param original_row_node_ids Column's reverse node ids in the original
   * graph.
   * @param original_edge_ids Mapping of subgraph edge IDs to original
   * FusedCSCSamplingGraph edge IDs.
   * @param type_per_edge Type id of each edge.
   * @param etype_offsets Edge offsets for the sampled edges for the sampled
   * edges that are sorted w.r.t. edge types.
   */
  FusedSampledSubgraph(
      torch::Tensor indptr, torch::optional<torch::Tensor> indices,
      torch::Tensor original_edge_ids,
      torch::optional<torch::Tensor> original_column_node_ids,
      torch::optional<torch::Tensor> original_row_node_ids = torch::nullopt,
      torch::optional<torch::Tensor> type_per_edge = torch::nullopt,
      torch::optional<torch::Tensor> etype_offsets = torch::nullopt)
      : indptr(indptr),
        indices(indices),
        original_edge_ids(original_edge_ids),
        original_column_node_ids(original_column_node_ids),
        original_row_node_ids(original_row_node_ids),
        type_per_edge(type_per_edge),
        etype_offsets(etype_offsets) {}

  FusedSampledSubgraph() = default;

  /**
   * @brief CSC format index pointer array, where the implicit node ids are
   * already compacted. And the original ids are stored in the
   * `original_column_node_ids` field. Its length is equal to:
   * 1 + \sum_{etype} #seeds with dst_node_type(etype)
   */
  torch::Tensor indptr;

  /**
   * @brief CSC format index array, where the node ids can be compacted ids or
   * original ids. If compacted, the original ids are stored in the
   * `original_row_node_ids` field. The indices are sorted w.r.t. their edge
   * types for the heterogenous case.
   *
   * @note This is optional if its fetch operation will be performed later using
   * the original_edge_ids tensor.
   */
  torch::optional<torch::Tensor> indices;

  /**
   * @brief Mapping of subgraph edge IDs to original FusedCSCSamplingGraph
   * edge IDs.
   *
   * In this subgraph, the edge at index i corresponds to the edge with ID
   * original_edge_ids[i] in the original FusedCSCSamplingGraph. Edges are
   * sorted by type for heterogeneous graphs.
   *
   * Note: To retrieve the actual original edge IDs for feature fetching, use
   * the `_ORIGINAL_EDGE_ID` edge attribute in FusedCSCSamplingGraph to map the
   * `original_edge_ids` agin, as IDs may have been remapped during conversion
   * to FusedCSCSamplingGraph.
   */
  torch::Tensor original_edge_ids;

  /**
   * @brief Column's reverse node ids in the original graph. A graph structure
   * can be treated as a coordinated row and column pair, and this is the the
   * mapped ids of the column.
   *
   * @note This is optional and the mapping relations can be inconsistent with
   * column's. It can be missing when the sampling algorithm is called via a
   * sliced sampled subgraph with missing seeds argument.
   */
  torch::optional<torch::Tensor> original_column_node_ids;

  /**
   * @brief Row's reverse node ids in the original graph. A graph structure
   * can be treated as a coordinated row and column pair, and this is the the
   * mapped ids of the row.
   *
   * @note This is optional and the mapping relations can be inconsistent with
   * row's.
   */
  torch::optional<torch::Tensor> original_row_node_ids;

  /**
   * @brief Type id of each edge, where type id is the corresponding index of
   * edge types. The length of it is equal to the number of edges in the
   * subgraph.
   *
   * @note This output is not created by the CUDA implementation as the edges
   * are sorted w.r.t edge types, one has to use etype_offsets to infer the edge
   * type information. This field is going to be deprecated. It can be generated
   * when needed by computing gb.expand_indptr(etype_offsets).
   */
  torch::optional<torch::Tensor> type_per_edge;

  /**
   * @brief Offsets of each etype,
   * type_per_edge[etype_offsets[i]: etype_offsets[i + 1]] == i
   * It has length equal to (1 + #etype), and the edges are guaranteed to be
   * sorted w.r.t. their edge types.
   */
  torch::optional<torch::Tensor> etype_offsets;
};

}  // namespace sampling
}  // namespace graphbolt

#endif  // GRAPHBOLT_FUSED_SAMPLED_SUBGRAPH_H_
