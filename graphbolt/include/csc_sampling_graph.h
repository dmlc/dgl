/**
 *  Copyright (c) 2023 by Contributors
 * @file graphbolt/include/csc_sampling_graph.h
 * @brief Header file of csc sampling graph.
 */

#include <torch/custom_class.h>
#include <torch/torch.h>

#include <string>
#include <vector>

namespace graphbolt {
namespace sampling {

using StringList = std::vector<std::string>;

/**
 * @brief Heterogeneous information in the Graph.
 */
struct HeteroInfo {
  HeteroInfo(
      const StringList& ntypes, const StringList& etypes,
      torch::Tensor& node_type_offset, torch::Tensor& type_per_edge)
      : node_types(ntypes),
        edge_types(etypes),
        node_type_offset(node_type_offset),
        type_per_edge(type_per_edge) {}
  /** @brief List of node types in the graph. */
  StringList node_types;
  /** @brief List of edge types in the graph. */
  StringList edge_types;
  /** @brief Offset array of node type. */
  torch::Tensor node_type_offset;
  /** @brief Type id of each edge, where type id is the corresponding index of
   * edge_types. */
  torch::Tensor type_per_edge;
};

/**
 * @brief A sampling oriented csc format graph.
 */
class CSCSamplingGraph : public torch::CustomClassHolder {
 public:
  /**
   * @brief Constructor for CSC with data.
   * @param num_rows The number of rows in the dense shape.
   * @param num_cols The number of columns in the dense shape.
   * @param indptr The CSC format index pointer array.
   * @param indices The CSC format index array.
   * @param hetero_info Heterogeneous graph information, if present. Nullptr
   * means it is a homogeneous graph.
   */
  CSCSamplingGraph(
      int64_t num_rows, int64_t num_cols, torch::Tensor& indptr,
      torch::Tensor& indices, const std::shared_ptr<HeteroInfo>& hetero_info);

  /**
   * @brief Create a CSC graph from tensors of CSC format.
   * @param indptr Index pointer array of the CSC.
   * @param indices Indices array of the CSC.
   * @param shape Shape of the sparse matrix.
   *
   * @return CSCSamplingGraph
   */
  static c10::intrusive_ptr<CSCSamplingGraph> FromCSC(
      const std::vector<int64_t>& shape, torch::Tensor indptr,
      torch::Tensor indices);

  /**
   * @brief Create a CSC graph from tensors of CSC format.
   * @param indptr Index pointer array of the CSC.
   * @param indices Indices array of the CSC.
   * @param shape Shape of the sparse matrix.
   * @param ntypes A list of node types, if present.
   * @param etypes A list of edge types, if present.
   * @param node_type_offset_ A tensor representing the offset of node types, if
   * present.
   * @param type_per_edge_ A tensor representing the type of each edge, if
   * present.
   *
   * @return CSCSamplingGraph
   */
  static c10::intrusive_ptr<CSCSamplingGraph> FromCSCWithHeteroInfo(
      const std::vector<int64_t>& shape, torch::Tensor indptr,
      torch::Tensor indices, const StringList& ntypes, const StringList& etypes,
      torch::Tensor node_type_offset, torch::Tensor type_per_edge);

  /** @brief Get the number of rows. */
  int64_t NumRows() const { return num_rows_; }

  /** @brief Get the number of columns. */
  int64_t NumCols() const { return num_cols_; }

  /** @brief Get the number of nodes. */
  int64_t NumNodes() const { return indptr_.size(0) - 1; }

  /** @brief Get the number of edges. */
  int64_t NumEdges() const { return indices_.size(0); }

  /** @brief Get the csc index pointer tensor. */
  const torch::Tensor CSCIndptr() const { return indptr_; }

  /** @brief Get the index tensor. */
  const torch::Tensor Indices() const { return indices_; }

  /** @brief Check if the graph is heterogeneous. */
  inline bool IsHeterogeneous() const { return hetero_info_ != nullptr; }

  /** @brief Get the node type offset tensor for a heterogeneous graph. */
  inline const torch::Tensor NodeTypeOffset() const {
    return hetero_info_->node_type_offset;
  }

  /** @brief Get the list of node types for a heterogeneous graph. */
  inline StringList& NodeTypes() const { return hetero_info_->node_types; }

  /** @brief Get the list of edge types for a heterogeneous graph. */
  inline const StringList& EdgeTypes() const {
    return hetero_info_->edge_types;
  }

  /** @brief Get the edge type tensor for a heterogeneous graph. */
  inline const torch::Tensor PerEdgeType() const {
    return hetero_info_->type_per_edge;
  }

 private:
  /** @brief The dense shape. */
  int64_t num_rows_ = 0, num_cols_ = 0;
  /** @brief CSC format index pointer array. */
  torch::Tensor indptr_;
  /** @brief CSC format index array. */
  torch::Tensor indices_;
  /** @brief Heterogeneous graph information, if present. */
  std::shared_ptr<HeteroInfo> hetero_info_;
};

}  // namespace sampling
}  // namespace graphbolt
