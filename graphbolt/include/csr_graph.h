/**
 *  Copyright (c) 2023 by Contributors
 * @file graphbolt/include/csr_graph.h
 * @brief Header file of csr graph.
 */

#include <torch/custom_class.h>
#include <torch/torch.h>

#include <string>
#include <vector>

namespace graphbolt {
namespace sampling {

struct HeteroMetadata {
  using StringList = std::vector<std::string>;
  /** @brief List of node types in the graph. */
  StringList node_types;
  /** @brief List of edge types in the graph. */
  StringList edge_types;
};

struct HeteroInfo {
  /** @brief Metadata for the heterogeneous graph. */
  HeteroMetadata metadata;
  /** @brief Offset array of node type. */
  torch::Tensor node_type_offset;
  /** @brief Type of each edge. */
  torch::Tensor type_per_edge;
};

class CSR : public torch::CustomClassHolder {
 public:
  /**
   * @brief Constructor for CSR with data.
   * @param num_rows The number of rows in the dense shape.
   * @param num_cols The number of columns in the dense shape.
   * @param indptr The CSR format index pointer array.
   * @param indices The CSR format index array.
   * @param hetero_info Heterogeneous graph information, if present.
   */
  CSR(int64_t num_rows, int64_t num_cols, const torch::Tensor& indptr,
      const torch::Tensor& indices,
      const torch::optional<torch::Tensor>& orig_type_edge_id,
      const torch::optional<HeteroInfo>& hetero_info);

  /**
   * @brief Create a CSR graph from tensors in CSR format.
   * @param shape Shape of the sparse matrix.
   * @param indptr Index pointer array of the CSR.
   * @param indices Indices array of the CSR.
   * @param hetero_info Heterogeneous graph information, if present.
   *
   * @return CSR
   */

  static c10::intrusive_ptr<CSR> FromCSR(
      const std::vector<int64_t>& shape, torch::Tensor indptr,
      torch::Tensor indices,
      const torch::optional<torch::Tensor>& orig_type_edge_id = torch::nullopt,
      const torch::optional<HeteroInfo>& hetero_info = torch::nullopt);

  inline int64_t Nnz() const { return indices_.size(0); }

  inline void SetHeteroInfo(HeteroInfo& hetero_info) {
    hetero_info_ = hetero_info;
  }

  /** @brief Get the number of rows. */
  int64_t NumRows() const { return num_rows_; }

  /** @brief Get the number of columns. */
  int64_t NumCols() const { return num_cols_; }

  /** @brief Get the index pointer tensor. */
  const torch::Tensor& IndPtr() const { return indptr_; }

  /** @brief Get the index tensor. */
  const torch::Tensor& Indices() const { return indices_; }

  /** @brief Get the heterogenous information, if present. */
  const torch::optional<HeteroInfo>& GetHeteroInfo() const {
    return hetero_info_;
  }

 private:
  /** @brief The dense shape. */
  int64_t num_rows_ = 0, num_cols_ = 0;
  /** @brief CSR format index pointer array. */
  torch::Tensor indptr_;
  /** @brief CSR format index array. */
  torch::Tensor indices_;
  /** @brief Heterogeneous edge id of each edge in the original graph. */
  torch::optional<torch::Tensor> orig_type_edge_id_;
  /** @brief Heterogeneous graph information, if present. */
  torch::optional<HeteroInfo> hetero_info_;
};

}  // namespace sampling
}  // namespace graphbolt
