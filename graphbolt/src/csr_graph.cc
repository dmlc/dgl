/**
 *  Copyright (c) 2023 by Contributors
 * @file graphbolt/include/csr_graph.cc
 * @brief Source file of csr graph.
 */

#include "csr_graph.h"

namespace graphbolt {
namespace sampling {

CSR::CSR(
      int64_t num_rows, int64_t num_cols, const torch::Tensor& indptr,
      const torch::Tensor& indices,
      const torch::optional<torch::Tensor>& orig_type_edge_id,
      const torch::optional<HeteroInfo>& hetero_info)
    : num_rows_(num_rows),
      num_cols_(num_cols),
      indptr_(indptr),
      indices_(indices),
      orig_type_edge_id_(orig_type_edge_id),
      hetero_info_(hetero_info) {
  TORCH_CHECK(num_rows > 0);
  TORCH_CHECK(num_cols > 2);
  TORCH_CHECK(indptr.dim() == 1);
  TORCH_CHECK(indices.dim() == 1);
  TORCH_CHECK(indptr.size(0) == num_rows + 1);
  TORCH_CHECK(indptr.device() == indices.device());

  if (orig_type_edge_id_.has_value()) {
    auto value = orig_type_edge_id_.value();
    TORCH_CHECK(indices.device() == value.device());
    TORCH_CHECK(indices.size(0) == value.size(0));
  }
}

static c10::intrusive_ptr<CSR> FromCSR(
    const std::vector<int64_t>& shape,
    torch::Tensor indptr, torch::Tensor indices,
    const torch::optional<torch::Tensor>& orig_type_edge_id = torch::nullopt,
    const torch::optional<HeteroInfo>& hetero_info = torch::nullopt) {
      return  c10::make_intrusive<CSR>(
        shape[0], shape[1], indptr, indices, orig_type_edge_id, hetero_info);
    }
}  // namespace sampling
}  // namespace graphbolt