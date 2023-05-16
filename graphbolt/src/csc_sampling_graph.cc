/**
 *  Copyright (c) 2023 by Contributors
 * @file graphbolt/include/csc_sampling_graph.cc
 * @brief Source file of sampling graph.
 */

#include "csc_sampling_graph.h"

namespace graphbolt {
namespace sampling {

CSCSamplingGraph::CSCSamplingGraph(
    int64_t num_rows, int64_t num_cols, torch::Tensor& indptr,
    torch::Tensor& indices, const std::shared_ptr<HeteroInfo>& hetero_info)
    : num_rows_(num_rows),
      num_cols_(num_cols),
      indptr_(indptr),
      indices_(indices),
      hetero_info_(hetero_info) {
  TORCH_CHECK(num_rows >= 0);
  TORCH_CHECK(num_cols >= 0);
  TORCH_CHECK(indptr.dim() == 1);
  TORCH_CHECK(indices.dim() == 1);
  TORCH_CHECK(indptr.size(0) == num_rows + 1);
  TORCH_CHECK(indptr.device() == indices.device());
}

c10::intrusive_ptr<CSCSamplingGraph> CSCSamplingGraph::FromCSC(
    const std::vector<int64_t>& shape, torch::Tensor indptr,
    torch::Tensor indices) {
  return c10::make_intrusive<CSCSamplingGraph>(
      shape[0], shape[1], indptr, indices, nullptr);
}

void CSCSamplingGraph::SetHeteroInfo(
    const StringList& ntypes, const StringList& etypes,
    torch::Tensor node_type_offset, torch::Tensor type_per_edge) {
  TORCH_CHECK(node_type_offset.size(0) > 0);
  TORCH_CHECK(node_type_offset.dim() == 1);
  TORCH_CHECK(type_per_edge.size(0) > 0);
  TORCH_CHECK(type_per_edge.dim() == 1);
  TORCH_CHECK(node_type_offset.device() == type_per_edge.device());
  TORCH_CHECK(type_per_edge.device() == indices_.device());
  TORCH_CHECK(!ntypes.empty());
  TORCH_CHECK(!etypes.empty());
  hetero_info_ = std::make_shared<HeteroInfo>(
      ntypes, etypes, node_type_offset, type_per_edge);
}
}  // namespace sampling
}  // namespace graphbolt
