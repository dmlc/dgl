/**
 *  Copyright (c) 2023 by Contributors
 * @file index_select.cc
 * @brief Index select operators.
 */
#include "./index_select.h"

#include <graphbolt/fused_csc_sampling_graph.h>

#include "./macro.h"
#include "./utils.h"

namespace graphbolt {
namespace ops {

torch::Tensor IndexSelect(torch::Tensor input, torch::Tensor index) {
  if (input.is_pinned() &&
      (index.is_pinned() || index.device().type() == c10::DeviceType::CUDA)) {
    GRAPHBOLT_DISPATCH_CUDA_ONLY_DEVICE(
        c10::DeviceType::CUDA, "UVAIndexSelect",
        { return UVAIndexSelectImpl(input, index); });
  }
  return input.index({index.to(torch::kLong)});
}

std::tuple<torch::Tensor, torch::Tensor> IndexSelectCSC(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor nodes) {
  TORCH_CHECK(
      indices.sizes().size() == 1, "IndexSelectCSC only supports 1d tensors");
  if (indices.is_pinned() && utils::is_accessible_from_gpu(indptr) &&
      utils::is_accessible_from_gpu(nodes)) {
    GRAPHBOLT_DISPATCH_CUDA_ONLY_DEVICE(
        c10::DeviceType::CUDA, "UVAIndexSelectCSC",
        { return UVAIndexSelectCSCImpl(indptr, indices, nodes); });
  } else if (
      indices.device().type() == c10::DeviceType::CUDA &&
      utils::is_accessible_from_gpu(indptr) &&
      utils::is_accessible_from_gpu(nodes)) {
    GRAPHBOLT_DISPATCH_CUDA_ONLY_DEVICE(
        c10::DeviceType::CUDA, "nodesSelectCSC",
        { return IndexSelectCSCImpl(indptr, indices, nodes); });
  }
  // For testing purposes, to compare with CPU implementation
  torch::optional<torch::Tensor> temp;
  torch::optional<sampling::FusedCSCSamplingGraph::EdgeAttrMap> temp2;
  sampling::FusedCSCSamplingGraph g(indptr, indices, temp, temp, temp2);
  const auto res = g.InSubgraph(nodes);
  return std::make_tuple(res->indptr, res->indices);
}

}  // namespace ops
}  // namespace graphbolt
