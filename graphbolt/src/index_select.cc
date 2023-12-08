/**
 *  Copyright (c) 2023 by Contributors
 * @file index_select.cc
 * @brief Index select operators.
 */
#include "./index_select.h"

#include <gnndlp/ops.h>
#include <graphbolt/fused_csc_sampling_graph.h>

#include "./macro.h"
#include "./utils.h"

namespace graphbolt {
namespace ops {

torch::Tensor IndexSelect(torch::Tensor input, torch::Tensor index) {
  if (input.is_pinned() && utils::is_accessible_from_gpu(index)) {
    GRAPHBOLT_DISPATCH_CUDA_ONLY_DEVICE(
        c10::DeviceType::CUDA, "UVAIndexSelect",
        { return gnndlp::cuda::ops::UVAIndexSelect(input, index); });
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
        c10::DeviceType::CUDA, "UVAIndexSelectCSC", {
          return gnndlp::cuda::ops::UVAIndexSelectCSC(indptr, indices, nodes);
        });
  } else if (
      indices.device().type() == c10::DeviceType::CUDA &&
      utils::is_accessible_from_gpu(indptr) &&
      utils::is_accessible_from_gpu(nodes)) {
    GRAPHBOLT_DISPATCH_CUDA_ONLY_DEVICE(
        c10::DeviceType::CUDA, "nodesSelectCSC",
        { return gnndlp::cuda::ops::IndexSelectCSC(indptr, indices, nodes); });
  }
  // For testing purposes, to compare with CPU implementation
  torch::optional<torch::Tensor> temp;
  torch::optional<sampling::FusedCSCSamplingGraph::NodeTypeToIDMap> temp2;
  torch::optional<sampling::FusedCSCSamplingGraph::EdgeTypeToIDMap> temp3;
  torch::optional<sampling::FusedCSCSamplingGraph::EdgeAttrMap> temp4;
  sampling::FusedCSCSamplingGraph g(
      indptr, indices, temp, temp, temp2, temp3, temp4);
  const auto res = g.InSubgraph(nodes);
  return std::make_tuple(res->indptr, res->indices);
}

}  // namespace ops
}  // namespace graphbolt
