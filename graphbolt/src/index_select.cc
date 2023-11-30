/**
 *  Copyright (c) 2023 by Contributors
 * @file index_select.cc
 * @brief Index select operators.
 */
#include "./index_select.h"

#include <graphbolt/fused_csc_sampling_graph.h>

#include "./macro.h"

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

c10::intrusive_ptr<sampling::FusedSampledSubgraph> IndexSelectCSC(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor index) {
  if (indptr.is_pinned() && indices.is_pinned() &&
      (index.is_pinned() || index.device().type() == c10::DeviceType::CUDA)) {
    GRAPHBOLT_DISPATCH_CUDA_ONLY_DEVICE(
        c10::DeviceType::CUDA, "UVAIndexSelectCSC", {
          const auto [subindptr, subindices] =
              UVAIndexSelectCSCImpl(indptr, indices, index);
          return c10::make_intrusive<sampling::FusedSampledSubgraph>(
              subindptr, subindices, index);
        });
  } else if (
      indptr.device().type() == c10::DeviceType::CUDA &&
      indices.device().type() == c10::DeviceType::CUDA &&
      index.device().type() == c10::DeviceType::CUDA) {
    GRAPHBOLT_DISPATCH_CUDA_ONLY_DEVICE(
        c10::DeviceType::CUDA, "IndexSelectCSC", {
          const auto [subindptr, subindices] =
              IndexSelectCSCImpl(indptr, indices, index);
          return c10::make_intrusive<sampling::FusedSampledSubgraph>(
              subindptr, subindices, index);
        });
  }
  torch::optional<torch::Tensor> temp;
  torch::optional<sampling::FusedCSCSamplingGraph::EdgeAttrMap> temp2;
  sampling::FusedCSCSamplingGraph g(indptr, indices, temp, temp, temp2);
  return g.InSubgraph(index);
}

}  // namespace ops
}  // namespace graphbolt
