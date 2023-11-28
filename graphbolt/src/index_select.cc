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
          const auto [indptr, indices, nodes] =
              UVAIndexSelectCSCImpl(indptr, indices, index);
          return c10::make_intrusive<FusedSampledSubgraph>(
              indptr, indices, nodes);
        });
  } else if (
      indptr.device().type() == c10::DeviceType::CUDA &&
      indices.device().type() == c10::DeviceType::CUDA &&
      index.device().type() == c10::DeviceType::CUDA) {
    GRAPHBOLT_DISPATCH_CUDA_ONLY_DEVICE(
        c10::DeviceType::CUDA, "IndexSelectCSC", {
          const auto [indptr, indices, nodes] =
              IndexSelectCSCImpl(indptr, indices, index);
          return c10::make_intrusive<FusedSampledSubgraph>(
              indptr, indices, nodes);
        });
  }
  sampling::FusedCSCSamplingGraph g(indptr, indices);
  return g.InSubgraph(index);
}

}  // namespace ops
}  // namespace graphbolt
