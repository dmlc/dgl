/**
 *  Copyright (c) 2023 by Contributors
 *  Copyright (c) 2023, GT-TDAlab (Muhammed Fatih Balin & Umit V. Catalyurek)
 * @file cuda/insubgraph.cu
 * @brief InSubgraph operator implementation on CUDA.
 */

#include <graphbolt/cuda_ops.h>
#include <graphbolt/cuda_sampling_ops.h>

#include <cub/cub.cuh>

#include "./common.h"

namespace graphbolt {
namespace ops {

c10::intrusive_ptr<sampling::FusedSampledSubgraph> InSubgraph(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor nodes,
    torch::optional<torch::Tensor> type_per_edge) {
  auto [output_indptr, output_indices] =
      IndexSelectCSCImpl(indptr, indices, nodes);
  torch::optional<torch::Tensor> output_type_per_edge;
  if (type_per_edge) {
    output_type_per_edge =
        std::get<1>(IndexSelectCSCImpl(indptr, type_per_edge.value(), nodes));
  }
  auto rows = CSCToCOOImpl(
      output_indptr, indices.scalar_type(), output_indices.size(0));
  auto [in_degree, sliced_indptr] = SliceCSCIndptr(indptr, nodes);
  auto i = torch::arange(output_indices.size(0), output_indptr.options());
  auto edge_ids =
      i - output_indptr.gather(0, rows) + sliced_indptr.gather(0, rows);

  return c10::make_intrusive<sampling::FusedSampledSubgraph>(
      output_indptr, output_indices, nodes, torch::nullopt, edge_ids,
      output_type_per_edge);
}

}  // namespace ops
}  // namespace graphbolt
