/**
 *   Copyright (c) 2023, GT-TDAlab (Muhammed Fatih Balin & Umit V. Catalyurek)
 *   All rights reserved.
 *
 *   Licensed under the Apache License, Version 2.0 (the "License");
 *   you may not use this file except in compliance with the License.
 *   You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *   Unless required by applicable law or agreed to in writing, software
 *   distributed under the License is distributed on an "AS IS" BASIS,
 *   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *   See the License for the specific language governing permissions and
 *   limitations under the License.
 *
 * @file cuda/insubgraph.cu
 * @brief InSubgraph operator implementation on CUDA.
 */

#include <graphbolt/cuda_ops.h>
#include <graphbolt/cuda_sampling_ops.h>

#include "./common.h"

namespace graphbolt {
namespace ops {

c10::intrusive_ptr<sampling::FusedSampledSubgraph> InSubgraph(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor nodes,
    torch::optional<torch::Tensor> type_per_edge) {
  auto [in_degree, sliced_indptr] = SliceCSCIndptr(indptr, nodes);
  auto [output_indptr, output_indices] = IndexSelectCSCImpl(
      in_degree, sliced_indptr, indices, nodes, indptr.size(0) - 2);
  const int64_t num_edges = output_indices.size(0);
  torch::optional<torch::Tensor> output_type_per_edge;
  if (type_per_edge) {
    output_type_per_edge = std::get<1>(IndexSelectCSCImpl(
        in_degree, sliced_indptr, type_per_edge.value(), nodes,
        indptr.size(0) - 2, num_edges));
  }
  auto edge_ids = IndptrEdgeIdsImpl(
      output_indptr, sliced_indptr.scalar_type(), sliced_indptr, num_edges);

  return c10::make_intrusive<sampling::FusedSampledSubgraph>(
      output_indptr, output_indices, edge_ids, nodes, torch::nullopt,
      output_type_per_edge);
}

}  // namespace ops
}  // namespace graphbolt
