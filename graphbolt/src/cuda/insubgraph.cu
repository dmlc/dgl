/**
 *  Copyright (c) 2023 by Contributors
 *  Copyright (c) 2023, GT-TDAlab (Muhammed Fatih Balin & Umit V. Catalyurek)
 * @file cuda/insubgraph.cu
 * @brief InSubgraph operator implementation on CUDA.
 */

#include <graphbolt/cuda_ops.h>

#include <cub/cub.cuh>

#include "./common.h"

namespace graphbolt {
namespace ops {

c10::intrusive_ptr<sampling::FusedSampledSubgraph> InSubgraph(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor nodes,
    torch::optional<torch::Tensor> type_per_edge) {}

}  // namespace ops
}  // namespace graphbolt
