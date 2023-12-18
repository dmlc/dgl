/**
 *  Copyright (c) 2023 by Contributors
 *  Copyright (c) 2023, GT-TDAlab (Muhammed Fatih Balin & Umit V. Catalyurek)
 * @file cuda/index_select_impl.cu
 * @brief Index select operator implementation on CUDA.
 */
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAStream.h>
#include <graphbolt/cuda_ops.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <cub/cub.cuh>
#include <numeric>

#include "./common.h"
#include "./utils.h"

namespace graphbolt {
namespace ops {
c10::intrusive_ptr<sampling::FusedSampledSubgraph> SampleNeighbors(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor nodes,
    const std::vector<int64_t>& fanouts, bool replace, bool layer,
    bool return_eids, torch::optional<torch::Tensor> type_per_edge,
    torch::optional<torch::Tensor> probs_or_mask) {}
}  //  namespace ops
}  //  namespace graphbolt
