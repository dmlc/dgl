/**
 *  Copyright (c) 2023 by Contributors
 *  Copyright (c) 2023, GT-TDAlab (Muhammed Fatih Balin & Umit V. Catalyurek)
 * @file graphbolt/cuda_sampling_ops.h
 * @brief Available CUDA sampling operations in Graphbolt.
 */

#include <graphbolt/fused_sampled_subgraph.h>
#include <torch/script.h>

namespace graphbolt {
namespace ops {

/**
 * @brief Return the subgraph induced on the inbound edges of the given nodes.
 * @param nodes Type agnostic node IDs to form the subgraph.
 *
 * @return FusedSampledSubgraph.
 */
c10::intrusive_ptr<sampling::FusedSampledSubgraph> InSubgraph(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor nodes,
    torch::optional<torch::Tensor> type_per_edge);

}  //  namespace ops
}  //  namespace graphbolt
