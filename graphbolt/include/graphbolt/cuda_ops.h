/**
 *  Copyright (c) 2023 by Contributors
 *  Copyright (c) 2023, GT-TDAlab (Muhammed Fatih Balin & Umit V. Catalyurek)
 * @file graphbolt/cuda_ops.h
 * @brief Available CUDA operations in Graphbolt.
 */

#include <graphbolt/fused_sampled_subgraph.h>
#include <torch/script.h>

namespace graphbolt {
namespace ops {

std::pair<torch::Tensor, torch::Tensor> Sort(torch::Tensor input, int num_bits);

std::tuple<torch::Tensor, torch::Tensor> IndexSelectCSCImpl(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor nodes);

std::tuple<torch::Tensor, torch::Tensor> UVAIndexSelectCSCImpl(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor nodes);

torch::Tensor UVAIndexSelectImpl(torch::Tensor input, torch::Tensor index);

std::tuple<torch::Tensor, torch::Tensor> SliceCSCIndptr(
    torch::Tensor indptr, torch::Tensor nodes);

torch::Tensor CSRToCOO(
    torch::Tensor indptr, torch::ScalarType indices_scalar_type);

c10::intrusive_ptr<sampling::FusedSampledSubgraph>
SampleNeighborsWithoutReplacement(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor nodes,
    const std::vector<int64_t>& fanouts, bool layer, bool return_eids,
    torch::optional<torch::Tensor> type_per_edge = torch::nullopt,
    torch::optional<torch::Tensor> probs_or_mask = torch::nullopt,
    torch::optional<int64_t> random_seed = torch::nullopt);

}  //  namespace ops
}  //  namespace graphbolt
