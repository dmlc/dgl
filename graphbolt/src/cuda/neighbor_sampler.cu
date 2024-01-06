/**
 *  Copyright (c) 2023 by Contributors
 *  Copyright (c) 2023, GT-TDAlab (Muhammed Fatih Balin & Umit V. Catalyurek)
 * @file cuda/neighbor_sampler.cu
 * @brief SampleNeighbor operator dispatch for indptr_t on CUDA.
 */
#include "./neighbor_sampler.cuh"

namespace graphbolt {
namespace ops {

extern INSTANTIATE_NEIGHBOR_SAMPLER(
    c10::impl::ScalarTypeToCPPTypeT<torch::kInt8>);
extern INSTANTIATE_NEIGHBOR_SAMPLER(
    c10::impl::ScalarTypeToCPPTypeT<torch::kUInt8>);
extern INSTANTIATE_NEIGHBOR_SAMPLER(
    c10::impl::ScalarTypeToCPPTypeT<torch::kInt16>);
extern INSTANTIATE_NEIGHBOR_SAMPLER(
    c10::impl::ScalarTypeToCPPTypeT<torch::kInt32>);
extern INSTANTIATE_NEIGHBOR_SAMPLER(
    c10::impl::ScalarTypeToCPPTypeT<torch::kInt64>);

c10::intrusive_ptr<sampling::FusedSampledSubgraph> SampleNeighbors(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor nodes,
    const std::vector<int64_t>& fanouts, bool replace, bool layer,
    bool return_eids, torch::optional<torch::Tensor> type_per_edge,
    torch::optional<torch::Tensor> probs_or_mask) {
  return AT_DISPATCH_INTEGRAL_TYPES(
      indptr.scalar_type(), "SampleNeighborsIndptr", ([&] {
        return SampleNeighborsIndptr<scalar_t>(
            indptr, indices, nodes, fanouts, replace, layer, return_eids,
            type_per_edge, probs_or_mask);
      }));
}

}  //  namespace ops
}  //  namespace graphbolt
