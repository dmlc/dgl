/**
 *  Copyright (c) 2023 by Contributors
 *  Copyright (c) 2023, GT-TDAlab (Muhammed Fatih Balin & Umit V. Catalyurek)
 * @file cuda/neighbor_sampler_int64.cu
 * @brief SampleNeighbor operator instantiation for kInt64 on CUDA.
 */
#include "./neighbor_sampler.cuh"

namespace graphbolt {
namespace ops {

INSTANTIATE_NEIGHBOR_SAMPLER(c10::impl::ScalarTypeToCPPTypeT<torch::kInt64>);

}  //  namespace ops
}  //  namespace graphbolt
