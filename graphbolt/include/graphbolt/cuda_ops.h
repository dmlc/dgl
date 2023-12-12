/**
 *  Copyright (c) 2023 by Contributors
 *  Copyright (c) 2023, GT-TDAlab (Muhammed Fatih Balin & Umit V. Catalyurek)
 * @file graphbolt/cuda_ops.h
 * @brief Available CUDA operations in Graphbolt.
 */

#include <torch/script.h>

namespace graphbolt {
namespace ops {

std::pair<torch::Tensor, torch::Tensor> Sort(torch::Tensor input, int num_bits = 0);

std::tuple<torch::Tensor, torch::Tensor> IndexSelectCSCImpl(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor nodes);

std::tuple<torch::Tensor, torch::Tensor> UVAIndexSelectCSCImpl(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor nodes);

torch::Tensor UVAIndexSelectImpl(torch::Tensor input, torch::Tensor index);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> UniqueAndCompact(
    const torch::Tensor src_ids, const torch::Tensor dst_ids,
    const torch::Tensor unique_dst_ids, int num_bits = 0);

}  //  namespace ops
}  //  namespace graphbolt
