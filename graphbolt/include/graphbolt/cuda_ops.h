/**
 *  Copyright (c) 2023 by Contributors
 *  Copyright (c) 2023, GT-TDAlab (Muhammed Fatih Balin & Umit V. Catalyurek)
 * @file graphbolt/cuda_ops.h
 * @brief Available CUDA operations in Graphbolt.
 */

#include <torch/script.h>


namespace graphbolt {
namespace ops {

std::pair<torch::Tensor, torch::Tensor> Sort(torch::Tensor input, int num_bits);

std::tuple<torch::Tensor, torch::Tensor> IndexSelectCSCImpl(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor nodes);

std::tuple<torch::Tensor, torch::Tensor> UVAIndexSelectCSCImpl(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor nodes);

torch::Tensor UVAIndexSelectImpl(torch::Tensor input, torch::Tensor index);

}  //  namespace ops
}  //  namespace graphbolt
