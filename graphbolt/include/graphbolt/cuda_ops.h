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

/**
 * @brief CSRToCOO implements conversion from a given indptr offset tensor to a
 * COO format tensor including ids in [0, indptr.size(0) - 1).
 *
 * @param input          A tensor containing IDs.
 * @param output_dtype   Dtype of output.
 *
 * @return
 * - The resulting tensor with output_dtype.
 */
torch::Tensor CSRToCOO(torch::Tensor indptr, torch::ScalarType output_dtype);

}  //  namespace ops
}  //  namespace graphbolt
