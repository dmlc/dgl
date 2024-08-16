/**
 *  Copyright (c) 2023 by Contributors
 *  Copyright (c) 2023, GT-TDAlab (Muhammed Fatih Balin & Umit V. Catalyurek)
 * @file expand_indptr.h
 * @brief ExpandIndptr operators.
 */
#ifndef GRAPHBOLT_EXPAND_INDPTR_H_
#define GRAPHBOLT_EXPAND_INDPTR_H_

#include <torch/script.h>

namespace graphbolt {
namespace ops {

/**
 * @brief ExpandIndptr implements conversion from a given indptr offset
 * tensor to a COO format tensor. If node_ids is not given, it is assumed to be
 * equal to torch::arange(indptr.size(0) - 1, dtype=dtype).
 *
 * @param indptr       The indptr offset tensor.
 * @param dtype        The dtype of the returned output tensor.
 * @param node_ids     1D tensor represents the node ids.
 * @param output_size  Optional, value of indptr[-1]. Passing it eliminates CPU
 * GPU synchronization.
 *
 * @return The resulting tensor.
 */
torch::Tensor ExpandIndptr(
    torch::Tensor indptr, torch::ScalarType dtype,
    torch::optional<torch::Tensor> node_ids = torch::nullopt,
    torch::optional<int64_t> output_size = torch::nullopt);

/**
 * @brief IndptrEdgeIdsImpl implements conversion from a given indptr offset
 * tensor to a COO edge ids tensor. For a given indptr [0, 2, 5, 7] and offset
 * tensor [0, 100, 200], the output will be [0, 1, 100, 101, 102, 201, 202]. If
 * offset was not provided, the output would be [0, 1, 0, 1, 2, 0, 1].
 *
 * @param indptr       The indptr offset tensor.
 * @param dtype        The dtype of the returned output tensor.
 * @param offset       The offset tensor.
 * @param output_size  Optional value of indptr[-1]. Passing it eliminates CPU
 * GPU synchronization.
 *
 * @return The resulting tensor.
 */
torch::Tensor IndptrEdgeIds(
    torch::Tensor indptr, torch::ScalarType dtype,
    torch::optional<torch::Tensor> offset,
    torch::optional<int64_t> output_size);

}  // namespace ops
}  // namespace graphbolt

#endif  // GRAPHBOLT_EXPAND_INDPTR_H_
