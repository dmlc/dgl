/**
 *  Copyright (c) 2023 by Contributors
 *  Copyright (c) 2023, GT-TDAlab (Muhammed Fatih Balin & Umit V. Catalyurek)
 * @file csc_to_coo.h
 * @brief CSCToCOO operators.
 */
#ifndef GRAPHBOLT_CSR_TO_COO_H_
#define GRAPHBOLT_CSR_TO_COO_H_

#include <torch/script.h>

namespace graphbolt {
namespace ops {

/**
 * @brief CSRToCOO implements conversion from a given indptr offset tensor to a
 * COO format tensor. If original_row_node_ids is not given, it is assumed to be
 * equal to torch::arange(indptr.size(0) - 1, dtype=output_dtype).
 *
 * @param indptr                 The indptr offset tensor.
 * @param output_dtype           Dtype of output.
 * @param num_edges              Optional number of edges, equal to indptr[-1].
 * @param original_row_node_ids  Optional original row ids for indptr.
 *
 * @return The resulting tensor with output_dtype.
 */
torch::Tensor CSRToCOO(
    torch::Tensor indptr, torch::ScalarType output_dtype,
    torch::optional<int64_t> num_edges = torch::nullopt,
    torch::optional<torch::Tensor> original_row_node_ids = torch::nullopt);

}  // namespace ops
}  // namespace graphbolt

#endif  // GRAPHBOLT_CSR_TO_COO_H_
