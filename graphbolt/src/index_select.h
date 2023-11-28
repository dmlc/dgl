/**
 *  Copyright (c) 2023 by Contributors
 * @file index_select.h
 * @brief Index select operators.
 */
#ifndef GRAPHBOLT_INDEX_SELECT_H_
#define GRAPHBOLT_INDEX_SELECT_H_

#include <torch/script.h>

namespace graphbolt {
namespace ops {

/** @brief Implemented in the cuda directory. */
c10::intrusive_ptr<sampling::FusedSampledSubgraph> UVAIndexSelectCSCImpl(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor index);

/** @brief Implemented in the cuda directory. */
c10::intrusive_ptr<sampling::FusedSampledSubgraph> IndexSelectCSCImpl(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor index);

/** @brief Implemented in the cuda directory. */
torch::Tensor UVAIndexSelectImpl(torch::Tensor input, torch::Tensor index);

/**
 * @brief Select rows from input tensor according to index tensor.
 *
 * NOTE:
 * 1. The shape of input tensor can be multi-dimensional, but the index tensor
 * must be 1-D.
 * 2. If input is on pinned memory and index is on pinned memory or GPU memory,
 * then UVAIndexSelectImpl will be called. Otherwise, torch::index_select will
 * be called.
 *
 * @param input Input tensor with shape (N, ...).
 * @param index Index tensor with shape (M,).
 * @return torch::Tensor Output tensor with shape (M, ...).
 */
torch::Tensor IndexSelect(torch::Tensor input, torch::Tensor index);

}  // namespace ops
}  // namespace graphbolt

#endif  // GRAPHBOLT_INDEX_SELECT_H_
