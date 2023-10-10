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

/**
 * @brief Template implementation function to be instantiated with different
 * device types, data types and index types.
 */
template <c10::DeviceType XPU, typename DType, typename IdType>
torch::Tensor UVAIndexSelectImpl(torch::Tensor input, torch::Tensor index);

/**
 * @brief Select rows from pinned memory tensor by GPU kernel.
 *
 * NOTE:
 * 1. The shape of input tensor can be multi-dimensional, but the index tensor
 * must be 1-D.
 * 2. The input tensor must be pinned memory tensor and the index tensor
 * can be on GPU or pinned memory.
 *
 * @param input Input tensor with shape (N, ...).
 * @param index Index tensor with shape (M,).
 * @return torch::Tensor Output tensor with shape (M, ...).
 */
torch::Tensor UVAIndexSelect(torch::Tensor input, torch::Tensor index);

}  // namespace ops
}  // namespace graphbolt

#endif  // GRAPHBOLT_INDEX_SELECT_H_
