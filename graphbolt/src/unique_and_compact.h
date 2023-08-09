/**
 *  Copyright (c) 2023 by Contributors
 *
 * @file unique_and_compact.h
 * @brief Unique and compact op.
 */
#ifndef GRAPHBOLT_SHM_UTILS_H_
#define GRAPHBOLT_SHM_UTILS_H_

#include <torch/torch.h>

namespace graphbolt {
namespace sampling {
/**
 * @brief Removes duplicate elements from the concatenated 'unique_dst_ids' and
 * 'src_ids' tensor and applies the uniqueness information to compact both
 * source and destination tensors.
 *
 * The function performs two main operations:
 *   1. Unique Operation: Concatenates 'unique_dst_ids' and 'src_ids' tensors
 * and extracts unique elements, ensuring no duplicates. The resulting tensor
 * represents all unique elements in both 'src_ids' and 'dst_ids', and the
 * indices correspond to the compacted IDs of the corresponding elements.
 *   2. Compact Operation: Utilizes the reverse mapping derived from the unique
 * operation to transform 'src_ids' and 'dst_ids' into condensed IDs, mapping
 * entries corresponding to the inverse IDs.
 *
 * @param src_ids         A tensor containing source IDs.
 * @param dst_ids         A tensor containing destination IDs.
 * @param unique_dst_ids  A tensor containing unique destination IDs, which is
 * exactly all the unique elements in 'dst_ids'.
 *
 * @return
 * - A tensor representing all unique elements in 'src_ids' and 'dst_ids' after
 * removing duplicates. The indices in this tensor precisely match the compacted
 * IDs of the corresponding elements.
 * - The tensor corresponding to the 'src_ids' tensor, where the entries are
 * mapped to inverse IDs.
 * - The tensor corresponding to the 'dst_ids' tensor, where the entries are
 * mapped to inverse IDs.
 *
 * @example
 *   torch::Tensor src_ids = src
 *   torch::Tensor dst_ids = dst
 *   torch::Tensor unique_dst_ids = torch::unique(dst);
 *   auto result = Unique_and_compact(src_ids, dst_ids, unique_dst_ids);
 *   torch::Tensor unique_result = std::get<0>(result);
 *   torch::Tensor unique_src_tensors = std::get<1>(result);
 *   torch::Tensor unique_dst_tensors = std::get<2>(result);
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> Unique_and_compact(
    const torch::Tensor& src_ids, const torch::Tensor& dst_ids,
    const torch::Tensor unique_dst_ids);

}  // namespace sampling
}  // namespace graphbolt

#endif  // GRAPHBOLT_SHM_UTILS_H_
