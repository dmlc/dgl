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

using TensorList = std::vector<torch::Tensor>;
/**
 * @brief Removes duplicate elements from the concatenated 'unique_dst_ids' and
 * 'src_ids' tensors and applies the uniqueness information to compact both
 * source and destination tensor lists.
 *
 * The function performs two main operations:
 *   1. Unique Operation: Concatenates 'unique_dst_ids' and 'src_ids' tensors
 * and extracts unique elements, ensuring no duplicates. The resulting tensor
 * represents all unique elements in both 'src_ids' and 'dst_ids', and the
 * indices correspond to the compacted IDs of the corresponding elements.
 *   2. Compact Operation: Uses the inverse mapping obtained in the unique
 * operation to modify 'src_ids' and 'dst_ids', mapping entries corresponding to
 * the inverse IDs.
 *
 * @param src_ids          A list of source tensors.
 * @param dst_ids          A list of destination tensors.
 * @param unique_dst_ids   A tensor containing unique destination IDs, which is
 * exactly all the unique elements in 'dst_ids'.
 *
 * @return
 * - A tensor representing all unique elements in 'src_ids' and 'dst_ids' after
 * removing duplicates. The indices in this tensor precisely match the compacted
 * IDs of the corresponding elements.
 * - The modified 'src_ids' tensor list with entries corresponding to the
 * inverse IDs mapped.
 * - The modified 'dst_ids' tensor list with entries corresponding to the
 * inverse IDs mapped.
 *
 * @example
 *   TensorList src_tensors = {src_tensor1, src_tensor2, src_tensor3};
 *   TensorList dst_tensors = {dst_tensor1, dst_tensor2, dst_tensor3};
 *   torch::Tensor unique_dst_ids = torch::unique(torch::cat(dst_tensors));
 *   auto result = Unique_and_compact(src_tensors, dst_tensors, unique_dst_ids);
 *   torch::Tensor unique_result = std::get<0>(result);
 *   TensorList unique_src_tensors = std::get<1>(result);
 *   TensorList unique_dst_tensors = std::get<2>(result);
 */
std::tuple<torch::Tensor, TensorList, TensorList> Unique_and_compact(
    const TensorList& src_ids, const TensorList& dst_ids,
    const torch::Tensor unique_dst_ids);

}  // namespace sampling
}  // namespace graphbolt

#endif  // GRAPHBOLT_SHM_UTILS_H_
