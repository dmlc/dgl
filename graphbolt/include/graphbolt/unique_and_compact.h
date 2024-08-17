/**
 *  Copyright (c) 2023 by Contributors
 *
 * @file unique_and_compact.h
 * @brief Unique and compact op.
 */
#ifndef GRAPHBOLT_UNIQUE_AND_COMPACT_H_
#define GRAPHBOLT_UNIQUE_AND_COMPACT_H_

#include <graphbolt/async.h>
#include <torch/torch.h>

namespace graphbolt {
namespace sampling {
/**
 * @brief Removes duplicate elements from the concatenated 'unique_dst_ids' and
 * 'src_ids' tensor and applies the uniqueness information to compact both
 * source and destination tensors.
 *
 * The function performs two main operations:
 *   1. Unique Operation: 'unique(concat(unique_dst_ids, src_ids))', in which
 * the unique operator will guarantee the 'unique_dst_ids' are at the head of
 * the result tensor.
 *   2. Compact Operation: Utilizes the reverse mapping derived from the unique
 * operation to transform 'src_ids' and 'dst_ids' into compacted IDs.
 *
 * @param src_ids         A tensor containing source IDs.
 * @param dst_ids         A tensor containing destination IDs.
 * @param unique_dst_ids  A tensor containing unique destination IDs, which is
 *                        exactly all the unique elements in 'dst_ids'.
 *
 * @return
 * - A tensor representing all unique elements in 'src_ids' and 'dst_ids' after
 * removing duplicates. The indices in this tensor precisely match the compacted
 * IDs of the corresponding elements.
 * - The tensor corresponding to the 'src_ids' tensor, where the entries are
 * mapped to compacted IDs.
 * - The tensor corresponding to the 'dst_ids' tensor, where the entries are
 * mapped to compacted IDs.
 *
 * @example
 *   torch::Tensor src_ids = src
 *   torch::Tensor dst_ids = dst
 *   torch::Tensor unique_dst_ids = torch::unique(dst);
 *   auto result = UniqueAndCompact(src_ids, dst_ids, unique_dst_ids);
 *   torch::Tensor unique_ids = std::get<0>(result);
 *   torch::Tensor compacted_src_ids = std::get<1>(result);
 *   torch::Tensor compacted_dst_ids = std::get<2>(result);
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> UniqueAndCompact(
    const torch::Tensor& src_ids, const torch::Tensor& dst_ids,
    const torch::Tensor unique_dst_ids);

std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>>
UniqueAndCompactBatched(
    const std::vector<torch::Tensor>& src_ids,
    const std::vector<torch::Tensor>& dst_ids,
    const std::vector<torch::Tensor> unique_dst_ids);

c10::intrusive_ptr<Future<
    std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>>>>
UniqueAndCompactBatchedAsync(
    const std::vector<torch::Tensor>& src_ids,
    const std::vector<torch::Tensor>& dst_ids,
    const std::vector<torch::Tensor> unique_dst_ids);

}  // namespace sampling
}  // namespace graphbolt

#endif  // GRAPHBOLT_UNIQUE_AND_COMPACT_H_
