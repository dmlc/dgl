/**
 *  Copyright (c) 2021 by Contributors
 * @file ndarray_partition.h
 * @brief DGL utilities for working with the partitioned NDArrays
 */

#ifndef DGL_PARTITION_PARTITION_OP_H_
#define DGL_PARTITION_PARTITION_OP_H_

#include <dgl/array.h>

#include <utility>

namespace dgl {
namespace partition {
namespace impl {

/**
 * @brief Create a permutation that groups indices by the part id when used for
 * slicing, via the remainder. That is, for the input indices A, find I
 * such that A[I] is grouped by part ID.
 *
 * For example, if we have the set of indices [3, 9, 2, 4, 1, 7] and two
 * partitions, the permutation vector would be [2, 3, 0, 1, 4, 5].
 *
 * @tparam XPU The type of device to run on.
 * @tparam IdType The type of the index.
 * @param array_size The total size of the partitioned array.
 * @param num_parts The number parts the array id divided into.
 * @param in_idx The array of indices to group by part id.
 *
 * @return The permutation to group the indices by part id, and the number of
 * indices in each part.
 */
template <DGLDeviceType XPU, typename IdType>
std::pair<IdArray, IdArray> GeneratePermutationFromRemainder(
    int64_t array_size, int num_parts, IdArray in_idx);

/**
 * @brief Generate the set of local indices from the global indices, using
 * remainder. That is, for each index `i` in `global_idx`, the local index
 * is computed as `global_idx[i] / num_parts`.
 *
 * @tparam XPU The type of device to run on.
 * @tparam IdType The type of the index.
 * @param num_parts The number parts the array id divided into.
 * @param global_idx The array of global indices to map.
 *
 * @return The array of local indices.
 */
template <DGLDeviceType XPU, typename IdType>
IdArray MapToLocalFromRemainder(int num_parts, IdArray global_idx);

/**
 * @brief Generate the set of global indices from the local indices, using
 * remainder. That is, for each index `i` in `local_idx`, the global index
 * is computed as `local_idx[i] * num_parts + part_id`.
 *
 * @tparam XPU The type of device to run on.
 * @tparam IdType The type of the index.
 * @param num_parts The number parts the array id divided into.
 * @param local_idx The array of local indices to map.
 * @param part_id The id of the current part.
 *
 * @return The array of global indices.
 */
template <DGLDeviceType XPU, typename IdType>
IdArray MapToGlobalFromRemainder(int num_parts, IdArray local_idx, int part_id);

/**
 * @brief Create a permutation that groups indices by the part id when used for
 * slicing. That is, for the input indices A, find I such that A[I] is grouped
 * by part ID.
 *
 * For example, if we have a range of [0, 5, 10] and the set of indices
 * [3, 9, 2, 4, 1, 7], the permutation vector would be [0, 2, 3, 4, 1, 5].
 *
 * @tparam XPU The type of device to run on.
 * @tparam IdType The type of the index.
 * @tparam RangeType THe type of the range.
 * @param array_size The total size of the partitioned array.
 * @param num_parts The number parts the array id divided into.
 * @param range The exclusive prefix-sum, representing the range of rows
 * assigned to each partition. Must be on the same context as `in_idx`.
 * @param in_idx The array of indices to group by part id.
 *
 * @return The permutation to group the indices by part id, and the number of
 * indices in each part.
 */
template <DGLDeviceType XPU, typename IdType, typename RangeType>
std::pair<IdArray, IdArray> GeneratePermutationFromRange(
    int64_t array_size, int num_parts, IdArray range, IdArray in_idx);

/**
 * @brief Generate the set of local indices from the global indices, using
 * remainder. That is, for each index `i` in `global_idx`, the local index
 * is computed as `global_idx[i] / num_parts`.
 *
 * @tparam XPU The type of device to run on.
 * @tparam IdType The type of the index.
 * @tparam RangeType THe type of the range.
 * @param num_parts The number parts the array id divided into.
 * @param range The exclusive prefix-sum, representing the range of rows
 * assigned to each partition. Must be on the same context as `global_idx`.
 * @param global_idx The array of global indices to map.
 *
 * @return The array of local indices.
 */
template <DGLDeviceType XPU, typename IdType, typename RangeType>
IdArray MapToLocalFromRange(int num_parts, IdArray range, IdArray global_idx);

/**
 * @brief Generate the set of global indices from the local indices, using
 * remainder. That is, for each index `i` in `local_idx`, the global index
 * is computed as `local_idx[i] * num_parts + part_id`.
 *
 * @tparam XPU The type of device to run on.
 * @tparam IdType The type of the index.
 * @tparam RangeType THe type of the range.
 * @param num_parts The number parts the array id divided into.
 * @param range The exclusive prefix-sum, representing the range of rows
 * assigned to each partition. Must be on the same context as `local_idx`.
 * @param local_idx The array of local indices to map.
 * @param part_id The id of the current part.
 *
 * @return The array of global indices.
 */
template <DGLDeviceType XPU, typename IdType, typename RangeType>
IdArray MapToGlobalFromRange(
    int num_parts, IdArray range, IdArray local_idx, int part_id);

}  // namespace impl
}  // namespace partition
}  // namespace dgl

#endif  // DGL_PARTITION_PARTITION_OP_H_
