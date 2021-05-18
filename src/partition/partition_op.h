/*!
 *  Copyright (c) 2021 by Contributors
 * \file ndarray_partition.h
 * \brief DGL utilities for working with the partitioned NDArrays
 */


#ifndef DGL_PARTITION_PARTITION_OP_H_
#define DGL_PARTITION_PARTITION_OP_H_

#include <dgl/array.h>
#include <utility>

namespace dgl {
namespace partition {
namespace impl {

/**
 * @brief Create a permutation that groups indices by the part id.
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
template <DLDeviceType XPU, typename IdType>
std::pair<IdArray, IdArray>
GeneratePermutationFromRemainder(
        int64_t array_size,
        int num_parts,
        IdArray in_idx);

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
template <DLDeviceType XPU, typename IdType>
IdArray MapToLocalFromRemainder(
    int num_parts,
    IdArray global_idx);


}  // namespace impl
}  // namespace partition
}  // namespace dgl

#endif  // DGL_PARTITION_PARTITION_OP_H_
