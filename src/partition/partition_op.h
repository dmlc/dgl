/*!
 *  Copyright (c) 2021 by Contributors
 * \file ndarray_partition.h 
 * \brief DGL utilities for working with the partitioned NDArrays 
 */


#ifndef DGL_PARTITION_PARTITION_OPS_H
#define DGL_PARTITION_PARTITION_OPS_H

#include <dgl/array.h>
#include <utility>

namespace dgl {
namespace partition {
namespace impl { 

template <DLDeviceType XPU, typename IdType>
std::pair<IdArray, IdArray>
GeneratePermutationFromRemainder(
        int64_t array_size,
        int num_parts,
        IdArray in_idx);

}
}
}

#endif

