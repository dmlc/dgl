/*!
 *  Copyright (c) 2021 by Contributors
 * \file ndarray_partition.cc
 * \brief DGL utilities for working with the partitioned NDArrays 
 */

#include "ndarray_partition.h"

namespace dgl {
namespace partition {


NDArrayPartition::NDArrayPartition(
    const int64_t array_size, const int num_parts) :
  array_size_(array_size),
  num_parts_(num_parts)
{
}

int64_t NDArrayPartition::ArraySize() const
{
  return array_size_;
}

int NDArrayPartition::NumParts() const
{
  return num_parts_;
}


class RemainderPartition : public NDArrayPartition {
 public:
  RemainderPartition::RemainderPartition(
      const int64_t array_size, const int num_parts) :
    NDArrayPartition(array_size, num_parts)
  {
  }

  std::tuple<IdArray, IdArray>
  GeneratePermutation(
      IdArray in_idx) const override
  {
    SWITCH_XPU();
  }
}

DGL_REGISTER_GLOBAL("partition._CAPI_DGLNDArrayPartitionCreateRemainderBased")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  int64_t array_size = args[0];
  int num_parts = args[1];

  *rv = NDArrayPartitionRef(std::make_shared<RemainderPartition>(
        array_size, num_parts));
});

}
}
