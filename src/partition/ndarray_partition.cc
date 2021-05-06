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
    auto ctx = in_idx->ctx;
    if (ctx.device_type == kDLGPU) {
      ATEN_ID_TYPE_SWITCH(in_idx->dtype, IdType, {
        return impl::GeneratePermutationFromRemainder<kDLGPU, IdType>(
            ArraySize(), NumParts(), in_idx);
      });
    }

    LOG(FATAL) << "Only GPU is supported";
    // should be unreachable
    return std::array<IdArray, IdArray>;
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
