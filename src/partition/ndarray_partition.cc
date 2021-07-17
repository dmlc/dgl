/*!
 *  Copyright (c) 2021 by Contributors
 * \file ndarray_partition.cc
 * \brief DGL utilities for working with the partitioned NDArrays
 */

#include "ndarray_partition.h"

#include <dgl/runtime/registry.h>
#include <dgl/runtime/packed_func.h>
#include <utility>
#include <memory>

#include "partition_op.h"

using namespace dgl::runtime;

namespace dgl {
namespace partition {

NDArrayPartition::NDArrayPartition(
    const int64_t array_size, const int num_parts) :
  array_size_(array_size),
  num_parts_(num_parts) {
}

int64_t NDArrayPartition::ArraySize() const {
  return array_size_;
}

int NDArrayPartition::NumParts() const {
  return num_parts_;
}


class RemainderPartition : public NDArrayPartition {
 public:
  RemainderPartition(
      const int64_t array_size, const int num_parts) :
    NDArrayPartition(array_size, num_parts) {
    // do nothing
  }

  std::pair<IdArray, NDArray>
  GeneratePermutation(
      IdArray in_idx) const override {
    auto ctx = in_idx->ctx;

#ifdef DGL_USE_CUDA
    if (ctx.device_type == kDLGPU) {
      ATEN_ID_TYPE_SWITCH(in_idx->dtype, IdType, {
        return impl::GeneratePermutationFromRemainder<kDLGPU, IdType>(
            ArraySize(), NumParts(), in_idx);
      });
    }
#endif

    LOG(FATAL) << "Remainder based partitioning for the CPU is not yet "
        "implemented.";
    // should be unreachable
    return std::pair<IdArray, NDArray>{};
  }

  IdArray MapToLocal(
      IdArray in_idx) const override {
    auto ctx = in_idx->ctx;
#ifdef DGL_USE_CUDA
    if (ctx.device_type == kDLGPU) {
      ATEN_ID_TYPE_SWITCH(in_idx->dtype, IdType, {
        return impl::MapToLocalFromRemainder<kDLGPU, IdType>(
            NumParts(), in_idx);
      });
    }
#endif

    LOG(FATAL) << "Remainder based partitioning for the CPU is not yet "
        "implemented.";
    // should be unreachable
    return IdArray{};
  }

  IdArray MapToGlobal(
      IdArray in_idx,
      const int part_id) const override {
    auto ctx = in_idx->ctx;
#ifdef DGL_USE_CUDA
    if (ctx.device_type == kDLGPU) {
      ATEN_ID_TYPE_SWITCH(in_idx->dtype, IdType, {
        return impl::MapToGlobalFromRemainder<kDLGPU, IdType>(
            NumParts(), in_idx, part_id);
      });
    }
#endif

    LOG(FATAL) << "Remainder based partitioning for the CPU is not yet "
        "implemented.";
    // should be unreachable
    return IdArray{};
  }

  int64_t PartSize(const int part_id) const override {
    CHECK_LT(part_id, NumParts()) << "Invalid part ID (" << part_id << ") for "
        "partition of size " << NumParts() << ".";
    return ArraySize() / NumParts() + (part_id < ArraySize() % NumParts());
  }
};

NDArrayPartitionRef CreatePartitionRemainderBased(
    const int64_t array_size,
    const int num_parts) {
  return NDArrayPartitionRef(std::make_shared<RemainderPartition>(
          array_size, num_parts));
}

DGL_REGISTER_GLOBAL("partition._CAPI_DGLNDArrayPartitionCreateRemainderBased")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  int64_t array_size = args[0];
  int num_parts = args[1];

  *rv = CreatePartitionRemainderBased(array_size, num_parts);
});

DGL_REGISTER_GLOBAL("partition._CAPI_DGLNDArrayPartitionGetPartSize")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  NDArrayPartitionRef part = args[0];
  int part_id = args[1];

  *rv = part->PartSize(part_id);
});

DGL_REGISTER_GLOBAL("partition._CAPI_DGLNDArrayPartitionMapToLocal")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  NDArrayPartitionRef part = args[0];
  IdArray idxs = args[1];

  *rv = part->MapToLocal(idxs);
});

DGL_REGISTER_GLOBAL("partition._CAPI_DGLNDArrayPartitionMapToGlobal")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  NDArrayPartitionRef part = args[0];
  IdArray idxs = args[1];
  const int part_id = args[2];

  *rv = part->MapToGlobal(idxs, part_id);
});


}  // namespace partition
}  // namespace dgl
