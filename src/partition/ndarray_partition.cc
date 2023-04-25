/**
 *  Copyright (c) 2021 by Contributors
 * @file ndarray_partition.cc
 * @brief DGL utilities for working with the partitioned NDArrays
 */

#include "ndarray_partition.h"

#include <dgl/runtime/packed_func.h>
#include <dgl/runtime/registry.h>

#include <memory>
#include <utility>

#include "../c_api_common.h"
#include "partition_op.h"

using namespace dgl::runtime;

namespace dgl {
namespace partition {

NDArrayPartition::NDArrayPartition(
    const int64_t array_size, const int num_parts)
    : array_size_(array_size), num_parts_(num_parts) {}

int64_t NDArrayPartition::ArraySize() const { return array_size_; }

int NDArrayPartition::NumParts() const { return num_parts_; }

class RemainderPartition : public NDArrayPartition {
 public:
  RemainderPartition(const int64_t array_size, const int num_parts)
      : NDArrayPartition(array_size, num_parts) {
    // do nothing
  }

  std::pair<IdArray, NDArray> GeneratePermutation(
      IdArray in_idx) const override {
#ifdef DGL_USE_CUDA
    auto ctx = in_idx->ctx;
    if (ctx.device_type == kDGLCUDA) {
      ATEN_ID_TYPE_SWITCH(in_idx->dtype, IdType, {
        return impl::GeneratePermutationFromRemainder<kDGLCUDA, IdType>(
            ArraySize(), NumParts(), in_idx);
      });
    }
#endif

    LOG(FATAL) << "Remainder based partitioning for the CPU is not yet "
                  "implemented.";
    // should be unreachable
    return std::pair<IdArray, NDArray>{};
  }

  IdArray MapToLocal(IdArray in_idx) const override {
#ifdef DGL_USE_CUDA
    auto ctx = in_idx->ctx;
    if (ctx.device_type == kDGLCUDA) {
      ATEN_ID_TYPE_SWITCH(in_idx->dtype, IdType, {
        return impl::MapToLocalFromRemainder<kDGLCUDA, IdType>(
            NumParts(), in_idx);
      });
    }
#endif

    LOG(FATAL) << "Remainder based partitioning for the CPU is not yet "
                  "implemented.";
    // should be unreachable
    return IdArray{};
  }

  IdArray MapToGlobal(IdArray in_idx, const int part_id) const override {
#ifdef DGL_USE_CUDA
    auto ctx = in_idx->ctx;
    if (ctx.device_type == kDGLCUDA) {
      ATEN_ID_TYPE_SWITCH(in_idx->dtype, IdType, {
        return impl::MapToGlobalFromRemainder<kDGLCUDA, IdType>(
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
    CHECK_LT(part_id, NumParts()) << "Invalid part ID (" << part_id
                                  << ") for "
                                     "partition of size "
                                  << NumParts() << ".";
    return ArraySize() / NumParts() + (part_id < ArraySize() % NumParts());
  }
};

class RangePartition : public NDArrayPartition {
 public:
  RangePartition(const int64_t array_size, const int num_parts, IdArray range)
      : NDArrayPartition(array_size, num_parts),
        range_(range),
        // We also need a copy of the range on the CPU, to compute partition
        // sizes. We require the input range on the GPU, as if we have multiple
        // GPUs, we can't know which is the proper one to copy the array to, but
        // we have only one CPU context, and can safely copy the array to that.
        range_cpu_(range.CopyTo(DGLContext{kDGLCPU, 0})) {
    auto ctx = range->ctx;
    if (ctx.device_type != kDGLCUDA) {
      LOG(FATAL) << "The range for an NDArrayPartition is only supported "
                    " on GPUs. Transfer the range to the target device before "
                    "creating the partition.";
    }
  }

  std::pair<IdArray, NDArray> GeneratePermutation(
      IdArray in_idx) const override {
#ifdef DGL_USE_CUDA
    auto ctx = in_idx->ctx;
    if (ctx.device_type == kDGLCUDA) {
      if (ctx.device_type != range_->ctx.device_type ||
          ctx.device_id != range_->ctx.device_id) {
        LOG(FATAL) << "The range for the NDArrayPartition and the input "
                      "array must be on the same device: "
                   << ctx << " vs. " << range_->ctx;
      }
      ATEN_ID_TYPE_SWITCH(in_idx->dtype, IdType, {
        ATEN_ID_TYPE_SWITCH(range_->dtype, RangeType, {
          return impl::GeneratePermutationFromRange<
              kDGLCUDA, IdType, RangeType>(
              ArraySize(), NumParts(), range_, in_idx);
        });
      });
    }
#endif

    LOG(FATAL) << "Remainder based partitioning for the CPU is not yet "
                  "implemented.";
    // should be unreachable
    return std::pair<IdArray, NDArray>{};
  }

  IdArray MapToLocal(IdArray in_idx) const override {
#ifdef DGL_USE_CUDA
    auto ctx = in_idx->ctx;
    if (ctx.device_type == kDGLCUDA) {
      ATEN_ID_TYPE_SWITCH(in_idx->dtype, IdType, {
        ATEN_ID_TYPE_SWITCH(range_->dtype, RangeType, {
          return impl::MapToLocalFromRange<kDGLCUDA, IdType, RangeType>(
              NumParts(), range_, in_idx);
        });
      });
    }
#endif

    LOG(FATAL) << "Remainder based partitioning for the CPU is not yet "
                  "implemented.";
    // should be unreachable
    return IdArray{};
  }

  IdArray MapToGlobal(IdArray in_idx, const int part_id) const override {
#ifdef DGL_USE_CUDA
    auto ctx = in_idx->ctx;
    if (ctx.device_type == kDGLCUDA) {
      ATEN_ID_TYPE_SWITCH(in_idx->dtype, IdType, {
        ATEN_ID_TYPE_SWITCH(range_->dtype, RangeType, {
          return impl::MapToGlobalFromRange<kDGLCUDA, IdType, RangeType>(
              NumParts(), range_, in_idx, part_id);
        });
      });
    }
#endif

    LOG(FATAL) << "Remainder based partitioning for the CPU is not yet "
                  "implemented.";
    // should be unreachable
    return IdArray{};
  }

  int64_t PartSize(const int part_id) const override {
    CHECK_LT(part_id, NumParts()) << "Invalid part ID (" << part_id
                                  << ") for "
                                     "partition of size "
                                  << NumParts() << ".";
    int64_t part_size = -1;
    ATEN_ID_TYPE_SWITCH(range_cpu_->dtype, RangeType, {
      const RangeType* const ptr =
          static_cast<const RangeType*>(range_cpu_->data);
      part_size = ptr[part_id + 1] - ptr[part_id];
    });
    return part_size;
  }

 private:
  IdArray range_;
  IdArray range_cpu_;
};

NDArrayPartitionRef CreatePartitionRemainderBased(
    const int64_t array_size, const int num_parts) {
  return NDArrayPartitionRef(
      std::make_shared<RemainderPartition>(array_size, num_parts));
}

NDArrayPartitionRef CreatePartitionRangeBased(
    const int64_t array_size, const int num_parts, IdArray range) {
  return NDArrayPartitionRef(
      std::make_shared<RangePartition>(array_size, num_parts, range));
}

DGL_REGISTER_GLOBAL("partition._CAPI_DGLNDArrayPartitionCreateRemainderBased")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      int64_t array_size = args[0];
      int num_parts = args[1];

      *rv = CreatePartitionRemainderBased(array_size, num_parts);
    });

DGL_REGISTER_GLOBAL("partition._CAPI_DGLNDArrayPartitionCreateRangeBased")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      const int64_t array_size = args[0];
      const int num_parts = args[1];
      IdArray range = args[2];

      *rv = CreatePartitionRangeBased(array_size, num_parts, range);
    });

DGL_REGISTER_GLOBAL("partition._CAPI_DGLNDArrayPartitionGetPartSize")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      NDArrayPartitionRef part = args[0];
      int part_id = args[1];

      *rv = part->PartSize(part_id);
    });

DGL_REGISTER_GLOBAL("partition._CAPI_DGLNDArrayPartitionMapToLocal")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      NDArrayPartitionRef part = args[0];
      IdArray idxs = args[1];

      *rv = part->MapToLocal(idxs);
    });

DGL_REGISTER_GLOBAL("partition._CAPI_DGLNDArrayPartitionMapToGlobal")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      NDArrayPartitionRef part = args[0];
      IdArray idxs = args[1];
      const int part_id = args[2];

      *rv = part->MapToGlobal(idxs, part_id);
    });

DGL_REGISTER_GLOBAL("partition._CAPI_DGLNDArrayPartitionGeneratePermutation")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      NDArrayPartitionRef part = args[0];
      IdArray idxs = args[1];

      std::pair<IdArray, NDArray> part_perm = part->GeneratePermutation(idxs);
      *rv =
          ConvertNDArrayVectorToPackedFunc({part_perm.first, part_perm.second});
    });

}  // namespace partition
}  // namespace dgl
