/*!
 *  Copyright (c) 2021 by Contributors
 * \file ndarray_partition.h 
 * \brief DGL utilities for working with the partitioned NDArrays 
 */


#ifndef DGL_PARTITION_NDARRAY_PARTITION_H_
#define DGL_PARTITION_NDARRAY_PARTITION_H_

#include <dgl/runtime/object.h>
#include <dgl/array.h>
#include <utility>

namespace dgl {
namespace partition {

class NDArrayPartition : public runtime::Object {
 public:
  NDArrayPartition(
      int64_t array_size,
      int num_parts);

  virtual ~NDArrayPartition() = default;

  static constexpr const char* _type_key = "partition.NDArrayPartition";

  DGL_DECLARE_OBJECT_TYPE_INFO(NDArrayPartition, Object);

  /**
   * @brief Create a mapping for the given indices to different partitions.
   *
   * @param in_idx The input indices to map.
   *
   * @return A pair containing 0) the permutation to re-order the indices by
   * partition, 1) the number of indices per partition.
   */
  virtual std::tuple<IdArray, IdArray>
  GeneratePermutation(
      const IdArray in_idx) const = 0;

  int64_t ArraySize() const;
  int NumParts() const;

 private:
  int64_t array_size_;
  int num_parts_;
};

DGL_DEFINE_OBJECT_REF(NDArrayPartitionRef, NDArrayPartition);


NDArrayPartitionRef CreatePartitionRemainderBased(
    int64_t array_size,
    int num_parts);

}  // namespace partition
}  // namespace dgl

#endif
