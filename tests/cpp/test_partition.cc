#include <gtest/gtest.h>

#include "../../src/partition/ndarray_partition.h"
#include "./common.h"

using namespace dgl;
using namespace dgl::partition;

template <DGLDeviceType XPU, typename IdType>
void _TestRemainder_GeneratePermutation() {
  const int64_t size = 160000;
  const int num_parts = 7;
  NDArrayPartitionRef part = CreatePartitionRemainderBased(size, num_parts);

  IdArray idxs =
      aten::Range(0, size / 10, sizeof(IdType) * 8, DGLContext{XPU, 0});

  std::pair<IdArray, IdArray> result = part->GeneratePermutation(idxs);

  // first part of result should be the permutation
  IdArray perm = result.first.CopyTo(DGLContext{kDGLCPU, 0});
  ASSERT_TRUE(perm.Ptr<IdType>() != nullptr);
  ASSERT_EQ(perm->shape[0], idxs->shape[0]);
  const IdType* const perm_cpu = static_cast<const IdType*>(perm->data);

  // second part of result should be the counts
  IdArray counts = result.second.CopyTo(DGLContext{kDGLCPU, 0});
  ASSERT_TRUE(counts.Ptr<int64_t>() != nullptr);
  ASSERT_EQ(counts->shape[0], num_parts);
  const int64_t* const counts_cpu = static_cast<const int64_t*>(counts->data);

  std::vector<int64_t> prefix(num_parts + 1, 0);
  for (int p = 0; p < num_parts; ++p) {
    prefix[p + 1] = prefix[p] + counts_cpu[p];
  }
  ASSERT_EQ(prefix.back(), idxs->shape[0]);

  // copy original indexes to cpu
  idxs = idxs.CopyTo(DGLContext{kDGLCPU, 0});
  const IdType* const idxs_cpu = static_cast<const IdType*>(idxs->data);

  for (int p = 0; p < num_parts; ++p) {
    for (int64_t i = prefix[p]; i < prefix[p + 1]; ++i) {
      EXPECT_EQ(idxs_cpu[perm_cpu[i]] % num_parts, p);
    }
  }
}

template <DGLDeviceType XPU, typename IdType>
void _TestRemainder_MapToX() {
  const int64_t size = 160000;
  const int num_parts = 7;
  NDArrayPartitionRef part = CreatePartitionRemainderBased(size, num_parts);

  for (int part_id = 0; part_id < num_parts; ++part_id) {
    IdArray local = aten::Range(
        0, part->PartSize(part_id), sizeof(IdType) * 8, DGLContext{XPU, 0});
    IdArray global = part->MapToGlobal(local, part_id);
    IdArray act_local = part->MapToLocal(global).CopyTo(CPU);

    // every global index should have the same remainder as the part id
    ASSERT_EQ(global->shape[0], local->shape[0]);
    global = global.CopyTo(CPU);
    for (int64_t i = 0; i < global->shape[0]; ++i) {
      EXPECT_EQ(Ptr<IdType>(global)[i] % num_parts, part_id)
          << "i=" << i << ", num_parts=" << num_parts
          << ", part_id=" << part_id;
    }

    // the remapped local indices to should match the original
    local = local.CopyTo(CPU);
    ASSERT_EQ(local->shape[0], act_local->shape[0]);
    for (int64_t i = 0; i < act_local->shape[0]; ++i) {
      EXPECT_EQ(Ptr<IdType>(local)[i], Ptr<IdType>(act_local)[i]);
    }
  }
}

TEST(PartitionTest, TestRemainderPartition) {
#ifdef DGL_USE_CUDA
  _TestRemainder_GeneratePermutation<kDGLCUDA, int32_t>();
  _TestRemainder_GeneratePermutation<kDGLCUDA, int64_t>();

  _TestRemainder_MapToX<kDGLCUDA, int32_t>();
  _TestRemainder_MapToX<kDGLCUDA, int64_t>();
#endif
  // CPU is not implemented
}

template <typename INDEX, typename RANGE>
int _FindPart(const INDEX idx, const RANGE* const range, const int num_parts) {
  for (int i = 0; i < num_parts; ++i) {
    if (range[i + 1] > idx) {
      return i;
    }
  }

  return -1;
}

template <DGLDeviceType XPU, typename IdType>
void _TestRange_GeneratePermutation() {
  const int64_t size = 160000;
  const int num_parts = 7;
  IdArray range = aten::NewIdArray(
      num_parts + 1, DGLContext{kDGLCPU, 0}, sizeof(IdType) * 8);
  for (int i = 0; i < num_parts; ++i) {
    range.Ptr<IdType>()[i] = (size / num_parts) * i;
  }
  range.Ptr<IdType>()[num_parts] = size;
  NDArrayPartitionRef part = CreatePartitionRangeBased(
      size, num_parts, range.CopyTo(DGLContext{XPU, 0}));

  IdArray idxs =
      aten::Range(0, size / 10, sizeof(IdType) * 8, DGLContext{XPU, 0});

  std::pair<IdArray, IdArray> result = part->GeneratePermutation(idxs);

  // first part of result should be the permutation
  IdArray perm = result.first.CopyTo(DGLContext{kDGLCPU, 0});
  ASSERT_TRUE(perm.Ptr<IdType>() != nullptr);
  ASSERT_EQ(perm->shape[0], idxs->shape[0]);
  const IdType* const perm_cpu = static_cast<const IdType*>(perm->data);

  // second part of result should be the counts
  IdArray counts = result.second.CopyTo(DGLContext{kDGLCPU, 0});
  ASSERT_TRUE(counts.Ptr<int64_t>() != nullptr);
  ASSERT_EQ(counts->shape[0], num_parts);
  const int64_t* const counts_cpu = static_cast<const int64_t*>(counts->data);

  std::vector<int64_t> prefix(num_parts + 1, 0);
  for (int p = 0; p < num_parts; ++p) {
    prefix[p + 1] = prefix[p] + counts_cpu[p];
  }
  ASSERT_EQ(prefix.back(), idxs->shape[0]);

  // copy original indexes to cpu
  idxs = idxs.CopyTo(DGLContext{kDGLCPU, 0});
  const IdType* const idxs_cpu = static_cast<const IdType*>(idxs->data);

  for (int p = 0; p < num_parts; ++p) {
    for (int64_t i = prefix[p]; i < prefix[p + 1]; ++i) {
      EXPECT_EQ(
          _FindPart(idxs_cpu[perm_cpu[i]], range.Ptr<IdType>(), num_parts), p);
    }
  }
}

template <DGLDeviceType XPU, typename IdType>
void _TestRange_MapToX() {
  const int64_t size = 160000;
  const int num_parts = 7;
  IdArray range = aten::NewIdArray(
      num_parts + 1, DGLContext{kDGLCPU, 0}, sizeof(IdType) * 8);
  for (int i = 0; i < num_parts; ++i) {
    Ptr<IdType>(range)[i] = (size / num_parts) * i;
  }
  range.Ptr<IdType>()[num_parts] = size;
  NDArrayPartitionRef part = CreatePartitionRangeBased(
      size, num_parts, range.CopyTo(DGLContext{XPU, 0}));

  for (int part_id = 0; part_id < num_parts; ++part_id) {
    IdArray local = aten::Range(
        0, part->PartSize(part_id), sizeof(IdType) * 8, DGLContext{XPU, 0});
    IdArray global = part->MapToGlobal(local, part_id);
    IdArray act_local = part->MapToLocal(global).CopyTo(CPU);

    ASSERT_EQ(global->shape[0], local->shape[0]);
    global = global.CopyTo(CPU);
    for (int64_t i = 0; i < global->shape[0]; ++i) {
      EXPECT_EQ(
          _FindPart(Ptr<IdType>(global)[i], Ptr<IdType>(range), num_parts),
          part_id)
          << "i=" << i << ", num_parts=" << num_parts << ", part_id=" << part_id
          << ", shape=" << global->shape[0];
    }

    // the remapped local indices to should match the original
    local = local.CopyTo(CPU);
    ASSERT_EQ(local->shape[0], act_local->shape[0]);
    for (int64_t i = 0; i < act_local->shape[0]; ++i) {
      EXPECT_EQ(Ptr<IdType>(local)[i], Ptr<IdType>(act_local)[i]);
    }
  }
}

TEST(PartitionTest, TestRangePartition) {
#ifdef DGL_USE_CUDA
  _TestRange_GeneratePermutation<kDGLCUDA, int32_t>();
  _TestRange_GeneratePermutation<kDGLCUDA, int64_t>();

  _TestRange_MapToX<kDGLCUDA, int32_t>();
  _TestRange_MapToX<kDGLCUDA, int64_t>();
#endif
  // CPU is not implemented
}
