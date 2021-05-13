#include <gtest/gtest.h>
#include "../../src/partition/ndarray_partition.h"


using namespace dgl;
using namespace dgl::partition;


template<DLDeviceType XPU, typename IdType>
void _TestRemainder()
{
  const int64_t size = 160000;
  const int num_parts = 7;
  NDArrayPartitionRef part = CreatePartitionRemainderBased(
      size,  num_parts);

  IdArray idxs = aten::Range(0, size/10, sizeof(IdType)*8,
      DGLContext{XPU, 0}); 

  std::pair<IdArray, IdArray> result = part->GeneratePermutation(idxs); 

  // first part of result should be the permutation
  IdArray perm = result.first.CopyTo(DGLContext{kDLCPU, 0});
  ASSERT_TRUE(perm.Ptr<IdType>() != nullptr);
  ASSERT_EQ(perm->shape[0], idxs->shape[0]);
  const IdType * const perm_cpu = static_cast<const IdType*>(perm->data);

  // second part of result should be the counts
  IdArray counts = result.second.CopyTo(DGLContext{kDLCPU, 0});
  ASSERT_TRUE(counts.Ptr<int64_t>() != nullptr);
  ASSERT_EQ(counts->shape[0], num_parts);
  const int64_t * const counts_cpu = static_cast<const int64_t*>(counts->data);

  std::vector<int64_t> prefix(num_parts+1, 0);
  for (int p = 0; p < num_parts; ++p) {
    prefix[p+1] = prefix[p] + counts_cpu[p];
  }
  ASSERT_EQ(prefix.back(), idxs->shape[0]);

  // copy original indexes to cpu
  idxs = idxs.CopyTo(DGLContext{kDLCPU, 0});
  const IdType * const idxs_cpu = static_cast<const IdType*>(idxs->data);

  for (int p = 0; p < num_parts; ++p) {
    for (int64_t i = prefix[p]; i < prefix[p+1]; ++i) {
      EXPECT_EQ(idxs_cpu[perm_cpu[i]] % num_parts, p);
    }
  }
}

TEST(PartitionTest, TestRemainderPartition) {
#ifdef DGL_USE_CUDA
  _TestRemainder<kDLGPU, int32_t>();
  _TestRemainder<kDLGPU, int64_t>();
#endif

  // CPU is not implemented
}



