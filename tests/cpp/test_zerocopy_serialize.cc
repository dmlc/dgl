#include <dgl/array.h>
#include <dgl/immutable_graph.h>
#include <dgl/zerocopy_serializer.h>
#include <dmlc/memory_io.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <iostream>
#include <vector>

#include "../../src/graph/heterograph.h"
#include "../../src/graph/unit_graph.h"
#include "./common.h"

#ifndef _WIN32

using namespace dgl;
using namespace dgl::aten;
using namespace dmlc;
// Function to convert an idarray to string
std::string IdArrayToStr(IdArray arr) {
  arr = arr.CopyTo(DLContext{kDLCPU, 0});
  int64_t len = arr->shape[0];
  std::ostringstream oss;
  oss << "(" << len << ")[";
  if (arr->dtype.bits == 32) {
    int32_t *data = static_cast<int32_t *>(arr->data);
    for (int64_t i = 0; i < len; ++i) {
      oss << data[i] << " ";
    }
  } else {
    int64_t *data = static_cast<int64_t *>(arr->data);
    for (int64_t i = 0; i < len; ++i) {
      oss << data[i] << " ";
    }
  }
  oss << "]";
  return oss.str();
}

TEST(ZeroCopySerialize, NDArray) {
  auto tensor1 = VecToIdArray<int64_t>({1, 2, 5, 3});
  auto tensor2 = VecToIdArray<int64_t>({6, 6, 5, 7});

  std::string nonzerocopy_blob;
  dmlc::MemoryStringStream ifs(&nonzerocopy_blob);
  static_cast<dmlc::Stream *>(&ifs)->Write(tensor1);
  static_cast<dmlc::Stream *>(&ifs)->Write(tensor2);

  std::string zerocopy_blob;
  StringStreamWithBuffer zc_write_strm(&zerocopy_blob);
  static_cast<dmlc::Stream *>(&zc_write_strm)->Write(tensor1);
  static_cast<dmlc::Stream *>(&zc_write_strm)->Write(tensor2);

  LOG(INFO) << nonzerocopy_blob.size() << std::endl;
  LOG(INFO) << zerocopy_blob.size() << std::endl;

  std::vector<void*> new_ptr_list;
  // Use memcpy to mimic remote machine reconstruction
  for (auto ptr : zc_write_strm.buffer_list()) {
    auto new_ptr = malloc(ptr.size);
    memcpy(new_ptr, ptr.data, ptr.size);
    new_ptr_list.emplace_back(new_ptr);
  }

  NDArray loadtensor1, loadtensor2;
  StringStreamWithBuffer zc_read_strm(&zerocopy_blob, new_ptr_list);
  static_cast<dmlc::Stream *>(&zc_read_strm)->Read(&loadtensor1);
  static_cast<dmlc::Stream *>(&zc_read_strm)->Read(&loadtensor2);
}

TEST(ZeroCopySerialize, SharedMem) {
  auto tensor1 = VecToIdArray<int64_t>({1, 2, 5, 3});
  DLDataType dtype = {kDLInt, 64, 1};
  std::vector<int64_t> shape{4};
  DLContext cpu_ctx = {kDLCPU, 0};
  auto shared_tensor =
    NDArray::EmptyShared("test", shape, dtype, cpu_ctx, true);
  shared_tensor.CopyFrom(tensor1);

  std::string nonzerocopy_blob;
  dmlc::MemoryStringStream ifs(&nonzerocopy_blob);
  static_cast<dmlc::Stream *>(&ifs)->Write(shared_tensor);

  std::string zerocopy_blob;
  StringStreamWithBuffer zc_write_strm(&zerocopy_blob, false);
  static_cast<dmlc::Stream *>(&zc_write_strm)->Write(shared_tensor);

  LOG(INFO) << nonzerocopy_blob.size();
  LOG(INFO) << zerocopy_blob.size();

  NDArray loadtensor1, loadtensor2;
  StringStreamWithBuffer zc_read_strm(&zerocopy_blob, false);
  static_cast<dmlc::Stream *>(&zc_read_strm)->Read(&loadtensor1);
}

TEST(ZeroCopySerialize, HeteroGraph) {
  auto src = VecToIdArray<int64_t>({1, 2, 5, 3});
  auto dst = VecToIdArray<int64_t>({1, 6, 2, 6});
  auto mg1 = dgl::UnitGraph::CreateFromCOO(2, 9, 8, src, dst);
  src = VecToIdArray<int64_t>({6, 2, 5, 1, 8});
  dst = VecToIdArray<int64_t>({5, 2, 4, 8, 0});
  auto mg2 = dgl::UnitGraph::CreateFromCOO(1, 9, 9, src, dst);
  std::vector<HeteroGraphPtr> relgraphs;
  relgraphs.push_back(mg1);
  relgraphs.push_back(mg2);
  src = VecToIdArray<int64_t>({0, 0});
  dst = VecToIdArray<int64_t>({1, 0});
  auto meta_gptr = ImmutableGraph::CreateFromCOO(3, src, dst);
  auto hrptr = std::make_shared<HeteroGraph>(meta_gptr, relgraphs);

  std::string nonzerocopy_blob;
  dmlc::MemoryStringStream ifs(&nonzerocopy_blob);
  static_cast<dmlc::Stream *>(&ifs)->Write(hrptr);

  std::string zerocopy_blob;
  StringStreamWithBuffer zc_write_strm(&zerocopy_blob, true);
  static_cast<dmlc::Stream *>(&zc_write_strm)->Write(hrptr);

  LOG(INFO) << "Non zerocopy size: " << nonzerocopy_blob.size() << std::endl;
  LOG(INFO) << "Zerocopy size: " << zerocopy_blob.size() << std::endl;

  std::vector<void*> new_ptr_list;
  // Use memcpy to mimic remote machine reconstruction
  for (auto ptr : zc_write_strm.buffer_list()) {
    auto new_ptr = malloc(ptr.size);
    memcpy(new_ptr, ptr.data, ptr.size);
    new_ptr_list.emplace_back(new_ptr);
  }

  auto gptr = dgl::Serializer::make_shared<HeteroGraph>();
  StringStreamWithBuffer zc_read_strm(&zerocopy_blob, new_ptr_list);
  static_cast<dmlc::Stream *>(&zc_read_strm)->Read(&gptr);

  EXPECT_EQ(gptr->NumVertices(0), 9);
  EXPECT_EQ(gptr->NumVertices(1), 8);
}

#endif  // _WIN32