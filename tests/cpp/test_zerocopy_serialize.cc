#include <dgl/array.h>
#include <dgl/zerocopy_serializer.h>
#include <dmlc/memory_io.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <iostream>
#include <vector>

#include "./common.h"

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
    int32_t* data = static_cast<int32_t*>(arr->data);
    for (int64_t i = 0; i < len; ++i) {
      oss << data[i] << " ";
    }
  } else {
    int64_t* data = static_cast<int64_t*>(arr->data);
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
  std::vector<Ptr_pair> ptr_list;
  ZeroCopyStream zc_write_strm(&zerocopy_blob, &ptr_list);
  static_cast<dmlc::Stream *>(&zc_write_strm)->Write(tensor1);
  static_cast<dmlc::Stream *>(&zc_write_strm)->Write(tensor2);

  LOG(INFO) << nonzerocopy_blob.size() << std::endl;
  LOG(INFO) << zerocopy_blob.size() << std::endl;

  std::vector<Ptr_pair> new_ptr_list;
  // Use memcpy to mimic remote machine reconstruction
  for (Ptr_pair ptr : ptr_list) {
    auto new_ptr = malloc(ptr.second);
    memcpy(new_ptr, ptr.first, ptr.second);
    new_ptr_list.emplace_back(new_ptr, ptr.second);
  }

  NDArray loadtensor1, loadtensor2;
  ZeroCopyStream zc_read_strm(&zerocopy_blob, &new_ptr_list, false);
  static_cast<dmlc::Stream *>(&zc_read_strm)->Read(&loadtensor1);
  static_cast<dmlc::Stream *>(&zc_read_strm)->Read(&loadtensor2);

//   auto ptr_list = zc_strm;



  //   dmlc::MemoryStringStream ofs(&blob);
}


TEST(ZeroCopySerialize, SharedMem) {  
  auto tensor1 = VecToIdArray<int64_t>({1, 2, 5, 3});
  DLDataType dtype = {kDLInt, 64, 1};
  std::vector<int64_t> shape {4};
  DLContext cpu_ctx = {kDLCPU, 0};
  auto shared_tensor = NDArray::EmptyShared("test", shape, dtype, cpu_ctx, true);
  shared_tensor.CopyFrom(tensor1);

  std::string nonzerocopy_blob;
  dmlc::MemoryStringStream ifs(&nonzerocopy_blob);
  static_cast<dmlc::Stream *>(&ifs)->Write(shared_tensor);

  std::string zerocopy_blob;
  std::vector<Ptr_pair> ptr_list;
  ZeroCopyStream zc_write_strm(&zerocopy_blob, &ptr_list);
  static_cast<dmlc::Stream *>(&zc_write_strm)->Write(shared_tensor);

  LOG(INFO) << nonzerocopy_blob.size();
  LOG(INFO) << zerocopy_blob.size();

  NDArray loadtensor1, loadtensor2;
  ZeroCopyStream zc_read_strm(&zerocopy_blob, nullptr, true);
  static_cast<dmlc::Stream *>(&zc_read_strm)->Read(&loadtensor1);

}