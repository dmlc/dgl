/*!
 *  Copyright (c) 2020 by Contributors
 * \file tests/cpp/test_allocations.cc
 * \brief Test the three categories of allocated space on CPU and GPU.
 */

#include <gtest/gtest.h>
#include <dgl/runtime/device_api.h>
#include "./common.h"

namespace {
constexpr size_t ALIGNMENT = sizeof(void*);
}  // namespace

using namespace dgl;

template<typename T>
void _TestAllocatedSpace(
    DGLContext ctx,
    T * space,
    const size_t num) {
  auto device = runtime::DeviceAPI::Get(ctx);
  std::vector<T> source(num, 0);
  for (size_t i = 0; i < num; ++i) {
    source[i] = static_cast<T>(i);
  }

  std::vector<T> dest(num);

  const size_t space_size = num*sizeof(T);

  // make sure copy to the allocation works
  device->CopyDataFromTo(
      source.data(), 0, space, 0, space_size,
      DGLContext{kDLCPU, 0}, ctx, DGLType{}, 0); 

  device->CopyDataFromTo(
      space, 0,
      dest.data(), 0,
      space_size,
      ctx, DGLContext{kDLCPU, 0}, DGLType{}, 0); 

  for (size_t i = 0; i < num; ++i) {
    ASSERT_EQ(dest[i], source[i]);
  }
}


template<typename T>
void _TestWorkspace(DGLContext ctx) {
  auto device = runtime::DeviceAPI::Get(ctx);
  const size_t num = 8123;
  const size_t workspace_size = num*sizeof(T);
  T * workspace = static_cast<T*>(device->AllocWorkspace(ctx, workspace_size));

  _TestAllocatedSpace(ctx, workspace, num);

  device->FreeWorkspace(ctx, workspace);
}


template<typename T>
void _TestDataspace(DGLContext ctx) {
  auto device = runtime::DeviceAPI::Get(ctx);
  const size_t num = 8123;
  const size_t dataspace_size = num*sizeof(T);
  T * dataspace = static_cast<T*>(device->AllocDataSpace(ctx,
      dataspace_size, ALIGNMENT, {}));

  _TestAllocatedSpace(ctx, dataspace, num);

  device->FreeDataSpace(ctx, dataspace);
}

template<typename T>
void _TestRawDataspace(DGLContext ctx) {
  auto device = runtime::DeviceAPI::Get(ctx);
  const size_t num = 8123;
  const size_t dataspace_size = num*sizeof(T);
  T * dataspace = static_cast<T*>(device->AllocRawDataSpace(ctx,
      dataspace_size, ALIGNMENT, {}));

  _TestAllocatedSpace(ctx, dataspace, num);

  device->FreeRawDataSpace(ctx, dataspace);
}

TEST(AllocationsTest, CPUWorkspace) {
  _TestWorkspace<int32_t>(CPU);
  _TestWorkspace<int64_t>(CPU);
}


TEST(AllocationsTest, CPUDataspace) {
  _TestDataspace<int32_t>(CPU);
  _TestDataspace<int64_t>(CPU);
}


TEST(AllocationsTest, CPURaw) {
  _TestRawDataspace<int32_t>(CPU);
  _TestRawDataspace<int64_t>(CPU);
}

#ifdef DGL_USE_CUDA
TEST(AllocationsTest, GPUWorkspace) {
  _TestWorkspace<int32_t>(GPU);
  _TestWorkspace<int64_t>(GPU);
}

TEST(AllocationsTest, GPURaw) {
  _TestDataspace<int32_t>(GPU);
  _TestDataspace<int64_t>(GPU);
}

TEST(AllocationsTest, GPU) {
  _TestRawDataspace<int32_t>(GPU);
  _TestRawDataspace<int64_t>(GPU);
}
#endif
