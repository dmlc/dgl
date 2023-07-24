/*!
 *  Copyright (c) 2022 by Contributors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 * \file gpu_cache.cu
 * \brief Implementation of wrapper HugeCTR gpu_cache routines.
 */

#ifndef DGL_RUNTIME_CUDA_GPU_CACHE_H_
#define DGL_RUNTIME_CUDA_GPU_CACHE_H_

#include <cuda_runtime.h>
#include <dgl/array.h>
#include <dgl/aten/array_ops.h>
#include <dgl/packed_func_ext.h>
#include <dgl/runtime/container.h>
#include <dgl/runtime/device_api.h>
#include <dgl/runtime/object.h>
#include <dgl/runtime/registry.h>

#include <nv_gpu_cache.hpp>

#include "../../runtime/cuda/cuda_common.h"

namespace dgl {
namespace runtime {
namespace cuda {

template <typename key_t>
class GpuCache : public runtime::Object {
  constexpr static int set_associativity = 2;
  constexpr static int WARP_SIZE = 32;
  constexpr static int bucket_size = WARP_SIZE * set_associativity;
  using gpu_cache_t = gpu_cache::gpu_cache<
      key_t, uint64_t, std::numeric_limits<key_t>::max(), set_associativity,
      WARP_SIZE>;

 public:
  static constexpr const char *_type_key =
      sizeof(key_t) == 4 ? "cuda.GpuCache32" : "cuda.GpuCache64";
  DGL_DECLARE_OBJECT_TYPE_INFO(GpuCache, Object);

  GpuCache(size_t num_items, size_t num_feats)
      : num_feats(num_feats),
        cache(std::make_unique<gpu_cache_t>(
            (num_items + bucket_size - 1) / bucket_size, num_feats)) {
    CUDA_CALL(cudaGetDevice(&cuda_device));
  }

  std::tuple<NDArray, IdArray, IdArray> Query(IdArray keys) {
    const auto &ctx = keys->ctx;
    cudaStream_t stream = dgl::runtime::getCurrentCUDAStream();
    auto device = dgl::runtime::DeviceAPI::Get(ctx);
    CHECK_EQ(ctx.device_type, kDGLCUDA)
        << "The keys should be on a CUDA device";
    CHECK_EQ(ctx.device_id, cuda_device)
        << "The keys should be on the correct CUDA device";
    CHECK_EQ(keys->ndim, 1)
        << "The tensor of requested indices must be of dimension one.";
    NDArray values = NDArray::Empty(
        {keys->shape[0], (int64_t)num_feats}, DGLDataType{kDGLFloat, 32, 1},
        ctx);
    IdArray missing_index = aten::NewIdArray(keys->shape[0], ctx, 64);
    IdArray missing_keys =
        aten::NewIdArray(keys->shape[0], ctx, sizeof(key_t) * 8);
    size_t *missing_len =
        static_cast<size_t *>(device->AllocWorkspace(ctx, sizeof(size_t)));
    cache->Query(
        static_cast<const key_t *>(keys->data), keys->shape[0],
        static_cast<float *>(values->data),
        static_cast<uint64_t *>(missing_index->data),
        static_cast<key_t *>(missing_keys->data), missing_len, stream);
    size_t missing_len_host;
    device->CopyDataFromTo(
        missing_len, 0, &missing_len_host, 0, sizeof(missing_len_host), ctx,
        DGLContext{kDGLCPU, 0}, keys->dtype);
    device->FreeWorkspace(ctx, missing_len);
    missing_index = missing_index.CreateView(
        {(int64_t)missing_len_host}, missing_index->dtype);
    missing_keys =
        missing_keys.CreateView({(int64_t)missing_len_host}, keys->dtype);
    return std::make_tuple(values, missing_index, missing_keys);
  }

  void Replace(IdArray keys, NDArray values) {
    cudaStream_t stream = dgl::runtime::getCurrentCUDAStream();
    CHECK_EQ(keys->ctx.device_type, kDGLCUDA)
        << "The keys should be on a CUDA device";
    CHECK_EQ(keys->ctx.device_id, cuda_device)
        << "The keys should be on the correct CUDA device";
    CHECK_EQ(values->ctx.device_type, kDGLCUDA)
        << "The values should be on a CUDA device";
    CHECK_EQ(values->ctx.device_id, cuda_device)
        << "The values should be on the correct CUDA device";
    CHECK_EQ(keys->shape[0], values->shape[0])
        << "First dimensions of keys and values must match";
    CHECK_EQ(values->shape[1], num_feats) << "Embedding dimension must match";
    cache->Replace(
        static_cast<const key_t *>(keys->data), keys->shape[0],
        static_cast<const float *>(values->data), stream);
  }

 private:
  size_t num_feats;
  std::unique_ptr<gpu_cache_t> cache;
  int cuda_device;
};

static_assert(sizeof(unsigned int) == 4);
DGL_DEFINE_OBJECT_REF(GpuCacheRef32, GpuCache<unsigned int>);
// The cu file in HugeCTR gpu cache uses unsigned int and long long.
// Changing to int64_t results in a mismatch of template arguments.
static_assert(sizeof(long long) == 8);                      // NOLINT
DGL_DEFINE_OBJECT_REF(GpuCacheRef64, GpuCache<long long>);  // NOLINT

/* CAPI **********************************************************************/

using namespace dgl::runtime;

DGL_REGISTER_GLOBAL("cuda._CAPI_DGLGpuCacheCreate")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      const size_t num_items = args[0];
      const size_t num_feats = args[1];
      const int num_bits = args[2];

      if (num_bits == 32)
        *rv = GpuCacheRef32(
            std::make_shared<GpuCache<unsigned int>>(num_items, num_feats));
      else
        *rv = GpuCacheRef64(std::make_shared<GpuCache<long long>>(  // NOLINT
            num_items, num_feats));
    });

DGL_REGISTER_GLOBAL("cuda._CAPI_DGLGpuCacheQuery")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      IdArray keys = args[1];

      List<ObjectRef> ret;
      if (keys->dtype.bits == 32) {
        GpuCacheRef32 cache = args[0];
        auto result = cache->Query(keys);

        ret.push_back(Value(MakeValue(std::get<0>(result))));
        ret.push_back(Value(MakeValue(std::get<1>(result))));
        ret.push_back(Value(MakeValue(std::get<2>(result))));
      } else {
        GpuCacheRef64 cache = args[0];
        auto result = cache->Query(keys);

        ret.push_back(Value(MakeValue(std::get<0>(result))));
        ret.push_back(Value(MakeValue(std::get<1>(result))));
        ret.push_back(Value(MakeValue(std::get<2>(result))));
      }

      *rv = ret;
    });

DGL_REGISTER_GLOBAL("cuda._CAPI_DGLGpuCacheReplace")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      IdArray keys = args[1];
      NDArray values = args[2];

      if (keys->dtype.bits == 32) {
        GpuCacheRef32 cache = args[0];
        cache->Replace(keys, values);
      } else {
        GpuCacheRef64 cache = args[0];
        cache->Replace(keys, values);
      }

      *rv = List<ObjectRef>{};
    });

}  // namespace cuda
}  // namespace runtime
}  // namespace dgl

#endif
