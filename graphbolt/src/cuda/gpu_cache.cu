/**
 *  Copyright (c) 2023 by Contributors
 *  Copyright (c) 2023, GT-TDAlab (Muhammed Fatih Balin & Umit V. Catalyurek)
 * @file cuda/gpu_cache.cu
 * @brief GPUCache implementation on CUDA.
 */
#include <numeric>

#include "./common.h"
#include "./gpu_cache.h"

namespace graphbolt {
namespace cuda {

GpuCache::GpuCache(const std::vector<int64_t> &shape, torch::ScalarType dtype) {
  TORCH_CHECK(shape.size() >= 2, "Shape must at least have 2 dimensions.");
  const auto num_items = shape[0];
  const int64_t num_feats =
      std::accumulate(shape.begin() + 1, shape.end(), 1ll, std::multiplies<>());
  const int element_size =
      torch::empty(1, torch::TensorOptions().dtype(dtype)).element_size();
  num_bytes = num_feats * element_size;
  num_float_feats = (num_bytes + sizeof(float) - 1) / sizeof(float);
  cache = std::make_unique<gpu_cache_t>(
      (num_items + bucket_size - 1) / bucket_size, num_float_feats);
  this->shape = shape;
  this->shape[0] = -1;
  this->dtype = dtype;
  device_id = cuda::GetCurrentStream().device_index();
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> GpuCache::Query(
    torch::Tensor keys) {
  TORCH_CHECK(keys.device().is_cuda(), "Keys should be on a CUDA device.");
  TORCH_CHECK(
      keys.device().index() == device_id,
      "Keys should be on the correct CUDA device.");
  TORCH_CHECK(keys.sizes().size() == 1, "Keys should be a 1D tensor.");
  keys = keys.to(torch::kLong);
  auto values = torch::empty(
      {keys.size(0), num_float_feats}, keys.options().dtype(torch::kFloat));
  auto missing_index =
      torch::empty(keys.size(0), keys.options().dtype(torch::kLong));
  auto missing_keys =
      torch::empty(keys.size(0), keys.options().dtype(torch::kLong));
  cuda::CopyScalar<size_t> missing_len;
  auto stream = cuda::GetCurrentStream();
  cache->Query(
      reinterpret_cast<const key_t *>(keys.data_ptr()), keys.size(0),
      values.data_ptr<float>(),
      reinterpret_cast<uint64_t *>(missing_index.data_ptr()),
      reinterpret_cast<key_t *>(missing_keys.data_ptr()), missing_len.get(),
      stream);
  values =
      values.view(torch::kByte).slice(1, 0, num_bytes).view(dtype).view(shape);
  // To safely read missing_len, we synchronize
  stream.synchronize();
  missing_index = missing_index.slice(0, 0, static_cast<size_t>(missing_len));
  missing_keys = missing_keys.slice(0, 0, static_cast<size_t>(missing_len));
  return std::make_tuple(values, missing_index, missing_keys);
}

void GpuCache::Replace(torch::Tensor keys, torch::Tensor values) {
  TORCH_CHECK(keys.device().is_cuda(), "Keys should be on a CUDA device.");
  TORCH_CHECK(
      keys.device().index() == device_id,
      "Keys should be on the correct CUDA device.");
  TORCH_CHECK(values.device().is_cuda(), "Keys should be on a CUDA device.");
  TORCH_CHECK(
      values.device().index() == device_id,
      "Values should be on the correct CUDA device.");
  TORCH_CHECK(
      keys.size(0) == values.size(0),
      "The first dimensions of keys and values must match.");
  TORCH_CHECK(
      std::equal(shape.begin() + 1, shape.end(), values.sizes().begin() + 1),
      "Values should have the correct dimensions.");
  TORCH_CHECK(
      values.scalar_type() == dtype, "Values should have the correct dtype.");
  keys = keys.to(torch::kLong);
  torch::Tensor float_values;
  if (num_bytes % sizeof(float) != 0) {
    float_values = torch::empty(
        {values.size(0), num_float_feats},
        values.options().dtype(torch::kFloat));
    float_values.view(torch::kByte)
        .slice(1, 0, num_bytes)
        .copy_(values.view(torch::kByte).view({values.size(0), -1}));
  } else {
    float_values = values.view(torch::kByte)
                       .view({values.size(0), -1})
                       .view(torch::kFloat)
                       .contiguous();
  }
  cache->Replace(
      reinterpret_cast<const key_t *>(keys.data_ptr()), keys.size(0),
      float_values.data_ptr<float>(), cuda::GetCurrentStream());
}

c10::intrusive_ptr<GpuCache> GpuCache::Create(
    const std::vector<int64_t> &shape, torch::ScalarType dtype) {
  return c10::make_intrusive<GpuCache>(shape, dtype);
}

}  // namespace cuda
}  // namespace graphbolt
