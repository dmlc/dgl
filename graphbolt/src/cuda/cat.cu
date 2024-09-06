/**
 *   Copyright (c) 2023, GT-TDAlab (Muhammed Fatih Balin & Umit V. Catalyurek)
 *   All rights reserved.
 *
 *   Licensed under the Apache License, Version 2.0 (the "License");
 *   you may not use this file except in compliance with the License.
 *   You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *   Unless required by applicable law or agreed to in writing, software
 *   distributed under the License is distributed on an "AS IS" BASIS,
 *   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *   See the License for the specific language governing permissions and
 *   limitations under the License.
 *
 * @file cuda/expand_indptr.cu
 * @brief ExpandIndptr operator implementation on CUDA.
 */
#include <cub/cub.cuh>
#include <limits>

#include "./common.h"
#include "./expand_indptr.cuh"

namespace graphbolt {
namespace ops {

torch::Tensor CatImpl(const std::vector<torch::Tensor>& tensors) {
  const int64_t num_batches = tensors.size();
  const int64_t original_feature_size = std::accumulate(
      tensors.at(0).sizes().begin() + 1, tensors.at(0).sizes().end(),
      tensors.at(0).element_size(), std::multiplies<>());
  auto pointers_and_offsets = torch::empty(
      num_batches * 2 + 1,
      c10::TensorOptions().dtype(torch::kInt64).pinned_memory(true));
  auto pointers_ptr =
      reinterpret_cast<std::byte**>(pointers_and_offsets.data_ptr());
  auto offsets_ptr = pointers_and_offsets.data_ptr<int64_t>() + num_batches;
  int64_t i = 0;
  offsets_ptr[0] = 0;
  for (const auto& tensor : tensors) {
    pointers_ptr[i++] = reinterpret_cast<std::byte*>(tensor.data_ptr());
    offsets_ptr[i] =
        offsets_ptr[i - 1] + tensor.size(0) * original_feature_size;
  }
  auto pointers_and_offsets_dev = torch::empty_like(
      pointers_and_offsets,
      tensors[0].options().dtype(pointers_and_offsets.scalar_type()));
  CUDA_CALL(cudaMemcpyAsync(
      pointers_and_offsets_dev.data_ptr<int64_t>(), pointers_ptr,
      sizeof(int64_t) * pointers_and_offsets.numel(), cudaMemcpyHostToDevice,
      cuda::GetCurrentStream()));
  auto shape = tensors[0].sizes().vec();
  shape[0] = offsets_ptr[num_batches] / original_feature_size;
  auto output = torch::empty(shape, tensors[0].options());
  auto output_ptr = reinterpret_cast<std::byte*>(output.data_ptr());

  pointers_ptr =
      reinterpret_cast<std::byte**>(pointers_and_offsets_dev.data_ptr());
  offsets_ptr = pointers_and_offsets_dev.data_ptr<int64_t>() + num_batches;

  thrust::counting_iterator<int64_t> iota(0);
  auto output_buffer = thrust::make_transform_iterator(
      iota, OutputBufferIndexer<int64_t, std::byte>{offsets_ptr, output_ptr});
  auto buffer_sizes = thrust::make_transform_iterator(
      iota, AdjacentDifference<int64_t>{offsets_ptr});

  constexpr int64_t max_copy_at_once = std::numeric_limits<int32_t>::max();

  for (int64_t i = 0; i < num_batches; i += max_copy_at_once) {
    CUB_CALL(
        DeviceMemcpy::Batched, pointers_ptr + i, output_buffer + i,
        buffer_sizes + i, std::min(num_batches - i, max_copy_at_once));
  }
  return output;
}

}  // namespace ops
}  // namespace graphbolt
