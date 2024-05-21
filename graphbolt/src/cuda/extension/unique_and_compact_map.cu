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
 * @file cuda/unique_and_compact_map.cu
 * @brief Unique and compact operator implementation on CUDA using hash table.
 */
#include <graphbolt/cuda_ops.h>
#include <thrust/gather.h>

#include <cuco/static_map.cuh>
#include <cuda/std/atomic>
#include <numeric>

#include "../common.h"
#include "../utils.h"
#include "./unique_and_compact.h"

namespace graphbolt {
namespace ops {

// Support graphs with up to 2^kNodeIdBits nodes.
constexpr int kNodeIdBits = 40;

template <typename index_t, typename map_t>
__global__ void _InsertAndSetMinBatched(
    const int64_t num_edges, const int32_t* const indexes, index_t** pointers,
    const int64_t* const offsets, map_t map) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;

  while (i < num_edges) {
    const int64_t tensor_index = indexes[i];
    const auto tensor_offset = i - offsets[tensor_index];
    const int64_t node_id = pointers[tensor_index][tensor_offset];
    const auto batch_index = tensor_index / 2;
    const int64_t key = node_id | (batch_index << kNodeIdBits);

    auto [slot, is_new_key] = map.insert_and_find(cuco::pair{key, i});

    if (!is_new_key) {
      auto ref = ::cuda::atomic_ref<int64_t, ::cuda::thread_scope_device>{
          slot->second};
      ref.fetch_min(i, ::cuda::memory_order_relaxed);
    }

    i += stride;
  }
}

template <typename index_t, typename map_t>
__global__ void _IsInsertedBatched(
    const int64_t num_edges, const int32_t* const indexes, index_t** pointers,
    const int64_t* const offsets, map_t map, int64_t* valid) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;

  while (i < num_edges) {
    const int64_t tensor_index = indexes[i];
    const auto tensor_offset = i - offsets[tensor_index];
    const int64_t node_id = pointers[tensor_index][tensor_offset];
    const auto batch_index = tensor_index / 2;
    const int64_t key = node_id | (batch_index << kNodeIdBits);

    auto slot = map.find(key);
    valid[i] = slot->second == i;

    i += stride;
  }
}

template <typename index_t, typename map_t>
__global__ void _GetInsertedBatched(
    const int64_t num_edges, const int32_t* const indexes, index_t** pointers,
    const int64_t* const offsets, map_t map, const int64_t* const valid,
    index_t* unique_ids) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;

  while (i < num_edges) {
    const auto valid_i = valid[i];

    if (valid_i + 1 == valid[i + 1]) {
      const int64_t tensor_index = indexes[i];
      const auto tensor_offset = i - offsets[tensor_index];
      const int64_t node_id = pointers[tensor_index][tensor_offset];
      const auto batch_index = tensor_index / 2;
      const int64_t key = node_id | (batch_index << kNodeIdBits);

      auto slot = map.find(key);
      const auto batch_offset = offsets[batch_index * 2];
      const auto new_id = valid_i - valid[batch_offset];
      unique_ids[valid_i] = node_id;
      slot->second = new_id;
    }

    i += stride;
  }
}

template <typename index_t, typename map_t>
__global__ void _MapIdsBatched(
    const int num_batches, const int64_t num_edges,
    const int32_t* const indexes, index_t** pointers,
    const int64_t* const offsets, map_t map, index_t* mapped_ids) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;

  while (i < num_edges) {
    const int64_t tensor_index = indexes[i];
    int64_t batch_index;

    if (tensor_index >= 2 * num_batches) {
      batch_index = tensor_index - 2 * num_batches;
    } else if (tensor_index & 1) {
      batch_index = tensor_index / 2;
    } else {
      batch_index = -1;
    }

    // Only map src or dst ids.
    if (batch_index >= 0) {
      const auto tensor_offset = i - offsets[tensor_index];
      const int64_t node_id = pointers[tensor_index][tensor_offset];
      const int64_t key = node_id | (batch_index << kNodeIdBits);

      auto slot = map.find(key);
      mapped_ids[i] = slot->second;
    }

    i += stride;
  }
}

std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> >
UniqueAndCompactBatchedHashMapBased(
    const std::vector<torch::Tensor>& src_ids,
    const std::vector<torch::Tensor>& dst_ids,
    const std::vector<torch::Tensor>& unique_dst_ids) {
  auto allocator = cuda::GetAllocator();
  auto stream = cuda::GetCurrentStream();
  auto scalar_type = src_ids.at(0).scalar_type();
  constexpr int BLOCK_SIZE = 512;
  const auto num_batches = src_ids.size();
  static_assert(
      sizeof(std::ptrdiff_t) == sizeof(int64_t),
      "Need to be compiled on a 64-bit system.");
  constexpr int batch_id_bits = sizeof(int64_t) * 8 - 1 - kNodeIdBits;
  TORCH_CHECK(
      num_batches <= (1 << batch_id_bits),
      "UniqueAndCompactBatched supports a batch size of up to ",
      1 << batch_id_bits);
  return AT_DISPATCH_INDEX_TYPES(
      scalar_type, "unique_and_compact", ([&] {
        // For 2 batches of inputs, stores the input tensor pointers in the
        // unique_dst, src, unique_dst, src, dst, dst order. Since there are
        // 3 * num_batches input tensors, we need the first 3 * num_batches to
        // store the input tensor pointers. Then, we store offsets in the rest
        // of the 3 * num_batches + 1 space as if they were stored contiguously.
        auto pointers_and_offsets = torch::empty(
            6 * num_batches + 1,
            c10::TensorOptions().dtype(torch::kInt64).pinned_memory(true));
        // Points to the input tensor pointers.
        auto pointers_ptr =
            reinterpret_cast<index_t**>(pointers_and_offsets.data_ptr());
        // Points to the input tensor storage logical offsets.
        auto offsets_ptr =
            pointers_and_offsets.data_ptr<int64_t>() + 3 * num_batches;
        for (std::size_t i = 0; i < num_batches; i++) {
          pointers_ptr[2 * i] = unique_dst_ids.at(i).data_ptr<index_t>();
          offsets_ptr[2 * i] = unique_dst_ids[i].size(0);
          pointers_ptr[2 * i + 1] = src_ids.at(i).data_ptr<index_t>();
          offsets_ptr[2 * i + 1] = src_ids[i].size(0);
          pointers_ptr[2 * num_batches + i] = dst_ids.at(i).data_ptr<index_t>();
          offsets_ptr[2 * num_batches + i] = dst_ids[i].size(0);
        }
        // Finish computing the offsets by taking a cumulative sum.
        std::exclusive_scan(
            offsets_ptr, offsets_ptr + 3 * num_batches + 1, offsets_ptr, 0ll);
        // Device version of the tensors defined above. We store the information
        // initially on the CPU, which are later copied to the device.
        auto pointers_and_offsets_dev = torch::empty(
            pointers_and_offsets.size(0),
            src_ids[0].options().dtype(pointers_and_offsets.scalar_type()));
        auto offsets_dev = pointers_and_offsets_dev.slice(0, 3 * num_batches);
        auto pointers_dev_ptr =
            reinterpret_cast<index_t**>(pointers_and_offsets_dev.data_ptr());
        auto offsets_dev_ptr = offsets_dev.data_ptr<int64_t>();
        CUDA_CALL(cudaMemcpyAsync(
            pointers_dev_ptr, pointers_ptr,
            sizeof(int64_t) * pointers_and_offsets.size(0),
            cudaMemcpyHostToDevice, stream));
        auto indexes = ExpandIndptrImpl(
            offsets_dev, torch::kInt32, torch::nullopt,
            offsets_ptr[3 * num_batches]);
        cuco::static_map map{
            offsets_ptr[2 * num_batches],
            0.5,  // load_factor
            cuco::empty_key{static_cast<int64_t>(-1)},
            cuco::empty_value{static_cast<int64_t>(-1)},
            {},
            cuco::linear_probing<1, cuco::default_hash_function<int64_t> >{},
            {},
            {},
            cuda::CUDAWorkspaceAllocator<cuco::pair<int64_t, int64_t> >{},
            cuco::cuda_stream_ref{stream},
        };
        C10_CUDA_KERNEL_LAUNCH_CHECK();  // Check the map constructor's success.
        const dim3 block(BLOCK_SIZE);
        const dim3 grid(
            (offsets_ptr[2 * num_batches] + BLOCK_SIZE - 1) / BLOCK_SIZE);
        CUDA_KERNEL_CALL(
            _InsertAndSetMinBatched, grid, block, 0,
            offsets_ptr[2 * num_batches], indexes.data_ptr<int32_t>(),
            pointers_dev_ptr, offsets_dev_ptr, map.ref(cuco::insert_and_find));
        auto valid = torch::empty(
            offsets_ptr[2 * num_batches] + 1,
            src_ids[0].options().dtype(torch::kInt64));
        CUDA_KERNEL_CALL(
            _IsInsertedBatched, grid, block, 0, offsets_ptr[2 * num_batches],
            indexes.data_ptr<int32_t>(), pointers_dev_ptr, offsets_dev_ptr,
            map.ref(cuco::find), valid.data_ptr<int64_t>());
        valid = ExclusiveCumSum(valid);
        auto unique_ids_offsets = torch::empty(
            num_batches + 1,
            c10::TensorOptions().dtype(torch::kInt64).pinned_memory(true));
        auto unique_ids_offsets_ptr = unique_ids_offsets.data_ptr<int64_t>();
        for (int64_t i = 0; i <= num_batches; i++) {
          unique_ids_offsets_ptr[i] = offsets_ptr[2 * i];
        }
        THRUST_CALL(
            gather, unique_ids_offsets_ptr,
            unique_ids_offsets_ptr + unique_ids_offsets.size(0),
            valid.data_ptr<int64_t>(), unique_ids_offsets_ptr);
        at::cuda::CUDAEvent unique_ids_offsets_event;
        unique_ids_offsets_event.record();
        auto unique_ids =
            torch::empty(offsets_ptr[2 * num_batches], src_ids[0].options());
        CUDA_KERNEL_CALL(
            _GetInsertedBatched, grid, block, 0, offsets_ptr[2 * num_batches],
            indexes.data_ptr<int32_t>(), pointers_dev_ptr, offsets_dev_ptr,
            map.ref(cuco::find), valid.data_ptr<int64_t>(),
            unique_ids.data_ptr<index_t>());
        auto mapped_ids =
            torch::empty(offsets_ptr[3 * num_batches], unique_ids.options());
        CUDA_KERNEL_CALL(
            _MapIdsBatched, grid, block, 0, num_batches,
            offsets_ptr[3 * num_batches], indexes.data_ptr<int32_t>(),
            pointers_dev_ptr, offsets_dev_ptr, map.ref(cuco::find),
            mapped_ids.data_ptr<index_t>());
        std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> >
            results;
        unique_ids_offsets_event.synchronize();
        for (int64_t i = 0; i < num_batches; i++) {
          results.emplace_back(
              unique_ids.slice(
                  0, unique_ids_offsets_ptr[i], unique_ids_offsets_ptr[i + 1]),
              mapped_ids.slice(
                  0, offsets_ptr[2 * i + 1], offsets_ptr[2 * i + 2]),
              mapped_ids.slice(
                  0, offsets_ptr[2 * num_batches + i],
                  offsets_ptr[2 * num_batches + i + 1]));
        }
        return results;
      }));
}

}  // namespace ops
}  // namespace graphbolt
