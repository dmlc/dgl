/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cooperative_groups.h>
#include <cuda_runtime_api.h>
#include <immintrin.h>

#include <atomic>
#include <iostream>
#include <limits>
#include <mutex>
#include <uvm_table.hpp>

namespace cg = cooperative_groups;

namespace {

constexpr int set_size = 4;
constexpr int block_size = 256;

template <typename key_type>
__host__ __device__ key_type hash(key_type key) {
  return key;
}

template <typename key_type>
__global__ void hash_add_kernel(const key_type* new_keys, const int num_keys, key_type* keys,
                                const int num_sets, int* set_sizes, const int max_set_size,
                                key_type* missing_keys, int* num_missing_keys) {
  __shared__ key_type s_missing_keys[block_size];
  __shared__ int s_missing_count;
  __shared__ size_t s_missing_idx;

  auto grid = cg::this_grid();
  auto block = cg::this_thread_block();

  if (block.thread_rank() == 0) {
    s_missing_count = 0;
  }
  block.sync();

  size_t idx = grid.thread_rank();
  if (idx < num_keys) {
    auto key = new_keys[idx];
    size_t idx_set = hash(key) % num_sets;
    int prev_set_size = atomicAdd(&set_sizes[idx_set], 1);
    if (prev_set_size < max_set_size) {
      keys[idx_set * max_set_size + prev_set_size] = key;
    } else {
      int count = atomicAdd(&s_missing_count, 1);
      s_missing_keys[count] = key;
    }
  }

  block.sync();
  if (block.thread_rank() == 0) {
    s_missing_idx = atomicAdd(num_missing_keys, s_missing_count);
  }
  block.sync();
  for (size_t i = block.thread_rank(); i < s_missing_count; i += block.num_threads()) {
    missing_keys[s_missing_idx + i] = s_missing_keys[i];
  }
}

template <typename key_type, typename index_type>
__global__ void hash_query_kernel(const key_type* query_keys, int* num_keys_ptr,
                                  const key_type* keys, const size_t num_sets,
                                  const int max_set_size, index_type* output_indices) {
  constexpr int tile_size = set_size;
  auto grid = cg::this_grid();
  auto block = cg::this_thread_block();
  auto tile = cg::tiled_partition<tile_size>(block);
  int num_keys = *num_keys_ptr;
  if (num_keys == 0) return;

#if (CUDA_VERSION < 11060)
  size_t num_threads_per_grid = grid.size();
#else
  size_t num_threads_per_grid = grid.num_threads();
#endif

  size_t step = (num_keys - 1) / num_threads_per_grid + 1;
  for (size_t i = 0; i < step; i++) {
    size_t idx = i * num_threads_per_grid + grid.thread_rank();
    key_type query_key = std::numeric_limits<key_type>::max();
    if (idx < num_keys) {
      query_key = query_keys[idx];
    }
    auto idx_set = hash(query_key) % num_sets;
    for (int j = 0; j < tile_size; j++) {
      auto current_idx_set = tile.shfl(idx_set, j);
      auto current_query_key = tile.shfl(query_key, j);
      if (current_query_key == std::numeric_limits<key_type>::max()) {
        continue;
      }
      auto candidate_key = keys[current_idx_set * set_size + tile.thread_rank()];
      int existed = tile.ballot(current_query_key == candidate_key);
      auto current_idx = tile.shfl(idx, 0) + j;
      if (existed) {
        int src_lane = __ffs(existed) - 1;
        size_t found_idx = current_idx_set * set_size + src_lane;
        output_indices[current_idx] = num_sets * src_lane + current_idx_set;
      } else {
        output_indices[current_idx] = std::numeric_limits<index_type>::max();
      }
    }
  }
}

template <typename key_type, typename index_type>
__global__ void hash_query_kernel(const key_type* query_keys, const int num_keys,
                                  const key_type* keys, const size_t num_sets,
                                  const int max_set_size, index_type* output_indices,
                                  key_type* missing_keys, int* missing_positions,
                                  int* missing_count) {
  __shared__ key_type s_missing_keys[block_size];
  __shared__ key_type s_missing_positions[block_size];
  __shared__ int s_missing_count;
  __shared__ int s_missing_idx;

  constexpr int tile_size = set_size;

  auto grid = cg::this_grid();
  auto block = cg::this_thread_block();
  auto tile = cg::tiled_partition<tile_size>(block);

  if (block.thread_rank() == 0) {
    s_missing_count = 0;
  }
  block.sync();

  size_t idx = grid.thread_rank();
  key_type query_key = std::numeric_limits<key_type>::max();
  if (idx < num_keys) {
    query_key = query_keys[idx];
  }
  auto idx_set = hash(query_key) % num_sets;

  for (int j = 0; j < tile_size; j++) {
    auto current_idx_set = tile.shfl(idx_set, j);
    auto current_query_key = tile.shfl(query_key, j);
    if (current_query_key == std::numeric_limits<key_type>::max()) {
      continue;
    }
    auto candidate_key = keys[current_idx_set * set_size + tile.thread_rank()];
    int existed = tile.ballot(current_query_key == candidate_key);
    if (existed) {
      int src_lane = __ffs(existed) - 1;
      size_t found_idx = current_idx_set * set_size + src_lane;
      output_indices[tile.shfl(idx, 0) + j] = num_sets * src_lane + current_idx_set;
    } else {
      auto current_idx = tile.shfl(idx, 0) + j;
      output_indices[current_idx] = std::numeric_limits<index_type>::max();
      if (tile.thread_rank() == 0) {
        int s_count = atomicAdd(&s_missing_count, 1);
        s_missing_keys[s_count] = current_query_key;
        s_missing_positions[s_count] = current_idx;
      }
    }
  }

  if (missing_keys == nullptr) {
    if (grid.thread_rank() == 0 && missing_count) {
      *missing_count = 0;
    }
    return;
  }
  block.sync();
  if (block.thread_rank() == 0) {
    s_missing_idx = atomicAdd(missing_count, s_missing_count);
  }
  block.sync();
  for (size_t i = block.thread_rank(); i < s_missing_count; i += block.num_threads()) {
    missing_keys[s_missing_idx + i] = s_missing_keys[i];
    missing_positions[s_missing_idx + i] = s_missing_positions[i];
  }
}

template <int warp_size>
__forceinline__ __device__ void warp_tile_copy(const size_t lane_idx,
                                               const size_t emb_vec_size_in_float,
                                               volatile float* d_dst, const float* d_src) {
  // 16 bytes align
  if (emb_vec_size_in_float % 4 != 0 || (size_t)d_dst % 16 != 0 || (size_t)d_src % 16 != 0) {
#pragma unroll
    for (size_t i = lane_idx; i < emb_vec_size_in_float; i += warp_size) {
      d_dst[i] = d_src[i];
    }
  } else {
#pragma unroll
    for (size_t i = lane_idx; i < emb_vec_size_in_float / 4; i += warp_size) {
      *(float4*)(d_dst + i * 4) = __ldg((const float4*)(d_src + i * 4));
    }
  }
}

template <typename index_type, typename vec_type>
__global__ void read_vectors_kernel(const index_type* query_indices, const int num_keys,
                                    const vec_type* vectors, const int vec_size,
                                    vec_type* output_vectors) {
  constexpr int warp_size = 32;

  auto grid = cg::this_grid();
  auto block = cg::this_thread_block();
  auto tile = cg::tiled_partition<warp_size>(block);

#if (CUDA_VERSION < 11060)
  auto num_threads_per_grid = grid.size();
#else
  auto num_threads_per_grid = grid.num_threads();
#endif

  for (int step = 0; step < (num_keys - 1) / num_threads_per_grid + 1; step++) {
    int key_num = step * num_threads_per_grid + grid.thread_rank();
    index_type idx = std::numeric_limits<index_type>::max();
    if (key_num < num_keys) {
      idx = query_indices[key_num];
    }
#pragma unroll 4
    for (size_t j = 0; j < warp_size; j++) {
      index_type current_idx = tile.shfl(idx, j);
      index_type idx_write = tile.shfl(key_num, 0) + j;
      if (current_idx == std::numeric_limits<index_type>::max()) continue;
      warp_tile_copy<warp_size>(tile.thread_rank(), vec_size, output_vectors + idx_write * vec_size,
                                vectors + current_idx * vec_size);
    }
  }
}

template <typename index_type, typename vec_type>
__global__ void distribute_vectors_kernel(const index_type* postions, const size_t num_keys,
                                          const vec_type* vectors, const int vec_size,
                                          vec_type* output_vectors) {
  constexpr int warp_size = 32;

  auto grid = cg::this_grid();
  auto block = cg::this_thread_block();
  auto tile = cg::tiled_partition<warp_size>(block);

#if (CUDA_VERSION < 11060)
  auto num_threads_per_grid = grid.size();
#else
  auto num_threads_per_grid = grid.num_threads();
#endif

  for (size_t step = 0; step < (num_keys - 1) / num_threads_per_grid + 1; step++) {
    size_t key_num = step * num_threads_per_grid + grid.thread_rank();
    index_type idx = std::numeric_limits<index_type>::max();
    if (key_num < num_keys) {
      idx = postions[key_num];
    }
#pragma unroll 4
    for (size_t j = 0; j < warp_size; j++) {
      size_t idx_write = tile.shfl(idx, j);
      size_t idx_read = tile.shfl(key_num, 0) + j;
      if (idx_write == std::numeric_limits<index_type>::max()) continue;
      warp_tile_copy<warp_size>(tile.thread_rank(), vec_size,
                                output_vectors + (size_t)idx_write * vec_size,
                                vectors + (size_t)idx_read * vec_size);
    }
  }
}

}  // namespace

namespace gpu_cache {
template <typename key_type, typename index_type, typename vec_type>
UvmTable<key_type, index_type, vec_type>::UvmTable(const size_t device_table_capacity,
                                                   const size_t host_table_capacity,
                                                   const int max_batch_size, const int vec_size,
                                                   const vec_type default_value)
    : max_batch_size_(std::max(100000, max_batch_size)),
      vec_size_(vec_size),
      num_set_((device_table_capacity - 1) / set_size + 1),
      num_host_set_((host_table_capacity - 1) / set_size + 1),
      table_capacity_(num_set_ * set_size),
      default_vector_(vec_size, default_value),
      device_table_(device_table_capacity, set_size, max_batch_size_),
      host_table_(host_table_capacity * 1.3, set_size, max_batch_size_) {
  CUDA_CHECK(cudaMalloc(&d_keys_buffer_, sizeof(key_type) * max_batch_size_));
  CUDA_CHECK(cudaMalloc(&d_vectors_buffer_, sizeof(vec_type) * max_batch_size_ * vec_size_));
  CUDA_CHECK(cudaMalloc(&d_vectors_, sizeof(vec_type) * device_table_.capacity * vec_size_));

  CUDA_CHECK(cudaMalloc(&d_output_indices_, sizeof(index_type) * max_batch_size_));
  CUDA_CHECK(cudaMalloc(&d_output_host_indices_, sizeof(index_type) * max_batch_size_));
  CUDA_CHECK(cudaMallocHost(&h_output_host_indices_, sizeof(index_type) * max_batch_size_));
  CUDA_CHECK(cudaMalloc(&d_missing_keys_, sizeof(key_type) * max_batch_size_));
  CUDA_CHECK(cudaMalloc(&d_missing_positions_, sizeof(int) * max_batch_size_));
  CUDA_CHECK(cudaMalloc(&d_missing_count_, sizeof(int)));
  CUDA_CHECK(cudaMemset(d_missing_count_, 0, sizeof(int)));
  CUDA_CHECK(cudaStreamCreate(&query_stream_));
  for (int i = 0; i < num_buffers_; i++) {
    int batch_size_per_buffer = ceil(1.0 * max_batch_size_ / num_buffers_);
    CUDA_CHECK(
        cudaMallocHost(&h_cpy_buffers_[i], sizeof(vec_type) * batch_size_per_buffer * vec_size));
    CUDA_CHECK(cudaMalloc(&d_cpy_buffers_[i], sizeof(vec_type) * batch_size_per_buffer * vec_size));
    CUDA_CHECK(cudaStreamCreate(&cpy_streams_[i]));
    CUDA_CHECK(cudaEventCreate(&cpy_events_[i]));
  }
  CUDA_CHECK(cudaMallocHost(&h_missing_keys_, sizeof(key_type) * max_batch_size_));
  CUDA_CHECK(cudaEventCreate(&query_event_));
  h_vectors_.resize(host_table_.capacity * vec_size_);
}

template <typename key_type, typename index_type, typename vec_type>
void UvmTable<key_type, index_type, vec_type>::add(const key_type* h_keys,
                                                   const vec_type* h_vectors,
                                                   const size_t num_keys) {
  std::vector<key_type> h_missing_keys;
  size_t num_batches = (num_keys - 1) / max_batch_size_ + 1;
  for (size_t i = 0; i < num_batches; i++) {
    size_t this_batch_size =
        i != num_batches - 1 ? max_batch_size_ : num_keys - i * max_batch_size_;
    CUDA_CHECK(cudaMemcpy(d_keys_buffer_, h_keys + i * max_batch_size_,
                          sizeof(*d_keys_buffer_) * this_batch_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_missing_count_, 0, sizeof(*d_missing_count_)));
    device_table_.add(d_keys_buffer_, this_batch_size, d_missing_keys_, d_missing_count_, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    int num_missing_keys;
    CUDA_CHECK(cudaMemcpy(&num_missing_keys, d_missing_count_, sizeof(num_missing_keys),
                          cudaMemcpyDeviceToHost));
    size_t prev_size = h_missing_keys.size();
    h_missing_keys.resize(prev_size + num_missing_keys);
    CUDA_CHECK(cudaMemcpy(h_missing_keys.data() + prev_size, d_missing_keys_,
                          sizeof(*d_missing_keys_) * num_missing_keys, cudaMemcpyDeviceToHost));
  }

  std::vector<key_type> h_final_missing_keys;
  num_batches = h_missing_keys.size() ? (h_missing_keys.size() - 1) / max_batch_size_ + 1 : 0;
  for (size_t i = 0; i < num_batches; i++) {
    size_t this_batch_size =
        i != num_batches - 1 ? max_batch_size_ : h_missing_keys.size() - i * max_batch_size_;
    CUDA_CHECK(cudaMemcpy(d_keys_buffer_, h_missing_keys.data() + i * max_batch_size_,
                          sizeof(*d_keys_buffer_) * this_batch_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_missing_count_, 0, sizeof(*d_missing_count_)));
    host_table_.add(d_keys_buffer_, this_batch_size, d_missing_keys_, d_missing_count_, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    int num_missing_keys;
    CUDA_CHECK(cudaMemcpy(&num_missing_keys, d_missing_count_, sizeof(num_missing_keys),
                          cudaMemcpyDeviceToHost));
    size_t prev_size = h_final_missing_keys.size();
    h_final_missing_keys.resize(prev_size + num_missing_keys);
    CUDA_CHECK(cudaMemcpy(h_final_missing_keys.data() + prev_size, d_missing_keys_,
                          sizeof(*d_missing_keys_) * num_missing_keys, cudaMemcpyDeviceToHost));
  }

  std::vector<key_type> h_keys_buffer(max_batch_size_);
  std::vector<index_type> h_indices_buffer(max_batch_size_);
  std::vector<int> h_positions_buffer(max_batch_size_);

  num_batches = (num_keys - 1) / max_batch_size_ + 1;

  size_t num_hit_keys = 0;
  for (size_t i = 0; i < num_batches; i++) {
    size_t this_batch_size =
        i != num_batches - 1 ? max_batch_size_ : num_keys - i * max_batch_size_;
    CUDA_CHECK(cudaMemcpy(d_keys_buffer_, h_keys + i * max_batch_size_,
                          sizeof(*d_keys_buffer_) * this_batch_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_missing_count_, 0, sizeof(*d_missing_count_)));
    device_table_.query(d_keys_buffer_, this_batch_size, d_output_indices_, d_missing_keys_,
                        d_missing_positions_, d_missing_count_, 0);
    CUDA_CHECK(cudaStreamSynchronize(0));

    CUDA_CHECK(cudaMemcpy(d_vectors_buffer_, h_vectors + i * max_batch_size_ * vec_size_,
                          sizeof(*d_vectors_) * this_batch_size * vec_size_,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaStreamSynchronize(0));
    if (num_hit_keys < device_table_.capacity) {
      distribute_vectors_kernel<<<(this_batch_size - 1) / block_size + 1, block_size, 0, 0>>>(
          d_output_indices_, this_batch_size, d_vectors_buffer_, vec_size_, d_vectors_);
      CUDA_CHECK(cudaStreamSynchronize(0));
    }

    int num_missing_keys;
    CUDA_CHECK(cudaMemcpy(&num_missing_keys, d_missing_count_, sizeof(num_missing_keys),
                          cudaMemcpyDeviceToHost));
    num_hit_keys += this_batch_size - num_missing_keys;
    host_table_.query(d_missing_keys_, num_missing_keys, d_output_indices_, nullptr, nullptr,
                      nullptr, 0);

    CUDA_CHECK(cudaMemcpy(h_keys_buffer.data(), d_missing_keys_,
                          sizeof(*d_missing_keys_) * num_missing_keys, cudaMemcpyDeviceToHost))

    CUDA_CHECK(cudaMemcpy(h_indices_buffer.data(), d_output_indices_,
                          sizeof(*d_output_indices_) * num_missing_keys, cudaMemcpyDeviceToHost))

    CUDA_CHECK(cudaMemcpy(h_positions_buffer.data(), d_missing_positions_,
                          sizeof(*d_missing_positions_) * num_missing_keys, cudaMemcpyDeviceToHost))

    for (int j = 0; j < num_missing_keys; j++) {
      if (h_indices_buffer[j] != std::numeric_limits<index_type>::max()) {
        memcpy(h_vectors_.data() + h_indices_buffer[j] * vec_size_,
               h_vectors + (i * max_batch_size_ + h_positions_buffer[j]) * vec_size_,
               sizeof(*h_vectors) * vec_size_);
      } else {
        size_t prev_idx = h_vectors_.size() / vec_size_;
        h_final_missing_items_.emplace(h_keys_buffer[j], prev_idx);
        h_vectors_.resize(h_vectors_.size() + vec_size_);
        memcpy(h_vectors_.data() + prev_idx * vec_size_,
               h_vectors + (i * max_batch_size_ + h_positions_buffer[j]) * vec_size_,
               sizeof(*h_vectors) * vec_size_);
      }
    }
  }
  CUDA_CHECK(cudaMemset(d_missing_count_, 0, sizeof(*d_missing_count_)));
}

template <typename key_type, typename index_type, typename vec_type>
void UvmTable<key_type, index_type, vec_type>::query(const key_type* d_keys, const int num_keys,
                                                     vec_type* d_vectors, cudaStream_t stream) {
  if (!num_keys) return;
  CUDA_CHECK(cudaEventRecord(query_event_, stream));
  CUDA_CHECK(cudaStreamWaitEvent(query_stream_, query_event_));

  static_assert(num_buffers_ >= 2);
  device_table_.query(d_keys, num_keys, d_output_indices_, d_missing_keys_, d_missing_positions_,
                      d_missing_count_, query_stream_);

  CUDA_CHECK(cudaEventRecord(query_event_, query_stream_));
  CUDA_CHECK(cudaStreamWaitEvent(cpy_streams_[0], query_event_));

  int num_missing_keys;
  CUDA_CHECK(cudaMemcpyAsync(&num_missing_keys, d_missing_count_, sizeof(*d_missing_count_),
                             cudaMemcpyDeviceToHost, cpy_streams_[0]));

  host_table_.query(d_missing_keys_, d_missing_count_, d_output_host_indices_, query_stream_);
  CUDA_CHECK(cudaStreamSynchronize(cpy_streams_[0]));

  CUDA_CHECK(cudaMemsetAsync(d_missing_count_, 0, sizeof(*d_missing_count_), query_stream_));

  CUDA_CHECK(cudaMemcpyAsync(h_output_host_indices_, d_output_host_indices_,
                             sizeof(index_type) * num_missing_keys, cudaMemcpyDeviceToHost,
                             query_stream_));

  CUDA_CHECK(cudaMemcpyAsync(h_missing_keys_, d_missing_keys_, sizeof(key_type) * num_missing_keys,
                             cudaMemcpyDeviceToHost, cpy_streams_[0]));

  read_vectors_kernel<<<(num_keys - 1) / block_size + 1, block_size, 0, cpy_streams_[1]>>>(
      d_output_indices_, num_keys, d_vectors_, vec_size_, d_vectors);

  CUDA_CHECK(cudaStreamSynchronize(query_stream_));
  CUDA_CHECK(cudaStreamSynchronize(cpy_streams_[0]));

  int num_keys_per_buffer = ceil(1.0 * num_missing_keys / num_buffers_);

  for (int buffer_num = 0; buffer_num < num_buffers_; buffer_num++) {
    int num_keys_this_buffer = buffer_num != num_buffers_ - 1
                                   ? num_keys_per_buffer
                                   : num_missing_keys - num_keys_per_buffer * buffer_num;
    if (!num_keys_this_buffer) break;
#pragma omp parallel for num_threads(8)
    for (size_t i = 0; i < static_cast<size_t>(num_keys_this_buffer); i++) {
      size_t idx_key = buffer_num * num_keys_per_buffer + i;
      index_type index = h_output_host_indices_[idx_key];
      if (index == std::numeric_limits<index_type>::max()) {
        key_type key = h_missing_keys_[idx_key];
        auto iterator = h_final_missing_items_.find(key);
        if (iterator != h_final_missing_items_.end()) {
          index = iterator->second;
        }
      }
      if (index != std::numeric_limits<index_type>::max()) {
        memcpy(h_cpy_buffers_[buffer_num] + i * vec_size_, h_vectors_.data() + index * vec_size_,
               sizeof(vec_type) * vec_size_);
      } else {
        memcpy(h_cpy_buffers_[buffer_num] + i * vec_size_, default_vector_.data(),
               sizeof(vec_type) * vec_size_);
      }
    }
    CUDA_CHECK(cudaMemcpyAsync(d_cpy_buffers_[buffer_num], h_cpy_buffers_[buffer_num],
                               sizeof(vec_type) * num_keys_this_buffer * vec_size_,
                               cudaMemcpyHostToDevice, cpy_streams_[buffer_num]));

    distribute_vectors_kernel<<<(num_keys_this_buffer - 1) / block_size + 1, block_size, 0,
                                cpy_streams_[buffer_num]>>>(
        d_missing_positions_ + buffer_num * num_keys_per_buffer, num_keys_this_buffer,
        d_cpy_buffers_[buffer_num], vec_size_, d_vectors);
  }

  for (int i = 0; i < num_buffers_; i++) {
    CUDA_CHECK(cudaEventRecord(cpy_events_[i], cpy_streams_[i]));
    CUDA_CHECK(cudaStreamWaitEvent(stream, cpy_events_[i]));
  }
}

template <typename key_type, typename index_type, typename vec_type>
void UvmTable<key_type, index_type, vec_type>::clear(cudaStream_t stream) {
  device_table_.clear(stream);
  host_table_.clear(stream);
}

template <typename key_type, typename index_type, typename vec_type>
UvmTable<key_type, index_type, vec_type>::~UvmTable() {
  CUDA_CHECK(cudaFree(d_keys_buffer_));
  CUDA_CHECK(cudaFree(d_vectors_buffer_));
  CUDA_CHECK(cudaFree(d_vectors_));

  CUDA_CHECK(cudaFree(d_output_indices_));
  CUDA_CHECK(cudaFree(d_output_host_indices_));
  CUDA_CHECK(cudaFreeHost(h_output_host_indices_));

  CUDA_CHECK(cudaFree(d_missing_keys_));
  CUDA_CHECK(cudaFree(d_missing_positions_));
  CUDA_CHECK(cudaFree(d_missing_count_));
  CUDA_CHECK(cudaFreeHost(h_missing_keys_));

  CUDA_CHECK(cudaStreamDestroy(query_stream_));
  CUDA_CHECK(cudaEventDestroy(query_event_));

  for (int i = 0; i < num_buffers_; i++) {
    CUDA_CHECK(cudaFreeHost(h_cpy_buffers_[i]));
    CUDA_CHECK(cudaFree(d_cpy_buffers_[i]));
    CUDA_CHECK(cudaStreamDestroy(cpy_streams_[i]));
    CUDA_CHECK(cudaEventDestroy(cpy_events_[i]));
  }
}

template <typename key_type, typename index_type>
HashBlock<key_type, index_type>::HashBlock(size_t expected_capacity, int set_size, int batch_size)
    : max_set_size_(set_size), batch_size_(batch_size) {
  if (expected_capacity) {
    num_sets = (expected_capacity - 1) / set_size + 1;
  } else {
    num_sets = 10000;
  }
  capacity = num_sets * set_size;
  CUDA_CHECK(cudaMalloc(&keys, sizeof(*keys) * capacity));
  CUDA_CHECK(cudaMalloc(&set_sizes_, sizeof(*set_sizes_) * num_sets));
  CUDA_CHECK(cudaMemset(set_sizes_, 0, sizeof(*set_sizes_) * num_sets));
}

template <typename key_type, typename index_type>
HashBlock<key_type, index_type>::~HashBlock() {
  CUDA_CHECK(cudaFree(keys));
  CUDA_CHECK(cudaFree(set_sizes_));
}

template <typename key_type, typename index_type>
void HashBlock<key_type, index_type>::query(const key_type* query_keys, const size_t num_keys,
                                            index_type* output_indices, key_type* missing_keys,
                                            int* missing_positions, int* num_missing_keys,
                                            cudaStream_t stream) {
  if (num_keys == 0) {
    return;
  }
  size_t num_batches = (num_keys - 1) / batch_size_ + 1;
  for (size_t i = 0; i < num_batches; i++) {
    size_t this_batch_size = i != num_batches - 1 ? batch_size_ : num_keys - i * batch_size_;
    hash_query_kernel<<<(this_batch_size - 1) / block_size + 1, block_size, 0, stream>>>(
        query_keys, this_batch_size, keys, num_sets, max_set_size_, output_indices, missing_keys,
        missing_positions, num_missing_keys);
  }
}

template <typename key_type, typename index_type>
void HashBlock<key_type, index_type>::query(const key_type* query_keys, int* num_keys,
                                            index_type* output_indices, cudaStream_t stream) {
  hash_query_kernel<<<128, 64, 0, stream>>>(query_keys, num_keys, keys, num_sets, max_set_size_,
                                            output_indices);
}

template <typename key_type, typename index_type>
void HashBlock<key_type, index_type>::add(const key_type* new_keys, const size_t num_keys,
                                          key_type* missing_keys, int* num_missing_keys,
                                          cudaStream_t stream) {
  if (num_keys == 0) {
    return;
  }
  size_t num_batches = (num_keys - 1) / batch_size_ + 1;
  for (size_t i = 0; i < num_batches; i++) {
    size_t this_batch_size = i != num_batches - 1 ? batch_size_ : num_keys - i * batch_size_;
    hash_add_kernel<<<(this_batch_size - 1) / block_size + 1, block_size, 0, stream>>>(
        new_keys + i * this_batch_size, this_batch_size, keys, num_sets, set_sizes_, max_set_size_,
        missing_keys, num_missing_keys);
  }
}

template <typename key_type, typename index_type>
void HashBlock<key_type, index_type>::clear(cudaStream_t stream) {
  CUDA_CHECK(cudaMemsetAsync(set_sizes_, 0, sizeof(*set_sizes_) * num_sets, stream));
}

template class HashBlock<int, size_t>;
template class HashBlock<int64_t, size_t>;
template class HashBlock<size_t, size_t>;
template class HashBlock<unsigned int, size_t>;
template class HashBlock<long long, size_t>;

template class UvmTable<int, size_t>;
template class UvmTable<int64_t, size_t>;
template class UvmTable<size_t, size_t>;
template class UvmTable<unsigned int, size_t>;
template class UvmTable<long long, size_t>;
}  // namespace gpu_cache