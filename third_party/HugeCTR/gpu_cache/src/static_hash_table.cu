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
#include <cuda.h>
#include <stdint.h>
#include <stdio.h>

#include <static_hash_table.hpp>

namespace gpu_cache {

template <typename T>
__device__ __forceinline__ T atomicCASHelper(T *address, T compare, T val) {
  return atomicCAS(address, compare, val);
}

template <>
__device__ __forceinline__ long long atomicCASHelper(long long *address, long long compare,
                                                     long long val) {
  return (long long)atomicCAS((unsigned long long *)address, (unsigned long long)compare,
                              (unsigned long long)val);
}

template <>
__device__ __forceinline__ int64_t atomicCASHelper(int64_t *address, int64_t compare, int64_t val) {
  return (int64_t)atomicCAS((unsigned long long *)address, (unsigned long long)compare,
                            (unsigned long long)val);
}

template <unsigned int group_size, typename key_type, typename size_type, typename hasher,
          typename CG>
__device__ size_type insert(key_type *table, size_type capacity, key_type key, const hasher &hash,
                            const CG &cg, const key_type empty_key, const size_type invalid_slot) {
  // If insert successfully, return its position in the table,
  // otherwise return invalid_slot.

  const size_type num_groups = capacity / group_size;
#if (CUDA_VERSION < 11060)
  unsigned long long num_threads_per_group = cg.size();
#else
  unsigned long long num_threads_per_group = cg.num_threads();
#endif
  const unsigned int num_tiles_per_group = group_size / num_threads_per_group;

  // Assuming capacity is a power of 2
  size_type slot = hash(key) & (capacity - 1);
  slot = slot - (slot & (size_type)(group_size - 1)) + cg.thread_rank();

  for (size_type step = 0; step < num_groups; ++step) {
    for (unsigned int i = 0; i < num_tiles_per_group; ++i) {
      key_type existed_key = table[slot];

      // Check if key already exists
      bool existed = cg.any(existed_key == key);
      if (existed) {
        return invalid_slot;
      }

      // Try to insert the target key into empty slot
      while (true) {
        int can_insert = cg.ballot(existed_key == empty_key);

        if (!can_insert) {
          break;
        }

        bool succeed = false;
        int src_lane = __ffs(can_insert) - 1;

        if (cg.thread_rank() == src_lane) {
          key_type old = atomicCASHelper(table + slot, empty_key, key);
          if (old == empty_key) {
            // Insert key successfully
            succeed = true;
          } else if (old == key) {
            // The target key was inserted by another thread
            succeed = true;
            slot = invalid_slot;
          } else {
            // The empty slot was occupied by another key,
            // update the existed_key for next loop.
            existed_key = old;
          }
        }

        succeed = cg.shfl(succeed, src_lane);
        if (succeed) {
          slot = cg.shfl(slot, src_lane);
          return slot;
        }
      }

      slot += num_threads_per_group;
    }
    slot = (slot + group_size * step) & (capacity - 1);
  }

  return invalid_slot;
}

template <unsigned int tile_size, unsigned int group_size, typename key_type, typename size_type,
          typename hasher>
__global__ void InsertKeyKernel(key_type *table_keys, size_type *table_indices, size_type capacity,
                                const key_type *keys, size_type num_keys, size_type offset,
                                hasher hash, const key_type empty_key,
                                const size_type invalid_slot) {
  static_assert(tile_size <= group_size, "tile_size cannot be larger than group_size");

  auto block = cooperative_groups::this_thread_block();
  auto tile = cooperative_groups::tiled_partition<tile_size>(block);

  int tile_idx = tile.meta_group_size() * block.group_index().x + tile.meta_group_rank();
  int tile_cnt = tile.meta_group_size() * gridDim.x;

  for (size_type i = tile_idx; i < num_keys; i += tile_cnt) {
    key_type key = keys[i];
    if (key == empty_key) {
      if (tile.thread_rank() == 0 && table_keys[capacity] != empty_key) {
        table_keys[capacity] = empty_key;
        table_indices[capacity] = i + offset;
      }
      continue;
    }
    size_type slot =
        insert<group_size>(table_keys, capacity, key, hash, tile, empty_key, invalid_slot);
    if (tile.thread_rank() == 0 && slot != invalid_slot) {
      table_indices[slot] = i + offset;
    }
  }
}

template <unsigned int group_size, typename key_type, typename size_type, typename hasher,
          typename CG>
__device__ size_type lookup(key_type *table, size_type capacity, key_type key, const hasher &hash,
                            const CG &cg, const key_type empty_key, const size_type invalid_slot) {
  // If lookup successfully, return the target key's position in the table,
  // otherwise return invalid_slot.

  const size_type num_groups = capacity / group_size;

#if (CUDA_VERSION < 11060)
  unsigned long long num_threads_per_group = cg.size();
#else
  unsigned long long num_threads_per_group = cg.num_threads();
#endif

  const unsigned int num_tiles_per_group = group_size / num_threads_per_group;

  // Assuming capacity is a power of 2
  size_type slot = hash(key) & (capacity - 1);
  slot = slot - (slot & (size_type)(group_size - 1)) + cg.thread_rank();

  for (size_type step = 0; step < num_groups; ++step) {
    for (unsigned int i = 0; i < num_tiles_per_group; ++i) {
      key_type existed_key = table[slot];

      // Check if key exists
      int existed = cg.ballot(existed_key == key);
      if (existed) {
        int src_lane = __ffs(existed) - 1;
        slot = cg.shfl(slot, src_lane);
        return slot;
      }

      // The target key doesn't exist
      bool contain_empty = cg.any(existed_key == empty_key);
      if (contain_empty) {
        return invalid_slot;
      }

      slot += num_threads_per_group;
    }
    slot = (slot + group_size * step) & (capacity - 1);
  }

  return invalid_slot;
}

template <int warp_size>
__forceinline__ __device__ void warp_tile_copy(const size_t lane_idx,
                                               const size_t emb_vec_size_in_float,
                                               volatile float *d_dst, const float *d_src) {
  // 16 bytes align
  if (emb_vec_size_in_float % 4 != 0 || (size_t)d_dst % 16 != 0 || (size_t)d_src % 16 != 0) {
#pragma unroll
    for (size_t i = lane_idx; i < emb_vec_size_in_float; i += warp_size) {
      d_dst[i] = d_src[i];
    }
  } else {
#pragma unroll
    for (size_t i = lane_idx; i < emb_vec_size_in_float / 4; i += warp_size) {
      *(float4 *)(d_dst + i * 4) = __ldg((const float4 *)(d_src + i * 4));
    }
  }
}

template <int warp_size>
__forceinline__ __device__ void warp_tile_copy(const size_t lane_idx,
                                               const size_t emb_vec_size_in_float,
                                               volatile float *d_dst, const float default_value) {
#pragma unroll
  for (size_t i = lane_idx; i < emb_vec_size_in_float; i += warp_size) {
    d_dst[i] = default_value;
  }
}

template <unsigned int tile_size, unsigned int group_size, typename key_type, typename value_type,
          typename size_type, typename hasher>
__global__ void LookupKernel(key_type *table_keys, size_type *table_indices, size_type capacity,
                             const key_type *keys, int num_keys, const value_type *values,
                             int value_dim, value_type *output, hasher hash,
                             const key_type empty_key, const value_type default_value,
                             const size_type invalid_slot) {
  static_assert(tile_size <= group_size, "tile_size cannot be larger than group_size");
  constexpr int WARP_SIZE = 32;
  static_assert(WARP_SIZE % tile_size == 0, "tile_size must be divisible by warp_size");

  auto grid = cooperative_groups::this_grid();
  auto block = cooperative_groups::this_thread_block();
  auto tile = cooperative_groups::tiled_partition<tile_size>(block);
  auto warp_tile = cooperative_groups::tiled_partition<WARP_SIZE>(block);

  int tile_idx = tile.meta_group_size() * block.group_index().x + tile.meta_group_rank();
  int tile_cnt = tile.meta_group_size() * gridDim.x;

  for (int it = 0; it < (num_keys - 1) / tile_cnt + 1; it++) {
    size_type slot = invalid_slot;
    int key_num = it * tile_cnt + tile_idx;
    if (key_num < num_keys) {
      key_type key = keys[key_num];
      if (key == empty_key) {
        if (tile.thread_rank() == 0 && table_keys[capacity] == key) {
          slot = capacity;
        }
      } else {
        slot = lookup<group_size>(table_keys, capacity, key, hash, tile, empty_key, invalid_slot);
      }
    }
    for (int i = 0; i < WARP_SIZE / tile_size; i++) {
      auto slot_to_read = warp_tile.shfl(slot, i * tile_size);
      int idx_to_write = warp_tile.shfl(key_num, 0) + i;
      if (idx_to_write >= num_keys) break;
      if (slot_to_read == invalid_slot) {
        warp_tile_copy<WARP_SIZE>(warp_tile.thread_rank(), value_dim,
                                  output + (size_t)value_dim * idx_to_write, default_value);
        continue;
      }
      auto index = table_indices[slot_to_read];
      warp_tile_copy<WARP_SIZE>(warp_tile.thread_rank(), value_dim,
                                output + (size_t)value_dim * idx_to_write,
                                values + (size_t)value_dim * index);
    }
  }
}

template <typename key_type, typename value_type, unsigned int tile_size, unsigned int group_size,
          typename hasher>
StaticHashTable<key_type, value_type, tile_size, group_size, hasher>::StaticHashTable(
    size_type capacity, int value_dim, hasher hash)
    : table_keys_(nullptr),
      table_indices_(nullptr),
      key_capacity_(capacity * 2),
      table_values_(nullptr),
      value_capacity_(capacity),
      value_dim_(value_dim),
      size_(0),
      hash_(hash) {
  // Check parameters
  if (capacity <= 0) {
    printf("Error: capacity must be larger than 0\n");
    exit(EXIT_FAILURE);
  }
  if (value_dim <= 0) {
    printf("Error: value_dim must be larger than 0\n");
    exit(EXIT_FAILURE);
  }

  // Make key_capacity_ be a power of 2
  size_t new_capacity = group_size;
  while (new_capacity < key_capacity_) {
    new_capacity *= 2;
  }
  key_capacity_ = new_capacity;

  // Allocate device memory
  size_t align_m = 16;
  size_t num_keys = key_capacity_ + 1;
  size_t num_values = (value_capacity_ * value_dim_ + align_m - 1) / align_m * align_m;
  CUDA_CHECK(cudaMalloc(&table_keys_, sizeof(key_type) * num_keys));
  CUDA_CHECK(cudaMalloc(&table_indices_, sizeof(size_type) * num_keys));
  CUDA_CHECK(cudaMalloc(&table_values_, sizeof(value_type) * num_values));

  // Initialize table_keys_
  CUDA_CHECK(cudaMemset(table_keys_, 0xff, sizeof(key_type) * key_capacity_));
  CUDA_CHECK(cudaMemset(table_keys_ + key_capacity_, 0, sizeof(key_type)));
}

template <typename key_type, typename value_type, unsigned int tile_size, unsigned int group_size,
          typename hasher>
void StaticHashTable<key_type, value_type, tile_size, group_size, hasher>::insert(
    const key_type *keys, const value_type *values, size_type num_keys, cudaStream_t stream) {
  if (num_keys == 0) {
    return;
  }
  if (num_keys <= 0 || (size() + num_keys) > capacity()) {
    printf("Error: Invalid num_keys to insert\n");
    exit(EXIT_FAILURE);
  }

  // Insert keys
  constexpr int block = 256;
  int grid = (num_keys - 1) / block + 1;
  InsertKeyKernel<tile_size, group_size>
      <<<grid, block, 0, stream>>>(table_keys_, table_indices_, key_capacity_, keys, num_keys,
                                   size_, hash_, empty_key, invalid_slot);
  // Copy values
  CUDA_CHECK(cudaMemcpyAsync(table_values_ + size_ * value_dim_, values,
                             sizeof(value_type) * num_keys * value_dim_, cudaMemcpyDeviceToDevice,
                             stream));
  size_ += num_keys;
}

template <typename key_type, typename value_type, unsigned int tile_size, unsigned int group_size,
          typename hasher>
void StaticHashTable<key_type, value_type, tile_size, group_size, hasher>::clear(
    cudaStream_t stream) {
  CUDA_CHECK(cudaMemsetAsync(table_keys_, 0xff, sizeof(key_type) * key_capacity_, stream));
  CUDA_CHECK(cudaMemsetAsync(table_keys_ + key_capacity_, 0, sizeof(key_type), stream));
  size_ = 0;
}

template <typename key_type, typename value_type, unsigned int tile_size, unsigned int group_size,
          typename hasher>
StaticHashTable<key_type, value_type, tile_size, group_size, hasher>::~StaticHashTable() {
  CUDA_CHECK(cudaFree(table_keys_));
  CUDA_CHECK(cudaFree(table_indices_));
  CUDA_CHECK(cudaFree(table_values_));
}

template <typename key_type, typename value_type, unsigned int tile_size, unsigned int group_size,
          typename hasher>
void StaticHashTable<key_type, value_type, tile_size, group_size, hasher>::lookup(
    const key_type *keys, value_type *values, int num_keys, value_type default_value,
    cudaStream_t stream) {
  if (num_keys == 0) {
    return;
  }

  constexpr int block = 256;
  const int grid = (num_keys - 1) / block + 1;
  // Lookup keys
  LookupKernel<tile_size, group_size><<<grid, block, 0, stream>>>(
      table_keys_, table_indices_, key_capacity_, keys, num_keys, table_values_, value_dim_, values,
      hash_, empty_key, default_value, invalid_slot);
}

template class StaticHashTable<long long, float>;
template class StaticHashTable<uint32_t, float>;
}  // namespace gpu_cache