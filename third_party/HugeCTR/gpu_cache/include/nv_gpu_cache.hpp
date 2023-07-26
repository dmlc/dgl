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
#pragma once

#include <nv_util.h>

#include <cstdio>
#include <hash_functions.cuh>
#include <limits>

#include "gpu_cache_api.hpp"
#ifdef LIBCUDACXX_VERSION
#include <cuda/std/atomic>
#include <cuda/std/semaphore>
#endif

#define SET_ASSOCIATIVITY 2
#define SLAB_SIZE 32
#define TASK_PER_WARP_TILE_MACRO 1

namespace gpu_cache {

// slab for static slab list
template <typename key_type, int warp_size>
struct static_slab {
  key_type slab_[warp_size];
};

// Static slablist(slabset) for GPU Cache
template <int set_associativity, typename key_type, int warp_size>
struct slab_set {
  static_slab<key_type, warp_size> set_[set_associativity];
};

///////////////////////////////////////////////////////////////////////////////////////////////////

// GPU Cache
template <typename key_type, typename ref_counter_type, key_type empty_key, int set_associativity,
          int warp_size, typename set_hasher = MurmurHash3_32<key_type>,
          typename slab_hasher = Mod_Hash<key_type, size_t>>
class gpu_cache : public gpu_cache_api<key_type> {
 public:
  // Ctor
  gpu_cache(const size_t capacity_in_set, const size_t embedding_vec_size);

  // Dtor
  ~gpu_cache();

  // Query API, i.e. A single read from the cache
  void Query(const key_type* d_keys, const size_t len, float* d_values, uint64_t* d_missing_index,
             key_type* d_missing_keys, size_t* d_missing_len, cudaStream_t stream,
             const size_t task_per_warp_tile = TASK_PER_WARP_TILE_MACRO) override;

  // Replace API, i.e. Follow the Query API to update the content of the cache to Most Recent
  void Replace(const key_type* d_keys, const size_t len, const float* d_values, cudaStream_t stream,
               const size_t task_per_warp_tile = TASK_PER_WARP_TILE_MACRO) override;

  // Update API, i.e. update the embeddings which exist in the cache
  void Update(const key_type* d_keys, const size_t len, const float* d_values, cudaStream_t stream,
              const size_t task_per_warp_tile = TASK_PER_WARP_TILE_MACRO) override;

  // Dump API, i.e. dump some slabsets' keys from the cache
  void Dump(key_type* d_keys, size_t* d_dump_counter, const size_t start_set_index,
            const size_t end_set_index, cudaStream_t stream) override;

 public:
  using slabset = slab_set<set_associativity, key_type, warp_size>;
#ifdef LIBCUDACXX_VERSION
  using atomic_ref_counter_type = cuda::atomic<ref_counter_type, cuda::thread_scope_device>;
  using mutex = cuda::binary_semaphore<cuda::thread_scope_device>;
#endif

 private:
  static const size_t BLOCK_SIZE_ = 64;

  // Cache data
  slabset* keys_;
  float* vals_;
  ref_counter_type* slot_counter_;

  // Global counter
#ifdef LIBCUDACXX_VERSION
  atomic_ref_counter_type* global_counter_;
#else
  ref_counter_type* global_counter_;
#endif
  // CUDA device
  int dev_;

  // Cache capacity
  size_t capacity_in_set_;
  size_t num_slot_;

  // Embedding vector size
  size_t embedding_vec_size_;

#ifdef LIBCUDACXX_VERSION
  // Array of mutex to protect (sub-)warp-level data structure, each mutex protect 1 slab set
  mutex* set_mutex_;
#else
  // Array of flag to protect (sub-)warp-level data structure, each flag act as a mutex and protect
  // 1 slab set 1 for unlock, 0 for lock
  int* set_mutex_;
#endif
};

}  // namespace gpu_cache
