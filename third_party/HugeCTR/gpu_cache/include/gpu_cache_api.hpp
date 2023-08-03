/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#define TASK_PER_WARP_TILE_MACRO 1

namespace gpu_cache {

///////////////////////////////////////////////////////////////////////////////////////////////////

// GPU Cache API
template <typename key_type>
class gpu_cache_api {
 public:
  virtual ~gpu_cache_api() noexcept(false) {}
  // Query API, i.e. A single read from the cache
  virtual void Query(const key_type* d_keys, const size_t len, float* d_values,
                     uint64_t* d_missing_index, key_type* d_missing_keys, size_t* d_missing_len,
                     cudaStream_t stream,
                     const size_t task_per_warp_tile = TASK_PER_WARP_TILE_MACRO) = 0;

  // Replace API, i.e. Follow the Query API to update the content of the cache to Most Recent
  virtual void Replace(const key_type* d_keys, const size_t len, const float* d_values,
                       cudaStream_t stream,
                       const size_t task_per_warp_tile = TASK_PER_WARP_TILE_MACRO) = 0;

  // Update API, i.e. update the embeddings which exist in the cache
  virtual void Update(const key_type* d_keys, const size_t len, const float* d_values,
                      cudaStream_t stream,
                      const size_t task_per_warp_tile = TASK_PER_WARP_TILE_MACRO) = 0;

  // Dump API, i.e. dump some slabsets' keys from the cache
  virtual void Dump(key_type* d_keys, size_t* d_dump_counter, const size_t start_set_index,
                    const size_t end_set_index, cudaStream_t stream) = 0;
};

}  // namespace gpu_cache
