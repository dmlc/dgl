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
#include <limits>
#include <static_hash_table.hpp>

namespace gpu_cache {

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename key_type>
class static_table {
 public:
  // Ctor
  static_table(const size_t table_size, const size_t embedding_vec_size,
               const float default_value = 0);

  // Dtor
  ~static_table(){};

  // Query API, i.e. A single read from the cache
  void Query(const key_type* d_keys, const size_t len, float* d_values, cudaStream_t stream);

  // Replace API, i.e. Follow the Query API to update the content of the cache to Most Recent
  void Init(const key_type* d_keys, const size_t len, const float* d_values, cudaStream_t stream);

  void Clear(cudaStream_t stream);

 private:
  StaticHashTable<key_type, float> static_hash_table_;
  // Embedding vector size
  size_t embedding_vec_size_;
  size_t table_size_;
  float default_value_;
};

}  // namespace gpu_cache
