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
#include <nv_util.h>

#include <iostream>
#include <static_hash_table.hpp>
#include <static_table.hpp>

namespace gpu_cache {

template <typename key_type>
static_table<key_type>::static_table(const size_t table_size, const size_t embedding_vec_size,
                                     const float default_value)
    : table_size_(table_size),
      embedding_vec_size_(embedding_vec_size),
      default_value_(default_value),
      static_hash_table_(table_size, embedding_vec_size) {
  if (embedding_vec_size_ == 0) {
    printf("Error: Invalid value for embedding_vec_size.\n");
    return;
  }
}

template <typename key_type>
void static_table<key_type>::Query(const key_type* d_keys, const size_t len, float* d_values,
                                   cudaStream_t stream) {
  static_hash_table_.lookup(d_keys, d_values, len, default_value_, stream);
}

template <typename key_type>
void static_table<key_type>::Init(const key_type* d_keys, const size_t len, const float* d_values,
                                  cudaStream_t stream) {
  static_hash_table_.insert(d_keys, d_values, len, stream);
}

template <typename key_type>
void static_table<key_type>::Clear(cudaStream_t stream) {
  static_hash_table_.clear(stream);
}

template class static_table<unsigned int>;
template class static_table<long long>;

}  // namespace gpu_cache
