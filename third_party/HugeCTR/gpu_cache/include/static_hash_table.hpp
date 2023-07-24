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

#include <hash_functions.cuh>

namespace gpu_cache {
template <typename key_type, typename value_type, unsigned int tile_size = 4,
          unsigned int group_size = 16, typename hasher = MurmurHash3_32<key_type>>
class StaticHashTable {
 public:
  using size_type = uint32_t;
  static_assert(sizeof(key_type) <= 8, "sizeof(key_type) cannot be larger than 8 bytes");
  static_assert(sizeof(key_type) >= sizeof(size_type),
                "sizeof(key_type) cannot be smaller than sizeof(size_type)");
  static_assert((group_size & (group_size - 1)) == 0, "group_size must be a power of 2");
  static_assert(group_size > 1, "group_size must be larger than 1");
  // User can use empty_key as input without affecting correctness,
  // since we will handle it inside kernel.
  constexpr static key_type empty_key = ~(key_type)0;
  constexpr static size_type invalid_slot = ~(size_type)0;

 public:
  StaticHashTable(size_type capacity, int value_dim = 1, hasher hash = hasher{});
  ~StaticHashTable();

  inline size_type size() const { return size_; }
  inline size_type capacity() const { return value_capacity_; }
  inline size_type key_capacity() const { return key_capacity_; }

  inline size_t memory_usage() const {
    size_t keys_bytes = sizeof(key_type) * (key_capacity_ + 1);
    size_t indices_bytes = sizeof(size_type) * (key_capacity_ + 1);
    size_t values_bytes = sizeof(value_type) * value_capacity_ * value_dim_;
    return keys_bytes + indices_bytes + values_bytes;
  }

  void clear(cudaStream_t stream = 0);

  // Note:
  // 1. Please make sure the key to be inserted is not duplicated.
  // 2. Please make sure the key to be inserted does not exist in the table.
  // 3. Please make sure (size() + num_keys) <= capacity().
  void insert(const key_type *keys, const value_type *values, size_type num_keys,
              cudaStream_t stream = 0);

  void lookup(const key_type *keys, value_type *values, int num_keys, value_type default_value = 0,
              cudaStream_t stream = 0);

 private:
  key_type *table_keys_;
  size_type *table_indices_;
  size_type key_capacity_;

  value_type *table_values_;
  size_type value_capacity_;
  int value_dim_;

  size_type size_;
  hasher hash_;
};
}  // namespace gpu_cache