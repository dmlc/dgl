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

#include <thread>
#include <unordered_map>
#include <vector>

namespace gpu_cache {

template <typename key_type, typename index_type>
class HashBlock {
 public:
  key_type* keys;
  size_t num_sets;
  size_t capacity;

  HashBlock(size_t expected_capacity, int set_size, int batch_size);
  ~HashBlock();
  void add(const key_type* new_keys, const size_t num_keys, key_type* missing_keys,
           int* num_missing_keys, cudaStream_t stream);
  void query(const key_type* query_keys, const size_t num_keys, index_type* output_indices,
             key_type* missing_keys, int* missing_positions, int* num_missing_keys,
             cudaStream_t stream);
  void query(const key_type* query_keys, int* num_keys, index_type* output_indices,
             cudaStream_t stream);
  void clear(cudaStream_t stream);

 private:
  int max_set_size_;
  int batch_size_;
  int* set_sizes_;
};

template <typename vec_type>
class H2HCopy {
 public:
  H2HCopy(int num_threads) : num_threads_(num_threads), working_(num_threads) {
    for (int i = 0; i < num_threads_; i++) {
      threads_.emplace_back(
          [&](int idx) {
            while (!terminate_) {
              if (working_[idx].load(std::memory_order_relaxed)) {
                working_[idx].store(false, std::memory_order_relaxed);
                if (num_keys_ == 0) continue;
                size_t num_keys_this_thread = (num_keys_ - 1) / num_threads_ + 1;
                size_t begin = idx * num_keys_this_thread;
                if (idx == num_threads_ - 1) {
                  num_keys_this_thread = num_keys_ - num_keys_this_thread * idx;
                }
                size_t end = begin + num_keys_this_thread;

                for (size_t i = begin; i < end; i++) {
                  size_t idx_vec = get_index_(i);
                  if (idx_vec == std::numeric_limits<size_t>::max()) {
                    continue;
                  }
                  memcpy(dst_data_ptr_ + i * vec_size_, src_data_ptr_ + idx_vec * vec_size_,
                         sizeof(vec_type) * vec_size_);
                }
                num_finished_workers_++;
              }
            }
            std::this_thread::sleep_for(std::chrono::microseconds(1));
          },
          i);
    }
  };

  void copy(vec_type* dst_data_ptr, vec_type* src_data_ptr, size_t num_keys, int vec_size,
            std::function<size_t(size_t)> get_index_func) {
    std::lock_guard<std::mutex> guard(submit_mutex_);
    dst_data_ptr_ = dst_data_ptr;
    src_data_ptr_ = src_data_ptr;
    get_index_ = get_index_func;
    num_keys_ = num_keys;
    vec_size_ = vec_size;
    num_finished_workers_.store(0, std::memory_order_acquire);

    for (auto& working : working_) {
      working.store(true, std::memory_order_relaxed);
    }

    while (num_finished_workers_ != num_threads_) {
      continue;
    }
  }

  ~H2HCopy() {
    terminate_ = true;
    for (auto& t : threads_) {
      t.join();
    }
  }

 private:
  vec_type* src_data_ptr_;
  vec_type* dst_data_ptr_;

  std::function<size_t(size_t)> get_index_;

  size_t num_keys_;
  int vec_size_;

  std::mutex submit_mutex_;
  const int num_threads_;
  std::vector<std::thread> threads_;
  std::vector<std::atomic<bool>> working_;
  volatile bool terminate_{false};
  std::atomic<int> num_finished_workers_{0};
};

template <typename key_type, typename index_type, typename vec_type = float>
class UvmTable {
 public:
  UvmTable(const size_t device_table_capacity, const size_t host_table_capacity,
           const int max_batch_size, const int vec_size,
           const vec_type default_value = (vec_type)0);
  ~UvmTable();
  void query(const key_type* d_keys, const int len, vec_type* d_vectors, cudaStream_t stream = 0);
  void add(const key_type* h_keys, const vec_type* h_vectors, const size_t len);
  void clear(cudaStream_t stream = 0);

 private:
  static constexpr int num_buffers_ = 2;
  key_type* d_keys_buffer_;
  vec_type* d_vectors_buffer_;
  vec_type* d_vectors_;

  index_type* d_output_indices_;
  index_type* d_output_host_indices_;
  index_type* h_output_host_indices_;

  key_type* d_missing_keys_;
  int* d_missing_positions_;
  int* d_missing_count_;

  std::vector<vec_type> h_vectors_;
  key_type* h_missing_keys_;

  cudaStream_t query_stream_;
  cudaEvent_t query_event_;

  vec_type* h_cpy_buffers_[num_buffers_];
  vec_type* d_cpy_buffers_[num_buffers_];
  cudaStream_t cpy_streams_[num_buffers_];
  cudaEvent_t cpy_events_[num_buffers_];

  std::unordered_map<key_type, index_type> h_final_missing_items_;

  int max_batch_size_;
  int vec_size_;
  size_t num_set_;
  size_t num_host_set_;
  size_t table_capacity_;
  std::vector<vec_type> default_vector_;

  HashBlock<key_type, index_type> device_table_;
  HashBlock<key_type, index_type> host_table_;
};
}  // namespace gpu_cache