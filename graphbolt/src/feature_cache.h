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
 * @file feature_cache.h
 * @brief Feature cache implementation on the CPU.
 */
#include <torch/custom_class.h>
#include <torch/torch.h>

#include <optional>
#include <unordered_map>
#include <vector>

namespace graphbolt {
namespace storage {

template <typename T>
struct circular_queue {
  circular_queue(const int64_t capacity)
      : head_{0},
        tail_{0},
        capacity_{capacity + 1},
        buffer_{new T[capacity + 1]} {}

  T* insert(const T& x) {
    auto insert_ptr = &buffer_[head_++];
    *insert_ptr = x;
    if (head_ >= capacity_) head_ -= capacity_;
    return insert_ptr;
  }

  T evict() {
    auto ret = buffer_[tail_++];
    if (tail_ >= capacity_) tail_ -= capacity_;
    return ret;
  }

  T& tail() const { return buffer_[tail_]; }

  bool is_full() const {
    const auto diff = head_ + 1 - tail_;
    return diff == 0 || diff == capacity_;
  }

  friend std::ostream& operator<<(std::ostream& os, const circular_queue& m) {
    for (auto i = m.tail_; i != m.head_; i++) {
      if (i >= m.capacity_) i -= m.capacity_;
      os << m.buffer_[i] << ", ";
    }
    return os << "\n";
  }

  bool is_empty() const { return head_ == tail_; }

  int64_t capacity() const { return capacity_ - 1; }

 private:
  int64_t head_;
  int64_t tail_;
  int64_t capacity_;
  std::unique_ptr<T[]> buffer_;
};

struct cache_key {
  int64_t freq : 3;
  int64_t key : 61;

  cache_key(int64_t _key = 0) : freq{0}, key{_key} {}

  void increment() { freq = std::min(3, freq + 1); }

  void decrement() { freq = std::max(0, freq - 1); }

  friend std::ostream& operator<<(std::ostream& os, const cache_key& m) {
    return os << '(' << m.key << ", " << m.freq << ")";
  }
};

struct S3FifoCachePolicy {
  S3FifoCachePolicy(const int64_t capacity);

  S3FifoCachePolicy() = default;

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> Query(
      torch::Tensor keys);

  torch::Tensor Replace(torch::Tensor keys);

  friend std::ostream& operator<<(
      std::ostream& os, const S3FifoCachePolicy& m) {
    return os << "S_: " << m.S_ << "\n"
              << "M_: " << m.M_ << "\n"
              << "position_q_: " << m.position_q_ << "\n"
              << "ghost_q_time_: " << m.ghost_q_time_ << "\n"
              << "capacity_: " << m.capacity_ << "\n";
  }

 private:
  int64_t evict_M() {
    for (;;) {
      auto evicted = M_.evict();
      if (evicted.freq > 0) {
        evicted.decrement();
        M_.insert(evicted);
      } else {
        auto it = position_map_.find(evicted.key);
        auto pos = it->second.second;
        position_map_.erase(it);
        return pos;
      }
    }
  }

  int64_t evict_S() {
    auto evicted = S_.evict();
    const auto is_M_full = M_.is_full();
    if (evicted.freq > 0 || !is_M_full) {
      auto it = position_map_.find(evicted.key);
      const auto pos = is_M_full ? evict_M() : position_q_++;
      it->second.first = M_.insert(evicted);
      return pos;
    } else {
      auto it = position_map_.find(evicted.key);
      auto pos = it->second.second;
      position_map_.erase(it);
      ghost_map_[evicted.key] = ghost_q_time_++;
      return pos;
    }
  }

  bool in_G(int64_t key) const {
    auto it = ghost_map_.find(key);
    return it != ghost_map_.end() &&
           ghost_q_time_ - it->second <= M_.capacity();
  }

  circular_queue<cache_key> S_, M_;
  std::unordered_map<int64_t, std::pair<cache_key*, int64_t>> position_map_;
  std::unordered_map<int64_t, int64_t> ghost_map_;
  int64_t position_q_;
  int64_t ghost_q_time_;
  int64_t capacity_;
};

struct FeatureCache : public torch::CustomClassHolder {
  FeatureCache(const std::vector<int64_t>& shape, torch::ScalarType dtype);

  FeatureCache() = default;

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> Query(
      torch::Tensor keys, bool pin_memory);

  void Replace(torch::Tensor keys, torch::Tensor values);

  static c10::intrusive_ptr<FeatureCache> Create(
      const std::vector<int64_t>& shape, torch::ScalarType dtype);

 private:
  torch::Tensor tensor_;
  S3FifoCachePolicy policy_;
};

}  // namespace storage
}  // namespace graphbolt
