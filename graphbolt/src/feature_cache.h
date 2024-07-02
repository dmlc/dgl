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
#ifndef GRAPHBOLT_FEATURE_CACHE_H_
#define GRAPHBOLT_FEATURE_CACHE_H_

#include <parallel_hashmap/phmap.h>
#include <torch/custom_class.h>
#include <torch/torch.h>

#include <vector>

namespace graphbolt {
namespace storage {

template <typename T>
struct CircularQueue {
  CircularQueue(const int64_t capacity)
      : tail_(0),
        head_(0),
        capacity_(capacity + 1),
        data_{new T[capacity + 1]} {}

  T* Push(const T& x) {
    auto insert_ptr = &data_[PostIncrement(tail_)];
    *insert_ptr = x;
    return insert_ptr;
  }

  T Pop() { return data_[PostIncrement(head_)]; }

  T& Front() const { return data_[head_]; }

  bool IsFull() const {
    const auto diff = tail_ + 1 - head_;
    return diff == 0 || diff == capacity_;
  }

  friend std::ostream& operator<<(
      std::ostream& os, const CircularQueue& queue) {
    for (auto i = queue.head_; i != queue.tail_; queue.PostIncrement(i)) {
      os << queue.data_[i] << ", ";
    }
    return os << "\n";
  }

  bool IsEmpty() const { return tail_ == head_; }

  int64_t Capacity() const { return capacity_ - 1; }

 private:
  int64_t PostIncrement(int64_t& i) const {
    const auto ret = i++;
    if (i >= capacity_) i -= capacity_;
    return ret;
  }

  int64_t tail_;
  int64_t head_;
  int64_t capacity_;
  std::unique_ptr<T[]> data_;
};

struct CacheKey {
  int64_t freq_ : 3;
  int64_t key_ : 61;
  int64_t position_;

  CacheKey(int64_t key, int64_t position)
      : freq_(0), key_(key), position_(position) {
    static_assert(sizeof(CacheKey) == 2 * sizeof(int64_t));
  }

  CacheKey() = default;

  void Increment() { freq_ = std::min(3, static_cast<int>(freq_ + 1)); }

  void Decrement() { freq_ = std::max(0, static_cast<int>(freq_ - 1)); }

  friend std::ostream& operator<<(std::ostream& os, const CacheKey& key_ref) {
    return os << '(' << key_ref.key_ << ", " << key_ref.freq_ << ", "
              << key_ref.position_ << ")";
  }
};

class S3FifoCachePolicy : public torch::CustomClassHolder {
 public:
  /**
   * @brief Constructor for the S3FifoCachePolicy class.
   *
   * @param capacity The capacity of the cache in terms of # elements.
   */
  S3FifoCachePolicy(int64_t capacity);

  S3FifoCachePolicy() = default;

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> Query(
      torch::Tensor keys);

  torch::Tensor Replace(torch::Tensor keys);

  static c10::intrusive_ptr<S3FifoCachePolicy> Create(int64_t capacity);

  friend std::ostream& operator<<(
      std::ostream& os, const S3FifoCachePolicy& m) {
    return os << "S_: " << m.S_ << "\n"
              << "M_: " << m.M_ << "\n"
              << "position_q_: " << m.position_q_ << "\n"
              << "ghost_q_time_: " << m.ghost_q_time_ << "\n"
              << "capacity_: " << m.capacity_ << "\n";
  }

 private:
  int64_t EvictM() {
    for (;;) {
      auto evicted = M_.Pop();
      auto it = position_map_.find(evicted.key_);
      if (evicted.freq_ > 0) {
        evicted.Decrement();
        it->second = M_.Push(evicted);
      } else {
        position_map_.erase(it);
        return evicted.position_;
      }
    }
  }

  int64_t EvictS() {
    auto evicted = S_.Pop();
    const auto is_M_full = M_.IsFull();
    auto it = position_map_.find(evicted.key_);
    if (evicted.freq_ > 0 || !is_M_full) {
      const auto pos = is_M_full ? EvictM() : position_q_++;
      it->second = M_.Push(evicted);
      return pos;
    } else {
      position_map_.erase(it);
      ghost_map_[evicted.key_] = ghost_q_time_++;
      return evicted.position_;
    }
  }

  // Is inside the ghost queue.
  bool InG(int64_t key) const {
    auto it = ghost_map_.find(key);
    return it != ghost_map_.end() &&
           ghost_q_time_ - it->second <= M_.Capacity();
  }

  CircularQueue<CacheKey> S_, M_;
  phmap::flat_hash_map<int64_t, CacheKey*> position_map_;
  phmap::flat_hash_map<int64_t, int64_t> ghost_map_;
  int64_t position_q_;
  int64_t ghost_q_time_;
  int64_t capacity_;
};

struct FeatureCache : public torch::CustomClassHolder {
  /**
   * @brief Constructor for the FeatureCache struct.
   *
   * @param shape The shape of the cache.
   * @param dtype The dtype of elements stored in the cache.
   */
  FeatureCache(const std::vector<int64_t>& shape, torch::ScalarType dtype);

  FeatureCache() = default;

  torch::Tensor Query(
      torch::Tensor positions, torch::Tensor indices, int64_t size,
      bool pin_memory);

  void Replace(
      torch::Tensor keys, torch::Tensor values, torch::Tensor positions);

  static c10::intrusive_ptr<FeatureCache> Create(
      const std::vector<int64_t>& shape, torch::ScalarType dtype);

 private:
  torch::Tensor tensor_;
};

}  // namespace storage
}  // namespace graphbolt

#endif  // GRAPHBOLT_FEATURE_CACHE_H_
