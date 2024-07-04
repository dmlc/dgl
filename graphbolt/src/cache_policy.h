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
 * @file cache_policy.h
 * @brief Cache policy implementation on the CPU.
 */
#ifndef GRAPHBOLT_CACHE_POLICY_H_
#define GRAPHBOLT_CACHE_POLICY_H_

#include <parallel_hashmap/phmap.h>
#include <torch/custom_class.h>
#include <torch/torch.h>

namespace graphbolt {
namespace storage {

template <typename T>
struct CircularQueue {
  CircularQueue(const int64_t capacity)
      : tail_(0),
        head_(0),
        // + 1 is needed to be able to differentiate empty and full states.
        capacity_(capacity + 1),
        data_{new T[capacity + 1]} {}

  CircularQueue() = default;

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
  CacheKey(int64_t key, int64_t position)
      : freq_(0), key_(key), position_in_cache_(position) {
    static_assert(sizeof(CacheKey) == 2 * sizeof(int64_t));
  }

  CacheKey() = default;

  auto getFreq() const { return freq_; }

  auto getKey() const { return key_; }

  auto getPos() const { return position_in_cache_; }

  void Increment() { freq_ = std::min(3, static_cast<int>(freq_ + 1)); }

  void Decrement() { freq_ = std::max(0, static_cast<int>(freq_ - 1)); }

  friend std::ostream& operator<<(std::ostream& os, const CacheKey& key_ref) {
    return os << '(' << key_ref.key_ << ", " << key_ref.freq_ << ", "
              << key_ref.position_in_cache_ << ")";
  }

 private:
  int64_t freq_ : 3;
  int64_t key_ : 61;
  int64_t position_in_cache_;
};

class BaseCachePolicy {
 public:
  /**
   * @brief The policy query function.
   * @param keys The keys to query the cache.
   *
   * @return (positions, indices, missing_keys), where positions has the
   * locations of the keys which were found in the cache, missing_keys has the
   * keys that were not found and indices is defined such that
   * keys[indices[:positions.size(0)]] gives us the found keys and
   * keys[indices[positions.size(0):]] is identical to missing_keys.
   */
  virtual std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> Query(
      torch::Tensor keys) = 0;

  /**
   * @brief The policy replace function.
   * @param keys The keys to query the cache.
   *
   * @return positions tensor is returned holding the locations of the replaced
   * entries in the cache.
   */
  virtual torch::Tensor Replace(torch::Tensor keys) = 0;
};

/**
 * @brief S3FIFO is a simple, scalable FIFObased algorithm with three static
 * queues (S3-FIFO). https://dl.acm.org/doi/pdf/10.1145/3600006.3613147
 **/
class S3FifoCachePolicy : public BaseCachePolicy {
 public:
  /**
   * @brief Constructor for the S3FifoCachePolicy class.
   *
   * @param capacity The capacity of the cache in terms of # elements.
   */
  S3FifoCachePolicy(int64_t capacity);

  S3FifoCachePolicy() = default;

  /**
   * @brief The policy query function.
   * @param keys The keys to query the cache.
   *
   * @return (positions, indices, missing_keys), where positions has the
   * locations of the keys which were found in the cache, missing_keys has the
   * keys that were not found and indices is defined such that
   * keys[indices[:positions.size(0)]] gives us the found keys and
   * keys[indices[positions.size(0):]] is identical to missing_keys.
   */
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> Query(
      torch::Tensor keys);

  /**
   * @brief The policy replace function.
   * @param keys The keys to query the cache.
   *
   * @return positions tensor is returned holding the locations of the replaced
   * entries in the cache.
   */
  torch::Tensor Replace(torch::Tensor keys);

  friend std::ostream& operator<<(
      std::ostream& os, const S3FifoCachePolicy& policy) {
    return os << "small_queue_: " << policy.small_queue_ << "\n"
              << "main_queue_: " << policy.main_queue_ << "\n"
              << "cache_usage_: " << policy.cache_usage_ << "\n"
              << "ghost_queue_time_: " << policy.ghost_queue_time_ << "\n"
              << "capacity_: " << policy.capacity_ << "\n";
  }

 private:
  int64_t EvictMainQueue() {
    while (true) {
      auto evicted = main_queue_.Pop();
      auto it = key_to_cache_key_.find(evicted.getKey());
      if (evicted.getFreq() > 0) {
        evicted.Decrement();
        it->second = main_queue_.Push(evicted);
      } else {
        key_to_cache_key_.erase(it);
        return evicted.getPos();
      }
    }
  }

  int64_t EvictSmallQueue() {
    auto evicted = small_queue_.Pop();
    const auto is_M_full = main_queue_.IsFull();
    auto it = key_to_cache_key_.find(evicted.getKey());
    if (evicted.getFreq() <= 0 && is_M_full) {
      key_to_cache_key_.erase(it);
      // No overflow is expected for any GNN workload.
      ghost_map_[evicted.getKey()] = ghost_queue_time_++;
      return evicted.getPos();
    } else {
      const auto pos = is_M_full ? EvictMainQueue() : cache_usage_++;
      it->second = main_queue_.Push(evicted);
      return pos;
    }
  }

  // Is inside the ghost queue.
  bool InGhostQueue(int64_t key) const {
    auto it = ghost_map_.find(key);
    return it != ghost_map_.end() &&
           ghost_queue_time_ - it->second <= main_queue_.Capacity();
  }

  void TrimGhostQueue() {
    if (static_cast<int64_t>(ghost_map_.size()) >= 2 * main_queue_.Capacity()) {
      // Here, we ensure that the ghost_map_ does not grow too much.
      phmap::priv::erase_if(ghost_map_, [&](const auto& key_value) {
        const auto timestamp = key_value.second;
        return ghost_queue_time_ - timestamp > main_queue_.Capacity();
      });
    }
  }

  CircularQueue<CacheKey> small_queue_, main_queue_;
  phmap::flat_hash_map<int64_t, CacheKey*> key_to_cache_key_;
  phmap::flat_hash_map<int64_t, int64_t> ghost_map_;
  // Keeps track of the number of insertions into the ghost queue so far. If an
  // item's time of insertion is older than main_queue_.Capacity(), then it is
  // not really considered in the ghost queue.
  int64_t ghost_queue_time_;
  int64_t capacity_;
  int64_t cache_usage_;
};

}  // namespace storage
}  // namespace graphbolt

#endif  // GRAPHBOLT_CACHE_POLICY_H_
