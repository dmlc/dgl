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

  auto Size() const {
    auto diff = tail_ - head_;
    if (diff < 0) diff += capacity_;
    return diff;
  }

  friend std::ostream& operator<<(
      std::ostream& os, const CircularQueue& queue) {
    for (auto i = queue.head_; i != queue.tail_; queue.PostIncrement(i)) {
      os << queue.data_[i] << ", ";
    }
    return os << "\n";
  }

  bool IsEmpty() const { return tail_ == head_; }

  auto Capacity() const { return capacity_ - 1; }

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

  CacheKey& SetFreq() {
    freq_ = 1;
    return *this;
  }

  CacheKey& ResetFreq() {
    freq_ = 0;
    return *this;
  }

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

  template <typename CachePolicy>
  static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> QueryImpl(
      CachePolicy& policy, torch::Tensor keys);

  template <typename CachePolicy>
  static torch::Tensor ReplaceImpl(CachePolicy& policy, torch::Tensor keys);
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
              << "capacity_: " << policy.capacity_ << "\n";
  }

  std::optional<int64_t> Read(int64_t key) {
    auto it = key_to_cache_key_.find(key);
    if (it != key_to_cache_key_.end()) {
      auto& cache_key = *it->second;
      cache_key.Increment();
      return cache_key.getPos();
    }
    return std::nullopt;
  }

  int64_t Insert(int64_t key) {
    const auto pos = Evict();
    const auto in_ghost_queue = ghost_set_.erase(key);
    auto& queue = in_ghost_queue ? main_queue_ : small_queue_;
    key_to_cache_key_[key] = queue.Push(CacheKey(key, pos));
    return pos;
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
    for (auto size = small_queue_.Size(); size > small_queue_size_target_;
         size--) {
      auto evicted = small_queue_.Pop();
      auto it = key_to_cache_key_.find(evicted.getKey());
      if (evicted.getFreq() <= 0) {
        key_to_cache_key_.erase(it);
        const auto evicted_key = evicted.getKey();
        if (ghost_queue_.IsFull()) {
          ghost_set_.erase(ghost_queue_.Pop());
        }
        ghost_set_.insert(evicted_key);
        ghost_queue_.Push(evicted_key);
        return evicted.getPos();
      } else {
        it->second = main_queue_.Push(evicted.ResetFreq());
      }
    }
    return -1;
  }

  int64_t Evict() {
    // If the cache has space, get an unused slot otherwise perform eviction.
    if (cache_usage_ < capacity_) return cache_usage_++;
    const auto pos = EvictSmallQueue();
    return pos >= 0 ? pos : EvictMainQueue();
  }

  CircularQueue<CacheKey> small_queue_, main_queue_;
  CircularQueue<int64_t> ghost_queue_;
  int64_t capacity_;
  int64_t cache_usage_;
  int64_t small_queue_size_target_;
  phmap::flat_hash_set<int64_t> ghost_set_;
  phmap::flat_hash_map<int64_t, CacheKey*> key_to_cache_key_;
};

/**
 * @brief SIEVE is a simple, scalable FIFObased algorithm with a single static
 * queue. https://www.usenix.org/system/files/nsdi24-zhang-yazhuo.pdf
 **/
class SieveCachePolicy : public BaseCachePolicy {
 public:
  /**
   * @brief Constructor for the SieveCachePolicy class.
   *
   * @param capacity The capacity of the cache in terms of # elements.
   */
  SieveCachePolicy(int64_t capacity);

  SieveCachePolicy() = default;

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
   * @return positions tensor is returned holding the locations of the
   * replaced entries in the cache.
   */
  torch::Tensor Replace(torch::Tensor keys);

  std::optional<int64_t> Read(int64_t key) {
    auto it = key_to_cache_key_.find(key);
    if (it != key_to_cache_key_.end()) {
      auto& cache_key = *it->second;
      cache_key.SetFreq();
      return cache_key.getPos();
    }
    return std::nullopt;
  }

  int64_t Insert(int64_t key) {
    const auto pos = Evict();
    queue_.push_front(CacheKey(key, pos));
    key_to_cache_key_[key] = queue_.begin();
    return pos;
  }

 private:
  int64_t Evict() {
    // If the cache has space, get an unused slot otherwise perform eviction.
    if (cache_usage_ < capacity_) return cache_usage_++;
    --hand_;
    while (hand_->getFreq()) {
      hand_->ResetFreq();
      if (hand_ == queue_.begin()) hand_ = queue_.end();
      --hand_;
    }
    TORCH_CHECK(key_to_cache_key_.erase(hand_->getKey()));
    const auto pos = hand_->getPos();
    const auto temp = hand_;
    if (hand_ == queue_.begin()) {
      hand_ = queue_.end();
    } else {
      ++hand_;
    }
    queue_.erase(temp);
    return pos;
  }

  std::list<CacheKey> queue_;
  decltype(queue_)::iterator hand_;
  int64_t capacity_;
  int64_t cache_usage_;
  phmap::flat_hash_map<int64_t, decltype(hand_)> key_to_cache_key_;
};

}  // namespace storage
}  // namespace graphbolt

#endif  // GRAPHBOLT_CACHE_POLICY_H_
