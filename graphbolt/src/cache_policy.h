/**
 *   Copyright (c) 2024, GT-TDAlab (Muhammed Fatih Balin & Umit V. Catalyurek)
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

#include <limits>
#include <mutex>

#include "./circular_queue.h"

namespace graphbolt {
namespace storage {

struct CacheKey {
  CacheKey(int64_t key, int64_t position)
      : freq_(0),
        key_(key),
        position_in_cache_(position),
        read_reference_count_(0),
        write_reference_count_(1) {
    static_assert(sizeof(CacheKey) == 2 * sizeof(int64_t));
  }

  CacheKey() = default;

  auto getFreq() const { return freq_; }

  auto getKey() const { return key_; }

  auto getPos() const { return position_in_cache_; }

  CacheKey& Increment() {
    freq_ = std::min(3, static_cast<int>(freq_ + 1));
    return *this;
  }

  CacheKey& Decrement() {
    freq_ = std::max(0, static_cast<int>(freq_ - 1));
    return *this;
  }

  CacheKey& SetFreq() {
    freq_ = 1;
    return *this;
  }

  CacheKey& ResetFreq() {
    freq_ = 0;
    return *this;
  }

  template <bool write>
  CacheKey& StartUse() {
    if constexpr (write) {
      TORCH_CHECK(
          write_reference_count_++ < std::numeric_limits<int16_t>::max());
    } else {
      TORCH_CHECK(read_reference_count_++ < std::numeric_limits<int8_t>::max());
    }
    return *this;
  }

  template <bool write>
  CacheKey& EndUse() {
    if constexpr (write) {
      --write_reference_count_;
    } else {
      --read_reference_count_;
    }
    return *this;
  }

  bool InUse() const { return read_reference_count_ || write_reference_count_; }

  bool BeingWritten() const { return write_reference_count_; }

  friend std::ostream& operator<<(std::ostream& os, const CacheKey& key_ref) {
    return os << '(' << key_ref.key_ << ", " << key_ref.freq_ << ", "
              << key_ref.position_in_cache_ << ")";
  }

 private:
  int64_t freq_ : 3;
  int64_t key_ : 61;
  int64_t position_in_cache_ : 40;
  int64_t read_reference_count_ : 8;
  // There could be a chain of writes so it is better to have larger bit count.
  int64_t write_reference_count_ : 16;
};

class BaseCachePolicy {
 public:
  BaseCachePolicy() = default;
  /**
   * @brief A virtual base class constructor ensures that the derived class
   * destructor gets called.
   */
  virtual ~BaseCachePolicy() = default;

  /**
   * @brief The policy query function.
   * @param keys The keys to query the cache.
   *
   * @return (positions, indices, missing_keys, found_ptrs), where positions has
   * the locations of the keys which were found in the cache, missing_keys has
   * the keys that were not found and indices is defined such that
   * keys[indices[:positions.size(0)]] gives us the keys for the found pointers
   * and keys[indices[positions.size(0):]] is identical to missing_keys.
   */
  virtual std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
  Query(torch::Tensor keys) = 0;

  /**
   * @brief The policy replace function.
   * @param keys The keys to query the cache.
   *
   * @return (positions, pointers), where positions has the locations of the
   * replaced entries and pointers point to their CacheKey pointers in the
   * cache.
   */
  virtual std::tuple<torch::Tensor, torch::Tensor> Replace(
      torch::Tensor keys) = 0;

  /**
   * @brief A reader has finished reading these keys, so they can be evicted.
   * @param pointers The CacheKey pointers in the cache to unmark.
   */
  virtual void ReadingCompleted(torch::Tensor pointers) = 0;

  /**
   * @brief A writer has finished writing these keys, so they can be evicted.
   * @param pointers The CacheKey pointers in the cache to unmark.
   */
  virtual void WritingCompleted(torch::Tensor pointers) = 0;

 protected:
  template <typename CachePolicy>
  static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
  QueryImpl(CachePolicy& policy, torch::Tensor keys);

  template <typename CachePolicy>
  static std::tuple<torch::Tensor, torch::Tensor> ReplaceImpl(
      CachePolicy& policy, torch::Tensor keys);

  template <bool write, typename CachePolicy>
  static void ReadingWritingCompletedImpl(
      CachePolicy& policy, torch::Tensor pointers);
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

  S3FifoCachePolicy(S3FifoCachePolicy&&) = default;

  virtual ~S3FifoCachePolicy() = default;

  /**
   * @brief See BaseCachePolicy::Query.
   */
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> Query(
      torch::Tensor keys);

  /**
   * @brief See BaseCachePolicy::Replace.
   */
  std::tuple<torch::Tensor, torch::Tensor> Replace(torch::Tensor keys);

  /**
   * @brief See BaseCachePolicy::ReadingCompleted.
   */
  void ReadingCompleted(torch::Tensor pointers);

  /**
   * @brief See BaseCachePolicy::WritingCompleted.
   */
  void WritingCompleted(torch::Tensor pointers);

  friend std::ostream& operator<<(
      std::ostream& os, const S3FifoCachePolicy& policy) {
    return os << "small_queue_: " << policy.small_queue_ << "\n"
              << "main_queue_: " << policy.main_queue_ << "\n"
              << "cache_usage_: " << policy.cache_usage_ << "\n"
              << "capacity_: " << policy.capacity_ << "\n";
  }

  template <bool write>
  std::optional<std::pair<int64_t, CacheKey*>> Read(int64_t key) {
    auto it = key_to_cache_key_.find(key);
    if (it != key_to_cache_key_.end()) {
      auto& cache_key = *it->second;
      if (write || !cache_key.BeingWritten())
        return std::make_pair(
            cache_key.Increment().StartUse<write>().getPos(), &cache_key);
    }
    return std::nullopt;
  }

  std::pair<int64_t, CacheKey*> Insert(int64_t key) {
    const auto pos = Evict();
    const auto in_ghost_queue = ghost_set_.erase(key);
    auto& queue = in_ghost_queue ? main_queue_ : small_queue_;
    auto cache_key_ptr = queue.Push(CacheKey(key, pos));
    key_to_cache_key_[key] = cache_key_ptr;
    return {pos, cache_key_ptr};
  }

  template <bool write>
  void Unmark(CacheKey* cache_key_ptr) {
    cache_key_ptr->EndUse<write>();
  }

 private:
  int64_t EvictMainQueue() {
    while (true) {
      auto evicted = main_queue_.Pop();
      auto it = key_to_cache_key_.find(evicted.getKey());
      if (evicted.getFreq() > 0 || evicted.InUse()) {
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
      if (evicted.getFreq() > 0 || evicted.InUse()) {
        it->second = main_queue_.Push(evicted.ResetFreq());
      } else {
        key_to_cache_key_.erase(it);
        const auto evicted_key = evicted.getKey();
        if (ghost_queue_.IsFull()) {
          ghost_set_.erase(ghost_queue_.Pop());
        }
        ghost_set_.insert(evicted_key);
        ghost_queue_.Push(evicted_key);
        return evicted.getPos();
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

  virtual ~SieveCachePolicy() = default;

  /**
   * @brief See BaseCachePolicy::Query.
   */
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> Query(
      torch::Tensor keys);

  /**
   * @brief See BaseCachePolicy::Replace.
   */
  std::tuple<torch::Tensor, torch::Tensor> Replace(torch::Tensor keys);

  /**
   * @brief See BaseCachePolicy::ReadingCompleted.
   */
  void ReadingCompleted(torch::Tensor pointers);

  /**
   * @brief See BaseCachePolicy::WritingCompleted.
   */
  void WritingCompleted(torch::Tensor pointers);

  template <bool write>
  std::optional<std::pair<int64_t, CacheKey*>> Read(int64_t key) {
    auto it = key_to_cache_key_.find(key);
    if (it != key_to_cache_key_.end()) {
      auto& cache_key = *it->second;
      if (write || !cache_key.BeingWritten())
        return std::make_pair(
            cache_key.SetFreq().StartUse<write>().getPos(), &cache_key);
    }
    return std::nullopt;
  }

  std::pair<int64_t, CacheKey*> Insert(int64_t key) {
    const auto pos = Evict();
    queue_.push_front(CacheKey(key, pos));
    auto cache_key_ptr = &queue_.front();
    key_to_cache_key_[key] = cache_key_ptr;
    return {pos, cache_key_ptr};
  }

  template <bool write>
  void Unmark(CacheKey* cache_key_ptr) {
    cache_key_ptr->EndUse<write>();
  }

 private:
  int64_t Evict() {
    // If the cache has space, get an unused slot otherwise perform eviction.
    if (cache_usage_ < capacity_) return cache_usage_++;
    --hand_;
    while (hand_->getFreq() || hand_->InUse()) {
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
  phmap::flat_hash_map<int64_t, CacheKey*> key_to_cache_key_;
};

/**
 * @brief LeastRecentlyUsed is a simple, scalable FIFObased algorithm with a
 * single static queue.
 **/
class LruCachePolicy : public BaseCachePolicy {
 public:
  /**
   * @brief Constructor for the LruCachePolicy class.
   *
   * @param capacity The capacity of the cache in terms of # elements.
   */
  LruCachePolicy(int64_t capacity);

  LruCachePolicy() = default;

  virtual ~LruCachePolicy() = default;

  /**
   * @brief See BaseCachePolicy::Query.
   */
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> Query(
      torch::Tensor keys);

  /**
   * @brief See BaseCachePolicy::Replace.
   */
  std::tuple<torch::Tensor, torch::Tensor> Replace(torch::Tensor keys);

  /**
   * @brief See BaseCachePolicy::ReadingCompleted.
   */
  void ReadingCompleted(torch::Tensor pointers);

  /**
   * @brief See BaseCachePolicy::WritingCompleted.
   */
  void WritingCompleted(torch::Tensor pointers);

  void MoveToFront(std::list<CacheKey>::iterator it) {
    std::list<CacheKey> temp;
    // Transfer the element to temp to keep references valid.
    auto next_it = it;
    std::advance(next_it, 1);
    temp.splice(temp.begin(), queue_, it, next_it);
    // Move the element to the beginning of the queue.
    queue_.splice(queue_.begin(), temp);
    // The iterators and references are not invalidated.
    TORCH_CHECK(it == queue_.begin());
  }

  template <bool write>
  std::optional<std::pair<int64_t, CacheKey*>> Read(int64_t key) {
    auto it = key_to_cache_key_.find(key);
    if (it != key_to_cache_key_.end()) {
      auto& cache_key = *it->second;
      if (write || !cache_key.BeingWritten()) {
        MoveToFront(it->second);
        return std::make_pair(cache_key.StartUse<write>().getPos(), &cache_key);
      }
    }
    return std::nullopt;
  }

  std::pair<int64_t, CacheKey*> Insert(int64_t key) {
    const auto pos = Evict();
    queue_.push_front(CacheKey(key, pos));
    key_to_cache_key_[key] = queue_.begin();
    return {pos, &queue_.front()};
  }

  template <bool write>
  void Unmark(CacheKey* cache_key_ptr) {
    cache_key_ptr->EndUse<write>();
  }

 private:
  int64_t Evict() {
    // If the cache has space, get an unused slot otherwise perform eviction.
    if (cache_usage_ < capacity_) return cache_usage_++;
    // Do not evict items that are still in use.
    for (auto cache_key = queue_.back(); cache_key.InUse();
         cache_key = queue_.back()) {
      auto it = queue_.end();
      std::advance(it, -1);
      // Move the last element to the front without invalidating references.
      MoveToFront(it);
      // The check below will hold true. Disabled for performance reasons.
      // TORCH_CHECK(key_to_cache_key_[cache_key.getKey()] == queue_.begin());
    }
    const auto& cache_key = queue_.back();
    TORCH_CHECK(key_to_cache_key_.erase(cache_key.getKey()));
    const auto pos = cache_key.getPos();
    queue_.pop_back();
    return pos;
  }

  std::list<CacheKey> queue_;
  int64_t capacity_;
  int64_t cache_usage_;
  phmap::flat_hash_map<int64_t, decltype(queue_)::iterator> key_to_cache_key_;
};

/**
 * @brief Clock (FIFO-Reinsertion) is a simple, scalable FIFObased algorithm
 * with a single static queue.
 * https://people.csail.mit.edu/saltzer/Multics/MHP-Saltzer-060508/bookcases/M00s/M0104%20074-12).PDF
 **/
class ClockCachePolicy : public BaseCachePolicy {
 public:
  /**
   * @brief Constructor for the ClockCachePolicy class.
   *
   * @param capacity The capacity of the cache in terms of # elements.
   */
  ClockCachePolicy(int64_t capacity);

  ClockCachePolicy() = default;

  ClockCachePolicy(ClockCachePolicy&&) = default;

  virtual ~ClockCachePolicy() = default;

  /**
   * @brief See BaseCachePolicy::Query.
   */
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> Query(
      torch::Tensor keys);

  /**
   * @brief See BaseCachePolicy::Replace.
   */
  std::tuple<torch::Tensor, torch::Tensor> Replace(torch::Tensor keys);

  /**
   * @brief See BaseCachePolicy::ReadingCompleted.
   */
  void ReadingCompleted(torch::Tensor pointers);

  /**
   * @brief See BaseCachePolicy::WritingCompleted.
   */
  void WritingCompleted(torch::Tensor pointers);

  template <bool write>
  std::optional<std::pair<int64_t, CacheKey*>> Read(int64_t key) {
    auto it = key_to_cache_key_.find(key);
    if (it != key_to_cache_key_.end()) {
      auto& cache_key = *it->second;
      if (write || !cache_key.BeingWritten())
        return std::make_pair(
            cache_key.SetFreq().StartUse<write>().getPos(), &cache_key);
    }
    return std::nullopt;
  }

  std::pair<int64_t, CacheKey*> Insert(int64_t key) {
    const auto pos = Evict();
    auto cache_key_ptr = queue_.Push(CacheKey(key, pos));
    key_to_cache_key_[key] = cache_key_ptr;
    return {pos, cache_key_ptr};
  }

  template <bool write>
  void Unmark(CacheKey* cache_key_ptr) {
    cache_key_ptr->EndUse<write>();
  }

 private:
  int64_t Evict() {
    // If the cache has space, get an unused slot otherwise perform eviction.
    if (cache_usage_ < capacity_) return cache_usage_++;
    CacheKey cache_key;
    while (true) {
      cache_key = queue_.Pop();
      if (cache_key.getFreq() || cache_key.InUse()) {
        key_to_cache_key_[cache_key.getKey()] =
            queue_.Push(cache_key.ResetFreq());
      } else
        break;
    }
    TORCH_CHECK(key_to_cache_key_.erase(cache_key.getKey()));
    return cache_key.getPos();
  }

  CircularQueue<CacheKey> queue_;
  int64_t capacity_;
  int64_t cache_usage_;
  phmap::flat_hash_map<int64_t, CacheKey*> key_to_cache_key_;
};

}  // namespace storage
}  // namespace graphbolt

#endif  // GRAPHBOLT_CACHE_POLICY_H_
