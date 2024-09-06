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

#include <torch/custom_class.h>
#include <torch/torch.h>
#include <tsl/robin_map.h>
#include <tsl/robin_set.h>

#include <cuda/std/atomic>
#include <limits>

#include "./circular_queue.h"

namespace graphbolt {
namespace storage {

struct CacheKey {
  auto getKey() const {
    return (static_cast<int64_t>(key_higher_16_bits_) << 32) +
           key_lower_32_bits_;
  }

  CacheKey(int64_t key) : CacheKey(key, std::numeric_limits<int64_t>::min()) {}

  CacheKey(int64_t key, int64_t position)
      : freq_(0),
        // EndUse<true>() should be called to reset the reference count.
        reference_count_(-1),
        key_higher_16_bits_(key >> 32),
        key_lower_32_bits_(key),
        position_in_cache_(position) {
    TORCH_CHECK(key == getKey());
    static_assert(sizeof(CacheKey) == 2 * sizeof(int64_t));
  }

  CacheKey() = default;

  auto getFreq() const { return freq_; }

  auto getPos() const { return position_in_cache_; }

  CacheKey& setPos(int64_t pos) {
    position_in_cache_ = pos;
    return *this;
  }

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

  CacheKey& StartRead() {
    ::cuda::std::atomic_ref ref(reference_count_);
    // StartRead runs concurrently only with EndUse. EndUse does not need to see
    // this modification at all. So we can use the relaxed memory order.
    const auto old_val = ref.fetch_add(1, ::cuda::std::memory_order_relaxed);
    TORCH_CHECK(
        old_val < std::numeric_limits<int8_t>::max(),
        "There are too many in-flight read requests to the same cache entry!");
    return *this;
  }

  template <bool write>
  CacheKey& EndUse() {
    ::cuda::std::atomic_ref ref(reference_count_);
    // The EndUse operation needs to synchronize with the InUse operation. So we
    // have an release-acquire ordering between the two.
    // https://en.cppreference.com/w/cpp/atomic/memory_order#Release-Acquire_ordering
    if constexpr (write) {
      ref.fetch_add(1, ::cuda::std::memory_order_release);
    } else {
      ref.fetch_add(-1, ::cuda::std::memory_order_release);
    }
    return *this;
  }

  bool InUse() const {
    ::cuda::std::atomic_ref ref(reference_count_);
    // The operations after a call to this function need to happen after the
    // load operation. Hence the acquire order.
    return ref.load(::cuda::std::memory_order_acquire);
  }

  bool BeingWritten() const {
    ::cuda::std::atomic_ref ref(reference_count_);
    // The only operation coming after this op is the StartRead operation. Since
    // StartRead is a refcount increment operation, it is fine if we don't
    // synchronize with EndUse ops.
    return ref.load(::cuda::std::memory_order_relaxed) < 0;
  }

  friend std::ostream& operator<<(std::ostream& os, const CacheKey& key_ref) {
    ::cuda::std::atomic_ref ref(key_ref.reference_count_);
    return os << '(' << key_ref.getKey() << ", " << key_ref.freq_ << ", "
              << key_ref.position_in_cache_ << ", " << ref.load() << ")";
  }

 private:
  int8_t freq_;
  // Negative values indicate writing while positive values indicate reading.
  // Access only through an std::atomic_ref instance atomically.
  int8_t reference_count_;
  // Keys are restricted to be 48-bit unsigned integers.
  uint16_t key_higher_16_bits_;
  uint32_t key_lower_32_bits_;
  int64_t position_in_cache_;
};

class BaseCachePolicy {
 public:
  BaseCachePolicy(int64_t capacity) : capacity_(capacity), cache_usage_(0) {}

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
   * @brief The policy query function.
   * @param keys The keys to query the cache.
   *
   * @return (positions, indices, pointers, missing_keys), where positions has
   * the locations of the keys which were emplaced into the cache, pointers
   * point to the emplaced CacheKey pointers in the cache, missing_keys has the
   * keys that were not found and just inserted and indices is defined such that
   * keys[indices[:keys.size(0) - missing_keys.size(0)]] gives us the keys for
   * the found keys and keys[indices[keys.size(0) - missing_keys.size(0):]] is
   * identical to missing_keys.
   */
  virtual std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
  QueryAndReplace(torch::Tensor keys) = 0;

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
  static void ReadingCompleted(torch::Tensor pointers);

  /**
   * @brief A writer has finished writing these keys, so they can be evicted.
   * @param pointers The CacheKey pointers in the cache to unmark.
   */
  static void WritingCompleted(torch::Tensor pointers);

 protected:
  template <typename K, typename V>
  using map_t = tsl::robin_map<K, V>;
  template <typename K>
  using set_t = tsl::robin_set<K>;
  template <typename iterator>
  static auto& mutable_value_ref(iterator it) {
    return it.value();
  }
  static constexpr int kCapacityFactor = 2;

  template <typename CachePolicy>
  static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
  QueryImpl(CachePolicy& policy, torch::Tensor keys);

  template <typename CachePolicy>
  static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
  QueryAndReplaceImpl(CachePolicy& policy, torch::Tensor keys);

  template <typename CachePolicy>
  static std::tuple<torch::Tensor, torch::Tensor> ReplaceImpl(
      CachePolicy& policy, torch::Tensor keys);

  template <typename T>
  static void MoveToFront(
      std::list<T>& from, std::list<T>& to,
      typename std::list<T>::iterator it) {
    std::list<T> temp;
    // Transfer the element to temp to keep references valid.
    auto next_it = it;
    std::advance(next_it, 1);
    temp.splice(temp.begin(), from, it, next_it);
    // Move the element to the beginning of the queue.
    to.splice(to.begin(), temp);
    // The iterators and references are not invalidated.
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(it == to.begin());
  }

  int64_t capacity_;
  int64_t cache_usage_;

 private:
  template <bool write>
  static void ReadingWritingCompletedImpl(torch::Tensor pointers);
};

/**
 * @brief S3FIFO is a simple, scalable FIFObased algorithm with three static
 * queues (S3-FIFO). https://dl.acm.org/doi/pdf/10.1145/3600006.3613147
 **/
class S3FifoCachePolicy : public BaseCachePolicy {
 public:
  using map_iterator = map_t<int64_t, CacheKey*>::iterator;
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
   * @brief See BaseCachePolicy::QueryAndReplace.
   */
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
  QueryAndReplace(torch::Tensor keys);

  /**
   * @brief See BaseCachePolicy::Replace.
   */
  std::tuple<torch::Tensor, torch::Tensor> Replace(torch::Tensor keys);

  CacheKey* Read(int64_t key) {
    auto it = key_to_cache_key_.find(key);
    if (it != key_to_cache_key_.end()) {
      auto& cache_key = it->second->Increment();
      if (!cache_key.BeingWritten()) {
        return &cache_key.StartRead();
      }
    }
    return nullptr;
  }

  auto getMapSentinelValue() const { return nullptr; }

  std::pair<map_iterator, bool> Emplace(int64_t key) {
    auto [it, inserted] = key_to_cache_key_.emplace(key, getMapSentinelValue());
    bool readable = false;
    if (!inserted) {
      auto& cache_key = it->second->Increment();
      if (!cache_key.BeingWritten()) {
        cache_key.StartRead();
        readable = true;
      }
    }
    return {it, readable};
  }

  CacheKey* Insert(map_iterator it) {
    const auto key = it->first;
    const auto in_ghost_queue = ghost_set_.erase(key);
    auto& queue = in_ghost_queue ? main_queue_ : small_queue_;
    queue.push_front(CacheKey(key));
    small_queue_size_ += 1 - in_ghost_queue;
    auto cache_key_ptr = &queue.front();
    mutable_value_ref(it) = cache_key_ptr;
    return &cache_key_ptr->setPos(Evict());
  }

 private:
  int64_t EvictMainQueue() {
    while (true) {
      auto& evicted = main_queue_.back();
      if (evicted.getFreq() > 0 || evicted.InUse()) {
        evicted.Decrement();
        auto it = main_queue_.end();
        std::advance(it, -1);
        MoveToFront(main_queue_, main_queue_, it);
      } else {
        key_to_cache_key_.erase(evicted.getKey());
        const auto evicted_pos = evicted.getPos();
        main_queue_.pop_back();
        return evicted_pos;
      }
    }
  }

  int64_t EvictSmallQueue() {
    while (small_queue_size_ > small_queue_size_target_) {
      --small_queue_size_;
      auto& evicted = small_queue_.back();
      if (evicted.getFreq() > 0 || evicted.InUse()) {
        evicted.ResetFreq();
        auto it = small_queue_.end();
        std::advance(it, -1);
        MoveToFront(small_queue_, main_queue_, it);
      } else {
        const auto evicted_key = evicted.getKey();
        key_to_cache_key_.erase(evicted_key);
        const auto evicted_pos = evicted.getPos();
        small_queue_.pop_back();
        if (ghost_queue_.IsFull()) {
          ghost_set_.erase(ghost_queue_.Pop());
        }
        ghost_set_.insert(evicted_key);
        ghost_queue_.Push(evicted_key);
        return evicted_pos;
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

  std::list<CacheKey> small_queue_, main_queue_;
  CircularQueue<int64_t> ghost_queue_;
  size_t small_queue_size_target_;
  // std::list<>::size() is O(N) before the CXX11 ABI which torch enforces.
  size_t small_queue_size_;
  set_t<int64_t> ghost_set_;
  map_t<int64_t, CacheKey*> key_to_cache_key_;
};

/**
 * @brief SIEVE is a simple, scalable FIFObased algorithm with a single static
 * queue. https://www.usenix.org/system/files/nsdi24-zhang-yazhuo.pdf
 **/
class SieveCachePolicy : public BaseCachePolicy {
 public:
  using map_iterator = map_t<int64_t, CacheKey*>::iterator;
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
   * @brief See BaseCachePolicy::QueryAndReplace.
   */
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
  QueryAndReplace(torch::Tensor keys);

  /**
   * @brief See BaseCachePolicy::Replace.
   */
  std::tuple<torch::Tensor, torch::Tensor> Replace(torch::Tensor keys);

  CacheKey* Read(int64_t key) {
    auto it = key_to_cache_key_.find(key);
    if (it != key_to_cache_key_.end()) {
      auto& cache_key = it->second->SetFreq();
      if (!cache_key.BeingWritten()) {
        return &cache_key.StartRead();
      }
    }
    return nullptr;
  }

  auto getMapSentinelValue() const { return nullptr; }

  std::pair<map_iterator, bool> Emplace(int64_t key) {
    auto [it, inserted] = key_to_cache_key_.emplace(key, getMapSentinelValue());
    bool readable = false;
    if (!inserted) {
      auto& cache_key = it->second->SetFreq();
      if (!cache_key.BeingWritten()) {
        cache_key.StartRead();
        readable = true;
      }
    }
    return {it, readable};
  }

  CacheKey* Insert(map_iterator it) {
    const auto key = it->first;
    queue_.push_front(CacheKey(key));
    auto cache_key_ptr = &queue_.front();
    mutable_value_ref(it) = cache_key_ptr;
    return &cache_key_ptr->setPos(Evict());
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
    key_to_cache_key_.erase(hand_->getKey());
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
  map_t<int64_t, CacheKey*> key_to_cache_key_;
};

/**
 * @brief LeastRecentlyUsed is a simple, scalable FIFObased algorithm with a
 * single static queue.
 **/
class LruCachePolicy : public BaseCachePolicy {
 public:
  using map_iterator = map_t<int64_t, std::list<CacheKey>::iterator>::iterator;
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
   * @brief See BaseCachePolicy::QueryAndReplace.
   */
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
  QueryAndReplace(torch::Tensor keys);

  /**
   * @brief See BaseCachePolicy::Replace.
   */
  std::tuple<torch::Tensor, torch::Tensor> Replace(torch::Tensor keys);

  CacheKey* Read(int64_t key) {
    auto it = key_to_cache_key_.find(key);
    if (it != key_to_cache_key_.end()) {
      auto& cache_key = *it->second;
      MoveToFront(queue_, queue_, it->second);
      if (!cache_key.BeingWritten()) {
        return &cache_key.StartRead();
      }
    }
    return nullptr;
  }

  auto getMapSentinelValue() { return queue_.end(); }

  std::pair<map_iterator, bool> Emplace(int64_t key) {
    auto [it, inserted] = key_to_cache_key_.emplace(key, getMapSentinelValue());
    bool readable = false;
    if (!inserted) {
      auto& cache_key = *it->second;
      MoveToFront(queue_, queue_, it->second);
      if (!cache_key.BeingWritten()) {
        cache_key.StartRead();
        readable = true;
      }
    }
    return {it, readable};
  }

  CacheKey* Insert(map_iterator it) {
    const auto key = it->first;
    queue_.push_front(CacheKey(key));
    mutable_value_ref(it) = queue_.begin();
    auto cache_key_ptr = &queue_.front();
    return &cache_key_ptr->setPos(Evict());
  }

 private:
  int64_t Evict() {
    // If the cache has space, get an unused slot otherwise perform eviction.
    if (cache_usage_ < capacity_) return cache_usage_++;
    // Do not evict items that are still in use.
    while (queue_.back().InUse()) {
      auto it = queue_.end();
      std::advance(it, -1);
      // Move the last element to the front without invalidating references.
      MoveToFront(queue_, queue_, it);
    }
    const auto& cache_key = queue_.back();
    key_to_cache_key_.erase(cache_key.getKey());
    const auto pos = cache_key.getPos();
    queue_.pop_back();
    return pos;
  }

  std::list<CacheKey> queue_;
  map_t<int64_t, decltype(queue_)::iterator> key_to_cache_key_;
};

/**
 * @brief Clock (FIFO-Reinsertion) is a simple, scalable FIFObased algorithm
 * with a single static queue.
 * https://people.csail.mit.edu/saltzer/Multics/MHP-Saltzer-060508/bookcases/M00s/M0104%20074-12).PDF
 **/
class ClockCachePolicy : public BaseCachePolicy {
 public:
  using map_iterator = map_t<int64_t, CacheKey*>::iterator;
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
   * @brief See BaseCachePolicy::QueryAndReplace.
   */
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
  QueryAndReplace(torch::Tensor keys);

  /**
   * @brief See BaseCachePolicy::Replace.
   */
  std::tuple<torch::Tensor, torch::Tensor> Replace(torch::Tensor keys);

  CacheKey* Read(int64_t key) {
    auto it = key_to_cache_key_.find(key);
    if (it != key_to_cache_key_.end()) {
      auto& cache_key = it->second->SetFreq();
      if (!cache_key.BeingWritten()) {
        return &cache_key.StartRead();
      }
    }
    return nullptr;
  }

  auto getMapSentinelValue() const { return nullptr; }

  std::pair<map_iterator, bool> Emplace(int64_t key) {
    auto [it, inserted] = key_to_cache_key_.emplace(key, getMapSentinelValue());
    bool readable = false;
    if (!inserted) {
      auto& cache_key = it->second->SetFreq();
      if (!cache_key.BeingWritten()) {
        cache_key.StartRead();
        readable = true;
      }
    }
    return {it, readable};
  }

  CacheKey* Insert(map_iterator it) {
    const auto key = it->first;
    queue_.push_front(CacheKey(key));
    auto cache_key_ptr = &queue_.front();
    mutable_value_ref(it) = cache_key_ptr;
    return &cache_key_ptr->setPos(Evict());
  }

 private:
  int64_t Evict() {
    // If the cache has space, get an unused slot otherwise perform eviction.
    if (cache_usage_ < capacity_) return cache_usage_++;
    while (true) {
      auto& cache_key = queue_.back();
      if (cache_key.getFreq() || cache_key.InUse()) {
        cache_key.ResetFreq();
        auto it = queue_.end();
        std::advance(it, -1);
        MoveToFront(queue_, queue_, it);
      } else {
        key_to_cache_key_.erase(cache_key.getKey());
        const auto evicted_pos = cache_key.getPos();
        queue_.pop_back();
        return evicted_pos;
      }
    }
  }

  std::list<CacheKey> queue_;
  map_t<int64_t, CacheKey*> key_to_cache_key_;
};

}  // namespace storage
}  // namespace graphbolt

#endif  // GRAPHBOLT_CACHE_POLICY_H_
