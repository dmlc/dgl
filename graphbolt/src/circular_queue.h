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
 * @file circular_queue.h
 * @brief Circular queue implementation.
 */
#ifndef GRAPHBOLT_CIRCULAR_QUEUE_H_
#define GRAPHBOLT_CIRCULAR_QUEUE_H_

#include <memory>

namespace graphbolt {

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

  void PopN(int64_t N) {
    head_ += N;
    if (head_ >= capacity_) head_ -= capacity_;
  }

  auto Clear() { head_ = tail_; }

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

}  // namespace graphbolt

#endif  // GRAPHBOLT_CIRCULAR_QUEUE_H_
