/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef DGL_RPC_TENSORPIPE_QUEUE_H_
#define DGL_RPC_TENSORPIPE_QUEUE_H_

#include <dmlc/logging.h>

#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <utility>

namespace dgl {
namespace rpc {

template <typename T>
class Queue {
 public:
  // Capacity isn't used actually
  explicit Queue(int capacity = 1) : capacity_(capacity) {}

  void push(T t) {
    std::unique_lock<std::mutex> lock(mutex_);
    // while (items_.size() >= capacity_) {
    //   cv_.wait(lock);
    // }
    items_.push_back(std::move(t));
    cv_.notify_all();
  }

  bool pop(T *msg, int timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (timeout == 0) {
      DLOG(WARNING) << "Will wait infinitely until message is popped...";
      cv_.wait(lock, [this] { return items_.size() > 0; });
    } else {
      if (!cv_.wait_for(lock, std::chrono::milliseconds(timeout), [this] {
            return items_.size() > 0;
          })) {
        DLOG(WARNING) << "Times out for popping message after " << timeout
                      << " milliseconds.";
        return false;
      }
    }
    *msg = std::move(items_.front());
    items_.pop_front();
    cv_.notify_all();
    return true;
  }

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  const int capacity_;
  std::deque<T> items_;
};
}  // namespace rpc
}  // namespace dgl

#endif  // DGL_RPC_TENSORPIPE_QUEUE_H_
