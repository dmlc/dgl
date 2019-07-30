/*!
 *  Copyright (c) 2019 by Contributors
 * \file msg_queue.cc
 * \brief Message queue for DGL distributed training.
 */
#include <dmlc/logging.h>
#include <cstring>

#include "msg_queue.h"

namespace dgl {
namespace network {

using std::string;

MessageQueue::MessageQueue(int64_t queue_size, int num_producers) {
  CHECK_GE(queue_size, 0);
  CHECK_GE(num_producers, 0);
  queue_size_ = queue_size;
  free_size_ = queue_size;
  num_producers_ = num_producers;
}

int64_t MessageQueue::Add(Message msg, bool is_blocking) {
  // check if message is too long to fit into the queue
  if (msg.size > queue_size_) {
    LOG(WARNING) << "Message is larger than the queue.";
    return -1;
  }
  if (msg.size <= 0) {
    LOG(WARNING) << "Message size (" << msg.size << ") is negative or zero.";
    return -1;
  }
  std::unique_lock<std::mutex> lock(mutex_);
  if (finished_producers_.size() >= num_producers_) {
    LOG(WARNING) << "Can't add to buffer when flag_no_more is set.";
    return -1;
  }
  if (msg.size > free_size_ && !is_blocking) {
    LOG(WARNING) << "Queue is full and message lost.";
    return 0;
  }
  cond_not_full_.wait(lock, [&]() {
    return msg.size <= free_size_;
  });
  // Add data pointer to queue
  queue_.push(msg);
  free_size_ -= msg.size;
  // not empty signal
  cond_not_empty_.notify_one();

  return msg.size;
}

int64_t MessageQueue::Remove(Message* msg, bool is_blocking) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (queue_.empty()) {
    if (!is_blocking) {
      LOG(WARNING) << "Queue is empty.";
      return 0;
    }
    if (finished_producers_.size() >= num_producers_) {
      return -1;
    }
  }

  cond_not_empty_.wait(lock, [this] {
    return !queue_.empty() || exit_flag_.load();
  });
  if (finished_producers_.size() >= num_producers_ && queue_.empty()) {
    return -1;
  }

  Message & old_msg = queue_.front();
  queue_.pop();
  msg->data = old_msg.data;
  msg->size = old_msg.size;
  msg->deallocator = old_msg.deallocator;
  free_size_ += old_msg.size;
  cond_not_full_.notify_one();

  return old_msg.size;
}

void MessageQueue::SignalFinished(int producer_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  finished_producers_.insert(producer_id);
  // if all producers have finished, consumers should be
  // waken up to get this signal
  if (finished_producers_.size() >= num_producers_) {
    exit_flag_.store(true);
    cond_not_empty_.notify_all();
  }
}

bool MessageQueue::Empty() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return queue_.size() == 0;
}

bool MessageQueue::EmptyAndNoMoreAdd() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return queue_.size() == 0 &&
         finished_producers_.size() >= num_producers_;
}

}  // namespace network
}  // namespace dgl
