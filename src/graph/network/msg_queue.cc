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
  if (queue_size < 0) {
    LOG(FATAL) << "queue_size cannot be a negative number.";
  }
  if (num_producers < 0) {
    LOG(FATAL) << "num_producers cannot be a negative number.";
  }

  queue_size_ = queue_size;
  free_size_ = queue_size;
  num_producers_ = num_producers;
}

int64_t MessageQueue::Add(const char* src, int64_t size, bool is_blocking) {
  // check if message is too long to fit into the queue
  if (size > queue_size_) {
    LOG(ERROR) << "Message is larger than the queue.";
    return -1;
  }
  if (size <= 0) {
    LOG(ERROR) << "Message size (" << size << ") is negative or zero.";
    return -1;
  }
  std::unique_lock<std::mutex> lock(mutex_);
  if (finished_producers_.size() >= num_producers_) {
    LOG(ERROR) << "Can't add to buffer when flag_no_more is set.";
    return -1;
  }
  if (size > free_size_ && !is_blocking) {
    LOG(WARNING) << "Queue is full and message lost.";
    return 0;
  }
  cond_not_full_.wait(lock, [&]() {
    return size <= free_size_;
  });
  // Add data pointer to queue
  queue_.push(std::make_pair(const_cast<char*>(src), size));
  free_size_ -= size;
  // not empty signal
  cond_not_empty_.notify_one();

  return size;
}

char* MessageQueue::Remove(int64_t* size, bool is_blocking) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (queue_.empty()) {
    if (!is_blocking) {
      *size = 0;
      return nullptr;
    }
    if (finished_producers_.size() >= num_producers_) {
      *size = 0;
      return nullptr;
    }
  }

  cond_not_empty_.wait(lock, [this] {
    return !queue_.empty() || exit_flag_.load();
  });
  if (finished_producers_.size() >= num_producers_ && 
      queue_.empty()) {
    *size = 0;
    return nullptr;
  }

  Message & msg = queue_.front();
  queue_.pop();
  *size = msg.second;
  free_size_ += msg.second;
  cond_not_full_.notify_one();

  return msg.first;
}

void MessageQueue::Signal(int producer_id) {
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
