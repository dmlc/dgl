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
    LOG(FATAL) << "queue_size cannot be a negative value.";
  }
  try {
    queue_ = new char[queue_size];
  } catch(const std::bad_alloc&) {
    LOG(FATAL) << "Not enough memory for message queue.";
  }
  memset(queue_, '\0', queue_size);

  queue_size_ = queue_size;
  free_size_ = queue_size;
  write_pointer_ = 0;
  num_producers_ = num_producers;
}

MessageQueue::~MessageQueue() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (nullptr != queue_) {
    delete [] queue_;
    queue_ = nullptr;
  }
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
  // Write data into buffer:
  // If there has enough space on tail of buffer, just append data
  // else, write till in the end of buffer and return to head of buffer
  message_positions_.push(std::make_pair(write_pointer_, size));
  free_size_ -= size;
  if (write_pointer_ + size <= queue_size_) {
    memcpy(&queue_[write_pointer_], src, size);
    write_pointer_ += size;
    if (write_pointer_ == queue_size_) {
      write_pointer_ = 0;
    }
  } else {
    int64_t size_partial = queue_size_ - write_pointer_;
    memcpy(&queue_[write_pointer_], src, size_partial);
    memcpy(queue_, &src[size_partial], size - size_partial);
    write_pointer_ = size - size_partial;
  }

  // not empty signal
  cond_not_empty_.notify_one();

  return size;
}

char* MessageQueue::Remove(int64_t* size, bool is_blocking) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (message_positions_.empty()) {
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
    return !message_positions_.empty() || exit_flag_.load();
  });
  if (finished_producers_.size() >= num_producers_ && 
      message_positions_.empty()) {
    *size = 0;
    return nullptr;
  }

  MessagePosition & pos = message_positions_.front();
  char* dest = new char[pos.second];

  // read from buffer:
  // if this message stores in consecutive memory, just read
  // else, read from buffer tail then return to the head
  if (pos.first + pos.second <= queue_size_) {
    memcpy(dest, &queue_[pos.first], pos.second);
  } else {
    int64_t size_partial = queue_size_ - pos.first;
    memcpy(dest, &queue_[pos.first], size_partial);
    memcpy(&dest[size_partial], queue_, pos.second - size_partial);
  }
  *size = pos.second;
  
  free_size_ += pos.second;
  message_positions_.pop();

  cond_not_full_.notify_one();

  return dest;
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
  return message_positions_.size() == 0;
}

bool MessageQueue::EmptyAndNoMoreAdd() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return message_positions_.size() == 0 &&
         finished_producers_.size() >= num_producers_;
}

}  // namespace network
}  // namespace dgl
