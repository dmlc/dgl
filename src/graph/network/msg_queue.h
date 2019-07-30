/*!
 *  Copyright (c) 2019 by Contributors
 * \file msg_queue.h
 * \brief Message queue for DGL distributed training.
 */
#ifndef DGL_GRAPH_NETWORK_MSG_QUEUE_H_
#define DGL_GRAPH_NETWORK_MSG_QUEUE_H_

#include <queue>
#include <set>
#include <string>
#include <utility>  // for pair
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <functional>

namespace dgl {
namespace network {

/*!
 * \brief Message used by network communicator and message queue.
 */
struct Message {
  /*!
   * \brief message data
   */
  char* data;
  /*!
   * \brief message size
   */
  int64_t size;
  /*!
   * \brief user-defined deallocator, which can be nullptr
   */
  std::function<void(Message*)> deallocator = nullptr;
};

/*!
 * \brief Free memory buffer for message
 */
inline void msg_clear(Message* msg) { delete [] msg->data; }

/*!
 * \brief Message Queue for network communication.
 *
 * MessageQueue is FIFO queue that adopts producer/consumer model for data message. 
 * It supports one or more producer threads and one or more consumer threads. 
 * Producers invokes Add() to push data message into the queue, and consumers 
 * invokes Remove() to pop data message from queue. Add() and Remove() use two condition
 * variables to synchronize producer threads and consumer threads. Each producer 
 * invokes SignalFinished(producer_id) to claim that it is about to finish, where 
 * producer_id is an integer uniquely identify a producer thread. This signaling mechanism 
 * prevents consumers from waiting after all producers have finished their jobs. 
 *
 * MessageQueue is thread-safe.
 * 
 */
class MessageQueue {
 public:
  /*!
   * \brief MessageQueue constructor
   * \param queue_size size (bytes) of message queue
   * \param num_producers number of producers, use 1 by default
   */
  MessageQueue(int64_t queue_size /* in bytes */,
               int num_producers = 1);

  /*!
   * \brief MessageQueue deconstructor
   */
  ~MessageQueue() {}

  /*!
   * \brief Add message to the queue
   * \param msg data message
   * \param is_blocking Blocking if cannot add, else return
   * \return bytes added to the queue
   *   > 0 : size of message
   *   = 0 : no enough space for this message (when is_blocking == false)
   *   -1  : error 
   */
  int64_t Add(Message msg, bool is_blocking = true);

  /*!
   * \brief Remove message from the queue
   * \param msg pointer of data msg
   * \param is_blocking Blocking if cannot remove, else return
   * \return bytes removed from the queue
   *   > 0 : size of message
   *   = 0 : queue is empty (when is_blocking == false)
   *   -1  : error
   */
  int64_t Remove(Message* msg, bool is_blocking = true);

  /*!
   * \brief Signal that producer producer_id will no longer produce anything
   * \param producer_id An integer uniquely to identify a producer thread
   */
  void SignalFinished(int producer_id);

  /*!
   * \return true if queue is empty.
   */
  bool Empty() const;

  /*!
   * \return true if queue is empty and all num_producers have signaled.
   */
  bool EmptyAndNoMoreAdd() const;

 protected:
  /*! 
   * \brief message queue 
   */
  std::queue<Message> queue_;

  /*! 
   * \brief Size of the queue in bytes 
   */
  int64_t queue_size_;

  /*! 
   * \brief Free size of the queue 
   */
  int64_t free_size_;

  /*! 
   * \brief Used to check all producers will no longer produce anything 
   */
  size_t num_producers_;

  /*! 
   * \brief Store finished producer id 
   */
  std::set<int /* producer_id */> finished_producers_;

  /*! 
   * \brief Condition when consumer should wait 
   */
  std::condition_variable cond_not_full_;

  /*! 
   * \brief Condition when producer should wait 
   */
  std::condition_variable cond_not_empty_;

  /*! 
   * \brief Signal for exit wait 
   */
  std::atomic<bool> exit_flag_{false};

  /*! 
   * \brief Protect all above data and conditions 
   */
  mutable std::mutex mutex_;
};

}  // namespace network
}  // namespace dgl

#endif  // DGL_GRAPH_NETWORK_MSG_QUEUE_H_
