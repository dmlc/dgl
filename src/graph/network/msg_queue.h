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

namespace dgl {
namespace network {

/*!
 * \brief Message Queue for network communication.
 *
 * MessageQueue is FIFO queue that adopts producer/consumer model for data pointer. 
 * It supports one or more producer threads and one or more consumer threads. 
 * Producers invokes Add() to push data pointers into the queue, and consumers 
 * invokes Remove() to pop data pointers. Add() and Remove() use two condition
 * variables to synchronize producer threads and consumer threads. Each producer 
 * invokes Signal(producer_id) to claim that it is about to finish, where producer_id 
 * is an integer uniquely identify a producer thread. This signaling mechanism 
 * prevents consumers from waiting after all producers have finished their jobs. 
 * Note that MessageQueue uses zero-copy technology to avoid large memory copy.
 *
 * MessageQueue is thread-safe.
 * 
 */
class MessageQueue {
 public:
  /*!
   * \brief MessageQueue constructor
   * \param queue_size size of message queue
   * \param num_producers number of producers, use 1 by default
   */
  MessageQueue(int64_t queue_size /* in bytes */,
               int num_producers = 1);

  /*!
   * \brief MessageQueue deconstructor
   */
  ~MessageQueue() {}

  /*!
   * \brief Add data pointer to the message queue
   * \param src data pointer
   * \param size size of data
   * \param is_blocking Blocking if cannot add, else return
   * \return bytes added to the queue
   *   > 0 : size of message
   *   = 0 : no enough space for this message (when is_blocking == false)
   *   - 1 : error 
   */
  int64_t Add(const char* src, int64_t size, bool is_blocking = true);

  /*!
   * \brief Remove data pointer from the queue
   * \param size size of data
   *   > 0 : size of data
   *   = 0 : queue is empty (is_blocking = false)
   *   < 0 : error
   * \param is_blocking Blocking if cannot remove, else return
   * \return pointer of data buffer
   */
  char* Remove(int64_t* size, bool is_blocking = true);

  /*!
   * \brief Signal that producer producer_id will no longer produce anything
   * \param producer_id An integer uniquely to identify a producer thread
   */
  void Signal(int producer_id);

  /*!
   * \return true if queue is empty.
   */
  bool Empty() const;

  /*!
   * \return true if queue is empty and all num_producers have signaled.
   */
  bool EmptyAndNoMoreAdd() const;

 protected:
  typedef std::pair<char*   /* data pointer */,
                    int64_t /* data size */> Message;

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
