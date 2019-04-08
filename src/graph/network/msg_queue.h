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
 * \brief Message Queue for DGL distributed training.
 *
 * MessageQueue is a circle queue for using the ring-buffer in a 
 * producer/consumer model. It supports one or more producer 
 * threads and one or more consumer threads. Producers invokes Add()
 * to push data elements into the queue, and consumers invokes
 * Remove() to pop data elements. Add() and Remove() use two condition
 * variables to synchronize producers and consumers. Each producer invokes
 * Signal(producer_id) to claim that it is about to finish, where 
 * producer_id is an integer uniquely identify a producer thread. This
 * signaling mechanism prevents consumers from waiting after all producers
 * have finished their jobs.
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
  ~MessageQueue();

  /*!
   * \brief Add data to the message queue
   * \param src The data pointer
   * \param size The size of data
   * \param is_blocking Block function if cannot add, else return
   * \return bytes added to the queue
   *   > 0 : size of message
   *   = 0 : no enough space for this message (when is_blocking = false)
   *   - 1 : error 
   */
  int64_t Add(const char* src, int64_t size, bool is_blocking = true);

  /*!
   * \brief Add data to the message queue
   * \param src The data string
   * \param is_blocking Block function if cannot add, else return
   * \return bytes added to queue
   *   > 0 : size of message
   *   = 0 : no enough space for this message (when is_blocking = false)
   *   - 1 : error 
   */
  int64_t Add(const std::string& src, bool is_blocking = true);

  /*!
   * \brief Remove message from the queue
   * \param dest The destination data pointer
   * \param max_size Maximal size of data
   * \param is_blocking Block function if cannot remove, else return
   * \return bytes removed from queue
   *   > 0 : size of message
   *   = 0 : queue is empty
   *   - 1 : error 
   */
  int64_t Remove(char *dest, int64_t max_size, bool is_blocking = true);

  /*!
   * \brief Remove message from the queue
   * \param dest The destination data string
   * \param is_blocking Block function if cannot remove, else return
   * \return bytes removed from queue
   *   > 0 : size of message
   *   = 0 : queue is empty
   *   - 1 : error 
   */
  int64_t Remove(std::string *dest, bool is_blocking = true);

  /*!
   * \brief Signal that producer producer_id will no longer produce anything
   * \param producer_id An integer uniquely to identify a producer thread
   */
  void Signal(int producer_id);

  /*!
   * \return true if queue is empty and all num_producers have signaled.
   */
  bool EmptyAndNoMoreAdd() const;

 protected:
  typedef std::pair<int64_t /* message_start_position in queue_ */,
                    int64_t /* message_length */> MessagePosition;

  /*! 
   * \brief Pointer to the queue 
   */
  char* queue_;

  /*! 
   * \brief Size of the queue in bytes 
   */
  int64_t queue_size_;

  /*! 
   * \brief Free size of the queue 
   */
  int64_t free_size_;

  /*! 
   * \brief Location in queue_ for where to write the next element 
   * Note that we do not need read_pointer since all messages were indexed
   * by message_postions_, and the first element in message_position_ 
   * denotes where we should read
   */
  int64_t write_pointer_;

  /*! 
   * \brief Used to check all producers will no longer produce anything 
   */
  size_t num_producers_;

  /*! 
   * \brief Messages in the queue 
   */
  std::queue<MessagePosition> message_positions_;

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
