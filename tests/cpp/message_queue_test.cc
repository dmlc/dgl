/*!
 *  Copyright (c) 2019 by Contributors
 * \file msg_queue.cc
 * \brief Message queue for DGL distributed training.
 */
#include <gtest/gtest.h>
#include <string>
#include <thread>
#include <vector>

#include "../src/graph/network/msg_queue.h"

using std::string;
using dgl::network::MessageQueue;

TEST(MessageQueueTest, AddRemove) {
  MessageQueue queue(5, 1);  // size:5, num_of_producer:1
  queue.Add("111", 3);
  queue.Add("22", 2);
  EXPECT_EQ(0, queue.Add("xxxx", 4, false));  // non-blocking add
  int64_t size = 0;
  char* data = queue.Remove(&size);
  EXPECT_EQ(string(data, size), string("111"));
  data = queue.Remove(&size);
  EXPECT_EQ(string(data, size), string("22"));
  queue.Add("33333", 5);
  data = queue.Remove(&size);
  EXPECT_EQ(string(data, size), string("33333"));
  EXPECT_EQ(nullptr, queue.Remove(&size, false));  // non-blocking remove
  EXPECT_EQ(queue.Add("666666", 6), -1);           // exceed queue size
  queue.Add("55555", 5);
  queue.Remove(&size);
  EXPECT_EQ(size, 5);
}

TEST(MessageQueueTest, EmptyAndNoMoreAdd) {
  MessageQueue queue(5, 2);  // size:5, num_of_producer:2
  EXPECT_EQ(queue.EmptyAndNoMoreAdd(), false);
  EXPECT_EQ(queue.Empty(), true);
  queue.Signal(1);
  queue.Signal(1);
  EXPECT_EQ(queue.EmptyAndNoMoreAdd(), false);
  queue.Signal(2);
  EXPECT_EQ(queue.EmptyAndNoMoreAdd(), true);
  int64_t size = 0;
  EXPECT_EQ(queue.Remove(&size), nullptr);
}

const int kNumOfProducer = 1;
const int kNumOfMessage = 1;

void start_add(MessageQueue* queue, int id) {
  for (int i = 0; i < kNumOfMessage; ++i) {
    int size = queue->Add("apple", 5);
    EXPECT_EQ(size, 5);
  }
  queue->Signal(id);
}

TEST(MessageQueueTest, MultiThread) {
  MessageQueue queue(100000, kNumOfProducer);
  EXPECT_EQ(queue.EmptyAndNoMoreAdd(), false);
  EXPECT_EQ(queue.Empty(), true);
  std::vector<std::thread*> thread_pool;
  for (int i = 0; i < kNumOfProducer; ++i) {
    thread_pool.push_back(new std::thread(start_add, &queue, i));
  }
  for (int i = 0; i < kNumOfProducer*kNumOfMessage; ++i) {
    int64_t size = 0;
    char* data = queue.Remove(&size);
    EXPECT_EQ(size, 5);
    EXPECT_EQ(string(data, size), string("apple"));
  }
  for (int i = 0; i < kNumOfProducer; ++i) {
    thread_pool[i]->join();
  }
  EXPECT_EQ(queue.EmptyAndNoMoreAdd(), true);
}