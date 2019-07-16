/*!
 *  Copyright (c) 2019 by Contributors
 * \file message_queue_test.cc
 * \brief Test MessageQueue
 */
#include <gtest/gtest.h>
#include <string>
#include <vector>
#include <thread>

#include "../src/graph/network/msg_queue.h"

using std::string;
using std::vector;
using dgl::network::MessageQueue;

TEST(MessageQueueTest, AddRemove) {
  MessageQueue queue(5, 1);  // size:5, num_of_producer:1
  vector<string> data;
  data.push_back("111");
  data.push_back("22");
  int64_t size = queue.Add(data[0].c_str(), 3);
  EXPECT_EQ(size, 3);
  size = queue.Add(data[1].c_str(), 2);
  EXPECT_EQ(size, 2);
  size = queue.Add("xxxx", 4, false);  // queue is full
  EXPECT_EQ(size, 0);
  char* ptr = queue.Remove(&size);
  EXPECT_EQ(size, 3);
  EXPECT_EQ(string(ptr, 3), data[0]);
  ptr = queue.Remove(&size);
  EXPECT_EQ(size, 2);
  EXPECT_EQ(string(ptr, 2), data[1]);
}

TEST(MessageQueueTest, EmptyAndNoMoreAdd) {
  MessageQueue queue(5, 2);  // size:5, num_of_producer:2
  EXPECT_EQ(queue.Empty(), true);
  EXPECT_EQ(queue.EmptyAndNoMoreAdd(), false);
  queue.Signal(1);
  queue.Signal(1);
  EXPECT_EQ(queue.Empty(), true);
  EXPECT_EQ(queue.EmptyAndNoMoreAdd(), false);
  queue.Signal(2);
  EXPECT_EQ(queue.EmptyAndNoMoreAdd(), true);
  EXPECT_EQ(queue.Empty(), true);
}

const int kNumOfProducer = 10;
const int kNumMessage = 10;
const std::string global_data("12345");

void start_producer(MessageQueue* queue) {
  for (int i = 0; i < kNumMessage; ++i) {
    int size = queue->Add(global_data.c_str(), 5);
    EXPECT_EQ(size, 5);
  }
}

TEST(MessageQueueTest, MultiThread) {
  MessageQueue queue(1000, kNumOfProducer);  // size: 1000, num_of_producer: 10
  // start 10 thread for producer
  vector<std::thread*> thread_pool;
  for (int i = 0; i < kNumOfProducer; ++i) {
    thread_pool.push_back(new std::thread(start_producer, &queue));
  }
  for (int i = 0; i < kNumOfProducer * kNumMessage; ++i) {
    int64_t size = 0;
    char* data = queue.Remove(&size);
    EXPECT_EQ(size, 5);
    EXPECT_EQ(string(data, 5), string("12345"));
  }
  for (int i = 0; i < kNumOfProducer; ++i) {
    thread_pool[i]->join();
  }
}