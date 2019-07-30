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
using dgl::network::Message;
using dgl::network::MessageQueue;

TEST(MessageQueueTest, AddRemove) {
  MessageQueue queue(5, 1);  // size:5, num_of_producer:1
  // msg 1
  Message msg_1;
  msg_1.data = "111";
  msg_1.size = 3;
  EXPECT_EQ(queue.Add(msg_1), 3);
  // msg 2
  Message msg_2;
  msg_2.data = "22";
  msg_2.size = 2;
  EXPECT_EQ(queue.Add(msg_2), 2);
  // msg 3
  Message msg_3;
  msg_3.data = "xxxx";
  msg_3.size = 4;
  EXPECT_EQ(0, queue.Add(msg_3, false));  // non-blocking add
  // msg 4
  Message msg_4;
  int64_t size = queue.Remove(&msg_4);
  EXPECT_EQ(size, 3);
  EXPECT_EQ(string(msg_4.data, msg_4.size), string("111"));
  // msg 5
  Message msg_5;
  size = queue.Remove(&msg_5);
  EXPECT_EQ(size, 2);
  EXPECT_EQ(string(msg_5.data, msg_5.size), string("22"));
  // msg 6
  Message msg_6;
  msg_6.data = "33333";
  msg_6.size = 5;
  EXPECT_EQ(queue.Add(msg_6), 5);
  // msg 7
  Message msg_7;
  size = queue.Remove(&msg_7);
  EXPECT_EQ(size, 5);
  EXPECT_EQ(string(msg_7.data, msg_7.size), string("33333"));
  // msg 8
  Message msg_8;
  EXPECT_EQ(0, queue.Remove(&msg_8, false));  // non-blocking remove
  // msg 9
  Message msg_9;
  msg_9.data = "666666";
  msg_9.size = 6;
  EXPECT_EQ(queue.Add(msg_9), -1);      // exceed queue size
  // msg 10
  Message msg_10;
  msg_10.data = "55555";
  msg_10.size = 5;
  EXPECT_EQ(queue.Add(msg_10), 5);
  // msg 11
  Message msg_11;
  size = queue.Remove(&msg_11);
  EXPECT_EQ(size, 5);
}

/*
TEST(MessageQueueTest, EmptyAndNoMoreAdd) {
  MessageQueue queue(5, 2);  // size:5, num_of_producer:2
  EXPECT_EQ(queue.EmptyAndNoMoreAdd(), false);
  EXPECT_EQ(queue.Empty(), true);
  queue.SignalFinished(1);
  queue.SignalFinished(1);
  EXPECT_EQ(queue.EmptyAndNoMoreAdd(), false);
  queue.SignalFinished(2);
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
  queue->SignalFinished(id);
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
}*/