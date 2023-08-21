/**
 *  Copyright (c) 2019 by Contributors
 * @file msg_queue.cc
 * @brief Message queue for DGL distributed training.
 */
#include <gtest/gtest.h>

#include <string>
#include <thread>
#include <vector>

#include "../src/rpc/network/msg_queue.h"

using dgl::network::Message;
using dgl::network::MessageQueue;
using std::string;

TEST(MessageQueueTest, AddRemove) {
  MessageQueue queue(5, 1);  // size:5, num_of_producer:1
  // msg 1
  std::string str_1("111");
  Message msg_1 = {const_cast<char*>(str_1.data()), 3};
  EXPECT_EQ(queue.Add(msg_1), ADD_SUCCESS);
  // msg 2
  std::string str_2("22");
  Message msg_2 = {const_cast<char*>(str_2.data()), 2};
  EXPECT_EQ(queue.Add(msg_2), ADD_SUCCESS);
  // msg 3
  std::string str_3("xxxx");
  Message msg_3 = {const_cast<char*>(str_3.data()), 4};
  EXPECT_EQ(queue.Add(msg_3, false), QUEUE_FULL);
  // msg 4
  Message msg_4;
  EXPECT_EQ(queue.Remove(&msg_4), REMOVE_SUCCESS);
  EXPECT_EQ(string(msg_4.data, msg_4.size), string("111"));
  // msg 5
  Message msg_5;
  EXPECT_EQ(queue.Remove(&msg_5), REMOVE_SUCCESS);
  EXPECT_EQ(string(msg_5.data, msg_5.size), string("22"));
  // msg 6
  std::string str_6("33333");
  Message msg_6 = {const_cast<char*>(str_6.data()), 5};
  EXPECT_EQ(queue.Add(msg_6), ADD_SUCCESS);
  // msg 7
  Message msg_7;
  EXPECT_EQ(queue.Remove(&msg_7), REMOVE_SUCCESS);
  EXPECT_EQ(string(msg_7.data, msg_7.size), string("33333"));
  // msg 8
  Message msg_8;
  EXPECT_EQ(queue.Remove(&msg_8, false), QUEUE_EMPTY);  // non-blocking remove
  // msg 9
  std::string str_9("666666");
  Message msg_9 = {const_cast<char*>(str_9.data()), 6};
  EXPECT_EQ(queue.Add(msg_9), MSG_GT_SIZE);  // exceed queue size
  // msg 10
  std::string str_10("55555");
  Message msg_10 = {const_cast<char*>(str_10.data()), 5};
  EXPECT_EQ(queue.Add(msg_10), ADD_SUCCESS);
  // msg 11
  Message msg_11;
  EXPECT_EQ(queue.Remove(&msg_11), REMOVE_SUCCESS);
}

TEST(MessageQueueTest, EmptyAndNoMoreAdd) {
  MessageQueue queue(5, 2);  // size:5, num_of_producer:2
  EXPECT_EQ(queue.EmptyAndNoMoreAdd(), false);
  EXPECT_EQ(queue.Empty(), true);
  queue.SignalFinished(1);
  queue.SignalFinished(1);
  EXPECT_EQ(queue.EmptyAndNoMoreAdd(), false);
  queue.SignalFinished(2);
  EXPECT_EQ(queue.EmptyAndNoMoreAdd(), true);
}

const int kNumOfProducer = 100;
const int kNumOfMessage = 100;

std::string str_apple("apple");

void start_add(MessageQueue* queue, int id) {
  for (int i = 0; i < kNumOfMessage; ++i) {
    Message msg = {const_cast<char*>(str_apple.data()), 5};
    EXPECT_EQ(queue->Add(msg), ADD_SUCCESS);
  }
  queue->SignalFinished(id);
}

TEST(MessageQueueTest, MultiThread) {
  MessageQueue queue(100000, kNumOfProducer);
  EXPECT_EQ(queue.EmptyAndNoMoreAdd(), false);
  EXPECT_EQ(queue.Empty(), true);
  std::vector<std::thread> thread_pool(kNumOfProducer);
  for (int i = 0; i < kNumOfProducer; ++i) {
    thread_pool[i] = std::thread(start_add, &queue, i);
  }
  for (int i = 0; i < kNumOfProducer * kNumOfMessage; ++i) {
    Message msg;
    EXPECT_EQ(queue.Remove(&msg), REMOVE_SUCCESS);
    EXPECT_EQ(string(msg.data, msg.size), string("apple"));
  }
  for (int i = 0; i < kNumOfProducer; ++i) {
    thread_pool[i].join();
  }
  EXPECT_EQ(queue.EmptyAndNoMoreAdd(), true);
}
