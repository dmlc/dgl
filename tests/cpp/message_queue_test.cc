/*!
 *  Copyright (c) 2019 by Contributors
 * \file msg_queue.cc
 * \brief Message queue for DGL distributed training.
 */
#include <gtest/gtest.h>
#include <string>

#include "../src/graph/network/msg_queue.h"

using std::string;
using dgl::network::MessageQueue;

TEST(MessageQueueTest, AddRemove) {
  MessageQueue queue(5, 1);  // size:5, num_of_producer:1
  char buff[10];
  queue.Add("111", 3);
  queue.Add("22", 2);
  EXPECT_EQ(0, queue.Add("xxxx", 4, false));  // non-blocking add
  queue.Remove(buff, 3);
  EXPECT_EQ(string(buff, 3), string("111"));
  queue.Remove(buff, 2);
  EXPECT_EQ(string(buff, 2), string("22"));
  queue.Add("33333", 5);
  queue.Remove(buff, 5);
  EXPECT_EQ(string(buff, 5), string("33333"));
  EXPECT_EQ(0, queue.Remove(buff, 10, false));  // non-blocking remove
  EXPECT_EQ(queue.Add("666666", 6), -1);  // exceed buffer size
  queue.Add("55555", 5);
  EXPECT_EQ(queue.Remove(buff, 3), -1);  // message too long
}

TEST(MessageQueueTest, EmptyAndNoMoreAdd) {
  MessageQueue queue(5, 2);  // size:5, num_of_producer:2
  char buff[10];
  EXPECT_EQ(queue.EmptyAndNoMoreAdd(), false);
  queue.Signal(1);
  queue.Signal(1);
  EXPECT_EQ(queue.EmptyAndNoMoreAdd(), false);
  queue.Signal(2);
  EXPECT_EQ(queue.EmptyAndNoMoreAdd(), true);
  EXPECT_EQ(queue.Remove(buff, 5), 0);
}