/*!
 *  Copyright (c) 2019 by Contributors
 * \file socket_communicator_test.cc
 * \brief Test SocketCommunicator
 */
#include <gtest/gtest.h>
#include <string.h>
#include <string>
#include <thread>
#include <vector>

#include "../src/graph/network/msg_queue.h"
#include "../src/graph/network/socket_communicator.h"

#ifndef WIN32
#include <unistd.h>
#else
#include <windows.h>
#pragma comment(lib, "ws2_32.lib")
#endif

using std::string;
using dgl::network::SocketSender;
using dgl::network::SocketReceiver;
using dgl::network::Message;

static void start_client();
static void start_server(int id);

const int     kNumSender = 3;
const int     kNumReceiver = 3;
const int     kNumMessage = 10;
const int64_t kQueueSize = 500 * 1024;

const char* ip_addr[] = {
  "socket://127.0.0.1:50091",
  "socket://127.0.0.1:50092",
  "socket://127.0.0.1:50093"
};


void Sleep(int seconds) {
#ifndef WIN32
  sleep(seconds);
#else
  Sleep(seconds * 1000);
#endif
}


TEST(SocketCommunicatorTest, SendAndRecv) {
  // start 10 client
  std::vector<std::thread*> client_thread;
  for (int i = 0; i < kNumSender; ++i) {
    client_thread.push_back(new std::thread(start_client));
  }
  // start 10 server
  std::vector<std::thread*> server_thread;
  for (int i = 0; i < kNumReceiver; ++i) {
    server_thread.push_back(new std::thread(start_server, i));
  }
  for (int i = 0; i < kNumSender; ++i) {
    client_thread[i]->join();
  }
}

void start_client() {
  Sleep(2); // wait server start
  SocketSender sender(kQueueSize);
  for (int i = 0; i < kNumReceiver; ++i) {
    sender.AddReceiver(ip_addr[i], i);
  }
  sender.Connect();
  for (int i = 0; i < kNumMessage; ++i) {
    for (int n = 0; n < kNumReceiver; ++n) {
      Message msg;
      msg.data = "123456789";
      msg.size = 9;
      sender.Send(msg, n);
    }
  }
  sender.Finalize();
}

void start_server(int id) {
  SocketReceiver receiver(kQueueSize);
  receiver.Wait(ip_addr[id], kNumSender);
  for (int i = 0; i < kNumMessage; ++i) {
    for (int n = 0; n < kNumSender; ++n) {
      Message msg;
      EXPECT_EQ(receiver.RecvFrom(&msg, n), 9);
      EXPECT_EQ(string(msg.data, msg.size), string("123456789"));
    }
  }
  receiver.Finalize();
}
