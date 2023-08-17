/**
 *  Copyright (c) 2019 by Contributors
 * @file socket_communicator_test.cc
 * @brief Test SocketCommunicator
 */
#include "../src/rpc/network/socket_communicator.h"

#include <gtest/gtest.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <chrono>
#include <fstream>
#include <streambuf>
#include <string>
#include <thread>
#include <vector>

#include "../src/rpc/network/msg_queue.h"

using std::string;

using dgl::network::DefaultMessageDeleter;
using dgl::network::Message;
using dgl::network::SocketReceiver;
using dgl::network::SocketSender;

const int64_t kQueueSize = 500 * 1024;
const int kThreadNum = 2;
const int kMaxTryTimes = 1024;

#ifndef WIN32

const int kNumSender = 3;
const int kNumReceiver = 3;
const int kNumMessage = 10;

const char* ip_addr[] = {
    "tcp://127.0.0.1:50091", "tcp://127.0.0.1:50092", "tcp://127.0.0.1:50093"};

static void start_client();
static void start_server(int id);

TEST(SocketCommunicatorTest, SendAndRecv) {
  // start 10 client
  std::vector<std::thread> client_thread(kNumSender);
  for (int i = 0; i < kNumSender; ++i) {
    client_thread[i] = std::thread(start_client);
  }
  // start 10 server
  std::vector<std::thread> server_thread(kNumReceiver);
  for (int i = 0; i < kNumReceiver; ++i) {
    server_thread[i] = std::thread(start_server, i);
  }
  for (int i = 0; i < kNumSender; ++i) {
    client_thread[i].join();
  }
  for (int i = 0; i < kNumReceiver; ++i) {
    server_thread[i].join();
  }
}

TEST(SocketCommunicatorTest, SendAndRecvTimeout) {
  std::atomic_bool stop{false};
  // start 1 client, connect to 1 server, send 2 messsage
  auto client = std::thread([&stop]() {
    SocketSender sender(kQueueSize, kThreadNum);
    sender.ConnectReceiver(ip_addr[0], 0);
    sender.ConnectReceiverFinalize(kMaxTryTimes);
    for (int i = 0; i < 2; ++i) {
      char* str_data = new char[9];
      memcpy(str_data, "123456789", 9);
      Message msg = {str_data, 9};
      msg.deallocator = DefaultMessageDeleter;
      EXPECT_EQ(sender.Send(msg, 0), ADD_SUCCESS);
    }
    while (!stop) {
    }
    sender.Finalize();
  });
  // start 1 server, accept 1 client, receive 2 message
  auto server = std::thread([&stop]() {
    SocketReceiver receiver(kQueueSize, kThreadNum);
    receiver.Wait(ip_addr[0], 1);
    Message msg;
    int recv_id;
    // receive 1st message
    EXPECT_EQ(receiver.RecvFrom(&msg, 0, 0), REMOVE_SUCCESS);
    EXPECT_EQ(string(msg.data, msg.size), string("123456789"));
    msg.deallocator(&msg);
    // receive 2nd message
    EXPECT_EQ(receiver.Recv(&msg, &recv_id, 0), REMOVE_SUCCESS);
    EXPECT_EQ(string(msg.data, msg.size), string("123456789"));
    msg.deallocator(&msg);
    // timed out
    EXPECT_EQ(receiver.RecvFrom(&msg, 0, 1000), QUEUE_EMPTY);
    EXPECT_EQ(receiver.Recv(&msg, &recv_id, 1000), QUEUE_EMPTY);
    stop = true;
    receiver.Finalize();
  });
  // join
  client.join();
  server.join();
}

void start_client() {
  SocketSender sender(kQueueSize, kThreadNum);
  for (int i = 0; i < kNumReceiver; ++i) {
    sender.ConnectReceiver(ip_addr[i], i);
  }
  sender.ConnectReceiverFinalize(kMaxTryTimes);
  for (int i = 0; i < kNumMessage; ++i) {
    for (int n = 0; n < kNumReceiver; ++n) {
      char* str_data = new char[9];
      memcpy(str_data, "123456789", 9);
      Message msg = {str_data, 9};
      msg.deallocator = DefaultMessageDeleter;
      EXPECT_EQ(sender.Send(msg, n), ADD_SUCCESS);
    }
  }
  for (int i = 0; i < kNumMessage; ++i) {
    for (int n = 0; n < kNumReceiver; ++n) {
      char* str_data = new char[9];
      memcpy(str_data, "123456789", 9);
      Message msg = {str_data, 9};
      msg.deallocator = DefaultMessageDeleter;
      EXPECT_EQ(sender.Send(msg, n), ADD_SUCCESS);
    }
  }
  sender.Finalize();
}

void start_server(int id) {
  sleep(5);
  SocketReceiver receiver(kQueueSize, kThreadNum);
  receiver.Wait(ip_addr[id], kNumSender);
  for (int i = 0; i < kNumMessage; ++i) {
    for (int n = 0; n < kNumSender; ++n) {
      Message msg;
      EXPECT_EQ(receiver.RecvFrom(&msg, n), REMOVE_SUCCESS);
      EXPECT_EQ(string(msg.data, msg.size), string("123456789"));
      msg.deallocator(&msg);
    }
  }
  for (int n = 0; n < kNumSender * kNumMessage; ++n) {
    Message msg;
    int recv_id;
    EXPECT_EQ(receiver.Recv(&msg, &recv_id), REMOVE_SUCCESS);
    EXPECT_EQ(string(msg.data, msg.size), string("123456789"));
    msg.deallocator(&msg);
  }
  receiver.Finalize();
}

TEST(SocketCommunicatorTest, TCPSocketBind) {
  dgl::network::TCPSocket socket;
  testing::internal::CaptureStderr();
  EXPECT_EQ(socket.Bind("127.0.0", 50001), false);
  const std::string stderr = testing::internal::GetCapturedStderr();
  EXPECT_NE(stderr.find("Invalid IP: 127.0.0"), std::string::npos);
}

#else

#include <windows.h>
#include <winsock2.h>

#pragma comment(lib, "ws2_32.lib")

void sleep(int seconds) { Sleep(seconds * 1000); }

static void start_client();
static bool start_server();

DWORD WINAPI _ClientThreadFunc(LPVOID param) {
  start_client();
  return 0;
}

DWORD WINAPI _ServerThreadFunc(LPVOID param) { return start_server() ? 1 : 0; }

TEST(SocketCommunicatorTest, SendAndRecv) {
  HANDLE hThreads[2];
  WSADATA wsaData;
  DWORD retcode, exitcode;

  srand((unsigned)time(NULL));
  int port = (rand() % (5000 - 3000 + 1)) + 3000;
  std::string ip_addr = "tcp://127.0.0.1:" + std::to_string(port);
  std::ofstream out("addr.txt");
  out << ip_addr;
  out.close();

  ASSERT_EQ(::WSAStartup(MAKEWORD(2, 2), &wsaData), 0);

  hThreads[0] =
      ::CreateThread(NULL, 0, _ClientThreadFunc, NULL, 0, NULL);  // client
  ASSERT_TRUE(hThreads[0] != NULL);
  hThreads[1] =
      ::CreateThread(NULL, 0, _ServerThreadFunc, NULL, 0, NULL);  // server
  ASSERT_TRUE(hThreads[1] != NULL);

  retcode = ::WaitForMultipleObjects(2, hThreads, TRUE, INFINITE);
  EXPECT_TRUE((retcode <= WAIT_OBJECT_0 + 1) && (retcode >= WAIT_OBJECT_0));

  EXPECT_EQ(::GetExitCodeThread(hThreads[1], &exitcode), TRUE);
  EXPECT_EQ(exitcode, 1);

  EXPECT_EQ(::CloseHandle(hThreads[0]), TRUE);
  EXPECT_EQ(::CloseHandle(hThreads[1]), TRUE);

  ::WSACleanup();
}

static void start_client() {
  std::ifstream t("addr.txt");
  std::string ip_addr(
      (std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
  t.close();
  SocketSender sender(kQueueSize, kThreadNum);
  sender.ConnectReceiver(ip_addr.c_str(), 0);
  sender.ConnectReceiverFinalize(kMaxTryTimes);
  char* str_data = new char[9];
  memcpy(str_data, "123456789", 9);
  Message msg = {str_data, 9};
  msg.deallocator = DefaultMessageDeleter;
  sender.Send(msg, 0);
  sender.Finalize();
}

static bool start_server() {
  sleep(5);
  std::ifstream t("addr.txt");
  std::string ip_addr(
      (std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
  t.close();
  SocketReceiver receiver(kQueueSize, kThreadNum);
  receiver.Wait(ip_addr.c_str(), 1);
  Message msg;
  EXPECT_EQ(receiver.RecvFrom(&msg, 0), REMOVE_SUCCESS);
  receiver.Finalize();
  return string("123456789") == string(msg.data, msg.size);
}

#endif
