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

void Sleep(int seconds) {
#ifndef WIN32
  sleep(seconds);
#else
  Sleep(seconds * 1000);
#endif
}

using std::string;
using dgl::network::SocketSender;
using dgl::network::SocketReceiver;
using dgl::network::Message;

#ifndef WIN32

const int     kNumSender = 3;
const int     kNumReceiver = 3;
const int     kNumMessage = 10;
const int64_t kQueueSize = 500 * 1024;

const char* ip_addr[] = {
  "socket://127.0.0.1:50091",
  "socket://127.0.0.1:50092",
  "socket://127.0.0.1:50093"
};

static void start_client();
static void start_server(int id);

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

#else

#include <windows.h>
#include <winsock2.h>

#pragma comment(lib, "ws2_32.lib")

DWORD WINAPI _ClientThreadFunc(LPVOID param) {
  start_client();
  return 0;
}

DWORD WINAPI _ServerThreadFunc(LPVOID param) {
  return start_server() ? 1 : 0;
}

TEST(SocketCommunicatorTest, SendAndRecv) {
  HANDLE hThreads[2];
  WSADATA wsaData;
  DWORD retcode, exitcode;

  ASSERT_EQ(::WSAStartup(MAKEWORD(2, 2), &wsaData), 0);

  hThreads[0] = ::CreateThread(NULL, 0, _ClientThreadFunc, NULL, 0, NULL);  // client
  ASSERT_TRUE(hThreads[0] != NULL);
  hThreads[1] = ::CreateThread(NULL, 0, _ServerThreadFunc, NULL, 0, NULL);  // server
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
  const char * msg = "123456789";
  sleep(1);
  SocketSender sender;
  sender.AddReceiver("127.0.0.1", 2049, 0);
  sender.Connect();
  sender.Send(msg, 9, 0);
  sender.Finalize();
}

static bool start_server() {
  char serbuff[10];
  memset(serbuff, '\0', 10);
  SocketReceiver receiver;
  receiver.Wait("127.0.0.1", 2049, 1, 500 * 1024);
  receiver.Recv(serbuff, 9);
  receiver.Finalize();
  return string("123456789") == string(serbuff);
}

#endif

