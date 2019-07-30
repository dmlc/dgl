/*!
 *  Copyright (c) 2019 by Contributors
 * \file socket_communicator_test.cc
 * \brief Test SocketCommunicator
 */
#include <gtest/gtest.h>
#include <string.h>
#include <string>

#include "../src/graph/network/msg_queue.h"
#include "../src/graph/network/socket_communicator.h"

using std::string;
using dgl::network::SocketSender;
using dgl::network::SocketReceiver;
using dgl::network::Message;

void start_client();
bool start_server();

#ifndef WIN32

#include <unistd.h>

TEST(SocketCommunicatorTest, SendAndRecv) {
  std::thread client_thread(start_client);
  start_server();
  client_thread.join();
}

#else   // WIN32

#include <windows.h>
#include <winsock2.h>

#pragma comment(lib, "ws2_32.lib")

void sleep(int seconds) {
  Sleep(seconds * 1000);
}

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

#endif  // WIN32

void start_client() {
  sleep(1); // wait server start
  SocketSender sender(500*1024);
  sender.AddReceiver("socket://127.0.0.1:50091", 0);
  sender.Connect();
  for (int i = 0; i < 100; ++i) {
    Message msg;
    msg.data = "123456789";
    msg.size = 9;
    sender.Send(msg, 0);
  }
  sender.Finalize();
}

bool start_server() {
  SocketReceiver receiver(500*1024);
  receiver.Wait("socket://127.0.0.1:50091", 1);
  int64_t size = 0;
  char* data = nullptr;
  Message msg;
  for (int i = 0; i < 100; ++i) {
    size = receiver.RecvFrom(&msg, 0);
  }
  receiver.Finalize();
  return string("123456789") == string(msg.data, msg.size);
}
