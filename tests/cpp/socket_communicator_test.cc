/*!
 *  Copyright (c) 2019 by Contributors
 * \file msg_queue.cc
 * \brief Message queue for DGL distributed training.
 */
#include <gtest/gtest.h>
#include <string.h>
#include <string>

#include "../src/graph/network/socket_communicator.h"

using std::string;
using dgl::network::SocketSender;
using dgl::network::SocketReceiver;

void start_client();
bool start_server();

#ifndef WIN32

#include <unistd.h>

TEST(SocketCommunicatorTest, SendAndRecv) {
  int pid = fork();
  ASSERT_GE(pid, 0);
  if (pid > 0) {
    EXPECT_TRUE(start_server());
  } else {
    start_client();
  }
}

#else   // WIN32

// Win32 doesn't have a fork() equivalent so use threads instead.

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
  const char * msg = "0123456789";
  sleep(1);
  SocketSender sender;
  sender.AddReceiver("127.0.0.1", 2049, 0);
  sender.Connect();
  sender.Send(msg, 10, 0);
  sender.Finalize();
}

bool start_server() {
  char serbuff[10];
  memset(serbuff, '\0', 10);
  SocketReceiver receiver;
  receiver.Wait("127.0.0.1", 2049, 1, 500);
  receiver.Recv(serbuff, 10);
  receiver.Finalize();
  return string("0123456789") == string(serbuff);
}
