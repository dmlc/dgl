/*!
 *  Copyright (c) 2019 by Contributors
 * \file msg_queue.cc
 * \brief Message queue for DGL distributed training.
 */
#include <gtest/gtest.h>
#include <string.h>
#include <unistd.h>
#include <string>

#include "../src/graph/network/socket_communicator.h"

using std::string;
using dgl::network::SocketSender;
using dgl::network::SocketReceiver;

TEST(SocketCommunicatorTest, SendAndRecv) {
  int pid = fork();
  const char * msg = "0123456789";
  ASSERT_GE(pid, 0);
  if (pid > 0) {  // parent: server
    char serbuff[10];
    memset(serbuff, '\0', 10);
    SocketReceiver receiver;
    receiver.Wait("127.0.0.1", 50051, 1, 500);
    receiver.Recv(serbuff, 10);
    ASSERT_EQ(string("0123456789"), string(serbuff, 10));
    receiver.Finalize();
  } else {  // child: client
    sleep(1);
    SocketSender sender;
    sender.AddReceiver("127.0.0.1", 50051, 0);
    sender.Connect();
    sender.Send(msg, 10, 0);
    sender.Finalize();
  }
}