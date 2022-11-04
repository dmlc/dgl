#ifndef DIST_TEST_RPC_BASE_H_
#define DIST_TEST_RPC_BASE_H_

#include <iostream>
#include <string>

#include "../../../src/rpc/rpc_msg.h"
#include "../../../src/rpc/tensorpipe/tp_communicator.h"
#include "../../cpp/common.h"

namespace {
const int kNumSender = 30;
const int kNumReceiver = 10;
const int kNumMessage = 1024;
const int kPort = 50090;
const int kSizeTensor = 10 * 1024;
const int kNumTensor = 30;

std::shared_ptr<tensorpipe::Context> InitTPContext() {
  auto context = std::make_shared<tensorpipe::Context>();
  auto transportContext = tensorpipe::transport::uv::create();
  context->registerTransport(0 /* priority */, "tcp", transportContext);
  auto basicChannel = tensorpipe::channel::basic::create();
  context->registerChannel(0 /* low priority */, "basic", basicChannel);
  return context;
}
}  // namespace

#endif
