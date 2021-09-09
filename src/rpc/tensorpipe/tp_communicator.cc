/*!
 *  Copyright (c) 2019 by Contributors
 * \file tp_communicator.cc
 * \brief Tensorpipe Communicator for DGL distributed training.
 */

#include "tp_communicator.h"

#include <time.h>
#include <unistd.h>

#include <future>
#include <memory>
#include <utility>

#include "../rpc.h"

namespace dgl {
namespace rpc {

using namespace tensorpipe;

void TPSender::AddReceiver(const std::string& addr, int recv_id) {
  receiver_addrs_[recv_id] = addr;
}

bool TPSender::Connect() {
  for (auto kv : receiver_addrs_) {
    std::shared_ptr<Pipe> pipe;
    for (;;) {
      pipe = context->connect(kv.second);
      std::promise<bool> done;
      tensorpipe::Message tpmsg;
      tpmsg.metadata = "dglconnect";
      pipe->write(tpmsg, [&done](const tensorpipe::Error& error) {
        if (error) {
          done.set_value(false);
        } else {
          done.set_value(true);
        }
      });
      if (done.get_future().get()) {
        break;
      } else {
        sleep(5);
        LOG(INFO) << "Cannot connect to remove server. Wait to retry";
      }
    }
    pipes_[kv.first] = pipe;
  }
  return true;
}

void TPSender::Send(RPCMessage msg, int recv_id) {
  auto pipe = pipes_[recv_id];
  tensorpipe::Message tp_msg;
  std::string* zerocopy_blob_ptr = &tp_msg.metadata;
  StreamWithBuffer zc_write_strm(zerocopy_blob_ptr, true);
  zc_write_strm.Write(msg);
  int32_t nonempty_ndarray_count = zc_write_strm.buffer_list().size();
  zerocopy_blob_ptr->append(reinterpret_cast<char*>(&nonempty_ndarray_count),
                            sizeof(int32_t));
  tp_msg.payloads.resize(nonempty_ndarray_count);
  auto ndarray_holder = std::make_shared<std::vector<NDArray>>();
  ndarray_holder->resize(nonempty_ndarray_count);
  auto& buffer_list = zc_write_strm.buffer_list();
  for (int i = 0; i < buffer_list.size(); i++) {
    auto& ptr = buffer_list[i];
    (*ndarray_holder.get())[i] = ptr.tensor;
    tp_msg.payloads[i].data = ptr.data;
    tp_msg.payloads[i].length = ptr.size;
    if (ptr.size == 0) {
      LOG(FATAL) << "Cannot send a empty NDArray.";
    }
  }
  pipe->write(tp_msg, [ndarray_holder](const tensorpipe::Error& error) {
    if (error) {
      LOG(FATAL) << error.what();
    }
  });
}

void TPSender::Finalize() {}
void TPReceiver::Finalize() {}

bool TPReceiver::Wait(const std::string& addr, int num_sender) {
  std::string addr_str(addr);
  listener = context->listen({addr});
  for (int i = 0; i < num_sender; i++) {
    std::promise<std::shared_ptr<Pipe>> pipeProm;
    listener->accept([&](const Error& error, std::shared_ptr<Pipe> pipe) {
      if (error) {
        LOG(WARNING) << error.what();
      }
      pipeProm.set_value(std::move(pipe));
    });
    std::shared_ptr<Pipe> pipe = pipeProm.get_future().get();
    std::promise<bool> checkConnect;
    pipe->readDescriptor(
      [pipe, &checkConnect](const Error& error, Descriptor descriptor) {
        Allocation allocation;
        checkConnect.set_value(descriptor.metadata == "dglconnect");
        pipe->read(allocation, [](const Error& error) {});
      });
    CHECK(checkConnect.get_future().get()) << "Invalid connect message.";
    pipes_[i] = pipe;
    ReceiveFromPipe(pipe, queue_);
  }
  return true;
}

void TPReceiver::ReceiveFromPipe(std::shared_ptr<Pipe> pipe,
                                 std::shared_ptr<RPCMessageQueue> queue) {
  pipe->readDescriptor([pipe, queue = std::move(queue)](const Error& error,
                                                        Descriptor descriptor) {
    if (error) {
      // Error may happen when the pipe is closed
      return;
    }
    Allocation allocation;
    int payloadsize = descriptor.payloads.size();
    if (descriptor.payloads.size() > 0) {
      allocation.payloads.resize(payloadsize);
      for (int i = 0; i < descriptor.payloads.size(); i++) {
        allocation.payloads[i].data = new char[descriptor.payloads[i].length];
      }
    }
    pipe->read(allocation, [allocation, descriptor, queue = std::move(queue),
                            pipe](const Error& error) {
      if (error) {
        // Error may happen when the pipe is closed
        return;
      }

      char* meta_msg_begin = const_cast<char*>(&descriptor.metadata[0]);
      std::vector<void*> buffer_list(descriptor.payloads.size());
      for (int i = 0; i < descriptor.payloads.size(); i++) {
        buffer_list[i] = allocation.payloads[i].data;
      }
      StreamWithBuffer zc_read_strm(
        meta_msg_begin, descriptor.metadata.size() - sizeof(int32_t),
        buffer_list);
      RPCMessage msg;
      zc_read_strm.Read(&msg);
      queue->push(msg);
      TPReceiver::ReceiveFromPipe(pipe, queue);
    });
  });
}

void TPReceiver::Recv(RPCMessage* msg) { *msg = std::move(queue_->pop()); }

}  // namespace rpc
}  // namespace dgl
