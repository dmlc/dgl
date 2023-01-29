/**
 *  Copyright (c) 2019 by Contributors
 * @file tp_communicator.cc
 * @brief Tensorpipe Communicator for DGL distributed training.
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

bool TPSender::ConnectReceiver(const std::string &addr, int recv_id) {
  if (pipes_.find(recv_id) != pipes_.end()) {
    LOG(WARNING) << "Duplicate recv_id[" << recv_id << "]. Ignoring...";
    return true;
  }
  std::shared_ptr<Pipe> pipe;
  pipe = context->connect(addr);
  auto done = std::make_shared<std::promise<bool>>();
  tensorpipe::Message tpmsg;
  tpmsg.metadata = "dglconnect";
  pipe->write(tpmsg, [done](const tensorpipe::Error &error) {
    done->set_value(!error);
  });
  if (!done->get_future().get()) {
    DLOG(WARNING) << "Failed to connect to receiver[" << addr << "].";
    return false;
  }
  pipes_[recv_id] = pipe;
  return true;
}

void TPSender::Send(const RPCMessage &msg, int recv_id) {
  auto pipe = pipes_[recv_id];
  tensorpipe::Message tp_msg;
  std::string *zerocopy_blob_ptr = &tp_msg.metadata;
  StreamWithBuffer zc_write_strm(zerocopy_blob_ptr, true);
  zc_write_strm.Write(msg);
  int32_t nonempty_ndarray_count = zc_write_strm.buffer_list().size();
  zerocopy_blob_ptr->append(
      reinterpret_cast<char *>(&nonempty_ndarray_count), sizeof(int32_t));
  tp_msg.tensors.resize(nonempty_ndarray_count);
  // Hold the NDArray that ensure it's valid until write operation completes
  auto ndarray_holder = std::make_shared<std::vector<NDArray>>();
  ndarray_holder->resize(nonempty_ndarray_count);
  auto &buffer_list = zc_write_strm.buffer_list();
  for (size_t i = 0; i < buffer_list.size(); i++) {
    auto &ptr = buffer_list[i];
    (*ndarray_holder.get())[i] = ptr.tensor;
    tensorpipe::CpuBuffer cpu_buffer;
    cpu_buffer.ptr = ptr.data;
    tp_msg.tensors[i].buffer = cpu_buffer;
    tp_msg.tensors[i].length = ptr.size;
    if (ptr.size == 0) {
      LOG(FATAL) << "Cannot send a empty NDArray.";
    }
  }
  // Let's write blockingly in case of congestion in underlying transports.
  auto done = std::make_shared<std::promise<void>>();
  pipe->write(
      tp_msg, [ndarray_holder, recv_id, done](const tensorpipe::Error &error) {
        if (error) {
          LOG(FATAL) << "Failed to send message to " << recv_id
                     << ". Details: " << error.what();
        }
        done->set_value();
      });
  done->get_future().wait();
}

void TPSender::Finalize() {
  for (auto &&p : pipes_) {
    if (p.second) {
      p.second->close();
    }
  }
  pipes_.clear();
}

void TPReceiver::Finalize() {
  if (listener_) {
    listener_->close();
  }
  for (auto &&p : pipes_) {
    if (p.second) {
      p.second->close();
    }
  }
  pipes_.clear();
}

bool TPReceiver::Wait(const std::string &addr, int num_sender, bool blocking) {
  if (listener_) {
    LOG(WARNING) << "TPReceiver::Wait() has been called already. Ignoring...";
    return true;
  }
  LOG(INFO) << "TPReceiver starts to wait on [" << addr << "].";
  listener_ = context->listen({addr});
  listener_->accept([this](const Error &error, std::shared_ptr<Pipe> pipe) {
    OnAccepted(error, pipe);
  });
  while (blocking && (num_sender != num_connected_)) {
  }
  return true;
}

void TPReceiver::OnAccepted(const Error &error, std::shared_ptr<Pipe> pipe) {
  if (error) {
    if (error.isOfType<ListenerClosedError>()) {
      // Expected.
    } else {
      LOG(WARNING) << "Unexpected error when accepting incoming pipe: "
                   << error.what();
    }
    return;
  }

  // Accept the next connection request
  listener_->accept([this](const Error &error, std::shared_ptr<Pipe> pipe) {
    OnAccepted(error, pipe);
  });

  // read the handshake message: "dglconnect"
  pipe->readDescriptor([pipe, this](const Error &error, Descriptor descriptor) {
    if (error) {
      LOG(ERROR) << "Unexpected error when reading from accepted pipe: "
                 << error.what();
      return;
    }
    Allocation allocation;
    pipe->read(allocation, [](const Error &error) {});
    CHECK(descriptor.metadata == "dglconnect") << "Invalid connect message.";
    pipes_[num_connected_] = pipe;
    ReceiveFromPipe(pipe, queue_);
    ++num_connected_;
  });
}

void TPReceiver::ReceiveFromPipe(
    std::shared_ptr<Pipe> pipe, std::shared_ptr<RPCMessageQueue> queue) {
  pipe->readDescriptor([pipe, queue = std::move(queue)](
                           const Error &error, Descriptor descriptor) {
    if (error) {
      // Error may happen when the pipe is closed
      return;
    }
    Allocation allocation;
    CHECK_EQ(descriptor.payloads.size(), 0) << "Invalid DGL RPC Message";

    int tensorsize = descriptor.tensors.size();
    if (tensorsize > 0) {
      allocation.tensors.resize(tensorsize);
      for (size_t i = 0; i < descriptor.tensors.size(); i++) {
        tensorpipe::CpuBuffer cpu_buffer;
        cpu_buffer.ptr = new char[descriptor.tensors[i].length];
        allocation.tensors[i].buffer = cpu_buffer;
      }
    }
    pipe->read(
        allocation, [allocation, descriptor = std::move(descriptor),
                     queue = std::move(queue), pipe](const Error &error) {
          if (error) {
            // Because we always have a read event posted to the epoll,
            // Therefore when pipe is closed, error will be raised.
            // But this error is expected.
            // Other error is not expected. But we cannot identify the error
            // with each Other for now. Thus here we skip handling for all
            // errors
            return;
          }

          char *meta_msg_begin = const_cast<char *>(&descriptor.metadata[0]);
          std::vector<void *> buffer_list(descriptor.tensors.size());
          for (size_t i = 0; i < descriptor.tensors.size(); i++) {
            buffer_list[i] =
                allocation.tensors[i].buffer.unwrap<CpuBuffer>().ptr;
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

RPCStatus TPReceiver::Recv(RPCMessage *msg, int timeout) {
  return queue_->pop(msg, timeout) ? kRPCSuccess : kRPCTimeOut;
}

}  // namespace rpc
}  // namespace dgl
