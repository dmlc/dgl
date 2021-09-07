#include "tp_communicator.h"

#include <future>

namespace dgl {
namespace network {
void TPSender::AddReceiver(const char* addr, int recv_id) {
  std::string addr_str(addr);
  auto pipe = context->connect(addr);
  receiver_addrs_[recv_id] = addr_str;
}

bool TPSender::Connect() {
  for (auto kv : receiver_addrs_) {
    auto pipe = context->connect(kv.second);
    pipes_[kv.first] = pipe;
  }
  return true;
};

STATUS TPSender::Send(Message msg, int recv_id) {
  auto pipe = pipes_[recv_id];
  tensorpipe::Message tp_msg;
  tp_msg.payloads.resize(1);
  //   tp_msg.payloads.push_back()
  tp_msg.payloads[0].data = msg.data;
  tp_msg.payloads[0].length = msg.size;
  pipe->write(tp_msg, [&msg](const Error& error) {
    if (msg.deallocator != nullptr) {
      msg.deallocator(&msg);
    }
  });
};

void TPSender::Finalize() { context->close(); };

bool TPReceiver::Wait(const char* addr, int num_sender) {
  std::string addr_str(addr);
  std::shared_ptr<Listener> listener = context->listen({addr});
  for (int i = 0; i < num_sender; i++) {
    std::promise<std::shared_ptr<Pipe>> pipeProm;
    listener->accept([&](const Error& error, std::shared_ptr<Pipe> pipe) {
      // TP_THROW_ASSERT_IF(error) << error.what();
      pipeProm.set_value(std::move(pipe));
    });
    std::shared_ptr<Pipe> pipe = pipeProm.get_future().get();
    pipes_[i] = pipe;
  }
}

void TPReceiver::AddReceiveRequest(int send_id) {
  auto pipe = pipes_[send_id];
  auto msg_q = msg_queue_[send_id];
  pipe->readDescriptor([pipe, msg_q](const Error& error,
                                     Descriptor descriptor) {
    Allocation allocation;
    allocation.payloads.resize(1);
    allocation.payloads[0].data = new char[descriptor.payloads[0].length]();
    pipe->read(allocation, [allocation, msg_q, descriptor](const Error& error) {
      Message msg;
      msg.data = (char*)allocation.payloads[0].data;
      msg.size = descriptor.payloads[0].length;
      msg.deallocator = DefaultMessageDeleter;
      msg_q->Add(std::move(msg));
    });
  });
}

STATUS TPReceiver::Recv(Message* msg, int* send_id) {
  for (;;) {
    for (; mq_iter_ != msg_queue_.end(); ++mq_iter_) {
      // We use non-block remove here
      STATUS code = mq_iter_->second->Remove(msg, false);
      if (code == QUEUE_EMPTY) {
        continue;  // jump to the next queue
      } else {
        *send_id = mq_iter_->first;
        AddReceiveRequest(*send_id);
        ++mq_iter_;
        return code;
      }
    }
    mq_iter_ = msg_queue_.begin();
  }
};

STATUS TPReceiver::RecvFrom(Message* msg, int send_id) {
  // Get message from specified message queue
  STATUS code = msg_queue_[send_id]->Remove(msg);
  return code;
}

}  // namespace network
}  // namespace dgl
