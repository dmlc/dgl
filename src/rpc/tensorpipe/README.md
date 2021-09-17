# Introduction to tensorpipe

## Process of setup communication:
```cpp
context = std::make_shared<tensorpipe::Context>();
// For Receiver
// Create listener to accept join request
listener = context->listen({addr});
// Accept join request and generate pipe
std::promise<std::shared_ptr<Pipe>> pipeProm;
listener->accept([&](const Error& error, std::shared_ptr<Pipe> pipe) {
    if (error) {
        LOG(WARNING) << error.what();
    }
    pipeProm.set_value(std::move(pipe));
});
std::shared_ptr<Pipe> pipe = pipeProm.get_future().get();

// For Sender
pipe = context->connect(addr);
// Note that the pipe may not be really available at this point
// For example if no listener listening the address, there won't be error raised
// The error will happen at the write/read operation. Thus we need to manually check this
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
```

## Read and Write

Message structure: https://github.com/pytorch/tensorpipe/blob/master/tensorpipe/core/message.h

There are three concepts, Message, Descriptor and Allocation. 
Message is the core struct for communication. Message contains three major field, metadata(string), payload(cpu memory buffers), tensor(cpu/gpu memory buffer, with device as attribute).

Descriptor and Allocation are for the read scenario. A typical read operation as follows

```cpp
pipe->readDescriptor(
      [](const Error& error, Descriptor descriptor) {
        // Descriptor contains metadata of the message, the data size of each payload, the device information of tensors and other metadatas other than the real buffer
        // User should allocate the proper memory based on the descriptor, and set back the allocated memory to Allocation object        
        Allocation allocation;
        // Then call pipe->read to ask pipe to receive the real buffer into allocations
        pipe->read(allocation, [](const Error& error) {});
      });
```

To send the message is much simpler
```cpp
// Resource cleaning should be handled in the callback
pipe->write(message, callback_fn)
```

## Register the underlying communication channel
There are two concept, transport and channel.
Transport is the basic component for communication like sockets, which only supports cpu buffers.
Channel is higher abstraction over transport, which can support gpu buffers, or utilize multiple transport method to acceelerate communication

Tensorpipe will try to setup the channel based on priority.

```cpp
// Register transport
auto context = std::make_shared<tensorpipe::Context>();
// uv is short for libuv, using epoll with sockets to communicate
auto transportContext = tensorpipe::transport::uv::create();
context->registerTransport(0 /* priority */, "tcp", transportContext);/
// basic channel just use the bare transport to communicate
auto basicChannel = tensorpipe::channel::basic::create();
context->registerChannel(0, "basic", basicChannel);
// Below is the mpt(multiplex transport) channel, which can use multiple uv transport to increase throughput
std::vector<std::shared_ptr<tensorpipe::transport::Context>> contexts = {
    tensorpipe::transport::uv::create(), tensorpipe::transport::uv::create(),
    tensorpipe::transport::uv::create()};
std::vector<std::shared_ptr<tensorpipe::transport::Listener>> listeners = {
    contexts[0]->listen("127.0.0.1"), contexts[1]->listen("127.0.0.1"),
    contexts[2]->listen("127.0.0.1")};
auto mptChannel = tensorpipe::channel::mpt::create(
    std::move(contexts), std::move(listeners));
context->registerChannel(10, "mpt", mptChannel);
```

There are more channels supported by tensorpipe, such as CUDA IPC (for cuda communication on the same machine), CMA(using shared memory on the same machine), CUDA GDR(using infiniband with CUDA GPUDirect for gpu buffer), CUDA Basic(using socket+seperate thread to copy buffer to CUDA memory.

Quote from tensorpipe:

Backends come in two flavors:

Transports are the connections used by the pipes to transfer control messages, and the (smallish) core payloads. They are meant to be lightweight and low-latency. The most basic transport is a simple TCP one, which should work in all scenarios. A more optimized one, for example, is based on a ring buffer allocated in shared memory, which two processes on the same machine can use to communicate by performing just a memory copy, without passing through the kernel.

Channels are where the heavy lifting takes place, as they take care of copying the (larger) tensor data. High bandwidths are a requirement. Examples include multiplexing chunks of data across multiple TCP sockets and processes, so to saturate the NIC's bandwidth. Or using a CUDA memcpy call to transfer memory from one GPU to another using NVLink.