/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/async_transferrer.h
 * \brief The AsyncTransferrer implementation. 
 */


#include "async_transferrer.h"

#include <dgl/runtime/device_api.h>
#include <cuda_runtime.h>
#include "../runtime/cuda/cuda_common.h"

namespace dgl {
namespace runtime {

using TransferId = AsyncTransferrer::TransferId;

struct AsyncTransferrer::Event {
  cudaEvent_t id;

  ~Event() {
    CUDA_CALL(cudaEventDestroy(id));
  }
};

AsyncTransferrer::AsyncTransferrer(
    DGLContext ctx) :
  ctx_(ctx),
  next_id_(0),
  transfers_(),
  stream_(DeviceAPI::Get(ctx_)->CreateStream(ctx_)) {
}


AsyncTransferrer::~AsyncTransferrer() {
  DeviceAPI::Get(ctx_)->FreeStream(ctx_, stream_);
}

TransferId AsyncTransferrer::CreateTransfer(
    NDArray src,
    DGLContext dst_ctx)
{
  const TransferId id = GenerateId();

  // get tensor information 
  DGLContext src_ctx = src->ctx;
  DLDataType dtype = src->dtype;
  std::vector<int64_t> shape(src->shape, src->shape+src->ndim);

  Transfer t;
  t.src = src;
  t.event.reset(new Event);
  CUDA_CALL(cudaEventCreate(&t.event->id));

  CHECK(dst_ctx == ctx_ || src_ctx == ctx_) <<
      "One side of the async copy must involve the AsyncTransfer's context";

  t.dst = NDArray::Empty(shape, dtype, dst_ctx); 

  t.dst.CopyFrom(t.src, stream_);

  CUDA_CALL(cudaEventRecord(t.event->id, static_cast<cudaStream_t>(stream_)));
  
  transfers_.emplace(id, std::move(t));

  return id;
}

NDArray AsyncTransferrer::Wait(
    const TransferId id)
{
  auto iter = transfers_.find(id);
  CHECK(iter != transfers_.end()) << "Unknown transfer: " << id;

  Transfer t = std::move(iter->second);
  transfers_.erase(iter);

  // wait for it
  CUDA_CALL(cudaEventSynchronize(t.event->id));

  return t.dst;
}

TransferId AsyncTransferrer::GenerateId()
{
  return ++next_id_;
}

}  // namespace runtime
}  // namespace dgl
