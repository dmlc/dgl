/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/async_transferer.cc
 * \brief The AsyncTransferer implementation.
 */


#include "async_transferer.h"

#include <dgl/packed_func_ext.h>
#include <dgl/runtime/registry.h>
#include <dgl/runtime/device_api.h>
#include <cuda_runtime.h>
#include <vector>
#include <utility>
#include "../../runtime/cuda/cuda_common.h"

namespace dgl {
namespace runtime {

using TransferId = AsyncTransferer::TransferId;

struct AsyncTransferer::Event {
  cudaEvent_t id;

  ~Event() {
    CUDA_CALL(cudaEventDestroy(id));
  }
};

AsyncTransferer::AsyncTransferer(
    DGLContext ctx) :
  ctx_(ctx),
  next_id_(0),
  transfers_(),
  stream_(DeviceAPI::Get(ctx_)->CreateStream(ctx_)) {
}


AsyncTransferer::~AsyncTransferer() {
  DeviceAPI::Get(ctx_)->FreeStream(ctx_, stream_);
}

TransferId AsyncTransferer::StartTransfer(
    NDArray src,
    DGLContext dst_ctx) {
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

NDArray AsyncTransferer::Wait(
    const TransferId id) {
  auto iter = transfers_.find(id);
  CHECK(iter != transfers_.end()) << "Unknown transfer: " << id;

  Transfer t = std::move(iter->second);
  transfers_.erase(iter);

  // wait for it
  CUDA_CALL(cudaEventSynchronize(t.event->id));

  return t.dst;
}

TransferId AsyncTransferer::GenerateId() {
  return ++next_id_;
}

DGL_REGISTER_GLOBAL("ndarray._CAPI_DGLAsyncTransfererCreate")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    DGLContext ctx = rv[0];
    *rv = AsyncTransfererRef(std::make_shared<AsyncTransferer>(ctx));
});

DGL_REGISTER_GLOBAL("ndarray._CAPI_DGLAsyncTransfererStartTransfer")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  AsyncTransfererRef ref = args[0];
  NDArray array = args[1];
  DGLContext ctx = args[2];
  int id = ref->StartTransfer(array, ctx);
  *rv = id;
});

DGL_REGISTER_GLOBAL("ndarray._CAPI_DGLAsyncTransfererWait")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  AsyncTransfererRef ref = args[0];
  int id = args[1];
  NDArray arr = ref->Wait(id);
  *rv = arr;
});

}  // namespace runtime
}  // namespace dgl
