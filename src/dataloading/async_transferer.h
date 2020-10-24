/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/async_transferer.h
 * \brief The AsyncTransferer class for copying the data to/from the GPU on a
 * separate stream. 
 */


#ifndef DGL_DATALOADING_ASYNC_TRANSFERER_H_
#define DGL_DATALOADING_ASYNC_TRANSFERER_H_

#include <dgl/runtime/c_runtime_api.h>
#include <dgl/runtime/ndarray.h>
#include <dgl/runtime/object.h>
#include <unordered_map>
#include <memory>

namespace dgl {
namespace dataloading {

class AsyncTransferer : public runtime::Object {
 public:
  using TransferId = int;

  explicit AsyncTransferer(
      DGLContext ctx);
  ~AsyncTransferer();

  // disable copying
  AsyncTransferer(
      const AsyncTransferer&) = delete;
  AsyncTransferer& operator=(
      const AsyncTransferer&) = delete;

  TransferId StartTransfer(
      runtime::NDArray data,
      DGLContext ctx);

  runtime::NDArray Wait(
      TransferId id);

  static constexpr const char* _type_key = "ndarray.AsyncTransferer";
  DGL_DECLARE_OBJECT_TYPE_INFO(AsyncTransferer, Object);

 private:
  struct Event;
  struct Transfer {
    std::unique_ptr<Event> event;
    bool recorded;
    runtime::NDArray src;
    runtime::NDArray dst;
  };

  DGLContext ctx_;
  TransferId next_id_;
  std::unordered_map<TransferId, Transfer> transfers_;
  DGLStreamHandle stream_;

  TransferId GenerateId();
};

DGL_DEFINE_OBJECT_REF(AsyncTransfererRef, AsyncTransferer);

}  // namespace dataloading
}  // namespace dgl

#endif  // DGL_DATALOADING_ASYNC_TRANSFERER_H_
