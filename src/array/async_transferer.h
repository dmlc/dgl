/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/async_transferrer.h
 * \brief The AsyncTransferrer class for copying the data to/from the GPU on a
 * separate stream. 
 */


#ifndef DGL_ARRAY_ASYNCTRANSFERRER_H_
#define DGL_ARRAY_ASYNCTRANSFERRER_H_

#include <dgl/runtime/c_runtime_api.h>
#include <dgl/runtime/ndarray.h>
#include <unordered_map>
#include <memory>

namespace dgl {
namespace runtime {

class AsyncTransferrer {
  public:
    using TransferId = int;

    AsyncTransferrer(
        DGLContext ctx);
    ~AsyncTransferrer();

    TransferId CreateTransfer(
        NDArray data,
        DGLContext ctx);

    NDArray Wait(
        TransferId id);

  private:
    struct Event;
    struct Transfer {
      std::unique_ptr<Event> event;
      bool recorded;
      NDArray src;
      NDArray dst;
    };

    DGLContext ctx_;
    TransferId next_id_;
    std::unordered_map<TransferId, Transfer> transfers_;
    DGLStreamHandle stream_;

    TransferId GenerateId();
};

}  // namespace runtime
}  // namespace dgl

#endif
