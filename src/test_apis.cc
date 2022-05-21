#include <thread>

#include <dgl/array.h>
#include <dgl/packed_func_ext.h>
#include <dgl/random.h>
#include <dgl/runtime/container.h>

#include "./c_api_common.h"

using namespace dgl::runtime;

namespace dgl {

DGL_REGISTER_GLOBAL("test._CAPI_DGLCallbackTestAPI")
.set_body([](DGLArgs args, DGLRetValue* rv) {
  LOG(INFO) << "Inside C API";
  PackedFunc fn = args[0];
  DGLValue cb_argv[] = {DGLValue{123}};
  int cb_argt[] = {kDLInt};
  DGLArgs cb_args(cb_argv, cb_argt, 1);
  DGLRetValue cb_rv;
  fn.CallPacked(cb_args, &cb_rv);
});

DGL_REGISTER_GLOBAL("test._CAPI_DGLAsyncCallbackTestAPI")
.set_body([](DGLArgs args, DGLRetValue* rv) {
  LOG(INFO) << "Inside C API";
  PackedFunc fn = args[0];
  auto thr = std::make_shared<std::thread>(
      [fn]() {
        LOG(INFO) << "Callback thread " << std::this_thread::get_id();
        DGLValue cb_argv[] = {DGLValue{123}};
        int cb_argt[] = {kDLInt};
        DGLArgs cb_args(cb_argv, cb_argt, 1);
        DGLRetValue cb_rv;
        fn.CallPacked(cb_args, &cb_rv);
      }
    );
  thr->join();
});


}
