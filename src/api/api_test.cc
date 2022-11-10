/**
 *  Copyright (c) 2022 by Contributors
 * @file api/api_test.cc
 * @brief C APIs for testing FFI
 */
#include <dgl/packed_func_ext.h>
#include <dgl/runtime/container.h>
#include <dgl/runtime/ndarray.h>
#include <dgl/runtime/registry.h>

#include <thread>

namespace dgl {
namespace runtime {

// Register an internal API for testing python callback.
// It receives two arguments:
//   - The python callback function.
//   - The argument to pass to the python callback
// It returns what python callback returns
DGL_REGISTER_GLOBAL("_TestPythonCallback")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      LOG(INFO) << "Inside C API";
      PackedFunc fn = args[0];
      DGLArgs cb_args(args.values + 1, args.type_codes + 1, 1);
      fn.CallPacked(cb_args, rv);
    });

// Register an internal API for testing python callback.
// It receives two arguments:
//   - The python callback function.
//   - The argument to pass to the python callback
// It returns what python callback returns
//
// The API runs the python callback in a separate thread to test
// python GIL is properly released.
DGL_REGISTER_GLOBAL("_TestPythonCallbackThread")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      LOG(INFO) << "Inside C API";
      PackedFunc fn = args[0];
      auto thr = std::make_shared<std::thread>([fn, args, rv]() {
        LOG(INFO) << "Callback thread " << std::this_thread::get_id();
        DGLArgs cb_args(args.values + 1, args.type_codes + 1, 1);
        fn.CallPacked(cb_args, rv);
      });
      thr->join();
    });

}  // namespace runtime
}  // namespace dgl
