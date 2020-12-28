/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \brief This is an all in one TVM runtime file.
 *
 *   You only have to use this file to compile libtvm/src_runtime to
 *   include in your project.
 *
 *  - Copy this file into your project which depends on tvm/src runtime.
 *  - Compile with -std=c++14
 *  - Add the following include path
 *     - /path/to/tvm/src/include/
 *     - /path/to/tvm/src/3rdparty/dmlc-core/include/
 *     - /path/to/tvm/src/3rdparty/dlpack/include/
 *   - Add -lpthread -ldl to the linked library.
 *   - You are good to go.
 *   - See the Makefile in the same folder for example.
 *
 *  The include files here are presented with relative path
 *  You need to remember to change it to point to the right file.
 *
 */
#include "../../third_party/tvm/src/runtime/c_runtime_api.cc"
#include "../../third_party/tvm/src/runtime/cpu_device_api.cc"
#include "../../third_party/tvm/src/runtime/file_utils.cc"
#include "../../third_party/tvm/src/runtime/library_module.cc"
#include "../../third_party/tvm/src/runtime/module.cc"
#include "../../third_party/tvm/src/runtime/ndarray.cc"
#include "../../third_party/tvm/src/runtime/object.cc"
#include "../../third_party/tvm/src/runtime/registry.cc"
#include "../../third_party/tvm/src/runtime/thread_pool.cc"
#include "../../third_party/tvm/src/runtime/threading_backend.cc"
#include "../../third_party/tvm/src/runtime/workspace_pool.cc"

// NOTE: all the files after this are optional modules
// that you can include remove, depending on how much feature you use.

// Likely we only need to enable one of the following
// If you use Module::Load, use dso_module
// For system packed library, use system_lib_module
#include "../../third_party/tvm/src/runtime/dso_library.cc"
#include "../../third_party/tvm/src/runtime/system_library.cc"

// Graph runtime
#include "../../third_party/tvm/src/runtime/graph/graph_runtime.cc"
#include "../../third_party/tvm/src/runtime/graph/graph_runtime_factory.cc"

// Uncomment the following lines to enable RPC
// #include "../../third_party/tvm/src/runtime/rpc/rpc_session.cc"
// #include "../../third_party/tvm/src/runtime/rpc/rpc_event_impl.cc"
// #include "../../third_party/tvm/src/runtime/rpc/rpc_server_env.cc"

// These macros enables the device API when uncommented.
#define TVM_CUDA_RUNTIME 1
#define TVM_METAL_RUNTIME 1
#define TVM_OPENCL_RUNTIME 1

// Uncomment the following lines to enable Metal
// #include "../../third_party/tvm/src/runtime/metal/metal_device_api.mm"
// #include "../../third_party/tvm/src/runtime/metal/metal_module.mm"

// Uncomment the following lines to enable CUDA
// #include "../../third_party/tvm/src/runtime/cuda/cuda_device_api.cc"
// #include "../../third_party/tvm/src/runtime/cuda/cuda_module.cc"

// Uncomment the following lines to enable OpenCL
// #include "../../third_party/tvm/src/runtime/opencl/opencl_device_api.cc"
// #include "../../third_party/tvm/src/runtime/opencl/opencl_module.cc"