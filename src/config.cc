/*!
 *  Copyright (c) 2017 by Contributors
 * \file config.cc
 * \brief Global Config for dgl
 */

#include <dgl/random.h>
#include <dgl/runtime/packed_func.h>
#include <dgl/runtime/registry.h>
#include <dmlc/omp.h>

using namespace dgl::runtime;

namespace dgl {

DGL_REGISTER_GLOBAL("backend._CAPI_SetCPUAlignment")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      int alignment = args[0];
      dgl::runtime::kAllocAlignment = alignment;
    });
};  // namespace dgl
