/*!
 *  Copyright (c) 2020 by Contributors
 * \file utils.cc
 * \brief DGL util functions
 */

#include <dmlc/omp.h>

#include <dgl/packed_func_ext.h>
#include "../c_api_common.h"

using namespace dgl::runtime;

namespace dgl {

DGL_REGISTER_GLOBAL("utils.internal._CAPI_DGLSetOMPThreads")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    int num_threads = args[0];
    omp_set_num_threads(num_threads);
  });

}  // namespace dgl
