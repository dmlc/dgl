/*!
 *  Copyright (c) 2017 by Contributors
 * \file random.cc
 * \brief Random number generator interfaces
 */

#include <dmlc/omp.h>
#include <dgl/runtime/registry.h>
#include <dgl/runtime/packed_func.h>
#include <dgl/random.h>

using namespace dgl::runtime;

namespace dgl {

DGL_REGISTER_GLOBAL("rng._CAPI_SetSeed")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
    int seed = args[0];
#pragma omp parallel for
    for (int i = 0; i < omp_get_max_threads(); ++i)
      RandomEngine::ThreadLocal()->SetSeed(seed);
  });

};  // namespace dgl
