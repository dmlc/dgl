/*!
 *  Copyright (c) 2017 by Contributors
 * \file random.cc
 * \brief Random number generator interfaces
 */

#include <dgl/runtime/registry.h>
#include <dgl/runtime/random.h>

namespace dgl {
namespace runtime {

DGL_REGISTER_GLOBAL("rng._CAPI_SetSeed")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
    int seed = args[0];
    Random::SetSeed(seed);
  });

}; // namespace runtime
};  // namespace dgl
