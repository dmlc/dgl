/*!
 *  Copyright (c) 2017 by Contributors
 * \file random.cc
 * \brief Random number generator interfaces
 */

#include <dmlc/omp.h>
#include <dgl/runtime/registry.h>
#include <dgl/runtime/packed_func.h>
#include <dgl/random.h>
#include <dgl/array.h>

using namespace dgl::runtime;

namespace dgl {

DGL_REGISTER_GLOBAL("rng._CAPI_SetSeed")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
    const int seed = args[0];
#pragma omp parallel for
    for (int i = 0; i < omp_get_max_threads(); ++i)
      RandomEngine::ThreadLocal()->SetSeed(seed);
  });

DGL_REGISTER_GLOBAL("rng._CAPI_Choice")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
    const int64_t num = args[0];
    const int64_t population = args[1];
    const NDArray prob = args[2];
    const bool replace = args[3];
    const int bits = args[4];
    CHECK(bits == 32 || bits == 64)
      << "Supported bit widths are 32 and 64, but got " << bits << ".";
    if (aten::IsNullArray(prob)) {
      if (bits == 32) {
        *rv = RandomEngine::ThreadLocal()->UniformChoice<int32_t>(num, population, replace);
      } else {
        *rv = RandomEngine::ThreadLocal()->UniformChoice<int64_t>(num, population, replace);
      }
    } else {
      if (bits == 32) {
        ATEN_FLOAT_TYPE_SWITCH(prob->dtype, FloatType, "probability", {
          *rv = RandomEngine::ThreadLocal()->Choice<int32_t, FloatType>(num, prob, replace);
        });
      } else {
        ATEN_FLOAT_TYPE_SWITCH(prob->dtype, FloatType, "probability", {
          *rv = RandomEngine::ThreadLocal()->Choice<int64_t, FloatType>(num, prob, replace);
        });
      }
    }
  });

};  // namespace dgl
