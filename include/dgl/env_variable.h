/**
 *  Copyright (c) 2023 by Contributors
 * @file dgl/env_variable.h
 * @brief Class about envrionment variables.
 */
#ifndef DGL_ENV_VARIABLE_H_
#define DGL_ENV_VARIABLE_H_

#include <cstdlib>

namespace dgl {

static const char* kDGLParallelForGrainSize =
    std::getenv("DGL_PARALLEL_FOR_GRAIN_SIZE");

}  // namespace dgl

#endif  // DGL_ENV_VARIABLE_H_
