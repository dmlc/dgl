/*!
 *  Copyright (c) 2017 by Contributors
 * \file dgl/runtime/env.h
 * \brief Structure for holding DGL global environment variables
 */

#ifndef DGL_RUNTIME_ENV_H_
#define DGL_RUNTIME_ENV_H_

#include <string>

/*!
 * \brief Global environment variables.
 */
struct Env {
  static Env* Global() {
    static Env inst;
    return &inst;
  }
  /*! \brief the path to the tensoradapter library */
  std::string ta_path;
};

#endif  // DGL_RUNTIME_ENV_H_
