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

  /*! \brief which backend DGL is using (pytorch, mxnet, tensorflow) */
  std::string backend;
  /*! \brief the directory containing the DGL C library */
  std::string dir;
};

#endif  // DGL_RUNTIME_ENV_H_
