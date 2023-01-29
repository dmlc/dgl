/**
 *  Copyright (c) 2016 by Contributors
 * @file runtime_base.h
 * @brief Base of all C APIs
 */
#ifndef DGL_RUNTIME_RUNTIME_BASE_H_
#define DGL_RUNTIME_RUNTIME_BASE_H_

#include <dgl/runtime/c_runtime_api.h>

#include <stdexcept>

/** @brief  macro to guard beginning and end section of all functions */
#define API_BEGIN() try {
/** @brief every function starts with API_BEGIN();
     and finishes with API_END() or API_END_HANDLE_ERROR */
#define API_END()                           \
  }                                         \
  catch (std::runtime_error & _except_) {   \
    return DGLAPIHandleException(_except_); \
  }                                         \
  return 0;  // NOLINT(*)
/**
 * @brief every function starts with API_BEGIN();
 *   and finishes with API_END() or API_END_HANDLE_ERROR
 *   The finally clause contains procedure to cleanup states when an error
 * happens.
 */
#define API_END_HANDLE_ERROR(Finalize)      \
  }                                         \
  catch (std::runtime_error & _except_) {   \
    Finalize;                               \
    return DGLAPIHandleException(_except_); \
  }                                         \
  return 0;  // NOLINT(*)

/**
 * @brief handle exception throwed out
 * @param e the exception
 * @return the return value of API after exception is handled
 */
inline int DGLAPIHandleException(const std::runtime_error &e) {
  DGLAPISetLastError(e.what());
  return -1;
}

#endif  // DGL_RUNTIME_RUNTIME_BASE_H_
