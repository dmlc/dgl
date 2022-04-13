/*!
 *  Copyright (c) 2021 by Contributors
 * \file semaphore_wrapper.h
 * \brief A simple corss platform semaphore wrapper
 */
#ifndef DGL_RUNTIME_SEMAPHORE_WRAPPER_H_
#define DGL_RUNTIME_SEMAPHORE_WRAPPER_H_

#ifdef _WIN32
#include <windows.h>
#else
#include <semaphore.h>
#endif


namespace dgl {
namespace runtime {

/*!
 * \brief A simple crossplatform Semaphore wrapper
 */
class Semaphore {
 public:
  /*!
   * \brief Semaphore constructor
   */
  Semaphore();
  /*!
   * \brief blocking wait, decrease semaphore by 1
   */
  void Wait();
  /*!
   * \brief increase semaphore by 1
   */
  void Post();
 private:
#ifdef _WIN32
  HANDLE sem_;
#else
  sem_t sem_;
#endif
};

}  // namespace runtime
}  // namespace dgl

#endif  // DGL_RUNTIME_SEMAPHORE_WRAPPER_H_
