/*!
 *  Copyright (c) 2021 by Contributors
 * \file semaphore_wrapper.cc
 * \brief A simple corss platform semaphore wrapper
 */
#include "semaphore_wrapper.h"

#include <dmlc/logging.h>

namespace dgl {
namespace runtime {

#ifdef _WIN32

Semaphore::Semaphore() {
  sem_ = CreateSemaphore(nullptr, 0, INT_MAX, nullptr);
  if (!sem_) {
    LOG(FATAL) << "Cannot create semaphore";
  }
}

void Semaphore::Wait() {
  WaitForSingleObject(sem_, INFINITE);
}

void Semaphore::Post() {
  ReleaseSemaphore(sem_, 1, nullptr);
}

#else

Semaphore::Semaphore() {
  sem_init(&sem_, 0, 0);
}

void Semaphore::Wait() {
  sem_wait(&sem_);
}

void Semaphore::Post() {
  sem_post(&sem_);
}

#endif

}  // namespace runtime
}  // namespace dgl
