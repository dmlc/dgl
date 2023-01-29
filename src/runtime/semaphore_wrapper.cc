/**
 *  Copyright (c) 2021 by Contributors
 * @file semaphore_wrapper.cc
 * @brief A simple corss platform semaphore wrapper
 */
#include "semaphore_wrapper.h"

#include <dmlc/logging.h>

#ifndef _WIN32
#include <errno.h>
#include <time.h>
#include <unistd.h>
#endif

namespace dgl {
namespace runtime {

#ifdef _WIN32

Semaphore::Semaphore() {
  sem_ = CreateSemaphore(nullptr, 0, INT_MAX, nullptr);
  if (!sem_) {
    LOG(FATAL) << "Cannot create semaphore";
  }
}

void Semaphore::Wait() { WaitForSingleObject(sem_, INFINITE); }

bool Semaphore::TimedWait(int) {
  // Timed wait is not supported on WIN32.
  Wait();
  return true;
}

void Semaphore::Post() { ReleaseSemaphore(sem_, 1, nullptr); }

#else

Semaphore::Semaphore() { sem_init(&sem_, 0, 0); }

void Semaphore::Wait() { sem_wait(&sem_); }

bool Semaphore::TimedWait(int timeout) {
  // sem_timedwait does not exist in Mac OS.
#ifdef __APPLE__
  DLOG(WARNING) << "Timeout is not supported in semaphore's wait on Mac OS.";
  Wait();
#else
  // zero timeout means wait infinitely
  if (timeout == 0) {
    DLOG(WARNING) << "Will wait infinitely on semaphore until posted.";
    Wait();
    return true;
  }
  timespec ts;
  if (clock_gettime(CLOCK_REALTIME, &ts) != 0) {
    LOG(ERROR) << "Failed to get current time via clock_gettime. Errno: "
               << errno;
    return false;
  }
  ts.tv_sec += timeout / MILLISECONDS_PER_SECOND;
  ts.tv_nsec +=
      (timeout % MILLISECONDS_PER_SECOND) * NANOSECONDS_PER_MILLISECOND;
  if (ts.tv_nsec >= NANOSECONDS_PER_SECOND) {
    ts.tv_nsec -= NANOSECONDS_PER_SECOND;
    ++ts.tv_sec;
  }
  int ret = 0;
  while ((ret = sem_timedwait(&sem_, &ts) != 0) && errno == EINTR) {
    continue;
  }
  if (ret != 0) {
    if (errno == ETIMEDOUT) {
      DLOG(WARNING) << "sem_timedwait timed out after " << timeout
                    << " milliseconds.";
    } else {
      LOG(ERROR) << "sem_timedwait returns unexpectedly. Errno: " << errno;
    }
    return false;
  }
#endif

  return true;
}

void Semaphore::Post() { sem_post(&sem_); }

#endif

}  // namespace runtime
}  // namespace dgl
