/*!
 *  Copyright (c) 2019 by Contributors
 * \file shared_mem.cc
 * \brief Shared memory management.
 */
#ifndef _WIN32
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#endif
#include <stdio.h>
#include <string.h>
#include <dmlc/logging.h>
#include <dgl/runtime/shared_mem.h>

#include "resource_manager.h"

namespace dgl {
namespace runtime {

#ifndef _WIN32
/*
 * Shared memory is a resource that cannot be cleaned up if the process doesn't
 * exit normally. We'll manage the resource with ResourceManager.
 */
class SharedMemoryResource: public Resource {
  std::string name;

 public:
  explicit SharedMemoryResource(const std::string &name) {
    this->name = name;
  }

  void Destroy() {
    // LOG(INFO) << "remove " << name << " for shared memory";
    shm_unlink(name.c_str());
  }
};
#endif  // _WIN32

SharedMemory::SharedMemory(const std::string &name) {
#ifndef _WIN32
  this->name = name;
  this->own_ = false;
  this->fd_ = -1;
  this->ptr_ = nullptr;
  this->size_ = 0;
#else
  LOG(FATAL) << "Shared memory is not supported on Windows.";
#endif  // _WIN32
}

SharedMemory::~SharedMemory() {
#ifndef _WIN32
  CHECK(munmap(ptr_, size_) != -1) << strerror(errno);
  close(fd_);
  if (own_) {
    // LOG(INFO) << "remove " << name << " for shared memory";
    shm_unlink(name.c_str());
    // The resource has been deleted. We don't need to keep track of it any more.
    DeleteResource(name);
  }
#else
  LOG(FATAL) << "Shared memory is not supported on Windows.";
#endif  // _WIN32
}

void *SharedMemory::CreateNew(size_t sz) {
#ifndef _WIN32
  this->own_ = true;

  // We need to create a shared-memory file.
  // TODO(zhengda) we need to report error if the shared-memory file exists.
  int flag = O_RDWR|O_CREAT;
  fd_ = shm_open(name.c_str(), flag, S_IRUSR | S_IWUSR);
  CHECK_NE(fd_, -1) << "fail to open " << name << ": " << strerror(errno);
  // Shared memory cannot be deleted if the process exits abnormally.
  AddResource(name, std::shared_ptr<Resource>(new SharedMemoryResource(name)));
  auto res = ftruncate(fd_, sz);
  CHECK_NE(res, -1)
      << "Failed to truncate the file. " << strerror(errno);
  ptr_ = mmap(NULL, sz, PROT_READ|PROT_WRITE, MAP_SHARED, fd_, 0);
  CHECK_NE(ptr_, MAP_FAILED)
      << "Failed to map shared memory. mmap failed with error " << strerror(errno);
  this->size_ = sz;
  return ptr_;
#else
  LOG(FATAL) << "Shared memory is not supported on Windows.";
#endif  // _WIN32
}

void *SharedMemory::Open(size_t sz) {
#ifndef _WIN32
  int flag = O_RDWR;
  fd_ = shm_open(name.c_str(), flag, S_IRUSR | S_IWUSR);
  CHECK_NE(fd_, -1) << "fail to open " << name << ": " << strerror(errno);
  ptr_ = mmap(NULL, sz, PROT_READ|PROT_WRITE, MAP_SHARED, fd_, 0);
  CHECK_NE(ptr_, MAP_FAILED)
      << "Failed to map shared memory. mmap failed with error " << strerror(errno);
  this->size_ = sz;
  return ptr_;
#else
  LOG(FATAL) << "Shared memory is not supported on Windows.";
#endif  // _WIN32
}

bool SharedMemory::Exist(const std::string &name) {
#ifndef _WIN32
  int fd_ = shm_open(name.c_str(), O_RDONLY, S_IRUSR | S_IWUSR);
  if (fd_ >= 0) {
    close(fd_);
    return true;
  } else {
    return false;
  }
#else
  LOG(FATAL) << "Shared memory is not supported on Windows.";
#endif  // _WIN32
}

}  // namespace runtime
}  // namespace dgl
