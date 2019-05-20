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

namespace dgl {
namespace runtime {

#ifndef _WIN32
SharedMemory::SharedMemory(const std::string &name) {
  this->name = name;
  this->own = false;
  this->fd = -1;
  this->ptr = nullptr;
  this->size = 0;
}

SharedMemory::~SharedMemory() {
  munmap(ptr, size);
  close(fd);
  if (own) {
    LOG(INFO) << "remove " << name << " for shared memory";
    shm_unlink(name.c_str());
  }
}

void *SharedMemory::create_new(size_t size) {
  this->own = true;

  int flag = O_RDWR|O_CREAT;
  fd = shm_open(name.c_str(), flag, S_IRUSR | S_IWUSR);
  CHECK_NE(fd, -1) << "fail to open " << name << ": " << strerror(errno);
  auto res = ftruncate(fd, size);
  CHECK_NE(res, -1)
      << "Failed to truncate the file. " << strerror(errno);
  ptr = mmap(NULL, size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
  CHECK_NE(ptr, MAP_FAILED)
      << "Failed to map shared memory. mmap failed with error " << strerror(errno);
  return ptr;
}

void *SharedMemory::open(size_t size) {
  int flag = O_RDWR;
  fd = shm_open(name.c_str(), flag, S_IRUSR | S_IWUSR);
  CHECK_NE(fd, -1) << "fail to open " << name << ": " << strerror(errno);
  ptr = mmap(NULL, size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
  CHECK_NE(ptr, MAP_FAILED)
      << "Failed to map shared memory. mmap failed with error " << strerror(errno);
  return ptr;
}
#endif  // _WIN32

}  // namespace runtime
}  // namespace dgl
