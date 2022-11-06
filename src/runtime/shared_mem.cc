/**
 *  Copyright (c) 2019 by Contributors
 * @file shared_mem.cc
 * @brief Shared memory management.
 */
#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif
#include <dgl/runtime/shared_mem.h>
#include <dmlc/logging.h>
#include <stdio.h>
#include <string.h>

#include "resource_manager.h"

namespace dgl {
namespace runtime {

/**
 * Shared memory is a resource that cannot be cleaned up if the process doesn't
 * exit normally. We'll manage the resource with ResourceManager.
 */
class SharedMemoryResource : public Resource {
  std::string name;

 public:
  explicit SharedMemoryResource(const std::string &name) { this->name = name; }

  void Destroy() {
    // LOG(INFO) << "remove " << name << " for shared memory";
#ifndef _WIN32
    shm_unlink(name.c_str());
#else  // _WIN32
    // NOTHING; Windows automatically removes the shared memory object once all
    // handles are unmapped.
#endif
  }
};

SharedMemory::SharedMemory(const std::string &name) {
  this->name = name;
  this->own_ = false;
#ifndef _WIN32
  this->fd_ = -1;
#else
  this->handle_ = nullptr;
#endif
  this->ptr_ = nullptr;
  this->size_ = 0;
}

SharedMemory::~SharedMemory() {
#ifndef _WIN32
  if (ptr_ && size_ != 0) CHECK(munmap(ptr_, size_) != -1) << strerror(errno);
  if (fd_ != -1) close(fd_);
  if (own_) {
    // LOG(INFO) << "remove " << name << " for shared memory";
    if (name != "") {
      shm_unlink(name.c_str());
      // The resource has been deleted. We don't need to keep track of it any
      // more.
      DeleteResource(name);
    }
  }
#else
  if (ptr_) CHECK(UnmapViewOfFile(ptr_)) << "Win32 Error: " << GetLastError();
  if (handle_) CloseHandle(handle_);
    // Windows do not need a separate shm_unlink step.
#endif  // _WIN32
}

void *SharedMemory::CreateNew(size_t sz) {
#ifndef _WIN32
  this->own_ = true;

  // We need to create a shared-memory file.
  // TODO(zhengda) we need to report error if the shared-memory file exists.
  int flag = O_RDWR | O_CREAT;
  fd_ = shm_open(name.c_str(), flag, S_IRUSR | S_IWUSR);
  CHECK_NE(fd_, -1) << "fail to open " << name << ": " << strerror(errno);
  // Shared memory cannot be deleted if the process exits abnormally in Linux.
  AddResource(name, std::shared_ptr<Resource>(new SharedMemoryResource(name)));
  auto res = ftruncate(fd_, sz);
  CHECK_NE(res, -1) << "Failed to truncate the file. " << strerror(errno);
  ptr_ = mmap(NULL, sz, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
  CHECK_NE(ptr_, MAP_FAILED)
      << "Failed to map shared memory. mmap failed with error "
      << strerror(errno);
  this->size_ = sz;
  return ptr_;
#else
  handle_ = CreateFileMapping(
      INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE,
      static_cast<DWORD>(sz >> 32), static_cast<DWORD>(sz & 0xFFFFFFFF),
      name.c_str());
  CHECK(handle_ != nullptr)
      << "fail to open " << name << ", Win32 error: " << GetLastError();
  ptr_ = MapViewOfFile(handle_, FILE_MAP_ALL_ACCESS, 0, 0, sz);
  if (ptr_ == nullptr) {
    LOG(FATAL) << "Memory mapping failed, Win32 error: " << GetLastError();
    CloseHandle(handle_);
    return nullptr;
  }
  this->size_ = sz;
  return ptr_;
#endif  // _WIN32
}

void *SharedMemory::Open(size_t sz) {
#ifndef _WIN32
  int flag = O_RDWR;
  fd_ = shm_open(name.c_str(), flag, S_IRUSR | S_IWUSR);
  CHECK_NE(fd_, -1) << "fail to open " << name << ": " << strerror(errno);
  ptr_ = mmap(NULL, sz, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
  CHECK_NE(ptr_, MAP_FAILED)
      << "Failed to map shared memory. mmap failed with error "
      << strerror(errno);
  this->size_ = sz;
  return ptr_;
#else
  handle_ = OpenFileMapping(FILE_MAP_ALL_ACCESS, FALSE, name.c_str());
  CHECK(handle_ != nullptr)
      << "fail to open " << name << ", Win32 Error: " << GetLastError();
  ptr_ = MapViewOfFile(handle_, FILE_MAP_ALL_ACCESS, 0, 0, sz);
  if (ptr_ == nullptr) {
    LOG(FATAL) << "Memory mapping failed, Win32 error: " << GetLastError();
    CloseHandle(handle_);
    return nullptr;
  }
  this->size_ = sz;
  return ptr_;
#endif  // _WIN32
}

bool SharedMemory::Exist(const std::string &name) {
#ifndef _WIN32
  int fd = shm_open(name.c_str(), O_RDONLY, S_IRUSR | S_IWUSR);
  if (fd >= 0) {
    close(fd);
    return true;
  } else {
    return false;
  }
#else
  HANDLE handle = OpenFileMapping(FILE_MAP_ALL_ACCESS, FALSE, name.c_str());
  if (handle != nullptr) {
    CloseHandle(handle);
    return true;
  } else {
    return false;
  }
#endif  // _WIN32
}

}  // namespace runtime
}  // namespace dgl
