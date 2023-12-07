/**
 *  Copyright (c) 2023 by Contributors
 * @file shared_memory.cc
 * @brief Source file of graphbolt shared memory.
 */
#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif  // !_WIN32

#include <graphbolt/shared_memory.h>
#include <stdio.h>
#include <string.h>
#include <torch/torch.h>

namespace graphbolt {
namespace sampling {

// Two processes opening the same path are guaranteed to access the same shared
// memory object if and only if path begins with a slash ('/') character.
constexpr char kSharedMemNamePrefix[] = "/dgl.graphbolt.";
constexpr char kSharedMemNameSuffix[] = ".lock";

// A prefix and a suffix are added to the name of the shared memory to create
// the name of the shared memory object.
inline std::string DecorateName(const std::string& name) {
  return kSharedMemNamePrefix + name + kSharedMemNameSuffix;
}

SharedMemory::SharedMemory(const std::string& name)
    : name_(name), size_(0), ptr_(nullptr) {
#ifdef _WIN32
  this->handle_ = nullptr;
#else   // _WIN32
  this->file_descriptor_ = -1;
  this->is_creator_ = false;
#endif  // _WIN32
}

#ifdef _WIN32

SharedMemory::~SharedMemory() {
  if (ptr_) CHECK(UnmapViewOfFile(ptr_)) << "Win32 Error: " << GetLastError();
  if (handle_) CloseHandle(handle_);
}

void* SharedMemory::Create(size_t size) {
  size_ = size;

  std::string decorated_name = DecorateName(name_);
  handle_ = CreateFileMapping(
      INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE,
      static_cast<DWORD>(size >> 32), static_cast<DWORD>(size & 0xFFFFFFFF),
      decorated_name.c_str());
  TORCH_CHECK(
      handle_ != nullptr, "Failed to open ", decorated_name,
      ", Win32 error: ", GetLastError());

  ptr_ = MapViewOfFile(handle_, FILE_MAP_ALL_ACCESS, 0, 0, size);
  TORCH_CHECK(
      ptr_ != nullptr, "Memory mapping failed, Win32 error: ", GetLastError());
  return ptr_;
}

void* SharedMemory::Open() {
  std::string decorated_name = DecorateName(name_);
  handle_ = OpenFileMapping(FILE_MAP_ALL_ACCESS, FALSE, decorated_name.c_str());
  TORCH_CHECK(
      handle_ != nullptr, "Failed to open ", decorated_name,
      ", Win32 Error: ", GetLastError());

  ptr_ = MapViewOfFile(handle_, FILE_MAP_ALL_ACCESS, 0, 0, 0);
  TORCH_CHECK(
      ptr_ != nullptr, "Memory mapping failed, Win32 error: ", GetLastError());

  // Obtain the size of the memory-mapped file.
  MEMORY_BASIC_INFORMATION memInfo;
  TORCH_CHECK(
      VirtualQuery(ptr_, &memInfo, sizeof(memInfo)) != 0,
      "Failed to get the size of shared memory: ", GetLastError());
  size_ = static_cast<size_t>(memInfo.RegionSize);

  return ptr_;
}

bool SharedMemory::Exists(const std::string& name) {
  std::string decorated_name = DecorateName(name);
  HANDLE handle =
      OpenFileMapping(FILE_MAP_ALL_ACCESS, FALSE, decorated_name.c_str());
  bool exists = handle != nullptr;
  if (exists) {
    CloseHandle(handle);
  }
  return exists;
}

#else  // _WIN32

SharedMemory::~SharedMemory() {
  if (ptr_ && size_ != 0) CHECK(munmap(ptr_, size_) != -1) << strerror(errno);
  if (file_descriptor_ != -1) close(file_descriptor_);

  std::string decorated_name = DecorateName(name_);
  if (is_creator_ && decorated_name != "") shm_unlink(decorated_name.c_str());
}

void *SharedMemory::Create(size_t size) {
  size_ = size;
  is_creator_ = true;

  // TODO(zhenkun): handle the error properly if the shared memory object
  // already exists.
  std::string decorated_name = DecorateName(name_);
  file_descriptor_ =
      shm_open(decorated_name.c_str(), O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
  TORCH_CHECK(file_descriptor_ != -1, "Failed to open: ", strerror(errno));

  auto status = ftruncate(file_descriptor_, size);
  TORCH_CHECK(status != -1, "Failed to truncate the file: ", strerror(errno));

  ptr_ =
      mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, file_descriptor_, 0);
  TORCH_CHECK(
      ptr_ != MAP_FAILED,
      "Failed to map shared memory, mmap failed with error: ", strerror(errno));
  return ptr_;
}

void *SharedMemory::Open() {
  std::string decorated_name = DecorateName(name_);
  file_descriptor_ =
      shm_open(decorated_name.c_str(), O_RDWR, S_IRUSR | S_IWUSR);
  TORCH_CHECK(
      file_descriptor_ != -1, "Failed to open ", decorated_name, ": ",
      strerror(errno));

  struct stat shm_stat;
  TORCH_CHECK(
      fstat(file_descriptor_, &shm_stat) == 0,
      "Failed to get the size of shared memory: ", strerror(errno));
  size_ = shm_stat.st_size;

  ptr_ = mmap(
      NULL, size_, PROT_READ | PROT_WRITE, MAP_SHARED, file_descriptor_, 0);
  TORCH_CHECK(
      ptr_ != MAP_FAILED,
      "Failed to map shared memory, mmap failed with error: ", strerror(errno));
  return ptr_;
}

bool SharedMemory::Exists(const std::string &name) {
  std::string decorated_name = DecorateName(name);
  int file_descriptor =
      shm_open(decorated_name.c_str(), O_RDONLY, S_IRUSR | S_IWUSR);
  bool exists = file_descriptor > 0;
  if (exists) {
    close(file_descriptor);
  }
  return exists;
}

#endif  // _WIN32

}  // namespace sampling
}  // namespace graphbolt
