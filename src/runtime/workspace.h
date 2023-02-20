/**
 *  Copyright (c) 2021 by Contributors
 * @file ndarray_partition.h
 * @brief Operations on partition implemented in CUDA.
 */

#ifndef DGL_RUNTIME_WORKSPACE_H_
#define DGL_RUNTIME_WORKSPACE_H_

#include <dgl/runtime/device_api.h>

#include <cassert>

namespace dgl {
namespace runtime {

template <typename T>
class Workspace {
 public:
  Workspace(DeviceAPI* device, DGLContext ctx, const size_t size)
      : device_(device),
        ctx_(ctx),
        size_(size * sizeof(T)),
        ptr_(static_cast<T*>(device_->AllocWorkspace(ctx_, size_))) {}

  ~Workspace() {
    if (*this) {
      free();
    }
  }

  operator bool() const { return ptr_ != nullptr; }

  T* get() {
    assert(size_ == 0 || *this);
    return ptr_;
  }

  T const* get() const {
    assert(size_ == 0 || *this);
    return ptr_;
  }

  void free() {
    assert(size_ == 0 || *this);
    device_->FreeWorkspace(ctx_, ptr_);
    ptr_ = nullptr;
  }

 private:
  DeviceAPI* device_;
  DGLContext ctx_;
  size_t size_;
  T* ptr_;
};

template <>
class Workspace<void> {
 public:
  Workspace(DeviceAPI* device, DGLContext ctx, const size_t size)
      : device_(device),
        ctx_(ctx),
        size_(size),
        ptr_(static_cast<void*>(device_->AllocWorkspace(ctx_, size_))) {}

  ~Workspace() {
    if (*this) {
      free();
    }
  }

  operator bool() const { return ptr_ != nullptr; }

  void* get() {
    assert(size_ == 0 || *this);
    return ptr_;
  }

  void const* get() const {
    assert(size_ == 0 || *this);
    return ptr_;
  }

  void free() {
    assert(size_ == 0 || *this);
    device_->FreeWorkspace(ctx_, ptr_);
    ptr_ = nullptr;
  }

 private:
  DeviceAPI* device_;
  DGLContext ctx_;
  size_t size_;
  void* ptr_;
};

}  // namespace runtime
}  // namespace dgl

#endif  // DGL_RUNTIME_WORKSPACE_H_
