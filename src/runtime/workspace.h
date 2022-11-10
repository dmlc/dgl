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
        ptr_(static_cast<T*>(device_->AllocWorkspace(ctx_, sizeof(T) * size))) {
  }

  ~Workspace() {
    if (*this) {
      free();
    }
  }

  operator bool() const { return ptr_ != nullptr; }

  T* get() {
    assert(*this);
    return ptr_;
  }

  T const* get() const {
    assert(*this);
    return ptr_;
  }

  void free() {
    assert(*this);
    device_->FreeWorkspace(ctx_, ptr_);
    ptr_ = nullptr;
  }

 private:
  DeviceAPI* device_;
  DGLContext ctx_;
  T* ptr_;
};

template <>
class Workspace<void> {
 public:
  Workspace(DeviceAPI* device, DGLContext ctx, const size_t size)
      : device_(device),
        ctx_(ctx),
        ptr_(static_cast<void*>(device_->AllocWorkspace(ctx_, size))) {}

  ~Workspace() {
    if (*this) {
      free();
    }
  }

  operator bool() const { return ptr_ != nullptr; }

  void* get() {
    assert(*this);
    return ptr_;
  }

  void const* get() const {
    assert(*this);
    return ptr_;
  }

  void free() {
    assert(*this);
    device_->FreeWorkspace(ctx_, ptr_);
    ptr_ = nullptr;
  }

 private:
  DeviceAPI* device_;
  DGLContext ctx_;
  void* ptr_;
};

}  // namespace runtime
}  // namespace dgl

#endif  // DGL_RUNTIME_WORKSPACE_H_
