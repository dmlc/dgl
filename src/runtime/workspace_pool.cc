/*!
 *  Copyright (c) 2017 by Contributors
 * \file workspace_pool.h
 * \brief Workspace pool utility.
 */
#include "workspace_pool.h"
#include <cstdint>
#include <memory>
#include "dgl/runtime/ndarray.h"

namespace dgl {
namespace runtime {

// page size.
constexpr size_t kWorkspacePageSize = 4 << 10;

class WorkspacePool::Pool {
 public:
  // constructor
  Pool() {
    // safe guard header on each list.
    Entry e;
    e.data = nullptr;
    e.size = 0;
    allocated_.push_back(e);
  }
  // allocate from pool
  void* Alloc(DGLContext ctx, DeviceAPI* device, size_t nbytes) {
    DGLType type;
    type.code = kDLUInt;
    type.bits = 8;
    type.lanes = 1;
    auto nd = NDArray::Empty({(int64_t)nbytes}, type, ctx);
    void* ptr = nd->data;
    allocated_.push_back(Entry{nd->data, nbytes, std::move(nd)});
    return ptr;
  }
  // free resource back to pool
  void Free(void* data) {
    Entry e;
    if (allocated_.back().data == data) {
      // quick path, last allocated.
      e = allocated_.back();
      allocated_.pop_back();
    } else {
      int index = static_cast<int>(allocated_.size()) - 2;
      for (; index > 0 && allocated_[index].data != data; --index) {}
      CHECK_GT(index, 0) << "trying to free things that has not been allocated";
      e = allocated_[index];
      allocated_.erase(allocated_.begin() + index);
    }
  }
  // Release all resources
  void Release(DGLContext ctx, DeviceAPI* device) {
    allocated_.clear();
  }

 private:
  /*! \brief a single entry in the pool */
  struct Entry {
    void* data;
    size_t size;
    NDArray nd;
  };
  /*! \brief List of allocated items */
  std::vector<Entry> allocated_;
};

WorkspacePool::WorkspacePool(DLDeviceType device_type, std::shared_ptr<DeviceAPI> device)
    : device_type_(device_type), device_(device) {
}

WorkspacePool::~WorkspacePool() {
  /*
   Comment out the destruct of WorkspacePool, due to Segmentation fault with MXNet
   Since this will be only called at the termination of process,
   not manually wiping out should not cause problems.
  */
  // for (size_t i = 0; i < array_.size(); ++i) {
  //   if (array_[i] != nullptr) {
  //     DGLContext ctx;
  //     ctx.device_type = device_type_;
  //     ctx.device_id = static_cast<int>(i);
  //     array_[i]->Release(ctx, device_.get());
  //     delete array_[i];
  //   }
  // }
}

void* WorkspacePool::AllocWorkspace(DGLContext ctx, size_t size) {
  if (static_cast<size_t>(ctx.device_id) >= array_.size()) {
    array_.resize(ctx.device_id + 1, nullptr);
  }
  if (array_[ctx.device_id] == nullptr) {
    array_[ctx.device_id] = new Pool();
  }
  return array_[ctx.device_id]->Alloc(ctx, device_.get(), size);
}

void WorkspacePool::FreeWorkspace(DGLContext ctx, void* ptr) {
  CHECK(static_cast<size_t>(ctx.device_id) < array_.size() &&
        array_[ctx.device_id] != nullptr);
  array_[ctx.device_id]->Free(ptr);
}

}  // namespace runtime
}  // namespace dgl
