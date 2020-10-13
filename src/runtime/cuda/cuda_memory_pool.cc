/*!
 *  Copyright (c) 2017-2020 by Contributors
 * \file cuda_device_api.cc
 * \brief Memory pool implementation for GPU. 
 */

#include "cuda_memory_pool.h"
#include "cuda_common.h"

#include <cuda_runtime.h>
#include <mutex>
#include <cstdlib>

namespace dgl {
namespace runtime {

namespace {

/*! \brief The minimum size of the pool before considering freeing space in
 * bytes. Currently set to 64MB. */
constexpr const size_t kMinPoolSize = 64 << 20;
constexpr const size_t kCudaPageSize = 4096;

/*! \brief The maximum ratio of allocation size, to allocation use. That is,
 * a multiple of 2 means that an allocation is 4MB, can be used for requests no
 * smaller than 2 MB.
 */
constexpr const size_t kMaxAllocMultiple = 2;

/*! \brief The maximum ratio of allocated memory to the maximum amount of
 * memory used so far. That is, for a multiple of 2, if we have use at most
 * 100 MB of memory at one time, we cannot have more than 200 MB allocated.
 */
constexpr const size_t kMaxTotalMultiple = 2;
}

class CudaMemoryPool::Pool {
 public:
  // constructor
  Pool() : total_(0), total_unused_(0), max_used_(0),
    free_list_(), allocated_(), mtx_() {
  }

  // allocate from pool
  void* Alloc(DGLContext ctx, DeviceAPI* const device, size_t nbytes) {
    std::lock_guard<std::mutex> guard(mtx_);

    // Allocate align to page.
    nbytes = AllocSize(nbytes);

    Entry e;
    auto iter = free_list_.lower_bound(nbytes);

    if (iter == free_list_.end() || iter->second.size > kMaxAllocMultiple*nbytes) {
      // if we have no allocation big enough, or that is not a large waste
      // of space, allocate new space

      if (TotalSize() > std::max(kMinPoolSize, kMaxTotalMultiple*max_used_)) {
        // If we have allocated more than twice the most we have ever used,
        // cleanup currently unused memory
        Release(ctx, device);
      }

      // allocate a new page
      DGLType type;
      type.code = kDLUInt;
      type.bits = 8;
      type.lanes = 1;

      e.size = nbytes;
      e.data = device->AllocRawDataSpace(ctx, nbytes, kTempAllocaAlignment, type);
      total_ += e.size;
    } else {
      // use previously allocated space
      e = iter->second;
      free_list_.erase(iter);
      total_unused_ -= e.size;
    }

    allocated_.emplace(e.data, e);

    if (TotalUsed() > max_used_) {
      max_used_ = TotalUsed();
    }

    return e.data;
  }

  // free resource back to pool
  void Free(void* data) {
    std::lock_guard<std::mutex> guard(mtx_);

    auto iter = allocated_.find(data);
    CHECK(iter != allocated_.end()) << "trying to free things that has not been allocated";

    Entry e = iter->second;
    allocated_.erase(iter);

    free_list_.emplace(e.size, e);
    total_unused_ += e.size;
  }

  size_t TotalUnused() const
  {
    return total_unused_;
  }

  size_t TotalUsed() const {
    return total_ - total_unused_;
  }

  size_t TotalSize() const
  {
    return total_;
  }

 private:
  /*! \brief a single entry in the pool */
  struct Entry {
    void* data;
    size_t size;
  };
  size_t total_;
  size_t total_unused_;
  size_t max_used_;

  /*! \brief Mapping of sizes to free items. There may be multiple items of
   * the same size. */
  std::multimap<size_t, Entry> free_list_;

  /*! \brief Mappings of addresses, to allocation data. Each address must
   * be unique. */
  std::map<void*, Entry> allocated_;

  std::mutex mtx_;

  size_t AllocSize(const size_t nbytes) const
  {
    return std::max(kCudaPageSize, (nbytes + (kCudaPageSize - 1)) / kCudaPageSize * kCudaPageSize);
  }

  void Release(DGLContext ctx, DeviceAPI* device) {
    for (auto kv : free_list_) {
      device->FreeRawDataSpace(ctx, kv.second.data);
      total_ -= kv.second.size;
      total_unused_ -= kv.second.size;
    }
    free_list_.clear();

    CHECK_EQ(total_unused_, 0);
  }
};

CudaMemoryPool::CudaMemoryPool(
    std::shared_ptr<DeviceAPI> device) : array_(), device_(device)
{
  int nDevices;
  CUDA_CALL(cudaGetDeviceCount(&nDevices));
  array_.resize(nDevices);
  for (int d = 0; d < nDevices; ++d) {
    array_[d] = new CudaMemoryPool::Pool();
  }
}

void* CudaMemoryPool::Alloc(DGLContext ctx, size_t size) {
  Pool* const pool = GetMemoryPool(ctx.device_id);
  return pool->Alloc(ctx, device_.get(), size);
}

void CudaMemoryPool::Free(DGLContext ctx, void* ptr) {
  Pool* const pool = GetMemoryPool(ctx.device_id);
  pool->Free(ptr);
}

CudaMemoryPool::Pool* const CudaMemoryPool::GetMemoryPool(const size_t device_id) {
  CHECK(device_id < array_.size());
  return array_[device_id];
}

}
}
