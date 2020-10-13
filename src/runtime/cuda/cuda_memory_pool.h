/*!
 *  Copyright (c) 2020 by Contributors
 * \file cuda_memory_pool.h
 * \brief Memory pool for GPU.
 */

#ifndef DGL_RUNTIME_CUDA_CUDA_MEMORY_POOL_H_
#define DGL_RUNTIME_CUDA_CUDA_MEMORY_POOL_H_

#include <memory>
#include <vector>
#include "dgl/runtime/c_runtime_api.h"
#include "dgl/runtime/device_api.h"

namespace dgl {
namespace runtime {

class CudaMemoryPool {
 public:
   /*!
    * \brief Create a new CudaMemoryPool for reducing repeat calls to
    * cudaMalloc() and cudaFree().
    *
    * @param device The cuda device API to use.
    */
  explicit CudaMemoryPool(std::shared_ptr<DeviceAPI> device);


  /*!
   * \brief Create an allocation in ctx of at least size bytes.
   *
   * @param ctx The context to get an allocation for.
   * @param size The size of the space required in bytes.
   *
   * @return A pointer to the allocated space.
   */
  void* Alloc(DGLContext ctx, size_t size);


  /**
   * @brief Free an allocation in ctx pointed to by ptr.
   *
   * @param ctx The context to release an allocation from.
   * @param ptr The pointer to the allocation.
   */
  void Free(DGLContext ctx, void* ptr);

 private:
  class Pool;
  std::vector<Pool*> array_;
  std::shared_ptr<DeviceAPI> device_;

  Pool* const GetMemoryPool(size_t device_id);
};

}  // namespace runtime
}  // namespace dgl

#endif  // DGL_RUNTIME_CUDA_CUDA_MEMORY_POOL_H_
