/*!
 *  Copyright (c) 2020 by Contributors
 * \file cuda_memory_pool.h
 * \brief Memory pool for GPU. 
 */

#ifndef DGL_RUNTIME_CUDA_CUDA_MEMORY_POOL_H_
#define DGL_RUNTIME_CUDA_CUDA_MEMORY_POOL_H_

#include "dgl/runtime/c_runtime_api.h"
#include "dgl/runtime/device_api.h"
#include <memory>
#include <vector>

namespace dgl {
namespace runtime {

class CudaMemoryPool
{
  public:
    CudaMemoryPool(std::shared_ptr<DeviceAPI> device);

    void* Alloc(DGLContext ctx, size_t size);

    void Free(DGLContext ctx, void* ptr);

  private:
    class Pool; 
    std::vector<Pool*> array_;
    std::shared_ptr<DeviceAPI> device_;

    Pool* const GetMemoryPool(size_t device_id);
};

}
}

#endif
