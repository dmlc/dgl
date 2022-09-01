/*!
 *  Copyright (c) 2020-2022 by Contributors
 * \file torch/torch.cpp
 * \brief Implementation of PyTorch adapter library.
 */

#include <tensoradapter_exports.h>
#include <c10/core/CPUAllocator.h>
#ifdef DGL_USE_CUDA
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#endif  // DGL_USE_CUDA

namespace tensoradapter {

extern "C" {

TA_EXPORTS void* CPURawAlloc(size_t nbytes) {
  return c10::GetCPUAllocator()->raw_allocate(nbytes);
}

TA_EXPORTS void CPURawDelete(void* ptr) {
  c10::GetCPUAllocator()->raw_deallocate(ptr);
}

#ifdef DGL_USE_CUDA
TA_EXPORTS void* CUDARawAlloc(size_t nbytes, cudaStream_t stream) {
  return c10::cuda::CUDACachingAllocator::raw_alloc_with_stream(
    nbytes, stream);
}

TA_EXPORTS void CUDARawDelete(void* ptr) {
  c10::cuda::CUDACachingAllocator::raw_delete(ptr);
}

TA_EXPORTS cudaStream_t GetTorchCUDAStream() {
  return at::cuda::getCurrentCUDAStream();
}
#endif  // DGL_USE_CUDA

};

};  // namespace tensoradapter
