/*!
 *  Copyright (c) 2020-2022 by Contributors
 * \file tensoradapter.h
 * \brief Header file for functions exposed by the adapter library.
 *
 * Functions in this library must be exported with extern "C" so that DGL can
 * locate them with dlsym(3) (or GetProcAddress on Windows).
 */

#ifndef TENSORADAPTER_H_
#define TENSORADAPTER_H_

#ifdef DGL_USE_CUDA
#include <cuda_runtime.h>
#endif  // DGL_USE_CUDA

namespace tensoradapter {

extern "C" {

/*!
 * \brief Allocate a piece of CPU memory via
 * PyTorch's CPUAllocator
 *
 * \param nbytes The size to be allocated.
 * \return Pointer to the allocated memory.
 */
void* CPURawAlloc(size_t nbytes);

/*!
 * \brief Free the CPU memory.
 *
 * \param ptr Pointer to the memory to be freed.
 */
void CPURawDelete(void* ptr);

#ifdef DGL_USE_CUDA
/*!
 * \brief Allocate a piece of GPU memory via
 * PyTorch's THCCachingAllocator.
 *
 * \param nbytes The size to be allocated.
 * \param stream The stream to be allocated on.
 * \return Pointer to the allocated memory.
 */
void* CUDARawAlloc(size_t nbytes, cudaStream_t stream);

/*!
 * \brief Free the GPU memory.
 *
 * \param ptr Pointer to the memory to be freed.
 */
void CUDARawDelete(void* ptr);

/*!
 * \brief Get the current CUDA stream.
 */
cudaStream_t CUDACurrentStream();

/*!
 * \brief Let the caching allocator know which streams are using this tensor.
 *
 * \param ptr Pointer of the tensor to be recorded.
 * \param stream The stream that is using this tensor.
 * \param device_id Device of the tensor.
 */
void RecordStream(void* ptr, cudaStream_t stream, int device_id);
#endif  // DGL_USE_CUDA
}

};  // namespace tensoradapter

#endif  // TENSORADAPTER_H_
