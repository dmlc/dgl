/*!
 *  Copyright (c) 2020 by Contributors
 * \file tensoradapter.h
 * \brief Header file for functions exposed by the adapter library.
 *
 * Functions in this library must be exported with extern "C" so that DGL can locate
 * them with dlsym(3) (or GetProcAddress on Windows).
 */

#ifndef TENSORADAPTER_H_
#define TENSORADAPTER_H_

#include <dlpack/dlpack.h>
#include <vector>

namespace tensoradapter {

extern "C" {

/*!
 * \brief Allocate an empty tensor.
 *
 * \param shape The shape
 * \param dtype The data type
 * \param ctx The device
 * \param stream The stream to be allcated.
 * \return The allocated tensor
 */
DLManagedTensor* TAempty(
    std::vector<int64_t> shape, DLDataType dtype, DLContext ctx, void* stream);

#ifdef DGL_USE_CUDA
/*!
 * \brief Allocate a piece of GPU memory via
 * PyTorch's THCCachingAllocator.
 *
 * \param nbytes The size to be allocated.
 * \param stream The stream to be allocated in.
 * \return Pointer to the allocated memory.
 */
void* RawAlloc(size_t nbytes, void* stream);

/*!
 * \brief Free the GPU memory.
 *
 * \param ptr Pointer to the memory to be freed.
 */
void RawDelete(void* ptr);
#endif  // DGL_USE_CUDA

}

};  // namespace tensoradapter

#endif  // TENSORADAPTER_H_
