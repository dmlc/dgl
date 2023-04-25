/**
 *  Copyright (c) 2020-2022 by Contributors
 * @file array/tensordispatch.h
 * @brief This file defines the dispatcher of tensor operators to
 * framework-specific implementations.
 *
 *  The dispatcher consists of a TensorDispatcher singleton in DGL C library and
 *  one separately-built shared library per supported backend.
 *
 *  Those shared libraries contain wrappers of the framework-specific operators.
 *  The wrappers are defined with extern "C", meaning that the C++ compiler will
 *  not do name mangling for those functions so that DGL can conveniently locate
 *  them using dlsym(3) (or GetProcAddress in Windows).
 *
 *  The TensorDispatcher singleton maintains a mapping from an array operator to
 *  the address of the corresponding symbol in the shared library.  During
 *  initialization, the TensorDispatcher checks which backend DGL is using.
 *  It then locates and opens the corresponding shared library using dlopen(3)
 * (or LoadLibrary in Windows), and populates the said mapping above with
 * dlsym(3) (or GetProcAddress in Windows).
 *
 *  A tensor operator in TensorDispatcher first checks whether the corresponding
 * symbol address is found in the mapping.  If so, it calls the function located
 * at the symbol address instead, allocate/free pieces of memory on CPU/GPU. If
 * not, it falls back to DeviceAPI::AllocWorkspace/FreeWorkspace.
 */

#ifndef DGL_RUNTIME_TENSORDISPATCH_H_
#define DGL_RUNTIME_TENSORDISPATCH_H_

#include <stddef.h>
#include <tensoradapter.h>
#if defined(WIN32) || defined(_WIN32)
#include <windows.h>
#endif  // WIN32
#ifdef DGL_USE_CUDA
#include <cuda_runtime.h>
#endif  // DGL_USE_CUDA
#include "ndarray.h"

/**
 * @brief Casts a pointer \c entry to a function pointer with signature of \c
 * func.
 */
#define FUNCCAST(func, entry) (*reinterpret_cast<decltype(&(func))>(entry))

namespace dgl {
namespace runtime {

/**
 * @brief Dispatcher that delegates the function calls to framework-specific C++
 * APIs.
 *
 * This class is not thread-safe.
 */
class TensorDispatcher {
 public:
  /** @brief Get the singleton instance. */
  static TensorDispatcher* Global() {
    static TensorDispatcher inst;
    return &inst;
  }

  /** @brief Whether an adapter library is available. */
  inline bool IsAvailable() { return available_; }

  /** @brief Load symbols from the given tensor adapter library path. */
  bool Load(const char* path_cstr);

  /**
   * @brief Allocate a piece of CPU memory via PyTorch's CPUAllocator.
   * Used in CPUDeviceAPI::AllocWorkspace().
   *
   * @param nbytes The size to be allocated.
   * @return Pointer to the allocated memory.
   */
  inline void* CPUAllocWorkspace(size_t nbytes) {
    auto entry = entrypoints_[Op::kCPURawAlloc];
    return FUNCCAST(tensoradapter::CPURawAlloc, entry)(nbytes);
  }

  /**
   * @brief Free the CPU memory.
   * Used in CPUDeviceAPI::FreeWorkspace().
   *
   * @param ptr Pointer to the memory to be freed.
   */
  inline void CPUFreeWorkspace(void* ptr) {
    auto entry = entrypoints_[Op::kCPURawDelete];
    FUNCCAST(tensoradapter::CPURawDelete, entry)(ptr);
  }

#ifdef DGL_USE_CUDA
  /**
   * @brief Allocate a piece of GPU memory via
   * PyTorch's THCCachingAllocator.
   * Used in CUDADeviceAPI::AllocWorkspace().
   *
   * @note THCCachingAllocator specify the device to allocate on
   * via cudaGetDevice(). Make sure to call cudaSetDevice()
   * before invoking this function.
   *
   * @param nbytes The size to be allocated.
   * @param stream The stream to be allocated on.
   * @return Pointer to the allocated memory.
   */
  inline void* CUDAAllocWorkspace(size_t nbytes, cudaStream_t stream) {
    auto entry = entrypoints_[Op::kCUDARawAlloc];
    return FUNCCAST(tensoradapter::CUDARawAlloc, entry)(nbytes, stream);
  }

  /**
   * @brief Free the GPU memory.
   * Used in CUDADeviceAPI::FreeWorkspace().
   *
   * @param ptr Pointer to the memory to be freed.
   */
  inline void CUDAFreeWorkspace(void* ptr) {
    auto entry = entrypoints_[Op::kCUDARawDelete];
    FUNCCAST(tensoradapter::CUDARawDelete, entry)(ptr);
  }

  /**
   * @brief Find the current PyTorch CUDA stream
   * Used in runtime::getCurrentCUDAStream().
   *
   * @note PyTorch pre-allocates/sets the current CUDA stream
   * on current device via cudaGetDevice(). Make sure to call cudaSetDevice()
   * before invoking this function.
   *
   * @return cudaStream_t stream handle
   */
  inline cudaStream_t CUDAGetCurrentStream() {
    auto entry = entrypoints_[Op::kCUDACurrentStream];
    return FUNCCAST(tensoradapter::CUDACurrentStream, entry)();
  }

  /**
   * @brief Allocate a piece of pinned CPU memory via PyTorch
   *     CachingHostAllocator.
   * @note Used in CUDADeviceAPI::AllocPinnedDataSpace().
   * @param nbytes The size to be allocated.
   * @param ctx Pointer to the PyTorch storage ctx ptr returned from the
   *     allocator.
   * @param deleter Pointer to the delete function ptr returned from the
   *     allocator.
   * @return Raw pointer to the allocated memory.
   */
  inline void* CUDAAllocHostWorkspace(
      size_t nbytes, void** ctx, void** deleter) {
    auto entry = entrypoints_[Op::kCUDARawHostAlloc];

    auto alloc_func = FUNCCAST(tensoradapter::CUDARawHostAlloc, entry);
    return alloc_func(nbytes, ctx, deleter);
  }

  /**
   * @brief Insert the pinned memory block (allocated via PyTorch
   *     CachingHostAllocator) back to the free list for future usage.(ref:
   *     pytorch/pytorch/blob/master/aten/src/ATen/cuda/CachingHostAllocator.cpp).
   * @note Used in CUDADeviceAPI::FreePinnedDataSpace().
   * @param deleter Pointer to the delete function ptr returned from the
   *     allocator.
   */
  inline void CUDAFreeHostWorkspace(void** deleter) {
    auto entry = entrypoints_[Op::kCUDARawHostDelete];
    FUNCCAST(tensoradapter::CUDARawHostDelete, entry)(deleter);
  }

  /**
   * @brief Invoke the record_event function call from PyTorch
   *     CachingHostAllocator.
   * @note This function assoicates a CUDA stream (used by a copy kernel) to the
   *     pinned data. In the free path of this data, which is achieved by
   *     calling CUDAFreeHostWorkspace, the set of associated streams is then
   *     consumed to ensure proper functionlity. (ref:
   *     pytorch/pytorch/blob/master/aten/src/ATen/cuda/CachingHostAllocator.cpp).
   *     Used in CUDADeviceAPI::RecordedCopyDataFromTo().
   *
   * @param data Pointer of the tensor to be recorded.
   * @param ctx PyTorch storage ctx ptr returned from the allocator.
   * @param stream The stream that currently consumes this tensor.
   * @param device_id Device of the tensor.
   */
  inline void CUDARecordHostAlloc(
      void* data, void* ctx, cudaStream_t stream, int device_id) {
    auto entry = entrypoints_[Op::kCUDARecordHostAlloc];
    auto recorded_alloc = FUNCCAST(tensoradapter::CUDARecordHostAlloc, entry);
    recorded_alloc(data, ctx, stream, device_id);
  }

  /**
   * @brief Release cached pinned memory allocations via cudaHostFree.
   * @note Used in CUDADeviceAPI::PinData() before pinning any host memory by
   *     DGL.
   */
  inline void CUDAHostAllocatorEmptyCache() {
    auto entry = entrypoints_[Op::kCUDAHostAllocatorEmptyCache];
    FUNCCAST(tensoradapter::CUDAHostAllocatorEmptyCache, entry)();
  }
#endif  // DGL_USE_CUDA

  /**
   * @brief Record streams that are using this tensor.
   * Used in NDArray::RecordStream().
   *
   * @param ptr Pointer of the tensor to be recorded.
   * @param stream The stream that is using this tensor.
   * @param device_id Device of the tensor.
   */
  inline void RecordStream(void* ptr, DGLStreamHandle stream, int device_id) {
#ifdef DGL_USE_CUDA
    auto entry = entrypoints_[Op::kRecordStream];
    FUNCCAST(tensoradapter::RecordStream, entry)
    (ptr, static_cast<cudaStream_t>(stream), device_id);
#endif
  }

 private:
  /** @brief ctor */
  TensorDispatcher() = default;
  /** @brief dtor */
  ~TensorDispatcher();

  /**
   * @brief List of symbols in the adapter library.
   *
   * Must match the functions in tensoradapter/include/tensoradapter.h.
   */
  static constexpr const char* names_[] = {
      "CPURawAlloc",         "CPURawDelete",
#ifdef DGL_USE_CUDA
      "CUDARawAlloc",        "CUDARawDelete",
      "CUDACurrentStream",   "RecordStream",
      "CUDARawHostAlloc",    "CUDARawHostDelete",
      "CUDARecordHostAlloc", "CUDAHostAllocatorEmptyCache",
#endif  // DGL_USE_CUDA
  };

  /** @brief Index of each function to the symbol list */
  class Op {
   public:
    static constexpr int kCPURawAlloc = 0;
    static constexpr int kCPURawDelete = 1;
#ifdef DGL_USE_CUDA
    static constexpr int kCUDARawAlloc = 2;
    static constexpr int kCUDARawDelete = 3;
    static constexpr int kCUDACurrentStream = 4;
    static constexpr int kRecordStream = 5;
    static constexpr int kCUDARawHostAlloc = 6;
    static constexpr int kCUDARawHostDelete = 7;
    static constexpr int kCUDARecordHostAlloc = 8;
    static constexpr int kCUDAHostAllocatorEmptyCache = 9;
#endif  // DGL_USE_CUDA
  };

  /** @brief Number of functions */
  static constexpr int num_entries_ = sizeof(names_) / sizeof(names_[0]);

  /** @brief Entrypoints of each function */
  void* entrypoints_[num_entries_] = {
      nullptr, nullptr,
#ifdef DGL_USE_CUDA
      nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
#endif  // DGL_USE_CUDA
  };

  bool available_ = false;
#if defined(WIN32) || defined(_WIN32)
  HINSTANCE handle_;
#else   // !WIN32
  void* handle_;
#endif  // WIN32
};

};  // namespace runtime
};  // namespace dgl

#undef FUNCCAST

#endif  // DGL_RUNTIME_TENSORDISPATCH_H_
