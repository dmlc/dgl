/**
 *  Copyright (c) 2017 by Contributors
 * @file cuda_common.h
 * @brief Common utilities for CUDA
 */
#ifndef DGL_RUNTIME_CUDA_CUDA_COMMON_H_
#define DGL_RUNTIME_CUDA_CUDA_COMMON_H_

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cusparse.h>
#include <dgl/runtime/packed_func.h>

#include <memory>
#include <string>

#include "../workspace_pool.h"

namespace dgl {
namespace runtime {

/*
  How to use this class to get a nonblocking thrust execution policy that uses
  DGL's memory pool and the current cuda stream

  runtime::CUDAWorkspaceAllocator allocator(ctx);
  const auto stream = runtime::getCurrentCUDAStream();
  const auto exec_policy = thrust::cuda::par_nosync(allocator).on(stream);

  now, one can pass exec_policy to thrust functions

  to get an integer array of size 1000 whose lifetime is managed by unique_ptr,
  use: auto int_array = allocator.alloc_unique<int>(1000); int_array.get() gives
  the raw pointer.
*/
class CUDAWorkspaceAllocator {
  DGLContext ctx;

 public:
  typedef char value_type;

  void operator()(void* ptr) const {
    runtime::DeviceAPI::Get(ctx)->FreeWorkspace(ctx, ptr);
  }

  explicit CUDAWorkspaceAllocator(DGLContext ctx) : ctx(ctx) {}

  CUDAWorkspaceAllocator& operator=(const CUDAWorkspaceAllocator&) = default;

  template <typename T>
  std::unique_ptr<T, CUDAWorkspaceAllocator> alloc_unique(
      std::size_t size) const {
    return std::unique_ptr<T, CUDAWorkspaceAllocator>(
        reinterpret_cast<T*>(runtime::DeviceAPI::Get(ctx)->AllocWorkspace(
            ctx, sizeof(T) * size)),
        *this);
  }

  char* allocate(std::ptrdiff_t size) const {
    return reinterpret_cast<char*>(
        runtime::DeviceAPI::Get(ctx)->AllocWorkspace(ctx, size));
  }

  void deallocate(char* ptr, std::size_t) const {
    runtime::DeviceAPI::Get(ctx)->FreeWorkspace(ctx, ptr);
  }
};

template <typename T>
inline bool is_zero(T size) {
  return size == 0;
}

template <>
inline bool is_zero<dim3>(dim3 size) {
  return size.x == 0 || size.y == 0 || size.z == 0;
}

#define CUDA_DRIVER_CALL(x)                                             \
  {                                                                     \
    CUresult result = x;                                                \
    if (result != CUDA_SUCCESS && result != CUDA_ERROR_DEINITIALIZED) { \
      const char* msg;                                                  \
      cuGetErrorName(result, &msg);                                     \
      LOG(FATAL) << "CUDAError: " #x " failed with error: " << msg;     \
    }                                                                   \
  }

#define CUDA_CALL(func)                                      \
  {                                                          \
    cudaError_t e = (func);                                  \
    CHECK(e == cudaSuccess || e == cudaErrorCudartUnloading) \
        << "CUDA: " << cudaGetErrorString(e);                \
  }

#define CUDA_KERNEL_CALL(kernel, nblks, nthrs, shmem, stream, ...)            \
  {                                                                           \
    if (!dgl::runtime::is_zero((nblks)) && !dgl::runtime::is_zero((nthrs))) { \
      (kernel)<<<(nblks), (nthrs), (shmem), (stream)>>>(__VA_ARGS__);         \
      cudaError_t e = cudaGetLastError();                                     \
      CHECK(e == cudaSuccess || e == cudaErrorCudartUnloading)                \
          << "CUDA kernel launch error: " << cudaGetErrorString(e);           \
    }                                                                         \
  }

#define CUSPARSE_CALL(func)                                         \
  {                                                                 \
    cusparseStatus_t e = (func);                                    \
    CHECK(e == CUSPARSE_STATUS_SUCCESS) << "CUSPARSE ERROR: " << e; \
  }

#define CUBLAS_CALL(func)                                       \
  {                                                             \
    cublasStatus_t e = (func);                                  \
    CHECK(e == CUBLAS_STATUS_SUCCESS) << "CUBLAS ERROR: " << e; \
  }

#define CURAND_CALL(func)                                                      \
  {                                                                            \
    curandStatus_t e = (func);                                                 \
    CHECK(e == CURAND_STATUS_SUCCESS)                                          \
        << "CURAND Error: " << dgl::runtime::curandGetErrorString(e) << " at " \
        << __FILE__ << ":" << __LINE__;                                        \
  }

inline const char* curandGetErrorString(curandStatus_t error) {
  switch (error) {
    case CURAND_STATUS_SUCCESS:
      return "CURAND_STATUS_SUCCESS";
    case CURAND_STATUS_VERSION_MISMATCH:
      return "CURAND_STATUS_VERSION_MISMATCH";
    case CURAND_STATUS_NOT_INITIALIZED:
      return "CURAND_STATUS_NOT_INITIALIZED";
    case CURAND_STATUS_ALLOCATION_FAILED:
      return "CURAND_STATUS_ALLOCATION_FAILED";
    case CURAND_STATUS_TYPE_ERROR:
      return "CURAND_STATUS_TYPE_ERROR";
    case CURAND_STATUS_OUT_OF_RANGE:
      return "CURAND_STATUS_OUT_OF_RANGE";
    case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
      return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
    case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
      return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
    case CURAND_STATUS_LAUNCH_FAILURE:
      return "CURAND_STATUS_LAUNCH_FAILURE";
    case CURAND_STATUS_PREEXISTING_FAILURE:
      return "CURAND_STATUS_PREEXISTING_FAILURE";
    case CURAND_STATUS_INITIALIZATION_FAILED:
      return "CURAND_STATUS_INITIALIZATION_FAILED";
    case CURAND_STATUS_ARCH_MISMATCH:
      return "CURAND_STATUS_ARCH_MISMATCH";
    case CURAND_STATUS_INTERNAL_ERROR:
      return "CURAND_STATUS_INTERNAL_ERROR";
  }
  // To suppress compiler warning.
  return "Unrecognized curand error string";
}

/**
 * @brief Cast data type to cudaDataType_t.
 */
template <typename T>
struct cuda_dtype {
  static constexpr cudaDataType_t value = CUDA_R_32F;
};

template <>
struct cuda_dtype<__half> {
  static constexpr cudaDataType_t value = CUDA_R_16F;
};

#if BF16_ENABLED
template <>
struct cuda_dtype<__nv_bfloat16> {
  static constexpr cudaDataType_t value = CUDA_R_16BF;
};
#endif  // BF16_ENABLED

template <>
struct cuda_dtype<float> {
  static constexpr cudaDataType_t value = CUDA_R_32F;
};

template <>
struct cuda_dtype<double> {
  static constexpr cudaDataType_t value = CUDA_R_64F;
};

/*
 * \brief Accumulator type for SpMM.
 */
template <typename T>
struct accum_dtype {
  typedef float type;
};

template <>
struct accum_dtype<__half> {
  typedef float type;
};

#if BF16_ENABLED
template <>
struct accum_dtype<__nv_bfloat16> {
  typedef float type;
};
#endif  // BF16_ENABLED

template <>
struct accum_dtype<float> {
  typedef float type;
};

template <>
struct accum_dtype<double> {
  typedef double type;
};

#if CUDART_VERSION >= 11000
/**
 * @brief Cast index data type to cusparseIndexType_t.
 */
template <typename T>
struct cusparse_idtype {
  static constexpr cusparseIndexType_t value = CUSPARSE_INDEX_32I;
};

template <>
struct cusparse_idtype<int32_t> {
  static constexpr cusparseIndexType_t value = CUSPARSE_INDEX_32I;
};

template <>
struct cusparse_idtype<int64_t> {
  static constexpr cusparseIndexType_t value = CUSPARSE_INDEX_64I;
};
#endif

/** @brief Thread local workspace */
class CUDAThreadEntry {
 public:
  /** @brief The cusparse handler */
  cusparseHandle_t cusparse_handle{nullptr};
  /** @brief The cublas handler */
  cublasHandle_t cublas_handle{nullptr};
  /** @brief thread local pool*/
  WorkspacePool pool;
  /** @brief constructor */
  CUDAThreadEntry();
  // get the threadlocal workspace
  static CUDAThreadEntry* ThreadLocal();
};

/** @brief Get the current CUDA stream */
cudaStream_t getCurrentCUDAStream();
}  // namespace runtime
}  // namespace dgl
#endif  // DGL_RUNTIME_CUDA_CUDA_COMMON_H_
