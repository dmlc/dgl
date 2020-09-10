/*!
 *  Copyright (c) 2017 by Contributors
 * \file cuda_common.h
 * \brief Common utilities for CUDA
 */
#ifndef DGL_RUNTIME_CUDA_CUDA_COMMON_H_
#define DGL_RUNTIME_CUDA_CUDA_COMMON_H_

#include <cublas_v2.h>
#include <cusparse.h>
#include <cuda_runtime.h>
#include <dgl/runtime/packed_func.h>
#include <string>
#include "../workspace_pool.h"

namespace dgl {
namespace runtime {

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
      const char *msg;                                                  \
      cuGetErrorName(result, &msg);                                     \
      LOG(FATAL)                                                        \
          << "CUDAError: " #x " failed with error: " << msg;            \
    }                                                                   \
  }

#define CUDA_CALL(func)                                            \
  {                                                                \
    cudaError_t e = (func);                                        \
    CHECK(e == cudaSuccess || e == cudaErrorCudartUnloading)       \
        << "CUDA: " << cudaGetErrorString(e);                      \
  }

#define CUDA_KERNEL_CALL(kernel, nblks, nthrs, shmem, stream, ...) \
  {                                                                \
    if (!dgl::runtime::is_zero((nblks)) &&                         \
        !dgl::runtime::is_zero((nthrs))) {                         \
      (kernel) <<< (nblks), (nthrs), (shmem), (stream) >>>         \
        (__VA_ARGS__);                                             \
      cudaError_t e = cudaGetLastError();                          \
      CHECK(e == cudaSuccess || e == cudaErrorCudartUnloading)     \
          << "CUDA kernel launch error: "                          \
          << cudaGetErrorString(e);                                \
    }                                                              \
  }

#define CUSPARSE_CALL(func)                                        \
  {                                                                \
    cusparseStatus_t e = (func);                                   \
    CHECK(e == CUSPARSE_STATUS_SUCCESS)                            \
        << "CUSPARSE ERROR: " << e;                                \
  }

#define CUBLAS_CALL(func)                                          \
  {                                                                \
    cublasStatus_t e = (func);                                     \
    CHECK(e == CUBLAS_STATUS_SUCCESS) << "CUBLAS ERROR: " << e;    \
  }


/*! \brief Thread local workspace */
class CUDAThreadEntry {
 public:
  /*! \brief The cuda stream */
  cudaStream_t stream{nullptr};
  /*! \brief The cusparse handler */
  cusparseHandle_t cusparse_handle{nullptr};
  /*! \brief The cublas handler */
  cublasHandle_t cublas_handle{nullptr};
  /*! \brief thread local pool*/
  WorkspacePool pool;
  /*! \brief constructor */
  CUDAThreadEntry();
  // get the threadlocal workspace
  static CUDAThreadEntry* ThreadLocal();
};
}  // namespace runtime
}  // namespace dgl
#endif  // DGL_RUNTIME_CUDA_CUDA_COMMON_H_
