/**
 *  Copyright (c) 2020 by Contributors
 * @file array/cuda/utils.h
 * @brief Utilities for CUDA kernels.
 */
#ifndef DGL_ARRAY_CUDA_UTILS_H_
#define DGL_ARRAY_CUDA_UTILS_H_

#include <dgl/runtime/c_runtime_api.h>
#include <dgl/runtime/device_api.h>
#include <dgl/runtime/ndarray.h>
#include <dmlc/logging.h>

#include <cub/cub.cuh>
#include <type_traits>

#include "../../runtime/cuda/cuda_common.h"

namespace dgl {
namespace cuda {

#define CUDA_MAX_NUM_BLOCKS_X 0x7FFFFFFF
#define CUDA_MAX_NUM_BLOCKS_Y 0xFFFF
#define CUDA_MAX_NUM_BLOCKS_Z 0xFFFF
// The max number of threads per block
#define CUDA_MAX_NUM_THREADS 256

/** @brief Calculate the number of threads needed given the dimension length.
 *
 * It finds the biggest number that is smaller than min(dim, max_nthrs)
 * and is also power of two.
 */
inline int FindNumThreads(int dim, int max_nthrs = CUDA_MAX_NUM_THREADS) {
  CHECK_GE(dim, 0);
  if (dim == 0) return 1;
  int ret = max_nthrs;
  while (ret > dim) {
    ret = ret >> 1;
  }
  return ret;
}

template <typename T>
int _NumberOfBits(const T& range) {
  if (range <= 1) {
    // ranges of 0 or 1 require no bits to store
    return 0;
  }

  int bits = 1;
  const auto urange = static_cast<std::make_unsigned_t<T>>(range);
  while (bits < static_cast<int>(sizeof(T) * 8) && (1ull << bits) < urange) {
    ++bits;
  }

  if (bits < static_cast<int>(sizeof(T) * 8)) {
    CHECK_EQ((range - 1) >> bits, 0);
  }
  CHECK_NE((range - 1) >> (bits - 1), 0);

  return bits;
}

/**
 * @brief Find number of blocks is smaller than nblks and max_nblks
 * on the given axis ('x', 'y' or 'z').
 */
template <char axis>
inline int FindNumBlocks(int nblks, int max_nblks = -1) {
  int default_max_nblks = -1;
  switch (axis) {
    case 'x':
      default_max_nblks = CUDA_MAX_NUM_BLOCKS_X;
      break;
    case 'y':
      default_max_nblks = CUDA_MAX_NUM_BLOCKS_Y;
      break;
    case 'z':
      default_max_nblks = CUDA_MAX_NUM_BLOCKS_Z;
      break;
    default:
      LOG(FATAL) << "Axis " << axis << " not recognized";
      break;
  }
  if (max_nblks == -1) max_nblks = default_max_nblks;
  CHECK_NE(nblks, 0);
  if (nblks < max_nblks) return nblks;
  return max_nblks;
}

template <typename T>
__device__ __forceinline__ T _ldg(T* addr) {
#if __CUDA_ARCH__ >= 350
  return __ldg(addr);
#else
  return *addr;
#endif
}

/**
 * @brief Return true if the given bool flag array is all true.
 * The input bool array is in int8_t type so it is aligned with byte address.
 *
 * @param flags The bool array.
 * @param length The length.
 * @param ctx Device context.
 * @return True if all the flags are true.
 */
bool AllTrue(int8_t* flags, int64_t length, const DGLContext& ctx);

/**
 * @brief CUDA Kernel of filling the vector started from ptr of size length
 *        with val.
 * @note internal use only.
 */
template <typename DType>
__global__ void _FillKernel(DType* ptr, size_t length, DType val) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride_x = gridDim.x * blockDim.x;
  while (tx < length) {
    ptr[tx] = val;
    tx += stride_x;
  }
}

/** @brief Fill the vector started from ptr of size length with val */
template <typename DType>
void _Fill(DType* ptr, size_t length, DType val) {
  cudaStream_t stream = runtime::getCurrentCUDAStream();
  int nt = FindNumThreads(length);
  int nb =
      (length + nt - 1) / nt;  // on x-axis, no need to worry about upperbound.
  CUDA_KERNEL_CALL(cuda::_FillKernel, nb, nt, 0, stream, ptr, length, val);
}

/**
 * @brief Search adjacency list linearly for each (row, col) pair and
 * write the data under the matched position in the indices array to the output.
 *
 * If there is no match, the value in \c filler is written.
 * If there are multiple matches, only the first match is written.
 * If the given data array is null, write the matched position to the output.
 */
template <typename IdType, typename DType>
__global__ void _LinearSearchKernel(
    const IdType* indptr, const IdType* indices, const IdType* data,
    const IdType* row, const IdType* col, int64_t row_stride,
    int64_t col_stride, int64_t length, const DType* weights, DType filler,
    DType* out) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride_x = gridDim.x * blockDim.x;
  while (tx < length) {
    int rpos = tx * row_stride, cpos = tx * col_stride;
    IdType v = -1;
    const IdType r = row[rpos], c = col[cpos];
    for (IdType i = indptr[r]; i < indptr[r + 1]; ++i) {
      if (indices[i] == c) {
        v = data ? data[i] : i;
        break;
      }
    }
    if (v == -1) {
      out[tx] = filler;
    } else {
      // The casts here are to be able to handle DType being __half.
      // GCC treats int64_t as a distinct type from long long, so
      // without the explcit cast to long long, it errors out saying
      // that the implicit cast results in an ambiguous choice of
      // constructor for __half.
      // The using statement is to avoid a linter error about using
      // long or long long.
      using LongLong = long long;  // NOLINT
      out[tx] = weights ? weights[v] : DType(LongLong(v));
    }
    tx += stride_x;
  }
}

#if BF16_ENABLED
/**
 * @brief Specialization for bf16 because conversion from long long to bfloat16
 * doesn't exist before SM80.
 */
template <typename IdType>
__global__ void _LinearSearchKernel(
    const IdType* indptr, const IdType* indices, const IdType* data,
    const IdType* row, const IdType* col, int64_t row_stride,
    int64_t col_stride, int64_t length, const __nv_bfloat16* weights,
    __nv_bfloat16 filler, __nv_bfloat16* out) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride_x = gridDim.x * blockDim.x;
  while (tx < length) {
    int rpos = tx * row_stride, cpos = tx * col_stride;
    IdType v = -1;
    const IdType r = row[rpos], c = col[cpos];
    for (IdType i = indptr[r]; i < indptr[r + 1]; ++i) {
      if (indices[i] == c) {
        v = data ? data[i] : i;
        break;
      }
    }
    if (v == -1) {
      out[tx] = filler;
    } else {
      // If the result is saved in bf16, it should be fine to convert it to
      // float first
      out[tx] = weights ? weights[v] : __nv_bfloat16(static_cast<float>(v));
    }
    tx += stride_x;
  }
}
#endif  // BF16_ENABLED

template <typename DType>
inline DType GetCUDAScalar(
    runtime::DeviceAPI* device_api, DGLContext ctx, const DType* cuda_ptr) {
  DType result;
  device_api->CopyDataFromTo(
      cuda_ptr, 0, &result, 0, sizeof(result), ctx, DGLContext{kDGLCPU, 0},
      DGLDataTypeTraits<DType>::dtype);
  return result;
}

/**
 * @brief Given a sorted array and a value this function returns the index
 * of the first element which compares greater than value.
 *
 * This function assumes 0-based index
 * @param A: ascending sorted array
 * @param n: size of the A
 * @param x: value to search in A
 * @return index, i, of the first element st. A[i]>x. If x>=A[n-1] returns n.
 * if x<A[0] then it returns 0.
 */
template <typename IdType>
__device__ IdType _UpperBound(const IdType* A, int64_t n, IdType x) {
  IdType l = 0, r = n, m = 0;
  while (l < r) {
    m = l + (r - l) / 2;
    if (x >= A[m]) {
      l = m + 1;
    } else {
      r = m;
    }
  }
  return l;
}

/**
 * @brief Given a sorted array and a value this function returns the index
 * of the element who is equal to val. If not exist returns n+1
 *
 * This function assumes 0-based index
 * @param A: ascending sorted array
 * @param n: size of the A
 * @param x: value to search in A
 * @return index, i, st. A[i]==x. If such an index not exists returns 'n'.
 */
template <typename IdType>
__device__ IdType _BinarySearch(const IdType* A, int64_t n, IdType x) {
  IdType l = 0, r = n - 1, m = 0;
  while (l <= r) {
    m = l + (r - l) / 2;
    if (A[m] == x) {
      return m;
    }
    if (A[m] < x) {
      l = m + 1;
    } else {
      r = m - 1;
    }
  }
  return n;  // not found
}

template <typename DType, typename BoolType>
void MaskSelect(
    runtime::DeviceAPI* device, const DGLContext& ctx, const DType* input,
    const BoolType* mask, DType* output, int64_t n, int64_t* rst,
    cudaStream_t stream) {
  size_t workspace_size = 0;
  CUDA_CALL(cub::DeviceSelect::Flagged(
      nullptr, workspace_size, input, mask, output, rst, n, stream));
  void* workspace = device->AllocWorkspace(ctx, workspace_size);
  CUDA_CALL(cub::DeviceSelect::Flagged(
      workspace, workspace_size, input, mask, output, rst, n, stream));
  device->FreeWorkspace(ctx, workspace);
}

inline void* GetDevicePointer(runtime::NDArray array) {
  void* ptr = array->data;
  if (array.IsPinned()) {
    CUDA_CALL(cudaHostGetDevicePointer(&ptr, ptr, 0));
  }
  return ptr;
}

}  // namespace cuda
}  // namespace dgl

#endif  // DGL_ARRAY_CUDA_UTILS_H_
