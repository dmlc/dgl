/**
 *  Copyright (c) 2021 by Contributors
 * @file array/cuda/negative_sampling.cu
 * @brief rowwise sampling
 */

#include <curand_kernel.h>
#include <dgl/array.h>
#include <dgl/array_iterator.h>
#include <dgl/random.h>

#include <cub/cub.cuh>

#include "../../runtime/cuda/cuda_common.h"
#include "./utils.h"

using namespace dgl::runtime;

namespace dgl {
namespace aten {
namespace impl {

namespace {

template <typename IdType>
__global__ void _GlobalUniformNegativeSamplingKernel(
    const IdType* __restrict__ indptr, const IdType* __restrict__ indices,
    IdType* __restrict__ row, IdType* __restrict__ col, int64_t num_row,
    int64_t num_col, int64_t num_samples, int num_trials,
    bool exclude_self_loops, int32_t random_seed) {
  int64_t tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride_x = gridDim.x * blockDim.x;

  curandStatePhilox4_32_10_t
      rng;  // this allows generating 4 32-bit ints at a time
  curand_init(random_seed * gridDim.x + blockIdx.x, threadIdx.x, 0, &rng);

  while (tx < num_samples) {
    for (int i = 0; i < num_trials; ++i) {
      uint4 result = curand4(&rng);
      // Turns out that result.x is always 0 with the above RNG.
      uint64_t y_hi = result.y >> 16;
      uint64_t y_lo = result.y & 0xFFFF;
      uint64_t z = static_cast<uint64_t>(result.z);
      uint64_t w = static_cast<uint64_t>(result.w);
      int64_t u = static_cast<int64_t>(((y_lo << 32L) | z) % num_row);
      int64_t v = static_cast<int64_t>(((y_hi << 32L) | w) % num_col);

      if (exclude_self_loops && (u == v)) continue;

      // binary search of v among indptr[u:u+1]
      int64_t b = indptr[u], e = indptr[u + 1] - 1;
      bool found = false;
      while (b <= e) {
        int64_t m = (b + e) / 2;
        if (indices[m] == v) {
          found = true;
          break;
        } else if (indices[m] < v) {
          b = m + 1;
        } else {
          e = m - 1;
        }
      }

      if (!found) {
        row[tx] = u;
        col[tx] = v;
        break;
      }
    }

    tx += stride_x;
  }
}

template <typename DType>
struct IsNotMinusOne {
  __device__ __forceinline__ bool operator()(const std::pair<DType, DType>& a) {
    return a.first != -1;
  }
};

/**
 * @brief Sort ordered pairs in ascending order, using \a tmp_major and \a
 * tmp_minor as temporary buffers, each with \a n elements.
 */
template <typename IdType>
void SortOrderedPairs(
    runtime::DeviceAPI* device, DGLContext ctx, IdType* major, IdType* minor,
    IdType* tmp_major, IdType* tmp_minor, int64_t n, cudaStream_t stream) {
  // Sort ordered pairs in lexicographical order by two radix sorts since
  // cub's radix sorts are stable.
  // We need a 2*n auxiliary storage to store the results form the first radix
  // sort.
  size_t s1 = 0, s2 = 0;
  void* tmp1 = nullptr;
  void* tmp2 = nullptr;

  // Radix sort by minor key first, reorder the major key in the progress.
  CUDA_CALL(cub::DeviceRadixSort::SortPairs(
      tmp1, s1, minor, tmp_minor, major, tmp_major, n, 0, sizeof(IdType) * 8,
      stream));
  tmp1 = device->AllocWorkspace(ctx, s1);
  CUDA_CALL(cub::DeviceRadixSort::SortPairs(
      tmp1, s1, minor, tmp_minor, major, tmp_major, n, 0, sizeof(IdType) * 8,
      stream));

  // Radix sort by major key next.
  CUDA_CALL(cub::DeviceRadixSort::SortPairs(
      tmp2, s2, tmp_major, major, tmp_minor, minor, n, 0, sizeof(IdType) * 8,
      stream));
  tmp2 = (s2 > s1) ? device->AllocWorkspace(ctx, s2)
                   : tmp1;  // reuse buffer if s2 <= s1
  CUDA_CALL(cub::DeviceRadixSort::SortPairs(
      tmp2, s2, tmp_major, major, tmp_minor, minor, n, 0, sizeof(IdType) * 8,
      stream));

  if (tmp1 != tmp2) device->FreeWorkspace(ctx, tmp2);
  device->FreeWorkspace(ctx, tmp1);
}

};  // namespace

template <DGLDeviceType XPU, typename IdType>
std::pair<IdArray, IdArray> CSRGlobalUniformNegativeSampling(
    const CSRMatrix& csr, int64_t num_samples, int num_trials,
    bool exclude_self_loops, bool replace, double redundancy) {
  auto ctx = csr.indptr->ctx;
  auto dtype = csr.indptr->dtype;
  const int64_t num_row = csr.num_rows;
  const int64_t num_col = csr.num_cols;
  const int64_t num_actual_samples =
      static_cast<int64_t>(num_samples * (1 + redundancy));
  IdArray row = Full<IdType>(-1, num_actual_samples, ctx);
  IdArray col = Full<IdType>(-1, num_actual_samples, ctx);
  IdArray out_row = IdArray::Empty({num_actual_samples}, dtype, ctx);
  IdArray out_col = IdArray::Empty({num_actual_samples}, dtype, ctx);
  IdType* row_data = row.Ptr<IdType>();
  IdType* col_data = col.Ptr<IdType>();
  IdType* out_row_data = out_row.Ptr<IdType>();
  IdType* out_col_data = out_col.Ptr<IdType>();
  auto device = runtime::DeviceAPI::Get(ctx);
  cudaStream_t stream = runtime::getCurrentCUDAStream();
  const int nt = cuda::FindNumThreads(num_actual_samples);
  const int nb = (num_actual_samples + nt - 1) / nt;
  std::pair<IdArray, IdArray> result;
  int64_t num_out;

  CUDA_KERNEL_CALL(
      _GlobalUniformNegativeSamplingKernel, nb, nt, 0, stream,
      csr.indptr.Ptr<IdType>(), csr.indices.Ptr<IdType>(), row_data, col_data,
      num_row, num_col, num_actual_samples, num_trials, exclude_self_loops,
      RandomEngine::ThreadLocal()->RandInt32());

  size_t tmp_size = 0;
  int64_t* num_out_cuda =
      static_cast<int64_t*>(device->AllocWorkspace(ctx, sizeof(int64_t)));
  IsNotMinusOne<IdType> op;
  PairIterator<IdType> begin(row_data, col_data);
  PairIterator<IdType> out_begin(out_row_data, out_col_data);
  CUDA_CALL(cub::DeviceSelect::If(
      nullptr, tmp_size, begin, out_begin, num_out_cuda, num_actual_samples, op,
      stream));
  void* tmp = device->AllocWorkspace(ctx, tmp_size);
  CUDA_CALL(cub::DeviceSelect::If(
      tmp, tmp_size, begin, out_begin, num_out_cuda, num_actual_samples, op,
      stream));
  num_out = cuda::GetCUDAScalar(device, ctx, num_out_cuda);

  if (!replace) {
    IdArray unique_row = IdArray::Empty({num_out}, dtype, ctx);
    IdArray unique_col = IdArray::Empty({num_out}, dtype, ctx);
    IdType* unique_row_data = unique_row.Ptr<IdType>();
    IdType* unique_col_data = unique_col.Ptr<IdType>();
    PairIterator<IdType> unique_begin(unique_row_data, unique_col_data);

    SortOrderedPairs(
        device, ctx, out_row_data, out_col_data, unique_row_data,
        unique_col_data, num_out, stream);

    size_t tmp_size_unique = 0;
    void* tmp_unique = nullptr;
    CUDA_CALL(cub::DeviceSelect::Unique(
        nullptr, tmp_size_unique, out_begin, unique_begin, num_out_cuda,
        num_out, stream));
    tmp_unique = (tmp_size_unique > tmp_size)
                     ? device->AllocWorkspace(ctx, tmp_size_unique)
                     : tmp;  // reuse buffer
    CUDA_CALL(cub::DeviceSelect::Unique(
        tmp_unique, tmp_size_unique, out_begin, unique_begin, num_out_cuda,
        num_out, stream));
    num_out = cuda::GetCUDAScalar(device, ctx, num_out_cuda);

    num_out = std::min(num_samples, num_out);
    result = {
        unique_row.CreateView({num_out}, dtype),
        unique_col.CreateView({num_out}, dtype)};

    if (tmp_unique != tmp) device->FreeWorkspace(ctx, tmp_unique);
  } else {
    num_out = std::min(num_samples, num_out);
    result = {
        out_row.CreateView({num_out}, dtype),
        out_col.CreateView({num_out}, dtype)};
  }

  device->FreeWorkspace(ctx, tmp);
  device->FreeWorkspace(ctx, num_out_cuda);
  return result;
}

template std::pair<IdArray, IdArray> CSRGlobalUniformNegativeSampling<
    kDGLCUDA, int32_t>(const CSRMatrix&, int64_t, int, bool, bool, double);
template std::pair<IdArray, IdArray> CSRGlobalUniformNegativeSampling<
    kDGLCUDA, int64_t>(const CSRMatrix&, int64_t, int, bool, bool, double);

};  // namespace impl
};  // namespace aten
};  // namespace dgl
