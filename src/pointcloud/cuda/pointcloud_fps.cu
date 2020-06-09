#include <dgl/array.h>
#include <cstdio>

#include "../../runtime/cuda/cuda_common.h"
#include "../../c_api_common.h"
#include "./pointcloud_fps.cuh"

#define THREADS 256

namespace dgl {
namespace aten {
namespace cuda {

template <typename DType>
__global__ void _FillKernel(DType* ptr, int64_t length, DType val) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride_x = gridDim.x * blockDim.x;
  while (tx < length) {
    ptr[tx] = val;
    tx += stride_x;
  }
}

template <typename DType>
__global__ void fps_kernel(const DType *array_data, const int64_t batch_size, const int64_t sample_points,
                           const DLContext ctx, const int64_t point_in_batch, const int64_t dim,
                           const int64_t *start_idx, DType *dist_data, int64_t *ret_data) {
  const int64_t thread_idx = threadIdx.x;
  const int64_t batch_idx = blockIdx.x;

  const int64_t array_start = point_in_batch * batch_idx;
  const int64_t ret_start = sample_points * batch_idx;

  __shared__ DType dist_max_ht[THREADS];
  __shared__ int64_t dist_argmax_ht[THREADS];

  // avoid race
  if (thread_idx == 0) {
    ret_data[ret_start] = array_start + 0; //start_idx[batch_idx];
  }

  // sample the rest `sample_points - 1` points
  for (auto i = 0; i < sample_points - 1; i++) {
    __syncthreads();

    // the last sampled point
    int64_t sample_idx = ret_data[ret_start + i];
    DType dist_max = (DType)(-1.);
    int64_t dist_argmax = 0;

    // multi-thread distance calculation
    for (auto j = thread_idx; j < point_in_batch; j += THREADS) {
      DType one_dist = (DType)(0.);
      for (auto d = 0; d < dim; d++) {
        DType tmp = array_data[(array_start + j) * dim + d] - array_data[sample_idx * dim + d];
        one_dist += tmp * tmp;
      }

      if (i == 0 || dist_data[array_start + j] > one_dist) {
        dist_data[array_start + j] = one_dist;
      }

      if (dist_data[array_start + j] > dist_max) {
        dist_argmax = j;
        dist_max = one_dist;
      }
    }

    dist_max_ht[thread_idx] = dist_max;
    dist_argmax_ht[thread_idx] = dist_argmax;

    // Parallel Reduction
    for (auto j = 1; j < THREADS; j *= 2) {
      __syncthreads();
      if ((thread_idx + j) < THREADS && dist_max_ht[thread_idx] < dist_max_ht[thread_idx + j]) {
          dist_max_ht[thread_idx] = dist_max_ht[thread_idx + j];
          dist_argmax_ht[thread_idx] = dist_argmax_ht[thread_idx + j];
      }
    }

    if (thread_idx == 0) {
      ret_data[ret_start + i + 1] = array_start + dist_argmax_ht[0];
    }
  }
}

template <typename DType>
IdArray _FPS_CUDA(NDArray array, int64_t batch_size, int64_t sample_points, DLContext ctx) {
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();

  const DType* array_data = static_cast<DType*>(array->data);

  const int64_t point_in_batch = array->shape[0] / batch_size;
  const int64_t dim = array->shape[1];

  // Init return value
  IdArray ret = NewIdArray(sample_points * batch_size, ctx, sizeof(int64_t) * 8);
  int64_t* ret_data = static_cast<int64_t*>(ret->data);
  // std::fill(ret_data, ret_data + sample_points * batch_size , 0);
  _FillKernel<<<(sample_points * batch_size + THREADS - 1) / THREADS, THREADS, 0, thr_entry->stream>>>(
    ret_data, sample_points * batch_size, static_cast<int64_t>(0));

  // Init distance
  NDArray dist = NDArray::Empty({array->shape[0]}, array->dtype, ctx);
  DType* dist_data = static_cast<DType*>(dist->data);
  // std::fill(dist_data, dist_data + point_in_batch, 1e9);
  _FillKernel<<<(array->shape[0] + THREADS - 1) / THREADS, THREADS, 0, thr_entry->stream>>>(
    dist_data, static_cast<int64_t>(array->shape[0]), static_cast<DType>(1e9));

  // Init sample for each cloud in the batch
  IdArray start_idx = NewIdArray(batch_size, ctx, sizeof(int64_t) * 8);
  int64_t* start_idx_data = static_cast<int64_t*>(start_idx->data);
  /*
  for (auto i = 0; i < batch_size; i++) {
    start_idx_data[i] = (int64_t)(rand() % point_in_batch);
  }
  */

  fps_kernel<<<batch_size, THREADS, 0, thr_entry->stream>>>(
    array_data, batch_size, sample_points, ctx,
    point_in_batch, dim, start_idx_data, dist_data, ret_data);
  return ret;
}

template IdArray _FPS_CUDA<float>(NDArray array, int64_t batch_size, int64_t sample_points, DLContext ctx);
template IdArray _FPS_CUDA<double>(NDArray array, int64_t batch_size, int64_t sample_points, DLContext ctx);

} // cuda
} // aten
} // dgl
