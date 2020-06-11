#include <cstdio>
#include <vector>
#include <dgl/array.h>

#include "../../runtime/cuda/cuda_common.h"
#include "../../c_api_common.h"
#include "../array_op.h"

#define THREADS 1024

namespace dgl {
namespace aten {
namespace impl {

template <typename FloatType>
__global__ void _FillKernel(FloatType* ptr, int64_t length, FloatType val) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride_x = gridDim.x * blockDim.x;
  while (tx < length) {
    ptr[tx] = val;
    tx += stride_x;
  }
}

template <typename FloatType>
__global__ void fps_kernel(const FloatType *array_data, const int64_t batch_size, const int64_t sample_points,
                           const int64_t point_in_batch, const int64_t dim,
                           const int64_t *start_idx, FloatType *dist_data, int64_t *ret_data) {
  const int64_t thread_idx = threadIdx.x;
  const int64_t batch_idx = blockIdx.x;

  const int64_t array_start = point_in_batch * batch_idx;
  const int64_t ret_start = sample_points * batch_idx;

  __shared__ FloatType dist_max_ht[THREADS];
  __shared__ int64_t dist_argmax_ht[THREADS];

  // start with random initialization
  if (thread_idx == 0) {
    ret_data[ret_start] = array_start + start_idx[batch_idx];
  }

  // sample the rest `sample_points - 1` points
  for (auto i = 0; i < sample_points - 1; i++) {
    __syncthreads();

    // the last sampled point
    int64_t sample_idx = ret_data[ret_start + i];
    FloatType dist_max = (FloatType)(-1.);
    int64_t dist_argmax = 0;

    // multi-thread distance calculation
    for (auto j = thread_idx; j < point_in_batch; j += THREADS) {
      FloatType one_dist = (FloatType)(0.);
      for (auto d = 0; d < dim; d++) {
        FloatType tmp = array_data[(array_start + j) * dim + d] - array_data[sample_idx * dim + d];
        one_dist += tmp * tmp;
      }

      if (i == 0 || dist_data[array_start + j] > one_dist) {
        dist_data[array_start + j] = one_dist;
      }

      if (dist_data[array_start + j] > dist_max) {
        dist_argmax = j;
        dist_max = dist_data[array_start + j];
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

template <DLDeviceType XPU, typename FloatType>
void FarthestPointSampler(NDArray array, int64_t batch_size, int64_t sample_points,
    NDArray dist, IdArray start_idx, IdArray result) {
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();

  const FloatType* array_data = static_cast<FloatType*>(array->data);

  const int64_t point_in_batch = array->shape[0] / batch_size;
  const int64_t dim = array->shape[1];

  // Init return value
  // IdArray ret = NewIdArray(sample_points * batch_size, ctx, sizeof(int64_t) * 8);
  int64_t* ret_data = static_cast<int64_t*>(result->data);
  // std::fill(ret_data, ret_data + sample_points * batch_size , 0);
  /*
  _FillKernel<<<(sample_points * batch_size + THREADS - 1) / THREADS, THREADS, 0, thr_entry->stream>>>(
    ret_data, sample_points * batch_size, static_cast<int64_t>(0));
  */

  // Init distance
  // NDArray dist = NDArray::Empty({array->shape[0]}, array->dtype, ctx);
  FloatType* dist_data = static_cast<FloatType*>(dist->data);
  // std::fill(dist_data, dist_data + point_in_batch, 1e9);
  /*
  _FillKernel<<<(array->shape[0] + THREADS - 1) / THREADS, THREADS, 0, thr_entry->stream>>>(
    dist_data, static_cast<int64_t>(array->shape[0]), static_cast<FloatType>(1e9));
  */

  // Init sample for each cloud in the batch
  // IdArray start_idx = NewIdArray(batch_size, ctx, sizeof(int64_t) * 8);
  int64_t* start_idx_data = static_cast<int64_t*>(start_idx->data);
  /*
  std::vector<int64_t> start_idx_cpu(batch_size);
  for (auto i = 0; i < batch_size; i++) {
    start_idx_cpu[i] = (int64_t)(rand() % point_in_batch);
  }
  cudaMemcpy(start_idx_data, start_idx_cpu.data(), batch_size*sizeof(int64_t), cudaMemcpyHostToDevice);
  */

  fps_kernel<<<batch_size, THREADS, 0, thr_entry->stream>>>(
    array_data, batch_size, sample_points,
    point_in_batch, dim, start_idx_data, dist_data, ret_data);
  // return ret;
}

template void FarthestPointSampler<kDLGPU, float>(NDArray array, int64_t batch_size, int64_t sample_points,
    NDArray dist, IdArray start_idx, IdArray result);
template void FarthestPointSampler<kDLGPU, double>(NDArray array, int64_t batch_size, int64_t sample_points,
    NDArray dist, IdArray start_idx, IdArray result);

} // impl
} // aten
} // dgl
