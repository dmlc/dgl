/*!
 *  Copyright (c) 2019 by Contributors
 * \file geometry/cuda/geometry_op_impl.cc
 * \brief Geometry operator CUDA implementation
 */
#include <dgl/array.h>

#include "../../runtime/cuda/cuda_common.h"
#include "../../c_api_common.h"
#include "../geometry_op.h"

#define THREADS 1024

namespace dgl {
namespace geometry {
namespace impl {

/*!
 * \brief Farthest Point Sampler without the need to compute all pairs of distance.
 * 
 * The input array has shape (N, d), where N is the number of points, and d is the dimension.
 * It consists of a (flatten) batch of point clouds.
 *
 * In each batch, the algorithm starts with the sample index specified by ``start_idx``.
 * Then for each point, we maintain the minimum to-sample distance.
 * Finally, we pick the point with the maximum such distance.
 * This process will be repeated for ``sample_points`` - 1 times.
 */
template <typename FloatType, typename IdType>
__global__ void fps_kernel(const FloatType *array_data, const int64_t batch_size,
                           const int64_t sample_points, const int64_t point_in_batch,
                           const int64_t dim, const IdType *start_idx,
                           FloatType *dist_data, IdType *ret_data) {
  const int64_t thread_idx = threadIdx.x;
  const int64_t batch_idx = blockIdx.x;

  const int64_t array_start = point_in_batch * batch_idx;
  const int64_t ret_start = sample_points * batch_idx;

  __shared__ FloatType dist_max_ht[THREADS];
  __shared__ int64_t dist_argmax_ht[THREADS];

  // start with random initialization
  if (thread_idx == 0) {
    ret_data[ret_start] = (IdType)(start_idx[batch_idx]);
  }

  // sample the rest `sample_points - 1` points
  for (auto i = 0; i < sample_points - 1; i++) {
    __syncthreads();

    // the last sampled point
    int64_t sample_idx = (int64_t)(ret_data[ret_start + i]);
    FloatType dist_max = (FloatType)(-1.);
    int64_t dist_argmax = 0;

    // multi-thread distance calculation
    for (auto j = thread_idx; j < point_in_batch; j += THREADS) {
      FloatType one_dist = (FloatType)(0.);
      for (auto d = 0; d < dim; d++) {
        FloatType tmp = array_data[(array_start + j) * dim + d] -
            array_data[(array_start + sample_idx) * dim + d];
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

    /*
     * \brief Parallel Reduction
     *
     * Suppose the maximum is dist_max_ht[k], where 0 <= k < THREAD.
     * After loop at j = 1, the maximum is propagated to [k-1].
     * After loop at j = 2, the maximum is propagated to the range [k-3] to [k].
     * After loop at j = 4, the maximum is propagated to the range [k-7] to [k].
     * After loop at any j < THREADS, we can see [k - 2*j + 1] to [k] are all covered by the maximum.
     * The max value of j is at least floor(THREAD / 2), and it is sufficient to cover [0] with the maximum.
     */

    for (auto j = 1; j < THREADS; j *= 2) {
      __syncthreads();
      if ((thread_idx + j) < THREADS && dist_max_ht[thread_idx] < dist_max_ht[thread_idx + j]) {
          dist_max_ht[thread_idx] = dist_max_ht[thread_idx + j];
          dist_argmax_ht[thread_idx] = dist_argmax_ht[thread_idx + j];
      }
    }

    if (thread_idx == 0) {
      ret_data[ret_start + i + 1] = (IdType)(dist_argmax_ht[0]);
    }
  }
}

template <DLDeviceType XPU, typename FloatType, typename IdType>
void FarthestPointSampler(NDArray array, int64_t batch_size, int64_t sample_points,
    NDArray dist, IdArray start_idx, IdArray result) {
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();

  const FloatType* array_data = static_cast<FloatType*>(array->data);

  const int64_t point_in_batch = array->shape[0] / batch_size;
  const int64_t dim = array->shape[1];

  // return value
  IdType* ret_data = static_cast<IdType*>(result->data);

  // distance
  FloatType* dist_data = static_cast<FloatType*>(dist->data);

  // sample for each cloud in the batch
  IdType* start_idx_data = static_cast<IdType*>(start_idx->data);

  fps_kernel<<<batch_size, THREADS, 0, thr_entry->stream>>>(
    array_data, batch_size, sample_points,
    point_in_batch, dim, start_idx_data, dist_data, ret_data);
}

template void FarthestPointSampler<kDLGPU, float, int32_t>(
    NDArray array, int64_t batch_size, int64_t sample_points,
    NDArray dist, IdArray start_idx, IdArray result);
template void FarthestPointSampler<kDLGPU, float, int64_t>(
    NDArray array, int64_t batch_size, int64_t sample_points,
    NDArray dist, IdArray start_idx, IdArray result);
template void FarthestPointSampler<kDLGPU, double, int32_t>(
    NDArray array, int64_t batch_size, int64_t sample_points,
    NDArray dist, IdArray start_idx, IdArray result);
template void FarthestPointSampler<kDLGPU, double, int64_t>(
    NDArray array, int64_t batch_size, int64_t sample_points,
    NDArray dist, IdArray start_idx, IdArray result);

}  // namespace impl
}  // namespace geometry
}  // namespace dgl
