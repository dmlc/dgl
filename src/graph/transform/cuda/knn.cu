/*!
 *  Copyright (c) 2020 by Contributors
 * \file graph/transform/knn.cu
 * \brief k-nearest-neighbor (KNN) implementation (cuda)
 */

#include <dgl/array.h>
#include <algorithm>
#include <string>
#include <limits>
#include "../../../runtime/cuda/cuda_common.h"
#include "../../../array/cuda/utils.h"
#include "../knn.h"

namespace dgl {
namespace transform {
namespace impl {

static constexpr int THREADS = 1024;

template <typename FloatType, typename IdType>
__global__ void _brute_force_knn_kernel(const FloatType* data_points, const IdType* data_offsets,
                                       const FloatType* query_points, const IdType* query_offsets,
                                       const int k, FloatType* dists, IdType* query_out,
                                       IdType* data_out, const int feature_size) {
  const int64_t batch_idx = blockIdx.x, thread_idx = threadIdx.x;
  const IdType data_start = data_offsets[batch_idx], data_end = data_offsets[batch_idx + 1];
  const IdType query_start = query_offsets[batch_idx], query_end = query_offsets[batch_idx + 1];

  for (IdType q_idx = query_start + thread_idx; q_idx < query_end; q_idx += blockDim.x) {
    for (IdType k_idx = 0; k_idx < k; ++k_idx) {
      query_out[q_idx * k + k_idx] = q_idx;
      dists[q_idx * k + k_idx] = std::numeric_limits<FloatType>::max();
    }
    FloatType worst_dist = std::numeric_limits<FloatType>::max();

    for (IdType d_idx = data_start; d_idx < data_end; ++d_idx) {
      FloatType tmp_dist = 0;
      IdType dim_idx = 0;
      bool early_stop = false;

      // expand loop (x4), #pragma unroll has too many if-conditions
      for (; dim_idx < feature_size - 3; dim_idx += 4) {
        FloatType diff0 = query_points[q_idx * feature_size + dim_idx]
        - data_points[d_idx * feature_size + dim_idx];
        FloatType diff1 = query_points[q_idx * feature_size + dim_idx + 1]
        - data_points[d_idx * feature_size + dim_idx + 1];
        FloatType diff2 = query_points[q_idx * feature_size + dim_idx + 2]
        - data_points[d_idx * feature_size + dim_idx + 2];
        FloatType diff3 = query_points[q_idx * feature_size + dim_idx + 3]
        - data_points[d_idx * feature_size + dim_idx + 3];
        tmp_dist += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;

        // stop if curent distance > all top-k distances.
        if (tmp_dist > worst_dist) {
          early_stop = true;
          dim_idx = feature_size;
          break;
        }
      }

      for (; dim_idx < feature_size; ++dim_idx) {
        FloatType diff = query_points[q_idx * feature_size + dim_idx]
          - data_points[d_idx * feature_size + dim_idx];
        tmp_dist += diff * diff;
      }

      if (early_stop) continue;

      // maintain a monotonic array by "insert sort"
      IdType out_offset = q_idx * k;
      for (IdType k1 = 0; k1 < k; ++k1) {
        if (dists[out_offset + k1] > tmp_dist) {
          for (IdType k2 = k - 1; k2 > k1; --k2) {
            dists[out_offset + k2] = dists[out_offset + k2 - 1];
            data_out[out_offset + k2] = data_out[out_offset + k2 - 1];
          }
          dists[out_offset + k1] = tmp_dist;
          data_out[out_offset + k1] = d_idx;
          worst_dist = dists[out_offset + k - 1];
          break;
        }
      }
    }
  }
}

template <typename FloatType, typename IdType>
__global__ void brute_force_knn_kernel(const FloatType* data_points, const IdType* data_offsets,
                                       const FloatType* query_points, const IdType* query_offsets,
                                       const int k, FloatType* dists, IdType* query_out,
                                       IdType* data_out, const int num_batches,
                                       const int feature_size) {
  const int64_t q_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t batch_idx = 0;
  for (IdType b = 0; b < num_batches + 1; ++b) {
    if (query_offsets[b] > q_idx) { batch_idx = b - 1; break; }
  }

  const IdType data_start = data_offsets[batch_idx], data_end = data_offsets[batch_idx + 1];

  for (IdType k_idx = 0; k_idx < k; ++k_idx) {
    query_out[q_idx * k + k_idx] = q_idx;
    dists[q_idx * k + k_idx] = std::numeric_limits<FloatType>::max();
  }
  FloatType worst_dist = std::numeric_limits<FloatType>::max();

  for (IdType d_idx = data_start; d_idx < data_end; ++d_idx) {
    FloatType tmp_dist = 0;
    IdType dim_idx = 0;
    bool early_stop = false;

    // expand loop (x4), #pragma unroll has too many if-conditions
    for (; dim_idx < feature_size - 3; dim_idx += 4) {
      FloatType diff0 = query_points[q_idx * feature_size + dim_idx]
      - data_points[d_idx * feature_size + dim_idx];
      FloatType diff1 = query_points[q_idx * feature_size + dim_idx + 1]
      - data_points[d_idx * feature_size + dim_idx + 1];
      FloatType diff2 = query_points[q_idx * feature_size + dim_idx + 2]
      - data_points[d_idx * feature_size + dim_idx + 2];
      FloatType diff3 = query_points[q_idx * feature_size + dim_idx + 3]
      - data_points[d_idx * feature_size + dim_idx + 3];
      tmp_dist += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;

      // stop if curent distance > all top-k distances.
      if (tmp_dist > worst_dist) {
        early_stop = true;
        dim_idx = feature_size;
        break;
      }
    }

    for (; dim_idx < feature_size; ++dim_idx) {
      FloatType diff = query_points[q_idx * feature_size + dim_idx]
        - data_points[d_idx * feature_size + dim_idx];
      tmp_dist += diff * diff;
    }

    if (early_stop) continue;

    // maintain a monotonic array by "insert sort"
    IdType out_offset = q_idx * k;
    for (IdType k1 = 0; k1 < k; ++k1) {
      if (dists[out_offset + k1] > tmp_dist) {
        for (IdType k2 = k - 1; k2 > k1; --k2) {
          dists[out_offset + k2] = dists[out_offset + k2 - 1];
          data_out[out_offset + k2] = data_out[out_offset + k2 - 1];
        }
        dists[out_offset + k1] = tmp_dist;
        data_out[out_offset + k1] = d_idx;
        worst_dist = dists[out_offset + k - 1];
        break;
      }
    }
  }
}

template <typename FloatType, typename IdType>
void BruteForceKNNCuda(const NDArray& data_points, const IdArray& data_offsets,
                       const NDArray& query_points, const IdArray& query_offsets,
                       const int k, IdArray result) {
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  const int64_t batch_size = data_offsets->shape[0] - 1;
  const int64_t feature_size = data_points->shape[1];
  const IdType* data_offsets_data = data_offsets.Ptr<IdType>();
  const IdType* query_offsets_data = query_offsets.Ptr<IdType>();
  const FloatType* data_points_data = data_points.Ptr<FloatType>();
  const FloatType* query_points_data = query_points.Ptr<FloatType>();
  IdType* query_out = result.Ptr<IdType>();
  IdType* data_out = query_out + k * query_points->shape[0];

  FloatType* dists;
  CUDA_CALL(cudaMalloc(
    reinterpret_cast<void **>(&dists), k * query_points->shape[0] * sizeof(FloatType)));

  // CUDA_KERNEL_CALL(brute_force_knn_kernel, batch_size, THREADS, 0, thr_entry->stream,
  //                  data_points_data, data_offsets_data, query_points_data, query_offsets_data,
  //                  k, dists, query_out, data_out, feature_size);
  const int64_t block_size = (query_points->shape[0] - 1) / THREADS + 1;
  CUDA_KERNEL_CALL(brute_force_knn_kernel, block_size, THREADS, 0, thr_entry->stream,
    data_points_data, data_offsets_data, query_points_data, query_offsets_data,
    k, dists, query_out, data_out, batch_size, feature_size);
  CUDA_CALL(cudaFree(dists));
}
}  // namespace impl

template <DLDeviceType XPU, typename FloatType, typename IdType>
void KNN(const NDArray& data_points, const IdArray& data_offsets,
         const NDArray& query_points, const IdArray& query_offsets,
         const int k, IdArray result, const std::string& algorithm) {
  if (algorithm == std::string("bruteforce")) {
    impl::BruteForceKNNCuda<FloatType, IdType>(
      data_points, data_offsets, query_points, query_offsets, k, result);
  } else {
    LOG(FATAL) << "Algorithm " << algorithm << " is not supported on CUDA";
  }
}

template void KNN<kDLGPU, float, int32_t>(
  const NDArray& data_points, const IdArray& data_offsets,
  const NDArray& query_points, const IdArray& query_offsets,
  const int k, IdArray result, const std::string& algorithm);
template void KNN<kDLGPU, float, int64_t>(
  const NDArray& data_points, const IdArray& data_offsets,
  const NDArray& query_points, const IdArray& query_offsets,
  const int k, IdArray result, const std::string& algorithm);
template void KNN<kDLGPU, double, int32_t>(
  const NDArray& data_points, const IdArray& data_offsets,
  const NDArray& query_points, const IdArray& query_offsets,
  const int k, IdArray result, const std::string& algorithm);
template void KNN<kDLGPU, double, int64_t>(
  const NDArray& data_points, const IdArray& data_offsets,
  const NDArray& query_points, const IdArray& query_offsets,
  const int k, IdArray result, const std::string& algorithm);

}  // namespace transform
}  // namespace dgl
