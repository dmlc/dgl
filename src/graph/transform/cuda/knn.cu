/**
 *  Copyright (c) 2020 by Contributors
 * @file graph/transform/cuda/knn.cu
 * @brief k-nearest-neighbor (KNN) implementation (cuda)
 */

#include <curand_kernel.h>
#include <dgl/array.h>
#include <dgl/random.h>
#include <dgl/runtime/device_api.h>

#include <algorithm>
#include <cub/cub.cuh>  // NOLINT
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include "../../../array/cuda/utils.h"
#include "../../../runtime/cuda/cuda_common.h"
#include "../knn.h"

namespace dgl {
namespace transform {
namespace impl {

/**
 * @brief Given input `size`, find the smallest value
 * greater or equal to `size` that is a multiple of `align`.
 *
 * e.g. Pow2Align(17, 4) = 20, Pow2Align(17, 8) = 24
 */
template <typename Type>
static __host__ __device__ std::enable_if_t<std::is_unsigned<Type>::value, Type>
Pow2Align(Type size, Type align) {
  if (align <= 1 || size <= 0) return size;
  return ((size - 1) | (align - 1)) + 1;
}

/**
 * @brief Utility class used to avoid linker errors with extern
 *  unsized shared memory arrays with templated type
 */
template <typename Type>
struct SharedMemory {
  __device__ inline operator Type*() {
    extern __shared__ int __smem[];
    return reinterpret_cast<Type*>(__smem);
  }

  __device__ inline operator const Type*() const {
    extern __shared__ int __smem[];
    return reinterpret_cast<Type*>(__smem);
  }
};

// specialize for double to avoid unaligned memory
// access compile errors
template <>
struct SharedMemory<double> {
  __device__ inline operator double*() {
    extern __shared__ double __smem_d[];
    return reinterpret_cast<double*>(__smem_d);
  }

  __device__ inline operator const double*() const {
    extern __shared__ double __smem_d[];
    return reinterpret_cast<double*>(__smem_d);
  }
};

/** @brief Compute Euclidean distance between two vectors in a cuda kernel */
template <typename FloatType, typename IdType>
__device__ FloatType
EuclideanDist(const FloatType* vec1, const FloatType* vec2, const int64_t dim) {
  FloatType dist = 0;
  IdType idx = 0;
  for (; idx < dim - 3; idx += 4) {
    FloatType diff0 = vec1[idx] - vec2[idx];
    FloatType diff1 = vec1[idx + 1] - vec2[idx + 1];
    FloatType diff2 = vec1[idx + 2] - vec2[idx + 2];
    FloatType diff3 = vec1[idx + 3] - vec2[idx + 3];

    dist += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
  }

  for (; idx < dim; ++idx) {
    FloatType diff = vec1[idx] - vec2[idx];
    dist += diff * diff;
  }

  return dist;
}

/**
 * @brief Compute Euclidean distance between two vectors in a cuda kernel,
 *  return positive infinite value if the intermediate distance is greater
 *  than the worst distance.
 */
template <typename FloatType, typename IdType>
__device__ FloatType EuclideanDistWithCheck(
    const FloatType* vec1, const FloatType* vec2, const int64_t dim,
    const FloatType worst_dist) {
  FloatType dist = 0;
  IdType idx = 0;
  bool early_stop = false;

  for (; idx < dim - 3; idx += 4) {
    FloatType diff0 = vec1[idx] - vec2[idx];
    FloatType diff1 = vec1[idx + 1] - vec2[idx + 1];
    FloatType diff2 = vec1[idx + 2] - vec2[idx + 2];
    FloatType diff3 = vec1[idx + 3] - vec2[idx + 3];

    dist += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
    if (dist > worst_dist) {
      early_stop = true;
      idx = dim;
      break;
    }
  }

  for (; idx < dim; ++idx) {
    FloatType diff = vec1[idx] - vec2[idx];
    dist += diff * diff;
    if (dist > worst_dist) {
      early_stop = true;
      break;
    }
  }

  if (early_stop) {
    return std::numeric_limits<FloatType>::max();
  } else {
    return dist;
  }
}

template <typename FloatType, typename IdType>
__device__ void BuildHeap(IdType* indices, FloatType* dists, int size) {
  for (int i = size / 2 - 1; i >= 0; --i) {
    IdType idx = i;
    while (true) {
      IdType largest = idx;
      IdType left = idx * 2 + 1;
      IdType right = left + 1;
      if (left < size && dists[left] > dists[largest]) {
        largest = left;
      }
      if (right < size && dists[right] > dists[largest]) {
        largest = right;
      }
      if (largest != idx) {
        IdType tmp_idx = indices[largest];
        indices[largest] = indices[idx];
        indices[idx] = tmp_idx;

        FloatType tmp_dist = dists[largest];
        dists[largest] = dists[idx];
        dists[idx] = tmp_dist;
        idx = largest;
      } else {
        break;
      }
    }
  }
}

template <typename FloatType, typename IdType>
__device__ void HeapInsert(
    IdType* indices, FloatType* dist, IdType new_idx, FloatType new_dist,
    int size, bool check_repeat = false) {
  if (new_dist > dist[0]) return;

  // check if we have it
  if (check_repeat) {
    for (IdType i = 0; i < size; ++i) {
      if (indices[i] == new_idx) return;
    }
  }

  IdType left = 0, right = 0, idx = 0, largest = 0;
  dist[0] = new_dist;
  indices[0] = new_idx;
  while (true) {
    left = idx * 2 + 1;
    right = left + 1;
    if (left < size && dist[left] > dist[largest]) {
      largest = left;
    }
    if (right < size && dist[right] > dist[largest]) {
      largest = right;
    }
    if (largest != idx) {
      IdType tmp_idx = indices[idx];
      indices[idx] = indices[largest];
      indices[largest] = tmp_idx;

      FloatType tmp_dist = dist[idx];
      dist[idx] = dist[largest];
      dist[largest] = tmp_dist;

      idx = largest;
    } else {
      break;
    }
  }
}

template <typename FloatType, typename IdType>
__device__ bool FlaggedHeapInsert(
    IdType* indices, FloatType* dist, bool* flags, IdType new_idx,
    FloatType new_dist, bool new_flag, int size, bool check_repeat = false) {
  if (new_dist > dist[0]) return false;

  // check if we have it
  if (check_repeat) {
    for (IdType i = 0; i < size; ++i) {
      if (indices[i] == new_idx) return false;
    }
  }

  IdType left = 0, right = 0, idx = 0, largest = 0;
  dist[0] = new_dist;
  indices[0] = new_idx;
  flags[0] = new_flag;
  while (true) {
    left = idx * 2 + 1;
    right = left + 1;
    if (left < size && dist[left] > dist[largest]) {
      largest = left;
    }
    if (right < size && dist[right] > dist[largest]) {
      largest = right;
    }
    if (largest != idx) {
      IdType tmp_idx = indices[idx];
      indices[idx] = indices[largest];
      indices[largest] = tmp_idx;

      FloatType tmp_dist = dist[idx];
      dist[idx] = dist[largest];
      dist[largest] = tmp_dist;

      bool tmp_flag = flags[idx];
      flags[idx] = flags[largest];
      flags[largest] = tmp_flag;

      idx = largest;
    } else {
      break;
    }
  }
  return true;
}

/**
 * @brief Brute force kNN kernel. Compute distance for each pair of input points
 * and get the result directly (without a distance matrix).
 */
template <typename FloatType, typename IdType>
__global__ void BruteforceKnnKernel(
    const FloatType* data_points, const IdType* data_offsets,
    const FloatType* query_points, const IdType* query_offsets, const int k,
    FloatType* dists, IdType* query_out, IdType* data_out,
    const int64_t num_batches, const int64_t feature_size) {
  const IdType q_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (q_idx >= query_offsets[num_batches]) return;
  IdType batch_idx = 0;
  for (IdType b = 0; b < num_batches + 1; ++b) {
    if (query_offsets[b] > q_idx) {
      batch_idx = b - 1;
      break;
    }
  }
  const IdType data_start = data_offsets[batch_idx],
               data_end = data_offsets[batch_idx + 1];

  for (IdType k_idx = 0; k_idx < k; ++k_idx) {
    query_out[q_idx * k + k_idx] = q_idx;
    dists[q_idx * k + k_idx] = std::numeric_limits<FloatType>::max();
  }
  FloatType worst_dist = std::numeric_limits<FloatType>::max();

  for (IdType d_idx = data_start; d_idx < data_end; ++d_idx) {
    FloatType tmp_dist = EuclideanDistWithCheck<FloatType, IdType>(
        query_points + q_idx * feature_size, data_points + d_idx * feature_size,
        feature_size, worst_dist);

    IdType out_offset = q_idx * k;
    HeapInsert<FloatType, IdType>(
        data_out + out_offset, dists + out_offset, d_idx, tmp_dist, k);
    worst_dist = dists[q_idx * k];
  }
}

/**
 * @brief Same as BruteforceKnnKernel, but use shared memory as buffer.
 *  This kernel divides query points and data points into blocks. For each
 *  query block, it will make a loop over all data blocks and compute distances.
 *  This kernel is faster when the dimension of input points is not large.
 */
template <typename FloatType, typename IdType>
__global__ void BruteforceKnnShareKernel(
    const FloatType* data_points, const IdType* data_offsets,
    const FloatType* query_points, const IdType* query_offsets,
    const IdType* block_batch_id, const IdType* local_block_id, const int k,
    FloatType* dists, IdType* query_out, IdType* data_out,
    const int64_t num_batches, const int64_t feature_size) {
  const IdType block_idx = static_cast<IdType>(blockIdx.x);
  const IdType block_size = static_cast<IdType>(blockDim.x);
  const IdType batch_idx = block_batch_id[block_idx];
  const IdType local_bid = local_block_id[block_idx];
  const IdType query_start = query_offsets[batch_idx] + block_size * local_bid;
  const IdType query_end =
      min(query_start + block_size, query_offsets[batch_idx + 1]);
  if (query_start >= query_end) return;
  const IdType query_idx = query_start + threadIdx.x;
  const IdType data_start = data_offsets[batch_idx];
  const IdType data_end = data_offsets[batch_idx + 1];

  // shared memory: points in block + distance buffer + result buffer
  FloatType* data_buff = SharedMemory<FloatType>();
  FloatType* query_buff = data_buff + block_size * feature_size;
  FloatType* dist_buff = query_buff + block_size * feature_size;
  IdType* res_buff = reinterpret_cast<IdType*>(Pow2Align<uint64_t>(
      reinterpret_cast<uint64_t>(dist_buff + block_size * k), sizeof(IdType)));
  FloatType worst_dist = std::numeric_limits<FloatType>::max();

  // initialize dist buff with inf value
  for (auto i = 0; i < k; ++i) {
    dist_buff[threadIdx.x + i * block_size] =
        std::numeric_limits<FloatType>::max();
  }

  // load query data to shared memory
  // TODO(tianqi): could be better here to exploit coalesce global memory
  // access.
  if (query_idx < query_end) {
    for (auto i = 0; i < feature_size; ++i) {
      // to avoid bank conflict, we use transpose here
      query_buff[threadIdx.x + i * block_size] =
          query_points[query_idx * feature_size + i];
    }
  }

  // perform computation on each tile
  for (auto tile_start = data_start; tile_start < data_end;
       tile_start += block_size) {
    // each thread load one data point into the shared memory
    IdType load_idx = tile_start + threadIdx.x;
    if (load_idx < data_end) {
      for (auto i = 0; i < feature_size; ++i) {
        data_buff[threadIdx.x * feature_size + i] =
            data_points[load_idx * feature_size + i];
      }
    }
    __syncthreads();

    // compute distance for one tile
    IdType true_block_size = min(data_end - tile_start, block_size);
    if (query_idx < query_end) {
      for (IdType d_idx = 0; d_idx < true_block_size; ++d_idx) {
        FloatType tmp_dist = 0;
        bool early_stop = false;
        IdType dim_idx = 0;

        for (; dim_idx < feature_size - 3; dim_idx += 4) {
          FloatType diff0 = query_buff[threadIdx.x + block_size * (dim_idx)] -
                            data_buff[d_idx * feature_size + dim_idx];
          FloatType diff1 =
              query_buff[threadIdx.x + block_size * (dim_idx + 1)] -
              data_buff[d_idx * feature_size + dim_idx + 1];
          FloatType diff2 =
              query_buff[threadIdx.x + block_size * (dim_idx + 2)] -
              data_buff[d_idx * feature_size + dim_idx + 2];
          FloatType diff3 =
              query_buff[threadIdx.x + block_size * (dim_idx + 3)] -
              data_buff[d_idx * feature_size + dim_idx + 3];

          tmp_dist +=
              diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;

          if (tmp_dist > worst_dist) {
            early_stop = true;
            dim_idx = feature_size;
            break;
          }
        }

        for (; dim_idx < feature_size; ++dim_idx) {
          const FloatType diff =
              query_buff[threadIdx.x + dim_idx * block_size] -
              data_buff[d_idx * feature_size + dim_idx];
          tmp_dist += diff * diff;

          if (tmp_dist > worst_dist) {
            early_stop = true;
            break;
          }
        }

        if (early_stop) continue;

        HeapInsert<FloatType, IdType>(
            res_buff + threadIdx.x * k, dist_buff + threadIdx.x * k,
            d_idx + tile_start, tmp_dist, k);
        worst_dist = dist_buff[threadIdx.x * k];
      }
    }
    __syncthreads();
  }

  // copy result to global memory
  if (query_idx < query_end) {
    for (auto i = 0; i < k; ++i) {
      dists[query_idx * k + i] = dist_buff[threadIdx.x * k + i];
      data_out[query_idx * k + i] = res_buff[threadIdx.x * k + i];
      query_out[query_idx * k + i] = query_idx;
    }
  }
}

/** @brief determine the number of blocks for each segment */
template <typename IdType>
__global__ void GetNumBlockPerSegment(
    const IdType* offsets, IdType* out, const int64_t batch_size,
    const int64_t block_size) {
  const IdType idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < batch_size) {
    out[idx] = (offsets[idx + 1] - offsets[idx] - 1) / block_size + 1;
  }
}

/** @brief Get the batch index and local index in segment for each block */
template <typename IdType>
__global__ void GetBlockInfo(
    const IdType* num_block_prefixsum, IdType* block_batch_id,
    IdType* local_block_id, size_t batch_size, size_t num_blocks) {
  const IdType idx = blockIdx.x * blockDim.x + threadIdx.x;
  IdType i = 0;

  if (idx < num_blocks) {
    for (; i < batch_size; ++i) {
      if (num_block_prefixsum[i] > idx) break;
    }
    i--;
    block_batch_id[idx] = i;
    local_block_id[idx] = idx - num_block_prefixsum[i];
  }
}

/**
 * @brief Brute force kNN. Compute distance for each pair of input points and
 * get the result directly (without a distance matrix).
 *
 * @tparam FloatType The type of input points.
 * @tparam IdType The type of id.
 * @param data_points NDArray of dataset points.
 * @param data_offsets offsets of point index in data points.
 * @param query_points NDArray of query points
 * @param query_offsets offsets of point index in query points.
 * @param k the number of nearest points
 * @param result output array
 */
template <typename FloatType, typename IdType>
void BruteForceKNNCuda(
    const NDArray& data_points, const IdArray& data_offsets,
    const NDArray& query_points, const IdArray& query_offsets, const int k,
    IdArray result) {
  cudaStream_t stream = runtime::getCurrentCUDAStream();
  const auto& ctx = data_points->ctx;
  auto device = runtime::DeviceAPI::Get(ctx);
  const int64_t batch_size = data_offsets->shape[0] - 1;
  const int64_t feature_size = data_points->shape[1];
  const IdType* data_offsets_data = data_offsets.Ptr<IdType>();
  const IdType* query_offsets_data = query_offsets.Ptr<IdType>();
  const FloatType* data_points_data = data_points.Ptr<FloatType>();
  const FloatType* query_points_data = query_points.Ptr<FloatType>();
  IdType* query_out = result.Ptr<IdType>();
  IdType* data_out = query_out + k * query_points->shape[0];

  FloatType* dists = static_cast<FloatType*>(device->AllocWorkspace(
      ctx, k * query_points->shape[0] * sizeof(FloatType)));

  const int64_t block_size = cuda::FindNumThreads(query_points->shape[0]);
  const int64_t num_blocks = (query_points->shape[0] - 1) / block_size + 1;
  CUDA_KERNEL_CALL(
      BruteforceKnnKernel, num_blocks, block_size, 0, stream, data_points_data,
      data_offsets_data, query_points_data, query_offsets_data, k, dists,
      query_out, data_out, batch_size, feature_size);

  device->FreeWorkspace(ctx, dists);
}

/**
 * @brief Brute force kNN with shared memory.
 *  This function divides query points and data points into blocks. For each
 *  query block, it will make a loop over all data blocks and compute distances.
 *  It will be faster when the dimension of input points is not large.
 *
 * @tparam FloatType The type of input points.
 * @tparam IdType The type of id.
 * @param data_points NDArray of dataset points.
 * @param data_offsets offsets of point index in data points.
 * @param query_points NDArray of query points
 * @param query_offsets offsets of point index in query points.
 * @param k the number of nearest points
 * @param result output array
 */
template <typename FloatType, typename IdType>
void BruteForceKNNSharedCuda(
    const NDArray& data_points, const IdArray& data_offsets,
    const NDArray& query_points, const IdArray& query_offsets, const int k,
    IdArray result) {
  cudaStream_t stream = runtime::getCurrentCUDAStream();
  const auto& ctx = data_points->ctx;
  auto device = runtime::DeviceAPI::Get(ctx);
  const int64_t batch_size = data_offsets->shape[0] - 1;
  const int64_t feature_size = data_points->shape[1];
  const IdType* data_offsets_data = data_offsets.Ptr<IdType>();
  const IdType* query_offsets_data = query_offsets.Ptr<IdType>();
  const FloatType* data_points_data = data_points.Ptr<FloatType>();
  const FloatType* query_points_data = query_points.Ptr<FloatType>();
  IdType* query_out = result.Ptr<IdType>();
  IdType* data_out = query_out + k * query_points->shape[0];
  constexpr size_t smem_align = std::max(sizeof(IdType), sizeof(FloatType));

  // get max shared memory per block in bytes
  // determine block size according to this value
  int max_sharedmem_per_block = 0;
  CUDA_CALL(cudaDeviceGetAttribute(
      &max_sharedmem_per_block, cudaDevAttrMaxSharedMemoryPerBlock,
      ctx.device_id));
  const int64_t single_shared_mem = static_cast<int64_t>(Pow2Align<size_t>(
      (k + 2 * feature_size) * sizeof(FloatType) + k * sizeof(IdType),
      smem_align));

  const int64_t block_size =
      cuda::FindNumThreads(max_sharedmem_per_block / single_shared_mem);

  // Determine the number of blocks. We first get the number of blocks for each
  // segment. Then we get the block id offset via prefix sum.
  IdType* num_block_per_segment = static_cast<IdType*>(
      device->AllocWorkspace(ctx, batch_size * sizeof(IdType)));
  IdType* num_block_prefixsum = static_cast<IdType*>(
      device->AllocWorkspace(ctx, batch_size * sizeof(IdType)));

  // block size for GetNumBlockPerSegment computation
  int64_t temp_block_size = cuda::FindNumThreads(batch_size);
  int64_t temp_num_blocks = (batch_size - 1) / temp_block_size + 1;
  CUDA_KERNEL_CALL(
      GetNumBlockPerSegment, temp_num_blocks, temp_block_size, 0, stream,
      query_offsets_data, num_block_per_segment, batch_size, block_size);
  size_t prefix_temp_size = 0;
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
      nullptr, prefix_temp_size, num_block_per_segment, num_block_prefixsum,
      batch_size, stream));
  void* prefix_temp = device->AllocWorkspace(ctx, prefix_temp_size);
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
      prefix_temp, prefix_temp_size, num_block_per_segment, num_block_prefixsum,
      batch_size, stream));
  device->FreeWorkspace(ctx, prefix_temp);

  // wait for results
  CUDA_CALL(cudaStreamSynchronize(stream));

  int64_t num_blocks = 0, final_elem = 0,
          copyoffset = (batch_size - 1) * sizeof(IdType);
  device->CopyDataFromTo(
      num_block_prefixsum, copyoffset, &num_blocks, 0, sizeof(IdType), ctx,
      DGLContext{kDGLCPU, 0}, query_offsets->dtype);
  device->CopyDataFromTo(
      num_block_per_segment, copyoffset, &final_elem, 0, sizeof(IdType), ctx,
      DGLContext{kDGLCPU, 0}, query_offsets->dtype);
  num_blocks += final_elem;
  device->FreeWorkspace(ctx, num_block_per_segment);

  // get batch id and local id in segment
  temp_block_size = cuda::FindNumThreads(num_blocks);
  temp_num_blocks = (num_blocks - 1) / temp_block_size + 1;
  IdType* block_batch_id = static_cast<IdType*>(
      device->AllocWorkspace(ctx, num_blocks * sizeof(IdType)));
  IdType* local_block_id = static_cast<IdType*>(
      device->AllocWorkspace(ctx, num_blocks * sizeof(IdType)));
  CUDA_KERNEL_CALL(
      GetBlockInfo, temp_num_blocks, temp_block_size, 0, stream,
      num_block_prefixsum, block_batch_id, local_block_id, batch_size,
      num_blocks);

  FloatType* dists = static_cast<FloatType*>(device->AllocWorkspace(
      ctx, k * query_points->shape[0] * sizeof(FloatType)));
  CUDA_KERNEL_CALL(
      BruteforceKnnShareKernel, num_blocks, block_size,
      single_shared_mem * block_size, stream, data_points_data,
      data_offsets_data, query_points_data, query_offsets_data, block_batch_id,
      local_block_id, k, dists, query_out, data_out, batch_size, feature_size);

  device->FreeWorkspace(ctx, num_block_prefixsum);
  device->FreeWorkspace(ctx, dists);
  device->FreeWorkspace(ctx, local_block_id);
  device->FreeWorkspace(ctx, block_batch_id);
}

/** @brief Setup rng state for nn-descent */
__global__ void SetupRngKernel(
    curandState* states, const uint64_t seed, const size_t n) {
  size_t id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < n) {
    curand_init(seed, id, 0, states + id);
  }
}

/**
 * @brief Randomly initialize neighbors (sampling without replacement)
 * for each nodes
 */
template <typename FloatType, typename IdType>
__global__ void RandomInitNeighborsKernel(
    const FloatType* points, const IdType* offsets, IdType* central_nodes,
    IdType* neighbors, FloatType* dists, bool* flags, const int k,
    const int64_t feature_size, const int64_t batch_size, const uint64_t seed) {
  const IdType point_idx = blockIdx.x * blockDim.x + threadIdx.x;
  IdType batch_idx = 0;
  if (point_idx >= offsets[batch_size]) return;
  curandState state;
  curand_init(seed, point_idx, 0, &state);

  // find the segment location in the input batch
  for (IdType b = 0; b < batch_size + 1; ++b) {
    if (offsets[b] > point_idx) {
      batch_idx = b - 1;
      break;
    }
  }

  const IdType segment_size = offsets[batch_idx + 1] - offsets[batch_idx];
  IdType* current_neighbors = neighbors + point_idx * k;
  IdType* current_central_nodes = central_nodes + point_idx * k;
  bool* current_flags = flags + point_idx * k;
  FloatType* current_dists = dists + point_idx * k;
  IdType segment_start = offsets[batch_idx];

  // reservoir sampling
  for (IdType i = 0; i < k; ++i) {
    current_neighbors[i] = i + segment_start;
    current_central_nodes[i] = point_idx;
  }
  for (IdType i = k; i < segment_size; ++i) {
    const IdType j = static_cast<IdType>(curand(&state) % (i + 1));
    if (j < k) current_neighbors[j] = i + segment_start;
  }

  // compute distances and set flags
  for (IdType i = 0; i < k; ++i) {
    current_flags[i] = true;
    current_dists[i] = EuclideanDist<FloatType, IdType>(
        points + point_idx * feature_size,
        points + current_neighbors[i] * feature_size, feature_size);
  }

  // build heap
  BuildHeap<FloatType, IdType>(neighbors + point_idx * k, current_dists, k);
}

/**
 * @brief Randomly select candidates from current knn and reverse-knn graph for
 *        nn-descent.
 */
template <typename IdType>
__global__ void FindCandidatesKernel(
    const IdType* offsets, IdType* new_candidates, IdType* old_candidates,
    IdType* neighbors, bool* flags, const uint64_t seed,
    const int64_t batch_size, const int num_candidates, const int k) {
  const IdType point_idx = blockIdx.x * blockDim.x + threadIdx.x;
  IdType batch_idx = 0;
  if (point_idx >= offsets[batch_size]) return;
  curandState state;
  curand_init(seed, point_idx, 0, &state);

  // find the segment location in the input batch
  for (IdType b = 0; b < batch_size + 1; ++b) {
    if (offsets[b] > point_idx) {
      batch_idx = b - 1;
      break;
    }
  }

  IdType segment_start = offsets[batch_idx],
         segment_end = offsets[batch_idx + 1];
  IdType* current_neighbors = neighbors + point_idx * k;
  bool* current_flags = flags + point_idx * k;

  // reset candidates
  IdType* new_candidates_ptr =
      new_candidates + point_idx * (num_candidates + 1);
  IdType* old_candidates_ptr =
      old_candidates + point_idx * (num_candidates + 1);
  new_candidates_ptr[0] = 0;
  old_candidates_ptr[0] = 0;

  // select candidates from current knn graph
  // here we use candidate[0] for reservoir sampling temporarily
  for (IdType i = 0; i < k; ++i) {
    IdType candidate = current_neighbors[i];
    IdType* candidate_array =
        current_flags[i] ? new_candidates_ptr : old_candidates_ptr;
    IdType curr_num = candidate_array[0];
    IdType* candidate_data = candidate_array + 1;

    // reservoir sampling
    if (curr_num < num_candidates) {
      candidate_data[curr_num] = candidate;
    } else {
      IdType pos = static_cast<IdType>(curand(&state) % (curr_num + 1));
      if (pos < num_candidates) candidate_data[pos] = candidate;
    }
    ++candidate_array[0];
  }

  // select candidates from current reverse knn graph
  // here we use candidate[0] for reservoir sampling temporarily
  IdType index_start = segment_start * k, index_end = segment_end * k;
  for (IdType i = index_start; i < index_end; ++i) {
    if (neighbors[i] == point_idx) {
      IdType reverse_candidate = (i - index_start) / k + segment_start;
      IdType* candidate_array =
          flags[i] ? new_candidates_ptr : old_candidates_ptr;
      IdType curr_num = candidate_array[0];
      IdType* candidate_data = candidate_array + 1;

      // reservoir sampling
      if (curr_num < num_candidates) {
        candidate_data[curr_num] = reverse_candidate;
      } else {
        IdType pos = static_cast<IdType>(curand(&state) % (curr_num + 1));
        if (pos < num_candidates) candidate_data[pos] = reverse_candidate;
      }
      ++candidate_array[0];
    }
  }

  // set candidate[0] back to length
  if (new_candidates_ptr[0] > num_candidates)
    new_candidates_ptr[0] = num_candidates;
  if (old_candidates_ptr[0] > num_candidates)
    old_candidates_ptr[0] = num_candidates;

  // mark new_candidates as old
  IdType num_new_candidates = new_candidates_ptr[0];
  for (IdType i = 0; i < k; ++i) {
    IdType neighbor_idx = current_neighbors[i];

    if (current_flags[i]) {
      for (IdType j = 1; j < num_new_candidates + 1; ++j) {
        if (new_candidates_ptr[j] == neighbor_idx) {
          current_flags[i] = false;
          break;
        }
      }
    }
  }
}

/** @brief Update knn graph according to selected candidates for nn-descent */
template <typename FloatType, typename IdType>
__global__ void UpdateNeighborsKernel(
    const FloatType* points, const IdType* offsets, IdType* neighbors,
    IdType* new_candidates, IdType* old_candidates, FloatType* distances,
    bool* flags, IdType* num_updates, const int64_t batch_size,
    const int num_candidates, const int k, const int64_t feature_size) {
  const IdType point_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (point_idx >= offsets[batch_size]) return;
  IdType* current_neighbors = neighbors + point_idx * k;
  bool* current_flags = flags + point_idx * k;
  FloatType* current_dists = distances + point_idx * k;
  IdType* new_candidates_ptr =
      new_candidates + point_idx * (num_candidates + 1);
  IdType* old_candidates_ptr =
      old_candidates + point_idx * (num_candidates + 1);
  IdType num_new_candidates = new_candidates_ptr[0];
  IdType num_old_candidates = old_candidates_ptr[0];
  IdType current_num_updates = 0;

  // process new candidates
  for (IdType i = 1; i <= num_new_candidates; ++i) {
    IdType new_c = new_candidates_ptr[i];

    // new/old candidates of the current new candidate
    IdType* twohop_new_ptr = new_candidates + new_c * (num_candidates + 1);
    IdType* twohop_old_ptr = old_candidates + new_c * (num_candidates + 1);
    IdType num_twohop_new = twohop_new_ptr[0];
    IdType num_twohop_old = twohop_old_ptr[0];
    FloatType worst_dist = current_dists[0];

    // new - new
    for (IdType j = 1; j <= num_twohop_new; ++j) {
      IdType twohop_new_c = twohop_new_ptr[j];
      FloatType new_dist = EuclideanDistWithCheck<FloatType, IdType>(
          points + point_idx * feature_size,
          points + twohop_new_c * feature_size, feature_size, worst_dist);

      if (FlaggedHeapInsert<FloatType, IdType>(
              current_neighbors, current_dists, current_flags, twohop_new_c,
              new_dist, true, k, true)) {
        ++current_num_updates;
        worst_dist = current_dists[0];
      }
    }

    // new - old
    for (IdType j = 1; j <= num_twohop_old; ++j) {
      IdType twohop_old_c = twohop_old_ptr[j];
      FloatType new_dist = EuclideanDistWithCheck<FloatType, IdType>(
          points + point_idx * feature_size,
          points + twohop_old_c * feature_size, feature_size, worst_dist);

      if (FlaggedHeapInsert<FloatType, IdType>(
              current_neighbors, current_dists, current_flags, twohop_old_c,
              new_dist, true, k, true)) {
        ++current_num_updates;
        worst_dist = current_dists[0];
      }
    }
  }

  // process old candidates
  for (IdType i = 1; i <= num_old_candidates; ++i) {
    IdType old_c = old_candidates_ptr[i];

    // new candidates of the current old candidate
    IdType* twohop_new_ptr = new_candidates + old_c * (num_candidates + 1);
    IdType num_twohop_new = twohop_new_ptr[0];
    FloatType worst_dist = current_dists[0];

    // old - new
    for (IdType j = 1; j <= num_twohop_new; ++j) {
      IdType twohop_new_c = twohop_new_ptr[j];
      FloatType new_dist = EuclideanDistWithCheck<FloatType, IdType>(
          points + point_idx * feature_size,
          points + twohop_new_c * feature_size, feature_size, worst_dist);

      if (FlaggedHeapInsert<FloatType, IdType>(
              current_neighbors, current_dists, current_flags, twohop_new_c,
              new_dist, true, k, true)) {
        ++current_num_updates;
        worst_dist = current_dists[0];
      }
    }
  }

  num_updates[point_idx] = current_num_updates;
}

}  // namespace impl

template <DGLDeviceType XPU, typename FloatType, typename IdType>
void KNN(
    const NDArray& data_points, const IdArray& data_offsets,
    const NDArray& query_points, const IdArray& query_offsets, const int k,
    IdArray result, const std::string& algorithm) {
  if (algorithm == std::string("bruteforce")) {
    impl::BruteForceKNNCuda<FloatType, IdType>(
        data_points, data_offsets, query_points, query_offsets, k, result);
  } else if (algorithm == std::string("bruteforce-sharemem")) {
    impl::BruteForceKNNSharedCuda<FloatType, IdType>(
        data_points, data_offsets, query_points, query_offsets, k, result);
  } else {
    LOG(FATAL) << "Algorithm " << algorithm << " is not supported on CUDA.";
  }
}

template <DGLDeviceType XPU, typename FloatType, typename IdType>
void NNDescent(
    const NDArray& points, const IdArray& offsets, IdArray result, const int k,
    const int num_iters, const int num_candidates, const double delta) {
  cudaStream_t stream = runtime::getCurrentCUDAStream();
  const auto& ctx = points->ctx;
  auto device = runtime::DeviceAPI::Get(ctx);
  const int64_t num_nodes = points->shape[0];
  const int64_t feature_size = points->shape[1];
  const int64_t batch_size = offsets->shape[0] - 1;
  const IdType* offsets_data = offsets.Ptr<IdType>();
  const FloatType* points_data = points.Ptr<FloatType>();

  IdType* central_nodes = result.Ptr<IdType>();
  IdType* neighbors = central_nodes + k * num_nodes;
  uint64_t seed;
  int warp_size = 0;
  CUDA_CALL(
      cudaDeviceGetAttribute(&warp_size, cudaDevAttrWarpSize, ctx.device_id));
  // We don't need large block sizes, since there's not much inter-thread
  // communication
  int64_t block_size = warp_size;
  int64_t num_blocks = (num_nodes - 1) / block_size + 1;

  // allocate space for candidates, distances and flags
  // we use the first element in candidate array to represent length
  IdType* new_candidates = static_cast<IdType*>(device->AllocWorkspace(
      ctx, num_nodes * (num_candidates + 1) * sizeof(IdType)));
  IdType* old_candidates = static_cast<IdType*>(device->AllocWorkspace(
      ctx, num_nodes * (num_candidates + 1) * sizeof(IdType)));
  IdType* num_updates = static_cast<IdType*>(
      device->AllocWorkspace(ctx, num_nodes * sizeof(IdType)));
  FloatType* distances = static_cast<FloatType*>(
      device->AllocWorkspace(ctx, num_nodes * k * sizeof(IdType)));
  bool* flags = static_cast<bool*>(
      device->AllocWorkspace(ctx, num_nodes * k * sizeof(IdType)));

  size_t sum_temp_size = 0;
  IdType total_num_updates = 0;
  IdType* total_num_updates_d =
      static_cast<IdType*>(device->AllocWorkspace(ctx, sizeof(IdType)));

  CUDA_CALL(cub::DeviceReduce::Sum(
      nullptr, sum_temp_size, num_updates, total_num_updates_d, num_nodes,
      stream));
  IdType* sum_temp_storage =
      static_cast<IdType*>(device->AllocWorkspace(ctx, sum_temp_size));

  // random initialize neighbors
  seed = RandomEngine::ThreadLocal()->RandInt<uint64_t>(
      std::numeric_limits<uint64_t>::max());
  CUDA_KERNEL_CALL(
      impl::RandomInitNeighborsKernel, num_blocks, block_size, 0, stream,
      points_data, offsets_data, central_nodes, neighbors, distances, flags, k,
      feature_size, batch_size, seed);

  for (int i = 0; i < num_iters; ++i) {
    // select candidates
    seed = RandomEngine::ThreadLocal()->RandInt<uint64_t>(
        std::numeric_limits<uint64_t>::max());
    CUDA_KERNEL_CALL(
        impl::FindCandidatesKernel, num_blocks, block_size, 0, stream,
        offsets_data, new_candidates, old_candidates, neighbors, flags, seed,
        batch_size, num_candidates, k);

    // update
    CUDA_KERNEL_CALL(
        impl::UpdateNeighborsKernel, num_blocks, block_size, 0, stream,
        points_data, offsets_data, neighbors, new_candidates, old_candidates,
        distances, flags, num_updates, batch_size, num_candidates, k,
        feature_size);

    total_num_updates = 0;
    CUDA_CALL(cub::DeviceReduce::Sum(
        sum_temp_storage, sum_temp_size, num_updates, total_num_updates_d,
        num_nodes, stream));
    device->CopyDataFromTo(
        total_num_updates_d, 0, &total_num_updates, 0, sizeof(IdType), ctx,
        DGLContext{kDGLCPU, 0}, offsets->dtype);

    if (total_num_updates <= static_cast<IdType>(delta * k * num_nodes)) {
      break;
    }
  }

  device->FreeWorkspace(ctx, new_candidates);
  device->FreeWorkspace(ctx, old_candidates);
  device->FreeWorkspace(ctx, num_updates);
  device->FreeWorkspace(ctx, distances);
  device->FreeWorkspace(ctx, flags);
  device->FreeWorkspace(ctx, total_num_updates_d);
  device->FreeWorkspace(ctx, sum_temp_storage);
}

template void KNN<kDGLCUDA, float, int32_t>(
    const NDArray& data_points, const IdArray& data_offsets,
    const NDArray& query_points, const IdArray& query_offsets, const int k,
    IdArray result, const std::string& algorithm);
template void KNN<kDGLCUDA, float, int64_t>(
    const NDArray& data_points, const IdArray& data_offsets,
    const NDArray& query_points, const IdArray& query_offsets, const int k,
    IdArray result, const std::string& algorithm);
template void KNN<kDGLCUDA, double, int32_t>(
    const NDArray& data_points, const IdArray& data_offsets,
    const NDArray& query_points, const IdArray& query_offsets, const int k,
    IdArray result, const std::string& algorithm);
template void KNN<kDGLCUDA, double, int64_t>(
    const NDArray& data_points, const IdArray& data_offsets,
    const NDArray& query_points, const IdArray& query_offsets, const int k,
    IdArray result, const std::string& algorithm);

template void NNDescent<kDGLCUDA, float, int32_t>(
    const NDArray& points, const IdArray& offsets, IdArray result, const int k,
    const int num_iters, const int num_candidates, const double delta);
template void NNDescent<kDGLCUDA, float, int64_t>(
    const NDArray& points, const IdArray& offsets, IdArray result, const int k,
    const int num_iters, const int num_candidates, const double delta);
template void NNDescent<kDGLCUDA, double, int32_t>(
    const NDArray& points, const IdArray& offsets, IdArray result, const int k,
    const int num_iters, const int num_candidates, const double delta);
template void NNDescent<kDGLCUDA, double, int64_t>(
    const NDArray& points, const IdArray& offsets, IdArray result, const int k,
    const int num_iters, const int num_candidates, const double delta);

}  // namespace transform
}  // namespace dgl
