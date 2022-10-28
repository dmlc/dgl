/*!
 *   Copyright (c) 2022, NVIDIA Corporation
 *   Copyright (c) 2022, GT-TDAlab (Muhammed Fatih Balin & Umit V. Catalyurek)  
 *   All rights reserved.
 *
 *   Licensed under the Apache License, Version 2.0 (the "License");
 *   you may not use this file except in compliance with the License.
 *   You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *   Unless required by applicable law or agreed to in writing, software
 *   distributed under the License is distributed on an "AS IS" BASIS,
 *   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *   See the License for the specific language governing permissions and
 *   limitations under the License.
 *
 * \file array/cuda/labor_sampling.cu
 * \brief labor sampling
 */

#include <dgl/random.h>
#include <dgl/runtime/device_api.h>
#include <dgl/aten/coo.h>

#include <thrust/execution_policy.h>
#include <thrust/shuffle.h>
#include <thrust/binary_search.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/zip_function.h>
#include <thrust/gather.h>

#include <curand_kernel.h>

#include <numeric>
#include <utility>
#include <type_traits>
#include <algorithm>
#include <limits>

#include "./dgl_cub.cuh"
#include "./functor.cuh"
#include "./spmm.cuh"

#include "../../array/cuda/atomic.cuh"
#include "../../runtime/cuda/cuda_common.h"
#include "../../array/cuda/utils.h"
#include "../../graph/transform/cuda/cuda_map_edges.cuh"

namespace dgl {
namespace aten {
namespace impl {

constexpr int BLOCK_SIZE = 128;
constexpr int CTA_SIZE = 128;
constexpr double eps = 0.0001;
constexpr bool labor = true;

namespace {

template <typename IdType>
struct TransformOp {
  const IdType * idx_coo;
  const IdType * rows;
  const IdType * indptr;
  const IdType * subindptr;
  const IdType * indices;
  const IdType * data_arr;
  __host__ __device__
  auto operator() (IdType idx) {
    const auto in_row = idx_coo[idx];
    const auto row = rows[in_row];
    const auto in_idx = indptr[row] + idx - subindptr[in_row];
    const auto u = indices[in_idx];
    const auto data = data_arr ? data_arr[in_idx] : in_idx;
    return thrust::make_tuple(row, u, data);
  }
};

template <typename IdType, typename FloatType, typename probs_t, typename A_t, typename B_t>
struct TransformOpImp {
  probs_t probs;
  A_t A;
  B_t B;
  const IdType * idx_coo;
  const IdType * rows;
  const FloatType * cs;
  const IdType * indptr;
  const IdType * subindptr;
  const IdType * indices;
  const IdType * data_arr;
  __host__ __device__
  auto operator() (IdType idx) {
    const auto ps = probs[idx];
    const auto in_row = idx_coo[idx];
    const auto c = cs[in_row];
    const auto row = rows[in_row];
    const auto in_idx = indptr[row] + idx - subindptr[in_row];
    const auto u = indices[in_idx];
    const auto w = A[in_idx];
    const auto w2 = B[in_idx];
    const auto data = data_arr ? data_arr[in_idx] : in_idx;
    return thrust::make_tuple(in_row, row, u, data, w / min((FloatType)1, c * w2 * ps));
  }
};

template <typename FloatType>
struct StencilOp {
  const FloatType * cs;
  template <typename IdType>
  __host__ __device__
  auto operator() (IdType in_row, FloatType ps, FloatType rnd) {
    return rnd <= cs[in_row] * ps;
  }
};

template <typename IdType, typename FloatType, typename ps_t, typename A_t>
struct StencilOpFused {
  const uint64_t rand_seed;
  const IdType * idx_coo;
  const FloatType * cs;
  const ps_t probs;
  const A_t A;
  const IdType * subindptr;
  const IdType * rows;
  const IdType * indptr;
  const IdType * indices;
  __device__
  auto operator() (IdType idx) {
    const auto in_row = idx_coo[idx];
    const auto ps = probs[idx];
    IdType rofs = idx - subindptr[in_row];
    const IdType row = rows[in_row];
    const auto in_idx = indptr[row] + rofs;
    const auto u = indices[in_idx];
    curandStatePhilox4_32_10_t rng;
    if (labor)
      curand_init(123123, rand_seed, u, &rng);
    else
      curand_init(rand_seed, idx, 0, &rng);
    float rnd;
    rnd = curand_uniform(&rng);
    return rnd <= cs[in_row] * A[in_idx] * ps;
  }
};

template <typename IdType, typename FloatType>
struct TransformOpMean {
  const IdType * ds;
  const FloatType * ws;
  __host__ __device__
  auto operator() (IdType idx, FloatType ps) {
    return ps * ds[idx] / ws[idx];
  }
};

struct TransformOpMinWith1 {
  template <typename FloatType>
  __host__ __device__
  auto operator() (FloatType x) {
    return min((FloatType)1, x);
  }
};

template <typename IdType>
struct IndptrFunc {
  const IdType * indptr;
  __host__ __device__
  auto operator() (IdType row) {
    return indptr[row];
  }
};

template <typename FloatType>
struct SquareFunc {
  __host__ __device__
  auto operator() (FloatType x) {
    return thrust::make_tuple(x, x * x);
  }
};

struct TupleSum {
    template <typename T>
    __host__ __device__
    T operator()(const T &a, const T &b) const {
        return thrust::make_tuple(thrust::get<0>(a) + thrust::get<0>(b),
            thrust::get<1>(a) + thrust::get<1>(b));
    }
};

template <typename IdType, typename FloatType>
struct DegreeFunc {
  const IdType num_picks;
  const IdType * rows;
  const IdType * indptr;
  const FloatType * ds;
  IdType * in_deg;
  FloatType * cs;
  __host__ __device__
  auto operator() (IdType tIdx) {
    const auto out_row = rows[tIdx];
    const auto d = indptr[out_row + 1] - indptr[out_row];
    in_deg[tIdx] = d;
    cs[tIdx] = num_picks / (ds ? ds[tIdx] : (FloatType)d);
  }
};

template <typename IdType, typename FloatType>
__global__ void _CSRRowWiseOneHopExtractorKernel(
  const uint64_t rand_seed,
  const IdType hop_size,
  const IdType * const rows,
  const IdType * const indptr,
  const IdType * const subindptr,
  const IdType * const indices,
  const IdType * const idx_coo,
  const FloatType * const A,
  FloatType * const rands,
  IdType * const hop,
  FloatType * const A_l) {
  IdType tx = static_cast<IdType>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int stride_x = gridDim.x * blockDim.x;

  curandStatePhilox4_32_10_t rng;
  if (!labor)
    curand_init(rand_seed, tx, 0, &rng);

  while (tx < hop_size) {
    IdType rpos = idx_coo[tx];
    IdType rofs = tx - subindptr[rpos];
    const IdType row = rows[rpos];
    const auto in_idx = indptr[row] + rofs;
    const auto u = indices[in_idx];
    hop[tx] = u;
    if (labor)
      curand_init(123123, rand_seed, u, &rng);
    float rnd;
    rnd = curand_uniform(&rng);
    if (A)
      A_l[tx] = A[in_idx];
    rands[tx] = (FloatType)rnd;
    tx += stride_x;
  }
}

template <typename IdType, typename FloatType, int BLOCK_CTAS, int TILE_SIZE>
__global__ void _CSRRowWiseLayerSampleDegreeKernel(
  const IdType num_picks,
  const IdType num_rows,
  const IdType * const rows,
  FloatType * const cs,
  const FloatType * const ds,
  const FloatType * const d2s,
  const IdType * const indptr,
  const FloatType * const probs,
  const FloatType * const A,
  const IdType * const subindptr) {
  typedef cub::BlockReduce<FloatType, BLOCK_SIZE> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ FloatType var_1_bcast[BLOCK_CTAS];

  // we assign one warp per row
  assert(blockDim.x == CTA_SIZE);
  assert(blockDim.y == BLOCK_CTAS);

  IdType out_row = blockIdx.x * TILE_SIZE + threadIdx.y;
  const auto last_row = min(static_cast<IdType>(blockIdx.x + 1) * TILE_SIZE, num_rows);

  constexpr FloatType ONE = 1;

  while (out_row < last_row) {
    const auto row = rows[out_row];

    const auto in_row_start = indptr[row];
    const auto out_row_start = subindptr[out_row];

    const IdType degree = indptr[row + 1] - in_row_start;

    if (degree > 0) {
      // stands for k in in arXiv:2210.13339, i.e. fanout
      const auto k = min(num_picks, degree);
      // slightly better than NS
      const FloatType d_ = ds ? ds[row] : degree;
      // stands for right handside of Equation (22) in arXiv:2210.13339
      FloatType var_target = d_ * d_ / k + (ds ? d2s[row] - d_ * d_ / degree : 0);

      auto c = cs[out_row];
      const int num_valid = min(degree, (IdType)CTA_SIZE);
      // stands for left handside of Equation (22) in arXiv:2210.13339
      FloatType var_1;
      do {
        var_1 = 0;
        if (A) {
          for (int idx = threadIdx.x; idx < degree; idx += CTA_SIZE) {
            const auto w = A[in_row_start + idx];
            const auto ps = probs ? probs[out_row_start + idx] : w;
            var_1 += w * w / min(ONE, c * ps);
          }
        } else {
          for (int idx = threadIdx.x; idx < degree; idx += CTA_SIZE) {
            const auto ps = probs[out_row_start + idx];
            var_1 += 1 / min(ONE, c * ps);
          }
        }
        var_1 = BlockReduce(temp_storage).Sum(var_1, num_valid);
        if (threadIdx.x == 0)
          var_1_bcast[threadIdx.y] = var_1;
        __syncthreads();
        var_1 = var_1_bcast[threadIdx.y];

        c *= var_1 / var_target;
      } while (min(var_1, var_target) / max(var_1, var_target) < 1 - eps);

      if (threadIdx.x == 0)
        cs[out_row] = c;
    }

    out_row += BLOCK_CTAS;
  }
}

template<typename IdType, int BLOCK_SIZE, IdType TILE_SIZE>
__global__ void map_vertex_ids_global(
  const IdType * const global,
  IdType * const new_global,
  const IdType num_vertices,
  const runtime::cuda::DeviceOrderedHashTable<IdType> table) {
  assert(BLOCK_SIZE == blockDim.x);

  using Mapping = typename OrderedHashTable<IdType>::Mapping;

  const IdType tile_start = TILE_SIZE * blockIdx.x;
  const IdType tile_end = min(TILE_SIZE * (blockIdx.x + 1), num_vertices);

  for (IdType idx = threadIdx.x + tile_start; idx < tile_end; idx += BLOCK_SIZE) {
    const Mapping& mapping = *table.Search(global[idx]);
    new_global[idx] = mapping.local;
  }
}

}  // namespace

/////////////////////////////// CSR ///////////////////////////////

template <DGLDeviceType XPU, typename IdType, typename FloatType>
std::pair<COOMatrix, FloatArray> CSRLaborSampling(CSRMatrix mat,
                  IdArray rows_arr,
                  const int64_t num_picks,
                  FloatArray prob_arr,
                  int importance_sampling) {
  const bool weights = !IsNullArray(prob_arr);

  const uint64_t max_log_num_vertices = [&]() -> int {
    for (int i = 0; i < static_cast<int>(sizeof(IdType)) * 8; i++)
      if (mat.num_cols <= ((IdType)1) << i)
        return i;
    return sizeof(IdType) * 8;
  }();

  const auto& ctx = rows_arr->ctx;

  runtime::workspace_memory_alloc<decltype(&ctx)> allocator(&ctx);

  const auto stream = runtime::getCurrentCUDAStream();
  const auto exec_policy = thrust::cuda::par_nosync(allocator).on(stream);

  auto device = runtime::DeviceAPI::Get(ctx);

  const IdType num_rows = rows_arr->shape[0];
  IdType * rows = static_cast<IdType*>(rows_arr->data);
  FloatType * A = static_cast<FloatType*>(prob_arr->data);

  const IdType * const indptr = static_cast<const IdType*>(mat.indptr->data);
  const IdType * const indices = static_cast<const IdType*>(mat.indices->data);
  const IdType * const data = CSRHasData(mat) ?
    static_cast<IdType*>(mat.data->data) : nullptr;

  // compute in-degrees
  auto in_deg = allocator.alloc_unique<IdType>(num_rows + 1);
  // cs stands for c_s in arXiv:2210.13339
  auto cs_arr = NewFloatArray(num_rows, ctx, sizeof(FloatType) * 8);
  auto cs = static_cast<FloatType*>(cs_arr->data);
  // ds stands for A_{*s} in arXiv:2210.13339
  auto ds_arr = weights ? NewFloatArray(num_rows, ctx, sizeof(FloatType) * 8) : NullArray();
  auto ds = static_cast<FloatType*>(ds_arr->data);
  // d2s stands for (A^2)_{*s} in arXiv:2210.13339, ^2 is elementwise.
  auto d2s_arr = weights ? NewFloatArray(num_rows, ctx, sizeof(FloatType) * 8) : NullArray();
  auto d2s = static_cast<FloatType*>(d2s_arr->data);

  if (weights) {
    auto b_offsets = thrust::make_transform_iterator(rows, IndptrFunc<IdType>{indptr});
    auto e_offsets = thrust::make_transform_iterator(rows, IndptrFunc<IdType>{indptr + 1});

    auto A_A2 = thrust::make_transform_iterator(A, SquareFunc<FloatType>{});
    auto ds_d2s = thrust::make_zip_iterator(ds, d2s);

    size_t prefix_temp_size = 0;
    CUDA_CALL(cub::DeviceSegmentedReduce::Reduce(
      nullptr, prefix_temp_size,
      A_A2,
      ds_d2s,
      num_rows,
      b_offsets,
      e_offsets,
      TupleSum{},
      thrust::make_tuple((FloatType)0, (FloatType)0),
      stream));
    auto temp = allocator.alloc_unique<char>(prefix_temp_size);
    CUDA_CALL(cub::DeviceSegmentedReduce::Reduce(
      temp.get(), prefix_temp_size,
      A_A2,
      ds_d2s,
      num_rows,
      b_offsets,
      e_offsets,
      TupleSum{},
      thrust::make_tuple((FloatType)0, (FloatType)0),
      stream));
  }

  thrust::counting_iterator<IdType> iota(0);
  thrust::for_each(exec_policy, iota, iota + num_rows, DegreeFunc<IdType, FloatType>{
    (IdType)num_picks, rows, indptr, weights ? ds : nullptr, in_deg.get(), cs});

  // fill subindptr
  auto subindptr_arr = NewIdArray(num_rows + 1, ctx, sizeof(IdType) * 8);
  auto subindptr = static_cast<IdType*>(subindptr_arr->data);

  IdType hop_size;
  {
    size_t prefix_temp_size = 0;
    CUDA_CALL(cub::DeviceScan::ExclusiveSum(nullptr, prefix_temp_size,
      in_deg.get(),
      subindptr,
      num_rows + 1,
      stream));
    auto temp = allocator.alloc_unique<char>(prefix_temp_size);
    CUDA_CALL(cub::DeviceScan::ExclusiveSum(temp.get(), prefix_temp_size,
      in_deg.get(),
      subindptr,
      num_rows + 1,
      stream));

    device->CopyDataFromTo(subindptr, num_rows * sizeof(hop_size), &hop_size, 0,
      sizeof(hop_size),
      ctx,
      DGLContext{kDGLCPU, 0},
      mat.indptr->dtype);
  }
  auto A_l_arr = weights ? NewFloatArray(hop_size, ctx, sizeof(FloatType) * 8) : NullArray();
  auto A_l = static_cast<FloatType*>(A_l_arr->data);
  auto hop_arr = NewIdArray(hop_size, ctx, sizeof(IdType) * 8);
  CSRMatrix smat(num_rows, mat.num_cols, subindptr_arr, hop_arr, NullArray(), mat.sorted);
  // Consider fusing CSRToCOO into StencilOpFused kernel
  auto smatcoo = CSRToCOO(smat, false);

  auto idx_coo_arr = smatcoo.row;
  auto idx_coo = static_cast<IdType*>(idx_coo_arr->data);

  auto hop_1 = static_cast<IdType*>(hop_arr->data);
  auto rands = allocator.alloc_unique<FloatType>(importance_sampling ? hop_size : 1);
  auto probs_found = allocator.alloc_unique<FloatType>(importance_sampling ? hop_size : 1);

  if (weights) {
    constexpr int BLOCK_CTAS = BLOCK_SIZE / CTA_SIZE;
    // the number of rows each thread block will cover
    constexpr int TILE_SIZE = BLOCK_CTAS;
    const dim3 block(CTA_SIZE, BLOCK_CTAS);
    const dim3 grid((num_rows + TILE_SIZE - 1) / TILE_SIZE);
    CUDA_KERNEL_CALL(
      (_CSRRowWiseLayerSampleDegreeKernel<IdType, FloatType, BLOCK_CTAS, TILE_SIZE>),
      grid, block, 0, stream,
      (IdType)num_picks,
      num_rows,
      rows,
      cs,
      ds,
      d2s,
      indptr,
      nullptr,
      A,
      subindptr);
  }

  const uint64_t random_seed = RandomEngine::ThreadLocal()->RandInt(1000000000);

  if (importance_sampling) {
    { // extracts the onehop neighborhood cols to a contiguous range into hop_1
      const dim3 block(BLOCK_SIZE);
      const dim3 grid((hop_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
      CUDA_KERNEL_CALL((_CSRRowWiseOneHopExtractorKernel<IdType, FloatType>),
        grid, block, 0, stream,
        random_seed, hop_size, rows,
        indptr, subindptr, indices, idx_coo, weights ? A : nullptr, rands.get(), hop_1, A_l);
    }
    int64_t hop_uniq_size = 0;
    auto hop_new_arr = NewIdArray(hop_size, ctx, sizeof(IdType) * 8);
    auto hop_new = static_cast<IdType*>(hop_new_arr->data);
    auto hop_unique = allocator.alloc_unique<IdType>(hop_size);
    // After this block, hop_unique holds the unique set of one-hop neighborhood
    // and hop_new holds the renamed hop_1, idx_coo already holds renamed destination.
    {
      auto hop_2 = allocator.alloc_unique<IdType>(hop_size);
      auto hop_3 = allocator.alloc_unique<IdType>(hop_size);

      device->CopyDataFromTo(hop_1, 0, hop_2.get(), 0,
          sizeof(IdType) * hop_size,
          ctx,
          ctx,
          mat.indptr->dtype);

      cub::DoubleBuffer<IdType> hop_b(hop_2.get(), hop_3.get());

      {
        std::size_t temp_storage_bytes = 0;
        CUDA_CALL(cub::DeviceRadixSort::SortKeys(
            nullptr, temp_storage_bytes, hop_b, hop_size, 0, max_log_num_vertices, stream));

        auto temp = allocator.alloc_unique<char>(temp_storage_bytes);

        CUDA_CALL(cub::DeviceRadixSort::SortKeys(
            temp.get(), temp_storage_bytes, hop_b, hop_size, 0, max_log_num_vertices, stream));
      }

      auto hop_counts = allocator.alloc_unique<IdType>(hop_size + 1);
      auto hop_unique_size = allocator.alloc_unique<int64_t>(1);

      {
        std::size_t temp_storage_bytes = 0;
        CUDA_CALL(cub::DeviceRunLengthEncode::Encode(
          nullptr, temp_storage_bytes, hop_b.Current(),
          hop_unique.get(), hop_counts.get(), hop_unique_size.get(), hop_size, stream));

        auto temp = allocator.alloc_unique<char>(temp_storage_bytes);

        CUDA_CALL(cub::DeviceRunLengthEncode::Encode(
            temp.get(), temp_storage_bytes, hop_b.Current(),
            hop_unique.get(), hop_counts.get(), hop_unique_size.get(), hop_size, stream));

        device->CopyDataFromTo(hop_unique_size.get(), 0, &hop_uniq_size, 0,
          sizeof(hop_uniq_size),
          ctx,
          DGLContext{kDGLCPU, 0},
          mat.indptr->dtype);
      }

      thrust::lower_bound(exec_policy,
          hop_unique.get(), hop_unique.get() + hop_uniq_size, hop_1, hop_1 + hop_size, hop_new);
    }

    // Consider creating a CSC because the SpMV will be done multiple times.
    COOMatrix rmat(
        num_rows, hop_uniq_size, idx_coo_arr, hop_new_arr, NullArray(), true, mat.sorted);

    BcastOff bcast_off;
    bcast_off.use_bcast = false;
    bcast_off.out_len = 1;
    bcast_off.lhs_len = 1;
    bcast_off.rhs_len = 1;

    auto probs_arr = NewFloatArray(hop_uniq_size, ctx, sizeof(FloatType) * 8);
    auto probs_1 = static_cast<FloatType*>(probs_arr->data);
    auto probs_arr_2 = NewFloatArray(hop_uniq_size, ctx, sizeof(FloatType) * 8);
    auto probs = static_cast<FloatType*>(probs_arr_2->data);
    auto arg_u = NewIdArray(hop_uniq_size, ctx, sizeof(IdType) * 8);
    auto arg_e = NewIdArray(hop_size, ctx, sizeof(IdType) * 8);

    double prev_ex_nodes = hop_uniq_size;

    for (int iters = 0; iters < importance_sampling || importance_sampling < 0; iters++) {
      if (weights && iters == 0) {
        cuda::SpMMCoo<IdType, FloatType,
          cuda::binary::Mul<FloatType>, cuda::reduce::Max<IdType, FloatType, true>>(
          bcast_off, rmat, cs_arr, A_l_arr, probs_arr_2, arg_u, arg_e);
      } else {
        cuda::SpMMCoo<IdType, FloatType,
          cuda::binary::CopyLhs<FloatType>, cuda::reduce::Max<IdType, FloatType, true>>(
          bcast_off, rmat, cs_arr, NullArray(), iters ? probs_arr : probs_arr_2, arg_u, arg_e);
      }

      if (iters)
        thrust::transform(exec_policy,
            probs_1, probs_1 + hop_uniq_size, probs, probs, thrust::multiplies<FloatType>{});

      thrust::gather(exec_policy, hop_new, hop_new + hop_size, probs, probs_found.get());

      {
        constexpr int BLOCK_CTAS = BLOCK_SIZE / CTA_SIZE;
        // the number of rows each thread block will cover
        constexpr int TILE_SIZE = BLOCK_CTAS;
        const dim3 block(CTA_SIZE, BLOCK_CTAS);
        const dim3 grid((num_rows + TILE_SIZE - 1) / TILE_SIZE);
        CUDA_KERNEL_CALL(
          (_CSRRowWiseLayerSampleDegreeKernel<IdType, FloatType, BLOCK_CTAS, TILE_SIZE>),
          grid, block, 0, stream,
          (IdType)num_picks,
          num_rows,
          rows,
          cs,
          weights ? ds : nullptr,
          weights ? d2s : nullptr,
          indptr,
          probs_found.get(),
          A,
          subindptr);
      }

      {
        auto probs_min_1 = thrust::make_transform_iterator(probs, TransformOpMinWith1{});
        double cur_ex_nodes = thrust::reduce(exec_policy,
            probs_min_1, probs_min_1 + hop_uniq_size, 0.0);
        // std::cerr << iters << ' ' << cur_ex_nodes << '\n';
        if (cur_ex_nodes / prev_ex_nodes >= 1 - eps)
          break;
        prev_ex_nodes = cur_ex_nodes;
      }
    }
  }

  IdArray picked_row = NewIdArray(hop_size, ctx, sizeof(IdType) * 8);
  IdArray picked_col = NewIdArray(hop_size, ctx, sizeof(IdType) * 8);
  IdArray picked_idx = NewIdArray(hop_size, ctx, sizeof(IdType) * 8);
  FloatArray picked_imp = importance_sampling || weights ?
      NewFloatArray(hop_size, ctx, sizeof(FloatType) * 8) : NullArray();

  IdType* const picked_row_data = static_cast<IdType*>(picked_row->data);
  IdType* const picked_col_data = static_cast<IdType*>(picked_col->data);
  IdType* const picked_idx_data = static_cast<IdType*>(picked_idx->data);
  FloatType* const picked_imp_data = static_cast<FloatType*>(picked_imp->data);

  auto picked_inrow = allocator.alloc_unique<IdType>(importance_sampling ? hop_size : 1);

  IdType num_edges;
  {
    thrust::constant_iterator<FloatType> one(1);
    if (importance_sampling) {
      auto output = thrust::make_zip_iterator(
          picked_inrow.get(), picked_row_data, picked_col_data, picked_idx_data, picked_imp_data);
      if (weights) {
        auto transformed_output = thrust::make_transform_output_iterator(output,
            TransformOpImp<IdType, FloatType, FloatType *, FloatType *, decltype(one)>{
            probs_found.get(), A, one, idx_coo, rows, cs, indptr, subindptr, indices, data});
        auto stencil = thrust::make_zip_iterator(idx_coo, probs_found.get(), rands.get());
        num_edges = thrust::copy_if(exec_policy,
            iota, iota + hop_size, stencil, transformed_output,
            thrust::make_zip_function(StencilOp<FloatType>{cs})) - transformed_output;
      } else {
        auto transformed_output = thrust::make_transform_output_iterator(output,
            TransformOpImp<IdType, FloatType, FloatType *, decltype(one), decltype(one)>{
            probs_found.get(), one, one, idx_coo, rows, cs, indptr, subindptr, indices, data});
        auto stencil = thrust::make_zip_iterator(idx_coo, probs_found.get(), rands.get());
        num_edges = thrust::copy_if(exec_policy,
            iota, iota + hop_size, stencil, transformed_output,
            thrust::make_zip_function(StencilOp<FloatType>{cs})) - transformed_output;
      }
    } else {
      if (weights) {
        auto output = thrust::make_zip_iterator(
            picked_inrow.get(), picked_row_data, picked_col_data, picked_idx_data, picked_imp_data);
        auto transformed_output = thrust::make_transform_output_iterator(output,
            TransformOpImp<IdType, FloatType, decltype(one), FloatType *, FloatType *>{
            one, A, A, idx_coo, rows, cs, indptr, subindptr, indices, data});
        const auto pred = StencilOpFused<IdType, FloatType, decltype(one), FloatType *>{
            random_seed, idx_coo, cs,
            one, A, subindptr, rows, indptr, indices};
        num_edges = thrust::copy_if(exec_policy,
            iota, iota + hop_size, iota, transformed_output, pred) - transformed_output;
      } else {
        auto output = thrust::make_zip_iterator(picked_row_data, picked_col_data, picked_idx_data);
        auto transformed_output = thrust::make_transform_output_iterator(output,
            TransformOp<IdType>{idx_coo, rows, indptr, subindptr, indices, data});
        const auto pred = StencilOpFused<IdType, FloatType, decltype(one), decltype(one)>{
            random_seed, idx_coo, cs,
            one, one, subindptr, rows, indptr, indices};
        num_edges = thrust::copy_if(exec_policy,
            iota, iota + hop_size, iota, transformed_output, pred) - transformed_output;
      }
    }
  }

  if (importance_sampling || weights) {
    thrust::constant_iterator<IdType> one(1);
    auto ds = allocator.alloc_unique<IdType>(num_rows);
    auto ws = allocator.alloc_unique<FloatType>(num_rows);
    auto ds_2 = allocator.alloc_unique<IdType>(num_rows);
    auto ws_2 = allocator.alloc_unique<FloatType>(num_rows);
    auto output_ = thrust::make_zip_iterator(ds.get(), ws.get());
    auto keys = allocator.alloc_unique<IdType>(num_rows);
    auto new_end = thrust::reduce_by_key(exec_policy,
        picked_inrow.get(), picked_inrow.get() + num_edges, one, keys.get(), ds.get());
    thrust::reduce_by_key(exec_policy,
        picked_inrow.get(), picked_inrow.get() + num_edges, picked_imp_data, keys.get(), ws.get());
    {
      thrust::constant_iterator<IdType> zero_int(0);
      thrust::constant_iterator<FloatType> zero_float(0);
      auto input = thrust::make_zip_iterator(zero_int, zero_float);
      auto output = thrust::make_zip_iterator(ds_2.get(), ws_2.get());
      thrust::copy(exec_policy, input, input + num_rows, output);
      {
        const auto num_rows_2 = new_end.first - keys.get();
        thrust::scatter(exec_policy, output_, output_ + num_rows_2, keys.get(), output);
      }
    }
    {
      auto input = thrust::make_zip_iterator(picked_inrow.get(), picked_imp_data);
      auto transformed_input = thrust::make_transform_iterator(input,
          thrust::make_zip_function(TransformOpMean<IdType, FloatType>{ds_2.get(), ws_2.get()}));
      thrust::copy(exec_policy, transformed_input, transformed_input + num_edges, picked_imp_data);
    }
  }

  picked_row = picked_row.CreateView({num_edges}, picked_row->dtype);
  picked_col = picked_col.CreateView({num_edges}, picked_col->dtype);
  picked_idx = picked_idx.CreateView({num_edges}, picked_idx->dtype);
  if (importance_sampling || weights)
    picked_imp = picked_imp.CreateView({num_edges}, picked_imp->dtype);

  return std::make_pair(COOMatrix(mat.num_rows, mat.num_cols, picked_row,
      picked_col, picked_idx), picked_imp);
}

template std::pair<COOMatrix, FloatArray> CSRLaborSampling<kDGLCUDA, int32_t, float>(
  CSRMatrix, IdArray, int64_t, FloatArray, int);
template std::pair<COOMatrix, FloatArray> CSRLaborSampling<kDGLCUDA, int64_t, float>(
  CSRMatrix, IdArray, int64_t, FloatArray, int);
template std::pair<COOMatrix, FloatArray> CSRLaborSampling<kDGLCUDA, int32_t, double>(
  CSRMatrix, IdArray, int64_t, FloatArray, int);
template std::pair<COOMatrix, FloatArray> CSRLaborSampling<kDGLCUDA, int64_t, double>(
  CSRMatrix, IdArray, int64_t, FloatArray, int);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
