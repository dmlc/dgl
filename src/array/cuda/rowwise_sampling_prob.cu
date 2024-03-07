/**
 *  Copyright (c) 2022 by Contributors
 * @file array/cuda/rowwise_sampling_prob.cu
 * @brief weighted rowwise sampling. The degree computing kernels and
 * host-side functions are partially borrowed from the uniform rowwise
 * sampling code rowwise_sampling.cu.
 * @author pengqirong (OPPO), dlasalle and Xin from Nvidia.
 */
#include <curand_kernel.h>
#include <dgl/random.h>
#include <dgl/runtime/device_api.h>

#include <cub/cub.cuh>
#include <numeric>

#include "../../array/cuda/atomic.cuh"
#include "../../runtime/cuda/cuda_common.h"
#include "./utils.h"

// require CUB 1.17 to use DeviceSegmentedSort
static_assert(
    CUB_VERSION >= 101700, "Require CUB >= 1.17 to use DeviceSegmentedSort");

namespace dgl {
using namespace cuda;
using namespace aten::cuda;
namespace aten {
namespace impl {

namespace {

constexpr int BLOCK_SIZE = 128;

/**
 * @brief Compute the size of each row in the sampled CSR, without replacement.
 * temp_deg is calculated for rows with deg > num_picks.
 * For these rows, we will calculate their A-Res values and sort them to get
 * top-num_picks.
 *
 * @tparam IdType The type of node and edge indexes.
 * @param num_picks The number of non-zero entries to pick per row.
 * @param num_rows The number of rows to pick.
 * @param in_rows The set of rows to pick.
 * @param in_ptr The index where each row's edges start.
 * @param out_deg The size of each row in the sampled matrix, as indexed by
 * `in_rows` (output).
 * @param temp_deg The size of each row in the input matrix, as indexed by
 * `in_rows` (output).
 */
template <typename IdType>
__global__ void _CSRRowWiseSampleDegreeKernel(
    const int64_t num_picks, const int64_t num_rows,
    const IdType* const in_rows, const IdType* const in_ptr,
    IdType* const out_deg, IdType* const temp_deg) {
  const int64_t tIdx = threadIdx.x + blockIdx.x * blockDim.x;

  if (tIdx < num_rows) {
    const int64_t in_row = in_rows[tIdx];
    const int64_t out_row = tIdx;
    const IdType deg = in_ptr[in_row + 1] - in_ptr[in_row];
    // temp_deg is used to generate ares_ptr
    temp_deg[out_row] = deg > static_cast<IdType>(num_picks) ? deg : 0;
    out_deg[out_row] = min(static_cast<IdType>(num_picks), deg);

    if (out_row == num_rows - 1) {
      // make the prefixsum work
      out_deg[num_rows] = 0;
      temp_deg[num_rows] = 0;
    }
  }
}

/**
 * @brief Compute the size of each row in the sampled CSR, with replacement.
 * We need the actual in degree of each row to store CDF values.
 *
 * @tparam IdType The type of node and edge indexes.
 * @param num_picks The number of non-zero entries to pick per row.
 * @param num_rows The number of rows to pick.
 * @param in_rows The set of rows to pick.
 * @param in_ptr The index where each row's edges start.
 * @param out_deg The size of each row in the sampled matrix, as indexed by
 * `in_rows` (output).
 * @param temp_deg The size of each row in the input matrix, as indexed by
 * `in_rows` (output).
 */
template <typename IdType>
__global__ void _CSRRowWiseSampleDegreeReplaceKernel(
    const int64_t num_picks, const int64_t num_rows,
    const IdType* const in_rows, const IdType* const in_ptr,
    IdType* const out_deg, IdType* const temp_deg) {
  const int64_t tIdx = threadIdx.x + blockIdx.x * blockDim.x;

  if (tIdx < num_rows) {
    const int64_t in_row = in_rows[tIdx];
    const int64_t out_row = tIdx;
    const IdType deg = in_ptr[in_row + 1] - in_ptr[in_row];
    temp_deg[out_row] = deg;
    out_deg[out_row] = deg == 0 ? 0 : static_cast<IdType>(num_picks);

    if (out_row == num_rows - 1) {
      // make the prefixsum work
      out_deg[num_rows] = 0;
      temp_deg[num_rows] = 0;
    }
  }
}

/**
 * @brief Equivalent to numpy expression: array[idx[off:off + len]]
 *
 * @tparam IdType The ID type used for indices.
 * @tparam FloatType The float type used for array values.
 * @param array The array to be selected.
 * @param idx_data The index mapping array.
 * @param index The index of value to be selected.
 * @param offset The offset to start.
 * @param out The selected value (output).
 */
template <typename IdType, typename FloatType>
__device__ void _DoubleSlice(
    const FloatType* const array, const IdType* const idx_data,
    const IdType idx, const IdType offset, FloatType* const out) {
  if (idx_data) {
    *out = array[idx_data[offset + idx]];
  } else {
    *out = array[offset + idx];
  }
}

/**
 * @brief Compute A-Res value. A-Res value needs to be calculated only if deg
 * is greater than num_picks in weighted rowwise sampling without replacement.
 *
 * @tparam IdType The ID type used for matrices.
 * @tparam FloatType The Float type used for matrices.
 * @tparam TILE_SIZE The number of rows covered by each threadblock.
 * @param rand_seed The random seed to use.
 * @param num_picks The number of non-zeros to pick per row.
 * @param num_rows The number of rows to pick.
 * @param in_rows The set of rows to pick.
 * @param in_ptr The indptr array of the input CSR.
 * @param data The data array of the input CSR.
 * @param prob The probability array of the input CSR.
 * @param ares_ptr The offset to write each row to in the A-res array.
 * @param ares_idxs The A-Res value corresponding index array, the index of
 * input CSR (output).
 * @param ares The A-Res value array (output).
 * @author pengqirong (OPPO)
 */
template <typename IdType, typename FloatType, int TILE_SIZE>
__global__ void _CSRAResValueKernel(
    const uint64_t rand_seed, const int64_t num_picks, const int64_t num_rows,
    const IdType* const in_rows, const IdType* const in_ptr,
    const IdType* const data, const FloatType* const prob,
    const IdType* const ares_ptr, IdType* const ares_idxs,
    FloatType* const ares) {
  int64_t out_row = blockIdx.x * TILE_SIZE;
  const int64_t last_row =
      min(static_cast<int64_t>(blockIdx.x + 1) * TILE_SIZE, num_rows);

  curandStatePhilox4_32_10_t rng;
  curand_init(rand_seed * gridDim.x + blockIdx.x, threadIdx.x, 0, &rng);

  while (out_row < last_row) {
    const int64_t row = in_rows[out_row];
    const int64_t in_row_start = in_ptr[row];
    const int64_t deg = in_ptr[row + 1] - in_row_start;
    // A-Res value needs to be calculated only if deg is greater than num_picks
    // in weighted rowwise sampling without replacement
    if (deg > num_picks) {
      const int64_t ares_row_start = ares_ptr[out_row];

      for (int64_t idx = threadIdx.x; idx < deg; idx += BLOCK_SIZE) {
        const int64_t in_idx = in_row_start + idx;
        const int64_t ares_idx = ares_row_start + idx;
        FloatType item_prob;
        _DoubleSlice<IdType, FloatType>(
            prob, data, idx, in_row_start, &item_prob);
        // compute A-Res value
        ares[ares_idx] = static_cast<FloatType>(
            __powf(curand_uniform(&rng), 1.0f / item_prob));
        ares_idxs[ares_idx] = static_cast<IdType>(in_idx);
      }
    }
    out_row += 1;
  }
}

/**
 * @brief Perform weighted row-wise sampling on a CSR matrix, and generate a COO
 * matrix, without replacement. After sorting, we select top-num_picks items.
 *
 * @tparam IdType The ID type used for matrices.
 * @tparam FloatType The Float type used for matrices.
 * @tparam TILE_SIZE The number of rows covered by each threadblock.
 * @param num_picks The number of non-zeros to pick per row.
 * @param num_rows The number of rows to pick.
 * @param in_rows The set of rows to pick.
 * @param in_ptr The indptr array of the input CSR.
 * @param in_cols The columns array of the input CSR.
 * @param data The data array of the input CSR.
 * @param out_ptr The offset to write each row to in the output COO.
 * @param ares_ptr The offset to write each row to in the ares array.
 * @param sort_ares_idxs The sorted A-Res value corresponding index array, the
 * index of input CSR.
 * @param out_rows The rows of the output COO (output).
 * @param out_cols The columns of the output COO (output).
 * @param out_idxs The data array of the output COO (output).
 * @author pengqirong (OPPO)
 */
template <typename IdType, typename FloatType, int TILE_SIZE>
__global__ void _CSRRowWiseSampleKernel(
    const int64_t num_picks, const int64_t num_rows,
    const IdType* const in_rows, const IdType* const in_ptr,
    const IdType* const in_cols, const IdType* const data,
    const IdType* const out_ptr, const IdType* const ares_ptr,
    const IdType* const sort_ares_idxs, IdType* const out_rows,
    IdType* const out_cols, IdType* const out_idxs) {
  // we assign one warp per row
  assert(blockDim.x == BLOCK_SIZE);

  int64_t out_row = blockIdx.x * TILE_SIZE;
  const int64_t last_row =
      min(static_cast<int64_t>(blockIdx.x + 1) * TILE_SIZE, num_rows);

  while (out_row < last_row) {
    const int64_t row = in_rows[out_row];
    const int64_t in_row_start = in_ptr[row];
    const int64_t out_row_start = out_ptr[out_row];
    const int64_t deg = in_ptr[row + 1] - in_row_start;

    if (deg > num_picks) {
      const int64_t ares_row_start = ares_ptr[out_row];
      for (int64_t idx = threadIdx.x; idx < num_picks; idx += BLOCK_SIZE) {
        // get in and out index, the in_idx is one of top num_picks A-Res value
        // corresponding index in input CSR.
        const int64_t out_idx = out_row_start + idx;
        const int64_t ares_idx = ares_row_start + idx;
        const int64_t in_idx = sort_ares_idxs[ares_idx];
        // copy permutation over
        out_rows[out_idx] = static_cast<IdType>(row);
        out_cols[out_idx] = in_cols[in_idx];
        out_idxs[out_idx] = static_cast<IdType>(data ? data[in_idx] : in_idx);
      }
    } else {
      for (int64_t idx = threadIdx.x; idx < deg; idx += BLOCK_SIZE) {
        // get in and out index
        const int64_t out_idx = out_row_start + idx;
        const int64_t in_idx = in_row_start + idx;
        // copy permutation over
        out_rows[out_idx] = static_cast<IdType>(row);
        out_cols[out_idx] = in_cols[in_idx];
        out_idxs[out_idx] = static_cast<IdType>(data ? data[in_idx] : in_idx);
      }
    }
    out_row += 1;
  }
}

// A stateful callback functor that maintains a running prefix to be applied
// during consecutive scan operations.
template <typename FloatType>
struct BlockPrefixCallbackOp {
  // Running prefix
  FloatType running_total;
  // Constructor
  __device__ BlockPrefixCallbackOp(FloatType running_total)
      : running_total(running_total) {}
  // Callback operator to be entered by the first warp of threads in the block.
  // Thread-0 is responsible for returning a value for seeding the block-wide
  // scan.
  __device__ FloatType operator()(FloatType block_aggregate) {
    FloatType old_prefix = running_total;
    running_total += block_aggregate;
    return old_prefix;
  }
};

/**
 * @brief Perform weighted row-wise sampling on a CSR matrix, and generate a COO
 * matrix, with replacement. We store the CDF (unnormalized) of all neighbors of
 * a row in global memory and use binary search to find inverse indices as
 * selected items.
 *
 * @tparam IdType The ID type used for matrices.
 * @tparam FloatType The Float type used for matrices.
 * @tparam TILE_SIZE The number of rows covered by each threadblock.
 * @param rand_seed The random seed to use.
 * @param num_picks The number of non-zeros to pick per row.
 * @param num_rows The number of rows to pick.
 * @param in_rows The set of rows to pick.
 * @param in_ptr The indptr array of the input CSR.
 * @param in_cols The columns array of the input CSR.
 * @param data The data array of the input CSR.
 * @param prob The probability array of the input CSR.
 * @param out_ptr The offset to write each row to in the output COO.
 * @param cdf_ptr The offset of each cdf segment.
 * @param cdf The global buffer to store cdf segments.
 * @param out_rows The rows of the output COO (output).
 * @param out_cols The columns of the output COO (output).
 * @param out_idxs The data array of the output COO (output).
 * @author pengqirong (OPPO)
 */
template <typename IdType, typename FloatType, int TILE_SIZE>
__global__ void _CSRRowWiseSampleReplaceKernel(
    const uint64_t rand_seed, const int64_t num_picks, const int64_t num_rows,
    const IdType* const in_rows, const IdType* const in_ptr,
    const IdType* const in_cols, const IdType* const data,
    const FloatType* const prob, const IdType* const out_ptr,
    const IdType* const cdf_ptr, FloatType* const cdf, IdType* const out_rows,
    IdType* const out_cols, IdType* const out_idxs) {
  // we assign one warp per row
  assert(blockDim.x == BLOCK_SIZE);

  int64_t out_row = blockIdx.x * TILE_SIZE;
  const int64_t last_row =
      min(static_cast<int64_t>(blockIdx.x + 1) * TILE_SIZE, num_rows);

  curandStatePhilox4_32_10_t rng;
  curand_init(rand_seed * gridDim.x + blockIdx.x, threadIdx.x, 0, &rng);

  while (out_row < last_row) {
    const int64_t row = in_rows[out_row];
    const int64_t in_row_start = in_ptr[row];
    const int64_t out_row_start = out_ptr[out_row];
    const int64_t cdf_row_start = cdf_ptr[out_row];
    const int64_t deg = in_ptr[row + 1] - in_row_start;
    const FloatType MIN_THREAD_DATA = static_cast<FloatType>(0.0f);

    if (deg > 0) {
      // Specialize BlockScan for a 1D block of BLOCK_SIZE threads
      typedef cub::BlockScan<FloatType, BLOCK_SIZE> BlockScan;
      // Allocate shared memory for BlockScan
      __shared__ typename BlockScan::TempStorage temp_storage;
      // Initialize running total
      BlockPrefixCallbackOp<FloatType> prefix_op(MIN_THREAD_DATA);

      int64_t max_iter = (1 + (deg - 1) / BLOCK_SIZE) * BLOCK_SIZE;
      // Have the block iterate over segments of items
      for (int64_t idx = threadIdx.x; idx < max_iter; idx += BLOCK_SIZE) {
        // Load a segment of consecutive items that are blocked across threads
        FloatType thread_data;
        if (idx < deg)
          _DoubleSlice<IdType, FloatType>(
              prob, data, idx, in_row_start, &thread_data);
        else
          thread_data = MIN_THREAD_DATA;
        thread_data = max(thread_data, MIN_THREAD_DATA);
        // Collectively compute the block-wide inclusive prefix sum
        BlockScan(temp_storage)
            .InclusiveSum(thread_data, thread_data, prefix_op);
        __syncthreads();

        // Store scanned items to cdf array
        if (idx < deg) {
          cdf[cdf_row_start + idx] = thread_data;
        }
      }
      __syncthreads();

      for (int64_t idx = threadIdx.x; idx < num_picks; idx += BLOCK_SIZE) {
        // get random value
        FloatType sum = cdf[cdf_row_start + deg - 1];
        FloatType rand = static_cast<FloatType>(curand_uniform(&rng) * sum);
        // get the offset of the first value within cdf array which is greater
        // than random value.
        int64_t item = cub::UpperBound<FloatType*, int64_t, FloatType>(
            &cdf[cdf_row_start], deg, rand);
        item = min(item, deg - 1);
        // get in and out index
        const int64_t in_idx = in_row_start + item;
        const int64_t out_idx = out_row_start + idx;
        // copy permutation over
        out_rows[out_idx] = static_cast<IdType>(row);
        out_cols[out_idx] = in_cols[in_idx];
        out_idxs[out_idx] = static_cast<IdType>(data ? data[in_idx] : in_idx);
      }
    }
    out_row += 1;
  }
}

template <typename IdType, typename DType, typename BoolType>
__global__ void _GenerateFlagsKernel(
    int64_t n, const IdType* idx, const DType* values, DType criteria,
    BoolType* output) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride_x = gridDim.x * blockDim.x;
  while (tx < n) {
    output[tx] = (values[idx ? idx[tx] : tx] != criteria);
    tx += stride_x;
  }
}

template <DGLDeviceType XPU, typename IdType, typename DType, typename MaskGen>
COOMatrix COOGeneralRemoveIf(const COOMatrix& coo, MaskGen maskgen) {
  using namespace dgl::cuda;

  const auto idtype = coo.row->dtype;
  const auto ctx = coo.row->ctx;
  const int64_t nnz = coo.row->shape[0];
  const IdType* row = coo.row.Ptr<IdType>();
  const IdType* col = coo.col.Ptr<IdType>();
  const IdArray& eid =
      COOHasData(coo) ? coo.data : Range(0, nnz, sizeof(IdType) * 8, ctx);
  const IdType* data = coo.data.Ptr<IdType>();
  IdArray new_row = IdArray::Empty({nnz}, idtype, ctx);
  IdArray new_col = IdArray::Empty({nnz}, idtype, ctx);
  IdArray new_eid = IdArray::Empty({nnz}, idtype, ctx);
  IdType* new_row_data = new_row.Ptr<IdType>();
  IdType* new_col_data = new_col.Ptr<IdType>();
  IdType* new_eid_data = new_eid.Ptr<IdType>();
  auto stream = runtime::getCurrentCUDAStream();
  auto device = runtime::DeviceAPI::Get(ctx);

  int8_t* flags = static_cast<int8_t*>(device->AllocWorkspace(ctx, nnz));
  int nt = dgl::cuda::FindNumThreads(nnz);
  int64_t nb = (nnz + nt - 1) / nt;

  maskgen(nb, nt, stream, nnz, data, flags);

  int64_t* rst =
      static_cast<int64_t*>(device->AllocWorkspace(ctx, sizeof(int64_t)));
  MaskSelect(device, ctx, row, flags, new_row_data, nnz, rst, stream);
  MaskSelect(device, ctx, col, flags, new_col_data, nnz, rst, stream);
  MaskSelect(device, ctx, data, flags, new_eid_data, nnz, rst, stream);

  int64_t new_len = GetCUDAScalar(device, ctx, rst);

  device->FreeWorkspace(ctx, flags);
  device->FreeWorkspace(ctx, rst);
  return COOMatrix(
      coo.num_rows, coo.num_cols, new_row.CreateView({new_len}, idtype, 0),
      new_col.CreateView({new_len}, idtype, 0),
      new_eid.CreateView({new_len}, idtype, 0));
}

template <DGLDeviceType XPU, typename IdType, typename DType>
COOMatrix _COORemoveIf(
    const COOMatrix& coo, const NDArray& values, DType criteria) {
  const DType* val = values.Ptr<DType>();
  auto maskgen = [val, criteria](
                     int nb, int nt, cudaStream_t stream, int64_t nnz,
                     const IdType* data, int8_t* flags) {
    CUDA_KERNEL_CALL(
        (_GenerateFlagsKernel<IdType, DType, int8_t>), nb, nt, 0, stream, nnz,
        data, val, criteria, flags);
  };
  return COOGeneralRemoveIf<XPU, IdType, DType, decltype(maskgen)>(
      coo, maskgen);
}

}  // namespace

/////////////////////////////// CSR ///////////////////////////////

/**
 * @brief Perform weighted row-wise sampling on a CSR matrix, and generate a COO
 * matrix. Use CDF sampling algorithm for with replacement:
 *   1) Calculate the CDF of all neighbor's prob.
 *   2) For each [0, num_picks), generate a rand ~ U(0, 1). Use binary search to
 *      find its index in the CDF array as a chosen item.
 * Use A-Res sampling algorithm for without replacement:
 *   1) For rows with deg > num_picks, calculate A-Res values for all neighbors.
 *   2) Sort the A-Res array and select top-num_picks as chosen items.
 *
 * @tparam XPU The device type used for matrices.
 * @tparam IdType The ID type used for matrices.
 * @tparam FloatType The Float type used for matrices.
 * @param mat The CSR matrix.
 * @param rows The set of rows to pick.
 * @param num_picks The number of non-zeros to pick per row.
 * @param prob The probability array of the input CSR.
 * @param replace Is replacement sampling?
 * @author pengqirong (OPPO), dlasalle and Xin from Nvidia.
 */
template <DGLDeviceType XPU, typename IdType, typename FloatType>
COOMatrix _CSRRowWiseSampling(
    const CSRMatrix& mat, const IdArray& rows, int64_t num_picks,
    const FloatArray& prob, bool replace) {
  const auto& ctx = rows->ctx;
  auto device = runtime::DeviceAPI::Get(ctx);
  cudaStream_t stream = runtime::getCurrentCUDAStream();

  const int64_t num_rows = rows->shape[0];
  const IdType* const slice_rows = static_cast<const IdType*>(rows->data);

  IdArray picked_row =
      NewIdArray(num_rows * num_picks, ctx, sizeof(IdType) * 8);
  IdArray picked_col =
      NewIdArray(num_rows * num_picks, ctx, sizeof(IdType) * 8);
  IdArray picked_idx =
      NewIdArray(num_rows * num_picks, ctx, sizeof(IdType) * 8);
  IdType* const out_rows = static_cast<IdType*>(picked_row->data);
  IdType* const out_cols = static_cast<IdType*>(picked_col->data);
  IdType* const out_idxs = static_cast<IdType*>(picked_idx->data);

  const IdType* in_ptr = static_cast<IdType*>(GetDevicePointer(mat.indptr));
  const IdType* in_cols = static_cast<IdType*>(GetDevicePointer(mat.indices));
  const IdType* data = CSRHasData(mat)
                           ? static_cast<IdType*>(GetDevicePointer(mat.data))
                           : nullptr;
  const FloatType* prob_data = static_cast<FloatType*>(GetDevicePointer(prob));

  // compute degree
  // out_deg: the size of each row in the sampled matrix
  // temp_deg: the size of each row we will manipulate in sampling
  //    1) for w/o replacement: in degree if it's greater than num_picks else 0
  //    2) for w/ replacement: in degree
  IdType* out_deg = static_cast<IdType*>(
      device->AllocWorkspace(ctx, (num_rows + 1) * sizeof(IdType)));
  IdType* temp_deg = static_cast<IdType*>(
      device->AllocWorkspace(ctx, (num_rows + 1) * sizeof(IdType)));
  if (replace) {
    const dim3 block(512);
    const dim3 grid((num_rows + block.x - 1) / block.x);
    CUDA_KERNEL_CALL(
        _CSRRowWiseSampleDegreeReplaceKernel, grid, block, 0, stream, num_picks,
        num_rows, slice_rows, in_ptr, out_deg, temp_deg);
  } else {
    const dim3 block(512);
    const dim3 grid((num_rows + block.x - 1) / block.x);
    CUDA_KERNEL_CALL(
        _CSRRowWiseSampleDegreeKernel, grid, block, 0, stream, num_picks,
        num_rows, slice_rows, in_ptr, out_deg, temp_deg);
  }

  // fill temp_ptr
  IdType* temp_ptr = static_cast<IdType*>(
      device->AllocWorkspace(ctx, (num_rows + 1) * sizeof(IdType)));
  size_t prefix_temp_size = 0;
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
      nullptr, prefix_temp_size, temp_deg, temp_ptr, num_rows + 1, stream));
  void* prefix_temp = device->AllocWorkspace(ctx, prefix_temp_size);
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
      prefix_temp, prefix_temp_size, temp_deg, temp_ptr, num_rows + 1, stream));
  device->FreeWorkspace(ctx, prefix_temp);
  device->FreeWorkspace(ctx, temp_deg);

  // TODO(Xin): The copy here is too small, and the overhead of creating
  // cuda events cannot be ignored. Just use synchronized copy.
  IdType temp_len;
  // copy using the internal current stream.
  device->CopyDataFromTo(
      temp_ptr, num_rows * sizeof(temp_len), &temp_len, 0, sizeof(temp_len),
      ctx, DGLContext{kDGLCPU, 0}, mat.indptr->dtype);
  device->StreamSync(ctx, stream);

  // fill out_ptr
  IdType* out_ptr = static_cast<IdType*>(
      device->AllocWorkspace(ctx, (num_rows + 1) * sizeof(IdType)));
  prefix_temp_size = 0;
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
      nullptr, prefix_temp_size, out_deg, out_ptr, num_rows + 1, stream));
  prefix_temp = device->AllocWorkspace(ctx, prefix_temp_size);
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
      prefix_temp, prefix_temp_size, out_deg, out_ptr, num_rows + 1, stream));
  device->FreeWorkspace(ctx, prefix_temp);
  device->FreeWorkspace(ctx, out_deg);

  cudaEvent_t copyEvent;
  CUDA_CALL(cudaEventCreate(&copyEvent));
  // TODO(dlasalle): use pinned memory to overlap with the actual sampling, and
  // wait on a cudaevent
  IdType new_len;
  // copy using the internal current stream.
  device->CopyDataFromTo(
      out_ptr, num_rows * sizeof(new_len), &new_len, 0, sizeof(new_len), ctx,
      DGLContext{kDGLCPU, 0}, mat.indptr->dtype);
  CUDA_CALL(cudaEventRecord(copyEvent, stream));

  // allocate workspace
  // 1) for w/ replacement, it's a global buffer to store cdf segments (one
  // segment for each row).
  // 2) for w/o replacement, it's used to store a-res segments (one segment for
  // each row with degree > num_picks)
  FloatType* temp = static_cast<FloatType*>(
      device->AllocWorkspace(ctx, temp_len * sizeof(FloatType)));

  const uint64_t rand_seed = RandomEngine::ThreadLocal()->RandInt(1000000000);

  // select edges
  // the number of rows each thread block will cover
  constexpr int TILE_SIZE = 128 / BLOCK_SIZE;
  if (replace) {  // with replacement.
    const dim3 block(BLOCK_SIZE);
    const dim3 grid((num_rows + TILE_SIZE - 1) / TILE_SIZE);
    CUDA_KERNEL_CALL(
        (_CSRRowWiseSampleReplaceKernel<IdType, FloatType, TILE_SIZE>), grid,
        block, 0, stream, rand_seed, num_picks, num_rows, slice_rows, in_ptr,
        in_cols, data, prob_data, out_ptr, temp_ptr, temp, out_rows, out_cols,
        out_idxs);
    device->FreeWorkspace(ctx, temp);
  } else {  // without replacement
    IdType* temp_idxs = static_cast<IdType*>(
        device->AllocWorkspace(ctx, (temp_len) * sizeof(IdType)));

    // Compute A-Res value. A-Res value needs to be calculated only if deg
    // is greater than num_picks in weighted rowwise sampling without
    // replacement.
    const dim3 block(BLOCK_SIZE);
    const dim3 grid((num_rows + TILE_SIZE - 1) / TILE_SIZE);
    CUDA_KERNEL_CALL(
        (_CSRAResValueKernel<IdType, FloatType, TILE_SIZE>), grid, block, 0,
        stream, rand_seed, num_picks, num_rows, slice_rows, in_ptr, data,
        prob_data, temp_ptr, temp_idxs, temp);

    // sort A-Res value array.
    FloatType* sort_temp = static_cast<FloatType*>(
        device->AllocWorkspace(ctx, temp_len * sizeof(FloatType)));
    IdType* sort_temp_idxs = static_cast<IdType*>(
        device->AllocWorkspace(ctx, temp_len * sizeof(IdType)));

    cub::DoubleBuffer<FloatType> sort_keys(temp, sort_temp);
    cub::DoubleBuffer<IdType> sort_values(temp_idxs, sort_temp_idxs);

    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    CUDA_CALL(cub::DeviceSegmentedSort::SortPairsDescending(
        d_temp_storage, temp_storage_bytes, sort_keys, sort_values, temp_len,
        num_rows, temp_ptr, temp_ptr + 1, stream));
    d_temp_storage = device->AllocWorkspace(ctx, temp_storage_bytes);
    CUDA_CALL(cub::DeviceSegmentedSort::SortPairsDescending(
        d_temp_storage, temp_storage_bytes, sort_keys, sort_values, temp_len,
        num_rows, temp_ptr, temp_ptr + 1, stream));
    device->FreeWorkspace(ctx, d_temp_storage);
    device->FreeWorkspace(ctx, temp);
    device->FreeWorkspace(ctx, temp_idxs);
    device->FreeWorkspace(ctx, sort_temp);
    device->FreeWorkspace(ctx, sort_temp_idxs);

    // select tok-num_picks as results
    CUDA_KERNEL_CALL(
        (_CSRRowWiseSampleKernel<IdType, FloatType, TILE_SIZE>), grid, block, 0,
        stream, num_picks, num_rows, slice_rows, in_ptr, in_cols, data, out_ptr,
        temp_ptr, sort_values.Current(), out_rows, out_cols, out_idxs);
  }

  device->FreeWorkspace(ctx, temp_ptr);
  device->FreeWorkspace(ctx, out_ptr);

  // wait for copying `new_len` to finish
  CUDA_CALL(cudaEventSynchronize(copyEvent));
  CUDA_CALL(cudaEventDestroy(copyEvent));

  picked_row = picked_row.CreateView({new_len}, picked_row->dtype);
  picked_col = picked_col.CreateView({new_len}, picked_col->dtype);
  picked_idx = picked_idx.CreateView({new_len}, picked_idx->dtype);

  return COOMatrix(
      mat.num_rows, mat.num_cols, picked_row, picked_col, picked_idx);
}

template <DGLDeviceType XPU, typename IdType, typename DType>
COOMatrix CSRRowWiseSampling(
    CSRMatrix mat, IdArray rows, int64_t num_picks, FloatArray prob,
    bool replace) {
  COOMatrix result;
  if (num_picks == -1) {
    // Basically this is UnitGraph::InEdges().
    COOMatrix coo = CSRToCOO(CSRSliceRows(mat, rows), false);
    IdArray sliced_rows = IndexSelect(rows, coo.row);
    result =
        COOMatrix(mat.num_rows, mat.num_cols, sliced_rows, coo.col, coo.data);
  } else {
    result = _CSRRowWiseSampling<XPU, IdType, DType>(
        mat, rows, num_picks, prob, replace);
  }
  // NOTE(BarclayII): I'm removing the entries with zero probability after
  // sampling. Is there a better way?
  return _COORemoveIf<XPU, IdType, DType>(result, prob, static_cast<DType>(0));
}

template COOMatrix CSRRowWiseSampling<kDGLCUDA, int32_t, float>(
    CSRMatrix, IdArray, int64_t, FloatArray, bool);
template COOMatrix CSRRowWiseSampling<kDGLCUDA, int64_t, float>(
    CSRMatrix, IdArray, int64_t, FloatArray, bool);
template COOMatrix CSRRowWiseSampling<kDGLCUDA, int32_t, double>(
    CSRMatrix, IdArray, int64_t, FloatArray, bool);
template COOMatrix CSRRowWiseSampling<kDGLCUDA, int64_t, double>(
    CSRMatrix, IdArray, int64_t, FloatArray, bool);
// These are not being called, but we instantiate them anyway to prevent missing
// symbols in Debug build
template COOMatrix CSRRowWiseSampling<kDGLCUDA, int32_t, int8_t>(
    CSRMatrix, IdArray, int64_t, FloatArray, bool);
template COOMatrix CSRRowWiseSampling<kDGLCUDA, int64_t, int8_t>(
    CSRMatrix, IdArray, int64_t, FloatArray, bool);
template COOMatrix CSRRowWiseSampling<kDGLCUDA, int32_t, uint8_t>(
    CSRMatrix, IdArray, int64_t, FloatArray, bool);
template COOMatrix CSRRowWiseSampling<kDGLCUDA, int64_t, uint8_t>(
    CSRMatrix, IdArray, int64_t, FloatArray, bool);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
