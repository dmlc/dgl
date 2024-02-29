/**
 *  Copyright (c) 2020 by Contributors
 * @file array/cuda/spmat_op_impl_csr.cu
 * @brief CSR operator CPU implementation
 */
#include <dgl/array.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>

#include <cub/cub.cuh>
#include <numeric>
#include <unordered_set>
#include <vector>

#include "../../runtime/cuda/cuda_common.h"
#include "./atomic.cuh"
#include "./utils.h"

namespace dgl {

using runtime::NDArray;
using namespace cuda;

namespace aten {
namespace impl {

///////////////////////////// CSRIsNonZero /////////////////////////////

template <DGLDeviceType XPU, typename IdType>
bool CSRIsNonZero(CSRMatrix csr, int64_t row, int64_t col) {
  cudaStream_t stream = runtime::getCurrentCUDAStream();
  const auto& ctx = csr.indptr->ctx;
  IdArray rows = aten::VecToIdArray<int64_t>({row}, sizeof(IdType) * 8, ctx);
  IdArray cols = aten::VecToIdArray<int64_t>({col}, sizeof(IdType) * 8, ctx);
  rows = rows.CopyTo(ctx);
  cols = cols.CopyTo(ctx);
  IdArray out = aten::NewIdArray(1, ctx, sizeof(IdType) * 8);
  const IdType* data = nullptr;
  // TODO(minjie): use binary search for sorted csr
  CUDA_KERNEL_CALL(
      dgl::cuda::_LinearSearchKernel, 1, 1, 0, stream, csr.indptr.Ptr<IdType>(),
      csr.indices.Ptr<IdType>(), data, rows.Ptr<IdType>(), cols.Ptr<IdType>(),
      1, 1, 1, static_cast<IdType*>(nullptr), static_cast<IdType>(-1),
      out.Ptr<IdType>());
  out = out.CopyTo(DGLContext{kDGLCPU, 0});
  return *out.Ptr<IdType>() != -1;
}

template bool CSRIsNonZero<kDGLCUDA, int32_t>(CSRMatrix, int64_t, int64_t);
template bool CSRIsNonZero<kDGLCUDA, int64_t>(CSRMatrix, int64_t, int64_t);

template <DGLDeviceType XPU, typename IdType>
NDArray CSRIsNonZero(CSRMatrix csr, NDArray row, NDArray col) {
  const auto rowlen = row->shape[0];
  const auto collen = col->shape[0];
  const auto rstlen = std::max(rowlen, collen);
  NDArray rst = NDArray::Empty({rstlen}, row->dtype, row->ctx);
  if (rstlen == 0) return rst;
  const int64_t row_stride = (rowlen == 1 && collen != 1) ? 0 : 1;
  const int64_t col_stride = (collen == 1 && rowlen != 1) ? 0 : 1;
  cudaStream_t stream = runtime::getCurrentCUDAStream();
  const int nt = dgl::cuda::FindNumThreads(rstlen);
  const int nb = (rstlen + nt - 1) / nt;
  const IdType* data = nullptr;
  const IdType* indptr_data =
      static_cast<IdType*>(GetDevicePointer(csr.indptr));
  const IdType* indices_data =
      static_cast<IdType*>(GetDevicePointer(csr.indices));
  // TODO(minjie): use binary search for sorted csr
  CUDA_KERNEL_CALL(
      dgl::cuda::_LinearSearchKernel, nb, nt, 0, stream, indptr_data,
      indices_data, data, row.Ptr<IdType>(), col.Ptr<IdType>(), row_stride,
      col_stride, rstlen, static_cast<IdType*>(nullptr),
      static_cast<IdType>(-1), rst.Ptr<IdType>());
  return rst != -1;
}

template NDArray CSRIsNonZero<kDGLCUDA, int32_t>(CSRMatrix, NDArray, NDArray);
template NDArray CSRIsNonZero<kDGLCUDA, int64_t>(CSRMatrix, NDArray, NDArray);

///////////////////////////// CSRHasDuplicate /////////////////////////////

/**
 * @brief Check whether each row does not have any duplicate entries.
 * Assume the CSR is sorted.
 */
template <typename IdType>
__global__ void _SegmentHasNoDuplicate(
    const IdType* indptr, const IdType* indices, int64_t num_rows,
    int8_t* flags) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride_x = gridDim.x * blockDim.x;
  while (tx < num_rows) {
    bool f = true;
    for (IdType i = indptr[tx] + 1; f && i < indptr[tx + 1]; ++i) {
      f = (indices[i - 1] != indices[i]);
    }
    flags[tx] = static_cast<int8_t>(f);
    tx += stride_x;
  }
}

template <DGLDeviceType XPU, typename IdType>
bool CSRHasDuplicate(CSRMatrix csr) {
  if (!csr.sorted) csr = CSRSort(csr);
  const auto& ctx = csr.indptr->ctx;
  cudaStream_t stream = runtime::getCurrentCUDAStream();
  auto device = runtime::DeviceAPI::Get(ctx);
  // We allocate a workspace of num_rows bytes. It wastes a little bit memory
  // but should be fine.
  int8_t* flags =
      static_cast<int8_t*>(device->AllocWorkspace(ctx, csr.num_rows));
  const int nt = dgl::cuda::FindNumThreads(csr.num_rows);
  const int nb = (csr.num_rows + nt - 1) / nt;
  CUDA_KERNEL_CALL(
      _SegmentHasNoDuplicate, nb, nt, 0, stream, csr.indptr.Ptr<IdType>(),
      csr.indices.Ptr<IdType>(), csr.num_rows, flags);
  bool ret = dgl::cuda::AllTrue(flags, csr.num_rows, ctx);
  device->FreeWorkspace(ctx, flags);
  return !ret;
}

template bool CSRHasDuplicate<kDGLCUDA, int32_t>(CSRMatrix csr);
template bool CSRHasDuplicate<kDGLCUDA, int64_t>(CSRMatrix csr);

///////////////////////////// CSRGetRowNNZ /////////////////////////////

template <DGLDeviceType XPU, typename IdType>
int64_t CSRGetRowNNZ(CSRMatrix csr, int64_t row) {
  const IdType cur = aten::IndexSelect<IdType>(csr.indptr, row);
  const IdType next = aten::IndexSelect<IdType>(csr.indptr, row + 1);
  return next - cur;
}

template int64_t CSRGetRowNNZ<kDGLCUDA, int32_t>(CSRMatrix, int64_t);
template int64_t CSRGetRowNNZ<kDGLCUDA, int64_t>(CSRMatrix, int64_t);

template <typename IdType>
__global__ void _CSRGetRowNNZKernel(
    const IdType* vid, const IdType* indptr, IdType* out, int64_t length) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride_x = gridDim.x * blockDim.x;
  while (tx < length) {
    const IdType vv = vid[tx];
    out[tx] = indptr[vv + 1] - indptr[vv];
    tx += stride_x;
  }
}

template <DGLDeviceType XPU, typename IdType>
NDArray CSRGetRowNNZ(CSRMatrix csr, NDArray rows) {
  cudaStream_t stream = runtime::getCurrentCUDAStream();
  const auto len = rows->shape[0];
  const IdType* vid_data = rows.Ptr<IdType>();
  const IdType* indptr_data =
      static_cast<IdType*>(GetDevicePointer(csr.indptr));
  NDArray rst = NDArray::Empty({len}, rows->dtype, rows->ctx);
  IdType* rst_data = static_cast<IdType*>(rst->data);
  const int nt = dgl::cuda::FindNumThreads(len);
  const int nb = (len + nt - 1) / nt;
  CUDA_KERNEL_CALL(
      _CSRGetRowNNZKernel, nb, nt, 0, stream, vid_data, indptr_data, rst_data,
      len);
  return rst;
}

template NDArray CSRGetRowNNZ<kDGLCUDA, int32_t>(CSRMatrix, NDArray);
template NDArray CSRGetRowNNZ<kDGLCUDA, int64_t>(CSRMatrix, NDArray);

////////////////////////// CSRGetRowColumnIndices //////////////////////////////

template <DGLDeviceType XPU, typename IdType>
NDArray CSRGetRowColumnIndices(CSRMatrix csr, int64_t row) {
  const int64_t len = impl::CSRGetRowNNZ<XPU, IdType>(csr, row);
  const int64_t offset =
      aten::IndexSelect<IdType>(csr.indptr, row) * sizeof(IdType);
  return csr.indices.CreateView({len}, csr.indices->dtype, offset);
}

template NDArray CSRGetRowColumnIndices<kDGLCUDA, int32_t>(CSRMatrix, int64_t);
template NDArray CSRGetRowColumnIndices<kDGLCUDA, int64_t>(CSRMatrix, int64_t);

///////////////////////////// CSRGetRowData /////////////////////////////

template <DGLDeviceType XPU, typename IdType>
NDArray CSRGetRowData(CSRMatrix csr, int64_t row) {
  const int64_t len = impl::CSRGetRowNNZ<XPU, IdType>(csr, row);
  const int64_t offset =
      aten::IndexSelect<IdType>(csr.indptr, row) * sizeof(IdType);
  if (aten::CSRHasData(csr))
    return csr.data.CreateView({len}, csr.data->dtype, offset);
  else
    return aten::Range(
        offset, offset + len, csr.indptr->dtype.bits, csr.indptr->ctx);
}

template NDArray CSRGetRowData<kDGLCUDA, int32_t>(CSRMatrix, int64_t);
template NDArray CSRGetRowData<kDGLCUDA, int64_t>(CSRMatrix, int64_t);

///////////////////////////// CSRSliceRows /////////////////////////////

template <DGLDeviceType XPU, typename IdType>
CSRMatrix CSRSliceRows(CSRMatrix csr, int64_t start, int64_t end) {
  const int64_t num_rows = end - start;
  const IdType st_pos = aten::IndexSelect<IdType>(csr.indptr, start);
  const IdType ed_pos = aten::IndexSelect<IdType>(csr.indptr, end);
  const IdType nnz = ed_pos - st_pos;
  IdArray ret_indptr = aten::IndexSelect(csr.indptr, start, end + 1) - st_pos;
  // indices and data can be view arrays
  IdArray ret_indices = csr.indices.CreateView(
      {nnz}, csr.indices->dtype, st_pos * sizeof(IdType));
  IdArray ret_data;
  if (CSRHasData(csr))
    ret_data =
        csr.data.CreateView({nnz}, csr.data->dtype, st_pos * sizeof(IdType));
  else
    ret_data =
        aten::Range(st_pos, ed_pos, csr.indptr->dtype.bits, csr.indptr->ctx);
  return CSRMatrix(
      num_rows, csr.num_cols, ret_indptr, ret_indices, ret_data, csr.sorted);
}

template CSRMatrix CSRSliceRows<kDGLCUDA, int32_t>(CSRMatrix, int64_t, int64_t);
template CSRMatrix CSRSliceRows<kDGLCUDA, int64_t>(CSRMatrix, int64_t, int64_t);

/**
 * @brief Copy data segment to output buffers
 *
 * For the i^th row r = row[i], copy the data from indptr[r] ~ indptr[r+1]
 * to the out_data from out_indptr[i] ~ out_indptr[i+1]
 *
 * If the provided `data` array is nullptr, write the read index to the
 * out_data.
 *
 */
template <typename IdType, typename DType>
__global__ void _SegmentCopyKernel(
    const IdType* indptr, const DType* data, const IdType* row, int64_t length,
    int64_t n_row, const IdType* out_indptr, DType* out_data) {
  IdType tx = static_cast<IdType>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int stride_x = gridDim.x * blockDim.x;
  while (tx < length) {
    IdType rpos = dgl::cuda::_UpperBound(out_indptr, n_row, tx) - 1;
    IdType rofs = tx - out_indptr[rpos];
    const IdType u = row[rpos];
    out_data[tx] = data ? data[indptr[u] + rofs] : indptr[u] + rofs;
    tx += stride_x;
  }
}

template <DGLDeviceType XPU, typename IdType>
CSRMatrix CSRSliceRows(CSRMatrix csr, NDArray rows) {
  cudaStream_t stream = runtime::getCurrentCUDAStream();
  const int64_t len = rows->shape[0];
  IdArray ret_indptr = aten::CumSum(aten::CSRGetRowNNZ(csr, rows), true);
  const int64_t nnz = aten::IndexSelect<IdType>(ret_indptr, len);

  const int nt = 256;  // for better GPU usage of small invocations
  const int nb = (nnz + nt - 1) / nt;

  // Copy indices.
  IdArray ret_indices = NDArray::Empty({nnz}, csr.indptr->dtype, rows->ctx);

  const IdType* indptr_data =
      static_cast<IdType*>(GetDevicePointer(csr.indptr));
  const IdType* indices_data =
      static_cast<IdType*>(GetDevicePointer(csr.indices));
  const IdType* data_data =
      CSRHasData(csr) ? static_cast<IdType*>(GetDevicePointer(csr.data))
                      : nullptr;

  CUDA_KERNEL_CALL(
      _SegmentCopyKernel, nb, nt, 0, stream, indptr_data, indices_data,
      rows.Ptr<IdType>(), nnz, len, ret_indptr.Ptr<IdType>(),
      ret_indices.Ptr<IdType>());
  // Copy data.
  IdArray ret_data = NDArray::Empty({nnz}, csr.indptr->dtype, rows->ctx);
  CUDA_KERNEL_CALL(
      _SegmentCopyKernel, nb, nt, 0, stream, indptr_data, data_data,
      rows.Ptr<IdType>(), nnz, len, ret_indptr.Ptr<IdType>(),
      ret_data.Ptr<IdType>());
  return CSRMatrix(
      len, csr.num_cols, ret_indptr, ret_indices, ret_data, csr.sorted);
}

template CSRMatrix CSRSliceRows<kDGLCUDA, int32_t>(CSRMatrix, NDArray);
template CSRMatrix CSRSliceRows<kDGLCUDA, int64_t>(CSRMatrix, NDArray);

///////////////////////////// CSRGetDataAndIndices /////////////////////////////

/**
 * @brief Generate a 0-1 mask for each index that hits the provided (row, col)
 *        index.
 *
 * Examples:
 * Given a CSR matrix (with duplicate entries) as follows:
 * [[0, 1, 2, 0, 0],
 *  [1, 0, 0, 0, 0],
 *  [0, 0, 1, 1, 0],
 *  [0, 0, 0, 0, 0]]
 * Given rows: [0, 1], cols: [0, 2, 3]
 * The result mask is: [0, 1, 1, 1, 0, 0]
 */
template <typename IdType>
__global__ void _SegmentMaskKernel(
    const IdType* indptr, const IdType* indices, const IdType* row,
    const IdType* col, int64_t row_stride, int64_t col_stride, int64_t length,
    IdType* mask) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride_x = gridDim.x * blockDim.x;
  while (tx < length) {
    int rpos = tx * row_stride, cpos = tx * col_stride;
    const IdType r = row[rpos], c = col[cpos];
    for (IdType i = indptr[r]; i < indptr[r + 1]; ++i) {
      if (indices[i] == c) {
        mask[i] = 1;
      }
    }
    tx += stride_x;
  }
}

/**
 * @brief Search for the insertion positions for needle in the hay.
 *
 * The hay is a list of sorted elements and the result is the insertion position
 * of each needle so that the insertion still gives sorted order.
 *
 * It essentially perform binary search to find lower bound for each needle
 * elements. Require the largest elements in the hay is larger than the given
 * needle elements. Commonly used in searching for row IDs of a given set of
 * coordinates.
 */
template <typename IdType>
__global__ void _SortedSearchKernel(
    const IdType* hay, int64_t hay_size, const IdType* needles,
    int64_t num_needles, IdType* pos) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride_x = gridDim.x * blockDim.x;
  while (tx < num_needles) {
    const IdType ele = needles[tx];
    // binary search
    IdType lo = 0, hi = hay_size - 1;
    while (lo < hi) {
      IdType mid = (lo + hi) >> 1;
      if (hay[mid] <= ele) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    pos[tx] = (hay[hi] == ele) ? hi : hi - 1;
    tx += stride_x;
  }
}

template <DGLDeviceType XPU, typename IdType>
std::vector<NDArray> CSRGetDataAndIndices(
    CSRMatrix csr, NDArray row, NDArray col) {
  const auto rowlen = row->shape[0];
  const auto collen = col->shape[0];
  const auto len = std::max(rowlen, collen);
  if (len == 0) return {NullArray(), NullArray(), NullArray()};

  const auto& ctx = row->ctx;
  const auto nbits = row->dtype.bits;
  const int64_t nnz = csr.indices->shape[0];
  const int64_t row_stride = (rowlen == 1 && collen != 1) ? 0 : 1;
  const int64_t col_stride = (collen == 1 && rowlen != 1) ? 0 : 1;
  cudaStream_t stream = runtime::getCurrentCUDAStream();

  const IdType* indptr_data =
      static_cast<IdType*>(GetDevicePointer(csr.indptr));
  const IdType* indices_data =
      static_cast<IdType*>(GetDevicePointer(csr.indices));

  // Generate a 0-1 mask for matched (row, col) positions.
  IdArray mask = Full(0, nnz, nbits, ctx);
  const int nt = dgl::cuda::FindNumThreads(len);
  const int nb = (len + nt - 1) / nt;
  CUDA_KERNEL_CALL(
      _SegmentMaskKernel, nb, nt, 0, stream, indptr_data, indices_data,
      row.Ptr<IdType>(), col.Ptr<IdType>(), row_stride, col_stride, len,
      mask.Ptr<IdType>());

  IdArray idx = AsNumBits(NonZero(mask), nbits);
  if (idx->shape[0] == 0)
    // No data. Return three empty arrays.
    return {idx, idx, idx};

  // Search for row index
  IdArray ret_row = NewIdArray(idx->shape[0], ctx, nbits);
  const int nt2 = dgl::cuda::FindNumThreads(idx->shape[0]);
  const int nb2 = (idx->shape[0] + nt - 1) / nt;
  CUDA_KERNEL_CALL(
      _SortedSearchKernel, nb2, nt2, 0, stream, indptr_data, csr.num_rows,
      idx.Ptr<IdType>(), idx->shape[0], ret_row.Ptr<IdType>());

  // Column & data can be obtained by index select.
  IdArray ret_col = IndexSelect(csr.indices, idx);
  IdArray ret_data = CSRHasData(csr) ? IndexSelect(csr.data, idx) : idx;
  return {ret_row, ret_col, ret_data};
}

template std::vector<NDArray> CSRGetDataAndIndices<kDGLCUDA, int32_t>(
    CSRMatrix csr, NDArray rows, NDArray cols);
template std::vector<NDArray> CSRGetDataAndIndices<kDGLCUDA, int64_t>(
    CSRMatrix csr, NDArray rows, NDArray cols);

///////////////////////////// CSRSliceMatrix /////////////////////////////

int64_t _UpPower(int64_t numel) {
  uint64_t ret = 1 << static_cast<uint64_t>(std::log2(numel) + 1);
  return ret;
}

/**
 * @brief Thomas Wang's 32 bit Mix Function.
 * Source link: https://gist.github.com/badboy/6267743
 */
__device__ inline uint32_t _Hash32Shift(uint32_t key) {
  key = ~key + (key << 15);
  key = key ^ (key >> 12);
  key = key + (key << 2);
  key = key ^ (key >> 4);
  key = key * 2057;
  key = key ^ (key >> 16);
  return key;
}

/**
 * @brief Thomas Wang's 64 bit Mix Function.
 * Source link: https://gist.github.com/badboy/6267743
 */
__device__ inline uint64_t _Hash64Shift(uint64_t key) {
  key = (~key) + (key << 21);
  key = key ^ (key >> 24);
  key = (key + (key << 3)) + (key << 8);
  key = key ^ (key >> 14);
  key = (key + (key << 2)) + (key << 4);
  key = key ^ (key >> 28);
  key = key + (key << 31);
  return key;
}

/**
 * @brief A hashmap designed for CSRSliceMatrix, similar in function to set. For
 * performance, it can only be created and called in the cuda kernel.
 */
template <typename IdType>
struct NodeQueryHashmap {
  __device__ inline NodeQueryHashmap(IdType* Kptr, size_t numel)
      : kptr_(Kptr), capacity_(numel) {}

  /**
   * @brief Insert a key. It must be called by cuda threads.
   *
   * @param key The key to be inserted.
   */
  __device__ inline void Insert(IdType key) {
    uint32_t delta = 1;
    uint32_t pos = Hash(key);
    IdType prev = dgl::aten::cuda::AtomicCAS(&kptr_[pos], kEmptyKey_, key);
    while (prev != key && prev != kEmptyKey_) {
      pos = Hash(pos + delta);
      delta += 1;
      prev = dgl::aten::cuda::AtomicCAS(&kptr_[pos], kEmptyKey_, key);
    }
  }

  /**
   * @brief Check whether a key exists within the hashtable. It must be called
   * by cuda threads.
   *
   * @param key The key to check for.
   * @return True if the key exists in the hashtable.
   */
  __device__ inline bool Query(IdType key) {
    uint32_t delta = 1;
    uint32_t pos = Hash(key);
    while (true) {
      if (kptr_[pos] == key) return true;
      if (kptr_[pos] == kEmptyKey_) return false;
      pos = Hash(pos + delta);
      delta += 1;
    }
    return false;
  }

  __device__ inline uint32_t Hash(int32_t key) {
    return _Hash32Shift(key) & (capacity_ - 1);
  }

  __device__ inline uint32_t Hash(uint32_t key) {
    return _Hash32Shift(key) & (capacity_ - 1);
  }

  __device__ inline uint32_t Hash(int64_t key) {
    return static_cast<uint32_t>(_Hash64Shift(key)) & (capacity_ - 1);
  }

  __device__ inline uint32_t Hash(uint64_t key) {
    return static_cast<uint32_t>(_Hash64Shift(key)) & (capacity_ - 1);
  }

  IdType kEmptyKey_{-1};
  IdType* kptr_;
  uint32_t capacity_{0};
};

/**
 * @brief Generate a 0-1 mask for each index whose column is in the provided
 * hashmap. It also counts the number of masked values per row.
 *
 * @tparam IdType The ID type used for matrices.
 * @tparam WARP_SIZE The number of cuda threads in a cuda warp.
 * @tparam BLOCK_WARPS The number of warps in a cuda block.
 * @tparam TILE_SIZE The number of rows covered by each threadblock.
 */
template <typename IdType, int WARP_SIZE, int BLOCK_WARPS, int TILE_SIZE>
__global__ void _SegmentMaskColKernel(
    const IdType* indptr, const IdType* indices, int64_t num_rows,
    IdType* hashmap_buffer, int64_t buffer_size, IdType* mask, IdType* count) {
  assert(blockDim.x == WARP_SIZE);
  assert(blockDim.y == BLOCK_WARPS);

  int warp_id = threadIdx.y;
  int laneid = threadIdx.x;
  IdType out_row = blockIdx.x * TILE_SIZE + threadIdx.y;
  IdType last_row =
      min(static_cast<IdType>((blockIdx.x + 1) * TILE_SIZE),
          static_cast<IdType>(num_rows));

  NodeQueryHashmap<IdType> hashmap(hashmap_buffer, buffer_size);
  typedef cub::WarpReduce<IdType> WarpReduce;
  __shared__ typename WarpReduce::TempStorage temp_storage[BLOCK_WARPS];

  while (out_row < last_row) {
    IdType local_count = 0;
    IdType in_row_start = indptr[out_row];
    IdType in_row_end = indptr[out_row + 1];
    for (int idx = in_row_start + laneid; idx < in_row_end; idx += WARP_SIZE) {
      bool is_in = hashmap.Query(indices[idx]);
      if (is_in) {
        local_count += 1;
        mask[idx] = 1;
      }
    }
    IdType reduce_count = WarpReduce(temp_storage[warp_id]).Sum(local_count);
    if (laneid == 0) {
      count[out_row] = reduce_count;
    }
    out_row += BLOCK_WARPS;
  }
}

template <DGLDeviceType XPU, typename IdType>
CSRMatrix CSRSliceMatrix(
    CSRMatrix csr, runtime::NDArray rows, runtime::NDArray cols) {
  cudaStream_t stream = runtime::getCurrentCUDAStream();
  const auto& ctx = rows->ctx;
  const auto& dtype = rows->dtype;
  const auto nbits = dtype.bits;
  const int64_t new_nrows = rows->shape[0];
  const int64_t new_ncols = cols->shape[0];

  if (new_nrows == 0 || new_ncols == 0)
    return CSRMatrix(
        new_nrows, new_ncols, Full(0, new_nrows + 1, nbits, ctx),
        NullArray(dtype, ctx), NullArray(dtype, ctx));

  // First slice rows
  csr = CSRSliceRows(csr, rows);

  if (csr.indices->shape[0] == 0)
    return CSRMatrix(
        new_nrows, new_ncols, Full(0, new_nrows + 1, nbits, ctx),
        NullArray(dtype, ctx), NullArray(dtype, ctx));

  // Generate a 0-1 mask for matched (row, col) positions.
  IdArray mask = Full(0, csr.indices->shape[0], nbits, ctx);
  // A count for how many masked values per row.
  IdArray count = NewIdArray(csr.num_rows, ctx, nbits);
  CUDA_CALL(
      cudaMemset(count.Ptr<IdType>(), 0, sizeof(IdType) * (csr.num_rows)));

  // Generate a NodeQueryHashmap buffer. The key of the hashmap is col.
  // For performance, the load factor of the hashmap is in (0.25, 0.5);
  // Because num_cols is usually less than 1 Million (on GPU), the
  // memory overhead is not significant (less than 31MB) at a low load factor.
  int64_t buffer_size = _UpPower(new_ncols) * 2;
  IdArray hashmap_buffer = Full(-1, buffer_size, nbits, ctx);

  using it = thrust::counting_iterator<int64_t>;
  runtime::CUDAWorkspaceAllocator allocator(ctx);
  const auto exec_policy = thrust::cuda::par_nosync(allocator).on(stream);
  thrust::for_each(
      exec_policy, it(0), it(new_ncols),
      [key = cols.Ptr<IdType>(), buffer = hashmap_buffer.Ptr<IdType>(),
       buffer_size] __device__(int64_t i) {
        NodeQueryHashmap<IdType> hashmap(buffer, buffer_size);
        hashmap.Insert(key[i]);
      });

  const IdType* indptr_data =
      static_cast<IdType*>(GetDevicePointer(csr.indptr));
  const IdType* indices_data =
      static_cast<IdType*>(GetDevicePointer(csr.indices));

  // Execute SegmentMaskColKernel
  const int64_t num_rows = csr.num_rows;
  constexpr int WARP_SIZE = 32;
  // With a simple fine-tuning, TILE_SIZE=16 gives a good performance.
  constexpr int TILE_SIZE = 16;
  constexpr int BLOCK_WARPS = CUDA_MAX_NUM_THREADS / WARP_SIZE;
  IdType nb =
      dgl::cuda::FindNumBlocks<'x'>((num_rows + TILE_SIZE - 1) / TILE_SIZE);
  const dim3 nthrs(WARP_SIZE, BLOCK_WARPS);
  const dim3 nblks(nb);
  CUDA_KERNEL_CALL(
      (_SegmentMaskColKernel<IdType, WARP_SIZE, BLOCK_WARPS, TILE_SIZE>), nblks,
      nthrs, 0, stream, indptr_data, indices_data, num_rows,
      hashmap_buffer.Ptr<IdType>(), buffer_size, mask.Ptr<IdType>(),
      count.Ptr<IdType>());

  IdArray idx = AsNumBits(NonZero(mask), nbits);
  if (idx->shape[0] == 0)
    return CSRMatrix(
        new_nrows, new_ncols, Full(0, new_nrows + 1, nbits, ctx),
        NullArray(dtype, ctx), NullArray(dtype, ctx));

  // Indptr needs to be adjusted according to the new nnz per row.
  IdArray ret_indptr = CumSum(count, true);

  // Column & data can be obtained by index select.
  IdArray ret_col = IndexSelect(csr.indices, idx);
  IdArray ret_data = CSRHasData(csr) ? IndexSelect(csr.data, idx) : idx;

  // Relabel column
  IdArray col_hash = NewIdArray(csr.num_cols, ctx, nbits);
  Scatter_(cols, Range(0, cols->shape[0], nbits, ctx), col_hash);
  ret_col = IndexSelect(col_hash, ret_col);

  return CSRMatrix(new_nrows, new_ncols, ret_indptr, ret_col, ret_data);
}

template CSRMatrix CSRSliceMatrix<kDGLCUDA, int32_t>(
    CSRMatrix csr, runtime::NDArray rows, runtime::NDArray cols);
template CSRMatrix CSRSliceMatrix<kDGLCUDA, int64_t>(
    CSRMatrix csr, runtime::NDArray rows, runtime::NDArray cols);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
