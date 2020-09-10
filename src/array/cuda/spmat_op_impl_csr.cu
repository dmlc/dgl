/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cuda/spmat_op_impl_csr.cu
 * \brief CSR operator CPU implementation
 */
#include <dgl/array.h>
#include <vector>
#include <unordered_set>
#include <numeric>
#include "../../runtime/cuda/cuda_common.h"
#include "./utils.h"

namespace dgl {

using runtime::NDArray;

namespace aten {
namespace impl {

/*!
 * \brief Search adjacency list linearly for each (row, col) pair and
 * write the data under the matched position in the indices array to the output.
 *
 * If there is no match, -1 is written.
 * If there are multiple matches, only the first match is written.
 * If the given data array is null, write the matched position to the output.
 */
template <typename IdType>
__global__ void _LinearSearchKernel(
    const IdType* indptr, const IdType* indices, const IdType* data,
    const IdType* row, const IdType* col,
    int64_t row_stride, int64_t col_stride,
    int64_t length, IdType* out) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride_x = gridDim.x * blockDim.x;
  while (tx < length) {
    int rpos = tx * row_stride, cpos = tx * col_stride;
    IdType v = -1;
    const IdType r = row[rpos], c = col[cpos];
    for (IdType i = indptr[r]; i < indptr[r + 1]; ++i) {
      if (indices[i] == c) {
        v = (data)? data[i] : i;
        break;
      }
    }
    out[tx] = v;
    tx += stride_x;
  }
}

///////////////////////////// CSRIsNonZero /////////////////////////////

template <DLDeviceType XPU, typename IdType>
bool CSRIsNonZero(CSRMatrix csr, int64_t row, int64_t col) {
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  const auto& ctx = csr.indptr->ctx;
  IdArray rows = aten::VecToIdArray<int64_t>({row}, sizeof(IdType) * 8, ctx);
  IdArray cols = aten::VecToIdArray<int64_t>({col}, sizeof(IdType) * 8, ctx);
  rows = rows.CopyTo(ctx);
  cols = cols.CopyTo(ctx);
  IdArray out = aten::NewIdArray(1, ctx, sizeof(IdType) * 8);
  const IdType* data = nullptr;
  // TODO(minjie): use binary search for sorted csr
  CUDA_KERNEL_CALL(_LinearSearchKernel,
      1, 1, 0, thr_entry->stream,
      csr.indptr.Ptr<IdType>(), csr.indices.Ptr<IdType>(), data,
      rows.Ptr<IdType>(), cols.Ptr<IdType>(),
      1, 1, 1,
      out.Ptr<IdType>());
  out = out.CopyTo(DLContext{kDLCPU, 0});
  return *out.Ptr<IdType>() != -1;
}

template bool CSRIsNonZero<kDLGPU, int32_t>(CSRMatrix, int64_t, int64_t);
template bool CSRIsNonZero<kDLGPU, int64_t>(CSRMatrix, int64_t, int64_t);

template <DLDeviceType XPU, typename IdType>
NDArray CSRIsNonZero(CSRMatrix csr, NDArray row, NDArray col) {
  const auto rowlen = row->shape[0];
  const auto collen = col->shape[0];
  const auto rstlen = std::max(rowlen, collen);
  NDArray rst = NDArray::Empty({rstlen}, row->dtype, row->ctx);
  if (rstlen == 0)
    return rst;
  const int64_t row_stride = (rowlen == 1 && collen != 1) ? 0 : 1;
  const int64_t col_stride = (collen == 1 && rowlen != 1) ? 0 : 1;
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  const int nt = cuda::FindNumThreads(rstlen);
  const int nb = (rstlen + nt - 1) / nt;
  const IdType* data = nullptr;
  // TODO(minjie): use binary search for sorted csr
  CUDA_KERNEL_CALL(_LinearSearchKernel,
      nb, nt, 0, thr_entry->stream,
      csr.indptr.Ptr<IdType>(), csr.indices.Ptr<IdType>(), data,
      row.Ptr<IdType>(), col.Ptr<IdType>(),
      row_stride, col_stride, rstlen,
      rst.Ptr<IdType>());
  return rst != -1;
}

template NDArray CSRIsNonZero<kDLGPU, int32_t>(CSRMatrix, NDArray, NDArray);
template NDArray CSRIsNonZero<kDLGPU, int64_t>(CSRMatrix, NDArray, NDArray);

///////////////////////////// CSRHasDuplicate /////////////////////////////

/*!
 * \brief Check whether each row does not have any duplicate entries.
 * Assume the CSR is sorted.
 */
template <typename IdType>
__global__ void _SegmentHasNoDuplicate(
    const IdType* indptr, const IdType* indices,
    int64_t num_rows, int8_t* flags) {
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


template <DLDeviceType XPU, typename IdType>
bool CSRHasDuplicate(CSRMatrix csr) {
  if (!csr.sorted)
    csr = CSRSort(csr);
  const auto& ctx = csr.indptr->ctx;
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  auto device = runtime::DeviceAPI::Get(ctx);
  // We allocate a workspace of num_rows bytes. It wastes a little bit memory but should
  // be fine.
  int8_t* flags = static_cast<int8_t*>(device->AllocWorkspace(ctx, csr.num_rows));
  const int nt = cuda::FindNumThreads(csr.num_rows);
  const int nb = (csr.num_rows + nt - 1) / nt;
  CUDA_KERNEL_CALL(_SegmentHasNoDuplicate,
      nb, nt, 0, thr_entry->stream,
      csr.indptr.Ptr<IdType>(), csr.indices.Ptr<IdType>(),
      csr.num_rows, flags);
  bool ret = cuda::AllTrue(flags, csr.num_rows, ctx);
  device->FreeWorkspace(ctx, flags);
  return !ret;
}

template bool CSRHasDuplicate<kDLGPU, int32_t>(CSRMatrix csr);
template bool CSRHasDuplicate<kDLGPU, int64_t>(CSRMatrix csr);

///////////////////////////// CSRGetRowNNZ /////////////////////////////

template <DLDeviceType XPU, typename IdType>
int64_t CSRGetRowNNZ(CSRMatrix csr, int64_t row) {
  const IdType cur = aten::IndexSelect<IdType>(csr.indptr, row);
  const IdType next = aten::IndexSelect<IdType>(csr.indptr, row + 1);
  return next - cur;
}

template int64_t CSRGetRowNNZ<kDLGPU, int32_t>(CSRMatrix, int64_t);
template int64_t CSRGetRowNNZ<kDLGPU, int64_t>(CSRMatrix, int64_t);

template <typename IdType>
__global__ void _CSRGetRowNNZKernel(
    const IdType* vid,
    const IdType* indptr,
    IdType* out,
    int64_t length) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride_x = gridDim.x * blockDim.x;
  while (tx < length) {
    const IdType vv = vid[tx];
    out[tx] = indptr[vv + 1] - indptr[vv];
    tx += stride_x;
  }
}

template <DLDeviceType XPU, typename IdType>
NDArray CSRGetRowNNZ(CSRMatrix csr, NDArray rows) {
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  const auto len = rows->shape[0];
  const IdType* vid_data = static_cast<IdType*>(rows->data);
  const IdType* indptr_data = static_cast<IdType*>(csr.indptr->data);
  NDArray rst = NDArray::Empty({len}, rows->dtype, rows->ctx);
  IdType* rst_data = static_cast<IdType*>(rst->data);
  const int nt = cuda::FindNumThreads(len);
  const int nb = (len + nt - 1) / nt;
  CUDA_KERNEL_CALL(_CSRGetRowNNZKernel,
      nb, nt, 0, thr_entry->stream,
      vid_data, indptr_data, rst_data, len);
  return rst;
}

template NDArray CSRGetRowNNZ<kDLGPU, int32_t>(CSRMatrix, NDArray);
template NDArray CSRGetRowNNZ<kDLGPU, int64_t>(CSRMatrix, NDArray);

///////////////////////////// CSRGetRowColumnIndices /////////////////////////////

template <DLDeviceType XPU, typename IdType>
NDArray CSRGetRowColumnIndices(CSRMatrix csr, int64_t row) {
  const int64_t len = impl::CSRGetRowNNZ<XPU, IdType>(csr, row);
  const int64_t offset = aten::IndexSelect<IdType>(csr.indptr, row) * sizeof(IdType);
  return csr.indices.CreateView({len}, csr.indices->dtype, offset);
}

template NDArray CSRGetRowColumnIndices<kDLGPU, int32_t>(CSRMatrix, int64_t);
template NDArray CSRGetRowColumnIndices<kDLGPU, int64_t>(CSRMatrix, int64_t);

///////////////////////////// CSRGetRowData /////////////////////////////

template <DLDeviceType XPU, typename IdType>
NDArray CSRGetRowData(CSRMatrix csr, int64_t row) {
  const int64_t len = impl::CSRGetRowNNZ<XPU, IdType>(csr, row);
  const int64_t offset = aten::IndexSelect<IdType>(csr.indptr, row) * sizeof(IdType);
  if (aten::CSRHasData(csr))
    return csr.data.CreateView({len}, csr.data->dtype, offset);
  else
    return aten::Range(offset, offset + len, csr.indptr->dtype.bits, csr.indptr->ctx);
}

template NDArray CSRGetRowData<kDLGPU, int32_t>(CSRMatrix, int64_t);
template NDArray CSRGetRowData<kDLGPU, int64_t>(CSRMatrix, int64_t);

///////////////////////////// CSRSliceRows /////////////////////////////

template <DLDeviceType XPU, typename IdType>
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
    ret_data = csr.data.CreateView({nnz}, csr.data->dtype, st_pos * sizeof(IdType));
  else
    ret_data = aten::Range(st_pos, ed_pos,
                           csr.indptr->dtype.bits, csr.indptr->ctx);
  return CSRMatrix(num_rows, csr.num_cols,
                   ret_indptr, ret_indices, ret_data,
                   csr.sorted);
}

template CSRMatrix CSRSliceRows<kDLGPU, int32_t>(CSRMatrix, int64_t, int64_t);
template CSRMatrix CSRSliceRows<kDLGPU, int64_t>(CSRMatrix, int64_t, int64_t);

/*!
 * \brief Copy data segment to output buffers
 * 
 * For the i^th row r = row[i], copy the data from indptr[r] ~ indptr[r+1]
 * to the out_data from out_indptr[i] ~ out_indptr[i+1]
 *
 * If the provided `data` array is nullptr, write the read index to the out_data.
 *
 */
template <typename IdType, typename DType>
__global__ void _SegmentCopyKernel(
    const IdType* indptr, const DType* data,
    const IdType* row, int64_t row_stride, int64_t length,
    const IdType* out_indptr, DType* out_data) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride_x = gridDim.x * blockDim.x;
  while (tx < length) {
    int rpos = tx * row_stride;
    const IdType r = row[rpos];
    DType* out_buf = out_data + out_indptr[tx];
    for (IdType i = indptr[r]; i < indptr[r + 1]; ++i) {
      *(out_buf++) = data? data[i] : i;
    }
    tx += stride_x;
  }
}

template <DLDeviceType XPU, typename IdType>
CSRMatrix CSRSliceRows(CSRMatrix csr, NDArray rows) {
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  const int64_t len = rows->shape[0];
  IdArray ret_indptr = aten::CumSum(aten::CSRGetRowNNZ(csr, rows), true);
  const int64_t nnz = aten::IndexSelect<IdType>(ret_indptr, len);

  const int nt = cuda::FindNumThreads(len);
  const int nb = (len + nt - 1) / nt;
  // Copy indices.
  IdArray ret_indices = NDArray::Empty({nnz}, csr.indptr->dtype, csr.indptr->ctx);
  CUDA_KERNEL_CALL(_SegmentCopyKernel,
      nb, nt, 0, thr_entry->stream,
      csr.indptr.Ptr<IdType>(), csr.indices.Ptr<IdType>(),
      rows.Ptr<IdType>(), 1, len,
      ret_indptr.Ptr<IdType>(), ret_indices.Ptr<IdType>());
  // Copy data.
  IdArray ret_data = NDArray::Empty({nnz}, csr.indptr->dtype, csr.indptr->ctx);
  CUDA_KERNEL_CALL(_SegmentCopyKernel,
      nb, nt, 0, thr_entry->stream,
      csr.indptr.Ptr<IdType>(), CSRHasData(csr)? csr.data.Ptr<IdType>() : nullptr,
      rows.Ptr<IdType>(), 1, len,
      ret_indptr.Ptr<IdType>(), ret_data.Ptr<IdType>());
  return CSRMatrix(len, csr.num_cols,
                   ret_indptr, ret_indices, ret_data,
                   csr.sorted);
}

template CSRMatrix CSRSliceRows<kDLGPU, int32_t>(CSRMatrix , NDArray);
template CSRMatrix CSRSliceRows<kDLGPU, int64_t>(CSRMatrix , NDArray);

///////////////////////////// CSRGetData /////////////////////////////

template <DLDeviceType XPU, typename IdType>
IdArray CSRGetData(CSRMatrix csr, NDArray row, NDArray col) {
  const int64_t rowlen = row->shape[0];
  const int64_t collen = col->shape[0];

  CHECK((rowlen == collen) || (rowlen == 1) || (collen == 1))
    << "Invalid row and col id array.";

  const int64_t row_stride = (rowlen == 1 && collen != 1) ? 0 : 1;
  const int64_t col_stride = (collen == 1 && rowlen != 1) ? 0 : 1;

  const int64_t rstlen = std::max(rowlen, collen);
  IdArray rst = NDArray::Empty({rstlen}, row->dtype, row->ctx);
  if (rstlen == 0)
    return rst;

  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  const int nt = cuda::FindNumThreads(rstlen);
  const int nb = (rstlen + nt - 1) / nt;
  // TODO(minjie): use binary search for sorted csr
  CUDA_KERNEL_CALL(_LinearSearchKernel,
      nb, nt, 0, thr_entry->stream,
      csr.indptr.Ptr<IdType>(), csr.indices.Ptr<IdType>(),
      CSRHasData(csr)? csr.data.Ptr<IdType>() : nullptr,
      row.Ptr<IdType>(), col.Ptr<IdType>(),
      row_stride, col_stride, rstlen,
      rst.Ptr<IdType>());
  return rst;
}

template NDArray CSRGetData<kDLGPU, int32_t>(CSRMatrix csr, NDArray rows, NDArray cols);
template NDArray CSRGetData<kDLGPU, int64_t>(CSRMatrix csr, NDArray rows, NDArray cols);

///////////////////////////// CSRGetDataAndIndices /////////////////////////////

/*!
 * \brief Generate a 0-1 mask for each index that hits the provided (row, col)
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
    const IdType* indptr, const IdType* indices,
    const IdType* row, const IdType* col,
    int64_t row_stride, int64_t col_stride,
    int64_t length, IdType* mask) {
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

/*!
 * \brief Search for the insertion positions for needle in the hay.
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
    const IdType* hay, int64_t hay_size,
    const IdType* needles, int64_t num_needles,
    IdType* pos) {
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
    pos[tx] = (hay[hi] == ele)? hi : hi - 1;
    tx += stride_x;
  }
}

template <DLDeviceType XPU, typename IdType>
std::vector<NDArray> CSRGetDataAndIndices(CSRMatrix csr, NDArray row, NDArray col) {
  const auto rowlen = row->shape[0];
  const auto collen = col->shape[0];
  const auto len = std::max(rowlen, collen);
  if (len == 0)
    return {NullArray(), NullArray(), NullArray()};

  const auto& ctx = row->ctx;
  const auto nbits = row->dtype.bits;
  const int64_t nnz = csr.indices->shape[0];
  const int64_t row_stride = (rowlen == 1 && collen != 1) ? 0 : 1;
  const int64_t col_stride = (collen == 1 && rowlen != 1) ? 0 : 1;
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();

  // Generate a 0-1 mask for matched (row, col) positions.
  IdArray mask = Full(0, nnz, nbits, ctx);
  const int nt = cuda::FindNumThreads(len);
  const int nb = (len + nt - 1) / nt;
  CUDA_KERNEL_CALL(_SegmentMaskKernel,
      nb, nt, 0, thr_entry->stream,
      csr.indptr.Ptr<IdType>(), csr.indices.Ptr<IdType>(),
      row.Ptr<IdType>(), col.Ptr<IdType>(),
      row_stride, col_stride, len,
      mask.Ptr<IdType>());

  IdArray idx = AsNumBits(NonZero(mask), nbits);
  if (idx->shape[0] == 0)
    // No data. Return three empty arrays.
    return {idx, idx, idx};

  // Search for row index
  IdArray ret_row = NewIdArray(idx->shape[0], ctx, nbits);
  const int nt2 = cuda::FindNumThreads(idx->shape[0]);
  const int nb2 = (idx->shape[0] + nt - 1) / nt;
  CUDA_KERNEL_CALL(_SortedSearchKernel,
      nb2, nt2, 0, thr_entry->stream,
      csr.indptr.Ptr<IdType>(), csr.num_rows,
      idx.Ptr<IdType>(), idx->shape[0],
      ret_row.Ptr<IdType>());

  // Column & data can be obtained by index select.
  IdArray ret_col = IndexSelect(csr.indices, idx);
  IdArray ret_data = CSRHasData(csr)? IndexSelect(csr.data, idx) : idx;
  return {ret_row, ret_col, ret_data};
}

template std::vector<NDArray> CSRGetDataAndIndices<kDLGPU, int32_t>(
    CSRMatrix csr, NDArray rows, NDArray cols);
template std::vector<NDArray> CSRGetDataAndIndices<kDLGPU, int64_t>(
    CSRMatrix csr, NDArray rows, NDArray cols);

///////////////////////////// CSRSliceMatrix /////////////////////////////

/*!
 * \brief Generate a 0-1 mask for each index whose column is in the provided set.
 *        It also counts the number of masked values per row.
 */
template <typename IdType>
__global__ void _SegmentMaskColKernel(
    const IdType* indptr, const IdType* indices, int64_t num_rows,
    const IdType* col, int64_t col_len,
    IdType* mask, IdType* count) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride_x = gridDim.x * blockDim.x;
  // TODO(minjie): consider putting the col array in shared memory.
  while (tx < num_rows) {
    IdType cnt = 0;
    for (IdType i = indptr[tx]; i < indptr[tx + 1]; ++i) {
      const IdType cur_c = indices[i];
      for (int64_t j = 0; j < col_len; ++j) {
        if (cur_c == col[j]) {
          mask[i] = 1;
          ++cnt;
          break;
        }
      }
    }
    count[tx] = cnt;
    tx += stride_x;
  }
}

template <DLDeviceType XPU, typename IdType>
CSRMatrix CSRSliceMatrix(CSRMatrix csr, runtime::NDArray rows, runtime::NDArray cols) {
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  const auto& ctx = rows->ctx;
  const auto& dtype = rows->dtype;
  const auto nbits = dtype.bits;
  const int64_t new_nrows = rows->shape[0];
  const int64_t new_ncols = cols->shape[0];

  if (new_nrows == 0 || new_ncols == 0)
    return CSRMatrix(new_nrows, new_ncols,
                     Full(0, new_nrows + 1, nbits, ctx),
                     NullArray(dtype, ctx), NullArray(dtype, ctx));

  // First slice rows
  csr = CSRSliceRows(csr, rows);

  if (csr.indices->shape[0] == 0)
    return CSRMatrix(new_nrows, new_ncols,
                     Full(0, new_nrows + 1, nbits, ctx),
                     NullArray(dtype, ctx), NullArray(dtype, ctx));

  // Generate a 0-1 mask for matched (row, col) positions.
  IdArray mask = Full(0, csr.indices->shape[0], nbits, ctx);
  // A count for how many masked values per row.
  IdArray count = NewIdArray(csr.num_rows, ctx, nbits);
  const int nt = cuda::FindNumThreads(csr.num_rows);
  const int nb = (csr.num_rows + nt - 1) / nt;
  CUDA_KERNEL_CALL(_SegmentMaskColKernel,
      nb, nt, 0, thr_entry->stream,
      csr.indptr.Ptr<IdType>(), csr.indices.Ptr<IdType>(), csr.num_rows,
      cols.Ptr<IdType>(), cols->shape[0],
      mask.Ptr<IdType>(), count.Ptr<IdType>());

  IdArray idx = AsNumBits(NonZero(mask), nbits);
  if (idx->shape[0] == 0)
    return CSRMatrix(new_nrows, new_ncols,
                     Full(0, new_nrows + 1, nbits, ctx),
                     NullArray(dtype, ctx), NullArray(dtype, ctx));

  // Indptr needs to be adjusted according to the new nnz per row.
  IdArray ret_indptr = CumSum(count, true);

  // Column & data can be obtained by index select.
  IdArray ret_col = IndexSelect(csr.indices, idx);
  IdArray ret_data = CSRHasData(csr)? IndexSelect(csr.data, idx) : idx;

  // Relabel column
  IdArray col_hash = NewIdArray(csr.num_cols, ctx, nbits);
  Scatter_(cols, Range(0, cols->shape[0], nbits, ctx), col_hash);
  ret_col = IndexSelect(col_hash, ret_col);

  return CSRMatrix(new_nrows, new_ncols, ret_indptr,
                   ret_col, ret_data);
}

template CSRMatrix CSRSliceMatrix<kDLGPU, int32_t>(
    CSRMatrix csr, runtime::NDArray rows, runtime::NDArray cols);
template CSRMatrix CSRSliceMatrix<kDLGPU, int64_t>(
    CSRMatrix csr, runtime::NDArray rows, runtime::NDArray cols);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
