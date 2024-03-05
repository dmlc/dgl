/**
 *  Copyright (c) 2020 by Contributors
 * @file array/cuda/csr2coo.cc
 * @brief CSR2COO
 */
#include <dgl/array.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <cub/cub.cuh>

#include "../../runtime/cuda/cuda_common.h"
#include "./utils.h"

namespace dgl {

using runtime::NDArray;

namespace aten {
namespace impl {

template <DGLDeviceType XPU, typename IdType>
COOMatrix CSRToCOO(CSRMatrix csr) {
  LOG(FATAL) << "Unreachable codes";
  return {};
}

template <>
COOMatrix CSRToCOO<kDGLCUDA, int32_t>(CSRMatrix csr) {
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  cudaStream_t stream = runtime::getCurrentCUDAStream();
  // allocate cusparse handle if needed
  if (!thr_entry->cusparse_handle) {
    CUSPARSE_CALL(cusparseCreate(&(thr_entry->cusparse_handle)));
  }
  CUSPARSE_CALL(cusparseSetStream(thr_entry->cusparse_handle, stream));

  NDArray indptr = csr.indptr, indices = csr.indices, data = csr.data;
  const int32_t* indptr_ptr = static_cast<int32_t*>(indptr->data);
  NDArray row =
      aten::NewIdArray(indices->shape[0], indptr->ctx, indptr->dtype.bits);
  int32_t* row_ptr = static_cast<int32_t*>(row->data);

  CUSPARSE_CALL(cusparseXcsr2coo(
      thr_entry->cusparse_handle, indptr_ptr, indices->shape[0], csr.num_rows,
      row_ptr, CUSPARSE_INDEX_BASE_ZERO));

  return COOMatrix(
      csr.num_rows, csr.num_cols, row, indices, data, true, csr.sorted);
}

struct RepeatIndex {
  template <typename IdType>
  __host__ __device__ auto operator()(IdType i) {
    return thrust::make_constant_iterator(i);
  }
};

template <typename IdType>
struct OutputBufferIndexer {
  const IdType* indptr;
  IdType* buffer;
  __host__ __device__ auto operator()(IdType i) { return buffer + indptr[i]; }
};

template <typename IdType>
struct AdjacentDifference {
  const IdType* indptr;
  __host__ __device__ auto operator()(IdType i) {
    return indptr[i + 1] - indptr[i];
  }
};

template <>
COOMatrix CSRToCOO<kDGLCUDA, int64_t>(CSRMatrix csr) {
  const auto& ctx = csr.indptr->ctx;
  cudaStream_t stream = runtime::getCurrentCUDAStream();

  const int64_t nnz = csr.indices->shape[0];
  const auto nbits = csr.indptr->dtype.bits;
  IdArray ret_row = NewIdArray(nnz, ctx, nbits);

  runtime::CUDAWorkspaceAllocator allocator(csr.indptr->ctx);
  thrust::counting_iterator<int64_t> iota(0);

  auto input_buffer = thrust::make_transform_iterator(iota, RepeatIndex{});
  auto output_buffer = thrust::make_transform_iterator(
      iota, OutputBufferIndexer<int64_t>{
                csr.indptr.Ptr<int64_t>(), ret_row.Ptr<int64_t>()});
  auto buffer_sizes = thrust::make_transform_iterator(
      iota, AdjacentDifference<int64_t>{csr.indptr.Ptr<int64_t>()});

  constexpr int64_t max_copy_at_once = std::numeric_limits<int32_t>::max();
  for (int64_t i = 0; i < csr.num_rows; i += max_copy_at_once) {
    std::size_t temp_storage_bytes = 0;
    CUDA_CALL(cub::DeviceCopy::Batched(
        nullptr, temp_storage_bytes, input_buffer + i, output_buffer + i,
        buffer_sizes + i, std::min(csr.num_rows - i, max_copy_at_once),
        stream));

    auto temp = allocator.alloc_unique<char>(temp_storage_bytes);

    CUDA_CALL(cub::DeviceCopy::Batched(
        temp.get(), temp_storage_bytes, input_buffer + i, output_buffer + i,
        buffer_sizes + i, std::min(csr.num_rows - i, max_copy_at_once),
        stream));
  }

  return COOMatrix(
      csr.num_rows, csr.num_cols, ret_row, csr.indices, csr.data, true,
      csr.sorted);
}

template COOMatrix CSRToCOO<kDGLCUDA, int32_t>(CSRMatrix csr);
template COOMatrix CSRToCOO<kDGLCUDA, int64_t>(CSRMatrix csr);

template <DGLDeviceType XPU, typename IdType>
COOMatrix CSRToCOODataAsOrder(CSRMatrix csr) {
  LOG(FATAL) << "Unreachable codes";
  return {};
}

template <>
COOMatrix CSRToCOODataAsOrder<kDGLCUDA, int32_t>(CSRMatrix csr) {
  COOMatrix coo = CSRToCOO<kDGLCUDA, int32_t>(csr);
  if (aten::IsNullArray(coo.data)) return coo;

  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  auto device = runtime::DeviceAPI::Get(coo.row->ctx);
  cudaStream_t stream = runtime::getCurrentCUDAStream();
  // allocate cusparse handle if needed
  if (!thr_entry->cusparse_handle) {
    CUSPARSE_CALL(cusparseCreate(&(thr_entry->cusparse_handle)));
  }
  CUSPARSE_CALL(cusparseSetStream(thr_entry->cusparse_handle, stream));

  NDArray row = coo.row, col = coo.col, data = coo.data;
  int32_t* row_ptr = static_cast<int32_t*>(row->data);
  int32_t* col_ptr = static_cast<int32_t*>(col->data);
  int32_t* data_ptr = static_cast<int32_t*>(data->data);

  size_t workspace_size = 0;
  CUSPARSE_CALL(cusparseXcoosort_bufferSizeExt(
      thr_entry->cusparse_handle, coo.num_rows, coo.num_cols, row->shape[0],
      data_ptr, row_ptr, &workspace_size));
  void* workspace = device->AllocWorkspace(row->ctx, workspace_size);
  CUSPARSE_CALL(cusparseXcoosortByRow(
      thr_entry->cusparse_handle, coo.num_rows, coo.num_cols, row->shape[0],
      data_ptr, row_ptr, col_ptr, workspace));
  device->FreeWorkspace(row->ctx, workspace);

  // The row and column field have already been reordered according
  // to data, thus the data field will be deprecated.
  coo.data = aten::NullArray();
  coo.row_sorted = false;
  coo.col_sorted = false;
  return coo;
}

template <>
COOMatrix CSRToCOODataAsOrder<kDGLCUDA, int64_t>(CSRMatrix csr) {
  COOMatrix coo = CSRToCOO<kDGLCUDA, int64_t>(csr);
  if (aten::IsNullArray(coo.data)) return coo;
  const auto& sorted = Sort(coo.data);

  coo.row = IndexSelect(coo.row, sorted.second);
  coo.col = IndexSelect(coo.col, sorted.second);

  // The row and column field have already been reordered according
  // to data, thus the data field will be deprecated.
  coo.data = aten::NullArray();
  coo.row_sorted = false;
  coo.col_sorted = false;
  return coo;
}

template COOMatrix CSRToCOODataAsOrder<kDGLCUDA, int32_t>(CSRMatrix csr);
template COOMatrix CSRToCOODataAsOrder<kDGLCUDA, int64_t>(CSRMatrix csr);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
