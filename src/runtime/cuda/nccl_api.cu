/*!
 *  Copyright (c) 2021-2022 by Contributors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 * \file nccl_api.cu
 * \brief Implementation of wrapper around NCCL routines.
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <dgl/array.h>
#include <dgl/aten/array_ops.h>
#include <dgl/packed_func_ext.h>
#include <dgl/runtime/container.h>
#include <dgl/runtime/device_api.h>
#include <dgl/runtime/registry.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "../../array/cuda/array_index_select.cuh"
#include "../../array/cuda/dgl_cub.cuh"
#include "../../partition/ndarray_partition.h"
#include "../../runtime/workspace.h"
#include "cuda_common.h"
#include "nccl_api.h"

#define NCCL_CALL(func)                                                   \
  {                                                                       \
    ncclResult_t result = func;                                           \
    if (result != ncclSuccess) {                                          \
      LOG(FATAL) << "NCCLError: " #func " failed with error: " << result; \
    }                                                                     \
  }

namespace dgl {

using namespace partition;

namespace runtime {
namespace cuda {

namespace {

#ifdef DGL_USE_NCCL

template <typename T>
ncclDataType_t NCCLType();
template <>
ncclDataType_t NCCLType<int32_t>() {
  return ncclInt32;
}
template <>
ncclDataType_t NCCLType<int64_t>() {
  return ncclInt64;
}
template <>
ncclDataType_t NCCLType<__half>() {
  return ncclHalf;
}
template <>
ncclDataType_t NCCLType<float>() {
  return ncclFloat32;
}
template <>
ncclDataType_t NCCLType<double>() {
  return ncclFloat64;
}

#endif  // DGL_USE_NCCL

template <typename IdType, typename DType>
__global__ void _DualPermKernel(
    const IdType* const in_idx, const DType* const in_value,
    const IdType* const perm, const int64_t num_in, const int64_t num_feat,
    IdType* const out_idx, DType* const out_value) {
  // set index permutation
  const int64_t tidx =
      blockDim.x * static_cast<int64_t>(blockIdx.x) + threadIdx.x;
  if (tidx < num_in) {
    const IdType perm_idx = perm[tidx];
    assert(perm_idx < num_in);
    out_idx[tidx] = in_idx[perm_idx];
  }

  if (num_feat > 1) {
    for (int d = 0; d < blockDim.x; ++d) {
      const int64_t bidx = blockDim.x * static_cast<int64_t>(blockIdx.x) + d;
      if (bidx < num_in) {
        const IdType perm_idx = perm[bidx];
        for (int64_t f = threadIdx.x; f < num_feat; f += blockDim.x) {
          out_value[bidx * num_feat + f] = in_value[perm_idx * num_feat + f];
        }
      }
    }
  } else {
    if (tidx < num_in) {
      const IdType perm_idx = perm[tidx];
      out_value[tidx] = in_value[perm_idx];
    }
  }
}

template <typename DType, typename IdType>
__global__ void _InversePermKernel(
    const DType* const array, const int64_t num_feat, int64_t length,
    const IdType* const perm, DType* const out) {
  int64_t in_row = blockIdx.x * blockDim.y + threadIdx.y;

  const int64_t stride = blockDim.y * gridDim.x;

  while (in_row < length) {
    int64_t col = threadIdx.x;
    const int64_t out_row = perm[in_row];
    while (col < num_feat) {
      out[out_row * num_feat + col] = array[in_row * num_feat + col];
      col += blockDim.x;
    }
    in_row += stride;
  }
}

template <typename IdType, typename DType>
std::pair<IdArray, NDArray> SparsePush(
    NCCLCommunicatorRef comm, IdArray in_idx, NDArray in_value,
    NDArrayPartitionRef part) {
  const auto& ctx = in_idx->ctx;
  CHECK_EQ(ctx, in_value->ctx) << "Indices and values must be on the same "
                                  "device";
  auto device = DeviceAPI::Get(ctx);

  cudaStream_t stream = runtime::getCurrentCUDAStream();

  CHECK_LE(in_idx->ndim, 1) << "The tensor of sending indices must be of "
                               "dimension one (or empty).";
  const int64_t num_in = in_idx->ndim > 0 ? in_idx->shape[0] : 0;

  CHECK_EQ(num_in, in_value->ndim > 0 ? in_value->shape[0] : 0)
      << "Leading dimension of indices (" << num_in
      << ") must match "
         "leading dimension of values ("
      << (in_value->ndim > 0 ? in_value->shape[0] : 0) << ").";

  int64_t num_feat = 1;
  for (int d = 1; d < in_value->ndim; ++d) {
    num_feat *= in_value->shape[d];
  }

  const int64_t comm_size = comm->size();

  if (comm_size == 1) {
    // nothing to do, just return original arrays
    return std::pair<IdArray, NDArray>(in_idx, in_value);
  }

  std::pair<IdArray, NDArray> part_perm = part->GeneratePermutation(in_idx);
  const IdType* const perm = static_cast<const IdType*>(part_perm.first->data);
  const int64_t* const send_sum =
      static_cast<const int64_t*>(part_perm.second->data);

  Workspace<IdType> send_idx(device, ctx, num_in);
  Workspace<DType> send_value(device, ctx, num_in * num_feat);

  // permute the indices and values
  if (num_in > 0) {
    const dim3 block(256);
    const dim3 grid((num_in + block.x - 1) / block.x);

    CUDA_KERNEL_CALL(
        _DualPermKernel, grid, block, 0, stream,
        static_cast<const IdType*>(in_idx->data),
        static_cast<const DType*>(in_value->data), perm, num_in, num_feat,
        send_idx.get(), send_value.get());
  }

  // compute the prefix sum of the send values
  Workspace<int64_t> send_prefix(device, ctx, comm_size + 1);
  {
    size_t prefix_workspace_size;
    CUDA_CALL(cub::DeviceScan::ExclusiveSum(
        nullptr, prefix_workspace_size, send_sum, send_prefix.get(),
        comm_size + 1, stream));

    Workspace<void> prefix_workspace(device, ctx, prefix_workspace_size);
    CUDA_CALL(cub::DeviceScan::ExclusiveSum(
        prefix_workspace.get(), prefix_workspace_size, send_sum,
        send_prefix.get(), comm_size + 1, stream));
  }

  std::vector<int64_t> send_prefix_host(comm_size + 1);
  // copy using the same stream (local current stream), no need to sync
  device->CopyDataFromTo(
      send_prefix.get(), 0, send_prefix_host.data(), 0,
      send_prefix_host.size() * sizeof(*send_prefix.get()), ctx,
      DGLContext{kDGLCPU, 0},
      DGLDataType{kDGLInt, sizeof(*send_prefix.get()) * 8, 1});
  send_prefix.free();

  CHECK_EQ(send_prefix_host.back(), num_in)
      << "Internal Error: "
         "send_prefix_host.back() = "
      << send_prefix_host.back() << ", and num_in = " << num_in;

  // communicate the amount to send
  Workspace<int64_t> recv_sum(device, ctx, comm_size + 1);
  comm->AllToAll(send_sum, recv_sum.get(), 1, stream);

  cudaEvent_t d2h;
  CUDA_CALL(cudaEventCreate(&d2h));

  // compute the prefix sum of the recv values
  Workspace<int64_t> recv_prefix(device, ctx, comm_size + 1);
  {
    size_t prefix_workspace_size;
    CUDA_CALL(cub::DeviceScan::ExclusiveSum(
        nullptr, prefix_workspace_size, recv_sum.get(), recv_prefix.get(),
        comm_size + 1, stream));

    Workspace<void> prefix_workspace(device, ctx, prefix_workspace_size);
    CUDA_CALL(cub::DeviceScan::ExclusiveSum(
        prefix_workspace.get(), prefix_workspace_size, recv_sum.get(),
        recv_prefix.get(), comm_size + 1, stream));
  }
  recv_sum.free();

  // finally copy the prefixsum sum down to the host
  std::vector<int64_t> recv_prefix_host(comm_size + 1);
  // copy using the same stream (local current stream), no need to sync
  device->CopyDataFromTo(
      recv_prefix.get(), 0, recv_prefix_host.data(), 0,
      recv_prefix_host.size() * sizeof(*recv_prefix.get()), ctx,
      DGLContext{kDGLCPU, 0},
      DGLDataType{kDGLInt, sizeof(*recv_prefix.get()) * 8, 1});
  recv_prefix.free();

  // use an event to track when copying is done
  CUDA_CALL(cudaEventRecord(d2h, stream));

  // allocate output space
  CUDA_CALL(cudaEventSynchronize(d2h));
  CUDA_CALL(cudaEventDestroy(d2h));

  IdArray recv_idx =
      aten::NewIdArray(recv_prefix_host.back(), ctx, sizeof(IdType) * 8);

  std::vector<int64_t> value_shape(in_value->ndim, 0);
  value_shape[0] = recv_prefix_host.back();
  for (int d = 1; d < in_value->ndim; ++d) {
    value_shape[d] = in_value->shape[d];
  }
  NDArray recv_value = NDArray::Empty(value_shape, in_value->dtype, ctx);

  // send data
  comm->SparseAllToAll(
      send_idx.get(), send_value.get(), num_feat, send_prefix_host.data(),
      static_cast<IdType*>(recv_idx->data),
      static_cast<DType*>(recv_value->data), recv_prefix_host.data(), stream);

  return std::pair<IdArray, NDArray>(recv_idx, recv_value);
}

template <typename IdType, typename DType>
NDArray SparsePull(
    NCCLCommunicatorRef comm, IdArray req_idx, NDArray local_tensor,
    NDArrayPartitionRef part) {
  const auto& ctx = req_idx->ctx;
  CHECK_EQ(ctx, local_tensor->ctx) << "The request indices and set of local "
                                      "values must be on the same device";
  auto device = DeviceAPI::Get(ctx);

  cudaStream_t stream = runtime::getCurrentCUDAStream();

  CHECK_LE(req_idx->ndim, 1) << "The tensor of requested indices must be of "
                                "dimension one (or empty).";
  const int64_t num_in = req_idx->ndim > 0 ? req_idx->shape[0] : 0;
  int64_t num_feat = 1;
  for (int d = 1; d < local_tensor->ndim; ++d) {
    num_feat *= local_tensor->shape[d];
  }

  const int64_t comm_size = comm->size();

  if (comm_size == 1) {
    // Just return index selection from current local_tensor
    return aten::IndexSelect(local_tensor, req_idx);
  }

  // First we need to send our requests to other processors. This means
  // re-ordering our index array to be contiguous among processors, and
  // counting the number of indices we are sending each processor. For now,
  // we assume a poorly partitioned graph, and that there exists the
  // possibility that each processor could request data from this one.

  // the buffer for us to re-order our requests in
  Workspace<IdType> send_idx(device, ctx, num_in);

  std::pair<IdArray, NDArray> part_perm = part->GeneratePermutation(req_idx);
  const IdType* const perm = static_cast<const IdType*>(part_perm.first->data);
  const int64_t* const send_sum =
      static_cast<const int64_t*>(part_perm.second->data);

  // permute requests
  if (num_in > 0) {
    const dim3 block(256);
    const dim3 grid((num_in + block.x - 1) / block.x);

    CUDA_KERNEL_CALL(
        aten::impl::IndexSelectSingleKernel, grid, block, 0, stream,
        static_cast<const IdType*>(req_idx->data), perm, num_in,
        req_idx->shape[0], send_idx.get());
  }

  // compute the prefix sum of the indexes this process is requesting
  Workspace<int64_t> request_prefix(device, ctx, comm_size + 1);
  {
    size_t prefix_workspace_size;
    CUDA_CALL(cub::DeviceScan::ExclusiveSum(
        nullptr, prefix_workspace_size, send_sum, request_prefix.get(),
        comm_size + 1, stream));

    Workspace<void> prefix_workspace(device, ctx, prefix_workspace_size);
    CUDA_CALL(cub::DeviceScan::ExclusiveSum(
        prefix_workspace.get(), prefix_workspace_size, send_sum,
        request_prefix.get(), comm_size + 1, stream));
  }

  cudaEvent_t d2h;
  CUDA_CALL(cudaEventCreate(&d2h));

  std::vector<int64_t> request_prefix_host(comm_size + 1);
  // copy using the same stream (local current stream), no need to sync
  device->CopyDataFromTo(
      request_prefix.get(), 0, request_prefix_host.data(), 0,
      request_prefix_host.size() * sizeof(*request_prefix.get()), ctx,
      DGLContext{kDGLCPU, 0},
      DGLDataType{kDGLInt, sizeof(*request_prefix.get()) * 8, 1});
  request_prefix.free();
  CHECK_EQ(request_prefix_host.back(), num_in)
      << "Internal Error: "
         "request_prefix_host.back() = "
      << request_prefix_host.back() << ", num_in = " << num_in;

  // communicate the amount requested
  Workspace<int64_t> recv_sum(device, ctx, comm_size + 1);
  comm->AllToAll(send_sum, recv_sum.get(), 1, stream);

  // compute the prefix sum of the requested indexes
  Workspace<int64_t> response_prefix(device, ctx, comm_size + 1);
  {
    size_t prefix_workspace_size;
    CUDA_CALL(cub::DeviceScan::ExclusiveSum(
        nullptr, prefix_workspace_size, recv_sum.get(), response_prefix.get(),
        comm_size + 1, stream));

    Workspace<void> prefix_workspace(device, ctx, prefix_workspace_size);
    CUDA_CALL(cub::DeviceScan::ExclusiveSum(
        prefix_workspace.get(), prefix_workspace_size, recv_sum.get(),
        response_prefix.get(), comm_size + 1, stream));
  }
  recv_sum.free();

  // finally copy the prefixsum sum down to the host
  std::vector<int64_t> response_prefix_host(comm_size + 1);
  // copy using the same stream (local current stream), no need to sync
  device->CopyDataFromTo(
      response_prefix.get(), 0, response_prefix_host.data(), 0,
      response_prefix_host.size() * sizeof(*response_prefix.get()), ctx,
      DGLContext{kDGLCPU, 0},
      DGLDataType{kDGLInt, sizeof(*response_prefix.get()) * 8, 1});
  response_prefix.free();

  // use an event to track when copying is done
  CUDA_CALL(cudaEventRecord(d2h, stream));

  // allocate output space
  CUDA_CALL(cudaEventSynchronize(d2h));
  CUDA_CALL(cudaEventDestroy(d2h));

  // gather requested indexes
  IdArray recv_idx =
      aten::NewIdArray(response_prefix_host.back(), ctx, sizeof(IdType) * 8);
  comm->AllToAllV(
      send_idx.get(), request_prefix_host.data(),
      static_cast<IdType*>(recv_idx->data), response_prefix_host.data(),
      stream);
  send_idx.free();

  // convert requested indices to local indices depending on partition
  if (response_prefix_host.back() > 0) {
    recv_idx = part->MapToLocal(recv_idx);
  }

  // and then index select them into place
  Workspace<DType> filled_response_value(
      device, ctx, response_prefix_host.back() * num_feat);
  if (response_prefix_host.back() > 0) {
    dim3 block(256, 1);
    while (block.x >= 2 * num_feat) {
      block.x /= 2;
      block.y *= 2;
    }
    const dim3 grid((response_prefix_host.back() + block.y - 1) / block.y);

    CUDA_KERNEL_CALL(
        aten::impl::IndexSelectMultiKernel, grid, block, 0, stream,
        static_cast<const DType*>(local_tensor->data), num_feat,
        static_cast<IdType*>(recv_idx->data), response_prefix_host.back(),
        local_tensor->shape[0], filled_response_value.get());
  }

  // we will collect recieved values in this array
  std::vector<int64_t> value_shape(local_tensor->ndim, 0);
  value_shape[0] = request_prefix_host.back();
  for (int d = 1; d < local_tensor->ndim; ++d) {
    value_shape[d] = local_tensor->shape[d];
  }
  Workspace<DType> filled_request_value(
      device, ctx, request_prefix_host.back() * num_feat);

  // multiply the prefixes by the number of features being sent
  for (auto& v : request_prefix_host) {
    v *= num_feat;
  }
  for (auto& v : response_prefix_host) {
    v *= num_feat;
  }

  // send the values
  comm->AllToAllV(
      filled_response_value.get(), response_prefix_host.data(),
      filled_request_value.get(), request_prefix_host.data(), stream);
  filled_response_value.free();

  // finally, we need to permute the values back into the requested order
  NDArray result = NDArray::Empty(value_shape, local_tensor->dtype, ctx);
  if (num_in > 0) {
    dim3 block(256, 1);
    while (block.x >= 2 * num_feat) {
      block.x /= 2;
      block.y *= 2;
    }
    const dim3 grid((num_in + block.y - 1) / block.y);

    CUDA_KERNEL_CALL(
        _InversePermKernel, grid, block, 0, stream, filled_request_value.get(),
        num_feat, num_in, perm, static_cast<DType*>(result->data));
  }

  return result;
}

}  // namespace

/* NCCLUniqueId **************************************************************/

NCCLUniqueId::NCCLUniqueId() : id_() {
#ifdef DGL_USE_NCCL
  // this ID is unique to the process, not to each call of this function
  NCCL_CALL(ncclGetUniqueId(&id_));
#else
  // when NCCL isn't enabled, use all zeros
  std::fill(
      id_.internal, id_.internal + NCCL_UNIQUE_ID_BYTES, static_cast<char>(0));
#endif
}

ncclUniqueId NCCLUniqueId::Get() const { return id_; }

std::string NCCLUniqueId::ToString() const {
  std::ostringstream oss;

  oss << std::hex;

  for (size_t b = 0; b < NCCL_UNIQUE_ID_BYTES; ++b) {
    const int num = static_cast<uint8_t>(id_.internal[b]);
    oss << std::setw(2) << std::setfill('0') << num;
  }

  std::string result = oss.str();
  CHECK_EQ(result.length(), NCCL_UNIQUE_ID_BYTES * 2)
      << "Invalid NCCL ID format: '" << result << "'";

  return result;
}

void NCCLUniqueId::FromString(const std::string& str) {
  // must be exactly 256 hex characters
  CHECK_EQ(str.length(), NCCL_UNIQUE_ID_BYTES * 2)
      << "Invalid NCCL ID format: '" << str << "'";

  for (size_t b = 0; b < NCCL_UNIQUE_ID_BYTES; ++b) {
    id_.internal[b] = std::strtol(str.substr(b * 2, 2).c_str(), nullptr, 16);
  }
}

/* NCCLCommunicator **********************************************************/

NCCLCommunicator::NCCLCommunicator(
    const int size, const int rank, ncclUniqueId id)
    : comm_(), size_(size), rank_(rank) {
  CHECK_LT(rank, size) << "The rank (" << rank
                       << ") must be smaller than "
                          "the size of the communicator ("
                       << size << ").";
  CHECK_GE(rank, 0) << "The rank (" << rank
                    << ") must be greater than or "
                       "equal to 0.";

#ifdef DGL_USE_NCCL
  NCCL_CALL(ncclCommInitRank(&comm_, size_, id, rank_));
#else
  CHECK_EQ(size, 1)
      << "Cannot create a communicator of size " << size
      << ". "
         "To use a communicator size greater than 1, compile DGL with NCCL "
         "support.";
#endif
}

NCCLCommunicator::~NCCLCommunicator() {
#ifdef DGL_USE_NCCL
  ncclCommDestroy(comm_);
#endif
}

ncclComm_t NCCLCommunicator::Get() { return comm_; }

template <typename DType>
void NCCLCommunicator::AllToAllV(
    const DType* const send, const int64_t* const send_prefix,
    DType* const recv, const int64_t* const recv_prefix, cudaStream_t stream) {
#ifdef DGL_USE_NCCL
  const ncclDataType_t type = NCCLType<DType>();

  NCCL_CALL(ncclGroupStart());
  for (int r = 0; r < size_; ++r) {
    const int64_t send_size = send_prefix[r + 1] - send_prefix[r];
    if (send_size > 0) {
      NCCL_CALL(
          ncclSend(send + send_prefix[r], send_size, type, r, comm_, stream));
    }
    const int64_t recv_size = recv_prefix[r + 1] - recv_prefix[r];
    if (recv_size > 0) {
      NCCL_CALL(
          ncclRecv(recv + recv_prefix[r], recv_size, type, r, comm_, stream));
    }
  }
  NCCL_CALL(ncclGroupEnd());
#else
  CHECK_EQ(send_prefix[1] - send_prefix[0], recv_prefix[1] - recv_prefix[0])
      << "Send message size must equal receive message size.";

  int dev_id;
  CUDA_CALL(cudaGetDevice(&dev_id));
  DGLContext ctx{kDGLCUDA, dev_id};

  auto device = runtime::DeviceAPI::Get(ctx);
  auto dtype = DGLDataTypeTraits<DType>::dtype;

  // copy using the same stream (local current stream), no need to sync
  device->CopyDataFromTo(
      send, send_prefix[0], recv, recv_prefix[0],
      sizeof(DType) * send_prefix[1] - send_prefix[0], ctx, ctx, dtype);
#endif
}

template void NCCLCommunicator::AllToAllV<int32_t>(
    const int32_t* const send, const int64_t* send_prefix, int32_t* const recv,
    const int64_t* recv_prefix, cudaStream_t stream);
template void NCCLCommunicator::AllToAllV<int64_t>(
    const int64_t* const send, const int64_t* send_prefix, int64_t* const recv,
    const int64_t* recv_prefix, cudaStream_t stream);
template void NCCLCommunicator::AllToAllV<float>(
    const float* const send, const int64_t* send_prefix, float* const recv,
    const int64_t* recv_prefix, cudaStream_t stream);
template void NCCLCommunicator::AllToAllV<__half>(
    const __half* const send, const int64_t* send_prefix, __half* const recv,
    const int64_t* recv_prefix, cudaStream_t stream);

template <typename IdType>
void NCCLCommunicator::AllToAll(
    const IdType* const send, IdType* const recv, const int64_t count,
    cudaStream_t stream) {
#ifdef DGL_USE_NCCL
  const ncclDataType_t type = NCCLType<IdType>();

  NCCL_CALL(ncclGroupStart());
  for (int r = 0; r < size_; ++r) {
    NCCL_CALL(ncclSend(send + (r * count), count, type, r, comm_, stream));
    NCCL_CALL(ncclRecv(recv + (r * count), count, type, r, comm_, stream));
  }
  NCCL_CALL(ncclGroupEnd());
#else
  int dev_id;
  CUDA_CALL(cudaGetDevice(&dev_id));
  DGLContext ctx{kDGLCUDA, dev_id};

  auto device = runtime::DeviceAPI::Get(ctx);
  auto dtype = DGLDataTypeTraits<IdType>::dtype;

  // copy using the same stream (local current stream), no need to sync
  device->CopyDataFromTo(send, 0, recv, 0, count, ctx, ctx, dtype);
#endif
}

template void NCCLCommunicator::AllToAll<int32_t>(
    const int32_t* const send, int32_t* const recv, const int64_t count,
    cudaStream_t stream);
template void NCCLCommunicator::AllToAll<int64_t>(
    const int64_t* const send, int64_t* const recv, const int64_t count,
    cudaStream_t stream);

template <typename IdType, typename DType>
void NCCLCommunicator::SparseAllToAll(
    const IdType* const send_idx, const DType* const send_value,
    const int64_t num_feat, const int64_t* const send_prefix,
    IdType* const recv_idx, DType* const recv_value,
    const int64_t* const recv_prefix, cudaStream_t stream) {
  // idxs
  AllToAllV(send_idx, send_prefix, recv_idx, recv_prefix, stream);

  // scale prefixes by number of features
  std::vector<int64_t> value_send_prefix(size_ + 1);
  for (int r = 0; r < size_ + 1; ++r) {
    value_send_prefix[r] = send_prefix[r] * num_feat;
  }
  std::vector<int64_t> value_recv_prefix(size_ + 1);
  for (int r = 0; r < size_ + 1; ++r) {
    value_recv_prefix[r] = recv_prefix[r] * num_feat;
  }
  AllToAllV(
      send_value, value_send_prefix.data(), recv_value,
      value_recv_prefix.data(), stream);
}

template void NCCLCommunicator::SparseAllToAll<int32_t, __half>(
    const int32_t* const send_idx, const __half* const send_value,
    const int64_t num_feat, const int64_t* const send_prefix,
    int32_t* const recv_idx, __half* const recv_value,
    const int64_t* const recv_prefix, cudaStream_t stream);
template void NCCLCommunicator::SparseAllToAll<int64_t, __half>(
    const int64_t* const send_idx, const __half* const send_value,
    const int64_t num_feat, const int64_t* const send_prefix,
    int64_t* const recv_idx, __half* const recv_value,
    const int64_t* const recv_prefix, cudaStream_t stream);

int NCCLCommunicator::size() const { return size_; }

int NCCLCommunicator::rank() const { return rank_; }

/* CAPI **********************************************************************/

DGL_REGISTER_GLOBAL("cuda.nccl._CAPI_DGLNCCLGetUniqueId")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      *rv = NCCLUniqueIdRef(std::make_shared<NCCLUniqueId>());
    });

DGL_REGISTER_GLOBAL("cuda.nccl._CAPI_DGLNCCLUniqueIdToString")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      NCCLUniqueIdRef idObj = args[0];
      *rv = idObj->ToString();
    });

DGL_REGISTER_GLOBAL("cuda.nccl._CAPI_DGLNCCLUniqueIdFromString")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      const std::string str = args[0];

      NCCLUniqueIdRef ref(std::make_shared<NCCLUniqueId>());
      ref->FromString(str);
      *rv = ref;
    });

DGL_REGISTER_GLOBAL("cuda.nccl._CAPI_DGLNCCLCreateComm")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      const int size = args[0];
      const int rank = args[1];
      NCCLUniqueIdRef idObj = args[2];

      *rv = NCCLCommunicatorRef(
          std::make_shared<NCCLCommunicator>(size, rank, idObj->Get()));
    });

DGL_REGISTER_GLOBAL("cuda.nccl._CAPI_DGLNCCLSparseAllToAllPush")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      NCCLCommunicatorRef comm = args[0];
      IdArray in_idx = args[1];
      NDArray in_values = args[2];
      NDArrayPartitionRef part = args[3];

      List<ObjectRef> ret;
      ATEN_ID_TYPE_SWITCH(in_idx->dtype, IdType, {
        ATEN_DTYPE_SWITCH(in_values->dtype, DType, "values", {
          auto result =
              SparsePush<IdType, DType>(comm, in_idx, in_values, part);
          ret.push_back(Value(MakeValue(result.first)));
          ret.push_back(Value(MakeValue(result.second)));
        });
      });

      *rv = ret;
    });

DGL_REGISTER_GLOBAL("cuda.nccl._CAPI_DGLNCCLSparseAllToAllPull")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      NCCLCommunicatorRef comm = args[0];
      // the indexes this process is requesting from others
      IdArray req_idx = args[1];

      // the tensor this process has to fulfill other requests
      NDArray tensor = args[2];
      NDArrayPartitionRef part = args[3];

      ATEN_ID_TYPE_SWITCH(req_idx->dtype, IdType, {
        ATEN_DTYPE_SWITCH(tensor->dtype, DType, "values", {
          *rv = SparsePull<IdType, DType>(comm, req_idx, tensor, part);
        });
      });
    });

DGL_REGISTER_GLOBAL("cuda.nccl._CAPI_DGLNCCLHasSupport")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
#ifndef DGL_USE_NCCL
      return false;
#else
      return true;
#endif
    });

}  // namespace cuda
}  // namespace runtime
}  // namespace dgl
