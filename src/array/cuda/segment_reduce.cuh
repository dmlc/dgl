/**
 *  Copyright (c) 2020 by Contributors
 * @file array/cuda/segment_reduce.cuh
 * @brief Segment reduce kernel function header.
 */
#ifndef DGL_ARRAY_CUDA_SEGMENT_REDUCE_CUH_
#define DGL_ARRAY_CUDA_SEGMENT_REDUCE_CUH_

#include <string>
#include <vector>

#include "../../runtime/cuda/cuda_common.h"
#include "./atomic.cuh"
#include "./utils.h"

namespace dgl {

using namespace cuda;
using namespace runtime;

namespace aten {
namespace cuda {

/**
 * @brief CUDA kernel of segment reduce.
 * @note each blockthread is responsible for aggregation on a row
 *       in the result tensor.
 */
template <typename IdType, typename DType, typename ReduceOp>
__global__ void SegmentReduceKernel(
    const DType* feat, const IdType* offsets, DType* out, IdType* arg,
    int64_t n, int64_t dim) {
  for (int row = blockIdx.x; row < n; row += gridDim.x) {
    int col = blockIdx.y * blockDim.x + threadIdx.x;
    while (col < dim) {
      typename accum_dtype<DType>::type local_accum = ReduceOp::zero();
      IdType local_arg = -1;
      for (IdType i = offsets[row]; i < offsets[row + 1]; ++i) {
        ReduceOp::Call(&local_accum, &local_arg, feat[i * dim + col], i);
      }
      out[row * dim + col] = static_cast<DType>(local_accum);
      if (ReduceOp::require_arg) arg[row * dim + col] = local_arg;
      col += gridDim.y * blockDim.x;
    }
  }
}

/**
 * @brief CUDA kernel of scatter add.
 * @note each blockthread is responsible for adding a row in feature tensor
 *       to a target row in output tensor.
 */
template <typename IdType, typename DType>
__global__ void ScatterAddKernel(
    const DType* feat, const IdType* idx, DType* out, int64_t n, int64_t dim) {
  for (int row = blockIdx.x; row < n; row += gridDim.x) {
    const int write_row = idx[row];
    int col = blockIdx.y * blockDim.x + threadIdx.x;
    while (col < dim) {
      cuda::AtomicAdd(out + write_row * dim + col, feat[row * dim + col]);
      col += gridDim.y * blockDim.x;
    }
  }
}

/**
 * @brief CUDA kernel to update gradients for reduce op max/min
 * @note each WARP (group of 32 threads) is responsible for adding a row in
 * feature tensor to a target row in output tensor.
 */

template <typename IdType, typename DType>
__global__ void UpdateGradMinMaxHeteroKernel(
    const DType* feat, const IdType* idx, const IdType* idx_type, DType* out,
    int64_t n, int64_t dim, int type) {
  unsigned int tId = threadIdx.x;
  unsigned int laneId = tId & 31;
  unsigned int gId = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int warpId = gId >> 5;
  unsigned int warp_size = 32;
  unsigned int row = warpId;

  while (row < n) {
    for (unsigned int col = laneId; col < dim; col += warp_size) {
      if (type == idx_type[row * dim + col]) {
        const int write_row = idx[row * dim + col];
        cuda::AtomicAdd(out + write_row * dim + col, feat[row * dim + col]);
      }
    }
    row += blockDim.x * gridDim.x;
  }
}

/**
 * @brief CUDA kernel of backward phase in segment min/max.
 * @note each blockthread is responsible for writing a row in the
 *       result gradient tensor by lookup the ArgMin/Max for index information.
 */
template <typename IdType, typename DType>
__global__ void BackwardSegmentCmpKernel(
    const DType* feat, const IdType* arg, DType* out, int64_t n, int64_t dim) {
  for (int row = blockIdx.x; row < n; row += gridDim.x) {
    int col = blockIdx.y * blockDim.x + threadIdx.x;
    while (col < dim) {
      int write_row = arg[row * dim + col];
      if (write_row >= 0) {
        out[write_row * dim + col] = feat[row * dim + col];
      }
      col += gridDim.y * blockDim.x;
    }
  }
}

/**
 * @brief CUDA implementation of forward phase of Segment Reduce.
 * @param feat The input tensor.
 * @param offsets The offsets tensor.
 * @param out The output tensor.
 * @param arg An auxiliary tensor storing ArgMax/Min information,
 */
template <typename IdType, typename DType, typename ReduceOp>
void SegmentReduce(NDArray feat, NDArray offsets, NDArray out, NDArray arg) {
  const DType* feat_data = feat.Ptr<DType>();
  const IdType* offsets_data = offsets.Ptr<IdType>();
  DType* out_data = out.Ptr<DType>();
  IdType* arg_data = arg.Ptr<IdType>();

  cudaStream_t stream = runtime::getCurrentCUDAStream();
  int64_t n = out->shape[0];
  int64_t dim = 1;
  for (int i = 1; i < out->ndim; ++i) dim *= out->shape[i];

  const int nbx = FindNumBlocks<'x'>(n);
  const int ntx = FindNumThreads(dim);
  const int nby = FindNumBlocks<'y'>((dim + ntx - 1) / ntx);
  const int nty = 1;
  const dim3 nblks(nbx, nby);
  const dim3 nthrs(ntx, nty);
  // TODO(zihao): try cub's DeviceSegmentedReduce and compare the performance.
  CUDA_KERNEL_CALL(
      (SegmentReduceKernel<IdType, DType, ReduceOp>), nblks, nthrs, 0, stream,
      feat_data, offsets_data, out_data, arg_data, n, dim);
}

/**
 * @brief CUDA implementation of Scatter Add (on first dimension).
 * @note math equation: out[idx[i], *] += feat[i, *]
 * @param feat The input tensor.
 * @param idx The indices tensor.
 * @param out The output tensor.
 */
template <typename IdType, typename DType>
void ScatterAdd(NDArray feat, NDArray idx, NDArray out) {
  const DType* feat_data = feat.Ptr<DType>();
  const IdType* idx_data = idx.Ptr<IdType>();
  DType* out_data = out.Ptr<DType>();

  cudaStream_t stream = runtime::getCurrentCUDAStream();
  int64_t n = feat->shape[0];
  int64_t dim = 1;
  for (int i = 1; i < out->ndim; ++i) dim *= out->shape[i];

  const int nbx = FindNumBlocks<'x'>(n);
  const int ntx = FindNumThreads(dim);
  const int nby = FindNumBlocks<'y'>((dim + ntx - 1) / ntx);
  const int nty = 1;
  const dim3 nblks(nbx, nby);
  const dim3 nthrs(ntx, nty);
  CUDA_KERNEL_CALL(
      (ScatterAddKernel<IdType, DType>), nblks, nthrs, 0, stream, feat_data,
      idx_data, out_data, n, dim);
}

/**
 * @brief CUDA implementation to update gradients for reduce op max/min
 * @param graph The input heterogeneous graph.
 * @param op The binary operator, could be `copy_u`, `copy_e'.
 * @param list_feat List of the input tensors.
 * @param list_idx  List of the indices tensors.
 * @param list_idx_etype List of the node- or edge-type tensors.
 * @param list_out List of the output tensors.
 */
template <typename IdType, typename DType>
void UpdateGradMinMax_hetero(
    const HeteroGraphPtr& graph, const std::string& op,
    const std::vector<NDArray>& list_feat, const std::vector<NDArray>& list_idx,
    const std::vector<NDArray>& list_idx_types,
    std::vector<NDArray>* list_out) {
  cudaStream_t stream = runtime::getCurrentCUDAStream();
  if (op == "copy_lhs" || op == "copy_rhs") {
    std::vector<std::vector<dgl_id_t>> src_dst_ntypes(
        graph->NumVertexTypes(), std::vector<dgl_id_t>());
    for (dgl_type_t etype = 0; etype < graph->NumEdgeTypes(); ++etype) {
      auto pair = graph->meta_graph()->FindEdge(etype);
      const dgl_id_t dst_ntype = pair.first;  // graph is reversed
      const dgl_id_t src_ntype = pair.second;
      auto same_src_dst_ntype = std::find(
          std::begin(src_dst_ntypes[dst_ntype]),
          std::end(src_dst_ntypes[dst_ntype]), src_ntype);
      // if op is "copy_lhs", relation type with same src and dst node type will
      // be updated once
      if (op == "copy_lhs" &&
          same_src_dst_ntype != std::end(src_dst_ntypes[dst_ntype]))
        continue;
      src_dst_ntypes[dst_ntype].push_back(src_ntype);
      const DType* feat_data = list_feat[dst_ntype].Ptr<DType>();
      const IdType* idx_data = list_idx[dst_ntype].Ptr<IdType>();
      const IdType* idx_type_data = list_idx_types[dst_ntype].Ptr<IdType>();
      int type = (op == "copy_lhs") ? src_ntype : etype;
      DType* out_data = (*list_out)[type].Ptr<DType>();
      int dim = 1;
      for (int i = 1; i < (*list_out)[type]->ndim; ++i)
        dim *= (*list_out)[type]->shape[i];
      int n = list_feat[dst_ntype]->shape[0];
      const int th_per_row = 32;
      const int ntx = 128;
      const int nbx = FindNumBlocks<'x'>((n * th_per_row + ntx - 1) / ntx);
      const dim3 nblks(nbx);
      const dim3 nthrs(ntx);
      CUDA_KERNEL_CALL(
          (UpdateGradMinMaxHeteroKernel<IdType, DType>), nblks, nthrs, 0,
          stream, feat_data, idx_data, idx_type_data, out_data, n, dim, type);
    }
  }
}

/**
 * @brief CUDA implementation of backward phase of Segment Reduce with Min/Max
 *        reducer.
 * @note math equation: out[arg[i, k], k] = feat[i, k]
 * @param feat The input
 *       tensor.
 * @param arg The ArgMin/Max information, used for indexing.
 * @param out The output tensor.
 */
template <typename IdType, typename DType>
void BackwardSegmentCmp(NDArray feat, NDArray arg, NDArray out) {
  const DType* feat_data = feat.Ptr<DType>();
  const IdType* arg_data = arg.Ptr<IdType>();
  DType* out_data = out.Ptr<DType>();

  cudaStream_t stream = runtime::getCurrentCUDAStream();
  int64_t n = feat->shape[0];
  int64_t dim = 1;
  for (int i = 1; i < out->ndim; ++i) dim *= out->shape[i];

  const int nbx = FindNumBlocks<'x'>(n);
  const int ntx = FindNumThreads(dim);
  const int nby = FindNumBlocks<'y'>((dim + ntx - 1) / ntx);
  const int nty = 1;
  const dim3 nblks(nbx, nby);
  const dim3 nthrs(ntx, nty);
  CUDA_KERNEL_CALL(
      (BackwardSegmentCmpKernel<IdType, DType>), nblks, nthrs, 0, stream,
      feat_data, arg_data, out_data, n, dim);
}

}  // namespace cuda
}  // namespace aten
}  // namespace dgl

#endif  // DGL_ARRAY_CUDA_SEGMENT_REDUCE_CUH_
