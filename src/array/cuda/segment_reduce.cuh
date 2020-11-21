/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cuda/segment_reduce.cuh
 * \brief Segment reduce kernel function header.
 */
#ifndef DGL_ARRAY_SEGMENT_REDUCE_CUH_
#define DGL_ARRAY_SEGMENT_REDUCE_CUH_

#include "../../runtime/cuda/cuda_common.h"
#include "./utils.h"

namespace dgl {

using namespace cuda;

namespace aten {
namespace cuda {

/*!
 * \brief CUDA kernel of segment reduce.
 */
template <typename IdType, typename DType,
          typename ReduceOp>
__global__ void SegmentReduceKernel(
    const DType* feat, const IdType* offsets,
    DType* out, IdType* arg,
    int64_t n, int64_t dim){
  // TODO(zihao)
}

/*!
 * \brief CUDA kernel of segment broadcast.
 */
template <typename IdType, typename DType, typename ReduceOp>
__global__ void SegmentBcastKernel(
    const DType* feat, const IdType* offsets,
    DType* out,
    int64_t n, int64_t dim) {
  // TODO(zihao)
}

template <typename IdType, typename DType, typename ReduceOp>
void SegmentReduce(
    NDArray feat,
    NDArray offsets,
    NDArray out,
    NDArray arg) {
  const DType* feat_data = feat.Ptr<DType>();
  const IdType* offsets_data = offsets.Ptr<IdType>();
  DType* out_data = out.Ptr<DType>();
  IdType* arg_data = arg.Ptr<IdType>();

  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  int64_t n = out->shape[0];
  int64_t dim = 1;
  for (int i = 1; i < out->ndim; ++i)
    dim *= out->shape[i];
  const int nbx = 1;
  const int nby = 1;
  const int ntx = 1;
  const int nty = 1;
  const dim3 nblks(nbx, nby);
  const dim3 nthrs(ntx, nty);
  CUDA_KERNEL_CALL((SegmentReduceKernel<IdType, DType, ReduceOp>),
      nblks, nthrs, 0, thr_entry->stream,
      feat_data, offsets_data, out_data, arg_data,
      n, dim);
}

template <typename IdType, typename DType>
void SegmentBcast(
    NDArray feat,
    NDArray offsets,
    NDArray out) {
  const DType *feat_data = feat.Ptr<DType>();
  const IdType *offsets_data = offsets.Ptr<IdType>();
  DType *out_data = out.Ptr<DType>();
  // TODO(zihao)
}

}  // namespace cuda
}  // namespace aten
}  // namespace dgl

#endif
