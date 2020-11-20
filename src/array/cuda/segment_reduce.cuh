/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cuda/segment_reduce.cuh
 * \brief Segment reduce kernel function header.
 */
#ifndef DGL_ARRAY_SEGMENT_REDUCE_CUH_
#define DGL_ARRAY_SEGMENT_REDUCE_CUH_

namespace dgl {

using namespace cuda;

namespace aten {
namespace cuda {

/*!
 * \brief CUDA kernel of segment reduce.
 */
template <typename Idx, typename DType,
          typename ReduceOp>
__global__ void SegmentReduceKernel(
  const DType* array,
  const Idx* offset,
  DType* out){
}


}  // namespace cuda
}  // namespace aten
}  // namespace dgl

#endif
