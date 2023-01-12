/**
 *  Copyright (c) 2020 by Contributors
 * @file array/cuda/segment_reduce.cu
 * @brief Segment reduce C APIs and definitions.
 */
#include <dgl/array.h>
#include <dgl/base_heterograph.h>

#include "./functor.cuh"
#include "./segment_reduce.cuh"
#include "./utils.h"

namespace dgl {

using namespace cuda;

namespace aten {

template <int XPU, typename IdType, typename DType>
void SegmentReduce(
    const std::string& op, NDArray feat, NDArray offsets, NDArray out,
    NDArray arg) {
  if (op == "sum") {
    cuda::SegmentReduce<IdType, DType, cuda::reduce::Sum<IdType, DType>>(
        feat, offsets, out, arg);
  } else if (op == "max") {
    cuda::SegmentReduce<IdType, DType, cuda::reduce::Max<IdType, DType>>(
        feat, offsets, out, arg);
  } else if (op == "min") {
    cuda::SegmentReduce<IdType, DType, cuda::reduce::Min<IdType, DType>>(
        feat, offsets, out, arg);
  } else {
    LOG(FATAL) << "Not implemented";
  }
}

template <int XPU, typename IdType, typename DType>
void ScatterAdd(NDArray feat, NDArray idx, NDArray out) {
  cuda::ScatterAdd<IdType, DType>(feat, idx, out);
}

template <int XPU, typename IdType, typename DType>
void UpdateGradMinMax_hetero(
    const HeteroGraphPtr& g, const std::string& op,
    const std::vector<NDArray>& feat, const std::vector<NDArray>& idx,
    const std::vector<NDArray>& idx_etype, std::vector<NDArray>* out) {
  cuda::UpdateGradMinMax_hetero<IdType, DType>(
      g, op, feat, idx, idx_etype, out);
}

template <int XPU, typename IdType, typename DType>
void BackwardSegmentCmp(NDArray feat, NDArray arg, NDArray out) {
  cuda::BackwardSegmentCmp<IdType, DType>(feat, arg, out);
}

template void SegmentReduce<kDGLCUDA, int32_t, __half>(
    const std::string& op, NDArray feat, NDArray offsets, NDArray out,
    NDArray arg);
template void SegmentReduce<kDGLCUDA, int64_t, __half>(
    const std::string& op, NDArray feat, NDArray offsets, NDArray out,
    NDArray arg);
#if BF16_ENABLED
template void SegmentReduce<kDGLCUDA, int32_t, __nv_bfloat16>(
    const std::string& op, NDArray feat, NDArray offsets, NDArray out,
    NDArray arg);
template void SegmentReduce<kDGLCUDA, int64_t, __nv_bfloat16>(
    const std::string& op, NDArray feat, NDArray offsets, NDArray out,
    NDArray arg);
#endif  // BF16_ENABLED
template void SegmentReduce<kDGLCUDA, int32_t, float>(
    const std::string& op, NDArray feat, NDArray offsets, NDArray out,
    NDArray arg);
template void SegmentReduce<kDGLCUDA, int64_t, float>(
    const std::string& op, NDArray feat, NDArray offsets, NDArray out,
    NDArray arg);
template void SegmentReduce<kDGLCUDA, int32_t, double>(
    const std::string& op, NDArray feat, NDArray offsets, NDArray out,
    NDArray arg);
template void SegmentReduce<kDGLCUDA, int64_t, double>(
    const std::string& op, NDArray feat, NDArray offsets, NDArray out,
    NDArray arg);

template void ScatterAdd<kDGLCUDA, int32_t, __half>(
    NDArray feat, NDArray idx, NDArray out);
template void ScatterAdd<kDGLCUDA, int64_t, __half>(
    NDArray feat, NDArray idx, NDArray out);
#if BF16_ENABLED
template void ScatterAdd<kDGLCUDA, int32_t, __nv_bfloat16>(
    NDArray feat, NDArray idx, NDArray out);
template void ScatterAdd<kDGLCUDA, int64_t, __nv_bfloat16>(
    NDArray feat, NDArray idx, NDArray out);
#endif  // BF16_ENABLED
template void ScatterAdd<kDGLCUDA, int32_t, float>(
    NDArray feat, NDArray idx, NDArray out);
template void ScatterAdd<kDGLCUDA, int64_t, float>(
    NDArray feat, NDArray idx, NDArray out);
template void ScatterAdd<kDGLCUDA, int32_t, double>(
    NDArray feat, NDArray idx, NDArray out);
template void ScatterAdd<kDGLCUDA, int64_t, double>(
    NDArray feat, NDArray idx, NDArray out);

template void UpdateGradMinMax_hetero<kDGLCUDA, int32_t, __half>(
    const HeteroGraphPtr& g, const std::string& op,
    const std::vector<NDArray>& feat, const std::vector<NDArray>& idx,
    const std::vector<NDArray>& idx_etype, std::vector<NDArray>* out);
template void UpdateGradMinMax_hetero<kDGLCUDA, int64_t, __half>(
    const HeteroGraphPtr& g, const std::string& op,
    const std::vector<NDArray>& feat, const std::vector<NDArray>& idx,
    const std::vector<NDArray>& idx_etype, std::vector<NDArray>* out);
#if BF16_ENABLED
template void UpdateGradMinMax_hetero<kDGLCUDA, int32_t, __nv_bfloat16>(
    const HeteroGraphPtr& g, const std::string& op,
    const std::vector<NDArray>& feat, const std::vector<NDArray>& idx,
    const std::vector<NDArray>& idx_etype, std::vector<NDArray>* out);
template void UpdateGradMinMax_hetero<kDGLCUDA, int64_t, __nv_bfloat16>(
    const HeteroGraphPtr& g, const std::string& op,
    const std::vector<NDArray>& feat, const std::vector<NDArray>& idx,
    const std::vector<NDArray>& idx_etype, std::vector<NDArray>* out);
#endif  // BF16_ENABLED
template void UpdateGradMinMax_hetero<kDGLCUDA, int32_t, float>(
    const HeteroGraphPtr& g, const std::string& op,
    const std::vector<NDArray>& feat, const std::vector<NDArray>& idx,
    const std::vector<NDArray>& idx_etype, std::vector<NDArray>* out);
template void UpdateGradMinMax_hetero<kDGLCUDA, int64_t, float>(
    const HeteroGraphPtr& g, const std::string& op,
    const std::vector<NDArray>& feat, const std::vector<NDArray>& idx,
    const std::vector<NDArray>& idx_etype, std::vector<NDArray>* out);
template void UpdateGradMinMax_hetero<kDGLCUDA, int32_t, double>(
    const HeteroGraphPtr& g, const std::string& op,
    const std::vector<NDArray>& feat, const std::vector<NDArray>& idx,
    const std::vector<NDArray>& idx_etype, std::vector<NDArray>* out);
template void UpdateGradMinMax_hetero<kDGLCUDA, int64_t, double>(
    const HeteroGraphPtr& g, const std::string& op,
    const std::vector<NDArray>& feat, const std::vector<NDArray>& idx,
    const std::vector<NDArray>& idx_etype, std::vector<NDArray>* out);

template void BackwardSegmentCmp<kDGLCUDA, int32_t, __half>(
    NDArray feat, NDArray arg, NDArray out);
template void BackwardSegmentCmp<kDGLCUDA, int64_t, __half>(
    NDArray feat, NDArray arg, NDArray out);
#if BF16_ENABLED
template void BackwardSegmentCmp<kDGLCUDA, int32_t, __nv_bfloat16>(
    NDArray feat, NDArray arg, NDArray out);
template void BackwardSegmentCmp<kDGLCUDA, int64_t, __nv_bfloat16>(
    NDArray feat, NDArray arg, NDArray out);
#endif  // BF16_ENABLED
template void BackwardSegmentCmp<kDGLCUDA, int32_t, float>(
    NDArray feat, NDArray arg, NDArray out);
template void BackwardSegmentCmp<kDGLCUDA, int64_t, float>(
    NDArray feat, NDArray arg, NDArray out);
template void BackwardSegmentCmp<kDGLCUDA, int32_t, double>(
    NDArray feat, NDArray arg, NDArray out);
template void BackwardSegmentCmp<kDGLCUDA, int64_t, double>(
    NDArray feat, NDArray arg, NDArray out);

}  // namespace aten
}  // namespace dgl
