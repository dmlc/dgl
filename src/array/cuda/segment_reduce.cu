/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cuda/segment_reduce.cu
 * \brief Segment reduce C APIs and definitions.
 */
#include <dgl/array.h>
#include <dgl/base_heterograph.h>
#include "./segment_reduce.cuh"
#include "./functor.cuh"
#include "./utils.h"


namespace dgl {

using namespace cuda;

namespace aten {


template <int XPU, typename IdType, int bits>
void SegmentReduce(const std::string& op,
                   NDArray feat,
                   NDArray offsets,
                   NDArray out,
                   NDArray arg) {
  SWITCH_BITS(bits, DType, {
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
  });
}


template <int XPU, typename IdType, int bits>
void ScatterAdd(NDArray feat,
                NDArray idx,
                NDArray out) {
  SWITCH_BITS(bits, DType, {
    cuda::ScatterAdd<IdType, DType>(feat, idx, out);
  });
}


template <int XPU, typename IdType, int bits>
void UpdateGradMinMax_hetero(const HeteroGraphPtr& g,
                const std::string& op,
                const std::vector<NDArray>& feat,
                const std::vector<NDArray>& idx,
                const std::vector<NDArray>& idx_etype,
                std::vector<NDArray>* out) {
  SWITCH_BITS(bits, DType, {
    cuda::UpdateGradMinMax_hetero<IdType, DType>(g, op, feat, idx, idx_etype, out);
  });
}


template <int XPU, typename IdType, int bits>
void BackwardSegmentCmp(NDArray feat,
                        NDArray arg,
                        NDArray out) {
  SWITCH_BITS(bits, DType, {
    cuda::BackwardSegmentCmp<IdType, DType>(feat, arg, out);
  });
}


template void SegmentReduce<kDLGPU, int32_t, 16>(
    const std::string& op,
    NDArray feat,
    NDArray offsets,
    NDArray out,
    NDArray arg);
template void SegmentReduce<kDLGPU, int64_t, 16>(
    const std::string &op,
    NDArray feat,
    NDArray offsets,
    NDArray out,
    NDArray arg);
template void SegmentReduce<kDLGPU, int32_t, 32>(
    const std::string& op,
    NDArray feat,
    NDArray offsets,
    NDArray out,
    NDArray arg);
template void SegmentReduce<kDLGPU, int64_t, 32>(
    const std::string &op,
    NDArray feat,
    NDArray offsets,
    NDArray out,
    NDArray arg);
template void SegmentReduce<kDLGPU, int32_t, 64>(
    const std::string &op,
    NDArray feat,
    NDArray offsets,
    NDArray out,
    NDArray arg);
template void SegmentReduce<kDLGPU, int64_t, 64>(
    const std::string &op,
    NDArray feat,
    NDArray offsets,
    NDArray out,
    NDArray arg);
template void ScatterAdd<kDLGPU, int32_t, 16>(
    NDArray feat,
    NDArray idx,
    NDArray out);
template void ScatterAdd<kDLGPU, int64_t, 16>(
    NDArray feat,
    NDArray idx,
    NDArray out);
template void ScatterAdd<kDLGPU, int32_t, 32>(
    NDArray feat,
    NDArray idx,
    NDArray out);
template void ScatterAdd<kDLGPU, int64_t, 32>(
    NDArray feat,
    NDArray idx,
    NDArray out);
template void ScatterAdd<kDLGPU, int32_t, 64>(
    NDArray feat,
    NDArray idx,
    NDArray out);
template void ScatterAdd<kDLGPU, int64_t, 64>(
    NDArray feat,
    NDArray idx,
    NDArray out);

template void UpdateGradMinMax_hetero<kDLGPU, int32_t, 16>(
    const HeteroGraphPtr& g, const std::string& op,
    const std::vector<NDArray>& feat, const std::vector<NDArray>& idx,
    const std::vector<NDArray>& idx_etype, std::vector<NDArray>* out);
template void UpdateGradMinMax_hetero<kDLGPU, int64_t, 16>(
    const HeteroGraphPtr& g, const std::string& op,
    const std::vector<NDArray>& feat, const std::vector<NDArray>& idx,
    const std::vector<NDArray>& idx_etype, std::vector<NDArray>* out);
template void UpdateGradMinMax_hetero<kDLGPU, int32_t, 32>(
    const HeteroGraphPtr& g, const std::string& op,
    const std::vector<NDArray>& feat, const std::vector<NDArray>& idx,
    const std::vector<NDArray>& idx_etype, std::vector<NDArray>* out);
template void UpdateGradMinMax_hetero<kDLGPU, int64_t, 32>(
    const HeteroGraphPtr& g, const std::string& op,
    const std::vector<NDArray>& feat, const std::vector<NDArray>& idx,
    const std::vector<NDArray>& idx_etype, std::vector<NDArray>* out);
template void UpdateGradMinMax_hetero<kDLGPU, int32_t, 64>(
    const HeteroGraphPtr& g, const std::string& op,
    const std::vector<NDArray>& feat, const std::vector<NDArray>& idx,
    const std::vector<NDArray>& idx_etype, std::vector<NDArray>* out);
template void UpdateGradMinMax_hetero<kDLGPU, int64_t, 64>(
    const HeteroGraphPtr& g, const std::string& op,
    const std::vector<NDArray>& feat, const std::vector<NDArray>& idx,
    const std::vector<NDArray>& idx_etype, std::vector<NDArray>* out);

template void BackwardSegmentCmp<kDLGPU, int32_t, 16>(
    NDArray feat,
    NDArray arg,
    NDArray out);
template void BackwardSegmentCmp<kDLGPU, int64_t, 16>(
    NDArray feat,
    NDArray arg,
    NDArray out);
template void BackwardSegmentCmp<kDLGPU, int32_t, 32>(
    NDArray feat,
    NDArray arg,
    NDArray out);
template void BackwardSegmentCmp<kDLGPU, int64_t, 32>(
    NDArray feat,
    NDArray arg,
    NDArray out);
template void BackwardSegmentCmp<kDLGPU, int32_t, 64>(
    NDArray feat,
    NDArray arg,
    NDArray out);
template void BackwardSegmentCmp<kDLGPU, int64_t, 64>(
    NDArray feat,
    NDArray arg,
    NDArray out);

}  // namespace aten
}  // namespace dgl
