/**
 *  Copyright (c) 2020 by Contributors
 * @file kernel/cpu/segment_reduce.cc
 * @brief Segment reduce C APIs and definitions.
 */
#include "./segment_reduce.h"

#include <dgl/array.h>

#include <string>

#include "./spmm_binary_ops.h"

namespace dgl {
namespace aten {

/** @brief Segment Reduce operator. */
template <int XPU, typename IdType, typename DType>
void SegmentReduce(
    const std::string& op, NDArray feat, NDArray offsets, NDArray out,
    NDArray arg) {
  if (op == "sum") {
    cpu::SegmentSum<IdType, DType>(feat, offsets, out);
  } else if (op == "max" || op == "min") {
    if (op == "max") {
      cpu::SegmentCmp<IdType, DType, cpu::op::Max<DType>>(
          feat, offsets, out, arg);
    } else {
      cpu::SegmentCmp<IdType, DType, cpu::op::Min<DType>>(
          feat, offsets, out, arg);
    }
  } else {
    LOG(FATAL) << "Unsupported reduce function " << op;
  }
}

/** @brief Scatter Add.*/
template <int XPU, typename IdType, typename DType>
void ScatterAdd(NDArray feat, NDArray idx, NDArray out) {
  cpu::ScatterAdd<IdType, DType>(feat, idx, out);
}

/** @brief Update gradients for reduce operator max/min on heterogeneous
 * graph.*/
template <int XPU, typename IdType, typename DType>
void UpdateGradMinMax_hetero(
    const HeteroGraphPtr& g, const std::string& op,
    const std::vector<NDArray>& feat, const std::vector<NDArray>& idx,
    const std::vector<NDArray>& idx_etype, std::vector<NDArray>* out) {
  cpu::UpdateGradMinMax_hetero<IdType, DType>(g, op, feat, idx, idx_etype, out);
}

/** @brief Backward function of segment cmp.*/
template <int XPU, typename IdType, typename DType>
void BackwardSegmentCmp(NDArray feat, NDArray arg, NDArray out) {
  cpu::BackwardSegmentCmp<IdType, DType>(feat, arg, out);
}

template void SegmentReduce<kDGLCPU, int32_t, BFloat16>(
    const std::string& op, NDArray feat, NDArray offsets, NDArray out,
    NDArray arg);
template void SegmentReduce<kDGLCPU, int64_t, BFloat16>(
    const std::string& op, NDArray feat, NDArray offsets, NDArray out,
    NDArray arg);
template void SegmentReduce<kDGLCPU, int32_t, float>(
    const std::string& op, NDArray feat, NDArray offsets, NDArray out,
    NDArray arg);
template void SegmentReduce<kDGLCPU, int64_t, float>(
    const std::string& op, NDArray feat, NDArray offsets, NDArray out,
    NDArray arg);
template void SegmentReduce<kDGLCPU, int32_t, double>(
    const std::string& op, NDArray feat, NDArray offsets, NDArray out,
    NDArray arg);
template void SegmentReduce<kDGLCPU, int64_t, double>(
    const std::string& op, NDArray feat, NDArray offsets, NDArray out,
    NDArray arg);

template <>
void ScatterAdd<kDGLCPU, int32_t, BFloat16>(
    NDArray feat, NDArray idx, NDArray out) {
  LOG(FATAL) << "Unsupported CPU kernel for ScatterAdd for BF16.";
}
template <>
void ScatterAdd<kDGLCPU, int64_t, BFloat16>(
    NDArray feat, NDArray idx, NDArray out) {
  LOG(FATAL) << "Unsupported CPU kernel for ScatterAdd for BF16.";
}
template void ScatterAdd<kDGLCPU, int32_t, float>(
    NDArray feat, NDArray idx, NDArray out);
template void ScatterAdd<kDGLCPU, int64_t, float>(
    NDArray feat, NDArray idx, NDArray out);
template void ScatterAdd<kDGLCPU, int32_t, double>(
    NDArray feat, NDArray idx, NDArray out);
template void ScatterAdd<kDGLCPU, int64_t, double>(
    NDArray feat, NDArray arg, NDArray out);

template <>
void UpdateGradMinMax_hetero<kDGLCPU, int32_t, BFloat16>(
    const HeteroGraphPtr& g, const std::string& op,
    const std::vector<NDArray>& feat, const std::vector<NDArray>& idx,
    const std::vector<NDArray>& idx_etype, std::vector<NDArray>* out) {
  LOG(FATAL) << "Unsupported CPU kernel for UpdateGradMinMax_hetero for BF16.";
}
template <>
void UpdateGradMinMax_hetero<kDGLCPU, int64_t, BFloat16>(
    const HeteroGraphPtr& g, const std::string& op,
    const std::vector<NDArray>& feat, const std::vector<NDArray>& idx,
    const std::vector<NDArray>& idx_etype, std::vector<NDArray>* out) {
  LOG(FATAL) << "Unsupported CPU kernel for UpdateGradMinMax_hetero for BF16.";
}
template void UpdateGradMinMax_hetero<kDGLCPU, int32_t, float>(
    const HeteroGraphPtr& g, const std::string& op,
    const std::vector<NDArray>& feat, const std::vector<NDArray>& idx,
    const std::vector<NDArray>& idx_etype, std::vector<NDArray>* out);
template void UpdateGradMinMax_hetero<kDGLCPU, int64_t, float>(
    const HeteroGraphPtr& g, const std::string& op,
    const std::vector<NDArray>& feat, const std::vector<NDArray>& idx,
    const std::vector<NDArray>& idx_etype, std::vector<NDArray>* out);
template void UpdateGradMinMax_hetero<kDGLCPU, int32_t, double>(
    const HeteroGraphPtr& g, const std::string& op,
    const std::vector<NDArray>& feat, const std::vector<NDArray>& idx,
    const std::vector<NDArray>& idx_etype, std::vector<NDArray>* out);
template void UpdateGradMinMax_hetero<kDGLCPU, int64_t, double>(
    const HeteroGraphPtr& g, const std::string& op,
    const std::vector<NDArray>& feat, const std::vector<NDArray>& idx,
    const std::vector<NDArray>& idx_etype, std::vector<NDArray>* out);

template void BackwardSegmentCmp<kDGLCPU, int32_t, BFloat16>(
    NDArray feat, NDArray arg, NDArray out);
template void BackwardSegmentCmp<kDGLCPU, int64_t, BFloat16>(
    NDArray feat, NDArray arg, NDArray out);
template void BackwardSegmentCmp<kDGLCPU, int32_t, float>(
    NDArray feat, NDArray arg, NDArray out);
template void BackwardSegmentCmp<kDGLCPU, int64_t, float>(
    NDArray feat, NDArray arg, NDArray out);
template void BackwardSegmentCmp<kDGLCPU, int32_t, double>(
    NDArray feat, NDArray arg, NDArray out);
template void BackwardSegmentCmp<kDGLCPU, int64_t, double>(
    NDArray feat, NDArray arg, NDArray out);

}  // namespace aten
}  // namespace dgl
