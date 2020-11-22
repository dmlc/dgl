/*!
 *  Copyright (c) 2020 by Contributors
 * \file kernel/cpu/segment_reduce.cc
 * \brief Segment reduce C APIs and definitions.
 */
#include "./segment_reduce.h"
#include <dgl/array.h>
#include <string>
#include "./spmm_binary_ops.h"

namespace dgl {
namespace aten {

/*! \brief Segment Reduce operator. */
template <int XPU, typename IdType, typename DType>
void SegmentReduce(
    const std::string& op,
    NDArray feat,
    NDArray offsets,
    NDArray out,
    NDArray arg) {
  if (op == "sum") {
    cpu::SegmentSum<IdType, DType>(feat, offsets, out);
  } else if (op == "max" || op == "min") {
    if (op == "max")
      cpu::SegmentCmp<IdType, DType, cpu::op::Max<DType>>(
          feat, offsets, out, arg);
    else
      cpu::SegmentCmp<IdType, DType, cpu::op::Min<DType>>(
          feat, offsets, out, arg);
  } else {
    LOG(FATAL) << "Unsupported reduce function " << op;
  }
}

/*! \brief Backward function of segment cmp.*/
template <int XPU, typename IdType, typename DType>
void BackwardSegmentCmp(
    NDArray feat,
    NDArray arg,
    NDArray out) {
  cpu::BackwardSegmentCmp<IdType, DType>(feat, arg, out);
}

template void SegmentReduce<kDLCPU, int32_t, float>(
    const std::string &op,
    NDArray feat,
    NDArray offsets,
    NDArray out,
    NDArray arg);
template void SegmentReduce<kDLCPU, int64_t, float>(
    const std::string &op,
    NDArray feat,
    NDArray offsets,
    NDArray out,
    NDArray arg);
template void SegmentReduce<kDLCPU, int32_t, double>(
    const std::string &op,
    NDArray feat,
    NDArray offsets,
    NDArray out,
    NDArray arg);
template void SegmentReduce<kDLCPU, int64_t, double>(
    const std::string &op,
    NDArray feat,
    NDArray offsets,
    NDArray out,
    NDArray arg);
template void BackwardSegmentCmp<kDLCPU, int32_t, float>(
    NDArray feat,
    NDArray arg,
    NDArray out);
template void BackwardSegmentCmp<kDLCPU, int64_t, float>(
    NDArray feat,
    NDArray arg,
    NDArray out);
template void BackwardSegmentCmp<kDLCPU, int32_t, double>(
    NDArray feat,
    NDArray arg,
    NDArray out);
template void BackwardSegmentCmp<kDLCPU, int64_t, double>(
    NDArray feat,
    NDArray arg,
    NDArray out);

}  // namespace aten
}  // namespace dgl
