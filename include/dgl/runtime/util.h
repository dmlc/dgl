/**
 *  Copyright (c) 2017 by Contributors
 * @file dgl/runtime/util.h
 * @brief Useful runtime util.
 */
#ifndef DGL_RUNTIME_UTIL_H_
#define DGL_RUNTIME_UTIL_H_

#include "c_runtime_api.h"

namespace dgl {
namespace runtime {

/**
 * @brief Check whether type matches the given spec.
 * @param t The type
 * @param code The type code.
 * @param bits The number of bits to be matched.
 * @param lanes The number of lanes sin the type.
 */
inline bool TypeMatch(DGLDataType t, int code, int bits, int lanes = 1) {
  return t.code == code && t.bits == bits && t.lanes == lanes;
}
}  // namespace runtime
}  // namespace dgl
// Forward declare the intrinsic id we need
// in structure fetch to enable stackvm in runtime
namespace dgl {
namespace ir {
namespace intrinsic {
/** @brief The kind of structure field info used in intrinsic */
enum DGLStructFieldKind : int {
  // array head address
  kArrAddr,
  kArrData,
  kArrShape,
  kArrStrides,
  kArrNDim,
  kArrTypeCode,
  kArrTypeBits,
  kArrTypeLanes,
  kArrByteOffset,
  kArrDeviceId,
  kArrDeviceType,
  kArrKindBound_,
  // DGLValue field
  kDGLValueContent,
  kDGLValueKindBound_
};
}  // namespace intrinsic
}  // namespace ir
}  // namespace dgl
#endif  // DGL_RUNTIME_UTIL_H_
