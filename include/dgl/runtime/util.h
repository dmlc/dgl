/*!
 *  Copyright (c) 2017 by Contributors
 * \file tvm/runtime/util.h
 * \brief Useful runtime util.
 */
#ifndef TVM_RUNTIME_UTIL_H_
#define TVM_RUNTIME_UTIL_H_

#include "c_runtime_api.h"

namespace tvm {
namespace runtime {

/*!
 * \brief Check whether type matches the given spec.
 * \param t The type
 * \param code The type code.
 * \param bits The number of bits to be matched.
 * \param lanes The number of lanes sin the type.
 */
inline bool TypeMatch(TVMType t, int code, int bits, int lanes = 1) {
  return t.code == code && t.bits == bits && t.lanes == lanes;
}
}  // namespace runtime
}  // namespace tvm
// Forward declare the intrinsic id we need
// in structure fetch to enable stackvm in runtime
namespace tvm {
namespace ir {
namespace intrinsic {
/*! \brief The kind of structure field info used in intrinsic */
enum TVMStructFieldKind : int {
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
  // TVMValue field
  kTVMValueContent,
  kTVMValueKindBound_
};
}  // namespace intrinsic
}  // namespace ir
}  // namespace tvm
#endif  // TVM_RUNTIME_UTIL_H_
