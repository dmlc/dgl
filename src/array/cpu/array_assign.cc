/*!
 *  Copyright (c) 2019 by Contributors
 * \file array/cpu/array_assign.cc
 * \brief Array assign CPU implementation
 */
#include <dgl/array.h>
#include <numeric>
#include "../arith.h"

namespace dgl {
using runtime::NDArray;
namespace aten {
namespace impl {

template<DLDeviceType XPU, typename DType>
void Assign(NDArray array, int64_t index, DType value) {
  DType *array_data = static_cast<DType *>(array->data);
  array_data[index] = value;
}

template void Assign<kDLCPU, int32_t>(
    NDArray array, int64_t index, int32_t value);
template void Assign<kDLCPU, int64_t>(
    NDArray array, int64_t index, int64_t value);
template void Assign<kDLCPU, float>(
    NDArray array, int64_t index, float value);
template void Assign<kDLCPU, double>(
    NDArray array, int64_t index, double value);

};  // namespace impl

};  // namespace aten

};  // namespace dgl
