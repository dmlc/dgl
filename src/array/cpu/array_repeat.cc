/**
 *  Copyright (c) 2020 by Contributors
 * @file array/cpu/array_repeat.cc
 * @brief Array repeat CPU implementation
 */
#include <dgl/array.h>

#include <algorithm>

namespace dgl {
using runtime::NDArray;
namespace aten {
namespace impl {

template <DGLDeviceType XPU, typename DType, typename IdType>
NDArray Repeat(NDArray array, IdArray repeats) {
  CHECK(array->shape[0] == repeats->shape[0])
      << "shape of array and repeats mismatch";

  const int64_t len = array->shape[0];
  const DType *array_data = static_cast<DType *>(array->data);
  const IdType *repeats_data = static_cast<IdType *>(repeats->data);

  IdType num_elements = 0;
  for (int64_t i = 0; i < len; ++i) num_elements += repeats_data[i];

  NDArray result = NDArray::Empty({num_elements}, array->dtype, array->ctx);
  DType *result_data = static_cast<DType *>(result->data);
  IdType curr = 0;
  for (int64_t i = 0; i < len; ++i) {
    std::fill(
        result_data + curr, result_data + curr + repeats_data[i],
        array_data[i]);
    curr += repeats_data[i];
  }

  return result;
}

template NDArray Repeat<kDGLCPU, int32_t, int32_t>(NDArray, IdArray);
template NDArray Repeat<kDGLCPU, int64_t, int32_t>(NDArray, IdArray);
template NDArray Repeat<kDGLCPU, float, int32_t>(NDArray, IdArray);
template NDArray Repeat<kDGLCPU, double, int32_t>(NDArray, IdArray);
template NDArray Repeat<kDGLCPU, int32_t, int64_t>(NDArray, IdArray);
template NDArray Repeat<kDGLCPU, int64_t, int64_t>(NDArray, IdArray);
template NDArray Repeat<kDGLCPU, float, int64_t>(NDArray, IdArray);
template NDArray Repeat<kDGLCPU, double, int64_t>(NDArray, IdArray);

};  // namespace impl
};  // namespace aten
};  // namespace dgl
