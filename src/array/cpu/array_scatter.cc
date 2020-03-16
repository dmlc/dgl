/*!
 *  Copyright (c) 2019 by Contributors
 * \file array/cpu/array_scatter.cc
 * \brief Array scatter CPU implementation
 */
#include <dgl/array.h>

namespace dgl {
using runtime::NDArray;
namespace aten {
namespace impl {

template <DLDeviceType XPU, typename DType, typename IdType>
NDArray Scatter(NDArray array, IdArray indices) {
  NDArray result = NDArray::Empty({indices->shape[0]}, array->dtype, array->ctx);

  const DType *array_data = static_cast<DType *>(array->data);
  const IdType *indices_data = static_cast<IdType *>(indices->data);
  DType *result_data = static_cast<DType *>(result->data);

  for (int64_t i = 0; i < indices->shape[0]; ++i)
    result_data[indices_data[i]] = array_data[i];

  return result;
}

template NDArray Scatter<kDLCPU, int32_t, int32_t>(NDArray, IdArray);
template NDArray Scatter<kDLCPU, int64_t, int32_t>(NDArray, IdArray);
template NDArray Scatter<kDLCPU, float, int32_t>(NDArray, IdArray);
template NDArray Scatter<kDLCPU, double, int32_t>(NDArray, IdArray);
template NDArray Scatter<kDLCPU, int32_t, int64_t>(NDArray, IdArray);
template NDArray Scatter<kDLCPU, int64_t, int64_t>(NDArray, IdArray);
template NDArray Scatter<kDLCPU, float, int64_t>(NDArray, IdArray);
template NDArray Scatter<kDLCPU, double, int64_t>(NDArray, IdArray);

};  // namespace impl
};  // namespace aten
};  // namespace dgl
