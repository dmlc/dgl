/**
 *  Copyright (c) 2019 by Contributors
 * @file array/cpu/array_scatter.cc
 * @brief Array scatter CPU implementation
 */
#include <dgl/array.h>
#include <dgl/runtime/parallel_for.h>

namespace dgl {
using runtime::NDArray;
namespace aten {
namespace impl {

template <DGLDeviceType XPU, typename DType, typename IdType>
NDArray Scatter(NDArray array, IdArray indices) {
  NDArray result =
      NDArray::Empty({indices->shape[0]}, array->dtype, array->ctx);

  const DType *array_data = static_cast<DType *>(array->data);
  const IdType *indices_data = static_cast<IdType *>(indices->data);
  DType *result_data = static_cast<DType *>(result->data);

  for (int64_t i = 0; i < indices->shape[0]; ++i)
    result_data[indices_data[i]] = array_data[i];

  return result;
}

template NDArray Scatter<kDGLCPU, int32_t, int32_t>(NDArray, IdArray);
template NDArray Scatter<kDGLCPU, int64_t, int32_t>(NDArray, IdArray);
template NDArray Scatter<kDGLCPU, float, int32_t>(NDArray, IdArray);
template NDArray Scatter<kDGLCPU, double, int32_t>(NDArray, IdArray);
template NDArray Scatter<kDGLCPU, int32_t, int64_t>(NDArray, IdArray);
template NDArray Scatter<kDGLCPU, int64_t, int64_t>(NDArray, IdArray);
template NDArray Scatter<kDGLCPU, float, int64_t>(NDArray, IdArray);
template NDArray Scatter<kDGLCPU, double, int64_t>(NDArray, IdArray);

template <DGLDeviceType XPU, typename DType, typename IdType>
void Scatter_(IdArray index, NDArray value, NDArray out) {
  const int64_t len = index->shape[0];
  const IdType *idx = index.Ptr<IdType>();
  const DType *val = value.Ptr<DType>();
  DType *outd = out.Ptr<DType>();
  runtime::parallel_for(0, len, [&](size_t b, size_t e) {
    for (auto i = b; i < e; ++i) {
      outd[idx[i]] = val[i];
    }
  });
}

template void Scatter_<kDGLCPU, int32_t, int32_t>(IdArray, NDArray, NDArray);
template void Scatter_<kDGLCPU, int64_t, int32_t>(IdArray, NDArray, NDArray);
template void Scatter_<kDGLCPU, float, int32_t>(IdArray, NDArray, NDArray);
template void Scatter_<kDGLCPU, double, int32_t>(IdArray, NDArray, NDArray);
template void Scatter_<kDGLCPU, int32_t, int64_t>(IdArray, NDArray, NDArray);
template void Scatter_<kDGLCPU, int64_t, int64_t>(IdArray, NDArray, NDArray);
template void Scatter_<kDGLCPU, float, int64_t>(IdArray, NDArray, NDArray);
template void Scatter_<kDGLCPU, double, int64_t>(IdArray, NDArray, NDArray);

};  // namespace impl
};  // namespace aten
};  // namespace dgl
