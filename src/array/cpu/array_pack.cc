/**
 *  Copyright (c) 2019 by Contributors
 * @file array/cpu/array_index_select.cc
 * @brief Array index select CPU implementation
 */
#include <dgl/array.h>
#include <dgl/runtime/parallel_for.h>

#include <tuple>
#include <utility>

namespace dgl {
using runtime::NDArray;
using runtime::parallel_for;
namespace aten {
namespace impl {

template <DGLDeviceType XPU, typename DType, typename IdType>
std::pair<NDArray, IdArray> ConcatSlices(NDArray array, IdArray lengths) {
  const int64_t rows = lengths->shape[0];
  const int64_t cols = (array->ndim == 1 ? array->shape[0] : array->shape[1]);
  const int64_t stride = (array->ndim == 1 ? 0 : cols);
  const DType *array_data = static_cast<DType *>(array->data);
  const IdType *length_data = static_cast<IdType *>(lengths->data);

  IdArray offsets = NewIdArray(rows, array->ctx, sizeof(IdType) * 8);
  IdType *offsets_data = static_cast<IdType *>(offsets->data);
  for (int64_t i = 0; i < rows; ++i)
    offsets_data[i] = (i == 0 ? 0 : length_data[i - 1] + offsets_data[i - 1]);
  const int64_t total_length = offsets_data[rows - 1] + length_data[rows - 1];

  NDArray concat = NDArray::Empty({total_length}, array->dtype, array->ctx);
  DType *concat_data = static_cast<DType *>(concat->data);

  parallel_for(0, rows, [=](size_t b, size_t e) {
    for (auto i = b; i < e; ++i) {
      for (int64_t j = 0; j < length_data[i]; ++j)
        concat_data[offsets_data[i] + j] = array_data[i * stride + j];
    }
  });

  return std::make_pair(concat, offsets);
}

template std::pair<NDArray, IdArray> ConcatSlices<kDGLCPU, int32_t, int32_t>(
    NDArray, IdArray);
template std::pair<NDArray, IdArray> ConcatSlices<kDGLCPU, int64_t, int32_t>(
    NDArray, IdArray);
template std::pair<NDArray, IdArray> ConcatSlices<kDGLCPU, float, int32_t>(
    NDArray, IdArray);
template std::pair<NDArray, IdArray> ConcatSlices<kDGLCPU, double, int32_t>(
    NDArray, IdArray);
template std::pair<NDArray, IdArray> ConcatSlices<kDGLCPU, int32_t, int64_t>(
    NDArray, IdArray);
template std::pair<NDArray, IdArray> ConcatSlices<kDGLCPU, int64_t, int64_t>(
    NDArray, IdArray);
template std::pair<NDArray, IdArray> ConcatSlices<kDGLCPU, float, int64_t>(
    NDArray, IdArray);
template std::pair<NDArray, IdArray> ConcatSlices<kDGLCPU, double, int64_t>(
    NDArray, IdArray);

template <DGLDeviceType XPU, typename DType>
std::tuple<NDArray, IdArray, IdArray> Pack(NDArray array, DType pad_value) {
  CHECK_NDIM(array, 2, "array");
  const DType *array_data = static_cast<DType *>(array->data);
  const int64_t rows = array->shape[0];
  const int64_t cols = array->shape[1];

  IdArray length = NewIdArray(rows, array->ctx);
  int64_t *length_data = static_cast<int64_t *>(length->data);
  parallel_for(0, rows, [=](size_t b, size_t e) {
    for (auto i = b; i < e; ++i) {
      int64_t j;
      for (j = 0; j < cols; ++j) {
        const DType val = array_data[i * cols + j];
        if (val == pad_value) break;
      }
      length_data[i] = j;
    }
  });

  auto ret = ConcatSlices<XPU, DType, int64_t>(array, length);
  return std::make_tuple(ret.first, length, ret.second);
}

template std::tuple<NDArray, IdArray, IdArray> Pack<kDGLCPU, int32_t>(
    NDArray, int32_t);
template std::tuple<NDArray, IdArray, IdArray> Pack<kDGLCPU, int64_t>(
    NDArray, int64_t);
template std::tuple<NDArray, IdArray, IdArray> Pack<kDGLCPU, float>(
    NDArray, float);
template std::tuple<NDArray, IdArray, IdArray> Pack<kDGLCPU, double>(
    NDArray, double);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
