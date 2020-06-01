/*!
 *  Copyright (c) 2019 by Contributors
 * \file array/cpu/array_op_impl.cc
 * \brief Array operator CPU implementation
 */
#include <dgl/array.h>
#include <numeric>
#include <vector>

namespace dgl {
using runtime::NDArray;
namespace aten {
namespace impl {

template <DLDeviceType XPU, typename DType, typename IdType>
IdArray FPS(NDArray array, IdArray batch_ptr, int64_t npoints, DLContext ctx) {
  const DType* array_data = static_cast<DType*>(array->data);
  const IdType* batch_data = static_cast<IdType*>(batch_ptr->data);
  const auto batch_size = batch_ptr->shape[0] - 1;
  const auto point_in_batch = batch_data[1] - batch_data[0];

  // Init distance
  IdArray dist = NewIdArray(point_in_batch, ctx, sizeof(IdType) * 8);
  IdType* dist_data = static_cast<IdType*>(dist->data);
  std::fill(dist_data, dist_data + point_in_batch, 0);

  // Init return value
  IdArray ret = NewIdArray(npoints * batch_size, ctx, sizeof(IdType) * 8);
  IdType* ret_data = static_cast<IdType*>(ret->data);

  IdType src_start = 0, out_start = 0, src_end, out_end;
  for (auto b = 0; b < point_in_batch; b++) {
    src_end = batch_data[b];
    out_end = npoints * b;

    // init start sample
    IdType start_idx = rand() % point_in_batch;
    ret_data[out_start] = src_start + start_idx;

    // Compute the first-sample distance, and get the max value
    IdType dist_argmax = 0, dist_max = -1;
    for (auto i = 0; i < src_end - src_start; i++) {
      DType d = array_data[i + src_start] - array_data[i + start_idx];
      dist_data[i] = d * d;
      if (dist_data[i] > dist_max) {
        dist_argmax = i;
        dist_max = dist_data[i];
      }
    }

    for (auto i = 1; i < out_end - out_start; i++) {
      // add the `dist_argmax`-th sample
      ret_data[out_start + i] = src_start + dist_argmax;
      // update distance and the argmax
      auto new_dist_argmax = 0;
      auto new_dist_max = -1;
      for (auto j = 0; j < src_end - src_start; j++) {
        DType d = array_data[j + src_start] - array_data[j + dist_argmax];
        DType dsqr = d * d;
        if (dist_data[j] > dsqr) {
          dist_data[j] = dsqr;
        }
        if (dist_data[j] > new_dist_max) {
          new_dist_argmax = j;
          new_dist_max = dist_data[j];
        }
      }
      dist_argmax = new_dist_argmax;
    }

    src_start = src_end;
    out_start = out_end;
  }
  return ret;
}

template IdArray FPS<kDLCPU, int32_t, int32_t>(NDArray array, IdArray batch_ptr, int64_t npoints, DLContext ctx);
template IdArray FPS<kDLCPU, int64_t, int32_t>(NDArray array, IdArray batch_ptr, int64_t npoints, DLContext ctx);
template IdArray FPS<kDLCPU, float, int32_t>(NDArray array, IdArray batch_ptr, int64_t npoints, DLContext ctx);
template IdArray FPS<kDLCPU, double, int32_t>(NDArray array, IdArray batch_ptr, int64_t npoints, DLContext ctx);
template IdArray FPS<kDLCPU, int32_t, int64_t>(NDArray array, IdArray batch_ptr, int64_t npoints, DLContext ctx);
template IdArray FPS<kDLCPU, int64_t, int64_t>(NDArray array, IdArray batch_ptr, int64_t npoints, DLContext ctx);
template IdArray FPS<kDLCPU, float, int64_t>(NDArray array, IdArray batch_ptr, int64_t npoints, DLContext ctx);
template IdArray FPS<kDLCPU, double, int64_t>(NDArray array, IdArray batch_ptr, int64_t npoints, DLContext ctx);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
