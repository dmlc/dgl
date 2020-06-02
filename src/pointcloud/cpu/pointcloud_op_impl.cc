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

inline int64_t get_index(int64_t dim1, int64_t dim2, int64_t dim) {
  return dim1 * dim + dim2;
}

template <DLDeviceType XPU, typename DType, typename IdType>
IdArray FPS(NDArray array, int64_t batch_size, int64_t sample_points, DLContext ctx) {
  const DType* array_data = static_cast<DType*>(array->data);

  const int64_t point_in_batch = array->shape[0] / batch_size;
  const int64_t dim = array->shape[1];

  // Init distance
  std::vector<DType> dist_data(point_in_batch);

  // Init return value
  IdArray ret = NewIdArray(sample_points * batch_size, ctx, sizeof(IdType) * 8);
  IdType* ret_data = static_cast<IdType*>(ret->data);
  std::fill(ret_data, ret_data + sample_points * batch_size, 0);

  IdType array_start = 0, ret_start = 0;
  // loop for each point cloud sample in this batch
  for (auto b = 0; b < batch_size; b++) {
    // random init start sample
    IdType sample_idx = rand() % point_in_batch;
    ret_data[ret_start] = array_start + sample_idx;

    // compute the first-sample distance, and get the max value
    IdType dist_argmax = 0;
    DType dist_max = -1;

    // sample the rest `sample_points - 1` points
    for (auto i = 0; i < sample_points - 1; i++) {
      // re-init distance and the argmax
      dist_argmax = 0;
      dist_max = -1;

      // update the distance
      for (auto j = 0; j < point_in_batch; j++) {
        // compute the distance on dimensions
        DType one_dist = 0;
        for (auto d = 0; d < dim; d++) {
          DType tmp = array_data[get_index(array_start + j, d, dim)] - array_data[get_index(array_start + sample_idx, d, dim)];
          one_dist += tmp * tmp;
        }

        // for each out-of-set point, keep its nearest to-the-set distance
        if (i == 0 || dist_data[j] > one_dist) {
          dist_data[j] = one_dist;
        }
        // look for the farthest sample
        if (dist_data[j] > dist_max) {
          dist_argmax = j;
          dist_max = dist_data[j];
        }
      }
      // sample the `dist_argmax`-th point
      sample_idx = dist_argmax;
      ret_data[ret_start + i + 1] = array_start + sample_idx;
    }

    array_start += point_in_batch;
    ret_start += sample_points;
  }
  return ret;
}

template IdArray FPS<kDLCPU, int32_t, int32_t>(NDArray array, int64_t batch_size, int64_t sample_points, DLContext ctx);
template IdArray FPS<kDLCPU, int64_t, int32_t>(NDArray array, int64_t batch_size, int64_t sample_points, DLContext ctx);
template IdArray FPS<kDLCPU, float, int32_t>(NDArray array, int64_t batch_size, int64_t sample_points, DLContext ctx);
template IdArray FPS<kDLCPU, double, int32_t>(NDArray array, int64_t batch_size, int64_t sample_points, DLContext ctx);
template IdArray FPS<kDLCPU, int32_t, int64_t>(NDArray array, int64_t batch_size, int64_t sample_points, DLContext ctx);
template IdArray FPS<kDLCPU, int64_t, int64_t>(NDArray array, int64_t batch_size, int64_t sample_points, DLContext ctx);
template IdArray FPS<kDLCPU, float, int64_t>(NDArray array, int64_t batch_size, int64_t sample_points, DLContext ctx);
template IdArray FPS<kDLCPU, double, int64_t>(NDArray array, int64_t batch_size, int64_t sample_points, DLContext ctx);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
