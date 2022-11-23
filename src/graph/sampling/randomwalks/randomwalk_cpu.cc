/**
 *  Copyright (c) 2018 by Contributors
 * @file graph/sampling/randomwalk_cpu.cc
 * @brief DGL sampler - CPU implementation of metapath-based random walk with
 * OpenMP
 */

#include <dgl/array.h>
#include <dgl/base_heterograph.h>
#include <dgl/runtime/device_api.h>

#include <algorithm>
#include <utility>
#include <vector>

#include "metapath_randomwalk.h"
#include "randomwalks_cpu.h"
#include "randomwalks_impl.h"

namespace dgl {

using namespace dgl::runtime;
using namespace dgl::aten;

namespace sampling {

namespace impl {

template <DGLDeviceType XPU, typename IdxType>
std::pair<IdArray, IdArray> RandomWalk(
    const HeteroGraphPtr hg, const IdArray seeds, const TypeArray metapath,
    const std::vector<FloatArray> &prob) {
  TerminatePredicate<IdxType> terminate = [](IdxType *data, dgl_id_t curr,
                                             int64_t len) { return false; };

  return MetapathBasedRandomWalk<XPU, IdxType>(
      hg, seeds, metapath, prob, terminate);
}

template <DGLDeviceType XPU, typename IdxType>
std::tuple<IdArray, IdArray, IdArray> SelectPinSageNeighbors(
    const IdArray src, const IdArray dst, const int64_t num_samples_per_node,
    const int64_t k) {
  CHECK(src->ctx.device_type == kDGLCPU) << "IdArray needs be on CPU!";
  int64_t len = src->shape[0] / num_samples_per_node;
  IdxType *src_data = src.Ptr<IdxType>();
  const IdxType *dst_data = dst.Ptr<IdxType>();
  std::vector<IdxType> res_src_vec, res_dst_vec, res_cnt_vec;
  for (int64_t i = 0; i < len; ++i) {
    int64_t start_idx = (i * num_samples_per_node);
    int64_t end_idx = (start_idx + num_samples_per_node);
    IdxType dst_node = dst_data[start_idx];
    std::sort(src_data + start_idx, src_data + end_idx);
    int64_t cnt = 0;
    std::vector<std::pair<IdxType, IdxType>> vec;
    for (int64_t j = start_idx; j < end_idx; ++j) {
      if ((j != start_idx) && (src_data[j] != src_data[j - 1])) {
        if (src_data[j - 1] != -1) {
          vec.emplace_back(std::make_pair(cnt, src_data[j - 1]));
        }
        cnt = 0;
      }
      ++cnt;
    }
    // add last count
    if (src_data[end_idx - 1] != -1) {
      vec.emplace_back(std::make_pair(cnt, src_data[end_idx - 1]));
    }
    std::sort(
        vec.begin(), vec.end(), std::greater<std::pair<IdxType, IdxType>>());
    int64_t len = std::min(vec.size(), static_cast<size_t>(k));
    for (int64_t j = 0; j < len; ++j) {
      auto pair_item = vec[j];
      res_src_vec.emplace_back(pair_item.second);
      res_dst_vec.emplace_back(dst_node);
      res_cnt_vec.emplace_back(pair_item.first);
    }
  }
  IdArray res_src = IdArray::Empty(
      {static_cast<int64_t>(res_src_vec.size())}, src->dtype, src->ctx);
  IdArray res_dst = IdArray::Empty(
      {static_cast<int64_t>(res_dst_vec.size())}, dst->dtype, dst->ctx);
  IdArray res_cnt = IdArray::Empty(
      {static_cast<int64_t>(res_cnt_vec.size())}, src->dtype, src->ctx);

  // copy data from vector to NDArray
  auto device = runtime::DeviceAPI::Get(src->ctx);
  device->CopyDataFromTo(
      static_cast<IdxType *>(res_src_vec.data()), 0, res_src.Ptr<IdxType>(), 0,
      sizeof(IdxType) * res_src_vec.size(), DGLContext{kDGLCPU, 0},
      res_src->ctx, res_src->dtype);
  device->CopyDataFromTo(
      static_cast<IdxType *>(res_dst_vec.data()), 0, res_dst.Ptr<IdxType>(), 0,
      sizeof(IdxType) * res_dst_vec.size(), DGLContext{kDGLCPU, 0},
      res_dst->ctx, res_dst->dtype);
  device->CopyDataFromTo(
      static_cast<IdxType *>(res_cnt_vec.data()), 0, res_cnt.Ptr<IdxType>(), 0,
      sizeof(IdxType) * res_cnt_vec.size(), DGLContext{kDGLCPU, 0},
      res_cnt->ctx, res_cnt->dtype);

  return std::make_tuple(res_src, res_dst, res_cnt);
}

template std::pair<IdArray, IdArray> RandomWalk<kDGLCPU, int32_t>(
    const HeteroGraphPtr hg, const IdArray seeds, const TypeArray metapath,
    const std::vector<FloatArray> &prob);
template std::pair<IdArray, IdArray> RandomWalk<kDGLCPU, int64_t>(
    const HeteroGraphPtr hg, const IdArray seeds, const TypeArray metapath,
    const std::vector<FloatArray> &prob);

template std::tuple<IdArray, IdArray, IdArray>
SelectPinSageNeighbors<kDGLCPU, int32_t>(
    const IdArray src, const IdArray dst, const int64_t num_samples_per_node,
    const int64_t k);
template std::tuple<IdArray, IdArray, IdArray>
SelectPinSageNeighbors<kDGLCPU, int64_t>(
    const IdArray src, const IdArray dst, const int64_t num_samples_per_node,
    const int64_t k);

};  // namespace impl

};  // namespace sampling

};  // namespace dgl
